"""Audit E20 frozen scores and certify old-cohort winner margins without refitting."""

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack


EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "e20_independent_confirmation_preparation"))
from predict_final_inputs import features


TAU = 1e-8
THRESHOLDS = (0, 1e-12, 1e-10, 1e-9, TAU, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
QUANTILES = (0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
CERTIFICATE_SUPPORT = 1e-8
DUAL_GRID = 2**40
REG = 0.003472222222222222
MODEL_SHA = "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92"
PREDICTION_SHA = "b790643119b6d85b370d34b6547e89e1316c26b26ad0d4a005e047babc521c75"
REWARD_SHA = "a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_npz(path, keys, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError(f"Unexpected frozen input identity: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def top_two(scores):
    ordered = np.sort(scores, axis=-1)
    return ordered[..., -1] - ordered[..., -2], scores.argmax(axis=-1)


def distributions(phi, rewards, weights, kept):
    """Find only dual weights using a linear residual-minimization problem."""
    transformed = 2 * (phi @ weights) - rewards
    maxima = rewards == rewards.max(axis=1, keepdims=True)
    alpha = maxima / maxima.sum(axis=1, keepdims=True)
    rows, actions = np.nonzero(kept[:, None] & (
        transformed >= transformed.max(axis=1, keepdims=True) - CERTIFICATE_SUPPORT
    ))
    denom, action_count = rewards.shape
    entries = len(rows)
    feature_count = phi.shape[1]
    # Gradient components are indexed by (action, feature).
    gradient_rows = (actions[:, None] * feature_count + np.arange(feature_count)).ravel()
    direction = coo_matrix(
        ((2 / denom * phi[rows]).ravel(),
         (gradient_rows, np.repeat(np.arange(entries), feature_count))),
        shape=(feature_count * action_count, entries),
    ).tocsr()
    fixed = (2 * REG * weights - 2 / denom * phi.T @ (kept[:, None] * alpha)).T.ravel()
    residual_column = csr_matrix(-np.ones((len(fixed), 1)))
    row_ids = np.cumsum(kept)[rows] - 1
    equality = csr_matrix((np.ones(entries), (row_ids, np.arange(entries))),
                          shape=(int(kept.sum()), entries + 1))
    result = linprog(np.r_[np.zeros(entries), 1.0],
                     A_eq=equality, b_eq=np.ones(equality.shape[0]),
                     A_ub=vstack([hstack([direction, residual_column]),
                                  hstack([-direction, residual_column])]),
                     b_ub=np.r_[-fixed, fixed], bounds=(0.0, None), method="highs",
                     options={"primal_feasibility_tolerance": 1e-10,
                              "dual_feasibility_tolerance": 1e-10})
    if not result.success:
        raise RuntimeError("Dual-witness recovery failed: " + result.message)
    proposed = np.zeros_like(rewards)
    proposed[rows, actions] = np.maximum(result.x[:-1], 0)
    if np.any(proposed[kept].sum(axis=1) <= 0):
        raise ValueError("A retained training row has no proposed dual mass")
    integer = np.zeros_like(rewards, dtype=np.int64)
    integer[kept] = np.floor(DUAL_GRID * proposed[kept] / proposed[kept].sum(axis=1, keepdims=True))
    integer[np.arange(denom), proposed.argmax(axis=1)] += np.where(
        kept, DUAL_GRID - integer.sum(axis=1), 0
    )
    if (integer < 0).any() or not np.array_equal(integer.sum(axis=1), kept * DUAL_GRID):
        raise ValueError("Quantized dual weights must be exactly row feasible")
    return integer, float(result.fun), entries


def exact_binary_rows(array):
    return [[Fraction(float(item)) for item in row] for row in array]


def exact_gap(phi, rewards, weights, kept, dual):
    """Exact upper bound F(W)-F(W*) for frozen binary64 inputs and lambda."""
    n, action_count = rewards.shape
    penalty = Fraction(REG)
    coefficients = exact_binary_rows(weights.T)
    feature_count = phi.shape[1]
    gradient = [[2 * penalty * coefficient for coefficient in row] for row in coefficients]
    max_slack = Fraction(0)
    for row in np.flatnonzero(kept):
        feature = [Fraction(float(value)) for value in phi[row]]
        reward = [Fraction(float(value)) for value in rewards[row]]
        top = max(reward)
        top_count = reward.count(top)
        transformed = [2 * sum(x * y for x, y in zip(feature, coefficient)) - value
                       for coefficient, value in zip(coefficients, reward)]
        row_max = max(transformed)
        for action in range(action_count):
            mass = Fraction(int(dual[row, action]), DUAL_GRID)
            max_slack += mass * (row_max - transformed[action]) / n
            offset = Fraction(1, top_count) if reward[action] == top else Fraction(0)
            direction = 2 * (mass - offset) / n
            if direction:
                for index in range(feature_count):
                    gradient[action][index] += direction * feature[index]
    norm_squared = sum(entry * entry for row in gradient for entry in row)
    gap = max_slack + norm_squared / (4 * penalty)
    if gap < 0 or max_slack < 0:
        raise AssertionError("The exact certificate components must be nonnegative")
    return gap, max_slack, norm_squared


def certified_near_tie_gaps(phi, weights, winners, chosen, gap):
    """Check all competitors exactly on the replayed binary64 design."""
    coefficient = exact_binary_rows(weights.T)
    exact_tau = Fraction(TAU)
    result = np.zeros(len(phi), dtype=bool)
    bounds = []
    for index in np.flatnonzero(chosen):
        feature = [Fraction(float(value)) for value in phi[index]]
        squared_bound = 2 * gap / Fraction(REG) * sum(value * value for value in feature)
        bounds.append(math.sqrt(float(squared_bound)))
        values = [sum(x * y for x, y in zip(feature, row)) for row in coefficient]
        winner = int(winners[index])
        best_other = max(value for action, value in enumerate(values) if action != winner)
        margin = values[winner] - best_other
        remainder = margin - exact_tau
        result[index] = remainder > 0 and remainder * remainder > squared_bound
    return result, [min(bounds), max(bounds)]


def grouped_thresholds(margins, winners, groups):
    changed = winners[0] != winners[1]
    output = []
    for threshold in THRESHOLDS:
        unique = margins > threshold
        both = unique.all(axis=0)
        rec = {
            "tau": threshold,
            "B_unique": int(unique[0].sum()),
            "C_unique": int(unique[1].sum()),
            "both_unique": int(both.sum()),
            "strict_changed": int(np.sum(both & changed)),
            "strict_changed_by_task": {task: int(np.sum(both & changed & mask))
                                       for task, mask in groups.items()},
        }
        output.append(rec)
    return output


def run(asset_root):
    p2 = asset_root / "research/e20_p2_execution/private"
    final_models = asset_root / "research/e20_independent_confirmation_preparation/private/final_models"
    p2_scores_path = p2 / "P2_PREDICTIONS_FROZEN.npz"
    embedding_path = p2 / "P2_INPUTS_ONLY.npz"
    train_outcomes = asset_root / "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
    public_p2 = EXPERIMENTS / "e20_p2_execution"
    evidence_path = EXPERIMENTS / "e20_independent_confirmation_preparation/FINAL_MODEL_NUMERICAL_AUDIT.json"
    main_evidence = load_json(evidence_path)
    encoder_receipt = public_p2 / "P2_ENCODER_RECEIPT.json"
    packet = load_npz(p2_scores_path, ("scores", "margins", "support_counts", "conditions",
                                      "question_ids", "canonical_hashes", "task_labels"), PREDICTION_SHA)
    encoded_sha = load_json(encoder_receipt)["vector_packet_sha256"]
    inputs = load_npz(embedding_path, ("embeddings", "question_ids", "canonical_hashes", "task_labels"),
                     encoded_sha)
    coefficient_keys = ("weights", "pca_mean", "pca_components", "scaler_mean", "scaler_scale",
                        "phi_h", "question_ids", "canonical_hashes", "conditions")
    state = load_npz(final_models / "FINAL600_MODELS_FROZEN.npz", coefficient_keys, MODEL_SHA)
    outcomes = load_npz(train_outcomes, ("aggregate_scores", "question_ids", "canonical_hashes",
                                        "generation_seeds"), REWARD_SHA)
    for key in ("question_ids", "canonical_hashes"):
        if not np.array_equal(inputs[key], packet[key]) or not np.array_equal(state[key], outcomes[key]):
            raise ValueError("Input and training identity order must match the frozen artifacts")
    if (not np.array_equal(inputs["task_labels"], packet["task_labels"])
            or not np.array_equal(state["conditions"], ["B", "C"])
            or not np.array_equal(packet["conditions"], ["B", "C"])
            or not np.array_equal(outcomes["generation_seeds"], [42, 137, 2024])):
        raise ValueError("Frozen action-head, task, or seed order differs")
    scores = packet["scores"]
    reward = outcomes["aggregate_scores"]
    if (scores.shape != (2, 2400, 20) or not np.isfinite(scores).all()
            or state["weights"].shape != (2, 17, 20) or state["phi_h"].shape != (600, 17)
            or reward.shape != (3, 600, 20) or not np.isin(reward, [0, 1]).all()
            or int(np.sum(packet["task_labels"] == "gsm8k")) != 1200
            or int(np.sum(packet["task_labels"] == "bbh")) != 1200
            or main_evidence["frozen_model_sha256"] != MODEL_SHA
            or main_evidence["final600_regularization"] != REG):
        raise ValueError("Expected E20 frozen train600 and P2 balanced new2400 objects")
    phi = features(inputs["embeddings"], state)
    replay = np.stack([phi @ np.asfortranarray(weight) for weight in state["weights"]])
    alternative = np.einsum("qd,hda->hqa", phi, state["weights"])
    margins, winners = top_two(scores)
    changed = winners[0] != winners[1]
    strict = changed & (margins > TAU).all(axis=0)
    if (not np.array_equal(margins, packet["margins"]) or int(strict.sum()) != 302
            or not np.array_equal((margins > TAU), packet["support_counts"] == 1)
            or float(np.max(np.abs(replay - scores))) > 1e-13):
        raise ValueError("Frozen P2 score, strict margin, or support reproduction failed")
    alternative_margins, alternative_winners = top_two(alternative)
    alternative_strict = (alternative_winners[0] != alternative_winners[1]) & (
        alternative_margins > TAU
    ).all(axis=0)
    mean_reward = reward.mean(axis=0)
    all_tied = np.all(mean_reward == mean_reward[:, :1], axis=1)
    if all_tied.sum() != 306:
        raise ValueError("All-action training tie count differs")
    certificates = {}
    combined = strict.copy()
    for index, head in enumerate(("B", "C")):
        log = main_evidence["model_diagnostics"][head]
        kept = np.ones(len(all_tied), dtype=bool) if head == "B" else ~all_tied
        weight = state["weights"][index]
        integer_dual, lp_residual, support_size = distributions(state["phi_h"], mean_reward, weight, kept)
        gap, slack, norm_squared = exact_gap(state["phi_h"], mean_reward, weight, kept, integer_dual)
        certified, pair_bounds = certified_near_tie_gaps(phi, weight, winners[index], strict, gap)
        combined &= certified
        certificates[head] = dict(
            historical_solver=dict(
                status=log["status"], stationarity_inf=log["stationarity_inf"],
                primal_dual_objective_gap=log["primal_dual_objective_gap"],
                primal_violation=log["primal_violation"], dual_violation=log["dual_violation"],
            ),
            feasible_witness_objective_gap_bound=float(gap),
            affine_minorant_loss_slack=float(slack), minorant_gradient_norm=math.sqrt(float(norm_squared)),
            recovery_lp_max_residual=lp_residual, recovery_lp_nonzero_candidates=support_size,
            pairwise_score_perturbation_bounds_range_among_strict_switches=pair_bounds,
            strict_switched_questions_with_certified_head_unique=int(certified.sum()),
        )
    group_masks = {task: packet["task_labels"] == task for task in ("gsm8k", "bbh")}
    return {
        "schema": "e24_old_p2_margin_audit_v1",
        "old_cohort_questions": len(strict),
        "frozen_tau": TAU,
        "input_hashes": {"predictions": PREDICTION_SHA, "model": MODEL_SHA, "encoded_inputs": encoded_sha,
                         "train_outcomes": REWARD_SHA, "numerical_audit": digest(evidence_path)},
        "source_sha256": digest(Path(__file__)),
        "quantile_probabilities": QUANTILES,
        "both_strict_switch_margin_quantiles": {
            label: np.quantile(values[strict], QUANTILES).tolist()
            for label, values in (("B", margins[0]), ("C", margins[1]), ("minimum_B_C", margins.min(axis=0)))
        },
        "fixed_thresholds": grouped_thresholds(margins, winners, group_masks),
        "inference_replay": {
            "frozen_operator_max_score_delta": float(np.max(np.abs(replay - scores))),
            "alternative_binary64_einsum_max_score_delta": float(np.max(np.abs(alternative - scores))),
            "raw_winner_differences_by_head": np.sum(alternative_winners != winners, axis=1).tolist(),
            "strict_switch_membership_differences": int(np.sum(alternative_strict != strict)),
            "original_strict_switch_winner_differences_by_head":
                np.sum((alternative_winners != winners)[:, strict], axis=1).tolist(),
        },
        "optimality_certificate": {
            "dual_candidate_gap_window": CERTIFICATE_SUPPORT,
            "dual_integer_denominator": DUAL_GRID,
            "exact_arithmetic_scope": "Stored binary64 train design, empirical reward means and lambda; replayed binary64 P2 design held fixed.",
            "heads": certificates,
            "strict_switches_certified_under_both_exact_optimizers": int(combined.sum()),
            "strict_switches_not_per_joint_bound": int(np.sum(strict & ~combined)),
            "certified_switch_counts_by_task": {task: int(np.sum(mask & combined)) for task, mask in group_masks.items()},
        },
        "new_head_fits": 0,
        "new_llm_calls": 0,
        "new_evaluation_gold_reads": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.assets)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
