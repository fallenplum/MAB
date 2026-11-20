"""Replay a fixed-K SkyLLM estimator and calculate reward-free policy bounds."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e24_training_deployment_replication"))
from replication import four_policies, group_weights, labels_for, read, sha, write


def load_fields(path, fields):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in fields}


def require_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Frozen file changed: {path.name}")
    return actual


def uniform(mask):
    return mask / mask.sum(axis=1, keepdims=True)


def effect(coefficient, weights, tasks):
    """Sharp point-cloud bound for a zero-sum contrast and scores in [0, 1]."""
    if np.max(np.abs(coefficient.sum(axis=1))) > 1e-12:
        raise ValueError("Policy contrast is not zero-sum")
    tv = np.abs(coefficient).sum(axis=1) / 2
    positive = np.maximum(coefficient, 0).sum(axis=1)
    if not np.allclose(tv, positive, atol=1e-12, rtol=0):
        raise AssertionError("Coefficient-sign extremizer did not reach the bound")
    return {
        "changed_questions": int((tv > 1e-12).sum()),
        "upper_pp": 100 * float(weights @ tv),
        "by_task_upper_pp": {
            task: 100 * float(np.average(tv[tasks == task], weights=weights[tasks == task]))
            for task in ("gsm8k", "bbh")
        },
    }


def tie_counts(success):
    widths = (success == success.max(axis=1, keepdims=True)).sum(axis=1)
    return {
        "rows": len(widths),
        "multiple_best": int((widths > 1).sum()),
        "full_tie": int((widths == success.shape[1]).sum()),
        "best_set_size_counts": dict(sorted(Counter(widths.tolist()).items())),
    }


def neighbor_scores(history, evaluation, seed_success):
    if len(history) % 10 or len(history) != len(seed_success):
        raise ValueError("Frozen 10% neighborhood requires one count row per historical input")
    k = len(history) // 10
    distances = cdist(evaluation, history, "sqeuclidean")
    ranking = np.argsort(distances, axis=1, kind="stable")
    tied_boundary = distances[np.arange(len(ranking)), ranking[:, k - 1]] == distances[
        np.arange(len(ranking)), ranking[:, k]
    ]
    neighbors = ranking[:, :k]
    complete_tie = seed_success.max(axis=1) == seed_success.min(axis=1)
    retained = seed_success[neighbors].sum(axis=1)
    muted = np.where(complete_tie[:, None], 0, seed_success)[neighbors].sum(axis=1)
    shift = retained - muted
    if not np.all(shift == shift[:, :1]):
        raise AssertionError("Full-tie contributions must be action-independent")
    centered_integer = lambda values: 20 * values - values.sum(axis=1, keepdims=True)
    if not np.array_equal(centered_integer(retained), centered_integer(muted)):
        raise AssertionError("Intervention changed a centered score")
    return retained, muted, seed_success.sum(axis=0), shift[:, 0], int(tied_boundary.sum()), k


def audit_estimator(history, evaluation, success, tasks, weights, tolerance):
    original, intervention, prior, shift, tied_boundary, k = neighbor_scores(history, evaluation, success)
    original_action = original.argmax(axis=1)
    intervention_action = intervention.argmax(axis=1)
    actions = np.eye(original.shape[1])
    best = original == original.max(axis=1, keepdims=True)
    best_intervention = intervention == intervention.max(axis=1, keepdims=True)
    if not np.array_equal(best, best_intervention):
        raise AssertionError("Intervention changed optimal action support")
    priority_count = np.where(best, prior, -1)
    priority = best & (priority_count == priority_count.max(axis=1, keepdims=True))
    margins = np.diff(np.sort(original, axis=1)[:, -2:], axis=1).ravel() / (3 * k)
    positive = margins[margins > 0]
    ordering = np.argsort(-original, axis=1, kind="stable")
    reordered = np.argsort(-intervention, axis=1, kind="stable")
    if not np.array_equal(ordering, reordered):
        raise AssertionError("Intervention changed the ranked actions")

    first = actions[original_action]
    equal_mix = uniform(best)
    historical_mix = uniform(priority)
    return {
        "train_feedback": tie_counts(success),
        "neighbors": k,
        "evaluation": tie_counts(original),
        "tied_l2_boundary_questions": tied_boundary,
        "margin_median": float(np.median(margins)),
        "minimum_positive_margin": float(positive.min()) if len(positive) else None,
        "positive_margin_at_most_1e_minus_8": int(((margins > 0) & (margins <= tolerance)).sum()),
        "absolute_scores_changed_questions": int(np.count_nonzero(shift)),
        "mean_common_score_shift": float(weights @ (shift / (3 * k))),
        "centered_scores_changed_questions": 0,
        "rankings_changed_questions": 0,
        "same_decoder_intervention": effect(first - actions[intervention_action], weights, tasks),
        "first_exact_max_frequency": np.bincount(original_action, minlength=20).tolist(),
        "counterfactual_added_decoders": {
            "uniform_vs_first": effect(equal_mix - first, weights, tasks),
            "history_priority_vs_first": effect(historical_mix - first, weights, tasks),
            "history_priority_vs_uniform": effect(historical_mix - equal_mix, weights, tasks),
        },
    }


def task_weights(tasks):
    return np.where(tasks == "gsm8k", 0.5 / (tasks == "gsm8k").sum(), 0.5 / (tasks == "bbh").sum())


def e25_bounds(path, index_key, e24, weights):
    packet = load_fields(path, (index_key, "scores", "question_ids"))
    if not np.array_equal(packet["question_ids"], e24["question_ids"]):
        raise ValueError("E25 and E24 question orders differ")
    if packet["scores"].shape != (3, 2, len(e24["question_ids"]), 20):
        raise ValueError("Unexpected E25 score dimensions")
    output = []
    for setting, scores in zip(packet[index_key].tolist(), packet["scores"], strict=True):
        b_u, b_p, c_u, c_p = four_policies(scores, e24["combo_success_counts"])
        b = b_u - b_p
        c = c_u - c_p
        tasks = e24["task_labels"]
        output.append({
            "setting": setting,
            "B_U_minus_P": effect(b, weights, tasks),
            "C_U_minus_P": effect(c, weights, tasks),
            "interaction": effect(c - b, weights, tasks),
        })
    return output


def run(root, output):
    protocol_path = HERE / "PROTOCOL.json"
    protocol = read(protocol_path)
    history = root / "artifacts/private/routing_redesign_20261104"
    inputs = history / "full_feedback_inputs/v1/selector_train_inputs.npz"
    rewards = history / "ff_v1/tr/selector_train_outcomes.npz"
    source_hashes = {
        "train_inputs": require_hash(inputs, "57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3"),
        "train_rewards": require_hash(rewards, "a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e"),
    }
    fields = ("embeddings", "question_ids", "canonical_hashes", "task_labels", "subtask_labels")
    train = load_fields(inputs, fields)
    outcome = load_fields(rewards, ("question_ids", "canonical_hashes", "aggregate_scores", "generation_seeds"))
    for key in ("question_ids", "canonical_hashes"):
        if not np.array_equal(train[key], outcome[key]) or len(set(train[key].tolist())) != 600:
            raise ValueError("Training identity mismatch or duplicate")
    y = outcome["aggregate_scores"]
    if y.shape != (3, 600, 20) or not np.isin(y, (0, 1)).all():
        raise ValueError("Expected complete original train600 binary feedback")
    if outcome["generation_seeds"].tolist() != protocol["seed_order"]:
        raise ValueError("Wrong generation seed order")
    success = y.astype(np.int64).sum(axis=0)
    e24_root = root / "research/e24_training_deployment_replication/private/cohort_v1"
    frozen = e24_root / "E24_POLICIES_FROZEN.npz"
    source_hashes["E24_frozen_policies"] = require_hash(
        frozen, "9595ee1ddf9e30e85a0c95e51af2155e6b3de4cbd36265114341f32f9c324cef"
    )
    e24_inputs = e24_root / "INPUTS_ONLY.npz"
    e24 = load_fields(e24_inputs, fields)
    fixed = load_fields(frozen, ("question_ids", "canonical_question_sha256", "combo_success_counts",
                                 "task_labels", "subtask_labels"))
    for key in ("question_ids", "task_labels", "subtask_labels"):
        if not np.array_equal(e24[key], fixed[key]):
            raise ValueError(f"E24 input/policy identity mismatch in {key}")
    if not np.array_equal(e24["canonical_hashes"], fixed["canonical_question_sha256"]):
        raise ValueError("E24 canonical question hashes differ")
    if any(set(train[key].tolist()) & set(e24[key].tolist()) for key in ("question_ids", "canonical_hashes")):
        raise ValueError("Train600 overlaps E24")
    if not np.array_equal(success.sum(axis=0), fixed["combo_success_counts"]):
        raise ValueError("Train600 priors differ from the frozen integer counts")
    source_hashes["E24_inputs"] = sha(e24_inputs)
    e24_protocol = HERE.parent / "e24_training_deployment_replication" / "E24_PROTOCOL.json"
    strata = labels_for(e24["task_labels"], e24["subtask_labels"])
    stratum_weights = group_weights(strata, read(e24_protocol))
    denominator = Counter(strata.tolist())
    weights = np.asarray([stratum_weights[name] / denominator[name] for name in strata])
    for samples, expected in ((train, 600), (e24, 1196)):
        if samples["embeddings"].shape != (expected, 384) or not np.isfinite(samples["embeddings"]).all():
            raise ValueError("Wrong frozen MiniLM input matrix")

    binding = read(HERE.parent / "e19_symmetric_tie_constraint" / "FROZEN_TSPO_REUSE_AUDIT.json")
    folds = []
    seen = []
    tolerance = protocol["numeric_diagnostic_tolerance"]
    for row in binding["folds"]:
        fold = row["fold"]
        fold_path = root / f"research/e13_partial_combination_feedback/private/inputs/fold{fold}/INPUTS_ONLY.npz"
        source_hashes[f"fold{fold}"] = require_hash(fold_path, row["input_packet_sha256"])
        split = load_fields(fold_path, ("H", "E"))
        # Sorting H makes the distance-boundary tie convention a property of the original row IDs.
        h, e = np.sort(split["H"]), split["E"]
        if len(h) != 480 or len(e) != 120 or sorted(np.r_[h, e].tolist()) != list(range(600)):
            raise ValueError("Invalid frozen H480/E120 split")
        tasks = train["task_labels"][e]
        seen.extend(e.tolist())
        folded = audit_estimator(train["embeddings"][h], train["embeddings"][e], success[h],
                                 tasks, task_weights(tasks), tolerance)
        folds.append(dict(fold=fold, **folded))
    if sorted(seen) != list(range(600)):
        raise ValueError("OOF evaluation does not cover each question once")
    independent_inputs = audit_estimator(train["embeddings"], e24["embeddings"], success,
                                         e24["task_labels"], weights, tolerance)
    e25_root = root / "research/e25_tie_mechanism/private"
    mlp = e25_root / "MLP_HEADS.npz"
    linear = e25_root / "LINEAR_HEADS.npz"
    source_hashes["E25_MLP"] = sha(mlp)
    source_hashes["E25_linear"] = sha(linear)
    stored_mlp = e25_bounds(mlp, "seeds", fixed, weights)
    if [row["setting"] for row in stored_mlp] != [1701, 1702, 1703]:
        raise ValueError("Wrong frozen MLP seed list")
    if stored_mlp[0]["interaction"]["upper_pp"] != 0:
        raise ValueError("Frozen MLP main-seed interaction is not zero")
    result = {
        "stage": "E26",
        "status": "STOP_NO_ACTION_EFFECT_SAME_DECODER",
        "protocol_sha256": sha(protocol_path),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": source_hashes,
        "train600_feedback": tie_counts(success),
        "existing_oof_fold_diagnostics": folds,
        "E24_input_only": independent_inputs,
        "E25_reward_free_interaction_bounds": {
            "linear": e25_bounds(linear, "lambda_factors", fixed, weights),
            "MLP": stored_mlp,
        },
        "resources": {
            "physical_member_calls": 0,
            "physical_aggregate_calls": 0,
            "new_tokens": 0,
            "evaluation_rewards_read_for_scoring": 0,
            "fresh400_rewards_read": 0,
            "sealed200_reads": 0,
        },
    }
    write(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.asset_root, args.output)
