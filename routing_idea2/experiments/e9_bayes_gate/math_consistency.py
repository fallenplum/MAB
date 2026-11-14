"""Small, deterministic mathematical qualifications for E9 Bayes gate.

This module uses synthetic arrays only. It never loads research reward caches.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.special import expit


def semantic_argmax(scores: np.ndarray, keys: list[str]) -> np.ndarray:
    """Break exact score ties by immutable semantic key, independent of columns."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape[-1] != len(keys) or len(set(keys)) != len(keys):
        raise ValueError("Action keys must be unique and match score columns")
    order = np.argsort(np.asarray(keys), kind="stable")
    return order[np.argmax(values[..., order], axis=-1)]


def set_spo_loss(q: np.ndarray, rewards: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Set-valued SPO surrogate for fixed nonnegative simplex weights."""
    q, rewards, weights = np.broadcast_arrays(
        np.asarray(q, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=-1), 1.0):
        raise ValueError("Weights must lie on the probability simplex")
    transformed = 2.0 * q - rewards
    return transformed.max(axis=-1) - (weights * transformed).sum(axis=-1)


def near_optimal_weights(rewards: np.ndarray, delta: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    if delta < 0:
        raise ValueError("delta must be nonnegative")
    support = rewards >= rewards.max(axis=-1, keepdims=True) - delta
    return support / support.sum(axis=-1, keepdims=True)


def proper_score_objective(
    flat_weights: np.ndarray,
    phi: np.ndarray,
    repeated_rewards: np.ndarray,
    penalty: float,
    score: str,
) -> tuple[float, np.ndarray]:
    """Independent-action BCE/Brier, mean over question, repeat and action.

    The constant column, when present in phi, receives the same L2 penalty.
    repeated_rewards has shape (question, generation repeat, action).
    Sigmoid-Brier is differentiable but is generally nonconvex in the logits.
    """
    phi = np.asarray(phi, dtype=np.float64)
    rewards = np.asarray(repeated_rewards, dtype=np.float64)
    if rewards.ndim != 3 or rewards.shape[0] != phi.shape[0]:
        raise ValueError("Question and repeated-reward axes do not match")
    if np.any((rewards < 0) | (rewards > 1)):
        raise ValueError("Rewards must lie in [0, 1]")
    weights = np.asarray(flat_weights, dtype=np.float64).reshape(phi.shape[1], rewards.shape[2])
    logits = phi @ weights
    probabilities = expit(logits)
    mean_rewards = rewards.mean(axis=1)
    if score == "bce":
        loss = np.mean(np.logaddexp(0.0, logits) - mean_rewards * logits)
        residual = probabilities - mean_rewards
    elif score == "brier":
        loss = np.mean((probabilities[:, None, :] - rewards) ** 2)
        residual = 2.0 * (probabilities - mean_rewards) * probabilities * (1.0 - probabilities)
    else:
        raise ValueError("Unknown proper score")
    loss += penalty * np.sum(weights**2)
    gradient = phi.T @ residual / (phi.shape[0] * rewards.shape[2]) + 2.0 * penalty * weights
    return float(loss), gradient.ravel()


def two_state_counterexample() -> dict:
    """Solve a two-state surrogate exactly using its linear epigraph."""
    rewards = np.asarray([[0.6, 0.5], [0.0, 1.0]], dtype=np.float64)
    state_probabilities = np.asarray([0.9, 0.1], dtype=np.float64)
    means = state_probabilities @ rewards
    # Variables are the predicted gap t=q_A-q_B and two nonnegative hinges.
    # Hinge 1 >= .1-2t, hinge 2 >= 1+2t.
    solution = linprog(
        np.asarray([0.0, 0.9, 0.1]),
        A_ub=np.asarray([[-2.0, -1.0, 0.0], [2.0, 0.0, -1.0]]),
        b_ub=np.asarray([-0.1, -1.0]),
        bounds=[(None, None), (0.0, None), (0.0, None)],
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    gap = float(solution.x[0])
    q = np.asarray([gap / 2.0, -gap / 2.0])
    weights = near_optimal_weights(rewards, 0.05)
    surrogate = float(state_probabilities @ set_spo_loss(q, rewards, weights))
    empirical_hindsight = rewards.max(axis=1)
    true_spo_risks = state_probabilities @ (empirical_hindsight[:, None] - rewards)
    correct_weights = state_probabilities @ (rewards - rewards.min(axis=1, keepdims=True))
    incorrectly_normalized_weights = state_probabilities @ (
        (rewards - rewards.min(axis=1, keepdims=True))
        / (rewards - rewards.min(axis=1, keepdims=True)).sum(axis=1, keepdims=True)
    )
    assert np.allclose(means, [0.54, 0.55])
    assert np.isclose(gap, 0.05)
    assert np.isclose(surrogate, 0.11)
    assert int(np.argmin(true_spo_risks)) == 1
    return {
        "status": "CONSTRUCTED_COUNTEREXAMPLE_REPRODUCED",
        "distribution": {"probabilities": state_probabilities.tolist(), "rewards": rewards.tolist()},
        "true_means": means.tolist(),
        "bayes_action": "B",
        "surrogate_optimal_predicted_gap_A_minus_B": gap,
        "surrogate_selected_action": "A",
        "expected_surrogate_loss": surrogate,
        "expected_true_spo_loss_per_action": true_spo_risks.tolist(),
        "expected_set_weights_delta_005": (state_probabilities @ weights).tolist(),
        "proper_expected_unnormalized_csce_weights": correct_weights.tolist(),
        "incorrect_row_normalized_weights_negative_control": incorrectly_normalized_weights.tolist(),
        "interpretation": "Known two-point world distribution, not an empirical MAB posterior fit",
        "solver_status": solution.message,
    }


def run_qualifications() -> dict:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    rng = np.random.default_rng(20261114109)
    q = rng.normal(size=(2000, 20))
    r = rng.uniform(size=(2000, 20))
    w = rng.dirichlet(np.ones(20), size=2000)
    q_other = rng.normal(size=q.shape)
    convex_mixing = rng.uniform(size=(2000, 1))
    left = set_spo_loss(convex_mixing * q + (1.0 - convex_mixing) * q_other, r, w)
    right = convex_mixing[:, 0] * set_spo_loss(q, r, w) + (1.0 - convex_mixing[:, 0]) * set_spo_loss(q_other, r, w)
    nonnegative_minimum = float(set_spo_loss(q, r, w).min())
    convexity_maximum_violation = float((left - right).max())
    assert nonnegative_minimum >= -1e-12
    assert convexity_maximum_violation <= 1e-12

    keys = [f"semantic-combination-{i:02d}" for i in range(20)]
    q[0] = 0.0
    q[1, :3] = 99.0
    permutation = rng.permutation(20)
    original_actions = semantic_argmax(q, keys)
    permuted_actions = semantic_argmax(q[:, permutation], [keys[i] for i in permutation])
    restored_actions = permutation[permuted_actions]
    assert np.array_equal(original_actions, restored_actions)
    shift = rng.normal(size=(len(q), 1))
    assert np.array_equal(original_actions, semantic_argmax(q + shift, keys))
    assert np.allclose(set_spo_loss(q, r, w), set_spo_loss(q[:, permutation], r[:, permutation], w[:, permutation]))

    # A lexicographic winner target prefers a particular semantic action even
    # when the observed reward vector contains no preference information.
    common_reward = np.full(20, 0.5)
    winner = np.zeros(20)
    winner[0] = 1.0
    prefer_first = np.zeros(20)
    prefer_first[0] = 0.5
    prefer_second = np.zeros(20)
    prefer_second[1] = 0.5
    lexical_first_loss = float(set_spo_loss(prefer_first, common_reward, winner))
    lexical_second_loss = float(set_spo_loss(prefer_second, common_reward, winner))
    assert lexical_first_loss == 0.0 and lexical_second_loss == 1.0
    uniform_weight = np.full(20, 1.0 / 20.0)
    assert np.isclose(set_spo_loss(np.zeros(20), common_reward, uniform_weight), 0.0)
    assert np.isclose(
        set_spo_loss(prefer_first, common_reward, uniform_weight),
        set_spo_loss(prefer_second, common_reward, uniform_weight),
    )

    phi = rng.normal(size=(8, 4))
    phi[:, 0] = 1.0
    repeated = rng.integers(0, 2, size=(8, 3, 5)).astype(np.float64)
    theta = rng.normal(scale=0.2, size=20)
    gradient_errors = {}
    step = 1e-6
    for score in ("bce", "brier"):
        _, analytic_gradient = proper_score_objective(theta, phi, repeated, 0.03, score)
        finite_difference = np.empty_like(theta)
        for index in range(len(theta)):
            plus = theta.copy()
            minus = theta.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference[index] = (
                proper_score_objective(plus, phi, repeated, 0.03, score)[0]
                - proper_score_objective(minus, phi, repeated, 0.03, score)[0]
            ) / (2.0 * step)
        error = float(np.max(np.abs(finite_difference - analytic_gradient)))
        gradient_errors[score] = error
        assert error < 1e-8

    constant_repeated = np.full((8, 3, 5), 0.5)
    constant_gradient = proper_score_objective(np.zeros(20), phi, constant_repeated, 0.03, "brier")[1]
    assert np.allclose(constant_gradient, 0.0)
    assert np.allclose(expit(0.0), 0.5)

    # Removing a common target component in independent-output Ridge changes
    # every predicted column by the same vector with matched design/penalty.
    target = repeated.mean(axis=1)
    common_target = rng.normal(size=(8, 1))
    hessian = phi.T @ phi + 0.7 * np.eye(phi.shape[1])
    raw_head = np.linalg.solve(hessian, phi.T @ target)
    shifted_head = np.linalg.solve(hessian, phi.T @ (target - common_target))
    head_difference = raw_head - shifted_head
    ridge_public_shift_max_error = float(np.max(np.abs(head_difference - head_difference[:, :1])))
    assert ridge_public_shift_max_error < 1e-12

    return {
        "status": "PASS",
        "evidence_kind": "Synthetic mathematical and numerical qualifications only",
        "seed": 20261114109,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "fixed_weight_set_spo_minimum_loss": nonnegative_minimum,
        "fixed_weight_set_spo_convexity_maximum_violation": convexity_maximum_violation,
        "action_permutation_rows": len(q),
        "action_permutation_mismatches": int(np.count_nonzero(original_actions != restored_actions)),
        "lexicographic_target_all_equal_reward": {
            "loss_prefer_first_semantic_action": lexical_first_loss,
            "loss_prefer_second_semantic_action": lexical_second_loss,
            "finding": "Historical lexicographic SPO winner target can encode an arbitrary preference on tied rewards",
        },
        "proper_score_finite_difference_max_errors": gradient_errors,
        "constant_brier_data_gradient_norm": float(np.linalg.norm(constant_gradient)),
        "ridge_common_target_shift_max_column_error": ridge_public_shift_max_error,
        "counterexample": two_state_counterexample(),
        "limitations": [
            "Finite numerical checks complement explicit proofs and do not establish statistical consistency",
            "Counterexample world distribution is constructed and does not establish posterior calibration on MAB",
            "Sigmoid-Brier is nonconvex in a linear head, BCE is convex in that head",
            "No real research reward assets or held-out data are read",
        ],
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def historical_operator_checks(asset_root: Path) -> dict:
    """Execute unchanged source function ASTs on constructed arrays only.

    Extracting function definitions avoids importing historical command runners
    or opening any real datasets at module import time.
    """
    e8_root = asset_root / "research/e8_decision_focused_combination_learning/public"
    ns_root = asset_root / "research/m4_noise_aware_set_learning/public/code"
    sources = [
        (e8_root / "qualification/e8_assets.py", ["exact_argmax"]),
        (e8_root / "training/e8_training_v1.py", ["_means", "true_optimal_weights", "spo_plus_value_subgradient"]),
        (ns_root / "ns_training.py", ["near_optimal_weights", "ns_value_gradient"]),
        (ns_root / "set_spo_operators.py", ["set_weights", "set_spo_sample_loss"]),
    ]
    namespace = {"np": np, "K": 20}
    provenance = []
    for source_path, functions in sources:
        source_bytes = source_path.read_bytes()
        tree = ast.parse(source_bytes.decode("utf-8-sig"))
        definitions = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name in functions]
        if {item.name for item in definitions} != set(functions):
            raise RuntimeError(f"Required historical operators missing in {source_path}")
        selected_tree = ast.Module(body=definitions, type_ignores=[])
        exec(compile(selected_tree, str(source_path), "exec"), namespace)
        provenance.append({
            "source": str(source_path),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "unchanged_functions_executed": functions,
        })
    rewards = np.zeros((10, 20))
    rewards[:9, :2] = [0.6, 0.5]
    rewards[9, :2] = [0.0, 1.0]
    coefficients = np.full((1, 20), -2.0)
    coefficients[0, :2] = [0.025, -0.025]
    m4_loss, _ = namespace["spo_plus_value_subgradient"](
        coefficients, np.ones((10, 1)), rewards, 0.0, "lexicographic"
    )
    ns_loss, _ = namespace["ns_value_gradient"](
        coefficients, np.ones((1, 1)), rewards[None, :, :], 0.05, 0.0
    )
    set_loss = namespace["set_spo_sample_loss"](
        coefficients,
        rewards,
        namespace["set_weights"](rewards, 0.05),
    ).mean()
    assert np.allclose([m4_loss, ns_loss, set_loss], 0.11)
    tied_rewards = np.full((1, 20), 0.5)
    tied_target = namespace["true_optimal_weights"](tied_rewards, "lexicographic")
    assert np.array_equal(tied_target, np.eye(20)[0:1])
    tie_losses = []
    for preferred in (0, 1):
        tied_q = np.zeros((1, 20))
        tied_q[0, preferred] = 0.5
        tie_losses.append(namespace["spo_plus_value_subgradient"](
            tied_q, np.ones((1, 1)), tied_rewards, 0.0, "lexicographic"
        )[0])
    assert np.allclose(tie_losses, [0.0, 1.0])
    return {
        "status": "PASS_WITH_KNOWN_HISTORICAL_TIE_PREFERENCE",
        "data_access": "source files and constructed arrays only",
        "source_functions": provenance,
        "constructed_twenty_action_counterexample_loss": {
            "historical_M4": float(m4_loss),
            "historical_NS_delta_005": float(ns_loss),
            "historical_set_operators_delta_005": float(set_loss),
        },
        "historical_M4_all_tied_rewards_losses_prefer_action_0_or_1": list(map(float, tie_losses)),
        "interpretation": "A valid lexicographic optimal-label convention creates a target-direction preference on tied empirical rewards",
        "real_reward_cache_reads": 0,
        "new_QP_fits": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-asset-root", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("Qualification receipt already exists")
    result = run_qualifications()
    if arguments.historical_asset_root is not None:
        result["historical_operator_checks"] = historical_operator_checks(arguments.historical_asset_root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
