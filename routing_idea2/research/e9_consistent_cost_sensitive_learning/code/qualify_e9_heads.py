"""Synthetic-only qualification for the frozen E9 predictor-head API."""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp, softmax

HERE = Path(__file__).resolve().parent
STAGED_OPERATORS = HERE.parent / "math" / "public"
if STAGED_OPERATORS.exists():
    sys.path.insert(0, str(STAGED_OPERATORS))

import csce_operators
import e9_heads


def finite_gradient(function, coefficient, step=1e-6):
    numerical = np.empty_like(coefficient)
    for location in np.ndindex(coefficient.shape):
        positive, negative = coefficient.copy(), coefficient.copy()
        positive[location] += step
        negative[location] -= step
        numerical[location] = (function(positive)[0] - function(negative)[0]) / (2.0 * step)
    return numerical


def qualify():
    rng = np.random.default_rng(2026111409)
    checks = {}
    scalar_assertions = 0

    def close(name, actual, expected, tolerance=1e-10):
        nonlocal scalar_assertions
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise AssertionError(name + " shape mismatch")
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        scalar_assertions += actual.size
        if not np.isfinite(error) or error > tolerance:
            raise AssertionError(name + " maximum error " + str(error))
        checks[name] = {"pass": True, "maximum_absolute_error": error, "scalar_values": int(actual.size)}

    def require(name, condition):
        nonlocal scalar_assertions
        if not condition:
            raise AssertionError(name)
        scalar_assertions += 1
        checks[name] = {"pass": True}

    def rejects(name, function):
        try:
            function()
        except ValueError:
            require(name, True)
        else:
            raise AssertionError(name + " failed to reject invalid input")

    phi = np.column_stack((np.ones(23), rng.normal(size=(23, 16))))
    labels = rng.uniform(0.1, 0.85, size=(len(phi), 20))
    coefficient = rng.normal(scale=0.2, size=(17, 20))
    regularization = 100.0 / (3.0 * len(phi) * 20)
    score = phi @ coefficient
    weights = labels - labels.min(axis=1, keepdims=True)
    total = weights.sum(axis=1)
    denominator = len(phi) * 20
    manual_loss = float(np.sum(total * logsumexp(score, axis=1) - np.sum(weights * score, axis=1)) / denominator)
    manual_gradient = phi.T @ (total[:, None] * softmax(score, axis=1) - weights) / denominator
    csce_loss, csce_gradient = csce_operators.csce_value_gradient(coefficient, phi, labels, regularization)
    close("csce_fixed_denominator_value", csce_loss, manual_loss + regularization * np.sum(coefficient**2))
    close("csce_fixed_denominator_gradient", csce_gradient, manual_gradient + 2.0 * regularization * coefficient)
    numerical = finite_gradient(
        lambda candidate: csce_operators.csce_value_gradient(candidate, phi, labels, regularization),
        coefficient,
    )
    close("csce_finite_difference_all340_coefficients", csce_gradient, numerical, 2e-8)
    mse_loss, mse_gradient = e9_heads.mse_value_gradient(coefficient, phi, labels, regularization)
    numerical = finite_gradient(
        lambda candidate: e9_heads.mse_value_gradient(candidate, phi, labels, regularization),
        coefficient,
    )
    close("mse_finite_difference_all340_coefficients", mse_gradient, numerical, 2e-8)
    close("mse_fixed_denominator_value", mse_loss, np.mean((score-labels)**2) + regularization*np.sum(coefficient**2))

    shifted_labels = labels + rng.uniform(-0.04, 0.04, size=(len(phi), 1))
    shifted_value, shifted_gradient = csce_operators.csce_value_gradient(coefficient, phi, shifted_labels, regularization)
    close("row_reward_shift_invariance_value", shifted_value, csce_loss)
    close("row_reward_shift_invariance_gradient", shifted_gradient, csce_gradient)
    score_shift = rng.normal(size=(17, 1))
    shifted_coefficient = coefficient + score_shift
    base_value, base_gradient = csce_operators.csce_value_gradient(coefficient, phi, labels, 0.0)
    shifted_value, shifted_gradient = csce_operators.csce_value_gradient(shifted_coefficient, phi, labels, 0.0)
    close("row_score_shift_unregularized_value", shifted_value, base_value)
    close("row_score_shift_unregularized_gradient", shifted_gradient, base_gradient)

    zero_labels = labels.copy()
    zero_labels[0] = 0.37
    full_value, full_gradient = csce_operators.csce_value_gradient(coefficient, phi, zero_labels, 0.0)
    reduced_value, reduced_gradient = csce_operators.csce_value_gradient(coefficient, phi[1:], zero_labels[1:], 0.0)
    close("zero_weight_row_retained_in_value_denominator", full_value, reduced_value*(len(phi)-1)/len(phi))
    close("zero_weight_row_retained_in_gradient_denominator", full_gradient, reduced_gradient*(len(phi)-1)/len(phi))
    all_equal = np.repeat(rng.uniform(size=(len(phi), 1)), 20, axis=1)
    equal_value, equal_gradient = csce_operators.csce_value_gradient(coefficient, phi, all_equal, 0.0)
    close("all_equal_rewards_zero_data_value", equal_value, 0.0)
    close("all_equal_rewards_zero_data_gradient", equal_gradient, np.zeros((17, 20)))
    normalized_weights = weights / total[:, None]
    row_normalized_loss = float(np.mean(logsumexp(score, axis=1) - np.sum(normalized_weights*score, axis=1))/20.0)
    require("unnormalized_reward_weights_distinguish_row_normalization", abs(manual_loss-row_normalized_loss) > 1e-3)
    half_labels = 0.5*labels
    half_value, half_gradient = csce_operators.csce_value_gradient(coefficient, phi, half_labels, 0.0)
    close("reward_spread_scales_data_loss", half_value, 0.5*base_value)
    close("reward_spread_scales_data_gradient", half_gradient, 0.5*base_gradient)
    constant_coefficient = np.zeros((17, 20))
    constant_coefficient[0] = rng.normal(size=20)
    penalized_value, penalized_gradient = csce_operators.csce_value_gradient(constant_coefficient, phi, labels, regularization)
    free_value, free_gradient = csce_operators.csce_value_gradient(constant_coefficient, phi, labels, 0.0)
    close("constant_coefficient_is_penalized_value", penalized_value-free_value, regularization*np.sum(constant_coefficient[0]**2))
    close("constant_coefficient_is_penalized_gradient", penalized_gradient[0]-free_gradient[0], 2.0*regularization*constant_coefficient[0])

    fit_diagnostics = []
    reference_heads = {}
    for family in ("B", "C", "D"):
        fitted, diagnostic = e9_heads.fit_head(family, phi, labels, 1.0)
        require("fit_" + family + "_accepted", diagnostic["accepted"])
        require("fit_" + family + "_float64", fitted.dtype == np.dtype("float64"))
        require("fit_" + family + "_costs_recorded", diagnostic["cpu_seconds"] >= 0.0 and diagnostic["wall_seconds"] >= 0.0 and diagnostic["peak_ram_bytes"] > 0)
        rediagnosed = e9_heads.diagnose_head(family, fitted, phi, labels, 1.0)
        close("fit_" + family + "_diagnostic_reconstruction", [diagnostic[key] for key in ("objective", "data_loss", "regularization_term", "coefficient_norm", "gradient_infinity_norm")], [rediagnosed[key] for key in ("objective", "data_loss", "regularization_term", "coefficient_norm", "gradient_infinity_norm")])
        fit_diagnostics.append(diagnostic)
        reference_heads[family] = fitted
    close("C_D_identical_kernel_and_labels", reference_heads["C"], reference_heads["D"], 0.0)
    for family in ("C", "D"):
        equal_head, equal_diag = e9_heads.fit_head(family, phi, all_equal, 1.0)
        close("equal_reward_" + family + "_zero_solution", equal_head, np.zeros((17,20)), 0.0)
        require("equal_reward_" + family + "_accepted", equal_diag["accepted"])
        fit_diagnostics.append(equal_diag)

    for mult in (0.1, 1.0, 10.0):
        fitted, diagnostic = e9_heads.fit_head("B", phi, labels, mult)
        u, singular, vt = np.linalg.svd(phi, full_matrices=False)
        historical = (vt.T * (singular/(singular**2 + 100.0*mult/3.0))) @ (u.T @ labels)
        close("ridge_matches_original_svd_mult_" + str(mult), fitted, historical)
        require("ridge_stationarity_mult_" + str(mult), diagnostic["accepted"])
        fit_diagnostics.append(diagnostic)

    repeated_rewards = rng.integers(0,2,size=(len(phi),3,20)).astype(np.float64)
    repeated_means = repeated_rewards.mean(axis=1)
    mean_mse = np.mean((score-repeated_means)**2)
    realized_mse = np.mean((score[:,None,:]-repeated_rewards)**2)
    within_question_variance = np.mean((repeated_rewards-repeated_means[:,None,:])**2)
    close("realized_three_repeat_mse_mapping", realized_mse, mean_mse+within_question_variance)
    repeated_closed = np.linalg.solve(3.0*phi.T@phi + 100.0*np.eye(17), 3.0*phi.T@repeated_means)
    mean_closed, repeated_diag = e9_heads.fit_head("B", phi, repeated_means, 1.0)
    close("original_three_repeat_ridge_to_mean_label_mapping", repeated_closed, mean_closed)
    fit_diagnostics.append(repeated_diag)

    task = np.array(["gsm8k"]*7 + ["bbh"]*(len(phi)-7))
    model_aliases = tuple("frozen_model_" + str(index) for index in range(6))
    keys = tuple(tuple(sorted(members)) for members in itertools.combinations(model_aliases, 3))
    permutation = rng.permutation(20)
    ties = np.zeros((len(phi), 20))
    ties[0, [3, 8, 17]] = 2.0
    original = e9_heads.routed_accuracy(ties, repeated_rewards, task, keys)
    permuted = e9_heads.routed_accuracy(ties[:,permutation], repeated_rewards[:,:,permutation], task, tuple(keys[index] for index in permutation))
    original_semantics = [keys[index] for index in original["selected_actions"]]
    permuted_keys = tuple(keys[index] for index in permutation)
    permuted_semantics = [permuted_keys[index] for index in permuted["selected_actions"]]
    require("exact_ties_preserve_semantic_selection_after_relabel", original_semantics == permuted_semantics)
    require("exact_zero_ties_choose_frozen_alias_minimum", all(key == min(keys) for key in original_semantics[1:]))
    close("relabel_preserves_realized_rewards", original["per_question_accuracy"], permuted["per_question_accuracy"], 0.0)
    close("relabel_preserves_balanced_accuracy", original["balanced_accuracy"], permuted["balanced_accuracy"], 0.0)
    per_question = np.array(original["per_question_accuracy"])
    close("gsm8k_bbh_exact_50_50_question_weighting", original["balanced_accuracy"], 0.5*per_question[:7].mean()+0.5*per_question[7:].mean())
    close("BBH_question_mean", original["task_accuracy"]["bbh"], per_question[7:].mean())
    means_result = e9_heads.routed_accuracy(ties, repeated_means, task, keys)
    close("repeat_and_mean_reward_evaluation_agree", means_result["per_question_accuracy"], original["per_question_accuracy"], 0.0)
    for family in ("B", "C", "D"):
        relabeled_head, relabeled_diag = e9_heads.fit_head(family, phi, labels[:,permutation], 1.0)
        close("fit_relabel_equivariance_" + family, relabeled_head, reference_heads[family][:,permutation], 1e-9)
        require("fit_relabel_stationarity_" + family, relabeled_diag["accepted"])
        fit_diagnostics.append(relabeled_diag)

    real_minimize = e9_heads.minimize
    try:
        e9_heads.minimize = lambda *args, **kwargs: SimpleNamespace(x=coefficient.ravel(),success=True,status=0,message="Synthetic false-positive optimizer success",nit=1,nfev=1)
        _, false_success = e9_heads.fit_head("C", phi, labels, 1.0)
    finally:
        e9_heads.minimize = real_minimize
    require("optimizer_success_alone_rejected_by_actual_gradient", false_success["success"] and not false_success["gradient_tolerance_met"] and not false_success["accepted"])

    history = {"ids": np.array(["history_a", "history_b"]), "global_indices": np.array([0,1])}
    evaluation = {"ids": np.array(["held_c"]), "global_indices": np.array([2])}
    require("evaluation_guard_accepts_disjoint_feature_package", e9_heads.validate_training_inputs(history, evaluation))
    rejects("evaluation_guard_rejects_reward", lambda: e9_heads.validate_training_inputs(history, {**evaluation, "reward": np.zeros((1,20))}))
    rejects("evaluation_guard_rejects_Y", lambda: e9_heads.validate_training_inputs(history, {**evaluation, "Y": np.zeros((1,3,20))}))
    rejects("evaluation_guard_rejects_identity_overlap", lambda: e9_heads.validate_training_inputs(history, {"ids": np.array(["history_a"])}))
    rejects("evaluation_guard_rejects_global_index_overlap", lambda: e9_heads.validate_training_inputs(history, {**evaluation, "global_indices": np.array([1])}))
    rejects("training_rejects_wrong_action_count", lambda: e9_heads.fit_head("B", phi, labels[:,:19], 1.0))
    rejects("training_rejects_wrong_feature_count", lambda: e9_heads.fit_head("B", phi[:,:16], labels, 1.0))
    rejects("training_rejects_nonfinite_labels", lambda: e9_heads.fit_head("C", phi, np.full_like(labels,np.nan), 1.0))
    rejects("training_rejects_out_of_range_labels", lambda: e9_heads.fit_head("C", phi, labels+1.0, 1.0))
    rejects("training_rejects_invalid_multiplier", lambda: e9_heads.fit_head("B", phi, labels, 0.0))
    rejects("training_rejects_invalid_family", lambda: e9_heads.fit_head("A", phi, labels, 1.0))
    rejects("routing_rejects_missing_task", lambda: e9_heads.routed_accuracy(ties, repeated_means, np.array(["bbh"]*len(phi)), keys))
    rejects("routing_rejects_duplicate_semantic_keys", lambda: e9_heads.routed_accuracy(ties, repeated_means, task, (keys[0],)*20))
    return {
        "status": "PASS",
        "scope": "Synthetic data only. No historical rewards or evaluation artifacts were loaded.",
        "random_seed": 2026111409,
        "checks": checks,
        "check_count": len(checks),
        "scalar_assertions": int(scalar_assertions),
        "fit_diagnostics": fit_diagnostics,
        "actual_gradient_rejection_probe": false_success,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(e9_heads.__file__), Path(csce_operators.__file__), Path(__file__))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE.parent / "public_qualification" / "E9_HEADS_QUALIFICATION.json")
    arguments = parser.parse_args()
    result = qualify()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "scalar_assertions": result["scalar_assertions"], "output": str(arguments.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
