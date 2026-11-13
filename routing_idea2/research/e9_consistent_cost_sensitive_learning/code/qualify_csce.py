"""Deterministic, synthetic-only qualification for the E9 CSCE operator."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

import csce_operators as operators


def qualify():
    rng = np.random.default_rng(20261114)
    action_keys = tuple(itertools.combinations(tuple(f"model_{i}" for i in range(6)), 3))
    phi = np.column_stack((np.ones(12), rng.normal(size=(12, 16))))
    rewards = rng.uniform(0.05, 0.95, size=(12, 20))
    coefficient = rng.normal(scale=0.2, size=(17, 20))
    regularization = 0.015
    checks = {}

    def record(name, passed, **details):
        checks[name] = dict(passed=bool(passed), **details)

    weights, total = operators.reward_weights(rewards)
    costs = 1 - rewards
    cost_weights = costs.max(axis=1, keepdims=True) - costs
    transform_error = float(np.max(np.abs(weights - cost_weights)))
    record("published_cost_to_reward_transform", transform_error < 1e-15,
           maximum_absolute_error=transform_error)

    value, gradient = operators.csce_value_gradient(coefficient, phi, rewards, regularization)
    scores = phi @ coefficient
    log_probabilities = scores - logsumexp(scores, axis=1, keepdims=True)
    reference_value = -np.sum(weights * log_probabilities) / (len(phi) * 20)
    reference_value += regularization * np.sum(coefficient ** 2)
    reference_gradient = phi.T @ (total[:, None] * softmax(scores, axis=1) - weights)
    reference_gradient /= len(phi) * 20
    reference_gradient += 2 * regularization * coefficient
    reference_error = max(float(abs(value - reference_value)),
                          float(np.max(np.abs(gradient - reference_gradient))))
    record("unnormalized_reference_loss_and_gradient", reference_error < 1e-13,
           maximum_absolute_error=reference_error, normalization="n_original_questions*20")

    step = 1e-6
    numerical_gradient = np.empty_like(coefficient)
    for index in np.ndindex(coefficient.shape):
        plus, minus = coefficient.copy(), coefficient.copy()
        plus[index] += step
        minus[index] -= step
        plus_value, _ = operators.csce_value_gradient(plus, phi, rewards, regularization)
        minus_value, _ = operators.csce_value_gradient(minus, phi, rewards, regularization)
        numerical_gradient[index] = (plus_value - minus_value) / (2 * step)
    gradient_error = float(np.max(np.abs(gradient - numerical_gradient)))
    record("finite_difference_gradient", gradient_error < 1e-8,
           maximum_absolute_error=gradient_error, coordinate_count=coefficient.size, step=step)

    constant_rewards = np.full((len(phi), 20), 0.4)
    constant_value, constant_gradient = operators.csce_value_gradient(
        coefficient, phi, constant_rewards, 0)
    original_value, original_gradient = operators.csce_value_gradient(coefficient, phi, rewards, 0)
    appended_value, appended_gradient = operators.csce_value_gradient(
        coefficient, np.vstack((phi, phi)), np.vstack((rewards, constant_rewards)), 0)
    zero_error = max(abs(constant_value), float(np.max(np.abs(constant_gradient))),
                     abs(appended_value - original_value / 2),
                     float(np.max(np.abs(appended_gradient - original_gradient / 2))))
    record("zero_weight_rows_and_original_question_denominator", zero_error < 1e-14,
           maximum_absolute_error=float(zero_error), original_rows=12, appended_rows=24)

    permutation = rng.permutation(20)
    permuted_keys = tuple(action_keys[index] for index in permutation)
    permuted_value, permuted_gradient = operators.csce_value_gradient(
        coefficient[:, permutation], phi, rewards[:, permutation], regularization)
    permutation_error = max(abs(value - permuted_value),
                            float(np.max(np.abs(gradient[:, permutation] - permuted_gradient))))
    original_actions = operators.semantic_argmax(scores, action_keys)
    permuted_actions = operators.semantic_argmax(scores[:, permutation], permuted_keys)
    decisions_match = np.array_equal(original_actions, permutation[permuted_actions])
    record("action_permutation_loss_gradient_and_semantic_decisions",
           permutation_error < 1e-13 and decisions_match,
           maximum_absolute_error=float(permutation_error), semantic_decisions_match=bool(decisions_match))

    tie_scores = np.full((3, 20), -1.0)
    tie_scores[0, [1, 10]] = 2.0
    tie_scores[1, :] = 0.0
    tie_scores[2, [2, 17]] = 3.0
    expected_tie_indices = np.array([1, 0, 2])
    tie_actions = operators.semantic_argmax(tie_scores, action_keys)
    permuted_tie_actions = operators.semantic_argmax(tie_scores[:, permutation], permuted_keys)
    tie_passed = np.array_equal(tie_actions, expected_tie_indices)
    tie_passed &= np.array_equal(permutation[permuted_tie_actions], expected_tie_indices)
    record("exact_ties_use_canonical_three_member_semantic_keys", tie_passed,
           selected_semantic_keys=[list(action_keys[index]) for index in tie_actions])

    fitted, fit_diagnostics = operators.fit_csce(phi, rewards, regularization)
    permuted_fit, permuted_fit_diagnostics = operators.fit_csce(
        phi, rewards[:, permutation], regularization)
    fit_error = float(np.max(np.abs(fitted[:, permutation] - permuted_fit)))
    fitted_actions = operators.semantic_argmax(phi @ fitted, action_keys)
    permuted_fitted_actions = operators.semantic_argmax(phi @ permuted_fit, permuted_keys)
    refit_decisions_match = np.array_equal(fitted_actions, permutation[permuted_fitted_actions])
    record("action_permutation_refit", fit_error < 1e-7 and refit_decisions_match,
           maximum_coefficient_absolute_error=fit_error,
           semantic_decisions_match=bool(refit_decisions_match),
           original_gradient_infinity_norm=fit_diagnostics["gradient_infinity_norm"],
           permuted_gradient_infinity_norm=permuted_fit_diagnostics["gradient_infinity_norm"])

    history = {"ids": np.array(["history_0", "history_1"]), "global_indices": np.array([0, 1])}
    evaluation = {"ids": np.array(["evaluation_0"]), "global_indices": np.array([2]), "features": np.zeros((1, 3))}
    valid_accepted = operators.validate_training_inputs(history, evaluation)
    outcome_rejections = {}
    for name in ("Y", "y", "reward", "rewards", "posterior_reward"):
        package = dict(evaluation, **{name: np.zeros((1, 20))})
        try:
            operators.validate_training_inputs(history, package)
        except ValueError:
            outcome_rejections[name] = True
        else:
            outcome_rejections[name] = False
    record("evaluation_outcome_keys_rejected", valid_accepted and all(outcome_rejections.values()),
           valid_outcome_free_package_accepted=bool(valid_accepted), rejected_keys=outcome_rejections)

    identity_rejections = {}
    identity_cases = {
        "same_question_id": dict(evaluation, ids=np.array(["history_1"])),
        "same_global_index": dict(evaluation, global_indices=np.array([1])),
        "missing_question_id": {"features": np.zeros((1, 3))},
    }
    for name, package in identity_cases.items():
        try:
            operators.validate_training_inputs(history, package)
        except ValueError:
            identity_rejections[name] = True
        else:
            identity_rejections[name] = False
    record("question_identity_overlap_rejected", all(identity_rejections.values()),
           rejection_cases=identity_rejections)

    two_action_rewards = np.array([[0.6, 0.5]] * 9 + [[0.0, 1.0]])
    two_action_weights, two_action_total = operators.reward_weights(two_action_rewards)
    reward_mean = two_action_rewards.mean(axis=0)
    expected_weight = two_action_weights.mean(axis=0)
    normalized_weight_mean = (two_action_weights / two_action_total[:, None]).mean(axis=0)
    conditional_optimum = np.log(expected_weight)[None, :]
    _, conditional_gradient = operators.csce_value_gradient(
        conditional_optimum, np.ones((10, 1)), two_action_rewards, 0)
    counterexample_passed = int(np.argmax(reward_mean)) == 1
    counterexample_passed &= int(np.argmax(expected_weight)) == 1
    counterexample_passed &= int(np.argmax(normalized_weight_mean)) == 0
    counterexample_passed &= float(np.max(np.abs(conditional_gradient))) < 1e-14
    record("per_question_normalization_changes_population_decision", counterexample_passed,
           reward_mean=reward_mean.tolist(), expected_unnormalized_weight=expected_weight.tolist(),
           expected_row_normalized_weight=normalized_weight_mean.tolist(),
           correct_action_one_based=2, row_normalized_action_one_based=1)

    ridge_regularization = 0.006
    ridge_operator = np.linalg.solve(
        phi.T @ phi + len(phi) * 20 * ridge_regularization * np.eye(phi.shape[1]), phi.T)
    raw_ridge = ridge_operator @ rewards
    centered_rewards = rewards - rewards.mean(axis=1, keepdims=True)
    centered_ridge = ridge_operator @ centered_rewards
    centered_reference = raw_ridge - raw_ridge.mean(axis=1, keepdims=True)
    baseline_rewards = rewards - rewards[:, [0]]
    baseline_ridge = ridge_operator @ baseline_rewards
    baseline_reference = raw_ridge - (ridge_operator @ rewards[:, [0]])
    ridge_error = max(float(np.max(np.abs(centered_ridge - centered_reference))),
                      float(np.max(np.abs(baseline_ridge - baseline_reference))))
    centered_actions = operators.semantic_argmax(phi @ centered_ridge, action_keys)
    baseline_actions = operators.semantic_argmax(phi @ baseline_ridge, action_keys)
    raw_actions = operators.semantic_argmax(phi @ raw_ridge, action_keys)
    ridge_decisions_match = np.array_equal(raw_actions, centered_actions)
    ridge_decisions_match &= np.array_equal(raw_actions, baseline_actions)
    record("ridge_centering_and_shared_baseline_only_change_common_offset",
           ridge_error < 1e-13 and ridge_decisions_match,
           maximum_absolute_error=ridge_error, semantic_decisions_match=bool(ridge_decisions_match))

    offset_coefficient = coefficient.copy()
    offset_coefficient[0, :] += 1e12
    shifted_value, shifted_gradient = operators.csce_value_gradient(offset_coefficient, phi, rewards, 0)
    offset_value_error = abs(original_value - shifted_value)
    offset_gradient_error = float(np.max(np.abs(original_gradient - shifted_gradient)))
    record("large_common_logit_offset_stability",
           offset_value_error < 1e-5 and offset_gradient_error < 1e-5,
           absolute_value_error=float(offset_value_error),
           maximum_gradient_absolute_error=offset_gradient_error, common_offset=1e12,
           tolerance_note="Input-score rounding at a 1e12 offset is retained in the comparison.")

    equal_scores_coefficient = np.zeros_like(coefficient)
    equal_scores_shifted_coefficient = equal_scores_coefficient.copy()
    equal_scores_shifted_coefficient[0, :] = 1e12
    equal_value, equal_gradient = operators.csce_value_gradient(
        equal_scores_coefficient, phi, rewards, 0)
    equal_shifted_value, equal_shifted_gradient = operators.csce_value_gradient(
        equal_scores_shifted_coefficient, phi, rewards, 0)
    equal_score_error = max(abs(equal_value - equal_shifted_value),
                            float(np.max(np.abs(equal_gradient - equal_shifted_gradient))))
    record("exact_equal_logits_common_offset", equal_score_error < 1e-13,
           maximum_absolute_error=float(equal_score_error), common_offset=1e12,
           tolerance_note="All logits are exactly equal and representable before and after the shift.")

    boundary_values = []
    for logit in (0.0, 2.0, 6.0, 20.0):
        boundary_value, _ = operators.csce_value_gradient(
            np.array([[logit, 0.0]]), np.ones((1, 1)), np.array([[1.0, 0.0]]), 0)
        boundary_values.append(boundary_value)
    boundary_passed = all(left > right > 0 for left, right in zip(boundary_values, boundary_values[1:]))
    record("finite_logit_boundary_approaches_infimum", boundary_passed,
           logits=[0.0, 2.0, 6.0, 20.0], objective_values=boundary_values)

    source_path = Path(operators.__file__).resolve()
    return dict(
        status="PASS" if all(check["passed"] for check in checks.values()) else "FAIL",
        scope="deterministic synthetic operator qualification",
        source_file=source_path.name,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        seed=20261114,
        check_count=len(checks),
        passed_count=sum(check["passed"] for check in checks.values()),
        formal_output_count=20,
        production_dimension_enforcement="Required in the experiment runner. Numerical kernel also supports synthetic action counts.",
        checks=checks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("CSCE_QUALIFICATION.json"))
    args = parser.parse_args()
    report = qualify()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], check_count=report["check_count"],
                          passed_count=report["passed_count"], output=str(args.output)), ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
