"""Independent, outcome-free qualifications for the E6 statistical operators."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np


SEED = 20261112


def maximum_error(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def weighted_classification_check(rng):
    rows = []
    for reward in (-1.0, 0.0, 1.0, -0.41, 0.27):
        for gate in (0.0, 1.0):
            lhs = abs(reward) * (gate != (reward > 0))
            rhs = max(reward, 0) - gate * reward
            rows.append(abs(lhs - rhs))
    errors = []
    for _ in range(1000):
        rewards = rng.choice([-1.0, 0.0, 1.0], size=(24, 3))
        gates = rng.integers(0, 2, size=(24, 1))
        lhs = np.mean(np.abs(rewards) * (gates != (rewards > 0)))
        rhs = np.mean(np.maximum(rewards, 0)) - np.mean(gates * rewards)
        errors.append(abs(lhs - rhs))
    return {
        "pointwise_cases": len(rows),
        "pointwise_max_error": max(rows),
        "random_joint_distributions": len(errors),
        "expectation_max_error": max(errors),
        "class_weight_counterexample": {
            "positive_mass": 0.04,
            "negative_mass": 0.06,
            "unweighted_gain": -0.02,
            "weighted_positive_mass_at_class_weight_2": 0.08,
            "weighted_negative_mass": 0.06,
            "decision_reversed": True,
        },
    }


def iid_covariance_check(rng):
    weights = np.asarray([0.2, 0.3, 0.5])
    probabilities = np.asarray([0.23, 0.58, 0.81])
    # Rows hold the conditional masses for D=-1,0,+1.
    reward_masses = np.asarray([[0.12, 0.58, 0.30], [0.40, 0.50, 0.10], [0.08, 0.80, 0.12]])
    gates = np.asarray([[0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
    repeat_count = 3
    conditional_mean = reward_masses[:, 2] - reward_masses[:, 0]
    conditional_second = reward_masses[:, 2] + reward_masses[:, 0]
    a = conditional_mean**2 + (conditional_second - conditional_mean**2) / repeat_count
    records, masses = [], []
    for h, outcomes in itertools.product(range(3), itertools.product(range(3), repeat=repeat_count)):
        outcome = float(np.mean(np.asarray(outcomes) - 1))
        mass = weights[h] * float(np.prod(reward_masses[h, list(outcomes)]))
        for acquired in (0, 1):
            acquisition_mass = probabilities[h] if acquired else 1 - probabilities[h]
            records.append(acquired * gates[:, h] * outcome / probabilities[h])
            masses.append(mass * acquisition_mass)
    records = np.asarray(records)
    masses = np.asarray(masses)
    mean = masses @ records
    covariance = (records.T * masses) @ records - np.outer(mean, mean)
    expected_mean = gates @ (weights * conditional_mean)
    expected_covariance = (gates * (weights * a / probabilities)) @ gates.T - np.outer(expected_mean, expected_mean)
    pair_errors = []
    for j, k in itertools.combinations(range(len(gates)), 2):
        exact = covariance[j, j] + covariance[k, k] - 2 * covariance[j, k]
        formula = float(np.sum(weights * (gates[j] - gates[k]) ** 2 * a / probabilities) - (mean[j] - mean[k]) ** 2)
        pair_errors.append(abs(exact - formula))
    draw_count = 200000
    picked = rng.choice(len(masses), size=draw_count, p=masses)
    observed_variance = np.var(records[picked, 2] - records[picked, 3], ddof=1)
    exact_variance = covariance[2, 2] + covariance[3, 3] - 2 * covariance[2, 3]
    return {
        "enumerated_joint_atoms": len(masses),
        "probability_sum_error": abs(float(masses.sum()) - 1),
        "mean_max_error": maximum_error(mean, expected_mean),
        "covariance_max_error": maximum_error(covariance, expected_covariance),
        "pair_variance_max_error": max(pair_errors),
        "conditional_a": a.tolist(),
        "mc_repetitions": draw_count,
        "mc_pair_variance": float(observed_variance),
        "exact_pair_variance": float(exact_variance),
        "mc_relative_error": abs(float(observed_variance / exact_variance) - 1),
        "requires_iid_questions": True,
        "requires_gate_constant_within_stratum_for_factored_formula": True,
        "repeat_independence_required_only_for_a_mu_rho_formula": True,
    }


def finite_pool_check():
    rewards = np.asarray([-1, 1, 0, 1, -1, 1 / 3, -2 / 3, 0], dtype=float)
    gates = np.asarray([[0] * 8, [1] * 8, [0, 0, 1, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0]], dtype=float)
    pilot = np.asarray([0, 1])
    rest = np.arange(2, 8)
    p = np.asarray([0.12, 0.25, 0.41, 0.56, 0.70, 0.92])
    records, masses = [], []
    known = gates[:, pilot] @ rewards[pilot]
    for acquisition in itertools.product((0, 1), repeat=len(rest)):
        acquisition = np.asarray(acquisition)
        mass = float(np.prod(np.where(acquisition, p, 1 - p)))
        estimate = (known + gates[:, rest] @ (acquisition * rewards[rest] / p)) / len(rewards)
        records.append(estimate)
        masses.append(mass)
    records, masses = np.asarray(records), np.asarray(masses)
    mean = masses @ records
    covariance = (records.T * masses) @ records - np.outer(mean, mean)
    true_value = gates @ rewards / len(rewards)
    coefficient = (1 - p) / p * rewards[rest] ** 2 / len(rewards) ** 2
    expected_covariance = (gates[:, rest] * coefficient) @ gates[:, rest].T
    errors = []
    for j, k in itertools.combinations(range(len(gates)), 2):
        exact = covariance[j, j] + covariance[k, k] - 2 * covariance[j, k]
        formula = float(np.sum(coefficient * (gates[j, rest] - gates[k, rest]) ** 2))
        errors.append(abs(exact - formula))
    return {
        "pool_size": len(rewards),
        "pilot_size": len(pilot),
        "enumerated_masks": len(masses),
        "mean_max_error": maximum_error(mean, true_value),
        "covariance_max_error": maximum_error(covariance, expected_covariance),
        "pair_variance_max_error": max(errors),
        "conditions": "fixed pool, fixed pilot outcomes, frozen gates and p, independent remaining Bernoulli draws",
    }


def objective_degeneracy_check(rng):
    population_errors, finite_pool_errors, gap_differences = [], [], []
    for _ in range(1000):
        gates = np.concatenate([np.zeros((1, 9)), np.ones((1, 9)), rng.integers(0, 2, size=(5, 9))])
        a = rng.uniform(0, 1, size=9)
        w = rng.dirichlet(np.ones(9))
        p = rng.uniform(0.05, 1, size=9)
        pairs = np.asarray([(gates[j] - gates[k]) ** 2 for j, k in itertools.combinations(range(len(gates)), 2)])
        marginal = float(np.max(gates @ (w * a / p)))
        contrast = float(np.max(pairs @ (w * a / p)))
        population_errors.append(abs(marginal - contrast))
        fp_marginal = float(np.max(gates @ (a * (1 - p) / p)))
        fp_contrast = float(np.max(pairs @ (a * (1 - p) / p)))
        finite_pool_errors.append(abs(fp_marginal - fp_contrast))
        values = gates @ rng.uniform(-0.1, 0.1, size=9)
        gaps = np.asarray([max(abs(values[j] - values[k]), 0.01) for j, k in itertools.combinations(range(len(gates)), 2)])
        normalized = float(np.max((pairs @ (w * a / p)) / gaps**2))
        gap_differences.append(normalized - contrast)
    # An explicit common-mode covariance example, independent reward estimation overcounts it.
    covariance_example = {
        "g_j": [1, 1],
        "g_k": [1, 0],
        "weights": [0.5, 0.5],
        "a": [1.0, 0.01],
        "p": [0.5, 0.5],
        "correct_pair_second_moment": 0.01,
        "incorrect_sum_of_marginal_second_moments": 2.01,
    }
    return {
        "random_non_nested_gate_families": 1000,
        "population_objective_max_error": max(population_errors),
        "finite_pool_objective_max_error": max(finite_pool_errors),
        "gap_normalized_numerically_distinct_cases": sum(abs(value) > 1e-10 for value in gap_differences),
        "equality_requires": "binary gates plus default and all-switch in the family; nonnegative second-moment coefficients",
        "not_a_result_of_nesting": True,
        "covariance_cancellation_example": covariance_example,
    }


def budget_check():
    masks = np.asarray(list(itertools.product((0, 1), repeat=2)))
    p = 0.5
    probabilities = np.full(4, 0.25)
    original_mean = float(probabilities @ masks.sum(axis=1))
    truncated = masks.copy()
    truncated[(masks[:, 0] == 1) & (masks[:, 1] == 1), 1] = 0
    true_inclusion = probabilities @ truncated
    return {
        "independent_expected_gold": original_mean,
        "nominal_gold_cap": 1,
        "probability_above_cap": float(probabilities @ (masks.sum(axis=1) > 1)),
        "true_inclusion_after_stop_at_one": true_inclusion.tolist(),
        "using_original_p_for_second_item_ht_expectation": float(true_inclusion[1] / p),
        "correct_expectation_for_second_item": 1.0,
        "naive_cap_introduces_bias": True,
    }


def main():
    rng = np.random.default_rng(SEED)
    result = {
        "seed": SEED,
        "uses_real_rewards": False,
        "weighted_classification": weighted_classification_check(rng),
        "iid_shared_feedback": iid_covariance_check(rng),
        "finite_pool_shared_feedback": finite_pool_check(),
        "all_switch_degeneracy": objective_degeneracy_check(rng),
        "bernoulli_budget_counterexample": budget_check(),
    }
    numerical_sections = [result["weighted_classification"], result["iid_shared_feedback"], result["finite_pool_shared_feedback"], result["all_switch_degeneracy"]]
    errors = [value for section in numerical_sections for name, value in section.items() if name.endswith("max_error")]
    assert all(value < 1e-12 for value in errors), errors
    assert result["iid_shared_feedback"]["mc_relative_error"] < 0.02
    result["status"] = "PASS"
    result["max_exact_algebra_error"] = max(errors)
    work = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由')) / 'research/e6_contrast_directed_feedback'
    target = work / 'public/THEORY_AND_NUMERICAL_CHECKS.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "max_exact_algebra_error": result["max_exact_algebra_error"], "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
