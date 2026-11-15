"""Known-truth E17 qualifications with fixed, explicitly enumerated rewards.

The synthetic reward blocks are deterministic finite representations of known
distributions. No language-model output, posterior sampler, real reward cache,
or confidence-interval routine is used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from objectives import fit_expected_spo, fit_spo, objective_value, select_actions


REGULARIZATION = .2


def _service_metrics(scores, mean, hindsight_oracle):
    """Separate Bayes-mean regret from the positive hindsight-oracle gap."""
    rows = {}
    for tolerance in (0., 1e-8):
        action = int(select_actions(scores[None, :], tolerance=tolerance)[0])
        value = float(mean[action])
        rows[str(tolerance)] = dict(
            chosen_physical_action=action,
            expected_reward=value,
            mean_regret=float(mean.max() - value),
            expected_hindsight_regret=float(hindsight_oracle - value),
        )
    return rows


def _case_record(name, support, counts, rule, expected_gap, mean_gap):
    support = np.asarray(support, dtype=float)
    counts = np.asarray(counts, dtype=int)
    probabilities = counts / counts.sum()
    draws = np.repeat(support, counts, axis=0)[None, :, :]
    mean = probabilities @ support
    finite_draw_mean = draws.mean(axis=1)[0]
    hindsight = float(probabilities @ support.max(axis=1))
    phi = np.ones((1, 1))
    before_cpu, before_wall = time.process_time(), time.perf_counter()
    expected, expected_diagnostic = fit_expected_spo(phi, draws, REGULARIZATION, rule)
    averaged, mean_diagnostic = fit_spo(phi, mean[None, :], REGULARIZATION, rule)
    actual_expected_gap = float(expected[0, 0] - expected[0, 1])
    actual_mean_gap = float(averaged[0, 0] - averaged[0, 1])
    if abs(actual_expected_gap - expected_gap) > 1e-6 or abs(actual_mean_gap - mean_gap) > 1e-6:
        raise AssertionError("Known two-action optimum was not recovered")
    # These are known distribution means, not estimated qualities of real data.
    if not np.allclose(finite_draw_mean, mean, rtol=0, atol=1e-15):
        raise AssertionError("The fixed reward block must represent its declared distribution")
    loss_difference_examples = []
    for gap in (-1., 0., 1.):
        coefficient = np.array([[gap / 2, -gap / 2]])
        expected_value = objective_value(coefficient, phi, draws, 0, rule)
        mean_value = objective_value(coefficient, phi, mean[None, :], 0, rule)
        loss_difference_examples.append(dict(
            score_gap=gap, loss_then_mean=expected_value, mean_then_loss=mean_value,
            difference=expected_value - mean_value,
        ))
    analytic_expected = np.array([expected_gap / 2, -expected_gap / 2])
    analytic_averaged = np.array([mean_gap / 2, -mean_gap / 2])
    return dict(
        name=name, support=support.tolist(), probabilities=probabilities.tolist(),
        fixed_draw_counts=counts.tolist(), questions=1, draws=int(counts.sum()),
        tie_rule=rule, regularization=REGULARIZATION,
        known_mean=mean.tolist(), working_mean=mean.tolist(),
        working_mean_error=0.0,
        fixed_draw_mean_error_max_abs=float(np.abs(finite_draw_mean - mean).max()),
        working_mean_scope="Known synthetic target mean; no real posterior-calibration claim",
        best_mean_action=int(mean.argmax()), optimal_expected_reward=float(mean.max()),
        expected_hindsight_oracle=hindsight,
        irreducible_hindsight_gap=float(hindsight - mean.max()),
        loss_then_mean=dict(
            analytic_optimal_score_gap=expected_gap,
            numerical_score_gap=actual_expected_gap,
            analytic_service=_service_metrics(analytic_expected, mean, hindsight),
            numerical_service=_service_metrics(expected[0], mean, hindsight),
            fit=expected_diagnostic,
        ),
        mean_then_loss=dict(
            analytic_optimal_score_gap=mean_gap,
            numerical_score_gap=actual_mean_gap,
            analytic_service=_service_metrics(analytic_averaged, mean, hindsight),
            numerical_service=_service_metrics(averaged[0], mean, hindsight),
            fit=mean_diagnostic,
        ),
        total_loss_difference_has_no_universal_sign=loss_difference_examples,
        cpu_seconds=time.process_time() - before_cpu,
        wall_seconds=time.perf_counter() - before_wall,
    )


def _qualify_known_risks(case_one, case_two):
    """Compare independently expanded piecewise risks with the direct operator."""
    grid = np.linspace(-2., 2., 4001)
    risks = []
    for case in (case_one, case_two):
        support = np.asarray(case["support"])
        counts = np.asarray(case["fixed_draw_counts"])
        samples = np.repeat(support, counts, axis=0)[None, :, :]
        actual = np.array([
            objective_value(np.array([[gap / 2, -gap / 2]]), [[1.]], samples, 0, case["tie_rule"])
            for gap in grid
        ])
        risks.append(actual)
    first_formula = .9 * np.maximum(0., .1 - 2 * grid) + .1 * np.maximum(0., 1 + 2 * grid)
    second_formula = np.select(
        [grid < -.5, grid < 0, grid <= .5],
        [.1 - .9 * grid, .3 - .5 * grid, .3 + .9 * grid],
        default=.2 + 1.1 * grid,
    )
    errors = [float(np.abs(risks[0] - first_formula).max()),
              float(np.abs(risks[1] - second_formula).max())]
    if max(errors) > 1e-12:
        raise AssertionError("Expanded known population risks differ from the SPO+ operator")
    minima = [float(grid[np.argmin(risk)]) for risk in risks]
    if not np.allclose(minima, [.05, 0.], rtol=0, atol=1e-12):
        raise AssertionError("Known unregularized risk minima differ")
    return dict(grid_points=len(grid), score_gap_range=[-2., 2.],
                direct_piecewise_max_abs_errors=errors,
                unregularized_minimum_score_gaps=minima)


def _qualify_all_ties():
    """For an all-tied-only dataset, J*=0 and positive L2 uniquely gives W=0."""
    phi = np.array([[1., -1.], [1., 0.], [1., 1.]])
    constants = np.array([[0., .5], [.25, .75], [1., .5]])
    records = []
    for actions in (2, 20):
        samples = np.repeat(constants[:, :, None], actions, axis=2)
        for rule in ("lexicographic", "uniform", "semantic"):
            identities = np.arange(actions)[::-1]
            coefficient, diagnostic = fit_expected_spo(phi, samples, .5, rule, identities)
            zero_value = objective_value(np.zeros_like(coefficient), phi, samples, .5, rule, identities)
            penalty = float(.5 * np.square(coefficient).sum())
            excess = float(diagnostic["objective"])
            if abs(zero_value) > 1e-12 or excess < -1e-12 or excess > 1e-7:
                raise AssertionError("Known zero optimum of an all-tied-only dataset failed")
            if penalty > excess + 1e-12:
                raise AssertionError("The positive-regularization norm bound failed")
            records.append(dict(
                actions=actions, questions=3, draws=2, tie_rule=rule,
                known_global_optimum=0., zero_coefficient_objective=zero_value,
                numerical_objective_gap=excess, regularization_penalty=penalty,
                coefficient_norm=float(np.linalg.norm(coefficient)),
                norm_bound_from_objective=float(np.sqrt(max(excess, 0.) / .5)),
                floating_point_allowance=1e-12, fit=diagnostic,
            ))
    return records


def _qualify_champion_identity():
    """Changing a true champion changes a mixed-data training objective term."""
    scores = np.array([.7, -.2, .4])
    reward = np.array([1., 1., .1])
    maximum = float(np.max(2 * scores - reward))
    loss_champion0 = maximum - 2 * scores[0] + reward.max()
    loss_champion1 = maximum - 2 * scores[1] + reward.max()
    difference = float(loss_champion1 - loss_champion0)
    identity = float(2 * (scores[0] - scores[1]))
    if abs(difference - identity) > 1e-12:
        raise AssertionError("True-tie champion loss-difference identity failed")
    return dict(loss_champion0=float(loss_champion0), loss_champion1=float(loss_champion1),
                observed_difference=difference, exact_difference_formula=identity,
                scope="A single tied row in a shared mixed-data objective; not a nonzero all-tied-only optimum")


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Synthetic qualification output already exists")
    before_cpu, before_wall = time.process_time(), time.perf_counter()
    case_one = _case_record(
        "same_known_mean_different_operator", [[.6, .5], [0., 1.]], [9, 1],
        "lexicographic", .05, -.005,
    )
    case_two = _case_record(
        "symmetric_true_ties_do_not_imply_mean_consistency", [[1., 0.], [0., 1.], [1., 1.]],
        [1, 2, 7], "uniform", 0., -.05,
    )
    # The symmetric population minimizer has a tied decision set. Its service
    # value depends on the independently specified inference-tie rule.
    case_two["loss_then_mean"]["analytic_optimal_action_set"] = [0, 1]
    case_two["loss_then_mean"]["uniform_inference_tie_expected_reward"] = .85
    case_two["loss_then_mean"]["uniform_inference_tie_mean_regret"] = .05
    case_two["loss_then_mean"]["uniform_inference_tie_hindsight_regret"] = .15
    qualification = _qualify_known_risks(case_one, case_two)
    all_ties = _qualify_all_ties()
    result = dict(
        schema="E17_fixed_known_truth_operator_qualification_v1", status="PASS",
        known_truth_cases=[case_one, case_two], piecewise_risk_qualification=qualification,
        all_tied_only_positive_regularization=all_ties,
        champion_identity_qualification=_qualify_champion_identity(),
        synthetic_qp_fits=10, real_model_fits=0, posterior_sampler_calls=0,
        new_language_model_calls=0, confidence_intervals=0, bootstrap_runs=0,
        true_reward_cache_reads=0,
        objective_source_sha256=hashlib.sha256(Path(__file__).with_name("objectives.py").read_bytes()).hexdigest(),
        synthetic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_seconds=time.process_time() - before_cpu,
        wall_seconds=time.perf_counter() - before_wall,
        scope="Known synthetic distributions and numerical implementation only; no empirical MAB performance claim",
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("synthetic_results.json"))
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps(dict(status=result["status"], synthetic_qp_fits=result["synthetic_qp_fits"],
                          wall_seconds=result["wall_seconds"], output=str(arguments.output)), ensure_ascii=False))
