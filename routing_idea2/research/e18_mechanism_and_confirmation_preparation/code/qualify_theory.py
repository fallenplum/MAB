"""Reward-free numerical checks of the E18 mathematical statements.

This program uses synthetic arrays and closed-form known distributions only.
It does not import E17 training code or open any experimental reward asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def weights(reward, uniform):
    maximum = reward == reward.max()
    if uniform:
        return maximum / maximum.sum()
    return np.eye(len(reward))[np.argmax(reward)]


def loss(score, reward, target_weights):
    return float(np.max(2 * score - reward) - 2 * score @ target_weights + reward.max())


def gradient(score, reward, target_weights):
    chosen = np.eye(len(reward))[np.argmax(2 * score - reward)]
    return 2 * (chosen - target_weights)


def qualify():
    cpu, wall = time.process_time(), time.perf_counter()
    rng = np.random.default_rng(180116)
    errors = {}
    cases = {}

    def close(name, actual, expected, tolerance=1e-11):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        if error > tolerance:
            raise AssertionError(name + " " + str(error))
        errors[name] = max(errors.get(name, 0.0), error)
        cases[name] = cases.get(name, 0) + 1

    for actions in (2, 20):
        for _ in range(50):
            score = rng.normal(size=actions)
            reward = rng.uniform(0, 0.8, actions)
            first, second = rng.choice(actions, 2, replace=False)
            reward[[first, second]] = 1
            e_first, e_second = np.eye(actions)[[first, second]]
            close("champion_loss_difference", loss(score, reward, e_first) - loss(score, reward, e_second),
                  2 * (score[second] - score[first]))
            close("champion_score_gradient", gradient(score, reward, e_first) - gradient(score, reward, e_second),
                  2 * (e_second - e_first))
            phi, future = rng.normal(size=(2, 7))
            grad_first = np.outer(phi, gradient(score, reward, e_first))
            grad_second = np.outer(phi, gradient(score, reward, e_second))
            close("shared_parameter_gradient", grad_first - grad_second, 2 * np.outer(phi, e_second - e_first))
            step = 0.03
            close("one_step_future_score_shift", -step * future @ (grad_first - grad_second),
                  -2 * step * (future @ phi) * (e_second - e_first))
            permutation = rng.permutation(actions)
            uniform = weights(reward, True)
            close("uniform_loss_permutation", loss(score[permutation], reward[permutation], weights(reward[permutation], True)),
                  loss(score, reward, uniform))
            close("uniform_gradient_permutation", gradient(score[permutation], reward[permutation], weights(reward[permutation], True)),
                  gradient(score, reward, uniform)[permutation])

    score, reward = np.array([1.0, 0.0]), np.ones(2)
    numeric_difference = loss(score, reward, weights(reward, False)) - loss(score[::-1], reward[::-1], weights(reward[::-1], False))
    close("numeric_first_noninvariance_example", numeric_difference, -2.0)

    regularization = 0.4
    for actions in (2, 20):
        phi = rng.normal(size=(13, 7))
        for uniform in (False, True):
            for _ in range(10):
                coefficient = rng.normal(size=(7, actions))
                reward = np.ones(actions) * 0.3
                empirical = np.mean([loss(q, reward, weights(reward, uniform)) for q in phi @ coefficient])
                objective = empirical + regularization * np.sum(coefficient ** 2)
                if empirical < -1e-12 or objective <= 0:
                    raise AssertionError("all-tied nonnegative loss and positive L2")
                if regularization * np.sum(coefficient ** 2) > objective + 1e-12:
                    raise AssertionError("coefficient norm objective-gap bound")
                close("pure_tie_zero_feasible", loss(np.zeros(actions), reward, weights(reward, uniform)), 0)

    grid = np.linspace(-2, 2, 4001)
    outcomes = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    probability = np.array([0.1, 0.2, 0.7])
    mean = probability @ outcomes
    close("binary_counterexample_mean", mean, [0.8, 0.9])
    risk = np.array([sum(p * loss(np.array([d, 0]), r, weights(r, True))
                         for p, r in zip(probability, outcomes)) for d in grid])
    piecewise = np.select([grid < -.5, grid < 0, grid <= .5],
                          [.1 - .9 * grid, .3 - .5 * grid, .3 + .9 * grid], default=.2 + 1.1 * grid)
    close("binary_piecewise_risk", risk, piecewise)
    close("binary_optimal_score_gap", grid[np.argmin(risk)], 0)
    close("uniform_prediction_tie_reward", mean.mean(), .85)
    close("uniform_prediction_tie_Bayes_gap", mean.max() - mean.mean(), .05)

    outcomes = np.array([[.6, .5], [0, 1]])
    probability = np.array([.9, .1])
    mean = probability @ outcomes
    close("order_counterexample_known_mean", mean, [.54, .55])
    differences = []
    for d in (-.1, 0, .1):
        q = np.array([d, 0])
        expected_loss = sum(p * loss(q, r, weights(r, False)) for p, r in zip(probability, outcomes))
        mean_loss = loss(q, mean, weights(mean, False))
        differences.append(expected_loss - mean_loss)
    if not (min(differences) < 0 < max(differences)):
        raise AssertionError("full operator difference must have no fixed Jensen ordering")
    # For a fixed difference d, minimum-norm scores are (d/2,-d/2).
    expected_regularized = np.array([
        .9 * max(0, .1 - 2*d) + .1 * max(0, 1 + 2*d) + .2 * d*d/2
        for d in grid
    ])
    close("order_expected_regularized_optimum", grid[np.argmin(expected_regularized)], .05)
    close("order_mean_regularized_solution_reward", mean[1] - mean[0], .01)
    # Direct evaluate both analytical MeanThenLoss minimizer and nearby points.
    d_star = -.005
    def mean_objective(d):
        q = np.array([d/2, -d/2])
        return loss(q, mean, weights(mean, False)) + .2 * q @ q
    if any(mean_objective(d_star) > mean_objective(d_star + offset) + 1e-12 for offset in (-.1, -.001, .001, .1)):
        raise AssertionError("mean-then-loss analytical minimum")
    mean_regularized = np.maximum(0, .01 + 2*grid) + .2 * grid*grid/2
    close("order_mean_regularized_optimum", grid[np.argmin(mean_regularized)], d_star)

    perturbed = np.array([1., 1.-1e-10, .1])
    epsilon = 1e-12
    for _ in range(50):
        noise = rng.uniform(-epsilon, epsilon, size=3)
        close("positive_margin_argmax_stability", np.argmax(perturbed + noise), 0)
    near_score=np.array([1.,1.-1e-10])
    priority=np.array([1,0])
    eligible=near_score>=near_score.max()-1e-8
    numeric_near_choice=priority[eligible[priority].argmax()]
    close("numeric_near_priority_changes_choice", numeric_near_choice-np.argmax(near_score), 1)
    distribution = np.array([.5, .5])
    close("uniform_random_inference_equivariance", distribution[::-1], distribution)

    return dict(status="PASS",synthetic_seed=180116,check_groups=len(cases),checks=sum(cases.values()),
                cases=cases,maximum_absolute_errors=errors,
                operator_order_difference_at_minus01_0_plus01=differences,
                theorem_scope="Numerical checks supplement algebraic proofs; they are not finite-sample generalization evidence",
                no_experiment_asset_files_opened=True,real_training_runs=0,physical_llm_calls=0,
                fresh_reward_reads=0,sealed_payload_reads=0,confidence_intervals=0,bootstrap_runs=0,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("A_THEORY_QUALIFICATION.json"))
    args=parser.parse_args()
    result=qualify()
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],check_groups=result["check_groups"],checks=result["checks"])))


if __name__=="__main__":
    main()
