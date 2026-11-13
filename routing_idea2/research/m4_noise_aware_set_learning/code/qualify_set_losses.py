"""Deterministic mathematical/operator qualification, with synthetic data only."""
from pathlib import Path
import hashlib
import json
import os
import sys

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from set_spo_operators import (set_weights, set_spo_sample_loss, ns_spo_value_subgradient,
                               davidson_value_gradient, posterior_pair_probabilities)

BASE = Path(__file__).resolve().parent.parent


def difference_gradient(function, value, step=1e-6):
    derivative = np.empty_like(value)
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += step
        minus[index] -= step
        derivative[index] = (function(plus) - function(minus)) / (2 * step)
    return derivative


def consistency_qp(regularization):
    """Independent20-action two-state epigraph QP, q bounded in[-1,1]."""
    sys.path.insert(0, r'E:\语言模型组合路由\research\e8_decision_focused_combination_learning\private\dependencies')
    import clarabel
    reward = np.zeros((2, 20))
    reward[0, :2] = [.6, .5]
    reward[1, :2] = [0., 1.]
    probability = np.array([.9, .1])
    weights = set_weights(reward, .05)
    linear = np.r_[-2 * probability @ weights, probability]
    quadratic = sparse.diags(np.r_[np.full(20, 2 * regularization), [0., 0.]], format='csc')
    constraints = np.zeros((40 + 40, 22))
    upper = np.r_[reward.ravel(), np.ones(40)]
    for state in range(2):
        for action in range(20):
            constraints[state * 20 + action, action] = 2.
            constraints[state * 20 + action, 20 + state] = -1.
    constraints[40:60, :20] = np.eye(20)
    constraints[60:80, :20] = -np.eye(20)
    settings = clarabel.DefaultSettings()
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    settings.max_iter = 1000
    settings.max_threads = 1
    settings.verbose = False
    problem = clarabel.DefaultSolver(quadratic, linear, sparse.csc_matrix(constraints), upper,
                                    [clarabel.NonnegativeConeT(80)], settings)
    solution = problem.solve()
    q = np.asarray(solution.x[:20])
    direct = probability @ set_spo_sample_loss(q, reward, weights) + regularization * np.sum(q ** 2)
    objective = solution.obj_val + np.sum(probability[:, None] * weights * reward)
    mean_reward = probability @ reward
    assert solution.status == clarabel.SolverStatus.Solved
    assert abs(direct - objective) < 1e-8
    assert q.argmax() == 0 and mean_reward.argmax() == 1
    return dict(regularization=regularization, solver_status=str(solution.status),
                optimal_score=q.tolist(), expected_reward=mean_reward.tolist(),
                surrogate_selected_action=int(q.argmax()), bayes_action=int(mean_reward.argmax()),
                posterior_mean_regret=float(mean_reward.max() - mean_reward[q.argmax()]),
                objective=float(direct), primal_dual_gap=float(abs(solution.obj_val - solution.obj_val_dual)))


def main():
    rng = np.random.default_rng(2026111404)
    checks = []

    def record(name, result=True, **detail):
        if not result:
            raise AssertionError(name)
        checks.append(dict(name=name, status='PASS', **detail))

    reward = rng.random((10000, 20))
    q = rng.uniform(-1, 1, size=(10000, 20))
    weights = set_weights(reward, .05)
    loss = set_spo_sample_loss(q, reward, weights)
    record('nonnegative_and_bounded', loss.min() >= -1e-12 and loss.max() <= 5 + 1e-12,
           cases=len(q), minimum=float(loss.min()), maximum=float(loss.max()), uniform_bound_Q1=5)
    regret = reward.max(1) - reward[np.arange(len(q)), q.argmax(1)]
    bias = reward.max(1) - np.sum(weights * reward, axis=1)
    record('approximate_set_decision_upper_bound', np.all(regret <= loss + bias + 1e-12),
           maximum_violation=float(np.max(regret - loss - bias)), maximum_bias=float(bias.max()), delta=.05)
    exact_weights = set_weights(reward, 0.)
    exact = set_spo_sample_loss(q, reward, exact_weights)
    original = np.max(2 * q - reward, axis=1) - (2 * q[np.arange(len(q)), reward.argmax(1)] - reward.max(1))
    record('singleton_set_recovers_original', np.allclose(exact, original, atol=1e-14),
           maximum_error=float(np.max(abs(exact - original))))
    unique_near = set_weights(np.array([[1., .98]]), .05)
    record('unique_argmax_does_not_imply_singleton_near_set', np.array_equal(unique_near, [[.5, .5]]),
           reward=[1., .98], delta=.05, weight=unique_near.tolist())
    near_q, near_reward, near_weight = np.array([0., .001]), np.array([1., .95]), np.array([.5, .5])
    near_loss = float(set_spo_sample_loss(near_q, near_reward, near_weight))
    record('unadjusted_surrogate_upper_bound_counterexample', near_loss < .05,
           loss=near_loss, greedy_regret=.05, delta=.05)
    all_tied_reward = np.full((7, 20), .5)
    tied_zero_error = float(np.max(abs(set_spo_sample_loss(np.zeros_like(all_tied_reward), all_tied_reward,
           set_weights(all_tied_reward)))))
    record('all_tied_zero_score_loss_zero', tied_zero_error < 1e-14, floating_error=tied_zero_error)
    permutation = rng.permutation(20)
    changed = set_spo_sample_loss(q[:, permutation], reward[:, permutation], weights[:, permutation])
    record('simultaneous_action_permutation_invariance', np.max(abs(changed - loss)) < 1e-12,
           max_error=float(np.max(abs(changed - loss))))
    raw_means = rng.integers(0, 4, size=(10000, 20)) / 3.
    record('three_repeat_empirical_delta_005_equals_delta0',
           np.array_equal(set_weights(raw_means, 0.), set_weights(raw_means, .05)), cases=10000)

    phi = np.column_stack((np.ones(7), rng.normal(size=(7, 2))))
    coefficient = rng.normal(scale=.1, size=(3, 20))
    samples = rng.random((7, 9, 20))
    sample_weights = set_weights(samples, .05)
    regularization = .03
    value, gradient = ns_spo_value_subgradient(coefficient, phi, samples, sample_weights, regularization)
    finite = difference_gradient(lambda w: ns_spo_value_subgradient(w, phi, samples, sample_weights, regularization)[0], coefficient)
    record('NS_full_MC_finite_difference_gradient', np.max(abs(gradient - finite)) < 1e-7,
           maximum_error=float(np.max(abs(gradient - finite))))
    other = rng.normal(scale=.1, size=coefficient.shape)
    convex_mix = .37
    left = ns_spo_value_subgradient(convex_mix * coefficient + (1 - convex_mix) * other,
                                  phi, samples, sample_weights, regularization)[0]
    right = convex_mix * value + (1 - convex_mix) * ns_spo_value_subgradient(other, phi, samples, sample_weights, regularization)[0]
    record('NS_convexity_numeric_check', left <= right + 1e-12, gap=float(right - left))
    tied = np.array([[1., 0.], [0., 1.]])
    tied_weight = set_weights(tied)
    full_mean = float(np.mean(set_spo_sample_loss(np.zeros(2), tied, tied_weight)))
    plugin = float(set_spo_sample_loss(np.zeros(2), tied.mean(0), tied_weight.mean(0)))
    record('mean_substitution_is_not_full_MC', full_mean == 1. and plugin == 0.,
           exact_MC=full_mean, substituted_mean_loss=plugin,
           E_weight_reward=float(np.mean(np.sum(tied_weight * tied, axis=1))),
           E_weight_dot_E_reward=float(tied_weight.mean(0) @ tied.mean(0)))

    targets = posterior_pair_probabilities(samples, .05)
    davidson_value, davidson_gradient, eta_gradient = davidson_value_gradient(coefficient, phi, targets, .2, regularization)
    finite = difference_gradient(lambda w: davidson_value_gradient(w, phi, targets, .2, regularization)[0], coefficient)
    eta_finite = (davidson_value_gradient(coefficient, phi, targets, .2 + 1e-6, regularization)[0]
                  - davidson_value_gradient(coefficient, phi, targets, .2 - 1e-6, regularization)[0]) / 2e-6
    record('Davidson_head_finite_difference_gradient', np.max(abs(davidson_gradient - finite)) < 1e-7,
           maximum_error=float(np.max(abs(davidson_gradient - finite))))
    record('Davidson_log_nu_finite_difference_gradient', abs(eta_gradient - eta_finite) < 1e-7,
           maximum_error=float(abs(eta_gradient - eta_finite)))
    extreme = coefficient * 10000
    extreme_value, extreme_gradient, extreme_eta = davidson_value_gradient(extreme, phi, targets, 20., regularization)
    record('Davidson_stable_extreme_logits', np.isfinite(extreme_value) and np.isfinite(extreme_gradient).all()
           and np.isfinite(extreme_eta), value=float(extreme_value))
    q_other = coefficient.copy()
    original_value = davidson_value_gradient(q_other, phi, targets, .2, regularization)[0]
    mirror = targets[..., ::-1]
    mirrored_value = davidson_value_gradient(-q_other, phi, mirror, .2, regularization)[0]
    record('Davidson_win_loss_swap_invariance', abs(original_value - mirrored_value) < 1e-12)

    hetero = np.full((2, 20), .5)
    hetero[:, 0] = [0., 1.]
    hetero_weights = set_weights(hetero, .05).mean(0)
    record('high_uncertainty_need_not_mean_uniform_weights', abs(hetero_weights[0] - .5) < 1e-15,
           all_mean_rewards=.5, high_variance_action_weight=float(hetero_weights[0]),
           each_other_action_weight=float(hetero_weights[1]), exchangeable_uniform_reference=.05)
    intervals = np.array([[0., .4], [.3, .7], [.6, 1.]])
    overlaps = np.minimum(intervals[:, None, 1], intervals[None, :, 1]) >= np.maximum(intervals[:, None, 0], intervals[None, :, 0])
    record('incomparability_overlap_not_transitive', overlaps[0, 1] and overlaps[1, 2] and not overlaps[0, 2],
           intervals=intervals.tolist())
    sample_d = np.array([-.9, -.2, .2, .9])
    pzero = float(np.mean(abs(sample_d) <= .05))
    record('absence_of_directional_certainty_not_equivalence', pzero == 0. and np.mean(sample_d > .05) == .5,
           p_plus=.5, p_practical_tie=pzero, p_minus=.5)
    counterexamples = [consistency_qp(lam) for lam in (0., .001, .01)]
    record('population_decision_consistency_counterexample', all(item['surrogate_selected_action'] == 0 for item in counterexamples),
           examples=counterexamples, analytic_sufficient_lambda_upper_bound=.08 / 1.3625)
    result = {'status': 'PASS_WITH_COUNTEREXAMPLES_AND_SCOPE_LIMITATIONS', 'synthetic_only': True,
              'real_reward_accesses': 0, 'physical_LLM_calls': 0,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'operator_sha256': hashlib.sha256((Path(__file__).parent / 'set_spo_operators.py').read_bytes()).hexdigest(),
              'checks': checks}
    path = Path(__file__).parent / 'MATHEMATICAL_SOFTWARE_CHECKS.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': len(checks),
                      'operator_gradient_errors': [item for item in checks if 'gradient' in item['name']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
