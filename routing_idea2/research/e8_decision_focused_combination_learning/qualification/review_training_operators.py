"""Independent derivatives and normal equations for the actual E8 objectives."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize


def main():
    root = Path(r'E:\语言模型组合路由')
    work = root / 'research/e8_decision_focused_combination_learning'
    sys.path.insert(0, str(work / 'public/qualification'))
    sys.path.insert(0, str(work / 'private/dependencies'))
    from e8_assets import exact_argmax, ridge_weights, matched_mean_loss_lambda, sha
    from qualify_e8 import finite_difference, ranknet_reference, spo_reference, put
    source = Path(__file__).parents[1] / 'training/e8_training_v1.py'
    source_hash = sha(source)
    spec = importlib.util.spec_from_file_location('e8_actual_training_reviewed', source)
    training = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training
    spec.loader.exec_module(training)
    started = time.perf_counter()
    rng = np.random.default_rng(2026111338)
    X = np.column_stack((np.ones(24), rng.normal(size=(24, 3))))
    Y = rng.binomial(3, .5, (24, 20)) / 3
    W = rng.normal(size=(4, 20))
    lam = .003
    rank_error = finite_difference(lambda w: training.ranknet_value_gradient(w, X, Y, lam), W)
    rank_actual = training.ranknet_value_gradient(W, X, Y, lam)
    rank_expected = ranknet_reference(W, X, Y, lam)
    assert abs(rank_actual[0] - rank_expected[0]) < 1e-12
    assert np.max(np.abs(rank_actual[1] - rank_expected[1])) < 1e-12
    list_error = finite_difference(lambda w: training.listnet_value_gradient(w, X, Y, lam, .3, .7), W)
    spo_checks = []
    for actual_rule, uniform in (('lexicographic', False), ('uniform', True)):
        error = finite_difference(lambda w: training.spo_plus_value_subgradient(w, X, Y, lam, actual_rule), W)
        actual = training.spo_plus_value_subgradient(W, X, Y, lam, actual_rule)
        expected = spo_reference(W, X, Y, lam, uniform)
        assert abs(actual[0] - expected[0]) < 1e-12
        assert np.max(np.abs(actual[1] - expected[1])) < 1e-12
        spo_checks.append(dict(rule=actual_rule, finite_difference_error=error,
                               independent_value_error=abs(actual[0] - expected[0]),
                               independent_gradient_error=float(np.max(np.abs(actual[1] - expected[1])))))

    base = ridge_weights(X, Y)
    centered = base @ (np.eye(20) - np.ones((20, 20)) / 20)
    pair_checks = []
    oof = rng.normal(size=(24, 20))
    for alpha in (0., 2., 5.):
        weights, top = training.competition_weights(oof, alpha, 3)
        assert weights.shape == (24, 190)
        assert np.max(np.abs(weights.mean(axis=1) - 1)) < 1e-12
        actual, diagnostic = training.fit_gap_ridge(X, Y, matched_mean_loss_lambda(24), weights)
        normal = np.zeros((80, 80))
        right = np.zeros((4, 20))
        incidence = training.INCIDENCE
        for q in range(24):
            laplacian = incidence.T @ (weights[q, :, None] * incidence)
            normal += np.kron(np.outer(X[q], X[q]), laplacian) / (24 * 400)
            right += np.outer(X[q], Y[q] @ laplacian) / (24 * 400)
        normal += matched_mean_loss_lambda(24) * np.eye(80)
        reference = np.linalg.solve(normal, right.ravel()).reshape(4, 20)
        coefficient_error = float(np.max(np.abs(reference - actual)))
        assert coefficient_error < 1e-10
        if alpha == 0:
            assert np.max(np.abs(actual - centered)) < 1e-10
            assert np.array_equal(exact_argmax(X @ actual), exact_argmax(X @ base))
        pair_checks.append(dict(alpha=alpha, mean_weight_error=float(np.max(np.abs(weights.mean(axis=1)-1))),
                                independent_normal_equation_error=coefficient_error,
                                objective=diagnostic['objective']))

    # Independent tiny exact epigraph solves, using C-order parameters and
    # explicit scalar constraints instead of the implementation's F-order blocks.
    phi = np.column_stack((np.ones(8), rng.normal(size=(8, 2))))
    rewards = rng.binomial(3, .5, (8, 20)) / 3
    qp_checks = []
    for rule in ('lexicographic', 'uniform'):
        Wqp, diagnostic = training.fit_spo_qp(phi, rewards, .1, rule, training.DEFAULT_SETTINGS)
        assert diagnostic['success']
        true = np.zeros_like(rewards)
        if rule == 'lexicographic':
            true[np.arange(8), exact_argmax(rewards)] = 1
        else:
            true = (rewards == rewards.max(axis=1, keepdims=True)).astype(float)
            true /= true.sum(axis=1, keepdims=True)
        size = 60
        def objective(vector):
            coefficient, epigraph = vector[:size].reshape(3, 20), vector[size:]
            value = np.mean(epigraph - 2 * np.sum((phi @ coefficient) * true, axis=1) + rewards.max(axis=1))
            value += .1 * np.sum(coefficient ** 2)
            gradient = np.r_[(-2 * phi.T @ true / 8 + .2 * coefficient).ravel(), np.full(8, 1 / 8)]
            return float(value), gradient
        def constraints(vector):
            coefficient, epigraph = vector[:size].reshape(3, 20), vector[size:]
            return (epigraph[:, None] - 2 * phi @ coefficient + rewards).ravel()
        jacobian = np.zeros((160, 68))
        for q in range(8):
            for action in range(20):
                jacobian[q * 20 + action, np.arange(3) * 20 + action] = -2 * phi[q]
                jacobian[q * 20 + action, 60 + q] = 1
        initial = np.r_[np.zeros(60), -rewards.min(axis=1)]
        independent = minimize(objective, initial, jac=True, method='SLSQP',
            constraints=dict(type='ineq', fun=constraints, jac=lambda _: jacobian),
            options=dict(maxiter=2000, ftol=1e-12))
        assert independent.success
        actual_value = training.spo_plus_value_subgradient(Wqp, phi, rewards, .1, rule)[0]
        discrepancy = abs(actual_value - independent.fun)
        assert discrepancy < 1e-6
        assert diagnostic['constraint_violation'] < 1e-7
        qp_checks.append(dict(true_tie_rule=rule, osqp_status=diagnostic['status'],
            independent_SLSQP_status=independent.message, objective_error=discrepancy,
            epigraph_direct_value_error=diagnostic['objective_discrepancy'],
            constraint_violation=diagnostic['constraint_violation']))
    # Run the actual objectives on the five fixed software controls.
    toy_X = np.column_stack((np.ones(64), np.tile([-1., 1.], 32)))
    toy_query = np.column_stack((np.ones(40), np.tile([-1., 1.], 20)))
    single = np.full((64, 20), .2)
    single_action = np.where(toy_X[:, 1] > 0, 19, 0)
    single[np.arange(64), single_action] = .8
    single_query = np.full((40, 20), .2)
    single_query[np.arange(40), np.where(toy_query[:, 1] > 0, 19, 0)] = .8
    sys.path.insert(0, str(root / 'research/e5_paired_gain_validation/public'))
    from theory_checks import strict_construction
    teacher, _, _, _, _, _ = strict_construction(0, .2, .1)
    ties = np.zeros(20); ties[[0, 1, 5]] = 1
    noise = np.zeros((64, 3, 20)); noise[:, :, 0] = 1
    noise[:, :, 18:] = rng.binomial(1, .5, (64, 3, 2))
    noisy_expected = np.zeros((40, 20)); noisy_expected[:, 0] = 1; noisy_expected[:, 18:] = .5
    toy_worlds = [
        ('all_equal', np.full((64, 20), .5), np.full((40, 20), .5)),
        ('feature_selectable', single, single_query),
        ('teacher_restricted_projection', np.tile(teacher, (64, 1)), np.tile(teacher, (40, 1))),
        ('multiple_true_optima', np.tile(ties, (64, 1)), np.tile(ties, (40, 1))),
        ('partial_generation_noise', noise.mean(axis=1), noisy_expected)]
    toy_checks = []
    for name, fitting_rewards, expected_rewards in toy_worlds:
        for method in ('M0_original', 'M1', 'M2', 'M3', 'M4'):
            parameters = dict(lambda_multiplier=1., target_temperature=.3, alpha=2.)
            coefficient, diagnostic = training.fit_coefficients(toy_X, fitting_rewards,
                method, parameters, oof_scores=rng.normal(size=(64, 20)))
            chosen = exact_argmax(toy_query @ coefficient)
            served = expected_rewards[np.arange(40), chosen]
            missed = expected_rewards.max(axis=1) - served
            assert missed.max() < 1e-10, (name, method, missed.max())
            toy_checks.append(dict(environment=name, method=method, query_questions=40,
                mean_expected_quality=float(served.mean()), true_expected_regret=float(missed.mean()),
                solver_status=diagnostic['status']))

    # Verify one fixed shuffled target block cannot influence its own legal OOF.
    toy_history = dict(E=rng.normal(size=(60, 384)), Y=rng.binomial(1, .5, (60, 3, 20)),
        ids=np.array([f'software{i}' for i in range(60)]),
        task=np.array(['gsm8k'] * 30 + ['bbh'] * 30), sub=np.array([''] * 30 + ['toy'] * 30))
    shuffled_rewards, mapping = training.permuted_rewards(toy_history, 2026111348)
    fitting = dict(toy_history, Y=shuffled_rewards)
    oof, log = training.legal_oof_m0(fitting, 2026111349, training.DEFAULT_SETTINGS, permutation=False)
    changed = dict(fitting, Y=shuffled_rewards.copy())
    changed['Y'][0] = 1 - changed['Y'][0]
    changed_oof, _ = training.legal_oof_m0(changed, 2026111349, training.DEFAULT_SETTINGS, permutation=False)
    assert np.array_equal(oof[0], changed_oof[0])
    for fold_record in log:
        fit_rows, held_rows = fold_record['fitting_rows'], fold_record['held_rows']
        assert not set(fit_rows) & set(held_rows)
        assert np.array_equal(fold_record['local_permutation'], np.arange(len(fit_rows)))
        assert fold_record['fitting_reward_hash'] == training.array_hash(fitting['Y'][fit_rows])
    shuffled_oof_check = dict(status='PASS', changed_held_block=0,
        held_prediction_max_error=float(np.max(np.abs(oof[0] - changed_oof[0]))),
        shuffle_mapping_fixed_once_before_OOF=True, OOF_internal_shuffle=False,
        source_fix='Inner/final fitting region is shuffled once, and OOF splits that already-shuffled target dataset')

    # One legal480-question M4 fit at the weakest prespecified regularization.
    history_path = work / 'private/qualification/fold_packs/fold0/history480.npz'
    with np.load(history_path, allow_pickle=False) as archive:
        history_E, history_Y = archive['E'], archive['Y']
    features = training.fit_features(history_E)
    _, real_qp_diagnostic = training.fit_coefficients(features.phi(history_E), history_Y.mean(axis=1),
        'M4', dict(lambda_multiplier=.1))
    real_qp_check = dict(status='PASS', input_scope='fold0 history480 only',
        lambda_multiplier=.1, solver=real_qp_diagnostic['solver'],
        iterations=real_qp_diagnostic['iterations'], constraint_violation=real_qp_diagnostic['constraint_violation'],
        objective_discrepancy=real_qp_diagnostic['objective_discrepancy'],
        primal_residual=real_qp_diagnostic['primal_residual'], dual_residual=real_qp_diagnostic['dual_residual'],
        wall_seconds=real_qp_diagnostic['wall_seconds'])
    if sha(source) != source_hash:
        raise RuntimeError('Training source changed during independent operator review')
    result = dict(schema='E8_actual_training_operator_review_v1', status='PASS',
        training_source_sha256=source_hash, ranknet_finite_difference_error=rank_error,
        listnet_finite_difference_error=list_error, listnet_target_temperature=.3, listnet_prediction_temperature=.7,
        SPOplus=spo_checks, M3_pair_quadratic=pair_checks, SPO_epigraph_QP=qp_checks,
        five_software_controls_actual_operators=toy_checks,
        shuffled_OOF_block_isolation=shuffled_oof_check, real_history480_M4_preflight=real_qp_check,
        actual_reward_scope='one legal fold0 history480 pack for prespecified solver preflight',
        evaluation_or_validation_or_fresh_or_sealed_reward_files_opened=0,
        caveat='Software and solver checks; actual new outer-fold decision benefit remains unevaluated',
        wall_seconds=time.perf_counter() - started)
    put(work / 'public/qualification/ACTUAL_OPERATOR_REVIEW.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
