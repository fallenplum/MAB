"""E8-A1/A2 prerequisites, fold-scoped packs, and small software controls."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit, logsumexp
from e8_assets import (ROOT, ALPHA_RAW, exact_argmax, fit_features, load_historical,
                       matched_mean_loss_lambda, ridge_weights, sha)

M = 20
C = np.eye(M) - np.ones((M, M)) / M
PAIRS = np.array(list(itertools.combinations(range(M), 2)))
INCIDENCE = np.eye(M)[PAIRS[:, 0]] - np.eye(M)[PAIRS[:, 1]]


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def build_fold_packs(data, splits, private):
    manifests = []
    for fold, (history, evaluation) in enumerate(splits):
        directory = private / 'fold_packs' / f'fold{fold}'
        directory.mkdir(parents=True, exist_ok=True)
        history_path = directory / 'history480.npz'
        input_path = directory / 'evaluation_inputs_only120.npz'
        metadata_path = directory / 'evaluation_metadata_only120.npz'
        np.savez_compressed(history_path, E=data['E'][history], Y=data['Y'][history],
            ids=data['ids'][history], canonical=data['canonical'][history],
            task=data['task'][history], sub=data['sub'][history], global_indices=history)
        np.savez_compressed(input_path, E=data['E'][evaluation], ids=data['ids'][evaluation],
            canonical=data['canonical'][evaluation], global_indices=evaluation)
        np.savez_compressed(metadata_path, task=data['task'][evaluation], sub=data['sub'][evaluation],
            global_indices=evaluation)
        with np.load(input_path, allow_pickle=False) as archive:
            assert set(archive.files) == {'E', 'ids', 'canonical', 'global_indices'}
        manifests.append(dict(fold=fold, history_questions=480, evaluation_questions=120,
            history=dict(path=str(history_path), sha256=sha(history_path),
                         keys=['E', 'Y', 'ids', 'canonical', 'task', 'sub', 'global_indices']),
            evaluation_inputs=dict(path=str(input_path), sha256=sha(input_path),
                                   keys=['E', 'ids', 'canonical', 'global_indices']),
            evaluation_metadata=dict(path=str(metadata_path), sha256=sha(metadata_path),
                                     keys=['task', 'sub', 'global_indices']),
            history_eval_ID_and_canonical_disjoint=not bool(set(data['ids'][history]) & set(data['ids'][evaluation]))
                and not bool(set(data['canonical'][history]) & set(data['canonical'][evaluation]))))
    return manifests


def restore_original(data, splits, root, private):
    legacy_path = root / 'research/conditional_gain_20261106/private/oof/direct20__pca16__a100.npz'
    with np.load(legacy_path, allow_pickle=False) as archive:
        original = archive['pred'].copy()
    assert original.shape == (600, 20)
    restored = np.zeros_like(original)
    folds = []
    for fold, (history, evaluation) in enumerate(splits):
        features = fit_features(data['E'][history])
        X, Xt = features.phi(data['E'][history]), features.phi(data['E'][evaluation])
        W = ridge_weights(X, data['Y'][history])
        restored[evaluation] = Xt @ W
        raw_X = np.repeat(X, 3, axis=0)
        raw_Y = data['Y'][history].reshape(-1, 20)
        raw_W = np.linalg.solve(raw_X.T @ raw_X + ALPHA_RAW * np.eye(17), raw_X.T @ raw_Y)
        errors = dict(original_prediction_max_abs_error=float(np.max(np.abs(restored[evaluation] - original[evaluation]))),
                      raw_repeat_vs_mean_max_abs_error=float(np.max(np.abs(Xt @ (raw_W - W)))))
        assert max(errors.values()) < 1e-10
        assert np.array_equal(exact_argmax(restored[evaluation]), exact_argmax(original[evaluation]))
        path = private / 'original_ridge' / f'fold{fold}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **features.arrays(), W=W, history_indices=history,
            evaluation_indices=evaluation, evaluation_ids=data['ids'][evaluation],
            evaluation_prediction=restored[evaluation], raw_alpha=np.array(ALPHA_RAW))
        folds.append(dict(fold=fold, history_questions=480, held_questions=120,
            per_fold_legacy_weights_were_saved=False, reconstruction='original feature recipe and exact raw-alpha100 objective',
            parameters_sha256=sha(path), recovered_choice_changes=0, **errors))
    rows = np.arange(600)[:, None]
    selected = data['Y'][rows, np.arange(3)[None, :], exact_argmax(restored)[:, None]]
    gaps = np.sort(restored, axis=1)[:, -1] - np.sort(restored, axis=1)[:, -2]
    exact_tie = np.zeros(20)
    near_tie = np.zeros(20); near_tie[1] = 1e-12
    assert int(exact_argmax(exact_tie)) == 0 and int(exact_argmax(near_tie)) == 1
    near_tolerance_rule = int(np.argmax(near_tie >= near_tie.max() - 1e-10))
    assert near_tolerance_rule == 0
    return dict(status='PASS', old_oof_source_sha256=sha(legacy_path), folds=folds,
        OOF_accuracy=float(selected.mean()), per_generation_accuracy=selected.mean(axis=0).tolist(),
        numpy_exact_tie_action=0, numpy_near_tie_action=1,
        different_legacy_E7_tolerance_rule_near_tie_action=near_tolerance_rule,
        minimum_real_predicted_top_gap=float(gaps.min()),
        interpretation='legacy train600 question-OOF, distinct from historical validation200 accuracy')


def equivalent_fit_check(X, Y, penalty, question_weights):
    weights = np.asarray(question_weights, dtype=float)
    weights = weights / weights.mean()
    G = X.T @ (weights[:, None] * X)
    rhs = X.T @ (weights[:, None] * Y)
    W0 = np.linalg.solve(G + penalty * np.eye(X.shape[1]), rhs)
    Wc = np.linalg.solve(G + penalty * np.eye(X.shape[1]), rhs @ C)
    laplacian = INCIDENCE.T @ INCIDENCE
    pair_normal = np.kron(G, laplacian) + penalty * M * np.eye(X.shape[1] * M)
    Wp = np.linalg.solve(pair_normal, (rhs @ laplacian).ravel()).reshape(X.shape[1], M)
    p0, pc, pp = X @ W0, X @ Wc, X @ Wp
    center_error = float(np.max(np.abs(p0 @ C - pc)))
    pair_error = float(np.max(np.abs(pc - pp)))
    assert center_error < 1e-10 and pair_error < 1e-10
    assert np.array_equal(exact_argmax(p0), exact_argmax(pc))
    assert np.array_equal(exact_argmax(p0), exact_argmax(pp))
    e = p0 - Y
    lhs = np.sum((e @ INCIDENCE.T) ** 2, axis=1)
    rhs_identity = M * np.sum((e @ C) ** 2, axis=1)
    identity_error = float(np.max(np.abs(lhs - rhs_identity)))
    assert identity_error < 1e-10
    return dict(questions=len(X), features=X.shape[1], raw_mean_objective_penalty=penalty,
                common_shift_centered_max_error=center_error, explicit_pair_solution_max_error=pair_error,
                gap_identity_max_error=identity_error, action_changes=0)


def check_equivalences(data, splits):
    rng = np.random.default_rng(2026111308)
    cases = []
    for scale in (.01, 1., 10.):
        X = np.column_stack((np.ones(61), rng.normal(size=(61, 5))))
        Y = rng.normal(size=(61, M))
        cases.append(equivalent_fit_check(X, Y, scale * ALPHA_RAW / 3, rng.uniform(.3, 2., 61)))
    for history, _ in splits:
        transform = fit_features(data['E'][history])
        cases.append(equivalent_fit_check(transform.phi(data['E'][history]), data['Y'][history].mean(axis=1),
                                          ALPHA_RAW / 3, np.ones(480)))
    return dict(status='PASS', cases=cases, formulas=dict(
        identity='sum_{i<j}(e_i-e_j)^2 = 20 ||C e||^2',
        original_mean_label_objective='sum_q ||phi_q W-rbar_q||^2 + (alpha_raw/3)||W||^2',
        matched_normalized_objective='mean_q mean_20outputs squared error + lambda||W||^2',
        M3_equal_weight_objective='mean_q [sum_190 gap_error^2/20^2] + lambda||W||^2',
        matched_lambda='alpha_raw/(3*n_questions*20)',
        lambda_for_480=matched_mean_loss_lambda(480), lambda_for_320=matched_mean_loss_lambda(320),
        constant='leading1 is penalized exactly as the remaining16 coefficients'))


def ranknet_reference(W, X, rewards, lam=.003):
    q = X @ W
    reward_gap = rewards[:, PAIRS[:, 0]] - rewards[:, PAIRS[:, 1]]
    sign, weight = np.sign(reward_gap), np.abs(reward_gap)
    score_gap = q[:, PAIRS[:, 0]] - q[:, PAIRS[:, 1]]
    denominator = len(X) * len(PAIRS)
    value = float(np.sum(weight * np.logaddexp(0., -sign * score_gap)) / denominator + lam * np.sum(W ** 2))
    pair_gradient = -weight * sign * expit(-sign * score_gap) / denominator
    score_gradient = pair_gradient @ INCIDENCE
    gradient = X.T @ score_gradient + 2 * lam * W
    return value, gradient


def spo_reference(W, X, rewards, lam=.003, true_tie_uniform=False):
    q = X @ W
    competing = exact_argmax(2 * q - rewards)
    true = exact_argmax(rewards)
    target = (rewards == rewards.max(axis=1, keepdims=True)).astype(float)
    target = target / target.sum(axis=1, keepdims=True) if true_tie_uniform else np.eye(M)[true]
    values = np.max(2 * q - rewards, axis=1) - 2 * np.sum(q * target, axis=1) + np.sum(rewards * target, axis=1)
    score_gradient = 2 * (np.eye(M)[competing] - target) / len(X)
    return float(values.mean() + lam * np.sum(W ** 2)), X.T @ score_gradient + 2 * lam * W


def finite_difference(function, W, epsilon=1e-6):
    numeric = np.zeros_like(W)
    for index in np.ndindex(W.shape):
        plus, minus = W.copy(), W.copy()
        plus[index] += epsilon; minus[index] -= epsilon
        numeric[index] = (function(plus)[0] - function(minus)[0]) / (2 * epsilon)
    exact = function(W)[1]
    error = float(np.max(np.abs(numeric - exact)))
    assert error < 1e-7
    return error


def gradient_checks():
    rng = np.random.default_rng(2026111318)
    X = np.column_stack((np.ones(6), rng.normal(size=(6, 3))))
    Y = rng.binomial(3, .5, size=(6, M)) / 3
    W = rng.normal(size=(4, M))
    rank_error = finite_difference(lambda w: ranknet_reference(w, X, Y), W)
    spo_error = finite_difference(lambda w: spo_reference(w, X, Y), W)
    uniform_error = finite_difference(lambda w: spo_reference(w, X, Y, true_tie_uniform=True), W)
    hinge_checks = []
    for _ in range(100):
        rewards = rng.random(M)
        q = rng.normal(size=M)
        chosen, best = int(exact_argmax(q)), int(exact_argmax(rewards))
        competitor = int(exact_argmax(2 * q - rewards))
        loss = float(np.max(2 * q - rewards) - 2 * q[best] + rewards[best])
        regret = float(rewards[best] - rewards[chosen])
        assert loss + 1e-12 >= regret >= -1e-12
        gradient = 2 * (np.eye(M)[competitor] - np.eye(M)[best])
        direction = rng.normal(size=M)
        changed = float(np.max(2 * (q + direction) - rewards) - 2 * (q + direction)[best] + rewards[best])
        assert changed >= loss + gradient @ direction - 1e-12
        hinge_checks.append(loss)
    return dict(status='PASS', ranknet_central_difference_max_error=rank_error,
        SPOplus_central_difference_max_error=spo_error,
        SPOplus_true_tie_uniform_central_difference_max_error=uniform_error,
        SPOplus_upper_bound_and_subgradient_trials=len(hinge_checks),
        SPOplus_formula='max_j(2q_j-r_j)-2q_jstar+r_jstar',
        SPOplus_gradient='2(e_argmax(2q-r)-e_jstar)',
        limitations='Reference-operator qualification. Actual training-operator equivalence is checked after implementation is frozen.')


def toy_controls(root):
    rng = np.random.default_rng(2026111328)
    X = np.column_stack((np.ones(80), np.tile([-1., 1.], 40)))
    query = np.column_stack((np.ones(40), np.tile([-1., 1.], 20)))
    all_equal = np.full((80, M), .5)
    We = ridge_weights(X, all_equal)
    equal_predictions = query @ We
    assert np.max(np.ptp(equal_predictions, axis=1)) < 1e-12
    learnable = np.full((80, M), .2)
    train_action = np.where(X[:, 1] > 0, 19, 0)
    learnable[np.arange(80), train_action] = .8
    W = ridge_weights(X, learnable)
    expected_action = np.where(query[:, 1] > 0, 19, 0)
    assert np.array_equal(exact_argmax(query @ W), expected_action)
    sys.path.insert(0, str(root / 'research/e5_paired_gain_validation/public'))
    from theory_checks import action_design, strict_construction
    A = action_design()
    reward, projection_expected, _, _, _, _ = strict_construction(0, .2, .1)
    projection = A @ np.linalg.pinv(A) @ reward
    assert np.max(np.abs(projection - projection_expected)) < 1e-12
    wrong, best = int(exact_argmax(projection)), int(exact_argmax(reward))
    assert wrong == 0 and reward[best] > reward[wrong]
    tied = np.zeros(M); tied[[0, 1, 5]] = 1.
    assert int(exact_argmax(tied)) == 0
    noise = np.zeros((80, 3, M)); noise[:, :, 0] = 1
    noise[:, :, 18:] = rng.binomial(1, .5, (80, 3, 2))
    Wn = ridge_weights(np.ones((80, 1)), noise)
    assert np.all(exact_argmax(np.ones((40, 1)) @ Wn) == 0)
    assert np.var(noise[:, :, :18], axis=1).max() == 0
    assert np.var(noise[:, :, 18:], axis=1).mean() > 0
    return dict(status='PASS', environments=[
        dict(name='all_actions_equal_expectation', true_reward=.5, centered_score_spread=float(np.ptp(equal_predictions, axis=1).max()),
             genuine_population_gain=0.),
        dict(name='single_feature_decides_best_action', query_questions=40, correct_action_predictions=40,
             expected_service_reward=.8, balanced_fixed_reward=.5),
        dict(name='manuscript_strict_quadratic_projection_counterexample', wrong_projection_action=wrong,
             example_true_best_action=best, genuine_loss=float(reward[best] - reward[wrong]),
             projection_error=float(np.max(np.abs(projection - projection_expected)))),
        dict(name='multiple_true_optimal_actions', true_optimal_actions=3, exact_lexicographic_action=0),
        dict(name='noise_only_in_some_actions', stochastic_actions=2, deterministic_actions=18,
             correct_query_predictions=40)],
        interpretation='Small mathematical software controls only; no real routing benefit inferred from them')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    work = root / 'research/e8_decision_focused_combination_learning'
    private = work / 'private/qualification'
    public = work / 'public/qualification'
    if (public / 'E8_QUALIFICATION.json').exists():
        raise FileExistsError('Existing qualification preserved; use a separate run for repetition')
    private.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data, refs, splits = load_historical(root)
    packs = build_fold_packs(data, splits, private)
    put(private / 'FOLD_PACKS_MANIFEST.json', dict(schema='E8_fold_scoped_packs_v1', folds=packs,
        privileged_builder=dict(reward_payload='train600 full raw matrix decoded by pack builder',
            outside_each_fold_is_inside_other_folds_history=True,
            formal_trainer_permissions='one history480 reward pack plus evaluation inputs only',
            new_method_evaluation_rewards='independent evaluator after all five model prediction freezes'),
        refs=refs))
    restored = restore_original(data, splits, root, private)
    equivalence = check_equivalences(data, splits)
    gradients = gradient_checks()
    toys = toy_controls(root)
    result = dict(schema='E8_prerequisite_qualification_v1', status='PASS_A1_A2_REFERENCE_A3',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), A1=restored, A2=equivalence,
        A3_reference_gradients=gradients, A3_toy_controls=toys,
        fold_packs=[{key: value for key, value in record.items()
                     if key not in ('history', 'evaluation_inputs', 'evaluation_metadata')}
                    | {key + '_sha256': record[key]['sha256'] for key in ('history', 'evaluation_inputs', 'evaluation_metadata')}
                    for record in packs],
        pack_manifest_sha256=sha(private / 'FOLD_PACKS_MANIFEST.json'),
        access=dict(loaded_reward_scope='train600 only', validation_rewards_opened=0,
                    fresh400_rewards_opened=0, sealed200_rewards_opened=0,
                    legacy_Ridge_restore_is_a_prerequisite_not_new_model_selection=True),
        source_sha256={Path(__file__).name: sha(__file__), 'e8_assets.py': sha(Path(__file__).with_name('e8_assets.py'))},
        new_LLM_or_API_calls=0, wall_seconds=time.perf_counter() - started)
    put(public / 'E8_QUALIFICATION.json', result)
    print(json.dumps(dict(status=result['status'], A1_accuracy=restored['OOF_accuracy'],
        A2_max_pair_error=max(r['explicit_pair_solution_max_error'] for r in equivalence['cases']),
        wall_seconds=result['wall_seconds']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
