"""Freeze B1 actors and train-only auxiliaries before evaluation outcomes open.

This process decodes train600 rewards and validation200 input features only.
The validation outcomes path is retained as metadata for a separate evaluator.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

ALPHA_RAW = 100.
AUX_ALPHA = 100.
TARGET_NAME = 'direct20__pca16__a100'
TEAMS = np.array(list(itertools.combinations(range(6), 3)), dtype=int)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def choose(values):
    # The restored original direct20 implementation uses numpy's exact argmax.
    return np.asarray(values).argmax(axis=-1)


class LegacyTransform:
    """Restore only attributes of the known historical common.Transform class."""
    def transform(self, embeddings):
        values = self.pca.transform(embeddings)
        return np.column_stack((np.ones(len(values)), self.scaler.transform(values)))


class RouterUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'common' and name == 'Transform':
            return LegacyTransform
        return super().find_class(module, name)


def fit_router(embeddings, rewards):
    pca = PCA(16, svd_solver='full').fit(embeddings)
    scaler = StandardScaler().fit(pca.transform(embeddings))
    X = np.column_stack((np.ones(len(embeddings)), scaler.transform(pca.transform(embeddings))))
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    W = (vt.T * (s / (s * s + ALPHA_RAW / 3))) @ (u.T @ rewards.mean(axis=1))
    default = int(choose(rewards.mean(axis=(0, 1))))
    return pca, scaler, W, default


def predict_router(model, embeddings):
    pca, scaler, W, default = model
    phi = np.column_stack((np.ones(len(embeddings)), scaler.transform(pca.transform(embeddings))))
    return phi @ W, phi, np.full(len(embeddings), default, dtype=int)


def paired_features(E, predictions, b):
    c = choose(predictions)
    m = np.clip(predictions[np.arange(len(E)), c] - predictions[np.arange(len(E)), b], -1, 1)
    active = b != c
    m[~active] = 0.
    members = np.array([[int(j in team) for j in range(6)] for team in TEAMS])
    overlap = (members[b] * members[c]).sum(axis=1)
    features = np.column_stack((E, predictions, np.eye(20)[b], np.eye(20)[c],
                               members[b], members[c], overlap, m))
    return features, b, c, m, active, overlap


def legal_pair_rewards(Y, b, c):
    rows = np.arange(len(Y))[:, None]
    seeds = np.arange(3)[None, :]
    differences = Y[rows, seeds, c[:, None]] - Y[rows, seeds, b[:, None]]
    return differences, differences.mean(axis=1)


def moments_fit(features, m, d, active):
    if active.sum() < 2:
        raise ValueError('Insufficient historical action disagreements for auxiliary fitting')
    scaler = StandardScaler().fit(features[active])
    target = np.column_stack(((d - m) ** 2, d ** 2))
    model = Ridge(alpha=AUX_ALPHA, fit_intercept=True, solver='cholesky').fit(
        scaler.transform(features[active]), target[active])
    return scaler, model


def moments_predict(fitted, features, active):
    scaler, model = fitted
    raw = model.predict(scaler.transform(features))
    safe = np.column_stack((np.clip(raw[:, 0], 1e-6, 4.), np.clip(raw[:, 1], 1e-6, 1.)))
    safe[~active] = 0.
    raw[~active] = 0.
    return raw, safe


def oof_predictions(E, Y, task, splits):
    predictions = np.zeros((len(E), 20))
    defaults = np.zeros(len(E), dtype=int)
    fold_index = np.full(len(E), -1, dtype=int)
    for fold, (history, held) in enumerate(splits):
        assert not set(history) & set(held)
        predictions[held], _, defaults[held] = predict_router(fit_router(E[history], Y[history]), E[held])
        fold_index[held] = fold
    assert (fold_index >= 0).all()
    return predictions, defaults, fold_index


def cost_predictions(costs, b, c):
    active = b != c
    result = {key: np.zeros(len(b)) for key in ('gold', 'member_calls', 'aggregator_calls',
              'scoring_ops', 'input_tokens', 'output_tokens', 'tokens', 'source_model_wall_s')}
    means = {name: np.asarray(value).mean(axis=(0, 1)) for name, value in costs.items()
             if name != 'aggregate_scores'}
    for q in np.flatnonzero(active):
        actions = [int(b[q]), int(c[q])]
        member_ids = np.unique(TEAMS[actions])
        result['gold'][q] = 1
        result['scoring_ops'][q] = 6
        for stem, ids, call_name in (('member', member_ids, 'member_calls'),
                                     ('aggregate', actions, 'aggregator_calls')):
            for suffix, resource in (('execution_counts', call_name), ('prompt_tokens', 'input_tokens'),
                                     ('completion_tokens', 'output_tokens'), ('wall_time_s', 'source_model_wall_s')):
                result[resource][q] += float(3 * means[stem + '_' + suffix][ids].sum())
        result['tokens'][q] = result['input_tokens'][q] + result['output_tokens'][q]
    return result, means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'E:\语言模型组合路由')
    parser.add_argument('--work', default=None)
    args = parser.parse_args()
    root = Path(args.root)
    work = Path(args.work) if args.work else root / 'research/b1_reliable_paired_evaluation'
    private = work / 'private/assets'
    public = work / 'public/assets'
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    if (private / 'FREEZE_RECEIPT.json').exists():
        raise FileExistsError('Frozen assets already exist. Use their recorded hashes for replay.')
    started = time.perf_counter()
    sys.path.insert(0, str(root / 'research/decision_acquisition_e3/public'))
    from assets import load_assets, BUNDLE_SHA, FOLDS_SHA
    train, train_refs = load_assets(root)
    E, Y = train['E'], train['Y']
    bundle_path = root / 'artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json'
    bundle = json.loads(bundle_path.read_text(encoding='utf-8-sig'))
    validation_ref = bundle['inputs']['selector_validation']
    assert sha(validation_ref['path']) == validation_ref['sha256']
    with np.load(validation_ref['path'], allow_pickle=False) as archive:
        expected_keys = {'question_ids', 'canonical_hashes', 'task_labels', 'subtask_labels', 'embeddings'}
        assert set(archive.files) == expected_keys
        val = {name: archive[name].copy() for name in expected_keys}
    assert val['embeddings'].shape == (200, 384)
    assert not set(val['question_ids']) & set(train['ids'])
    assert not set(val['canonical_hashes']) & set(train['canonical'])
    assert np.sum(val['task_labels'] == 'gsm8k') == np.sum(val['task_labels'] == 'bbh') == 100

    source_root = root / 'research/conditional_gain_20261106'
    state_path = source_root / 'private/FITTED_CANDIDATE_STATES.pkl'
    source_code = source_root / 'repo/routing_idea2/research/conditional_gain_20261106/code'
    selection_path = source_code.parent / 'TRAIN_ONLY_STRUCTURE_FEATURE_SELECTION.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8-sig'))
    assert selection['structure_winners']['direct20'] == TARGET_NAME
    assert selection['validation_not_used_for_this_selection'] is True
    with state_path.open('rb') as stream:
        pack = RouterUnpickler(stream).load()
    saved = pack['states'][TARGET_NAME]
    transform, W = saved['transform'], saved['state']['W']
    assert saved['config']['family'] == 'direct20' and saved['config']['feature'] == 'pca16'
    assert float(saved['config']['alpha']) == ALPHA_RAW
    assert np.array_equal(transform.train_ids, train['ids'])
    assert np.array_equal(pack['defaults_train_ids'], train['ids'])
    assert transform.kind == 'pca16' and W.shape == (17, 20)
    phi = transform.transform(val['embeddings'])
    prediction = phi @ W
    default_action = int(choose(Y.mean(axis=(0, 1))))
    base = np.full(200, default_action, dtype=int)
    features, b, c, m, active, overlap = paired_features(val['embeddings'], prediction, base)

    # Independent algebra check leaves restored weights and predictions untouched.
    refit = fit_router(E, Y)
    refit_prediction, _, _ = predict_router(refit, val['embeddings'])
    restoration_error = float(np.max(np.abs(prediction - refit_prediction)))
    assert restoration_error < 1e-10
    X = transform.transform(E)
    raw_X = np.repeat(X, 3, axis=0)
    raw_Y = Y.reshape(1800, 20)
    raw_W = np.linalg.solve(raw_X.T @ raw_X + ALPHA_RAW * np.eye(17), raw_X.T @ raw_Y)
    alpha_equivalence_error = float(np.max(np.abs(X @ raw_W - X @ W)))
    assert alpha_equivalence_error < 1e-10

    fold_path = source_root / 'private/FOLDS_PRIVATE.npz'
    assert sha(fold_path) == FOLDS_SHA
    with np.load(fold_path, allow_pickle=False) as archive:
        splits = [(archive[f'fold{f}_train'].astype(int), archive[f'fold{f}_holdout'].astype(int))
                  for f in range(5)]
    oof_pred, oof_b, fold_idx = oof_predictions(E, Y, train['task'], splits)
    oof_features, oof_b, oof_c, oof_m, oof_active, _ = paired_features(E, oof_pred, oof_b)
    oof_D, oof_d = legal_pair_rewards(Y, oof_b, oof_c)
    aux = moments_fit(oof_features, oof_m, oof_d, oof_active)
    raw_moments, safe_moments = moments_predict(aux, features, active)

    # Strict nested historical diagnostic. The held-out question never influences
    # its router, default, target residuals, auxiliary fit, or standardization.
    nested_prediction = np.zeros((600, 2))
    nested_rows = []
    for fold, (history, held) in enumerate(splits):
        local_E, local_Y = E[history], Y[history]
        stratifier = StratifiedKFold(5, shuffle=True, random_state=20261113 + fold)
        inner = list(stratifier.split(np.arange(len(history)), train['task'][history]))
        local_pred, local_b, _ = oof_predictions(local_E, local_Y, train['task'][history], inner)
        local_features, local_b, local_c, local_m, local_active, _ = paired_features(local_E, local_pred, local_b)
        _, local_d = legal_pair_rewards(local_Y, local_b, local_c)
        local_aux = moments_fit(local_features, local_m, local_d, local_active)
        _, nested_prediction[held] = moments_predict(local_aux, oof_features[held], oof_active[held])
        nested_rows.append(dict(fold=fold, historical_questions=len(history), held_questions=len(held),
                                inner_splits=5, inner_seed=20261113 + fold,
                                held_out_fitting_or_feedback_access=False))

    predicted_cost, cost_means = cost_predictions(train['cost'], b, c)
    np.savez_compressed(private / 'FROZEN_INPUTS.npz',
        N=np.array(200), generation_seeds=np.array([42, 137, 2024]), teams=TEAMS,
        val_ids=val['question_ids'], canonical_hashes=val['canonical_hashes'],
        task=val['task_labels'], sub=val['subtask_labels'], embeddings=val['embeddings'],
        phi=phi, predicted_scores=prediction, auxiliary_features=features, b=b, c=c, m=m,
        active=active, overlap=overlap, vhat_raw=raw_moments[:, 0], vhat=safe_moments[:, 0],
        rawmom_raw=raw_moments[:, 1], rawmom=safe_moments[:, 1],
        cost_prediction=predicted_cost['tokens'],
        **{'predicted_' + name: value for name, value in predicted_cost.items()})
    np.savez_compressed(private / 'TRAIN_OOF.npz', ids=train['ids'], fold=fold_idx,
        predicted_scores=oof_pred, b=oof_b, c=oof_c, m=oof_m, active=oof_active,
        D=oof_D, d=oof_d, residual=oof_d - oof_m, auxiliary_features=oof_features,
        nested_moment_predictions=nested_prediction)
    np.savez_compressed(private / 'FROZEN_MODEL.npz',
        pca_mean=transform.pca.mean_, pca_components=transform.pca.components_,
        scaler_mean=transform.scaler.mean_, scaler_scale=transform.scaler.scale_, W=W,
        global_default=np.array(default_action), raw_alpha=np.array(ALPHA_RAW),
        mean_alpha=np.array(ALPHA_RAW / 3), train_ids=train['ids'],
        aux_scaler_mean=aux[0].mean_, aux_scaler_scale=aux[0].scale_,
        aux_coef=aux[1].coef_, aux_intercept=aux[1].intercept_, aux_alpha=np.array(AUX_ALPHA),
        **{'cost_mean_' + name: value for name, value in cost_means.items()})

    proposal_path = root / 'artifacts/private/routing_redesign_20261104/PARTITION_PROPOSAL_v2.json'
    roles = json.loads(proposal_path.read_text(encoding='utf-8-sig'))['roles']
    role_summary = {}
    for name, info in roles.items():
        if not isinstance(info, dict) or 'records' not in info:
            continue
        ids = {row['question_id'] for row in info['records']}
        role_summary[name] = dict(count=len(ids), old_split=info.get('old_split'),
                                  validation_id_overlap=len(ids & set(val['question_ids'])))
    assert role_summary['selector_validation']['validation_id_overlap'] == 200
    old_diagnostics = [v for v in role_summary.values() if v['old_split'] == 'e1_diagnostic_evaluation']
    assert len(old_diagnostics) == 1 and old_diagnostics[0]['validation_id_overlap'] == 0
    target_residual = (oof_d - oof_m) ** 2
    diagnostic = dict(questions=600, action_disagreements=int(oof_active.sum()),
        residual_second_moment=float(target_residual[oof_active].mean()),
        raw_second_moment=float((oof_d[oof_active] ** 2).mean()),
        nested_residual_moment_MSE=float(np.mean((nested_prediction[oof_active, 0] - target_residual[oof_active]) ** 2)),
        nested_raw_moment_MSE=float(np.mean((nested_prediction[oof_active, 1] - oof_d[oof_active] ** 2) ** 2)),
        nested_folds=nested_rows,
        in_sample_uncertainty_is_not_a_calibrated_true_error_bound=True)
    npz_hashes = {name: sha(private / name) for name in ('FROZEN_INPUTS.npz', 'FROZEN_MODEL.npz', 'TRAIN_OOF.npz')}
    evidence = dict(schema='B1_actor_auxiliary_freeze_v1', status='FROZEN_BEFORE_VALIDATION_OUTCOMES',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), new_llm_calls=0,
        validation_reward_payloads_opened=0, fresh400_reward_payloads_opened=0,
        sealed200_reward_payloads_opened=0, validation_actual_costs_opened=0,
        questions=200, train_questions=600, generation_seeds=[42, 137, 2024],
        router=dict(name=TARGET_NAME, restored=True, source_sha256=sha(state_path),
                    weights_train_questions=600, raw_alpha=ALPHA_RAW, mean_alpha=ALPHA_RAW / 3,
                    components=16, constant_penalized=True, task_or_subtask_routing_input=False,
                    tie_rule='numpy exact argmax, first lexicographic combination',
                    restored_vs_independent_closed_form_error=restoration_error,
                    raw_repeat_vs_mean_objective_error=alpha_equivalence_error),
        default=dict(action=default_action, members=TEAMS[default_action].tolist(),
                     source='all train600, three-repeat mean then global argmax'),
        active_questions=int(active.sum()), structural_zero_questions=int((~active).sum()),
        auxiliary=dict(model='Ridge', alpha=AUX_ALPHA, tuning=False, fit_intercept=True,
            targets=['question-mean paired residual squared', 'question-mean paired reward squared'],
            training_rows='active disagreements in question-grouped five-fold OOF train600',
            input_dimension=features.shape[1], task_or_subtask_features=False,
            standardization='auxiliary fitting rows only', residual_floor=1e-6,
            residual_cap=4., raw_moment_cap=1., nested_diagnostic=diagnostic),
        cost_model=dict(source='train600 only, average each member/combo over question and generation repeats',
            final_cost_unit='prompt plus completion tokens',
            distinct_pair_member_calls='3 times union of members', distinct_pair_aggregate_calls=6,
            scoring_ops=6, gold_reveals=1, structural_zero_evaluation_cost=0,
            validation_output_lengths_or_correctness_used=False,
            projected_total_tokens=float(predicted_cost['tokens'].sum()),
            projected_total_member_calls=int(predicted_cost['member_calls'].sum()),
            projected_total_aggregate_calls=int(predicted_cost['aggregator_calls'].sum())),
        data_lineage=dict(bundle_sha256=BUNDLE_SHA, training_refs=train_refs,
            validation_inputs=validation_ref,
            validation_outcomes_metadata_only=bundle['outcomes']['selector_validation'],
            validation_receipt_metadata_only=bundle['matrix_receipts']['selector_validation'],
            partition_sha256=sha(proposal_path), role_summary=role_summary,
            validation_is_earlier_aggregator200=False,
            validation_is_new_independent_test=False, train_validation_ID_and_canonical_disjoint=True),
        sources={str(p.relative_to(root)): sha(p) for p in
                 (source_code / 'common.py', source_code / 'run_cpu_ablation.py', selection_path)},
        files_sha256=npz_hashes, script_sha256=sha(__file__),
        wall_seconds=time.perf_counter() - started)
    put(private / 'FREEZE_RECEIPT.json', evidence)
    public_evidence = {key: value for key, value in evidence.items() if key != 'data_lineage'}
    public_evidence['data_lineage'] = {key: value for key, value in evidence['data_lineage'].items()
        if key not in ('training_refs', 'validation_inputs', 'validation_outcomes_metadata_only',
                       'validation_receipt_metadata_only')}
    public_evidence['data_lineage']['train_input_sha256'] = train_refs['inputs']['sha256']
    public_evidence['data_lineage']['train_outcome_sha256'] = train_refs['outcomes']['sha256']
    public_evidence['data_lineage']['validation_input_sha256'] = validation_ref['sha256']
    public_evidence['data_lineage']['validation_outcome_sha256_expected'] = bundle['outcomes']['selector_validation']['sha256']
    put(public / 'ASSET_FREEZE_PUBLIC.json', public_evidence)
    put(private / 'ACCESS_LOG.json', dict(
        reward_payloads_decoded=['selector_train'],
        input_payloads_decoded=['selector_train', 'selector_validation'],
        existing_model_state_payload_decoded='historical train600 fitted model states',
        validation_rewards_and_costs_decoded=False, fresh400_or_sealed200_payloads_decoded=False,
        inference_scope='application-level function and file-access boundary, not OS sandbox isolation'))
    print(json.dumps(dict(status=evidence['status'], active=int(active.sum()),
                          default=default_action, hashes=npz_hashes,
                          wall_seconds=evidence['wall_seconds']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
