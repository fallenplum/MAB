"""Fixed E10 PCA16/Ridge heads with sealed input-only outer predictions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ASSETS = Path(r'E:\语言模型组合路由')
ROOT = ASSETS/'research/e10_input_prefill_matching_audit'
E9 = ASSETS/'research/e9_consistent_cost_sensitive_learning'
sys.path.insert(0, str(E9/'public/code'))
from e9_common import ACTION_KEYS, fold_packs, fit_features, permute_region
from csce_operators import semantic_argmax, validate_training_inputs
from e8_assets import ridge_weights


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    protocol_path = ROOT/'public/E10_PROTOCOL.json'
    protocol_sha = sha(protocol_path)
    freeze = json.loads((ROOT/'private/PROTOCOL_FREEZE_RECEIPT.json').read_text())
    assert freeze['protocol_sha256'] == protocol_sha
    assert freeze['sources']['e10_train.py'] == sha(Path(__file__))
    receipt = json.loads((ROOT/'private/features/FEATURE_FREEZE_RECEIPT.json').read_text())
    assert receipt['protocol_sha256'] == protocol_sha
    feature_path = ROOT/'private/features/PREFILL_FEATURES_FROZEN.npz'
    assert sha(feature_path) == receipt['feature_sha256']
    feature = np.load(feature_path, allow_pickle=False)
    assert np.array_equal(feature['global_indices'], np.arange(600))
    variants = ['R_full']
    if receipt['n_truncated'] > 0 and receipt['r_match_available']:
        variants.append('R_match')
    names = variants + [f'R_full_perm{seed}' for seed in (20261114091,20261114092,20261114093)]
    all_scores = {name:np.empty((600,20)) for name in names}
    all_actions = {name:np.empty(600, dtype=int) for name in names}
    folds = np.full(600, -1, dtype=int)
    identity = np.empty(600, dtype='U256')
    logs = []
    for outer in range(5):
        history, evaluation, _, sources = fold_packs(outer)
        validate_training_inputs(history,evaluation)
        fit_rows, eval_rows = history['global_indices'], evaluation['global_indices']
        assert not set(history['ids']) & set(evaluation['ids'])
        assert not set(fit_rows) & set(eval_rows)
        folds[eval_rows] = outer
        identity[eval_rows] = evaluation['ids']
        for variant in variants:
            embeddings = feature['full'] if variant == 'R_full' else feature['matched']
            start = time.perf_counter()
            feature_map = fit_features(embeddings[fit_rows])
            fit_phi = feature_map.phi(embeddings[fit_rows])
            eval_phi = feature_map.phi(embeddings[eval_rows])
            feature_seconds = time.perf_counter()-start
            actual_names = [variant]
            if variant == 'R_full':
                actual_names += [f'R_full_perm{seed}' for seed in (20261114091,20261114092,20261114093)]
            expected_diagonal = len(fit_rows)*np.r_[1.,feature_map.scaler.var_/feature_map.scaler.scale_**2]
            gram_error = float(abs(fit_phi.T@fit_phi-np.diag(expected_diagonal)).max())
            assert gram_error < 1e-7
            for name in actual_names:
                folder = ROOT/f'private/models/fold{outer}/{name}'
                folder.mkdir(parents=True, exist_ok=True)
                if (folder/'FROZEN_RECEIPT.json').exists():
                    old = json.loads((folder/'FROZEN_RECEIPT.json').read_text())
                    assert old['protocol_sha256'] == protocol_sha
                    assert sha(folder/'MODEL.npz') == old['model_sha256']
                    assert sha(folder/'PREDICTIONS.npz') == old['predictions_sha256']
                    saved = np.load(folder/'PREDICTIONS.npz', allow_pickle=False)
                    all_scores[name][eval_rows] = saved['scores']
                    all_actions[name][eval_rows] = saved['actions']
                    logs.append(old)
                    continue
                fit_history = history
                mapping = None
                if '_perm' in name:
                    fit_history, mapping = permute_region(history, int(name.split('_perm')[1]))
                labels = fit_history['Y'].mean(axis=1)
                cpu_start, wall_start = time.process_time(), time.perf_counter()
                coefficient = ridge_weights(fit_phi, labels, raw_alpha=1000.)
                cpu, wall = time.process_time()-cpu_start, time.perf_counter()-wall_start
                error = fit_phi@coefficient-labels
                regularization = 1000./(3*len(fit_rows)*20)
                gradient = 2*fit_phi.T@error/(len(fit_rows)*20)+2*regularization*coefficient
                gradient_max = float(abs(gradient).max())
                assert gradient_max < 1e-8
                prediction_start = time.perf_counter()
                scores = eval_phi@coefficient
                prediction_wall = time.perf_counter()-prediction_start
                actions = semantic_argmax(scores, ACTION_KEYS)
                restore = ((embeddings[eval_rows].astype(float)-feature_map.pca.mean_)@feature_map.pca.components_.T
                    -feature_map.scaler.mean_)/feature_map.scaler.scale_
                restored_scores = np.column_stack((np.ones(len(restore)),restore))@coefficient
                restoration_error = float(abs(scores-restored_scores).max())
                assert restoration_error < 1e-12
                assert np.array_equal(semantic_argmax(restored_scores,ACTION_KEYS), actions)
                np.savez_compressed(folder/'MODEL.npz', coefficient=coefficient, **feature_map.arrays())
                np.savez_compressed(folder/'PREDICTIONS.npz', scores=scores, actions=actions,
                    evaluation_ids=evaluation['ids'], evaluation_indices=eval_rows)
                if mapping is not None:
                    np.savez_compressed(folder/'PERMUTATION.npz', mapping=mapping,
                        fitting_indices=fit_rows, fitting_ids=history['ids'])
                record = dict(name=name, outer_fold=outer, completed_utc=now(),
                    protocol_sha256=protocol_sha, source_sha256=sha(Path(__file__)),
                    input_feature_sha256=receipt['feature_sha256'], history_sources=sources,
                    fit_ids_sha256=array_hash(history['ids']), evaluation_ids_sha256=array_hash(evaluation['ids']),
                    labels_sha256=array_hash(labels), model_sha256=sha(folder/'MODEL.npz'),
                    predictions_sha256=sha(folder/'PREDICTIONS.npz'),
                    parameters=340, fitting_questions=len(fit_rows), evaluation_questions=len(eval_rows),
                    raw_alpha=1000., lambda_multiplier=10., regularization=regularization,
                    data_loss=float(np.sum(error*error)/(len(fit_rows)*20)),
                    penalty=float(regularization*np.sum(coefficient*coefficient)),
                    gradient_infinity_norm=gradient_max, gram_max_error=gram_error,
                    design_rank=int(np.linalg.matrix_rank(fit_phi)),
                    exact_zero_variance_pca_components=int(sum(feature_map.scaler.var_==0)),
                    restoration_max_error=restoration_error, cpu_seconds=cpu, wall_seconds=wall,
                    evaluation_head_batch_wall_seconds=prediction_wall,
                    preprocessing_wall_seconds=feature_seconds if name==variant else 0.,
                    coefficient_norm=float(np.linalg.norm(coefficient)),
                    evaluation_rewards_used=False, inner_searches=0, constants_penalized=True)
                put(folder/'FROZEN_RECEIPT.json', record)
                logs.append(record)
                all_scores[name][eval_rows] = scores
                all_actions[name][eval_rows] = actions
                print('HEAD_FROZEN', outer,name,flush=True)
    assert np.all(folds>=0) and len(set(identity)) == 600
    destination = ROOT/'private/ALL_E10_PREDICTIONS_FROZEN.npz'
    assert not destination.exists()
    np.savez_compressed(destination, fold=folds, ids=identity,
        **{'scores__'+name:scores for name,scores in all_scores.items()},
        **{'actions__'+name:actions for name,actions in all_actions.items()})
    put(ROOT/'private/ALL_PREDICTIONS_FREEZE_RECEIPT.json', dict(status='COMPLETE', frozen_utc=now(),
        protocol_sha256=protocol_sha, prediction_sha256=sha(destination), names=names,
        formal_head_fits=len(logs), outer_reward_reads=0, heads=logs,
        labels='original three-generation mean, one row per question', new_teachers=0))
    print('ALL_E10_PREDICTIONS_FROZEN',flush=True)


if __name__=='__main__':
    main()
