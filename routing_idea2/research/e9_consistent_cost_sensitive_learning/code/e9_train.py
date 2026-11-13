"""Locked E9 nested training and all matched content controls."""
from pathlib import Path
import time
from e9_common import (ACTION_KEYS, MULTIPLIERS, PERMUTATION_SEEDS, ROOT, array_hash,
    archive, fit_features, fold_packs, now, permute_region, put, read, sha, subset, verify_sources)
import numpy as np
from csce_operators import semantic_argmax, validate_training_inputs
from e9_heads import fit_head, diagnose_head, routed_accuracy
from e9_teachers import build_mean, reuse_outer_mean


def save_fit(folder, family, phi, labels, mult, evaluation_phi, fit_ids, evaluation_ids,
             feature_state, teacher_ref, derived=None):
    folder = Path(folder)
    expected = dict(family=family, multiplier=mult, fit_ids_hash=array_hash(fit_ids),
        evaluation_ids_hash=array_hash(evaluation_ids), labels_hash=array_hash(labels),
        training_source_sha256=sha(__file__), head_source_sha256=sha(Path(__file__).parent/'e9_heads.py'),
        teacher_reference=teacher_ref)
    receipt_path = folder / 'FROZEN_RECEIPT.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if any(receipt[key] != value for key,value in expected.items()):
            raise AssertionError('Existing fit source or data differs')
        if sha(folder/'MODEL.npz') != receipt['model_sha256'] or sha(folder/'PREDICTIONS.npz') != receipt['prediction_sha256'] or sha(folder/'FIT_LOG.json') != receipt['fit_log_sha256']:
            raise AssertionError('Frozen fit artifact changed')
        return archive(folder/'MODEL.npz')['coefficient'], archive(folder/'PREDICTIONS.npz')['scores'], read(folder/'FIT_LOG.json')
    if folder.exists():
        raise FileExistsError('Unsealed model attempt requires explicit recovery')
    folder.mkdir(parents=True)
    if derived is None:
        coefficient, diagnostic = fit_head(family, phi, labels, mult)
        diagnostic['actual_head_fit'] = True
    else:
        started_cpu, started_wall = time.process_time(), time.perf_counter()
        n = len(phi)
        coefficient = derived * (n + 100/3) / (n + 100*mult/3)
        diagnostic = diagnose_head(family, coefficient, phi, labels, mult)
        diagnostic.update(actual_head_fit=False, solver='verified positive Ridge scale cache',
            derived_from_multiplier=1., cpu_seconds=time.process_time()-started_cpu,
            wall_seconds=time.perf_counter()-started_wall,
            gram_max_error=float(abs(phi.T@phi-n*np.eye(17)).max()))
        if not diagnostic['accepted'] or diagnostic['gram_max_error'] > 1e-8:
            coefficient, diagnostic = fit_head(family, phi, labels, mult)
            diagnostic.update(actual_head_fit=True, scale_cache_fallback=True)
    scores = evaluation_phi @ coefficient
    np.savez_compressed(folder/'MODEL.npz', coefficient=coefficient, **feature_state)
    np.savez_compressed(folder/'PREDICTIONS.npz', scores=scores,
        actions=semantic_argmax(scores,ACTION_KEYS), evaluation_ids=evaluation_ids)
    put(folder/'FIT_LOG.json', diagnostic)
    if not diagnostic['accepted']:
        put(folder/'FAILED_ATTEMPT.json', dict(**expected, failed_utc=now(), diagnostic=diagnostic))
        raise RuntimeError('Predictor optimization failed qualification '+str(folder))
    put(receipt_path, dict(**expected, frozen_utc=now(), model_sha256=sha(folder/'MODEL.npz'),
        prediction_sha256=sha(folder/'PREDICTIONS.npz'), fit_log_sha256=sha(folder/'FIT_LOG.json'),
        data_normalization='n*20', constant_penalized=True, evaluation_rewards_used=False))
    return coefficient, scores, diagnostic


def run_fold(outer):
    verify_sources('HEAD_SOURCE_LOCK.json')
    if not (ROOT/f'private/teachers/fold{outer}/FOLD_TEACHERS_READY.json').exists():
        raise RuntimeError('All legal teacher regions must be ready')
    history, evaluation, splits, source = fold_packs(outer)
    n = len(history['ids'])
    fold = ROOT / f'private/models/fold{outer}'
    fold.mkdir(parents=True, exist_ok=True)
    teachers = ROOT / f'private/teachers/fold{outer}'
    predictions = {family:np.empty((3,n,20)) for family in ('B','C','D')}
    logs = {family:[] for family in predictions}
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    for inner,(fit,held) in enumerate(splits):
        fitting, validation = subset(history,fit), subset(history,held)
        validation_inputs = {key:value for key,value in validation.items() if key!='Y'}
        validate_training_inputs(fitting,validation_inputs)
        fmap = fit_features(fitting['E'])
        phi, validation_phi = fmap.phi(fitting['E']), fmap.phi(validation['E'])
        posterior, teacher_receipt = build_mean(fitting,outer,teachers/f'inner{inner}',
            f'outer{outer}/inner{inner}',tuple(validation['ids'])+tuple(evaluation['ids']))
        ref = dict(path=str(teachers/f'inner{inner}/TEACHER_MEAN.npz'),
            sha256=teacher_receipt['artifact_sha256'], receipt_sha256=sha(teachers/f'inner{inner}/FROZEN_RECEIPT.json'))
        empirical = fitting['Y'].mean(axis=1)
        base = None
        for family in ('B','C','D'):
            labels = empirical if family=='C' else posterior
            order = (1.,.1,10.) if family=='B' else MULTIPLIERS
            for mult in order:
                coefficient,scores,diagnostic = save_fit(fold/f'inner{inner}/{family}_{mult:g}',
                    family,phi,labels,mult,validation_phi,fitting['ids'],validation['ids'],fmap.arrays(),
                    None if family=='C' else ref,derived=base if family=='B' and mult!=1. else None)
                if family=='B' and mult==1.:
                    base = coefficient
                if family=='B' and mult!=1. and not np.array_equal(
                    semantic_argmax(scores,ACTION_KEYS), semantic_argmax(validation_phi@base,ACTION_KEYS)):
                    raise AssertionError('Positive Ridge scale changed action choices')
                predictions[family][MULTIPLIERS.index(mult),held] = scores
                logs[family].append({**diagnostic, 'inner':inner, 'multiplier':mult})
    selected = {}
    for family,values in predictions.items():
        accuracies = [routed_accuracy(scores,history['Y'],history['task'],ACTION_KEYS)['balanced_accuracy'] for scores in values]
        maximum = max(accuracies)
        index = max((i for i,v in enumerate(accuracies) if v>=maximum-1e-12),key=lambda i:MULTIPLIERS[i])
        selected[family] = dict(multiplier=MULTIPLIERS[index],inner_accuracy=accuracies[index],
            grid_accuracy=accuracies,grid_boundary=index in (0,2),inner_heads=logs[family])
    selection_path = fold/'SELECTED_PARAMETERS_FROZEN.json'
    if selection_path.exists():
        old_selection = read(selection_path)
        if old_selection['selected'] != selected or old_selection['source'] != source:
            raise AssertionError('Frozen inner selection changed')
    else:
        put(selection_path,dict(selected=selected, frozen_utc=now(), source=source,
            selected_without_outer_rewards=True, inner_split_seed=20261113+outer))
    fmap = fit_features(history['E'])
    phi,evaluation_phi = fmap.phi(history['E']),fmap.phi(evaluation['E'])
    posterior, reused = reuse_outer_mean(history,outer,teachers/'outer_main')
    raw, action, references = {},{},{}
    for family in ('B','C','D'):
        labels = history['Y'].mean(axis=1) if family=='C' else posterior
        _,scores,_ = save_fit(fold/f'outer/{family}',family,phi,labels,selected[family]['multiplier'],
            evaluation_phi,history['ids'],evaluation['ids'],fmap.arrays(), None if family=='C' else reused)
        raw[family] = scores
    for seed in PERMUTATION_SEEDS:
        permuted,mapping = permute_region(history,seed)
        m,teacher_receipt = build_mean(permuted,outer,teachers/f'permutation{seed}',
            f'outer{outer}/permutation{seed}',evaluation['ids'])
        ref = dict(path=str(teachers/f'permutation{seed}/TEACHER_MEAN.npz'),
            sha256=teacher_receipt['artifact_sha256'], permutation_seed=seed)
        for family in ('B','C','D'):
            labels = permuted['Y'].mean(axis=1) if family=='C' else m
            name = f'{family}_perm{seed}'
            _,scores,_ = save_fit(fold/f'controls/{name}',family,phi,labels,selected[family]['multiplier'],
                evaluation_phi,history['ids'],evaluation['ids'],fmap.arrays(),None if family=='C' else ref)
            raw[name] = scores
    output = fold/'ALL_OUTER_PREDICTIONS_FROZEN.npz'
    if output.exists():
        old = archive(output)
        if any(not np.array_equal(old['scores__'+key],value) for key,value in raw.items()):
            raise AssertionError('Refusing different frozen outer predictions')
    else:
        np.savez_compressed(output, evaluation_indices=evaluation['global_indices'],
            evaluation_ids=evaluation['ids'], **{'scores__'+key:value for key,value in raw.items()},
            **{'actions__'+key:semantic_argmax(value,ACTION_KEYS) for key,value in raw.items()})
    receipt_path = fold/'FOLD_FROZEN_RECEIPT.json'
    if not receipt_path.exists():
        put(receipt_path,dict(fold=outer, completed_utc=now(),
            predictions_sha256=sha(output), training_source_sha256=sha(__file__), source=source,
            variants=list(raw), models_and_controls_complete=True, outer_rewards_read=False,
            cpu_seconds=time.process_time()-started_cpu, wall_seconds=time.perf_counter()-started_wall))
    elif read(receipt_path)['predictions_sha256'] != sha(output):
        raise AssertionError('Frozen fold receipt differs')
    print('FOLD_ALL_MODELS_AND_CONTROLS_FROZEN',outer,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,choices=range(5),required=True)
    args=parser.parse_args()
    run_fold(args.fold)
