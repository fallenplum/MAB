"""Variable-size, question-cross-fitted posterior means with strict provenance."""
from pathlib import Path
import time
from e9_common import (ASSETS, E8, NS, ROOT, archive, array_hash, fold_packs,
                       inner_splits, now, permute_region, put, read, sha, subset, verify_sources)
import numpy as np
import joint_noise_model as model

OLD_MODEL_HASH = 'cc3f4b9d42656e37d50f6858f082fc93f5799196073ef36d653c8c7afc8519ff'


def verify_model_source():
    path = NS / 'private/runtime/noise/joint_noise_model.py'
    if sha(path) != OLD_MODEL_HASH:
        raise AssertionError('Frozen NS posterior operator changed')
    return dict(path=str(path), sha256=OLD_MODEL_HASH)


def build_mean(region, outer, folder, context, forbidden_ids=()):
    """Only this fitting region supplies noise, prior and observations."""
    if set(region['ids'].tolist()) & set(forbidden_ids):
        raise ValueError('Teacher fitting region overlaps evaluation identities')
    verify_model_source()
    folder = Path(folder)
    filename = folder / 'TEACHER_MEAN.npz'
    receipt_path = folder / 'FROZEN_RECEIPT.json'
    expected = dict(context=context, outer=outer, n_questions=len(region['ids']),
        question_ids_hash=array_hash(region['ids']), indices_hash=array_hash(region['global_indices']),
        raw_rewards_hash=array_hash(region['Y']), model_source_sha256=OLD_MODEL_HASH,
        runner_source_sha256=sha(__file__))
    if receipt_path.exists():
        receipt = read(receipt_path)
        if any(receipt[key] != value for key, value in expected.items()) or sha(filename) != receipt['artifact_sha256'] or sha(folder / 'PRIOR_AND_SPLITS.npz') != receipt['prior_artifact_sha256']:
            raise AssertionError('Teacher cache identity or source differs')
        saved = archive(filename)
        if not np.array_equal(saved['history_indices'], region['global_indices']):
            raise AssertionError('Teacher cache question order differs')
        return saved['posterior_mean'], receipt
    if folder.exists():
        raise FileExistsError('Unsealed teacher directory requires explicit failure recovery')
    folder.mkdir(parents=True)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    n = len(region['ids'])
    means, fallback = np.empty((n, 20)), np.zeros(n, bool)
    rhat, ess = np.empty(n), np.empty(n)
    splits = inner_splits(region, np.arange(n), 20261113)
    logs, private_fits = [], {}
    for inner, (fitting_rows, target_rows) in enumerate(splits):
        if set(fitting_rows) & set(target_rows):
            raise AssertionError('Teacher hyperparameters contain their target')
        prior = model.fit_working_prior(region['Y'][fitting_rows])
        seed = 2026111400 + 1000 * outer + 100 * inner + 3
        sampled = model.sample_bounded(prior, region['Y'][target_rows], seed)
        values = sampled['draws'].mean(axis=1)
        rejected = ~sampled['accepted']
        values[rejected] = region['Y'][target_rows][rejected].mean(axis=1)
        means[target_rows] = values
        fallback[target_rows] = rejected
        rhat[target_rows] = sampled['rhat'].max(axis=1)
        ess[target_rows] = sampled['ess'].min(axis=1)
        for key in ('mean', 'noise', 'prior_covariance', 'between_raw'):
            private_fits[f'inner{inner}_{key}'] = prior[key]
        private_fits[f'inner{inner}_fit_indices'] = region['global_indices'][fitting_rows]
        private_fits[f'inner{inner}_target_indices'] = region['global_indices'][target_rows]
        logs.append(dict(inner=inner, fitting_questions=len(fitting_rows),
            target_questions=len(target_rows), fitting_ids_hash=array_hash(region['ids'][fitting_rows]),
            target_ids_hash=array_hash(region['ids'][target_rows]), seed=seed,
            fallback=int(rejected.sum()), maximum_Rhat=float(sampled['rhat'].max()),
            minimum_ESS=float(sampled['ess'].min())))
    if not np.isfinite(means).all() or means.min() < 0 or means.max() > 1:
        raise AssertionError('Invalid teacher posterior means')
    np.savez_compressed(filename, posterior_mean=means, history_indices=region['global_indices'],
        history_ids=region['ids'], fallback_mask=fallback, rhat_max=rhat, ess_min=ess)
    np.savez_compressed(folder / 'PRIOR_AND_SPLITS.npz', **private_fits)
    receipt = dict(**expected, created_utc=now(), artifact_sha256=sha(filename),
        prior_artifact_sha256=sha(folder / 'PRIOR_AND_SPLITS.npz'), inner_fits=logs,
        total_prior_fits=3, total_sampling_calls=3, draws_per_question=512,
        fallback_questions=int(fallback.sum()), cpu_seconds=time.process_time()-started_cpu,
        wall_seconds=time.perf_counter()-started_wall,
        teacher_hyperparameters_exclude_each_target=True,
        evaluation_reward_access=False, physical_LLM_calls=0)
    put(receipt_path, receipt)
    return means, receipt


def reuse_outer_mean(region, outer, folder):
    verify_model_source()
    source = NS / f'private/runtime/noise/private/fold{outer}'
    path = source / 'POSTERIOR_TRAINING.npz'
    old_receipt = read(source / 'POSTERIOR_FREEZE_RECEIPT.json')
    if sha(path) != old_receipt['posterior_sha256']:
        raise AssertionError('Old posterior artifact changed')
    history_path = E8 / f'private/qualification/fold_packs/fold{outer}/history480.npz'
    if sha(history_path) != old_receipt['input_history_sha256']:
        raise AssertionError('Old posterior history differs')
    original = archive(history_path)
    if not np.array_equal(region['Y'], original['Y']):
        raise AssertionError('Outer posterior cannot label a different reward assignment')
    values = archive(path)
    if not np.array_equal(values['history_indices'], region['global_indices']) or not np.array_equal(values['history_ids'], region['ids']):
        raise AssertionError('Old posterior identity mapping differs')
    if old_receipt['source_sha256']['joint_noise_model.py'] != OLD_MODEL_HASH:
        raise AssertionError('Old posterior source differs')
    for name, digest in old_receipt['source_sha256'].items():
        if sha(NS / 'private/runtime/noise' / name) != digest:
            raise AssertionError('Old sampler source provenance differs')
    means = values['posterior_mean']
    if means.shape != (len(region['ids']), 20) or not np.isfinite(means).all():
        raise AssertionError('Old mean target shape differs')
    folder = Path(folder)
    expected = dict(status='REUSED_FROZEN_512_DRAW_MEAN', source_path=str(path),
        source_sha256=sha(path), source_receipt_sha256=sha(source / 'POSTERIOR_FREEZE_RECEIPT.json'),
        question_ids_hash=array_hash(region['ids']), indices_hash=array_hash(region['global_indices']),
        raw_rewards_hash=array_hash(region['Y']), mean_hash=array_hash(means),
        n_questions=len(means), new_prior_fits=0, new_sampling_calls=0,
        posterior_mean_not_eight_draw_mean=True, code_source_sha256=sha(__file__))
    target = folder / 'REUSE_RECEIPT.json'
    if target.exists() and read(target) != expected:
        raise AssertionError('Outer mean reuse receipt changed')
    put(target, expected)
    return means, expected


def prepare_fold(outer, destination=ROOT):
    verify_sources('TEACHER_SOURCE_LOCK.json')
    history, evaluation, splits, source = fold_packs(outer)
    base = Path(destination) / f'private/teachers/fold{outer}'
    reuse_outer_mean(history, outer, base / 'outer_main')
    for inner, (fit, held) in enumerate(splits):
        region = subset(history, fit)
        build_mean(region, outer, base / f'inner{inner}', f'outer{outer}/inner{inner}',
                   tuple(history['ids'][held]) + tuple(evaluation['ids']))
    from e9_common import PERMUTATION_SEEDS
    for seed in PERMUTATION_SEEDS:
        region, mapping = permute_region(history, seed)
        folder = base / f'permutation{seed}'
        build_mean(region, outer, folder, f'outer{outer}/permutation{seed}', evaluation['ids'])
        mapping_path = folder / 'PERMUTATION.npz'
        if not mapping_path.exists():
            np.savez_compressed(mapping_path, mapping=mapping, ids=history['ids'],
                source_reward_ids=history['ids'][mapping])
    put(base / 'FOLD_TEACHERS_READY.json', dict(outer=outer, source=source,
        outer_mean_reused=True, new_regions=6, new_prior_fits=18, numerical_model=OLD_MODEL_HASH))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--out', type=Path, default=ROOT)
    args = parser.parse_args()
    prepare_fold(args.fold, args.out)
    print('TEACHERS_READY', args.fold, flush=True)
