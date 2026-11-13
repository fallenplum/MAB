"""Fold-local empirical Gaussian noise study with bounded latent draws."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from scipy.stats import truncnorm
import joint_noise_model as model

DEFAULT_ASSETS = Path(r'E:\语言模型组合路由')
MODULE = Path(__file__).resolve().parent
SOURCE_NAMES = ('joint_noise_model.py', 'run_joint_noise.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()) + '\n', encoding='utf-8')


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def freeze(out, assets):
    filename = out / 'public/JOINT_NOISE_PROTOCOL.json'
    sources = {name: sha(MODULE / name) for name in SOURCE_NAMES}
    if filename.exists():
        previous = read(filename)
        if previous['source_sha256'] != sources:
            raise AssertionError('Frozen noise sources changed')
        return previous
    protocol = dict(status='FROZEN_BEFORE_REAL_NOISE_REWARD_ACCESS',
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        assets_root=str(assets), target='bounded20-dimensional Gaussian working posterior',
        prior='constant historical mean, Gaussian between-question covariance',
        noise='pooled within-question sample covariance, n independent question clusters',
        shrinkage=model.SHRINKAGE, numerical_ridge=model.RIDGE,
        between_projection='PSD projection of Cov(question means)-noise/3, then fixed diagonal shrinkage',
        likelihood='Gaussian moment approximation for mean reward with covariance noise/L',
        latent_box=[0., 1.], sampler='two conditional truncated-normal Gibbs chains',
        burn_sweeps=model.BURN, thin_sweeps=model.THIN, draws_per_chain=model.CHAIN_DRAWS,
        total_posterior_draws=2 * model.CHAIN_DRAWS,
        main_training_positions=[0, 85, 170, 255, 256, 341, 426, 511],
        alternative_training_positions=[32, 96, 160, 224, 288, 352, 416, 480],
        max_Rhat=model.RHAT_LIMIT, min_autocorrelation_ESS=model.ESS_LIMIT,
        fallback='empirical three-repeat reward vector, repeated8, uniform empirical T_delta',
        delta_main=.05, delta_control=0.,
        target_hyperparameters='three-fold crossfit within each history480, other320 only',
        predictive_diagnostic='two observed generations, predict third, three rotations',
        predictive_probability_scope='E(mu) marginal probability and Gaussian pair moment diagnostics; no latent-event calibration claim',
        forbidden_reward_splits=['validation200', 'fresh400', 'sealed200'],
        physical_LLM_calls=0, parameter_searches=0,
        seed_recipe='2026111400+1000*outerfold+100*innerfold+L/held tag')
    put(filename, protocol)
    return protocol


def synthetic_qualification(out):
    rng = np.random.default_rng(20261114081)
    covariance = np.eye(20) * .05
    prior = dict(mean=np.full(20, .5), prior_covariance=covariance,
                 noise=covariance)
    observation = np.full((2, 2, 20), .5)
    sample = model.sample_bounded(prior, observation, 20261114082)
    values = sample['draws']
    sd = np.sqrt(.05 / 3)
    expected = truncnorm.stats(-.5 / sd, .5 / sd, loc=.5, scale=sd, moments='mv')
    assert np.all((values >= 0) & (values <= 1))
    assert abs(values.mean() - expected[0]) < .01
    assert abs(values.var() - expected[1]) < .002
    assert sample['accepted'].all()
    permutation = rng.permutation(20)
    pm = dict(mean=prior['mean'][permutation],
              prior_covariance=covariance[np.ix_(permutation, permutation)],
              noise=covariance[np.ix_(permutation, permutation)])
    mean0, covariance0, _ = model.conjugate_parameters(prior, observation)
    mean1, covariance1, _ = model.conjugate_parameters(pm, observation[:, :, permutation])
    assert np.allclose(mean1, mean0[:, permutation])
    assert np.allclose(covariance1, covariance0[np.ix_(permutation, permutation)])
    weights = model.summaries(values, .05)
    assert np.allclose(weights['pair_probability'].sum(axis=2), 1)
    assert np.allclose(weights['weights'].sum(axis=1), 1)
    payload = model.training_payload(sample, observation, .05)
    assert payload['mu_samples'].shape == payload['mu_samples_alt'].shape == (2, 8, 20)
    # A forced failed chain visibly switches to the empirical-set fallback.
    sample['accepted'][0] = False
    fallback = model.training_payload(sample, observation, .05)
    assert fallback['fallback_mask'][0]
    assert np.all(fallback['mu_samples'][0] == .5)
    assert np.allclose(fallback['weights_delta005'][0], .05)
    stuck = np.zeros((1, 2, 256, 20))
    stuck[:, 1] = 1.
    stuck_rhat, stuck_ess = model.chain_diagnostics(stuck)
    assert np.isinf(stuck_rhat).all() and np.all(stuck_ess == 0)
    stuck[:] = 0.
    constant_rhat, constant_ess = model.chain_diagnostics(stuck)
    assert np.all(constant_rhat == 1) and np.all(constant_ess == 0)
    result = dict(status='PASS_SOFTWARE_QUALIFICATION', bounded_samples=True,
        independent_truncated_normal_expected_mean=float(expected[0]),
        observed_mean=float(values.mean()), expected_variance=float(expected[1]),
        observed_variance=float(values.var()), posterior_parameter_permutation_equivariant=True,
        probability_simplex_checked=True, disjoint_MC_batches_checked=True,
        explicit_failure_fallback_checked=True,
        degenerate_different_chains_rejected=True, constant_chains_zero_ESS=True,
        max_Rhat=float(sample['rhat'].max()), minimum_ESS=float(sample['ess'].min()),
        scope='software sampler sanity only, no empirical posterior calibration guarantee')
    put(out / 'public/JOINT_NOISE_SOFTWARE_QUALIFICATION.json', result)
    print(json.dumps(result), flush=True)


def covariance_summary(fitted):
    noise = fitted['noise']
    scale = np.sqrt(np.diag(noise))
    correlation = noise / np.outer(scale, scale)
    teams = list(itertools.combinations(range(6), 3))
    overlap = np.array([len(set(teams[i]) & set(teams[j]))
                       for i, j in zip(model.PAIR_I, model.PAIR_J)])
    eigenvalues = fitted['between_raw_eigenvalues']
    return dict(noise_trace=float(np.trace(noise)),
        noise_eigenvalues=np.linalg.eigvalsh(noise),
        prior_eigenvalues=np.linalg.eigvalsh(fitted['prior_covariance']),
        negative_between_eigenvalues=int(np.sum(eigenvalues < 0)),
        negative_between_eigenvalue_mass=float(-np.sum(eigenvalues[eigenvalues < 0])),
        average_noise_correlation_by_shared_members={str(count): float(
            correlation[model.PAIR_I, model.PAIR_J][overlap == count].mean()) for count in (0, 1, 2)},
        noise_diagonal_min=float(np.diag(noise).min()), noise_diagonal_max=float(np.diag(noise).max()))


def posterior_summary(result, payload):
    return dict(questions=len(payload['fallback_mask']),
        fallback_questions=int(payload['fallback_mask'].sum()),
        numeric_failures=int(result['numeric_failed'].sum()),
        max_Rhat=float(result['rhat'].max()) if np.isfinite(result['rhat']).all() else None,
        minimum_ESS=float(result['ess'].min()),
        zero_variance_coordinates=int(result['zero_variance_coordinates'].sum()),
        Rhat_median=float(np.median(payload['rhat_max'])),
        minimum_ESS_median=float(np.median(payload['ess_min'])),
        effective_combinations_delta005_mean=float(payload['effective_combinations_delta005'].mean()),
        effective_combinations_delta000_mean=float(payload['effective_combinations_delta000'].mean()),
        largest_mean_weight_delta005=float(payload['weights_delta005'].max(axis=1).mean()),
        largest_mean_weight_delta000=float(payload['weights_delta000'].max(axis=1).mean()),
        mean_sample_set_size_delta005=float(payload['mean_set_size_delta005'].mean()),
        mean_sample_set_size_delta000=float(payload['mean_set_size_delta000'].mean()),
        slow_tail_updates=result['slow_tail_updates'])


def build_fold(out, assets, fold):
    started = time.perf_counter()
    freeze(out, assets)
    if not (out / 'public/JOINT_NOISE_SOFTWARE_QUALIFICATION.json').exists():
        raise RuntimeError('Software qualification must precede real data')
    dest = out / f'private/fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'POSTERIOR_TRAINING.npz').exists():
        raise FileExistsError('Frozen posterior targets already exist')
    parent = assets / 'research/e8_decision_focused_combination_learning'
    spec = read(parent / 'private/qualification/FOLD_PACKS_MANIFEST.json')['folds'][fold]
    history_path = Path(spec['history']['path'])
    if sha(history_path) != spec['history']['sha256']:
        raise AssertionError('Frozen history pack changed')
    opened = datetime.now(timezone.utc).isoformat()
    history = arrays(history_path)
    if len(history['Y']) != 480:
        raise AssertionError('Expected480 history questions')
    sys.path.insert(0, str(parent / 'public/qualification'))
    from e8_assets import inner_splits
    splits = inner_splits(history, np.arange(480), 20261113)
    combined, fitted_summaries, diagnostics, private_fits = {}, [], [], {}
    diagnostic_arrays = {}
    for inner, (fit_rows, target_rows) in enumerate(splits):
        fitted = model.fit_working_prior(history['Y'][fit_rows])
        if set(fit_rows) & set(target_rows):
            raise AssertionError('Noise hyperparameter fit contains its target question')
        seed = 2026111400 + 1000 * fold + 100 * inner
        sampled = model.sample_bounded(fitted, history['Y'][target_rows], seed + 3)
        payload = model.training_payload(sampled, history['Y'][target_rows], .05)
        for key, value in payload.items():
            if key not in combined:
                combined[key] = np.empty((480, *value.shape[1:]), dtype=value.dtype)
            combined[key][target_rows] = value
        item = dict(inner=inner, fitted_questions=len(fit_rows), target_questions=len(target_rows),
                    covariance=covariance_summary(fitted), posterior=posterior_summary(sampled, payload))
        fitted_summaries.append(item)
        private_fits[f'fit_rows_inner{inner}'] = fit_rows
        private_fits[f'target_rows_inner{inner}'] = target_rows
        for key in ('mean', 'noise', 'noise_raw', 'prior_covariance', 'between_raw'):
            private_fits[f'{key}_inner{inner}'] = fitted[key]
        for held in range(3):
            observed_indices = [index for index in range(3) if index != held]
            observed = history['Y'][target_rows][:, observed_indices]
            check = model.sample_bounded(fitted, observed, seed + 20 + held)
            weak = model.training_payload(check, observed, .05)
            # Preserve a declared empirical two-repeat fallback in predictive diagnostics.
            prediction_samples = check['draws'].copy()
            prediction_samples[weak['fallback_mask']] = observed.mean(axis=1)[weak['fallback_mask'], None, :]
            checks = model.predictive_diagnostics(prediction_samples, fitted['noise'], history['Y'][target_rows, held])
            observed_mean = observed.mean(axis=1)
            checks['empirical_two_seed_brier'] = ((observed_mean - history['Y'][target_rows, held]) ** 2).mean(axis=1)
            checks['historical_prior_brier'] = ((fitted['mean'] - history['Y'][target_rows, held]) ** 2).mean(axis=1)
            checks['weak_Jeffreys_brier'] = (((observed.sum(axis=1) + .5) / 3 - history['Y'][target_rows, held]) ** 2).mean(axis=1)
            checks['fallback'] = weak['fallback_mask']
            for key, value in checks.items():
                value = np.asarray(value)
                if value.ndim == 1 and len(value) == len(target_rows):
                    if key not in diagnostic_arrays:
                        diagnostic_arrays[key] = np.empty((480, 3), dtype=value.dtype)
                    diagnostic_arrays[key][target_rows, held] = value
            diagnostics.append(dict(inner=inner, held_generation=[42, 137, 2024][held],
                target_questions=len(target_rows), fallback_questions=int(weak['fallback_mask'].sum()),
                summary={key: float(np.asarray(value).mean()) for key, value in checks.items()
                         if key not in ('empirical_same_question_latent_events_observed', 'fallback')}))
        print(f'fold={fold} inner={inner} targets=160 fallback={item["posterior"]["fallback_questions"]} elapsed={time.perf_counter()-started:.2f}', flush=True)
    combined['history_indices'] = history['global_indices']
    combined['history_ids'] = history['ids']
    combined['history_task'] = history['task']
    combined['history_subtask'] = history['sub']
    target_file = dest / 'POSTERIOR_TRAINING.npz'
    np.savez_compressed(target_file, **combined)
    np.savez_compressed(dest / 'JOINT_PRIOR_FITS.npz', **private_fits)
    np.savez_compressed(dest / 'LOSO_PREDICTIVE_RECORDS.npz',
        global_indices=history['global_indices'], task=history['task'], subtask=history['sub'], **diagnostic_arrays)
    report = dict(status='FROZEN_POSTERIOR_TRAINING_TARGETS', fold=fold,
        protocol_sha256=sha(out / 'public/JOINT_NOISE_PROTOCOL.json'),
        source_sha256={name: sha(MODULE / name) for name in SOURCE_NAMES},
        input_history_sha256=sha(history_path), input_history_questions=480,
        hyperparameter_crossfit=True, outer_evaluation_rewards_opened=False,
        posterior_shape=list(combined['mu_samples'].shape),
        posterior_sha256=sha(target_file), inner_fit_summaries=fitted_summaries,
        training_fallback_questions=int(combined['fallback_mask'].sum()),
        predictive_rotations=diagnostics,
        predictive_summary={key: float(value.mean()) for key, value in diagnostic_arrays.items()},
        raw_record_count=480 * 3, independent_question_note='Each outer history reuses600 source questions across folds; predictive records are dependent repeats',
        full_three_seed_target_has_no_outer_rewards=True,
        latent_event_calibration='unidentified from a single held generation',
        Gaussian_pair_intervals='moment diagnostics, not valid joint-Bernoulli confidence intervals',
        physical_LLM_calls=0, elapsed_seconds=time.perf_counter()-started)
    put(dest / 'POSTERIOR_FREEZE_RECEIPT.json', report)
    put(out / f'public/FOLD{fold}_JOINT_NOISE_SUMMARY.json', report)
    put(dest / 'REWARD_ACCESS_LOG.json', dict(first_history_pack_opened_utc=opened,
        decoded_reward_scope=f'outerfold{fold} history480 only',
        never_opened=['validation200', 'fresh400', 'sealed200', f'outerfold{fold} evaluation120'],
        protocol_sha256=report['protocol_sha256'], history_sha256=sha(history_path),
        application_level_crossfit=True, OS_access_isolation_claim=False))
    print(json.dumps(dict(status='COMPLETE_FOLD', fold=fold, posterior=str(target_file),
        fallback=report['training_fallback_questions'], elapsed=report['elapsed_seconds'])), flush=True)


def source_audit(out, assets):
    files = ['scripts/v2/stage2_aggregation_adapter.py', 'scripts/v2/package_stage2_generation.py',
             'src/budget_router_v2/full_feedback.py',
             'artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_matrix_receipt.json']
    text = (assets / files[0]).read_text(encoding='utf-8-sig')
    package = (assets / files[1]).read_text(encoding='utf-8-sig')
    checks = {
        'member_provenance_bound_to_question_alias_generation': 'expected_members[qid, alias]' in text,
        'aggregator_seed_fixed42': '"aggregation_seed": 42' in text,
        'shared_six_member_group_per_question_generation': "key = (node['question_id'], node['split'], node['generation_seed'])" in package,
        'twenty_future_combinations_share_same_dependency_group': "'future_K3_aggregation_jobs': 20" in package,
    }
    if not all(checks.values()):
        raise AssertionError('Expected cache dependency audit markers absent')
    put(out / 'public/JOINT_REPLICATE_PROVENANCE_AUDIT.json', dict(status='PASS_STATIC_LINEAGE_AUDIT',
        files_sha256={name: sha(assets / name) for name in files}, checks=checks,
        interpretation='Three member-generation blocks42/137/2024 each share cached members across20 combinations; aggregator seed42 remains fixed',
        limitations=['Static provenance and frozen matrix receipts establish execution identity',
                     'Distinct seed labels do not prove iid randomization or independent generator error',
                     'Covariance estimates use20-dimensional repeated blocks and assume exchangeable generation distribution',
                     'No new physical calls or preserved outcome decoding performed']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['qualify', 'build'])
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--asset-root', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--out', type=Path, default=MODULE)
    args = parser.parse_args()
    freeze(args.out, args.asset_root)
    if args.mode == 'qualify':
        synthetic_qualification(args.out)
        source_audit(args.out, args.asset_root)
    else:
        if args.fold is None:
            parser.error('build requires --fold')
        build_fold(args.out, args.asset_root, args.fold)


if __name__ == '__main__':
    main()
