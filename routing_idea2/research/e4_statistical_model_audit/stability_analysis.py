"""Train600-only cross-generation stability and finite-repeat Oracle audit."""

import argparse
import itertools
from pathlib import Path

from e4_common import PUBLIC, PRIVATE, SEEDS, load_train, e4_splits, put, sha, semantic
import numpy as np
from e4_metrics import PAIRS, SCOPES, scope_means, paired_summary


SEED = 20261107


def finite_list(values):
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values).ravel()]


def fixed_routes(train, history, query):
    reward = train['Y'][history].mean(axis=1)
    tasks, subs = train['task'][history], train['sub'][history]
    task_means = {task: reward[tasks == task].mean(axis=0) for task in ('gsm8k', 'bbh')}
    global_action = int(np.argmax(np.mean(list(task_means.values()), axis=0)))
    task_actions = {task: int(np.argmax(value)) for task, value in task_means.items()}
    sub_actions = {sub: int(np.argmax(reward[(tasks == 'bbh') & (subs == sub)].mean(axis=0)))
                   for sub in sorted(set(subs[tasks == 'bbh']))}
    task = np.asarray([task_actions[name] for name in train['task'][query]])
    sub = np.asarray([sub_actions.get(str(name), task_actions['bbh']) if domain == 'bbh' else task_actions['gsm8k']
                      for domain, name in zip(train['task'][query], train['sub'][query])])
    return {'global_fixed': np.full(len(query), global_action, dtype=int),
            'task_fixed': task, 'subtask_fixed': sub}


def oof_routes(train, splits):
    n = len(train['Y'])
    choices = {name: np.full(n, -1, dtype=int) for name in ('global_fixed', 'task_fixed', 'subtask_fixed')}
    seen = np.zeros(n, dtype=int)
    for fold in splits['folds']:
        history, query = np.asarray(fold['history']), np.asarray(fold['stream'])
        if len(history) != 480 or len(query) != 120 or set(history) & set(query):
            raise ValueError('Require the frozen480-history/120-query disjoint folds')
        for name, values in fixed_routes(train, history, query).items():
            choices[name][query] = values
        seen[query] += 1
    if not np.all(seen == 1):
        raise ValueError('All600 original questions must be queried in exactly one fold')
    return choices


def chosen_rewards(Y, actions):
    return Y[np.arange(len(Y))[:, None], np.arange(Y.shape[1])[None, :], np.asarray(actions)[:, None]]


def split_summary(values, task, sub, reps):
    return paired_summary(values, task, sub, reps=reps, seed=SEED)


def covariance_components(Y):
    mean = Y.mean(axis=1)
    centered = Y - mean[:, None, :]
    noise_by_question = np.einsum('isa,isb->iab', centered, centered) / (Y.shape[1] - 1)
    noise = noise_by_question.mean(axis=0)
    between = np.cov(mean, rowvar=False, ddof=1)
    signal = between - noise / Y.shape[1]
    eigenvalues = np.linalg.eigvalsh(signal)
    return {'original_questions': len(Y), 'generation_noise_covariance': noise.tolist(),
            'observed_mean_reward_covariance': between.tolist(),
            'noise_corrected_latent_reward_covariance': signal.tolist(),
            'noise_corrected_covariance_eigenvalues': eigenvalues.tolist(),
            'negative_signal_eigenvalue_count': int(np.sum(eigenvalues < -1e-10)),
            'diagonal_generation_noise_mean': float(np.diag(noise).mean()),
            'diagonal_raw_between_question_variance_mean': float(np.diag(between).mean()),
            'diagonal_noise_corrected_signal_variance_mean': float(np.diag(signal).mean()),
            'noise_covariance_rank': int(np.linalg.matrix_rank(noise)),
            'assumption': 'generation draws conditionally independent given each question; actions within a draw may be correlated',
            'projection': 'raw estimated signal covariance retained, including negative eigenvalues'}


def gap_task_statistics(Y, task, sub, reps):
    gaps = Y[:, :, PAIRS[:, 0]] - Y[:, :, PAIRS[:, 1]]
    labels = np.where(task == 'gsm8k', 'gsm8k', sub)
    names = ['gsm8k'] + sorted(set(sub[task == 'bbh']))
    rng = np.random.default_rng(SEED)
    rows, signal_draws, raw_signal_draws, noise_draws = [], {}, {}, {}
    for name in names:
        indexes = np.flatnonzero(labels == name)
        data = gaps[indexes]
        means = data.mean(axis=1)
        noise = data.var(axis=1, ddof=1)
        n = len(indexes)
        between = means.var(axis=0, ddof=1) if n > 1 else np.full(len(PAIRS), np.nan)
        signal = between - noise.mean(axis=0) / 3
        pair_rows = []
        for source, target in itertools.combinations(range(3), 2):
            a, b = data[:, source], data[:, target]
            a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
            denominator = np.sqrt((a0 ** 2).sum(axis=0) * (b0 ** 2).sum(axis=0))
            valid = denominator > 0
            correlation = np.divide((a0 * b0).sum(axis=0), denominator,
                                    out=np.full(len(PAIRS), np.nan), where=valid)
            strict = (a != 0) & (b != 0)
            concordant = strict & (a == b)
            pair_rows.append({'source_seed': SEEDS[source], 'target_seed': SEEDS[target],
                              'pairs_with_identifiable_correlation': int(valid.sum()),
                              'mean_identifiable_pair_correlation': float(correlation[valid].mean()) if valid.any() else None,
                              'pair_correlations': finite_list(correlation),
                              'joint_strict_comparison_cells': int(strict.sum()),
                              'strict_direction_agreement': float(concordant.sum() / strict.sum()) if strict.any() else None,
                              'cells_are_independent_sample_size': False})
        signal_mean, noise_mean = float(signal.mean()), float(noise.mean())
        rows.append({'task': name, 'original_questions': n,
                     'observed_between_question_gap_variance_mean': float(between.mean()),
                     'generation_gap_noise_variance_mean': noise_mean,
                     'noise_corrected_gap_signal_variance_mean': signal_mean,
                     'noise_corrected_signal_by_pair': finite_list(signal),
                     'noise_by_pair': noise.mean(axis=0).tolist(),
                     'unresolved_or_negative_signal_pair_fraction': float(np.mean(signal <= 0)),
                     'signal_noise_ratio': signal_mean / noise_mean if signal_mean > 0 and noise_mean > 0 else None,
                     'seed_pair_stability': pair_rows})
        if n <= 1:
            signal_draws[name] = np.full(reps, np.nan)
            raw_signal_draws[name] = np.full(reps, np.nan)
            noise_draws[name] = np.full(reps, np.nan)
            continue
        weights = rng.multinomial(n, np.full(n, 1 / n), size=reps)
        resampled_means = weights @ means / n
        second = weights @ (means ** 2).mean(axis=1) / n
        noise_sample = weights @ noise.mean(axis=1) / n
        raw_signal_draws[name] = n / (n - 1) * (second - (resampled_means ** 2).mean(axis=1)) - noise_sample / 3
        # Conditional bootstrap sample variance loses between/n in expectation.
        # Recenter only its sampling distribution and retain the raw sensitivity.
        signal_draws[name] = raw_signal_draws[name] + between.mean() / n
        noise_draws[name] = noise_sample
    result = {'per_semantic_task': rows, 'pair_order': PAIRS.tolist(),
              'population_decomposition': 'Var_X(E_seed[D|X]) + E_X Var_seed(D|X); within BBH subtask centering removes task-level mean-gap shifts',
              'finite_repeat_estimator': 'sample_var_question(mean_seed_D) - mean_question(sample_var_seed_D)/3',
              'negative_estimates': 'retained; finite-three-seed moments do not identify a nonnegative population signal precisely'}
    row_lookup = {row['task']: row for row in rows}
    bbh_names = names[1:]
    counts = np.asarray([row_lookup[name]['original_questions'] for name in bbh_names])
    for label, keys, weights in (('gsm', ['gsm8k'], np.ones(1)),
                                ('bbh_macro', bbh_names, np.ones(len(bbh_names))),
                                ('bbh_micro', bbh_names, counts)):
        point_signal = np.average([row_lookup[key]['noise_corrected_gap_signal_variance_mean'] for key in keys], weights=weights)
        point_noise = np.average([row_lookup[key]['generation_gap_noise_variance_mean'] for key in keys], weights=weights)
        sdraw = np.average([signal_draws[key] for key in keys], axis=0, weights=weights)
        raw_sdraw = np.average([raw_signal_draws[key] for key in keys], axis=0, weights=weights)
        wdraw = np.average([noise_draws[key] for key in keys], axis=0, weights=weights)
        result[label] = {'signal_variance': float(point_signal), 'noise_variance': float(point_noise),
                         'signal_variance_question_bootstrap_ci95': np.quantile(sdraw, (.025, .975)).tolist(),
                         'signal_variance_original_percentile_ci95': np.quantile(raw_sdraw, (.025, .975)).tolist(),
                         'signal_ci_method': 'recentered descriptive bootstrap; add each stratum observed between-question variance/n to correct conditional finite-stratum bootstrap bias',
                         'noise_variance_question_bootstrap_ci95': np.quantile(wdraw, (.025, .975)).tolist(),
                         'bootstrap_replicates': reps, 'bootstrap_seed': SEED,
                         'scope': 'within semantic task, original-question resampling recomputes question means; all3 seeds and190 contrasts kept together'}
    return result


def analyze_stability(train, splits, reps=5000):
    Y = np.asarray(train['Y'], dtype=float)
    task, sub = np.asarray(train['task']).astype(str), np.asarray(train['sub']).astype(str)
    if Y.shape != (600, 3, 20) or not np.isin(Y, (0, 1)).all():
        raise ValueError('E4 stability accepts only the frozen600-by3-by20 binary train matrix')
    choices = oof_routes(train, splits)
    quality = {name: chosen_rewards(Y, actions) for name, actions in choices.items()}
    legacy = fixed_routes(train, np.arange(600), np.arange(600))
    legacy_quality = {name: chosen_rewards(Y, actions) for name, actions in legacy.items()}
    cross_actions = np.zeros((600, 3), dtype=int)
    cross_quality = np.zeros((600, 3))
    uniform_quality = np.zeros((600, 3))
    tie_counts = np.zeros((600, 3), dtype=int)
    discriminative = np.zeros((600, 3), dtype=bool)
    for target in range(3):
        source = [seed for seed in range(3) if seed != target]
        mean = Y[:, source].mean(axis=1)
        actions = np.argmax(mean, axis=1)
        highest = mean == mean.max(axis=1, keepdims=True)
        cross_actions[:, target] = actions
        cross_quality[:, target] = Y[np.arange(600), target, actions]
        uniform_quality[:, target] = (Y[:, target] * highest).sum(axis=1) / highest.sum(axis=1)
        tie_counts[:, target] = highest.sum(axis=1)
        discriminative[:, target] = mean.max(axis=1) > mean.min(axis=1)
    post = Y.max(axis=2).mean(axis=1)
    plugin = Y.mean(axis=1).max(axis=1)
    if np.any(post < plugin - 1e-12) or np.any(plugin < cross_quality.mean(axis=1) - 1e-12):
        raise ValueError('Empirical Oracle ordering changed; inspect source/target indexing')
    names = ['observed_post_oracle', 'empirical_3seed_mean_maximum', 'cross_seed_oracle_lex',
             'cross_seed_oracle_uniform_tie', 'cross_minus_oof_global', 'cross_minus_oof_task',
             'cross_minus_oof_subtask', 'post_minus_cross', 'plugin_minus_cross']
    values = [post, plugin, cross_quality.mean(axis=1), uniform_quality.mean(axis=1),
              cross_quality.mean(axis=1) - quality['global_fixed'].mean(axis=1),
              cross_quality.mean(axis=1) - quality['task_fixed'].mean(axis=1),
              cross_quality.mean(axis=1) - quality['subtask_fixed'].mean(axis=1),
              post - cross_quality.mean(axis=1), plugin - cross_quality.mean(axis=1)]
    joint = split_summary(np.stack(values, axis=1), task, sub, reps)
    summary = {'counts': {'original_questions': 600, 'generation_repeats': 3,
                           'combinations': 20, 'score_cells': 36000, 'bbh_subtasks': len(set(sub[task == 'bbh']))},
               'oracle_metric_order': names, 'oracle_paired_summary': joint,
               'oracle_per_seed': [{'heldout_seed': SEEDS[target],
                                    'cross_seed_value': scope_means(cross_quality[:, target], task, sub),
                                    'cross_minus_oof_subtask': scope_means(cross_quality[:, target] - quality['subtask_fixed'][:, target], task, sub),
                                    'source_maximum_tie_sizes': {str(count): int(np.sum(tie_counts[:, target] == count)) for count in sorted(set(tie_counts[:, target]))}}
                                   for target in range(3)],
               'oof_fixed_accuracy': {name: scope_means(value.mean(axis=1), task, sub) for name, value in quality.items()},
               'legacy_in_sample_train600_fixed_accuracy': {name: scope_means(value.mean(axis=1), task, sub) for name, value in legacy_quality.items()},
               'seed_combo_accuracy': [{'seed': SEEDS[s], 'accuracy_by_combination': scope_means(Y[:, s], task, sub)} for s in range(3)],
               'three_seed_score_agreement': scope_means((Y.max(axis=1) == Y.min(axis=1)).mean(axis=1), task, sub),
               'generation_covariance': {name: covariance_components(Y[mask]) for name, mask in
                                         (('all600_pooled', np.ones(600, bool)), ('gsm8k', task == 'gsm8k'), ('bbh_pooled', task == 'bbh'))},
               'within_subtask_gap_stability': gap_task_statistics(Y, task, sub, reps),
               'interpretation': {'oracle': 'cross-seed selection reads two true outcome vectors of this same question and is a privileged reproducibility diagnostic',
                                  'population_bound': 'finite empirical maxima and cross-seed values do not bound the true population stable Oracle',
                                  'legacy_88_percent': 'historical validation200/V2 two-task macro only; E4 uses train600 and also reports BBH27-subtask macro',
                                  'fixed_routes': 'primary fixed routes use only all3 generation outcomes in the other480 historical questions; legacy600 in-sample routes remain separately labelled',
                                  'seed_independence': 'variance correction assumes conditional independence across actual generation repeats; score agreement alone does not prove independent or duplicate generation',
                                  'uncertainty': 'question/task resampling preserves all generation and combination cells; overlapping fitted-fold histories remain conditioned upon'}}
    arrays = {'question_ids': np.asarray(train['ids']), 'task': task, 'sub': sub,
              'cross_seed_actions': cross_actions, 'cross_seed_quality': cross_quality,
              'uniform_tie_cross_seed_quality': uniform_quality, 'source_tie_counts': tie_counts,
              'source_discriminative': discriminative, 'observed_post_oracle': post,
              'empirical_mean_maximum': plugin, 'oracle_metric_values': np.stack(values, axis=1)}
    for name in choices:
        arrays['oof_' + name + '_actions'] = choices[name]
        arrays['oof_' + name + '_quality'] = quality[name]
    return summary, arrays


def self_test():
    n = 600
    Y = np.zeros((n, 3, 20))
    Y[np.arange(n), :, np.arange(n) % 20] = 1
    task = np.asarray(['gsm8k'] * 300 + ['bbh'] * 300)
    sub = np.asarray([''] * 300 + ['sub' + str(i % 27) for i in range(300)])
    train = {'Y': Y, 'task': task, 'sub': sub, 'ids': np.arange(n).astype(str)}
    folds = [{'history': np.flatnonzero(np.arange(n) % 5 != f), 'stream': np.flatnonzero(np.arange(n) % 5 == f)} for f in range(5)]
    summary, arrays = analyze_stability(train, {'folds': folds}, reps=100)
    assert np.all(arrays['cross_seed_quality'] == 1)
    assert summary['generation_covariance']['all600_pooled']['diagonal_generation_noise_mean'] == 0
    assert summary['within_subtask_gap_stability']['bbh_macro']['noise_variance'] == 0
    assert summary['oracle_paired_summary']['balanced_subtask']['difference'][0] == 1
    print('E4_STABILITY_SOFTWARE_PASS600_3_20')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    train, references = load_train()
    splits = e4_splits(train)
    summary, arrays = analyze_stability(train, splits, reps=args.replicates)
    summary['source_hashes'] = references
    summary['code_sha256'] = {'stability_analysis.py': sha(Path(__file__)),
                              'e4_metrics.py': sha(Path(__file__).with_name('e4_metrics.py'))}
    summary['new_llm_calls'] = 0
    PRIVATE.mkdir(parents=True, exist_ok=True)
    pending = PRIVATE / 'STABILITY_PER_QUESTION.pending.npz'
    final = PRIVATE / 'STABILITY_PER_QUESTION.npz'
    np.savez_compressed(pending, **arrays)
    pending.replace(final)
    put(PUBLIC / 'STABILITY_AND_ORACLE.json', summary)
    print('TRAIN600_STABILITY_AND_ORACLE_COMPLETE')


if __name__ == '__main__':
    main()

