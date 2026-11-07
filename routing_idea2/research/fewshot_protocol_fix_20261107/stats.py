"""Paired inference after integrating all legal Support/Query kernels.

Historical training data and fitted predictors are conditioned upon. BBH task
blocks retain every original question and every Support role. GSM inference uses
delete-one-question jackknife pseudo-values, removing all roles of that question.
Only rewards are bootstrapped. Other metrics are exact finite-pool summaries.
"""

from math import comb

import numpy as np

from models import METHODS


GENERATION_SEEDS = (42, 137, 2024)
SOURCE_TARGET_PAIRS = tuple((source, target) for source in range(3) for target in range(3))
DIAGONAL = np.array([0, 4, 8])
OFF_DIAGONAL = np.array([1, 2, 3, 5, 6, 7])
METRIC_NAMES = (
    'reward', 'correction_vs_base', 'harm_vs_base', 'correction_vs_fixed',
    'harm_vs_fixed', 'cpu_seconds', 'top1_regret', 'switch_vs_base',
)


def _validate_total(item, metric_count=None):
    total = np.asarray(item['total'], dtype=float)
    if total.ndim != 3 or total.shape[:2] != (9, len(METHODS)):
        raise ValueError('total must have shape (9, len(METHODS), metric_count)')
    if total.shape[2] < 6 or (metric_count is not None and total.shape[2] != metric_count):
        raise ValueError('Metric count must be consistent and include the first six metrics')
    count = int(item['count'])
    if count <= 0 or not np.isfinite(total).all():
        raise ValueError('A positive kernel count and finite accumulated metrics are required')
    point = total / count
    reward = point[:, :, 0]
    if np.any(reward < -1e-10) or np.any(reward > 1 + 1e-10):
        raise ValueError('Mean rewards must lie in [0,1]')
    if not np.allclose(point[:, :, 1] - point[:, :, 2], reward - reward[:, :1], atol=1e-9):
        raise ValueError('Base correction minus harm must equal the paired reward gain')
    if not np.allclose(point[:, :, 3] - point[:, :, 4], reward - reward[:, 1:2], atol=1e-9):
        raise ValueError('Fixed correction minus harm must equal the paired reward gain')
    if total.shape[2] > 6:
        regret_change = point[:, :, 6] - point[:, :1, 6]
        if not np.allclose(regret_change, -(reward - reward[:, :1]), atol=1e-9):
            raise ValueError('Regret and reward changes must share the same observed oracle')
    return total, count, point


def _gsm_reward_bootstrap(gsm, reps, rng):
    total, count, point = _validate_total(gsm)
    roles = np.asarray(gsm['roles'], dtype=float)
    if roles.ndim != 4 or roles.shape[1:] != total.shape or not np.isfinite(roles).all():
        raise ValueError('roles must have shape (original_questions, 9, methods, metrics)')
    n = roles.shape[0]
    k = int(gsm['k'])
    if n <= k + 1 or count != comb(n, k) * (n - k):
        raise ValueError('GSM requires complete Support enumeration and at least k+2 original questions')
    deleted_count = comb(n - 1, k) * (n - 1 - k)
    if not np.allclose(roles.sum(axis=0), (k + 1) * total, atol=1e-7):
        raise ValueError('Each kernel must contribute to every Support and Query role exactly once')
    deleted_means = (total[None, :, :, 0] - roles[:, :, :, 0]) / deleted_count
    reward = point[:, :, 0]
    pseudo = n * reward[None, :, :] - (n - 1) * deleted_means
    if not np.allclose(pseudo.mean(axis=0), reward, atol=1e-9):
        raise ValueError('Delete-one pseudo-values must average to the complete-enumeration estimate')
    centered = (pseudo - reward[None, :, :]).reshape(n, -1)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=reps)
    draws = reward[None, :, :] + (weights @ centered / n).reshape(reps, 9, len(METHODS))
    diagonal_deleted_means = deleted_means[:, DIAGONAL].mean(axis=1)
    jackknife_variance = (n - 1) / n * np.sum(
        (diagonal_deleted_means - diagonal_deleted_means.mean(axis=0)) ** 2, axis=0
    )
    return point, draws, jackknife_variance, n, deleted_count


def _interval(samples, point):
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        'difference': float(point),
        'ci95': [float(max(-1, lower)), float(min(1, upper))],
        'approximate_unadjusted': True,
    }


def _paired(point, draws, method_index):
    reward = point[:, 0]
    result = {}
    for key, reference_index in (('vs_base', 0), ('vs_fixed', 1)):
        result[key] = _interval(
            draws[:, method_index] - draws[:, reference_index],
            reward[method_index] - reward[reference_index],
        )
    name = METHODS[method_index]
    matched_name = name + '_shuffle'
    if matched_name in METHODS:
        reference_index = METHODS.index(matched_name)
        result['vs_matched_shuffle'] = _interval(
            draws[:, method_index] - draws[:, reference_index],
            reward[method_index] - reward[reference_index],
        )
        result['matched_shuffle_method'] = matched_name
    else:
        result['vs_matched_shuffle'] = None
        result['matched_shuffle_method'] = None
    return result


def _values(point, method_index):
    values = point[method_index]
    result = {
        'accuracy': float(values[0]),
        'correction_vs_base': float(values[1]),
        'harm_vs_base': float(values[2]),
        'net_correction_vs_base': float(values[1] - values[2]),
        'correction_vs_fixed': float(values[3]),
        'harm_vs_fixed': float(values[4]),
        'net_correction_vs_fixed': float(values[3] - values[4]),
        'mean_cpu_seconds_per_query_kernel': float(values[5]),
    }
    if len(values) > 6:
        result['top1_regret'] = float(values[6])
    if len(values) > 7:
        result['switch_rate_vs_base'] = float(values[7])
    return result


def _scope_summary(point, draws, task_points, task_names):
    result = {}
    for method_index, name in enumerate(METHODS):
        diagonal_point = point[DIAGONAL].mean(axis=0)
        diagonal_draws = draws[:, DIAGONAL].mean(axis=1)
        item = _values(diagonal_point, method_index)
        item['paired_ci'] = _paired(diagonal_point, diagonal_draws, method_index)
        item.update(item['paired_ci'])
        task_gains = (
            task_points[:, DIAGONAL, method_index, 0].mean(axis=1)
            - task_points[:, DIAGONAL, 0, 0].mean(axis=1)
        )
        item['negative_transfer_task_count'] = int(np.sum(task_gains < -1e-12))
        item['negative_transfer_task_fraction'] = float(np.mean(task_gains < -1e-12))
        item['semantic_task_count'] = len(task_names)
        item['per_task_gain_vs_base'] = {
            task_name: float(gain) for task_name, gain in zip(task_names, task_gains)
        }
        per_seed = []
        for pair_index in DIAGONAL:
            source, target = SOURCE_TARGET_PAIRS[int(pair_index)]
            seed_item = _values(point[pair_index], method_index)
            seed_item.update(source_seed=GENERATION_SEEDS[source], target_seed=GENERATION_SEEDS[target])
            seed_item['paired_ci'] = _paired(point[pair_index], draws[:, pair_index], method_index)
            per_seed.append(seed_item)
        item['perdiagseed'] = per_seed
        off_point = point[OFF_DIAGONAL].mean(axis=0)
        off_draws = draws[:, OFF_DIAGONAL].mean(axis=1)
        item['offdiagmean'] = _values(off_point, method_index)
        item['offdiagmean']['paired_ci'] = _paired(off_point, off_draws, method_index)
        transfer = []
        for pair_index, (source, target) in enumerate(SOURCE_TARGET_PAIRS):
            cell = _values(point[pair_index], method_index)
            cell.update(source_seed=GENERATION_SEEDS[source], target_seed=GENERATION_SEEDS[target])
            cell['paired_ci'] = _paired(point[pair_index], draws[:, pair_index], method_index)
            transfer.append(cell)
        item['source_target_matrix'] = transfer
        result[name] = item
    return result


def summarize(gsm, bbh, reps=10000):
    """Return JSON-friendly task-balanced metrics and paired approximate 95% CIs.

    ``total`` sums complete enumerated Query kernels, not episode means. ``roles``
    accumulates a kernel once for every original question used in its Support or
    Query. BBH items also provide ``n`` so V2-compatible micro weights are based
    on original question counts rather than numbers of enumerated kernels.
    """
    if not isinstance(reps, int) or reps < 100:
        raise ValueError('At least 100 bootstrap replicates are required')
    if len(bbh) != 27 or len({item['task'] for item in bbh}) != 27:
        raise ValueError('All 27 unique BBH subtasks are required')
    rng = np.random.default_rng(20261107)
    gsm_point, gsm_draws, gsm_variance, gsm_n, deleted_count = _gsm_reward_bootstrap(gsm, reps, rng)
    bbh_points = []
    question_counts = []
    for item in bbh:
        _, _, point = _validate_total(item, metric_count=gsm_point.shape[-1])
        bbh_points.append(point)
        question_counts.append(int(item['n']))
    bbh_points = np.asarray(bbh_points)
    question_counts = np.asarray(question_counts, dtype=float)
    if np.any(question_counts <= 0):
        raise ValueError('BBH original question counts must be positive')
    task_weights = rng.multinomial(27, np.full(27, 1 / 27), size=reps)
    rewards = bbh_points[:, :, :, 0]
    macro_point = bbh_points.mean(axis=0)
    macro_draws = (task_weights @ rewards.reshape(27, -1) / 27).reshape(reps, 9, len(METHODS))
    micro_point = np.average(bbh_points, axis=0, weights=question_counts)
    weighted_rewards = (rewards * question_counts[:, None, None]).reshape(27, -1)
    micro_draws = (
        (task_weights @ weighted_rewards) / (task_weights @ question_counts)[:, None]
    ).reshape(reps, 9, len(METHODS))
    all_points = np.concatenate((gsm_point[None], bbh_points), axis=0)
    all_names = ['gsm8k'] + [str(item['task']) for item in bbh]
    result = {
        'inference': {
            'scope': 'approximate conditional-on-frozen-history paired inference',
            'multiplicity': 'descriptive unadjusted intervals; no confirmatory familywise claim',
            'bootstrap_replicates': reps,
            'bootstrap_seed': 20261107,
            'gsm_original_questions': gsm_n,
            'gsm_k': int(gsm['k']),
            'gsm_delete_one_kernel_count': deleted_count,
            'gsm_method': 'delete-original-question in all roles, jackknife pseudo-value bootstrap',
            'gsm_diagonal_jackknife_se_by_method': {
                name: float(np.sqrt(gsm_variance[index]))
                for index, name in enumerate(METHODS)
            },
            'gsm_se_note': 'diagonal seed mean is formed before jackknife variance; repeated-seed covariance retained',
            'bbh_method': 'whole-subtask block bootstrap preserving every question and Support role',
            'bbh_subtasks': 27,
            'generation_replicates': 'all three seeds share identical resampling weights',
            'source_target_order': [list(pair) for pair in SOURCE_TARGET_PAIRS],
            'metric_order': list(METRIC_NAMES[:gsm_point.shape[-1]]),
            'negative_transfer_definition': 'fraction of semantic task clusters with diagonal mean gain below zero',
            'per_subtask_question_population_ci': 'not estimated from only three/four questions',
        },
        'gsm': _scope_summary(gsm_point, gsm_draws, gsm_point[None], ['gsm8k']),
        'bbh_macro': _scope_summary(macro_point, macro_draws, bbh_points, all_names[1:]),
        'bbh_micro': _scope_summary(micro_point, micro_draws, bbh_points, all_names[1:]),
        'balanced_subtask_macro': _scope_summary(
            0.5 * (gsm_point + macro_point), 0.5 * (gsm_draws + macro_draws),
            all_points, all_names,
        ),
        'balanced_v2_macro': _scope_summary(
            0.5 * (gsm_point + micro_point), 0.5 * (gsm_draws + micro_draws),
            all_points, all_names,
        ),
    }
    return result
