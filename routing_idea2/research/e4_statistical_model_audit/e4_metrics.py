"""Original-question metrics and conditional paired inference for E4."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from itertools import combinations
from statistics import NormalDist
import numpy as np


SEED = 20261107
SCOPES = ('gsm', 'bbh_macro', 'bbh_micro', 'balanced_subtask', 'balanced_v2')
PAIRS = np.asarray(list(combinations(range(20), 2)), dtype=int)


def _native(value):
    array = np.asarray(value)
    return float(array) if array.ndim == 0 else array.tolist()


def _labels(task, sub, n):
    task, sub = np.asarray(task).astype(str), np.asarray(sub).astype(str)
    if task.shape != (n,) or sub.shape != (n,) or not np.isin(task, ('gsm8k', 'bbh')).all():
        raise ValueError('Task and subtask labels must identify each GSM8K/BBH original question')
    return task, sub


def _scope_arrays(values, task, sub):
    result = {}
    gsm, bbh = task == 'gsm8k', task == 'bbh'
    if gsm.any():
        result['gsm'] = values[gsm].mean(axis=0)
    names = sorted(set(sub[bbh]))
    if names:
        result['bbh_macro'] = np.mean([values[bbh & (sub == name)].mean(axis=0) for name in names], axis=0)
        result['bbh_micro'] = values[bbh].mean(axis=0)
    if gsm.any() and names:
        result['balanced_subtask'] = 0.5 * (result['gsm'] + result['bbh_macro'])
        result['balanced_v2'] = 0.5 * (result['gsm'] + result['bbh_micro'])
    return result


def scope_means(values, task, sub):
    """Return task weights shared by every E4 metric, permitting trailing axes."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        raise ValueError('Values require an original-question axis')
    task, sub = _labels(task, sub, len(values))
    valid = np.isfinite(values.reshape(len(values), -1)).all(axis=1)
    result = dict.fromkeys(SCOPES)
    result.update({name: _native(value) for name, value in _scope_arrays(values[valid], task[valid], sub[valid]).items()})
    result['bbh'] = result['bbh_micro']
    return result


def _ci(point, draws):
    bounds = np.quantile(draws, (0.025, 0.975), axis=0)
    p = (1 + np.sum(np.abs(draws - point) >= np.abs(point) - 1e-15, axis=0)) / (len(draws) + 1)
    return {'difference': _native(point), 'ci95': np.stack((bounds[0], bounds[1]), axis=-1).tolist(),
            'p_value_descriptive_two_sided': _native(p)}


def paired_summary(diff, task, sub, reps=5000, seed=SEED):
    """Pair original questions, keeping all already-averaged seed/action repeats.

    Primary intervals resample questions within fixed semantic strata. Additional
    BBH whole-task intervals expose task-composition sensitivity. Both condition
    on fitted predictors and the observed generation cache. CV training-history
    changes require a separate refit/replay analysis.
    """
    values = np.asarray(diff, dtype=float)
    if values.ndim == 0 or len(values) == 0 or reps < 100:
        raise ValueError('Nonempty question values and at least 100 replicates are required')
    task, sub = _labels(task, sub, len(values))
    valid = np.isfinite(values.reshape(len(values), -1)).all(axis=1)
    values, task, sub = values[valid], task[valid], sub[valid]
    if not len(values):
        result = {name: {'difference': None, 'ci95': None, 'question_stratified_ci95': None,
                         'task_cluster_ci95': None} for name in SCOPES}
        result['bbh'] = dict(result['bbh_micro'])
        result['metadata'] = {'question_support': 0, 'status': 'NO_FINITE_QUESTION_VALUES'}
        return result
    tail = values.shape[1:]
    flat = values.reshape(len(values), -1)
    rng = np.random.default_rng(seed)
    points = _scope_arrays(flat, task, sub)
    question_draws = {}
    if np.any(task == 'gsm8k'):
        x = flat[task == 'gsm8k']
        weights = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=reps)
        question_draws['gsm'] = weights @ x / len(x)
    names = sorted(set(sub[task == 'bbh']))
    task_means, task_sizes, strata_draws = [], [], []
    support = {'gsm8k': int(np.sum(task == 'gsm8k'))}
    for name in names:
        x = flat[(task == 'bbh') & (sub == name)]
        weights = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=reps)
        strata_draws.append(weights @ x / len(x))
        task_means.append(x.mean(axis=0))
        task_sizes.append(len(x))
        support['bbh/' + name] = len(x)
    cluster_draws = dict(question_draws)
    if names:
        task_sizes = np.asarray(task_sizes)
        question_draws['bbh_macro'] = np.mean(strata_draws, axis=0)
        question_draws['bbh_micro'] = np.average(strata_draws, axis=0, weights=task_sizes)
        weights = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=reps)
        means = np.asarray(task_means)
        cluster_draws['bbh_macro'] = weights @ means / len(names)
        cluster_draws['bbh_micro'] = (weights @ (means * task_sizes[:, None])) / (weights @ task_sizes)[:, None]
    for draws in (question_draws, cluster_draws):
        if 'gsm' in draws and 'bbh_macro' in draws:
            draws['balanced_subtask'] = 0.5 * (draws['gsm'] + draws['bbh_macro'])
            draws['balanced_v2'] = 0.5 * (draws['gsm'] + draws['bbh_micro'])
    result = {}
    for name, point in points.items():
        p = point.reshape(tail)
        q = question_draws[name].reshape((reps,) + tail)
        c = cluster_draws[name].reshape((reps,) + tail)
        item = _ci(p, q)
        item['question_stratified_ci95'] = item['ci95']
        item['task_cluster_ci95'] = _ci(p, c)['ci95']
        result[name] = item
    if 'bbh_micro' in result:
        result['bbh'] = dict(result['bbh_micro'])
    result['metadata'] = {'question_support': len(values), 'excluded_nonfinite_questions': int(np.sum(~valid)),
                          'strata_support': support, 'bbh_subtasks_present': len(names),
                          'minimum_nonempty_stratum_support': min(value for value in support.values() if value),
                          'bootstrap_replicates': reps, 'bootstrap_seed': seed,
                          'ci95_default': 'original-question bootstrap within fixed task/subtask strata',
                          'task_cluster_ci95': 'whole BBH task blocks; GSM questions retained as its only semantic task',
                          'scope': 'conditional on fixed OOF fits and cached generation repeats; descriptive unadjusted',
                          'cv_training_history_changes_included': False,
                          'generation_or_combination_cells_are_new_questions': False,
                          'bbh_alias': 'bbh_micro'}
    return result


def _variance(value, shape, name):
    value = np.asarray(value, dtype=float)
    try:
        result = np.broadcast_to(value, shape).copy()
    except ValueError as error:
        raise ValueError(name + ' must broadcast to question-by-combination marginal variances') from error
    if not np.isfinite(result).all() or np.any(result < -1e-10):
        raise ValueError(name + ' must contain finite nonnegative marginal variances')
    return np.maximum(result, 0)


def calibration_table(probability, rewards, bins=10):
    """Fixed-width diagnostic bins; cell counts remain distinct from question counts."""
    n, actions = probability.shape
    target = rewards.mean(axis=1)
    membership = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        mask = membership == index
        count = int(mask.sum())
        predicted = float(probability[mask].mean()) if count else None
        observed = float(target[mask].mean()) if count else None
        if count:
            ece += count / (n * actions) * abs(predicted - observed)
        rows.append({'lower': index / bins, 'upper': (index + 1) / bins,
                     'question_combination_cells': count, 'generation_score_cells': count * rewards.shape[1],
                     'original_question_support': int(np.sum(mask.any(axis=1))),
                     'predicted_probability': predicted, 'observed_reward_mean': observed})
    return {'bins': rows, 'ece': ece,
            'weighting': 'uniform question-combination cells with generation repeats averaged within cell',
            'independent_sample_size': n,
            'interpretation': 'descriptive binned calibration; task-weighted scoring uses per_question Brier/log loss'}


def prediction_metrics(pred, Y, *, latent_variance=None, observation_variance=None,
                       levels=(0.5, 0.8, 0.95), prediction_tie_tolerance=0.0):
    """Score frozen predictions against one or more generated binary reward vectors.

    ``pred`` is n by20. ``Y`` is n byL by20, or n by20 for one repeat.
    Returned per_question values are numeric NumPy arrays suitable for NPZ.
    Undefined ranking and one-repeat noise corrections use floating NaN.
    Gaussian serving choices always use raw prediction argmax. Probability
    scoring explicitly clips raw values, while log loss additionally uses eps.
    """
    pred, Y = np.asarray(pred, dtype=float), np.asarray(Y, dtype=float)
    if Y.ndim == 2:
        Y = Y[:, None, :]
    if (pred.ndim != 2 or pred.shape[1] != 20 or Y.ndim != 3 or Y.shape[0] != len(pred)
            or Y.shape[2] != 20 or not np.isfinite(pred).all() or not np.isin(Y, (0, 1)).all()):
        raise ValueError('Require finite n-by20 predictions and binary n-byL-by20 rewards')
    if not len(pred) or Y.shape[1] < 1 or prediction_tie_tolerance < 0:
        raise ValueError('Nonempty questions, generation repeats and a nonnegative tie tolerance are required')
    n, repeats = len(pred), Y.shape[1]
    probability = np.clip(pred, 0, 1)
    log_probability = np.clip(probability, 1e-12, 1 - 1e-12)
    mean_reward = Y.mean(axis=1)
    selected = np.argmax(pred, axis=1)
    selected_reward = Y[np.arange(n)[:, None], np.arange(repeats)[None, :], selected[:, None]]
    pair_pred = pred[:, PAIRS[:, 0]] - pred[:, PAIRS[:, 1]]
    pair_reward = Y[:, :, PAIRS[:, 0]] - Y[:, :, PAIRS[:, 1]]
    strict = pair_reward != 0
    predicted_tie = np.abs(pair_pred) <= prediction_tie_tolerance
    rank = np.where(predicted_tie[:, None, :], 0.5, (pair_pred[:, None, :] * pair_reward > 0).astype(float))
    strict_count = strict.sum(axis=(1, 2))
    ranking = np.divide((rank * strict).sum(axis=(1, 2)), strict_count,
                        out=np.full(n, np.nan), where=strict_count > 0)
    mean_gap = mean_reward[:, PAIRS[:, 0]] - mean_reward[:, PAIRS[:, 1]]
    mean_strict = mean_gap != 0
    mean_rank = np.where(predicted_tie, 0.5, (pair_pred * mean_gap > 0).astype(float))
    mean_count = mean_strict.sum(axis=1)
    mean_ranking = np.divide((mean_rank * mean_strict).sum(axis=1), mean_count,
                             out=np.full(n, np.nan), where=mean_count > 0)
    mean_mse = ((pred - mean_reward) ** 2).mean(axis=1)
    noise_correction = Y.var(axis=1, ddof=1).mean(axis=1) / repeats if repeats > 1 else np.full(n, np.nan)
    per_question = {
        'raw_mse': ((pred[:, None, :] - Y) ** 2).mean(axis=(1, 2)),
        'mean_mse': mean_mse,
        'brier': ((probability[:, None, :] - Y) ** 2).mean(axis=(1, 2)),
        'log_loss': -(Y * np.log(log_probability[:, None, :]) +
                      (1 - Y) * np.log1p(-log_probability[:, None, :])).mean(axis=(1, 2)),
        'pair_gap_mae': np.abs(pair_pred - mean_gap).mean(axis=1),
        'non_tie_rank_accuracy': ranking,
        'accuracy': selected_reward.mean(axis=1),
        'top1_regret': (Y.max(axis=2) - selected_reward).mean(axis=1),
        'outside_rate': ((pred < 0) | (pred > 1)).mean(axis=1),
        'noise_corrected_mean_mse': mean_mse - noise_correction,
        'estimated_mean_reward_measurement_variance': noise_correction,
        'strict_pair_support': strict_count.astype(float),
        'mean_reward_rank_accuracy': mean_ranking,
        'mean_reward_strict_pair_support': mean_count.astype(float),
        'empirical_mean_oracle_regret': mean_reward.max(axis=1) - selected_reward.mean(axis=1),
    }
    interval = {'latent_true_probability_coverage': None,
                'latent_reference_note': 'three-seed reward mean is a noisy reference; its inclusion does not establish true latent-p coverage',
                'gaussian_future_note': 'working Gaussian observation intervals include aleatoric noise; binary coverage is accompanied by width and contains-both outcomes rate',
                'levels': list(levels)}
    if latent_variance is not None:
        latent = _variance(latent_variance, pred.shape, 'latent_variance')
        per_question['latent_variance_mean'] = latent.mean(axis=1)
        observation = _variance(observation_variance, pred.shape, 'observation_variance') if observation_variance is not None else None
        for level in levels:
            if not 0 < level < 1:
                raise ValueError('Interval levels must lie in (0,1)')
            suffix = format(100 * level, 'g').replace('.', '_')
            z = NormalDist().inv_cdf(0.5 + level / 2)
            half_width = z * np.sqrt(latent)
            lower, upper = pred - half_width, pred + half_width
            per_question['latent_reference_mean_inclusion_' + suffix] = ((mean_reward >= lower) & (mean_reward <= upper)).mean(axis=1)
            per_question['latent_interval_width_' + suffix] = (2 * half_width).mean(axis=1)
            if observation is not None:
                total_variance = latent + observation
                half_width = z * np.sqrt(total_variance)
                lower, upper = pred - half_width, pred + half_width
                per_question['gaussian_observation_coverage_' + suffix] = ((Y >= lower[:, None, :]) & (Y <= upper[:, None, :])).mean(axis=(1, 2))
                per_question['gaussian_observation_width_' + suffix] = (2 * half_width).mean(axis=1)
                per_question['gaussian_observation_contains_both_' + suffix] = ((lower <= 0) & (upper >= 1)).mean(axis=1)
        interval['latent_intervals_available'] = True
        interval['gaussian_observation_intervals_available'] = observation is not None
    else:
        interval['latent_intervals_available'] = False
        interval['gaussian_observation_intervals_available'] = False
    return {'per_question': per_question, 'selected_actions': selected,
            'calibration': calibration_table(probability, Y), 'interval_diagnostics': interval,
            'metadata': {'original_questions': n, 'generation_repeats': repeats, 'combinations': 20,
                         'score_cells': int(Y.size), 'ranking_prediction_tie_tolerance': prediction_tie_tolerance,
                         'non_tie_rank_target': 'strict binary reward differences per generation; prediction ties score0.5',
                         'mean_reward_target': 'finite-generation empirical mean, distinct from true latent probability',
                         'noise_corrected_mse_assumption': 'conditionally independent generation repeats and prediction fitted without these query outcomes',
                         'noise_corrected_mse_available': repeats > 1,
                         'probability_scoring': 'Brier clips raw predictions to [0,1]; log loss additionally clips to [1e-12,1-1e-12]',
                         'service_selection': 'raw prediction argmax, first action on an exact tie',
                         'regret_oracle': 'observed20-combination maximum separately in each generation',
                         'undefined_metrics': 'numeric floating NaN in per_question arrays; report support when aggregating'}}


def design_diagnostics(X, B, alpha=100.0, repeats=3):
    """Kronecker design spectrum, including original null directions and Ridge df."""
    X, B = np.asarray(X, dtype=float), np.asarray(B, dtype=float)
    if X.ndim != 2 or B.ndim != 2 or alpha <= 0 or repeats < 1 or not np.isfinite(X).all() or not np.isfinite(B).all():
        raise ValueError('Finite question/action designs, positive alpha and repeat weight are required')
    sx = np.linalg.svd(X, compute_uv=False)
    sb = np.linalg.svd(B, compute_uv=False)
    singular = np.sort((np.sqrt(repeats) * np.outer(sx, sb)).ravel())[::-1]
    dimension = X.shape[1] * B.shape[1]
    tolerance = max(X.shape[0] * B.shape[0] * repeats, dimension) * np.finfo(float).eps * (singular[0] if len(singular) else 0)
    positive = singular[singular > tolerance]
    rank = len(positive)
    squares = singular ** 2
    total = squares.sum()
    weights = squares[squares > 0] / total if total > 0 else np.asarray([])
    minimum_precision = alpha if rank < dimension else alpha + positive[-1] ** 2
    return {'independent_questions': len(X), 'generation_repeat_weight': repeats,
            'question_feature_dimension': X.shape[1], 'action_feature_dimension': B.shape[1],
            'nominal_parameter_dimension': dimension,
            'question_design_rank': int(np.linalg.matrix_rank(X)),
            'action_design_rank': int(np.linalg.matrix_rank(B)),
            'observable_joint_rank': rank, 'likelihood_unidentified_dimensions': dimension - rank,
            'score_observation_rows': int(len(X) * len(B) * repeats),
            'singular_values': singular.tolist(), 'svd_tolerance': float(tolerance),
            'raw_condition_number': float(positive[0] / positive[-1]) if rank == dimension and rank else None,
            'observable_condition_number': float(positive[0] / positive[-1]) if rank else None,
            'regularized_precision_condition_number': float((alpha + (singular[0] ** 2 if len(singular) else 0)) / minimum_precision),
            'ridge_effective_df': float(np.sum(squares / (squares + alpha))),
            'stable_rank': float(total / singular[0] ** 2) if total > 0 else 0.0,
            'entropy_effective_rank_squared_spectrum': float(np.exp(-np.sum(weights * np.log(weights)))) if len(weights) else 0.0,
            'nullspace_prior_variance_trace': float((dimension - rank) / alpha),
            'likelihood': 'independent unit-noise score observations with repeated identical design rows',
            'rank_note': 'Ridge regularization makes precision invertible; likelihood-null directions retain prior information'}

