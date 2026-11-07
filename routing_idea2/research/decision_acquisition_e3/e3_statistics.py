"""Paired summaries for frozen E3 online replays.

Generation seeds and orders repeat original questions. Conditional-trace
intervals preserve those repeats but condition on every realized learning path.
They do not estimate uncertainty from changing a feedback history. End-to-end
delete-group replays must be produced by the execution engine separately.
"""

from collections import defaultdict
import hashlib
import json

import numpy as np


CONTEXT_FIELDS = ('predictor', 'stage', 'method', 'B', 'history_size')
PATH_FIELDS = ('fold', 'gen', 'order')
RESOURCE_FIELDS = (
    'cumulative_gold', 'cumulative_member_calls',
    'cumulative_aggregate_calls', 'cumulative_tokens',
)
VALUE_NAMES = (
    'accuracy', 'no_feedback_accuracy', 'fixed_accuracy', 'top1_regret',
    'correction_vs_no_feedback', 'harm_vs_no_feedback',
    'correction_vs_fixed', 'harm_vs_fixed', 'switch_rate_vs_no_feedback',
    'planning_cpu_seconds', 'update_cpu_seconds',
)
DEFAULT_SEED = 20261107


def _meta(run):
    meta = run.get('meta', run)
    missing = [key for key in CONTEXT_FIELDS + PATH_FIELDS if key not in meta]
    if missing:
        raise ValueError('Missing run metadata fields ' + ', '.join(missing))
    return meta


def _context(meta, include_method=True):
    return tuple(meta[key] for key in CONTEXT_FIELDS if include_method or key != 'method')


def _path(meta):
    return tuple(meta[key] for key in PATH_FIELDS)


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _regret(trace):
    if trace.get('regret') is not None:
        return float(trace['regret'])
    if trace.get('top1_regret') is not None:
        return float(trace['top1_regret'])
    if trace.get('oracle_reward') is not None:
        return float(trace['oracle_reward']) - float(trace['reward'])
    return np.nan


def _trace_values(trace):
    reward = float(trace['reward'])
    base = float(trace['no_feedback_reward'])
    fixed = float(trace['fixed_reward'])
    if not all(value in (0.0, 1.0) for value in (reward, base, fixed)):
        raise ValueError('Each raw trace must carry binary rewards before repeat averaging')
    return np.asarray((
        reward, base, fixed, _regret(trace), float(reward > base),
        float(reward < base), float(reward > fixed), float(reward < fixed),
        float(trace['main'] != trace['no_feedback_main']),
        float(trace['plan_cpu_s']), float(trace['update_cpu_s']),
    ))


def _validate_run(run):
    meta = _meta(run)
    traces = run['traces']
    if not traces:
        raise ValueError('An online run must contain at least one service trace')
    questions = [int(trace['q']) for trace in traces]
    if len(set(questions)) != len(questions):
        raise ValueError('An original question may be served only once in one online path')
    counters = np.asarray([[float(trace[key]) for key in RESOURCE_FIELDS] for trace in traces])
    if not np.isfinite(counters).all() or np.any(counters < 0) or np.any(np.diff(counters, axis=0) < -1e-9):
        raise ValueError('Resource counters must be finite, nonnegative and cumulative')
    for trace in traces:
        values = _trace_values(trace)
        if not np.isfinite(values[np.arange(len(values)) != 3]).all():
            raise ValueError('Non-regret trace metrics must be finite')
        if values[9] < 0 or values[10] < 0:
            raise ValueError('CPU times must be nonnegative')
        if np.isfinite(values[3]) and not -1e-9 <= values[3] <= 1 + 1e-9:
            raise ValueError('Observed binary-reward Top-1 regret must lie in [0,1]')
    return meta, traces


def _question_table(runs, horizon=None):
    values = defaultdict(list)
    labels = {}
    positions = defaultdict(list)
    paths = set()
    qfold = {}
    for run in runs:
        meta, traces = _validate_run(run)
        path = _path(meta)
        if path in paths:
            raise ValueError('Duplicate fold, generation seed and order in one method context')
        paths.add(path)
        for position, trace in enumerate(traces, 1):
            if horizon is not None and position > horizon:
                break
            q = int(trace['q'])
            label = (str(trace['task']), str(trace['sub']))
            if q in labels and labels[q] != label:
                raise ValueError('Original question semantic labels changed across repeated paths')
            if q in qfold and qfold[q] != meta['fold']:
                raise ValueError('The same original question cannot be online in multiple CV folds')
            labels[q] = label
            qfold[q] = meta['fold']
            values[q].append(_trace_values(trace))
            positions[q].append(position)
    ids = sorted(values)
    if not ids:
        raise ValueError('No service observations at the requested arrival horizon')
    table = np.stack([np.asarray(values[q]).mean(axis=0) for q in ids])
    counts = np.asarray([len(values[q]) for q in ids])
    return ids, table, labels, counts, positions


def _metric_dict(values):
    result = {key: None if not np.isfinite(value) else float(value)
              for key, value in zip(VALUE_NAMES, values)}
    result['gain_vs_no_feedback'] = result['accuracy'] - result['no_feedback_accuracy']
    result['gain_vs_fixed'] = result['accuracy'] - result['fixed_accuracy']
    result['net_correction_vs_no_feedback'] = (
        result['correction_vs_no_feedback'] - result['harm_vs_no_feedback']
    )
    result['net_correction_vs_fixed'] = result['correction_vs_fixed'] - result['harm_vs_fixed']
    return result


def _scope_values(ids, table, labels, occurrence_weights=None):
    gsm = np.asarray([labels[q][0] == 'gsm8k' for q in ids])
    bbh = np.asarray([labels[q][0] == 'bbh' for q in ids])
    if not (gsm | bbh).all():
        raise ValueError('Only gsm8k and bbh semantic task labels are supported')
    weights = np.ones(len(ids)) if occurrence_weights is None else np.asarray(occurrence_weights)
    scopes = {}
    if gsm.any():
        scopes['gsm'] = np.average(table[gsm], axis=0, weights=weights[gsm])
    subtasks = sorted({labels[q][1] for q, flag in zip(ids, bbh) if flag})
    if subtasks:
        task_means = []
        for task in subtasks:
            mask = np.asarray([labels[q][0] == 'bbh' and labels[q][1] == task for q in ids])
            task_means.append(np.average(table[mask], axis=0, weights=weights[mask]))
        scopes['bbh_macro'] = np.mean(task_means, axis=0)
        scopes['bbh_micro'] = np.average(table[bbh], axis=0, weights=weights[bbh])
    if 'gsm' in scopes and 'bbh_macro' in scopes:
        scopes['balanced_subtask'] = 0.5 * (scopes['gsm'] + scopes['bbh_macro'])
        scopes['balanced_v2'] = 0.5 * (scopes['gsm'] + scopes['bbh_micro'])
    return scopes


def _costs(runs, horizon=None):
    rows = []
    folds = set()
    for run in runs:
        meta = _meta(run)
        traces = run['traces'] if horizon is None else run['traces'][:horizon]
        if not traces:
            continue
        last = traces[-1]
        rows.append([float(last[key]) for key in RESOURCE_FIELDS] + [
            sum(float(trace['plan_cpu_s']) for trace in traces),
            sum(float(trace['update_cpu_s']) for trace in traces),
            len(traces),
        ])
        folds.add(meta['fold'])
    array = np.asarray(rows)
    names = ('gold_questions', 'member_calls', 'aggregate_calls', 'tokens',
             'planning_cpu_seconds', 'update_cpu_seconds', 'served_questions')
    mean = array.mean(axis=0)
    result = {
        'mean_per_complete_path': dict(zip(names, mean.tolist())),
        'pooled_per_served_question': dict(zip(names[:-1], (array[:, :-1].sum(0) / array[:, -1].sum()).tolist())),
        'path_count': len(rows),
        'fold_count': len(folds),
    }
    repetition_counts = defaultdict(int)
    for run in runs:
        repetition_counts[_meta(run)['fold']] += 1
    if len(set(repetition_counts.values())) == 1:
        repeats = next(iter(repetition_counts.values()))
        result['expected_full_oof_totals'] = dict(zip(names, (array.sum(0) / repeats).tolist()))
        result['repeat_paths_per_fold'] = repeats
    else:
        result['expected_full_oof_totals'] = None
    return result


def _direction_rows(runs, field):
    subsets = defaultdict(list)
    for run in runs:
        subsets[_meta(run)[field]].append(run)
    result = []
    for value, subset in sorted(subsets.items(), key=lambda item: str(item[0])):
        ids, table, labels, counts, _ = _question_table(subset)
        result.append({field: _native(value), 'original_questions': len(ids),
                       'metrics': {scope: _metric_dict(point) for scope, point in _scope_values(ids, table, labels).items()},
                       'minimum_repeats_per_question': int(counts.min()),
                       'maximum_repeats_per_question': int(counts.max())})
    return result


def _arrival_curve(runs):
    lengths = [len(run['traces']) for run in runs]
    common_horizon = min(lengths)
    checkpoints = sorted({min(step, common_horizon) for step in (20, 40, 60, 80, 100, 120, common_horizon)})
    curve = []
    for checkpoint in checkpoints:
        ids, table, labels, counts, _ = _question_table(runs, horizon=checkpoint)
        values = _scope_values(ids, table, labels, occurrence_weights=counts)
        curve.append({'arrival_horizon_per_fold': checkpoint,
                      'served_original_questions': len(ids),
                      'observed_bbh_subtasks': len({labels[q][1] for q in ids if labels[q][0] == 'bbh'}),
                      'metrics': {scope: _metric_dict(point) for scope, point in values.items()},
                      'resources': _costs(runs, horizon=checkpoint)})
    future_windows = []
    for start in sorted({common_horizon // 2, 2 * common_horizon // 3} - {0}):
        remaining = [{'meta': _meta(run), 'traces': run['traces'][start:common_horizon]} for run in runs]
        ids, table, labels, counts, _ = _question_table(remaining)
        values = _scope_values(ids, table, labels, occurrence_weights=counts)
        future_windows.append({'start_arrival_exclusive': start, 'end_arrival_inclusive': common_horizon,
                               'served_original_questions': len(ids),
                               'metrics': {scope: _metric_dict(point) for scope, point in values.items()}})
    return {'common_horizon': common_horizon, 'path_lengths_equal': len(set(lengths)) == 1,
            'curve': curve, 'common_future_windows': future_windows,
            'interpretation': 'common arrival prefixes, same service questions for paired methods; never method-specific budget-exhaustion suffixes',
            'early_macro_note': 'prefix BBH macro includes only observed subtasks; prefix comparisons require identical paired question paths',
            'future_window_note': 'future service rewards retain the actual complete learning history and intervening updates; no policy state is reset'}


def aggregate(runs):
    """Summarize exact cached replays without treating 45 paths as independent.

    Input runs contain ``meta`` and ``traces``. Each original question is online
    in one fold and is averaged over its available generation/order repeats.
    Missing observed oracle rewards produce a missing regret field.
    """
    grouped = defaultdict(list)
    for run in runs:
        meta, _ = _validate_run(run)
        grouped[_context(meta)].append(run)
    output = []
    for context, subset in grouped.items():
        ids, table, labels, counts, _ = _question_table(subset)
        metrics = {scope: _metric_dict(point) for scope, point in _scope_values(ids, table, labels).items()}
        task_gains = []
        for name in sorted({'gsm8k' if labels[q][0] == 'gsm8k' else labels[q][1] for q in ids}):
            mask = np.asarray([('gsm8k' if labels[q][0] == 'gsm8k' else labels[q][1]) == name for q in ids])
            gain = float((table[mask, 0] - table[mask, 1]).mean())
            task_gains.append({'task': name, 'questions': int(mask.sum()), 'gain_vs_no_feedback': gain,
                               'metrics': _metric_dict(table[mask].mean(0))})
        output.append({
            'meta': dict(zip(CONTEXT_FIELDS, map(_native, context))),
            'original_questions': len(ids), 'path_count': len(subset),
            'minimum_repeats_per_question': int(counts.min()),
            'maximum_repeats_per_question': int(counts.max()),
            'question_set_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'complete_5fold_3seed_3order_question_coverage': bool(
                len(ids) == 600 and len(subset) == 45 and np.all(counts == 9)
                and all(len(run['traces']) == 120 for run in subset)
                and len({labels[q][1] for q in ids if labels[q][0] == 'bbh'}) == 27
                and len({_meta(run)['fold'] for run in subset}) == 5
                and len({_meta(run)['gen'] for run in subset}) == 3
                and len({_meta(run)['order'] for run in subset}) == 3
            ),
            'metrics': metrics, 'per_task': task_gains,
            'negative_transfer_task_fraction': float(np.mean([row['gain_vs_no_feedback'] < 0 for row in task_gains])),
            'expected_corrections_on_original_questions': float(table[:, 4].sum()),
            'expected_harms_on_original_questions': float(table[:, 5].sum()),
            'expected_net_corrections_on_original_questions': float((table[:, 4] - table[:, 5]).sum()),
            'expected_corrections_vs_fixed_on_original_questions': float(table[:, 6].sum()),
            'expected_harms_vs_fixed_on_original_questions': float(table[:, 7].sum()),
            'resources': _costs(subset),
            'per_seed': _direction_rows(subset, 'gen'),
            'per_fold': _direction_rows(subset, 'fold'),
            'per_order': _direction_rows(subset, 'order'),
            'arrival_curves': _arrival_curve(subset),
            'regret_status': 'observed_oracle_available' if np.isfinite(table[:, 3]).all() else 'not_available_for_all_traces',
        })
    budget_groups = defaultdict(list)
    for item in output:
        meta = item['meta']
        key = tuple(meta[field] for field in CONTEXT_FIELDS if field != 'B')
        budget_groups[key].append(item)
    budget_curves = []
    for context, items in budget_groups.items():
        horizons = {item['arrival_curves']['common_horizon'] for item in items}
        comparable = (len(horizons) == 1
                      and all(item['arrival_curves']['path_lengths_equal'] for item in items)
                      and len({item['question_set_sha256'] for item in items}) == 1)
        points = []
        for item in sorted(items, key=lambda value: float(value['meta']['B'])):
            points.append({'gold_cap': item['meta']['B'], 'metrics': item['metrics'],
                           'actual_resources': item['resources'],
                           'arrival_horizon_per_fold': item['arrival_curves']['common_horizon']})
        auc = {}
        if len(points) >= 2 and comparable:
            x = np.asarray([point['gold_cap'] for point in points], dtype=float)
            for scope in ('gsm', 'bbh_macro', 'balanced_subtask', 'balanced_v2'):
                if all(scope in point['metrics'] for point in points) and x[-1] > x[0]:
                    y = np.asarray([point['metrics'][scope]['accuracy'] for point in points])
                    area = np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / (x[-1] - x[0])
                    auc[scope] = float(area)
        budget_curves.append({'meta': dict(zip([field for field in CONTEXT_FIELDS if field != 'B'], context)),
                              'points': points, 'same_arrival_horizon_and_original_question_set': comparable,
                              'normalized_accuracy_auc_over_observed_gold_cap_range': auc,
                              'gold_cap_range': [points[0]['gold_cap'], points[-1]['gold_cap']],
                              'actual_cost_interpolation': 'none; actual resource points retain the common final service horizon'})
    return {'groups': output, 'budget_curves': budget_curves,
            'inference': {'unit': 'original question averaged over generation seeds and stream orders',
                          'path_count_is_independent_sample_size': False,
                          'fold_history_overlap': 'retained; no independent-fold significance claim',
                          'aggregation': 'GSM mean and BBH subtask macro, then 50/50; V2 BBH question weights separate',
                          'regret_definition': 'observed per-generation 20-combination oracle minus service reward',
                          'new_llm_calls': 0}}


def _interval(point, draws):
    low, high = np.quantile(draws, (0.025, 0.975))
    centered = draws - point
    p_value = (1 + np.sum(np.abs(centered) >= abs(point) - 1e-15)) / (len(draws) + 1)
    return {'difference': float(point), 'ci95': [float(low), float(high)],
            'p_value_descriptive_two_sided': float(p_value),
            'interval_scope': 'conditional on frozen realized learning paths; approximate descriptive nominal 95%'}


def _matched_runs(runs, method_a, method_b, filters):
    selected = [run for run in runs if all(_meta(run).get(key) == value for key, value in filters.items())]
    left = [run for run in selected if _meta(run)['method'] == method_a]
    right = [run for run in selected if _meta(run)['method'] == method_b]
    contexts = {_context(_meta(run), include_method=False) for run in left + right}
    if len(contexts) != 1 or not left or not right:
        raise ValueError('Filters must resolve one common predictor/stage/budget/history context and both methods')
    left_paths = {_path(_meta(run)): run for run in left}
    right_paths = {_path(_meta(run)): run for run in right}
    if len(left_paths) != len(left) or len(right_paths) != len(right) or left_paths.keys() != right_paths.keys():
        raise ValueError('Methods must have identical unique fold/generation/order paths')
    for path, run_a in left_paths.items():
        run_b = right_paths[path]
        if [trace['q'] for trace in run_a['traces']] != [trace['q'] for trace in run_b['traces']]:
            raise ValueError('Paired methods must serve identical original questions in identical arrival order')
    return left, right, contexts.pop()


def paired_compare(runs, methodA, methodB, filters=None, reps=10000, seed=DEFAULT_SEED, **filter_values):
    """Descriptive paired intervals after fixing all realized learning paths.

    GSM original-question clusters and BBH whole-subtask blocks are resampled.
    Repeated seeds/orders stay together. Changing training observations, feedback
    decisions, or later model states requires separate end-to-end replay.
    """
    if not isinstance(reps, int) or reps < 100:
        raise ValueError('At least 100 bootstrap replicates are required')
    selection = dict(filters or {})
    selection.update(filter_values)
    left, right, context = _matched_runs(runs, methodA, methodB, selection)
    ids, values_a, labels, repeats, _ = _question_table(left)
    ids_b, values_b, labels_b, repeats_b, _ = _question_table(right)
    if ids != ids_b or labels != labels_b or not np.array_equal(repeats, repeats_b):
        raise ValueError('Paired original-question identities, labels and repetition counts must agree')
    if not np.allclose(values_a[:, 1:3], values_b[:, 1:3]):
        raise ValueError('No-feedback and fixed references changed between paired methods')
    differences = values_a[:, 0] - values_b[:, 0]
    gsm = np.asarray([labels[q][0] == 'gsm8k' for q in ids])
    tasks = sorted({labels[q][1] for q in ids if labels[q][0] == 'bbh'})
    if not gsm.any() or not tasks:
        raise ValueError('Both GSM8K and BBH are required for the preregistered balanced contrast')
    task_means = []
    task_counts = []
    for task in tasks:
        mask = np.asarray([labels[q][0] == 'bbh' and labels[q][1] == task for q in ids])
        task_means.append(differences[mask].mean())
        task_counts.append(mask.sum())
    task_means = np.asarray(task_means)
    task_counts = np.asarray(task_counts)
    rng = np.random.default_rng(seed)
    gsm_values = differences[gsm]
    gsm_weights = rng.multinomial(len(gsm_values), np.full(len(gsm_values), 1 / len(gsm_values)), size=reps)
    gsm_draws = gsm_weights @ gsm_values / len(gsm_values)
    task_weights = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=reps)
    bbh_draws = task_weights @ task_means / len(tasks)
    micro_draws = (task_weights @ (task_means * task_counts)) / (task_weights @ task_counts)
    points = {'gsm': gsm_values.mean(), 'bbh_macro': task_means.mean(),
              'bbh_micro': np.average(task_means, weights=task_counts)}
    draws = {'gsm': gsm_draws, 'bbh_macro': bbh_draws, 'bbh_micro': micro_draws}
    points['balanced_subtask'] = 0.5 * (points['gsm'] + points['bbh_macro'])
    points['balanced_v2'] = 0.5 * (points['gsm'] + points['bbh_micro'])
    draws['balanced_subtask'] = 0.5 * (gsm_draws + bbh_draws)
    draws['balanced_v2'] = 0.5 * (gsm_draws + micro_draws)
    direction = {}
    for field in PATH_FIELDS:
        rows = []
        for value in sorted({_meta(run)[field] for run in left}, key=str):
            subset_a = [run for run in left if _meta(run)[field] == value]
            subset_b = [run for run in right if _meta(run)[field] == value]
            q, a, lab, _, _ = _question_table(subset_a)
            _, b, _, _, _ = _question_table(subset_b)
            delta = a.copy()
            delta[:, 0] = a[:, 0] - b[:, 0]
            rows.append({field: _native(value), 'original_questions': len(q),
                         'accuracy_difference': {scope: float(point[0])
                                                 for scope, point in _scope_values(q, delta, lab).items()}})
        direction[field] = rows
    cost_a = _costs(left)
    cost_b = _costs(right)
    resource_delta = {key: value - cost_b['mean_per_complete_path'][key]
                      for key, value in cost_a['mean_per_complete_path'].items()}
    return {'method_a': methodA, 'method_b': methodB,
            'meta': dict(zip([field for field in CONTEXT_FIELDS if field != 'method'], map(_native, context))),
            'original_questions': len(ids), 'paired_paths': len(left),
            'minimum_repeats_per_question': int(repeats.min()), 'maximum_repeats_per_question': int(repeats.max()),
            'metrics': {scope: _interval(points[scope], draws[scope]) for scope in points},
            'per_direction': direction,
            'per_task_accuracy_difference': {task: float(value) for task, value in zip(tasks, task_means)},
            'actual_resource_difference_per_complete_path': resource_delta,
            'inference': {'method': 'conditional_trace_question_and_bbh_task_bootstrap95',
                          'gsm_unit': 'original question with all seed/order repeats averaged first',
                          'bbh_unit': 'whole semantic subtask retaining all original questions and repeats',
                          'bootstrap_replicates': reps, 'bootstrap_seed': seed,
                          'replays_learning_path_changes': False, 'confirmatory': False,
                          'limitation': 'adaptive feedback dependencies and overlapping CV training histories remain conditioned upon',
                          'multiplicity': 'unadjusted; use a preregistered family with holm_adjust; adjustment does not make conditional traces confirmatory'}}


def holm_adjust(p_values, alpha=0.05):
    """Return Holm adjusted p-values for an explicitly supplied comparison family."""
    if not 0 < alpha < 1 or not p_values:
        raise ValueError('A nonempty frozen family and alpha in (0,1) are required')
    values = {str(name): float(value) for name, value in p_values.items()}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError('All p-values must be finite and lie in [0,1]')
    ordered = sorted(values, key=values.get)
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * values[name])
        adjusted[name] = min(1.0, running)
    return {'family_size': len(values), 'alpha': alpha,
            'comparisons': {name: {'p_value': values[name], 'holm_adjusted_p': adjusted[name],
                                   'passes_holm_threshold': adjusted[name] <= alpha}
                            for name in values},
            'scope_note': 'Validity inherits the supplied p-values; descriptive trace p-values remain descriptive after adjustment'}


def jackknife_ci(delete_values, full_value, n_groups=20, reps=10000, seed=DEFAULT_SEED):
    """Approximate paired CI from engine-produced end-to-end deleted-group runs.

    Supply one metric or an aligned vector of paired contrasts. The same group
    weights are used for every vector coordinate. Every deletion must remove
    the original questions from all CV history, reference and online roles and
    replay the entire remaining learning path with shared random keys.
    """
    full = np.asarray(full_value, dtype=float)
    deleted = np.asarray(delete_values, dtype=float)
    if n_groups < 3 or deleted.shape != (n_groups,) + full.shape:
        raise ValueError('Delete values must have shape (n_groups, *full_value.shape)')
    if reps < 100 or not np.isfinite(full).all() or not np.isfinite(deleted).all():
        raise ValueError('Finite aligned estimates and at least 100 replicates are required')
    deleted_mean = deleted.mean(axis=0)
    variance = (n_groups - 1) / n_groups * np.sum((deleted - deleted_mean) ** 2, axis=0)
    standard_error = np.sqrt(variance)
    pseudo = n_groups * full - (n_groups - 1) * deleted
    centered = (pseudo - pseudo.mean(axis=0)).reshape(n_groups, -1)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=reps)
    perturbation = weights @ centered / n_groups * np.sqrt(n_groups / (n_groups - 1))
    draws = full.reshape(-1)[None] + perturbation
    lower, upper = np.quantile(draws, (0.025, 0.975), axis=0)
    p_values = (1 + np.sum(np.abs(perturbation) >= np.abs(full.reshape(-1))[None] - 1e-15, axis=0)) / (reps + 1)
    scales = standard_error.reshape(-1)
    valid = scales > 0
    maximum = np.max(np.abs(perturbation[:, valid] / scales[valid]), axis=1) if valid.any() else np.zeros(reps)
    cutoff = float(np.quantile(maximum, 0.95))
    return {
        'estimate': full.tolist(),
        'ci95': np.stack((lower, upper), axis=-1).reshape(full.shape + (2,)).tolist(),
        'jackknife_standard_error': standard_error.tolist(),
        'p_value_approximate_two_sided': p_values.reshape(full.shape).tolist(),
        'pseudo_value_bias_estimate': (pseudo.mean(axis=0) - full).tolist(),
        'simultaneous_ci95': np.stack((full - cutoff * standard_error, full + cutoff * standard_error), axis=-1).tolist(),
        'simultaneous_family_size': int(full.size),
        'n_groups': n_groups, 'bootstrap_replicates': reps, 'bootstrap_seed': seed,
        'same_resampling_for_all_vector_coordinates': True,
        'inference': 'approximate end-to-end delete-group jackknife pseudo-value bootstrap with finite-group variance correction',
        'conditions': 'balanced original-question deletion groups; all history and online roles deleted; complete replay; smooth-enough statistic',
        'limitations': 'budget thresholds and discrete selection are nonsmooth; 20 deletion groups do not establish finite-sample coverage or independent new-task generalization',
    }

