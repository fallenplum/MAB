"""Frozen five-contrast E3 statistics from streamed, complete replay receipts."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path

from run_e3 import CONFIG, PRIVATE, PUBLIC, ROOT, put, sha256
import numpy as np
from assets import make_splits
from e3_statistics import paired_compare, jackknife_ci, holm_adjust


CONTRASTS = (
    ('primary_B1_decision_vs_ig_cost', 'B1', 'decision_cost', 'ig_cost'),
    ('secondary_B1_decision_vs_random', 'B1', 'decision_cost', 'random'),
    ('secondary_B1_decision_vs_boundary_cost', 'B1', 'decision_cost', 'boundary_cost'),
    ('secondary_B2_decision_vs_ig', 'B2', 'decision_cost', 'ig'),
    ('secondary_B2_decision_vs_random_neighbor', 'B2', 'decision_cost', 'random_neighbor'),
)
SCOPES = ('gsm', 'bbh_macro', 'bbh_micro', 'balanced_subtask', 'balanced_v2')
PATH_FIELDS = ('fold', 'gen', 'order')
RESOURCE_FIELDS = ('member_calls', 'aggregator_calls', 'gold', 'scoring_ops',
                   'input_tokens', 'output_tokens', 'tokens', 'source_model_wall_s')
SEED = 20261107


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_index(stage):
    path = PRIVATE / (stage + '_INDEX.json')
    if not path.exists():
        return None
    content = path.read_bytes()
    rows = json.loads(content)
    if not isinstance(rows, list):
        raise ValueError('An execution index must contain a list of receipt records')
    return {'rows': rows, 'sha256': hashlib.sha256(content).hexdigest(), 'name': path.name}


def select_rows(index, stage, method, predictor='P2', budget=20, group=None):
    if index is None:
        return []
    selected = []
    for row in index['rows']:
        meta = row['meta']
        if (meta['stage'], meta['method'], meta['predictor'], meta['B'], meta['history_size'], meta.get('delete_group')) == (
                stage, method, predictor, budget, 240, group):
            selected.append(row)
    return selected


def load_run(row):
    path = Path(row['path']).resolve()
    if not path.is_relative_to((PRIVATE / 'runs').resolve()):
        raise ValueError('Replay source must remain inside private/runs')
    run = json.loads(gzip.decompress(path.read_bytes()))
    if run['meta'] != row['meta'] or len(run['traces']) != run['ledger']['stream_questions']:
        raise ValueError('Indexed replay identity or service completeness changed')
    if run['runtime']['new_llm_calls'] != 0:
        raise ValueError('The frozen cached replay contract requires zero new LLM calls')
    return run


def require_paths(rows):
    expected = set(itertools.product(range(5), CONFIG['generation_seeds'], CONFIG['order_seeds']))
    paths = [tuple(row['meta'][key] for key in PATH_FIELDS) for row in rows]
    if len(paths) != 45 or set(paths) != expected:
        raise ValueError('A statistics context requires all 45 unique fold/generation/order paths')


def scope_means(values, labels):
    questions = sorted(labels)
    gsm = [q for q in questions if labels[q][0] == 'gsm8k']
    bbh = [q for q in questions if labels[q][0] == 'bbh']
    names = sorted({labels[q][1] for q in bbh})
    if not gsm or not bbh or len(names) != 27:
        raise ValueError('Every primary estimate must retain GSM8K and all 27 BBH subtasks')
    gsm_mean = float(np.mean(values[gsm]))
    bbh_mean = float(np.mean([np.mean(values[[q for q in bbh if labels[q][1] == name]]) for name in names]))
    micro = float(np.mean(values[bbh]))
    return {'gsm': gsm_mean, 'bbh_macro': bbh_mean, 'bbh_micro': micro,
            'balanced_subtask': 0.5 * (gsm_mean + bbh_mean),
            'balanced_v2': 0.5 * (gsm_mean + micro)}


def stream_point(rows):
    """Accumulate original-question rewards without retaining whole replay payloads."""
    require_paths(rows)
    sums = np.zeros((600, 3))
    counts = np.zeros(600, dtype=int)
    labels = {}
    qfolds = {}
    history_counts = defaultdict(set)
    order_hashes = {}
    cap_hashes = {}
    resources = dict.fromkeys(RESOURCE_FIELDS + ('priced_units', 'planning_cpu_s', 'update_cpu_s'), 0.0)
    identities = set()
    for row in rows:
        run = load_run(row)
        meta = run['meta']
        identities.add(run['identity'])
        path = tuple(meta[key] for key in PATH_FIELDS)
        qs = []
        for trace in run['traces']:
            q = int(trace['q'])
            if not 0 <= q < 600 or not trace['service_frozen_before_feedback']:
                raise ValueError('Original question identity or pre-feedback service lock is invalid')
            reward = [trace['reward'], trace['no_feedback_reward'], trace['fixed_reward']]
            if any(value not in (0, 1) for value in reward):
                raise ValueError('Raw reward observations must be binary')
            label = (trace['task'], trace['sub'])
            if q in labels and labels[q] != label:
                raise ValueError('A semantic label changed across generation/order repeats')
            if q in qfolds and qfolds[q] != meta['fold']:
                raise ValueError('An original question was online in multiple folds')
            labels[q] = label
            qfolds[q] = meta['fold']
            sums[q] += reward
            counts[q] += 1
            qs.append(q)
        if len(qs) != len(set(qs)):
            raise ValueError('An online path repeated an original question')
        order_hashes[path] = digest(qs)
        cap_hashes[path] = digest({'service': run['ledger']['service_caps'], 'audit': run['ledger']['audit_caps']})
        history_counts[meta['fold']].add(int(run['historical_initialization_resources']['gold']))
        for name in RESOURCE_FIELDS:
            resources[name] += float(run['ledger']['ledger']['total'][name])
        resources['priced_units'] += float(run['ledger']['priced_units'])
        resources['planning_cpu_s'] += float(run['runtime']['planning_cpu_s'])
        resources['update_cpu_s'] += float(run['runtime']['update_cpu_s'])
    present = counts > 0
    if np.any(counts[present] != 9) or len(identities) != 1:
        raise ValueError('Question repeats must be nine and the execution identity must be common')
    means = np.zeros_like(sums)
    means[present] = sums[present] / counts[present, None]
    if any(len(values) != 1 for values in history_counts.values()):
        raise ValueError('Historical initialization size changed across repeated paths')
    return {'means': means, 'counts': counts, 'labels': labels, 'qfolds': qfolds,
            'accuracy': scope_means(means[:, 0], labels),
            'no_feedback_accuracy': scope_means(means[:, 1], labels),
            'fixed_accuracy': scope_means(means[:, 2], labels),
            'history_counts': {fold: next(iter(values)) for fold, values in history_counts.items()},
            'order_hashes': order_hashes, 'cap_hashes': cap_hashes,
            'resources_mean_per_120_or_remaining_question_path': {key: value / 45 for key, value in resources.items()},
            'identity': identities.pop()}


def compare_points(left, right):
    if (left['labels'] != right['labels'] or not np.array_equal(left['counts'], right['counts'])
            or left['order_hashes'] != right['order_hashes'] or left['cap_hashes'] != right['cap_hashes']):
        raise ValueError('Paired methods changed original questions, orders, repeats or hard resource caps')
    if not np.array_equal(left['means'][:, 1:], right['means'][:, 1:]):
        raise ValueError('No-feedback and fixed references changed between paired strategies')
    return scope_means(left['means'][:, 0] - right['means'][:, 0], left['labels'])


def context_points(indexes, group=None, predictor='P2', budget=20):
    result = {}
    for _, stage, left, right in CONTRASTS:
        for method in (left, right):
            key = (stage, method)
            if key in result:
                continue
            index = indexes['DELETE'] if group is not None else indexes[stage]
            result[key] = stream_point(select_rows(index, stage, method, predictor, budget, group))
    identities = {point['identity'] for point in result.values()}
    if len(identities) != 1:
        raise ValueError('The frozen primary family mixes execution identities')
    return result


def contrast_vector(points):
    details = []
    for name, stage, left_name, right_name in CONTRASTS:
        left, right = points[(stage, left_name)], points[(stage, right_name)]
        delta = compare_points(left, right)
        cost_left = left['resources_mean_per_120_or_remaining_question_path']
        cost_right = right['resources_mean_per_120_or_remaining_question_path']
        details.append({'name': name, 'stage': stage, 'method_a': left_name, 'method_b': right_name,
                        'accuracy_difference': delta,
                        'accuracy_a': left['accuracy'], 'accuracy_b': right['accuracy'],
                        'no_feedback_accuracy': left['no_feedback_accuracy'],
                        'actual_resource_difference_mean_per_path': {key: cost_left[key] - cost_right[key] for key in cost_left},
                        'actual_resources_a_mean_per_path': cost_left, 'actual_resources_b_mean_per_path': cost_right})
    return np.asarray([item['accuracy_difference']['balanced_subtask'] for item in details]), details


def no_feedback_vector(points):
    details = []
    for stage in ('B1', 'B2'):
        point = points[(stage, 'decision_cost')]
        delta = scope_means(point['means'][:, 0] - point['means'][:, 1], point['labels'])
        details.append({'name': stage + '_decision_cost_vs_no_feedback',
                        'stage': stage, 'accuracy_difference': delta,
                        'decision_accuracy': point['accuracy'],
                        'no_feedback_accuracy': point['no_feedback_accuracy']})
    return np.asarray([item['accuracy_difference']['balanced_subtask'] for item in details]), details


def delete_readiness(index):
    needed = {(group, stage, method) for group in range(20)
              for _, stage, left, right in CONTRASTS for method in (left, right)}
    missing = []
    for group, stage, method in sorted(needed):
        rows = select_rows(index, stage, method, group=group)
        try:
            require_paths(rows)
        except ValueError:
            missing.append({'delete_group': group, 'stage': stage, 'method': method, 'paths_available': len(rows), 'paths_required': 45})
    return {'ready': not missing, 'required_contexts': len(needed), 'required_paths': 6300,
            'indexed_paths': len(index['rows']) if index else 0, 'incomplete_contexts': missing}


def reconstruct_design(labels):
    if set(labels) != set(range(600)):
        raise ValueError('The full OOF primary family must cover exactly 600 original questions')
    train = {'task': np.asarray([labels[q][0] for q in range(600)]),
             'sub': np.asarray([labels[q][1] for q in range(600)]), 'asset_root': ROOT}
    splits = make_splits(train)
    groups = [[] for _ in range(20)]
    rng = np.random.default_rng(SEED)
    for task in ('gsm8k', 'bbh'):
        rows = rng.permutation(np.flatnonzero(train['task'] == task))
        for position, q in enumerate(rows):
            groups[position % 20].append(int(q))
    return splits, [set(group) for group in groups]


def verify_roles(point, splits, deleted):
    expected_present = set(range(600)) - deleted
    if set(point['labels']) != expected_present:
        raise ValueError('Deleted questions differ from the frozen original-question groups')
    history, references, online = {}, {}, {}
    for fold in splits['folds']:
        number = fold['fold']
        expected_history = len(set(fold['nested'][240]) - deleted)
        if point['history_counts'][number] != expected_history:
            raise ValueError('Recorded historical initialization size violates all-role deletion')
        history[number] = {'before': 240, 'after': expected_history, 'removed': 240 - expected_history}
        after_reference = len(set(fold['references'][240]) - deleted)
        references[number] = {'before': 54, 'reconstructed_after': after_reference, 'removed': 54 - after_reference}
        online[number] = len(set(fold['stream']) - deleted)
        for gen, order in itertools.product(CONFIG['generation_seeds'], CONFIG['order_seeds']):
            remaining = [int(q) for q in fold['orders'][order] if int(q) not in deleted]
            if point['order_hashes'][(number, gen, order)] != digest(remaining):
                raise ValueError('The remaining online order changed after deletion')
    return {'retained_original_questions': len(expected_present), 'deleted_original_questions': len(deleted),
            'retained_gsm': sum(label[0] == 'gsm8k' for label in point['labels'].values()),
            'retained_bbh': sum(label[0] == 'bbh' for label in point['labels'].values()),
            'repeats_per_retained_question': 9, 'history_removal_by_fold': history,
            'reference_removal_expected_by_fold': references, 'online_questions_by_fold': online,
            'reference_verification_scope': 'Expected identity counts reconstructed from labels and frozen folds; actual reference rows are bound to frozen template_for code and absent from replay receipts',
            'remaining_order_relative_sequence_verified': True}


def conditional_statistics(indexes, predictors, budgets, reps):
    results = []
    for predictor, budget in itertools.product(predictors, budgets):
        for name, stage, left, right in CONTRASTS:
            rows_a = select_rows(indexes[stage], stage, left, predictor, budget)
            rows_b = select_rows(indexes[stage], stage, right, predictor, budget)
            require_paths(rows_a)
            require_paths(rows_b)
            runs = [load_run(row) for row in rows_a + rows_b]
            comparison = paired_compare(runs, left, right, predictor=predictor, stage=stage,
                                        B=budget, history_size=240, reps=reps, seed=SEED)
            comparison['contrast_name'] = name
            results.append(comparison)
            print(json.dumps({'phase': 'CONDITIONAL_TRACE', 'predictor': predictor, 'B': budget,
                              'contrast': name, 'balanced': comparison['metrics']['balanced_subtask']}), flush=True)
    family = {item['contrast_name']: item['metrics']['balanced_subtask']['p_value_descriptive_two_sided']
              for item in results if item['meta']['predictor'] == 'P2' and item['meta']['B'] == 20}
    return {'classification': 'conditional_on_realized_paths_descriptive', 'comparisons': results,
            'holm_five_preregistered_contrasts': holm_adjust(family) if len(family) == 5 else None,
            'learning_history_changes_replayed': False,
            'predictor_budget_sensitivity_scope': 'P1/P3 and gold40 are descriptive sensitivities; main family remains P2/gold20',
            'statistics_code_sha256': sha256(Path(__file__).with_name('e3_statistics.py')),
            'index_hashes': {stage: indexes[stage]['sha256'] for stage in ('B1', 'B2')}}


def self_test():
    full = np.arange(1, 6, dtype=float) / 100
    deleted = full + np.arange(-2, 3)[:, None] * np.arange(1, 6)[None] / 10000
    result = jackknife_ci(deleted, full, n_groups=5, reps=200, seed=SEED)
    assert np.asarray(result['ci95']).shape == (5, 2)
    assert np.asarray(result['simultaneous_ci95']).shape == (5, 2)
    assert np.asarray(result['p_value_approximate_two_sided']).shape == (5,)
    diagnostic = jackknife_ci(deleted[:, :2], full[:2], n_groups=5, reps=200, seed=SEED)
    assert np.asarray(diagnostic['ci95']).shape == (2, 2)
    labels = {q: ('gsm8k', '') if q < 300 else ('bbh', str((q - 300) % 27)) for q in range(600)}
    assert scope_means(np.ones(600), labels)['balanced_subtask'] == 1
    json.dumps(result, allow_nan=False)
    print(json.dumps({'software_qualification': 'PASS', 'synthetic_deletion_groups': 5,
                      'contrast_vector_dimension': 5, 'actual_DELETE_results_evaluated': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--conditional-only', action='store_true')
    parser.add_argument('--skip-conditional', action='store_true')
    parser.add_argument('--predictors', nargs='+', default=['P1', 'P2', 'P3'], choices=['P1', 'P2', 'P3'])
    parser.add_argument('--budgets', nargs='+', type=int, default=[20, 40])
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    indexes = {stage: read_index(stage) for stage in ('B1', 'B2', 'DELETE')}
    if indexes['B1'] is None or indexes['B2'] is None:
        raise ValueError('Complete formal B1 and B2 indexes are required')
    points = context_points(indexes)
    full, point_details = contrast_vector(points)
    no_feedback_full, no_feedback_details = no_feedback_vector(points)
    labels = points[('B1', 'decision_cost')]['labels']
    splits, deletion_groups = reconstruct_design(labels)
    for point in points.values():
        verify_roles(point, splits, set())
    conditional = None
    conditional_path = PRIVATE / 'PRIMARY_CONDITIONAL_TRACE.json'
    if not args.skip_conditional:
        conditional = conditional_statistics(indexes, args.predictors, args.budgets, args.replicates)
        put(conditional_path, conditional)
    elif conditional_path.exists():
        cached = json.loads(conditional_path.read_text(encoding='utf-8'))
        if (cached['index_hashes'] == {stage: indexes[stage]['sha256'] for stage in ('B1', 'B2')}
                and cached['statistics_code_sha256'] == sha256(Path(__file__).with_name('e3_statistics.py'))):
            conditional = cached
    readiness = delete_readiness(indexes['DELETE'])
    public = {'schema': 'frozen_e3_primary_statistics_v1', 'status': 'WAITING_FOR_COMPLETE_DELETE_REPLAYS',
              'predictor': 'P2', 'gold_budget_per_fold': 20, 'nominal_history_size': 240,
              'primary_scope': 'balanced_subtask', 'family_order': [row[0] for row in CONTRASTS],
              'full_data_point_contrasts': point_details, 'conditional_trace_diagnostics': conditional,
              'delete_replay_readiness': readiness, 'end_to_end': None,
              'secondary_no_feedback': {
                  'point_contrasts': no_feedback_details, 'end_to_end': None,
                  'multiplicity': 'unadjusted diagnostic marginal intervals outside the frozen five-contrast Holm family',
                  'reference': 'per-trace no_feedback_reward from the same original or deleted-history initial prior',
                  'additional_strategy_replays': 0},
              'code_sha256': sha256(Path(__file__)),
              'statistics_code_sha256': sha256(Path(__file__).with_name('e3_statistics.py')),
              'source_indexes': {stage: {'name': index['name'], 'sha256': index['sha256']} for stage, index in indexes.items() if index},
              'data_scope': 'original train600 development replay; validation/fresh400/sealed payloads unopened',
              'new_llm_calls': 0}
    if args.conditional_only or not readiness['ready']:
        if args.conditional_only:
            public['status'] = 'CONDITIONAL_TRACE_DIAGNOSTICS_ONLY'
        put(PUBLIC / 'PRIMARY_STATISTICS.json', public)
        print(json.dumps({'status': public['status'], 'delete_ready': readiness['ready'],
                          'incomplete_delete_contexts': len(readiness['incomplete_contexts'])}), flush=True)
        return 0 if args.conditional_only else 2
    deleted_values = []
    no_feedback_deleted_values = []
    group_checks = []
    identities = {point['identity'] for point in points.values()}
    for group in range(20):
        deleted_points = context_points(indexes, group=group)
        identity = {point['identity'] for point in deleted_points.values()}
        if identity != identities:
            raise ValueError('Deleted and full replays use different frozen execution identities')
        audits = [verify_roles(point, splits, deletion_groups[group]) for point in deleted_points.values()]
        if any(audit != audits[0] for audit in audits):
            raise ValueError('Role removal differs between methods within one deletion group')
        audit = audits[0]
        if (audit['deleted_original_questions'], audit['retained_gsm'], audit['retained_bbh']) != (30, 285, 285):
            raise ValueError('Each deleted group must remove 15 GSM and 15 BBH questions')
        vector, _ = contrast_vector(deleted_points)
        no_feedback_deleted, _ = no_feedback_vector(deleted_points)
        deleted_values.append(vector)
        no_feedback_deleted_values.append(no_feedback_deleted)
        group_checks.append({'delete_group': group, **audit})
        print(json.dumps({'phase': 'END_TO_END_DELETE', 'completed_groups': group + 1,
                          'retained_original_questions': audit['retained_original_questions'],
                          'contrast_vector': vector.tolist()}), flush=True)
    deleted_values = np.asarray(deleted_values)
    no_feedback_deleted_values = np.asarray(no_feedback_deleted_values)
    uncertainty = jackknife_ci(deleted_values, full, n_groups=20, reps=args.replicates, seed=SEED)
    no_feedback_uncertainty = jackknife_ci(no_feedback_deleted_values, no_feedback_full,
                                         n_groups=20, reps=args.replicates, seed=SEED)
    no_feedback_uncertainty.pop('simultaneous_ci95')
    no_feedback_uncertainty.pop('simultaneous_family_size')
    family_p = dict(zip(public['family_order'], uncertainty['p_value_approximate_two_sided']))
    private = {'family_order': public['family_order'], 'full_vector': full.tolist(),
               'delete_values_shape': list(deleted_values.shape), 'delete_vectors': deleted_values.tolist(),
               'secondary_no_feedback_full_vector': no_feedback_full.tolist(),
               'secondary_no_feedback_delete_shape': list(no_feedback_deleted_values.shape),
               'secondary_no_feedback_delete_vectors': no_feedback_deleted_values.tolist(),
               'group_checks': group_checks,
               'removal_rule': '15 GSM and 15 BBH original question IDs removed simultaneously from all historical, reference and online roles; remaining stream relative order retained; gold cap remains20',
               'source_indexes': public['source_indexes'], 'code_sha256': public['code_sha256']}
    put(PRIVATE / 'PRIMARY_DELETE_VALUES.json', private)
    public['status'] = 'END_TO_END_DEVELOPMENT_COMPLETE'
    public['secondary_no_feedback']['end_to_end'] = no_feedback_uncertainty
    public['end_to_end'] = {'balanced_subtask_five_contrast_jackknife': uncertainty,
                          'holm_five_preregistered_contrasts': holm_adjust(family_p),
                          'deleted_groups': 20, 'retained_questions_per_group': 570,
                          'generation_order_repeats_per_original_question': 9,
                          'scope': 'preregistered approximate inference on the CPU development replay; external generalization remains a separate evaluation',
                          'limitations': 'discrete decisions and budget thresholds are nonsmooth; overlapping-fold and feedback-history changes are included by replay, but finite-sample coverage is approximate'}
    put(PUBLIC / 'PRIMARY_STATISTICS.json', public)
    print(json.dumps({'status': public['status'], 'delete_vector_shape': list(deleted_values.shape),
                      'full_contrast_vector': full.tolist(), 'holm': public['end_to_end']['holm_five_preregistered_contrasts']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

