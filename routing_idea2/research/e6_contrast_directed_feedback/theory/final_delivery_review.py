"""Independent delivery review of E6 frozen artifacts. No policy changes."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
WORK = ROOT / 'research/e6_contrast_directed_feedback'
PUBLIC = WORK / 'public'
PRIVATE = WORK / 'private'
sys.path.insert(0, str(ROOT / 'research/e5_paired_gain_validation/public'))
from e5_common import load_train, TEAMS

CHECKS = 0
MAX_ERROR = 0.


def equal(actual, expected, tolerance=1e-12):
    global CHECKS, MAX_ERROR
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (a.shape, b.shape)
    CHECKS += max(1, a.size)
    if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
        error = float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0.))
        MAX_ERROR = max(MAX_ERROR, error)
        assert error <= tolerance, error
    else:
        assert np.array_equal(a, b)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def independent_resources(data, rows, sets, gold=0):
    result = {name: 0 for name in ('gold', 'member_calls', 'aggregator_calls', 'scoring_ops',
                                   'input_tokens', 'output_tokens', 'tokens')}
    result['source_model_wall_s'] = 0.
    for q, actions in zip(rows, sets):
        actions = sorted(set(map(int, actions)))
        members = sorted({model for action in actions for model in TEAMS[action]})
        for generation in range(3):
            for stem, selected, count_name in (('member', members, 'member_calls'),
                                                ('aggregate', actions, 'aggregator_calls')):
                for suffix, destination in (('execution_counts', count_name), ('prompt_tokens', 'input_tokens'),
                    ('completion_tokens', 'output_tokens'), ('wall_time_s', 'source_model_wall_s')):
                    values = data['cost'][stem + '_' + suffix][generation, q, selected]
                    result[destination] += float(values.sum()) if suffix == 'wall_time_s' else int(values.sum())
            result['scoring_ops'] += len(actions)
    result['gold'] = gold
    result['tokens'] = result['input_tokens'] + result['output_tokens']
    return result


def main():
    python_files = sorted(PUBLIC.rglob('*.py'))
    for path in python_files:
        compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
    data, references = load_train()
    A = read(PUBLIC / 'a/A_RESULTS.json')
    B = read(PUBLIC / 'b/E6_B_RESULTS.json')
    C = read(PUBLIC / 'c/E6_C_RESULTS.json')
    STATS = read(PUBLIC / 'E6_PRIMARY_STATISTICS.json')
    COST = read(PUBLIC / 'E6_PROVENANCE_AND_COST.json')
    COMPLETE = read(PUBLIC / 'E6_COMPLETE_RESULTS.json')
    CONFIG = read(PUBLIC / 'E6_RUN_CONFIG.json')
    text_summary = (PUBLIC / 'EXPERIMENT_SUMMARY.md').read_text(encoding='utf-8')
    equal(list(COMPLETE['main_real_accuracy'].values()), [v['accuracy'] for v in C['table']['36'].values()])
    equal(COMPLETE['default_Q_accuracy'], C['default_accuracy'])
    for key, value in independent_resources(data, np.arange(600), [range(20)] * 600, gold=600).items():
        equal(COST['source_cache_once'][key], value)
    assert not CONFIG['fresh400_evaluated'] if 'fresh400_evaluated' in CONFIG else True
    assert COST['fresh400_evaluated'] is False and COST['sealed200_evaluated'] is False
    equal(COST['physical_new_member_calls'], 0)
    equal(COST['physical_new_aggregator_calls'], 0)
    for record in COST['reference_inputs'].values():
        if isinstance(record, dict) and 'path' in record:
            path = Path(record['path'])
            assert path.exists()
            equal(hashlib.sha256(path.read_bytes()).hexdigest(), record['sha256'])

    b_counts = dict(gap_lower=0, ci_better=0, ci_worse=0, worse_than_no_pilot=0)
    assert len(B['cases']) == 12
    for case in B['cases']:
        label = (f"{case['family']}_ann{int(case['annotation_fraction'] * 100)}_"
                 f"exec{int(case['execution_fraction'] * 100)}_pilot{case['pilot_per_stratum']}")
        with np.load(PRIVATE / f'b/cases/{label}.npz') as archive:
            estimates = archive['estimates']
            truth = archive['true_gains']
            selected = np.argmax(estimates >= estimates.max(axis=-1, keepdims=True) - 1e-12, axis=-1)
            regret = truth.max() - truth[selected]
            equal(selected, archive['selected'])
            equal(regret, archive['regret'])
            equal(estimates[:, 1], estimates[:, 2])
            equal(estimates[:, 4], estimates[:, 5])
        for j, method in enumerate(B['config']['methods']):
            saved = case['methods'][method]['policy_selection']
            equal(saved['mean_value_regret'], regret[:, j].mean())
            equal(saved['nonoptimal_probability'], (regret[:, j] > 1e-12).mean())
            equal(saved['probability_regret_above_epsilon'], (regret[:, j] > .01 + 1e-12).mean())
        paired = regret[:, 3] - regret[:, 0]
        mean = paired.mean()
        se = paired.std(ddof=1) / np.sqrt(len(paired))
        ci = np.asarray([mean - 1.96 * se, mean + 1.96 * se])
        stored = case['paired_comparisons']['Estimated-GapContrast_minus_Uniform']['regret']
        equal(stored['mean'], mean)
        equal(stored['paired_mc_ci95'], ci)
        b_counts['gap_lower'] += int(mean < -1e-12)
        b_counts['ci_better'] += int(ci[1] < 0)
        b_counts['ci_worse'] += int(ci[0] > 0)
        b_counts['worse_than_no_pilot'] += int(regret[:, 3].mean() > regret[:, 7].mean() + 1e-12)
    equal(list(b_counts.values()), [9, 3, 1, 12])

    with np.load(PRIVATE / 'a/PRIMARY_A_CONTRAST.npz') as archive:
        point_a = float(archive['observed'])
        draws_a = archive['bootstrap']
        equal(archive['question_contribution'].mean(), point_a)
    equal(STATS['main_comparisons'][0]['point'], point_a)
    equal(STATS['main_comparisons'][0]['centered_bootstrap_p'],
          (1 + np.sum(np.abs(draws_a - point_a) >= abs(point_a) - 1e-15)) / (len(draws_a) + 1))
    equal(STATS['main_comparisons'][0]['ci95'], np.quantile(draws_a, [.025, .975]))
    equal(STATS['main_comparisons'][0]['ci_family_bonferroni'], np.quantile(draws_a, [.05 / 6, 1 - .05 / 6]))

    score_parts, seed_parts, baseline_parts, row_parts, costs, service_parts, mc_parts = [], [], [], [], [], [], []
    for fold in range(5):
        with np.load(PRIVATE / f'c/C_f{fold}.npz') as archive:
            raw = {key: archive[key] for key in archive.files}
        with np.load(PRIVATE / f'c/EVALUATED_C_f{fold}.npz') as archive:
            evaluated = {key: archive[key] for key in archive.files}
        selected = raw['selected']
        Q = raw['Q']
        Rb = evaluated['RbQ']
        D = evaluated['DQ']
        masks = raw['masksQ']
        # Select each randomized policy's frozen mask, then average randomization within a question.
        expected_switch = np.take(masks, selected, axis=0).mean(axis=2)
        seed_score = Rb[None, None] + expected_switch[..., None] * D[None, None]
        seed_parts.append(seed_score)
        score_parts.append(seed_score.mean(axis=-1))
        baseline_parts.append(Rb.mean(axis=1))
        row_parts.extend(Q.tolist())
        mc_parts.append(Rb.mean() + evaluated['gainsQ'][selected])
        costs.append(raw['purchases'].astype(float) @ raw['costmatrix'])
        with np.load(PRIVATE / f'a/fold{fold}_policies.npz') as archive:
            E, base, candidate = archive['E'], int(archive['base']), archive['candidate']
        lookup = {int(q): i for i, q in enumerate(E)}
        P = raw['P']
        costnames = raw['costnames'].tolist()
        for i, q in enumerate(P):
            actual = independent_resources(data, [int(q)], [[base, int(candidate[lookup[int(q)]])]], gold=1)
            equal(raw['costmatrix'][i], [actual[name] for name in costnames], tolerance=1e-9)
        strategy_costs = []
        for mask in masks:
            actions = [[int(candidate[lookup[int(q)]]) if mask[i] else base] for i, q in enumerate(Q)]
            measured = independent_resources(data, Q, actions)
            strategy_costs.append([measured[name] for name in costnames])
        strategy_costs = np.asarray(strategy_costs)
        service = np.empty((3, 5, len(costnames)))
        for bi in range(3):
            for mi in range(5):
                proportions = np.bincount(selected[bi, mi], minlength=len(masks)) / selected.shape[2]
                service[bi, mi] = proportions @ strategy_costs
        service_parts.append(service)
        equal(selected[:, 1], selected[:, 2])
        equal(raw['purchases'][:, 1], raw['purchases'][:, 2])

    score = np.concatenate(score_parts, axis=2)
    seed_score = np.concatenate(seed_parts, axis=2)
    baseline = np.concatenate(baseline_parts)
    Q = np.asarray(row_parts)
    assert len(Q) == len(set(Q)) == 300
    task, sub = data['task'][Q], data['sub'][Q]
    folds = np.repeat(np.arange(5), 60)
    rng = np.random.default_rng(20261112)
    bootstrap_acc = []
    bootstrap_gain = []
    for _ in range(5000):
        indices = np.concatenate([rng.choice(cell, len(cell), replace=True)
            for fold in range(5) for name in ('gsm8k', 'bbh')
            for cell in [np.flatnonzero((folds == fold) & (task == name))]])
        bootstrap_acc.append(score[:, :, indices].mean(axis=2))
        bootstrap_gain.append((score[:, :, indices] - baseline[indices]).mean(axis=2))
    bootstrap_acc, bootstrap_gain = np.asarray(bootstrap_acc), np.asarray(bootstrap_gain)
    costmean = np.mean(np.stack(costs), axis=(0, 3))
    servicemean = np.mean(np.stack(service_parts), axis=0)
    mc = np.stack(mc_parts)
    with np.load(PRIVATE / 'c/PAIRED_C_RESULTS.npz') as archive:
        equal(archive['score'], score)
        equal(archive['score_seed'], seed_score)
        equal(archive['base'], baseline)
        equal(archive['bootstrap_accuracy'], bootstrap_acc)
        equal(archive['bootstrap_gain'], bootstrap_gain)
        equal(archive['cost_mean'], costmean, tolerance=1e-7)
        equal(archive['service_mean'], servicemean, tolerance=1e-7)
        equal(archive['MC_accuracy'], mc)

    for bi, budget in enumerate(C['budgets']):
        for mi, name in enumerate(C['methods']):
            row = C['table'][str(budget)][name]
            equal(row['accuracy'], score[bi, mi].mean())
            equal(row['seed_accuracy'], seed_score[bi, mi].mean(axis=0))
            task_values = [score[bi, mi, task == name].mean() for name in ('gsm8k', 'bbh')]
            equal(list(row['task'].values()), task_values)
            equal(row['task_macro'], np.mean(task_values))
            sub_values = []
            for subname, stored in row['BBH_subtasks'].items():
                value = score[bi, mi, (task == 'bbh') & (sub == subname)].mean()
                equal(stored, value)
                sub_values.append(value)
            equal(row['GSM_BBH_subtask_macro'], .5 * task_values[0] + .5 * np.mean(sub_values))
            equal(row['gain_default']['ci95'], np.quantile(bootstrap_gain[:, bi, mi], [.025, .975]))
            equal(list(row['feedback_cost'].values()), costmean[bi, mi], tolerance=1e-7)
            equal(list(row['selected_Q_service_cost'].values()), servicemean[bi, mi], tolerance=1e-7)
            equal(list(row['feedback_plus_Q_service_cost'].values()), costmean[bi, mi] + servicemean[bi, mi], tolerance=1e-7)
            equal(row['feedback_cost']['gold'], budget)
            equal(row['feedback_cost']['aggregator_calls'], 6 * budget)
        for label, first, second in (('Contrast-Marginal', 2, 1), ('Contrast-Uniform', 2, 0),
            ('GapContrast-Uniform', 3, 0), ('GapContrast-Marginal', 3, 1), ('NoPilotUniform-Uniform', 4, 0)):
            contrast = C['paired_comparisons'][f'{label}@{budget}']
            difference = score[bi, first] - score[bi, second]
            equal(contrast['point'], difference.mean())
            draws = bootstrap_acc[:, bi, first] - bootstrap_acc[:, bi, second]
            equal(contrast['ci95'], np.quantile(draws, [.025, .975]))
            equal(contrast['centered_bootstrap_p'],
                  (1 + np.sum(np.abs(draws - contrast['point']) >= abs(contrast['point']) - 1e-15)) / (len(draws) + 1))
            equal(contrast['seed_difference'], (seed_score[bi, first] - seed_score[bi, second]).mean(axis=0))
            cluster = (mc[:, bi, first] - mc[:, bi, second]).reshape(5, 50, 40).mean(axis=2)
            se = np.sqrt(np.sum(cluster.var(axis=1, ddof=1) / 50)) / 5
            equal(contrast['paired_MC']['cluster_SE'], se)
    pvalues = np.asarray([row['centered_bootstrap_p'] for row in STATS['main_comparisons']])
    order = np.argsort(pvalues)
    adjusted = np.empty(3)
    previous = 0.
    for rank, j in enumerate(order):
        previous = max(previous, min(1., (3 - rank) * pvalues[j]))
        adjusted[j] = previous
    equal([row['Holm_p'] for row in STATS['main_comparisons']], adjusted)

    # Actual private identifiers and full question strings must not appear in public payloads.
    public_files = [path for path in PUBLIC.rglob('*') if path.is_file()]
    decoded = {path: path.read_text(encoding='utf-8-sig') for path in public_files}
    forbidden = [str(x) for x in data['ids']] + [str(x) for x in data['canonical']]
    with np.load(Path(references['inputs']['path']), allow_pickle=False) as archive:
        for column in archive.files:
            array = archive[column]
            if array.dtype.kind in 'US' and array.ndim == 1:
                forbidden.extend(str(x) for x in array if len(str(x)) >= 60)
    hits = []
    for path, text in decoded.items():
        for value in forbidden:
            if len(value) >= 12 and value in text:
                hits.append(dict(file=str(path.relative_to(PUBLIC)), kind='private_id_or_full_question'))
                break
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (RSA |OPENSSH )?PRIVATE KEY', text)
    assert not hits, hits
    # Fitted direction model coefficients are private. Public synthetic design vectors are permitted.
    for name in ('A_RESULTS.json', 'A_SUPPORT_AND_COST_DIAGNOSTICS.json'):
        payload = (PUBLIC / 'a' / name).read_text(encoding='utf-8')
        assert '"beta"' not in payload and '"heads"' not in payload and '"scaler_mean"' not in payload
    assert all(path.suffix in ('.py', '.md', '.json') for path in public_files)
    temporary_path = str(Path.cwd() / ('.' + 'e6_' + 'staging')).casefold()
    temporary_path_forward = temporary_path.replace('\\', '/')
    assert all(temporary_path not in text.casefold() and temporary_path_forward not in text.casefold()
               for text in decoded.values())

    missing = []
    for name in ('REPRODUCE.md', 'prepare_e6.py', 'verify_e6_delivery.py', 'PUBLIC_DELIVERY_MANIFEST.json'):
        if not (PUBLIC / name).exists():
            missing.append(name + ' is referenced but not yet present at review time')
    audit = dict(status='PASS' if not missing else 'PASS_WITH_PENDING_DOCUMENT', scalar_checks=CHECKS,
        maximum_numeric_reconstruction_error=MAX_ERROR,
        python_compile_count=len(python_files), compiled_files=[str(p.relative_to(PUBLIC)) for p in python_files],
        reviewed_files={str(path.relative_to(PUBLIC)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in public_files if path.name not in
                        ('E6_FINAL_REVIEW.json', 'PUBLIC_DELIVERY_MANIFEST.json', 'E6_DELIVERY_AUDIT.json')},
        review_file_self_hash_excluded=True,
        reviewed_file_hash_exclusions=['E6_FINAL_REVIEW.json', 'PUBLIC_DELIVERY_MANIFEST.json', 'E6_DELIVERY_AUDIT.json'],
        independently_reconstructed=['all Q question and seed scores', '5000 paired Q bootstraps',
            'default covariance in gain intervals', 'all task and BBH27 macros', '15 actual-cost means and service accounts',
            'all private P pair execution costs', '15 MC paired cluster standard errors', 'Holm3 family', 'source cache totals',
            '12 synthetic selection arrays and regret comparisons'],
        synthetic_summary_counts=b_counts,
        privacy=dict(public_files=len(public_files), actual_private_identifier_or_question_matches=0,
            credentials_pattern_hits=0, real_fitted_head_parameters_public=False),
        scientific_boundary='existing train600 development, overlapping outer histories, P-conditional Q intervals, no fresh or sealed evaluation',
        pending=missing,
        caution='Raw Python compilation verifies syntax. Full replay entrypoints require private historical asset tree and documented environment.')
    (PUBLIC / 'E6_FINAL_REVIEW.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in audit.items() if key not in ('reviewed_files', 'compiled_files')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
