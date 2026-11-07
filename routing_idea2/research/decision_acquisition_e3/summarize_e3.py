"""Stream indexed E3 receipts one context at a time into aggregate-only reports."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path

from run_e3 import CONFIG, PRIVATE, PUBLIC, jobs_for, sensitivity_jobs, put, sha256
import e3_statistics
from e3_statistics import aggregate


CONTEXT_FIELDS = ('predictor', 'stage', 'method', 'B', 'history_size', 'delete_group')
PATH_FIELDS = ('fold', 'gen', 'order')
RESOURCE_NAMES = ('member_calls', 'aggregator_calls', 'gold', 'scoring_ops',
                  'input_tokens', 'output_tokens', 'tokens', 'source_model_wall_s')


def context_key(meta):
    return tuple(meta.get(key) for key in CONTEXT_FIELDS)


def path_key(meta):
    return tuple(meta[key] for key in PATH_FIELDS)


def expected_contexts(stage):
    if stage in ('A', 'B1', 'B2', 'J', 'D'):
        return {context_key(job['meta']) for job in jobs_for(stage, 'summary-design-only')}
    if stage == 'SENSITIVITY':
        return {context_key(job['meta']) for job in sensitivity_jobs('summary-design-only')}
    return None


def load_context(receipts):
    runs = []
    roots = (PRIVATE / 'runs').resolve()
    for receipt in receipts:
        path = Path(receipt['path']).resolve()
        if not path.is_relative_to(roots):
            raise ValueError('An indexed receipt must be inside the private runs directory')
        run = json.loads(gzip.decompress(path.read_bytes()))
        if run['meta'] != receipt['meta']:
            raise ValueError('Receipt metadata differs from its run payload')
        if len(run['traces']) != run['ledger']['stream_questions']:
            raise ValueError('The indexed run has incomplete service coverage')
        if not all(trace['service_frozen_before_feedback'] for trace in run['traces']):
            raise ValueError('A current service decision was changed after feedback')
        if int(run['runtime']['new_llm_calls']) != 0:
            raise ValueError('This cached replay report requires zero new LLM calls')
        runs.append(run)
    paths = [path_key(run['meta']) for run in runs]
    if len(set(paths)) != len(paths):
        raise ValueError('Repeated indexed fold/generation/order path in a context')
    identities = {run['identity'] for run in runs}
    if len(identities) != 1:
        raise ValueError('A context mixes execution identities')
    return runs, identities.pop()


def complete_grid(runs):
    expected = set(itertools.product(range(CONFIG['folds']),
                                     CONFIG['generation_seeds'], CONFIG['order_seeds']))
    return {path_key(run['meta']) for run in runs} == expected


def mean_resources(rows):
    return {name: sum(float(row[name]) for row in rows) / len(rows) for name in RESOURCE_NAMES}


def execution_accounts(runs):
    accounts = {}
    for account in ('service', 'audit', 'total'):
        accounts[account] = mean_resources([run['ledger']['ledger'][account] for run in runs])
    historical = [run['historical_initialization_resources'] for run in runs]
    history_names = ('gold', 'member_calls', 'aggregator_calls', 'tokens', 'source_model_wall_s')
    return {
        'mean_online_resources_per_path': accounts,
        'mean_historical_initialization_resources_per_path': {
            name: sum(float(row[name]) for row in historical) / len(historical) for name in history_names
        },
        'mean_original_score_observations_per_path': sum(run['ledger']['original_score_observations'] for run in runs) / len(runs),
        'mean_audit_skips_per_path': sum(len(run['ledger']['audit_skips']) for run in runs) / len(runs),
        'mean_priced_units_per_path': sum(float(run['ledger']['priced_units']) for run in runs) / len(runs),
        'mean_runtime_process_time_seconds_per_path': sum(run['runtime']['process_time_s'] for run in runs) / len(runs),
        'gpu_time_seconds': None,
        'source_wall_note': 'Source model wall-time sums are execution cost proxies; GPU kernel time is unavailable',
        'history_note': 'Historical initialization is a shared already-observed cost and is separate from online service and audit accounts',
        'new_physical_model_calls': 0,
        'new_physical_gold_acquisition_calls': 0,
    }


def future_snapshot_audits(runs):
    names = ('remaining_questions', 'changed', 'corrected', 'harmed', 'net_reward')
    totals = dict.fromkeys(names, 0)
    events = positive = negative = purchases = 0
    for run in runs:
        for trace in run['traces']:
            purchases += int(trace['purchase'])
            audit = trace['future_feedback_influence']
            if audit is None:
                continue
            events += 1
            for name in names:
                totals[name] += int(audit[name])
            if audit['net_reward'] != audit['corrected'] - audit['harmed']:
                raise ValueError('Future snapshot net reward differs from correction minus harm')
            if not 0 <= audit['corrected'] + audit['harmed'] <= audit['changed'] <= audit['remaining_questions']:
                raise ValueError('Future snapshot influence counts are inconsistent')
            positive += int(audit['net_reward'] > 0)
            negative += int(audit['net_reward'] < 0)
    exposure = totals['remaining_questions']
    return {
        'reviewed_paths': len(runs), 'accepted_purchase_events': purchases,
        'snapshot_events': events, 'positive_net_snapshot_events': positive,
        'negative_net_snapshot_events': negative,
        'raw_repeated_path_totals': totals,
        'mean_per_path': {name: value / len(runs) for name, value in totals.items()},
        'mean_per_snapshot_event': {name: value / events if events else None for name, value in totals.items()},
        'net_reward_per_remaining_question_exposure': totals['net_reward'] / exposure if exposure else None,
        'scope': 'auditor-only frozen before/after current-feedback snapshots on the same remaining questions; later feedback updates omitted in each snapshot',
        'dependence_note': 'Snapshot future windows overlap and original questions repeat across seeds/orders. These counts are diagnostics and cannot be added to realized service corrections or independent sample size',
    }


def aggregate_immutable_context(runs):
    """Reuse pure validation and metric conversions within one immutable batch.

    The statistics implementation examines the same traces for task, seed,
    order and arrival summaries. Keep these two pure results only for the
    current context. Strong references prevent Python object ID reuse, and
    restoring both functions leaves subsequent callers unchanged.
    """
    original_values = e3_statistics._trace_values
    original_validate = e3_statistics._validate_run
    value_cache = {}
    validation_cache = {}

    def trace_values(trace):
        key = id(trace)
        if key not in value_cache:
            value_cache[key] = (trace, original_values(trace))
        return value_cache[key][1]

    def validate_run(run):
        key = id(run)
        if key not in validation_cache:
            validation_cache[key] = (run, original_validate(run))
        return validation_cache[key][1]

    e3_statistics._trace_values = trace_values
    e3_statistics._validate_run = validate_run
    try:
        return aggregate(runs)
    finally:
        e3_statistics._trace_values = original_values
        e3_statistics._validate_run = original_validate


def summarize_context(receipts, require_complete):
    runs, identity = load_context(receipts)
    full_grid = complete_grid(runs)
    if require_complete and not full_grid:
        raise ValueError('Formal stage summary requires all 45 paired paths in every context')
    result = aggregate_immutable_context(runs)
    if len(result['groups']) != 1:
        raise ValueError('The context loader must pass exactly one context to aggregate')
    item = result['groups'][0]
    if 'delete_group' in runs[0]['meta']:
        item['meta']['delete_group'] = runs[0]['meta']['delete_group']
    item['execution_identity'] = identity
    item['complete_path_grid'] = full_grid
    item['classification'] = 'complete_development_replay' if full_grid else 'incomplete_qualification_replay'
    item['execution_accounts'] = execution_accounts(runs)
    item['future_feedback_influence'] = future_snapshot_audits(runs)
    return item


def budget_curves(groups):
    contexts = defaultdict(list)
    for item in groups:
        meta = item['meta']
        key = tuple(meta.get(field) for field in CONTEXT_FIELDS if field != 'B')
        contexts[key].append(item)
    curves = []
    for key, items in contexts.items():
        items = sorted(items, key=lambda item: item['meta']['B'])
        comparable = (len({item['question_set_sha256'] for item in items}) == 1
                      and len({item['arrival_curves']['common_horizon'] for item in items}) == 1
                      and all(item['arrival_curves']['path_lengths_equal'] for item in items))
        points = [{'gold_cap': item['meta']['B'], 'metrics': item['metrics'],
                   'actual_resources': item['resources'],
                   'arrival_horizon_per_fold': item['arrival_curves']['common_horizon']}
                  for item in items]
        auc = {}
        if comparable and len(points) > 1:
            width = points[-1]['gold_cap'] - points[0]['gold_cap']
            if width > 0:
                for scope in ('gsm', 'bbh_macro', 'balanced_subtask', 'balanced_v2'):
                    if all(scope in point['metrics'] for point in points):
                        area = sum((right['gold_cap'] - left['gold_cap']) *
                                   (left['metrics'][scope]['accuracy'] + right['metrics'][scope]['accuracy']) / 2
                                   for left, right in zip(points[:-1], points[1:]))
                        auc[scope] = area / width
        targets = {}
        for scope in ('gsm', 'bbh_macro', 'balanced_subtask', 'balanced_v2'):
            if not all(scope in point['metrics'] for point in points):
                continue
            target = points[0]['metrics'][scope]['no_feedback_accuracy'] + 0.005
            attained = next((point for point in points if point['metrics'][scope]['accuracy'] >= target), None)
            targets[scope] = {'quality_target': target, 'rule': 'frozen no-feedback accuracy +0.005 absolute',
                              'status': 'attained_on_tested_grid' if attained else 'not_reached_on_tested_grid',
                              'first_attained_gold_cap': attained['gold_cap'] if attained else None,
                              'actual_resources': attained['actual_resources'] if attained else None,
                              'maximum_tested_gold_cap': points[-1]['gold_cap'],
                              'monotonic_quality_assumed': False}
        curves.append({'meta': dict(zip([field for field in CONTEXT_FIELDS if field != 'B'], key)),
                       'points': points, 'same_arrival_horizon_and_question_set': comparable,
                       'normalized_accuracy_auc_over_observed_gold_cap_range': auc,
                       'gold_cap_range': [points[0]['gold_cap'], points[-1]['gold_cap']],
                       'target_quality_cost': targets,
                       'actual_cost_interpolation': 'none; final quality and actual resources share the complete service horizon'})
    return curves


def percent(value):
    return '缺失' if value is None else f'{100 * value:.3f}'


def signed_percent(value):
    return '缺失' if value is None else f'{100 * value:+.3f}'


def render_feedback_report(summary):
    groups = [item for item in summary['groups'] if item['meta']['stage'] == 'A']
    if not groups:
        return None
    precheck = summary['classification'] == 'precheck_qualification_only'
    title = '# E3A反馈能否改善未来服务' + (' · 单路径预检' if precheck else '')
    rows = [title, '', '## 研究问题', '',
            '在每题服务决策冻结后获取真实组合评分，是否改善之后题目的选队。分别检查随机有限反馈、主组合全量反馈和20组合充分信息诊断。', '',
            '## 实验设置', '',
            '六模型、每题三模型、20组合及V2冻结聚合器保持原设置。历史初始化240题，正式设计为五折、三个生成种子与三个固定订单，每条流120题。', '',
            '主要口径为GSM8K准确率与BBH27子任务宏平均各占一半。原题先平均生成种子和订单，BBH按原题加权的V2口径单列。所有成本来自合法缓存执行集合。', '']
    if precheck:
        rows += ['**本稿只覆盖fold0、生成种子42、订单17的软件预检路径。数值用于检查汇总与账本流程，正式阶段报告另行生成。**', '']
    rows += ['## 对照与数值结果', '',
             '|预测器|方法|gold上限|路径数|GSM%|BBH子任务宏%|50/50宏%|V2宏%|相对不反馈pp|相对固定pp|期望纠错/破坏题数|Regret%|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    ordered = sorted(groups, key=lambda item: (item['meta']['predictor'], item['meta']['method'], item['meta']['B']))
    for item in ordered:
        meta = item['meta']
        metrics = item['metrics']
        main = metrics['balanced_subtask']
        rows.append(f"|{meta['predictor']}|{meta['method']}|{meta['B']}|{item['path_count']}|{percent(metrics['gsm']['accuracy'])}|{percent(metrics['bbh_macro']['accuracy'])}|{percent(main['accuracy'])}|{percent(metrics['balanced_v2']['accuracy'])}|{signed_percent(main['gain_vs_no_feedback'])}|{signed_percent(main['gain_vs_fixed'])}|{item['expected_corrections_on_original_questions']:.3f}/{item['expected_harms_on_original_questions']:.3f}|{percent(main['top1_regret'])}|")
    rows += ['', '纠错和破坏题数为各原题在生成种子、订单上的期望计数之和。正式完整网格对应600道原题，预检对应实际覆盖的120道原题。三次生成和重复订单共同进入平均。', '',
             '### 服务与反馈执行成本', '',
             '|预测器|方法|gold上限|实际gold/流|服务成员/流|审计成员/流|服务聚合/流|审计聚合/流|总Token/流|确定性评分/流|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in ordered:
        meta = item['meta']
        account = item['execution_accounts']['mean_online_resources_per_path']
        rows.append(f"|{meta['predictor']}|{meta['method']}|{meta['B']}|{account['total']['gold']:.3f}|{account['service']['member_calls']:.3f}|{account['audit']['member_calls']:.3f}|{account['service']['aggregator_calls']:.3f}|{account['audit']['aggregator_calls']:.3f}|{account['total']['tokens']:.1f}|{account['total']['scoring_ops']:.3f}|")
    rows += ['', 'none不购买反馈。full_observable为每道已服务题购买主组合评分，full_information为每道题购买20组合评分，后者单独承担额外成员及聚合开销。随机反馈按各自gold上限执行。', '',
             '### 反馈影响的未来快照诊断', '',
             '|预测器|方法|gold上限|快照事件/流|剩余题曝光/流|改变选队/流|快照纠错/流|快照破坏/流|快照净纠错/流|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in ordered:
        meta = item['meta']
        audit = item['future_feedback_influence']
        means = audit['mean_per_path']
        rows.append(f"|{meta['predictor']}|{meta['method']}|{meta['B']}|{audit['snapshot_events']/item['path_count']:.3f}|{means['remaining_questions']:.3f}|{means['changed']:.3f}|{means['corrected']:.3f}|{means['harmed']:.3f}|{means['net_reward']:+.3f}|")
    rows += ['', '每次快照比较同一反馈更新前后、在相同剩余题上的冻结预测。快照暂不执行后续反馈更新。不同快照覆盖的未来题重叠，上述计数用于反馈响应诊断，服务净纠错另由第一张表给出。', '',
             '### 生成种子、订单与折级方向', '']
    for field, title, key in (('gen', '生成种子', 'per_seed'), ('order', '订单', 'per_order'), ('fold', '折', 'per_fold')):
        rows += [f'#### {title}', '', '|预测器|方法|gold上限|方向|50/50宏%|相对不反馈pp|相对固定pp|',
                 '|---|---|---:|---|---:|---:|---:|']
        for item in ordered:
            meta = item['meta']
            for direction in item[key]:
                main = direction['metrics']['balanced_subtask']
                rows.append(f"|{meta['predictor']}|{meta['method']}|{meta['B']}|{direction[field]}|{percent(main['accuracy'])}|{signed_percent(main['gain_vs_no_feedback'])}|{signed_percent(main['gain_vs_fixed'])}|")
        rows.append('')
    rows += ['## 统计结论', '',
             '本阶段保留全部预定预测器与反馈配置。表中给出当前原题池上的完整配对点估计和各方向结果，主机制比较按预冻结P2、历史240题、gold20设置执行。', '',
             '正式45条路径共享原始600题，五折历史集重叠。后续配对区间分别标明条件trace描述性统计与端到端删组重放近似统计，单路径预检只用于软件资格核验。', '',
             '## 失败原因与局限', '',
             '出现负增益的配置完整保留。有限反馈、组合价值预测误差和生成随机性共同影响更新方向，单凭本阶段的总准确率无法分离具体原因。', '',
             '优势是当前题服务先冻结、后续反馈合法揭示，能核验反馈对未来预测的响应。风险是BBH每折每子任务只有2至3题，种子与订单重复、未来快照重叠，快照总和与实际服务增益具有不同含义。', '',
             '## 下一步', '',
             '按已冻结配置继续B1择时、B2邻居选择与联合动作机制比较。P1、P2、P3结果全部保留，A阶段结果不用于重选预测器或调整超参数。独立fresh评价在方法冻结后另行安排。', '',
             '完整分任务结果、固定到达时刻曲线、来源wall-time成本代理及逐配置统计保存在私有阶段汇总中。公开报告只展示聚合量。']
    return '\n'.join(rows) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('A', 'B1', 'B2', 'J', 'D', 'PRECHECK', 'SENSITIVITY', 'DELETE'))
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--report-name')
    args = parser.parse_args()
    index_path = PRIVATE / (args.stage + '_INDEX.json')
    index_bytes = index_path.read_bytes()
    index_hash = hashlib.sha256(index_bytes).hexdigest()
    receipts = json.loads(index_bytes)
    if not isinstance(receipts, list) or not receipts:
        raise ValueError('A stage index must contain completed receipt records')
    grouped = defaultdict(list)
    for receipt in receipts:
        grouped[context_key(receipt['meta'])].append(receipt)
    precheck = args.stage == 'PRECHECK'
    required_contexts = expected_contexts(args.stage)
    if required_contexts is not None:
        if not set(grouped).issubset(required_contexts):
            raise ValueError('Stage index contains a context outside its frozen stage design')
        if not args.allow_partial and set(grouped) != required_contexts:
            raise ValueError('Stage index is missing frozen contexts')
    groups = []
    identities = set()
    for number, (key, context_receipts) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0])), 1):
        item = summarize_context(context_receipts, require_complete=not precheck and not args.allow_partial)
        groups.append(item)
        identities.add(item['execution_identity'])
        print(json.dumps({'stage': args.stage, 'context': item['meta'],
                          'completed_contexts': number, 'total_contexts': len(grouped),
                          'loaded_paths_in_this_context': len(context_receipts)}, ensure_ascii=False), flush=True)
    if len(identities) != 1:
        raise ValueError('Stage index mixes execution identities')
    complete = all(item['complete_path_grid'] for item in groups)
    summary = {
        'schema': 'e3_streamed_stage_aggregate_v1', 'stage': args.stage,
        'classification': 'precheck_qualification_only' if precheck else ('complete_development_replay' if complete else 'partial_development_replay'),
        'execution_identity': identities.pop(), 'stage_index_sha256': index_hash,
        'summary_code_sha256': sha256(Path(__file__)),
        'statistics_code_sha256': sha256(Path(__file__).with_name('e3_statistics.py')),
        'completed_contexts': len(groups), 'indexed_paths': len(receipts),
        'groups': groups, 'budget_curves': budget_curves(groups),
        'input_access': 'completed private train600 replay receipts only; no source matrices or fresh/sealed payloads opened',
        'selection': 'P2/history240/gold20 and primary contrasts remain frozen; no predictor or hyperparameter selection from this summary',
        'new_llm_calls': 0,
    }
    output = PRIVATE / ('summary_' + args.stage + '.json')
    put(output, summary)
    report = render_feedback_report(summary)
    report_path = None
    if report is not None:
        name = args.report_name or ('PRECHECK_01_FEEDBACK_RESPONSE.md' if precheck else '01_FEEDBACK_RESPONSE.md')
        if Path(name).name != name or not name.endswith('.md'):
            raise ValueError('The public report name must be one Markdown filename')
        report_path = PUBLIC / name
        pending = report_path.with_suffix('.md.pending')
        pending.write_text(report, encoding='utf-8')
        pending.replace(report_path)
    print(json.dumps({'status': 'STAGE_AGGREGATION_COMPLETE', 'stage': args.stage,
                      'classification': summary['classification'], 'contexts': len(groups),
                      'paths': len(receipts), 'private_summary': str(output),
                      'public_report': str(report_path) if report_path else None}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

