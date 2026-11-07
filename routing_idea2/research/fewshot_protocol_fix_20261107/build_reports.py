"""Publish aggregate diagnostics and prepare an outcome-blind fresh input plan."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from models import METHODS, initialize, semantic_labels
from run_corrected import PUBLIC, PRIVATE, ROOT, load, sha, put, inputs


def pct(value):
    return f'{100 * value:.3f}'


def interval(item):
    return f"{100 * item['difference']:+.3f} [{100 * item['ci95'][0]:+.3f}, {100 * item['ci95'][1]:+.3f}]"


def write(name, content):
    (PUBLIC / name).write_text(content.rstrip() + '\n', encoding='utf-8')


def main():
    results = json.loads((PUBLIC / 'EXPERIMENT_RESULTS.json').read_text(encoding='utf-8'))
    assert len(results) == 12
    trainval, refs = load()
    previous = json.loads((PRIVATE / 'preliminary_engine_results.json').read_text(encoding='utf-8'))
    comparison = []
    for cfg, block in results.items():
        for scope in ('gsm', 'bbh_macro', 'bbh_micro', 'balanced_subtask_macro', 'balanced_v2_macro'):
            for method in METHODS:
                for field in ('accuracy', 'top1_regret', 'net_correction_vs_base'):
                    comparison.append(abs(block[scope][method][field] - previous[cfg][scope][method][field]))
        for baseline in ('no_update', 'fixed'):
            zero = results[cfg[:-1] + '0']
            assert abs(block['balanced_subtask_macro'][baseline]['accuracy'] - zero['balanced_subtask_macro'][baseline]['accuracy']) < 1e-12
            source_target = block['gsm'][baseline]['source_target_matrix']
            for target in range(3):
                assert np.ptp([cell['accuracy'] for cell in source_target if cell['target_seed'] == (42, 137, 2024)[target]]) < 1e-12
    assert max(comparison) < 1e-12
    put(PUBLIC / 'REPLAY_QUALIFICATION.json', {'status': 'PASS', 'compared_configurations': 12,
                                            'maximum_numerical_metric_difference': max(comparison),
                                            'checks': ['k0_k1_k2_frozen_baselines_equal', 'frozen_choices_source_seed_invariant',
                                                       'initial_and_hardened_cache_engine_metrics_equal'],
                                            'new_llm_calls': 0})
    tables = ['# 修正协议完整对照表', '', '## 研究问题', '',
              '历史600题训练后，在新200题每个语义环境收到0、1、2道Support反馈，剩余Query能否受益。所有划分穷举积分。', '',
              '## 实验设置', '',
              '完整表包括12个配置、5个统计范围、12个方法。数值单位为百分比或百分点。括号区间为配对95%开发性近似区间。', '',
              'full20观察同一Support题20个组合，single仅观察冻结基础Ridge事先选择的组合。带shuffle后缀的方法使用匹配错误任务历史反馈。', '',
              '## 对照与数值结果', '']
    scopes = {'gsm': 'GSM8K', 'bbh_macro': 'BBH 27子任务宏平均', 'bbh_micro': 'BBH按原题加权',
              'balanced_subtask_macro': '主口径 50% GSM与50% BBH子任务宏平均',
              'balanced_v2_macro': 'V2口径 50% GSM与50% BBH按题平均'}
    for cfg, block in results.items():
        for scope, title in scopes.items():
            tables += [f'### {cfg} · {title}', '', '|方法|准确率|相对不更新及95%CI|相对固定及95%CI|相对匹配打乱及95%CI|事后Regret|纠错率/破坏率|',
                       '|---|---:|---|---|---|---:|---:|']
            for method in METHODS:
                r = block[scope][method]
                control = interval(r['vs_matched_shuffle']) if r['vs_matched_shuffle'] else '—'
                tables.append(f"|{method}|{pct(r['accuracy'])}|{interval(r['vs_base'])}|{interval(r['vs_fixed'])}|{control}|{pct(r['top1_regret'])}|{pct(r['correction_vs_base'])}/{pct(r['harm_vs_base'])}|")
            tables.append('')
    tables += ['## 统计结论', '', '区间条件于冻结的历史训练。GSM删除原题的全部Support和Query角色构造jackknife伪值，BBH按27个完整子任务抽样。生成种子共用重采样权重。', '',
               '多方法多配置区间未作多重比较校正。单一正区间只能用于开发诊断。Regret使用每个生成种子内观测20组合的事后最大奖励。', '',
               '## 失败原因', '', '按任务逐项的诊断与跨种子结果见配套报告。固定策略在所有k下保持相同边际，避免将Query构成改变误认为适应。', '',
               '## 下一步', '', '冻结主要诊断候选与输入计划，依据科学关口决定后续评价。']
    write('COMPLETE_COMPARISON.md', '\n'.join(tables))

    task_rows = []
    costs = json.loads((PUBLIC / 'COSTS.json').read_text(encoding='utf-8'))
    for cfg, blocks in costs['cached_counterfactual_costs'].items():
        for info in blocks:
            path = PRIVATE / info['private_file']
            with np.load(path, allow_pickle=False) as source:
                count = math.comb(info['n'], info['k']) * (info['n'] - info['k'])
                metric = source['total'] / count
                for method_index, method in enumerate(METHODS):
                    diag = metric[[0, 4, 8], method_index].mean(0)
                    task_rows.append({'configuration': cfg, 'task': info['task'], 'questions': info['n'],
                                      'support_per_episode': info['k'], 'query_per_episode': info['n'] - info['k'],
                                      'support_sets': info['support_sets'], 'method': method,
                                      'accuracy': float(diag[0]), 'correction_rate': float(diag[1]), 'harm_rate': float(diag[2]),
                                      'net_corrections_per_episode': float((diag[1] - diag[2]) * (info['n'] - info['k'])),
                                      'corrections_per_episode': float(diag[1] * (info['n'] - info['k'])),
                                      'harms_per_episode': float(diag[2] * (info['n'] - info['k'])),
                                      'cpu_seconds_per_query': float(diag[5]), 'top1_regret': float(diag[6]),
                                      'source_target_metrics': metric[:, method_index].tolist()})
    put(PUBLIC / 'TASK_RESULTS.json', {'rows': task_rows, 'metric_order': ['reward', 'correction', 'harm', 'correction_vs_fixed', 'harm_vs_fixed', 'cpu_s', 'regret', 'switch'],
                                     'counts_note': 'Counts are exact expectations over Support choices, so they need not be integers. Repeated episodes are not added into one independent sample size.'})
    detail = ['# 分任务、负迁移与生成随机性诊断', '', '## 研究问题', '', '少量反馈是否携带能迁移到同任务其他问题和其他生成种子的能力信息。', '',
              '## 实验设置', '', '重点为预先指定的single、k2层级更新。各BBH子任务只有3或4题，每episode只剩1或2道Query。', '', '## 对照与数值结果', '']
    for mode in ('seen', 'unseen'):
        cfg = mode + '/single/k2'
        r = results[cfg]['balanced_subtask_macro']['hierarchical']
        detail += [f'### {mode} 层级收缩', '', f"相对不更新为{interval(r['vs_base'])}pp，相对固定为{interval(r['vs_fixed'])}pp。", '',
                   '|任务|准确率|纠错期望题数|破坏期望题数|净纠错期望题数|原题数/Query数|', '|---|---:|---:|---:|---:|---:|']
        for row in task_rows:
            if row['configuration'] == cfg and row['method'] == 'hierarchical':
                detail.append(f"|{row['task']}|{pct(row['accuracy'])}|{row['corrections_per_episode']:.4f}|{row['harms_per_episode']:.4f}|{row['net_corrections_per_episode']:+.4f}|{row['questions']}/{row['query_per_episode']}|")
        detail += ['', f"28个语义环境负迁移比例为{pct(r['negative_transfer_task_fraction'])}%。BBH-only负迁移比例为{pct(results[cfg]['bbh_macro']['hierarchical']['negative_transfer_task_fraction'])}%。", '']
    detail += ['## 跨生成种子结果', '', '|配置与方法|同种子平均增益pp|异种子平均增益pp|同种子三方向增益pp|', '|---|---:|---:|---|']
    for cfg in results:
        if cfg.endswith('k0'):
            continue
        for method in ('online_ridge', 'hierarchical', 'knn', 'adapter2', 'adapter4'):
            r = results[cfg]['balanced_subtask_macro'][method]
            diag = ', '.join(f"{100 * cell['net_correction_vs_base']:+.3f}" for cell in r['perdiagseed'])
            detail.append(f"|{cfg} {method}|{100*r['net_correction_vs_base']:+.3f}|{100*r['offdiagmean']['net_correction_vs_base']:+.3f}|{diag}|")
    detail += ['', '## 正负迁移任务的共同观测特征', '',
               '以下按seen、single k2层级更新的净收益事后分组，只用于解释已观察到的差异。分组没有用于选择候选、特征、超参数或测试任务。', '',
               '|任务组|BBH子任务数|平均历史题数|平均新题数|更新前准确率%|同题组合生成方差|', '|---|---:|---:|---:|---:|---:|']
    characteristics = {'正向': [], '负向': [], '零变化': []}
    labels = semantic_labels(trainval['selector_validation'])
    historical_labels = semantic_labels(trainval['selector_train'])
    for task in sorted(set(labels) - {'gsm8k'}):
        row = next(item for item in task_rows if item['configuration'] == 'seen/single/k2' and item['task'] == task and item['method'] == 'hierarchical')
        base = next(item for item in task_rows if item['configuration'] == 'seen/single/k2' and item['task'] == task and item['method'] == 'no_update')
        gain = row['net_corrections_per_episode']
        group = '正向' if gain > 1e-12 else ('负向' if gain < -1e-12 else '零变化')
        noise = trainval['selector_validation']['Y'][labels == task].var(axis=1, ddof=1).mean()
        characteristics[group].append([int(np.sum(historical_labels == task)), row['questions'], base['accuracy'], noise])
    for group, values in characteristics.items():
        if values:
            mean = np.asarray(values).mean(0)
            detail.append(f'|{group}|{len(values)}|{mean[0]:.3f}|{mean[1]:.3f}|{100*mean[2]:.3f}|{mean[3]:.5f}|')
    detail += ['', '组合生成方差按题、组合计算三个种子的样本方差后平均，单位为二值奖励平方。该量与任务内条件能力变化混合，当前三种子不足以确定其总体值。', '',
               '正负迁移组的更新前准确率分别约63%，均低于零变化组的95%。仅靠问题难度无法区分有益与有害更新。负迁移组的观测生成方差更高，这只是5个正向与6个负向任务的事后关联。', '',
               '## 统计结论', '', '同题三个生成种子属于重复测量。异种子六个方向共用同一批原题和Support集合，只作为随机性迁移诊断。完整九格方向及配对区间见压缩JSON。', '',
               '## 失败原因', '', '任务级常量校正无法表达同一子任务内部的问题差异。受限单组合反馈对其他组合的直接信息稀疏，低秩方法依赖历史残差方向能迁移的额外假设。', '',
               '某些方法的同种子增益在异种子减弱，提示生成随机性会影响支持标签的可迁移性。这些现象不能单独证明“任务没有可用信息”或“模型永远学不会”。', '',
               '每子任务3至4题使逐任务区间难以可靠估计，正增益任务清单属于描述性探索。当前数据不足以识别这些任务共有的语义机制。', '',
               '## 下一步', '', '先保留完整负结果及任务支持数，不从开发Query中搜索更多超参数或按正收益任务筛选评价集。']
    write('TASK_AND_SEED_DIAGNOSTICS.md', '\n'.join(detail))

    bundle = json.loads((ROOT / 'artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json').read_text(encoding='utf-8-sig'))
    reference = bundle['inputs']['fresh_dev_test']
    assert sha(reference['path']) == reference['sha256']
    with np.load(reference['path'], allow_pickle=False) as source:
        fresh = {'E': source['embeddings'].astype(float), 'ids': source['question_ids'], 'task': source['task_labels'],
                 'sub': source['subtask_labels'], 'canonical': source['canonical_hashes']}
    assert len(fresh['ids']) == 400
    for split in trainval.values():
        assert not set(fresh['ids']) & set(split['ids'])
        assert not set(fresh['canonical']) & set(split['canonical'])
    actions = np.full(400, -1, dtype=int)
    support_mask = np.zeros(400, dtype=bool)
    planning = []
    for name in sorted(set(semantic_labels(fresh))):
        rows = np.flatnonzero(semantic_labels(fresh) == name)
        ordering = sorted(rows, key=lambda row: hashlib.sha256((str(fresh['ids'][row]) + '|fresh_support_v1').encode()).hexdigest())
        support_mask[ordering[:2]] = True
        state = initialize(trainval['selector_train'], inputs(fresh, rows), name, 'seen')
        actions[rows] = np.argmax(state.query_prediction, axis=1)
        planning.append({'task': name, 'questions': len(rows), 'support': 2, 'query': len(rows)-2})
    np.savez_compressed(PRIVATE / 'FRESH400_INPUT_ONLY_PLAN.npz', question_ids=fresh['ids'], support=support_mask, frozen_support_actions=actions)
    freeze = {'candidate': 'hierarchical', 'evaluation_mode': 'seen', 'feedback': 'single', 'support_questions_per_environment': 2,
              'ridge_alpha': 100, 'historical_repeat_weight': 3, 'support_repeat_weight': 1,
              'pca_components': 16, 'hierarchical_lambda': 4, 'selection_basis': 'predeclared diagnostic candidate; no development Query tuning',
              'candidate_status': 'FROZEN_FOR_DIAGNOSTIC_CONFIRMATION_ONLY', 'fresh_execution': 'NOT_STARTED',
              'fresh_input_sha256': reference['sha256'], 'fresh_scores_read': False, 'sealed_scores_read': False,
              'planning': planning, 'support_total': int(support_mask.sum()), 'query_total': int((~support_mask).sum()),
              'support_rule': 'first two sha256(question_id|fresh_support_v1) within semantic environment',
              'query_rule': 'after legal Support feedback update, freeze Query choices and union before reading Query scores',
              'history_hashes': {k: v['sha256'] for k, v in refs.items()},
              'code_hashes': {p.name: sha(p) for p in PUBLIC.glob('*.py')}}
    put(PUBLIC / 'FROZEN_CANDIDATE.json', freeze)

    summary = ['# MAB少样本适应协议修正结果', '', '## 研究问题', '', '用历史600题训练组合预测器，检验新200题上的1至2道Support反馈能否改善剩余Query选择。', '',
               '## 实验设置', '', '六模型、K3、20组合和V2 Qwen3聚合器保持冻结。复用已有48,000次组合评分，其中训练36,000、验证12,000，无新增语言模型调用。已见任务使用600题，未见BBH子任务在历史中全部排除。', '',
               '主要指标为50% GSM准确率加50% BBH27子任务宏平均。原V2的50% GSM与50% BBH按题平均另外报告。原200题已用于开发，当前结果属于修正后的开发验证。', '',
               '## 对照与数值结果', '', '|模式/反馈|方法，k2|GSM%|BBH27宏%|50/50宏%|相对不更新及95%CI，pp|相对固定及95%CI，pp|', '|---|---|---:|---:|---:|---|---|']
    for mode in ('seen', 'unseen'):
        for feedback in ('single', 'full20'):
            cfg = f'{mode}/{feedback}/k2'
            for method in ('no_update', 'fixed', 'online_ridge', 'hierarchical', 'knn', 'adapter2', 'adapter4'):
                r = results[cfg]['balanced_subtask_macro'][method]
                summary.append(f"|{mode}/{feedback}|{method}|{pct(results[cfg]['gsm'][method]['accuracy'])}|{pct(results[cfg]['bbh_macro'][method]['accuracy'])}|{pct(r['accuracy'])}|{interval(r['vs_base'])}|{interval(r['vs_fixed'])}|")
    primary = results['seen/single/k2']['balanced_subtask_macro']['hierarchical']
    gate_checks = []
    for cfg, block in results.items():
        if cfg.endswith('k0'):
            continue
        for method in ('hierarchical', 'adapter2', 'adapter4'):
            r = block['balanced_subtask_macro'][method]
            supported = all(r[field]['ci95'][0] > 0 for field in ('vs_base', 'vs_fixed', 'vs_matched_shuffle'))
            directions = sum(item['net_correction_vs_base'] > 0 for item in r['perdiagseed'])
            gate_checks.append({'configuration': cfg, 'method': method,
                                'positive_unadjusted_ci_vs_all_three_references': supported,
                                'positive_same_seed_directions': directions,
                                'cross_seed_gain': r['offdiagmean']['net_correction_vs_base'],
                                'passes_diagnostic_screen': supported and directions >= 2 and r['offdiagmean']['net_correction_vs_base'] > 0})
    passed = any(item['passes_diagnostic_screen'] for item in gate_checks)
    put(PUBLIC / 'SCIENTIFIC_GATE.json', {'status': 'DIAGNOSTIC_SIGNAL_REQUIRES_CONFIRMATION' if passed else 'NOT_MET', 'meta_learning_started': False,
                                        'fresh_queue_started': False, 'checks': gate_checks,
                                        'note': 'Screen failure is descriptive. Passing would still require multiplicity-aware independent confirmation.'})
    summary += ['', '## 修正前后变化', '', '|记录|准确率%|相对不更新，pp|', '|---|---:|---:|',
                '|旧28环境等权层级single k2|82.4010|+0.9384|', '|旧预测仅改50/50权重|86.3300|+1.1007|',
                f"|新历史训练与更新，主口径|{pct(primary['accuracy'])}|{100*primary['net_correction_vs_base']:+.4f}|", '',
                '旧验证实现曾用其他验证题评分训练基础模型，旧低秩基底未经学习，旧Online更新未继承历史精度，旧区间未充分处理题目复用。新旧差异同时包含这些修正，无法分解为单一因素的因果效应。', '',
                '## 统计结论', '', ('**出现通过开发诊断筛查的候选，仍需独立确认。**' if passed else '**当前层级收缩与学习型低秩适应器没有达到稳定优于不更新和固定路由的关口。元学习阶段的启动条件未满足。**'), '',
                'Online Ridge的部分同种子小幅收益未形成同时超过固定策略、匹配错误任务对照及跨种子迁移的完整证据。所有区间为条件于历史的开发性近似区间，未校正多重比较。', '',
                '## 失败原因', '', '当前证据显示，1至2道反馈下的轻量任务适应缺乏稳定收益。不能由此断言不存在任务信息或组合路由不可学。任务样本少、Support标签生成方差和模型限制仍未被分离。', '',
                '优势是历史与Query边界、更新公式和相关性可逐项重放核验，所有LLM结果得到复用。风险是每个BBH环境仅3至4题，区间依赖任务抽样和jackknife近似，历史训练随机性未纳入区间。', '',
                '两项具体实现选择需要区分。KNN预测采用基础Ridge加历史记忆增量残差，使k0统一。负对照使用合法历史错任务反馈，维持相同动作掩码、更新器及账面成本，但不能保证与当前Support反馈完全相同的经验奖励边际分布。', '',
                '## 下一步', '', f"预先指定的hierarchical single k2已冻结为诊断候选，fresh400输入计划已生成，共{freeze['support_total']}道Support和{freeze['query_total']}道Query。候选选择未使用开发Query调参，fresh评分和sealed200均未读取。", '',
                'fresh正式调用队列尚未启动。后续先审查负迁移与跨种子稳定性，再依据科学关口推进独立确认。ANIL、MAML及反馈采集阶段没有启动。', '',
                '## 文件索引', '', '- COMPLETE_COMPARISON.md 保存全部720条总体方法对照。', '- TASK_AND_SEED_DIAGNOSTICS.md 保存分任务纠错、负迁移及跨种子差异。',
                '- EXPERIMENT_RESULTS.json.gz 保存完整原始聚合指标及区间。', '- TASK_RESULTS.json.gz 保存每任务每方法的九格种子指标。',
                '- COSTS.json.gz 保存合法成员与聚合缓存成本，不将反事实episodes相加计费。', '- QUALIFICATION.json和REPLAY_QUALIFICATION.json 保存软件与重放验证。',
                '- CORRECTION_AUDIT.md和PROTOCOL.md 保存修正细节与协议。', '- FROZEN_CANDIDATE.json 保存候选、输入哈希和后续评价规则。']
    write('EXPERIMENT_SUMMARY.md', '\n'.join(summary))
    accounting = ['# 完整反馈与服务成本', '', '## 研究问题', '', '反馈权限是否通过隐藏的成员或聚合调用获得额外优势。', '',
                  '## 实验设置', '', '本轮只执行CPU训练和真实缓存重放，新增LLM、成员生成、聚合、评价调用均为0。', '',
                  '下表是单个部署episode的反事实成本。每环境Support提供k道不同题的gold，full20为6k次成员与20k次聚合，single为3k次成员与k次聚合。gold均为k。历史训练矩阵视作共同已付成本。', '',
                  'CPU统计为适应更新和预测的实测合计除Query数，历史PCA与基础拟合不包含在该项。GPU时间为原缓存调用耗时之和，不能用来预测新硬件延迟或并发墙钟。', '',
                  '## 对照与数值结果', '', '|配置|方法|每Query prompt tokens|每Query completion tokens|每Query缓存GPU秒|每Query更新预测CPU毫秒|', '|---|---|---:|---:|---:|---:|']
    for cfg, blocks in costs['cached_counterfactual_costs'].items():
        if cfg.endswith('k0'):
            continue
        for method in ('no_update', 'fixed', 'online_ridge', 'hierarchical', 'knn', 'adapter2', 'adapter4'):
            index = METHODS.index(method)
            task_values = np.asarray([np.asarray(item['service_cost_mean'])[:, index].mean(0) for item in blocks])
            value = .5 * task_values[0] + .5 * task_values[1:].mean(0)
            cpu = results[cfg]['balanced_subtask_macro'][method]['mean_cpu_seconds_per_query_kernel']
            accounting.append(f'|{cfg}|{method}|{value[1]:.2f}|{value[2]:.2f}|{value[3]:.4f}|{1000*cpu:.4f}|')
    accounting += ['', '## 统计结论', '', '每条服务决策固定3成员加1聚合。Support成本与Query服务成本分列。无更新和固定策略可以不购买Support，表中的共同Support预算只是实验对齐额度。', '',
                   '当前成本文件仅报告三个同种子主对角的服务成本。跨种子六方向用于随机性诊断，其服务成本没有混入主对角账单。', '',
                   '## 失败原因与风险', '', '历史矩阵的48,000条组合评分可共享其成员生成与gold。把每个组合评分重复计为一次新gold，或者把穷举的互斥Support部署episode相加，都会夸大费用。', '',
                   '## 下一步', '', 'fresh先购买冻结Support动作，再冻结Query选择并计算合法缓存并集，之后才允许读取Query评分。']
    write('COST_ACCOUNTING.md', '\n'.join(accounting))
    for name in ('EXPERIMENT_RESULTS.json', 'TASK_RESULTS.json', 'COSTS.json'):
        content = (PUBLIC / name).read_bytes()
        (PUBLIC / (name + '.gz')).write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
    put(PUBLIC / 'PUBLICATION_MANIFEST.json', {'base_commit': '3a5b15020f06eeeec15bc5c87a439dffb20be215',
                                            'branch': 'codex/fewshot-protocol-fix-20261107',
                                            'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in PUBLIC.iterdir()
                                                      if p.suffix in ('.md', '.py', '.gz') or p.name in ('FROZEN_CANDIDATE.json', 'QUALIFICATION.json', 'REPLAY_QUALIFICATION.json', 'COMPLETION.json', 'SCIENTIFIC_GATE.json')}})
    print('REPORTS_AND_OUTCOME_BLIND_FRESH_PLAN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
