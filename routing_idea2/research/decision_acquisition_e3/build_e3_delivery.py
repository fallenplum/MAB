"""Build complete E3 reports from closed, hash-bound replay summaries."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from run_e3 import ROOT, PUBLIC, PRIVATE, CONFIG, signature, put
from assets import load_assets, priced_units
from mechanism_diagnostics import no_reuse_accounting

STAGES = {'A': 945, 'B1': 4320, 'B2': 3240, 'J': 4320, 'SENSITIVITY': 1620, 'D': 1800}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def write(name, lines):
    (PUBLIC / name).write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def pc(value):
    return f'{100*value:.4f}'


def pp(value):
    return f'{100*value:+.4f}'


def main_metric(group):
    return group['metrics']['balanced_subtask']


def compact_group(group):
    keep = ('meta', 'original_questions', 'path_count', 'minimum_repeats_per_question',
            'maximum_repeats_per_question', 'metrics', 'resources', 'expected_corrections_on_original_questions',
            'expected_harms_on_original_questions', 'expected_net_corrections_on_original_questions',
            'negative_transfer_task_fraction', 'future_feedback_influence')
    result = {key: group[key] for key in keep if key in group}
    result['per_task'] = []
    metric_names = ('accuracy', 'gain_vs_no_feedback', 'gain_vs_fixed', 'top1_regret',
                    'correction_vs_no_feedback', 'harm_vs_no_feedback')
    for task in group['per_task']:
        result['per_task'].append({'task': task['task'], 'questions': task['questions'],
                                   **{name: task['metrics'][name] for name in metric_names}})
    for direction_name, key in (('per_seed', 'gen'), ('per_fold', 'fold'), ('per_order', 'order')):
        result[direction_name] = []
        for direction in group[direction_name]:
            result[direction_name].append({key: direction[key], 'original_questions': direction['original_questions'],
                                           'metrics': {name: direction['metrics']['balanced_subtask'][name] for name in metric_names},
                                           'gsm_accuracy': direction['metrics']['gsm']['accuracy'],
                                           'bbh_macro_accuracy': direction['metrics']['bbh_macro']['accuracy']})
    return result


def group_table(groups, include_resources=False):
    lines = ['|P|阶段|历史题数|gold上限|方法|GSM%|BBH27宏%|50/50宏%|相对不反馈pp|相对固定pp|',
             '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for group in sorted(groups, key=lambda g: tuple(str(g['meta'][k]) for k in ('predictor', 'stage', 'history_size', 'B', 'method'))):
        meta, metric = group['meta'], main_metric(group)
        lines.append(f"|{meta['predictor']}|{meta['stage']}|{meta['history_size']}|{meta['B']}|{meta['method']}|{pc(group['metrics']['gsm']['accuracy'])}|{pc(group['metrics']['bbh_macro']['accuracy'])}|{pc(metric['accuracy'])}|{pp(metric['gain_vs_no_feedback'])}|{pp(metric['gain_vs_fixed'])}|")
    return lines


def reuse_report():
    train, _ = load_assets(ROOT)
    rows = []
    for stage in ('B2', 'J'):
        for receipt in read(PRIVATE / (stage + '_INDEX.json')):
            meta = receipt['meta']
            if meta['predictor'] not in ('P1', 'P2') or meta['method'] != 'decision_cost':
                continue
            run = read(receipt['path'])
            rows.append(no_reuse_accounting(run, train))
    groups = []
    keys = sorted({(row['meta']['stage'], row['meta']['predictor'], row['meta']['B']) for row in rows})
    for stage, predictor, budget in keys:
        subset = [r for r in rows if (r['meta']['stage'], r['meta']['predictor'], r['meta']['B']) == (stage, predictor, budget)]
        assert len(subset) == 45
        groups.append({'stage': stage, 'predictor': predictor, 'B': budget, 'paths': 45,
                       'mean_comparison_events': float(np.mean([r['neighbor_events'] for r in subset])),
                       'mean_saved_members': float(np.mean([r['shared_member_savings']['member_calls'] for r in subset])),
                       'mean_saved_tokens': float(np.mean([r['shared_member_savings']['tokens'] for r in subset])),
                       'mean_saved_source_wall_s': float(np.mean([r['shared_member_savings']['source_model_wall_s'] for r in subset])),
                       'mean_actual_price': float(np.mean([r['baseline_priced_units'] for r in subset])),
                       'mean_counterfactual_price': float(np.mean([r['no_reuse_priced_units'] for r in subset]))})
    put(PRIVATE / 'NO_REUSE_ACCOUNTING.json', {'scope': 'ledger counterfactual, identical existing output and policy history', 'rows': rows})
    put(PUBLIC / 'REUSE_ACCOUNTING.json', {'groups': groups, 'new_model_calls': 0,
                                       'scope': 'cost accounting counterfactual only; no regenerated outputs or alternate budget-feasible policy',
                                       'gold_reuse': 'one canonical question reference permits both executed aggregate scores; all primary policies share this rule'})
    return groups


def plot_curves(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = Path(r'C:\Windows\Fonts\msyh.ttc')
    if font.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    colors = {'random': '#777777', 'random_neighbor': '#777777', 'boundary_cost': '#30936c',
              'pair_uncertainty': '#30936c', 'ig_cost': '#3872be', 'ig': '#3872be', 'decision_cost': '#cf6c29'}
    labels = {'random': 'Random', 'random_neighbor': 'Random Neighbor', 'boundary_cost': 'Boundary/Cost',
              'pair_uncertainty': 'Pairwise Uncertainty', 'ig_cost': 'Parameter IG/Cost', 'ig': 'Parameter IG',
              'decision_cost': 'Decision Aware/Cost'}
    for by_cost, filename in ((False, 'feedback_quality_curves.png'), (True, 'quality_complete_token_curves.png')):
        fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2), constrained_layout=True)
        for row, stage in enumerate(('B1', 'B2')):
            methods = ('random', 'boundary_cost', 'ig_cost', 'decision_cost') if stage == 'B1' else ('random_neighbor', 'pair_uncertainty', 'ig', 'decision_cost')
            for col, predictor in enumerate(('P1', 'P2', 'P3')):
                ax = axes[row, col]
                for method in methods:
                    subset = sorted([g for g in groups if g['meta']['stage'] == stage and g['meta']['predictor'] == predictor
                                     and g['meta']['method'] == method and g['meta']['history_size'] == 240], key=lambda g: g['meta']['B'])
                    assert len(subset) == 4
                    x = [g['resources']['mean_per_complete_path']['tokens']/1000 if by_cost else g['meta']['B'] for g in subset]
                    y = [100*main_metric(g)['gain_vs_no_feedback'] for g in subset]
                    ax.plot(x, y, marker='o', markersize=3.5, linewidth=1.4, color=colors[method], label=labels[method])
                ax.axhline(0, color='black', linewidth=.7)
                ax.set_title(stage + ' · ' + predictor)
                ax.set_xlabel('每120题完整服务与反馈Token（千）' if by_cost else '每120题gold揭示上限')
                ax.set_ylabel('相对无反馈准确率差（百分点）')
                ax.grid(alpha=.18)
                if not by_cost:
                    ax.set_xticks(CONFIG['gold_budgets'])
                if col == 0:
                    ax.legend(fontsize=7, frameon=False)
        fig.suptitle('固定120题服务时域 · 原题与重复生成按任务汇总', fontsize=12)
        fig.savefig(PUBLIC / filename, dpi=180, facecolor='white')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-reuse', action='store_true')
    args = parser.parse_args()
    summaries = {stage: read(PRIVATE / ('summary_' + stage + '.json')) for stage in STAGES}
    for stage, expected in STAGES.items():
        assert summaries[stage]['indexed_paths'] == expected and summaries[stage]['classification'] == 'complete_development_replay'
    identity, sources = signature()
    assert all(s['execution_identity'] == identity for s in summaries.values())
    primary = read(PUBLIC / 'PRIMARY_STATISTICS.json')
    assert primary['status'] == 'END_TO_END_DEVELOPMENT_COMPLETE'
    precision = read(PUBLIC / 'KG_PRECISION_AUDIT.json')['overall']
    groups = [group for stage in STAGES for group in summaries[stage]['groups']]
    assert len(groups) == 361
    reuse = read(PUBLIC / 'REUSE_ACCOUNTING.json')['groups'] if args.skip_reuse else reuse_report()
    uncertainty = primary['end_to_end']['balanced_subtask_five_contrast_jackknife']
    holm = primary['end_to_end']['holm_five_preregistered_contrasts']
    family = primary['family_order']
    comparisons = []
    for i, name in enumerate(family):
        comparisons.append({'name': name, 'difference': uncertainty['estimate'][i], 'ci95': uncertainty['ci95'][i],
                            'simultaneous_ci95': uncertainty['simultaneous_ci95'][i],
                            **holm['comparisons'][name]})
    passed = [item['name'] for item in comparisons if item['difference'] > 0 and item['passes_holm_threshold']]
    b1_supported = all(comparisons[i]['difference'] > 0 and comparisons[i]['passes_holm_threshold'] for i in (0, 1, 2))
    b2_supported = all(comparisons[i]['difference'] > 0 and comparisons[i]['passes_holm_threshold'] for i in (3, 4))
    overview = [f"本轮完成{sum(STAGES.values())}条主网格路径，以及6300条删组回放。每条主路径服务120道问题，600个原题分别作为一次折外题，生成种子和问题顺序各重复3次。",
                '主指标为50% GSM8K准确率加50% BBH27子任务宏平均。原V2按BBH题数加权的口径另行保留。']

    report = ['# E3-B 反馈采集机制对照', '', '## 研究问题', '', '固定同一预测器、历史、更新器和执行权限，仅改变反馈购买规则。', '',
              '## 实验设置', '', *overview, '',
              'B1只购买当前已执行主组合评分。B2固定全部方法共同的gold时点，再选择一个单成员替换邻居。初始两道gold用于共同主评分预热。联合策略允许skip、main和main加邻居。', '',
              '## 对照', '', '|方法|购买依据|', '|---|---|', '|Random|分段内随机购买时点或随机邻居|',
              '|Max-Uncertainty|反馈包内潜在价值方差之和|', '|Boundary|当前竞争边界概率，联合包使用预计边界方差缩减|',
              '|Parameter-IG|Gaussian参数信息增益|', '|KG|历史参考输入上的更新后最大预测价值增量|',
              '|Cost归一化|固定gold、成员、聚合、Token与评分次数工程价格|', '',
              '## 数值结果', '', *group_table([g for g in groups if g['meta']['stage'] in ('B1', 'B2', 'J')]), '',
              '## 统计结论', '', '主比较采用预冻结P2、历史240题、gold20。完整端到端区间与多重比较结果见统计报告。所有非主预算与跨预测器结果完整保留。', '',
              'B1的主评分增量推理成本为0，IG与IG/Cost、Boundary与Boundary/Cost具有相同的购买排序。该恒等性已逐轨迹核验。', '',
              '## 失败原因与局限', '', 'KG是线性Gaussian工作模型中的决策价值代理。实际二值聚合质量、模型表示误差和有限生成重复会影响其排序。', '',
              '联合matched_random沿用源KG已实现的购买时点和包类型，只用于条件化随机邻居诊断。它不进入独立同权限策略的主要显著性比较。', '',
              '## 下一步', '', '依据预冻结统计家族判断机制证据，保留所有正负预算档位。']
    write('02_ACQUISITION_BASELINES.md', report)

    report = ['# E3-C 跨预测器与历史规模结果', '', '## 研究问题', '', '决策定向反馈的收益能否跨原二阶、直接20组合及任务先验Ridge复现。', '',
              '## 实验设置', '', 'P1、P2历史预测逐值复现原函数，P3使用固定历史任务及子任务标识。alpha100与PCA16预先固定。历史120、240、480为嵌套样本，参考54题始终一致。', '',
              '## 对照与数值结果', '', *group_table([g for g in groups if g['meta']['B'] == 20 and g['meta']['stage'] in ('B1', 'B2')
                                                     and g['meta']['method'] in ('random', 'random_neighbor', 'ig', 'ig_cost', 'decision_cost')]), '',
              '## 统计结论', '', '不同预测器及历史规模的正负方向分别呈现，保持预先指定的P2主模型。不能从这里挑选最优配置后重复宣称同一开发集上的独立收益。', '',
              '## 失败原因与局限', '', '初始化已经含每历史题三种子、20组合的充分反馈。少量在线评分对已学后验的增量有限，A阶段的实际未来纠错与破坏支持反馈响应偏弱。初始化强弱的关系以敏感性实测为准。', '',
              '## 下一步', '', '先区分反馈响应、KG代理排序及成本归一化各自的贡献，再判断独立评价条件。']
    write('03_CROSS_PREDICTOR_RESULTS.md', report)

    report = ['# E3-D 核心机制消融', '', '## 研究问题', '', '决策相关性、成本归一化、比较反馈和同题相关建模分别影响什么。', '',
              '## 实验设置', '', 'P1和P2使用同一历史初始后验。独立噪声消融仅改变在线规划及更新的噪声矩阵。无成本、只买main及随机替换成员分别为独立行为干预。', '',
              '## 对照与数值结果', '', *group_table([g for g in groups if g['meta']['stage'] == 'D' or
                                                    (g['meta']['stage'] == 'J' and g['meta']['method'] == 'decision_cost' and g['meta']['predictor'] != 'P3')]), '',
              '## 统计结论', '', '参数IG保留同一更新器，移除KG目标；no_cost移除价格归一化；main_only移除比较包；independent_noise保留共同初始状态；random_replacement在选中比较包后随机邻居。', '',
              '成员执行复用另作冻结行为的账本反事实，详见成本报告。未执行的候选始终不向学习器提供真实评分。', '',
              '## 失败原因与局限', '', '动作类型和后续服务相互影响，单个消融总差值无法唯一归因于某一道Support。机制审计补充了同一后验下的未来冻结快照比较。', '',
              '## 下一步', '', '将实际未来净纠错与KG、IG和方差对照，保留代理预测失效的具体任务。']
    write('04_MECHANISM_ABLATIONS.md', report)

    report = ['# 完整成本与执行复用', '', '## 研究问题', '', '比较反馈能否通过合法gold及成员复用改善质量与完整资源的关系。', '',
              '## 实验设置', '', '真实模型输出与原始评分已在既有缓存生成。本轮新增语言模型、judge、gold采购及确定性评分器调用均为0，实际新增工作为CPU回放和统计。', '',
              '每个正常服务执行3成员加1聚合。购买main评分消耗1个gold和1次合法评分揭示，推理增量0。邻居比较额外1成员、1聚合和1评分，同题gold共用。', '',
              '## 对照与数值结果', '', '|P|阶段|gold上限|方法|实际gold|成员|聚合|Token|规划CPU秒|更新CPU秒|', '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        meta = group['meta']
        if meta['B'] != 20 or meta['history_size'] != 240 or meta['stage'] not in ('B1', 'B2', 'J'):
            continue
        if meta['method'] not in ('random', 'random_neighbor', 'ig', 'ig_cost', 'decision_cost'):
            continue
        cost = group['resources']['mean_per_complete_path']
        report.append(f"|{meta['predictor']}|{meta['stage']}|20|{meta['method']}|{cost['gold_questions']:.2f}|{cost['member_calls']:.2f}|{cost['aggregate_calls']:.2f}|{cost['tokens']:.2f}|{cost['planning_cpu_seconds']:.5f}|{cost['update_cpu_seconds']:.5f}|")
    report += ['', '### 共享成员账本反事实', '', '|P|阶段|gold上限|平均比较次数|少计的重复成员调用|节约Token|', '|---|---|---:|---:|---:|---:|']
    for row in reuse:
        report.append(f"|{row['predictor']}|{row['stage']}|{row['B']}|{row['mean_comparison_events']:.2f}|{row['mean_saved_members']:.2f}|{row['mean_saved_tokens']:.2f}|")
    report += ['', '## 统计结论', '', '表中为完整120题服务时域每条路径的实际执行等价花费。相同gold上限同时列出实际调用和Token，不能由上限相同推定花费相同。', '',
               '相同实际Token资源下只能比较共同服务时域的已测试预算点。曲线连接属于描述性展示，没有通过截取各方法不同结束时点的服务后缀制造配对。', '',
               '历史初始化资源另存在每条原始回放中。规模h使用18h次成员、60h次聚合与h个共享gold，表示三个生成种子的20组合历史。政策内新增成本与共同历史成本分列。', '',
               '## 失败原因与局限', '', '无复用表仅把两个已共享成员按原缓存费用重复计价，行为与质量使用原轨迹。它不模拟重新解码，也不保证该反事实仍满足原预算。', '',
               '标量价格未给CPU定价，CPU实测另列。来源模型调用wall-time不是GPU kernel时间，后者在原缓存中缺失。合法评分揭示次数与实际新增评分器调用分别报告。', '',
               '## 下一步', '', '把执行复用的费用优势与反馈选择的质量优势分别归因，V2的同题gold规则由全部主方法共享。']
    write('05_COST_AND_REUSE.md', report)

    report = ['# 配对统计与预冻结比较', '', '## 研究问题', '', '机制收益能否在保持原题依赖和完整学习路径变化的统计下得到支持。', '',
              '## 实验设置', '', '主预算20、初始化240、预测器P2预先固定。五折各120题，三个生成种子和三个问题顺序属于同600题重复测量。', '',
              '主要五维差值同时使用20个删组回放。每组删除15道GSM与15道BBH原题，同时移除历史、参考输入和在线角色，剩余流相对顺序保持，重新拟合并执行全部合法反馈。', '',
              '## 对照与数值结果', '', '|预冻结比较|准确率差pp|配对95%CI pp|Holm p|伪值偏差诊断pp|', '|---|---:|---|---:|---:|']
    for i, item in enumerate(comparisons):
        report.append(f"|{item['name']}|{pp(item['difference'])}|[{pp(item['ci95'][0])}, {pp(item['ci95'][1])}]|{item['holm_adjusted_p']:.6f}|{pp(uncertainty['pseudo_value_bias_estimate'][i])}|")
    no_feedback = primary['secondary_no_feedback']['end_to_end']
    report += ['', '### 相对No Feedback的诊断', '', '|场景|准确率差pp|近似95%CI pp|', '|---|---:|---|']
    for i, stage in enumerate(('B1', 'B2')):
        report.append(f"|{stage} Decision-Aware/Cost|{pp(no_feedback['estimate'][i])}|[{pp(no_feedback['ci95'][i][0])}, {pp(no_feedback['ci95'][i][1])}]|")
    report += ['', '这两项为单列的未校正诊断，主要Holm家族仍为预先冻结的五项。']
    report += ['', '## 统计结论', '', f"正差且通过五比较Holm阈值的条目数为{len(passed)}。具体比较名称和同时置信区间完整保存在PRIMARY_STATISTICS.json。", '',
               '各预算和各预测器的条件trace区间保留为描述性诊断。它们固定已实现学习历史，不能替代端到端删组回放。', '',
               '## 失败原因与局限', '', '20组jackknife为近似推断，离散动作和预算阈值并不光滑，有限样本覆盖没有严格保证。五折历史重叠通过删去原题所有角色纳入，生成与顺序重复不能增加独立题数。', '',
               f"主比较的伪值偏差诊断为{pp(uncertainty['pseudo_value_bias_estimate'][0])}个百分点，远大于原点差{pp(comparisons[0]['difference'])}个百分点。该诊断表明平滑删组近似对当前离散选择较敏感，区间不能包装成严格覆盖保证或总体零收益证明。", '',
               '参考点的删除后计数根据冻结规则重建，实际执行绑定template_for源码。原路径回执没有逐条reference ID列表，因此参考角色核验的范围为源码及设计重建。', '',
               f"主实验使用8对Gaussian假想值。事后{precision['snapshots']}个固定快照的8对与原默认32对数值审计中，B2/J候选选择一致{precision['cost_argmax_agreement_count']}/{precision['cost_argmax_comparison_snapshots']}，另有{precision['tolerance_g8_zero_g32_positive_candidates']}/{precision['candidate_values']}候选从近零变为正值。因此8对存在可见积分敏感性，结果不能推广为所有KG数值实现的结论。该检查没有重选主参数或改写主实验。", '',
               '全部题属于已使用的开发谱系，外部新题泛化仍需单独验证。', '',
               '## 下一步', '', '将主要比较、强通用对照、实际费用以及跨预测器方向合并判断，保留未经支持的对比。']
    write('06_STATISTICAL_ANALYSIS.md', report)

    def find(stage, predictor, method, budget=20, history=240):
        return next(g for g in groups if all(g['meta'][key] == value for key, value in
                                           (('stage', stage), ('predictor', predictor), ('method', method), ('B', budget), ('history_size', history))))
    b1 = {name: main_metric(find('B1', 'P2', name)) for name in ('decision_cost', 'random', 'boundary_cost', 'ig_cost')}
    b2 = {name: main_metric(find('B2', 'P2', name)) for name in ('decision_cost', 'random_neighbor', 'ig')}
    report = ['# E3研究结论', '', '## 研究问题', '', '相同学习器下决策定向采集是否更快改善未来问题，质量与成本收益是否独立于预测器。', '',
              '## 实验设置', '', *overview, '', '## 对照与数值结果', '', '|P2，gold20|准确率%|相对无反馈pp|', '|---|---:|---:|']
    for prefix, mapping in (('B1', b1), ('B2', b2)):
        for name, metric in mapping.items():
            report.append(f"|{prefix} {name}|{pc(metric['accuracy'])}|{pp(metric['gain_vs_no_feedback'])}|")
    report += ['', '## 统计结论', '',
               f"B1的Decision-Aware/Cost相对Random为{pp(b1['decision_cost']['accuracy']-b1['random']['accuracy'])}个百分点，相对IG/Cost为{pp(b1['decision_cost']['accuracy']-b1['ig_cost']['accuracy'])}个百分点。",
               f"B2相对Random Neighbor为{pp(b2['decision_cost']['accuracy']-b2['random_neighbor']['accuracy'])}个百分点，相对IG为{pp(b2['decision_cost']['accuracy']-b2['ig']['accuracy'])}个百分点。", '',
               '以上差值的端到端区间及Holm结果见第六份报告。不能将对一个较弱对照的正差替代同时优于随机和最强通用采集的证据。', '',
               ('预冻结主要机制比较出现正向统计支持，跨预测器与独立新题证据继续单列。' if b1_supported or b2_supported else '**本轮未建立决策定向采集同时优于随机与通用信息增益对照的主要证据。**'), '',
               '### 四个问题的回答', '',
               '1. 是否比随机更好，需同时看B1与B2的配对差和区间，主P2/B20点值未显示优于随机。',
               '2. 决策相关性是否具有独立价值，B1对IG的小正差与B2近零差分开保留，跨预算及预测器结果用于检查稳定性。',
               '3. 执行复用能降低重复成员费用，同题gold能支持多个合法已执行评分。这些规则由所有主方法共享，费用节约与DA独有的质量收益分开判断。',
               '4. 跨预测器的一致性以完整矩阵为准，主模型和主预算保持预冻结身份。', '',
               '## 失败原因与局限', '', 'A阶段显示当前学习器对少量反馈的响应偏弱，更多反馈也会增加破坏正确决策。KG工作模型的期望最大预测值不能保证提升真实二值服务质量。', '',
               '历史参考覆盖所有27个BBH子任务，但每子任务仅一个参考输入，输入覆盖和奖励生成随机性仍有限。分段预算pacing限制跨时段分配，结论条件于该规则。', '',
               'B1排序策略先观察时段前三分之一，Random在整段均匀选时点，因此策略效果也包含等待代价。B2共同gold安排更直接隔离邻居选择。这一工程选择与纯采集评分函数的作用需要区分。', '',
               '优势是复用了真实生成矩阵并核验权限、更新与费用；风险是所有结果属于600题开发回放，统计近似及工程价格会影响结论范围。', '',
               '## 下一步', '', '保留固定预测器的全部正负结果，优先分析反馈响应、奖励随机性和KG代理与实际未来纠错的偏差。独立fresh400及sealed200的评分本轮未读取。']
    write('07_RESEARCH_CONCLUSION.md', report)

    plot_curves(groups)
    archives = []
    for stage in STAGES:
        path = PRIVATE / ('summary_' + stage + '.json')
        target = PUBLIC / ('FULL_' + stage + '_STATISTICS.json.gz')
        target.write_bytes(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))
        archives.append({'name': target.name, 'sha256': sha(target), 'decompressed_sha256': sha(path),
                         'stage': stage, 'contexts': len(summaries[stage]['groups'])})
    results = {'schema': 'decision_acquisition_e3_complete_v1', 'status': 'COMPLETE_DEVELOPMENT_REPLAY',
               'execution_identity': identity, 'source_hashes': sources, 'main_paths': sum(STAGES.values()),
               'delete_replay_paths': 6300, 'stage_paths': STAGES, 'original_questions': 600,
               'generation_seeds': CONFIG['generation_seeds'], 'order_seeds': CONFIG['order_seeds'],
               'independent_sample_unit': 'original question; repeated generations/orders and overlapping CV histories are retained',
               'groups': [compact_group(g) for g in groups], 'primary_comparisons': comparisons,
               'secondary_no_feedback': primary.get('secondary_no_feedback'),
               'budget_curves': {stage: [{key: value for key, value in curve.items() if key != 'points'}
                                         for curve in summaries[stage]['budget_curves']] for stage in STAGES},
               'science_gate': {'B1_primary_supported': b1_supported, 'B2_primary_supported': b2_supported,
                                'cross_predictor_independent_support': 'not established by direction alone',
                                'fresh_evaluation_started': False},
               'reuse_accounting': reuse, 'full_stage_archives': archives,
               'data_access': {'decoded_source_splits': ['selector_train'], 'validation_used': False,
                               'fresh400_used': False, 'sealed200_used': False},
               'new_llm_calls': 0, 'new_judge_calls': 0, 'new_gold_purchase_calls': 0,
               'scope': 'development mechanism evidence, with full negative and positive cells retained',
               'detail_note': 'All per-task metrics, repeated-measure directions, prefix curves, costs and accounts are retained in the hashed full-stage archives. This JSON is the complete main-results index.'}
    put(PUBLIC / 'E3_COMPLETE_RESULTS.json', results)
    (PUBLIC / 'E3_COMPLETE_RESULTS.json.gz').write_bytes(gzip.compress((PUBLIC / 'E3_COMPLETE_RESULTS.json').read_bytes(), compresslevel=9, mtime=0))
    put(PUBLIC / 'E3_COMPLETION.json', {'status': 'COMPLETE_DEVELOPMENT', 'main_paths': sum(STAGES.values()),
                                      'delete_paths': 6300, 'contexts': len(groups), 'new_llm_calls': 0,
                                      'execution_identity': identity, 'primary_statistics': primary['status'],
                                      'fresh_and_sealed_payloads_opened': False})
    print('E3_REPORTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
