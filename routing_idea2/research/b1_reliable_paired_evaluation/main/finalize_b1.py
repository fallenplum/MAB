"""Assemble the bounded B1 study, including native-reproduction limitations."""
from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from run_b1 import BASE, put, sha
from audit_and_report import text_file, verify_main


def read(relative):
    return json.loads((BASE/'public'/relative).read_text(encoding='utf-8'))


def main():
    verify=verify_main()
    real=read('REAL_FIXED_POOL_RESULTS.json')
    synthetic=read('inference/SYNTHETIC_COMPLETE_RESULTS.json')
    supplement=read('SUPPLEMENTARY_FINITE_POOL_RESULTS.json')
    native=read('NATIVE_REFERENCE_RESULTS.json')
    assert synthetic['complete'] and len(synthetic['settings'])==36
    assert len(real['methods'])==27 and len(supplement)==2
    assert all(x['repetitions']==2000 and len(x['results'])==6 for x in synthetic['settings'])
    replays=sorted((BASE/'private/reproductions').glob('*/REPRODUCTION_AUDIT.json'))
    assert replays and json.loads(replays[-1].read_text())['status']=='PASS'
    rows=[]
    for setting in synthetic['settings']:
        if setting['n']!=5000 or setting['correlation']!='conditionally_independent':continue
        if setting['world'] not in ['positive_3pp','heterogeneous','wrong_uncertainty']:continue
        for method in ['U1','ASI-adapted','Robust-ASI-adapted','T1']:
            v=setting['results'][method]['threshold_0']
            rows.append(f"| {setting['world']} | {method} | {100*v['rmse']:.3f} | {100*v['population_bernstein']['adopt_probability']:.2f}% | {v['mean_gold']:.2f} | {v['mean_actual_tokens']:,.0f} |")
    text_file('04_POPULATION_RELIABILITY_AND_POWER.md',f"""
# B1 总体可靠性与检出能力

## 研究问题

在已知总体真值的合法二值配对环境中，评价平均收益估计、错误批准和正收益检出。固定池覆盖与未来问题总体覆盖分别计算。

## 设置与对照

六类环境为0、−2、+1、+3个百分点、费用与分歧异质的+3个百分点，以及保持同一异质环境但颠倒工作不确定性方向。每种环境取200、1000、5000道独立问题，每题三次生成，题内条件独立与题内完全相关分别执行。36个设置，每个2000次Monte Carlo，六种方法合计432000条估计记录。

历史600题独立于评价题。工作辅助、购买函数和费用模型在当前总体样本产生前固定。合成总体的类别权重与预计费用视为已知，当前输入池未用于重新归一化购买概率。完整环境定义、联合二值概率与种子保存在inference/SYNTHETIC_PROTOCOL.json。

## 代表数值

| 环境，5000题独立重复 | 方法 | RMSE百分点 | 保守检出率 | 平均Gold | 平均实际合成Token |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 统计结论

全部216个方法设置的保守95%区间覆盖最低99.9%。零或负收益环境的最大观测保守错误批准率0.05%，即2000次中1次。其Monte Carlo区间与每个单元的三态决定均保存。正收益异质环境的T1检出率95.05%，有效推断具有实际检出能力。

Wald最低覆盖93.1%，发生于200题、生成重复完全相关、工作残差方向错误的T1设置。Wald的较高检出率与较窄区间需连同该覆盖不足一起解释。

T1在同质费用+3个百分点环境中RMSE降低，但保守检出率72.70%，U1为77.80%。平方根采样带来的较小购买概率增大了保守误差界。RMSE改善尚不足以证明可靠采用判断更高效。

## 成本和范围

异质正收益环境各方法约使用1400万Token，但Gold使用从约787到1249不等。T1未超过Robust适配的最低RMSE。颠倒残差方向的T1约用879万Token，U1约1400万。这一反例属于共同预算上限下的比较，实际Token差异完整保留。

达到80%检出率的资源表使用预设题量网格上的点估计首达。部分Monte Carlo区间跨越80%，其解释限于该粗网格。生成完全相关的敏感性实验沿用原独立重复历史辅助，包含相关性失配。合成结果没有覆盖未知总体类别权重或未知预计费用的额外估计误差。

## 失败原因与下一步

主动采样的条件性收益依赖残差预测方向。当前T1没有相对强公开适配方法的普遍优势。对真实题池的外推需结合03报告，其中T1的精度与保守区间均未体现优势。

优势是总体真值、全输入支持界和独立问题簇均可核验。风险是离散合成结构、已知类别权重、保守界宽度及残差失配。数值重现的置信度高，真实任务的机制外推证据有限。
""")
    tables=[]
    for method in ['U0','U1','A1_ASI','A2_Robust_ASI','T0_RawMoment','T1']:
        v=real['methods']['primary__'+method]
        tables.append(f"| {method} | {100*v['RMSE']:.3f} | {v['resources']['gold']['mean']:.2f} | {v['resources']['tokens']['mean']:,.0f} |")
    text_file('EXPERIMENT_SUMMARY.md',f"""
# B1 冻结路由器的可靠低成本配对评估

## 研究目标与执行状态

本轮从codex/ces-traceable-baseline-20261112创建codex/b1-reliable-paired-evaluation。恢复原train600直接20组合Ridge权重，与train600选择的全局固定组合比较。六个冻结语言模型、K=3、20种组合、V2 Qwen3聚合器及评分器均按原资产复用。

已完成主真实池27格、每格2000次抽样，36个合成设置、每个2000次Monte Carlo，两项补充基线各2000次，以及四项机制消融。原始采样、合法揭示和成本账本均留在私有目录。新增物理LLM调用为0。

## 研究问题与主要数值

原V2 validation200的全量默认准确率86.00%，候选86.333%，有限池平均收益+0.333个百分点。该批题已有开发使用。184题动作不同，16题为已知零增益。三个生成重复始终属于同一原始问题。

主预算为46次期望Gold与543,186.88预测Token。下表均为相同信息权限、相同配对估计框架下的真实有限池回放。

| 方法 | RMSE百分点 | 平均Gold | 缓存记录复算Token |
|---|---:|---:|---:|
{chr(10).join(tables)}

**T1没有通过真实题池的效率验证。** 达到U1主档同一精确设计RMSE的事后反解，T1约需84.60道Gold和981,084 Token，U1为46道和539,991 Token。该反解使用全量有限池真值，解释范围限于当前题池的条件性效率。

## 为什么没有通过

历史嵌套交叉拟合显示，残差平方预测与真实目标相关0.0553。184道待评题中59道工作二阶矩被截到1e-6，下限购买概率0.05将其漏判风险转化为较大的逆概率方差。费用预测变异系数只有4.04%，双预算通常由其中一个约束决定。

主预算T1的Wald覆盖92.4%。有效保守区间观测覆盖100%，平均宽度138.27个百分点。当前没有可靠性保持下的精度或费用优势。

合成实验支持主动估计在部分条件下改善RMSE，T1未普遍优于ASI或Robust适配。方向失准反例保留。有效保守推断在全部合成单元最低覆盖99.9%。T1在异质正收益场景的检出率为95.05%。

## 公开方法证据链及范围

ASI作者AlphaFold原例子完成20预算×1000次重复。Active Testing标准二次函数例子完成100次，原配置为5000次。Fixed-Size作者CPS包执行2000次和全部17项测试，并在本题池使用实际一二阶纳入概率完成2000次回放。

Robust ASI原Pew数据实验仍为部分复现，缺ATPW79.sav。原minimax运算已与独立实现核对，主表使用明确标注的历史反馈适配。Pew官方下载要求账号登录，取得该文件后可补原生实验。Fixed-Size ASI任务书列出的完整实验仓库当前返回404，本轮使用论文关联的作者CPS包。原生实例与MAB适配的完整区别见01报告。

## 科学判断

- H1的预测辅助部分在当前题池未获支持，U1误差略高于U0。配对结构是本轮共同设计，未单独设置非配对采样轴。
- H2未通过。T1劣于均匀与强公开适配对照，双预算和残差目标未带来实际优势。
- H3的可靠性取决于推断。设计匹配的保守方法有效但较宽，普通Wald的名义覆盖不足。检出能力与错误批准均已联合报告。

保留全部负结果，第一轮到此停止。后续多候选阶段的依据尚未建立。

## 审计、修复及局限

冻结代码、购买概率与预算逐项通过独立审阅。主27格在防覆盖入口中完整重放，全部指标逐项一致。交付过程中发现并恢复了冻结后代码清理造成的源码漂移，原始哈希完全恢复，6000个数值输出等价核对通过。Fixed-Size补充适配的184活跃题与200总题分母索引错误已修复并重跑。

访问隔离由独立设计与评价入口、冻结哈希和应用层日志实现。实际Token来自已执行缓存的逐调用记录，历史模型wall time属于记录费用代理。总体推断的可靠性证据来自已知真值合成实验。

优势是完整保留真实配对输出、公开对照、数学资格检查及成本账本。主要风险是旧开发题池、失准辅助、有限费用异质性及原生复现资料范围。对当前固定池T1无优势的结论置信度高。

## 文件入口

- 01_PUBLIC_BASELINES_AND_ASSUMPTIONS.md记录原生复现与权限。
- 02_PAIRED_DESIGN_AND_INFERENCE.md给出估计、分配与推断。
- 03_REAL_FIXED_POOL_RESULTS.md给出真实题池主表。
- 04_POPULATION_RELIABILITY_AND_POWER.md给出可靠性与检出。
- 05_ABLATIONS_COST_AND_RESEARCH_DECISION.md给出消融与费用。
- B1_COMPLETE_RESULTS.json索引完整原始指标与完成状态。
- REPRODUCE.md提供防覆盖复现命令。

## 拓展思考与可能关注的问题

本轮最直接的发现是，合法的无偏修正和最优工作模型分配可以同时成立，而实际评测效率仍然下降。能否改善取决于历史残差信息对新题的有效性，以及保守判断所需的样本量。当前证据支持保留简单公开评估方法作为后续研究基准。
""")
    text_file('REPRODUCE.md',r"""
# B1 复现入口

## 环境与资产

规范实验目录为E:​\语言模型组合路由\research\b1_reliable_paired_evaluation，环境为项目原Python3.11及NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2。下面PowerShell变量使用真实完整路径。全部主计算使用CPU。

```powershell
$b1Work = 'E:\语言模型组合路由\research\b1_reliable_paired_evaluation'
$b1Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\main\reproduce_guarded.py') verify
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\main\reproduce_guarded.py') replay
```

verify先核验原协议中的全部源码、辅助模型和概率设计哈希。replay只创建带UTC时间的新私有目录，重放完整27格并与原始公开指标逐项比较，原始实验产物保留。

## 隐藏收益与费用边界

private/assets保存冻结模型和输入，不含验证真实评分。private/design保存预先冻结购买概率。评价入口才读取原selector_validation评分，并只在已购买位置构造反馈修正与当次费用。private/evaluator保存独立真值，private/replays保留27格购买矩阵、逐次估计、区间及Token。fresh400和sealed200载荷继续封存。

公开仓库保存聚合结果和重建代码，原始问题、评分矩阵、权重和逐题账本位于本机私有目录。GitHub研究目录routing_idea2/research/b1_reliable_paired_evaluation的内容对应本机规范目录中的public子目录。复制时保持该映射，使用项目已有的私有V2资产及记录哈希运行。

## 数学资格检查

```powershell
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\inference\qualify_inference.py') --output (Join-Path $b1Work 'private\qualification_reproduction')
```

资格产物包含固定题池完整枚举、总体1728状态枚举、LTT合法iid有界例子及无标签回退。公开方法下载与原生例子的命令另见PUBLIC_METHODS_REPRODUCE.md。

## 合成重放

```powershell
$b1Replay = Join-Path $b1Work ('private\synthetic_reproduction_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\inference\run_synthetic.py') --output (Join-Path $b1Replay 'public') --private (Join-Path $b1Replay 'traces') --methods-directory (Join-Path $b1Work 'public\methods') --design-directory (Join-Path $b1Work 'public\main') --repetitions 2000
```

该命令执行全部36个预设设置。预定的三个题量与两个相关性世界分别保留，三个生成重复合并为一道题。

## 补充与原生复现范围

run_supplements.py由独立评价入口调用已冻结方法，在原始私有Oracle已生成后重放CPS与LURE两项补充。默认创建全新的private/supplement_reproductions时间目录，也支持--output指定新的空目录。程序拒绝写回原始规范目录或任何已存在目录，并将重放指标与原始结果比较。正式结果位于SUPPLEMENTARY_FINITE_POOL_RESULTS.json及private/supplements。

Robust ASI原Pew完整实验需用户从Pew官方注册下载ATPW79.sav。当前已完成其原生运算资格和MAB历史反馈适配。Fixed-Size原完整实验仓库不可访问，作者CPS包的实际复现范围见01报告。
""".replace('E:\u200b','E:').replace('\\\\', '\\'))
    source_correction=dict(status='RESOLVED_AND_REPLAY_VERIFIED',
        frozen_sha256='238f0c58035f5d435ac56caf87f5cd07f662b6cb5fe5b9bd343e42ff010ce5f1',
        restored_sha256=sha(BASE/'public/inference/b1_inference.py'),
        original_numerical_artifacts_preserved=True,
        source_cleanup_archived_privately=True,all_27_main_cells_reproduced_exactly=True,
        supplemental_active_vs_total_denominator_error='corrected, regression qualified, both supplements fully rerun')
    put(BASE/'public/SOURCE_AND_REPRODUCTION_CORRECTIONS.json',source_correction)
    matrix=BASE/'public/04_EXPERIMENT_MATRIX.csv'
    with matrix.open(encoding='utf-8') as stream:existing=list(csv.DictReader(stream))
    existing=[x for x in existing if not x['cell'].startswith(('synthetic__','supplement__','native__'))]
    for s in synthetic['settings']:
        existing.append(dict(cell=f"synthetic__{s['world']}__{s['n']}__{s['correlation']}",
            role='known-population reliability and power',methods='six matched-estimator methods',budget='expected .25 answer/.25 execution',
            repetitions=2000,independent_unit='iid original question cluster',status='COMPLETE',artifact='inference/SYNTHETIC_COMPLETE_RESULTS.json'))
    for name in supplement:
        existing.append(dict(cell='supplement__'+name,role='finite-pool public method supplement',methods=name,budget='primary',
            repetitions=2000,independent_unit='original question cluster',status='COMPLETE_TASK_ADAPTER',artifact='SUPPLEMENTARY_FINITE_POOL_RESULTS.json'))
    with matrix.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(existing[0]));writer.writeheader();writer.writerows(existing)
    refs=['B1_PROTOCOL.json','REAL_FIXED_POOL_RESULTS.json','SUPPLEMENTARY_FINITE_POOL_RESULTS.json',
        'inference/SYNTHETIC_PROTOCOL.json','inference/SYNTHETIC_COMPLETE_RESULTS.json',
        'NATIVE_REFERENCE_RESULTS.json','PUBLIC_METHOD_RESULTS.json','UPSTREAM_SOURCE_MANIFEST.json',
        'EQUAL_RMSE_COST_DIAGNOSTIC.json','assets/ASSET_FREEZE_PUBLIC.json','assets/AUXILIARY_DIAGNOSTICS.json',
        'assets/ROOT_PIPELINE_REVIEW.json','assets/SYNTHETIC_REVIEW.json','inference/INFERENCE_QUALIFICATION.json',
        'SOURCE_AND_REPRODUCTION_CORRECTIONS.json']
    put(BASE/'public/B1_COMPLETE_RESULTS.json',dict(
        schema='MAB_B1_study_index_v1',status='COMPLETED_BOUNDED_STUDY_WITH_NATIVE_REPRODUCTION_LIMITS',
        scientific_decision='T1_REAL_POOL_EFFICIENCY_NOT_SUPPORTED',
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        counts=dict(real_cells=27,real_sampling_repetitions_per_cell=2000,
            synthetic_settings=36,synthetic_methods_per_setting=6,synthetic_repetitions=2000,
            synthetic_method_estimates=432000,supplements=2,supplement_repetitions=2000,
            new_physical_LLM_calls=0),
        real_finite_pool_gain=real['ground_truth'],
        native_reproduction=dict(ASI='author AlphaFold full budget grid executed',
            Robust_ASI='PARTIAL_missing_registered_Pew_dataset',
            Active_Testing='author standard example executed with 100 versus original 5000 repeats',
            Fixed_Size_ASI='author CPS package executed; requested full experiment repo HTTP404',
            LTT='native operator and valid iid bounded-loss example only'),
        evaluation_scope='old V2 development pool, plus known-truth synthetic populations',
        results={name:dict(path=name,sha256=sha(BASE/'public'/name)) for name in refs},
        all_main_metrics_reproduced_exactly=True,router_weights_updated=False,
        fresh400_outcomes_accessed=False,sealed200_outcomes_accessed=False,
        next_stage='STOP_FIRST_ROUND_NO_MULTI_CANDIDATE_EXPANSION',
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        artifact_roles=dict(public='code, provenance, aggregated metrics and reports',
            private='weights, question identities, true score matrices, purchased feedback, individual execution ledgers')))
    put(BASE/'public/B1_DELIVERY_AUDIT.json',dict(status='PASS_LOCAL',
        frozen_main=verify,requirements=dict(main_three_budgets=True,all_requested_samplers_or_explicit_native_limits=True,
            both_targets_separate=True,all_27_main_cells_and_36_synthetic_settings_complete=True,
            four_ablations=True,shared_execution_reuse=True,history_costs_separate=True,
            zero_labels_qualified=True,conservative_and_asymptotic_intervals_separate=True,
            no_new_LLM_calls=True),native_reproduction_limits=True,
        original_27_main_cells_independently_replayed=True))
    print(json.dumps(dict(status='REPORTS_ASSEMBLED',scientific_decision='T1_REAL_POOL_EFFICIENCY_NOT_SUPPORTED')))


if __name__=='__main__':main()
