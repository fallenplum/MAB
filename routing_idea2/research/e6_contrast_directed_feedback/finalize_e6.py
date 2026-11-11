"""Aggregate frozen E6 stage artifacts without changing any evaluated policy."""
import os
from pathlib import Path
import sys
import json
import hashlib
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e6_contrast_directed_feedback'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
sys.path.insert(0,str(ROOT/'research/e5_paired_gain_validation/public'))
from e5_common import load_train,sha,put
from cost_and_parameters import resources


def holm(p):
    order=np.argsort(p)
    result=np.zeros(len(p))
    previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(p)-rank)*p[index]))
        result[index]=previous
    return result


def pp(x):
    return f'{100*x:+.4f}'


def ci(x):
    return '['+', '.join(pp(v) for v in x)+']'


def main():
    A=json.loads((PUBLIC/'a/A_RESULTS.json').read_text())
    support=json.loads((PUBLIC/'a/A_SUPPORT_AND_COST_DIAGNOSTICS.json').read_text())
    B=json.loads((PUBLIC/'b/E6_B_RESULTS.json').read_text())
    C=json.loads((PUBLIC/'c/E6_C_RESULTS.json').read_text())
    with np.load(PRIVATE/'a/PRIMARY_A_CONTRAST.npz') as a:
        av=float(a['observed'])
        ab=a['bootstrap']
    records=[dict(name='A3-task-matched-random',point=av,ci95=np.quantile(ab,[.025,.975]).tolist(),
        ci_family_bonferroni=np.quantile(ab,[.05/6,1-.05/6]).tolist(),
        centered_bootstrap_p=float((1+np.sum(abs(ab-av)>=abs(av)-1e-15))/(len(ab)+1))),
        dict(name='C-Contrast-Marginal@36',**C['paired_comparisons']['Contrast-Marginal@36']),
        dict(name='C-Contrast-Uniform@36',**C['paired_comparisons']['Contrast-Uniform@36'])]
    adjusted=holm([row['centered_bootstrap_p'] for row in records])
    for row,p in zip(records,adjusted):
        row['Holm_p']=float(p)
        row['positive_evidence_after_family_control']=bool(p<.05 and row['ci_family_bonferroni'][0]>0)
    stats=dict(status='COMPLETE',family_size=3,main_comparisons=records,
        method='Two-sided centered original-question bootstrap tests, Holm family correction, percentile95CI and Bonferroni98.3333CI',
        A_units=600,C_Q_units=300,LLM_repeats_per_question=3,
        independent_unit='original question',acquisition_MC_unit='pilot cluster, 40 purchases nested within each of50 pilot draws per fold',
        limitations=['frozen-policy conditional bootstrap, no outer-history retraining',
                     'five outer histories overlap', 'Q inference conditional on P pool',
                     'train600 reused development, no fresh or sealed evaluation',
                     'bootstrap p and percentile intervals are descriptive and not exact inversions'])
    put(PUBLIC/'E6_PRIMARY_STATISTICS.json',stats)
    diagnostics=[]
    token_exceeded={name:0 for name in C['methods'][:4]}
    design_count=0
    max_conditional_token_ratio=0.
    call_fallback=0
    bounds=dict(max_call_violation=0,max_token_violation=0,max_gold_violation=0)
    reuse=[]
    for f in range(5):
        summary=C['fold_summaries'][f]
        gp=np.asarray(summary['gainsP'])
        gq=np.asarray(summary['gainsQ'])
        best=np.flatnonzero(gp>=gp.max()-1e-12)[0]
        correlation=None if np.std(gp)==0 or np.std(gq)==0 else float(np.corrcoef(gp,gq)[0,1])
        diagnostics.append(dict(fold=f,names=summary['names'],gainsP=gp.tolist(),gainsQ=gq.tolist(),
            pool_best_name=summary['names'][best],pool_best_Q_gain=float(gq[best]),
            Q_oracle_gain=float(gq.max()),pool_Q_policy_value_correlation=correlation,
            P_Q_split_source='question ID hash fixed before E6-A evaluation',
            baseline_Q_accuracy=float(np.mean(np.load(PRIVATE/f'c/EVALUATED_C_f{f}.npz')['RbQ']))))
        plans=json.loads((PRIVATE/f'c/PLANNING_C_f{f}.json').read_text())
        for row in plans:
            design_count+=1
            call_fallback+=int(row['fallback'])
            for name in token_exceeded:
                ratio=row['conditional_expected_tokens'][name]/row['predicted_token_budget']
                token_exceeded[name]+=int(ratio>1+1e-10)
                max_conditional_token_ratio=max(max_conditional_token_ratio,ratio)
        for Bgold in [24,36,48]:
            for name in C['methods']:
                row=summary['methods'][str(Bgold)][name]
                reuse.append(dict(fold=f,budget=Bgold,method=name,
                    member_calls_with_reuse=row['cost_mean']['member_calls'],
                    member_calls_without_reuse=18*Bgold,
                    member_calls_saved=18*Bgold-row['cost_mean']['member_calls'],
                    gold_with_repeat_reuse=Bgold,gold_if_three_repeats_rebilled=3*Bgold,
                    aggregator_calls=row['cost_mean']['aggregator_calls']))
    execution=dict(conditional_expected_token_exceed_count=token_exceeded,designs_per_method=design_count,
        maximum_conditional_to_proposal_token_budget_ratio=max_conditional_token_ratio,
        hard_call_cap_fallbacks=call_fallback,
        token_constraint_scope='F-only expected-token constraint belongs to Bernoulli proposal. Conditioning may exceed that expectation. Frozen worst-case token cap and exact call/gold caps remain enforced.',
        replay_costs='actual cached output lengths charged only after a purchase, unique members per question/seed',reuse=reuse)
    put(PUBLIC/'E6_COST_AND_POOL_DIAGNOSTICS.json',dict(status='COMPLETE',pool_Q_diagnostics=diagnostics,execution=execution))
    table=[]
    for name,row in C['table']['36'].items():
        table.append('| '+name+' | '+f"{100*row['accuracy']:.4f}%"+' | '+pp(row['gain_default']['point'])+' '+ci(row['gain_default']['ci95'])+' | '+f"{100*row['pool_regret']:.4f}"+' | '+f"{100*row['Q_regret']:.4f}"+' | '+f"{row['feedback_cost']['tokens']:.1f}"+' |')
    quality=[]
    for budget,rows in C['table'].items():
        for name,row in rows.items():
            quality.append(f"| {budget} | {name} | {100*row['accuracy']:.4f}% | {row['feedback_cost']['member_calls']:.3f} | {row['feedback_cost']['aggregator_calls']:.0f} | {row['feedback_cost']['tokens']:.1f} | {row['feedback_plus_Q_service_cost']['tokens']:.1f} |")
    d0=diagnostics[0]
    ctext=f'''# E6-C 真实缓存上的有限反馈策略识别

## 研究问题

固定F/G学习的候选策略，仅改变在P60采集哪些配对反馈，评价最终选中策略在不同Q60题上的准确率。主Gold预算为每折36，先导12题包含在预算中。

## 实验设置

五折P/Q划分在E6-A读取外折评分前持久保存，按题目ID哈希并在GSM/BBH内各分30题。合计300道P题、300道Q题，三个生成重复在题内平均。全部均属于旧train600开发资产。每折50份共享随机先导，每份40次采样，共2000次采集回放。五折共150000个方法×预算×随机化策略选择记录。

候选为默认、全切换、E5 gap、E5因子化、A2和A3，按整个E区输入上的动作去重。A4不进入。所有方法共用默认及P2候选，不更新原始价值预测器。

先导反馈按门控签名估计题级Dbar²，4个伪观测向先导池平均收缩。间距截断固定0.01。求解器使用独立Bernoulli有限池方差代理的系数与扣除常数。分配采用F区历史Token预测，未知当前输出长度未用于购买决策。

## 预算实现与对照

独立Bernoulli无法保证每次恰好满足Gold硬上限。主回放将分配概率转为条件Bernoulli分布，固定样本数并约束输入可知的成员与聚合调用数。动态规划精确计算实际一阶纳入概率，再执行HT估计。一般依赖采样方差需要pi_ij，任务书中的独立Bernoulli公式仅作为另列诊断。条件化后的实际方差没有被声称为凸代理的最优值。

调用上限为3×floor(7.5×B)，即每道Gold最多平均22.5次历史成员/聚合调用。正纳入不可行时仅根据输入调用数回退24×B。本次回退{call_fallback}份设计。Token硬界为159744×B，来自冻结输入和生成长度上限。每题一次Gold复用三个生成种子，成员及聚合调用全部付费。NoPilotUniform将全部Gold用于普通随机子集，同样执行上限。

F-only预计Token约束属于Bernoulli提案。条件化后相应预计Token超出次数为{json.dumps(token_exceeded,ensure_ascii=False)}，每方法共{design_count}份设计，最大比值{max_conditional_token_ratio:.6f}。这些预期偏移没有被当成硬预算保证。实际Gold、调用数及最坏Token界均逐记录检查。

## 数值结果

Q区默认准确率为{100*C['default_accuracy']:.4f}%。表中成本为平均每外折60道P池的已购反馈成本，非新增物理调用。

| 方法 | Q准确率 | 相对默认及95%CI，百分点 | P选择损失，百分点 | Q选择损失，百分点 | 反馈Token |
|---|---|---|---|---|---|
{chr(10).join(table)}

Contrast减Uniform为{pp(C['paired_comparisons']['Contrast-Uniform@36']['point'])}个百分点，题级95%CI为{ci(C['paired_comparisons']['Contrast-Uniform@36']['ci95'])}。Contrast与Marginal全部样本逐字节相同。GapContrast减Uniform为{pp(C['paired_comparisons']['GapContrast-Uniform@36']['point'])}个百分点，95%CI为{ci(C['paired_comparisons']['GapContrast-Uniform@36']['ci95'])}。

## 质量与实际成本

下表全部保留。Gold为每折预算，聚合和成员为全部三个生成重复的平均源执行调用。最后一列加上所选策略在Q60服务的Token，Q事后双组合评价另记研究诊断成本。

| Gold | 方法 | Q准确率 | 反馈成员调用 | 反馈聚合调用 | 反馈Token | 反馈＋Q服务Token |
|---|---|---|---|---|---|---|
{chr(10).join(quality)}

主比较具有相同Gold和执行上限，各算法实际Token花费不同，不能把相同上限描述为相同实际花费。完整Token、输入/输出Token、成员/聚合、评分次数及源模型wall-time均保存在JSON和私有逐题账本。物理新增LLM调用为0。

## 统计结论

对Q的5000次题级bootstrap按外折×GSM/BBH分层，保留三重复及全部策略，条件于P反馈。Monte Carlo区间另外按250个先导簇处理，每簇40次购买。主Contrast减Uniform的Monte Carlo区间为{ci(C['paired_comparisons']['Contrast-Uniform@36']['paired_MC']['ci95_normal'])}个百分点。这个区间衡量固定池上的采样均值误差，Q题级区间衡量另一种不确定性，不能混用。统一三项主检验见E6_PRIMARY_STATISTICS.json。

## 失败原因

默认和全切换使未归一化目标恒等。主方法对Uniform的微小正点差同样由旧Marginal取得，不能归为新增差值目标的价值。归一化目标在本次真实回放没有稳定改进。

反馈池最优策略也可能无法迁移到Q。外折0中，P区全切换收益为{pp(d0['gainsP'][1])}个百分点，Q区为{pp(d0['gainsQ'][1])}个百分点。即使完全知道该P池，选中P最优也可能损害Q表现。这是已观察到的有限样本池间差异，不足以单独确定其总体来源。各折全策略P/Q值和相关性完整保存。

## 优势与风险

优势是固定候选、共享先导、严格Gold和调用上限、精确纳入概率以及独立Q评价。风险包括先导占用25%—50%的Gold、题内标签相关、每折仅60道反馈池及60道Q题、训练历史重叠、旧开发题复用，以及代理最优分配经过条件化后改变的抽样性质。研究对象为冻结有限策略族。

## 下一步

本轮没有得到可将新增机制推进到fresh400的正向证据。已完成计划中的低成本回放并保存负结果。若后续继续，须先明确能改变该退化的竞争集合或在不同问题上验证更可靠的候选差异，再另行冻结新协议。当前不追加模型搜索。
'''
    (PUBLIC/'c/03_REAL_CACHE_POLICY_IDENTIFICATION.md').write_text(ctext,encoding='utf-8')
    statrows=[]
    for row in records:
        statrows.append(f"| {row['name']} | {pp(row['point'])} | {ci(row['ci95'])} | {ci(row['ci_family_bonferroni'])} | {row['Holm_p']:.6f} |")
    (PUBLIC/'04_PRIMARY_STATISTICAL_ANALYSIS.md').write_text('''# E6统一主统计

## 研究问题与设置

预先固定A3相对任务内同覆盖随机、C未归一化Contrast相对Marginal、C Contrast相对Uniform三项。C主Gold=36，A主模型A3不按E区结果改变。仅原始问题作为独立抽样记录，三种子和同题策略结果共同保留。

## 数值结果

| 主比较 | 差值，百分点 | 95%CI | 家族Bonferroni98.3333%CI | Holm双侧p |
|---|---|---|---|---|
'''+chr(10).join(statrows)+'''

## 统计结论

三项均未通过统一家族检验。Contrast减Marginal的零差由目标恒等性决定。C的Monte Carlo均值误差区间与真实Q题的不确定性分开。其余模型、预算、先导与合成格属于探索性诊断，未选择性保留。

## 优势与风险

优势是题级配对、匹配随机重新计算切换数、明确三项检验与同题重复处理。区间条件于已训练策略及已观察P池，没有重新拟合重叠外折历史，也没有建立独立最终测试保证。

## 下一步

保留所有正负方向与完整原始回放，不将单个正Monte Carlo区间转换为真实泛化确认。
''',encoding='utf-8')
    data,refs=load_train()
    cache=resources(data,np.arange(600),[range(20)]*600,gold=600)
    provenance=dict(status='COMPLETE',reference_inputs=refs,source_cache_once=cache,
        task_sha256=sha(PRIVATE/'E6_USER_TASK.txt'),frozen_protocol_sha256=sha(PRIVATE/'PROTOCOL_FROZEN.json'),
        split_sha256=sha(PRIVATE/'E6_FGE_PQ_FROZEN.npz'),
        physical_new_member_calls=0,physical_new_aggregator_calls=0,
        decoded_experimental_split='selector_train only',fresh400_evaluated=False,sealed200_evaluated=False,
        isolation_scope='Array-level planner/fit permissions and explicit outcome perturbations, not OS-wide access forensics',
        old_branch='codex/e5-paired-gain-validation',old_commit='7e06c632787f1de093732aed0250c8a30d3035b7')
    put(PUBLIC/'E6_PROVENANCE_AND_COST.json',provenance)
    complete=dict(status='COMPLETE',stages=dict(A='COMPLETE',B='COMPLETE',C='COMPLETE'),
        scientific_hypotheses=dict(H1='NOT_ESTABLISHED',H2='NOT_ESTABLISHED',
            H3='UNNORMALIZED_OBJECTIVE_IDENTICAL; GAP_NORMALIZATION_NOT_STABLE_END_TO_END'),
        main_statistics=stats,main_real_accuracy={k:v['accuracy'] for k,v in C['table']['36'].items()},
        default_Q_accuracy=C['default_accuracy'],provenance=provenance,
        stage_files=['a/A_RESULTS.json','b/E6_B_RESULTS.json','c/E6_C_RESULTS.json'],
        judgments='Implementation/numerical checks are distinct from scientific benefit. No new independent confirmation claimed.')
    put(PUBLIC/'E6_COMPLETE_RESULTS.json',complete)
    a3=next(row for row in A['configurations'] if row['configuration']=='A3')
    summary=f'''# MAB E6 实验总结

## 研究问题

固定6模型、K=3、20组合和V2聚合器，检验换队判断的信息与监督，以及冻结策略族的差值导向反馈分配。A、B、C均已完成，新增物理LLM调用0。

## 实验设置与对照

沿用五折F360/G120/E120，主候选A3。G内原题三折选择λ∈{{0.01,0.1,1}}，服务阈值0。A4仅扩展监督，不进入C。B复用E5两类奖励世界、三预算、两先导规模、每格2000次配对模拟。C按预存题目ID划分P60/Q60，在相同Gold与调用上限下比较Uniform、Marginal、Contrast、GapContrast、NoPilotUniform。

## 数值结果

- A3准确率85.1111%，默认85.0556%，收益+0.0556个百分点，95%CI[-1.1667,+1.2222]。A3相对任务匹配随机为{pp(av)}个百分点，95%CI{ci(records[0]['ci95'])}。A3相对A1为0.0000个百分点。
- A4准确率85.2778%，相比A3增加0.1667个百分点，95%CI[-0.2778,+0.6111]。有效方向支持题并集从91增至202，独立G题仍为413，每折额外6480次历史聚合。
- 默认和全切换共同存在时，最坏Marginal与未归一化Contrast目标完全恒等。B和C逐样本验证两方法同概率、同反馈、同选择。
- 合成GapContrast相对Uniform的选择损失在9/12格点估计降低，未经跨格校正的区间3格全负、1格全正。计入先导总成本后，相对无先导Uniform在12/12格更差。
- C主预算36时，Uniform准确率84.4211%，Marginal/Contrast84.4794%，GapContrast84.3879%，无先导Uniform84.3743%，默认84.1111%。Contrast减Uniform为{pp(records[2]['point'])}个百分点，题级95%CI{ci(records[2]['ci95'])}，此变化同样由旧目标获得。

## 统计结论

三项主比较统一Holm校正后均无正向支持。H1、H2未建立当前特征和辅助监督带来稳定条件切换收益。H3的未归一化目标受候选族结构支配，归一化代理也未建立稳定、覆盖先导成本的实际优势。以上结论限于已冻结方法与旧开发数据。

## 失败原因与边界

方向监督仍集中在少量不同问题，新增同题组合条数不增加独立支持数。两种未归一化目标代数恒等，无法隔离新机制优势。少量先导估计收益间距不稳定，且先导占用资源。P池策略优势与Q区可能方向不同，外折0全切换为P+1.1111/Q−3.3333个百分点。

C包含硬预算抽样的实现修正。主回放采用精确固定样本数与调用上限的条件Bernoulli，HT使用真实纳入概率；独立Bernoulli方差公式另行验证。条件化后的预计Token可能超过提案的期望约束，该偏移完整报告。实际硬Gold、调用数和冻结Token上界均核验。源模型wall-time为代理，原GPU kernel与原编码耗时不可恢复。

## 优势与风险

优势是固定候选、同题配对、沿用原世界、完整资源复用计量、保存负结果和独立算子审计。主要风险是外折历史重叠、旧开发题复用、小P/Q池及有限生成重复。A的600题与C的300道Q题共享来源，不能叠加为900道独立样本。

## 重要文件

- `RELATED_WORK_MAPPING.md`及`THEORY_AND_NUMERICAL_CHECKS.md`记录参考边界、证明与数值测试。
- `a/01_SWITCH_INFORMATION_AND_SUPERVISION.md`、`a/A_RESULTS.json`记录A0—A4、支持数和配对统计。
- `b/E6_B_SYNTHETIC_ALLOCATION.md`、`b/E6_B_RESULTS.json`记录全部合成预算和先导结果。
- `c/03_REAL_CACHE_POLICY_IDENTIFICATION.md`、`c/E6_C_RESULTS.json`记录真实池、Q结果、实际预算和采样统计。
- `04_PRIMARY_STATISTICAL_ANALYSIS.md`及`E6_PRIMARY_STATISTICS.json`记录统一三项检验。
- `REPRODUCE.md`列明复现入口。题目、逐题奖励、参数及完整反馈账本保存在同根private。

## 下一步

本轮计划全部执行完毕，fresh400及sealed200继续未评价。本轮没有提供推进独立确认的正向机制证据。后续若授权新实验，应先解决目标退化、先导成本与P到Q的策略收益不稳定，保持当前负结果。

理论恒等和已保存数值结果的置信度为高。对整体研究方向是否存在可行方法，本轮证据无法作否定性定论。
'''
    (PUBLIC/'EXPERIMENT_SUMMARY.md').write_text(summary,encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',primary=records,execution=execution|{'reuse':'private-length table public JSON'}),ensure_ascii=False))


if __name__=='__main__':
    main()
