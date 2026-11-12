"""Joint E7 primary statistics, resource accounts and bounded research decision."""
import os
from pathlib import Path
import sys
import json
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
sys.path.insert(0,str(PUBLIC))
from e7_common import load_train,put,sha,resources,paired_interval,holm,splits


def number(x):
    return f'{100*x:+.4f}'


def ci(x):
    return '['+', '.join(number(v) for v in x)+']'


def main():
    data,inputs=load_train()
    A=json.loads((PUBLIC/'a/E7_A_RESULTS.json').read_text())
    B=json.loads((PUBLIC/'b/E7_B_RESULTS.json').read_text())
    C=json.loads((PUBLIC/'c/E7_C_RESULTS.json').read_text())
    with np.load(PRIVATE/'b/PRIMARY_B_COMPARISONS.npz') as a:
        ids=a['question_indices']
        task=a['task']
        fold=a['fold']
        learning=a['learning_contribution']
        content=a['content_contribution']
        oldb={name:a[name] for name in a.files}
    with np.load(PRIVATE/'c/E7_C_PAIRED.npz') as a:
        q=a['Q']
        score=a['score']
        perq=a['perq']
        ct=a['task']
        cf=a['fold']
    assert np.array_equal(ids,q) and np.array_equal(task,ct) and np.array_equal(fold,cf)
    contributions=np.asarray([learning,content,perq[1,2,3]-perq[1,0,3],perq[1,2,3]-perq[1,1,3]])
    labels=['P3 n480 minus n240','P3 original n480 minus task-internal shuffled n480 mean',
            'P3 DecisionAware minus Random at Gold60','P3 DecisionAware minus ParameterIG at Gold60']
    records=[]
    boot=[]
    for label,contribution in zip(labels,contributions):
        r,draws=paired_interval(contribution,task,fold,family=4)
        r['comparison']=label
        records.append(r)
        boot.append(draws)
    corrected=holm([r['centered_bootstrap_p'] for r in records])
    for r,p in zip(records,corrected):
        r['Holm_p']=float(p)
        r['positive_evidence_after_family_control']=bool(p<.05 and r['ci_family_bonferroni'][0]>0)
    np.savez_compressed(PRIVATE/'E7_PRIMARY_PAIRED.npz',question_indices=ids,task=task,fold=fold,
        contributions=contributions,bootstrap=np.asarray(boot))
    stats=dict(status='COMPLETE',family_size=4,main_comparisons=records,
        resamples=5000,independent_original_questions=600,generation_repeats=3,
        unit='Original question with all methods, three generation repeats, algorithm seeds and shuffled runs retained together',
        weights='GSM .5, BBH .5 question mean',BBH27_macro='supplementary only',
        inference='Frozen cross-fitted development prediction bootstrap, no repeated outer-history refit',
        repetitions='Neither algorithm seeds, shuffled trainings nor generation repeats add independent questions',
        multiple_comparisons='Holm4 centered bootstrap p; unadjusted95CI and Bonferroni98.75CI',
        overlapping_histories=True,independent_final_confirmation=False)
    put(PUBLIC/'E7_PRIMARY_STATISTICS.json',stats)
    source=resources(data,np.arange(600),[range(20)]*600,gold=600)
    Icost=[]
    for f in range(5):
        I=splits(f)['I']
        Icost.append(resources(data,I,[range(20)]*len(I),gold=240))
    initial={key:float(np.mean([r[key] for r in Icost])) for key in Icost[0]}
    costs=dict(status='COMPLETE',physical_new_LLM_calls=0,source_cache_once=source,
        initial_I240_counterfactual_mean_per_fold=initial,
        packet=dict(gold=1,member_calls=18,aggregator_calls=60,scoring_ops=60,Token_upper_bound=602112),
        Gold60=dict(gold=60,member_calls=1080,aggregator_calls=3600,scoring_ops=3600,Token_upper_bound=36126720),
        source_cache_note='The original cache was prepared once. Overlapping folds and alternative replay paths are counterfactual per-system costs, not new physical expenses.',
        CPU_note='Per-step process_time on Windows can quantize short IG/update intervals to0. Path wall time separately saved. Original encoder/GPU kernel timing unavailable.',
        no_equal_actual_Token_claim=True)
    put(PUBLIC/'E7_COST_SUMMARY.json',costs)
    statrows=[f"| {r['comparison']} | {number(r['point'])} | {ci(r['ci95'])} | {ci(r['ci_family_bonferroni'])} | {r['Holm_p']:.6f} |" for r in records]
    costrows=[]
    for k in ['P1','P3']:
        for method in ['Random','ParameterIG','DecisionAware']:
            row=C['table'][k][method]['60']
            cc=row['feedback_cost']
            costrows.append(f"| {k} | {method} | {100*row['accuracy']:.4f}% | {cc['gold']:.0f} | {cc['member_calls']:.0f} | {cc['aggregator_calls']:.0f} | {cc['tokens']:.1f} | {row['full_path_planning_CPU_mean']:.4f} | {row['full_path_update_CPU_mean']:.4f} |")
    report=f'''# E7 成本与统一统计

## 研究问题

区分增加独立反馈的学习收益、文本对应的内容收益和固定学习器下的采集贡献，同时计量完整反馈包的执行费用。

## 实验设置与对照

主权重为GSM8K、BBH各50%，BBH内部按题平均。BBH27子任务宏平均另列。五个外折构成600道不同E题，算法三种子、三次生成及20次置乱保留在题内。主要比较在新评价前固定为下表四项。C主Gold60，20和40为学习曲线。

## 四项主比较

单位均为百分点。区间由5000次外折×任务分层原题bootstrap得到。Holm校正覆盖全部四项，家族区间为Bonferroni98.75%。

| 比较 | 差值 | 95%CI | 家族98.75%CI | Holm双侧p |
|---|---|---|---|---|
{chr(10).join(statrows)}

## 完整执行成本

历史I240平均每外折成本为240 Gold、{initial['member_calls']:.0f}次成员生成、{initial['aggregator_calls']:.0f}次聚合、{initial['tokens']:.1f} Token。I对所有方法共享，学习器初始化计算与反馈回放开销分开保存。

每新增原题反馈包揭示一次Gold，执行六成员×三生成，共18次成员调用，以及20组合×三生成，共60次聚合和60次评分。60题对应1080次成员、3600次聚合。原题Gold跨生成与组合复用，全部成员及聚合执行仍付费。

下表是平均每外折/算法种子的源执行开销和实测CPU。P1为原直接20组合PCA16 Ridge，P3为核岭回归。

| 学习器 | 采集 | E准确率 | Gold | 成员调用 | 聚合调用 | 实际Token | 规划CPU秒 | 更新CPU秒 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(costrows)}

实际Token随购买题不同而变化。相同Gold和调用数构成同权限对照，不能解释为同实际Token花费。C报告还给出反馈加E120服务的Token、单独源wall-time及各预算曲线。source wall-time为历史模型执行代理，当前CPU计算没有新增GPU推理。

完整原缓存只准备一次，600题共{source['member_calls']}次成员调用、{source['aggregator_calls']}次聚合、{source['tokens']} Token。本轮所有LLM物理新增调用为0。正式90条路径合计5400次回放购买事件，其中同一原始问题可出现在不同反事实路径中，不能把5400当成新独立题数或真实新增Gold。

## 统计结论

四项统一检验均未建立正向家族证据。P3增加监督的+1.6111个百分点具有未经多重校正的95%正区间，家族98.75%区间跨零，Holm p约0.10。内容对应和主动采集的独立收益均未得到支持。

## 优势与风险

优势是预先固定四项主比较、严格原题配对、完整反馈包计费和先冻结路径再评价。风险是旧开发题反复使用、外折历史交叠、置乱训练间相关、工作后验失配，以及有限算法随机重复。当前区间条件于保存的模型和路径，没有覆盖所有模型选择不确定性。

## 失败原因与下一步

误差下降不能替代最终服务质量。核模型的监督规模信号尚未转化为稳定内容证据，DecisionAware对同一核学习器没有超过Random或IG。按任务书保留这些结果，下一轮先处理候选召回与信息覆盖，采用独立冻结设计检验新机制。
'''
    (PUBLIC/'04_COST_AND_STATISTICS.md').write_text(report,encoding='utf-8')
    decision=f'''# E7 研究结论与下一步

## 研究问题与设置

本轮固定6个冻结模型、K=3、20组合、V2聚合器和真实评分，只使用既有train600。A沿用E6的F360默认及P2候选。B、C改用每折I240/U240/E120，I确定的默认在学习期间固定。两种默认的口径已经区分。

## 好组合有没有进入可选择范围？

**部分进入，当前二选一丢失超过一半可观察的纠错机会。** 全20中有197条可纠错生成记录，来自122道不同题，原默认加P2 Top1保留90条，保留率45.6853%。107条在进入二选一前丢失。保留的90条中，E6-A3纠正46条、拒绝44条，另破坏45条默认正确记录。

默认加Top3、Top5保留75.1269%、82.7411%，但P2排序与同规模随机接近。扩大集合的覆盖不能证明排序本身有效。二候选跨生成重复评价86.7222%，全20为88.3889%，并列均匀的补充结果方向一致。该成绩使用同题另两生成反馈，属于事后诊断。

## 只看题目，能不能学到哪些组合更好？

**有增加独立监督的开发信号，尚未建立稳定的同任务内容收益。** 核岭回归从240到480题准确率85.1667%→86.7778%，增益+1.6111个百分点，95%CI为{ci(records[0]['ci95'])}，统一Holm p为{records[0]['Holm_p']:.6f}。PCA16 Ridge和完整向量线性Ridge的同类曲线没有稳定提升。

核模型原始对应相对任务内整题置乱平均为+0.7333个百分点，95%CI为{ci(records[1]['ci95'])}。其原始对应的中心化MSE反而比置乱平均高0.000688，不能仅凭训练误差下降解释为学到了题目特定组合优势。模型中的任务/子任务标签只用于划分、置乱与统计。

## 在同一个学习器下，我们的采集是否比随机反馈更有效？

**本轮没有获得正向证据。** 核学习器Gold60时，Random85.4259%、ParameterIG86.0000%、DecisionAware85.3889%。DecisionAware减Random为{number(records[2]['point'])}个百分点，95%CI为{ci(records[2]['ci95'])}；减IG为{number(records[3]['point'])}个百分点，95%CI为{ci(records[3]['ci95'])}。原Ridge同条件Random86.2593%、IG86.6667%、DecisionAware86.2037%。两学习器下DecisionAware均没有独立采集优势。

## 科学结论

数值正确、软件正负控制通过及路径完整，都不能证明工作模型的决策价值代理有效。当前结果定位到三个具体问题。候选构造前丢失54.31%的已观察机会，同任务内容证据有限，主动采集的预测价值与未来实际收益尚未对应。增加新独立题的信号与采集机制的贡献必须分开解释。

反馈确实改变了模型与服务动作，六个学习器/采集配置在Gold60时的中心化误差均下降，但相对初始化的准确率收益95%区间全部跨零。模型误差改善和后续选队改善分别保留。

本轮没有证据支持把连续反馈后验收缩直接当作准确率改善。核模型的独立监督曲线可作为探索性线索，未达到统一家族证据要求。

## 优势与风险

优势是完整候选机会分解、三个固定数学预测器、语义组内置乱、同一学习器更新规则、90条前瞻反馈路径、360个模型快照以及逐题成本和访问日志。风险是旧开发资产、每子任务支持少、三生成有限重复、独立输出Gaussian工作假设，以及只采用I输入参考的决策代理可能与未见题分布不一致。

PCA16经过历史标准化，而完整向量线性Ridge保留原归一化尺度。相同raw正则100不构成两个特征先验完全等价，不能将其差异唯一归因于压缩。核模型的不确定性仅为工作后验，尚无实际概率校准结论。64次对称幻想是KG近似，具有数值积分误差。

## 下一步

根据本轮结果，优先处理候选推荐丢失的机会，再检验输入表示及不同独立问题的覆盖。保留原Ridge作为固定基础学习器，以及核模型作为充分监督诊断。当前采集效用没有得到独立正证据，暂不扩大采集网格。

本轮交付已有矩阵上的完整诊断，没有新增数据执行计划。若后续另行提出补数据方案，应按BBH27子任务覆盖和预先固定输入需求确定类别与题数，并明确训练、开发、最终评价角色、每题组合/重复及成员/聚合/Token上限。现有真值不能用于挑选更容易得到正结果的问题。

已保存的数值与候选漏失诊断置信度为高。对总体输入可学习性及其他采集机制的结论仍受本轮范围限制。
'''
    (PUBLIC/'05_E7_RESEARCH_DECISION.md').write_text(decision,encoding='utf-8')
    outputs=['01_CANDIDATE_OPPORTUNITY.md','02_INPUT_SIGNAL_AND_LEARNING_CURVES.md','03_ACTIVE_FEEDBACK_LEARNING.md',
        '04_COST_AND_STATISTICS.md','05_E7_RESEARCH_DECISION.md','RELATED_WORK_MAPPING.md','E7_PROTOCOL.json','REPRODUCE.md']
    result=dict(status='COMPLETE',stages=dict(A='COMPLETE',B='COMPLETE',C='COMPLETE'),
        branch='codex/e7-opportunity-and-feedback-learning',base_commit='0c34a38bede9fcf46359316058759b27fa9f5fea',
        fixed=dict(N=6,K=3,combinations=20,generation_seeds=[42,137,2024]),
        primary_statistics=stats,source_costs=costs,inputs=inputs,
        answers=dict(opportunity='Top1 pair retains45.6853% observed opportunities; loss occurs both before and inside gate',
            input_learning='P3 independent-supervision growth signal, but no family-supported within-task content benefit',
            acquisition='No evidence DecisionAware improves Random/IG under either fixed learner'),
        original_questions=600,formal_replay_paths=90,model_snapshots=360,physical_new_LLM_calls=0,
        development_only=True,fresh400_evaluated=False,sealed200_evaluated=False,
        protocol_sha256=sha(PUBLIC/'E7_PROTOCOL.json'),split_sha256=sha(PRIVATE/'E7_SPLITS_FROZEN.npz'),
        required_reports=outputs,stage_results=['a/E7_A_RESULTS.json','b/E7_B_RESULTS.json','c/E7_C_RESULTS.json'])
    put(PUBLIC/'E7_COMPLETE_RESULTS.json',result)
    print(json.dumps(dict(status='COMPLETE',primary=records),ensure_ascii=False))


if __name__=='__main__':
    main()
