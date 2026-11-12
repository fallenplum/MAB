"""Verify frozen inputs and build evidence-bounded B1 deliverables."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
import numpy as np
from run_b1 import BASE, ROOT, put, sha, frozen_inputs, load_npz
from b1_design import optimal_probabilities
from public_methods import asi_probabilities, robust_probabilities, scale_to_budgets


def text_file(name, content):
    (BASE/'public'/name).write_text(content.strip()+'\n',encoding='utf-8')


def verify_main():
    p=json.loads((BASE/'public/B1_PROTOCOL.json').read_text(encoding='utf-8'))
    checks={relative:sha(BASE/relative)==digest for relative,digest in p['code_hashes'].items()}
    for relative,ok in checks.items():
        if not ok:
            raise AssertionError('Frozen source hash mismatch '+relative)
    a,receipt=frozen_inputs()
    designs=load_npz(BASE/'private/design/FROZEN_DESIGNS.npz')
    assert sha(BASE/'private/design/FROZEN_DESIGNS.npz')==p['freeze_hashes']['probability_designs']
    assert sha(BASE/'public/HISTORICAL_METHOD_TUNING.json')==p['freeze_hashes']['method_tuning']
    summary=json.loads((BASE/'public/DESIGN_SUMMARY.json').read_text(encoding='utf-8'))
    for key,record in summary.items():
        q=designs[key+'__p']
        assert abs(record['expected_answers']-q.sum())<1e-8
        assert abs(record['expected_tokens']-q@a['cost_prediction'])<1e-5
        assert np.all(q[~a['active']]==0)
        assert np.all(q[a['active']]>=.05-1e-10)
        assert abs(record['answer_cap']-len(q[a['active']])*p['budgets'][record['budget']][0])<1e-8
    return dict(status='PASS',frozen_source_hashes=checks,design_cells=len(summary),
        protocol_sha256=sha(BASE/'public/B1_PROTOCOL.json'),
        design_summary_sha256=sha(BASE/'public/DESIGN_SUMMARY.json'),
        validation_is_original_V2=True,source_and_probability_hashes_preserved=True,
        same_actions_are_structural_zeros=True,fresh400_and_sealed200_outcomes_opened=False)


def cost_to_rmse():
    """Retrospective evaluator-only resource inversion, with frozen work models.

    The target is the primary U1 exact sampling RMSE. This is a conditional
    finite-pool diagnostic, not an implementable reward-aware stopping rule.
    """
    a,_=frozen_inputs(); oracle=load_npz(BASE/'private/evaluator/ORACLE_PAIRED.npz')
    r=json.loads((BASE/'public/REAL_FIXED_POOL_RESULTS.json').read_text(encoding='utf-8'))
    tuning=json.loads((BASE/'public/HISTORICAL_METHOD_TUNING.json').read_text(encoding='utf-8'))['primary']
    mask=a['active'];v=a['vhat'][mask]; c=a['cost_prediction'][mask];d=oracle['d'][mask]
    target=r['methods']['primary__U1']['exact_design_variance']

    def design(method,rate):
        ba,be=rate*len(c),rate*c.sum()
        if method=='T1':return optimal_probabilities(v,c,ba,be)[0]
        if method=='A1_ASI':return asi_probabilities(v,c,ba,be,tau=tuning['asi']['tau'])['p']
        if method=='A2_Robust_ASI':return robust_probabilities(v,c,ba,be,c_radius=tuning['robust']['c_radius'])['p']
        return scale_to_budgets(np.ones(len(c)),c,ba,be)['p']

    answers={}
    for method in ['U0','U1','A1_ASI','A2_Robust_ASI','T1']:
        m=np.zeros_like(d) if method=='U0' else a['m'][mask]
        lo,hi=.05000001,1.
        for _ in range(45):
            rate=(lo+hi)/2;p=design(method,rate)
            variance=np.sum((1-p)/p*(d-m)**2)/200**2
            if variance>target:lo=rate
            else:hi=rate
        p=design(method,hi)
        answers[method]=dict(equal_RMSE_target=float(np.sqrt(target)),
            expected_joint_budget_fraction=hi,expected_gold=float(p.sum()),
            expected_predicted_tokens=float(p@c),
            expected_cache_recorded_tokens=float(p@oracle['tokens'][mask]),
            exact_design_RMSE=float(np.sqrt(np.sum((1-p)/p*(d-m)**2)/200**2)))
    output=dict(scope='retrospective finite-pool evaluator-only expected-resource inversion',
        target='primary U1 exact design RMSE',methods=answers,
        probabilities_use_only_frozen_nuisances=True,
        target_matching_uses_privileged_pool_truth=True,
        operational_claim='does not establish population cost-to-power or an online stopping procedure')
    put(BASE/'public/EQUAL_RMSE_COST_DIAGNOSTIC.json',output)
    return output


def build_real_reports():
    audit=verify_main();put(BASE/'public/MAIN_REPLAY_INTEGRITY_AUDIT.json',audit)
    r=json.loads((BASE/'public/REAL_FIXED_POOL_RESULTS.json').read_text(encoding='utf-8'))
    a,receipt=frozen_inputs(); costs=cost_to_rmse()
    rows=[]
    for method in ['U0','U1','A1_ASI','A2_Robust_ASI','T0_RawMoment','T1']:
        v=r['methods']['primary__'+method]
        rows.append(f"| {method} | {100*v['bias']:+.3f} | {100*v['RMSE']:.3f} | {v['resources']['gold']['mean']:.2f} | {v['resources']['tokens']['mean']:,.0f} | {100*v['inference']['asymptotic_wald']['coverage']:.2f}% | {100*v['inference']['finite_bernstein']['coverage']:.2f}% | {100*v['inference']['finite_bernstein']['width']['mean']:.2f} |")
    subgroup=[]
    for name,v in r['groups'].items():
        subgroup.append(f"| {name} | {v['questions']} | {100*v['default_accuracy']:.2f}% | {100*v['candidate_accuracy']:.2f}% | {100*v['gain']:+.3f} |")
    text_file('03_REAL_FIXED_POOL_RESULTS.md',f"""
# B1 真实有限题池评估

## 研究问题

在原V2 selector_validation200上，隐藏尚未购买的配对收益，以有限评测恢复这200题、三个已冻结生成重复的平均收益。该题池已有历史开发使用。

## 实验设置与对照

恢复原train600直接20组合Ridge权重，PCA16，原始三重复损失的正则化系数100。默认组合为train600全局最优的组合5，成员索引0、2、4。路由输入未使用任务或BBH子任务标签。主预算期望Gold为46，历史预测Token预算543,186.88。

200题中184题动作不同，16题为已知零收益。分母始终保留200。同题三个生成重复先取平均。各方法使用同一2000×200均匀随机数矩阵。完整评分只由评价进程使用，设计进程在其读取之前退出并冻结。

## 完整题池真值

| 范围 | 原始问题数 | 默认准确率 | 候选准确率 | 收益百分点 |
|---|---:|---:|---:|---:|
{chr(10).join(subgroup)}

全池收益为+0.333个百分点。三个生成重复分别为+1.50、−1.00、+0.50个百分点。这是已观测有限题池的精确算术量，不提供未来总体正收益证据。

## 主预算结果

| 方法 | 偏差百分点 | RMSE百分点 | 平均Gold | 缓存记录复算Token | Wald覆盖 | 保守覆盖 | 保守平均宽度百分点 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

ASI与Robust ASI行使用公开公式对应的历史辅助适配。共同估计器、共同推断和共同信息权限下，只改变购买概率。作者原生实例及其区间另列于公开方法报告。

## 统计结论

T1相对U1、ASI适配和Robust ASI适配的MSE均更高。三项配对Monte Carlo比较经过Holm处理，具体区间与p值保留在REAL_FIXED_POOL_RESULTS.json。它们描述这一固定题池上采样误差比较的模拟精度。重复2000次没有增加语言问题样本量。

普通Wald区间未达到标称95%覆盖。主表保守区间采用抽样设计匹配的有限样本界，当前各方法均观测到100%覆盖，但T1区间宽度约138个百分点，不能据此声称判断足够精确。可靠性与实用精度需同时评价。

## 失败原因与机制证据

历史嵌套OOF残差平方预测与真实目标相关仅0.0553。预测最低五分位的平均预测为1e-6，真实残差平方均值0.0583。184道待评题有59个工作二阶矩落在冻结下限。T1相应将59道题分配到购买概率0.05。其无偏修正仍然成立，但大的逆概率权重放大误差。

费用预测的变异系数仅4.04%。主档T1只有执行约束生效，答案约束略松，另外两档分别退化为执行或答案约束。双约束形式没有在这一题池上产生独立的精度优势。

## 下一步与局限

保留当前冻结配置和负结果。真实有限池不支持T1优于强公开适配对照。合成实验另行评价已知总体真值下的错误批准和检出率。

优势是目标真值、权限、配对修正和执行费用可以逐次核对。风险是验证题池已有开发使用，残差模型失准，且费用差异很小。对本题池的负结果置信度高，推广到其他任务的置信度有限。
""")
    text_file('02_PAIRED_DESIGN_AND_INFERENCE.md',r"""
# B1 配对估计、分配与有效推断

## 研究问题与对象

冻结默认与候选动作，原始问题为统计单位。令d_i为同题三个生成重复的配对收益平均，m_i为冻结辅助预测。动作相同的题已知d_i=m_i=0，购买概率为0，保留在N中。其余p_i至少0.05。

## 估计器与证明

独立Bernoulli购买A_i时，

\[
\widehat G_N=N^{-1}\sum_i[m_i+A_i(d_i-m_i)/p_i].
\]

给定固定题池及冻结m、p，有E[A_i/p_i]=1，故E_A[估计]=G_N。辅助预测有偏不会破坏这一设计无偏性。独立购买使协方差项为零，因此

\[
V_N=N^{-2}\sum_i(1-p_i)(d_i-m_i)^2/p_i.
\]

对每个被购入项再除一次p_i，得到无偏Vhat。三个生成重复已包含在d_i内。这个方差未包含未来问题及新生成重复的抽样变化。

## 双期望预算分配

最小化sum_i vhat_i/p_i，约束sum p≤B_answer、sum chat_i p_i≤B_execution，p∈[.05,1]。Lagrangian对内部坐标求导得−vhat_i/p_i²+lambda_answer+lambda_execution chat_i=0。由严格凸性及可行域凸性，满足KKT的截断平方根解为全局最优。

\[
p_i=\operatorname{clip}_{[.05,1]}
\sqrt{\widehat v_i/(\lambda_a+\lambda_e\widehat c_i)}.
\]

数值程序通过两个对偶乘子的单调求根求解，报告原始与对偶目标、互补松弛和约束绑定状态。NUMERICAL_QUALIFICATION.json含独立约束优化器对照。工作二阶矩的预测准确性不由KKT保证。

## 固定池保守区间

记e_i=d_i−m_i，b_i=1+|m_i|。Vhat未除N²的非负随机项A_i(1−p_i)e_i²/p_i²的共同上界为c。独立有界非负和的lower-tail Chernoff不等式可反解得到

\[
V_{\rm sum}\le\widehat V_{\rm sum}+ct+
\sqrt{2ct\widehat V_{\rm sum}+c^2t^2}
\]

以至少1−delta_v成立，其中t=log(1/delta_v)。还可与确定性的sum(1−p_i)b_i²/p_i取较小值。对中心化的A_i e_i/p_i−e_i使用Bernstein界，界限M=max_i b_i max(1,(1−p_i)/p_i)。双侧尾概率与方差上界失效概率采用union bound。单侧决定采用各自5%界，阈值0和0.01分别报告。

完整枚举验证估计器与Vhat无偏。保守界证明及进一步随机小池枚举由独立模块审阅。没有标签时返回由结构性零和有界收益给出的宽区间。工作后验与普通Wald区间单列为渐近诊断。

## 未来总体与LTT

总体推断使用独立同分布原始问题簇，p(x)、m(x)由独立历史固定，且有全输入支持上的IPW变量界。采用总体二阶矩上界及固定时点Bernstein推断。当前输入池归一化概率一般会让变量相互依赖，不能直接调用该iid公式。

LTT是检验组织框架。单候选对应一个检验。原生有界损失检验只在合法iid有界观测下使用，IPW变量超出[0,1]时不调用原生Bernoulli公式。当前研究不进行反复查看后提前停止。

## 优势、风险与后续

设计无偏性允许辅助预测存在偏差，这是可直接证明的性质。逆概率权重在失准时会放大方差，保守界也可能很宽。固定池和总体目标的覆盖范围各自限定，后续是否值得独立评价取决于完整精度与费用证据。
""")
    budget_rows=[]
    for budget in ['primary','execution_tight','answer_tight']:
        v=r['methods'][budget+'__T1'];dg=v['diagnostic']
        budget_rows.append(f"| {budget} | {v['answer_cap']:.1f} | {v['predicted_token_cap']:,.2f} | {v['expected_answers']:.4f} | {v['expected_tokens']:,.2f} | {dg['answer_binding']} | {dg['execution_binding']} |")
    eq_rows=[]
    for method,v in costs['methods'].items():
        eq_rows.append(f"| {method} | {v['expected_gold']:.2f} | {v['expected_cache_recorded_tokens']:,.0f} | {v['expected_joint_budget_fraction']:.4f} |")
    ablation_rows=[]
    for method in ['T1','T1_no_execution_constraint','T1_m0','T0_RawMoment','T1_permuted_moment']:
        v=r['methods']['primary__'+method]
        ablation_rows.append(f"| {method} | {100*v['RMSE']:.3f} | {v['resources']['tokens']['mean']:,.0f} | {100*v['inference']['finite_bernstein']['width']['mean']:.2f} |")
    text_file('05_ABLATIONS_COST_AND_RESEARCH_DECISION.md',f"""
# B1 成本、最小消融与研究判断

## 研究问题与预算设置

完整配对评估184道不同动作问题，预测Token总费用2,172,747.52，缓存记录的完整实际Token总量2,159,964。共同历史投入单列为600道Gold、10,800次成员生成、36,000次聚合和37,274,127 Token。本轮全部为缓存回放，新增物理LLM调用为0。

| 预算档 | Gold上限 | 预测Token上限 | T1期望Gold | T1期望预测Token | 答案约束生效 | 执行约束生效 |
|---|---:|---:|---:|---:|---|---|
{chr(10).join(budget_rows)}

以上为期望约束。每次运行的Gold和Token会波动，超出期望预算的比例均在原始JSON中。预测预算满足不保证实际Token的硬上限。

## 四项机制消融

| 配置 | 主预算RMSE百分点 | 平均缓存记录Token | 保守区间宽度百分点 |
|---|---:|---:|---:|
{chr(10).join(ablation_rows)}

T1_m0对应E[d_i²]，保留完整配对问题簇单位。导师稿件第22页Eq79明确D是L次重复平均，Eq80使用E[D²]，因此T0使用d_i²。单重复分歧率mean_l D_il²是另一个量。该符号核对没有更改已冻结目标。

去掉执行约束未形成可靠优势。m=0减少当前辅助预测带来的误差，但仍不优于均匀主对照。置乱工作二阶矩进一步恶化。结论均来自冻结配置，未依据验证收益调整辅助模型或概率下限。

## 达到共同RMSE的成本

以主档U1精确设计RMSE为共同目标，下面使用冻结的工作模型和采样规则，在独立评价器中反解期望预算。该计算使用完整有限池真值，属于事后条件性效率诊断。

| 方法 | 目标所需期望Gold | 目标所需缓存记录Token | 共同预算比例 |
|---|---:|---:|---:|
{chr(10).join(eq_rows)}

这些数值不提供未来总体的停止规则，也不提供同覆盖率的保证。对相同可靠性，主表同时给出相同保守界的覆盖和宽度。T1保守区间更宽，当前没有可靠性保持下的效率优势。

## 共享执行与费用权限

每题Gold至多一次，三个生成重复下成员数为3乘以两个三模型组合的成员并集大小，两个不同组合合计6次聚合。所有方法拥有相同复用权限。若故意重复共同成员，完整题池Token从2,159,964增至2,573,626，成员调用从2,511增至3,312。这是账本反事实，不是T1的独有优势。

## 科学结论与停止决定

H1中预测辅助降低误差的部分未得到支持，U1的RMSE略高于U0。所有主方法都采用配对反馈，当前表未单独识别配对相对非配对方案的因果优势。

H2未通过。T1在真实有限池上劣于均匀和强公开适配对照，费用异质性较小且辅助残差严重失准。

H3需区分区间。普通Wald的标称覆盖未实现。有效保守界保留覆盖，但宽度限制了判断能力，不能把总是不批准当成有效检出。

保留B1全部结果，当前不进入最多三个候选的第二阶段，不启动fresh400或新LLM生成。合成部分的可检出性与错误批准另见04报告。

## 拓展思考与可能关注的问题

本轮能检验一个没有稳定收益的冻结系统是否被误判为有效。它已经暴露出残差分配失准的代价。下一阶段的价值取决于可验证的辅助校准和真实执行费用差异，当前证据不支持叠加新的分配公式。
""")
    with (BASE/'public/04_EXPERIMENT_MATRIX.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['cell','role','methods','budget','repetitions','independent_unit','status','artifact'])
        writer.writeheader()
        for key,v in r['methods'].items():
            writer.writerow(dict(cell=key,role='main' if v['method'] in ['U0','U1','A1_ASI','A2_Robust_ASI','T0_RawMoment','T1'] else 'ablation',
                methods=v['method'],budget=v['budget'],repetitions=2000,independent_unit='original question cluster; fixed pool design repeats',
                status='COMPLETE',artifact='REAL_FIXED_POOL_RESULTS.json'))
    return audit


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='verify':
        print(json.dumps(verify_main()))
    else:
        print(json.dumps(build_real_reports()))
