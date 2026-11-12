"""Generate E7-C report from actual frozen-path evaluation and audit totals."""
from pathlib import Path
import os
import json
import sys
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
sys.path.insert(0,str(PUBLIC))
from e7_common import load_train,sha,put,splits,resources


def pct(x):
    return f'{100*x:.4f}%'


def pp(x):
    return f'{100*x:+.4f}'


def ci(values):
    return '['+', '.join(pp(x) for x in values)+']'


def check_report_inputs(result,paired,index):
    data,_=load_train()
    checks=0
    maximum_error=0.
    def close(actual,expected,tolerance=1e-12):
        nonlocal checks,maximum_error
        actual=np.asarray(actual,dtype=float)
        expected=np.asarray(expected,dtype=float)
        assert actual.shape==expected.shape,(actual.shape,expected.shape)
        error=float(np.max(abs(actual-expected))) if actual.size else 0.
        maximum_error=max(maximum_error,error)
        assert error<tolerance or error==0,error
        checks+=actual.size
    labels=result['learners']
    methods=result['methods']
    budgets=result['checkpoints']
    score=paired['score']
    perq=score.mean(axis=(2,5))
    close(perq,paired['perq'])
    rows=paired['Q']
    Y=data['Y'][rows]
    for ki,kind in enumerate(labels):
        for mi,method in enumerate(methods):
            for si in range(3):
                for bi,B in enumerate(budgets):
                    action=paired['actions'][ki,mi,si,bi]
                    actual=Y[np.arange(600)[:,None],np.arange(3)[None,:],action[:,None]]
                    close(actual,score[ki,mi,si,bi])
            for bi,B in enumerate(budgets):
                row=result['table'][kind][method][str(B)]
                initial=score[ki,mi,:,0]
                updated=score[ki,mi,:,bi]
                close(row['accuracy'],perq[ki,mi,bi].mean())
                close(row['gain_initial'],(updated-initial).mean())
                close(row['corrected_question_equivalents'],(updated>initial).mean(axis=(0,2)).sum())
                close(row['harmed_question_equivalents'],(updated<initial).mean(axis=(0,2)).sum())
                close(row['net_correction'],600*row['gain_initial'])
                close(row['seed_gain_initial'],(updated-initial).mean(axis=(0,1)))
                close(row['feedback_cost']['gold'],B)
                close(row['feedback_cost']['member_calls'],18*B)
                close(row['feedback_cost']['aggregator_calls'],60*B)
                close(row['centered_error'],paired['centered_error'][ki,mi,:,bi].mean())
                close(row['common_offset_error'],paired['common_offset_error'][ki,mi,:,bi].mean())
                close(row['centered_error_change'],(paired['centered_error'][ki,mi,:,bi]-paired['centered_error'][ki,mi,:,0]).mean())
                task=paired['task'].astype(str)
                for t in ('gsm8k','bbh'):
                    close(row['GSM_BBH'][t],perq[ki,mi,bi,task==t].mean())
                for sub,value in row['BBH_subtasks'].items():
                    mask=(task=='bbh')&(paired['sub'].astype(str)==sub)
                    close(value,perq[ki,mi,bi,mask].mean())
    initial_cost=[]
    for f in range(5):
        r=splits(f)
        initial_cost.append(resources(data,r['I'],[range(20)]*240,gold=240))
    cost_keys=list(initial_cost[0])
    average={key:float(np.mean([cost[key] for cost in initial_cost])) for key in cost_keys}
    for item in index['private_purchase_index']:
        f=item['fold']
        for key in cost_keys:
            close(item['initial_cost'][key],initial_cost[f][key])
    qualification=json.loads((PUBLIC/'c/E7_C_QUALIFICATION.json').read_text())
    assert qualification['status']=='PASS' and qualification['packages']==10
    guard=json.loads((PRIVATE/'c/ALL_PATHS_FROZEN.json').read_text())
    assert guard['status']=='ALL90_FROZEN_BEFORE_EVALUATION' and guard['path_count']==90
    assert result['private_paired_sha256']==sha(PRIVATE/'c/E7_C_PAIRED.npz')
    for item in index['path_metadata']:
        assert item['steps']==60 and item['checkpoints']==[0,20,40,60]
        assert item['E_rewards_used_for_any_purchase'] is False
        assert item['query_evaluation_after_all90_frozen'] is True
    return dict(status='PASS',scalar_checks=checks,maximum_reconstruction_error=maximum_error,
        reward_values_rebuilt=int(score.size),formal_paths=90,snapshots=360,
        default_initial_cost_mean_per_fold=average,
        original_evaluation_questions=600,LLM_repeats_per_question=3,algorithm_repeats=3,
        output_roles='Report reconstruction, frozen policy predictions and purchased-feedback metrics, no acquisition changes')


def main():
    source=PUBLIC/'c/E7_C_RESULTS.json'
    result=json.loads(source.read_text())
    assert result['status']=='COMPLETE'
    with np.load(PRIVATE/'c/E7_C_PAIRED.npz') as a:
        paired={name:a[name] for name in a.files}
    index=json.loads((PRIVATE/'c/E7_C_PATH_COST_AND_ACCESS_INDEX.json').read_text())
    audit=check_report_inputs(result,paired,index)
    audit['results_sha256']=sha(source)
    put(PUBLIC/'c/E7_C_REPORT_AUDIT.json',audit)
    labels={'P1':'P1，原P2 PCA16 Ridge','P3':'P3，RBF核岭'}
    model_rows=[]
    curve_rows=[]
    cost_rows=[]
    task_rows=[]
    seed_rows=[]
    CPU_rows=[]
    per_fold_rows=[]
    for kind in result['learners']:
        for method in result['methods']:
            for B in result['checkpoints']:
                row=result['table'][kind][method][str(B)]
                curve_rows.append(f"| {kind} | {method} | {B} | {pct(row['accuracy'])} | {pp(row['gain_initial'])} {ci(row['gain_initial_ci95'])} | {row['centered_error']:.6f} | {row['common_offset_error']:.6f} |")
                cost=row['feedback_cost']
                cost_rows.append(f"| {kind} | {method} | {B} | {cost['member_calls']:.0f} | {cost['aggregator_calls']:.0f} | {cost['tokens']:.1f} | {row['E_service_cost']['tokens']:.1f} | {row['feedback_plus_E_service_cost']['tokens']:.1f} |")
            row=result['table'][kind][method]['60']
            model_rows.append(f"| {kind} | {method} | {pct(row['accuracy'])} | {pp(row['gain_initial'])} {ci(row['gain_initial_ci95'])} | {pp(row['gain_I_fixed'])} | {row['corrected_question_equivalents']:.3f} | {row['harmed_question_equivalents']:.3f} | {row['changed_decisions']:.3f} | {row['centered_error_change']:+.6f} |")
            task_rows.append(f"| {kind} | {method} | {pct(row['GSM_BBH']['gsm8k'])} | {pct(row['GSM_BBH']['bbh'])} | {pct(row['GSM_BBH_task_macro'])} | {pct(row['GSM_BBH27_macro'])} |")
            seed_rows.append(f"| {kind} | {method} | "+' | '.join(pp(x) for x in row['seed_gain_initial'])+' |')
            CPU_rows.append(f"| {kind} | {method} | {row['full_path_planning_CPU_mean']:.6f} | {row['full_path_update_CPU_mean']:.6f} | {row['full_path_wall_mean']:.6f} |")
        for entry in result['per_fold']:
            if entry['learner']!=kind:
                continue
            values=np.asarray(entry['accuracy'])
            initial=values[0,0]
            per_fold_rows.append(f"| {kind} | {entry['fold']} | {pct(initial)} | {pct(values[0,-1])} | {pct(values[1,-1])} | {pct(values[2,-1])} | {pp(values[2,-1]-values[0,-1])} |")
    compare_rows=[]
    for kind in result['learners']:
        for B in result['checkpoints']:
            for label in ('DA-Random','DA-IG'):
                row=result['paired_comparisons'][f'{kind}:{label}@{B}']
                compare_rows.append(f"| {kind} | {B} | {label} | {pp(row['point'])} | {ci(row['ci95'])} | "+', '.join(pp(x) for x in row['seed_direction'])+' |')
    sub_sections=[]
    for kind in result['learners']:
        subrows=[]
        sub=paired['sub'].astype(str)
        task=paired['task'].astype(str)
        for name in sorted(result['table'][kind]['Random']['60']['BBH_subtasks']):
            mask=(task=='bbh')&(sub==name)
            values=[]
            for method in result['methods']:
                updated=result['table'][kind][method]['60']['BBH_subtasks'][name]
                initial=result['table'][kind][method]['0']['BBH_subtasks'][name]
                values.append(f"{pct(updated)} ({pp(updated-initial)})")
            subrows.append(f"| {name} | {mask.sum()} | "+' | '.join(values)+' |')
        sub_sections.append(f'''### {labels[kind]}

| BBH子任务 | 原题数 | Random准确率及增益，百分点 | IG准确率及增益，百分点 | DA准确率及增益，百分点 |
|---|---:|---|---|---|
{chr(10).join(subrows)}
''')
    icost=audit['default_initial_cost_mean_per_fold']
    primary=result['paired_comparisons']['P3:DA-Random@60']
    primaryIG=result['paired_comparisons']['P3:DA-IG@60']
    positive_learning=[]
    for kind in result['learners']:
        for method in result['methods']:
            row=result['table'][kind][method]['60']
            if row['gain_initial_ci95'][0]>0:
                positive_learning.append(f'{kind}/{method}')
    text=f'''# E7-C 固定学习器的主动反馈学习

## 研究问题

在固定预测器、特征、更新单位与初始化后，购买不同问题的完整组合反馈能否改善后续未见E题的选队，Decision-Aware是否优于Random和Parameter-IG。

## 实验设置

沿用E7的五折I240/U240/E120。I初始化后，采集器读取U题目输入，每步购买一道此前未购买问题的完整20组合×3生成评分，再更新同一个组合价值预测器。预测器随后直接选择其20个组合中的最高预测分。每次回答执行三名成员。

共2学习器×3采集×5折×3算法种子=90条正式回放路径。算法种子17、29、43，生成种子42、137、2024为题内重复。每条路径保存0、20、40、60个反馈包的模型快照，共360份。所有90条路径的动作和预测在评价器读取E奖励前冻结。E区600道原始题在外折间互斥，外折历史有重叠。

本报告P1是E7-B的P1，即历史命名P2的PCA16直接20组合Ridge。P3是E7-B的完整MiniLM RBF核岭。P1沿用原Ridge正则并正确换算三重复均值单位。P3的均值偏置、带宽和正则从I固定，后续更新保持原单位。

两个学习器均使用固定的Gaussian工作后验，20个输出在该工作模型下独立，实际同题组合相关性没有因此消失。均值、协方差、真实更新和假想更新使用同一套核与噪声。后验不确定性没有经过真实正确概率的校准。

## 采集对照

- Random从剩余U题中随机购买一个包。
- Parameter-IG对20输出完整包使用10 log(1+v/σ²)。
- Decision-Aware使用E3的未来决策价值定义，在共享I区参考输入上计算假想反馈更新后的加权最大预测价值增量。使用64个20维反向配对Gaussian幻想样本，候选与参考共享随机数。

参考集合为I中27道GSM题和27个BBH子任务各1道题。GSM总权重0.5，BBH总权重0.5，BBH子任务按I频率加权。参考输入与权重在全部采集对照中共享。采集评分及选择在当前付费反馈揭示前保存，真实输出长度在购买后进入账本。

资格检查在外折0上对P3的三种采集各购买10包，验证顺序更新与整体重拟合，拒绝E题反馈请求。学习器级软件核验见E7_B_MODEL_QUALIFICATION.json。资格路径与90条正式路径分开保存。

## 主预算60的数值结果

I240确定的固定组合在600道E题上准确率为{pct(result['default_accuracy'])}。表中纠错、破坏为三算法种子和三生成种子平均后的原题等价数，净纠错除以600等于相对初始化收益。

| 学习器 | 采集 | E准确率 | 相对初始化及95%CI，百分点 | 相对I固定组合，百分点 | 纠错 | 破坏 | 最终动作变化题数 | 中心化误差变化 |
|---|---|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(model_rows)}

## 完整学习曲线

| 学习器 | 采集 | Gold | E准确率 | 相对初始化及95%CI，百分点 | 中心化误差 | 共同偏移误差 |
|---|---|---:|---:|---|---:|---:|
{chr(10).join(curve_rows)}

原始三生成奖励在题内平均后用于价值预测误差。误差分解相对于20组合真实三重复均值，中心化误差和共同偏移误差的和为原绝对均方误差。训练区拟合误差没有代替E区服务质量。

## 采集机制配对比较

以下为题级95%区间，E7四项主比较的统一Holm与Bonferroni区间由04_COST_AND_STATISTICS.md和主结果JSON提供。

| 学习器 | Gold | 比较 | 准确率差，百分点 | 95%CI | 三生成种子差值方向，百分点 |
|---|---:|---|---:|---|---|
{chr(10).join(compare_rows)}

主P3、Gold60时，DA减Random为{pp(primary['point'])}个百分点，95%CI为{ci(primary['ci95'])}。DA减IG为{pp(primaryIG['point'])}个百分点，95%CI为{ci(primaryIG['ci95'])}。

## 分任务与生成重复

主权重为GSM8K和BBH各50%，BBH内部按原题平均。BBH27子任务宏平均另列，其权重与主指标不同。

| 学习器 | 采集 | GSM准确率 | BBH准确率 | 任务50/50宏平均 | GSM/BBH27宏平均 |
|---|---|---:|---:|---:|---:|
{chr(10).join(task_rows)}

下表为Gold60相对同学习器初始化的增益，各列保留同题生成重复，不能当成三批独立问题。

| 学习器 | 采集 | seed42，百分点 | seed137，百分点 | seed2024，百分点 |
|---|---|---:|---:|---:|
{chr(10).join(seed_rows)}

### 外折方向

| 学习器 | 外折 | 初始化 | Random60 | IG60 | DA60 | DA减Random，百分点 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_fold_rows)}

## BBH全部子任务

表中括号为Gold60相对同方法初始化的增益。每个原题的三生成共同保留。

{chr(10).join(sub_sections)}

## 完整执行成本

每个问题包只揭示一次Gold，三生成下六个成员输出共18次生成、20组合共60次聚合及60次评分。初始I240全部方法共享，平均每外折Gold240、成员{icost['member_calls']:.0f}、聚合{icost['aggregator_calls']:.0f}、Token{icost['tokens']:.1f}。该初始历史成本与新购反馈分列，各方法重复回放没有重新产生物理模型调用。

下表为平均每外折／路径源成本。E服务列为最终所选组合在E120上的三生成服务成本。Gold和调用数相同，实际Token不同。两者分别列示。

| 学习器 | 采集 | Gold | 反馈成员调用 | 反馈聚合调用 | 反馈Token | E服务Token | 反馈＋E服务Token |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(cost_rows)}

这轮完整反馈包的执行成本较高，研究问题聚焦于购买哪些问题。当前结果仅覆盖完整包的池式反馈学习，减少每题执行组合数的联合问题仍需另行实验。全部结果来自已有缓存，新增物理LLM调用为0。源模型wall-time作为历史执行代理保存，GPU kernel时间和原始编码耗时没有被重建。

## 采集与更新CPU

下表为完整60步路径的平均CPU秒及墙钟秒，含不同算法种子和外折。Windows process_time的计时粒度可令短操作累计为0，这种0值表示计时器未分辨到CPU时间。CPU列与墙钟列共同报告。

| 学习器 | 采集 | 规划CPU秒 | 更新CPU秒 | 路径墙钟秒 |
|---|---|---:|---:|---:|
{chr(10).join(CPU_rows)}

## 统计结论

全部区间按原始问题配对抽样5000次，分层为外折×GSM/BBH。同题三个生成重复、算法种子及全部对照共同保留。算法种子用于稳健性检查，独立题数仍为600。当前单项95%区间明确支持相对初始化正收益的学习器／采集配置为{('、'.join(positive_learning) if positive_learning else '无')}。采集机制的正式判断结合E7统一四项校正结果。

## 失败原因与边界

随机反馈学习和主动采集的附加效应分开解释。若新反馈没有提高E服务质量，不能仅根据方差下降或中心化误差改善宣称机制有效。Decision-Aware优化的是Gaussian工作模型上的参考集价值代理，参考分布、独立输出假设、核和线性表示的失配都可能削弱它与真实纠错的关系。这些因素属于解释范围，单次回放没有识别各因素的独立因果贡献。

90条路径的完整购买序列、合法反馈、前瞻性采集分数、模型快照、应用级奖励访问记录和源成本账本保存在private/c。入口为E7_C_PATH_COST_AND_ACCESS_INDEX.json。应用级访问隔离已经通过合法购买边界与先冻结后评价实现，未声称操作系统级访问取证。

## 优势与风险

优势是同一学习器下只改变采集规则、真实组合评分、完整包付费、顺序更新与重拟合对齐、全部路径冻结后统一E评价。风险是旧开发数据复用、外折历史重叠、仅三个算法种子、Gaussian工作模型失配、参考输入有限以及完整20组合反馈的高执行开销。

## 下一步

后续研究决策应同时读取E7-A的候选机会和E7-B的独立题学习曲线。当前学习器及采集结果按原协议完整保留，下一轮新增数据或缩减执行组合需要单独冻结计划。
'''
    (PUBLIC/'03_ACTIVE_FEEDBACK_LEARNING.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',report='03_ACTIVE_FEEDBACK_LEARNING.md',audit=audit,
        primary=primary,primary_IG=primaryIG),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
