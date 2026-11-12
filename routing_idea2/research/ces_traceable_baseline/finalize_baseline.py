"""Traceable CES evidence chain, limits and six-model transfer results."""
from ces_common import *


def pp(value):
    return f'{100*value:+.4f}'


def ci(values):
    return '['+', '.join(pp(x) for x in values)+']'


def main():
    result=json.loads((PUBLIC/'MIGRATION_RESULTS.json').read_text())
    native=json.loads((PUBLIC/'NATIVE_AUDIT.json').read_text())
    assets=json.loads((PUBLIC/'CES_PUBLIC_ASSET_AUDIT.json').read_text())
    protocol=json.loads((PUBLIC/'CES_PROTOCOL.json').read_text())
    labels={'ProfileOnly-SingleCombo':'仅能力先验＋单组合反馈',
            'HistoryFit-SingleCombo':'完整历史预训练＋单组合反馈',
            'HistoryFit-Full20':'完整历史预训练＋20组合反馈'}
    table=[]
    curves=[]
    for track in result['tracks']:
        for mode in ['UCB','MeanOnlyDiagnostic']:
            row=result['table'][track][mode]['60']
            history=row['initial_cost']
            budget=row['feedback_cost']
            table.append(f"| {labels[track]} | {mode} | {100*row['accuracy']:.4f}% | {pp(row['gain_own_initial']['point'])} {ci(row['gain_own_initial']['ci95'])} | {budget['member_calls']:.0f}/{budget['aggregator_calls']:.0f} | {budget['tokens']:.1f} | {history['tokens']:.1f} |")
            for B in [0,20,40,60]:
                r=result['table'][track][mode][str(B)]
                curves.append(f"| {labels[track]} | {mode} | {B} | {100*r['accuracy']:.4f}% | {r['feedback_cost']['tokens']:.1f} | {r['complete_system_cost']['tokens']:.1f} |")
    stats=[]
    for record in result['primary_comparisons']:
        stats.append(f"| {record['name']} | {pp(record['point'])} | {ci(record['ci95'])} | {record['Holm_p']:.6f} |")
    text=f'''# CES六模型缓存迁移结果与完整成本

## 研究问题

验证公开CES作者代码在六个冻结语言模型、固定K=3和真实聚合奖励下的表现。原算法算子、信息输入和执行费用均有独立来源记录。原论文的在线主表成绩当前无法由公开材料原数复算，详见01和02。

## 实验设置

继承冻结E7的五折I240/U240/E120，只使用train600开发数据。15个fold×算法种子生成特征，每条轨道15路径，共45条。0/20/40/60保存180个模型快照。全部购买和E动作先冻结，再由评价器读取真实E奖励。

选择与更新直接调用作者`LinUCBSharedCombinatorial`，参数alpha3.5、ridge0.1，build_phi的beta和gamma均3，d_x32，总维96。保留原集合UCB边际贪心，K固定3，cost penalty0。主服务规则是原代码UCB；alpha0仅作为预先固定的均值服务诊断，未替换主比较。

问题输入使用既有MiniLM-L6向量及原8维手工通道，明确替换官方MiniLM-L12编码。能力先验由I240真实成员评分形成GSM/BBH两项及历史Token费用，共3维。原SVD初始化及raw-profile应用直接保留。每个最终组合三生成均值作为一次原update观测。I预训练每题20个均值标签，共4800次更新；这与官方MCQ theta0初始化不同。

## 信息与反馈对照

三条轨道分别保留原式theta0能力先验、提供完整I组合历史后的单组合反馈、以及同I历史加完整20组合反馈。所有U题顺序精确使用冻结E7 Random的相同fold/seed购买安排。本轮没有引入新的采集公式。

完整包轨道与E7 Random具有相同I/U原题及最终组合评分监督。CES另外使用I成员正确性profile。E7原学习器未接收该侧信息，因此这是一项明确记录的完整系统比较，无法单独归因到学习器更新或服务规则。额外成员评分计4320次，I的Gold和成员生成可以复用。

## 数值结果

I全局固定组合在E上的准确率85.0000%。表中区间为相对自身初始化的原题配对95%区间，尚未对所有探索轨道校正。成本是平均每外折/算法种子的反事实源执行费用，包含全部三个生成重复。

| 历史与反馈权限 | 服务规则 | Gold60准确率 | 相对自身初始，百分点及95%CI | U成员/聚合调用 | U实际Token | I历史Token |
|---|---|---|---|---|---|---|
{chr(10).join(table)}

仅能力先验的I准备为4320次成员调用、0聚合、240Gold和4320成员评分。完整历史另需14400聚合与评分，Gold和成员输出复用。单组合U60为540成员、180聚合，完整包为1080成员、3600聚合。两者同Gold但执行开销不同。

## 与冻结E7对照

Gold60时，E7 Random PCA16 Ridge为86.2593%，E7 Random核岭为85.4259%。CES完整历史/完整包的原UCB为85.8519%，均值诊断为86.0556%。主比较预先固定为原UCB对两种E7 Random学习器，统一Holm2。

| 主比较 | 差值，百分点 | 原题95%CI | Holm双侧p |
|---|---|---|---|
{chr(10).join(stats)}

两项均未建立稳定优劣。均值诊断的点估计不能事后替换主服务规则。全部GSM/BBH、27子任务、外折和生成种子结果及中心化误差保存在MIGRATION_RESULTS.json。

## 质量与成本曲线

末列包含I历史、U反馈和所选规则的E120服务。历史只计一次，跨折重复和不同轨道属于反事实比较，本轮物理LLM调用为0。

| 权限轨道 | 服务 | U Gold | E准确率 | U Token | I＋U＋E服务Token |
|---|---|---|---|---|---|
{chr(10).join(curves)}

## 统计结论

600道原始E题各出现一次，三个生成和三个算法重复共同留在题内。5000次bootstrap按外折×GSM/BBH分层，主权重各50%，BBH内按题平均，27子任务宏平均另列。外折历史重叠、旧开发数据复用和模型侧信息差异均限制因果解释。

## 失败原因与边界

原代码可运行并具有真实反馈更新，当前六模型迁移也没有出现稳定的后续收益。CES的集合均值是成员特征和，其20组合设计秩最多6、中心化秩最多5，不能表示任意组合交互。但当前比较不足以确定表示限制、先验替换、编码变化、UCB服务、任务本身或样本量分别贡献了多少。

单组合反馈在完整历史下的原UCB从85.7222%到85.7593%，增加完整包后为85.8519%。执行量增加没有对应稳定质量证据。原theta0轨道起始UCB85.9444%，60题后85.3148%，区间跨零，也不支持将反馈自动视为改善。

## 优势与风险

优势是直接作者代码、完整源哈希、真实聚合评分、合法I先验、逐包账本以及E7不变。风险包括原论文/代码差异、原画像与轨迹缺口、迁移的额外侧信息、模型池/编码/聚合器变化和有限开发样本。当前结果支持一个可复核的作者代码迁移实例。

## 下一步

先补足原协议逐题轨迹、模型版本、运行配置和评价日志，或在明确修订与费用授权后规划小规模原协议运行。现有六模型结果可继续用于定位表示与服务规则，任何新增消融应另行冻结。本轮不将不显著结果外推为组合路由没有研究价值。
'''
    (PUBLIC/'04_MIGRATION_RESULTS_AND_COST.md').write_text(text,encoding='utf-8')
    decision='''# CES可追溯基线证据链与结论

## 公开方法原协议表现

已核对原论文表3—5、官方master、正式归档和默认runner。作者报告CES的Alpaca LC63.12%、MT平均8.94、ARC97.5%。公开仓库和归档缺少CES原始逐题生成、选择、反馈、裁判及usage轨迹，当前原论文主表成绩尚未独立复现。

已实际执行作者数学核心，重构原选择、集合更新和密集解，并重新计算公开能力profile的聚合算术。15个模型中14个算术一致，1个profile的correct为非整数且与acc字段不一致。现有证据无法恢复其原始预测，原文件保留。

## 它获得的数据、特征与反馈

原MCQ使用15模型、10项MMLU能力加费用先验、多语MiniLM-L12与手工复杂度、共享集合参数及最终聚合评分。已发布profile来自MMLU test且缺少题级ID，无法核对与MMLU runner测题的重叠情况。

论文与代码在选队、特征缩放、先验协方差和默认K上存在差异。本轮忠实保留官方代码作为迁移来源，未静默改成纸面算法。FLASK runner的profile缺口会导致原样初始化失败，445/534是解决该前提后、89题的名义在线调用量。

## 原协议调用预算

下列为冷缓存、无失败、单次runner在线部分的逻辑模型调用。最终独立评价、离线profile、重复种子和重试另外计量。

| 任务 | 题数 | README K2 | paper K3 |
|---|---|---:|---:|
| ARC MCQ示例 | 200 | 600 | 800 |
| AlpacaEval | 805 | 4025 | 4830 |
| MT-Bench | 80题×2轮 | 720 | 880 |
| FLASK-Hard bundled | 89 | 445 | 534 |

原公开能力profile对应约8550个模型—题目槽位，缺少真实调用日志。按README每subject20题重新准备会名义需要17100次成员调用，并改变已发布先验。最终Alpaca的805项、MT的160项和FLASK的89项评价还需按具体批处理计费。原usage缺失，美元与完整Token预算不能可靠还原。[官方固定版本](https://github.com/JinkunXu/CES/tree/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7)

## 六模型真实聚合评分迁移

已完成45条零API缓存路径，保留6模型、K3、20组合、Qwen3聚合器和三个生成重复。直接复用原作者build_phi、集合UCB贪心与update。替换项包括模型池、编码器、能力profile、聚合器与题均值反馈，均列入冻结协议。

同I/U题和相同完整组合评分包下，CES原UCB Gold60为85.8519%，冻结E7 Random Ridge86.2593%、Random核岭85.4259%。两项主差的95%区间均跨零，Holm p均1。CES额外使用I成员级能力先验，完整输入权限仍有差异。均值服务诊断86.0556%属于预先固定的辅助结果。

## 对当前瓶颈的判断

**公开作者代码已被追溯并真实运行，原论文主表的独立复现证据仍有缺口。六模型迁移没有展示稳定超越，也没有建立有益反馈更新。**

这表明弱反馈响应在本轮公开基线迁移中也出现，当前现象不限于Decision-Aware公式。现有结果仍无法唯一归因于任务设置或自己的学习器。缺少成功的原协议端到端复现，加上输入先验、函数表示和服务规则变化，使这种归因尚不闭合。

后续优先补原运行的可追溯材料，并围绕已确认差异设计最小、冻结的对照。扩大在线API调用量不能替代原始轨迹、权限和统计口径的核对。当前Decision-Aware和E7结论按哈希冻结。

## 优势与风险

优势是证据链包含论文、官方commit、完整归档、原算子执行、真实缓存迁移与实际成本。风险是原始结果缺失、profile异常、纸面与代码差异、迁移权限变化和旧开发题复用。源码/算术/已保存数值的置信度为高，原论文主成绩独立复现尚未完成。
'''
    (PUBLIC/'05_EVIDENCE_CHAIN_AND_NEXT_STEPS.md').write_text(decision,encoding='utf-8')
    complete=dict(status='COMPLETE_FOR_AUDIT_AND_CACHE_TRANSFER',native_benchmark='NOT_REPRODUCED_WITH_RELEASED_ARTIFACTS',
        original_source_operators='EXECUTED_AND_INDEPENDENTLY_VERIFIED',migration='COMPLETE',
        paper_reported_results=native['reported_results'],main_migration_comparisons=result['primary_comparisons'],
        frozen_baseline='DecisionAware and E7',E7_freeze_sha256=sha(PUBLIC/'E7_FREEZE_RECEIPT.json'),
        protocol_sha256=sha(PUBLIC/'CES_PROTOCOL.json'),formal_paths=45,model_snapshots=180,
        extra_CES_I_side_information=True,physical_new_model_API_calls=0,new_acquisition_formula=False,
        scope='Existing train600, controlled author-code port, no fresh/sealed evaluation',
        reports=['01_NATIVE_REPRODUCIBILITY_AUDIT.md','02_DATA_FEEDBACK_AND_COST_PERMISSIONS.md',
            '03_ADAPTATION_CONTRACT_AND_OPERATOR_AUDIT.md','04_MIGRATION_RESULTS_AND_COST.md','05_EVIDENCE_CHAIN_AND_NEXT_STEPS.md','REPRODUCE.md'],
        native_assets_status=assets['status'],migration_result_sha256=sha(PUBLIC/'MIGRATION_RESULTS.json'),
        unresolved=['Original generated/graded/usage trajectories missing','Published profile one-row inconsistency',
            'Paper-code behavior/configuration discrepancies','FLASK runner profile-key coverage gap'])
    put(PUBLIC/'CES_COMPLETE_RESULTS.json',complete)
    print(json.dumps(dict(status=complete['status'],native=complete['native_benchmark'],
        transfer_primary=complete['main_migration_comparisons']),ensure_ascii=False))


if __name__=='__main__':
    main()
