# 论文结论与证据

## 研究问题

在有限生成重复产生经验并列奖励的语言模型路由中，训练监督怎样通过共享参数改变其他问题的动作排序？预测近并列时的部署规则又怎样影响这些变化对应的最终答案质量？

当前论文主线为 **E17–E23**。本文引用的 E17–E22 文件来自 E22 提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`，E23 文件来自提交 `3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`。全部实验的各自版本见 [全部实验](../EXPERIMENTS.md)，研究方向的变化见 [研究历史](../docs/HISTORY.md)。P2 是原始独立评价；E23 在结果已知的同一 2,400 题上进行补充比较。

## 定义与评价方式

- **经验全并列行 T**：对应训练区中，有限次生成得到的所有动作经验平均奖励完全相同的题目。并列指这些观测结果，动作的真实期望奖励仍可能不同。
- **B/C 干预**：B 保留 T 的损失，C 将其权重设为零；其他训练行、归一化分母、表示、函数类和正则保持不变。
- **严格选择变化**：在同一预定解码规则下，两边均有唯一最高分动作，且所选动作不同。选择变化和答案质量分别计量。
- **原选择规则**：在距最大分不超过 1e−8 的动作间均匀分配，按预先确定的动作概率计算期望准确率。
- **raw-exact**：对已有浮点分数使用零容差最大集合。这是解码敏感性分析，数值仍来自浮点预测。
- **独立题数**：按原始问题计数。同题的多次生成以及不同动作、温度、掩码和重编号均按同一道题计数，各评测集分别报告。
- **外部固定基线**：E21 在 GPQA 与 MATH-500 各自训练区选择固定模型，再等权汇总测试准确率，结果为 62.194%。
- **E23 训练先验选择规则**：在原近并列候选中，选择 train600 的三个生成重复中整数成功数最高的组合；成功数仍并列时均匀选择。B_prior 与 B_uniform 共用预测头，C_prior 与 C_uniform 的全部动作相同。
- **E23 MatchedMean**：三个预定匹配控制策略的等权动作概率混合，各策略用全部 600 题训练。控制沿用 E22 的任务/子任务×特征范数分箱规则与种子，使用 `final600` 作为掩码生成的固定标记；E22 则使用五折交叉验证的留出预测。涉及均匀策略或混合策略的纠错、破坏按耦合期望计数。

## 结论与证据

| 编号 | 结论 | 结果 | 来源 | 范围 |
|---|---|---|---|---|
| C1 | 按编号选择经验奖励最高的动作，会改变当前 SPO+ 的组合选择 | 100 次重编号后，按编号处理并列的 SPO+ 实际组合选择一致率均值约 6.13%，Ridge 与均匀处理并列的 TSPO+ 为 100% | [E17 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/EXPERIMENT_SUMMARY.md) · [重编号报告](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/02_RANDOM_RENUMBERING.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/E17_COMPLETE_RESULTS.json) | 原 600 题开发集；计量实际组合选择的稳定性 |
| C2 | 对奖励最高的并列动作采用均匀监督，独立评价的准确率差接近零 | 2,998 题上 TSPO_uniform 对 SPO_coordinate 为 +0.011749 个百分点，377 次纠错、376 次破坏 | [E18 结论](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e18_confirmatory_execution/final_results/E18_RESEARCH_CONCLUSION.md) · [准确率与配对结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json) | 两道不可评分题共同排除；比较训练中并列最优动作的处理方式 |
| C3 | 有限次生成可使期望奖励不同的动作呈现经验并列；对称约束也可通过共享线性预测器改变其他输入的唯一最高分动作 | 已有有限重复构造、三输入凸构造和数值实现检查；三输入例子的阈值为 β=46/51 | [E19 理论与预测](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/01_THEORY_AND_PREDICTIONS.md) · [P0 实现检查](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/02_P0_QUALIFICATION_REPORT.md) | 结论基于理论构造及其假设；46/51 是人工构造例子中的阈值 |
| C4 | 在原系统固定表示和正则下，关闭全并列行约束能改变留出题目的唯一最高分动作 | 600 题中心化分数均变化，203 题的 B/C 均有唯一最高分动作，其中 150 题选择不同；这些选择变化的记录中纠错 38、破坏 19 | [E19 P1](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/03_P1_FIVE_FOLD_RESULTS.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json) | 原 600 题的五折交叉验证留出预测，该题集长期用于开发 |
| C5 | 原系统冻结 B/C 策略在独立新题上，C 的准确率高于 B | 2,400 题，B 85.355%、C 86.681%、C−B +1.326 个百分点；GSM8K +0.248、BBH +2.404 | [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) · [准确率与配对结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json) · [诊断](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_DIAGNOSTICS.json) | 一套六模型、三成员聚合配置；原冻结策略包含预定选择规则，E23 另作部署规则比较 |
| C6 | P2 中出现严格选择变化，准确率差的大小随近并列处理规则变化 | 严格选择变化 302 题、整数纠错 75/破坏 56；raw-exact 准确率差 +0.639 个百分点，小于主规则 +1.326 | [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) · [动作概率诊断](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_DIAGNOSTICS.json) | 总准确率差包含发生严格选择变化的题目和预测近并列的题目所作的贡献；raw-exact 是浮点敏感性诊断 |
| C7 | 另一损失和外部单模型选择中仍出现严格选择变化，准确率差的方向不一致 | 原题集 ListNet 严格选择变化 135 题、准确率差 −0.278；外部 L=3 TSPO+/ListNet 严格选择变化 113/41 题、准确率差 +0.145/−0.202 个百分点 | [P3 报告](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_RESULTS.md) · [原题集 ListNet](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json) · [外部结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json) | 外部 548 题、六模型单选、两任务；复用 DARS 固定评分缓存 |
| C8 | 外部主要比较中，C 的准确率低于按任务在训练区选择的固定模型 | 按任务固定基线等权 62.194%，TSPO+ C 57.933%、ListNet C 61.255% | [P3 外部结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json) · [任务内拟合和固定基线源码](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/run_external.py) | 按任务分别选择后再平均，固定基线的选择发生在各自训练区 |
| C9 | E22 开发集中，一般训练行降权也会引起严格选择变化；全并列行降权对匹配控制的准确率差随损失变化 | TSPO+ C 对同总权重 Uniform 为 +1.343，对三掩码平均为 +0.352；ListNet 分别为 −0.278、−0.019 个百分点 | [E22 审计](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/AUDIT_REPORT.md) · [全部控制](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/controls.csv) · [汇总](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/SUMMARY.json) | 原 600 题开发集，报告全部三个预定掩码及其均值；与 E23 用全部 600 题训练的预测头分别计量 |
| C10 | B→C 的参数和得分变化超出共同平移与统一正比例缩放，并能在固定凸目标中作梯度分解 | TSPO+ 最佳正比例残差占 C 范数 76.0%–87.4%，ListNet 23.7%–34.4%；十组损失与交叉验证折的组合通过残差修正几何检查 | [E22 几何](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/GEOMETRY.md) · [几何实现](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/geometry.py) · [几何表](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/geometry.csv) | 固定凸目标下的参数几何；合法次梯度分量可能不唯一，且分量间存在抵消 |
| C11 | 原系统独立评价有完整调用和 token 记录 | P2 成员 41,289 次、聚合 106,536 次，合计 147,825 次，116,097,715 tokens | [P2 执行账本](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_EXECUTION_LEDGER.json) · [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) | 覆盖原冻结策略全部候选动作的实验评价成本；单策略部署成本需按其执行路径计量 |
| C12 | 只修改 B 的部署规则，即可缩小原 P2 的大部分 B/C 准确率差 | B_prior 86.50000%，C 86.68056%；B_prior−B_uniform +1.14542，C−B_prior +0.18056 个百分点 | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`baseline_strategies`、`B_deployment_gain_pp`、`remaining_gap_pp` · [先验选择与冻结实现](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/analyze_references.py)：`train_prior`、`freeze` | 只改变 B 的 2,048 道近并列题；先验来自 train600，原模型权重及候选集合保持固定 |
| C13 | 改用训练先验后，原严格选择变化题目的贡献仍为正，原近并列题目的贡献转为负 | 发生严格选择变化的 302 道题贡献 +0.26389 个百分点，纠错 75、破坏 56；原 B 近并列区贡献 −0.08333，合计 +0.18056 | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`paired_comparisons.C_prior_minus_B_prior.regions` · [完整分解](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/EXPERIMENT_SUMMARY.md) | 区域贡献均以全 2,400 题为分母；纠错、破坏对应每题三个生成中的记录事件 |
| C14 | 在 E23 固定 TSPO+ 训练条件下，C 的准确率高于同总权重 Uniform 和三份匹配控制 | C−Uniform +1.37173；C 对三个 Matched 分别 +0.56944、+0.54167、+0.45833；对等权混合 +0.52315 个百分点，混合比较在两任务和三个 seed 上均为正 | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`matching`、`paired_comparisons` · [控制协议](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json) · [混合与评分实现](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/finalize_results.py)：`summarize` | 原固定正则、三个预定掩码、四个用全部 600 题新训练的预测头；与 E22 跨损失五折结果共同解释 |
| C15 | E23 在同一评测集比较的各方法中，Ridge 总体准确率最高 | Ridge 86.87500%，C 86.68056%，GlobalFixed 86.48611%；C 对二者分别 −0.19444、+0.19444 个百分点 | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`baseline_strategies`、`paired_comparisons` · [对照方法恢复实现](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/analyze_references.py)：`freeze` | Ridge 使用 E17 原先用全部 600 题训练的预测头及对应预处理；GlobalFixed 按训练成功数选组合 5；各方法沿用各自冻结超参数 |
| C16 | E23 十二种冻结策略所需的调用已补齐，成本记录完整 | 新增成员 1,770 次、聚合 4,263 次，共 6,033 次调用、4,486,492 tokens；未知或失败尝试 0；原 P2 与补充累计 120,584,207 tokens | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`native_supplement_cost`、`combined_execution_cost` · [原生收据核验与汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/finalize_results.py) | 复用原 P2 评测集与已付费缓存，额外购买答案账户 0；费用按十二种策略需要的调用去重统计 |

## 理论与实验的联系

全并列行在当前两种目标中具有不同形式。TSPO+ 的逐行项为 `2 × (max(q) − mean(q))`；单位温度 ListNet 的逐行项为 `logsumexp(q) − mean(q) − log(A)`。二者对动作编号对称，同时约束动作分差。经验全并列可能来自真实均值接近，也可能来自有限重复；当前真实数据中，两种来源的占比尚未区分。[理论构造](../routing_idea2/research/e19_symmetric_tie_constraint/01_THEORY_AND_PREDICTIONS.md)与[参数几何](../routing_idea2/research/e22_mechanism_specificity/GEOMETRY.md)分别说明损失结构和共享参数中的传递。

论文围绕五个问题展开：经验并列的来源与损失结构；固定表示、函数类和正则下的严格选择变化；原系统独立质量结果；部署规则对同一预测头的影响；跨损失、外部模型池和训练行控制。E23 补充部署规则的比较：原 P2 的 +1.32598 个百分点可分解为 B 训练先验相对原均匀规则的 +1.14542，以及 C 相对 B 训练先验的 +0.18056。该分解描述这组冻结策略的准确率差，后者仍随任务和生成 seed 改变方向。

E17/E18 研究训练中并列最优动作的处理方式，E19–E23 研究全并列约束强度及其后续部署。E18 的接近零结果、P2 原冻结策略的正准确率差、P3 跨损失和外部评测集的不同结果、E22 的开发控制，以及 E23 在原评测集上的补充比较各自对应不同问题。

## E23 对论文判断的更新

| 判断 | E23 增加的证据 | 写作调整 |
|---|---|---|
| 训练约束与部署规则各贡献多少 | 同一 B 预测头用训练先验使准确率提高 1.14542 个百分点；C−B_prior 仅 +0.18056 | 将原 +1.32598 定义为完整策略差，分别展示部署对照和发生严格选择变化的题目贡献 |
| 全并列定向降权相对一般降权的表现 | C 对三份预定匹配控制均为正，对混合 +0.52315；E22 的 ListNet 结果仍为 −0.019 | 增加 TSPO+、固定正则及本轮对照下的准确率结果，保留跨损失差异 |
| 作为完整路由算法的性能排序 | Ridge 86.87500% 高于 C 86.68056%，GlobalFixed 为 86.48611% | 论文贡献聚焦训练与部署机制；完整性能表同时列出这些对照方法 |
| 使用的评价数据 | 本轮使用结果已知的 P2 评测集，策略在补齐调用前冻结 | 保留 P2 原独立结果，将 E23 列为补充比较 |
| 复现与执行记录 | 十二种策略完整评分、四个新控制头及 6,033 次补充调用完成 | 最终数值以 `RESULTS_FINAL.json` 为准，`RESULTS.json` 记录准备阶段与缓存覆盖检查 |

## 后续整理与扩展问题

| 事项 | 当前状态 | 后续工作 |
|---|---|---|
| 论文统一结果与图 | E17–E23 的结论、所用数据和完整结果已列入文档 | 围绕训练约束、部署规则和准确率差组织正文，保留正负结果 |
| 完整复现资产 | 公开源码与脱敏汇总已归档；逐题矩阵、权重和原生缓存为私有依赖，E18/P2 生成链有四个源文件待补齐 | 按 [复现说明](../docs/REPRODUCIBILITY.md) 恢复环境、依赖和资产，分别核对公开表格、合成例子的实现检查、重训与生成 |
| 同等调参机会下的性能排序 | E23 各对照方法沿用既有超参数，控制用于固定条件下的机制比较 | 若研究最优性能，另设只使用训练区的统一调参协议 |
| 跨系统质量 | 一套三成员聚合池；外部实验为六模型单选，质量方向随损失变化 | 扩展到新的聚合模型池并评价准确率 |
| 非线性或大容量预测器 | 当前主体是固定线性头及指定正则 | 将函数类变化作为独立问题，分别考察参数、严格选择变化和准确率 |
| 有限反馈采集效率 | E1–E16 保留早期探索，当前主线使用冻结奖励与策略 | 若恢复采集研究，单独报告相同实际预算下的策略质量与成本 |

## 报告规则

新汇总报告准确率、配对差值、纠错与破坏、任务/子任务/生成重复分解及完整成本。期望计数与整数记录事件分别标注。沿用项目约定，不新增置信区间或 bootstrap；历史冻结文件保留原文和原哈希。
