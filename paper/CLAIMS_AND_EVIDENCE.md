# 论文主张与证据

## 当前论文问题

在有限生成重复产生经验并列奖励的语言模型路由中，训练监督怎样通过共享参数改变其他问题的动作排序？预测近并列时的部署规则又怎样影响这些变化对应的最终答案质量？

当前主线由 **E17–E23** 构成，早期反馈采集探索保留在 [研究历史](../docs/HISTORY.md)。E1–E22 证据固定到提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`，E23 固定到 `3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`。P2 保留原独立评价的身份；E23 是在已经曝光的同一 2,400 题上进行的补充比较。

## 术语和评价口径

- **经验全并列行 T**：仅用对应训练区的生成重复，得到所有动作完全相同的经验平均奖励。它不证明动作真实条件期望相同。
- **B/C 干预**：B 保留 T 的损失，C 将其权重设为零；其他训练行、归一化分母、表示、函数类和正则保持不变。
- **严格换队**：在同一预定解码规则下，两端均具有严格唯一胜者，且选择不同真实动作。动作变化和答案质量分别计量。
- **原主解码**：在距最大分不超过 1e−8 的动作间均匀分配，以完整冻结支持计算期望质量。
- **raw-exact**：对已有浮点分数使用零容差最大集合。这是解码敏感性分析，数值仍来自浮点预测。
- **独立题数**：以原始问题计数。同题生成、动作、温度、掩码及重编号不增加独立问题数。不同实验池不直接合并为一个统一测试集。
- **外部固定基线**：E21 在 GPQA 与 MATH-500 各自训练区选择固定模型，再等权汇总测试准确率，结果为 62.194%。
- **E23 训练先验部署**：在原近并列支持集合内，选择 train600 的三个生成重复中整数成功数最高的组合；剩余计数并列采用均匀策略。B_prior 与 B_uniform 共用预测头，C_prior 与 C_uniform 的全部动作相同。
- **E23 MatchedMean**：三个预定全量 600 题匹配控制策略的等权动作概率混合。控制沿用 E22 的任务/子任务×特征范数分箱规则与种子，使用 `final600` 盐值重新拟合；E22 则是五折 OOF。涉及均匀策略或混合策略的纠错、破坏按耦合期望计数。

## 已有证据及可写表述

| 编号 | 可写入论文的限定主张 | 已完成结果 | 准确来源 | 范围 |
|---|---|---|---|---|
| C1 | 经验冠军的任意编号规则能够改变当前 SPO+ 预测器对真实组合的选择 | 100 次重编号后，坐标冠军 SPO+ 真实组合一致率均值约 6.13%，Ridge 与均匀冠军 TSPO+ 为 100% | [E17 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/EXPERIMENT_SUMMARY.md) · [重编号报告](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/02_RANDOM_RENUMBERING.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/E17_COMPLETE_RESULTS.json) | 原 600 题开发池；此处计量真实组合选择的稳定性 |
| C2 | 对称化经验冠军规则在独立确认中的质量点差接近零 | 2,998 题上 TSPO_uniform 对 SPO_coordinate 为 +0.011749 个百分点，377 次纠错、376 次破坏 | [E18 结论](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e18_confirmatory_execution/final_results/E18_RESEARCH_CONCLUSION.md) · [点结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json) | 两道不可评分题共同排除；冠军方向干预的独立结果 |
| C3 | 有限重复可使严格有信息的奖励差呈现经验并列；对称约束也可在共享线性预测器中改变其他输入的严格决策 | 已有有限重复构造、三输入凸构造和数值资格；三输入例子的阈值为 β=46/51 | [E19 理论与预测](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/01_THEORY_AND_PREDICTIONS.md) · [P0 资格](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/02_P0_QUALIFICATION_REPORT.md) | 存在性构造及明确假设；46/51 不是从真实数据估计的阈值 |
| C4 | 在原系统固定表示和正则下，关闭全并列行约束能改变外折题目的严格动作排序 | 600 题中心化分数均变化，203 题两端均严格唯一，其中 150 题换队；严格事件纠错 38、破坏 19 | [E19 P1](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/03_P1_FIVE_FOLD_RESULTS.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json) | 原 600 题的五折 OOF，题池长期开发复用 |
| C5 | 原系统两支冻结 B/C 策略在独立新题上有正向质量差异 | 2,400 题，B 85.355%、C 86.681%、C−B +1.326 个百分点；GSM8K +0.248、BBH +2.404 | [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) · [点结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json) · [诊断](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_DIAGNOSTICS.json) | 一套六模型、三成员聚合配置；原冻结策略包含预定解码，E23 另作部署规则补充 |
| C6 | P2 中的质量差和严格变化同时存在，但效果大小受数值并列规则影响 | 严格换队 302 题、整数纠错 75/破坏 56；raw-exact 点差 +0.639 个百分点，小于主规则 +1.326 | [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) · [动作概率诊断](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_DIAGNOSTICS.json) | 总点差包含严格换队区和预测近并列区；raw-exact 是浮点敏感性诊断 |
| C7 | 严格排序影响在另一损失和外部单选模型池中仍可观察，质量收益没有一致方向 | 原池 ListNet 严格换队 135 题、质量差 −0.278；外部 L=3 TSPO+/ListNet 严格换队 113/41 题、质量差 +0.145/−0.202 个百分点 | [P3 报告](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_RESULTS.md) · [原池 ListNet](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json) · [外部结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json) | 外部 548 题、六模型单选、两任务；复用 DARS 固定评分缓存 |
| C8 | 外部主对照的 C 未超过训练区按任务选择的固定模型 | 按任务固定基线等权 62.194%，TSPO+ C 57.933%、ListNet C 61.255% | [P3 外部结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json) · [任务内拟合和固定基线源码](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/run_external.py) | 按任务分别选择后再平均，固定基线的选择发生在各自训练区 |
| C9 | E22 开发池中，一般行降权也会引起广泛严格换队；全并列定向降权对匹配控制的质量差跨损失不一致 | TSPO+ C 对同总权重 Uniform 为 +1.343，对三掩码平均为 +0.352；ListNet 分别为 −0.278、−0.019 个百分点 | [E22 审计](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/AUDIT_REPORT.md) · [全部控制](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/controls.csv) · [汇总](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/SUMMARY.json) | 原 600 题开发审计，报告全部三个预定掩码及其均值；与 E23 的全量 600 题头分别计量 |
| C10 | B→C 的参数和得分变化超出共同平移与统一正比例缩放，并能在固定凸目标中作梯度分解 | TSPO+ 最佳正比例残差占 C 范数 76.0%–87.4%，ListNet 23.7%–34.4%；十个损失×折单元通过残差修正几何检查 | [E22 几何](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/GEOMETRY.md) · [几何实现](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/geometry.py) · [几何表](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/geometry.csv) | 固定凸目标下的参数几何；合法次梯度分量可能不唯一，且分量间存在抵消 |
| C11 | 原系统独立评价有完整调用和 token 账本 | P2 成员 41,289 次、聚合 106,536 次，合计 147,825 次，116,097,715 tokens | [P2 执行账本](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_EXECUTION_LEDGER.json) · [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) | 覆盖原冻结策略支持集合的实验评价成本；单策略部署成本需按其执行路径计量 |
| C12 | 原 P2 的大部分 B/C 点差可通过只修改 B 的部署规则缩小 | B_prior 86.50000%，C 86.68056%；B_prior−B_uniform +1.14542，C−B_prior +0.18056 个百分点 | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`baseline_strategies`、`B_deployment_gain_pp`、`remaining_gap_pp` · [先验选择与冻结实现](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/analyze_references.py)：`train_prior`、`freeze` | 只改变 B 的 2,048 道近并列题；先验来自 train600，原权重及支持集合保持固定 |
| C13 | 改用训练先验后，原严格换队区域仍有正贡献，近并列区域转为负贡献 | 302 道严格换队题贡献 +0.26389 个百分点，纠错 75、破坏 56；原 B 近并列区贡献 −0.08333，合计 +0.18056 | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`paired_comparisons.C_prior_minus_B_prior.regions` · [完整分解](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/EXPERIMENT_SUMMARY.md) | 区域贡献均以全 2,400 题为分母；纠错、破坏对应每题三个生成中的记录事件 |
| C14 | 在 E23 的固定 TSPO+ 训练控制下，C 优于同总权重 Uniform 和三份匹配控制 | C−Uniform +1.37173；C 对三个 Matched 分别 +0.56944、+0.54167、+0.45833；对等权混合 +0.52315 个百分点，混合比较在两任务和三个 seed 上均为正 | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`matching`、`paired_comparisons` · [控制协议](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json) · [混合与评分实现](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/finalize_results.py)：`summarize` | 原固定正则、三个预定掩码、四个新拟合的全量 600 题头；与 E22 跨损失五折结果共同解释 |
| C15 | E23 在同一评测集比较的各方法中，Ridge 总体准确率最高 | Ridge 86.87500%，C 86.68056%，GlobalFixed 86.48611%；C 对二者分别 −0.19444、+0.19444 个百分点 | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`baseline_strategies`、`paired_comparisons` · [对照方法恢复实现](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/analyze_references.py)：`freeze` | Ridge 恢复 E17 原全量 600 题头和预处理；GlobalFixed 按训练成功数选组合 5；各方法沿用各自冻结超参数 |
| C16 | E23 已完成十二份冻结策略支持并集的缺失调用和成本记录 | 新增成员 1,770 次、聚合 4,263 次，共 6,033 次调用、4,486,492 tokens；未知或失败尝试 0；原 P2 与补充累计 120,584,207 tokens | [E23 最终汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)：`native_supplement_cost`、`combined_execution_cost` · [原生收据核验与汇总](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/finalize_results.py) | 复用原 P2 题池与已付费缓存，额外购买答案账户 0；覆盖十二份策略的评价费用 |

## 理论应怎样与实验连接

全并列行在当前两种目标中具有不同形式。TSPO+ 的逐行项为 `2 × (max(q) − mean(q))`；单位温度 ListNet 的逐行项为 `logsumexp(q) − mean(q) − log(A)`。二者对动作编号对称，同时约束动作分差。经验全并列可能来自真实均值接近，也可能来自有限重复；当前真实数据中，两种来源的占比尚未区分。[理论构造](../routing_idea2/research/e19_symmetric_tie_constraint/01_THEORY_AND_PREDICTIONS.md)与[参数几何](../routing_idea2/research/e22_mechanism_specificity/GEOMETRY.md)分别说明损失结构和共享参数中的传递。

论文叙事依次连接五件事：经验并列的来源与损失结构；固定表示、函数类和正则下的严格动作变化；原系统独立质量结果；部署规则对同一预测头的影响；跨损失、外部模型池和训练行控制。E23 将部署选择加入这一链条：原 P2 的 +1.32598 个百分点可分解为 B 训练先验相对原均匀规则的 +1.14542，以及 C 相对 B 训练先验的 +0.18056。该分解描述这组冻结策略的质量差，后者仍随任务和生成 seed 改变方向。

E17/E18 研究经验冠军的方向规则，E19–E23 研究全并列约束强度及其后续部署。E18 的接近零结果、P2 原冻结策略的正点差、P3 跨损失和外部池的不同方向、E22 的开发控制，以及 E23 在原评测集上的补充比较各自对应不同问题。

## E23 对论文判断的更新

| 判断 | E23 增加的证据 | 写作调整 |
|---|---|---|
| 训练约束与部署规则各贡献多少 | 同一 B 预测头用训练先验使准确率提高 1.14542 个百分点；C−B_prior 仅 +0.18056 | 将原 +1.32598 定义为完整策略差，单独展示部署对照和严格区域贡献 |
| 全并列定向降权相对一般降权的表现 | C 对三份预定匹配控制均为正，对混合 +0.52315；E22 的 ListNet 结果仍为 −0.019 | 增加 TSPO+、固定正则和本轮控制范围内的正向证据，保留跨损失差异 |
| 作为完整路由算法的性能排序 | Ridge 86.87500% 高于 C 86.68056%，GlobalFixed 为 86.48611% | 论文贡献聚焦训练与部署机制；完整性能表同时列出这些对照方法 |
| 评价数据的角色 | 本轮使用已曝光 P2 题池，策略在补齐调用前冻结 | 保留 P2 原独立结果，将 E23 列为补充比较 |
| 复现与执行记录 | 十二份策略完整评分、四个新控制头及 6,033 次补充调用完成 | 最终数值以 `RESULTS_FINAL.json` 为准，`RESULTS.json` 保留准备与覆盖审计身份 |

## 后续整理与扩展问题

| 事项 | 当前状态 | 后续工作 |
|---|---|---|
| 论文统一结果与图 | E17–E23 的主张、数据角色和完整结果已列入入口 | 围绕训练约束、部署规则和质量差组织正文，保留正负结果 |
| 完整复现资产 | 公开源码与脱敏汇总已归档；逐题矩阵、权重和原生缓存为私有依赖，E18/P2 生成链有四个源文件待补齐 | 按 [复现说明](../docs/REPRODUCIBILITY.md) 恢复环境、依赖和资产，分别核对公开表格、合成资格、重训与生成 |
| 同等调参机会下的性能排序 | E23 各对照方法沿用既有超参数，控制用于固定条件下的机制比较 | 若研究最优性能，另设只使用训练区的统一调参协议 |
| 跨系统质量 | 一套三成员聚合池；外部实验为六模型单选，质量方向随损失变化 | 更广质量主张需要新的聚合池与对应评价 |
| 非线性或大容量预测器 | 当前主体是固定线性头及指定正则 | 将函数类变化作为独立问题，分别考察参数、严格排序和质量 |
| 有限反馈采集效率 | E1–E16 保留早期探索，当前主线使用冻结奖励与策略 | 若恢复采集研究，单独报告相同实际预算下的策略质量与成本 |

当前证据的优势是受控端点、严格动作变化、参数几何与部署对照相互补充。主要限制是原 600 题长期开发复用、有限生成重复、固定函数类和正则，以及 E23 仅三份匹配掩码并使用已曝光题池。当前论文可围绕这些具体机制与适用范围完成写作，扩展问题按拟新增的科学主张安排。

## 报告规则

新汇总报告准确率、配对差值、纠错与破坏、任务/子任务/生成重复分解及完整成本。期望计数与整数记录事件分别标注。沿用项目约定，不新增置信区间或 bootstrap；历史冻结文件保留原文和原哈希。
