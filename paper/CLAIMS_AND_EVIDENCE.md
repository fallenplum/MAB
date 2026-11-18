# 论文主张与证据

## 当前论文问题

在有限生成重复产生经验并列奖励的语言模型路由中，训练监督怎样通过共享参数改变其他问题的动作排序？这种变化是否改善最终答案质量，又在哪些控制和系统条件下不成立？

当前主线由 E17–E22 构成，早期反馈采集探索保留在 [研究历史](../docs/HISTORY.md)。本页是写作与核对入口，不代表已经投稿或接收。历史结果来源固定为提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`，下方证据链接全部指向该提交的真实文件。

## 术语和评价口径

- **经验全并列行 T**：仅用对应训练区的生成重复，得到所有动作完全相同的经验平均奖励。它不证明动作真实条件期望相同。
- **B/C 干预**：B 保留 T 的损失，C 将其权重设为零；其他训练行、归一化分母、表示、函数类和正则保持不变。
- **严格换队**：在同一预定解码规则下，两端均具有严格唯一胜者，且选择不同真实动作。它排除“仅在预测并列中换规则”的解释，不保证质量提高。
- **主解码**：在距最大分不超过 1e−8 的动作间均匀分配，以完整冻结支持计算期望质量。
- **raw-exact**：对已有浮点分数使用零容差最大集合；它不是数学精确求解，也不是有真值权限的 oracle。
- **独立题数**：以原始问题计数。同题生成、动作、温度、掩码及重编号不增加独立问题数。不同实验池不直接合并为一个统一测试集。
- **外部固定基线**：E21 在 GPQA 与 MATH-500 各自训练区选择固定模型，再等权汇总测试准确率。报告中的 62.194% 不对应跨两任务共同选定的一个全局模型。

## 已有证据及可写表述

| 编号 | 可写入论文的限定主张 | 已完成结果 | 准确来源 | 范围 |
|---|---|---|---|---|
| C1 | 经验冠军的任意编号规则能够显著改变当前 SPO+ 预测器对真实组合的选择 | 100 次重编号后，坐标冠军 SPO+ 真实组合一致率均值约 6.13%，Ridge 与均匀冠军 TSPO+ 为 100% | [E17 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/EXPERIMENT_SUMMARY.md) · [重编号报告](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/02_RANDOM_RENUMBERING.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e17_supervision_distortion_audit/E17_COMPLETE_RESULTS.json) | 原 600 题开发池；不把选择稳定性解释为质量优势 |
| C2 | 对称化经验冠军规则并未在独立确认中带来实际质量收益 | 2,998 题上 TSPO_uniform 对 SPO_coordinate 为 +0.011749 个百分点，377 次纠错、376 次破坏 | [E18 结论](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e18_confirmatory_execution/final_results/E18_RESEARCH_CONCLUSION.md) · [点结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json) | 两道不可评分题共同排除；原冻结策略的负结果必须保留 |
| C3 | 有限重复可使严格有信息的奖励差呈现经验并列；对称约束也可在共享线性预测器中改变其他输入的严格决策 | 已有有限重复构造、三输入凸构造和数值资格；三输入例子的阈值为 β=46/51 | [E19 理论与预测](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/01_THEORY_AND_PREDICTIONS.md) · [P0 资格](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/02_P0_QUALIFICATION_REPORT.md) | 存在性构造及明确假设；46/51 不是从真实数据估计的阈值 |
| C4 | 在原系统固定表示和正则下，关闭全并列行约束能改变外折题目的严格动作排序 | 600 题中心化分数均变化，203 题两端均严格唯一，其中 150 题换队；严格事件纠错 38、破坏 19 | [E19 P1](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/03_P1_FIVE_FOLD_RESULTS.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json) | 五折共享训练历史且题池长期开发复用；不能写成五次独立验证 |
| C5 | 原系统两支冻结 B/C 策略在独立新题上有正向质量差异 | 2,400 题，B 85.355%、C 86.681%、C−B +1.326 个百分点；GSM8K +0.248、BBH +2.404 | [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) · [点结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json) · [诊断](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_DIAGNOSTICS.json) | 一套六模型、三成员聚合配置；比较完整冻结策略，包含预定解码的影响 |
| C6 | P2 中的质量差和严格变化同时存在，但效果大小受数值并列规则影响 | 严格换队 302 题、整数纠错 75/破坏 56；raw-exact 点差 +0.639 个百分点，小于主规则 +1.326 | [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) · [动作概率诊断](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_DIAGNOSTICS.json) | 不能将总收益全部归因于严格换队；raw-exact 是浮点敏感性诊断 |
| C7 | 严格排序影响在另一损失和外部单选模型池中仍可观察，质量收益没有一致方向 | 原池 ListNet 严格换队 135 题、质量差 −0.278；外部 L=3 TSPO+/ListNet 严格换队 113/41 题、质量差 +0.145/−0.202 个百分点 | [P3 报告](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_RESULTS.md) · [原池 ListNet](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json) · [外部结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json) | 外部 548 题、六模型单选、两任务；不能说已复现 DARS 原算法或第二套三成员聚合 |
| C8 | 外部主对照的 C 未超过训练区按任务选择的固定模型 | 按任务固定基线等权 62.194%，TSPO+ C 57.933%、ListNet C 61.255% | [P3 外部结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json) · [任务内拟合和固定基线源码](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e21_p3_external_mechanism/run_external.py) | 按任务选择后再平均；不得误写成两个任务共享一个固定模型 |
| C9 | 一般行降权同样可以引起广泛严格换队；目前缺少定向关闭全并列行的特殊质量优势 | TSPO+ C 对同总权重 Uniform 为 +1.343，对三掩码平均为 +0.352；ListNet 分别为 −0.278、−0.019 个百分点 | [E22 审计](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/AUDIT_REPORT.md) · [全部控制](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/controls.csv) · [汇总](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/SUMMARY.json) | 原 600 题开发审计，三个预固定掩码未增加独立题数；不选择最优随机掩码作为主对照 |
| C10 | B→C 的参数和得分变化超出共同平移与统一正比例缩放，并能在固定凸目标中作梯度分解 | TSPO+ 最佳正比例残差占 C 范数 76.0%–87.4%，ListNet 23.7%–34.4%；十个损失×折单元通过残差修正几何检查 | [E22 几何](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/GEOMETRY.md) · [几何实现](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/geometry.py) · [几何表](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/geometry.csv) | 合法次梯度分量可能不唯一；分量符号不能单独证明中介因果路径或质量特异性 |
| C11 | 原系统独立评价有完整调用和 token 账本 | P2 成员 41,289 次、聚合 106,536 次，合计 147,825 次，116,097,715 tokens | [P2 执行账本](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_EXECUTION_LEDGER.json) · [P2 结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) | 这是实验评价成本，不是推理加速、服务费用更低或反馈采集效率优势的证明 |

## 理论应怎样与实验连接

全并列行在当前两种目标中具有不同形式。TSPO+ 的逐行项为 `2 × (max(q) − mean(q))`；单位温度 ListNet 的逐行项为 `logsumexp(q) − mean(q) − log(A)`。二者不偏好特定动作编号，但都约束动作分差。经验全并列可能来自真实均值接近，也可能来自有限重复；两种原因在当前真实数据中没有被完全识别。

写作顺序应是：先给出经验并列如何产生及其损失结构，再用固定函数类与正则的端点干预证明严格动作可改变，随后报告独立原系统质量，最后用跨损失、外部模型池和匹配控制限定结论。参数梯度分解解释指定目标中的传递关系，不能代替实际质量实验。

E17 的冠军方向干预与 E19–E22 的对称约束强度干预是不同问题。E18 接近零的独立结果说明“去编号偏好”没有自动带来质量提升；P2 的正结果不修改 E18 的历史结论。

## 尚待完成的整理与证据缺口

| 事项 | 当前状态 | 在主张中如何处理 |
|---|---|---|
| 论文统一术语、结果表和图 | 各阶段报告已有内容，需整合成同一论文叙事 | 用上表限定主张，始终保留 P2 正结果与 E18/P3/E22 边界 |
| 外部固定基线命名 | 历史 P3 摘要写“全局固定”，执行代码实际按任务选择 | 新文稿写“按任务训练区固定基线”；保留固定历史文件以供追溯 |
| 完整复现资产入口 | 公开源码与脱敏汇总存在；逐题矩阵、拟合权重、原生缓存仍是私有依赖，E18/P2 全生成链尚缺四个必需源文件 | 不能声称克隆仓库即可重现全部数值；按 [复现说明](../docs/REPRODUCIBILITY.md) 区分公开复核、合成资格、真实资产重训与全生成 |
| 全并列降权的质量特异性 | E22 控制结果不足以支持 | 当前论文报告这种限制；若要提出“优于一般行降权”，需独立相应证据 |
| 第二套三成员聚合池 | P3 仅为外部单模型选择 | 当前只称跨单选池机制，不称跨聚合系统质量泛化 |
| 非线性或大容量预测器中的普遍行为 | 当前主体为固定线性头及指定正则 | 不把线性目标的严格凸性或参数几何直接外推 |
| 有限反馈采集效率 | 早期阶段未形成当前可承诺的稳定优势，E17–E22 非采集算法主实验 | 不以“预算最优采集”“减少标注”等作为已完成贡献 |

上述条目区分写作整理与增加科学主张需要的证据；不要求为了保留当前有限机制结论而把所有扩展都做完。

## 当前不能主张的结论

1. **关闭全并列训练行普遍提高路由准确率。** P3 和 E22 已提供反向或不支持结果。
2. **原理具有独有的质量优势。** 匹配随机行降权也能改变严格排序，TSPO+ 相对匹配平均的正点差不能替代普遍特异性证据。
3. **无编号偏好意味着无监督影响。** 对称项仍可通过共享参数产生影响，且参数影响和质量影响不同。
4. **全部收益都来自严格换队。** P2 主策略包含数值并列支持上的期望表现，raw-exact 点差明显较小。
5. **三次生成、五折或三份掩码等于独立重复研究。** 原题、训练历史和缓存存在重用。
6. **外部实验复现了 DARS 的完整风险成本目标。** 当前只使用其固定评分缓存验证损失端点。
7. **现有结果证明图剪枝、边信用再校准或多轮协作优势。** 当前动作是组合选择，旧图方案没有对应当前主线实验。
8. **主结果证明有限预算采集更有效或部署更便宜。** E17–E22 的机制对照和评价账本不提供这项结论。
9. **本仓库已经具有已投稿或已接收 ACL 的身份。** 目前是论文整理与复现实验仓库。

## 报告规则

新入口保留准确率、实际点差、纠错与破坏、任务/子任务/生成重复分解和完整成本。不新增或重算置信区间，不为其执行 bootstrap。历史冻结文件保留原文和原哈希，历史统计内容不被删去，也不在新主张表中重新作为推进门槛。分数期望计数与整数实际事件分开标注。
