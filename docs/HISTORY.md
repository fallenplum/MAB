# 研究历史与版本定位

## 当前主线与历史记录

MAB 起初研究有限评测预算下的模型组合路由，随后依次检查聚合协议、预测器、主动反馈、表征、监督平滑和损失结构。当前 E17–E23 聚焦经验并列监督怎样通过共享参数影响路由决策，以及训练约束和部署规则怎样共同影响答案质量。

本页按阶段记录研究问题、结果与后续决定。各实验分支保留当时的源码、报告、数据设置和协议，早期进展按报告日期阅读。

**全部阶段的报告、代码和分支入口见 [全部实验入口](../EXPERIMENTS.md)。** 早期实验的完整记录继续保存在原实验分支，主线提供统一导航。

E1–E22 的来源链接固定到提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`，E23 固定到 `3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`。固定提交链接随分支更名保持有效。当前论文结论见 [主张与证据](../paper/CLAIMS_AND_EVIDENCE.md)。

## 三份项目来源材料

以下三份原件保留在项目来源中。本节记录其题名、日期、研究内容及与当前论文的关系。

| 材料 | 日期与身份 | 研究内容 | 与当前论文的关系 |
|---|---|---|---|
| 《有限评测预算下的语言模型组合路由》 | 文内日期 2026-10-26；研究方案 | 冻结模型池，按问题选集合，决策相关反馈采集与完整成本约束 | 提供最初问题定义、协议动机和拟议实验安排 |
| 《图结构演化下的多小模型协同推理》 | 文内日期 2026-10-26；另一研究方案 | 固定节点与消息协议，研究图变化后的边信用陈旧、复检及恢复 | 保留为独立方案，动作空间和执行依赖另有设置 |
| *From Hindsight Oracles to Reliable Ensemble Routing* | 项目英文 PDF；文件生成于 2026-11-08；29 页旧稿 | 输入问题路由的 oracle 差距、配对切换认证与双预算理论 | 记录早期结果和拟议理论，为 E17–E23 提供历史背景；数值按原开发队列保留 |

英文旧稿对应文件 SHA-256 为 `b2c9c7f9bf09a3183f4907eb7daedd3a7537dc9f6460f62ccfc246e525bb9e99`，与 [E5 摘要登记](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e5_paired_gain_validation/EXPERIMENT_SUMMARY.md)一致。旧稿主要实验使用 600 训练题与 200 开发题；在该旧稿写作时，认证和预算采集程序仍处于方案阶段。E5 后续对其中若干理论与诊断完成了对应实验，结果见 E5 记录。

旧稿曾记录 subtask default 87.00%、conditional-gain MLP 86.17%、事后 oracle 96.17%。它们使用当时的 200 题开发集；E19 的 train600 外折、P2 的 2,400 新题和 P3 外部 548 题分别列出各自结果。

## E1–E7 从系统搭建到反馈采集审查

| 阶段 | 主要问题与完成范围 | 历史定位 | 固定来源 |
|---|---|---|---|
| E1 | 初始真实调用、聚合器与输入/答案审计、额外因果诊断 | 暴露早期聚合配置与成本质量问题；分别检查事后最优组合上限与实际部署路由收益 | [E1 文件索引](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/E1_FILE_INDEX.md) · [审计指标](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/results/e1/audit_metrics.json) · [聚合因果诊断](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/results/e1/causal_diagnostic_summary.json) |
| E2 | 300 组固定 K=3 路由回放及独立评价，统一权限与费用 | 技术执行通过；共享二阶预测器在原预算下的排序主结果约 51.16% | [E2 完整汇总](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/release/e2_20261104/E2_AGGREGATED_RESULTS.json) · [CSV](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/release/e2_20261104/E2_AGGREGATED_RESULTS.csv) |
| V2 协议 | 改进聚合器和生成重复，区分充分反馈预测与有限反馈采集 | 参考答案预算按题/真值版本去重；V1/E2 按组合评分条数计数，两阶段使用不同预算口径 | [V2 实验设计](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/protocol/decision_feedback_v2/EXPERIMENT_DESIGN_v2.txt) · [V2 文件索引](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/results/decision_feedback_v2/FILE_INDEX_v1.json) |
| E3 | 决策定向采集、随机与信息增益，跨预测器及预算回放 | 16,245 条主网格路径与 6,300 条删组回放中，该方法未同时优于随机采集和通用信息增益 | [E3 结论](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/decision_acquisition_e3/07_RESEARCH_CONCLUSION.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/decision_acquisition_e3/E3_COMPLETE_RESULTS.json) |
| E4 | 数据与评分映射、Ridge 后验、噪声、二值奖励模型及反馈代理审计 | 36,000 个聚合评分重算一致；概率误差或设计协方差有改善，部署动作收益仍无稳定提升 | [E4 结论](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e4_statistical_model_audit/08_E4_RESEARCH_CONCLUSION.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e4_statistical_model_audit/E4_COMPLETE_RESULTS.json) |
| E5 | 英文旧稿的理论构造、中心化误差、门控与双预算配置对应验证 | 共同偏移解释了部分误差下降；估计效率有改善，策略选择的收益随设置变化 | [E5 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e5_paired_gain_validation/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e5_paired_gain_validation/E5_COMPLETE_RESULTS.json) |
| E6 | 切换信息、方向监督和差值导向反馈分配 | 部分候选族下两个采集目标代数恒等；前期采集成本和不同题目的数量限制了实际收益 | [E6 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e6_contrast_directed_feedback/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e6_contrast_directed_feedback/E6_COMPLETE_RESULTS.json) |
| E7 | 候选覆盖、输入学习曲线与同一学习器下主动采集 | 扩大候选集合保留了更多事后最优机会；核模型有开发线索，决策采集未显示独立优势 | [E7 决定](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e7_opportunity_and_feedback_learning/05_E7_RESEARCH_DECISION.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e7_opportunity_and_feedback_learning/E7_COMPLETE_RESULTS.json) |

## E8–E16 从决策损失到监督结构

| 阶段 | 主要问题与完成范围 | 历史定位 | 固定来源 |
|---|---|---|---|
| E8 | Ridge、竞争加权回归、SPO+ 等决策目标与排序诊断 | SPO+ 在开发集出现准确率提升，误差改善主要集中于零差值标签；跨数据适用性仍待验证 | [E8 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e8_decision_focused_combination_learning/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e8_decision_focused_combination_learning/E8_COMPLETE_RESULTS.json) |
| E8 M4 审计 | 组合集中、冠军规则、题目内容与稳定非零差值 | 原 M4 在 467/600 题选组合 0；只更换训练冠军并列规则便有 594/600 题动作改变，成为 E17 的直接动机 | [M4 审计摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e8_m4_mechanism_audit/EXPERIMENT_SUMMARY.md) · [科学复核](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e8_m4_mechanism_audit/SCIENTIFIC_FINAL_REVIEW.md) |
| E9 | 经验标签/后验均值与 MSE/CSCE 的受控比较 | 冻结 MiniLM 线性头下，未观察到一致的质量优势；分别记录标签估计、训练损失和部署准确率 | [E9 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e9_consistent_cost_sensitive_learning/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e9_consistent_cost_sensitive_learning/E9_COMPLETE_RESULTS.json) |
| E10 | MiniLM 输入覆盖、Qwen3 完整题目 prefill 表征及组内匹配 | R_full 86.3333% 对 A 85.2778% 有正点差，但内容控制更强；此处“完整输入”覆盖整个问题文本，路由时使用问题内容 | [E10 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e10_input_prefill_matching_audit/EXPERIMENT_SUMMARY.md) · [覆盖审计](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e10_input_prefill_matching_audit/INPUT_COVERAGE_AUDIT.md) |
| E11 | 训练组均值监督收缩 | 固定 0.5 收缩未改善 R_full，直接使用组均值时达到 87.00%，引出组能力估计路线 | [E11 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e11_group_mean_shrinkage/EXPERIMENT_SUMMARY.md) · [结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e11_group_mean_shrinkage/E11_RESULTS.json) |
| E12 | 有限已揭示原题下的组能力共享 | 减少的是已揭示题目数，每个揭示题仍有全部 20×3 奖励；预算和表示对结果有交互影响 | [E12 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e12_shared_feedback_learning/EXPERIMENT_SUMMARY.md) · [结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e12_shared_feedback_learning/E12_RESULTS.json) |
| E13 | 部分组合反馈与 SaveRouter-FixedMask 对照 | 统一 logger 和权限后，Group 未形成超越 RawCentered 或适配基线的独立优势 | [E13 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e13_partial_combination_feedback/EXPERIMENT_SUMMARY.md) · [结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e13_partial_combination_feedback/E13_RESULTS.json) |
| E14 | 组识别、能力估计与同实际 token 前缀 | 组识别误差非唯一瓶颈，稀疏支持与误差相关；相同成本下配对反馈对不同学习器方向不同 | [E14 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e14_mechanism_diagnostics/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e14_mechanism_diagnostics/E14_COMPLETE_RESULTS.json) |
| E15 | 固定 token 下 Main、邻居 Pair、随机 Pair 与三学习器交互 | 主预算 Group 邻居对 Main +0.8519 个百分点，但跨预算方向不一致；未达到原方案的自动扩展条件 | [E15 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e15_feedback_structure_interaction/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e15_feedback_structure_interaction/E15_COMPLETE_RESULTS.json) |
| E16 | 方差、相同 nL 的独立题与重复生成、表示对照 | nL=480 时更多不同题的 L=1 比 L=3 高 0.9444 个百分点；该比较固定 nL，两组实际 gold/token 消耗不同 | [E16 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e16_signal_replication_representation/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e16_signal_replication_representation/E16_COMPLETE_RESULTS.json) |

早期记录还包含 [e9_bayes_gate 报告](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/results/e9_bayes_gate/07_scientific_report_zh.md)。它与 `research/e9_consistent_cost_sensitive_learning` 分别保留各自的协议、数据设置和结果。

## E17–E23 形成当前机制主线

1. **E17：分开监督方向、运算顺序与真实服务。** 重编号干预表明任意经验冠军规则可塑造真实组合选择；均匀冠军与 Ridge 的稳定性提供控制。
2. **E18：独立评价监督动作的选择规则。** 2,998 道可评分新题上 TSPO_uniform 对 SPO_coordinate 的准确率差为 +0.011749 个百分点，纠错 377、破坏 376。
3. **E19：从“偏向哪个动作”转到“对称项是否仍施加约束”。** 在固定共享预测器和正则下只改变全并列行权重；出现 150 道严格换队，理论构造说明共享参数可能把约束影响传到其他题目。
4. **E20 / P2：独立确认完整冻结策略。** 2,400 道新题上 C−B 为 +1.326 个百分点；raw-exact 敏感性为 +0.639，主要收益来自 BBH。
5. **E21 / P3：不同损失与外部模型池对照。** ListNet 及 DARS 缓存上的单模型选择仍有严格动作变化，但质量差没有统一方向。外部固定基线在每个任务的训练区分别选择模型。
6. **E22：匹配控制与参数几何。** 参数变化超出统一正比例缩放，一般行降权也会改变严格排序。C 对匹配控制平均的准确率差，TSPO+ 为 +0.352、ListNet 为 −0.019 个百分点，跨损失方向不同。
7. **E23：部署规则与同一评测集的对照方法。** B 在原近并列支持内采用训练先验，准确率从 85.35458% 提高到 86.50000%，使原 C−B 的 +1.32598 缩至 +0.18056 个百分点。C 为 86.68056%，比三个匹配控制的等权混合高 0.52315 个百分点；Ridge 为 86.87500%。原严格换队区仍贡献 +0.26389 个百分点，近并列区在改进 B 部署后贡献 −0.08333。该结果将训练约束和部署选择分别纳入论文解释。[E23 摘要](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/EXPERIMENT_SUMMARY.md) · [最终结果](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)

E23 沿用已曝光的 P2 原 2,400 题，属于补充比较。四个新控制头用全部 600 道训练题拟合，沿用 E22 的匹配规则和三个既定种子，使用 `final600` 盐值；E22 的控制保留原五折 OOF 结果。十二份策略在补齐调用前冻结，随后新增成员生成 1,770 次、聚合 4,263 次，共 6,033 次调用、4,486,492 tokens。[协议](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json)与 [analyze_references.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/analyze_references.py)记录策略定义和冻结顺序；`RESULTS.json` 保留准备阶段，`RESULTS_FINAL.json` 记录完整比较和成本，[finalize_results.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/finalize_results.py)实现最终核验、评分与汇总。

各阶段的题池、干预和比较对象见 [主张与证据](../paper/CLAIMS_AND_EVIDENCE.md)：E18 比较监督动作的选择规则，P2 保留原独立评价，P3 使用外部单模型池，E22 使用五折控制，E23 在已曝光的 P2 题池上补充比较。

## 历史归档与后续汇总

历史源码、协议与结果保留其冻结版本，分支名称只承担导航作用。论文入口介绍当前主线，各阶段链接指向固定提交；旧报告的计划和进度按原日期保留。

后续汇总按现有约定报告点值，不运行新的置信区间或 bootstrap；历史统计分析保留原字节。历史停止条件按当时协议记录，后续阶段使用各自的实验方案。负结果、工程失败、数据缺陷、原题数和执行成本与正结果共同保存。

公开归档提供源码、协议和脱敏汇总；原题、标准答案、逐题矩阵、拟合权重及原生输出缓存保留在私有工件中，各步骤所需资产列在复现说明。
