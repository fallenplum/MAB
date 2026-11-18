# 研究历史与版本定位

## 当前主线与历史记录

MAB 起初研究有限评测预算下的模型组合路由，随后依次检查聚合协议、预测器、主动反馈、表征、监督平滑和损失结构。当前 E17–E22 聚焦经验并列监督通过共享参数影响路由决策的机制，以及这种变化的质量边界。

本页说明各阶段为何保留、哪些结论已经被后续结果补充。历史成功、失败和停止决定均保留其当时的数据与协议身份。早期报告中的“下一步”“尚未完成”只描述写作当时的状态，不能据此判断当前仓库。

以下仓库链接全部固定到提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`，不依赖实验分支名称。后续分支重命名不改变这些来源链接。当前论文允许使用的主张见 [主张与证据](../paper/CLAIMS_AND_EVIDENCE.md)。

## 三份项目来源材料

以下为项目提供的历史来源材料，原件不因这次仓库整理新增到仓库中。本节仅记录其题名、日期、角色和内容边界，不提供不存在的仓库附件链接。

| 材料 | 日期与身份 | 研究内容 | 与当前论文的关系 |
|---|---|---|---|
| 《有限评测预算下的语言模型组合路由》 | 文内日期 2026-10-26；研究方案 | 冻结模型池，按问题选集合，决策相关反馈采集与完整成本约束 | 提供最初问题定义及协议动机；文中的拟议实验不能当作已完成结果 |
| 《图结构演化下的多小模型协同推理》 | 文内日期 2026-10-26；另一研究方案 | 固定节点与消息协议，研究图变化后的边信用陈旧、复检及恢复 | 动作空间和执行依赖不同；没有并入当前集合路由的已完成贡献 |
| *From Hindsight Oracles to Reliable Ensemble Routing* | 项目英文 PDF；文件生成于 2026-11-08；29 页旧稿 | 输入问题路由的 oracle 差距、配对切换认证与双预算理论 | 记录早期结果和拟议理论，不代表 E17–E22；其历史数值与后来新题结果分开保留 |

英文旧稿对应文件 SHA-256 为 `b2c9c7f9bf09a3183f4907eb7daedd3a7537dc9f6460f62ccfc246e525bb9e99`，与 [E5 摘要登记](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e5_paired_gain_validation/EXPERIMENT_SUMMARY.md)一致。旧稿主要实验使用 600 训练题与 200 开发题；当时提出的认证和预算采集程序尚未验证。E5 后续对其中若干理论与诊断进行了对应实验，因此“旧稿未做”不能延伸为“仓库从未做”。

旧稿曾记录 subtask default 87.00%、conditional-gain MLP 86.17%、事后 oracle 96.17%。它们属于当时的 200 题开发队列，不能与 E19 的 train600 外折、P2 的 2,400 新题或 P3 外部 548 题直接混合比较。

## E1–E7 从系统搭建到反馈采集审查

| 阶段 | 主要问题与完成范围 | 历史定位 | 固定来源 |
|---|---|---|---|
| E1 | 初始真实调用、聚合器与输入/答案审计、额外因果诊断 | 暴露早期聚合配置与成本质量问题；候选 oracle 空间不等于可部署路由收益 | [E1 文件索引](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/E1_FILE_INDEX.md) · [审计指标](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/results/e1/audit_metrics.json) · [聚合因果诊断](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/results/e1/causal_diagnostic_summary.json) |
| E2 | 300 组固定 K=3 路由回放及独立评价，统一权限与费用 | 技术执行通过；原预算下共享二阶预测器的排序主结果约 51.16%，未形成可靠选队证据 | [E2 完整汇总](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/release/e2_20261104/E2_AGGREGATED_RESULTS.json) · [CSV](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/release/e2_20261104/E2_AGGREGATED_RESULTS.csv) |
| V2 协议 | 改进聚合器和生成重复，区分充分反馈预测与有限反馈采集 | 参考答案预算按题/真值版本去重，原 V1/E2 的逐组合评分条数不能直接换算成 V2 预算 | [V2 实验设计](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/protocol/decision_feedback_v2/EXPERIMENT_DESIGN_v2.txt) · [V2 文件索引](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/results/decision_feedback_v2/FILE_INDEX_v1.json) |
| E3 | 决策定向采集、随机与信息增益，跨预测器及预算回放 | 16,245 条主网格路径与 6,300 条删组回放仍未建立同时优于随机及通用信息增益的主证据 | [E3 结论](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/decision_acquisition_e3/07_RESEARCH_CONCLUSION.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/decision_acquisition_e3/E3_COMPLETE_RESULTS.json) |
| E4 | 数据与评分映射、Ridge 后验、噪声、二值奖励模型及反馈代理审计 | 36,000 个聚合评分重算一致；概率误差或设计协方差改善未自动转化为服务动作收益 | [E4 结论](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e4_statistical_model_audit/08_E4_RESEARCH_CONCLUSION.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e4_statistical_model_audit/E4_COMPLETE_RESULTS.json) |
| E5 | 英文旧稿的理论构造、中心化误差、门控与双预算配置对应验证 | 发现共同偏移可解释误差下降；估计效率改善未稳定改善策略选择，不能只以理论资格作为算法胜出证据 | [E5 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e5_paired_gain_validation/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e5_paired_gain_validation/E5_COMPLETE_RESULTS.json) |
| E6 | 切换信息、方向监督和差值导向反馈分配 | 部分候选族下两个采集目标代数恒等；先导成本与有限不同问题支持限制实际收益 | [E6 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e6_contrast_directed_feedback/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e6_contrast_directed_feedback/E6_COMPLETE_RESULTS.json) |
| E7 | 候选覆盖、输入学习曲线与同一学习器下主动采集 | 扩大候选可保留更多事后机会；核模型有开发线索，但决策采集没有独立优势，输入可学性与 oracle 分开判断 | [E7 决定](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e7_opportunity_and_feedback_learning/05_E7_RESEARCH_DECISION.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e7_opportunity_and_feedback_learning/E7_COMPLETE_RESULTS.json) |

## E8–E16 从决策损失到监督结构

| 阶段 | 主要问题与完成范围 | 历史定位 | 固定来源 |
|---|---|---|---|
| E8 | Ridge、竞争加权回归、SPO+ 等决策目标与排序诊断 | SPO+ 出现正向开发点值，但主要误差改善集中于零差值，尚未形成可推广质量贡献 | [E8 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e8_decision_focused_combination_learning/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e8_decision_focused_combination_learning/E8_COMPLETE_RESULTS.json) |
| E8 M4 审计 | 组合集中、冠军规则、题目内容与稳定非零差值 | 原 M4 在 467/600 题选组合 0；只更换训练冠军并列规则便有 594/600 题动作改变，成为 E17 的直接动机 | [M4 审计摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e8_m4_mechanism_audit/EXPERIMENT_SUMMARY.md) · [科学复核](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e8_m4_mechanism_audit/SCIENTIFIC_FINAL_REVIEW.md) |
| E9 | 经验标签/后验均值与 MSE/CSCE 的受控比较 | 当前冻结 MiniLM 线性头下未得到一致质量优势；标签估计、损失和真实服务分别评估 | [E9 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e9_consistent_cost_sensitive_learning/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e9_consistent_cost_sensitive_learning/E9_COMPLETE_RESULTS.json) |
| E10 | MiniLM 输入覆盖、Qwen3 完整题目 prefill 表征及组内匹配 | R_full 86.3333% 对 A 85.2778% 有正点差，但内容控制更强；此处“完整输入”指问题文本覆盖，不是读取候选答案后路由 | [E10 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e10_input_prefill_matching_audit/EXPERIMENT_SUMMARY.md) · [覆盖审计](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e10_input_prefill_matching_audit/INPUT_COVERAGE_AUDIT.md) |
| E11 | 训练组均值监督收缩 | 固定 0.5 收缩未改善 R_full，组均值端点达到 87.00%，引出组能力估计路线 | [E11 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e11_group_mean_shrinkage/EXPERIMENT_SUMMARY.md) · [结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e11_group_mean_shrinkage/E11_RESULTS.json) |
| E12 | 有限已揭示原题下的组能力共享 | 每个揭示题仍有全部 20×3 奖励；不同预算和表示存在交互，不能等同于真正部分动作反馈 | [E12 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e12_shared_feedback_learning/EXPERIMENT_SUMMARY.md) · [结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e12_shared_feedback_learning/E12_RESULTS.json) |
| E13 | 真正部分组合反馈与 SaveRouter-FixedMask 对照 | 统一 logger 和权限后，Group 未形成超越 RawCentered 或适配基线的独立优势 | [E13 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e13_partial_combination_feedback/EXPERIMENT_SUMMARY.md) · [结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e13_partial_combination_feedback/E13_RESULTS.json) |
| E14 | 组识别、能力估计与同实际 token 前缀 | 组识别误差非唯一瓶颈，稀疏支持与误差相关；相同成本下配对反馈对不同学习器方向不同 | [E14 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e14_mechanism_diagnostics/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e14_mechanism_diagnostics/E14_COMPLETE_RESULTS.json) |
| E15 | 固定 token 下 Main、邻居 Pair、随机 Pair 与三学习器交互 | 主预算 Group 邻居对 Main +0.8519 个百分点，但跨预算方向不一致；原自动扩展门槛未通过 | [E15 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e15_feedback_structure_interaction/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e15_feedback_structure_interaction/E15_COMPLETE_RESULTS.json) |
| E16 | 方差、相同 nL 的独立题与重复生成、表示对照 | nL=480 时更多不同题的 L=1 比 L=3 高 0.9444 个百分点；实际 gold/token 不同，不能称预算完全相等的普遍结论 | [E16 摘要](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e16_signal_replication_representation/EXPERIMENT_SUMMARY.md) · [完整结果](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e16_signal_replication_representation/E16_COMPLETE_RESULTS.json) |

早期记录还包含 [e9_bayes_gate 报告](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/results/e9_bayes_gate/07_scientific_report_zh.md)。它与 `research/e9_consistent_cost_sensitive_learning` 是不同路径的历史工件，不能仅凭同名 E9 合并结果。

## E17–E22 形成当前机制主线

1. **E17：分开监督方向、运算顺序与真实服务。** 重编号干预表明任意经验冠军规则可塑造真实组合选择；均匀冠军与 Ridge 的稳定性提供控制。
2. **E18：保留独立质量负结果。** 2,998 道可评分新题上 TSPO_uniform 对 SPO_coordinate 的点差接近零，未因机制现象存在就宣称质量受益。
3. **E19：从“偏向哪个动作”转到“对称项是否仍施加约束”。** 在固定共享预测器和正则下只改变全并列行权重；出现 150 道严格换队，理论构造给出跨输入传播可能性。
4. **E20 / P2：独立确认完整冻结策略。** 2,400 道新题上 C−B 为 +1.326 个百分点；raw-exact 敏感性为 +0.639，主要收益来自 BBH。
5. **E21 / P3：跨损失和外部池限定。** ListNet 及 DARS 缓存上的单模型选择仍有严格动作变化，但质量差没有统一方向。外部固定基线是按任务选择，不是跨任务单一模型。
6. **E22：匹配控制与参数几何。** 总权重控制不能解释所有变化，但一般行降权也会改变排序；定向关闭全并列行的特殊质量优势证据不足。

各阶段准确来源、数据角色和不得外推的范围见 [主张与证据](../paper/CLAIMS_AND_EVIDENCE.md)。这些阶段不是同一个方法在同一测试集上的连续调参成绩，不能只摘取最终正数拼成总体增益。

## 保留与报告原则

历史源码、协议与结果保留其冻结版本，分支名称只承担导航作用。新的论文入口反映当前主线，同时指向固定提交，避免旧报告中的计划段落被误读为当前进展。

新汇总不报告或重算置信区间，不为其运行 bootstrap。历史文件中已存在的统计分析保留原字节，不将过去的停止门槛自动当作未来科研行动的门槛。负结果、工程失败、数据缺陷、原题数和执行成本与正结果共同保存。

公开归档不新增原题、标准答案、逐题私有矩阵、拟合权重、原生输出缓存、认证信息或执行进程状态。公开源码及汇总不等于已经开放这些资产；复现说明须明确每一步实际依赖。
