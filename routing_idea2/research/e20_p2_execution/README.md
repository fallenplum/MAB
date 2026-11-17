# E20 P2独立确认

## 当前阶段

本目录记录新2400题的独立确认执行。输入准入、冻结MiniLM编码、B/C输入预测、执行并集和盲评资格已完成。启动时间、进程、GPU分配、实际调用及完成数量由运行层后续收据更新。准备收据中的调用计数0对应其生成时点。

研究对象继承E20最终train600模型，B固定β=1、γ=0，C固定β=0、γ=0。主部署为τ=1e−8均匀argmax，raw-exact为预声明敏感性诊断。主要评价为C−B最终聚合正确率，GSM8K/BBH各50%，三个生成seed保留在同一原始问题内。

## 已冻结规模

| 项目 | 数量 |
|---|---:|
| 独立原始问题 | 2400 |
| GSM8K／BBH | 1200／1200 |
| BBH子任务 | 27 |
| 成员生成并集 | 41289 |
| 组合聚合并集 | 106536 |
| B平均支持动作 | 14.6533 |
| C平均支持动作 | 1 |
| raw-exact新增支持动作 | 0 |

BBH含原共同子池39题和完整官方源扩展1161题，来源层在结果中单列。原fresh400已曝光开发身份继续记录，sealed200保持答案封存并从本次2400题排除。

## 重要文件

| 文件 | 内容 |
|---|---|
| P2_PRESELECTION_PROTOCOL.json | 用户授权、样本量、采样规则、模型、部署与统计约定 |
| P2_INPUT_SELECTION_FROZEN.json | 选题前固定的seed、子任务配额和去重规则 |
| P2_INPUT_ADMISSION_REPORT.json | 历史排除、来源、配额和输入准入结果 |
| P2_ENCODER_ENVIRONMENT.json | 正式E虚拟环境及一次0前向版本拒绝 |
| P2_ENCODER_RECEIPT.json | 2400题冻结编码实际时间与来源 |
| P2_FROZEN_EXECUTION_UNION.json | B/C精确支持并集及调用预算 |
| P2_REFERENCE_LOCATOR_QUALIFICATION.json | 2400个官方源位置的输入侧资格 |
| P2_REAL_SCOPE_QUALIFICATION.json | 实际输入、概率和执行并集的独立复核 |
| P2_BLIND_AND_SCOPE_QUALIFICATION.json | 作用域、权限和期望效用的软件控制 |
| P2_DATA_AND_FROZEN_SCOPE.md | 数据、输入侧机制量与成本说明 |
| P2_NATIVE_COLLECTION_README.md | 闭合包导入、offset恢复、成功成本与完整输出评分门槛 |
| P2_NATIVE_COLLECTION_REAL_PARTIAL.json | 两包1840个实际成功输出的导入资格快照 |
| P2_NATIVE_COLLECTOR_SOFTWARE_QUALIFICATION.json | 12项恢复、并发、去重及完整冻结资格 |
| P2_SINGLE_AGGREGATION_MIGRATION.json | 单次迁移在所记录UTC时点的接续结果 |
| P2_RUNTIME_SCALING_REVIEW.json | 本地速率敏感性及执行器扩展边界分析 |
| P2_THREE_SHARD_ACTIVATION.json | 原actor安全停止、三分片实际启动与首成功的脱敏快照 |
| P2_V3_TOOLING_QUALIFICATION.json | 原准备阶段34项合成资格及来源哈希 |
| P2_V3_REGISTERED_DISPATCH_HANDOFF.md | 显式来源、互斥分片与部署追加说明 |
| P2_FIVE_SHARD_ACTIVATION.json | GPU1/6增量授权后的五路接续与首成功快照 |
| P2_FIVE_SHARD_TOOLING_QUALIFICATION.json | 原16项五路软件资格，保留准备时点状态 |
| P2_FIVE_SHARD_HANDOFF.md | 五路调度接口与实际首成功追加说明 |
| P2_MEMBER_PARALLEL_HANDOFF.md | 慢成员4的互斥接续及现有worker接口 |
| P2_MEMBER_PARALLEL_QUALIFICATION.json | 原17项成员接续软件资格，保留准备时点状态 |
| P2_RUNTIME_FILL_ACTIVATION.json | 成员4四路、聚合六路的安全接续及实际首成功快照 |
| REPRODUCE.md | 真实入口、环境、私有依赖和执行顺序 |

P2_PREPARATION_SUMMARY.md给出准备阶段结论。实际质量报告在完整输出冻结及合法评分后生成。

新增运行源码的交付采用显式ready清单。FILE_INDEX.json逐项记录发布字节哈希、原始来源哈希、脱敏字段和工件角色。各次接续的本地资格与实际部署快照分别保留。当前运行快照优先指向已生成的P2_RUNTIME_FILL_ACTIVATION.json，再依次保留五路、三路及原启动记录。该打包清单不产生新的部署事件。

## 私有资产与两个scope

私有根目录为`E:\语言模型组合路由\research\e20_p2_execution\private`。其下ADMITTED2400_INPUTS_ONLY.jsonl保存已准入输入，P2_INPUTS_ONLY.npz保存冻结向量，P2_PREDICTIONS_FROZEN.npz保存完整得分与动作概率，P2_REFERENCE_LOCATORS_FROZEN.json保存盲评源定位元数据。

`private/EXECUTION_SCOPE.json`为科学执行清单，直接从冻结B/C正概率支持集生成。`private/runtime_launch_v2/inputs/EXECUTION_SCOPE_RUNTIME.json`为运行适配器的权限scope，额外绑定协议、预算、模型、预测和执行批准。两者成员及聚合键集合逐一一致。工作器只接受运行scope，盲评可核对科学清单和运行scope的相同支持并集。

远端运行目录为`/data2/chendong/e20_p2_execution`。模型与原E18推理环境复用既有数据盘资产。标准答案、题目原文、身份映射、权重与逐题评分存私有工件，公开交付使用脱敏汇总。

## 结果边界

输入侧严格换队、近并列或动作分布变化均为机制指标。服务质量由已批准2400题的冻结评价决定。统计单位为原题，后续报告准确率、配对点差、p值、纠错/破坏和实际费用。本阶段沿用项目约定，新置信区间和仅用于区间的bootstrap均为0。

优势是选题先于预测、完整历史与保护题排除、B/C参数和执行并集已冻结。风险是官方BBH采样框扩展、固定近重复筛查的语义覆盖边界、公开数据的预训练曝光，以及B宽支持带来的执行成本。
