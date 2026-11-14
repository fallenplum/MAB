# E10 固定输入 prefill 与组内匹配审计

本轮已完成 P0、P1、P2 和 P3。结论为 `STOP_FIXED_PREFILL_PCA16_RIDGE`，证据状态为 `EXPLORATORY_REUSED_TRAIN600`。

## 阅读入口

- `EXPERIMENT_SUMMARY.md`保存研究问题、完整数值、任务与生成重复分解、成本和局限。
- `MECHANISM_AND_STOP_DECISION.md`逐项核对预声明 GO，并给出停止范围及功效规划。
- `E10_PROTOCOL.json`与冻结收据保存唯一候选、唯一主比较及事前科学规则。
- `E9_REVIEW_TIMELINE_CLARIFICATION.json`及说明追加历史审查时间澄清，原 E9 文件保持原字节。
- `INPUT_COVERAGE_AUDIT.json`为协议冻结时审计，`INPUT_COVERAGE_AUDIT_COMPLETION.json`追加独立CPU重编码完成证据。
- `FROZEN_POLICY_MATCHING_DECOMPOSITION.json`保存 A/B/C/D/NS 的精确组内分解、动作频率和支持数。
- `E10_MAIN_RESULTS.json`、`E10_PRIMARY_STATISTICS.json`和`E10_CONTENT_CONTROLS.json`保存冻结 P2 评价结果。
- `COST.json`与`INFERENCE_CPU_PROFILE.json`分别保存实际 prefill、历史执行账本及冻结头的 CPU 推理剖析。
- `REPRODUCE.md`给出本机实际通过的验证和评分重放入口。

## 范围与权限

沿用六模型、K=3、20组合、原 V2 聚合评分、train600、三个生成重复和原五外折。新答案生成、新 gold 和新组合聚合均为0。实际新增606次正式 Qwen prefill，另有10次合成输入处理。完整历史监督成本作为共同投入单列。

源记录 `source_refs.identity.new_llm_calls=0`由既有缓存加载器继承，其含义限定于该加载器读取历史缓存的行为。本轮资源以 `COST.json`中的真实 prefill 计数为准。

真实 BBH 子任务只用于分层、内容置乱及诊断。主选队头只接收原文的冻结表征。validation200、fresh400、sealed200的奖励访问均为 false。

## 数学与实现证据

P1解析分解的数学资格、固定 Ridge 合成资格、原文 token/offset 映射、模型 checkpoint 文件 SHA、prefill 层位与padding资格、25头全部预测封存及独立统计复核分别保留对应收据。

GSM8K原始子任务字段为空。初次 P1 在纯分解函数的非空元数据校验处退出，失败源码和日志已保留。实际入口在分解前把该字段规范为协议既定的 GSM8K 单组。冻结核心源码保持原字节，薄适配及其事前来源收据单独保存。

E9已有两次冻结模型评分重放。原 E9 full-refit 入口的存在和完整重算实际执行分别记录，当前来源未提供后者的执行证据。

## 私有资产

canonical题目、offset前缀、逐题 token、向量、模型权重来源、折内 PCA/Scaler/系数、逐题奖励和预测、原始日志及失败记录保存在 E10 的 `private/`。完整原执行档案归档到 `private/runtime/`。

## 优势与风险

受控输入替换与解析分解使动作先验变化和细粒度匹配可以分别核对。限制包括 BBH 细组通常只有2至3题、旧600题的反复开发、固定层/PCA16可能损失信息，以及新增真实 prefill 成本。区间条件于已拟合模型，跨独立问题总体的收益仍需新的证据。
