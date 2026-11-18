# E1：E1 与旧 main 备份

保存 E1 原始汇报、E2 结果和 V2 生成与预测诊断。

## 实验记录

| 内容 | 文件 |
|---|---|
| 报告 | [E1_流程与实验结果汇报_20261103.docx](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/reports/e1/E1_%E6%B5%81%E7%A8%8B%E4%B8%8E%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C%E6%B1%87%E6%8A%A5_20261103.docx) |
| 结果 | [e1_diagnostics.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/results/e1/e1_diagnostics.json) · [audit_metrics.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/results/e1/audit_metrics.json) · [causal_diagnostic_summary.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/results/e1/causal_diagnostic_summary.json) |
| 代码 | [scripts](https://github.com/fallenplum/MAB/tree/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/scripts) · [budget_router](https://github.com/fallenplum/MAB/tree/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/src/budget_router) · [audit](https://github.com/fallenplum/MAB/tree/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/artifacts/audit) |
| 运行说明 | [README.md](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/README.md) |
| 实验设置 | [E0_LOCK.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/protocol/E0_LOCK.json) |
| 补充报告 | [E1_FILE_INDEX.md](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/E1_FILE_INDEX.md) |


### V2 生成缓存与预测器诊断

[报告](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/decision_feedback_v2_diagnostic_20261106_v1/REPORT_zh.md) · [运行说明](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/decision_feedback_v2_diagnostic_20261106_v1/FILE_INDEX_zh.md) · [实验设置](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/protocol/decision_feedback_v2/EXPERIMENT_PROTOCOL_v2.json) · [VALIDATION_AND_COST_DECISION_v1.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/decision_feedback_v2_diagnostic_20261106_v1/VALIDATION_AND_COST_DECISION_v1.json) · [PUBLIC_SELECTOR_DIAGNOSTIC_v1.json.gz](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/decision_feedback_v2_diagnostic_20261106_v1/selector/PUBLIC_SELECTOR_DIAGNOSTIC_v1.json.gz) · [PUBLIC_AGGREGATE_SEED_STABILITY_v1.json.gz](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/decision_feedback_v2_diagnostic_20261106_v1/seed_stability/PUBLIC_AGGREGATE_SEED_STABILITY_v1.json.gz) · [source_snapshot](https://github.com/fallenplum/MAB/tree/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/decision_feedback_v2_diagnostic_20261106_v1/source_snapshot)

### 组合评分与跨生成分析

[报告](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/step1/README_zh.md) · [运行说明](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/FILE_INDEX_zh.md) · [实验设置](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/protocol/decision_feedback_v2/EXPERIMENT_PROTOCOL_v2.json) · [STEP1_COMPLETE_SCIENTIFIC_STATISTICS_v1.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/step1/STEP1_COMPLETE_SCIENTIFIC_STATISTICS_v1.json) · [STEP1_CROSS_SEED_TRANSFER_v1.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/step1/STEP1_CROSS_SEED_TRANSFER_v1.json) · [source_snapshot](https://github.com/fallenplum/MAB/tree/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/decision_feedback_v2_diagnostic_20261106_v1/source_snapshot)

### 独立 Top3 选择与评价

[报告](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/step2/README_zh.md) · [运行说明](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/FILE_INDEX_zh.md) · [实验设置](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/protocol/decision_feedback_v2/EXPERIMENT_PROTOCOL_v2.json) · [STEP2_INDEPENDENT_TOP3_VALIDATION_v1.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/step2/STEP2_INDEPENDENT_TOP3_VALIDATION_v1.json) · [STEP2_INDEPENDENT_TOP3_ACCURACY_BY_TASK_v1.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/step2/STEP2_INDEPENDENT_TOP3_ACCURACY_BY_TASK_v1.json) · [STEP2_TOP3_FREEZE_AND_LINEAGE_PUBLIC_v1.json](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/v2_step1_step2_statistics_20261106_v1/step2/STEP2_TOP3_FREEZE_AND_LINEAGE_PUBLIC_v1.json) · [source_snapshot](https://github.com/fallenplum/MAB/tree/bf1db55079fb38b3fd91311f4f3731e405db714a/routing_idea2/release/decision_feedback_v2_diagnostic_20261106_v1/source_snapshot)


## 版本

- 分支：[`archive/main-before-paper-20261118`](https://github.com/fallenplum/MAB/tree/archive/main-before-paper-20261118)
- 原实验提交：[`bf1db55079fb38b3fd91311f4f3731e405db714a`](https://github.com/fallenplum/MAB/commit/bf1db55079fb38b3fd91311f4f3731e405db714a)
- 原首页：[README.md](https://github.com/fallenplum/MAB/blob/bf1db55079fb38b3fd91311f4f3731e405db714a/README.md)

[返回全部实验](../../EXPERIMENTS.md)
