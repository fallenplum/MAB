# E20 准备 复现说明

冻结 train600 的 B/C 最终模型、输入预测和后续评价方案。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/EXPERIMENT_SUMMARY.md) |
| 结果 | [E20_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/E20_COMPLETE_RESULTS.json) |
| 代码 | [fit_final_models.py](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/fit_final_models.py) · [predict_final_inputs.py](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/predict_final_inputs.py) · [audit_data_and_budget.py](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/audit_data_and_budget.py) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/REPRODUCE.md) |
| 实验设置 | [E20_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/E20_PROTOCOL.json) |
| 补充报告 | [01_FINAL_MODEL_FREEZE.md](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/01_FINAL_MODEL_FREEZE.md) |
| 补充报告 | [02_DATA_PERMISSIONS.md](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/02_DATA_PERMISSIONS.md) |
| 补充报告 | [03_POWER_ANALYSIS.md](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/03_POWER_ANALYSIS.md) |
| 补充报告 | [04_EXECUTION_BUDGET.md](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/04_EXECUTION_BUDGET.md) |
| 补充报告 | [05_P2_FROZEN_PROTOCOL.md](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/05_P2_FROZEN_PROTOCOL.md) |
| 补充报告 | [06_READINESS_REVIEW.md](https://github.com/fallenplum/MAB/blob/031d32ee34911f58c535803c078584c424c8f6e3/routing_idea2/research/e20_independent_confirmation_preparation/06_READINESS_REVIEW.md) |


## 获取代码

```bash
git clone --single-branch --branch e20-independent-confirmation-preparation https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`031d32ee34911f58c535803c078584c424c8f6e3`](https://github.com/fallenplum/MAB/commit/031d32ee34911f58c535803c078584c424c8f6e3)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e20-independent-confirmation-preparation) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
