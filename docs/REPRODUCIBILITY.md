# CES 对照 复现说明

检查公开 CES 实现，并在本项目模型池中作对应比较。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [05_EVIDENCE_CHAIN_AND_NEXT_STEPS.md](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/05_EVIDENCE_CHAIN_AND_NEXT_STEPS.md) |
| 结果 | [CES_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/CES_COMPLETE_RESULTS.json) · [MIGRATION_RESULTS.json](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/MIGRATION_RESULTS.json) |
| 代码 | [run_migration.py](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/run_migration.py) · [evaluate_migration.py](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/evaluate_migration.py) · [freeze_ces.py](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/freeze_ces.py) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/REPRODUCE.md) |
| 实验设置 | [CES_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/CES_PROTOCOL.json) |
| 补充报告 | [01_NATIVE_REPRODUCIBILITY_AUDIT.md](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/01_NATIVE_REPRODUCIBILITY_AUDIT.md) |
| 补充报告 | [02_DATA_FEEDBACK_AND_COST_PERMISSIONS.md](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/02_DATA_FEEDBACK_AND_COST_PERMISSIONS.md) |
| 补充报告 | [03_ADAPTATION_CONTRACT_AND_OPERATOR_AUDIT.md](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/03_ADAPTATION_CONTRACT_AND_OPERATOR_AUDIT.md) |
| 补充报告 | [04_MIGRATION_RESULTS_AND_COST.md](https://github.com/fallenplum/MAB/blob/a701eabf0720b834d584c3f47ccd8e82fdcb4440/routing_idea2/research/ces_traceable_baseline/04_MIGRATION_RESULTS_AND_COST.md) |


## 获取代码

```bash
git clone --single-branch --branch ces-traceable-baseline-20261112 https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`a701eabf0720b834d584c3f47ccd8e82fdcb4440`](https://github.com/fallenplum/MAB/commit/a701eabf0720b834d584c3f47ccd8e82fdcb4440)。

[本分支首页](https://github.com/fallenplum/MAB/tree/ces-traceable-baseline-20261112) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
