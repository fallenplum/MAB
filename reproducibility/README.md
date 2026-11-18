# E18.1 复现说明

分析 E18 已完成评价的任务、种子和机制差异。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/EXPERIMENT_SUMMARY.md) |
| 结果 | [E18_1_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/E18_1_COMPLETE_RESULTS.json) |
| 代码 | [analyze_e18_1.py](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/analyze_e18_1.py) · [accounting_global_fixed_audit.py](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/accounting_global_fixed_audit.py) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/REPRODUCE.md) |
| 实验设置 | [E18_1_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/E18_1_PROTOCOL.json) |
| 补充报告 | [01_STATISTICAL_REPORT.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/01_STATISTICAL_REPORT.md) |
| 补充报告 | [02_TASK_HETEROGENEITY.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/02_TASK_HETEROGENEITY.md) |
| 补充报告 | [03_SELECTION_AND_QUALITY.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/03_SELECTION_AND_QUALITY.md) |
| 补充报告 | [04_TSPO_GLOBAL_FIXED.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/04_TSPO_GLOBAL_FIXED.md) |
| 补充报告 | [05_ACCOUNTING_AND_GLOBAL_FIXED.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/05_ACCOUNTING_AND_GLOBAL_FIXED.md) |
| 补充报告 | [06_JOINT_EVIDENCE.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/06_JOINT_EVIDENCE.md) |
| 补充报告 | [07_RESULTS_AND_DISCUSSION.md](https://github.com/fallenplum/MAB/blob/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c/routing_idea2/research/e18_1_statistics_and_mechanisms/07_RESULTS_AND_DISCUSSION.md) |


## 获取代码

```bash
git clone --single-branch --branch e18-1-statistics-and-mechanisms https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`341a8a3c48ee95b7a38c1a5713ac659fccba8f6c`](https://github.com/fallenplum/MAB/commit/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e18-1-statistics-and-mechanisms) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
