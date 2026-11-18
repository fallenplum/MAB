# E9 Bayes 对照 复现说明

比较独立奖励预测、层级收缩和默认策略回退规则。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [07_scientific_report_zh.md](https://github.com/fallenplum/MAB/blob/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/results/e9_bayes_gate/07_scientific_report_zh.md) |
| 结果 | [03_main_results.json](https://github.com/fallenplum/MAB/blob/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/results/e9_bayes_gate/03_main_results.json) · [03_main_results.csv](https://github.com/fallenplum/MAB/blob/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/results/e9_bayes_gate/03_main_results.csv) · [02_signal_diagnostics.json](https://github.com/fallenplum/MAB/blob/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/results/e9_bayes_gate/02_signal_diagnostics.json) |
| 代码 | [e9_bayes_gate](https://github.com/fallenplum/MAB/tree/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/experiments/e9_bayes_gate) |
| 运行说明 | [README.md](https://github.com/fallenplum/MAB/blob/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/experiments/e9_bayes_gate/README.md) |
| 实验设置 | [protocol_frozen.json](https://github.com/fallenplum/MAB/blob/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/experiments/e9_bayes_gate/protocol_frozen.json) |
| 补充报告 | [01_math_audit.md](https://github.com/fallenplum/MAB/blob/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/results/e9_bayes_gate/01_math_audit.md) |
| 补充报告 | [02_signal_report.md](https://github.com/fallenplum/MAB/blob/fd68c774642bf3612fbfea09957fdb6fd555a05c/routing_idea2/results/e9_bayes_gate/02_signal_report.md) |


## 获取代码

```bash
git clone --single-branch --branch experiment/e9-bayes-gate https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`fd68c774642bf3612fbfea09957fdb6fd555a05c`](https://github.com/fallenplum/MAB/commit/fd68c774642bf3612fbfea09957fdb6fd555a05c)。

[本分支首页](https://github.com/fallenplum/MAB/tree/experiment/e9-bayes-gate) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
