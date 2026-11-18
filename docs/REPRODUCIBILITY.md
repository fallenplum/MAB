# E9 复现说明

在相同输入与模型容量下比较标签估计和损失函数。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/6502fd62932e91deb35e3b09449506afcf9f7e1c/routing_idea2/research/e9_consistent_cost_sensitive_learning/EXPERIMENT_SUMMARY.md) |
| 结果 | [E9_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/6502fd62932e91deb35e3b09449506afcf9f7e1c/routing_idea2/research/e9_consistent_cost_sensitive_learning/E9_COMPLETE_RESULTS.json) · [E9_MAIN_RESULTS.json](https://github.com/fallenplum/MAB/blob/6502fd62932e91deb35e3b09449506afcf9f7e1c/routing_idea2/research/e9_consistent_cost_sensitive_learning/E9_MAIN_RESULTS.json) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/6502fd62932e91deb35e3b09449506afcf9f7e1c/routing_idea2/research/e9_consistent_cost_sensitive_learning/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/6502fd62932e91deb35e3b09449506afcf9f7e1c/routing_idea2/research/e9_consistent_cost_sensitive_learning/REPRODUCE.md) |
| 实验设置 | [E9_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/6502fd62932e91deb35e3b09449506afcf9f7e1c/routing_idea2/research/e9_consistent_cost_sensitive_learning/E9_PROTOCOL.json) |


## 获取代码

```bash
git clone --single-branch --branch e9-consistent-cost-sensitive-learning https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`6502fd62932e91deb35e3b09449506afcf9f7e1c`](https://github.com/fallenplum/MAB/commit/6502fd62932e91deb35e3b09449506afcf9f7e1c)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e9-consistent-cost-sensitive-learning) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
