# 少样本适应 复现说明

比较少量 Support 反馈后的路由更新方法。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/3a5b15020f06eeeec15bc5c87a439dffb20be215/routing_idea2/research/fewshot_adaptation_20261107/EXPERIMENT_SUMMARY.md) |
| 结果 | [EXPERIMENT_RESULTS_COMPACT.json](https://github.com/fallenplum/MAB/blob/3a5b15020f06eeeec15bc5c87a439dffb20be215/routing_idea2/research/fewshot_adaptation_20261107/EXPERIMENT_RESULTS_COMPACT.json) |
| 代码 | [fewshot_adaptation.py](https://github.com/fallenplum/MAB/blob/3a5b15020f06eeeec15bc5c87a439dffb20be215/routing_idea2/research/fewshot_adaptation_20261107/fewshot_adaptation.py) |
| 运行说明 | [CPU_EXPERIMENT_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/3a5b15020f06eeeec15bc5c87a439dffb20be215/routing_idea2/research/fewshot_adaptation_20261107/CPU_EXPERIMENT_PROTOCOL.json) |
| 实验设置 | [CPU_EXPERIMENT_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/3a5b15020f06eeeec15bc5c87a439dffb20be215/routing_idea2/research/fewshot_adaptation_20261107/CPU_EXPERIMENT_PROTOCOL.json) |
| 补充报告 | [01_episode_construction.md](https://github.com/fallenplum/MAB/blob/3a5b15020f06eeeec15bc5c87a439dffb20be215/routing_idea2/research/fewshot_adaptation_20261107/01_episode_construction.md) |
| 补充报告 | [02_fewshot_adaptation_baselines.md](https://github.com/fallenplum/MAB/blob/3a5b15020f06eeeec15bc5c87a439dffb20be215/routing_idea2/research/fewshot_adaptation_20261107/02_fewshot_adaptation_baselines.md) |


## 获取代码

```bash
git clone --single-branch --branch fewshot-adaptation-20261107 https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`3a5b15020f06eeeec15bc5c87a439dffb20be215`](https://github.com/fallenplum/MAB/commit/3a5b15020f06eeeec15bc5c87a439dffb20be215)。

[本分支首页](https://github.com/fallenplum/MAB/tree/fewshot-adaptation-20261107) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
