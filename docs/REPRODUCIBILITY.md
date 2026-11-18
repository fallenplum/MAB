# E16 复现说明

比较独立题数、生成重复和输入表示对路由学习的影响。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/EXPERIMENT_SUMMARY.md) |
| 结果 | [E16_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/E16_COMPLETE_RESULTS.json) · [B_SEED_RESULTS.csv](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/B_SEED_RESULTS.csv) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/REPRODUCE.md) |
| 实验设置 | [E16_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/E16_PROTOCOL.json) |
| 补充报告 | [01_VARIANCE_AND_STABILITY.md](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/01_VARIANCE_AND_STABILITY.md) |
| 补充报告 | [02_QUESTIONS_VS_REPEATS.md](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/02_QUESTIONS_VS_REPEATS.md) |
| 补充报告 | [03_REPRESENTATION_ABLATION.md](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/03_REPRESENTATION_ABLATION.md) |
| 补充报告 | [04_RESEARCH_DECISION.md](https://github.com/fallenplum/MAB/blob/1f012ae7b9af54adc6a68fcf4417fab81a269982/routing_idea2/research/e16_signal_replication_representation/04_RESEARCH_DECISION.md) |


## 获取代码

```bash
git clone --single-branch --branch e16-signal-replication-representation https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`1f012ae7b9af54adc6a68fcf4417fab81a269982`](https://github.com/fallenplum/MAB/commit/1f012ae7b9af54adc6a68fcf4417fab81a269982)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e16-signal-replication-representation) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
