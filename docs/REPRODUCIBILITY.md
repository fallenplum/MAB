# E6 复现说明

比较差值监督、切换信息与反馈分配方法。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/EXPERIMENT_SUMMARY.md) |
| 结果 | [E6_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/E6_COMPLETE_RESULTS.json) · [E6_B_RESULTS.json](https://github.com/fallenplum/MAB/blob/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/b/E6_B_RESULTS.json) · [E6_C_RESULTS.json](https://github.com/fallenplum/MAB/blob/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/c/E6_C_RESULTS.json) |
| 代码 | [a](https://github.com/fallenplum/MAB/tree/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/a) · [b](https://github.com/fallenplum/MAB/tree/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/b) · [c](https://github.com/fallenplum/MAB/tree/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/c) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/REPRODUCE.md) |
| 实验设置 | [E6_RUN_CONFIG.json](https://github.com/fallenplum/MAB/blob/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/E6_RUN_CONFIG.json) |
| 补充报告 | [04_PRIMARY_STATISTICAL_ANALYSIS.md](https://github.com/fallenplum/MAB/blob/0c34a38bede9fcf46359316058759b27fa9f5fea/routing_idea2/research/e6_contrast_directed_feedback/04_PRIMARY_STATISTICAL_ANALYSIS.md) |


## 获取代码

```bash
git clone --single-branch --branch e6-contrast-directed-feedback https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`0c34a38bede9fcf46359316058759b27fa9f5fea`](https://github.com/fallenplum/MAB/commit/0c34a38bede9fcf46359316058759b27fa9f5fea)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e6-contrast-directed-feedback) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
