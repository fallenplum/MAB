# E11 复现说明

比较逐题标签、训练组均值和固定混合监督。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/84af4c1bcc63389f9ef9224680292a0e33fb212c/routing_idea2/research/e11_group_mean_shrinkage/EXPERIMENT_SUMMARY.md) |
| 结果 | [E11_RESULTS.json](https://github.com/fallenplum/MAB/blob/84af4c1bcc63389f9ef9224680292a0e33fb212c/routing_idea2/research/e11_group_mean_shrinkage/E11_RESULTS.json) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/84af4c1bcc63389f9ef9224680292a0e33fb212c/routing_idea2/research/e11_group_mean_shrinkage/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/84af4c1bcc63389f9ef9224680292a0e33fb212c/routing_idea2/research/e11_group_mean_shrinkage/REPRODUCE.md) |
| 实验设置 | [E11_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/84af4c1bcc63389f9ef9224680292a0e33fb212c/routing_idea2/research/e11_group_mean_shrinkage/E11_PROTOCOL.json) |


## 获取代码

```bash
git clone --single-branch --branch e11-group-mean-shrinkage https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`84af4c1bcc63389f9ef9224680292a0e33fb212c`](https://github.com/fallenplum/MAB/commit/84af4c1bcc63389f9ef9224680292a0e33fb212c)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e11-group-mean-shrinkage) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
