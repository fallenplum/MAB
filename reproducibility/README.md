# E13 复现说明

在相同已知组合评分下比较 Group、RawCentered 和固定掩码对照。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82/routing_idea2/research/e13_partial_combination_feedback/EXPERIMENT_SUMMARY.md) |
| 结果 | [E13_RESULTS.json](https://github.com/fallenplum/MAB/blob/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82/routing_idea2/research/e13_partial_combination_feedback/E13_RESULTS.json) · [E13_SUMMARY.csv](https://github.com/fallenplum/MAB/blob/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82/routing_idea2/research/e13_partial_combination_feedback/E13_SUMMARY.csv) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82/routing_idea2/research/e13_partial_combination_feedback/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82/routing_idea2/research/e13_partial_combination_feedback/REPRODUCE.md) |
| 实验设置 | [E13_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82/routing_idea2/research/e13_partial_combination_feedback/E13_PROTOCOL.json) |


## 获取代码

```bash
git clone --single-branch --branch e13-partial-combination-feedback https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`0114a7f744eb39aff24e7e2eb2dbb82af66cdd82`](https://github.com/fallenplum/MAB/commit/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e13-partial-combination-feedback) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
