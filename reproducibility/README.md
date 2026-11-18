# E12 复现说明

比较有限已评分问题下组能力和逐题监督的作用。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/46e475f2c6d394a20c8292c12ff1c3194a0be9b3/routing_idea2/research/e12_shared_feedback_learning/EXPERIMENT_SUMMARY.md) |
| 结果 | [E12_RESULTS.json](https://github.com/fallenplum/MAB/blob/46e475f2c6d394a20c8292c12ff1c3194a0be9b3/routing_idea2/research/e12_shared_feedback_learning/E12_RESULTS.json) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/46e475f2c6d394a20c8292c12ff1c3194a0be9b3/routing_idea2/research/e12_shared_feedback_learning/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/46e475f2c6d394a20c8292c12ff1c3194a0be9b3/routing_idea2/research/e12_shared_feedback_learning/REPRODUCE.md) |
| 实验设置 | [E12_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/46e475f2c6d394a20c8292c12ff1c3194a0be9b3/routing_idea2/research/e12_shared_feedback_learning/E12_PROTOCOL.json) |


## 获取代码

```bash
git clone --single-branch --branch e12-shared-feedback-learning https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`46e475f2c6d394a20c8292c12ff1c3194a0be9b3`](https://github.com/fallenplum/MAB/commit/46e475f2c6d394a20c8292c12ff1c3194a0be9b3)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e12-shared-feedback-learning) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
