# E23 复现说明

在原 P2 题池上比较训练先验部署、固定组合和训练行控制。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/EXPERIMENT_SUMMARY.md) |
| 结果 | [RESULTS_FINAL.json](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json) · [RESULTS.json](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS.json) |
| 代码 | [analyze_references.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/analyze_references.py) · [finalize_results.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/finalize_results.py) · [supplement_adapter.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/supplement_adapter.py) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/REPRODUCE.md) |
| 实验设置 | [PROTOCOL.json](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json) |


## 获取代码

```bash
git clone --single-branch --branch e23-deployment-prior-and-references https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`](https://github.com/fallenplum/MAB/commit/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e23-deployment-prior-and-references) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
