# E23：部署规则与对照方法

在原 P2 题池上比较训练先验部署、固定组合和训练行控制。

## 实验记录

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/EXPERIMENT_SUMMARY.md) |
| 结果 | [RESULTS_FINAL.json](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json) · [RESULTS.json](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS.json) |
| 代码 | [analyze_references.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/analyze_references.py) · [finalize_results.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/finalize_results.py) · [supplement_adapter.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/supplement_adapter.py) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/REPRODUCE.md) |
| 实验设置 | [PROTOCOL.json](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json) |


### E23 策略冻结与缺口记录

RESULTS.json 保留补齐前的状态和拟合日志；完整参照结果见 RESULTS_FINAL.json。

[报告](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json) · [运行说明](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/REPRODUCE.md) · [实验设置](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json) · [RESULTS.json](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/RESULTS.json) · [analyze_references.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/analyze_references.py) · [cache_budget.py](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/routing_idea2/research/e23_deployment_prior_and_references/cache_budget.py)


## 版本

- 分支：[`e23-deployment-prior-and-references`](https://github.com/fallenplum/MAB/tree/e23-deployment-prior-and-references)
- 原实验提交：[`3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`](https://github.com/fallenplum/MAB/commit/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2)
- 原首页：[README.md](https://github.com/fallenplum/MAB/blob/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2/README.md)

[本阶段复现说明](reproducibility/README.md) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md)
