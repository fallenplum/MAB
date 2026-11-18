# E17 复现说明

通过重编号和训练顺序对照检查监督对选队的影响。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/EXPERIMENT_SUMMARY.md) |
| 结果 | [E17_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/E17_COMPLETE_RESULTS.json) · [LOSS_ORDER_RESULTS.json](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/LOSS_ORDER_RESULTS.json) · [PERMUTATION_SUMMARIES.json](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/PERMUTATION_SUMMARIES.json) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/REPRODUCE.md) |
| 实验设置 | [E17_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/E17_PROTOCOL.json) |
| 补充报告 | [01_LITERATURE_AND_TARGET.md](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/01_LITERATURE_AND_TARGET.md) |
| 补充报告 | [02_RANDOM_RENUMBERING.md](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/02_RANDOM_RENUMBERING.md) |
| 补充报告 | [03_SAME_DRAWS_OPERATOR_ORDER.md](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/03_SAME_DRAWS_OPERATOR_ORDER.md) |
| 补充报告 | [04_INDEPENDENT_CONFIRMATION_PLAN.md](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/04_INDEPENDENT_CONFIRMATION_PLAN.md) |
| 补充报告 | [05_MECHANISM_AND_RESEARCH_DECISION.md](https://github.com/fallenplum/MAB/blob/94a8579f9c8dc16d1f6afb943cee6678c5a9792a/routing_idea2/research/e17_supervision_distortion_audit/05_MECHANISM_AND_RESEARCH_DECISION.md) |


## 获取代码

```bash
git clone --single-branch --branch e17-supervision-distortion-audit https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`94a8579f9c8dc16d1f6afb943cee6678c5a9792a`](https://github.com/fallenplum/MAB/commit/94a8579f9c8dc16d1f6afb943cee6678c5a9792a)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e17-supervision-distortion-audit) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
