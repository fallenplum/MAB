# B1 复现说明

比较有限答案与 Token 预算下的配对评价方法。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/EXPERIMENT_SUMMARY.md) |
| 结果 | [B1_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/B1_COMPLETE_RESULTS.json) · [REAL_FIXED_POOL_RESULTS.json](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/REAL_FIXED_POOL_RESULTS.json) · [PUBLIC_METHOD_RESULTS.json](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/PUBLIC_METHOD_RESULTS.json) |
| 代码 | [main](https://github.com/fallenplum/MAB/tree/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/main) · [inference](https://github.com/fallenplum/MAB/tree/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/inference) · [methods](https://github.com/fallenplum/MAB/tree/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/methods) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/REPRODUCE.md) |
| 实验设置 | [B1_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/B1_PROTOCOL.json) |
| 补充报告 | [01_PUBLIC_BASELINES_AND_ASSUMPTIONS.md](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/01_PUBLIC_BASELINES_AND_ASSUMPTIONS.md) |
| 补充报告 | [02_PAIRED_DESIGN_AND_INFERENCE.md](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/02_PAIRED_DESIGN_AND_INFERENCE.md) |
| 补充报告 | [03_REAL_FIXED_POOL_RESULTS.md](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/03_REAL_FIXED_POOL_RESULTS.md) |
| 补充报告 | [04_POPULATION_RELIABILITY_AND_POWER.md](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/04_POPULATION_RELIABILITY_AND_POWER.md) |
| 补充报告 | [05_ABLATIONS_COST_AND_RESEARCH_DECISION.md](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/05_ABLATIONS_COST_AND_RESEARCH_DECISION.md) |


## 获取代码

```bash
git clone --single-branch --branch b1-reliable-paired-evaluation https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`a8ca3eb660136de0cc0793d153e3561b8315283b`](https://github.com/fallenplum/MAB/commit/a8ca3eb660136de0cc0793d153e3561b8315283b)。

[本分支首页](https://github.com/fallenplum/MAB/tree/b1-reliable-paired-evaluation) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
