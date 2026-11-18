# B1：低成本配对评价

比较有限答案与 Token 预算下的配对评价方法。

## 实验记录

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


### 公开评价方法的原例子与适配

[报告](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/01_PUBLIC_BASELINES_AND_ASSUMPTIONS.md) · [运行说明](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/PUBLIC_METHODS_REPRODUCE.md) · [实验设置](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/B1_PROTOCOL.json) · [NATIVE_REFERENCE_RESULTS.json](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/NATIVE_REFERENCE_RESULTS.json) · [PUBLIC_METHOD_RESULTS.json](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/PUBLIC_METHOD_RESULTS.json) · [methods](https://github.com/fallenplum/MAB/tree/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/methods)

### 评价可靠性与检出能力

[报告](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/04_POPULATION_RELIABILITY_AND_POWER.md) · [运行说明](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/REPRODUCE.md) · [实验设置](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/inference/SYNTHETIC_PROTOCOL.json) · [SYNTHETIC_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/inference/SYNTHETIC_COMPLETE_RESULTS.json) · [inference](https://github.com/fallenplum/MAB/tree/a8ca3eb660136de0cc0793d153e3561b8315283b/routing_idea2/research/b1_reliable_paired_evaluation/inference)


## 版本

- 分支：[`b1-reliable-paired-evaluation`](https://github.com/fallenplum/MAB/tree/b1-reliable-paired-evaluation)
- 原实验提交：[`a8ca3eb660136de0cc0793d153e3561b8315283b`](https://github.com/fallenplum/MAB/commit/a8ca3eb660136de0cc0793d153e3561b8315283b)
- 原首页：[README.md](https://github.com/fallenplum/MAB/blob/a8ca3eb660136de0cc0793d153e3561b8315283b/README.md)

[本阶段复现说明](reproducibility/README.md) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md)
