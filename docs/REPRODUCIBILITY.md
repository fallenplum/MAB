# E8 复现说明

比较 Ridge、加权回归和 SPO+ 等训练目标。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/EXPERIMENT_SUMMARY.md) |
| 结果 | [E8_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/E8_COMPLETE_RESULTS.json) |
| 代码 | [training](https://github.com/fallenplum/MAB/tree/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/training) · [evaluation](https://github.com/fallenplum/MAB/tree/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/evaluation) · [qualification](https://github.com/fallenplum/MAB/tree/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/qualification) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/REPRODUCE.md) |
| 实验设置 | [E8_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/E8_PROTOCOL.json) |
| 补充报告 | [01_LITERATURE_AND_METHOD_MAPPING.md](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/01_LITERATURE_AND_METHOD_MAPPING.md) |
| 补充报告 | [02_MATHEMATICAL_AND_SOFTWARE_CHECKS.md](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/02_MATHEMATICAL_AND_SOFTWARE_CHECKS.md) |
| 补充报告 | [03_TRAINING_OBJECTIVES_AND_IMPLEMENTATION.md](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/03_TRAINING_OBJECTIVES_AND_IMPLEMENTATION.md) |
| 补充报告 | [04_MAIN_COMPARISONS.md](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/04_MAIN_COMPARISONS.md) |
| 补充报告 | [05_NOISE_AND_RANKING_DIAGNOSTICS.md](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/05_NOISE_AND_RANKING_DIAGNOSTICS.md) |
| 补充报告 | [06_STATISTICAL_COST_AND_RESEARCH_DECISION.md](https://github.com/fallenplum/MAB/blob/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f/routing_idea2/research/e8_decision_focused_combination_learning/06_STATISTICAL_COST_AND_RESEARCH_DECISION.md) |


## 获取代码

```bash
git clone --single-branch --branch e8-decision-focused-combination-learning https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`45767047d556e6cd9cac0e8905cf6dd81d5e3d4f`](https://github.com/fallenplum/MAB/commit/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e8-decision-focused-combination-learning) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
