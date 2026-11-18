# 条件增益 复现说明

比较二阶预测、直接组合预测和相对默认组合的增益。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/EXPERIMENT_SUMMARY.md) |
| 结果 | [CPU_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/CPU_COMPLETE_RESULTS.json) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/code) |
| 运行说明 | [CPU_EXPERIMENT_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/CPU_EXPERIMENT_PROTOCOL.json) |
| 实验设置 | [CPU_EXPERIMENT_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/CPU_EXPERIMENT_PROTOCOL.json) |
| 补充报告 | [01_stability_and_oracle_analysis.md](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/01_stability_and_oracle_analysis.md) |
| 补充报告 | [02_second_order_vs_direct20.md](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/02_second_order_vs_direct20.md) |
| 补充报告 | [03_feature_ablation.md](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/03_feature_ablation.md) |
| 补充报告 | [04_conditional_gain_router.md](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/04_conditional_gain_router.md) |
| 补充报告 | [05_theory_and_proofs.md](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/05_theory_and_proofs.md) |
| 补充报告 | [06_fresh400_evaluation.md](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/06_fresh400_evaluation.md) |
| 补充报告 | [07_feedback_acquisition.md](https://github.com/fallenplum/MAB/blob/8e8093c3e1f3748edd1417564a7665f025b53e78/routing_idea2/research/conditional_gain_20261106/07_feedback_acquisition.md) |


## 获取代码

```bash
git clone --single-branch --branch conditional-gain-20261106 https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`8e8093c3e1f3748edd1417564a7665f025b53e78`](https://github.com/fallenplum/MAB/commit/8e8093c3e1f3748edd1417564a7665f025b53e78)。

[本分支首页](https://github.com/fallenplum/MAB/tree/conditional-gain-20261106) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
