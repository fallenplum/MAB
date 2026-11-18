# E14 复现说明

检查组识别、能力估计以及相同实际 Token 下的反馈差异。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/EXPERIMENT_SUMMARY.md) |
| 结果 | [E14_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/E14_COMPLETE_RESULTS.json) · [A_FOLD_RESULTS.csv](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/A_FOLD_RESULTS.csv) · [C_SEED_RESULTS.csv](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/C_SEED_RESULTS.csv) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/REPRODUCE.md) |
| 实验设置 | [E14_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/E14_PROTOCOL.json) |
| 补充报告 | [01_GROUP_RECOGNITION.md](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/01_GROUP_RECOGNITION.md) |
| 补充报告 | [02_CAPABILITY_ESTIMATION.md](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/02_CAPABILITY_ESTIMATION.md) |
| 补充报告 | [03_TOKEN_MATCHED_FEEDBACK.md](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/03_TOKEN_MATCHED_FEEDBACK.md) |
| 补充报告 | [04_RESEARCH_DECISION.md](https://github.com/fallenplum/MAB/blob/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526/routing_idea2/research/e14_mechanism_diagnostics/04_RESEARCH_DECISION.md) |


## 获取代码

```bash
git clone --single-branch --branch e14-controlled-mechanism-diagnostics https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`9f84007bf4d5ebf516f20a8e0fc74b5e721c4526`](https://github.com/fallenplum/MAB/commit/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e14-controlled-mechanism-diagnostics) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
