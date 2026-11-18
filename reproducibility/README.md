# M4 噪声分析 复现说明

估计奖励噪声，并检查集合式 SPO+ 的训练结果。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/EXPERIMENT_SUMMARY.md) |
| 结果 | [NS_SPO_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/NS_SPO_COMPLETE_RESULTS.json) · [NS_TRAINING_RESULTS.json](https://github.com/fallenplum/MAB/blob/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/NS_TRAINING_RESULTS.json) · [JOINT_NOISE_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/JOINT_NOISE_COMPLETE_RESULTS.json) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/REPRODUCE.md) |
| 实验设置 | [NS_SPO_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/NS_SPO_PROTOCOL.json) |
| 补充报告 | [01_THEORY_AND_LITERATURE_AUDIT.md](https://github.com/fallenplum/MAB/blob/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/01_THEORY_AND_LITERATURE_AUDIT.md) |
| 补充报告 | [03_SET_LOSS_TRAINING_RESULTS.md](https://github.com/fallenplum/MAB/blob/19ba4fd5316bd897036622dab6b02fcfd0e036fd/routing_idea2/research/m4_noise_aware_set_learning/03_SET_LOSS_TRAINING_RESULTS.md) |


## 获取代码

```bash
git clone --single-branch --branch m4-noise-aware-set-learning https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`19ba4fd5316bd897036622dab6b02fcfd0e036fd`](https://github.com/fallenplum/MAB/commit/19ba4fd5316bd897036622dab6b02fcfd0e036fd)。

[本分支首页](https://github.com/fallenplum/MAB/tree/m4-noise-aware-set-learning) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
