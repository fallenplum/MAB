# E3 复现说明

在相同预测器与预算下比较决策导向、随机和信息增益采集。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [07_RESEARCH_CONCLUSION.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/07_RESEARCH_CONCLUSION.md) |
| 结果 | [E3_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/E3_COMPLETE_RESULTS.json) |
| 代码 | [run_e3.py](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/run_e3.py) · [acquisition.py](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/acquisition.py) · [predictors.py](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/predictors.py) |
| 运行说明 | [README.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/README.md) |
| 实验设置 | [E3_PROTOCOL.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/E3_PROTOCOL.md) |
| 补充报告 | [01_FEEDBACK_RESPONSE.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/01_FEEDBACK_RESPONSE.md) |
| 补充报告 | [02_ACQUISITION_BASELINES.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/02_ACQUISITION_BASELINES.md) |
| 补充报告 | [03_CROSS_PREDICTOR_RESULTS.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/03_CROSS_PREDICTOR_RESULTS.md) |
| 补充报告 | [04_MECHANISM_ABLATIONS.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/04_MECHANISM_ABLATIONS.md) |
| 补充报告 | [05_COST_AND_REUSE.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/05_COST_AND_REUSE.md) |
| 补充报告 | [06_STATISTICAL_ANALYSIS.md](https://github.com/fallenplum/MAB/blob/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6/routing_idea2/research/decision_acquisition_e3/06_STATISTICAL_ANALYSIS.md) |


## 获取代码

```bash
git clone --single-branch --branch decision-acquisition-e3-20261107 https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6`](https://github.com/fallenplum/MAB/commit/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6)。

[本分支首页](https://github.com/fallenplum/MAB/tree/decision-acquisition-e3-20261107) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
