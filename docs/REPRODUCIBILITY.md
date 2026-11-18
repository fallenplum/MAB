# E8 M4 检查 复现说明

检查组合集中、并列冠军规则及题目内容的作用。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/EXPERIMENT_SUMMARY.md) |
| 结果 | [M4_MECHANISM_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/M4_MECHANISM_COMPLETE_RESULTS.json) · [M4_DISTRIBUTION_AND_TIE_RESULTS.json](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/M4_DISTRIBUTION_AND_TIE_RESULTS.json) · [M4_STABLE_GAP_RESULTS.json](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/M4_STABLE_GAP_RESULTS.json) |
| 代码 | [code](https://github.com/fallenplum/MAB/tree/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/code) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/REPRODUCE.md) |
| 实验设置 | [M4_MECHANISM_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/M4_MECHANISM_PROTOCOL.json) |
| 补充报告 | [01_SELECTION_DISTRIBUTION_AND_TIE_AUDIT.md](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/01_SELECTION_DISTRIBUTION_AND_TIE_AUDIT.md) |
| 补充报告 | [02_STABLE_NONZERO_GAPS.md](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/02_STABLE_NONZERO_GAPS.md) |
| 补充报告 | [03_CONTENT_AND_INDEPENDENT_CONFIRMATION_PLAN.md](https://github.com/fallenplum/MAB/blob/0dece5c291c5ec644f690f892638f81543aed691/routing_idea2/research/e8_m4_mechanism_audit/03_CONTENT_AND_INDEPENDENT_CONFIRMATION_PLAN.md) |


## 获取代码

```bash
git clone --single-branch --branch e8-m4-mechanism-audit https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`0dece5c291c5ec644f690f892638f81543aed691`](https://github.com/fallenplum/MAB/commit/0dece5c291c5ec644f690f892638f81543aed691)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e8-m4-mechanism-audit) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
