# E22 复现说明

比较全并列行干预、相同总权重和匹配删除控制。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [AUDIT_REPORT.md](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/AUDIT_REPORT.md) |
| 结果 | [RESULTS.json](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/RESULTS.json) · [SUMMARY.json](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/SUMMARY.json) · [controls.csv](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/controls.csv) |
| 代码 | [run_audit.py](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/run_audit.py) · [row_weighted.py](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/row_weighted.py) · [geometry.py](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/geometry.py) |
| 运行说明 | [README.md](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/README.md) |
| 实验设置 | [PROTOCOL.json](https://github.com/fallenplum/MAB/blob/8f490a06ebe1153be2fcbeb42ad3944d63075b7d/routing_idea2/research/e22_mechanism_specificity/PROTOCOL.json) |


## 获取代码

```bash
git clone --single-branch --branch e22-mechanism-specificity https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`8f490a06ebe1153be2fcbeb42ad3944d63075b7d`](https://github.com/fallenplum/MAB/commit/8f490a06ebe1153be2fcbeb42ad3944d63075b7d)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e22-mechanism-specificity) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
