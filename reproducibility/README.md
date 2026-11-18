# 少样本协议修正 复现说明

修正训练和评价边界后，重新比较少量反馈的作用。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/EXPERIMENT_SUMMARY.md) |
| 结果 | [EXPERIMENT_RESULTS.json.gz](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/EXPERIMENT_RESULTS.json.gz) · [TASK_RESULTS.json.gz](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/TASK_RESULTS.json.gz) · [COSTS.json.gz](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/COSTS.json.gz) |
| 代码 | [run_corrected.py](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/run_corrected.py) · [models.py](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/models.py) · [stats.py](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/stats.py) |
| 运行说明 | [PROTOCOL.md](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/PROTOCOL.md) |
| 实验设置 | [PROTOCOL.md](https://github.com/fallenplum/MAB/blob/7a21e158fabe56acbe3a40a3dd0e09c139c6c493/routing_idea2/research/fewshot_protocol_fix_20261107/PROTOCOL.md) |


## 获取代码

```bash
git clone --single-branch --branch fewshot-protocol-fix-20261107 https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`7a21e158fabe56acbe3a40a3dd0e09c139c6c493`](https://github.com/fallenplum/MAB/commit/7a21e158fabe56acbe3a40a3dd0e09c139c6c493)。

[本分支首页](https://github.com/fallenplum/MAB/tree/fewshot-protocol-fix-20261107) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
