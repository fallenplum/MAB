# E20/P2 复现说明

在原系统的 2400 道新题上评价冻结 B/C 策略。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [P2_FINAL_RESULTS.md](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) |
| 结果 | [P2_POINT_RESULTS.json](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json) · [P2_FINAL_DIAGNOSTICS.json](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/P2_FINAL_DIAGNOSTICS.json) · [P2_FINAL_EXECUTION_LEDGER.json](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/P2_FINAL_EXECUTION_LEDGER.json) |
| 代码 | [blind_evaluation.py](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/blind_evaluation.py) · [score_paid_outputs.py](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/score_paid_outputs.py) · [p2_execution_adapter.py](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/p2_execution_adapter.py) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/REPRODUCE.md) |
| 实验设置 | [P2_PRESELECTION_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/21209351651a76abc782662c00ff4523703bbd8a/routing_idea2/research/e20_p2_execution/P2_PRESELECTION_PROTOCOL.json) |


## 获取代码

```bash
git clone --single-branch --branch e20-p2-independent-confirmation https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`21209351651a76abc782662c00ff4523703bbd8a`](https://github.com/fallenplum/MAB/commit/21209351651a76abc782662c00ff4523703bbd8a)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e20-p2-independent-confirmation) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
