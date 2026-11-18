# E2 复现说明

保存 E2 全部回放结果和后续 E3 固定协议任务书。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [README.md](https://github.com/fallenplum/MAB/blob/1fe57f540970ead0eac70b20cf1a29571fd30390/README.md) |
| 结果 | [E2_AGGREGATED_RESULTS.json](https://github.com/fallenplum/MAB/blob/1fe57f540970ead0eac70b20cf1a29571fd30390/routing_idea2/release/e2_20261104/E2_AGGREGATED_RESULTS.json) · [E2_AGGREGATED_RESULTS.csv](https://github.com/fallenplum/MAB/blob/1fe57f540970ead0eac70b20cf1a29571fd30390/routing_idea2/release/e2_20261104/E2_AGGREGATED_RESULTS.csv) · [E2_FILE_INDEX.json](https://github.com/fallenplum/MAB/blob/1fe57f540970ead0eac70b20cf1a29571fd30390/routing_idea2/release/e2_20261104/E2_FILE_INDEX.json) |
| 代码 | [evaluate_e2.py](https://github.com/fallenplum/MAB/blob/1fe57f540970ead0eac70b20cf1a29571fd30390/routing_idea2/scripts/evaluate_e2.py) · [evaluate_e3.py](https://github.com/fallenplum/MAB/blob/1fe57f540970ead0eac70b20cf1a29571fd30390/routing_idea2/scripts/evaluate_e3.py) |
| 运行说明 | [README.md](https://github.com/fallenplum/MAB/blob/1fe57f540970ead0eac70b20cf1a29571fd30390/routing_idea2/README.md) |
| 实验设置 | [E0_LOCK.json](https://github.com/fallenplum/MAB/blob/1fe57f540970ead0eac70b20cf1a29571fd30390/routing_idea2/protocol/E0_LOCK.json) |


## 获取代码

```bash
git clone --single-branch --branch e3-locked-mechanism-review-20261104 https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`1fe57f540970ead0eac70b20cf1a29571fd30390`](https://github.com/fallenplum/MAB/commit/1fe57f540970ead0eac70b20cf1a29571fd30390)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e3-locked-mechanism-review-20261104) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
