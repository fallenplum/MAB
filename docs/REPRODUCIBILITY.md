# E2/E3 复现说明

保存固定 K=3 的 E2/E3 任务书和运行代码。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [CODEX_TASK_A_E2_E3.md](https://github.com/fallenplum/MAB/blob/3750e75711a6877a2258020be3db8c52f4c7dd37/routing_idea2/CODEX_TASK_A_E2_E3.md) |
| 代码 | [evaluate_e2.py](https://github.com/fallenplum/MAB/blob/3750e75711a6877a2258020be3db8c52f4c7dd37/routing_idea2/scripts/evaluate_e2.py) · [evaluate_e3.py](https://github.com/fallenplum/MAB/blob/3750e75711a6877a2258020be3db8c52f4c7dd37/routing_idea2/scripts/evaluate_e3.py) · [budget_router](https://github.com/fallenplum/MAB/tree/3750e75711a6877a2258020be3db8c52f4c7dd37/routing_idea2/src/budget_router) |
| 运行说明 | [README.md](https://github.com/fallenplum/MAB/blob/3750e75711a6877a2258020be3db8c52f4c7dd37/routing_idea2/README.md) |
| 实验设置 | [E0_LOCK.json](https://github.com/fallenplum/MAB/blob/3750e75711a6877a2258020be3db8c52f4c7dd37/routing_idea2/protocol/E0_LOCK.json) |


## 获取代码

```bash
git clone --single-branch --branch a-fixed-k3-e2-e3 https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`3750e75711a6877a2258020be3db8c52f4c7dd37`](https://github.com/fallenplum/MAB/commit/3750e75711a6877a2258020be3db8c52f4c7dd37)。

[本分支首页](https://github.com/fallenplum/MAB/tree/a-fixed-k3-e2-e3) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
