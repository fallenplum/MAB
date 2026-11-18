# E21/P3 复现说明

用 ListNet 和外部 DARS 缓存检查机制与质量差异。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [P3_RESULTS.md](https://github.com/fallenplum/MAB/blob/692e481ee344d1dd7632c77633716882de599c55/routing_idea2/research/e21_p3_external_mechanism/P3_RESULTS.md) |
| 结果 | [P3_INTERNAL_LISTNET_RESULTS.json](https://github.com/fallenplum/MAB/blob/692e481ee344d1dd7632c77633716882de599c55/routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json) · [P3_EXTERNAL_DARS_RESULTS.json](https://github.com/fallenplum/MAB/blob/692e481ee344d1dd7632c77633716882de599c55/routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json) |
| 代码 | [run_internal.py](https://github.com/fallenplum/MAB/blob/692e481ee344d1dd7632c77633716882de599c55/routing_idea2/research/e21_p3_external_mechanism/run_internal.py) · [run_external.py](https://github.com/fallenplum/MAB/blob/692e481ee344d1dd7632c77633716882de599c55/routing_idea2/research/e21_p3_external_mechanism/run_external.py) · [endpoints.py](https://github.com/fallenplum/MAB/blob/692e481ee344d1dd7632c77633716882de599c55/routing_idea2/research/e21_p3_external_mechanism/endpoints.py) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/692e481ee344d1dd7632c77633716882de599c55/routing_idea2/research/e21_p3_external_mechanism/REPRODUCE.md) |
| 实验设置 | [P3_PROTOCOL.json](https://github.com/fallenplum/MAB/blob/692e481ee344d1dd7632c77633716882de599c55/routing_idea2/research/e21_p3_external_mechanism/P3_PROTOCOL.json) |


## 获取代码

```bash
git clone --single-branch --branch e21-p3-external-mechanism https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`692e481ee344d1dd7632c77633716882de599c55`](https://github.com/fallenplum/MAB/commit/692e481ee344d1dd7632c77633716882de599c55)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e21-p3-external-mechanism) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
