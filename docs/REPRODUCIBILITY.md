# E5 复现说明

核对旧稿中的理论构造、预测误差和预算配置方法。

## 文件入口

| 内容 | 文件 |
|---|---|
| 报告 | [EXPERIMENT_SUMMARY.md](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/EXPERIMENT_SUMMARY.md) |
| 结果 | [E5_COMPLETE_RESULTS.json](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/E5_COMPLETE_RESULTS.json) · [CENTERED_DIAGNOSTICS_RESULTS.json](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/CENTERED_DIAGNOSTICS_RESULTS.json) · [SWITCH_RANKING_RESULTS.json](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/SWITCH_RANKING_RESULTS.json) |
| 代码 | [run_base.py](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/run_base.py) · [run_gates.py](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/run_gates.py) · [run_allocation_synthetic.py](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/run_allocation_synthetic.py) |
| 运行说明 | [REPRODUCE.md](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/REPRODUCE.md) |
| 实验设置 | [E5_EXPERIMENT_CONFIG.json](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/E5_EXPERIMENT_CONFIG.json) |
| 补充报告 | [01_THEORY_IMPLEMENTATION_CHECKS.md](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/01_THEORY_IMPLEMENTATION_CHECKS.md) |
| 补充报告 | [02_CENTERED_ERROR_DIAGNOSTICS.md](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/02_CENTERED_ERROR_DIAGNOSTICS.md) |
| 补充报告 | [03_SWITCH_RANKING_AND_MATCHED_CONTROLS.md](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/03_SWITCH_RANKING_AND_MATCHED_CONTROLS.md) |
| 补充报告 | [04_CERTIFICATION_VALIDITY_AND_POWER.md](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/04_CERTIFICATION_VALIDITY_AND_POWER.md) |
| 补充报告 | [05_TWO_RESOURCE_ALLOCATION.md](https://github.com/fallenplum/MAB/blob/7e06c632787f1de093732aed0250c8a30d3035b7/routing_idea2/research/e5_paired_gain_validation/05_TWO_RESOURCE_ALLOCATION.md) |


## 获取代码

```bash
git clone --single-branch --branch e5-paired-gain-validation https://github.com/fallenplum/MAB.git
cd MAB
```

按本阶段运行说明准备依赖、数据和模型文件，再执行其中的命令。原实验中的参数、数据划分、随机种子和输出文件位置见上面的实验设置与代码。

原实验版本为 [`7e06c632787f1de093732aed0250c8a30d3035b7`](https://github.com/fallenplum/MAB/commit/7e06c632787f1de093732aed0250c8a30d3035b7)。

[本分支首页](https://github.com/fallenplum/MAB/tree/e5-paired-gain-validation) · [全部实验](https://github.com/fallenplum/MAB/blob/main/EXPERIMENTS.md) · [论文复现说明](https://github.com/fallenplum/MAB/blob/main/docs/REPRODUCIBILITY.md)
