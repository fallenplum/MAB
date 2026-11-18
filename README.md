# MAB 模型组合路由

**当前论文主线是 E17–E23。全部 E1–E23 及补充实验见 [全部实验](EXPERIMENTS.md) 和 [实验目录](experiments/README.md)。** 各阶段的报告、源码和协议保留在对应实验分支，目录提供直接入口。

项目研究有限次生成产生经验并列奖励时，训练目标怎样通过共享参数改变其他问题的动作排序，以及部署时的选择规则怎样影响最终答案质量。

## 常用入口

| 内容 | 入口 |
|---|---|
| 论文结论与证据 | [主张与证据](paper/CLAIMS_AND_EVIDENCE.md) |
| 论文结果表 | [结果与来源](paper/RESULTS.md) · [CSV](paper/results.csv) |
| 研究进展与方向变化 | [研究历史](docs/HISTORY.md) |
| 实验分支 | [分支索引](docs/BRANCHES.md) |
| 环境、数据和执行步骤 | [复现说明](docs/REPRODUCIBILITY.md) |

## 核心设定

原系统从六个冻结语言模型中选择三个，共 20 个组合。成员独立回答，由冻结的 Qwen3-8B 聚合器输出最终答案，以答案正确率评价。路由发生在生成前，只读取问题表征。

B/C 对照使用冻结 MiniLM 表征、训练区拟合的 PCA16 和标准化，以及带截距的 17×20 线性头。训练区中，所有动作经验平均奖励相同的题目称为全并列行。**B 保留这些行的损失，C 将其权重设为零**；其他训练行、归一化分母、表示和正则保持相同。严格选择变化指在同一预定选择规则下，两边均有唯一最高分动作，且所选动作不同。

E23 进一步比较部署规则：在原预测分数距最大值不超过 `1e-8` 的候选中，优先选择 train600 三次生成累计成功数最高的组合，计数仍并列时均匀选择。该规则得到 B_prior，C 的全部动作保持原样。具体定义见 [E23 协议](routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json)。

## 主要结果

准确率差以百分点计，各阶段的数据和比较对象如下。

| 阶段 | 评价数据 | 主要结果 |
|---|---|---|
| [E17](routing_idea2/research/e17_supervision_distortion_audit/EXPERIMENT_SUMMARY.md) | 原 600 题开发集 | 100 次重编号后，按编号处理并列的 SPO+ 实际组合选择一致率约 6.13%；Ridge 与均匀处理并列的 TSPO+ 为 100% |
| [E18](routing_idea2/research/e18_confirmatory_execution/final_results/E18_RESEARCH_CONCLUSION.md) | 2,998 道可评分独立问题 | TSPO_uniform 对 SPO_coordinate 的准确率差为 +0.011749，纠错 377、破坏 376 |
| [E19 / P1](routing_idea2/research/e19_symmetric_tie_constraint/03_P1_FIVE_FOLD_RESULTS.md) | 原 600 题五折分析 | B/C 在 150 题上发生严格选择变化，C−B 为 +1.442 |
| [E20 / P2](routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) | 2,400 道独立新题 | 原冻结策略 B 85.355%、C 86.681%，C−B 为 +1.326；302 题发生严格选择变化 |
| [E21 / P3](routing_idea2/research/e21_p3_external_mechanism/P3_RESULTS.md) | 原 600 题与外部 548 题 | 原集 ListNet 的 C−B 为 −0.278；外部 L=3 的 TSPO+ / ListNet 分别为 +0.145 / −0.202 |
| [E22](routing_idea2/research/e22_mechanism_specificity/AUDIT_REPORT.md) | 原 600 题五折匹配控制 | C 相对三份匹配控制平均，TSPO+ 为 +0.352、ListNet 为 −0.019 |
| [E23](routing_idea2/research/e23_deployment_prior_and_references/EXPERIMENT_SUMMARY.md) | 已曝光 P2 原 2,400 题的补充比较 | B_prior **86.50000%**、C **86.68056%**，差 **+0.18056**；Ridge **86.87500%**；C 对 MatchedMean 为 **+0.52315** |

E23 只改进 B 的部署规则，就将其准确率提高 1.14542 个百分点。原先发生严格选择变化的 302 道题对 C−B_prior 的贡献仍为 +0.26389，原 B 近并列区的贡献为 −0.08333。训练约束和部署规则共同决定了最终的准确率差；Ridge 在本轮对照方法中的总体准确率最高。[完整结果与分组比较](routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)

P2 的 raw-exact 使用浮点分数的零容差最大集合，准确率差为 +0.639。P3 外部实验为六模型单选，其 62.194% 固定基线由各任务分别在训练区选模型，再等权汇总。E23 的四个新控制头使用全部 600 题拟合；MatchedMean 是三份预定匹配策略的等权混合，与 E22 的五折控制分别报告。均匀或混合策略的纠错、破坏按期望计数。

P2 原评价使用 147,825 次调用、116,097,715 tokens。E23 补充 1,770 次成员生成和 4,263 次聚合，共 **6,033 次调用、4,486,492 tokens**。这些费用用于评价多份冻结策略的支持集合，账目见 [E23 最终汇总](routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)。

## 复现

在仓库根目录检查文件与公开结果表：

```bash
python scripts/verify_repository.py
python scripts/build_paper_tables.py --check
```

E18/P2 全生成链尚缺 `typed_aggregation_execution.py`、`typed_operator_contract.py`、`stage1_review_common.py` 和 `budget_router_v2/scoring.py` 四个源文件；完整重训所需的环境、私有资产和各阶段步骤见 [复现说明](docs/REPRODUCIBILITY.md)。
