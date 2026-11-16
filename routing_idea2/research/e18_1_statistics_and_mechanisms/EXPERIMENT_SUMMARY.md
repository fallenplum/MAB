# E18.1统计与机制分析总结

## 主要结论

| TSPO比较对象 | 点差pp | 题级Bootstrap95%CI | 原配对t p | 原次要Holm p | 纠错 | 破坏 |
|---|---|---|---|---|---|---|
| SPO_coordinate | 0.012 | [-0.667, 0.690] | 0.973048 | 未定义 | 377 | 376 |
| Ridge | -0.256 | [-0.845, 0.334] | 0.394418 | 1.000000 | 269 | 292 |
| GlobalFixed | -0.456 | [-1.145, 0.234] | 0.200390 | 0.994965 | 344 | 385 |

主比较的总体点差接近零，区间包含零。任务交互为1.890pp，95%CI[0.533, 3.247]，p=0.006619。GSM8K下降、BBH上升的抵消需要与总体结果同时呈现。

SPO组合0占比83.02%，TSPO降至4.64%，但这种集中度变化尚未表现为总体服务质量改善。相对GlobalFixed，TSPO离开组合5的2,720题中纠错344次、破坏385次。

## 统计与数据边界

20,000次按任务分层的题级配对Bootstrap保留同题三seed，主权重50/50。子任务全部报告，边际区间和Holm调整p分别呈现。E18.1是观察E18后进行的统计补充。E17的100次重编号是训练规则重复，独立评价支持数仍为2,998个可评分问题。

原计划另有两道不可评分BBH题，全部执行支出保留。原E18统计、确定性缺失范围和负面结果完整继承。新增LLM调用、模型训练与调参均为0。

## 文件

- 01_STATISTICAL_REPORT.md给出主区间与推断口径。
- 02_TASK_HETEROGENEITY.md给出交互、seed和27子任务。
- 03_SELECTION_AND_QUALITY.md给出熵、HHI与真实动作变化。
- 04_TSPO_GLOBAL_FIXED.md给出344/385的任务分解。
- 05_ACCOUNTING_AND_GLOBAL_FIXED.md核对执行账目与训练来源。
- 06_JOINT_EVIDENCE.md和07_RESULTS_AND_DISCUSSION.md组织论文证据与草稿。
- E18_1_COMPLETE_RESULTS.json、全部CSV及私有BOOTSTRAP_REPLICATES.npz保存结果。
