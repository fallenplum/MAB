# 论文结果表

本页由 `python scripts/build_paper_tables.py` 从归档 JSON/CSV 生成。
脚本仅提取已归档的汇总字段、换算百分比单位，并按原协议恢复任务宏平均；
不访问原题，不重训或重新评分，不计算新统计检验、置信区间或 bootstrap。
逐行来源字段、计算口径与事件单位见 [results.csv](results.csv)。

**不同阶段使用不同题池、训练目标或模型池，不构成可直接排序的统一算法榜单。**
E19/P2/P3/E22 主解码为 1e−8 容差内最大动作集合的均匀策略；E18 沿用原冻结策略。
raw-exact 是原浮点分数的零容差最大集合，
不表示数学精确最优解或可用真值的 oracle。原题数不因生成重复、模型动作或匹配掩码增加。

## E18 独立确认的冻结策略

任务等权；已共同排除不可评分题。

| 范围 | 策略 | 原题数 | 准确率 % |
|---|---|---:|---:|
| 两任务等权 | GlobalFixed | 2998 | 86.863448 |
| 两任务等权 | Ridge | 2998 | 86.663715 |
| 两任务等权 | SPO_coordinate | 2998 | 86.396128 |
| 两任务等权 | TSPO_uniform | 2998 | 86.407877 |

来源：[E18_POINT_RESULTS.json](../routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json)。

## E18 冠军方向干预

这是 TSPO_uniform 与 SPO_coordinate 的历史主比较，不能替换为 P2 的 B/C 干预。

| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |
|---|---|---:|---:|---:|---:|---:|
| 两任务等权 | TSPO_uniform − SPO_coordinate | 2998 | 86.396128 | 86.407877 | +0.011749 | — |
| gsm8k | TSPO_uniform − SPO_coordinate | 1500 | 95.266667 | 94.333333 | -0.933333 | — |
| bbh | TSPO_uniform − SPO_coordinate | 1498 | 77.525590 | 78.482421 | +0.956831 | — |

来源：[E18_POINT_RESULTS.json](../routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json)。

## E19 对称约束开发分析

原 train600 五折外折分析，训练历史重叠且题池长期用于开发。只保留归档点值。

| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |
|---|---|---:|---:|---:|---:|---:|
| 两任务等权 | TSPO+ C − B | 600 | 85.057981 | 86.500000 | +1.442019 | 150 |
| 两任务等权 raw-exact | TSPO+ C − B | 600 | 85.500000 | 86.500000 | +1.000000 | — |
| gsm8k | TSPO+ C − B | 300 | 92.720899 | 93.222222 | +0.501323 | — |
| bbh | TSPO+ C − B | 300 | 77.395062 | 79.777778 | +2.382716 | — |

来源：[E19_COMPLETE_RESULTS.json](../routing_idea2/research/e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json)。

## P2 原系统独立新题

同一原系统的冻结 B/C 比较。总体准确率按源文件中的任务权重恢复；总体差值直接读取归档。主效果包含预定解码的影响。

| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |
|---|---|---:|---:|---:|---:|---:|
| 两任务等权 | TSPO+ C − B | 2400 | 85.354575 | 86.680556 | +1.325980 | 302 |
| gsm8k | TSPO+ C − B | 1200 | 93.751634 | 94.000000 | +0.248366 | — |
| bbh | TSPO+ C − B | 1200 | 76.957516 | 79.361111 | +2.403595 | — |
| 两任务等权 raw-exact | TSPO+ C − B | 2400 | 86.041667 | 86.680556 | +0.638889 | — |

来源：[P2_POINT_RESULTS.json](../routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json)。

## P3 原池 ListNet

原 train600 的跨损失开发机制对照，不是独立新题。

| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |
|---|---|---:|---:|---:|---:|---:|
| 两任务等权 | ListNet C − B | 600 | 85.944444 | 85.666667 | -0.277778 | 135 |
| gsm8k | ListNet C − B | 300 | 93.222222 | 93.555556 | +0.333333 | — |
| bbh | ListNet C − B | 300 | 78.666667 | 77.777778 | -0.888889 | — |

来源：[P3_INTERNAL_LISTNET_RESULTS.json](../routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json)。

## P3 外部单模型池 L3

DARS 缓存上的 GPQA 与 MATH-500；部署选择单模型，测试配置固定为温度 0.7。总体严格换队数为两个任务计数之和。

| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |
|---|---|---:|---:|---:|---:|---:|
| 两任务等权 | TSPO+ L3 C − B | 548 | 57.787634 | 57.932796 | +0.145161 | 113 |
| gpqa | TSPO+ L3 C − B | 248 | 42.741935 | 41.532258 | -1.209677 | 51 |
| math-500 | TSPO+ L3 C − B | 300 | 72.833333 | 74.333333 | +1.500000 | 62 |
| 两任务等权 | ListNet L3 C − B | 548 | 61.456989 | 61.255376 | -0.201613 | 41 |
| gpqa | ListNet L3 C − B | 248 | 47.580645 | 47.177419 | -0.403226 | 13 |
| math-500 | ListNet L3 C − B | 300 | 75.333333 | 75.333333 | +0.000000 | 28 |

来源：[P3_EXTERNAL_DARS_RESULTS.json](../routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json)。

## P3 外部零容差敏感性

按原协议将两个任务的已归档 raw-exact 点值等权汇总；没有重新求解模型。

| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |
|---|---|---:|---:|---:|---:|---:|
| 两任务等权 raw-exact | TSPO+ L3 C − B | 548 | 58.037634 | 57.932796 | -0.104839 | — |
| 两任务等权 raw-exact | ListNet L3 C − B | 548 | 61.456989 | 61.255376 | -0.201613 | — |

来源：[P3_EXTERNAL_DARS_RESULTS.json](../routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json)。

## P3 按任务固定模型基线

各任务分别从自己的训练区选择固定模型，再对两个任务等权汇总。源字段名含 global_fixed，但不表示跨任务共用一个全局模型。

| 范围 | 策略 | 原题数 | 准确率 % |
|---|---|---:|---:|
| gpqa | TaskFixed | 248 | 48.387097 |
| math-500 | TaskFixed | 300 | 76.000000 |
| 两任务等权 | TaskFixed | 548 | 62.193548 |

来源：[P3_EXTERNAL_DARS_RESULTS.json](../routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json)。

## E22 全部行权重控制

B、C、同总权重 Uniform 与每个预固定匹配掩码全部保留。每行差值及严格换队的参照均为同损失 B；B 的零值是自比较。

| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |
|---|---|---:|---:|---:|---:|---:|
| 两任务等权 | TSPO+ B − B | 600 | 85.057981 | 85.057981 | +0.000000 | 0 |
| 两任务等权 | TSPO+ C − B | 600 | 85.057981 | 86.500000 | +1.442019 | 150 |
| 两任务等权 | TSPO+ Uniform − B | 600 | 85.057981 | 85.156746 | +0.098765 | 3 |
| 两任务等权 | TSPO+ Matched2026111801 − B | 600 | 85.057981 | 87.000000 | +1.942019 | 137 |
| 两任务等权 | TSPO+ Matched2026111802 − B | 600 | 85.057981 | 85.833333 | +0.775353 | 143 |
| 两任务等权 | TSPO+ Matched2026111803 − B | 600 | 85.057981 | 85.611111 | +0.553131 | 149 |
| 两任务等权 | ListNet B − B | 600 | 85.944444 | 85.944444 | +0.000000 | 0 |
| 两任务等权 | ListNet C − B | 600 | 85.944444 | 85.666667 | -0.277778 | 135 |
| 两任务等权 | ListNet Uniform − B | 600 | 85.944444 | 85.944444 | +0.000000 | 2 |
| 两任务等权 | ListNet Matched2026111801 − B | 600 | 85.944444 | 86.277778 | +0.333333 | 327 |
| 两任务等权 | ListNet Matched2026111802 − B | 600 | 85.944444 | 85.000000 | -0.944444 | 321 |
| 两任务等权 | ListNet Matched2026111803 − B | 600 | 85.944444 | 85.777778 | -0.166667 | 314 |

来源：[controls.csv](../routing_idea2/research/e22_mechanism_specificity/controls.csv)。

## E22 特异性主对照

MatchedMean 为三份预固定掩码策略的等权平均；不把最优掩码替代平均对照。原开发题池的质量影响因损失和降权位置而异。

| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |
|---|---|---:|---:|---:|---:|---:|
| 两任务等权 | TSPO+ C − Uniform | 600 | 85.156746 | 86.500000 | +1.343254 | — |
| 两任务等权 | TSPO+ C − MatchedMean | 600 | 86.148148 | 86.500000 | +0.351852 | — |
| 两任务等权 | ListNet C − Uniform | 600 | 85.944444 | 85.666667 | -0.277778 | — |
| 两任务等权 | ListNet C − MatchedMean | 600 | 85.685185 | 85.666667 | -0.018519 | — |

来源：[SUMMARY.json](../routing_idea2/research/e22_mechanism_specificity/SUMMARY.json)。

## 主比较的纠错与破坏

事件计数读取归档。整数观测与均匀策略耦合期望分别标注；期望值可能为小数。
任务等权准确率差不能一律由净事件数除以合并分母恢复。

| 阶段 | 比较 | 纠错 | 破坏 | 事件单位 |
|---|---|---:|---:|---|
| E18 | TSPO_uniform − SPO_coordinate | 377.000000 | 376.000000 | 整数题目×生成事件 |
| E19/P1 | TSPO+ C − B | 89.646825 | 63.690476 | 冻结均匀策略耦合期望 |
| E20/P2 | TSPO+ C − B | 379.000000 | 283.529412 | 冻结均匀策略耦合期望 |
| E21/P3 | ListNet C − B | 16.000000 | 21.000000 | 冻结策略耦合期望；本比较两端均唯一 |
| E22 | TSPO+ C − Uniform | 88.924603 | 64.746032 | 冻结策略耦合期望 |
| E22 | TSPO+ C − MatchedMean | 63.000000 | 56.666667 | 冻结策略耦合期望 |
| E22 | ListNet C − Uniform | 16.000000 | 21.000000 | 冻结策略耦合期望 |
| E22 | ListNet C − MatchedMean | 47.666667 | 48.000000 | 冻结策略耦合期望 |

## 输入文件校验值

以下 SHA-256 来自本次构表实际读取的文件字节。表格内容与输入哈希均为确定性输出。

| 输入文件 | SHA-256 |
|---|---|
| [routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json](../routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json) | `91c3070b4a1fde80e3f72a6220e74eacfa25eb2bdf170c7351c79cf874d625a1` |
| [routing_idea2/research/e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json](../routing_idea2/research/e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json) | `b9d0d2d0c340c26572be7441a92c20ea68807c436d252e2c01198a1692812435` |
| [routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json](../routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json) | `54aa84a760e70e5189bf2ed7ee0ba92056365497b409c6d146540ab8e2cf2758` |
| [routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json](../routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json) | `ee01a82450c7c1a6951b007effc856b28a2329a13b610c22b322a242b50396d0` |
| [routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json](../routing_idea2/research/e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json) | `d8dd54fd2f83732acb9f222a5028623d191d535f0b7c8272629f1fb1392cd778` |
| [routing_idea2/research/e22_mechanism_specificity/SUMMARY.json](../routing_idea2/research/e22_mechanism_specificity/SUMMARY.json) | `ec8bcedcbe2b2880e2ef4a1fe70828c53c654553fa8aa0c9ad740e0165fac389` |
| [routing_idea2/research/e22_mechanism_specificity/controls.csv](../routing_idea2/research/e22_mechanism_specificity/controls.csv) | `c097e7449f0d1cee7bab5e89ac604d250a657a2e6644f4a128960230667b05b7` |

## 复核命令

```bash
python scripts/build_paper_tables.py
python scripts/build_paper_tables.py --check
```

`--check` 只在内存中重建并逐字节比较两个输出，不创建目录、不修改文件。
退出码 0 表示一致；1 表示缺少或漂移的输出；2 表示输入、字段或参数错误。
这项检查验证公开汇总到论文表格的映射，不等于复现原始训练、推理与逐题评价。
