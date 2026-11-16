# E18.1 TSPO与GlobalFixed机制分解

## 主对照结果

GlobalFixed始终选择训练区确定的组合5。训练来源与排除题成本见05_ACCOUNTING_AND_GLOBAL_FIXED.md。TSPO相对GlobalFixed的任务宏平均差为−0.455570pp，题级95%CI为[-1.145, 0.234]pp，原配对t检验p=0.200390。

TSPO共有344次纠错、385次破坏，净少41条正确生成事件。两种策略在相同题和相同生成重复上配对。

## 何时离开组合5

| 范围 | 动作 | 题数 | 该范围占比% | 条件点差pp | 纠错 | 破坏 | 净纠错 |
|---|---|---|---|---|---|---|---|
| all_evaluable | stay_combination5 | 278 | 9.27 | 0.000 | 0 | 0 | 0 |
| all_evaluable | leave_combination5 | 2720 | 90.73 | -0.502 | 344 | 385 | -41 |
| gsm8k | stay_combination5 | 116 | 7.73 | 0.000 | 0 | 0 | 0 |
| gsm8k | leave_combination5 | 1384 | 92.27 | -0.963 | 64 | 104 | -40 |
| bbh | stay_combination5 | 162 | 10.81 | 0.000 | 0 | 0 | 0 |
| bbh | leave_combination5 | 1336 | 89.19 | -0.025 | 280 | 281 | -1 |

TSPO离开组合5共2,720题，占90.73%。留在组合5的278题上，两策略完全复用相同执行结果，纠错和破坏均为0。全部344次纠错和385次破坏均发生在离开组合5的题上。

表中条件点差使用当前子集原题的自然平均。离开集合中GSM8K与BBH题数不相等，其条件点差−0.502451pp具有该子集口径。主结果−0.455570pp继续使用全体可评分题及固定任务50/50权重。

GSM8K离开组合5后的64次纠错伴随104次破坏，净−40。BBH为280次纠错和281次破坏，净−1。整体质量代价主要由当前GSM8K样本贡献，BBH内部还存在相互抵消的子任务差异。

## BBH全部子任务分解

| BBH子任务 | 离开5题数 | 占子任务% | 纠错 | 破坏 | 净纠错 | 条件点差pp |
|---|---|---|---|---|---|---|
| boolean_expressions | 56 | 100.00 | 0 | 4 | -4 | -2.381 |
| causal_judgement | 52 | 94.55 | 12 | 9 | 3 | 1.923 |
| date_understanding | 56 | 100.00 | 4 | 9 | -5 | -2.976 |
| disambiguation_qa | 55 | 100.00 | 29 | 19 | 10 | 6.061 |
| dyck_languages | 18 | 32.14 | 0 | 14 | -14 | -25.926 |
| formal_fallacies | 55 | 100.00 | 10 | 12 | -2 | -1.212 |
| geometric_shapes | 55 | 100.00 | 9 | 12 | -3 | -1.818 |
| hyperbaton | 56 | 100.00 | 4 | 14 | -10 | -5.952 |
| logical_deduction_five_objects | 55 | 100.00 | 9 | 15 | -6 | -3.636 |
| logical_deduction_seven_objects | 56 | 100.00 | 7 | 9 | -2 | -1.190 |
| logical_deduction_three_objects | 41 | 74.55 | 3 | 1 | 2 | 1.626 |
| movie_recommendation | 52 | 96.30 | 24 | 8 | 16 | 10.256 |
| multistep_arithmetic_two | 49 | 85.96 | 2 | 1 | 1 | 0.680 |
| navigate | 56 | 100.00 | 5 | 8 | -3 | -1.786 |
| object_counting | 56 | 100.00 | 11 | 14 | -3 | -1.786 |
| penguins_in_a_table | 55 | 100.00 | 1 | 13 | -12 | -7.273 |
| reasoning_about_colored_objects | 53 | 94.64 | 1 | 5 | -4 | -2.516 |
| ruin_names | 54 | 100.00 | 63 | 9 | 54 | 33.333 |
| salient_translation_error_detection | 56 | 100.00 | 17 | 25 | -8 | -4.762 |
| snarks | 55 | 100.00 | 31 | 23 | 8 | 4.848 |
| sports_understanding | 51 | 92.73 | 14 | 20 | -6 | -3.922 |
| temporal_sequences | 42 | 76.36 | 3 | 4 | -1 | -0.794 |
| tracking_shuffled_objects_five_objects | 44 | 78.57 | 2 | 5 | -3 | -2.273 |
| tracking_shuffled_objects_seven_objects | 44 | 80.00 | 8 | 11 | -3 | -2.273 |
| tracking_shuffled_objects_three_objects | 36 | 64.29 | 0 | 6 | -6 | -5.556 |
| web_of_lies | 56 | 100.00 | 5 | 10 | -5 | -2.976 |
| word_sorting | 22 | 39.29 | 6 | 1 | 5 | 7.576 |

所有任务组均保留，表格用于解释已冻结决策的得失分布。它未参与本轮模型或默认组合选择。

## 条件分组的局限

留在组合5的题与离开组合5的题由冻结路由器选择，两个子集的难度与内容可能不同。其绝对准确率差不能归因于离开动作本身。可解释的服务差是同题TSPO与GlobalFixed的配对结果，现有结果尚未支持离开组合5带来总体正收益。
