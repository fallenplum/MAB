# E9 条件均值监督与成本敏感交叉熵

## 当前状态

本轮完成P1数学与权限资格、正式2×2矩阵、全部45个内容控制、统一评分和独立复算。两项主比较均未获得Holm校正后的正向收益支持。全部结论属于重复使用的train600上的探索性开发证据。

本机资产根目录为 `E:\语言模型组合路由`，当前研究目录为 `research\e9_consistent_cost_sensitive_learning`。Git分支为 `codex/e9-consistent-cost-sensitive-learning`。

## 重要文件

| 文件 | 内容 |
|---|---|
| `EXPERIMENT_SUMMARY.md` | 两主问题、四格机制、分任务结果、内容控制、功效与决定 |
| `MECHANISM_AND_STOP_DECISION.md` | 失败假说、证据范围及当前停止决定 |
| `LITERATURE_AND_TARGET.md` | CSCE一手来源、MSE与真实regret目标、数学证明和实现边界 |
| `E9_PROTOCOL.json` | 不可覆盖的预声明协议，私有收据保存初始哈希 |
| `E9_MAIN_RESULTS.json` | 19个方法/控制的聚合指标、26组比较、超参数和195个拟合槽位日志 |
| `E9_PRIMARY_STATISTICS.json` | C−A、B−A的普通95%配对区间及Holm调整p |
| `E9_CONTENT_CONTROLS.json` | 三次独立置乱各自结果与敏感性解释 |
| `COST.json` | 正式训练、教师、资格、历史反馈、失败与缺失计时分账 |
| `MAIN_SUMMARY.csv`、`COMPARISONS.csv`、`GRID.csv`、`CONTENT_CONTROLS.csv` | 可直接查看的汇总表 |
| `E9_ARCHIVE_VERIFICATION.json` | 三个父研究哈希保全、模型和教师恢复验证 |
| `E9_FINAL_SCIENTIFIC_REVIEW.json` | 独立统计复算及时间顺序核查 |
| `REPRODUCE.md` | 真实运行命令、依赖、重放与完整重算 |

## 方法与训练范围

A为经完整过程核验后复用的经验Ridge，B为后验均值Ridge，C为经验CSCE，D为后验均值CSCE。主比较固定为C−A和B−A，D相关比较属于机制层。

每折480题历史区中使用原三折320/160划分。B/D的内层教师在对应320题内再交叉拟合，支持213/214题的先验拟合区。每个合法拟合区单独估计PCA和标准化器。全部学生使用17×20线性系数，常数项受罚。

原三重复均值是经验标签，后验标签使用既有工作模型的512样本均值。教师只调用冻结的联合先验估计与有界采样，E9不使用近优集合或组合胜者概率作为标签。

## 实现与一手来源

CSCE源于Mozannar与Sontag的ICML2020论文式(3)和命题1。本轮按奖励到成本的映射实现已有目标，数学核、头拟合器、合法教师和评价器分别进行核查。[论文记录](https://proceedings.mlr.press/v119/mozannar20b.html)

论文记录链接的作者仓库描述了学习转交专家的专门任务与LCE形式。本轮的20组合线性头是按一般CSCE公式实现的任务适配，作者仓库核查范围包括官方README，完整原论文实验复现未作为本轮交付。[作者仓库](https://github.com/clinicalml/learn-to-defer)

## 私有资产清单

| 私有资产 | 角色 |
|---|---|
| 原E8 `private/qualification/fold_packs` | 冻结历史、输入、身份与外折划分 |
| 原NS `private/runtime/noise/private/fold*/POSTERIOR_TRAINING.npz` | 经来源核验的outer512样本均值 |
| E9 `private/teachers` | 30个新合法教师、重建先验、索引、置乱来源与收据 |
| E9 `private/models` | 195槽位的权重、特征、分数、动作、日志与冻结收据 |
| E9 `private/ALL_E9_PREDICTIONS_FROZEN.npz` | 统一评价前封存的主与控制预测 |
| E9 `private/E9_PAIRED_OUTCOMES.npz` | 原题级配对真实评分及分组 |
| E9 `private/PARENT_SNAPSHOTS.json` | 原E8/M4/NS共1652文件的初始哈希 |

公开产物只包含源码、聚合统计、配置和报告。题目文本、答案、真实私有矩阵与系数保存在私有资产中。

## 阻塞与局限

已授权的本轮实验交付没有技术阻塞。个别早期资格失败和审查开销只有进程级计时，算子独立耗时以空值记录。全部正式拟合和新教师的日志齐备。

科学局限包括旧开发题多次曝光、外折历史重叠、每题三生成、固定表征与有限网格。三次置乱的检验分辨率有限，成本敏感softmax具有归一权重解释，正确率预测需要单独的任务定义。
