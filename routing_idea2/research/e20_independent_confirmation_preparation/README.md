# E20独立确认准备

## 完成状态

本阶段完成最终train600 B/C训练、输入侧策略冻结、数据权限与近重复审计、配对功效规划、精确调用预算及P2执行前审查。入口汇总为EXPERIMENT_SUMMARY.md，机器可读结果为E20_COMPLETE_RESULTS.json。

正式独立评价等待用户批准。fresh400已有E18奖励曝光，sealed200保持答案封存并有1道近重复预警待处置。新增独立题池仅提出2400题方案，尚未选题或入池。

## 报告索引

| 文件 | 用途 |
|---|---|
| 01_FINAL_MODEL_FREEZE.md | 最终B/C训练配置、正则来源、优化资格与冻结收据 |
| 02_DATA_PERMISSIONS.md | fresh/sealed来源、曝光历史、去重及新题来源余量 |
| 03_POWER_ANALYSIS.md | 双侧功效、目标效应与样本量、199题待审批情景 |
| 04_EXECUTION_BUDGET.md | B/C正概率支持并集、成员及聚合调用、缓存与Token |
| 05_P2_FROZEN_PROTOCOL.md | 主比较、部署、权限、执行顺序和停止条件 |
| 06_READINESS_REVIEW.md | 工程、独立性、功效与执行批准的审查结论 |
| E20_PROTOCOL.json | 完整模型与P2规则、来源和候选题池方案 |
| E20_PROTOCOL_FREEZE_RECEIPT.json | 最终协议与源码的固定哈希 |

## 私有资产

工件根目录为`E:\语言模型组合路由\research\e20_independent_confirmation_preparation\private`。

- `final_models/FINAL600_MODELS_FROZEN.npz`包含两支权重与train600预处理状态。
- `final_models/FINAL600_PREFIT_CONFIG_FROZEN.json`与`FINAL600_MODEL_FREEZE_RECEIPT.json`记录训练前与训练后冻结。
- `final_models/*_INPUT_ONLY_PREDICTIONS_ORIGINAL_OPERATOR_FROZEN.npz`是正式两池输入预测。
- `inputs/`保存隔离sealed输入、冻结MiniLM向量、修复前后元数据及权限收据。
- `budget/`保存逐题所需、可复用及尚缺的成员/聚合执行键。

原题、标准答案、逐题评分、向量、权重和真实身份映射保留于私有目录。公开目录只含源码、来源哈希、脱敏汇总与报告。

## 历史输入依赖

本轮需要原train600输入与真实组合评分、E17冻结full600配置和日志、E19操作符及OOF配对效用、E18输入排除索引和闭合缓存成本元数据。完整路径与源SHA在相应JSON中列明。运行预算复核需要D盘现有E18运行时的元数据账本或其已闭合私有归档。

源码依赖父提交中的同级`e19_symmetric_tie_constraint/weighted_objectives.py`与`e18_mechanism_and_confirmation_preparation/`。同时支持原本地`experiments/`布局。请从研究分支checkout的同级研究目录运行，保留这些父阶段源码。私有资产盘和冻结编码器为必要运行依赖。

## 当前边界

新增预测头拟合2次，冻结MiniLM编码200题，新增成员生成0，新增聚合0，fresh奖励解码0，sealed答案解码0。按最新项目约定，本轮新置信区间和bootstrap均为0。功效是基于E19开发方差的解析规划。

一次输入元数据拒绝及修复、原数值算子回放、预算状态大小写预检修复和审查端精度对照更正均保留记录。实际LLM评价未启动。
