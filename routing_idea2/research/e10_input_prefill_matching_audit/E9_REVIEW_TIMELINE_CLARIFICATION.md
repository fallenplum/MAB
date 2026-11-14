# E9 审查时间、原执行布局与复现证据澄清

## 结论

**两份原“事前通过”状态当前无法获得实际审查完成时点的原始日志支持，按事后审查解释。** 两份文件在公开目录与 `private/runtime/audit/` 的归档副本逐字节一致。审查开始与完成均记为 null。原 JSON 的 `created_utc` 作为文件内声明的写入时点保存，其语义与实际审查完成时点分别保留。

本澄清追加于 E10 工作区。原 E9 协议、审查文件、receipt、时间字段和模型均保留原字节。每个已知时间的相对原始路径、字段与完整 SHA-256 保存在同名 JSON。私有来源清单保存所读 429 个文件的哈希，并在完成读取后逐项复核为 PASS。

## 时间线

下表全部为 2026-11-13 UTC。完整路径均相对于原 E9 根目录。

| 事件 | 时间 | 原始证据 |
|---|---|---|
| 两次审查的开始、完成 | null | 未找到带绝对时间的独立审查事件日志 |
| 首次实际拟合开始 | null | FIT_LOG 记录 CPU/wall 时长，未记录绝对开始时间 |
| 最早实际拟合的已知封存上界 | 19:16:07.921613 | private/models/fold0/inner0/B_1/FROZEN_RECEIPT.json，frozen_utc |
| fold0 完成 | 19:16:16.836875 | private/models/fold0/FOLD_FROZEN_RECEIPT.json，completed_utc |
| 编排审查文件写入声明 | 19:17:46.404899 | public/E9_ORCHESTRATOR_REVIEW.json，created_utc |
| 最后一折封存 | 19:19:08.974626 | private/models/fold4/FOLD_FROZEN_RECEIPT.json，completed_utc |
| 全折完成收据 | 19:19:09.116528 | private/head_execution/ALL_FOLDS_COMPLETE.json，completed_utc |
| 统一预测封存 | 19:25:54.081866 | private/EVALUATION_FREEZE_RECEIPT.json，sealed_utc |
| 首次 outer evaluator 奖励读取 | 19:25:54.120852 | private/REWARD_ACCESS_RECEIPT.json，first_outer_evaluator_reward_access_utc |
| 评价源码审查文件写入声明 | 19:27:19.453064 | public/E9_EVALUATION_SOURCE_REVIEW.json，created_utc |

读取 195 组模型 FROZEN_RECEIPT 与 FIT_LOG 后，最早实际头的 FIT_LOG 明确记录 `actual_head_fit=true`。由封存时点可得该次拟合已完成的上界。相对运行时长缺少共同起点，因此首次拟合开始保留 null。

**预测统一封存在评分奖励读取之前。** 两份收据的预测 SHA-256 一致，为 `d8ac242dfa0af454d5b2d3603d8c5fb9383ac07c3873c5cfde8981bc8c1ddc8e`。五折 receipt 均早于统一封存。该证据支持预测冻结顺序。审查时点缺口仍需单独保留。仅由 review 文件创建时刻，无法建立数据泄漏结论。

## 两份原状态的解释

| 原文件 | 原状态 | 本次追加解释 |
|---|---|---|
| E9_ORCHESTRATOR_REVIEW.json | PASS_BEFORE_FORMAL_HEAD_FITS | 当前无法支持，按事后审查解释 |
| E9_EVALUATION_SOURCE_REVIEW.json | PASS_BEFORE_FIRST_OUTER_SCORING | 当前无法支持，按事后审查解释 |

此结论的置信度为高，依据为已保存原始收据、文件内时间声明以及缺失的事件字段。审查实际是否曾更早完成仍属未知。

## 三种复现状态

| 命题 | 已核对状态 | 支持范围 |
|---|---|---|
| 冻结模型评分重放实际运行并通过 | 已支持 | 两份私有 scoring-replay REPRODUCTION_VERIFIED.json 为 PASS_REPRODUCED，models_refitted=false，公开收据对应第二次 |
| full-refit 入口存在 | 已支持 | public/code/e9_reproduce.py 显式提供 full-refit 分支、教师重建、五折训练与科学字段比较 |
| 完整重新拟合已实际执行 | 未证实，字段为 null | 所检查 private/reproductions 的成功收据均为 scoring-replay，未发现 full-refit 执行收据 |

代码存在与真实执行分别由源码和运行 receipt 支持。本次只读检查使用这些现存记录。正式 reproducer 的 `verify` 会写公开核验文件，执行任何模式均有独立运行语义，本次读取源码和 receipt 的审计边界保持可追溯。

## stage wrapper 布局

七份 `public/orchestration/` 源码是原 stage 编排档案。它们与 `private/runtime/` 同名源码逐字节一致。依赖相对源码路径的脚本使用自身 parent 下的 `code`，组装脚本还使用 `math/public`、`audit`、`reports`、`public_qualification`。这些目录存在于原 `private/runtime/` 布局。

公开布局中 `public/code/` 与 `public/orchestration/` 并列，`public/orchestration/code/` 缺失。因此这些 stage 源码按公开位置直接启动时，所需相对依赖存在布局缺口。其用途与历史原执行路径如下。

| wrapper | 原用途 |
|---|---|
| initialize_e9.py | 初始化历史根、协议和父资产清单，使用冻结绝对根路径 |
| qualify_and_start_teachers.py | 冻结教师源码并执行合成资格与启动准备 |
| build_all_teachers.py | 按原 stage/code 路径编排教师子进程 |
| freeze_heads.py | 冻结并复制正式头源码 |
| train_remaining_folds.py | 在 fold0 完成后编排其余头训练 |
| freeze_and_evaluate.py | 冻结评价源码后进入历史统一评分 |
| assemble_delivery.py | 按原 stage 的 code/math/audit/reports 组装交付 |

正式 `public/code/e9_reproduce.py` 的导入与运行位于同一 code 目录，源码未依赖上述 stage wrappers。它继续依赖原私有资产根和既有环境。这里通过入口说明记录布局与副作用，路径适配状态记为未新增。已检查范围内的布局事实置信度为高，full-refit 完整运行仍需执行证据。

## 优势与劣势（风险）

**优势**为逐时点绑定来源哈希、未知时间保留 null、评分冻结链可独立核对，并对归档布局和正式入口给出可复查范围。

**劣势（风险）**为原始 review 事件日志缺失、FIT_LOG 缺少绝对起止、完整重拟合缺少实际执行收据。本次审计依赖现存 receipt 的记录真实性，对缺失历史事件的事实保留未知。

## 拓展思考与可能想知道

若后来找到原始 review 事件日志，可在追加记录中登记日志路径、哈希与其支持的实际时间，并保留本次澄清版本。历史文件中的状态与时间继续可追溯。
