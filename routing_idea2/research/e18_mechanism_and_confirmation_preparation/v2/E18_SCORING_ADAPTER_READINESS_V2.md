# E18 V2 稀疏评分与资源账本准备

## 当前状态

**人工 toy reference、FakeBackend 和 FakeTokenizer 的端到端资格通过 15 项检查。GPU 与真实数据执行均未实测。** 真实 fresh/sealed 答案读取为 0，真实模型推理为 0。

`e18_scoring_adapter.py` 直接加载原确定性评分器 `safe_exact_v2_0_terminal_final_heading_fix`，核验源码 SHA256 为 `6fc1d2f3cc81b3e954350143abd1071d6556e1cfe4f321d9333f2016baf2150d`，并核验其原始评分依赖。评分器输出的 reference 字段留在调用栈内，持久化评分收据不保存标准答案。

## 接口

- `build_scoring_scope(execution_scope, reference_scope)` 绑定共同五项来源、完整稀疏选中组合并集、原 scorer 身份，以及 execution/reference 两个 scope 的哈希。参考组件的额外来源 bindings 随评分 scope 保存。
- `SparseScoringAdapter(approval_ref, scope_contract, reference_provider, journal_path, receipt_directory, software_test_only=...)` 构造适配器。
- `score_one(execution_receipt, reference_version)` 返回 `sparse_row`、`scoring_receipt` 与 `from_scoring_cache`。
- `ledger()` 返回评分尝试、gold provider 统计及按物理执行身份去重的上游资源引用。
- `resource_ledger_from_execution_receipts(receipts)` 汇总成员与聚合原生执行收据，按 `physical_attempt_id` 去重。

每次评分均调用共享 `validate_scope_approval(..., operation="score_outputs")`。参考组件使用同一 approval 的 `read_reference` 权限。假 approval 明确标记 `software_test_only=True`，生产模式拒绝使用。

评分接收完整 `required_aggregate_keys` 中已执行的组合，包括合法缓存。未执行组合不能产生评分行。输出行严格匹配 `sparse_policy_projection.FIELDS`，可直接传入 `project_frozen_policy_scores`。`reference_version`、参考来源、实际 input/output Token、原生记录 SHA 和 CPU/墙钟耗时保存在单独收据和账本中。

## 复用与失败语义

同一 canonical question/reference version 复用参考组件的内存缓存和 gold 账本。同一个已评分组合再次用于其他冻结政策时复用原评分收据。同一物理聚合执行的 Token 只作为已由执行组件计费的资源引用。

评分使用 SQLite 持久化尝试、成功行、gold 来源及资源身份。中断或失败后保留 `FAILED_REQUIRES_REVIEW` 或未完成状态，随后调用需要先协调处理。参考值解码后发生错误时，provider 的已消费记录与前后统计仍被保存。执行失败且 Token 未知时，完整成本返回 `null` 并标记待协调。

没有原生执行记录的 backend 初始化失败由 execution owner 的 reservation/fence 记录覆盖，资源汇总函数明确将其标为列表外未物化 reservation，需与执行日志共同检查。

## 合成资格

入口为 `qualify_e18_scoring_adapter.py`，使用执行组件提供的 FakeBackend/FakeTokenizer、原 MemberStore、typed ExecutionStore 与真实原 scorer。两道人工问题保留三个生成重复，执行两个选中组合后投影到四个冻结政策，逐项核对预设奖励。

资格覆盖 approval 缺失、操作未批准、假生产、错误 scope/canonical/组合、原生记录篡改、不可覆盖收据与恢复、gold 复用、失败后停用，以及成员/聚合实际 Token 去重与未知费用保留。所有模型调用均为软件 fixture。

正式收据为 `E18_SCORING_ADAPTER_QUALIFICATION_V2.json`。前三次资格记录保留。第一次在 toy 元数据阶段发现 BBH 来源角色填写错误，修正为原 test 角色。第二次发现将小写目录名称用作模型池语义 alias，已改为冻结模型池的精确语义名称。第三次 15 项全部通过，累计执行 45 次测试入口，三次均未读取真实答案。

## 执行边界

后续真实评分仍需明确批准的 E18 execution/reference/scoring 三组件 scope、合法输出及 reference source。当前准备只确认软件接口和合成失败处理，真实 GPU、真实文件来源和完整多工作进程调度仍需各自的运行核验。
