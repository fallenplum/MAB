# B4 远端聚合缓存范围补充核验 V3

## 核验结果

**已扫描的聚合根、槽位、bundle 和 exports 范围内，可确认复用的 fresh 聚合缓存为 0。** 现有成员缓存仍以 V2 已核验的 5,330 个逻辑 key 为准。该结论只覆盖本节列出的实际目录边界。

主机 ed25519 pin 与已有服务器一致，认证后仅通过 SFTP 列举、stat 和读取允许的元数据。用时 35.258 秒。累计完整列举 282 个目录、1735 个条目，未触及深度或文件数上限，没有扫描内目录读取失败。

130 份文件通过键名防护后投影了身份元数据。68 个 `outputs.jsonl` 均未获得 fresh scope 计划及 context 绑定，因此未打开其模型输出原文。没有发现 fresh scope 的 RUN_PLAN 或 context。

## 实际扫描边界

- routing 根下实际存在的聚合、typed 与 stage2 相关目录。
- 三个 `primary-aggregator-slot` 的 `artifacts/private`、`artifacts/preparation` 与 `results/raw`。这些槽位的 raw 元数据只见 `online_stream` 和 `online_calibration`。
- 共享 qwen3 worker 中符合聚合作业命名的目录，以及上述范围中已枚举的 shard、bundle 和 exports。
- 36 个原训练/验证 typed 主聚合输出容器与 32 个历史诊断、单答、强模型等输出容器。仅核验路径与安全元数据。

| 输出根目录 | 已列举输出容器数 |
|---|---:|
| `decision_feedback_v2_stage1_aggcal42_20261104` | 1 |
| `routing_v2_stage2_typed_train_g137_20261105` | 8 |
| `routing_v2_stage2_typed_train_g2024_20261105` | 8 |
| `routing_v2_stage2_typed_train_g42_20261105` | 8 |
| `routing_v2_stage2_typed_validation_g137_20261105` | 4 |
| `routing_v2_stage2_typed_validation_g2024_20261105` | 4 |
| `routing_v2_stage2_typed_validation_g42_20261105` | 4 |
| `routing_v2_typed_alone_g42_20261104` | 2 |
| `routing_v2_typed_diag200_g42_20261104` | 4 |
| `routing_v2_typed_heterogeneous_g137_20261104` | 4 |
| `routing_v2_typed_heterogeneous_g2024_20261105` | 4 |
| `routing_v2_typed_strong_g137_20261105` | 4 |
| `routing_v2_typed_strong_g2024_20261105` | 4 |
| `routing_v2_typed_strong_g42_20261104` | 4 |
| `tasktyped_calibration_9ef6c02bc9fb_20261104` | 5 |

109 个域外目录和 8 个受保护命名项均保留了显式边界记录。成员工作区由独立 V2 成员审计处理。模型、环境、qualification/diagnostics、受保护数据目录，以及 routing 根以外未登记位置继续标记 UNKNOWN。

## 成本与权限

本次新增确认的 fresh 聚合逻辑 key、物理执行及其计入账本的 Token 均为 0。未知位置可能存在的聚合开销仍为 UNKNOWN。已扫描历史训练、验证和诊断输出的原文与成本未加入 fresh 账本。

本模块 fresh gold/reward 值解码 0，sealed 文件打开 0，远端命令启动 0，远端写入 0，新增生成、评分与训练任务均为 0。数据模块另有单向 sealed 输入指纹审计授权，项目总边界由对应模块记录。

成功唯一 key 的执行量只说明成功结果预算。原执行器遇失败停止，跨人工续跑的 `max_attempts` 总上限未锁定，含失败重试总开销另列。

## 依据与局限

执行脚本 SHA256 为 `d85acfc66bce9a7765cb45e58cb46aae21e2cf89d08f146eb33067ec2ce00c66`。私有完整目录元数据收据 SHA256 为 `2442a6d4eb1df5f3d10157f004d2310c6b7c7a4b7ef540dc5fbc65d10db3c7c5`，体积 2,201,327 字节。

本次未发现 fresh 聚合候选，输出 record、jobs/bridge 和成员依赖的完整核验路径未用于纳入任何实际 fresh 聚合缓存。后续若定位到新目录，需要以该目录身份和未评分 schema 重新核验后再入账。
