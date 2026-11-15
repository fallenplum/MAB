# E18 v2 单任务执行适配

## 当前资格

`e18_execution_adapter.py` 已提供输入作用域构造、原始任务工厂、审批闸口和单任务执行函数。`qualify_execution_adapter.py` 使用临时合成输入、FakeBackend 和原始 Store 完成 33 项软件资格。

| 项目 | 状态 |
|---|---|
| 原 member/typed aggregation 任务与 Store 复用 | 合成资格通过 |
| 审批哈希、目标键、预算与输出拥有者绑定 | 合成资格通过 |
| 成功缓存复用、失败/未知消耗围栏、软件缓存模式隔离 | 合成资格通过 |
| 成员生成 seed 42/137/2024 与聚合 seed 42 | 合成资格通过 |
| 实际 Transformers/GPU 推理 | 尚未运行 |
| 完整 cohort 多任务调度器 | 尚未实现 |

软件资格读取原 V2 源码和配置元数据。资格记录中的真实模型调用、真实答案读取及真实任务提交均为 0。它与未来 GPU 运行资格分别记录。

## 可执行的软件资格入口

在仓库根目录执行以下命令。新输出路径应尚未存在。

```powershell
E:\语言模型组合路由\.venv\Scripts\python.exe -B -X utf8 experiments/e18_mechanism_and_confirmation_preparation/qualify_execution_adapter.py --output experiments/e18_mechanism_and_confirmation_preparation/EXECUTION_ADAPTER_LOCAL_RECHECK_v2.json
```

该入口实际执行合成成员任务和合成 typed 聚合任务，并核验相同任务再次调用时的缓存行为。临时拥有者目录按系统临时目录边界检查后创建，退出时清理。

接口说明入口可单独执行。

```powershell
E:\语言模型组合路由\.venv\Scripts\python.exe -B -X utf8 experiments/e18_mechanism_and_confirmation_preparation/e18_execution_adapter.py --describe-only
```

## 冻结作用域与审批

`build_scope_contract` 仅接收问题输入、四套已冻结动作、共同来源绑定及待执行的稀疏键。正式执行通过 `E18ExecutionAdapter` 重新计算并核对作用域。

每个执行键为 `[question_id, member_generation_seed, member_or_combination_index]`。所需成员及组合并集独立保留，新增付费键是其中经过缓存核对后的子集。

审批文件使用 `e18_scope_approval_v2`。五项共同绑定为 `protocol_sha256`、`target_inputs_sha256`、`frozen_actions_sha256`、`budget_sha256` 和 `target_ids_sha256`。每个组件的完整作用域哈希写入 `scope_sha256_by_role`。

执行审批还须提供绝对路径 `approved_output_root` 和两个阶段的 `execution_limits`。调用传入的输出根经 `resolve` 后须与审批路径一致。相同作用域、阶段和键只拥有一个目录，目录中的持久围栏在后端构造前写入。

`software_test_only` 必须与调用模式一致。成功缓存的围栏必须绑定相同的模式、任务、原始记录哈希和作用域。合成缓存进入正式模式时立即停止。

单任务公开接口如下。

```python
receipt = run_approved_task(
    adapter,
    {
        "stage": "member_generation",
        "question_id": approved_question_id,
        "generation_seed": 137,
        "member_index": approved_member_index,
    },
    approval_ref,
    approved_output_root,
    backend_factory,
    software_test_only=False,
)
```

聚合任务将 `stage` 改为 `aggregation`，使用 `combination_index`，并传入选定三个成员的原始记录及冻结聚合器 tokenizer。聚合任务沿用原厂匿名排序与解码 seed 42。生成重复 137/2024 通过三个成员的原始任务身份和 provenance 绑定。

## 正式后端来源

`original_backend_factory(asset_root)` 延迟调用原始 `make_backend`，支持冻结 logical path 上的本地模型。`mapped_backend_factory` 延迟调用原始私有 MappedBackend，支持经原始来源收据验证的物理加载路径。两者均通过原 `TransformersBackend` 的来源校验。

映射模式额外需要原始 helper 文件、其 SHA 绑定、映射文件及其 SHA、物理模型来源收据、原 E0/发现收据和对应物理根。私有路径与收据由运行端提供。公开仓库保留验证代码及脱敏身份。

运行模块加载源码时保持 `CUDA_VISIBLE_DEVICES` 原值。CPU 准备模块 `e18_common` 的环境设置不进入此运行入口。

## 执行前仍须完成的核对

1. 最终协议、输入、冻结动作、缓存对账和精确预算使用同一版本。
2. reference、execution、scoring 三个组件的作用域哈希由同一审批绑定。
3. 正式模型路径、tokenizer 与原始来源收据可访问，实际 GPU 后端通过运行资格。
4. 原 failed/unknown 调用完成费用对账，拥有者围栏状态明确。
5. 完整 cohort 调度层明确并发数、单任务接纳顺序和暂停/续跑行为，并经过无付费软件资格。

## 优势与风险

原始工厂、来源身份、Store、压缩和聚合语义共同复用，稀疏必要组合可以按冻结策略执行。付费任务的审批、唯一拥有者与实际用量收据有可核对的链路。

当前证据覆盖合成单任务路径。真实 GPU 环境、跨主机文件可见性和完整 cohort 调度仍需运行资格。任何失败或消耗不明的拥有者会阻止自动重试，需要先核实实际记录及费用。
