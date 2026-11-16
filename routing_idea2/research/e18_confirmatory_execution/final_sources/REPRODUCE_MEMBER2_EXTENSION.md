# E18成员2调度扩展复现说明

## 安装位置

把本扩展文件按INVENTORY.json的publication_path追加到77文件final_sources包，保留原路径，不覆盖其既有源。基包INVENTORY SHA为05e72127b26140792c96e5405dc236ce41ae1907664c03dd68bcc357fd773f74。

新增code/e18_confirmatory_execution/*.py与基包同目录。准备阶段adapter/fixture仍位于code/e18_mechanism_and_confirmation_preparation。无需重复拷贝原76历史V2源；其来源、配置、环境和私有资产要求沿基包REPRODUCE.md。

## 既有源码依赖

- 原member4_checkpoint_broker b2dc379c、e18_gpu_worker17f74192和dispatcher5d273799。
- 原run_original_plan_shard_v2 2650c784及其原模块。
- 原incremental_aggregation_planner56c89563与原资格fixture。
- 原adapter21e76f6a、原聚合workerde5c3cc8/dispatcher715fb151及其资格fixture。
- 原合法模型/config/Store、科学算子与private scope/approval/native围栏，仍由调用者提供。

INVENTORY逐项保存原始与发布SHA，软件摘要source_bindings区分当前源与未打包的历史0dd8源。

## 合成软件检查入口

SUITE_ROOT指向合并后code目录，输出使用新的文件名。

```powershell
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/qualify_member2_worker_v1.py" --output NEW_MEMBER2_WORKER_CHECK.json
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/qualify_member2_checkpoint_broker.py" --output NEW_MEMBER2_BROKER_CHECK.json
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/qualify_incremental_aggregation_planner_v4.py" --output NEW_PLANNER_V4_CHECK.json
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/qualify_original_aggregation_partial_resume.py" --output NEW_AGG_RESUME_CHECK.json
```

这些fixture沿原源码/元数据依赖，使用临时合成输入与FakeBackend。部分资格需要文件hard-link权限。v1权限失败保留为软件失败，不计入LLM失败。本次扩展暂存未运行上述命令。

## 获准生产入口

以下只展示真实接口。它们不能替代实际审批/资源接纳，本包不携带真实运行JSON。

```text
member2_checkpoint_broker.py --plan PLAN --plan-sha256 PLAN_SHA --dispatcher ORIGINAL_DISPATCHER --former-metadata APPROVED_METADATA --former-metadata-sha256 METADATA_SHA --lease-root LEASE_ROOT --child-plan CHILD_PLAN
```

broker默认执行元数据检查。只有获准控制流程添加--execute，原双former身份、metadata哈希、原claims与backend/cost资格必须一致。两个控制PID常量及其绑定不能由发布者静默替换；其他部署需新的有来源版本和相应软件资格。

```text
e18_member2_dispatcher_v1.py --plan CHILD_PLAN --plan-sha256 CHILD_SHA --gpu GPU --gpu-slot SLOT --resource-policy POLICY --resource-policy-sha256 POLICY_SHA
incremental_aggregation_planner_v4.py --config CONFIG --config-sha256 CONFIG_SHA --extra-source-plan CHILD_PLAN --extra-source-plan-sha256 CHILD_SHA --member2-worker-source MEMBER2_SOURCE --member2-worker-source-sha256 MEMBER2_SOURCE_SHA --once
```

复用base56c原planner同名模块并和v4放置于同目录。新增planner仅注册受控journal来源，原配置、SQLite schema、ready计划和ALL_DONE身份保持原协议。生产follow循环沿原--follow接口。原聚合部分恢复资格检验成功前缀缓存和PAUSED恢复，未修改原聚合执行源。

## 证据与风险

优势是保留原始执行字节、源与资格配对，以及失败历史。风险在于checkpoint控制guard具有特定来源绑定，硬链接权限/跨进程锁及GPU余量依实际系统；源码/合成资格不能代替真实GPU接纳或训练重现。模型、题目、反馈权限和生成结果均沿原科学设置，新增独立研究题和新LLM调用由获准运行流程计量，本次暂存为0。
