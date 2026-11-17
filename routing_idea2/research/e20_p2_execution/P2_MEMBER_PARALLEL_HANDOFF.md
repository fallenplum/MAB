# P2 慢成员并行接续

## 准备结果

本地合成资格17项通过，记录于`P2_MEMBER_PARALLEL_QUALIFICATION.json`。本次准备的SSH、生产进程修改、语言模型调用及标准答案访问均为0。

`prepare_member_shards.py`为一个成员建立剩余任务分片，默认成员4及4份任务。`launch_registered_member.py`复用原活跃任务交集检查、GPU准入、命令构造和进程启动协议。正式执行继续调用原`p2_member_worker.py`。

原成员worker、来源注册表和聚合follower的文件摘要均与准备前一致。新增manifest保留原scope、approval、模型backend、题目输入及执行owner，变化限于worker身份、计划身份、任务子集引用和技术GPU锁目录。

## 安全停止

`stop_registered_workers.py`仅处理显式PID列表。每个PID必须匹配已登记启动收据中的完整命令、manifest及worker源码。一个停止组必须来自同一stage，成员停止组还要求来自同一成员profile。它发送SIGTERM，等待当前原生调用完成并写入成功记录。

以下为服务器命令，变量均仅用于P2。

```bash
P2_ROOT=/data2/chendong/e20_p2_execution
P2_PY=/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python
P2_REG=$P2_ROOT/control/manifest_registry_v3.jsonl

$P2_PY -B "$P2_ROOT/code/stop_registered_workers.py" \
  --registry "$P2_REG" --pids 2383366 \
  --record-dir "$P2_ROOT/control/member4_parallel_20261117" --phase request

$P2_PY -B "$P2_ROOT/code/stop_registered_workers.py" \
  --registry "$P2_REG" \
  --record-dir "$P2_ROOT/control/member4_parallel_20261117" --phase verify
```

`verify`在当前调用尚未完成时返回`WAITING_FOR_CURRENT_NATIVE_CALLS_TO_COMMIT`。成功后生成不可覆盖的`02_STOPPED_CHECKPOINT.json`。

成员原worker的SUMMARY没有PID、scope和COMPLETE状态字段。检查使用其`schema`、`manifest_sha256`、`stage`、`profile`、`software_test_only`、`planned`、`completed`、`stop_requested`、`failures`及文件时刻，并与本次启动后的实际成功journal逐条对应。自然完成需要`completed==planned`和空失败列表。安全停止允许`stop_requested=true`及部分完成。聚合checkpoint继续使用原PID、scope及`SAFELY_STOPPED`。

## 建立四份任务

```bash
$P2_PY -B "$P2_ROOT/code/prepare_member_shards.py" \
  --registry "$P2_REG" \
  --output-dir "$P2_ROOT/control/member_dispatch" \
  --worker-source "$P2_ROOT/code/p2_member_worker.py" \
  --member-index 4 --shards 4
```

标准输出提供`plan_ref`、已经成功的唯一任务数及四份剩余任务数量。对应目录包含四份`shardN_registry_addition.json`、四份manifest及精确的任务键列表。先逐一ADD，再读取`DISPATCH_PLAN.json`中的实际worker身份启动。

```bash
$P2_PY -B "$P2_ROOT/code/p2_manifest_registry.py" \
  --registry "$P2_REG" --add /ABSOLUTE/PLAN/shard0_registry_addition.json

$P2_PY -B "$P2_ROOT/code/launch_registered_member.py" \
  --registry "$P2_REG" --worker-id EXACT_REGISTERED_WORKER_ID \
  --dispatch-plan /ABSOLUTE/PLAN/DISPATCH_PLAN.json \
  --gpu 5 --required-free-mib 14336 --shared-authorized
```

父任务拟采用GPU分布`5、4、4、5`，每次启动仍以实际显存和利用率检查为准。代码提供2至8份技术分片，当前准备目标固定为4份。

## 执行与证据边界

- 历史成功集合只包含当前成员profile。其他成员继续产出会更新其各自journal，也继续支持聚合。
- 全部历史成功与四份新任务两两互斥，其并集严格等于原成员全部所需任务。
- 同计划所有successor manifest共同排除于历史快照推进检查，先启动的兄弟已产出不会阻止后续兄弟启动。
- 原成员worker继续使用原成功owner缓存和失败围栏。新任务列表排除已完成键，原成功文件保留。
- 聚合follower通过ADD-only注册表读取新增成员journal，并保留已读offset。此成员接续无需停止正在运行的聚合进程。
- 启动后的实际首成功、没有重复付费调用及吞吐变化由父任务核验。本地合成资格只覆盖执行工具与权限边界。

## 本地复现

```powershell
& 'D:\Anaconda\envs\tslib\python.exe' -B -X utf8 .\qualify_member_parallel.py
```

结果包含源码摘要及每项合成资格。合成输入使用人工任务键，题目、标准答案及私有模型权重访问均为0。
