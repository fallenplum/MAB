# P2 显式来源注册与互斥聚合分片

## 实际部署追加记录

原完整聚合actor在北京时间2026年11月17日00:53:35安全退出，保留586个唯一成功结果。来源注册先初始化原七份manifest，再ADD三份互斥分片。剩余聚合键分别为34883、35826、35241，使用GPU0、1、3。00:55:43快照中三份均已首成功，成功数为3、4、2，九个P2 actor活跃，重复成功及成功键额外付费attempt均为0。

新增P2_THREE_SHARD_ACTIVATION.json保存本次实际停止、计划、准入、启动、首成功和本地回传的脱敏证据及来源摘要。以下原本地准备资格及三个资格JSON保留当时状态。

## 原本地准备资格

本地执行工具准备时，注册与分片资格20项、监控与导出资格14项均通过。该准备任务的真实模型调用、Gold读取、SSH操作、生产进程变更与部署次数均为0。

本地准备时，原生产六成员与一个聚合actor由主任务控制。下列命令构成所交付接口，后续实际执行由新增上线快照记录。

## 文件与接口

| 文件 | 接口 |
|---|---|
| `p2_manifest_registry.py` | `create_registry`、`add_manifest`、`load_registry`及命令行初始化/ADD |
| `prepare_aggregation_shards_v3.py` | `prepare`、`validate_dispatch_plan`，默认2份，`--shards 3`指定3份 |
| `p2_aggregation_follower_v3.py` | 显式`--manifest`、`--manifest-sha256`、`--worker-id`、`--registry`、`--gpu` |
| `launch_registered_worker_v3.py` | 依据注册的worker身份、任务集合、安全停止检查点与实测显存启动 |
| `monitor_native_status_v3.py` | `collect_status`及`--registry PATH` |
| `p2_export_native_v3.py` | `export_batch`及`--registry PATH`，复用既有SQLite导出索引 |

注册表首行INIT，后续仅允许ADD。每条记录具有序号、前一条摘要及自身摘要。原六member和原full-aggregate来源持续保留。每个manifest内部重复键、worker身份重定义、同manifest另起别名均拒绝。

历史来源可以覆盖相同任务。进程启动另行检查活跃actor的stage与任务键交集。旧full-aggregate仍活跃时，任何覆盖其剩余键的shard启动均拒绝。

## 执行顺序

以下为服务器本地命令，路径变量仅用于本任务。先上传新增文件及其资格收据，保留原v2源文件。

```bash
P2_ROOT=/data2/chendong/e20_p2_execution
P2_PY=/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python
P2_REG=$P2_ROOT/control/manifest_registry_v3.jsonl

$P2_PY -B "$P2_ROOT/code/p2_manifest_registry.py" \
  --registry "$P2_REG" --initialize-original-root "$P2_ROOT"
```

一次性初始化严格核对原七份v2 manifest及原member/follower的已知源码摘要。若需要显式构建初始化JSON，文件内容为七个对象的数组，每个对象恰含`worker_id`、`manifest_ref`及`worker_source_ref`。两个ref均恰含绝对`path`与`sha256`。

主任务对旧聚合actor发送SIGTERM，等待当前调用成功落盘及进程退出。`SUMMARY.json`应为`SAFELY_STOPPED`且`stopped=true`，PID、scope与前驱启动收据一致。准备器本身不发送信号。

```bash
$P2_PY -B "$P2_ROOT/code/prepare_aggregation_shards_v3.py" \
  --registry "$P2_REG" \
  --output-dir "$P2_ROOT/control/aggregation_dispatch_v3" \
  --worker-source "$P2_ROOT/code/p2_aggregation_follower_v3.py" --shards 3
```

stdout给出`plan_ref`及各shard数量。将输出目录中的`shardN_registry_addition.json`逐一ADD，再使用`DISPATCH_PLAN.json`中对应的真实`worker_id`启动。下面以一个shard演示，GPU编号由主任务按实时状态选择。

```bash
$P2_PY -B "$P2_ROOT/code/p2_manifest_registry.py" \
  --registry "$P2_REG" --add /ABSOLUTE/PLAN/shard0_registry_addition.json

$P2_PY -B "$P2_ROOT/code/launch_registered_worker_v3.py" \
  --registry "$P2_REG" --worker-id EXACT_REGISTERED_WORKER_ID \
  --dispatch-plan /ABSOLUTE/PLAN/DISPATCH_PLAN.json \
  --gpu 0 --required-free-mib 14336 --shared-authorized
```

同一计划的各shard可顺序启动。其一先产出成功结果不会使同计划其余shard失效。原历史actor在准备后继续推进则使计划过期，须在其安全退出后重建剩余键计划。空shard无需启动。

## 分片与恢复资格

`D_completed`来自已注册的全部旧/新聚合成功journal。各剩余shard按`SHA256([question_id,generation_seed]) mod shard_count`划分，每道题同一seed的待执行动作保留在一份中。

检查保证`D_completed`与各`Pi`两两互斥，且并集严格等于原required aggregate集合。原scope、approval、owner路径、backend、ModelSpec、问题身份和科学参数均通过原引用保持绑定。manifest的变化限于worker ID、调度计划ID、task subset引用与技术GPU锁目录。

新follower一次构建原persistent backend，以当前manifest的subset执行。启动时从所有注册聚合journal恢复本subset成功键。它持续读取注册member来源，保留已读offset，追加member manifest后可见新journal。末尾未完成JSON行等待后续完整追加，完整坏行报错。

## 监控与传输

```bash
$P2_PY -B "$P2_ROOT/code/monitor_native_status_v3.py" --registry "$P2_REG"

$P2_PY -B "$P2_ROOT/code/p2_export_native_v3.py" \
  --registry "$P2_REG" --max-tasks 500
```

监控按stage与原语义键计算唯一成功数，按stage与physical attempt ID计算已付费成功调用。`from_cache=true`的重放不增加Token，同一付费attempt出现在多份journal中仅计一次。相同任务出现不同付费attempt时保留额外真实成本，并给出异常数量。

导出继续使用`exports/exported.sqlite`的stage/digest主键，读取全部显式注册来源。导出前验证原native三文件、scope、approval、任务键、记录摘要和真实Token。切换导出入口前应结束旧导出器的调度，防止旧版本与新版本同时竞争同一索引。v3导出器自身通过SQLite事务串行化。

## 优势与风险

- 优势为聚合剩余键互斥接续、旧/新证据统一可见、成功缓存复用及付费Token去重。每个shard复用原模型上下文，逐任务执行沿用原算子。
- 资源风险由主任务实测GPU状态控制。共享GPU的显存与利用率会随其他作业变化，启动器记录连续三次显存观测与授权标记。额外并行的吞吐需通过真实运行验证。
- 原合成资格覆盖执行工具的边界与账本行为。实际部署后的首成功及重复计数已由新增上线快照核验，后续吞吐和持续资源状态继续由主任务监控。

## 本地复现

```powershell
& 'D:\Anaconda\envs\tslib\python.exe' -B -X utf8 .\qualify_registry_consumers.py
& 'D:\Anaconda\envs\tslib\python.exe' -B -X utf8 .\qualify_registry_dispatch.py
```

两项均使用短生命周期合成工件，输入不含真实问题、答案或私有模型权重。源码摘要见`P2_V3_TOOLING_QUALIFICATION.json`。
