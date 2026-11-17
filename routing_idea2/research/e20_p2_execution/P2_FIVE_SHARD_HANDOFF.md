# P2 五路聚合调度准备

## 实际首成功追加记录

UTC 2026年11月17日03:51:56，原三个聚合actor安全退出，保留20842个历史唯一成功。其余85694个键被划为17536、16900、16882、17051、17325五份，在GPU0、1、3、1、6执行。原注册来源10份保留，再ADD五份至15。

UTC 03:56:25首成功快照中，五路分别成功14、14、15、14、17次，总聚合20916，成员26436，活跃actor10。重复成功及成功键额外付费attempt均为0。P2_FIVE_SHARD_ACTIVATION.json保存该时点及实际控制响应摘要。该记录验证启动与接续，持续吞吐由另行观测时间窗评价。

## 原本地准备资格

本工具准备由GPU 1和GPU 6各增加一个聚合进程的授权触发。16项本地合成资格通过，结果保存在`P2_FIVE_SHARD_TOOLING_QUALIFICATION.json`。该准备阶段的SSH操作、生产进程变更、Gold读取与真实模型调用均为0。原JSON保持准备时点状态，实际部署由上述新增快照记录。

原注册表、聚合follower、v3准备器和v3启动器字节保持各自已冻结版本。v4准备器采用同一种按题与生成seed分组的互斥划分，技术分片数支持2至8，当前默认5。v4启动器复用原活跃任务交集、安全停止检查点、显存准入和命令构造函数。

## 操作顺序

上传两个v4工具，保留服务器原v3文件。主任务核对GPU资源并让当前三个v3聚合actor完成在途调用后安全退出，逐一核验PID、scope和`SAFELY_STOPPED`检查点。六成员任务依其原运行状态继续。

以下命令在服务器执行，路径变量仅属于本任务。

```bash
P2_ROOT=/data2/chendong/e20_p2_execution
P2_PY=/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python
P2_REG=$P2_ROOT/control/manifest_registry_v3.jsonl

$P2_PY -B "$P2_ROOT/code/prepare_aggregation_shards_v4.py" \
  --registry "$P2_REG" \
  --output-dir "$P2_ROOT/control/aggregation_dispatch_v4" \
  --worker-source "$P2_ROOT/code/p2_aggregation_follower_v3.py" --shards 5
```

stdout给出完整`plan_ref`和五份待执行数量。将计划目录内的`shard0_registry_addition.json`至`shard4_registry_addition.json`依次ADD到原注册表。既有完整来源及旧分片来源继续保留。

```bash
$P2_PY -B "$P2_ROOT/code/p2_manifest_registry.py" \
  --registry "$P2_REG" --add /ABSOLUTE/PLAN/shard0_registry_addition.json

$P2_PY -B "$P2_ROOT/code/launch_registered_worker_v4.py" \
  --registry "$P2_REG" --worker-id EXACT_REGISTERED_WORKER_ID \
  --dispatch-plan /ABSOLUTE/PLAN/DISPATCH_PLAN.json \
  --gpu 0 --required-free-mib 14336 --shared-authorized
```

五个shard依次使用GPU `0、1、3、1、6`。每次启动前重新测量显存，启动收据保存连续三次观测。空分片跳过启动。已启动的同计划兄弟分片先产出成功结果时，其余分片仍可继续启动。

## 原执行与账本复用

全部已注册旧journal的成功任务组成`D_completed`。它与五份待执行任务两两互斥，并集严格等于原106536个required聚合键。底层仍使用原`p2_aggregation_follower_v3.py`，scope、approval、owned输出根、backend、ModelSpec、解码设置和随机种子均通过原引用校验。

监控和原生导出继续使用现有`monitor_native_status_v3.py`与`p2_export_native_v3.py`，保留同一个SQLite导出索引和所有来源。软件资格仅验证调度与账本实现。部署后需核对每个新增actor的首个成功、重复付费计数和实际吞吐。

## 优势与风险

优势是复用原算子和成功缓存，利用GPU 1及GPU 6的可用资源提高并行量。运行风险来自共享GPU的动态显存与计算竞争，实际吞吐需要通过部署后的时间窗口核验。

## 本地资格复现

```powershell
& 'D:\Anaconda\envs\tslib\python.exe' -B -X utf8 .\qualify_five_shard_dispatch.py
```

资格程序使用短生命周期合成问题及模拟journal，保留全部旧源码哈希。它覆盖全并集、互斥、旧成功恢复、同计划兄弟启动、旧活跃任务拒绝、后端变化拒绝和冻结旧文件字节一致性。
