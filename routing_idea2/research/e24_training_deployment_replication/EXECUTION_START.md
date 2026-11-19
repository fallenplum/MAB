# E24 正式执行准备与启动状态

本阶段根据用户的开始指令准备正式物理执行。使用已冻结的 1,196 道题、B–U、B–P、C–U、C–P 和三个生成 seed。统一支持并集为 20,661 个成员键和 53,661 个聚合键。训练和题池冻结收据沿用准备阶段。

## 本地准备

`package_approved_inputs.py` 从冻结策略概率独立重建支持并集，与私有调用键逐项比对。只打包题面、输入身份、四份冻结动作概率、执行键和公开协议。正式评分用的本地参考位置表通过源快照和题面核对，执行包仅绑定其 SHA256。标准答案留在本地来源。

`runtime/prepare_execution.py` 从原 P2 后端清单重用原科学算子和模型配置，形成 E24 独立 scope、批准收据和互斥任务分片。默认成员模型分片数为 `2,2,2,2,4,2`，聚合为 8 片。调度分片只影响执行归属。`runtime/supplement_worker.py` 重用 E23 常驻推理器，按有界间隔检查未闭合的成员依赖。

`monitor_e24.py` 读取已注册 journal 汇总实际完成量和计费。使用 `--verify-owners` 校验围栏，使用 `--export-final` 导出按阶段分片压缩的原生回执。闭合前的失败或未知付费需要单独核对。真实评分仅发生在产出闭合之后，独立分析入口为 `replication.py evaluate`。

运行配置、监控和闭合评分分别通过 14、17、8 项本地资格检查。运行资格包含真实输入包的只读范围重建，其余检查使用合成执行记录或参考答案。评分检查覆盖原评分器、题级答案复用、按阶段计费、成员来源核对、封存交集拒绝及损坏分片拒绝。实际 GPU 产出尚未验证。

## 服务器核验与启动状态

通过既有主机指纹登录了原推理服务器。检查时 GPU 1、2、5、7 空闲，数据盘 `/data2` 可用约 16 GiB。Python 3.11.9、PyTorch 2.4.1+cu121、Transformers 4.51.3 和 bitsandbytes 0.48.2 均可导入，原 P2 注册表含 6 份成员后端和 1 份聚合后端。该资源快照仅用于本次启动准备，实际并发仍依据启动时的显存和进程状态。

自动审批拒绝向服务器传送 `INPUTS.jsonl`。原因是此前的数据留本地约定与本次题面传输权限缺少明确对应。回读确认服务器 E24 根目录、题面文件、执行计划和启动收据均不存在。**本轮新增物理调用 0，E24 评价答案解码 0。**

服务器侧只需运行输入包和科学执行器。题面传输权限明确后，以已锁定的输入包配置原任务并集，逐步启动成员和聚合进程，使用原生成功回执确认写入和计费。配置与监控日志保留在各自的独立目录。

## 部署文件名与命令

新运行根目录的 `code/` 采用以下文件布局。原始依赖来自 E23 远端已归档代码，逐字节复制并校验引用摘要。

| 文件 | 来源 |
|---|---|
| `prepare_execution.py`、`supplement_adapter.py`、`supplement_worker.py` | 本目录 `runtime/` |
| `monitor_e24.py` | 本目录 |
| `e23_supplement_adapter_original.py`、`e23_supplement_worker_original.py` | E23 同名无前缀源码，部署时重命名 |
| `launch_supplement.py`、`execution_monitor.py` | E23 原源码 |
| `p2_original_execution_adapter.py`、`p2_scope_freeze.py`、`p2_member_worker.py`、`p2_aggregation_worker.py` | E23 服务器 `code/` 绑定的原 P2 源码 |

在新的运行根目录从 `inputs/` 保存已冻结的执行包，然后用原推理 Python 执行 `code/prepare_execution.py --root RUN_ROOT`。发起正式推理前，确认输出为 `READY_NOT_LAUNCHED`，并与 `INPUT_ONLY_BUDGET.json` 逐项核对任务数。按 `code/launch_supplement.py --root RUN_ROOT --worker WORKER_NAME --gpu GPU_INDEX` 启动绑定清单。监控和导出分别使用 `code/monitor_e24.py --root RUN_ROOT` 和 `code/monitor_e24.py --root RUN_ROOT --export-final`。

本地收回已闭合的原生 gzip 分片与 `OUTPUTS_FROZEN.json` 后，使用 `score_closed_execution.py` 核对预测、参考位置、阶段任务与实际成本。默认入口验证原生回执。开启 `--execute-scoring` 时需要绑定同一闭合工件的本地评分收据。评分产生的受保护文本、原生数据、逐题得分、标准答案和定位器均保留在私有工件中。
