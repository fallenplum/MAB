# E18操作源码追加包

## 范围

本包追加18份原字节Python文件，其中13份为当前操作入口或资格源码，5份为历史源码。另提供15份软件资格白名单摘要。每项原始来源SHA、公开文件SHA及目标路径记录在INDEX.json中。

本包叠加在commit `d819d6bde0e60db9d94ff83d1820eb0729cf7cac` 的95文件公开源码包上，代码目录继续使用 `code/e18_confirmatory_execution`。保留原目录拓扑，并将同名模块保持在既有相对位置。原95文件及原运行源保持各自字节。

## 当前入口与依赖

- `safe_idle_retirement.py` 通过明确PID、进程出生时间、完整argv及冻结源码身份，等待原claim文件描述符空闲窗口。原执行器保持各自冻结版本。选择完成但尚未领取的batch仍可能完整执行，实际调用和费用以原生账本为准。
- `cooperative_slot_admission_v2.py` 包裹调用者持有的同一原member4模块。资源准入记录显式own/foreign进程身份、GPU UUID及余量，保持原ModelSpec和付费调用路径。
- `run_member4_fine_cooperative.py` 使用当前fine控制器与资源上下文。policy的requester源码引用应绑定实际薄runner文件。
- `member4_tail_partition.py` 在原claim与两层lease下细分剩余原键，已完成原生前缀沿原Store继续复用。
- `incremental_aggregation_planner_v5.py` 依次依赖既有v4与base planner。运行来源、原配置SHA和原生完成身份分别保存。
- `native_walltime_metadata.py` 提取计时元数据。调用级ledger和真实运行资产由使用者私下提供。

通用代码使用Python标准库，部分资格和既有报告入口还需要NumPy、原项目冻结科学源码、配置与来源receipt。涉及原API的fixture应以 `MAB_ASSET_ROOT` 指向使用者依法持有的源资产。完整模型运行另需原环境、冻结权重、合法数据和私有scope/approval/plan/lease/cache。源码完整性与软件资格的证据范围限定为列明的文件与检查，实际生产重放、训练重现和科学结论需各自执行证据。

## 软件复核

将本包与既有95文件的代码按相对目录合并后，可分别运行以下入口，输出到使用者新建的私有位置。

```bash
python -B code/e18_confirmatory_execution/qualify_safe_idle_retirement.py --output <new-private-receipt.json>
python -B code/e18_confirmatory_execution/qualify_cooperative_slot_admission_v2.py --output <new-private-receipt.json>
python -B code/e18_confirmatory_execution/qualify_member4_fine_cooperative_runner.py --output <new-private-receipt.json>
python -B code/e18_confirmatory_execution/qualify_member4_tail_partition.py --output <new-private-receipt.json>
python -B code/e18_confirmatory_execution/qualify_member4_tail_partial_resume.py --output <new-private-receipt.json>
python -B code/e18_confirmatory_execution/qualify_incremental_aggregation_planner_v5.py --output <new-private-receipt.json>
python -B code/e18_confirmatory_execution/qualify_native_walltime_metadata.py --output <new-private-receipt.json>
```

这些命令复核软件与合成fixture。打包阶段的测试重跑数为0。资格摘要仅保留状态、原始检查计数、源SHA、异常类别、合成模式与原记录的零真实调用计数。原始traceback、临时目录和实际部署记录保持私有。

## 修订与失败历史

- tail controller初版资格记录21项software errors。
- planner v5前两份资格分别记录12项和2项software errors。早期收据的异常类型细节未记录，公开摘要按原计数保留。
- cooperative首版26项PASS对应较窄的argv入口假设。v2增加显式dispatcher路径与伪入口回归并记录29项PASS。两个版本的源与摘要分开保存。
- `software_archive/tail_partition_b921` 保存修正前的运行身份约束版本。它属于追溯材料，当前生产入口使用最新45c控制源。
- `software_archive/planner_v5_provisional_b62` 保存早期registry版本，最终planner使用5f517源。

部分更早controller和资格源码SHA尚无对应副本进入本包，记录保留这一历史重跑缺口。当前最终入口的源与资格绑定可逐项核对。

## 发布边界

本包使用通用源码、历史修订源码、软件资格摘要和本说明。真实题目、标准答案、奖励、逐调用输出与计时ledger、scope、authorization和运行policy继续留在私有工件。真实单记录walltime事实及相关检查文件保留于原私有说明。

优势是代码字节、版本关系和软件负记录可按SHA追踪。风险是私有资产、实际硬件与进程资源环境仍由使用者提供和核验，软件检查计数的依赖与重叠也需保留其原含义。
