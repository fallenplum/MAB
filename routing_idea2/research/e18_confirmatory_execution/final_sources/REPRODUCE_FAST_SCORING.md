# E18 正式评分快速入口

## 执行依据

用户后续批准直接开展正式答案评分，并复用已经完成的native全量审计。快速入口依次流式载入155份已审计归档中的73,490份新调用收据、4,717份历史成员缓存，核对选定集合及已记录Token总数，构造原评分器所需context。

原始`e18_final_scoring.py`的`execute_scoring`负责正式评分，原`ScopeMemo`提供冻结scope的摘要复用。此次实际入口的`original9b_execution_gate_rerun`为`false`，完成审计的SHA继续作为执行接纳依据。

主执行进程报告，正式运行已于2026年11月16日19时16分载入全部收据并记录`ENTERING_FROZEN_SCORING`。最终质量与费用结果由正式输出另行归档。本源码包记录执行入口与已有人工软件资格。

## 冻结身份

| 资产 | SHA256 |
|---|---|
| FINAL_CPU_SCORING_MANIFEST.json | `8341f6ef86f063c64b3701f24846cbbff3177ffc67ffcf3ab9388192f69bf40e` |
| CLOSED_NATIVE_INDEPENDENT_AUDIT_FINAL.json | `8367d50ef3138c4c0d7327fe18ebc883a93b709d28bdfcdcb59f6c19ef8f8655` |
| fast_scoring_from_audit.py | `75bbc675947696d1420c20a1aab8e1a38d9110de9a97fb35b612cd1b999f9c7e` |
| cpu_scope_memo.py | `7fd45e003f09c8fc342e5b90f72b0e96f18e08bca63045c90fe13a424d15ed75` |
| e18_final_scoring.py | `9b164c40ad6b3a1b21f84c7dc446438a11ce6421d56991d82d224ae2c4b0daa8` |

## 实际命令结构

下列变量对应已授权的原有私有资产和CPU工作目录。将已发布原E18源码、操作层源码、CPU memo及本入口放在同一代码目录，使用原已记录依赖环境。

```powershell
$e18Private = '<E18私有资产目录>'
$e18Code = '<E18完整冻结源码目录>'
$e18NativeAudit = '<CLOSED_NATIVE_INDEPENDENT_AUDIT_FINAL.json完整路径>'
$e18Runtime = '<E18 CPU运行目录>/scoring'

python -B "$e18Code/fast_scoring_from_audit.py" `
  --manifest "$e18Private/FINAL_CPU_SCORING_MANIFEST.json" `
  --manifest-sha256 8341f6ef86f063c64b3701f24846cbbff3177ffc67ffcf3ab9388192f69bf40e `
  --completed-audit "$e18NativeAudit" `
  --completed-audit-sha256 8367d50ef3138c4c0d7327fe18ebc883a93b709d28bdfcdcb59f6c19ef8f8655 `
  --archive-directory "$e18Private/native_exports" `
  --historical-zip "$e18Private/HISTORICAL4717_MEMBER_NATIVE_CACHE.zip" `
  --runtime-root "$e18Runtime" `
  --performance-receipt "$e18Runtime/PERFORMANCE_AUDIT_REUSE_SCORING_v1.json" `
  --execute-scoring
```

性能收据使用新的文件名。断点恢复和单题评分收据遵循原`execute_scoring`的既有规则。完整复现需要授权私有资产以及上述冻结版本。

## 已有软件资格

`FAST_SCORING_FROM_AUDIT_QUALIFICATION_v3.json`记录8项人工软件测试全部通过，测试源绑定`5482599e2afd1daea946ff1a561db29235ccc41a79c28da3d6ff1458fdb612bf`，入口绑定`75bbc675947696d1420c20a1aab8e1a38d9110de9a97fb35b612cd1b999f9c7e`。该版本采用2个人工归档任务，真实收据、真实答案、真实题目读取以及模型与评分调用均为0。

本次发布整理复用这一已有资格，测试重跑数为0。公开QA保留测试计数和源码SHA。真实运行收据、题目标识、答案、逐题评分及私有矩阵归入私有工件。

## 优势与风险

该入口利用已经完成的物理输出完整审计，直接调用冻结评分实现，减少正式评分前重复扫描的时间。其适用前提是原manifest、完整审计、归档目录和历史缓存身份与已完成审计一致。实际性能及科学结论依据本次正式输出，人工软件资格仅覆盖实现检查。
