# E18 批量正式评分入口

本入口沿用冻结manifest、已经完成的native审计、原reference provider与`score_answer`。它在单进程内批量保存评分receipt与事件，使用已授权的审计复用context，并调用原`e18_final_scoring.execute_scoring`。

本次用户指定结果采用点值呈现。入口的运行态统计适配保留原点估计和原配对检验公式，区间计算次数为0。中文最终报告由独立`generate_point_only_report.py`生成。

## 执行

使用原已记录依赖环境，将本入口与已有E18冻结源码、`fast_scoring_from_audit.py`及`cpu_scope_memo.py`放在同一代码目录。

```powershell
$e18Private = '<E18私有资产目录>'
$e18Code = '<E18完整冻结源码目录>'
$e18NativeAudit = '<CLOSED_NATIVE_INDEPENDENT_AUDIT_FINAL.json完整路径>'
$e18Runtime = '<全新E18 bulk CPU工作目录>'
$e18Public = '<全新E18完成结果目录>'

python -B "$e18Code/bulk_scoring_from_audit.py" `
  --manifest "$e18Private/FINAL_CPU_SCORING_MANIFEST.json" `
  --manifest-sha256 8341f6ef86f063c64b3701f24846cbbff3177ffc67ffcf3ab9388192f69bf40e `
  --completed-audit "$e18NativeAudit" `
  --archive-directory "$e18Private/native_exports" `
  --historical-zip "$e18Private/HISTORICAL4717_MEMBER_NATIVE_CACHE.zip" `
  --runtime-root "$e18Runtime" `
  --public-root "$e18Public" `
  --performance-receipt "$e18Runtime/PERFORMANCE_BULK_SCORING.json" `
  --execute-scoring
```

完成审计SHA固定为`8367d50ef3138c4c0d7327fe18ebc883a93b709d28bdfcdcb59f6c19ef8f8655`，由继承的快速入口校验。科学输入与选队保持原manifest身份，运行目录单独记录。

## 已有资格与限制

v2软件资格记录4项人工测试通过，入口SHA为`812e72bed58ae6ef2aa50fd4464318965d3a6223f67da46afb45b75d89e3afd0`，资格源码SHA为`3ff1b849a8804374847d44a560544dfa2318aca3c7a77238c9e11b49ad78c74f`。其真实Gold读取、模型调用、原执行门禁重跑、原区间辅助函数调用及分位点计算次数均为0。

本次staging复用已有资格，新增测试数为0。正式运行的逐条等价核对和完整结果由主执行进程交付。私有评分receipt、事件、原题和答案按原工件规则保存。

批量I/O减少逐题SQLite和文件操作，保留原确定性判分函数。中间日志采用本入口的批量格式，故障恢复及来源核验需要同时保留对应入口、输出目录和运行收据。完整科学结论依据正式完成结果。
