# P2原生输出收集

## 实际部分导入

两份已闭合TAR包的资格样本任务数为444与1396。该次导入取得1840个唯一成功输出，其中成员1582、聚合258。成功输出实际Token为501844输入、248893输出，合计750737。

这些数字对应已导入成功输出。失败及未知attempt的账目继续单独闭合，`failure_closure_pending`为true，全部attempt的总Token保持null。累计调用墙钟求和单列，不解释为独占GPU时间。

原始闭合包继续保存在E盘。收集器以单向gzip/TAR流读取当前条目，核对owner的三份文件及末尾manifest，不在exFAT展开小文件，也不对已导入且文件身份未变化的包重复扫描。

## 入口与私有工作视图

```powershell
python -B -X utf8 collect_native_exports.py
```

默认来源目录为`E:\语言模型组合路由\research\e20_p2_execution\private\native_exports`。

默认NTFS工作目录为`D:\develop\CV\多臂老虎机\.e20_p2_cpu_runtime`。

| 工件 | 用途 |
|---|---|
| NATIVE_COLLECTION.sqlite | 已提交输出键、原生SHA、包身份、成本与持久化offset |
| MEMBER_NATIVE_RECEIPTS.jsonl | 唯一成功成员的原生P2 receipt |
| AGGREGATION_NATIVE_RECEIPTS.jsonl | 唯一成功聚合的原生P2 receipt |
| COLLECTION_PROGRESS.json | 脱敏进度、成功Token及失败成本待闭合状态 |
| OUTPUTS_FROZEN.json | 全scope输出完整时才产生的最终评分入口收据 |

需要指定单包或其他派生工作视图时，可使用`--archive`与`--work-root`。原归档始终以只读方式打开。

## 提交与恢复

SQLite使用WAL及FULL同步。每个包导入期间持有`BEGIN IMMEDIATE`写事务，跨进程共享同一份offset和去重索引。

两份原生JSONL使用无缓冲二进制追加。所有新增字节先flush及fsync，再提交SQLite中的offset、输出键与闭合包记录。中断后，新的导入器取得写事务，并将文件截断到数据库已提交offset。已经提交的输出继续保留，未提交尾部被回收。

同一stage、语义任务键与native SHA只计一次。同一语义任务出现不同成功native时会报冲突，保留失败记录，不自动选择其中一个。相同physical attempt不能重复收费。

SQLite写事务同时保护最终快照，避免在其他导入器追加时计算全量文件哈希。部分进度查询只读取已提交成本与计数。全量JSONL哈希在首次完整输出冻结时计算。

## 原生身份资格

复用已完成E18的独立owner验证函数，只适配外层技术schema及cohort命名。原native payload和其SHA保持科学身份检查对象。核对真实P2 scope、审批、目标键、canonical题身份、owner fence、冻结ModelSpec、prompt、seed、成员组合、实际Token和调用墙钟。

归档中的三文件字节SHA与长度逐项匹配末尾manifest。归档必须完整结束并通过gzip校验。任何晚出现的manifest或native冲突都会回滚本包的数据库变更和JSONL未提交尾部。

## 最终门槛

只有全部41289个成员和106536个聚合输出齐全时，才生成`ALL_REQUIRED_OUTPUTS_FROZEN`收据。其scope文件SHA固定为`835dc9fa321ff3b20c95d5a9410ce18e79084e01453e66a94b6e3501e9ca0d06`，canonical scope摘要为`b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9`。

收据同时绑定预测b790…、模型b8b…、当前AUTHORIZATION_v2以及参考定位计划b24…。成员及聚合JSONL的完整哈希供`score_paid_outputs.py`使用。失败成本尚未闭合的状态继续明确保留，它不阻止记录已完成输出。

当前1840条部分输出没有触发最终门槛。参考答案读取、评分和新增LLM调用均为0。

## 资格与限制

`qualify_native_collector.py`的12项合成检查通过，覆盖重入去重、重叠包、晚失败回滚、实际进程硬退出后的offset恢复、两个进程并发导入以及完整覆盖后的文件哈希门槛。

优势是输出本体保持原生、逐物理attempt去重、恢复位置明确，且既有归档不会被重写。风险包括源包尚未回传完整、导出范围仅覆盖已成功提交的输出，以及未物化失败或未知attempt费用。完整质量分析仍须等待全部输出完成并冻结。
