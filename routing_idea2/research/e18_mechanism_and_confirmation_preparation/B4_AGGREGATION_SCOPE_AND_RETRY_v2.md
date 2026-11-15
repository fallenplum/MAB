# 聚合缓存查验范围与失败重试预算

## 新端点已检查的范围

新端点已通过原主机pin并完成只读认证。本次实际扫描的两个项目目录如下，目录内只列名称、文件类型、大小和mtime。

- `/data1/data2/chendong/routing`，32个父级目录项。
- `/data1/data2/chendong/routing/parallel_workers_20261103`，12个父级目录项。

另检查六成员原native及DeepSeek两个明确variant工作区的成员JSONL。主机地址只保存在私有握手收据，公开结果用host散列标识。

## 聚合结果的身份结论

本次已核验可复用fresh400未评分聚合输出为0。这个0表示尚未取得身份核验通过的聚合输出，全集是否存在其他缓存保持UNKNOWN。当前没有打开聚合结果载荷。

父目录metadata中已经看到typed train/validation、旧诊断及strong/heterogeneous聚合目录，以及primary-aggregator-slot0/1/2三个目录项。该观察只覆盖名称与stat，并未进入这些目录读取聚合文件。其它MAB副本、其它数据根、尚未展开的export档案也保持UNKNOWN。

## 原失败重试约束

原native成员worker与正式stage2聚合worker均无自动付费重试。首次失败会停止并要求核对，unknown-cost/PENDING journal和共享成本fence阻止未经核对的再执行。

member的max-new-jobs仅限制一次invocation内的新成功调用数，默认未设数值。底层Store没有累计max_attempts参数。36份正式聚合RUN_PLAN没有max_attempts/max_retries/retry_limit/attempt_cap/max_calls字段。跨人工恢复及重新授权后的总attempt数值上限尚未锁定。

因此，完成全部唯一成功key所需的调用数与Token上限，应明确标为one-success资源口径。包括后续人工批准重试的全流程hard cap在原协议中缺少数值约束，当前记为null。未知失败成本需要保留并核对，不能因自动重试为0而填成0。

## 优势与限制

优势是区分已观察目录、已核验输出和未覆盖位置，并将成功唯一输出预算与失败恢复预算分别记录。限制是本次没有完整扫描所有聚合payload或导出档案。缓存存在性仍有未完成核查范围，当前结论限于已记录的只读观察。模块内fresh标准答案、真实评分、sealed文件读取及新生成任务均为0。
