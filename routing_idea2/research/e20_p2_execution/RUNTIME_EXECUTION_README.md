# P2原生执行与接续

## 当前阶段

正式2400题执行持续进行，目前采用五个注册的互斥聚合分片，成员profile按原队列继续运行或自然完成。P2_ACTIVATION_STATUS.json保留首次启动时点，P2_THREE_SHARD_ACTIVATION.json和P2_FIVE_SHARD_ACTIVATION.json分别记录两次扩展的实际首成功时点。最终质量评价需等待全部所需输出闭合。

实际作用域要求41289次成员生成和106536次聚合。每道题按B/C正概率支持集的并集执行，三个生成seed为42、137、2024，聚合seed为42。raw-exact动作已包含于主支持并集。模型参数、提示、生成参数、压缩和评分算子身份沿用冻结协议。

## 代码与数据角色

| 文件 | 职责 |
|---|---|
| prepare_runtime_sources.py | 从既有合格E18执行器生成P2概率作用域与worker技术适配 |
| p2_execution_adapter.py | 两策略概率支持并集、hash绑定权限与原生单任务执行 |
| p2_scope_freeze.py | 沿用E18只读JSON容器，支持不可变scope的单次摘要缓存 |
| p2_member_worker.py | 每模型常驻后端，逐个执行冻结成员键并保留原生断点 |
| p2_aggregation_worker.py | 沿用成熟Qwen3常驻上下文与tokenizer物理路径映射 |
| p2_aggregation_follower.py | 增量读取已提交成员journal，依赖齐套时执行冻结聚合 |
| prepare_launch.py | 输入、预测、预算、作用域与审批绑定，生成v2生产清单 |
| launch_native_profile.py | 依据当前GPU进程、显存和利用率启动指定profile |
| monitor_native_status.py | 每个profile取最近启动记录，避免同journal重复计数 |
| p2_export_native.py | 已提交原生owner增量闭合归档，SQLite按语义键去重 |
| record_activation_status.py | 从真实运行元数据生成脱敏启动快照与私有控制交接 |
| collect_native_exports.py | 单向流式导入闭合包，以SQLite及持久offset管理NTFS原生JSONL |
| qualify_native_collector.py | 中断恢复、重入去重、跨进程并发和最终完整性门槛的软件资格 |
| migrate_aggregation_once.py | 已有单聚合进程的安全接续，绑定既有语义任务与原生检查点 |
| p2_manifest_registry.py | 原七来源的显式登记与追加三份互斥聚合manifest |
| prepare_aggregation_shards_v3.py | 从历史成功并集生成互斥剩余键分片 |
| p2_aggregation_follower_v3.py | 原上下文与算子下执行当前注册subset，并恢复旧/新成功键 |
| launch_registered_worker_v3.py | 核对分片互斥、旧actor检查点及实测显存后启动注册worker |
| monitor_native_status_v3.py | 读取全部十份注册来源，按语义键与physical attempt分别去重 |
| p2_export_native_v3.py | 从全部注册来源闭合原生owner，继续复用导出索引 |
| switch_aggregation_to_three_shards.py | 本次旧完整actor的精确身份安全退出与检查点核验 |
| record_three_shard_activation.py | 从既有本地收据生成脱敏上线快照 |
| prepare_aggregation_shards_v4.py / launch_registered_worker_v4.py | 同一冻结作用域下的五路互斥计划及原准入函数复用 |
| scale_aggregation_to_five.py | 精确核对并安全停止原三个聚合actor |
| record_five_shard_activation.py | 仅凭既有本地控制响应生成五路首成功快照 |
| prepare_member_shards.py / launch_registered_member.py | 慢成员剩余键互斥分片及原worker接续 |
| stop_registered_workers.py | 显式自有PID的安全停止与实际journal检查点 |
| record_runtime_fill_activation.py | 从本地控制响应核对四成员及六聚合全部首成功 |

独立题目输入、身份、概率、部署清单、原生输出和控制日志存入私有工件。生成侧不加载标准答案。fresh400与sealed200均在本次独立2400题作用域之外。

## 并发与持久上下文

三分片首成功快照中的成员profile0至5分别位于GPU4、7、7、7、5、6，聚合shard0至2分别位于GPU0、1、3。资源准入依据实际进程及连续三次显存观测，共享GPU授权继续记录。冻结profile为NF4，保留各其他作业进程。

每个成员profile只创建一次原冻结后端。首次构造执行既有原始模型源验证，后续任务持续复用该后端。聚合进程只加载一次作用域和原始Qwen3上下文，按成员成功journal增量建立依赖队列。

旧完整聚合actor在北京时间2026年11月17日00:53:35安全退出，保留586个唯一成功结果。注册表先保留原七份来源，再依次ADD三份分片manifest。三份剩余集合分别含34883、35826、35241个聚合键，总数105950，与旧586个成功键共同覆盖原106536键。

北京时间00:55:43快照显示三分片各有3、4、2个成功结果，六成员加三分片共9个活跃actor，重复成功行和成功键额外付费attempt均为0。显存准入的三次观测下限分别为22755、35231、36245 MiB。该记录证明上线和接续路径，吞吐改善继续依据后续匹配时间窗统计。

用户随后指定在GPU1和GPU6增加聚合。UTC 03:51:56，原三个actor完成在途调用后安全停止，历史唯一成功为20842。原106536键的剩余85694键重新互斥分成五份，GPU布局为0、1、3、1、6。原10份注册来源继续保留，再追加5份至15。

UTC 03:56:25首成功快照中，五份各有14、14、15、14、17次成功，合计新增74，总聚合20916，成员26436。该时点活跃actor为10，重复成功及额外付费attempt均为0。成员自然完成后活跃数会下降，其完成状态和后续吞吐分别由监控快照记录。旧三路报告和16项本地资格的原数字及准备状态继续保留。

后续慢成员4采用四份接续，GPU布局为5、4、4、5；聚合采用六份接续，GPU布局为0、1、3、1、6、3。原成员检查点1778与四份1145、1205、1231、1214严格恢复6573个原需调用。原聚合检查点38939与六份11529、11821、11784、11131、10490、10842严格恢复106536个原需调用。现有后端、生成参数、权限scope与缓存owner继续沿用。新增注册后共25份来源。

四成员及六聚合的实际启动与成功由P2_RUNTIME_FILL_ACTIVATION.json单列。其入口仅处理本地完成的控制响应，核对各新actor均有成功原生记录、当前仍活跃、重复成功及成功键额外付费attempt为0，再写入不可覆盖收据。17项原合成资格保持原准备状态。吞吐改善使用后续观测窗口另行记录。

## 运行时目录和接口

远端执行根为`/data2/chendong/e20_p2_execution`，模型与Python环境复用既有数据盘目录。主要子目录包括`code`、`inputs`、`manifests`、`owned`、`control`和`exports`。

本地私有工件位于`E:\语言模型组合路由\research\e20_p2_execution\private`。私有`control/RUNTIME_HANDOFF_PRIVATE.json`提供当前控制交接与监控入口。凭据只存在已认证连接的进程内存中。

现行轻量监控调用`monitor_native_status_v3.py --registry PATH`。增量回传调用`p2_export_native_v3.py --registry PATH --max-tasks 500`，返回已闭合TAR路径、任务数、Token及归档SHA，再将归档取回本地`private/native_exports`。旧入口及其收据继续归档。只读核验直接流式读取TAR，不在exFAT展开大量小文件。

## 本地收集与评分门槛

调用`collect_native_exports.py`将已闭合包导入私有NTFS工作视图。每次SQLite写事务串行保护成员和聚合JSONL，字节先fsync、offset及去重键随后提交。中断后截断至已提交offset，再继续接收。已经导入且文件身份未变化的包直接命中导入索引。

两包真实资格样本共1840个输出，成员1582、聚合258，已知成功输出Token为750737。该数字描述两包收集资格时点，实时队列进度由监控快照记录。

三分片切换后的第四份闭合包通过现有收集器，00:57:00本地快照累计3915个输出，成员3298、聚合617，已知成功输出Token为1629791。两包资格原始数字继续保留。远端首成功和本地收集分别标注各自时点。

全部41289个成员与106536个聚合输出齐备后，收集器才生成`ALL_REQUIRED_OUTPUTS_FROZEN`，绑定runtime scope、预测、模型、审批、参考定位计划及两份原生JSONL的完整哈希。盲评入口核对该收据，再开放获批2400题的参考读取及评分。部分输出阶段的参考读取与评分计数保持0。

费用状态分别记录成功输出和全部attempt。成功输出实际Token从唯一原生记录累计；失败及未知attempt由独立账本闭合，待闭合时`failure_closure_pending`为true、`all_attempt_total_tokens`为null。输出完整性与费用完整性分别标记。成功调用墙钟求和继续保留其原生含义。

## 交付快照与定稿范围

发布清单使用显式ready文件表，保留源码及结果快照各自的角色。收集器资格、两包实际导入、单次聚合迁移和扩容分析分别归档。扩容分析JSON记录本地源码复核与速率敏感性，其部署状态由独立运行收据确定。

registry及v3分片入口已完成本次实际部署，其首成功证据单列于P2_THREE_SHARD_ACTIVATION.json。20项dispatch、14项consumer及合计34项工具资格的JSON继续保留原本地准备阶段状态。清单生成属于本地打包操作，实际部署状态以新增运行收据及其UTC时点为准。

## 已处理的启动失败

初版审批遗漏原生执行器要求的execution_limits，两支worker在native owner预留前安全退出，实际LLM调用0。v2审批与manifest补齐两阶段精确上限，绑定绝对输出根，原失败日志和审批版本保留。

聚合follower初版将原helper返回的`(tokenizer, receipt)`整体当作tokenizer，在make_job阶段退出。实际聚合调用0、原生owner预留0。按原接口解包并保存tokenizer别名资格收据后续跑，已经产生真实聚合成功及实际Token。首条真实聚合通过既有原验证器的科学语义兼容检查。

## 继续与完成条件

成功语义键继续复用。失败预留owner须先按原生状态分辨是否已产生未知用量，保留失败及重试成本。禁止把已成功任务重新调用。

全部41289成员及106536聚合输出完成后，冻结原生输出清单和预测投影，再由盲评分接口读取获批新2400题的标准答案。成功输出成本、失败和未知attempt费用分别核算并保留闭合状态。sealed200答案保持封存。最终报告保留完整正负结果、不可评分题、原题统计单位和任务各50%的主权重，按项目当前约定输出点差与配对检验。
