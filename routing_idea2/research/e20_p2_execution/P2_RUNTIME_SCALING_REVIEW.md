# P2运行吞吐与最小安全扩容审查

## 范围与依据

仅作本地源码审查与父任务提供的计数计算。未连接SSH、未修改任何进程或生产文件、未新增LLM调用。研究对象、作用域、两支模型、提示、生成参数与任务身份保持既定冻结版本。

以15时35分至15时51分约16分钟窗口粗算。

| Profile | 成功增量 | 每分钟成功调用 |
|---|---:|---:|
| Member0 | 85 | 5.31 |
| Member1 | 70 | 4.38 |
| Member2 | 52 | 3.25 |
| Member3 | 130 | 8.13 |
| Member4 | 32 | 2.00 |
| Member5 | 195 | 12.19 |
| Aggregation | 117 | 7.31 |

成员合计每分钟约35.25次。完整清单的聚合与成员调用比为106536/41289，约2.58。当前窗口聚合完成增量与成员增量之比为117/564，约0.21。该比率说明聚合处理容量值得优先检查，单凭比率还不能区分依赖等待与GPU服务速度。

按当前单聚合窗口速率外推，剩余106406次聚合约需243小时。Member4剩余6529次按每分钟2次外推约54小时。这些量仅用于压力测试当前调度容量，题型顺序、输出长度、共享GPU及启动阶段会改变后续速率，不作完成时间承诺。

## 先取得两项轻量事实

1. 读取聚合SUMMARY的ready、observed_member_successes、状态和最近成功时间。ready持续为正且逐步增加，支持聚合服务侧积压。反复为零且主要处于WAITING_FOR_COMMITTED_MEMBERS，优先看成员齐套。
2. 读取native实际completion_tokens及wall_time_s，分开看每分钟成功调用、每秒输出Token和调用长度。Member4长回答与共享GPU竞争应分别判断。

这些诊断只使用执行元数据，不涉及答案或奖励。当前源码每完成聚合就更新SUMMARY，30秒等待状态也更新，适合直接使用。

## 最小运维路径

第一种较小改动为单聚合迁移。若GPU4持续具备余量，SIGTERM原GPU7聚合进程，等待当前题成功提交、SAFELY_STOPPED汇总和进程退出，再用同一manifest、作用域和输出根在GPU4恢复。原follower从同一journal恢复成功集，原native owner继续去重。该方案无需先改并行任务协议，能够释放GPU7上三支成员共用的资源。GPU4的其他作业波动仍可能抵消收益。

若现有ready积压支持增加聚合容量，首轮并行上限设为两支聚合worker，分别承载互斥剩余键。可在资源合适时使用GPU7与GPU4。先评估匹配时间窗的总成功调用与Token速率，再决定后续资源利用。该设置只改变执行并发，完整执行键及科学配置保持既定身份。

### 两支聚合的安全切换顺序

1. 请求原聚合进程安全停止，等待当前调用完成、成功journal写入和进程退出。避免SIGKILL打断已预留的实际调用。
2. 从原journal与对应SUCCESS_COMMITTED owner确认已完成键D。若存在RESERVED或未知用量owner，先按原状态核对，暂不让新worker竞争该键。
3. 令待执行集合P为完整required_aggregate_keys减D。按问题与生成seed的固定哈希将P分成两份，确保同题同seed的全部组合交由同一worker，有利于成员输出复用局部性。
4. 核对D、P0、P1两两不交，三者并集精确等于原required集合。两个新manifest引用原scope、审批、模型映射、owned_output_root与原始科学算子，仅task_keys_ref及调度身份不同。
5. 新follower先登记旧成功集合或仅使用剩余P0/P1，并记录显式旧完成键来源。原paid cache继续按同一语义owner定位。
6. 两worker各使用自己的journal及worker身份，逐题记录实际Token。汇总成功时按语义任务键去重。

现有run_approved_task具有原生原子预留fence。相同键的两个worker竞争会引发异常并使follower退出，因此复制全量keys并发不具备可靠的工作分配行为。

## 扩容需要同步修复的消费者

| 现有假设 | 具体位置 | 扩容要求 |
|---|---|---|
| 只监听六份memberN_v2.json journal | p2_aggregation_follower.py第85行起 | 成员拆分前改为显式已批准journal注册表，并保留所有旧源 |
| completed只从当前manifest journal恢复 | p2_aggregation_follower.py第70行起 | 新manifest需要旧完成键来源或显式remaining-only集合 |
| 同profile只保留最新start记录 | monitor_native_status.py第17行起 | 多副本使用不同worker_id，按model_profile聚合，避免隐藏仍活跃副本 |
| 仅扫描名称匹配*_v2.json的manifest | p2_export_native.py第48行 | 扩为显式完整manifest注册表，包含旧、新所有生产journal |
| launcher按固定profile名寻找manifest | launch_native_profile.py | 新启动器需绑定明确manifest路径、SHA和worker_id |

只改task_keys_ref会遗漏这些运行时消费者，可能造成新成员输出不被聚合看见、计数遗漏或输出未回传。原scope、模型与预算无需因此更换。

## 后续慢成员路径

当前Member4为成员侧最慢，Member2次之。先补充其成功调用长度、completion达到2048上限比例、实际wall_time和当前GPU竞争，保持原生成上限与采样参数。

若增加聚合后成员齐套成为瓶颈，可对Member4的剩余键采用同样安全停止、已完成集恢复与互斥拆分。在增加成员副本前先启用显式member journal注册表，否则现有聚合进程不会观察新manifest下的成功记录。两个成员副本保持相同冻结逻辑profile、每题seed与batch1行为，继续共用语义owner去重。

待Member5或Member3等完整profile自然完成并释放上下文后，可将该资源分配给尚有积压的聚合或Member4副本。具体优先级依据ready积压与剩余执行量，使用执行元数据作运维判断。

## 优势与风险

优势是无需改变研究策略或增加科学调用集合，现有原生结果及cost ledger仍可复用。分步先迁移、再两份互斥聚合，是较小的运维改动路径。

风险包括共享GPU负载波动、初期BBH与后续GSM题长变化、模型重复加载费用、新旧journal遗漏以及过多上下文导致总吞吐下降。按实际峰值加KV与加载临时内存余量判断资源，并保留其他作业。

瓶颈判断置信度为中，依据为16分钟完成计数和源码结构。并发冲突、硬编码journal以及导出遗漏风险的判断置信度为高，依据为当前实现的直接控制流。

本审查尚未实施任何迁移或拆分。
