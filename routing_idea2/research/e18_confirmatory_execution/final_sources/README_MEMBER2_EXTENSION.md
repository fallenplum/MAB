# E18成员2调度扩展通用源码

## 范围与依赖

本扩展提供8份冻结原字节源码和6份脱敏软件记录。它叠加于先前77文件的final_sources包，保持code下e18_confirmatory_execution与e18_mechanism_and_confirmation_preparation两个同级目录。既有文件不重复复制、不修改。

四份新增实现为成员2持久worker/dispatcher、双former checkpoint broker和聚合planner v4，另外四份为对应资格及原聚合部分分片恢复资格。科学ModelSpec、生成参数、三重复、成员/聚合语义键和稀疏任务并集沿原冻结协议。

## 资格记录与失败历史

最新worker/dispatcher32项、broker32项、planner11项共75项合成软件检查通过。原聚合部分恢复另有8项，绑定旧de5c worker及715f dispatcher原字节。检查条目与继承用例不能当作独立研究样本。本次打包未重跑这些检查，也未读取Gold或正式奖励。

history保留planner v1的os.link权限失败。该11项运行记录0个断言失败和4个软件错误，模型调用及生产提交为0，属于软件环境权限问题。中间v2在0dd8源版本11项通过。最终1d76源版本由v3资格覆盖，历史记录分别保留，不能相互冒称覆盖。

## 公开边界

原源包含模型index、源码/不透明冻结metadata哈希、任务计数以及broker的TARGETS进程控制常量。这些是保留原SHA所需的程序guard，不是原题或逐题数据。TARGETS已绑定那次checkpoint来源，不能直接作为新部署的信号授权。实际source metadata、scope、approval、lease、resource policy、队列plan、进程快照及机器路径映射均保留私有。

MEMBER2_CHECKPOINT_READINESS原件含实际控制上下文，未复制。这里仅提供占位入口和依赖说明。SSH信息、原题/答案、奖励矩阵、native和模型权重不在包内。

源码和既有合成资格提供可审计的软件证据。实际Linux锁/遥测、GPU推理及产出完整性另沿真实执行来源核验，本次打包没有重新验证生产运行，也没有修改冻结源或推送。
