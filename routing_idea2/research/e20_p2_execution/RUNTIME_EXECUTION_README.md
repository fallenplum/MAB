# P2原生执行与接续

## 当前阶段

正式2400题执行已启动。六支冻结成员模型及一个依赖驱动聚合进程均已进入生产队列。P2_ACTIVATION_STATUS.json记录固定UTC时点的真实成功调用及Token，运行状态为RUNNING。最终质量评价需等待全部所需输出闭合。

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

独立题目输入、身份、概率、部署清单、原生输出和控制日志存入私有工件。生成侧不加载标准答案。fresh400与sealed200均在本次独立2400题作用域之外。

## 并发与持久上下文

当前使用GPU4、5、6、7。GPU4、5、6与既有其他作业共享，GPU7承载多支P2 NF4上下文。资源准入依据实际进程及多次利用率观测。冻结profile为NF4，已加载显存实测用于新增上下文判断，保留各其他作业进程。

每个成员profile只创建一次原冻结后端。首次构造执行既有原始模型源验证，后续任务持续复用该后端。聚合进程只加载一次作用域和原始Qwen3上下文，按成员成功journal增量建立依赖队列。

## 运行时目录和接口

远端执行根为`/data2/chendong/e20_p2_execution`，模型与Python环境复用既有数据盘目录。主要子目录包括`code`、`inputs`、`manifests`、`owned`、`control`和`exports`。

本地私有工件位于`E:\语言模型组合路由\research\e20_p2_execution\private`。私有`control/RUNTIME_HANDOFF_PRIVATE.json`提供当前控制交接与监控入口。凭据只存在已认证连接的进程内存中。

轻量监控调用`monitor_native_status.py`。增量回传调用`p2_export_native.py --max-tasks 500`，返回已闭合TAR路径、任务数、Token及归档SHA，再将归档取回本地`private/native_exports`。只读核验可直接流式读取TAR，不必在exFAT展开大量小文件。

## 已处理的启动失败

初版审批遗漏原生执行器要求的execution_limits，两支worker在native owner预留前安全退出，实际LLM调用0。v2审批与manifest补齐两阶段精确上限，绑定绝对输出根，原失败日志和审批版本保留。

聚合follower初版将原helper返回的`(tokenizer, receipt)`整体当作tokenizer，在make_job阶段退出。实际聚合调用0、原生owner预留0。按原接口解包并保存tokenizer别名资格收据后续跑，已经产生真实聚合成功及实际Token。首条真实聚合通过既有原验证器的科学语义兼容检查。

## 继续与完成条件

成功语义键继续复用。失败预留owner须先按原生状态分辨是否已产生未知用量，保留失败及重试成本。禁止把已成功任务重新调用。

全部41289成员及106536聚合输出完成后，冻结原生输出清单、真实成本与预测投影，再由盲评分接口读取获批新2400题的标准答案。sealed200答案保持封存。最终报告保留完整正负结果、不可评分题、原题统计单位和任务各50%的主权重，按项目当前约定输出点差与配对检验。
