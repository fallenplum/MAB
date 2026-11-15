# E18执行作用域与入口就绪审计

## 当前结论

**本轮完成准备方案及纯软件资格。面向3000题、四个冻结策略调用并集的付费runner状态为NOT_IMPLEMENTED。** 原V2底层科学算子和缓存资产具备复用基础，新cohort的作用域适配、稀疏队列物化、native工作计划及实际评分读取入口仍需实现并核验。计划DAG或元数据资格PASS仅覆盖对应准备工件。

本审计没有启动新生成任务，也没有读取Gold、新聚合输出或sealed payload。完整科学operator继续固定为`9ef6c02b…cb440e`，基础模型配置指纹为`32aed607…861d3`。typed prompt、随机种子、压缩、tokenizer与parser保持原科学身份。

## 旧入口的实际限制

| 原入口 | 已核查的约束 | E18需要的适配 |
| --- | --- | --- |
| `budget_router_v2.full_feedback.read_inputs` | `COUNTS`固定600/200/400，fresh固定旧400题 | 新输入cohort独立manifest、3000题身份与来源层绑定 |
| `full_feedback.read_outcomes` | 每题完整20组合、三生成重复的奖励与成本矩阵 | 仅执行冻结策略并集的稀疏评分连接 |
| `full_feedback.fit_bundle/predict_bundle` | 原V2 selector及四方法schema | 直接使用E17冻结的Ridge/SPO/TSPO/GlobalFixed动作 |
| `full_feedback.evaluate_predictions` | 原方法名、完整矩阵、旧bootstrap比较协议 | 使用E18已冻结的题级配对双侧统计及次要比较约定 |
| `stage2_member_adapter.load_member_api` | 原800题与固定fresh400的qid、输入hash和manifest | 绑定新3000题输入且保持旧模块隔离 |
| `stage2_fresh_admission.validate_fresh_admission` | 800个开发ID、原model schema、旧四方法400×20分数 | 绑定E17 full600头、E18四策略3000个动作及新来源层 |
| `run_stage2_members_native.main` | 每模型全部phase共2800条、旧RUN_PLAN与host目录 | 从新调用key并集生成独立phase及工作计划 |
| `stage2_aggregation_adapter.load_aggregation_api` | 1200题投影及固定成员job身份 | 仅接受新manifest中的qid与同生成重复三成员 |
| `materialize_stage2_aggregation.materialize` | 每题六成员齐全，再构造全部20队 | 按已冻结qid/seed/team集合仅物化必要三成员依赖 |
| `build_stage2_aggregation_bundle.task_scope` | cohort600/200/400，任务数为题数×20 | 新稀疏任务manifest与精确分片覆盖 |
| `run_stage2_aggregation_native.validate_shard_jobs` | 每题20队、旧cohort与旧host/assignment根 | 新集合覆盖守卫及独立输出/assignment命名空间 |

因此，修改一个`COUNTS`常量仍不足以完成新作用域支持。旧fresh准入使用的模型schema与E17冻结四头不同，完整20队矩阵也与本轮30921次聚合并集不同。现有入口的精确函数签名、源码位置与SHA已记录在`EXECUTION_ADAPTER_READINESS.json`。

## 可复用的科学内核

### 成员生成

`budget_router.member_execution.make_member_job`接收问题ID、任务、原题、成员alias、seed、固定ModelSpec、split、subtask及样本角色。它构造原`independent_user_only_brief_final_v2`提示及内容寻址job。`validate_member_job`核对提示、模型、seed和成员角色。`MemberStore`提供既有缓存与实际调用记录语义。

E18需要新的input-bound scope adapter。该适配器应在独立模块中接受冻结cohort，核对qid、canonical hash、原题hash、task/subtask、固定模型及所需成员key。原底层split集合还包含其他实验角色，新适配器应明确接受本轮cohort。作用域扩展无需修改原科学源码。

### typed聚合

`budget_router.execution.make_job`负责匿名顺序、512-source-token压缩和基础消息。`typed_aggregation_execution.upgrade_job`附加原task-typed输出合同，`validate_job`核验合同及来源hash。旧stage2适配器以隔离模块绑定原代码，并将所用三份成员结果逐项绑定到对应成员job、record hash及生成重复。

E18应沿用该隔离方式，在新的cohort适配器中绑定实际三成员依赖。每份聚合只接收已成功且已核验的成员输出。所需新代码属于输入成员资格和执行作用域，科学消息、合同、压缩及parser继续使用原冻结字节。

### seed必须分别记录

| 字段 | 固定约定 |
| --- | --- |
| 成员生成seed | 42、137、2024 |
| 成员sample_index | 0 |
| 成员sampling_control | false |
| 成员采样 | temperature0.7、top_p0.95、max_new_tokens2048 |
| 聚合调用seed | 42 |
| 聚合来源的member_generation_repeat | 42、137、2024之一，与三份成员provenance一致 |
| 聚合模型采样 | temperature0、top_p1、max_new_tokens2048 |

聚合job的seed与成员生成重复承担不同角色。缓存key中的`generation_seed`表示成员生成重复，聚合调用仍使用42。旧聚合worker也显式核对这两个字段。

## 最小新cohort适配计划

1. 冻结新的执行输入manifest，引用Unicode修复v2、原qid、canonical、task/subtask、source_frame及冻结动作。1254道BBH full-source扩展单列来源层。字节和题目角色核对先于任何实际调用。
2. 新增`load_e18_member_api(contract)`类型的隔离作用域入口。它包装原成员工厂/validator，将成员生成key限制到精确付费并集。对既有缓存使用已审核的语义key与原物理执行receipt建立复用桥。
3. 新增按缺失key创建成员bundle的物化器和native计划入口，继承既有模型映射、唯一任务归属、journal、实际Token和未决成本处理语义。新的phase数量应来自本轮key集合。
4. 新增`load_e18_aggregation_api(contract, member_generation_seed)`类型的作用域入口。它包装原匿名工厂和typed合同，核对每个必要team的三份同seed成员记录。
5. 新增稀疏聚合物化器和native计划入口。精确覆盖已冻结team集合，允许每题不同数量的策略去重组合。输出根、assignment与计划schema独立于旧stage2。
6. 在评分开放前冻结实际评分读取入口、允许的Gold源投影、qid/canonical映射、scorer SHA、评分receipt和最终统计脚本。Gold按题复用，各实际执行输出单独判分。
7. 通过稀疏分数投影形成`[policy,3,question]`最终正确率数组，再调用E18题级配对统计。全部策略动作使用既有冻结版本。

这些是批准后第一步应完成的非付费软件适配及资格范围。新的native runner、真实receipt读取和实际Gold投影目前均为NOT_IMPLEMENTED。涉及物化消息或开启backend的付费执行命令留待实际实现和资格后提供。

## 已补充的纯稀疏结果接口

`sparse_policy_projection.project_frozen_policy_scores`接收冻结动作与已核验评分记录，按qid、canonical、成员生成重复、实际team、operator和scorer连接。输出形状为`[policy, generation, original_question]`。其自身没有文件读取、Gold访问、模型调用或执行授权。

22项合成资格已通过，覆盖稀疏选中分数连接、共享动作复用、缺失、重复、异operator/scorer、错误seed、qid/canonical不符、team列错配、非法分数、provenance格式和同一聚合record被错误赋予多个key。合成只供应45条选中评分，完整20队矩阵需要360条，接口按必要集合投影。实际源receipt的字节和合法评分权限由未来输入适配器核验。

资格结果为`SPARSE_RESULT_PROJECTION_QUALIFICATION.json`。其`paid_runner_status`、`gold_reader_status`和`actual_scoring_adapter_status`均保持NOT_IMPLEMENTED。对合成连接器的PASS无需真实Gold或新模型输出。

## 计划DAG与可执行队列的边界

仅审阅的计划DAG可以记录成员/聚合key、依赖、原输入定位和冻结策略引用，并声明`authorizes_execution=false`。它不会自动满足native worker的计划、job schema、physical mapping及模型backend条件。

旧native入口读取`RUN_PLAN.json`、`logical_jobs.jsonl`、`physical_jobs.jsonl`、`JOB_ID_MAP.json`及相应native schema。新审阅DAG使用独立名称和明确的非执行状态。当前`materialize_remote_members_v4.py`是已存在成员缓存的只读SFTP镜像工具，其角色与新成员生成runner不同。

## 优势、风险与下一步

优势是科学operator已具备完整身份核验，冻结动作、输入编码及精确调用并集可复用。主要技术风险为旧qid/phase假设、完整20队物化、任务源域扩展及旧bootstrap统计入口被误用于新协议。旧聚合输入helper还使用`read_text().splitlines()`，新的输入加载器应使用文件逐物理行迭代，保留题目字符串内的Unicode行分隔符。

下一步技术门槛是将上述最小作用域适配和真实receipt读取实现为可检查的非付费预检，再决定实际执行。当前就绪程度应报告为“科学内核与输入/预算准备已核验，3000题native执行适配待实现”。
