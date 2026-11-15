# E18执行准备的软件资格补充

## 当前入口

`code/e18_execution_adapter.py`提供精确范围构造、审批绑定、原MemberStore和typed聚合Store的单任务入口。`code/e18_reference_adapter.py`提供批准题目集合内的单题reference投影。`code/e18_scoring_adapter.py`使用冻结确定性评分器并保存稀疏评分和执行费用账本。`code/sparse_policy_projection.py`将已执行且有合法评分收据的组合投影至四套冻结策略。

资格使用临时合成输入、FakeBackend和原Store接口。各资格JSON与源码哈希随本版封存。真实GPU加载、完整3000题调度及真实Gold投影属于批准后的运行阶段。

## 运行和审批边界

准备阶段只创建审阅用DAG和冻结预测。收到明确执行批准后，执行者将批准范围写入hash绑定收据，引用协议、题目身份、冻结动作、来源、预算及各组件scope。用户无需手工编写收据。原逻辑transport使用online_stream，外部研究身份固定为e18_confirm，原全局数据划分保持原字节。

执行前在目标主机核对模型路径映射、权重、tokenizer、运行依赖与实际GPU状态。以精确缺失key分配任务，合法历史缓存走已核验的复用桥。原成本journal及未决费用围栏持续生效。成员seed为42、137、2024，聚合decode与匿名排序使用原seed42，成员重复通过provenance绑定。

Gold投影scope按已批准源及原行定位冻结。部分格式的源SHA和结构定位需要扫描其他记录的编码字节，扫描权限在源scope中显式记录，未授权reference值保持未解码状态。格式或定位无法满足要求时入口拒绝处理。正式评分的reference按canonical_question与reference_version复用，输出分别计分。

## 优势与风险

优势是复用原提示、typed合同、实际Token记录和Store接口，同时将执行范围限制到已冻结策略并集。合成软件资格覆盖权限、身份、seed、复用及失败处理。主要风险为正式GPU环境、实际源文件与完整队列尚未进行生产端到端验证。通过软件资格只支持实现完整性结论。

BBH扩展来源1254题及3000题总体范围仍需用户确认。未知远端缓存范围、人工续跑的累计费用和语义近重复残留均沿用v1报告中的限制。
