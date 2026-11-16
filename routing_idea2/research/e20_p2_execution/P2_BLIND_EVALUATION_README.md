# P2盲评接口与资格

## 已完成资格

- 真实2400题输入预测与科学scope通过核对。GSM8K与BBH各1200题，27个BBH子任务的44/45题配额及稀缺子任务44题配额保持。
- 完整正概率支持并集为35512个题目–组合，三个生成重复要求106536次聚合及41289次合法去重成员输出。
- 两支模型、τ=1e−8部署及raw-exact概率由冻结得分逐值重建。模型、聚合算子、生成seed与输入身份匹配。
- 22项合成scope与统计资格通过，覆盖成员复用、执行越界、重复键、scope不可变快照、审批摘要、期望效用与题级统计。
- 11项参考投影与冻结评分器合成资格通过，覆盖完整输出门槛、封存排除、来源字节变化、标准答案去重以及GSM8K和BBH两种源格式。
- 真实参考定位计划核对2400条记录、28个来源、200条封存排除。参考答案解码0。

软件资格用于核查实现与信息权限，最终质量结果仍需完整真实输出与正式评分。

## 文件及入口

`blind_evaluation.py`提供输入侧资格、精确概率效用投影、原题配对双侧点检验、纠错与破坏、任务及seed分解。默认CLI仅读取输入侧预测，不读取任何分数或答案。

`score_paid_outputs.py`在完整输出冻结后，验证P2原生聚合receipt并调用既有冻结评分器。默认CLI仅检查完整输出门槛，只有显式`--execute-scoring`才打开经批准的新2400题答案。

```powershell
python -B -X utf8 qualify_blind_evaluation.py
python -B -X utf8 qualify_paid_scorer.py
python -B -X utf8 blind_evaluation.py --scope SCOPE.json --predictions P2_PREDICTIONS_FROZEN.npz --output INPUT_ONLY_QUALIFICATION.json
```

正式评分的`--scope`使用`EXECUTION_SCOPE_RUNTIME.json`，包含`bindings.final_model_sha256`及冻结策略概率。科学scope文件`EXECUTION_SCOPE.json`用于研究输入身份与执行需求记录。两者的成员及聚合并集应逐项一致，物理文件SHA分别保存。原生receipt的`scope_sha256`使用运行时scope的canonical JSON摘要，完整输出冻结记录的`scope_sha256`使用运行时scope的物理文件SHA。

生成侧的完整receipt按行存入JSONL，保留原`p2_execution_task_receipt_v1`结构。评分器只为复用旧语义验证函数临时映射技术schema及cohort别名，真实native payload、真实P2 scope、审批哈希和实际receipt摘要仍按新队列核对。

正式运行时scope已完成真实输入侧资格，私有路径为`E:\语言模型组合路由\research\e20_p2_execution\private\runtime_launch_v2\inputs\EXECUTION_SCOPE_RUNTIME.json`。物理文件SHA为`835dc9fa321ff3b20c95d5a9410ce18e79084e01453e66a94b6e3501e9ca0d06`，canonical摘要为`b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9`。原科学scope物理SHA为`6dce6da5f78604b8bcddde969598e1e923ff9dd75383b57d49f69d80a0fbf2db`。两者全部成员及聚合并集逐项一致。授权文件`AUTHORIZATION_v2.json`已包含生成、聚合、参考读取和评分四项操作。

## 完整输出冻结记录

正式评分要求一份包含以下字段的执行完成记录。该记录只能在所有所需成员与聚合输出完成后生成。

```json
{
  "status": "ALL_REQUIRED_OUTPUTS_FROZEN",
  "scope_sha256": "runtime execution scope file SHA256",
  "prediction_sha256": "b790643119b6d85b370d34b6547e89e1316c26b26ad0d4a005e047babc521c75",
  "final_model_sha256": "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92",
  "completed_required_member_calls": 41289,
  "completed_required_aggregate_calls": 106536,
  "aggregate_records_sha256": "complete native aggregation receipt JSONL SHA256",
  "approval_sha256": "P2 approval file SHA256",
  "reference_plan_sha256": "b24b870647fcd70c0b9479375e2eed2317e76f3dc963a21c830f231e9ae18060"
}
```

```powershell
python -B -X utf8 score_paid_outputs.py --scope EXECUTION_SCOPE_RUNTIME.json --predictions P2_PREDICTIONS_FROZEN.npz --outputs-frozen OUTPUTS_FROZEN.json --aggregate-records AGGREGATION_RECEIPTS.jsonl --reference-plan P2_REFERENCE_LOCATORS_FROZEN.json --approval P2_APPROVAL.json --output-root PRIVATE_SCORING --execute-scoring
```

该入口先完成所有原生receipt的支持键、模型、prompt、seed、Token、科学payload和完整JSONL身份核验，再实例化答案提供器。按官方源行号只解码已批准题的目标答案，其余记录值继续在语法层跳过。标准答案账户以canonical question与reference version去重。

## 统计及成本

真实组合评分为二值。冻结均匀策略效用是正概率组合评分的加权和，因此允许位于[0,1]。三个生成seed先在题内平均，独立统计单位为原始题。主任务权重固定GSM8K与BBH各50%，BBH子任务宏平均另列。

主比较为C−B，双侧α=0.05。任务分层配对均值采用Satterthwaite自由度的t点检验，只报告点差与p值。raw-exact、seed和子任务为预声明诊断。新置信区间与bootstrap均为0。

纠错与破坏延续E19最大同动作重合耦合。两支均严格唯一的换队子集另列整数事件计数。若参考答案与原评分器不兼容，整题记录为不可评分，保留全部执行费用，并在原任务分母下提供确定性缺失值范围。来源、canonical身份或模型完整性错误会中止评分，不能归为不可评分题而被排除。

真实成员、聚合、Token及失败调用账本由执行器维护，评分结果只增加标准答案账户、确定性评分次数与CPU开销。再次评分不产生LLM调用。

## 优势与风险

优势是支持并集与冻结策略逐项对应、答案访问位于完整输出冻结之后、所有对照共享合法执行复用权限，且期望效用没有被误当成二值观测。

风险包括B策略近并列支持较宽、旧开发配对方差与新源框架的差异、官方参考格式可能无法由冻结解析器评分，以及五折开发证据与独立确认的范围区别。合成资格不能代替真实新题质量结果。

最初一项合成夹具采用了冻结解析器不接受的`FINAL:`格式，已经改为其原有`Final answer:`格式。评分器源码与解析规则保持冻结版本。官方BBH源示例由原输入审计函数返回文本，本适配器在进入冻结字节投影前显式转换为UTF-8字节，合成BBH案例已覆盖该接口。
