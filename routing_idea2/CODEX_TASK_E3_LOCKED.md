# CODEX TASK｜E3 固定 K=3 决策定向采集机制验证（基于已完成 E2）

日期：2026-11-04  
来源：fallenplum/MAB main 当前 E2 归档；从 main 拉取，不从旧版 `codex/a-fixed-k3-e2-e3` 继续。  
定位：**锁定 E3 正式对照的执行/审计任务说明**，不是已完成 E3 的证明。

## 研究结论边界

- E2 完成固定 N=6/K=3、20 组合、5 策略 × 5 流种子 × 4 评分预算 × 3 延迟的 300 组回放。工程 `E2_PASS`，但科学 `INCONCLUSIVE`。
- 预指定 `none/25条每百题/延迟0` 未见题组合排序 51.1573%，题级分任务95%区间49.9871%–52.2770%，跨越50%。317/600 题支持非平局排序（BBH203/GSM8K114）。
- 描述性条件 `delay=0`、5流种子平均：rate25 时固定73.33%、contextual_topk74.80%、none74.73%、linucb74.90%、thompson73.67%；rate50 时分别73.33%、75.50%、74.23%、73.60%、72.93%。未经题级配对区间，不宣称显著或模型结构因果优劣。
- E2 使用1份真实生成缓存，5个是算法流随机种子；原20,800次物理调用与300组逻辑回放资源分账。已有缓存仅E0锁定的Qwen2.5/256原V1聚合器；此前Qwen3/2048后验因果诊断是**强模型三次采样**，不能替代异构组合奖励，也不能与V1拼接。
- 单模型作为辅助系统参照，不把超越单模型设为E3算法核心成立条件；**主要证据是同执行/反馈约束下决策定向采集的独立贡献**。

## A. 进入 E3 前：只读验证，禁止覆盖锁定代码/数据

逐一读取 README、E0_LOCK、experiment_v1、E1/E2 manifests、`release/e2_20261104/E2_AGGREGATED_RESULTS.json`、`DELIVERY_MANIFEST.json`、`scripts/run_online.py`、`scripts/evaluate_e2.py`、`scripts/evaluate_e3.py`、`src/budget_router/{simulation,acquisition,learner,budget,features,stage_validation}.py` 及测试。

1. 读取/独立校验本地正式 `e2_gate.json`、全部 300 组 episode receipts、200 题校准矩阵、600 题在线矩阵及 manifest SHA，确认与 E2 公共交付中的 run_identity `2eb9e98505dcd88aadcbbc590cf200125c77340f6741b09df314d9c3bd829c96` 相符。
2. 重新检查原 `E0` 锁、PCA fit只在200道 online_calibration；无E1题/参考/embedding进E2/E3，没有封存集提前评测；在线 episode 当前题 gold、当前reward、真实未来长度不会参与选队或购买。
3. 确认在 `main_feedback_naturally_arrives=false` 时：购买`main_only`花1条标注；购买`paired`花2条标注且新增模型、聚合器调用完整计费；已到达标签才更新，原始标签只更新一次；共用候选在实际诊断中可缓存，但部署逻辑服务成本不能免算。
4. **不得修改** `stage_validation.SOURCE_FILES` 的文件及原协议后直接沿用旧 E2 技术门！其源码哈希是绑定的一部分。没有 E2 Gate 或当前资产不匹配时停止为 `BLOCKED_E3_INPUT_IDENTITY`，列出缺失，而不是重新算一个没有可比性的 gate。
5. 首先只做软件级安全测试和 gate/binding 检验；如果软件测试需要修复核心代码，保留原锁定源码/结果快照，另开修改版协议和重新验证链，**不把新实现冒充旧E3**。

## B. 原样执行 E3：不再基于 E2 结果调公式

E3 原设 `E3_STRATEGIES` 共7项：
`none,random,periodic,max_uncertainty,max_uncertainty_per_cost,no_cost,decision`。

要求 **7 策略×5 流种子×4反馈率×3延迟=420组** 的完整逻辑回放，使用 E2 已执行的原聚合奖励缓存；不应有新增模型生成。所有训练规则、label权限、预算和warmup不变。E3复用已通过哈希验证的E2作为比较基线，绝不重跑或篡改E2成绩。

操作上根据原实验机真实路径构造：`python scripts/run_online.py --stage e3 --online-matrix <真实文件> --online-manifest <真实文件> --calibration-matrix <真实文件> --calibration-manifest <真实文件> --output-dir <全新目录> --e2-gate <原真实e2_gate.json>`。完整 grid 不加`--strategies/--rates/--delays`子集参数，已存在相同 identity 的未完成目录可`--resume`，不得覆盖不同 identity 结果。E3完整网格完成后再用`python scripts/evaluate_e3.py --runs-dir <相同目录>`独立评价；没有全部420组不可称E3技术PASS。

所有原始结果保存不可变 SHA、trace/模型动作/预测值/真实逻辑资源/控制器 CPU 与 wall time，出具实验账本。

## C. 预声明主诊断与必要对照

保持`scripts/evaluate_e3.py`中**已经声明**的主要测试不变：
- 条件：25条最终评分每百题、delay0、5个流种子；
- 主实验：`E3/decision` 对 `E3/max_uncertainty_per_cost`；
- 主要门槛：服务质量配对差95% CI下界>0，且工程定价的实际成本差95% CI上界≤0，才可宣称严格质量-成本支配。该门槛并非证明非零净效用的唯一方法。
- 同时报告`decision vs random`、`decision vs none`、`decision vs no_cost`、`decision vs max_uncertainty`，以及 E2 的固定、Top-3、LinUCB和Thompson。
- 主图：最终已冻结服务质量 vs 累计实际完整逻辑成本；副图：服务质量 vs 累计反馈标签。另报每法实际标签数、审计动作数、额外调用数、token、时间。预算上限一样不等于真实花费一样。
- 按题做配对分析，5流seed在同题先合并，按GSM8K/BBH分层重采样；不可将3000次同题重复看作3000个独立题，也不可将2000 Bootstrap视作生成种子。
- 预先保留全部格点结果；只对已预注册主格点作确认性阐释，其他格点是描述性/稳健性分析，避免事后挑选好看结果。

## D. 需要附加的机制审计（**不能篡改锁定E3主实验**）

对已有 trace 做只读诊断：每种策略的`main_only`/`paired`/不购买分布、平均每标签新增模型调用与聚合次数、9种单成员替换邻居选择分布、审计任务类型分布、预测信息收益与实际下一阶段错误修正的关系。

尤其需指出：当前`random`和`decision`可能选择不同的动作**类型**，前者均匀从可行动作选而后者按信息收益/成本选；因此“优于随机”可能混合了购买数量/类型和比较对象选择的影响。

建议**在完成并锁定原 E3 后**另行预注册`E3b action-type-matched random`作为补充实验：在决策前匹配合法购买类型与可承担的标签预算，再随机选同类型的替换邻居。代码改动必须形成新的source hash/单独 run identity，不能混同原 E2/E3验证链；本次先列设计和预期信息权限，必要时报告用户审批。

其他只读诊断：若 E3 仍失败，检查活跃组合的二阶价值预测器排序在不同任务/预算上的稳定性；注意名义特征 `17×22=374` 而反馈稀缺，同时不可将固定K下回归参数当作唯一的“真实协作强度”。不在当前600题上事后调整特征或采集阈值，然后把同一批题当独立的支持性证据。

## E. 汇报与下一步 Gate

最终交付：
1. `reports/e3/E3_PROTOCOL_AND_PROVENANCE.md`（和E2版本绑定）；
2. `reports/e3/E3_CORE_RESULTS.md`（主比较与五种子、四预算、三延迟及分任务结果，带置信区间）；
3. `reports/e3/E3_COST_AND_ACTION_AUDIT.md`（执行、标签、tokens、控制器成本、主/备动作类型）；
4. `reports/e3/E3_RESEARCH_DECISION.json`：分别标注`technical_status`、`scientific_primary_conclusion`、`single_cache_limitation`、`E4_readiness`和`next_recommended_action`。

如果E3主比较缺显著质量增益或者不满足成本条件，如实报告`INCONCLUSIVE/NOT_SUPPORTED`；不把`E3_TECHNICAL_PASS`等同于方法有效。此时先分析二阶预测器是否失配、反馈样本是否有用，以及同权限成员正确性监督是否可改进预测，而不是直接投入E4全部近邻复现。若E3主机制产生可复核的独立增益，才进入E4与CES、C2MAB-V同权限对照。

正式代码/报告推送到本研究分支，保留原 E0/E1/E2 和正式输出不变；任何外发文件用隐私白名单，不能推真实题、参考、生成日志、权重或认证信息。最终向用户给出链接、主要数值、全部不利结果及科学结论。
