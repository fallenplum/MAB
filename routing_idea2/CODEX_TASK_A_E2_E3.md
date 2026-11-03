# CODEX_TASK_A：固定 K=3 的组合路由核心机制验证（E2→E3）

日期：2026-11-04  
仓库：fallenplum/MAB  
状态：任务规范；**不是已完成的实验结果**  
研究方案：A，固定 N=6、K=3、20 个无序组合，冻结原模型池与聚合器；不扩展 K，不微调语言模型。

## 0. 背景与不可变事实

请先完整阅读以下真实仓库文件，而非仅依赖本任务：
1. `README.md`、`routing_idea2/README.md`、`routing_idea2/E1_FILE_INDEX.md`；
2. `routing_idea2/protocol/E0_LOCK.json`、`routing_idea2/protocol/EXECUTION_CONTRACT.md`、`routing_idea2/configs/experiment_v1.json`；
3. `routing_idea2/results/e1/e1_diagnostics.json`、`routing_idea2/results/e1/causal_diagnostic_summary.json`；
4. `routing_idea2/src/budget_router/{features,learner,acquisition,simulation,ranking_evaluation,stage_validation}.py`；
5. `routing_idea2/scripts/{prepare_features,prepare_online_matrices,run_online,evaluate_e2,evaluate_e3}.py` 和相关测试。

已核实：原 E1 使用 300 道开发题、6 个模型、20 种真实三模型聚合，原始 8800 次调用，`PASS` **仅意味着逐题事后 Oracle 机会检查通过**，不意味着可学习或成本收益。后续补充 600 次聚合器诊断只覆盖同一个 Qwen3 的三采样；原聚合器 Qwen2.5/256 的 E1 奖励与后验 Qwen3/2048 的输出**不是同一奖励环境**。不得把两套结果拼接、把诊断结果冒充新鲜测试。GSM8K E1 来自官方 train；没有请求内金标泄漏证据，不代表训练污染已排除。

仓库归档不含逐题真值、原始生成、权重、运行认证与服务器状态。必须在 Codex 可访问的原实验环境恢复并验证私有资产；若缺失，输出 `BLOCKED_REAL_ASSETS` 和缺失清单，不能以造数、成员正确率或投票替代真实聚合分数。

## 1. 确认研究假设与科学门槛

主问题：固定组合空间中，在有限在线执行及评测反馈预算内，是否能用**决策定向的单成员替换对照**高效减少当前竞争组合的排序不确定性，提高未来问题的最终聚合质量。

- H2（可学习性）：共享二阶组合价值预测器在**未见在线问题**上的真实聚合奖励排序、top-1 决策损失和服务准确率，优于仅随机信息或无条件固定选择的相关强基线。仅有大 Oracle/大覆盖率不算证据。
- H3（采集增量）：共享**同一个**价值预测器、服务策略、共同预热、任务流与预算，仅改变标签购买与备选组合选择规则，本文 decision-aware information gain / incremental cost 的后续服务质量-成本曲线优于随机审计与最大不确定性审计；无成本归一化、无决策方向做单变量消融。
- 区分工程 Gate 与科学 Gate：`evaluate_e2.py` 中的 `E2_PASS` 是**技术完整性**，即使 `scientific_learnability_conclusion=NOT_SUPPORTED/INCONCLUSIVE` 也不能描述为算法有效；E3 技术 PASS 同理。单独出具 `scientific_gate.json`，保留负结果、不自适应修改阈值追阳性。

## 2. 先执行只读仓库/资产审计（不得先启动大规模 GPU 推理）

1. 检查仓库 main HEAD 与本分支来源、`release_manifest.json` 和 `E0_LOCK` 哈希，确认 E1 字节完整；记录 commit SHA、运行环境、数据和资产 SHA256；**不改写原 E1**。
2. 逐项确认原本机可用的 `online_calibration=200`、`online_stream=600`、`sealed=200` 分割及来源；检查规范化文本/来源 ID 去重、题目的任务归属和 BBH 子任务分层。E1 的 300 题及标签/embedding 不得进入后续 PCA、warmup、在线和最终评价。
3. 查看原完整数据与冻结生成缓存是否真的存在；不得因为 GitHub 有 `prepare_online_matrices.py` 就认为 800×20 个组合输出已经生成。在线 600 道题×20 组合=12000 次真实聚合，校准集 200×20=4000 次，共 16000 次；成员额外 800×6=4800 次，需复用**同一版本**已经有效的缓存。若缺资源，先输出具体所需新增调用与预算账目，不能自动大规模执行。
4. 核对所有线上矩阵只使用原 E0 V1 聚合器得到的真实组合最终评分；新聚合器只能在另一个单独锁版本、独立诊断与重做对应机会验证后作为次要研究协议。若业务上决定切换主聚合器，应显式提出 `E0_v2/E1_v2` 变更单并等待批准，不能沿用 V1 的 E1 PASS。
5. 快速静态复核 `features.py`、`learner.py`、`acquisition.py` 和 simulator 的前瞻性：当前题的原始输出、gold、执行后 token 长度、隐藏奖励不能参与当前决策；延迟奖励到达前不能更新；同一原始标签只入库一次；差值不得当额外样本；预算超过上限须拒绝。

## 3. E2：先证明路由价值可学习（不把系统算法与采集混在一起）

主特征保留原设定：冻结编码器 `all-MiniLM-L6-v2`；**仅** online_calibration 的 200 道题 fit PCA16，`phi(x)∈R^17`；组合基底 `[1,z_i,z_i z_j]` 22 维，Kronecker 特征 `psi(x,S)∈R^374`；固定 20 个组合全枚举。该表示因固定 K 存在不可辨识性，模型系数不是物理协同归因，不得画图宣称恢复真实协同矩阵。

真实的反馈学习：共享、计费的随机组合少量预热（不超过 32 标签和总预算 20%），后续仅消费已经到达的合法反馈；评测者保管在线全信息矩阵，路由器只能看到 `phi`、历史反馈和历史/开发集成本估计。

现有 E2 对照（优先复用原实现并审计定义）：
- paid-warmup best-fixed combination；
- legitimately observed member-score-based independent contextual Top-3；
- shared second-order ridge greedy（无额外主动审计）；
- same feature second-order LinUCB；
- same feature second-order Thompson sampling。

先运行单个正式预设条件（例：label_rate=25/100，delay=0），**5 个流/算法随机种子**作技术/科学预检查，再运行预注册 E2 全网格（反馈预算10、25、50、100/100题；延迟0、5、20；5种子）。这 5 个是算法/题目流种子，**不是重新生成 LLM 的 5 份缓存**。

必须输出：
- pre-decision 未见问题的非平局组合对排序准确率（0.5 偶然基准）及按问题分层配对 95% CI；
- 预测与实际组合价值差 MAE、top-1 决策 regret，正确/错误平局规则；
- 每道题最终服务准确率（GSM8K、BBH 和宏平均），质量-累计完整逻辑成本曲线；
- 服务调用、审计调用、反馈条数、Token、模型时间与路由 CPU/墙钟分开；
- 固定最优参照只能用已合法到达的付费反馈选，不能用在线标签构造“可部署最佳固定”；
- 任务分层以及任务切换前后可学习性的单列诊断。

**E2 科学停止条件（工程建议，不是先验定理）：** 如果未见题排序的配对 95% 区间不支持超过 0.5，且服务质量相对于合理固定/Top-3 对照没有正向改善，则给出 `E2_SCIENTIFIC_INCONCLUSIVE_OR_NEGATIVE`，先解释输入表示/样本稀缺/奖励失配，不把技术 PASS 当继续大规模 E3 的实证理由；不准在相同验证题上无限调参后当新测试。

## 4. E3：关键是隔离「采集」的独立贡献

每个对照使用同一 learner/主服务决策/校准预热/预算/题序/评分揭示协议：
- none（无自愿审计）；
- random audit；
- periodic audit；
- max reward uncertainty；
- max reward uncertainty / predicted incremental cost；
- decision information gain only（去成本分母）；
- **ours：decision-direction information gain / predicted full incremental cost**。

同时新增一个**动作类型匹配的随机策略**（如果主方案允许 `main_only` 与 `paired` 两种购买动作，普通随机可能改变反馈条数和购买类型；在给定相同反馈标签数/审计类型与可行集合的条件下，只随机挑选替换成员，作为更清楚的因果对照）。必须提前规定比较方法；不可利用运行后的处理条件挑选对子集。若匹配策略需要前瞻性随机化，则在决策前实现并记录。

本文工作代理：
`G(a)=sum_{d in D_comp} w_d*(d^T V^{-1} psi_a)^2/(sigma^2 + psi_a^T V^{-1} psi_a)`，
`Score(a)=G(a)/C_hat_incremental(a)`。
共同成员的既有回答可复用，新增成员、聚合、标签购买、失败调用都必须正确计费。该公式是**线性设计的代理信息收益**，不是对真实奖励改进的保证；若主答案没有自然反馈，则需同时购买主/备标签，绝不能把主标签视为免费。

比较在**相同反馈预算、相同执行预算上限**进行，同时出具各策略**实际使用成本**和可插值的质量-成本 Pareto 前沿；相同 cap≠实际消耗完全相等。主要报告连续流 service reward 与分任务结果，次要报告被选择审计的任务、竞争边、预测信息收益与实际后续错误修复；不许用当前题新增金标修改当前主输出。

建议首个预注册核心检验：
`ours vs action-matched random` 及 `ours vs max_uncertainty_per_cost`，反馈率 25/100、delay=0，在相同执行与反馈权限下作题级配对区间；其他预算档位与延迟作为稳健性扫描，不要从大量格点中事后挑最好声称 confirmatory。

## 5. 评测结论、数据边界与强近邻

- 评价基于固定缓存/一个生成种子时，只能说对该缓存条件的结果；至少 5 个流种子并不自动证明对模型采样随机性稳健。
- 高 Oracle 不能推导可学习性；高候选覆盖率不能代替聚合准确率；技术 E2/E3 PASS 不等于学术成功。
- 主线实验对照为固定/随机/独立 Top-3、结构化 LinUCB/TS、随机与最大不确定性审计；单模型可放辅助表，不要求在 E2 内超过单模型才允许开展算法比较。
- E4 强近邻预先准备 CES、C2MAB-V，检查原论文与官方源码对应的反馈权限、聚合机制、离线预训练信息与 K 设定；不同权限必须分成原设定与同权限适配版。其他需要核实的研究：AAAI 2025 `Cost-Aware Near-Optimal Policy Learning`，ICML 2026 `The Cost of Information: Phase Transitions in Contextual Bandits with Paid Observations`；主动采集/成本感知不是无人做过的新概念，创新需落实到当前具体模型组合与差分反馈、实证对照。
- 保留最强单模型和强模型三采样/直接投票结果作系统参考；不把超越单模型写成方案 A 的必要科学 Gate，但不能隐瞒系统负结果。
- 原 E1 使用官方 GSM8K train；不声明官方 test 或排除预训练污染，留出 200 题未授权不得提前动用。

## 6. 必须做的代码测试

实现/检查测试，至少覆盖：
1. 当下问题的 gold、聚合分数和实际 token 长度变化但历史与题干不变时，当前选择及审计动作不变（lookahead leakage test）。
2. 未到达延迟标签拒绝更新；标签重复不会双计；不同题共享候选生成按服务逻辑计费。
3. 固定 K=3 全枚举20组合，单成员替换恰有9个独特邻居；非法动作阻断。
4. 同题配对 Bootstrap 以题而非组合为重采样单位，种子变化不会冒充独立新 LLM 生成。
5. `E2_PASS` 技术完成但科学结论 `NOT_SUPPORTED` 时，不应自动写成研究有效；E3 技术完成同理。
6. `random` 与 `ours` 的合法反馈权限、动作成本、预热标签数和不同购买类型有账本审计。
7. 任何 E1 题、标签、embedding 出现在 PCA/在线学习时直接报错；任何封存标签提前出现在 evaluator 以外时阻断。
8. 软件合成测试与真实模型实验输出严格隔离；结果目录不可覆盖已有正式结果。

## 7. 交付顺序与停止规则

阶段 0（当前优先执行，不触发大规模模型生成）：
- `reports/route_A/00_repo_asset_audit.md`、`00_blockers.json`、`00_cost_plan.json`；
- 修复必要的公平性/数据泄漏/账本测试并提供 pytest 结果；
- 显式列出当前是否拥有 E2 所需 800 问×20 组合的**真实**评分矩阵；没有就 `BLOCKED_REAL_ASSETS`，不可伪造“E2通过”。

阶段 1（真实资产存在且协议一致时）：
- E2 有限预算可学习性运行记录、五种方法对照与 `E2_scientific_gate.json`；
- 正负结论均保留原始 traces、不可变哈希与任务分层结果。

阶段 2（E2 可学习性得到支持时）：
- E3 主机制对照与必要的动作匹配随机策略；`E3_scientific_gate.json`；
- 主图：未来问题最终质量 vs 累计**真实服务 + 探索 + 反馈**完整成本；另报准确率 vs 购买标签数量。

每阶段主要实验与审计通过后，将代码与可公开的指标/报告提交到**新的 codex 分支**，保留真实数据与权重于受控实验环境，不要 push 原始逐题题目和金标、密钥或服务器私有路径。禁止将 E2/E3 gate 拟成“研究完整完成”；用户和导师决定下一协议阶段。

在需要正式执行大量 LLM 生成或切换主聚合器前，先汇报预计新增调用数量、已经可复用的有效缓存及费用边界，并停止进一步生成，避免在不一致协议上重复烧算力。
