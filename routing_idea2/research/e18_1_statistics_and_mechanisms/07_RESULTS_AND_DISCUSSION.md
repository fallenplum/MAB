# Results and Discussion draft

## Analysis status

E17 used 600 repeatedly examined development questions. Its 100 action renumberings were controlled interventions on the same questions and recorded generations. E18 evaluated four frozen policies on independently reserved questions. Of 3,000 requested questions, 2,998 were scorable, comprising 1,500 GSM8K and 1,498 BBH questions. Two BBH questions had incompatible visible-choice and reference mappings, and their rewards remained missing. E18.1 was specified after inspection of the E18 results. Its 20,000 stratified paired bootstrap resamples retained all three generation repetitions within each original-question cluster. The intervals below are marginal 95% percentile intervals conditional on the frozen policies, present source mixture and recorded generations.

## Results

### A champion convention altered the learned preference over physical teams

Empirical best-team ties occurred in 462–467 of the 480 training questions in each E17 outer fold. Renumbering the 20 actions changed the designated champion while preserving team membership, rewards, inputs and model capacity. After mapping predictions back to physical team identities and fixing semantic-exact inference, SPO_coordinate retained the same team on an average of 6.135% of questions across 100 renumberings. Its accuracy ranged from 82.889% to 87.389%. Ridge and TSPO_uniform retained 100% identity consistency, with constant accuracies of 85.278% and 85.500%, respectively. The original SPO model selected combination 0 on 467 of 600 questions. These interventions demonstrated sensitivity of the fitted SPO router to the champion convention under this training protocol.

Training symmetry and inference priority contributed separate effects. TSPO_uniform had 397 of 600 questions with a top-score gap at most 10⁻⁸, despite a maximum score-equivariance residual of 6.25×10⁻¹¹. Its identity consistency was 100% under semantic near-tie priority and 37.308% under numeric near-tie priority. Holding the fitted weights fixed, changing numeric near-tie priority changed an average of 376.15 selections. This inference control localized a second source of selection sensitivity to the deployment tie rule.

### The order of reward averaging changed the supervision objective

Using the same eight joint reward draws, features, capacity and regularization, MeanThenLoss achieved 85.111% development accuracy and LossThenMean achieved 86.000%. MeanThenLoss changed 404 selections and produced 54 corrected and 70 damaged generation outcomes, yielding −0.889 percentage points relative to LossThenMean. Its differences were negative for all three recorded generation seeds. A known-truth two-action distribution with mean rewards (0.54, 0.55) gave different proxy-optimal decisions under the two operators even when the reward mean was exact. The controlled example established an operator-level distinction, while the real-data comparison retained the negative result for MeanThenLoss.

### The frozen evaluation separated selection dispersion from quality

On the 2,998 scorable E18 questions, task-macro accuracies were 86.6637% for Ridge, 86.3961% for SPO_coordinate, 86.4079% for TSPO_uniform and 86.8634% for GlobalFixed. SPO_coordinate assigned 83.02% of questions to combination 0. TSPO_uniform used all 20 combinations, and its most frequent combination accounted for 9.27%. The two policies selected different teams on 2,778 questions. Across the 8,994 recorded generation events, TSPO corrected 377 outcomes and damaged 376, giving a net improvement of one event.

| Frozen-policy comparison | Task-macro difference, pp | Marginal 95% interval, pp | Original paired-t p |
|---|---:|---|---:|
| TSPO_uniform − SPO_coordinate | +0.011749 | [−0.666934, +0.690062] | 0.973048 |
| TSPO_uniform − Ridge | −0.255837 | [−0.845142, +0.333526] | 0.394418 |
| TSPO_uniform − GlobalFixed | −0.455570 | [−1.145424, +0.233987] | 0.200390 |

The primary estimate was close to zero, with uncertainty spanning both quality gains and losses. The recorded task-macro seed differences were +0.2010, +0.2676 and −0.4334 pp. These results leave the practical quality benefit of the symmetric training treatment unresolved within the frozen setting.

### Opposing task effects explained the aggregate cancellation

TSPO_uniform lost 0.933333 pp to SPO_coordinate on GSM8K, with a marginal interval of [−1.555556, −0.333333] pp. On BBH, its point gain was +0.956831 pp, with an interval of [−0.245327, +2.158433] pp. The exploratory difference between the BBH and GSM8K effects was +1.890165 pp, with an interval of [+0.532644, +3.247382] pp and a two-sided Welch p of 0.006619. Task labels were observational. The BBH marginal interval retained substantial uncertainty around its positive point estimate. All 27 BBH subtask results were retained in the accompanying analysis, with their support counts and multiplicity diagnostics.

## Discussion

The strongest evidence concerns the mechanism connecting an arbitrary champion to a shared predictor. For two tied champions s and t, the SPO+ loss difference is 2(q_s−q_t), giving different gradient contributions under a shared linear head. E17 perturbed that convention directly while controlling the semantic inference rule. E18 then measured the quality consequences of frozen policies on separate questions. Together, the experiments support a reproducible change in selection behavior and delimit the observed quality effect to a near-zero overall difference with opposing task directions.

The interpretation requires three separate properties. Action-renaming equivariance concerns whether a semantic relabeling preserves physical-team decisions. Reward consistency concerns the optimizer of a population surrogate risk. Generalization concerns the performance of a fitted finite-capacity model on new inputs. Our known-truth examples expose gaps between these properties. Squared loss targets conditional mean rewards in the unrestricted population problem. The original SPO+ analysis provides distributional conditions for consistency, including continuity and conditional central symmetry. The present finite binary reward vectors and regularized linear class require their own scope analysis. [Smart “Predict, then Optimize”](https://arxiv.org/html/1710.08005v5)

Prior routing research already connects sampled outcomes to supervision instability, and identifies deterministic winner construction as a source of label priors. Our evidence adds a loss-specific gradient identity, semantic-preserving renumbering controls, inference-only tie controls, an identical-draw operator comparison and a frozen quality evaluation. This contribution is bounded to the audited ensemble-routing setting. DARS studies distribution-aware supervision, while EquiRouter analyzes persistent model preferences and cost-aware ordering. [DARS](https://aclanthology.org/2026.emnlp-main.333/), [EquiRouter](https://arxiv.org/html/2602.03478v2)

Several constraints shape the empirical conclusion. E17's repeated development use limits its role to mechanism diagnosis. Three recorded generations provide limited information about per-question conditional reward means. The E18 source mixture includes a BBH expansion, and two input-reference incompatibilities reduced the evaluable set. The bootstrap describes approximate question-sampling uncertainty under the present mixture, conditional on the fitted models and observed generation repetitions. The intervals have marginal coverage targets. E18.1's post-result specification gives the task interaction an exploratory role. The deterministic missing-outcome range for the full 3,000-question primary difference, [−0.055556, +0.077778] pp, retains a separate finite-record interpretation.

The current evidence favors an audit contribution centered on how supervision conventions alter routing behavior. Further claims about beneficial routing require a frozen candidate whose quality improvement is supported in a separately defined evaluation. The observed task heterogeneity provides a concrete target for that design, alongside explicit control of training champions, inference tie priority and source compatibility.

## 中文Results草稿

### 任意冠军规则改变了模型对真实组合的偏好

E17每个480题训练折中，462—467题存在多个经验最优组合。我们保持实际组合、输入、奖励、容量和预处理不变，同步改变20个动作的编号。将输出映射回原组合身份并固定语义精确推理规则后，SPO_coordinate在100次重编号中的平均身份一致率为6.135%，准确率变化范围为82.889%—87.389%。Ridge与TSPO_uniform的身份一致率均为100%，对应准确率分别固定为85.278%和85.500%。原SPO在600题中有467题选择组合0。该干预表明，在当前训练协议下，冠军构造约定塑造了拟合路由器的组合选择。

训练对称性与推理优先级还存在独立影响。TSPO的分数等变最大残差为6.25×10⁻¹¹，但397道题的最高分间隔不超过10⁻⁸。语义近并列规则保持100%身份一致率，数字坐标近并列规则得到37.308%。固定训练权重后，仅改变数字近并列优先级，平均改变376.15道题的选择。该控制将推理数值边界与冠军标签的训练效应分别定位。

### 相同奖励样本上的运算顺序会改变监督结果

在同一批8个联合奖励draw及相同特征、容量、正则下，MeanThenLoss准确率为85.111%，LossThenMean为86.000%。前者改变404道题，纠错54个生成事件、破坏70个，点差为−0.889pp，三个生成seed方向均为负。已知均值(0.54,0.55)的两动作反例中，两种运算顺序产生不同代理最优选择，即使奖励均值完全准确，这种差异仍然存在。反例说明算子差异，真实数据保留MeanThenLoss的负结果。

### 独立题目上的质量差接近零，并伴随任务抵消

E18原计划3000题，其中2998题可评分，包含1500道GSM8K及1498道BBH。另2道BBH题存在可见选项与官方目标无法唯一对应的输入缺陷，其奖励保持缺失。Ridge、SPO_coordinate、TSPO_uniform和GlobalFixed的任务宏平均分别为86.6637%、86.3961%、86.4079%和86.8634%。SPO在83.02%的问题上选择组合0，TSPO使用全部20个组合，最高频组合占9.27%。两者在2778题上选择不同，8994个可评分生成事件中纠错377次、破坏376次，净纠错+1。

主点差为+0.011749pp，边际95%区间为[−0.666934,+0.690062]pp，原配对t检验p=0.973048。TSPO相对Ridge为−0.255837pp，区间[−0.845142,+0.333526]pp。相对GlobalFixed为−0.455570pp，区间[−1.145424,+0.233987]pp。主点估计接近零，当前误差范围同时容纳质量收益和损失。

GSM8K主差为−0.933333pp，区间[−1.555556,−0.333333]pp。BBH点差为+0.956831pp，区间[−0.245327,+2.158433]pp。探索性“BBH增益减GSM8K增益”为+1.890165pp，区间[+0.532644,+3.247382]pp，Welch双侧p=0.006619。三个生成seed的任务宏平均差分别为+0.2010、+0.2676和−0.4334pp。任务标签为观测分组，BBH自身正向点值仍有较大误差范围。

## 中文Discussion草稿

当前最明确的发现是任意冠军约定进入共享预测器的机制。真实目标向量上并列的两个冠军，会产生2(q_s−q_t)的SPO+损失差及相应的梯度差。E17通过同步重编号和固定语义推理直接干预这一约定，E18随后评价冻结策略在独立题目上的质量。联合证据支持组合偏好发生了可复现变化，独立质量结果则表现为总体近零及任务方向抵消。

编号等变、总体代理目标与有限样本泛化各有不同问题。置换对称控制回答语义改名是否保持真实组合选择，已知真值反例回答代理风险的最优解是否对应条件均值最优，独立评价回答实际拟合模型在新题上的服务质量。平方损失的无约束总体最优仍为条件均值，有限线性容量与正则需要单独解释。原SPO+一致性理论有明确分布条件，当前离散正确性奖励保留自身适用边界。[原始SPO+](https://arxiv.org/html/1710.08005v5)

已有研究已经讨论随机生成导致的路由监督不稳定，以及确定性冠军标签诱导的先验。本研究的可辩护增量在于损失层梯度关系、语义重编号、推理并列隔离、相同draw运算顺序和冻结质量评价构成的受控链条。DARS侧重分布感知监督，EquiRouter研究模型级偏好与费用排序。当前组合设置保留该文献边界。[DARS](https://aclanthology.org/2026.emnlp-main.333/), [EquiRouter](https://arxiv.org/html/2602.03478v2)

E18.1在查看E18后确定统计补充方案，采用20000次分任务题级配对percentile bootstrap。同一原题的三个生成重复保持成簇，100次E17重编号仍共享600题。区间为边际近似区间，条件于冻结模型、当前题源混合与已记录生成。BBH扩展来源、三次生成、两道输入缺陷及E17开发重复使用共同限制外推。完整3000题主差的确定性缺失范围[−0.055556,+0.077778]pp保留其有限记录含义。

本轮更适合以监督约定如何改变路由行为为核心贡献。任务异质性提出了可检验的后续问题，下一阶段的质量收益主张需要另行冻结候选、数据角色、推理规则与确认方案。

## 写作与证据说明

- 主体结构按冠军干预、算子隔离、独立质量及任务异质性展开，每段对应一项观测或解释。
- E18.1数值来自主执行进程的`E18_1_COMPLETE_RESULTS.json`，本写作子任务未执行bootstrap。
- 区间与p值各自保留方法身份。区间来自题级bootstrap，表中p值来自原配对t方法，任务交互p来自Welch方法。
- 原E17/E18文件保持既有版本。文稿作为E18.1新增解释，尚需作者审阅。
- 本机draw.io CLI缺失。联合图保留可编辑XML，SVG与PNG由同一几何场景本地渲染，渲染来源写入`figure.qa.json`。
