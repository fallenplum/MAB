# E19论文段落工作稿

## 证据状态

理论构造、小型数值资格、P0与P1均已完成。主执行进程实际运行10次主拟合、2次合法映射拟合及预定10000次分任务题级配对bootstrap。全部效用结果来自反复使用的train600外折开发资产，E17、E18与E18.1保留各自历史身份。本文稿以正式`E19_COMPLETE_RESULTS.json`为数值来源，SHA为`b9d0d2d0c340c26572be7441a92c20ea68807c436d252e2c01198a1692812435`。

## Theory draft

### Finite repetitions can collapse a strictly informative score difference

Consider two actions with rewards R(A)=0 and R(B) distributed as Bernoulli(a), for 0<a<1. A router receives L independent repetitions per question and fits a constant score difference d=q_A−q_B. With probability t=(1−a)^L, all observations tie at zero. If action A receives champion weight p on this event, the expected SPO+ objective with a fixed positive quadratic penalty has a unique optimum. Its score difference is zero exactly when tp≥1−t, and is strictly negative otherwise. The first positive integer repetition count yielding a strictly correct ordering is max{1, floor(log(1+p)/[−log(1−a)])+1}. Fixed L determines this population phenomenon. Increasing the number of independently observed training questions improves estimation of the same risk.

A zero score difference leaves a deployment choice unresolved. Under a fixed preference for A, the true mean-reward regret is a. Uniform selection gives regret a/2. The construction therefore connects empirical feedback ambiguity, a surrogate optimum and the separately specified service rule.

### A symmetric tied-row constraint can alter strict decisions on other inputs

We next fix three input types with probabilities 0.3, 0.6 and 0.1, features (1,4,4), (1,4,0) and (1,0,4), and reward vectors (1,1), (0,1) and (0,1). Uniform champion weights on the first type yield a symmetric absolute-gap penalty. Scaling only this term by β gives the convex objective

\[
F_\beta(\theta)=0.3\beta|\theta^\top\phi_T|
+0.6[2\theta^\top\phi_1+1]_+
+0.1[2\theta^\top\phi_2+1]_+
+30\|\theta\|^2.
\]

For every β∈[0,1], the exact minimizer is

\[
\theta_\beta=(-7/300,-2/25,-1/75)
+\beta(1/200,1/50,1/50).
\]

The score difference on the third input is (−46+51β)/600. It crosses zero at β=46/51 while the other strictly informative input keeps the correct ordering. With uniform selection on an exact predicted tie, expected service reward is 1 below this threshold, 0.95 at the threshold and 0.9 above it. The solution satisfies a common active region and has a zero gradient under a 60-strongly-convex objective. This example establishes a cross-input consequence of a direction-neutral constraint under fixed representation and regularization geometry.

### The empirical intervention separates direction from strength

For 20-action reward vectors whose empirical means all tie, uniform champion weights give 2(max q−mean q). We fix uniform weights on all empirical best-action sets and scale this term only on fully tied training rows. The empirical objectives satisfy F_β(W)=F₀(W)+βJ_T(W), where J_T=(2/N)Σᵢ∈T(maxⱼqᵢⱼ−meanⱼqᵢⱼ), with the original N retained. Both endpoints use the same fold-local features, 340-parameter linear head and original coefficient penalty, including bias. For exact endpoint minimizers, 0≤F₀(W_B)−F₀(W_C)≤J_T(W_C)−J_T(W_B). This inequality supplies an optimization check before interpreting changes in held-out scores or decisions.

The three-dimensional construction uses its stated penalty of 30||θ||². The empirical head retains c_f||W||² with c_f=0.003472222222222222 in folds 0–3 and 0.00034722222222222224 in fold 4. Parameter coordinates and normalization remain explicit.

## Results draft

### P0 confirmed the legal endpoint and renumbering implementation

On the preselected first fold, the B and C fits and their legal reverse-numbering checks required four real fits. The largest mapped centered-score residual over training and evaluation inputs was 2.723×10⁻¹⁰, relative to the frozen tolerance of 10⁻⁷. Uniform-policy mapping errors were zero for both branches. All four fits reported solved status, with a maximum stationarity residual of 3.619×10⁻¹² and primal–dual objective gap of 3.091×10⁻¹¹. Cross-evaluating the endpoint objectives gave 0≤ΔF₀=0.0271468≤J_T(W_C)−J_T(W_B)=0.1091279. The implementation gate passed, permitting the remaining folds under the predeclared continuation rule.

The decrease in the fully tied-row term was accompanied by separate changes in the remaining proxy loss and coefficient penalty. Moving from C to B increased the remaining data term by 0.0275547 and decreased the penalty by 0.000407851, giving the recorded ΔF₀ of 0.0271468. The same penalty coefficient was used at both endpoints.

The first fold contained 120 held-out questions. Under the frozen 10⁻⁸ uniform-argmax rule, B had 100 questions with multiple qualified maximum actions, while C had none. Both policies had strict winners on 20 questions and chose different strict winners on 12. This strict-change region produced three corrected and zero damaged recorded generation outcomes. Its contribution to the full-fold gain was +0.8333 pp. The score-tie region contributed the remaining +1.6667 pp under the specified randomized policy. Overall C−B was +2.5000 pp, with coupling-based expected corrections and damages of 18.6111 and 9.6111 generation-outcome equivalents. These fractional expectations were computed from the fixed cache and policy probabilities.

The raw floating-point exact-argmax sensitivity gave B accuracy 87.2222%, compared with 86.1111% under the qualified numerical-tie policy. C achieved 88.6111% under both definitions. We therefore retained strict-decision effects and score-tie policy effects as separate components of the first-fold result.

### P1 measured strict score effects separately from tie-mediated effects

Across the 600 outer-fold questions, centered scores changed beyond the frozen 10⁻⁷ threshold on every question. The maximum absolute centered-score change was 0.548920, with a mean rowwise L2 change of 0.290533. The largest common-score shift was 1.12×10⁻¹¹. Both endpoints had strict unique winners on 203 questions, and the winners differed on 150. Those strict changes produced 38 corrected and 19 damaged recorded generation outcomes, contributing +1.055556 pp to the full-pool task-macro difference. The remaining 397 questions had a numerical tie in at least one endpoint and contributed +0.386464 pp. This decomposition localized an effect beyond common offsets and predicted-tie resolution.

Under the prespecified 10⁻⁸ uniform-argmax policy, B achieved 85.057981% and C achieved 86.500000% task-macro accuracy. The all-question C−B difference was +1.442019 pp, with a 10000-resample stratified paired question-bootstrap 95% interval of [+0.190691,+2.722239] pp. Under the frozen maximal same-action overlap coupling with independent residual draws, expected corrections and damages were 89.646825 and 63.690476 generation-outcome equivalents, giving +25.956349 net equivalents. These fractional values summarize the specified randomized deployment policies. Strict-change counts retain their observed integer interpretation.

The task-specific differences were +0.501323 pp on GSM8K, with an interval of [−0.644235,+1.640653] pp, and +2.382716 pp on BBH, with an interval of [+0.143728,+4.671539] pp. The three generation-seed differences were +1.015212, +2.427910 and +0.882937 pp. Fold differences were +2.5000, +0.7341, +3.4491, +0.5864 and −0.0595 pp. The overlapping training histories retain the five folds as parts of one development analysis. All 27 BBH subtask results and support counts are retained.

The bootstrap kept both policies and all three recorded generations together within each sampled question. Its uncertainty conditions on the fitted heads and the repeatedly examined development pool. Task and numerical-sensitivity intervals have marginal coverage targets.

All five folds satisfied the endpoint inequality. The completed 12 real fits required 18.6875 CPU seconds and 19.1052 summed fit-wall seconds. The maximum KKT stationarity residual was 5.622×10⁻¹² and maximum primal–dual gap was 4.838×10⁻¹¹. The mean policy total variation was 0.872097, yielding a fixed-pool effect bound of 87.2097 pp. This conservative bound exceeded the declared 1 pp practical reference and supplied limited quantitative restriction on the observed effect size.

### The quality result depended on the declared numerical tie policy

The prespecified raw floating-point exact-argmax-uniform diagnostic gave B accuracy 85.5000%, C accuracy 86.5000% and C−B +1.0000 pp. Its corresponding 95% interval was [−0.333333,+2.333333] pp. Thus, the main policy had a positive development interval, while the raw exact diagnostic retained both negative and positive effects. The service definition materially affected the baseline and uncertainty. The main interval also spanned effects below the predeclared 1 pp practical reference.

The historical-cache comparison supplied a further precision boundary. Under the same 10⁻⁸ policy, eight fold 1 questions changed their maximum-set support from 13 to 14 between the historical E17 B cache and the newly fitted B. This changed pooled 600-question B accuracy by −0.008242 pp. Fold 3 had a maximum centered-score difference of 4.83×10⁻⁷ with unchanged policy support. The old–new objective differences were 1.405×10⁻¹¹ and 1.041×10⁻¹¹ in folds 1 and 3, compared with historical primal–dual gaps of 2.242×10⁻¹⁰ and 4.156×10⁻¹⁰. The numerical review found the observed coefficient differences compatible with the reported finite-residual certificate diagnostics. Their interpretation retains those residual conditions. Both current endpoints used fresh fits under the unified 10⁻¹⁰ solver tolerances, and the comparison uses newly fitted B and C throughout. The historical differences are recorded without a unique causal attribution.

### Historical interventions motivated the isolated strength test

E17's 100 renumberings reused 600 development questions and gave semantic-exact physical-team agreement of 6.135% for coordinate-champion SPO and 100% for Ridge and uniform-champion TSPO. E18's frozen policies later differed on 2778 of 2998 scorable questions, while TSPO's task-macro gain over SPO was +0.011749 pp with 377 corrected and 376 damaged generation events. E18.1 added a post-result marginal interval of [−0.666934,+0.690062] pp. These observations motivate the isolated β intervention and retain their original data and analysis roles.

## Discussion draft

The controlled endpoints showed that a direction-neutral tied-row term could alter strict decisions on other outer-fold questions in the present shared predictor. Uniform empirical champion weights were fixed, and the only objective change was the weight of the fully tied training rows. Centered scores changed on all 600 evaluation questions, and 150 questions had different winners with both endpoint gaps exceeding the declared numerical threshold. These observations extend the earlier champion-direction and inference-priority audits to a symmetric constraint-strength intervention.

The practical effect retained two components. Strict changes contributed +1.055556 pp, while questions with a numerical tie contributed +0.386464 pp under the main deployment policy. The all-question positive interval supports a conditional development signal for the prespecified policy. The raw exact diagnostic crossed zero, and the main interval included values below the practical 1 pp reference. The evidence therefore motivates an implementation-specific mechanism claim, with quality robustness across numerical tie definitions and independent data still to be established.

The theory distinguishes two sources of constraint. Limited repetitions can produce empirical ties despite a strict mean advantage. Symmetric penalties can also propagate across a shared predictor when the tied input is truly equivalent. The two constructions establish possible mechanisms under explicit assumptions. Their joint prevalence in the model pool remains an empirical question.

The endpoint control retains all 480 training rows, the original 1/N factor and the original penalty in each fold. This prevents loss normalization from silently changing effective regularization. Score changes and held-out action changes provide separate measurements, while the declared uniform deployment policy resolves predicted ties consistently. The 5 outer folds reuse overlapping training histories, so the bootstrap quantifies conditional development uncertainty.

Each fold used the same original penalty at both endpoints, although the inherited fold4 penalty was smaller than in folds0–3. Turning off the tied term increased coefficient penalties while decreasing the remaining proxy loss. This behavior is consistent with the stated interaction between symmetric constraints, shared representation and fixed regularization geometry. The finite-repeat construction and the three-input construction establish possible mechanisms under different assumptions. The real endpoint experiment identifies objective-level propagation within this pipeline, while latent conditional reward advantages remain incompletely observed from three generations.

The current decision is to retain the strict mechanism evidence and the full positive and negative sensitivity results within the completed P0/P1 study. A separate independent-confirmation design would need to freeze the same endpoints, deployment rule and practical effect target. Any further values of β, γ or λ, new representations, additional feedback or external evaluations require a later protocol. This work completed P0 and P1 only.

## 中文Results与Discussion要点

1. P0四次拟合已通过，合法回映射中心化分数最大误差2.723×10⁻¹⁰，两支均匀策略误差均为0。端点满足0≤0.0271468≤0.1091279。
2. P0的120题中，两支均严格唯一的有20题，其中12题严格换队，产生3次整数纠错、0次破坏，对全折点差贡献+0.8333pp。B数值分数并列的100题贡献其余+1.6667pp。主均匀策略总差+2.5pp，耦合期望纠错/破坏为18.6111/9.6111，分别保留整数严格事件及分数期望的口径。
3. 浮点精确argmax的B准确率87.2222%，预定数值并列均匀策略下为86.1111%，C均为88.6111%。数值并列与严格机制作用分别解释。
4. P1的600题中心化得分变化均超过10⁻⁷。203题两支均为严格唯一胜出，其中150题改变胜者，整数纠错38、破坏19、净19，对全600题贡献+1.055556pp。另397道至少一支数值分数并列的题贡献+0.386464pp，两部分合为全题差。
5. 预定10⁻⁸ argmax集合均匀策略下，B准确率85.057981%，C为86.500000%，C−B=+1.442019pp，题级配对95%区间[+0.190691,+2.722239]pp。GSM8K为+0.501323pp，区间[−0.644235,+1.640653]pp；BBH为+2.382716pp，区间[+0.143728,+4.671539]pp。
6. 原始浮点精确argmax均匀诊断下，B为85.5%、C为86.5%，差+1pp，区间[−0.333333,+2.333333]pp。该差异必须正面披露。主策略区间跨越1pp实用参考值，当前质量线索依赖已声明的推理数值定义。
7. 全题耦合期望纠错89.646825、破坏63.690476、净25.956349，属于固定耦合和均匀策略的分数期望。严格改变的38/19保持整数观测口径。三个生成seed主差均正，五折有4正1负，折历史相互重叠。
8. γ对应指定冠军方向，β对应已固定均匀冠军下的对称约束强度。150道严格胜者变化支持当前共享预测器中的约束强度作用，全部600题中心化变化也超出了共同平移。
9. 理论46/51属于三输入构造。真实数据的局部活动面及斜率仍需另外核验，当前两个端点没有被解释为真实阈值的估计。
10. 本轮实际完成10+2次真实拟合及10000次题级bootstrap，保留原600题开发身份。质量区间以冻结拟合模型为条件。当前结果支持继续设计独立确认方案，后续P2及参数网格另需授权和协议。

## 证据来源与作者核对

- P0来源为正式`P0_QUALIFICATION.json`与`FOLD0_POINT_RESULTS.json`，P1来源为正式`E19_COMPLETE_RESULTS.json`。
- 主结果与raw exact诊断同时保留，任何总体稳健性或独立质量主张均需后续证据。
- 独立数值复核已完成。旧E17缓存B与本轮新B在相同10⁻⁸部署规则下，fold1有8道题的最大集合support由13变14，整体600题B准确率仅变化−0.008242pp；fold3中心化得分最大差4.83×10⁻⁷，但策略support不变。旧新F1差分别为1.405×10⁻¹¹、1.041×10⁻¹¹，与旧primal–dual gap为2.242×10⁻¹⁰、4.156×10⁻¹⁰的有限残差诊断相容。新B/C均按统一10⁻¹⁰求解容差重新拟合，保留历史数值差异及其范围。
- 数值与访问复核来源为`NUMERICAL_AND_ACCESS_REVIEW.json`，SHA`4fc39686625858059bb66a212c5b71ab44404cc7ae6740f3933c71bddf8f9533`；历史恢复诊断为`HISTORICAL_NUMERICAL_RECOVERY.json`，SHA`314cc81f2f72a57ee15475a2509ef03a3fac56d41bde099e3d3106fe5032a3d8`。
- 完整数学证明及数值资格分别见`01_THEORY_AND_PREDICTIONS.md`和`THEORY_CHECKS.json`。
- 本稿的构造结论依据已读导师PDF，历史数值沿原E17/E18/E18.1版本引用。原文件均保持既有字节。
