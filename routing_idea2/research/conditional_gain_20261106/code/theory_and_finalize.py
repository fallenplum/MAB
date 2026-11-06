from common import *
from audit_and_freeze import raw_predict
import pickle

THEORY=r'''# 05 有限样本条件增益与资源分配：命题、证明和实证对应

## 研究问题
区分事后选队、跨生成稳定价值、输入表示可实现价值；在问题与重复两级采样下估计冻结策略增益，并说明安全切换何时成立。以下是条件明确的标准概率论推导及本问题的应用，不宣称这些恒等式本身构成算法新颖性。

## 实验设置与符号
有限动作集 $\mathcal S$ 为6选3的20组合。每次执行返回真实冻结聚合器奖励 $R(x,S,\xi)\in[0,1]$。令 $q_S(x)=E_\xi R(x,S,\xi)$，$Z=\phi(X)$。策略/默认组合在评价问题与生成噪声揭示前冻结；默认组合可依赖训练获得的task/subtask规则。训练数据一律条件化为固定；不把策略重拟合的不确定性悄悄并入下述采样方差。

## 命题1：Oracle层级及必要的信息限制
定义 $V_{post}=E_{X,\xi}\max_S R$、$V_{stable}=E_X\max_S q_S(X)$、$V_\phi=E_Z\max_S E[q_S(X)\mid Z]$。
若 $\pi$ 仅使用 $Z$，其随机化与评价噪声条件独立，则
$$V_{post}\ge V_{stable}\ge V_\phi\ge V(\pi).$$
**证明。** 固定$x$时，$\max_S E_\xi R\le E_\xi\max_S R$；积分得到第一不等式。给定$Z$，有$\max_S E[q_S(X)\mid Z]\le E[\max_S q_S(X)\mid Z]$，再用塔式法则得到第二项。策略给定$Z$的动作概率为$p_S(Z)$，故$E[R(X,\pi(Z),\xi)\mid Z]=\sum_S p_S(Z)E[q_S(X)\mid Z]\le\max_S E[q_S(X)\mid Z]$，积分得最后一项。有限动作保证最大值可测、可用固定索引处理同分。证毕。

**限制和反例。** 若$Z$为常量，而$X$指示哪个动作必对，能看完整$X$的策略可得1，$V_\phi$却只有1/2。因此最后一项不能不加条件地覆盖使用比$Z$更多信息的策略，也不能覆盖先看候选回答再选队。若噪声受动作选择所揭示的信息影响，须重定义条件分布。

**有限三seed不识别总体界。** 每题$\hat q_S=L^{-1}\sum_lR_{Sl}$，$E_\xi\max_S\hat q_S\ge\max_S q_S$，所以三seed均值再取最大通常乐观。观测的$\hat V_{post}-\hat V_{stable}$只是同一小样本的代数差，不是总体噪声比例。对每个$x$，有限动作和独立重复下联合Hoeffding给$\max_S|\hat q_S-q_S|\le\sqrt{\log(40/\delta)/(2L)}$，概率至少$1-\delta$；$L=3$时通常太宽，裁到[0,1]也不能得到紧界。800个不同输入每个仅三次结果，无法无模型假设地识别$E[R\mid Z,S]$。交叉拟合/未见题策略价值可提供实际可达到的证据，但不能把拟合类最优值直接当$V_\phi$。

## 命题2：二阶组合基底恰为秩15
令$C$为20个三元组对15个二元组的包含指示矩阵。三元组上$\sum_{i<j}z_iz_j=3$，且$\sum_{j\ne i}z_iz_j=2z_i$，所以常数和一阶列均在$C$列空间内，原22列基底与$C$同列空间。
**证明满列秩。** $C^TC$对角为4，两二元组共一个元素时为1、不相交时为0。令$A$为完全图$K_6$的顶点—边关联矩阵，则$A^TA=2I+H$，$H$为线图邻接矩阵；$AA^T=4I+J$，特征值为10（1重）和4（5重）。故$H$谱为8（1重）、2（5重）、−2（9重），$C^TC=4I+H$谱为12、6、2，均正。因此rank(C)=15，rank(B)=15。证毕。
**经验检验。** 对每题均值向量$\hat q$计算$\hat q(I-P_B)$。它混合了真实不可表达成分和有限生成噪声，不能直接称为真实结构偏差。二阶Ridge还使用非正交22列上的系数正则，改变组合表示同时改变谱收缩；不能将直接20回归的差异全归到秩。

## 命题3：普通线性相对标签的排序不变性
所有动作使用同一线性平滑算子$H_\lambda$和同一训练权重。标签矩阵$Y$改成$Y-b\mathbf1^T$（$b$可随训练题的默认动作变化），有
$$H_\lambda(Y-b\mathbf1^T)=H_\lambda Y-(H_\lambda b)\mathbf1^T.$$
**证明。** 由线性分配律直接成立。对任意两个动作相减，共同项消失。故动作得分的argmax集合不变；若同分规则一致，选队不变。预测时再减去默认动作预测也不改变集合。证毕。
因此“线性标签相对化”本身不是改进来源。阈值、默认动作条件交互、不同正则或非线性模型可以改变决策，但必须分别归因。交互模型容量增加不能包装成仅标签变化的收益。

## 命题4：冻结策略配对均值的两级方差
令$D_{il}=R(X_i,\pi(X_i),\xi_{il})-R(X_i,\pi_0(X_i),\xi_{il})\in[-1,1]$，同题两个策略使用同一生成重复配对。设不同$X_i$独立同分布，给定$X_i$的$L$个重复条件独立同分布，题间噪声独立。令$m(X)=E[D\mid X]$、$s^2(X)=Var(D\mid X)$、$g=ED$、$v=Var_Xm(X)$、$w=E_Xs^2(X)$。
$$\hat g={1\over n}\sum_i\bar D_i,\qquad Var(\hat g)={v\over n}+{w\over nL}.$$
**证明。** 条件于$X_i$，$E\bar D_i=m(X_i)$、$Var(\bar D_i\mid X_i)=s^2(X_i)/L$。全方差公式给$Var(\bar D_i)=v+w/L$，独立题的均值方差再除以$n$。证毕。
若重复条件相关，则增加$L^{-2}\sum_{l\ne l'}Cov(D_{il},D_{il'}\mid X_i)$的期望项；固定的伪随机seed或混合硬件不能自动保证其为0。若task/subtask分层配额固定，则方差是$\sum_h p_h^2(v_h/n_h+w_h/(n_hL))$，而非直接套总体iid式。跨折OOF模型共享训练数据还会诱导题间依赖，故OOF拟合后选择的统计量不直接享有此冻结策略保证。

**估计。** $\hat w=n^{-1}\sum_i s_i^2$，$\hat v=sampleVar(\bar D_i)-\hat w/L$。未截断估计可负，反映有限样本不确定性；同时报告原值及仅规划时截到0的值，不能把截断说成无偏。各策略必须在这些评价题外冻结。仅3重复难以验证条件独立；当前估计均作开发诊断。

## 命题5：有界奖励集中界与所需题数
在命题4假设下，题簇均值$U_i=\bar D_i\in[-1,1]$独立，方差$\sigma_L^2=v+w/L$。有
$$P(|\hat g-g|\ge\epsilon)\le2\exp\{-n\epsilon^2/(2\sigma_L^2+4\epsilon/3)\}.$$
**证明。** 令$Z_i=U_i-EU_i$，$|Z_i|\le2$。对$0<t<3/2$，展开MGF，使用$E Z_i=0$、$E|Z_i|^k\le2^{k-2}EZ_i^2$及$k!\ge2\,3^{k-2}$，得
$$\log E e^{tZ_i}\le{t^2 EZ_i^2\over2(1-2t/3)}.$$
独立性将MGF相乘。Chernoff取$t=\epsilon/(\sigma_L^2+2\epsilon/3)$，得单侧$\exp[-n\epsilon^2/(2\sigma_L^2+4\epsilon/3)]$；负侧同理并合并。证毕。
于是充分的题数为$n\ge(2\sigma_L^2+4\epsilon/3)\log(2/\delta)/\epsilon^2$。未知$v,w$不能把点估计直接代入并声称严格置信保证；保守Hoeffding为$n\ge2\log(2/\delta)/\epsilon^2$。若已知真实正向增益至少$\gamma$，取精度$\epsilon<\gamma/2$可使同半径下界为正；没有正向效应假设时，不能通过增加预算保证发现正增益。
可选经验Bernstein界：对$n\ge2$，以题簇样本方差$S_U^2$，应用区间缩放与双侧union bound，半径为$\sqrt{2 S_U^2\log(4/\delta)/n}+14\log(4/\delta)/(3(n-1))$。这是[Maurer与Pontil，2009，定理4](https://arxiv.org/abs/0907.3740)的缩放形式，不是本研究新定理；自适应从多个策略中挑赢家须控制模型类/多重比较，不能沿用固定策略界。

## 命题6：简化预算下的生成重复最优分配
设新增一道独立题固定成本$a\ge0$，每个生成重复边际成本$b>0$，预算$B$，连续放松且所有预算用完，$n=B/(a+bL)$。则
$$Var(\hat g)={(a+bL)(v+w/L)\over B}={av+bw+bvL+aw/L\over B}.$$
当$a,w,v>0$时，导数$(bv-aw/L^2)/B$，二阶导数$2aw/(BL^3)>0$，唯一连续最优
$$L^*=\sqrt{aw/(bv)}.$$
**整数与边界。** 实验至少$L\ge1$，且$n=\lfloor B/(a+bL)\rfloor\ge n_{min}$；连续情形先裁到可行区间。由于floor出现台阶，实际应枚举所有可行整数$L$，不能只保证相邻两个整数最优。$a=0$或$w=0$且$v>0$优先最小$L$；$v=0,a w>0$时随$L$增加方差下降，连续最优在上边界，但不能以只有一道题替代问题泛化；$v=w=0$时方差均0。$a=b=0$不构成有意义预算约束。
实际gold、成员、聚合、token、规划和GPU资源为多维约束，可有不同单位和重用，不能直接折成该式。公共基准gold已经下载，不等于真实新增标注采购收费；当前未测得同单位$a$，因此仅报告$a/b$敏感性，绝不伪造唯一最优重复数。

## 命题7：条件增益安全切换
对固定输入$x$，若事件$E$上所有候选都满足$|\hat\Delta(x,S)-\Delta(x,S)|\le u(x,S)$，则任何满足$\hat\Delta-u>0$的动作都满足$\Delta>0$。
**证明。** 在$E$上，$\Delta\ge\hat\Delta-u>0$。默认动作的真实增益为0，保持默认不会引入负增益；在该事件上按任何合格动作切换均非劣。证毕。
若$P(E)\ge1-\delta$，结论以同一概率成立；必须保证同时覆盖全部候选，因为动作是看过预测后挑出的。单动作点态CI不自动同时有效；有限题/动作可union bound，所有未知$x$的保证则需要更强统一复杂度或分布假设。若事件外奖励差最低−1，则仅有该保证时总体最坏损失至多$\delta$，不能直接说无条件严格正收益。模型失配、分布漂移、同题相关重复和校准集复用均可能使$E$失效。
本轮proper480/cal120给出的阈值覆盖目标是带有限生成噪声的三seed平均标签，不是真实$\Delta$。预测方差、工作模型后验或一个经验残差分位数均不等于严格的同时误差上界。故本轮校准变体只作实证检查，不提供population安全证明。

## 数值结果、统计结论与可证伪预测
真实数据的$v,w$、$a/b$敏感性见CPU_COMPLETE_RESULTS.json；方差恒等式的独立已知分布Monte Carlo检验见THEORY_NUMERICAL_CHECKS.json，与真实LLM实验分开记录。
预测1：当$v>0$时固定$n$增加$L$存在$v/n$方差地板；新增独立题同时降低两项。预测2：普通线性相对标签不改变argmax，已在真实冻结矩阵上验证。预测3：严格校准切换阈值可能导致零切换；零切换不证明逐题路由有价值。预测4：增加参数维度可降低训练损失但不保证未见题服务质量；结构/特征消融专门检验这一点。

## 失败原因与局限
现有三seed、200题开发验证不支持精确的每题条件期望或紧致理论功效估计。理论条件不是已经由数据证明的事实。Oracle差值、Ridge系数、方差保留率均不具有自动因果解释；全部统计是开发性而非sealed确认性评价。

## 下一步
先验证可重复开发增益，再用冻结策略一次性评价fresh400；仅在fresh正向CI及至少2/3seed同向时进入有限反馈。采集机制的新颖性还需和既有决策价值/信息获取前史区分。[RankNet原始论文](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)仅作为排序基线的文献来源，不作为本研究效果的证据。
'''

def main():
    r=read(PUBLIC/'CPU_COMPLETE_RESULTS.json');rng=np.random.default_rng(20261106);rows=[]
    # Software-only known distribution: D in{-1,1}, E[D|X]=.4X, v=.16,w=.84.
    for n in [50,200]:
        for L in [1,3,10]:
            x=rng.choice([-1,1],size=(2000,n));p=.5+.2*x
            d=2*(rng.random((2000,n,L))<p[:,:,None])-1
            empirical=float(d.mean((1,2)).var(ddof=1));expected=.16/n+.84/(n*L)
            assert abs(empirical/expected-1)<.15
            rows.append({'n':n,'L':L,'known_v':.16,'known_w':.84,'theoretical_variance':expected,'monte_carlo_variance':empirical,'relative_error':empirical/expected-1})
    put(PUBLIC/'THEORY_NUMERICAL_CHECKS.json',{'scope':'SOFTWARE_ONLY_KNOWN_DISTRIBUTION_NOT_LLM_RESULT','replicates':2000,'seed':20261106,'rows':rows,'new_llm_calls':0})
    (PUBLIC/'05_theory_and_proofs.md').write_text(THEORY,encoding='utf8')
    freeze=read(PUBLIC/'FRESH_POLICY_FREEZE_INPUT_ONLY.json');main=r['conditional_switches']['conditional_main_thresholded']
    stage5='# 06 fresh400独立开发留出评价\n\n## 研究问题\n改进方法能否在未见题上相对子任务固定组合稳定获益。\n\n## 实验设置与对照\n计划保留原三生成seed与九类对照；原sealed200封存。六种组合路由的逐题选队已仅依据输入冻结，私有预测SHA及并集见FRESH_POLICY_FREEZE_INPUT_ONLY.json。\n\n## 数值结果\n**NOT_EXECUTED，不能填入准确率或配对CI。** 六组合策略需要'+str(freeze['unique_question_team_calls_per_seed'])+'次聚合/seed，三seed共'+str(freeze['unique_question_team_calls_three_seeds'])+'次；这不含最强单模型、三采样投票/融合、聚合器独答，不是完整九对照执行成本。未重启原7200成员或24000聚合队列。已有5330成员成功输出的暂停备份及费用完整保留。\n\n## 统计结论\n训练OOF主候选87.67%，开发validation86.17%对原subtask87%，95%配对CI[-3.33,+1.83]pp，只有1/3生成seed正向；尚未满足用户要求的“明确可重复开发增益，再冻结进行实验5”前提。选队冻结是准备完成，绝不是独立评价完成。\n\n## 失败原因\n开发收益没有迁移，BBH净纠错为负，阈值未抑制过度切换。此处失败指开发前提，不能冒称fresh假设被否定。\n\n## 下一步\n当前保持fresh评分封存，不按validation挑另一个看起来更好的模型。若继续一次性fresh确认，应先明确修订开发推进条件，锁定全部九类对照的精确job、缓存复用和完整费用，再执行三seed评价。此处不将仅六路由并集冒充完整执行预算。\n'
    (PUBLIC/'06_fresh400_evaluation.md').write_text(stage5,encoding='utf8')
    fb='# 07 有限反馈采集机制执行状态\n\n## 研究问题\n在已确认有益条件切换后，以更低完整反馈成本学会切换。\n\n## 实验设置与对照\n用户指定Random、Boundary、ParameterIG、V1决策定向IG/Cost、KG-informed/Cost、条件切换采集；统一selector/features/update/default/service与gold复用。\n\n## 数值结果\n**NOT_EXECUTED_PREREQUISITE_NOT_MET**。未执行60/150/300/600预算回放，未生成CES或C2MAB-V结果，未将适配策略冒称原生复现。\n\n## 统计结论\nfresh400主要条件尚未评价，不能进入该阶段。\n\n## 失败原因\n尚无稳定未见题条件切换收益；更快学习一个未证实有用的切换规则不足以支持实际价值。\n\n## 下一步\n达到前阶段条件后，固定完整执行资源口径与动作复用图，再仅改变采集规则；现有训练、聚合和评分不能在回放里被当作免费真实支出。\n'
    (PUBLIC/'07_feedback_acquisition.md').write_text(fb,encoding='utf8')
    summary='# MAB 条件增益研究：本轮阶段结果\n\n## 研究问题\n原二阶预测器为何未学到可泛化逐题优势，默认组合与相对增益能否改善。\n\n## 实验设置与对照\n从main建立codex/conditional-gain-20261106。train600内部五折、按task/BBH子任务分层，同题三seed同折；四个正则/模型，折内PCA/标准化。44个配置，模型池、K、冻结聚合器和评分矩阵均保持原身份。val200仅开发诊断。\n\n## 数值结果\n|阶段|状态|关键结果|\n|---|---|---|\n|实验1 稳定性|完成|val两seed→留出seed88%对87%，原配对区间跨0|\n|实验2 结构|完成|训练OOF选中二阶86.44%；对应val85.67%；同PCA16直接20/低秩/排序val86.33/86.50/86.50%|\n|实验2 特征|完成|direct20完整向量+元数据OOF86.78%，val84.33%；增维训练更好不代表决策更好|\n|实验3 条件增益|完成开发训练与诊断|主MLP阈值.025，OOF87.67%，val86.17%；94%题切换，25纠正/30破坏题×seed实例|\n|实验4 理论|完成|Oracle层级、基底秩、线性delta等价性、两级方差、集中界、资源最优与安全切换证明|\n|实验5 fresh400|未执行：开发前提未满足|六组合策略输入预测已冻结，1807聚合/seed；完整九对照预算仍须单列|\n|实验6 有限反馈|未执行：前提未满足|没有生成或填造任何采集胜出结果|\n\n## 统计结论\n主候选相对子任务固定−0.83pp，95%题簇CI[-3.33,+1.83]pp；逐seed差−2.5/+1.0/−1.0pp，仅1/3为正。所有配置均公开，未按val换赢家。OOF是在同一训练数据上用于选参的值，有模型/阈值选择乐观偏差。\n\n## 错误来源分析\n二阶仅有15组合方向，但直接20/排序并未带来稳定服务提升，因此不能归因只有二阶欠表达。PCA增维和完整向量降低训练误差也未迁移；题数600、BBH每子任务11–12、val每子任务3–4题，与高维/交互/MLP复杂度不匹配。线性delta在统一算子下有排序等价性，改变标签本身无效。条件MLP过度切换，GSM净+2、BBH净−7个题×seed实例；无法证明同一BBH子任务内稳定可预测有益切换。\n\n## 口径修正与复核\n初次CPU报告把delta数值误差和绝对奖励作比较；已用同题同seed配对target零重拟合回放修正，选队与准确率变化0。独立Top3只有冻结选队，不能用哨兵分数解释其MSE/排序，相关值已置null。原初版本保留在私有目录，METRIC_AND_REPLAY_AUDIT公开证据。普通线性与直接20在匹配表示/正则下OOF选择相同，数值等价性检验通过。完整旧sourceview成本链复用原已核验receipt；本轮重验矩阵/receipt/冻结身份哈希，不冒称重新验证全部历史推理。\n\n## 成本与局限\n本轮LLM调用0、答案评分0、新增GPU推理0。首轮CPU搜索与最终拟合用时见JSON；零fit审计、数值模拟和回放另外记录。输入编码已缓存，选择器基准不含新题MiniLM编码耗时；历史model.generate墙钟之和不是GPU kernel或日历时间。风险还包括三seed有限、seed42部分混合硬件、探索性多重比较和CV选择偏差。事实与数值置信度高，具体瓶颈因果归因与未见题收益置信度低。\n\n## 下一步\n当前证据不支持扩大有限反馈。保持训练OOF选择及负结果，先明确是否修订开发推进条件再做一次性fresh确认；不能换成val赢家或把全部20组合生成当学习成功。所有新增代码/脱敏统计/阶段报告在本研究分支，逐题预测、模型pickle、npz和原题/gold只放私有目录。\n'
    (PUBLIC/'EXPERIMENT_SUMMARY.md').write_text(summary,encoding='utf8')
    put(PUBLIC/'STAGE_STATUS.json',{'experiment1':'COMPLETE','experiment2':'COMPLETE_DEVELOPMENT_ONLY','experiment3':'COMPLETE_DEVELOPMENT_ONLY','experiment4':'COMPLETE_WITH_ASSUMPTIONS','experiment5':'NOT_EXECUTED_DEVELOPMENT_REPEATABILITY_PRECONDITION_NOT_MET','experiment6':'NOT_EXECUTED_FRESH_GATE_NOT_MET','research_complete':False,'fresh_scores_opened':False,'sealed_scores_opened':False,'new_llm_calls':0,'new_gpu_inference':0})
    # A precise source/artifact manifest; no recursive private output upload.
    files=[{'file':p.relative_to(PUBLIC).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(PUBLIC.rglob('*')) if p.is_file() and p.name!='DELIVERY_MANIFEST.json' and '__pycache__' not in p.parts]
    put(PUBLIC/'DELIVERY_MANIFEST.json',{'files':files,'private_arrays_and_fitted_models_excluded':True,'not_a_final_research_success':True})
    print(json.dumps({'reports':8,'public_files':len(files)+1,'fresh_status':'NOT_EVALUATED','stage6':'NOT_RUN','actual_new_llm_calls':0}),flush=True)
if __name__=='__main__':main()
