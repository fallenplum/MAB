"""Render E8 reports directly from sealed metrics and audited costs."""
from __future__ import annotations
import csv
import json
from pathlib import Path

BASE=Path(r'E:\语言模型组合路由\research\e8_decision_focused_combination_learning')


def read(name):return json.loads((BASE/'public'/name).read_text(encoding='utf-8'))


def write(name,value):(BASE/'public'/name).write_text(value.strip()+'\n',encoding='utf-8')


def pp(value):return f'{100*value:+.3f}'


def interval(values):return '['+', '.join(pp(v) for v in values)+']'


def main():
    results=read('E8_COMPLETE_RESULTS.json');stats=read('E8_PRIMARY_STATISTICS.json')
    noise=read('E8_NOISE_DIAGNOSTICS.json');shrink=read('E8_SHRINKAGE_DIAGNOSTIC.json')
    costs=read('training/TRAINING_COST_SUMMARY.json');audits=read('E8_TRAINING_AND_RESTORATION_AUDIT.json')
    methods=results['methods'];main_names=['M0_original','M0_tuned','M1','M2','M3','M4']
    configs=[]
    for fold in range(5):
        m3=audits['training'][f'fold{fold}__M3']['selected_parameters']
        m4=audits['training'][f'fold{fold}__M4']['selected_parameters']
        m2=audits['training'][f'fold{fold}__M2']['selected_parameters']
        configs.append(f"| {fold} | {m3['lambda_multiplier']:g} | {m3['alpha']:g} | {m4['lambda_multiplier']:g} | {m2['lambda_multiplier']:g} | {m2['target_temperature']:g} |")
    objectives_math=r"""
## 完整学习目标

记r为同题三个生成奖励的20维均值，q=phi W，T为合法OOF预测的Top3组合。每个目标的正则项均包含受罚常数坐标。

\[
L_0=\frac1{20n}\sum_x\|q(x)-r(x)\|_2^2+\lambda\|W\|_F^2.
\]

\[
L_1=\frac1{190n}\sum_x\sum_{i<j}|r_i-r_j|
\log\left(1+\exp[-\operatorname{sign}(r_i-r_j)(q_i-q_j)]\right)
+\lambda\|W\|_F^2.
\]

真实差为0时，M1的该项严格为0。

\[
t=\operatorname{softmax}(r/\tau_y),\qquad
p=\operatorname{softmax}(q/\tau_p),\qquad
L_2=-\frac1n\sum_x\sum_jt_j\log p_j+\lambda\|W\|_F^2.
\]

\[
\widetilde w_{ij}=1+\alpha\mathbf1\{i\in T\ \lor\ j\in T\},\qquad
w_{ij}=\frac{\widetilde w_{ij}}{190^{-1}\sum_{a<b}\widetilde w_{ab}},
\]

\[
L_3=\frac1{400n}\sum_x\sum_{i<j}w_{ij}
\left[(q_i-q_j)-(r_i-r_j)\right]^2+\lambda\|W\|_F^2.
\]

\[
L_4=\frac1n\sum_x\left[\max_j(2q_j-r_j)-2q_{j^\star}+r_{j^\star}\right]
+\lambda\|W\|_F^2,\qquad j^\star=\arg\max_jr_j.
\]

精确QP以每题t_x作epigraph变量，约束t_x≥2q_j−r_j，并优化相同的线性项加二次正则。均匀真实tie变体将q_{j*}替换为真实并列最优集合上的平均预测分数，保持r_{j*}=max r。原始目标与适配变体分别记录。
"""
    write('03_TRAINING_OBJECTIVES_AND_IMPLEMENTATION.md',f"""
# E8 训练目标与实现

## 研究问题

在相同MiniLM问题输入、PCA16、标准化和17×20线性头下，只改变监督目标，检查竞争相对价值与决策导向训练能否提高后续选队。常数项与其他参数一并受到二范数正则。

## 实验设置

沿用train600的原五个外折，每折480题历史、120题评价。三折内层划分按原始问题完成。所有预处理均在当前合法拟合区学习。三次生成先取同题20组合均值。训练进程只读取history480和evaluation_inputs_only120两类包。

| 组 | 数学目标与归一化 | 候选 |
|---|---|---|
| M0_original | 按题和20输出平均的绝对MSE | 原raw alpha100固定 |
| M0_tuned | 相同M0目标 | 原有效正则乘0.1、1、10 |
| M1 | 真实非零gap的绝对差加权RankNet，固定190对分母，再题平均 | 三个正则 |
| M2 | Top-one ListNet交叉熵，题平均 | 目标温度0.1、0.3、1及三个正则，预测温度1 |
| M3 | OOF Top3竞争加权平方gap，除400，再题平均 | alpha0、2、5及三个正则 |
| M4 | 精确SPO+ epigraph凸QP，真实最优按字典序 | 三个正则 |

{objectives_math}

统一正则为lambda=100×multiplier/(3×nfit×20)。nfit为当前实际拟合问题数。M3每题190个权重的平均值为1，alpha0恢复中心化Ridge排序。零真实gap在M1中贡献0。

### 原Ridge正则搜索的结构性退化

当前PCA正交投影后，历史标准化令每个主成分的训练方差为1，常数列与其正交。在统一题权重下，Phi转置乘Phi约等于nI。M0的系数因此为Phi转置乘Y再除以n+raw_alpha/3，改变正则只产生全头共同的正比例缩放。

五折训练Gram与480I的最大误差7.30e-12，三个候选正则的缩放核验误差小于1e-12，600个评价输入的选队变化为0。M0_tuned在内层并列后选择倍率10，其名义三档搜索没有增加当前表示下的路由自由度。该诊断记录在qualification/PCA_RIDGE_GEOMETRY_DIAGNOSTIC.json。

M3的竞争Top3来自合法OOF原Ridge，表示20个组合中得分最高的三个三成员组合，用于训练权重。服务阶段仍在全部20组合中选择一个三成员组合。内层训练区另行交叉拟合，其验证题标签不参与竞争权重构造。正式代码保留所有区域的输入哈希、标签哈希、OOF行归属与配置。

## 模型选择与敏感性

选择标准为历史内层OOF真实聚合准确率，GSM8K与BBH各50%。相同分数优先强正则，再优先较小M3 alpha或较大ListNet目标温度。外折选择使用原精确numpy argmax，完全并列选择字典序第一个。

原固定Ridge是两项主比较的基线，调参Ridge为同范围调参对照。Top5复用Top3选出的参数，均匀真实最优并列变体复用主SPO+的正则。六个置乱控制复用各自主参数，在语义任务组内置乱整题20×3奖励块，作为一次固定配置内容诊断。置乱控制没有重新选择超参数。

| 外折 | M3正则乘数 | M3 alpha | M4正则乘数 | M2正则乘数 | M2目标温度 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(configs)}

## 数值校准与评价边界

每个目标使用同一历史OOF拟合程序，得到一个非负全局斜率和共享截距。该映射用于把原分数转换到奖励数值单位，数值误差同时报告原始与校准版本。路由动作始终来自原分数。校准系数、权重和逐题记录留在私有目录。

RankNet、ListNet和SPO+原分数具有各自尺度。其原始绝对MSE只作诊断。正文比较相对价值时采用共同校准程序后的中心化MSE和gap误差，并同时检查严格排序和最终准确率。

## 求解与失败记录

RankNet与ListNet使用稳定logaddexp/softmax和解析梯度的L-BFGS-B，最多2000次迭代，ftol1e-12、gtol1e-7。M3通过340维正规方程Cholesky求解。SPO+主实现为精确epigraph QP，原OSQP上限200000次、eps1e-8。

原运行在均匀真实并列敏感性中出现solved inaccurate，保留49个已封存模型和失败日志。外折评价前的执行修订用同P、q、A、b的Clarabel0.11.1、tol1e-9补齐21个未封存模型。原源码、主模型与原配置有完整哈希链。数学目标与超参数沿原协议执行。

## 数值结果与结论

五折×14变体共70模型封存。成功模型日志中的610次目标拟合均达到各自声明的停止规则，失败尝试另存，额外OOF闭式Ridge拟合费用计入总CPU。五个主M4模型跨求解器的600个输入选择一致。均匀真实并列敏感性存在近零分数间隔，解释范围见05报告。

优势是容量、标签、输入与服务决策规则统一，能够定位目标差别。风险是离散噪声奖励、内层选择变化、校准尺度和近并列数值效应。实际选队证据见04报告，后续条件见06报告。
""")
    main_rows=[]
    for name in main_names:
        v=methods[name]['all']
        main_rows.append(f"| {name} | {100*v['accuracy']:.3f}% | {100*methods[name]['gsm8k']['accuracy']:.3f}% | {100*methods[name]['bbh']['accuracy']:.3f}% | {v['corrected_question_generation_events']} | {v['destroyed_question_generation_events']} | {v['net_corrections_question_generation_events']} |")
    primary_rows=[]
    for row in stats['primary_comparisons']:
        primary_rows.append(f"| {row['method']} − M0_original | {pp(row['gain'])} | {interval(row['paired95'])} | {row['p_Holm']:.4f} | {row['scientific_positive_gate']} |")
    seeds=[]
    base=methods['M0_original']['all']['per_generation_rewards']
    for name in main_names:
        values=methods[name]['all']['per_generation_rewards']
        differences=[v-b for v,b in zip(values,base)]
        seeds.append('| '+name+' | '+' | '.join(f'{100*v:.3f}%' for v in values)+' | '+', '.join(pp(v) for v in differences)+' |')
    task_rows=[]
    for sub in sorted(methods['M0_original']['bbh_subtasks']):
        values=[methods[m]['bbh_subtasks'][sub] for m in ['M0_original','M3','M4']]
        task_rows.append(f"| {sub} | {values[0]['questions']} | {100*values[0]['accuracy']:.2f}% | {100*values[1]['accuracy']:.2f}% | {100*values[2]['accuracy']:.2f}% |")
    write('04_MAIN_COMPARISONS.md',f"""
# E8 主比较

## 研究问题与设置

检查M3竞争加权Ridge和M4精确SPO+是否优于原直接20输出Ridge。600个独立原始问题在五外折各评价一次，三个生成重复属于同一问题簇。所有70模型和历史校准先封存，再由独立评价器读取评分。

## 最终服务准确率与纠错

外折历史区选出的全局固定组合准确率为{100*methods['M0_original']['all']['default_accuracy']:.3f}%。下表纠错、破坏和净收益是相对该逐折固定组合的题×生成事件数，共1800次已记录结果，统计独立单位仍为600题。

| 目标 | 总准确率 | GSM8K | BBH | 纠错事件 | 破坏事件 | 净纠错事件 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_rows)}

## 两项预定主检验

| 比较 | 增益百分点 | 配对95%区间，百分点 | Holm p | 正收益关口 |
|---|---:|---|---:|---|
{chr(10).join(primary_rows)}

采用10000次原始问题配对bootstrap，按外折×任务分层，三生成与所有方法共同抽样。p值来自中心化双侧bootstrap检验，再对两项主比较统一Holm。两个区间均跨零，本轮证据等级为探索性。

调参M0准确率与原固定M0相同。两项主方法相对该同范围调参对照的区间与上述相同。数值分数和正则配置仍分别保存，没有将相同准确率解释为模型完全相同。

## 生成重复方向

| 目标 | seed42 | seed137 | seed2024 | 对M0逐seed增益百分点 |
|---|---:|---:|---:|---|
{chr(10).join(seeds)}

M4三个生成重复及GSM8K、BBH两个任务的点增益均为正。问题级配对区间仍较宽，当前不足以排除开发样本变化。M3的seed137方向为负，GSM8K点增益为0。

## BBH子任务支持量

| 子任务 | 原始问题数 | M0 | M3 | M4 |
|---|---:|---:|---:|---:|
{chr(10).join(task_rows)}

子任务分解属于支持量诊断，各任务仅约11—12题，完整逐折结果与子任务宏平均保存在E8_COMPLETE_RESULTS.json。当前未从最有利子任务选择方法版本。

## 统计结论、失败原因与下一步

两项主比较均未通过预定科学关口。M4保留一致正向点值，当前仍属于探索性。相对固定组合的收益区间也跨零。中心化误差、竞争误差和置乱控制见05报告，当前不扩展反馈采集。

优势是配对评价复用同一真实聚合评分。风险是训练历史重叠、旧开发资产重复使用以及有限独立问题数量。本表区间用于当前条件性开发交叉验证，推广证据尚待独立确认。
""")
    metric_rows=[]
    for name in main_names:
        v=methods[name]['all']
        metric_rows.append(f"| {name} | {v['raw_absolute_MSE']:.6f} | {v['calibrated_absolute_MSE']:.6f} | {v['calibrated_centered_MSE']:.6f} | {v['calibrated_competitor_gap_MAE']:.6f} | {100*v['strict_pair_rank_accuracy']:.2f}% | {v['posthoc_three_repeat_selection_loss']:.6f} |")
    shrink_rows=[]
    decomposition_rows=[]
    for name in ['M0_original','M3','M4','ZERO_GAP_NUMERICAL_COMPARATOR']:
        v=shrink['metrics'][name]['all']
        shrink_rows.append(f"| {name} | {v['all_competitor']['MAE']:.6f} | {v['strict_competitor']['MAE']:.6f} | {v['tied_competitor']['MAE']:.6f} |")
        d=shrink['additive_error_decomposition'][name]
        decomposition_rows.append(f"| {name} | {d['strict_contribution']:.6f} | {d['tied_contribution']:.6f} | {d['total']:.6f} |")
    negative_rows=[]
    for name in main_names:
        real=methods[name]['all']['accuracy'];wrong=methods['SHUFFLE_'+name]['all']['accuracy']
        contrast=stats['secondary'][name].get('versus_matching_shuffle')
        negative_rows.append(f"| {name} | {100*real:.3f}% | {100*wrong:.3f}% | {pp(real-wrong)} | {interval(contrast['paired95'])} |")
    noise_rows=[]
    for label in ['all','gsm8k','bbh']:
        for row in noise[label]['cross_generation_pair_direction']:
            if row['gap_group']=='all_nonzero':
                noise_rows.append(f"| {label} | {row['held_generation']} | {row['prior_strict_pair_events']} | {100*row['same_per_prior_strict']:.2f}% | {100*row['same_among_held_non_ties']:.2f}% | {100*row['held_tie_fraction']:.2f}% |")
    write('05_NOISE_AND_RANKING_DIAGNOSTICS.md',f"""
# E8 噪声、排序与相对误差

## 研究问题

区分可预测的竞争差异、真实奖励并列与生成波动，检查训练误差改善是否对应有意义的选队能力。

## 数值误差与排序

原分数按各目标定义保留。校准误差使用同一历史OOF非负仿射程序，校准过程与路由动作分离。

校准是历史内层训练规模到完整480题拟合的迁移。正则只改变缩放的M0，在320题与480题时缩放比不同，因此相同选队仍可能得到不同校准误差。数值表的结论限于该共同校准程序，排序和准确率独立报告。

| 目标 | 原始绝对MSE | 校准绝对MSE | 校准中心化MSE | 竞争gap MAE | 严格pair排序准确率 | 三重复事后选队损失 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

竞争集合使用共同的外折M0预测Top3，任一端属于该集合的54对。排序以真实平均差非零的组合对计算，预测数值并列计半分，按有支持的原题平均，支持量完整保留。各对没有作为独立样本。

M3的中心化MSE下降0.000214，探索性95%区间为[-0.000402,-0.000022]。服务准确率主检验未通过。M4中心化MSE点值略高于M0，竞争MAE却降低，严格排序准确率也降低，需要进一步区分误差来源。

## 事后零差值诊断

这一诊断在主评价结束后提出，已明确标记POSTHOC_DESCRIPTIVE_DIAGNOSTIC。零差值向量仅用于数值误差比较，不作为新增路由器，不进入主检验或模型选择。

| 数值预测 | 全部竞争对MAE | 每题非零竞争对MAE，294题平均 | 真实竞争差为0时的MAE |
|---|---:|---:|---:|
{chr(10).join(shrink_rows)}

32400个竞争pair-events中，25541个真实三重复平均差为0。M3、M4的整体MAE下降主要伴随零差值误差缩小。非零指标先在每题仅对非零竞争对计算，再平均294个有支持问题，两者误差均较高。零差值数值对照的整体MAE更低，说明整体MAE改善本身不足以证明学会预测竞争组合优劣。

该294题指标是按有支持问题平均的条件性描述，pair-events数量没有扩增独立题量。这里没有将事后诊断升级为确认性结论。

另外以共同32400个竞争对为分母作精确可加分解，各行前两个误差贡献之和等于总MAE。

| 数值预测 | 非零差值项误差贡献 | 零差值项误差贡献 | 总MAE |
|---|---:|---:|---:|
{chr(10).join(decomposition_rows)}

在该分解中，M3与M4的非零差值误差贡献上升，零差值项贡献减少。当前整体MAE改善的数值来源可以直接定位。

## 跨生成方向

600题中，579题存在多个三重复经验最优组合，306题的20个组合三重复平均奖励全部并列。后者占51%。三重复均值仍是有噪声的经验量。

先由两个生成重复确定非零差方向，再用第三个重复评价。下面同时报告持出重复仍为严格差时的同向率和持出并列比例。

| 范围 | 持出seed | 前两重复严格pair-events | 同向/前两严格 | 同向/持出仍严格 | 持出并列比例 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(noise_rows)}

较大经验差的严格方向具有更多一致性，gap=1时持出仍严格的同向率约86.8%—90.2%。同时仍有约52%—54%的持出并列。该诊断支持存在部分稳定差异，无法将所有经验优势视为真实条件期望。

## 内容负对照

整题奖励块仅在合法训练区域的任务/BBH子任务内部置乱，保持组合与生成结构。主超参数固定复用，控制未重新调参，解释范围为固定配置的内容关联诊断。

| 目标 | 原始内容对应 | 置乱对应 | 原始−置乱百分点 | 配对95%区间，百分点 |
|---|---:|---:|---:|---|
{chr(10).join(negative_rows)}

M3、M4相对匹配置乱的区间均跨零。当前训练配置下，同语义组内问题内容与具体组合成绩对应的额外价值尚未得到可靠证据。该结果没有穷尽问题输入的潜在信息。

## 并列和数值敏感性

若一题20奖励全相同，lex SPO+仍为2(max q−q_lex)，均匀真实tie变体为2(max q−mean q)。前者产生索引相关偏好，后者促进分数相同。真实奖励并列并不自动消除训练信号。

主M4五折跨OSQP和Clarabel审阅的600个输入选择变化为0。均匀tie敏感性的一折中，两后端目标差仅约6.35e-10，却有83/120个exactargmax变化，变化题旧分数间隔最大4.29e-10。原模型保留。这使均匀tie点值的纯并列效应解释受到数值近并列影响。

Top5相对Top3增益{pp(stats['sensitivities']['Top5_minus_Top3']['gain'])}个百分点，区间{interval(stats['sensitivities']['Top5_minus_Top3']['paired95'])}。均匀tie相对主SPO增益{pp(stats['sensitivities']['uniform_true_ties_minus_lexicographic']['gain'])}个百分点，区间{interval(stats['sensitivities']['uniform_true_ties_minus_lexicographic']['paired95'])}。两项均为预声明敏感性，未据外折重新选择。

## 结论、限制与下一步

优势是能够把总体误差、竞争差异、零差值收缩和实际决策分开核对。风险是三重复有限、奖励并列多、校准尺度及数值近并列。当前阻碍包括未确认的服务增益，以及非零竞争差异上的误差。唯一失效原因尚无法识别，下一步判断见06报告。
""")
    cost_rows=[]
    for name in main_names+['M3_top5','M4_uniformtie']:
        v=costs['variants'][name];service=results['service_cache_costs'][name]
        cost_rows.append(f"| {name} | {v['sum_training_cpu_seconds']:.3f} | {v['sum_training_wall_seconds']:.3f} | {1000*v['sum_inference_wall_seconds']:.3f} | {v['max_process_peak_working_set_bytes']/2**20:.1f} | {service['tokens']:,.0f} |")
    write('06_STATISTICAL_COST_AND_RESEARCH_DECISION.md',f"""
# E8 统计、成本与研究判断

## 研究问题与证据范围

用固定输入、真实聚合奖励和相同线性容量检验训练目标。原600题已在历史研究中使用，五折训练历史重叠。当前区间为固定开发交叉验证的条件性证据。原V2 validation200未参与本轮选择，fresh400和sealed200继续封存。

## 统计协议

主指标是按题平均的三生成聚合正确性。GSM8K与BBH各300题，主权重各50%，BBH内部按题平均。10000次配对bootstrap在外折×任务内抽原始问题，所有方法和三生成一起移动。仅M3、M4对原固定M0的两项检验进行Holm控制，其他指标和敏感性属于探索性。BBH子任务宏平均与27个支持量分解单列。

M3增益+0.111个百分点，95%CI[-0.611,+0.833]。M4为+0.667，95%CI[-0.944,+2.278]。两项Holm p均为0.8839。当前符合点值提升但区间跨零的情况D，M3同时呈现中心化误差下降而服务增益未确认的情况B。

原特征的PCA正交加训练标准化令M0正则网格仅缩放所有分数，调参版与固定版路由一致。这是该基线搜索的结构退化，当前没有通过放宽表示或另加参数搜索改变这一设置。

## 成本与求解状态

70个最终模型、610次目标拟合完成。训练CPU还包含额外预处理和OOF Ridge。下表为成功模型日志求和，wall为跨折拟合耗时之和。

| 模型 | 成功训练CPU秒 | 成功训练wall秒之和 | 五个120题向量批次推理毫秒 | 进程峰值MiB | 服务缓存Token |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cost_rows)}

全部成功模型训练CPU合计{costs['total_training_cpu_seconds']:.2f}秒。原失败运行捕获CPU下界{costs['original_run_cpu_lower_bound']:.2f}秒，恢复运行下界{costs['recovery_run_cpu_lower_bound']:.2f}秒。两次运行可合并为至少{costs['original_run_cpu_lower_bound']+costs['recovery_run_cpu_lower_bound']:.2f}秒的CPU下界。该下界与成功模型日志存在覆盖，二者不再累加。

索引记录的23.155秒是恢复运行wall，不是完整研究运行时间。失败计算与数值资格费用保留在私有日志，完整总CPU没有精确计量，已报告可核验下界。内存为Windows进程自启动峰值，非某次拟合独占增量。极短CPU推理计时可能因分辨率记为0。

服务Token来自已有执行缓存复算。每个路由器在600题×3生成中对应5400次成员调用和1800次聚合。本轮实际新增LLM/API推理为0。向量推理计时仅覆盖PCA和线性头，历史MiniLM编码与语言模型服务时间另有范围。

## 四个最终问题

### M3或M4是否真正优于原Ridge

两者点值分别提高0.111和0.667个百分点。预定问题级配对区间均跨零，Holm检验未通过，可靠改进证据尚未成立。M4在三个生成种子与两个任务方向均正，保留为探索性信号。

### 是否降低竞争相对价值误差

整体校准竞争MAE下降，M3中心化MSE也略降。事后分解显示大部分竞争差为0，零差值数值对照的MAE更低。每题先仅计算非零竞争对的MAE，再平均294个有支持题，M3、M4分别0.40211、0.41427，M0为0.38830。当前未建立更强的竞争差异辨识能力。

### 改善是否稳定

M4跨seed点方向一致，其600个输入跨求解器选择也一致。主准确率区间仍宽，真实内容对应相对语义组内置乱的区间同样跨零。当前无法把正点值确认为独立问题上的稳定收益。均匀tie敏感性另受近并列数值影响。

### 是否进入有限反馈研究

当前科学关口未通过，有限反馈采集保持停止。本轮没有启动Random/IG等新采集网格，也未使用保留集追求显著性。

## 可反驳失效假设

| 假设 | 当前证据 | 尚缺的区分证据 |
|---|---|---|
| 当前表示容量限制 | 相同PCA16线性头下多个目标均无确认增益 | 本轮没有新表示/容量干预，不能指定为唯一原因 |
| 奖励噪声与并列 | 51%题全20经验奖励并列，持出生成大量并列，uniform数值敏感 | 三重复不足以精确识别条件期望 |
| 题目内容缺乏可用组合信号 | 固定配置内容−置乱区间跨零 | 当前控制没有穷尽特征或所有可学习函数 |
| 独立训练题数量不足 | 每子任务总支持约11—12，主收益区间较宽 | 本轮未进行独立题数量学习曲线，数量因果作用未知 |
| 开发集不稳定差异 | 主区间跨零，旧数据多轮开发，M3一seed方向负 | 独立确认尚无授权执行 |

## 优势、风险与下一步

优势是完整固定目标、容量、真实标签、数值资格和配对统计，负结果与失败重试均可追溯。风险集中在有限经验奖励、训练历史重叠、校准与近并列解释。对本轮计算与未通过关口的结论置信度高，对失效唯一原因及未来总体提升的证据有限。后续研究方向由新的明确实验任务决定。
""")
    write('EXPERIMENT_SUMMARY.md',f"""
# E8 结果摘要

## 执行状态

全部合法缓存实验、数学资格、作者目标核对和配对评价完成。五折×14变体，共70模型。新增物理LLM/API推理为0。保留原49模型，数值执行修订补齐21个未封存模型。原研究结果保持存档。

## 主结果

| 方法 | 准确率 | 对原Ridge增益百分点 | 配对95%CI百分点 | Holm p |
|---|---:|---:|---|---:|
| 原直接20组合Ridge | 85.278% | 0 | — | — |
| M3竞争加权Ridge | 85.389% | +0.111 | [-0.611,+0.833] | 0.8839 |
| M4精确SPO+ | 85.944% | +0.667 | [-0.944,+2.278] | 0.8839 |

两个主比较均未通过科学关口。外折训练选出的全局固定组合为84.944%。调参Ridge为85.278%。M4的三个生成重复与两个任务方向均正，当前属于探索性点值。

## 四个回答

**M3/M4选队更好吗**

点值改善，问题级区间跨零。当前可靠改进证据不足。

**相对价值误差改善了吗**

整体竞争MAE下降。先在每题仅计算非零竞争对MAE，再平均294个有支持题，M0误差0.38830，M3为0.40211，M4为0.41427。零差值收缩解释了主要误差变化，整体零差值数值对照MAE0.09337还低于M4的0.09610。该事后描述不作为新方法或主检验。

**稳定收益还是偶然差异**

M4逐seed点方向与跨求解器主动作一致。准确率区间与内容−置乱区间均跨零，旧开发数据和三生成经验奖励尚不能确立稳定泛化。

**能进入有限反馈了吗**

本轮保持停止。具体阻碍为未通过主检验、非零竞争差异误差与内容控制证据不足。容量、题量与奖励噪声是不同假设，当前证据不能指定唯一原因。

## 重要实现选择与局限

保留原固定正则Ridge及同范围调参Ridge，两项主检验均对原固定版本。所有数值误差使用同一历史OOF非负仿射校准，服务动作使用原分数。置乱控制固定主超参数，未重新调参。

PCA与训练标准化使M0正则搜索退化为统一正比例缩放，其调参准确率相同具有结构原因。校准误差还包含320题OOF到480题重拟合的尺度迁移，排序与服务准确率保持单列。

精确SPO+采用凸QP。OSQP均匀tie敏感性超出固定迭代精度后，经预评价执行修订使用数学相同Clarabel问题补齐。五个主M4模型600个输入的跨求解器动作一致。均匀tie的一折却有83/120个近并列动作变化，其点值无法纯粹解释为tie规则效应。

完整CPU成本只能给出成功模型日志和全运行捕获下界。当前区间解释限于原600题的条件开发交叉验证。

## 文件入口

01报告为文献与方法映射，02为数学软件资格，03为目标与实现，04为主比较，05为噪声排序，06为统计成本和最终判断。E8_PROTOCOL.json、E8_EXECUTION_AMENDMENT_01.json、E8_COMPLETE_RESULTS.json与E8_PRIMARY_STATISTICS.json保留完整配置与指标。REPRODUCE.md提供防覆盖重放入口。
""")
    rows=[]
    for fold in range(5):
        for variant in results['variants']:
            rows.append(dict(cell=f'fold{fold}__{variant}',outer_fold=fold,variant=variant,
                role='primary' if variant in ['M3','M4'] else 'comparator_or_diagnostic',
                independent_evaluation_questions=120,generation_repeats=3,status='FROZEN_AND_EVALUATED'))
    with (BASE/'public/EXPERIMENT_MATRIX.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='REPORTS_WRITTEN',research_decision=results['next_stage'])))


if __name__=='__main__':main()
