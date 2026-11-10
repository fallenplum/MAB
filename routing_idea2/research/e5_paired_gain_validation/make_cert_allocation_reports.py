"""Write E5-D/E reports from completed numerical artifacts without choosing policies."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np


BASE=Path(__file__).resolve().parent
ROOT=Path(r'E:\语言模型组合路由\research\e5_paired_gain_validation')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def percent(x):return f'{100*x:.4f}%'
def number(x):return '未定义' if x is None else f'{x:,.0f}'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result_input(filename,legacy_subdirectory):
    flat=BASE/filename
    if flat.is_file():return flat
    return BASE/legacy_subdirectory/filename


def selection_numerical_sensitivity_section(output_directory):
    path=Path(output_directory)/'ALLOCATION_SELECTION_NUMERICAL_SENSITIVITY.json'
    if not path.is_file():return []
    result=read(path)
    if result.get('status')!='COMPLETE':
        raise ValueError('Allocation selection sensitivity must be complete before reporting')
    candidates=[Path(output_directory)/'ALLOCATION_SYNTHETIC_RESULTS.json',
                Path(output_directory)/'allocation_synthetic/ALLOCATION_SYNTHETIC_RESULTS.json']
    existing=[candidate for candidate in candidates if candidate.is_file()]
    if existing and sha(existing[0])!=result['original_ALLOCATION_SYNTHETIC_RESULTS_sha256']:
        raise ValueError('Selection sensitivity belongs to another primary allocation result')
    if result.get('original_results_modified') is not False:
        raise ValueError('Selection sensitivity must preserve the frozen primary result')
    s=result['summary']
    counts=s['nonnested_estimated_selection_loss_worse_than_uniform']
    tolerance=result['tolerance']
    return ['', '## 选择器数值边界与稳健性', '',
        f"独立审查在{s['total_scenario_method_MC_draws']:,}个场景×方法×MC状态中发现{s['positive_near_zero_estimates']}个最大收益估计落在(0, {tolerance:g}]。这是共享模拟世界中的数值状态计数，独立题量仍按各配置定义。", '',
        f"第一项敏感性将最大估计不超过{tolerance:g}的状态归入零收益默认，正分数之间仍采用原精确argmax。第二项同时将距离正最大值不超过{tolerance:g}的规则视为并列，按固定索引选队。两项均从已冻结估计数组计算，原严格正值与精确argmax主规则、原MC表继续保留。", '',
        '| 数值选择规则 | nonnested中Estimated损失高于Uniform的预算格数 |',
        '|---|---:|',
        f"| 原主规则 | {counts['original']}/{s['nonnested_scenarios']} |",
        f"| 仅零阈值容差 | {counts['zero_tolerance']}/{s['nonnested_scenarios']} |",
        f"| 零阈值与正分数并列容差 | {counts['full_tolerance']}/{s['nonnested_scenarios']} |", '',
        f"两项敏感性对Estimated−Uniform平均策略价值损失差的最大绝对改变量分别为{100*s['maximum_absolute_zero_tolerance_change_in_estimated_vs_uniform_contrast']:.7f}和{100*s['maximum_absolute_full_tolerance_change_in_estimated_vs_uniform_contrast']:.7f}个百分点。最坏门控MSE比较仍为{s['estimated_worst_MSE_better_than_uniform']}/{s['total_MSE_comparisons']}格Estimated较低，MSE比较变化数量为{s['MSE_comparison_changed_count']}。", '',
        '由此，零附近浮点选择会影响部分插件策略价值的末位数值，但本轮“估计效率改善而最终策略选择未改善”的方向在两项诊断中均保留。该诊断仅针对合成插件选择器，E5-C的Text-only主门控及附录F固定时域证书各自保持原评价范围。', '',
        f"`ALLOCATION_SELECTION_NUMERICAL_SENSITIVITY.json`的SHA256为`{sha(path)}`。它记录原主结果SHA、全部18份来源NPZ的SHA及逐方法与配对MC误差，原主结果修改标志为`{str(result['original_results_modified']).lower()}`。", '']


def certification_report():
    synthetic_path=result_input('CERTIFICATION_SYNTHETIC_RESULTS.json','cert_synthetic')
    power_path=result_input('CERTIFICATION_DEVELOPMENT_POWER.json','power_development')
    s,p=read(synthetic_path),read(power_path)
    rows=s['rows']
    worst=max(rows,key=lambda r:r['results']['false_nonpositive_certificate']['rate'])
    worst_result=worst['results']['false_nonpositive_certificate']
    maxharm=max(r['results']['harmful_selected']['rate'] for r in rows)
    maxcoverage=max(r['results']['simultaneous_noncoverage']['rate'] for r in rows)
    report=[
        '# 04 配对认证有效性与功效', '',
        '## 研究问题', '',
        '检查导师稿件正文第5—6节、附录D/F的固定家族配对证书是否按公式运行，以及当前独立题量能否支持严格正收益下界。安全性、正收益检出率和默认返回率分别评价。', '',
        '## 实验设置与实现对应', '',
        '已经视觉核对PDF第19页式(66)及第26页式(96)。完整反馈以每个原题的三个配对差值均值为一个观测，样本方差采用n−1分母。实际用于选择的全部非默认规则计入J，已知收益为0的默认选项另列。', '',
        r'完整反馈半径为 $r_j=\sqrt{2s_j^2\log(6J/\delta)/n}+14\log(6J/\delta)/[3(n-1)]$。只接受严格正下界，固定并列规则；固定n后执行，无前缀停止。', '',
        r'主动反馈接口实现附录F的 $r_T=\sqrt{2u/(Tp_{\min})}+4u/(3Tp_{\min})$，$u=\log(2J/\delta)$。该接口保持全部进入问题为分母，未获取反馈的潜在活跃题仍需正购买概率，所有规则不切换的区域贡献为已知0。', '',
        '合成设置为ρ∈{0.02,0.10,0.30,0.60}，名义Δ/ρ∈{−0.5,0,0.1,0.5}，n∈{200,400,1000,3000,10000}，L∈{1,3}，重复独立与完全相关两种机制，J∈{1,5}，δ=0.05。每配置10000次Monte Carlo。J=5共享四等权输入层与同题D，并同时包含负、零、正增益门控。', '',
        '共640条配置记录。L=1的两种相关性标签具有相同分布，程序使用同一随机流检查等价，因此独特的模拟随机流为480组。各配置的MC区间是单配置Clopper–Pearson区间，未解释为640配置同时置信保证。分层与cluster-mean的多项分布计数是完整样本均值、平方和的精确充分统计模拟。', '',
        '### 必要界所对应的分布类', '',
        '正文式(13)、附录D式(70)直接针对单门控、L=1的正负三值两点检验。允许任意题内相关的重复分布类包含“所有重复完全相同”，所以该类的最坏情形仍受同一独立题数必要界约束。该证明还同时要求负分布下的安全性与正分布下的检出能力，并未给出log J必要界。', '',
        '严格条件独立的L=3是更窄的分布类。JSON在这类行中保留原L=1数值作为大类参照，并明确标记reference_only，该数值不作为严格独立三重复子类的必要题量。未在本实验另行推导该子类的信息论必要界。', '',
        '## 对照', '',
        '始终返回默认的安全错误率为0、默认比例为1、正收益检出率为0。这个对照单独保存，避免将未检出有益策略当作成功识别。L=3独立重复和完全相同重复使用相同独立题量n。', '',
        '## 合成数值结果', '',
        f"最大错误非正认证率为{percent(worst_result['rate'])}，出现于ρ={worst['rho']}、Δ={worst['nominal_delta']}、n={worst['n']}、L={worst['L']}、{worst['dependence']}、J={worst['J']}。该配置{worst_result['count']}/10000，MC95%区间[{percent(worst_result['mc_ci95'][0])}, {percent(worst_result['mc_ci95'][1])}]。", '',
        f'所有配置的最大有害策略选择率为{percent(maxharm)}，最大同时区间未覆盖率为{percent(maxcoverage)}，低于设定δ=5%。这属于预定有限网格的数值验证，不能替代定理证明。', '',
        '| ρ | 真增益 | n | L | 同题重复 | 正收益检出率 | MC95%区间 | 平均区间宽度 |',
        '|---:|---:|---:|---:|---|---:|---|---:|'
    ]
    selected=[r for r in rows if r['J']==1 and r['gain_ratio']==.5 and
              ((r['rho']==.02 and r['n'] in (400,10000)) or (r['rho']==.1 and r['n']==1000)) and
              not(r['L']==1 and r['dependence']=='perfectly_correlated')]
    for r in selected:
        power=r['results']['positive_detection_power']
        report.append(f"| {r['rho']:.2f} | {100*r['nominal_delta']:.2f}pp | {r['n']} | {r['L']} | {r['dependence']} | {percent(power['rate'])} | [{percent(power['mc_ci95'][0])}, {percent(power['mc_ci95'][1])}] | {r['mean_interval_width']:.6f} |")
    report += ['', 'ρ=0.02且真增益1pp时，400题在本证书下全部返回默认，10000题才取得较高检出率。ρ=0.10且真增益5pp时，1000题的L=3独立重复检出率显著高于完全相关重复。题内重复对精度的帮助依赖联合分布，独立问题数仍然为n。', '',
        '## 现有矩阵上的开发诊断', '',
        '每个外折的F/G冻结其本身20条规则，E有120个不同问题。对GSM8K、BBH各60题分别构造区间，δ_fold=0.01、δ_dataset=0.005，然后按1/2、1/2相加。五折联合显示采用失败概率之和0.05和各折均值/半径的平均；五折事件的union bound不需要事件之间独立。该平均对象是五个折特定流程，不能据此将600题直接作为同一个冻结策略的iid校准样本。', '',
        '| 外折 | G选定的因子化主规则 | 全20规则最大LCB | 家族平均半径 | 认证选择 |',
        '|---:|---|---:|---:|---|']
    for f in p['fold_certificates']:
        report.append(f"| {f['fold']} | {f['primary_factorized_rule']} | {max(f['certificate']['lcb']):.6f} | {np.mean(f['certificate']['radius']):.6f} | {f['selected_rule']} |")
    report += ['', f"五折平均家族的最大LCB为{max(p['fivefold_family_average_diagnostic']['lcb']):.6f}，全部返回默认。代码只读取冻结门控的E差值与输入元标签，未为功效分析重新拟合门控。", '',
        '### 三种样本需求量的边界', '',
        '第一类是指定经验Bernstein证书的充分数量。JSON分别保存正文式(12)/附录式(69)在假设iid目标混合下的数量，以及由附录F.1分层组合与式(68)下界推导的平衡采样数量。真实总体ρ与方差未知，用开发估计代入得到的是该证书的规划插件值。表中约125,000等数量属于这一身份，通用最小题量仍未由本实验确定，不能据此限制其他可能更有效的程序。', '',
        r'对两个等权层、总题量n且各n/2，令 $A=(\sqrt{\rho_1}+\sqrt{\rho_2})/2$、$t=\log(12J/\delta)$。由式(68)逐层相加，$LCB\ge G-6A\sqrt{t/n}-24t/n$。代码直接解二次不等式并向上取偶数。它与较松的式(69)充分界使用不同常数化步骤，两个数量不应被当作同一个公式的复现。', '',
        '第二类是已知三值分布下的真实合成检出率，上表和完整640配置给出。第三类是每折原有120题经验池上的分层整题重采样，目标模拟题数仍固定为200、400、1000、3000、10000，每格2000次。原三个生成重复及20条规则共同移动。其结果仅描述给定开发池的经验分布，原始独立问题支持数仍是120。', '',
        '| 外折 | 因子化主规则 | 开发净收益 | 开发ρ GSM/BBH | 假定400题F.1插件半径 | 目标+1pp的F.1插件充分题量 |',
        '|---:|---|---:|---|---:|---:|']
    for f in p['fold_certificates']:
        r=next(x for x in p['planning'] if x['fold']==f['fold'] and x['rule']==f['primary_factorized_rule'])
        radius=next(x['F1_plugin_half_width'] for x in r['empirical_radius_planning'] if x['total_independent_questions']==400)
        needed=next(x['fixed_balanced_F1_plugin_sufficient_n'] for x in r['target_gains'] if x['gain']==.01)
        report.append(f"| {f['fold']} | {r['rule']} | {100*r['empirical_gain']:+.4f}pp | {r['empirical_rho_by_dataset'][0]:.6f}/{r['empirical_rho_by_dataset'][1]:.6f} | {radius:.6f} | {number(needed)} |")
    report += ['', '零观察分歧不证明总体ρ为0。程序对超过当前经验ρ的正收益目标返回未定义，保留估计不足；没有借助零方差把正收益认证需求写成极小题量。', '',
        '## 统计结论', '',
        '固定规则、独立题与共享反馈的合成验证符合预期，安全性与检出能力呈明显区别。现有小样本在当前保守家族证书下不足以产生正下界。单纯观察默认返回率不能判断输入排序有没有收益，该问题须看E5-C的真实纠错和匹配控制。', '',
        '## 优势、失败原因与风险', '',
        '优势是每个门控共享同题原始配对反馈，保持正确的独立题量和有限家族J。数值实现核对了默认、方差分母、F.1分层加权和固定时域IPW。', '',
        '风险来自开发题已被反复使用、折间历史重叠、有限生成重复和未知总体分布。相关seed标识并不能证明条件独立。实际认证模型允许题内重复相关，但仍要求同一部署边际与独立校准题。本文开发诊断不主张这些总体前提已经得到数据证明。', '',
        '## 下一步', '',
        '保持固定评价时点和候选族。是否需要新增独立题应先按候选主收益和可接受半径规划，保留理论充分数、合成实际功效和经验池重采样三套结果，继续遵守fresh400与sealed200的角色边界。', '',
        '## 产物', '',
        f'`CERTIFICATION_SYNTHETIC_RESULTS.json`包含完整640条配置，源SHA256为`{sha(synthetic_path)}`。', '',
        f'`CERTIFICATION_DEVELOPMENT_POWER.json`包含五折F.1、100条门控功效规划与25个经验池重采样网格，源SHA256为`{sha(power_path)}`。', '']
    lines=[(line.replace(chr(92)*2,chr(92)) if '$' in line else line).replace('；','。') for line in report]
    (BASE/'04_CERTIFICATION_VALIDITY_AND_POWER.md').write_text('\n'.join(lines),encoding='utf-8')


def allocation_report():
    path=result_input('ALLOCATION_SYNTHETIC_RESULTS.json','allocation_synthetic')
    a=read(path)
    rows=a['cases']
    non=[x for x in rows if x['family']=='nonnested']
    reductions=[1-x['methods']['Estimated-two-resource']['worst_second_moment']/x['methods']['Uniform']['worst_second_moment'] for x in non]
    selection_worse=sum(x['methods']['Estimated-two-resource']['plugin_policy_selection']['mean_value_loss']>x['methods']['Uniform']['plugin_policy_selection']['mean_value_loss'] for x in non)
    theoretical_improvements=sum(x['methods']['Estimated-two-resource']['worst_gate_MSE']<x['methods']['Uniform']['worst_gate_MSE'] for x in rows)
    maxgap=max(abs(x['solver_details']['Oracle-two-resource']['KKT']['duality_gap']) for x in rows)
    sensitivitygap=max(abs(r['solver_duality_gap']) for f in a['estimated_weights_costs_sensitivity'] for r in f['rows'])
    warnings=sum(r['solver_warning_count'] for f in a['estimated_weights_costs_sensitivity'] for r in f['rows'])
    gate_path=BASE/'REAL_ALLOCATION_GATE.json'
    if not gate_path.is_file():gate_path=ROOT/'public/REAL_ALLOCATION_GATE.json'
    gate=read(gate_path) if gate_path.exists() else None
    report=['# 05 双资源反馈分配', '', '## 研究问题', '',
        '对已经冻结的默认、候选和门控族，检验稿件第二矩代理的两资源最优设计，区分估计效率、最终策略选择和保守证书检出能力。该阶段不更新基础预测器。', '',
        '## 实验设置', '',
        '合成总体含6个活跃输入层和一个所有门控不切换的层。活跃质量为0.9，未活跃质量0.1仍进入估计分母。每题L=3个条件独立三值差值，a_h直接计算题级均值平方的期望。主设计知道真实w_h、期望执行费用c_h，仅从独立先导估计a_h。', '',
        '先导在每个活跃层条件采样30道独立问题，共180题。先导购买180次gold、945单位执行资源、2700成员调用和1080聚合调用。若依赖逐层拒绝采样获得同样条件样本，预计还需1277.78次输入筛查，报告记录这个额外条件采样假设。先导信息对主机制对照完全共享，另列无先导Uniform，将相同总预算投入反馈，达到活跃全反馈后保留剩余预算。', '',
        '反馈阶段有400个iid进入问题。双预算分别设置为活跃区全量配对所需资源的25%、50%、75%，形成9个组合。nested3门控与预定nonnested4门控各9格，每格2000次共享世界Monte Carlo。所有方法共享同一H、配对均值和购买Uniform随机数；门控共享执行一次配对，J不倍增执行费用。', '',
        '## 对照与求解', '',
        'Uniform、Cost-only、Disagreement-based、Single-resource、Estimated two-resource、Oracle two-resource和No-pilot Uniform均报告。Single-resource先求仅annotation约束下的形状，再按共同规则回退到执行预算内，保留回退系数。Oracle仅供已知矩的理论/审计对照。', '',
        '使用解析Jacobian的凸epigraph求解，恢复非负gate/resource对偶变量并计算真正的dual objective。所有设计记录可行性、对偶间隙、点坐标最小化与互补松弛。λ由最坏门控问题求出，未以任意平均门控权重代替。', '',
        f'主oracle18格最大primal-dual间隙为{maxgap:.3g}。900个估计w/c敏感设计最大间隙为{sensitivitygap:.3g}，记录{warnings}条SLSQP中间越界后裁剪警告，每个最终解仍逐项验证边界、可行域和对偶目标。', '',
        '## 门控支配与退化', '',
        'nested族的最大覆盖门控逐点支配其余门控，9格目标均退化为单目标。额外数值资格将all-one活跃门控加到任意预定nonnested族，确认最大覆盖规则再次支配全部。真实完整门控族若含恒正r评分的零阈值规则，也会出现相同结构，应保留并报告。', '',
        '预定nonnested4族无逐点支配，完整保留独立敏感性结果。它不能替代真实完整族的退化目标。', '',
        '## 数值结果', '',
        '| 门控族 | ann预算 | exec预算 | Uniform最坏二阶矩 | Estimated最坏二阶矩 | Oracle最坏二阶矩 | Estimated策略价值损失 | Uniform损失 | No-pilot Uniform损失 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        m=x['methods']
        report.append(f"| {x['family']} | {x['annotation_fraction']:.0%} | {x['execution_fraction']:.0%} | {m['Uniform']['worst_second_moment']:.6f} | {m['Estimated-two-resource']['worst_second_moment']:.6f} | {m['Oracle-two-resource']['worst_second_moment']:.6f} | {m['Estimated-two-resource']['plugin_policy_selection']['mean_value_loss']:.6f} | {m['Uniform']['plugin_policy_selection']['mean_value_loss']:.6f} | {m['No-pilot-Uniform']['plugin_policy_selection']['mean_value_loss']:.6f} |")
    report += ['', f'在预定nonnested族，Estimated相对Uniform的最坏二阶矩下降范围为{100*min(reductions):.2f}%至{100*max(reductions):.2f}%，9格插件策略价值损失更高的数量为{selection_worse}。18格中Estimated的实测最坏门控MSE低于Uniform的数量为{theoretical_improvements}，完整MC误差区间保存在JSON。估计效率和选策略质量没有共同单调关系。', '',
        '### 中间预算点的全部方法', '',
        '| 方法 | 最坏二阶矩 | 最坏门控MSE | 策略价值损失 | 含先导期望gold | 含先导期望执行单位 | 证书返回默认 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    middle=next(x for x in non if x['annotation_fraction']==.5 and x['execution_fraction']==.5)
    for name,m in middle['methods'].items():
        report.append(f"| {name} | {m['worst_second_moment']:.6f} | {m['worst_gate_MSE']:.8f} | {m['plugin_policy_selection']['mean_value_loss']:.6f} | {m['expected_resources']['gold_including_pilot']:.3f} | {m['expected_resources']['execution_including_pilot']:.3f} | {percent(m['fixed_horizon_certified_selection']['default_rate'])} |")
    report += ['', '全部18格的7方法均使用附录F固定T、共同p_min=0.05的保守半径，认证选择全部返回默认。主选择损失表使用未认证插件argmax并明确区分，两者不可合并为有证书的质量收益。', '',
        '主先导估a与真实a的比较沿用真实w/c和同一可行域。18格全部满足数值形式的Theorem19代理误差界，原真实目标超额与2E_str均保存。', '',
        '### 估计可行域误差', '',
        '| 门控族 | 独立pilot数 | ann真实期望约束违反比例 | exec真实期望约束违反比例 |',
        '|---|---:|---:|---:|']
    for row in a['estimated_weights_costs_sensitivity']:
        report.append(f"| {row['family']} | {row['independent_pilot_repeats']}×9预算 | {percent(row['true_annotation_violation_rate'])} | {percent(row['true_execution_violation_rate'])} |")
    report += ['', '该敏感性用50个独立iid先导估计w/c/a，共900个设计。它们对估计可行域求解成功，真实期望约束却可能违反。这个比例描述特定合成估计过程，不能套用仅估a、固定w/c的同域稳健性定理。实际随机消费超出期望预算的概率另列，未使用硬预算停止。', '',
        '## 真实矩阵回放关口', '']
    if gate is None:
        report += ['真实矩阵上的双资源回放尚未启动，等待独立记录的E5-C预锁关口。当前合成数值完整，未以合成估计效率替代真实收益证据。']
    else:
        report += ['真实矩阵双预算回放状态为`NOT_RUN_GATE_NOT_PASSED`。E5-C预定的两项主要比较没有同时通过正向关口，因此保留完整合成结果，真实采集区实验按协议分流停止。执行判据及结果来自`REAL_ALLOCATION_GATE.json`，具体冻结记录如下。', '', '```json',json.dumps(gate,ensure_ascii=False,indent=2),'```']
    report += ['', '## 优势、失败原因与风险', '',
        '优势是两种资源分开、所有方法共享合法反馈和成员复用、先导收费、真实购买概率与全部进入问题的分母明确。已知参数合成可以精确计算目标和策略价值。', '',
        '风险包括保守证书、有限独立先导、成本/分层估计误差和目标退化。最坏二阶矩与确切最坏方差相差各门控的G_j平方，策略选择又依赖收益间隔和共享估计相关性。当前nonnested例子提供了方差代理改善却策略选择更差的负结果，不能宣称回答质量随分配改进。', '',
        '合成执行单位、成员/聚合调用是声明世界的资源，不是实测LLM Token或GPU时间。真实语言模型新调用为0。原始模拟估计向量和资源保存在18份NPZ，逐案参数、概率、MC误差与预算偏差均保留。', '',
        '## 下一步', '',
        '依照E5-C科学关口决定是否执行真实矩阵分配。若门控排序尚无正证据，保留当前完整合成与负结果，继续研究有益切换识别，不扩大真实生成或认证调用。', '',
        '## 产物', '',
        f'`ALLOCATION_SYNTHETIC_RESULTS.json`的SHA256为`{sha(path)}`。包含18格主设计、126方法结果及900个估计可行域敏感设计。', '']
    report.extend(selection_numerical_sensitivity_section(BASE)[1:])
    (BASE/'05_TWO_RESOURCE_ALLOCATION.md').write_text('\n'.join(line.replace('；','。') for line in report),encoding='utf-8')


if __name__=='__main__':
    certification_report()
    allocation_report()
    print('CERTIFICATION_AND_ALLOCATION_REPORTS_WRITTEN')
