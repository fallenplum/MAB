"""E5-C question-record bootstrap with re-computed matched-control counts."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from collections import defaultdict
import numpy as np


def _cell_indices(task, fold):
    return [(f, name, np.flatnonzero((fold == f) & (task == name)))
            for f in sorted(set(fold.tolist())) for name in ('gsm8k', 'bbh')]


def _holm(pvalues):
    result = {}
    previous = 0.
    names = sorted(pvalues, key=lambda name: pvalues[name])
    for index, name in enumerate(names):
        previous = max(previous, min(1., (len(names) - index) * pvalues[name]))
        result[name] = previous
    return result


def _intervals(point, draws):
    point = np.asarray(point)
    draws = np.asarray(draws)
    q95 = np.quantile(draws, [.025, .975], axis=0)
    q975 = np.quantile(draws, [.0125, .9875], axis=0)
    centered_p = (1 + np.sum(np.abs(draws - point) >= np.abs(point) - 1e-15, axis=0)) / (len(draws) + 1)
    return dict(point=point, ci95=np.stack(q95, axis=-1),
                ci97_5_bonferroni_two_comparisons=np.stack(q975, axis=-1),
                centered_bootstrap_two_sided_p=centered_p,
                p_interval_note='Centered bootstrap p and percentile intervals are descriptive and need not invert exactly')


def matched_point_estimates(gates, D, task, sub, fold):
    """Each row is a frozen gate. Match counts separately in each outer fold."""
    gates = np.asarray(gates, dtype=bool)
    D = np.asarray(D, dtype=float)
    task, sub, fold = np.asarray(task).astype(str), np.asarray(sub).astype(str), np.asarray(fold)
    n = len(task)
    if gates.ndim != 2 or gates.shape[1] != n or D.ndim != 2 or D.shape[0] != n:
        raise ValueError('Original question axes must agree')
    if not np.isin(D, (-1, 0, 1)).all():
        raise ValueError('Only original ternary paired rewards are supported')
    mean_D = D.mean(axis=1)
    weights = np.zeros(n)
    for name in ('gsm8k', 'bbh'):
        mask = task == name
        if not mask.any():
            raise ValueError('Primary GSM/BBH mixture requires both tasks')
        weights[mask] = .5 / mask.sum()
    observed_perq = gates * mean_D[None]
    controls = {name: np.zeros_like(observed_perq) for name in ('global', 'task', 'subtask')}
    for f in sorted(set(fold.tolist())):
        rows = np.flatnonzero(fold == f)
        definitions = dict(global_=[rows], task=[rows[task[rows] == name] for name in ('gsm8k', 'bbh')])
        definitions['subtask'] = [rows[task[rows] == 'gsm8k']] + [
            rows[(task[rows] == 'bbh') & (sub[rows] == name)] for name in sorted(set(sub[rows][task[rows] == 'bbh']))]
        for scheme, groups in definitions.items():
            scheme = 'global' if scheme == 'global_' else scheme
            for indices in groups:
                if not len(indices):
                    continue
                inclusion = gates[:, indices].mean(axis=1)
                controls[scheme][:, indices] = inclusion[:, None] * mean_D[indices]
    output = dict(gain=observed_perq @ weights,
                  coverage=gates @ weights,
                  rho=(gates * (D ** 2).mean(axis=1)) @ weights,
                  correction=(gates * (D == 1).mean(axis=1)) @ weights,
                  harm=(gates * (D == -1).mean(axis=1)) @ weights,
                  per_question_gain=observed_perq,
                  per_question_expected_random=controls)
    for scheme, expected in controls.items():
        output['random_' + scheme + '_gain'] = expected @ weights
        output['excess_' + scheme] = (observed_perq - expected) @ weights
    return output


def question_record_bootstrap(gates, D, task, sub, fold, reps=5000, seed=20261110):
    """Re-sample complete original records, then re-match every control count.

    Cells are outer-fold by broad task. All generation repetitions and all gate
    masks travel with the sampled question. The fitted rules remain frozen.
    """
    gates = np.asarray(gates, dtype=bool)
    D = np.asarray(D, dtype=float)
    task, sub, fold = np.asarray(task).astype(str), np.asarray(sub).astype(str), np.asarray(fold)
    if reps < 100:
        raise ValueError('Use at least one hundred bootstrap draws')
    point = matched_point_estimates(gates, D, task, sub, fold)
    k, n = gates.shape
    mean_D = D.mean(axis=1)
    mean_disagreement = (D ** 2).mean(axis=1)
    corrected = (D == 1).mean(axis=1)
    harmed = (D == -1).mean(axis=1)
    rng = np.random.default_rng(seed)
    observed = np.zeros((reps, k))
    coverage = np.zeros((reps, k))
    rho = np.zeros((reps, k))
    correction = np.zeros((reps, k))
    harm = np.zeros((reps, k))
    random_task = np.zeros((reps, k))
    random_subtask = np.zeros((reps, k))
    fold_totals = {}
    resample_checks = []
    for f, name, indices in _cell_indices(task, fold):
        if not len(indices):
            raise ValueError('Every original outer fold must have both broad tasks')
        draws = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=reps)
        broad_denominator = np.sum(task == name)
        weight = .5 / broad_denominator
        g = gates[:, indices].T.astype(float)
        d = mean_D[indices]
        gate_counts = draws @ g
        sum_D = draws @ d
        observed += weight * (draws @ (g * d[:, None]))
        coverage += weight * gate_counts
        rho += weight * (draws @ (g * mean_disagreement[indices, None]))
        correction += weight * (draws @ (g * corrected[indices, None]))
        harm += weight * (draws @ (g * harmed[indices, None]))
        random_task += weight * gate_counts / len(indices) * sum_D[:, None]
        if f not in fold_totals:
            fold_totals[f] = dict(gates=np.zeros((reps, k)), weighted_D=np.zeros(reps), size=0)
        fold_totals[f]['gates'] += gate_counts
        fold_totals[f]['weighted_D'] += weight * sum_D
        fold_totals[f]['size'] += len(indices)
        if name == 'gsm8k':
            random_subtask += weight * gate_counts / len(indices) * sum_D[:, None]
        else:
            for name_sub in sorted(set(sub[indices])):
                local = sub[indices] == name_sub
                support = draws[:, local].sum(axis=1)
                local_gate = draws[:, local] @ g[local]
                local_D = draws[:, local] @ d[local]
                frequency = np.divide(local_gate, support[:, None], out=np.zeros_like(local_gate), where=support[:, None] > 0)
                random_subtask += weight * frequency * local_D[:, None]
        resample_checks.append(dict(fold=int(f), task=name, original_questions=len(indices),
            draw_size_fixed=bool(np.all(draws.sum(axis=1) == len(indices)))))
    random_global = np.zeros((reps, k))
    for values in fold_totals.values():
        random_global += values['gates'] / values['size'] * values['weighted_D'][:, None]
    arrays = dict(gain=observed, coverage=coverage, rho=rho, correction=correction, harm=harm,
                  random_global_gain=random_global, random_task_gain=random_task, random_subtask_gain=random_subtask,
                  excess_global=observed-random_global, excess_task=observed-random_task,
                  excess_subtask=observed-random_subtask)
    summaries = {name: _intervals(point[name], value) for name, value in arrays.items()}
    return dict(point=point, summaries=summaries, bootstrap_draws=arrays,
        metadata=dict(seed=seed, replicates=reps, independent_questions=n, rule_rows=k,
            strata='outer fold crossed with GSM8K/BBH', resampling_checks=resample_checks,
            controls_recomputed_inside_each_draw=True,
            matched_control_rule='counts matched within each original outer fold; expected uniform-control contribution recomputed from resampled complete records',
            generation_repeats_are_new_questions=False,
            scope='conditional development intervals with fixed fitted rules and cached rewards; overlapping training histories and historical model selection not re-fitted'))


def qualification():
    rng = np.random.default_rng(25)
    n = 48
    task = np.tile(np.asarray(['gsm8k']*12+['bbh']*12), 2)
    fold = np.repeat(np.arange(2), 24)
    sub = np.where(task == 'gsm8k', '', np.tile(np.asarray(['a']*6+['b']*6+['a']*6+['b']*6), 2))
    D = rng.choice([-1.,0.,1.], size=(n,3))
    gates = np.vstack((np.zeros(n,dtype=bool),np.ones(n,dtype=bool),rng.random(n)>.5))
    result = question_record_bootstrap(gates,D,task,sub,fold,reps=300,seed=22)
    for scheme in ('global','task','subtask'):
        assert np.allclose(result['bootstrap_draws']['excess_'+scheme][:,:2],0)
    assert np.max(abs(result['bootstrap_draws']['gain']-result['bootstrap_draws']['correction']+result['bootstrap_draws']['harm']))<1e-12
    # Reconstruct the first complete draw from a copied RNG and evaluate the
    # matching operation directly on the duplicated original records.
    repeated_rng = np.random.default_rng(22)
    sampled_rows = []
    for _, _, indices in _cell_indices(task,fold):
        draws = repeated_rng.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=300)
        sampled_rows.extend(np.repeat(indices,draws[0]).tolist())
    sampled_rows = np.asarray(sampled_rows)
    direct = matched_point_estimates(gates[:,sampled_rows],D[sampled_rows],task[sampled_rows],sub[sampled_rows],fold[sampled_rows])
    for name in result['bootstrap_draws']:
        assert np.max(abs(direct[name]-result['bootstrap_draws'][name][0]))<1e-12
    assert all(row['draw_size_fixed'] for row in result['metadata']['resampling_checks'])
    assert result['metadata']['independent_questions']==48
    return dict(status='PASS',checks=['original_record_resampling','stratum_counts_fixed',
        'all_three_match_counts_recomputed','all_generation_repeats_shared','zero_and_all_gates_match_exactly',
        'paired_correction_harm_identity','direct_duplicate_record_matching_equivalence'],maximum_identity_error=float(np.max(abs(result['bootstrap_draws']['gain']-result['bootstrap_draws']['correction']+result['bootstrap_draws']['harm']))),new_llm_calls=0)


def _native(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:_native(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_native(item) for item in value]
    return value


def _compact_statistics(value):
    return {name:item for name,item in value.items() if not name.startswith('per_question_')}


def _format_interval(item, name='ci95'):
    bounds=item[name]
    return f"{100*item['point']:+.4f} [{100*bounds[0]:+.4f}, {100*bounds[1]:+.4f}]"


def _one_rule_interval(summary,index):
    return dict(point=float(summary['point'][index]),ci95=summary['ci95'][index].tolist(),
        ci97_5_bonferroni_two_comparisons=summary['ci97_5_bonferroni_two_comparisons'][index].tolist(),
        centered_bootstrap_two_sided_p=float(summary['centered_bootstrap_two_sided_p'][index]),
        p_interval_note=summary['p_interval_note'])


def report_text(result):
    table=result['table']
    primary=result['primary_comparisons']
    gate=result['real_allocation_gate']
    lines=['# 03 有益切换排序与同覆盖率控制','',
        '## 研究问题','',
        '在同一个F区默认组合和同一个P2候选推荐器下，换队评分能否在未参与当前拟合的问题上富集正配对收益，尤其能否超过任务内部同覆盖率随机选择。','',
        '## 实验设置','',
        '使用train600原题的五折开发诊断。每折F360训练默认与P2推荐器，G120学习概率门控和阈值，E120仅供冻结后的评价。三个生成重复始终跟随同一题，不扩大独立题数。主输入只用文本，默认由F区全局平均奖励选择，候选为P2在其余19组合中预测最高者。','',
        '概率头使用两维原始P2预测值，分别为默认与候选价值。r估计正确性分歧概率，u在D非零记录中估计正方向。r的每道题三重复权重各1/3，u过滤D非零时保留原1/3，因此每题u累计权重等于其经验分歧频率。各头C仅从0.1、1、10内在G题级三折中选择，G分数采用嵌套题级OOF，E使用full-G拟合头。缺类以Beta(1,1)加权常数平滑回退。','',
        'G OOF头每次看到80题，E头使用120题，二者有明确的stacking拟合规模差异。阈值与规则在读取E反馈前持久冻结。G区每类评分固定四个目标覆盖率分位点10%、25%、50%、75%与数值零点，严格接受score>threshold。主阈值在五规则中取G配对收益最大者，数值1e-12内并列时依次偏好较低覆盖、更高阈值和较低索引。','',
        '方向消融使用2u−1，保留u原概率，方向零点对应u=.5。仅分歧r的零点只表示数值score0，方向性未由该分数给出。每折完整非默认家族保守计J=20，默认策略精确收益0。各外折的主阈值可以不同。','',
        '主指标为GSM8K和BBH各50%，BBH内部按题平均。27个BBH子任务宏平均单列。真实任务标签只用于分层、诊断与匹配，同权限元数据补充由METADATA_PERMISSION_CONTROLS.json单独记录，并保持在Text-only主门控及继续门槛之外。','',
        '## 对照与统计方法','',
        '三个随机控制分别在每个外折内固定总接受数、各任务接受数、各BBH子任务接受数，只改变哪些题接受，候选组合始终不变。三生成重复共用一个题级决定。点估计使用有限池均匀抽样的解析期望，1000次预定随机化另外描述条件控制分布，不计作新增题。','',
        '置信区间以outerfold×GSM/BBH为分层单位，重新抽样完整原题记录。每次重新计算匹配组大小和接受数，包括可能缺失的BBH子任务，未把原样本接受概率固定到所有bootstrap中。五折600个不同E题支持5000次配对bootstrap。模型与阈值保持冻结，区间条件于这批缓存及历史拟合，未重新拟合相互重叠的F/G历史。','',
        '预先指定的主比较只有factorized主门控相对默认、相对任务内同覆盖率随机两项。开发继续标准为两项Bonferroni97.5%区间下界均严格大于1e-12，另保留95%描述区间及Holm校正的centered-bootstrap p。分位区间与centered p在离散有限样本下不必互逆，使用已冻结的区间标准分流。','',
        '## 数值结果','',
        '### 四类冻结主门控','',
        '|评分|E切换覆盖率|平均纠错题数|平均破坏题数|总体净收益及95%区间 pp|相对任务内随机收益及95%区间 pp|',
        '|---|---:|---:|---:|---:|---:|']
    for score in ('gap','factorized','disagreement','direction'):
        row=table['main/'+score]
        lines.append(f"|{score}|{100*row['coverage']:.3f}%|{row['expected_corrected_questions']:.3f}|{row['expected_harmed_questions']:.3f}|{_format_interval(row['gain_interval'])}|{_format_interval(row['matched']['task']['excess_interval'])}|")
    lines.extend(['','纠错与破坏为三个生成重复平均后的题数，两个方向可在同一题的不同重复中同时出现。总体净收益的分母始终是全部600题，切换区域内条件收益在机器结果中另列。','',
        '### 两项预先指定比较','',
        '|比较|差值与95%区间 pp|Bonferroni97.5%区间 pp|Holm调整p（开发描述）|',
        '|---|---:|---:|---:|'])
    for name,row in primary.items():
        lines.append(f"|{name}|{_format_interval(row)}|{_format_interval(row,'ci97_5_bonferroni_two_comparisons')}|{row['holm_adjusted_centered_bootstrap_p']:.6f}|")
    lines.extend(['',f"按预设继续标准，本阶段判定为 **{gate['status']}**。这项分流属于开发证据判断，正式配对认证的定理及功效另见04。",'',
        '### 配对分歧—方向恒等式','',
        '|主门控|kappa|rho|eta|G=rho(2eta−1)绝对误差|切换区条件收益|',
        '|---|---:|---:|---:|---:|---:|'])
    for score in ('gap','factorized','disagreement','direction'):
        row=table['main/'+score]
        eta='未定义' if row['eta'] is None else f"{row['eta']:.6f}"
        conditional='未定义' if row['gain_among_switched'] is None else f"{100*row['gain_among_switched']:+.4f} pp"
        lines.append(f"|{score}|{row['coverage']:.6f}|{row['rho']:.6f}|{eta}|{row['identity_error']:.3e}|{conditional}|")
    lines.extend(['','rho使用同一题各重复的D平方平均，再乘门控，体现正确性分歧。双资源分配中的a_h使用题级平均D的平方，两者在L=3时需要区分。','',
        '### 分任务与生成重复方向','',
        '|主门控|GSM净收益 pp|BBH净收益 pp|BBH27宏净收益 pp|生成42 pp|生成137 pp|生成2024 pp|',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for score in ('gap','factorized','disagreement','direction'):
        row=table['main/'+score]
        lines.append(f"|{score}|{100*row['by_task']['gsm8k']['gain']:+.4f}|{100*row['by_task']['bbh']['gain']:+.4f}|{100*row['bbh_subtask_macro_gain']:+.4f}|"+'|'.join(f"{100*x:+.4f}" for x in row['per_generation_gain'])+'|')
    lines.extend(['','### G区方向支持','',
        '|外折|r有分歧独立题|u正向支持题|u负向支持题|u有效独立题|u总权重|r C|u C|',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for item in result['fold_diagnostics']:
        r,u=item['r_support'],item['u_support']
        lines.append(f"|{item['fold']}|{r['positive_question_support']}|{u['positive_question_support']}|{u['negative_question_support']}|{u['active_original_questions']}|{u['total_training_weight']:.4f}|{item['r_C']}|{item['u_C']}|")
    lines.extend(['','五个G区互有历史重叠，表内支持数不能直接加成600道独立门控训练题。正负方向支持也可在同一题重叠。','',
        '### 预固定覆盖率曲线','',
        '|评分|阈值索引|E覆盖率|净收益 pp|对global随机 pp|对task随机 pp|对subtask随机 pp|',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for score in ('gap','factorized','disagreement','direction'):
        for index in range(5):
            row=table[f'curve/{score}/{index}']
            lines.append(f"|{score}|{index}|{100*row['coverage']:.3f}%|{100*row['gain']:+.4f}|{100*row['matched']['global']['observed_minus_expected_random_gain']:+.4f}|{100*row['matched']['task']['observed_minus_expected_random_gain']:+.4f}|{100*row['matched']['subtask']['observed_minus_expected_random_gain']:+.4f}|")
    lines.extend(['','曲线用于诊断评分排序，外折事后最高点保持探索性角色，没有替换G区冻结主阈值。各门控的三个匹配控制条件分布、MC误差、分任务结果和置信区间完整保存在JSON与private文件。','',
        '## 统计结论','',
        result['interpretation'],'',
        '完整r零分数门控的E覆盖率为100%，因而逐点支配其余门控。对全部20规则求最坏二阶矩时，目标会由该全切换规则决定。不同评分拼接的完整家族仍有这种退化，双预算阶段需要显式记录。四个固定50%分位规则的敏感性家族另检查支配关系。','',
        '## 失败原因与风险','',
        '- 该实验检验当前P2候选和两维价值特征所支持的排序能力，无法据此排除其他候选或更丰富输入的价值。候选和门控的贡献在本设计中已经分开。',
        '- 少量G区有分歧题与生成重复的正负方向变化限制概率头估计，平滑回退和条件似然权重已经逐折保存。',
        '- G OOF与full-G评分存在拟合规模差异，开发阈值选择与既往模型选择影响仍在。outerfold×task区间只对冻结程序作条件诊断。',
        '- 子任务内匹配可使小组中0或全接受的部分完全固定，识别同子任务排序的有效随机化空间有限。',
        '- 优势是同一候选、同一配对反馈、同一原题覆盖与重复权重，能够区分门控排序和任务构成。主要风险是正收益稀疏与开发数据复用，真实总体泛化需要独立证据。','',
        '## 下一步','',
        '真实双预算回放仅依照REAL_ALLOCATION_GATE的预设分流执行。理论合成与认证功效分析可以继续，新的语言模型生成调用保持为0。收益和区间尚未支持主门控时，保留正区域及失效信息，优先解释排序局限。','',
        '## 产物','',
        '- SWITCH_RANKING_RESULTS.json记录所有四主门控、20预固定曲线、两主比较与匹配控制。',
        '- REAL_ALLOCATION_GATE.json绑定主结果SHA256与下一阶段判断。',
        '- private/switching内保存原题配对数组、bootstrap draws、随机化分布及来源哈希。',''])
    return '\n'.join(lines)


def run_summary(output=None,private_output=None,reps=5000,random_reps=1000,seed=20261110):
    from pathlib import Path
    import hashlib,json,time
    from e5_common import PUBLIC,PRIVATE,load_train,put,sha
    from gate_models import SCORE_NAMES,gate_statistics,matched_controls
    output=PUBLIC if output is None else Path(output)
    private_output=PRIVATE/'switching' if private_output is None else Path(private_output)
    output.mkdir(parents=True,exist_ok=True)
    private_output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    data,_=load_train()
    all_masks=np.zeros((4,5,600),dtype=bool)
    main_masks=np.zeros((4,600),dtype=bool)
    D=np.zeros((600,3))
    default_rewards=np.zeros((600,3))
    folds=np.full(600,-1,dtype=int)
    candidate=np.zeros(600,dtype=int)
    default=np.zeros(600,dtype=int)
    counts=np.zeros(600,dtype=int)
    descriptions=[]
    source={}
    for f in range(5):
        path=PRIVATE/'gates'/f'GATES_f{f}.npz'
        with np.load(path) as a:
            E=a['E']
            assert len(E)==120 and len(np.unique(E))==120
            assert a['gate_masks_E'].shape==(4,5,120) and a['D_E'].shape==(120,3)
            with np.load(PRIVATE/'gates'/f'FROZEN_DECISIONS_f{f}.npz') as frozen_decisions:
                for name in ('E','gate_masks_E','main_indices','base','candidate_E'):
                    assert np.array_equal(a[name],frozen_decisions[name])
            b=int(a['base'])
            c=a['candidate_E']
            expected=data['Y'][E[:,None],np.arange(3)[None,:],c[:,None]]-data['Y'][E,:,b]
            assert np.array_equal(expected,a['D_E'])
            assert np.array_equal(data['Y'][E,:,b],a['Rb_E'])
            all_masks[:,:,E]=a['gate_masks_E']
            for score_index in range(4):main_masks[score_index,E]=a['gate_masks_E'][score_index,int(a['main_indices'][score_index])]
            D[E]=a['D_E']
            default_rewards[E]=a['Rb_E']
            folds[E]=f
            default[E]=b
            candidate[E]=c
            counts[E]+=1
        metadata=json.loads((PRIVATE/'gates'/f'FROZEN_GATE_f{f}.json').read_text(encoding='utf-8'))
        assert metadata['score_names']==list(SCORE_NAMES)
        assert metadata['E_outcome_used_for_freeze'] is False
        receipt=json.loads((PRIVATE/'gates'/f'GATES_f{f}.json').read_text(encoding='utf-8'))
        assert receipt['eval_after_freeze'] and receipt['status']=='COMPLETE'
        heads=metadata['fitted_heads']
        r_support={key:value for key,value in heads['r']['fit_info']['support'].items() if key!='per_question_training_weight'}
        u_support={key:value for key,value in heads['u']['fit_info']['support'].items() if key!='per_question_training_weight'}
        descriptions.append(dict(fold=f,main_indices=metadata['frozen']['main_indices'],
            thresholds=metadata['frozen']['thresholds'],
            r_support=r_support,u_support=u_support,
            r_C=heads['r']['C'],u_C=heads['u']['C'],
            unique_score_threshold_pairs=metadata['frozen']['unique_score_threshold_pairs'],
            unique_G_gate_patterns=metadata['frozen']['unique_G_gate_patterns']))
        source[f'GATES_f{f}.npz']=sha(path)
        source[f'FROZEN_GATE_f{f}.json']=sha(PRIVATE/'gates'/f'FROZEN_GATE_f{f}.json')
        source[f'FROZEN_DECISIONS_f{f}.npz']=sha(PRIVATE/'gates'/f'FROZEN_DECISIONS_f{f}.npz')
    assert np.all(counts==1) and np.all(folds>=0)
    assert np.all(all_masks[SCORE_NAMES.index('disagreement'),4])
    labels=['main/'+name for name in SCORE_NAMES]+[f'curve/{name}/{j}' for name in SCORE_NAMES for j in range(5)]
    gates=np.concatenate([main_masks,all_masks.reshape(20,600)],axis=0)
    boot=question_record_bootstrap(gates,D,data['task'],data['sub'],folds,reps=reps,seed=seed)
    private_MC={}
    table={}
    primary_weights=np.where(data['task']=='gsm8k',.5/np.sum(data['task']=='gsm8k'),.5/np.sum(data['task']=='bbh'))
    default_accuracy=float(primary_weights@default_rewards.mean(1))
    for index,(label,gate) in enumerate(zip(labels,gates)):
        row=_compact_statistics(gate_statistics(gate,D,data['task'],data['sub']))
        row.update(rule_key=label,type='primary_threshold' if label.startswith('main/') else 'fixed_curve',
                   final_accuracy=default_accuracy+row['gain'],default_accuracy=default_accuracy,
                   gain_interval=_one_rule_interval(boot['summaries']['gain'],index),
                   coverage_interval=_one_rule_interval(boot['summaries']['coverage'],index),matched={})
        MC_by_scheme={scheme:[] for scheme in ('global','task','subtask')}
        exact_by_scheme={scheme:[] for scheme in MC_by_scheme}
        randomization_details=[]
        for f in range(5):
            selected=np.flatnonzero(folds==f)
            controls=matched_controls(gate[selected],D[selected],data['task'][selected],data['sub'][selected],
                reps=random_reps,seed=seed+1000*index+17*f)
            for scheme,control in controls.items():
                MC_by_scheme[scheme].append(control['randomization_gains'])
                exact_by_scheme[scheme].append(control['exact_expected_random_gain'])
                private_MC[f'{index}_{f}_{scheme}_gain']=control['randomization_gains']
                private_MC[f'{index}_{f}_{scheme}_gain_by_generation']=control['randomization_gain_by_generation']
                randomization_details.append(dict(fold=f,scheme=scheme,seed=control['seed'],groups=control['groups']))
        for scheme,draws in MC_by_scheme.items():
            draw=np.mean(draws,axis=0)
            exact=float(boot['point']['random_'+scheme+'_gain'][index])
            assert abs(np.mean(exact_by_scheme[scheme])-exact)<1e-12
            row['matched'][scheme]=dict(exact_expected_random_gain=exact,
                observed_minus_expected_random_gain=row['gain']-exact,
                excess_interval=_one_rule_interval(boot['summaries']['excess_'+scheme],index),
                randomization_mean=float(draw.mean()),randomization_distribution_quantile95=np.quantile(draw,[.025,.975]).tolist(),
                randomization_mean_MC_standard_error=float(draw.std(ddof=1)/np.sqrt(random_reps)),
                one_sided_randomization_tail_probability=float((1+np.sum(draw>=row['gain']-1e-15))/(random_reps+1)))
        row['randomization_details']=randomization_details
        table[label]=row
    p_index=labels.index('main/factorized')
    primary=dict(factorized_minus_default=_one_rule_interval(boot['summaries']['gain'],p_index),
                 factorized_minus_task_matched=_one_rule_interval(boot['summaries']['excess_task'],p_index))
    adjusted=_holm({key:row['centered_bootstrap_two_sided_p'] for key,row in primary.items()})
    for key,row in primary.items():row['holm_adjusted_centered_bootstrap_p']=adjusted[key]
    passed=all(row['ci97_5_bonferroni_two_comparisons'][0]>1e-12 for row in primary.values())
    continue_gate=dict(status='PASSED' if passed else 'NOT_PASSED',passed=passed,
        criterion='both primary conditional-development Bonferroni97.5% interval lower bounds > 1e-12',
        primary_comparisons=primary, independent_questions=600,primary_method='factorized',
        scope='development mechanism gate, no independent fresh400 confirmation or theorem certification claim')
    pos=D.mean(axis=1)>0
    empirical_opportunity=dict(positive_paired_mean_questions=int(pos.sum()),negative_paired_mean_questions=int(np.sum(D.mean(1)<0)),
        zero_paired_mean_questions=int(np.sum(D.mean(1)==0)),
        empirical_three_repeat_positive_region_gain=float(primary_weights@np.maximum(D.mean(1),0)),
        qualifier='finite-repeat hindsight paired region, not population or input-learnable upper bound')
    pg=table['main/factorized']['gain']
    pe=table['main/factorized']['matched']['task']['observed_minus_expected_random_gain']
    if passed:
        interpretation='因子化冻结主门控通过两项预设开发区间门槛，支持继续有限矩阵上的双预算机制检验。该结论仍条件于当前开发数据和已选择预测器。'
    elif pg>0 and pe>0:
        interpretation='因子化主门控的净收益和任务内匹配超额收益点值均正，预设联合区间门槛仍未通过。当前证据不足以扩大真实双预算回放。'
    elif pg>0:
        interpretation='因子化主门控有正净收益点值，任务内匹配后未保留正超额收益。任务构成差异需要与逐题排序能力分开，当前机制关口未通过。'
    else:
        interpretation='因子化冻结主门控未得到正总体净收益，预设机制关口未通过。其他曲线的局部正值仍按开发诊断记录，保持主比较和失败分流。'
    result=dict(schema='e5_switch_ranking_v1',status='COMPLETE',table=table,primary_comparisons=primary,
        real_allocation_gate=continue_gate,interpretation=interpretation,empirical_opportunity=empirical_opportunity,
        fold_diagnostics=descriptions,primary_method='factorized',independent_original_questions=600,
        generations=list(map(int,(42,137,2024))),generation_repeats=3,gate_family_per_fold=20,
        frozen_primary_procedures=4,diagnostic_curve_procedures=20,
        family_note='Each outer fold freezes twenty indexed rules; cross-fold primary threshold choices may differ. Pooled OOF evaluation is not a single iid certification set.',
        metadata=boot['metadata'],randomization_repeats=random_reps,new_llm_calls=0,
        source=source,summary_source_sha256=sha(__file__),wall_s=time.perf_counter()-started)
    result['primary_table']= {name:table['main/'+name] for name in SCORE_NAMES}
    np.savez_compressed(private_output/'SWITCH_PAIRED.npz',gates=gates,D=D,fold=folds,task=data['task'],sub=data['sub'],
        default_rewards=default_rewards,default=default,candidate=candidate,rule_names=np.asarray(labels))
    np.savez_compressed(private_output/'SWITCH_BOOTSTRAP.npz',**boot['bootstrap_draws'])
    np.savez_compressed(private_output/'SWITCH_MATCHED_RANDOMIZATIONS.npz',**private_MC)
    put(output/'SWITCH_RANKING_RESULTS.json',_native(result))
    continue_gate['switch_results_sha256']=sha(output/'SWITCH_RANKING_RESULTS.json')
    put(output/'REAL_ALLOCATION_GATE.json',_native(continue_gate))
    (output/'03_SWITCH_RANKING_AND_MATCHED_CONTROLS.md').write_text(report_text(result),encoding='utf-8')
    print(json.dumps(_native(dict(status='COMPLETE',gate=continue_gate['status'],primary=primary,
        empirical_opportunity=empirical_opportunity,wall_s=time.perf_counter()-started)),ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument('--qualify',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--private-output',type=Path)
    parser.add_argument('--reps',type=int,default=5000)
    parser.add_argument('--random-reps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20261110)
    args=parser.parse_args()
    if args.qualify:print(json.dumps(qualification(),indent=2))
    else:run_summary(args.output,args.private_output,args.reps,args.random_reps,args.seed)
