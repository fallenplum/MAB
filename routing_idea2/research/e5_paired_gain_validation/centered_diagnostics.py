"""E5-B centered error decomposition, observed cross-repeat risks and routing."""
from e5_common import *
from theory_checks import centering_check,C
from collections import defaultdict
from time import perf_counter

NAMES=('P1','P2','P3_Probit')
PAIRS=np.asarray(list(__import__('itertools').combinations(range(20),2)))


def fold_task_paired(diff,fold,task,reps=5000,seed=20261110):
    diff=np.asarray(diff,float)
    if diff.ndim==1:diff=diff[:,None]
    assert np.isfinite(diff).all() and len(diff)==600
    rng=np.random.default_rng(seed)
    draws=np.zeros((reps,diff.shape[1]))
    point=np.zeros(diff.shape[1])
    for domain in ('gsm8k','bbh'):
        total=int(np.sum(task==domain))
        for f in range(5):
            mask=(task==domain)&(fold==f)
            values=diff[mask]
            assert len(values)==60
            weights=rng.multinomial(len(values),np.ones(len(values))/len(values),size=reps)
            draws+=.5*(weights@values)/total
            point+=.5*values.sum(0)/total
    bounds=np.quantile(draws,[.025,.975],axis=0)
    centered_null=draws-point
    p=(1+np.sum(abs(centered_null)>=abs(point)-1e-15,axis=0))/(reps+1)
    return dict(difference=point.tolist(),ci95=np.stack(bounds,axis=-1).tolist(),
        descriptive_centered_bootstrap_p=p.tolist(),question_support=600,bootstrap_reps=reps,seed=seed,
        strata='outer-fold by broad-task, original questions resampled with their complete reward blocks',
        scope='Conditional on frozen folds, historical fitting and the cached three generation repeats',
        selection_uncertainty_included=False)


def model_metrics(pred,rewards,default):
    pred,rewards=np.asarray(pred,float),np.asarray(rewards,float)
    n,L,M=rewards.shape
    observed_mean=rewards.mean(1)
    error=pred-observed_mean
    centered=error-error.mean(1,keepdims=True)
    absolute=(error**2).mean(1)
    perpendicular=(centered**2).mean(1)
    offset=error.mean(1)**2
    decomposition_error=float(np.max(abs(absolute-perpendicular-offset)))
    assert decomposition_error<1e-12
    centered_single=pred[:,None,:]-rewards
    centered_single-=centered_single.mean(2,keepdims=True)
    ordered_sum=np.zeros(n)
    for a in range(L):
        for b in range(L):
            if a!=b:ordered_sum+=np.sum(centered_single[:,a]*centered_single[:,b],axis=1)
    cross_repeat=ordered_sum/(M*L*(L-1))
    deviations=rewards-observed_mean[:,None,:]
    deviations-=deviations.mean(2,keepdims=True)
    correction=np.sum(deviations**2,axis=(1,2))/(M*L*(L-1))
    cross_identity_error=float(np.max(abs(cross_repeat-(perpendicular-correction))))
    assert cross_identity_error<1e-12
    chosen=argmax(pred)
    route=rewards[np.arange(n)[:,None],np.arange(L)[None,:],chosen[:,None]]
    base=rewards[np.arange(n)[:,None],np.arange(L)[None,:],np.asarray(default)[:,None]]
    gain=(route-base).mean(1)
    correct=((route==1)&(base==0)).mean(1)
    harm=((route==0)&(base==1)).mean(1)
    assert np.max(abs(gain-correct+harm))==0
    metrics=prediction_metrics(pred,rewards,prediction_tie_tolerance=TOLERANCE)['per_question']
    clipped=np.clip(pred,0,1)
    clip_error=clipped-observed_mean
    clip_center=clip_error-clip_error.mean(1,keepdims=True)
    brier_center=(clip_center**2).mean(1)
    brier_offset=clip_error.mean(1)**2
    noise=((rewards-observed_mean[:,None,:])**2).mean((1,2))
    brier=((clipped[:,None,:]-rewards)**2).mean((1,2))
    assert np.max(abs(brier-brier_center-brier_offset-noise))<1e-12
    values=dict(absolute_mean_mse=absolute,centered_mean_mse=perpendicular,common_offset_mse=offset,
        cross_repeat_T=cross_repeat,observed_center_noise_correction=correction,
        brier_clipped=brier,brier_centered_component=brier_center,brier_offset_component=brier_offset,
        observation_noise_mean=noise,accuracy=route.mean(1),gain_vs_default=gain,correct=correct,harm=harm,
        retrospective_generation_gap=(rewards.max(2)-route).mean(1),
        retrospective_mean_gap=observed_mean.max(1)-route.mean(1),
        non_tie_binary_pair_ranking=metrics['non_tie_rank_accuracy'],
        non_tie_mean_reward_ranking=metrics['mean_reward_rank_accuracy'],
        raw_value_outside_01_rate=((pred<0)|(pred>1)).mean(1),
        prediction_changed_from_default=(chosen!=np.asarray(default)).astype(float))
    top_gap=np.sort(pred,axis=1)[:,-1]-np.sort(pred,axis=1)[:,-2]
    return values,dict(decomposition_error=decomposition_error,cross_repeat_identity_error=cross_identity_error,
        cross_repeat_negative_questions=int(np.sum(cross_repeat<0)),minimum_cross_repeat_T=float(cross_repeat.min()),
        near_tie_questions=int(np.sum(top_gap<=TOLERANCE)),minimum_top_score_gap=float(top_gap.min()),
        native_argmax_vs_declared_tolerance_mismatches=int(np.sum(np.argmax(pred,axis=1)!=chosen))),route,chosen


def single_repeat_error(pred,reward):
    error=pred-reward
    return dict(observed_absolute_mse=(error**2).mean(1),
        observed_centered_mse=((error-error.mean(1,keepdims=True))**2).mean(1),
        observed_offset_mse=error.mean(1)**2)


def scope_table(values,data):
    return {name:scope_means(value,data['task'],data['sub']) for name,value in values.items()}


def main():
    start=perf_counter()
    data,_=load_train()
    folds=np.full(600,-1,dtype=int)
    default=np.zeros(600,dtype=int)
    before=np.empty((3,600,20))
    after=np.empty((3,3,600,20))
    train_rows=[]
    center_rows=[]
    probit_receipts=[]
    sources={}
    for f in range(5):
        path=PRIVATE/'base'/f'BASE_f{f}.npz'
        metadata=json.loads(path.with_suffix('.json').read_text())
        for name in ('e5_common.py','run_base.py','theory_checks.py','E5_EXPERIMENT_CONFIG.json'):
            assert metadata['source'][name]==sha(PUBLIC/name), 'Frozen base source changed: '+name
        assert metadata['source']['E4_Gaussian']==sha(E4/'e4_gaussian.py')
        assert metadata['source']['E4_Probit']==sha(E4/'e4_binary.py')
        probit_receipts.append(dict(fold=f,**metadata['Probit_fit']))
        a=np.load(path)
        F,G,E=a['F'],a['G'],a['E']
        assert len(F)==360 and len(G)==len(E)==120
        assert not set(F)&set(G) and not set(F)&set(E) and not set(G)&set(E)
        assert set(a['feedback_G20'])<=set(G)
        assert np.all(folds[E]==-1)
        folds[E]=f
        default[E]=int(a['base'])
        before[:,E]=a['pred_E']
        after[:,:,E]=a['updated_pred_E']
        sources[path.name]=sha(path)
        sources[path.with_suffix('.json').name]=sha(path.with_suffix('.json'))
        replay=centering_check(a['X_F'],data['Y'][F].mean(1),a['X_E'],100.,
            question_weights=np.ones(len(F))*3,eval_rewards=data['Y'][E],case_id=f'fold{f}_real360_120_independent_replay')
        coefficient=np.linalg.solve(3*a['X_F'].T@a['X_F']+100*np.eye(17),3*a['X_F'].T@data['Y'][F].mean(1))
        replay['frozen_P2_prediction_replay_max_error']=float(np.max(abs(a['X_E']@coefficient-a['pred_E'][1])))
        assert replay['frozen_P2_prediction_replay_max_error']<1e-10
        assert metadata['centering_check']['declared_lexicographic_selection_mismatches']==0
        center_rows.append(replay)
        for m,name in enumerate(NAMES):
            # Training-role metrics are descriptive. These 5x360 rows overlap across folds.
            v,check,route,chosen=model_metrics(a['pred_F'][m],data['Y'][F],np.repeat(int(a['base']),len(F)))
            train_rows.append(dict(fold=f,model=name,role='F-training only',questions=360,
                metrics={key:scope_means(value,data['task'][F],data['sub'][F]) for key,value in v.items()}))
    assert np.all(folds>=0)
    raw={}
    audits={}
    route_before={}
    route_after={}
    for m,name in enumerate(NAMES):
        v,check,r,chosen=model_metrics(before[m],data['Y'],default)
        raw[name]=v
        audits[name]=check
        route_before[name]=r
        all_values=[]
        all_routes=[]
        all_chosen=[]
        for source in range(3):
            values,check,rr,cc=model_metrics(after[source,m],data['Y'],default)
            all_values.append(values)
            all_routes.append(rr)
            all_chosen.append(cc)
            audits[f'{name}_updated_source{SEEDS[source]}']=check
        raw[name+'_updated']={key:np.mean([v[key] for v in all_values],axis=0) for key in all_values[0]}
        route_after[name]=np.asarray(all_routes)
        raw[name+'_updated']['update_correct']=np.mean((route_after[name]==1)&(r[None]==0),axis=(0,2))
        raw[name+'_updated']['update_harm']=np.mean((route_after[name]==0)&(r[None]==1),axis=(0,2))
        raw[name+'_updated']['update_changed_action_rate']=np.mean(np.asarray(all_chosen)!=chosen[None],axis=0)
    comparisons={}
    comparisons_spec=[('P1_to_P2','P1','P2'),('P2_to_Probit','P2','P3_Probit')]
    comparisons_spec += [(name+'_feedback_before_to_after',name,name+'_updated') for name in NAMES]
    paired_keys=['absolute_mean_mse','centered_mean_mse','common_offset_mse','cross_repeat_T','accuracy',
                 'brier_clipped','brier_centered_component','brier_offset_component']
    vectors=[]
    for label,old,new in comparisons_spec:
        delta={key:raw[new][key]-raw[old][key] for key in paired_keys}
        assert np.max(abs(delta['absolute_mean_mse']-delta['centered_mean_mse']-delta['common_offset_mse']))<1e-12
        assert np.max(abs(delta['cross_repeat_T']-delta['centered_mean_mse']))<1e-12
        assert np.max(abs(delta['brier_clipped']-delta['brier_centered_component']-delta['brier_offset_component']))<1e-12
        for key in paired_keys:vectors.append(delta[key])
        comparisons[label]=dict(old=old,new=new,delta=scope_table(delta,data),
            cross_repeat_difference_identity_max_error=float(np.max(abs(delta['cross_repeat_T']-delta['centered_mean_mse']))))
    combined=fold_task_paired(np.asarray(vectors).T,folds,data['task'])
    index=0
    for label,old,new in comparisons_spec:
        entry=comparisons[label]
        entry['primary_fold_task_paired']={}
        for key in paired_keys:
            entry['primary_fold_task_paired'][key]=dict(difference=combined['difference'][index],
                ci95=combined['ci95'][index],descriptive_centered_bootstrap_p=combined['descriptive_centered_bootstrap_p'][index])
            index+=1
        absolute=entry['delta']['absolute_mean_mse']['balanced_v2']
        offset=entry['delta']['common_offset_mse']['balanced_v2']
        entry['offset_fraction_of_absolute_improvement']=offset/absolute if absolute < -1e-12 else None
        brier=entry['delta']['brier_clipped']['balanced_v2']
        brier_offset=entry['delta']['brier_offset_component']['balanced_v2']
        entry['offset_fraction_of_clipped_brier_improvement']=brier_offset/brier if brier < -1e-12 else None
        entry['secondary_BBH_macro_and_task_cluster_sensitivity']=paired_summary(raw[new]['centered_mean_mse']-raw[old]['centered_mean_mse'],data['task'],data['sub'])
    table={key:dict(metrics=scope_table(values,data),negative_cross_repeat_questions=int(np.sum(values['cross_repeat_T']<0)),
        minimum_cross_repeat_T=float(values['cross_repeat_T'].min()),independent_query_questions=600,
        per_task={task:{metric:float(np.nanmean(value[semantic(data)==task])) if np.isfinite(value[semantic(data)==task]).any() else None for metric,value in values.items()} for task in sorted(set(semantic(data)))}) for key,values in raw.items()}
    seed_results={}
    for m,name in enumerate(NAMES):
        base=route_before[name]
        observed_before=[single_repeat_error(before[m],data['Y'][:,g]) for g in range(3)]
        observed_after=[[single_repeat_error(after[s,m],data['Y'][:,g]) for g in range(3)] for s in range(3)]
        seed_results[name]=dict(before_accuracy={str(SEEDS[g]):scope_means(base[:,g],data['task'],data['sub']) for g in range(3)},
            after_accuracy_by_support_query_seed=[[scope_means(route_after[name][s,:,g],data['task'],data['sub']) for g in range(3)] for s in range(3)],
            gain_by_support_query_seed=[[scope_means(route_after[name][s,:,g]-base[:,g],data['task'],data['sub']) for g in range(3)] for s in range(3)],
            observed_error_before_by_query_seed={str(SEEDS[g]):scope_table(observed_before[g],data) for g in range(3)},
            observed_error_after_by_support_query_seed=[[scope_table(observed_after[s][g],data) for g in range(3)] for s in range(3)],
            observed_error_change_by_support_query_seed=[[scope_table({k:observed_after[s][g][k]-observed_before[g][k] for k in observed_before[g]},data) for g in range(3)] for s in range(3)],
            seed_error_scope='Single-repeat observed loss, distinct from the three-repeat-mean target in the primary table')
    output=dict(schema='e5_centered_diagnostics_v1',status='COMPLETE',models=list(NAMES),
        code_sha256=sha(Path(__file__)),configuration_sha256=sha(PUBLIC/'E5_EXPERIMENT_CONFIG.json'),
        independent_E_questions=600,generation_repeats=3,source_feedback_paths=3,
        primary_weight='GSM8K .5, BBH question-average .5',secondary_weight='BBH27 macro .5',
        model_table=table,comparisons=comparisons,seed_results=seed_results,training_role_metrics=train_rows,
        frozen_Probit_fit_receipts=probit_receipts,
        algebra_audits=audits,actual_centering_checks=center_rows,input_sha256=sources,
        bootstrap={k:v for k,v in combined.items() if k not in ('difference','ci95','descriptive_centered_bootstrap_p')},
        multiplicity='All B diagnostics are descriptive unadjusted intervals. They are distinct from the two prespecified C primary comparisons.',
        cross_repeat_interpretation='Algebraic identity verified. Recorded distinct seeds and common frozen versions do not establish conditional independence. Ti is not asserted to be an identified unbiased absolute risk; paired centered-loss differences are primary.',
        source_protocol='Three member-generation seeds42/137/2024, frozen V2 aggregator seed42; E4 execution-identity audit supports different runs, not conditional iid.',
        raw_gaussian_probability_note='Absolute diagnostics use original raw Ridge values. Clipped Brier decomposition is separate and never changes the service action.',
        wall_s=perf_counter()-start,new_llm_calls=0)
    np.savez_compressed(PRIVATE/'CENTERED_DIAGNOSTICS_PAIRED.npz',fold=folds,
        **{name+'__'+metric:value for name,values in raw.items() for metric,value in values.items()})
    output['private_paired_sha256']=sha(PRIVATE/'CENTERED_DIAGNOSTICS_PAIRED.npz')
    put(PUBLIC/'CENTERED_DIAGNOSTICS_RESULTS.json',output)
    put(PUBLIC/'ACTUAL_CENTERING_RESULTS.json',dict(status='PASS',folds=center_rows,new_llm_calls=0))
    write_report(output)
    append_actual_A(center_rows)
    print(json.dumps(dict(status='COMPLETE',independent_E_questions=600,model_contexts=len(table),
                         algebra_decomposition_error=max(x['decomposition_error'] for x in audits.values()),
                         real_centering_folds=5,wall_s=output['wall_s'])),flush=True)


def format_ci(item,scale=1):
    return f"{item['difference']*scale:+.6f} [{item['ci95'][0]*scale:+.6f}, {item['ci95'][1]*scale:+.6f}]"


def write_report(d):
    lines=['# 02 中心化误差与真实选队诊断','',
        '## 研究问题','',
        '价值预测改善来自共同题目难度偏移，还是组合之间的相对误差。中心化风险、概率预测与最终路由分别评价。','',
        '## 实验设置与对照','',
        '沿用五折，每折F360、G120、E120。P1/P2原Ridge惩罚100，P3沿用E4文本Probit固定惩罚10。五个外折覆盖600道不同评价题，三个生成重复属于同一题。主部署输入只有冻结文本向量。默认组合由F区全局均值选择，任务及BBH子任务只用于离线分组。','',
        '反馈诊断采用G区预先固定20题、同一P2默认/候选组合对。三种Support生成种子分别更新基础模型，在同一E区三生成评分上评价。主结果先按原题平均三条反馈路径和三个Query生成重复，不增加独立题量。','',
        '## 数值结果','',
        '|模型/阶段|绝对均值误差|中心化误差|共同偏移误差|Clipped Brier|准确率|默认纠错期望题数|默认破坏期望题数|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,item in d['model_table'].items():
        metrics=item['metrics']
        get=lambda key:metrics[key]['balanced_v2']
        lines.append(f"|{name}|{get('absolute_mean_mse'):.8f}|{get('centered_mean_mse'):.8f}|{get('common_offset_mse'):.8f}|{get('brier_clipped'):.8f}|{100*get('accuracy'):.4f}%|{600*get('correct'):.3f}|{600*get('harm'):.3f}|")
    lines.extend(['','纠错和破坏以同题同生成评分配对后平均，所以期望题数可为小数。分母为全部600题。实际独立题数仍为600。','',
        '### 原始数值误差变化','',
        '|模型变化|绝对误差差|中心化误差差|共同偏移误差差|准确率差，百分点|共同偏移占绝对改善比例|',
        '|---|---:|---:|---:|---:|---:|'])
    for name,item in d['comparisons'].items():
        ci=item['primary_fold_task_paired']
        share=item['offset_fraction_of_absolute_improvement']
        lines.append(f"|{name}|{format_ci(ci['absolute_mean_mse'])}|{format_ci(ci['centered_mean_mse'])}|{format_ci(ci['common_offset_mse'])}|{format_ci(ci['accuracy'],100)}|{share:.3f}"+'|' if share is not None else f"|{name}|{format_ci(ci['absolute_mean_mse'])}|{format_ci(ci['centered_mean_mse'])}|{format_ci(ci['common_offset_mse'])}|{format_ci(ci['accuracy'],100)}|绝对误差未下降，比例不定义|")
    lines.extend(['','差值均为新减旧，误差负值表示改善。比例只在绝对误差下降时计算，超过1表示共同偏移改善被相对误差恶化部分抵消。主95%描述区间按外折×GSM/BBH对600原题重采样，完整三重复保留。历史训练区重叠与已冻结方案选择影响未纳入。B的多项诊断区间保留未校正描述性质，主门控判定使用E5-C另定比较。','',
        '### Clipped Brier的独立分解','',
        '|模型变化|Brier差|Clipped中心化分量差|Clipped共同偏移差|共同偏移占Brier改善比例|',
        '|---|---:|---:|---:|---:|'])
    for name,item in d['comparisons'].items():
        delta=item['delta']
        share=item['offset_fraction_of_clipped_brier_improvement']
        share_text=f'{share:.3f}' if share is not None else '未改善，比例不定义'
        lines.append(f"|{name}|{delta['brier_clipped']['balanced_v2']:+.8f}|{delta['brier_centered_component']['balanced_v2']:+.8f}|{delta['brier_offset_component']['balanced_v2']:+.8f}|{share_text}|")
    lines.extend(['','Brier使用[0,1]裁剪概率，真实服务动作继续使用冻结原始评分和并列规则。Brier等于裁剪后中心化误差、共同偏移误差以及同题生成观测噪声之和，跨模型差中最后一项抵消。原始Ridge数值误差分解与此表分开。','',
        '### 事后差距与排序','',
        '|模型/阶段|逐生成事后最佳差距|三重复均值后事后最佳差距|非平局二值排序准确率|非平局均值排序准确率|Ti负估计题数|',
        '|---|---:|---:|---:|---:|---:|'])
    for name,item in d['model_table'].items():
        m=item['metrics']
        lines.append(f"|{name}|{m['retrospective_generation_gap']['balanced_v2']:.8f}|{m['retrospective_mean_gap']['balanced_v2']:.8f}|{m['non_tie_binary_pair_ranking']['balanced_v2']:.6f}|{m['non_tie_mean_reward_ranking']['balanced_v2']:.6f}|{item['negative_cross_repeat_questions']}|")
    lines.extend(['','两种事后差距分别基于已观察生成结果及有限三生成均值，它们都不能直接视为总体条件均值Regret。全组合奖励相同的题没有严格排序对，排序统计按有效原题报告。','',
        '## 跨重复风险的统计解释','',
        '程序逐个计算L(L−1)个有序交叉项，并独立核验Ti等于中心化观测误差减去trace形式的噪声项。各固定模型及三反馈路径、各题均验证代数恒等式。负估计保留。','',
        '冻结成员、聚合器、提示及评分版本支持同一目标协议的意向。三个成员种子确有不同执行，聚合器种子固定42。但条件独立与共同条件边际分布不能由整数种子或现有三次观测识别，硬件/执行条件也需要额外假设。因此主要解释采用冻结预测器的配对中心化损失差，Ti仅作为带假设的绝对风险诊断量。Ti差与中心化观测损失差的题级代数相等不依赖重复独立。','',
        '## 统计结论',''])
    for name in ('P1_to_P2','P2_to_Probit'):
        item=d['comparisons'][name]
        delta=item['delta']
        lines.append(f"{name}的绝对误差变化为{delta['absolute_mean_mse']['balanced_v2']:+.8f}，中心化变化{delta['centered_mean_mse']['balanced_v2']:+.8f}，共同偏移变化{delta['common_offset_mse']['balanced_v2']:+.8f}。准确率变化为{100*delta['accuracy']['balanced_v2']:+.4f}个百分点，其描述性区间见主表。")
        lines.append('')
    lines.extend(['共同偏移与相对误差的贡献以实测分量分别给出，不能从绝对误差改善自动推导换队排序有效。各任务、27个BBH子任务、三Query生成重复和Support×Query种子增益矩阵均保存至数值JSON。','',
        f"真实评价的裸argmax与冻结1e−10并列规则差异合计{sum(v['native_argmax_vs_declared_tolerance_mismatches'] for v in d['algebra_audits'].values())}。所有实际模型及反馈路径的最小top-2分数间隔为{min(v['minimum_top_score_gap'] for v in d['algebra_audits'].values()):.6g}。真实服务奖励与事后差距均由冻结并列规则重算。",'',
        f"Probit沿用固定惩罚10和原L-BFGS拟合。五折优化器收敛标志均为{all(v['success'] for v in d['frozen_Probit_fit_receipts'])}，原梯度无穷范数最大{max(v['gradient_infinity'] for v in d['frozen_Probit_fit_receipts']):.6g}。梯度精度作为原拟合的数值属性披露，本诊断保持已冻结参数。",'',
        '## 优势与风险','',
        '优势是原题级精确误差恒等式、三重复配对及固定F/G/E权限。风险在于旧开发题复用、五折历史重叠和三生成重复分布假设不足。Ti本身出现负数无需修正为0，Gaussian原始预测的有界风险结论需另行检查其[0,1]条件。','',
        '## 下一步','',
        'E5-C检验同一候选组合是否能在任务内部由门控分数筛出正净收益区域。B阶段诊断提供误差来源证据，不替代冻结门控与匹配随机对照的服务收益检验。','',
        '## 数值资格与来源','',
        f"真实五折P2中心化预测重算与冻结P2输出最大差{max(row['frozen_P2_prediction_replay_max_error'] for row in d['actual_centering_checks']):.3g}，中心化最大误差{max(row['prediction_centering_max_error'] for row in d['actual_centering_checks']):.3g}，五折选队与准确率差均为0。完整题级配对数组在private/CENTERED_DIAGNOSTICS_PAIRED.npz，公开文件保留其SHA256及各输入哈希。新增LLM调用为0。",''])
    (PUBLIC/'02_CENTERED_ERROR_DIAGNOSTICS.md').write_text('\n'.join(lines),encoding='utf8')


def append_actual_A(rows):
    path=PUBLIC/'01_THEORY_IMPLEMENTATION_CHECKS.md'
    marker='\n## 真实五折中心化一致性复核\n'
    original=path.read_text(encoding='utf8').split(marker)[0]
    lines=[marker,'','E5-B读取已完成的F360/E120设计，对每折独立重算相同三重复损失权重的P2。F区之外的评分仅用于结果评价。','',
        '|外折|系数等价误差|中心化预测误差|冻结P2重算误差|动作差异|准确率差|',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row['case_id']}|{row['coefficient_identity_max_error']:.3g}|{row['prediction_centering_max_error']:.3g}|{row['frozen_P2_prediction_replay_max_error']:.3g}|{row['declared_lexicographic_selection_mismatches']}|{row['outer_accuracy_difference']:.3g}|")
    lines.extend(['','五个外折合计600个不同E题，中心化等价与准确率差为0得到真实设计重算验证。该结果核验相同拟合算子的等价性，不构成新方法的准确率提升。逐折JSON见ACTUAL_CENTERING_RESULTS.json。',''])
    path.write_text(original+'\n'.join(lines),encoding='utf8')


if __name__=='__main__':main()
