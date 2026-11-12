"""Original-question statistics for fixed E7-B models and negative controls."""
from run_learning_curves import *
from run_learning_curves import _json


def _weights(task):
    task=np.asarray(task)
    return np.where(task=='gsm8k',.5/np.sum(task=='gsm8k'),.5/np.sum(task=='bbh'))


def _interval(point,draws):
    draws=np.asarray(draws)
    return dict(point=float(point),ci95=np.quantile(draws,[.025,.975]).tolist(),
        ci_family_bonferroni=np.quantile(draws,[.05/8,1-.05/8]).tolist(),
        centered_bootstrap_p=float((1+np.sum(abs(draws-point)>=abs(point)-1e-15))/(len(draws)+1)))


def _metrics(values,task,sub,weights):
    service=values['service']
    means=service.mean(axis=1)
    base=values['base'].mean(axis=1)
    by_task={name:float(means[task==name].mean()) for name in ['gsm8k','bbh']}
    by_sub={name:float(means[(task=='bbh')&(sub==name)].mean()) for name in sorted(set(sub[task=='bbh']))}
    task_gain={name:float((means-base)[task==name].mean()) for name in ['gsm8k','bbh']}
    sub_gain={name:float((means-base)[(task=='bbh')&(sub==name)].mean()) for name in by_sub}
    result=dict(accuracy=float(means@weights),seed_accuracy=(weights@service).tolist(),
        task_accuracy=by_task,BBH_subtask_accuracy=by_sub,task_gain_default=task_gain,BBH_subtask_gain_default=sub_gain,
        GSM_BBH_subtask_macro=float(.5*by_task['gsm8k']+.5*np.mean(list(by_sub.values()))),
        corrections_expected_questions=float(values['correction'].sum()),harm_expected_questions=float(values['harm'].sum()),
        net_corrections_expected_questions=float((values['correction']-values['harm']).sum()))
    for key in ['absolute_mse','centered_mse','offset_mse','contrast_mae','regret']:
        result[key]=float(values[key]@weights)
    result['non_tied_pairwise_ranking_accuracy']=None if values['pair_count'].sum()==0 else float(values['pair_correct'].sum()/values['pair_count'].sum())
    result['non_tied_pairwise_comparisons']=int(values['pair_count'].sum())
    result['ranking_tie_rule']='True ties omitted; predicted ties credit0.5; combination pairs within an original question are not independent questions'
    return result


def controls():
    """Known reward means identify null/signal properties independently of noise."""
    controls=[]
    for kind_environment in ['equal_expected_reward','known_feature_switch']:
        rng=np.random.default_rng(np.random.SeedSequence([20261112,0 if kind_environment=='equal_expected_reward' else 1,771]))
        N=420
        x=rng.normal(scale=.04,size=(N,24))
        signs=np.r_[np.ones(N//2),-np.ones(N-N//2)]
        rng.shuffle(signs)
        x[:,0]=signs
        x/=np.linalg.norm(x,axis=1)[:,None]
        truth=np.full((N,20),.5)
        if kind_environment=='known_feature_switch':
            truth[:]=.15
            truth[:,0]=np.where(signs>0,.95,.1)
            truth[:,1]=np.where(signs<0,.95,.1)
        raw=(rng.random((N,3,20))<truth[:,None]).astype(float)
        I=np.arange(240)
        U=np.arange(240,300)
        E=np.arange(300,420)
        tasks=np.where(np.arange(N)%2==0,'gsm8k','bbh')
        default=int(argmax(raw[I].mean(axis=(0,1))))
        row=dict(environment=kind_environment,seed=[20261112,0 if kind_environment=='equal_expected_reward' else 1,771],
                 initial=240,added=60,query=120,default=default,models={})
        for kind in KINDS:
            spec=build_spec(x,I,raw[I],kind,tasks=tasks,cv_seed=20261112)
            snapshots=[]
            for n,H in [(240,I),(300,np.r_[I,U])]:
                predicted=batch_predict(spec,H,raw[H],E)
                values=error_arrays(predicted,raw[E],default)
                actions=values['actions']
                population=truth[E,actions]
                baseline=truth[E,default]
                snapshots.append(dict(n=n,empirical_accuracy=float(values['service'].mean()),
                    known_expected_accuracy=float(population.mean()),known_expected_default=float(baseline.mean()),
                    known_expected_gain=float((population-baseline).mean()),
                    centered_mse=float(values['centered_mse'].mean()),offset_mse=float(values['offset_mse'].mean())))
                if kind_environment=='equal_expected_reward':
                    assert np.all(population-baseline==0)
            row['models'][kind]=snapshots
        if kind_environment=='known_feature_switch':
            assert row['models']['P3'][-1]['known_expected_gain']>.25
        controls.append(row)
    result=dict(status='PASS',cases=controls,
        interpretation='Software-only known mechanisms, separate from real routing evidence; all methods use the same fit/prediction/metric functions',
        null_rule='Known means are exactly equal, so any noisy realized difference is never classified as a true advantage')
    put(PUBLIC/'E7_B_SOFTWARE_CONTROLS.json',result)
    return result


def summarize():
    data,_=load_train()
    main=[]; shuffled=[]; Ys=[]; tasks=[]; subs=[]; folds=[]; defaults=[]; rows=[]; training=[]; costs=[]
    for f in range(5):
        with np.load(PRIVATE/f'EVALUATED_B_f{f}.npz') as a:
            E=a['E'];main.append(a['main_predictions']);shuffled.append(a['permutation_predictions'])
            Ys.append(a['Y']);tasks.append(a['task']);subs.append(a['sub']);defaults.extend([int(a['default'])]*len(E))
            rows.extend(E.tolist());folds.extend([f]*len(E))
        receipt=json.loads((PRIVATE/f'FROZEN_B_f{f}.json').read_text())
        training.append(receipt['training'])
        role=splits(f)
        for n in SIZES:
            H=np.r_[role['I'],role['U'][:n-240]]
            source=resources(data,H,[range(20)]*n,gold=n)
            source.update(fold=f,n=n,shared_by_all_models=True,additional_gold_over_I=n-240,
                additional_member_calls_over_I=(n-240)*18,additional_aggregator_calls_over_I=(n-240)*60)
            costs.append(source)
    predicted=np.concatenate(main,axis=2)
    permuted=np.concatenate(shuffled,axis=2)
    Y=np.concatenate(Ys)
    task=np.concatenate(tasks).astype(str);sub=np.concatenate(subs).astype(str);fold=np.asarray(folds);default=np.asarray(defaults)
    assert len(set(rows))==600
    weights=_weights(task)
    scores=np.zeros((3,4,600))
    perm_scores=np.zeros((20,4,600))
    centered=np.zeros_like(scores)
    perm_centered=np.zeros_like(perm_scores)
    baseline=None
    records={}
    for j,kind in enumerate(KINDS):
        records[kind]={}
        for k,n in enumerate(SIZES):
            values=error_arrays(predicted[j,k],Y,default)
            scores[j,k]=values['service'].mean(axis=1)
            centered[j,k]=values['centered_mse']
            if baseline is None:baseline=values['base'].mean(axis=1)
            metric=_metrics(values,task,sub,weights)
            record=dict(metrics=metric,training_by_fold=[training[f][j*4+k] for f in range(5)])
            record['mean_training_absolute_mse']=float(np.mean([v['train_absolute_mse'] for v in record['training_by_fold']]))
            record['mean_training_centered_mse']=float(np.mean([v['train_centered_mse'] for v in record['training_by_fold']]))
            record['source_supervision_cost_by_fold']=[c for c in costs if c['n']==n]
            records[kind][str(n)]=record
    perm_metrics=[]
    for p,seed in enumerate(PERMUTATIONS):
        per_n={}
        for k,n in enumerate(SIZES):
            values=error_arrays(permuted[p,k],Y,default)
            perm_scores[p,k]=values['service'].mean(axis=1)
            perm_centered[p,k]=values['centered_mse']
            per_n[str(n)]=_metrics(values,task,sub,weights)
        perm_metrics.append(dict(seed=seed,sizes=per_n))
    # All randomizations/repetitions remain inside each original-question record.
    contributions=np.concatenate([scores.reshape(12,600),perm_scores.reshape(80,600),baseline[None]],axis=0)
    draws=paired_bootstrap(contributions,task,fold,reps=5000,seed=20261112)
    main_boot=draws[:,:12].reshape(5000,3,4)
    perm_boot=draws[:,12:92].reshape(5000,20,4)
    baseline_boot=draws[:,92]
    for j,kind in enumerate(KINDS):
        for k,n in enumerate(SIZES):
            record=records[kind][str(n)]
            record['gain_default']=_interval(float((scores[j,k]-baseline)@weights),main_boot[:,j,k]-baseline_boot)
            record['gain_self240']=_interval(float((scores[j,k]-scores[j,0])@weights),main_boot[:,j,k]-main_boot[:,j,0])
            record['seed_gain_default']=(weights[:,None]*(error_arrays(predicted[j,k],Y,default)['service']-Y[np.arange(600),:,default])).sum(axis=0).tolist()
            record['seed_gain_self240']=(weights[:,None]*(error_arrays(predicted[j,k],Y,default)['service']-error_arrays(predicted[j,0],Y,default)['service'])).sum(axis=0).tolist()
            record['fold_accuracy']={str(f):float(scores[j,k,fold==f].mean()) for f in range(5)}
            record['fold_gain_default']={str(f):float((scores[j,k]-baseline)[fold==f].mean()) for f in range(5)}
            record['fold_gain_self240']={str(f):float((scores[j,k]-scores[j,0])[fold==f].mean()) for f in range(5)}
    learning=scores[2,3]-scores[2,0]
    content=scores[2,3]-perm_scores[:,3].mean(axis=0)
    learning_result=_interval(float(learning@weights),main_boot[:,2,3]-main_boot[:,2,0])
    content_result=_interval(float(content@weights),main_boot[:,2,3]-perm_boot[:,:,3].mean(axis=1))
    content_error=centered[2,3]-perm_centered[:,3].mean(axis=0)
    error_result,error_boot=paired_interval(content_error,task,fold,reps=5000,seed=20261112,family=4)
    content_task={t:float(content[task==t].mean()) for t in ['gsm8k','bbh']}
    content_sub={s:float(content[(task=='bbh')&(sub==s)].mean()) for s in sorted(set(sub[task=='bbh']))}
    result=dict(status='COMPLETE',original_E_questions=600,generation_repeats_per_question=3,
        models=list(KINDS),sizes=list(SIZES),default_accuracy=float(baseline@weights),
        configurations=records,permutations=perm_metrics,
        primary_learning=learning_result,primary_content=content_result,
        content_centered_error_difference=error_result,content_task_gain=content_task,content_BBH_subtask_gain=content_sub,
        permutation_rule='I and currently authorized U_n separately shuffled within GSM/BBHsubtask, whole20x3 reward blocks. I-only nested CV rerun. U mappings may vary by n.',
        permutation_count=20,permutation_repeats_are_not_new_questions=True,
        primary_content_n=480,permutation_service_accuracy_mean=float(np.mean([v['sizes']['480']['accuracy'] for v in perm_metrics])),
        permutation_service_accuracy_range=[float(min(v['sizes']['480']['accuracy'] for v in perm_metrics)),float(max(v['sizes']['480']['accuracy'] for v in perm_metrics))],
        interpretation='Frozen within-fold preprocessing and regularization, only authorized question packages increase. Conditional development question bootstrap, overlapping training histories.',
        physical_new_llm_calls=0)
    prepared=json.loads((PRIVATE/'PREPARED_ALL.json').read_text())
    result['P3_lambdas']=[row['models']['P3']['metadata']['lambda_value'] for row in prepared]
    result['P3_bandwidths']=[row['models']['P3']['metadata']['bandwidth'] for row in prepared]
    result['fit_runtime_by_fold']=[{key:json.loads((PRIVATE/f'FROZEN_B_f{f}.json').read_text())[key] for key in ['fit_wall_s','fit_cpu_s']} for f in range(5)]
    put(PUBLIC/'E7_B_RESULTS.json',_json(result))
    np.savez_compressed(PRIVATE/'PAIRED_B_RESULTS.npz',E=np.asarray(rows),task=task,sub=sub,fold=fold,default=default,Y=Y,
        main_predictions=predicted,permutation_predictions=permuted,main_scores=scores,permutation_scores=perm_scores,
        main_centered_error=centered,permutation_centered_error=perm_centered,baseline=baseline,
        bootstrap_main=main_boot,bootstrap_permutation=perm_boot,bootstrap_baseline=baseline_boot)
    np.savez_compressed(PRIVATE/'PRIMARY_B_COMPARISONS.npz',question_indices=np.asarray(rows),task=task,fold=fold,
        learning_contribution=learning,learning_bootstrap=main_boot[:,2,3]-main_boot[:,2,0],learning_point=learning_result['point'],
        content_contribution=content,content_bootstrap=main_boot[:,2,3]-perm_boot[:,:,3].mean(axis=1),content_point=content_result['point'])
    software=controls()
    table=[]
    for kind in KINDS:
        for n in SIZES:
            r=records[kind][str(n)]
            m=r['metrics']
            table.append(f"| {kind} | {n} | {100*m['accuracy']:.4f}% | {100*r['gain_self240']['point']:+.4f} | {m['centered_mse']:.6f} | {m['offset_mse']:.6f} | {m['non_tied_pairwise_ranking_accuracy']:.4f} |")
    def pp_ci(r):return f"{100*r['point']:+.4f} [{100*r['ci95'][0]:+.4f}, {100*r['ci95'][1]:+.4f}]"
    controls_table=[]
    for case in software['cases']:
        for kind in KINDS:
            m=case['models'][kind][-1]
            controls_table.append(f"| {case['environment']} | {kind} | {100*m['empirical_accuracy']:.4f}% | {100*m['known_expected_accuracy']:.4f}% | {100*m['known_expected_gain']:+.4f} |")
    text=f'''# E7-B 题目信号与独立问题学习曲线

## 研究问题

直接预测20组合并选最高分，检验更多独立题目是否改善同一预测器，以及完整文本向量是否提供同任务内部的组合选择信息。每次服务仍执行一个K=3组合。

## 实验设置

五个外折H480/E120，H按预存问题ID划为I240和顺序U240。学习曲线使用I加U的前0、60、120、240题，所有模型共享。I确定全局默认并固定。P1保留原P2的PCA16、标准化与带惩罚常数，raw3正则100等价于三重复均值目标正则100/3。P2使用原归一化MiniLM完整向量加带惩罚常数，相同正则。两种Ridge的Gaussian表示为线性核XX'/100与均值观测噪声1/3。

P3使用RBF核，I内距离中位数定带宽，λ∈{{0.1,1,10}}在I内原题分层三折按服务准确率选择，并列更强正则。内折均值和带宽仅由内训练题拟合。最终μI、带宽、λ在后续增量监督中固定。五折λ为{result['P3_lambdas']}。

原三重复与均值正则换算、真实I上的原E3预测和协方差、逐包更新与整体重拟合均通过资格检查。所有外折预测先写入私有冻结文件，再由评价器读取E评分。

## 对照

主P3每折执行20次预固定语义组内整块置乱。I只在I内置乱，新增U_n只在当前合法U_n内置乱，避免未来U标签进入I调参。保留每个20组合×3生成奖励块。置乱重新选择I内λ，评价区原样。不同n的U置乱映射可能变化，这属于内容诊断曲线。所有源映射均保留，重复不增加独立题量。

## 数值结果

I固定默认准确率{100*result['default_accuracy']:.4f}%。下表误差对三重复均值定义，centered误差去掉每题跨组合共同偏移。排序仅计算真实非平局组合对，预测并列计0.5。

| 模型 | 历史题数 | E准确率 | 相对自身240，百分点 | 中心化MSE | 共同偏移MSE | 非平局排序准确率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

P3从240到480题的配对收益为{pp_ci(learning_result)}个百分点。原P3@480相对20个任务内置乱模型均值的内容收益为{pp_ci(content_result)}个百分点。置乱模型@480平均准确率{100*result['permutation_service_accuracy_mean']:.4f}%，范围{[round(100*v,4) for v in result['permutation_service_accuracy_range']]}%。原始减置乱的中心化MSE差为{error_result['point']:+.6f}，95%CI{error_result['ci95']}。

GSM内容收益{100*content_task['gsm8k']:+.4f}、BBH内容收益{100*content_task['bbh']:+.4f}个百分点。全部BBH子任务、逐种子、纠错与破坏、Top1Regret、组合差值MAE和每折训练误差见E7_B_RESULTS.json。训练误差未作为科学通过依据。

## 软件正负控制

两个已知总体奖励的环境复用相同fit、argmax与指标代码。无条件优势环境中所有组合期望0.5，任何实际样本差异都具有精确零总体增益。明确可学习环境中，输入首特征的符号确定最优组合，P3真实期望收益大于25个百分点。

| 环境 | 模型 | 噪声样本准确率 | 已知总体准确率 | 已知总体相对默认，百分点 |
|---|---|---:|---:|---:|
{chr(10).join(controls_table)}

## 统计结论

5000次原题bootstrap按外折×GSM/BBH分层，所有三生成及20置乱模型共同重抽样。主学习比较和主内容比较交由总报告统一Holm4。这里的95%CI是单项描述区间。学习器改善与主动采集的独立贡献分开检验。

## 成本与支持数

新增完整题包各含18成员、60聚合、60原始评分、1次Gold。240→480每折增加240道不同题，源账本新增4320成员、14400聚合与240 Gold。现有缓存无需物理生成。各n的Token、源wall-time、有效方向问题数和每折成本逐项保留。五折历史相交，不能相加作为独立题数。

## 优势与风险

优势是三个固定数学学习器、I内调参、冻结表示和对齐标签权限的内容负对照。风险是每BBH子任务训练题少、完整包信息量大、工作后验可能失配、置乱并非独立新数据，以及外折历史重叠。完整向量与核模型的表现仅支持本协议内的诊断。

## 失败原因与下一步

若原始比置乱没有稳定优势，更多监督带来的提升也可能来自任务层面的组合频率估计，不能直接推出同任务问题内容已可用。若特定学习器的独立问题曲线有收益，再在E7-C固定该学习器检验Random、ParameterIG和DecisionAware各自是否改善后续未见题。当前结果不触发额外LLM调用。
'''
    (ALL_PUBLIC/'02_INPUT_SIGNAL_AND_LEARNING_CURVES.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',learning=learning_result,content=content_result,
                         controls=software['status'],P3=[records['P3'][str(n)]['metrics']['accuracy'] for n in SIZES])),flush=True)


if __name__=='__main__':
    summarize()
