"""Question-paired evaluation and complete cost accounts for E6-C."""
from run_real_pool import *


def interval(point, draws, family=3):
    draws=np.asarray(draws)
    return dict(point=float(point),ci95=np.quantile(draws,[.025,.975]).tolist(),
                ci_family_bonferroni=np.quantile(draws,[.05/(2*family),1-.05/(2*family)]).tolist(),
                centered_bootstrap_p=float((1+np.sum(abs(draws-point)>=abs(point)-1e-15))/(len(draws)+1)))


def summarize():
    data,_=load_train()
    summaries=[]
    arrays=[]
    question_scores=[]
    seed_scores=[]
    sourceP=[]
    sourceQ=[]
    baseline=[]
    cpu=0.
    service_cost=[]
    costs=[]
    mc_scores=[]
    for f in range(5):
        summary=json.loads((WORK/f'private/c/SUMMARY_C_f{f}.json').read_text())
        summaries.append(summary)
        cpu+=summary['cpu_s']
        with np.load(WORK/f'private/c/C_f{f}.npz') as a:
            raw={key:a[key] for key in a.files}
        with np.load(WORK/f'private/c/EVALUATED_C_f{f}.npz') as a:
            val={key:a[key] for key in a.files}
        selected=raw['selected']
        qmasks=raw['masksQ']
        # Mean over acquisition randomization inside each original question.
        gate_probability=qmasks[selected].mean(axis=2)
        score=val['RbQ'][None,None]+gate_probability[...,None]*val['DQ'][None,None]
        seed_scores.append(score)
        question_scores.append(score.mean(axis=-1))
        sourceP.extend(raw['P'].tolist())
        sourceQ.extend(raw['Q'].tolist())
        baseline.append(val['RbQ'].mean(axis=1))
        mc_scores.append(val['RbQ'].mean()+val['gainsQ'][selected])
        with np.load(WORK/f'private/a/fold{f}_policies.npz') as a:
            E,base,candidate=a['E'],int(a['base']),a['candidate']
        qrows=raw['Q']
        clocal=np.asarray([np.flatnonzero(E==q)[0] for q in qrows])
        cnames=raw['costnames'].tolist()
        strategy_service=[]
        for mask in qmasks:
            chosen=np.where(mask,candidate[clocal],base)
            strategy_service.append(resources(data,qrows,[[int(c)] for c in chosen],gold=0))
        strategy_service=np.asarray([[entry[name] for name in cnames] for entry in strategy_service])
        strategy_counts=np.zeros((3,5,len(strategy_service)))
        for bi in range(3):
            for mi in range(5):
                strategy_counts[bi,mi]=np.bincount(selected[bi,mi],minlength=len(strategy_service))/selected.shape[2]
        service_cost.append(strategy_counts@strategy_service)
        costs.append(raw['purchases'].astype(float)@raw['costmatrix'])
        arrays.append((raw,val))
    qscore=np.concatenate(question_scores,axis=2)
    sscore=np.concatenate(seed_scores,axis=2)
    base=np.concatenate(baseline)
    Q=np.asarray(sourceQ)
    assert len(set(Q))==300
    assert len(set(sourceP))==300 and not set(Q)&set(sourceP)
    task=data['task'][Q]
    sub=data['sub'][Q]
    fold=np.repeat(np.arange(5),60)
    rng=np.random.default_rng(20261112)
    boot=[]
    for _ in range(5000):
        indices=np.concatenate([rng.choice(rows,len(rows),replace=True)
            for f in range(5) for t in ['gsm8k','bbh']
            for rows in [np.flatnonzero((fold==f)&(task==t))]])
        boot.append(qscore[:,:,indices].mean(axis=2))
    boot=np.asarray(boot)
    point=qscore.mean(axis=2)
    costmean=np.mean(np.stack(costs),axis=(0,3))
    servicemean=np.mean(np.stack(service_cost),axis=0)
    results=dict(status='COMPLETE',N_Q_original=300,N_P_original=300,pilot_draws_per_fold=50,
        purchase_draws_per_pilot=40,randomizations_per_fold=2000,folds=5,methods=METHODS,budgets=BUDGETS,
        main_budget=36,new_llm_calls=0,cpu_s=cpu,default_accuracy=float(base.mean()),
        statistical_scope='Q-question bootstrap conditional on frozen P observations, source predictors and policies; outer training overlaps. 50 pilot draws x40 conditional purchases are dependent clusters, never 2000 independent questions.',
        execution_scope='Counterfactual existing-cache source costs, physical new calls0; Q service and evaluation costs reported separately.',
        sampler='fixed count and call-cap conditional-Bernoulli with exact inclusion probabilities',
        policy_identity='Marginal and Contrast identical due default/all, identical proposal, subsets, estimates and selections',
        paired_comparisons={},table={},fold_summaries=summaries,
        main_original_question_ids_not_public=True)
    contrasts=[('Contrast-Marginal',2,1),('Contrast-Uniform',2,0),('GapContrast-Uniform',3,0),
               ('GapContrast-Marginal',3,1),('NoPilotUniform-Uniform',4,0)]
    mc=np.stack(mc_scores)
    for bi,B in enumerate(BUDGETS):
        table={}
        for mi,name in enumerate(METHODS):
            score=qscore[bi,mi]
            task_values={t:float(score[task==t].mean()) for t in ['gsm8k','bbh']}
            sub_values={s:float(score[(task=='bbh')&(sub==s)].mean()) for s in sorted(set(sub[task=='bbh']))}
            task_macro=.5*task_values['gsm8k']+.5*task_values['bbh']
            sub_macro=.5*task_values['gsm8k']+.5*np.mean(list(sub_values.values()))
            seed_acc=sscore[bi,mi].mean(axis=0)
            table[name]=dict(accuracy=float(point[bi,mi]),task=task_values,BBH_subtasks=sub_values,
                task_macro=float(task_macro),GSM_BBH_subtask_macro=float(sub_macro),seed_accuracy=seed_acc.tolist(),
                gain_default=interval(float(point[bi,mi]-base.mean()),boot[:,bi,mi]-np.asarray([base.mean()]*len(boot))),
                feedback_cost=dict(zip(cnames,costmean[bi,mi].tolist())),
                selected_Q_service_cost=dict(zip(cnames,servicemean[bi,mi].tolist())),
                feedback_plus_Q_service_cost=dict(zip(cnames,(costmean[bi,mi]+servicemean[bi,mi]).tolist())),
                pool_regret=float(np.mean([s['methods'][str(B)][name]['pool_regret'] for s in summaries])),
                Q_regret=float(np.mean([s['methods'][str(B)][name]['Q_regret'] for s in summaries])),
                pool_wrong_rate=float(np.mean([s['methods'][str(B)][name]['pool_wrong_rate'] for s in summaries])),
                Q_wrong_rate=float(np.mean([s['methods'][str(B)][name]['Q_wrong_rate'] for s in summaries])),
                pilot_gold_fraction=0 if mi==4 else 12/B)
        results['table'][str(B)]=table
        for label,a,b in contrasts:
            p=float(point[bi,a]-point[bi,b])
            record=interval(p,boot[:,bi,a]-boot[:,bi,b])
            paired=mc[:,bi,a]-mc[:,bi,b]
            cluster_means=paired.reshape(5,50,40).mean(axis=2)
            se=float(np.sqrt(np.sum(cluster_means.var(axis=1,ddof=1)/50))/5)
            record['paired_MC']=dict(point=float(paired.mean()),cluster_SE=se,
                ci95_normal=[float(paired.mean()-1.96*se),float(paired.mean()+1.96*se)],
                independent_pilot_clusters=250,purchases_per_cluster=40)
            record['seed_difference']=(sscore[bi,a]-sscore[bi,b]).mean(axis=0).tolist()
            results['paired_comparisons'][f'{label}@{B}']=record
    # Recompute baseline differences on the same original-question draws.
    # This separate bootstrap preserves default/question covariance exactly.
    rng=np.random.default_rng(20261112)
    gain_boot=[]
    for _ in range(5000):
        ix=np.concatenate([rng.choice(rows,len(rows),replace=True)
            for f in range(5) for t in ['gsm8k','bbh']
            for rows in [np.flatnonzero((fold==f)&(task==t))]])
        gain_boot.append((qscore[:,:,ix]-base[ix]).mean(axis=2))
    gain_boot=np.asarray(gain_boot)
    for bi,B in enumerate(BUDGETS):
        for mi,name in enumerate(METHODS):
            results['table'][str(B)][name]['gain_default']=interval(float(point[bi,mi]-base.mean()),gain_boot[:,bi,mi])
    np.savez_compressed(WORK/'private/c/PAIRED_C_RESULTS.npz',Q=Q,task=task,sub=sub,fold=fold,
        score=qscore,score_seed=sscore,base=base,bootstrap_accuracy=boot,bootstrap_gain=gain_boot,
        MC_accuracy=mc,cost_mean=costmean,service_mean=servicemean)
    put(WORK/'public/c/E6_C_RESULTS.json',results)
    print(json.dumps(dict(status='COMPLETE',default=base.mean(),main_accuracy=point[1].tolist(),
        contrast_uniform=results['paired_comparisons']['Contrast-Uniform@36'])),flush=True)


if __name__=='__main__':
    summarize()
