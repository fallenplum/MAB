"""Independent last-axis routing, paired statistics and cost reconstruction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
sys.path.insert(0,str(WORK/'public'))
from e7_common import load_train,splits,resources


def main():
    data,_=load_train()
    kinds=['P1','P3']
    methods=['Random','ParameterIG','DecisionAware']
    seeds=[17,29,43]
    budgets=[0,20,40,60]
    count=0
    maxerror=0.
    def equal(a,b):
        nonlocal count,maxerror
        a,b=np.asarray(a),np.asarray(b)
        assert a.shape==b.shape,(a.shape,b.shape)
        error=float(np.max(abs(a.astype(float)-b.astype(float)))) if a.size else 0.
        maxerror=max(maxerror,error)
        assert error<1e-12,error
        count+=a.size
    score=np.empty((2,3,3,4,600,3))
    actions=np.empty((2,3,3,4,600),int)
    centered=np.empty((2,3,3,4,600))
    offset=np.empty_like(centered)
    baseline=[]
    qrows=[]
    tasks=[]
    subtasks=[]
    costs=[]
    service_costs=[]
    for f in range(5):
        roles=splits(f)
        I,E=roles['I'],roles['E']
        truth=data['Y'][E]
        mean_truth=truth.mean(axis=1)
        base_value=data['Y'][I].mean(axis=(0,1))
        base=int(np.flatnonzero(base_value>=base_value.max()-1e-10)[0])
        baseline.append(truth[:,:,base])
        qrows.extend(E.tolist())
        tasks.extend(data['task'][E].tolist())
        subtasks.extend(data['sub'][E].tolist())
        fcost=[]
        fservice=[]
        for ki,kind in enumerate(kinds):
            for mi,method in enumerate(methods):
                for si,seed in enumerate(seeds):
                    folder=WORK/f'private/c/f{f}_{kind}_{method}_s{seed}'
                    with np.load(folder/'PATH_RECORDS.npz') as a:
                        predicted=a['E_predictions']
                    # Explicit action-axis choice from each original query.
                    selected=np.asarray([[int(np.flatnonzero(row>=row.max()-1e-10)[0]) for row in snapshot] for snapshot in predicted])
                    evaluated=np.asarray([truth[np.arange(120)[:,None],np.arange(3)[None,:],row[:,None]] for row in selected])
                    assert selected.shape==(4,120)
                    assert evaluated.shape==(4,120,3)
                    qslice=slice(f*120,(f+1)*120)
                    score[ki,mi,si,:,qslice]=evaluated
                    actions[ki,mi,si,:,qslice]=selected
                    residual=predicted-mean_truth
                    common=residual.mean(axis=-1)
                    centered[ki,mi,si,:,qslice]=np.square(residual-common[...,None]).mean(axis=-1)
                    offset[ki,mi,si,:,qslice]=common**2
                    log=json.loads((folder/'ACCESS_AND_COST_LOG.json').read_text())
                    names=list(log['ledger'][0])
                    ledger=np.asarray([[record[key] for key in names] for record in log['ledger']])
                    fcost.append(np.asarray([ledger[:B].sum(axis=0) for B in budgets]))
                    qcost=[]
                    for row in selected:
                        record=resources(data,E,[[int(action)] for action in row],gold=0)
                        qcost.append([record[key] for key in names])
                    fservice.append(qcost)
        costs.append(np.asarray(fcost).reshape(2,3,3,4,len(names)))
        service_costs.append(np.asarray(fservice).reshape(2,3,3,4,len(names)))
    baseline=np.concatenate(baseline)
    task=np.asarray(tasks)
    sub=np.asarray(subtasks)
    fold=np.repeat(np.arange(5),120)
    perq=score.mean(axis=(2,5))
    feedback_cost=np.stack(costs).mean(axis=(0,3))
    query_cost=np.stack(service_costs).mean(axis=(0,3))
    with np.load(WORK/'private/c/E7_C_PAIRED.npz') as a:
        for key,value in [('score',score),('actions',actions),('centered_error',centered),('common_offset_error',offset),
                          ('default_score',baseline),('perq',perq),('feedback_cost',feedback_cost),('E_service_cost',query_cost)]:
            equal(value,a[key])
        saved_boot=a['bootstrap']
        saved_learn=a['learn_bootstrap']
    rng=np.random.default_rng(20261112)
    boot=[]
    learning=[]
    comparison_contributions=[perq[ki,2,bi]-perq[ki,other,bi]
        for ki in range(2) for other in [0,1] for bi in range(4)]
    comparison_boot=[]
    for rep in range(5000):
        chosen=np.concatenate([rng.choice(indices,len(indices),replace=True)
            for f in range(5) for broad in ['gsm8k','bbh']
            for indices in [np.flatnonzero((fold==f)&(task==broad))]])
        w=np.where(task[chosen]=='gsm8k',.5/np.sum(task[chosen]=='gsm8k'),.5/np.sum(task[chosen]=='bbh'))
        expected=perq[...,chosen]@w
        gains=(perq-perq[:,:,:1])[...,chosen]@w
        equal(expected,saved_boot[rep])
        equal(gains,saved_learn[rep])
        boot.append(expected)
        learning.append(gains)
        # Preserve the declared scalar paired-contribution dot product.
        # Subtracting two separately rounded accuracy bootstraps can change a
        # boundary count in the discrete centered-bootstrap p calculation.
        comparison_boot.append([float(contribution[chosen]@w) for contribution in comparison_contributions])
    boot=np.asarray(boot)
    learning=np.asarray(learning)
    comparison_boot=np.asarray(comparison_boot)
    result=json.loads((WORK/'public/c/E7_C_RESULTS.json').read_text())
    equal(result['default_accuracy'],baseline.mean())
    for ki,kind in enumerate(kinds):
        for mi,method in enumerate(methods):
            for bi,B in enumerate(budgets):
                row=result['table'][kind][method][str(B)]
                values=perq[ki,mi,bi]
                equal(row['accuracy'],values.mean())
                equal(row['ci95'],np.quantile(boot[:,ki,mi,bi],[.025,.975]))
                equal(row['gain_initial'],np.mean(values-perq[ki,mi,0]))
                equal(row['gain_initial_ci95'],np.quantile(learning[:,ki,mi,bi],[.025,.975]))
                equal(row['seed_accuracy'],score[ki,mi,:,bi].mean(axis=(0,1)))
                equal(row['seed_gain_initial'],(score[ki,mi,:,bi]-score[ki,mi,:,0]).mean(axis=(0,1)))
                equal(row['algorithm_seed_accuracy'],score[ki,mi,:,bi].mean(axis=(1,2)))
                corrected=np.mean(score[ki,mi,:,bi]>score[ki,mi,:,0],axis=(0,2)).sum()
                harmed=np.mean(score[ki,mi,:,bi]<score[ki,mi,:,0],axis=(0,2)).sum()
                equal(row['corrected_question_equivalents'],corrected)
                equal(row['harmed_question_equivalents'],harmed)
                equal(row['net_correction'],corrected-harmed)
                equal(row['net_correction']/600,row['gain_initial'])
                equal(row['centered_error'],centered[ki,mi,:,bi].mean())
                equal(row['common_offset_error'],offset[ki,mi,:,bi].mean())
                for key,name in enumerate(names):
                    equal(row['feedback_cost'][name],feedback_cost[ki,mi,bi,key])
                    equal(row['E_service_cost'][name],query_cost[ki,mi,bi,key])
                    equal(row['feedback_plus_E_service_cost'][name],feedback_cost[ki,mi,bi,key]+query_cost[ki,mi,bi,key])
                for broad in ['gsm8k','bbh']:
                    equal(row['GSM_BBH'][broad],values[task==broad].mean())
                actual_sub=[float(values[(task=='bbh')&(sub==name)].mean()) for name in sorted(set(sub[task=='bbh']))]
                equal(row['GSM_BBH27_macro'],.5*values[task=='gsm8k'].mean()+.5*np.mean(actual_sub))
        for ci,(comparison,other) in enumerate([('DA-Random',0),('DA-IG',1)]):
            for bi,B in enumerate(budgets):
                row=result['paired_comparisons'][f'{kind}:{comparison}@{B}']
                contribution=perq[ki,2,bi]-perq[ki,other,bi]
                weights=np.where(task=='gsm8k',.5/np.sum(task=='gsm8k'),.5/np.sum(task=='bbh'))
                point=float(contribution@weights)
                draws=comparison_boot[:,ki*8+ci*4+bi]
                equal(row['point'],point)
                equal(row['ci95'],np.quantile(draws,[.025,.975]))
                equal(row['ci_family_bonferroni'],np.quantile(draws,[.05/8,1-.05/8]))
                p=(1+np.sum(abs(draws-point)>=abs(point)-1e-15))/(len(draws)+1)
                equal(row['centered_bootstrap_p'],p)
    output=dict(status='PASS',scalar_comparisons=count,maximum_absolute_error=maxerror,
        independently_rebuilt_paths=90,query_predictions=90*4*120,
        original_question_count=600,all_generation_and_algorithm_repeats_inside_question=True,
        bootstrap_draws_reconstructed=5000,paired_initial_covariance_retained=True,
        cost_mean_scope='same average per fold and algorithm path as report, initial history separate',
        decision_axis='20-combination last axis on checkpoint-by-question-by-action predictions',
        scientific_scope='Old-development conditional paired inference, no independent confirmation')
    (WORK/'public/theory/ACTIVE_FEEDBACK_STATISTICS_AUDIT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':
    main()
