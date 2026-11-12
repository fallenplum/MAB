"""Independent reconstruction of all E7-A sets and held-repeat metrics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import hashlib
import sys
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
sys.path.insert(0,str(ROOT/'research/e5_paired_gain_validation/public'))
from e5_common import load_train


def main():
    data,_=load_train()
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
    rows=[]
    outcomes=[]
    uniform=[]
    retained=[]
    opportunity=[]
    baselines=[]
    labels=[]
    chosen_by_family={name:[] for name in ['P2','Fixed','Random']}
    for f in range(5):
        path=WORK/f'private/a/CANDIDATES_f{f}.npz'
        receipt=json.loads((WORK/f'private/a/FROZEN_CANDIDATES_f{f}.json').read_text())
        assert receipt['candidate_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a:
            arrays={key:a[key] for key in a.files}
        E=arrays['E']
        b=int(arrays['default'])
        y=data['Y'][E]
        baseline=y[:,:,b]
        opp=(baseline==0)&(y.max(axis=2)==1)
        configurations=[]
        uniform_configs=[]
        retained_configs=[]
        for family in ['P2','Fixed','Random']:
            family_choices=[]
            masks=arrays[{'P2':'p2_masks','Fixed':'fixed_masks','Random':'random_masks'}[family]]
            for mi,m in enumerate([1,3,5,19]):
                mask=masks[mi]
                if mask.ndim==2:
                    mask=mask[:,None,:]
                assert np.all(mask[:,:,b])
                equal(mask.sum(axis=2),np.full(mask.shape[:2],m+1))
                if mi:
                    previous=masks[mi-1]
                    if previous.ndim==2:
                        previous=previous[:,None,:]
                    assert np.all(previous<=mask)
                n,draws,_=mask.shape
                scores=np.empty((n,draws,3))
                expected=np.empty_like(scores)
                chosen=np.empty_like(scores,dtype=int)
                for held in range(3):
                    other=y.sum(axis=1)-y[:,held]
                    scores_allowed=np.where(mask,other[:,None,:],-np.inf)
                    tied=mask&(scores_allowed==scores_allowed.max(axis=2,keepdims=True))
                    index=tied.argmax(axis=2)
                    scores[:,:,held]=y[np.arange(n)[:,None],held,index]
                    expected[:,:,held]=(tied*y[:,None,held]).sum(axis=2)/tied.sum(axis=2)
                    chosen[:,:,held]=index
                configurations.append(scores.mean(axis=1))
                uniform_configs.append(expected.mean(axis=1))
                covered=(mask[:,:,None,:]*y[:,None,:,:]).sum(axis=3)>0
                retained_configs.append((covered*opp[:,None,:]).mean(axis=1))
                family_choices.append(chosen[:,0] if draws==1 else chosen)
            chosen_by_family[family].append(np.asarray(family_choices))
        rows.extend(E.tolist())
        outcomes.append(np.asarray(configurations))
        uniform.append(np.asarray(uniform_configs))
        retained.append(np.asarray(retained_configs))
        opportunity.append(opp)
        baselines.append(baseline)
    outcome=np.concatenate(outcomes,axis=1)
    tie_uniform=np.concatenate(uniform,axis=1)
    retention=np.concatenate(retained,axis=1)
    baseline=np.concatenate(baselines)
    opp=np.concatenate(opportunity)
    with np.load(WORK/'private/a/OPPORTUNITY_PAIRED.npz') as a:
        equal(outcome,a['outcomes'])
        equal(tie_uniform,a['uniform_outcomes'])
        equal(retention,a['retained'])
        equal(opp,a['opportunity'])
        equal(rows,a['rows'])
        for family in ['P2','Fixed','Random']:
            equal(np.concatenate(chosen_by_family[family],axis=1),a[family+'_choices'])
        tasks=a['task']
        fold=a['fold']
        alt=a['alternative']
        gate=a['A3_mask']
        saved_labels=a['labels']
    result=json.loads((WORK/'public/a/E7_A_RESULTS.json').read_text())
    for index,row in enumerate(result['configurations']):
        equal(row['accuracy'],outcome[index].mean())
        equal(row['uniform_tie_accuracy'],tie_uniform[index].mean())
        equal(row['retained_opportunity_records'],retention[index].sum())
        equal(row['opportunity_retention'],retention[index].sum()/opp.sum())
    equal(result['observable_opportunity_records'],opp.sum())
    equal(result['original_questions_with_any_opportunity'],opp.any(axis=1).sum())
    decomposition=result['two_choice_decomposition']
    equal(decomposition['lost_before_gate_records'],np.sum(opp&(alt==0)))
    equal(decomposition['retained_but_A3_declined_records'],np.sum(opp&(alt==1)&(~gate[:,None])))
    equal(decomposition['A3_corrected_records'],np.sum(opp&(alt==1)&gate[:,None]))
    equal(decomposition['A3_harmed_records'],np.sum((baseline==1)&(alt==0)&gate[:,None]))
    with np.load(WORK/'private/a/OPPORTUNITY_BOOTSTRAP.npz') as a:
        saved=a['bootstrap']
    rng=np.random.default_rng(np.random.SeedSequence([20261112,752]))
    for rep in range(5000):
        indices=np.concatenate([rng.choice(cell,len(cell),replace=True)
            for f in range(5) for task in ['gsm8k','bbh']
            for cell in [np.flatnonzero((fold==f)&(tasks==task))]])
        accuracy=outcome[:,indices].mean(axis=(1,2))
        base=baseline[indices].mean()
        full=outcome[3,indices].mean()
        retained_rate=retention[:,indices].sum(axis=(1,2))/opp[indices].sum()
        expected=np.column_stack((accuracy,accuracy-base,accuracy-full,retained_rate))
        equal(saved[rep],expected)
    output=dict(status='PASS',scalar_comparisons=count,maximum_absolute_error=maxerror,
        folds=5,original_questions=600,question_generation_repeats=3,
        random_sets_per_question=200,held_repeat_choices_reconstructed=True,
        independent_uniform_tie_reconstruction=True,
        candidate_nested_sizes=[2,4,6,20],bootstrap_records_reconstructed=5000,
        A3_opportunity_decomposition_reconstructed=True,
        source_sha256=hashlib.sha256((WORK/'public/a/run_opportunity.py').read_bytes()).hexdigest(),
        scope='Post-freeze old-development outcome diagnosis. No deployment or stable-population upper-bound claim')
    target=WORK/'public/theory/CANDIDATE_OPPORTUNITY_AUDIT.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':
    main()
