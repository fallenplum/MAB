"""Independently reconstruct E7-C snapshots, costs, scores and access guards."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import uuid
import numpy as np
from scipy.linalg import cho_factor,cho_solve

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
sys.path.insert(0,str(WORK/'public'))
sys.path.insert(0,str(WORK/'public/b'))
sys.path.insert(0,str(ROOT/'research/e5_paired_gain_validation/public'))
from e7_common import load_train,splits,resources
from e7_models import ModelSpec


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense(spec,rows,labels):
    H=np.asarray(rows,dtype=int)
    factor=cho_factor(spec.K[np.ix_(H,H)]+spec.noise*np.eye(len(H)),lower=True)
    cross=spec.K[:,H]
    mean=spec.offset+cross@cho_solve(factor,labels-spec.offset)
    variance=np.diag(spec.K)-np.sum(cross*cho_solve(factor,cross.T).T,axis=1)
    return mean,variance


def source_module():
    location=WORK/'public/c/active_feedback.py'
    module_spec=importlib.util.spec_from_file_location('independent_E7_C_source',location)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def main():
    data,source_refs=load_train()
    source=WORK/'public/c/active_feedback.py'
    count=0
    maxerror=0.
    def equal(a,b,tol=1e-10):
        nonlocal count,maxerror
        a,b=np.asarray(a),np.asarray(b)
        assert a.shape==b.shape,(a.shape,b.shape)
        nan=np.isnan(a.astype(float))|np.isnan(b.astype(float))
        assert np.array_equal(np.isnan(a.astype(float)),np.isnan(b.astype(float)))
        error=float(np.max(abs(a.astype(float)[~nan]-b.astype(float)[~nan]))) if np.any(~nan) else 0.
        maxerror=max(maxerror,error)
        assert error<tol,error
        count+=a.size
    paths=[]
    for f in range(5):
        roles=splits(f)
        I,U,E=roles['I'],roles['U'],roles['E']
        for kind in ['P1','P3']:
            specpath=WORK/f'private/b/SPEC_f{f}_{kind}.npz'
            spec=ModelSpec.load(specpath)
            for method in ['Random','ParameterIG','DecisionAware']:
                for seed in [17,29,43]:
                    name=f'f{f}_{kind}_{method}_s{seed}'
                    folder=WORK/'private/c'/name
                    receipt=json.loads((folder/'FROZEN_PATH.json').read_text())
                    assert receipt['source_sha256']==digest(source)
                    assert receipt['spec_sha256']==digest(specpath)
                    assert receipt['record_sha256']==digest(folder/'PATH_RECORDS.npz')
                    with np.load(folder/'PATH_RECORDS.npz') as a:
                        raw={key:a[key] for key in a.files}
                    assert len(raw['purchased'])==len(set(raw['purchased']))==60
                    assert set(raw['purchased'])<=set(U)
                    assert not set(raw['purchased'])&set(E)
                    equal(raw['raw_feedback'],data['Y'][raw['purchased']])
                    access=json.loads((folder/'ACCESS_AND_COST_LOG.json').read_text())
                    roles_access=[row['role'] for row in access['access_log']]
                    assert roles_access==['historical_initialization']+['paid_feedback']*60
                    assert access['Q_reward_reads_in_actor']==0
                    equal(access['access_log'][0]['questions'],I)
                    equal([row['question'] for row in access['access_log'][1:]],raw['purchased'])
                    for step,q in enumerate(raw['purchased']):
                        cost=resources(data,[q],[range(20)],gold=1)
                        for key,value in cost.items():
                            equal(value,access['ledger'][step][key])
                        equal(cost['member_calls'],18)
                        equal(cost['aggregator_calls'],60)
                    for key,value in resources(data,I,[range(20)]*240,gold=240).items():
                        equal(value,access['initial_shared_cost'][key])
                    for si,budget in enumerate([0,20,40,60]):
                        state_path=folder/f'MODEL_B{budget}.npz'
                        assert digest(state_path)==receipt['snapshots'][state_path.name]
                        with np.load(state_path) as state:
                            rows=np.r_[I,raw['purchased'][:budget]]
                            labels=np.concatenate((data['Y'][I].mean(axis=1),raw['raw_feedback'][:budget].mean(axis=1)))
                            equal(state['rows'],rows)
                            equal(state['received_means'],labels)
                            equal(state['noise'],spec.noise)
                            equal(state['offset'],spec.offset)
                            mean,variance=dense(spec,rows,labels)
                            equal(state['mean'],mean)
                            equal(state['variance'],variance)
                            equal(raw['E_predictions'][si],mean[E])
                    prospective=json.loads((folder/'PROSPECTIVE_CHOICES.json').read_text())
                    equal([entry['question'] for entry in prospective],raw['purchased'])
                    equal([entry['score'] for entry in prospective],raw['selected_score'])
                    for step,q in enumerate(raw['purchased']):
                        remaining=U[~np.isin(U,raw['purchased'][:step])]
                        assert np.isfinite(raw['scores'][step,remaining]).all()
                        assert np.isnan(raw['scores'][step,np.setdiff1d(np.arange(600),remaining)]).all()
                        equal(prospective[step]['remaining'],len(remaining))
                        equal(prospective[step]['noise'],spec.noise)
                    paths.append(dict(name=name,steps=60,checkpoints=4,hard_gold=60,
                        member_calls=1080,aggregate_calls=3600,all_updates_mean_unit=True))
    # Independent formula comparison at two checkpoints on actual posterior
    # states. This direct candidate loop does not call the actor score routine.
    score_checks=[]
    for kind in ['P1','P3']:
        spec=ModelSpec.load(WORK/f'private/b/SPEC_f0_{kind}.npz')
        role=splits(0)
        for method in ['ParameterIG','DecisionAware']:
            folder=WORK/f'private/c/f0_{kind}_{method}_s17'
            with np.load(folder/'PATH_RECORDS.npz') as a:
                raw={key:a[key] for key in a.files}
            for previous in [0,20]:
                rows=np.r_[raw['I'],raw['purchased'][:previous]]
                labels=np.concatenate((data['Y'][raw['I']].mean(axis=1),raw['raw_feedback'][:previous].mean(axis=1)))
                H=np.asarray(rows,int)
                factor=cho_factor(spec.K[np.ix_(H,H)]+spec.noise*np.eye(len(H)),lower=True)
                cross=spec.K[:,H]
                mean=spec.offset+cross@cho_solve(factor,labels-spec.offset)
                covariance=spec.K-cross@cho_solve(factor,cross.T)
                remaining=raw['U'][~np.isin(raw['U'],raw['purchased'][:previous])]
                if method=='ParameterIG':
                    predicted=np.asarray([.5*np.linalg.slogdet(np.eye(20)+covariance[q,q]/spec.noise*np.eye(20))[1] for q in remaining])
                else:
                    rng=np.random.default_rng(np.random.SeedSequence([20261112,0,17,previous+1,731]))
                    z=rng.standard_normal((32,20))
                    z=np.r_[z,-z]
                    refs=raw['refs']
                    weights=raw['reference_weights']
                    current=float(np.sum(weights*np.max(mean[refs],axis=1)))
                    predicted=[]
                    for q in remaining:
                        shift=covariance[refs,q]/np.sqrt(covariance[q,q]+spec.noise)
                        updates=mean[refs][None]+z[:,None,:]*shift[None,:,None]
                        after=np.sum(np.max(updates,axis=2)*weights[None],axis=1).mean()
                        predicted.append(max(0.,float(after-current)))
                    predicted=np.asarray(predicted)
                equal(predicted,raw['scores'][previous,remaining])
                score_checks.append(dict(kind=kind,method=method,step=previous+1,candidates=len(remaining),max_error=float(np.max(abs(predicted-raw['scores'][previous,remaining])))))
    # Re-run complete10-package prefixes while every unpaid reward/cost and
    # every E reward/cost is NaN. These mutations would expose unauthorized
    # planning, hypothetical labels or future-token lookups.
    actor=source_module()
    temporary=WORK/'private/audit_replays'/('hidden_prefix_nan_'+uuid.uuid4().hex)
    for relative in ['public/E7_PROTOCOL.json','private/PROTOCOL_FROZEN.json',
                     'private/b/SPEC_f0_P1.npz','private/b/SPEC_f0_P3.npz']:
        target=temporary/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(WORK/relative,target)
    actor.PUBLIC=temporary/'public'
    actor.PRIVATE=temporary/'private'
    mutations=[]
    for kind in ['P1','P3']:
        for method in ['Random','ParameterIG','DecisionAware']:
            name=f'f0_{kind}_{method}_s17'
            with np.load(WORK/'private/c'/name/'PATH_RECORDS.npz') as a:
                original={key:a[key] for key in a.files}
            retained=original['purchased'][:10]
            hidden=np.setdiff1d(np.arange(600),np.r_[original['I'],retained])
            poisoned=dict(data)
            poisoned['Y']=data['Y'].copy()
            poisoned['Y'][hidden]=np.nan
            poisoned['cost']={key:np.asarray(value,float).copy() for key,value in data['cost'].items()}
            for value in poisoned['cost'].values():
                value[:,hidden]=np.nan
            actor.load_train=lambda:(poisoned,source_refs)
            actor.run_path((0,kind,method,17,10,True))
            altered=temporary/'private/qualification'/name/'PATH_RECORDS.npz'
            with np.load(altered) as a:
                for key in ['purchased','raw_feedback','scores','route_before','route_after','variance_E','selected_score']:
                    equal(a[key],original[key][:10])
                equal(a['E_predictions'][0],original['E_predictions'][0])
            mutations.append(dict(kind=kind,method=method,retained_paid_packages=10,
                unpaid_U_and_all_E_rewards_NaN=True,unpaid_U_and_all_E_costs_NaN=True,
                choices_scores_and_updates_identical=True))
    result=dict(status='PASS',scalar_comparisons=count,maximum_absolute_error=maxerror,
        formal_paths_reconstructed=len(paths),checkpoints_per_path=4,
        total_paid_packets=90*60,paths=paths,formula_checks=score_checks,
        hidden_reward_cost_mutation=mutations,source_sha256=digest(source),
        units='Original question, three-repeat20-output package; one gold18 members60 aggregates',
        scope='Array-level permission and explicit outcome/cost perturbation checks, not OS-wide file-access forensics')
    (WORK/'public/theory/ACTIVE_FEEDBACK_OPERATOR_ACCESS_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',checks=count,maximum_absolute_error=maxerror,formal_paths=90,mutation_prefixes=len(mutations)),ensure_ascii=False))


if __name__=='__main__':
    main()
