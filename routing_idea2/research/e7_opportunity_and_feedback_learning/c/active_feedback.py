"""E7 pool feedback acquisition with a fixed twenty-output Gaussian learner.

The planner receives only a working posterior and input indices. Raw rewards
and source output lengths are accessible through a permission-checked paid
package environment. Evaluation is a separate operation after path freezing.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import argparse
import json
import hashlib
from time import perf_counter,process_time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
sys.path.insert(0,str(PUBLIC))
sys.path.insert(0,str(PUBLIC/'b'))
from e7_common import load_train,splits,put,sha,resources,argmax
from e7_models import ModelSpec

METHODS=['Random','ParameterIG','DecisionAware']
KINDS=['P1','P3']
ALGSEEDS=[17,29,43]
CHECKPOINTS=[0,20,40,60]


def fantasies(fold,seed,step,pairs=32):
    rng=np.random.default_rng(np.random.SeedSequence([20261112,fold,seed,step,731]))
    z=rng.standard_normal((pairs,20))
    return np.concatenate((z,-z))


def acquisition_scores(model,pool,refs,weights,method,z=None,chunk=12):
    """The same scalar kernel/noise governs real and hypothetical updates."""
    pool=np.asarray(pool,int)
    latent=np.maximum(0,np.diag(model.cov)[pool])
    noise=model.spec.noise
    if method=='ParameterIG':
        return 10*np.log1p(latent/noise)
    if method!='DecisionAware':
        raise ValueError('Prospective scoring supports IG or decision value')
    refmean=model.mean[refs]
    before=float(refmean.max(axis=1)@weights)
    result=np.empty(len(pool))
    for start in range(0,len(pool),chunk):
        qs=pool[start:start+chunk]
        coefficient=model.cov[np.ix_(refs,qs)].T/np.sqrt(np.maximum(0,np.diag(model.cov)[qs])+noise)[:,None]
        # One twenty-arm innovation is shared by every reference for a given
        # fantasy. All candidate measurements use common random numbers.
        changes=coefficient[:,None,:,None]*z[None,:,None,:]
        after=(refmean[None,None]+changes).max(axis=-1)@weights
        result[start:start+len(qs)]=np.maximum(0,after.mean(axis=1)-before)
    if not np.isfinite(result).all():
        raise ArithmeticError('Nonfinite prospective decision scores')
    return result


def choose(scores,pool,priority):
    tied=np.flatnonzero(scores>=np.max(scores)-1e-12)
    return int(tied[np.argmin(priority[np.asarray(pool)[tied]])])


class PacketEnvironment:
    def __init__(self,data,initial,pool,evaluation):
        self.__data=data
        self.initial=np.asarray(initial,int)
        self.allowed=set(map(int,pool))
        self.evaluation=set(map(int,evaluation))
        self.purchased=[]
        self.access_log=[]
        self.ledger=[]
        self.raw_feedback=[]
        self.mean_estimated_cost={k:3*float(v[:,self.initial].sum(axis=-1).mean()) for k,v in data['cost'].items()}

    def historical(self):
        self.access_log.append(dict(role='historical_initialization',questions=self.initial.tolist(),raw_reward_values=240*3*20))
        return self.__data['Y'][self.initial].copy()

    def buy(self,q):
        q=int(q)
        if q not in self.allowed or q in self.purchased or q in self.evaluation:
            self.access_log.append(dict(role='denied_reward_access',question=q))
            raise PermissionError('Unpaid or duplicate package request')
        if len(self.purchased)>=60:
            raise PermissionError('The frozen Gold cap is exhausted')
        self.purchased.append(q)
        self.access_log.append(dict(role='paid_feedback',question=q,purchase_number=len(self.purchased),raw_reward_values=60))
        received=self.__data['Y'][q].copy()
        cost=resources(self.__data,[q],[range(20)],gold=1)
        if cost['member_calls']!=18 or cost['aggregator_calls']!=60 or cost['scoring_ops']!=60:
            raise AssertionError('Wrong full package execution unit')
        self.raw_feedback.append(received)
        self.ledger.append(cost)
        return received,cost


def path_name(f,kind,method,seed):
    return f'f{f}_{kind}_{method}_s{seed}'


def run_path(args):
    f,kind,method,seed,steps,qualification=args
    name=path_name(f,kind,method,seed)
    folder=PRIVATE/('qualification' if qualification else 'c')/name
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'FROZEN_PATH.json'
    protocol=PUBLIC/'E7_PROTOCOL.json'
    protocol_sha=sha(protocol)
    source_sha=sha(__file__)
    specpath=PRIVATE/f'b/SPEC_f{f}_{kind}.npz'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['status']=='FROZEN_BEFORE_E_REWARD_EVALUATION' and old['steps']==steps and old['protocol_sha256']==protocol_sha and old['source_sha256']==source_sha and old['spec_sha256']==sha(specpath):
            return dict(path=name,status='REUSED_FROZEN_PATH',wall_s=old['wall_s'])
        raise RuntimeError('Conflicting path receipt must be diagnosed before reuse')
    frozen=json.loads((PRIVATE/'PROTOCOL_FROZEN.json').read_text())
    assert frozen['protocol_sha256']==protocol_sha
    roles=splits(f)
    I,U,E,refs,weights=(roles[k] for k in ['I','U','E','refs','weights'])
    data,source=load_train()
    env=PacketEnvironment(data,I,U,E)
    spec=ModelSpec.load(specpath)
    assert np.array_equal(spec.initial,I)
    model=spec.posterior(I,env.historical())
    initial_cost=resources(data,I,[range(20)]*len(I),gold=len(I))
    rng=np.random.default_rng(np.random.SeedSequence([20261112,f,seed,733]))
    ordering=rng.permutation(U)
    priority=np.full(600,10000,dtype=int)
    priority[ordering]=np.arange(len(ordering))
    means=[]
    snapshot_steps=[]
    selected=[]
    planned_scores=[]
    score_indices=[]
    prospective=[]
    selected_score=[]
    planning_cpu=[]
    update_cpu=[]
    route_before=[]
    route_after=[]
    variance_traces=[]
    start,cpu=perf_counter(),process_time()
    def snapshot(step):
        model.save_state(folder/f'MODEL_B{step}.npz',include_covariance=False)
        means.append(model.mean[E].copy())
        snapshot_steps.append(step)
    snapshot(0)
    for step in range(1,steps+1):
        remaining=U[~np.isin(U,selected)]
        started=process_time()
        if method=='Random':
            scores=np.zeros(len(remaining))
            index=int(rng.integers(len(remaining)))
        else:
            z=fantasies(f,seed,step) if method=='DecisionAware' else None
            scores=acquisition_scores(model,remaining,refs,weights,method,z)
            index=choose(scores,remaining,priority)
        planning_cpu.append(process_time()-started)
        q=int(remaining[index])
        fullscores=np.full(600,np.nan)
        fullscores[remaining]=scores
        planned_scores.append(fullscores)
        selected_score.append(float(scores[index]))
        predicted=model.mean[q].copy()
        var=float(model.cov[q,q])
        cross_ref=model.cov[refs,q].copy()
        before=argmax(model.mean[E])
        # The complete prospective choice and scores are saved before reveal.
        prospective.append(dict(step=step,question=q,method=method,score=float(scores[index]),
            remaining=len(remaining),latent_variance=var,noise=spec.noise,
            reference_covariance=cross_ref.tolist(),predicted_mean=predicted.tolist()))
        put(folder/'PROSPECTIVE_CHOICES.json',prospective)
        received,cost=env.buy(q)
        update_start=process_time()
        model.update(q,received)
        update_cpu.append(process_time()-update_start)
        selected.append(q)
        route_before.append(before)
        route_after.append(argmax(model.mean[E]))
        variance_traces.append(np.diag(model.cov)[E].copy())
        if step in CHECKPOINTS or qualification and step==steps:
            snapshot(step)
            put(folder/'PROGRESS.json',dict(path=name,completed_packages=step,total=steps,wall_s=perf_counter()-start,
                Gold=step,member_calls=18*step,aggregate_calls=60*step,physical_llm_calls=0))
    if qualification:
        batch=spec.posterior(np.r_[I,selected],np.concatenate((data['Y'][I],np.asarray(env.raw_feedback)),axis=0))
        mean_error=float(np.max(abs(batch.mean-model.mean)))
        covariance_error=float(np.max(abs(batch.cov-model.cov)))
        assert mean_error<1e-10 and covariance_error<1e-10
        try:
            env.buy(int(E[0]))
            raise AssertionError('Unauthorized E label was accepted')
        except PermissionError:
            pass
    else:
        mean_error=covariance_error=None
    rawpath=folder/'PATH_RECORDS.npz'
    np.savez_compressed(rawpath,I=I,U=U,E=E,refs=refs,reference_weights=weights,
        purchased=np.asarray(selected),raw_feedback=np.asarray(env.raw_feedback),scores=np.asarray(planned_scores),
        snapshot_steps=snapshot_steps,E_predictions=np.asarray(means),
        route_before=np.asarray(route_before),route_after=np.asarray(route_after),
        variance_E=np.asarray(variance_traces),planning_cpu=planning_cpu,update_cpu=update_cpu,
        selected_score=selected_score)
    put(folder/'ACCESS_AND_COST_LOG.json',dict(access_log=env.access_log,ledger=env.ledger,initial_shared_cost=initial_cost,
        I_only_estimated_package=env.mean_estimated_cost,feedback_permission='full20 x3 scores per original question',
        Q_reward_reads_in_actor=0,source_input_identity=source['identity']))
    metadata=dict(status='FROZEN_BEFORE_E_REWARD_EVALUATION',path=name,fold=f,learner=kind,method=method,algorithm_seed=seed,
        steps=steps,checkpoints=snapshot_steps,source_sha256=source_sha,protocol_sha256=protocol_sha,
        spec_sha256=sha(specpath),record_sha256=sha(rawpath),snapshots={f'MODEL_B{b}.npz':sha(folder/f'MODEL_B{b}.npz') for b in snapshot_steps},
        planning_cpu_s=float(np.sum(planning_cpu)),update_cpu_s=float(np.sum(update_cpu)),
        wall_s=perf_counter()-start,total_cpu_s=process_time()-cpu,physical_new_llm_calls=0,
        qualification=dict(sequential_batch_mean_error=mean_error,sequential_batch_covariance_error=covariance_error) if qualification else None,
        E_rewards_used_for_any_purchase=False,observed_questions=len(I)+steps)
    put(receipt,metadata)
    print(json.dumps(dict(path=name,status='FROZEN',steps=steps,planning_s=metadata['planning_cpu_s'],wall_s=metadata['wall_s'])),flush=True)
    return dict(path=name,status='FROZEN',wall_s=metadata['wall_s'])


def qualify():
    results=[run_path((0,'P3',method,17,10,True)) for method in METHODS]
    put(PUBLIC/'c/E7_C_QUALIFICATION.json',dict(status='PASS',outer_fold=0,packages=10,methods=METHODS,
        no_E_outcome_scoring=True,paths=results,formal_path_limit=90))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['qualify','run'])
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if args.stage=='qualify':
        qualify()
        return
    q=json.loads((PUBLIC/'c/E7_C_QUALIFICATION.json').read_text())
    assert q['status']=='PASS'
    tasks=[(f,k,m,s,60,False) for f in range(5) for k in KINDS for m in METHODS for s in ALGSEEDS]
    assert len(tasks)==90
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(run_path,tasks))
    put(PRIVATE/'c/ALL_PATHS_FROZEN.json',dict(status='ALL90_FROZEN_BEFORE_EVALUATION',paths=results,
        path_count=90,new_llm_calls=0,protocol_sha256=sha(PUBLIC/'E7_PROTOCOL.json')))


if __name__=='__main__':
    main()
