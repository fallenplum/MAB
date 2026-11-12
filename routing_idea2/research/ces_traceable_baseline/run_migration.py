"""Bounded zero-API CES author-code migration on existing V2 caches."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import json
import argparse
from time import perf_counter,process_time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/ces_traceable_baseline'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
sys.path.insert(0,str(PUBLIC))
from ces_common import *

TRACKS=['ProfileOnly-SingleCombo','HistoryFit-SingleCombo','HistoryFit-Full20']
SEEDS=[17,29,43]
CHECKPOINTS=[0,20,40,60]
TEAM_IDS={tuple(team):i for i,team in enumerate(TEAMS)}


def save_state(path,agent,received_questions):
    np.savez_compressed(path,V_inv=agent.V_inv,b=agent.b,w_hat=agent.w_hat,
        alpha=np.array(agent.alpha),ridge=np.array(agent.lam),received_questions=np.asarray(received_questions))


def select(agent,phis):
    chosen=agent.select_topk(list(phis),k=3,costs=None,lam_cost=0.)
    members=tuple(sorted(map(int,chosen)))
    assert members in TEAM_IDS
    return TEAM_IDS[members],members


def scores(agent,phi):
    actions=[]
    means=[]
    selected=[]
    for row in phi:
        q,m=select(agent,row)
        actions.append(q)
        selected.append(m)
        means.append([float(np.sum(row[list(team)],axis=0)@agent.w_hat) for team in TEAMS])
    return np.asarray(actions),np.asarray(selected),np.asarray(means)


def prepare():
    config=json.loads((PUBLIC/'CES_PROTOCOL.json').read_text())
    assert config['trials']['formal_paths']==45
    data,refs=load_train()
    member,member_source=load_member_scores(data)
    with np.load(PRIVATE/'MANUAL_INPUTS.npz') as a:
        manual=a['features']
        assert np.array_equal(a['question_ids'],data['ids'])
    Core,build_phi,initialize,Encoder=author_operators()
    output=PRIVATE/'prepared'
    output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for f in range(5):
        roles=splits(f)
        I=roles['I']
        ordered=np.asarray(sorted(I,key=lambda q:str(data['ids'][q])),int)
        for seed in SEEDS:
            started=perf_counter()
            phi,feature=make_features(data,member,I,seed,manual)
            native=Core(p=96,alpha=3.5,lam=.1)
            warm=Core(p=96,alpha=3.5,lam=.1)
            qmean=data['Y'][ordered].mean(axis=1)
            for q,row in zip(ordered,qmean):
                for team,reward in zip(TEAMS,row):
                    z=phi[q,list(team)].sum(axis=0)
                    warm.update(z,float(reward))
            feature_path=output/f'FEATURES_f{f}_s{seed}.npz'
            np.savez_compressed(feature_path,phi=phi,I=I,E=roles['E'],U=roles['U'],**feature)
            save_state(output/f'WARM_f{f}_s{seed}.npz',warm,ordered)
            # A deterministic full-data ridge solution independently checks the
            # author update sequence without replacing the preserved actor.
            Z=np.asarray([phi[q,list(team)].sum(axis=0) for q in ordered for team in TEAMS])
            y=qmean.reshape(-1)
            dense=np.linalg.solve(.1*np.eye(96)+Z.T@Z,Z.T@y)
            error=float(np.max(abs(dense-warm.w_hat)))
            assert error<1e-8
            profile_cost=resources(data,I,[[]]*len(I),gold=240)
            # An empty action list represents no generation, so price member
            # profiling explicitly from all6 source executions at each seed.
            for key in ['member_calls','input_tokens','output_tokens','source_model_wall_s']:
                suffix={'member_calls':'execution_counts','input_tokens':'prompt_tokens',
                        'output_tokens':'completion_tokens','source_model_wall_s':'wall_time_s'}[key]
                value=data['cost']['member_'+suffix][:,I].sum()
                profile_cost[key]=float(value) if key=='source_model_wall_s' else int(value)
            profile_cost['tokens']=profile_cost['input_tokens']+profile_cost['output_tokens']
            profile_cost['scoring_ops']=6*3*len(I)
            full_cost=resources(data,I,[range(20)]*len(I),gold=240)
            full_cost['scoring_ops']+=6*3*len(I)
            pathdata=[]
            for kind in ['P1','P3']:
                old=E7/f'private/c/f{f}_{kind}_Random_s{seed}/PATH_RECORDS.npz'
                with np.load(old) as a:
                    pathdata.append(a['purchased'])
            assert np.array_equal(pathdata[0],pathdata[1])
            assert len(set(pathdata[0]))==60 and set(pathdata[0])<=set(roles['U'])
            report=dict(fold=f,seed=seed,feature_sha256=sha(feature_path),warm_sha256=sha(output/f'WARM_f{f}_s{seed}.npz'),
                historical_mean_observations=4800,raw_aggregate_scoring_ops=14400,mean_dense_error=error,
                profile_cost=profile_cost,full_initial_cost=full_cost,
                question_schedule=pathdata[0].tolist(),schedule_source_sha256=sha(E7/f'private/c/f{f}_P1_Random_s{seed}/PATH_RECORDS.npz'),
                prepare_wall_s=perf_counter()-started)
            put(output/f'PREPARED_f{f}_s{seed}.json',report)
            reports.append(report)
    put(PRIVATE/'PREPARED_ALL.json',dict(status='COMPLETE',models=reports,source_pins=SOURCE_PINS,
        protocol_sha256=sha(PUBLIC/'CES_PROTOCOL.json'),member_source=member_source))
    print(json.dumps(dict(status='PREPARED',feature_geometries=15,warmfits=15)),flush=True)


class FeedbackPool:
    def __init__(self,data,I,U,E):
        self.__data=data
        self.I=set(map(int,I))
        self.U=set(map(int,U))
        self.E=set(map(int,E))
        self.paid=set()
        self.log=[]
        self.cost=[]
        self.labels=[]

    def buy(self,q,actions):
        q=int(q)
        actions=tuple(sorted(set(map(int,actions))))
        if q not in self.U or q in self.paid or q in self.E or not 1<=len(actions)<=20:
            raise PermissionError('Unpaid, duplicate or evaluator feedback')
        if len(self.paid)>=60:
            raise PermissionError('Gold budget exhausted')
        self.paid.add(q)
        # Only this already selected package is returned to the learner.
        received=self.__data['Y'][q][:,actions].copy()
        actual=resources(self.__data,[q],[actions],gold=1)
        self.log.append(dict(role='paid_U_feedback',question=q,actions=list(actions),raw_reward_count=3*len(actions),Gold=1))
        self.cost.append(actual)
        self.labels.append(dict(question=q,actions=list(actions),raw=received.tolist()))
        return received.mean(axis=0),actual


def run_path(args):
    f,seed,track=args
    started,cpu=perf_counter(),process_time()
    data,refs=load_train()
    roles=splits(f)
    I,U,E=(roles[key] for key in ['I','U','E'])
    prep=PRIVATE/'prepared'
    meta=json.loads((prep/f'PREPARED_f{f}_s{seed}.json').read_text())
    with np.load(prep/f'FEATURES_f{f}_s{seed}.npz') as a:
        phi=a['phi']
    Core,_,_,_=author_operators()
    agent=Core(p=96,alpha=3.5,lam=.1)
    if track!='ProfileOnly-SingleCombo':
        with np.load(prep/f'WARM_f{f}_s{seed}.npz') as a:
            agent.load_state(V_inv=a['V_inv'],b=a['b'],w_hat=a['w_hat'])
    folder=PRIVATE/'replay'/f'f{f}_{track}_s{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    pool=FeedbackPool(data,I,U,E)
    decisions=[]
    E_actions=[]
    E_mean_actions=[]
    E_predictions=[]
    snapshots=[]
    planning=[]
    updating=[]
    Uactions=[]
    def snapshot(step):
        save_state(folder/f'MODEL_B{step}.npz',agent,meta['question_schedule'][:step])
        chosen,members,predictions=scores(agent,phi[E])
        previous_alpha=agent.alpha
        agent.alpha=0.
        mean_chosen,mean_members,unused=scores(agent,phi[E])
        agent.alpha=previous_alpha
        E_actions.append(chosen)
        E_mean_actions.append(mean_chosen)
        E_predictions.append(predictions)
        snapshots.append(step)
    snapshot(0)
    for step,q in enumerate(meta['question_schedule'],start=1):
        planstart=process_time()
        current,members=select(agent,phi[q])
        actions=tuple(range(20)) if track=='HistoryFit-Full20' else (current,)
        planning.append(process_time()-planstart)
        decisions.append(dict(step=step,question=q,served_combo=current,members=list(members),feedback_actions=list(actions),
            prior_quality_mean=float(phi[q,list(members)].sum(axis=0)@agent.w_hat)))
        put(folder/'PROSPECTIVE_DECISIONS.json',decisions)
        values,cost=pool.buy(q,actions)
        updatestart=process_time()
        for combo,reward in zip(actions,values):
            agent.update(phi[q,list(TEAMS[combo])].sum(axis=0),float(reward))
        updating.append(process_time()-updatestart)
        Uactions.append(current)
        if step in CHECKPOINTS:
            snapshot(step)
    raw=folder/'FROZEN_PATH.npz'
    np.savez_compressed(raw,I=I,U=U,E=E,purchased=meta['question_schedule'],U_selected=Uactions,
        E_actions=np.asarray(E_actions),E_mean_actions=np.asarray(E_mean_actions),E_predicted_means=np.asarray(E_predictions),
        checkpoints=snapshots,planning_CPU=planning,update_CPU=updating)
    put(folder/'FEEDBACK_ACCESS_AND_COST.json',dict(log=pool.log,labels=pool.labels,cost=pool.cost,
        initial_cost=meta['profile_cost'] if track=='ProfileOnly-SingleCombo' else meta['full_initial_cost'],
        initial_aggregate_mean_observations=0 if track=='ProfileOnly-SingleCombo' else 4800,
        initial_profile_member_values=4320,query_Y_reads=0))
    receipt=dict(status='FROZEN_BEFORE_EVALUATION',fold=f,seed=seed,track=track,path=folder.name,
        record_sha256=sha(raw),protocol_sha256=sha(PUBLIC/'CES_PROTOCOL.json'),core_source_sha256=SOURCE_PINS,
        feature_sha256=meta['feature_sha256'],warm_source_sha256=meta['warm_sha256'] if track!='ProfileOnly-SingleCombo' else None,
        source_sha256=sha(__file__),checkpoints=snapshots,snapshot_sha256={f'MODEL_B{b}.npz':sha(folder/f'MODEL_B{b}.npz') for b in snapshots},
        planning_CPU_s=float(np.sum(planning)),update_CPU_s=float(np.sum(updating)),
        total_CPU_s=process_time()-cpu,wall_s=perf_counter()-started,new_LLM_calls=0)
    put(folder/'FROZEN_PATH.json',receipt)
    print(json.dumps(dict(path=folder.name,status='FROZEN',wall_s=receipt['wall_s'])),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare()
        return
    jobs=[(f,seed,track) for f in range(5) for seed in SEEDS for track in TRACKS]
    assert len(jobs)==45
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results=list(executor.map(run_path,jobs))
    put(PRIVATE/'ALL45_FROZEN.json',dict(status='ALL45_FROZEN_BEFORE_EVALUATION',paths=results,
        protocol_sha256=sha(PUBLIC/'CES_PROTOCOL.json'),new_LLM_calls=0))


if __name__=='__main__':
    main()
