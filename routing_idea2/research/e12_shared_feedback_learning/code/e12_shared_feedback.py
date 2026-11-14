"""E12 total-budget learning with shared legal rewards and frozen H480 features."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

ASSETS=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
ROOT=ASSETS/'research/e12_shared_feedback_learning'
E7=ASSETS/'research/e7_opportunity_and_feedback_learning'
E8=ASSETS/'research/e8_decision_focused_combination_learning'
E9=ASSETS/'research/e9_consistent_cost_sensitive_learning'
E10=ASSETS/'research/e10_input_prefill_matching_audit'
E11=ASSETS/'research/e11_group_mean_shrinkage'
sys.path.insert(0,str(E9/'public/code'))
from e9_common import ACTION_KEYS,fold_packs
from csce_operators import semantic_argmax,validate_training_inputs
from e8_assets import load_historical

BUDGETS=(60,120,240,300,360,480)
METHODS=('M_raw','M_group_all','P_raw','P_group_all','P_group_revealed','Global_B')
ETA=1000/3
STATE_KEYS=('pca_mean','pca_components','scaler_mean','scaler_scale')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda item:item.tolist() if isinstance(item,np.ndarray) else item.item())+'\n',encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def archive(path,keys=None):
    with np.load(path,allow_pickle=False) as source:
        return {key:source[key].copy() for key in (source.files if keys is None else keys)}


def transform(values,state):
    projected=(np.asarray(values,dtype=float)-state['pca_mean'])@state['pca_components'].T
    scaled=(projected-state['scaler_mean'])/state['scaler_scale']
    return np.column_stack((np.ones(len(scaled)),scaled))


def asset_paths():
    return {
        'E7_splits':E7/'private/E7_SPLITS_FROZEN.npz',
        'prefill_features':E10/'private/features/PREFILL_FEATURES_FROZEN.npz',
        'E10_predictions':E10/'private/ALL_E10_PREDICTIONS_FROZEN.npz',
        'E11_predictions':E11/'private/ALL_PREDICTIONS_FROZEN.npz',
        'E9_background':E9/'private/ALL_E9_PREDICTIONS_FROZEN.npz',
        **{f'M_model{f}':E8/f'private/training/fold{f}/M0_tuned/model_state.npz' for f in range(5)},
        **{f'P_model{f}':E10/f'private/models/fold{f}/R_full/MODEL.npz' for f in range(5)},
        **{f'Group_model{f}':E11/f'private/fold{f}/GROUP_AND_HALF_MODEL.npz' for f in range(5)},
        **{f'RBF_predictions{f}':E7/f'private/b/FROZEN_B_f{f}.npz' for f in range(5)}}


def verify_sources():
    protocol=read(ROOT/'public/E12_PROTOCOL.json')
    assert sha(Path(__file__))==protocol['source_sha256']
    paths=asset_paths()
    for name,path in paths.items():
        assert sha(path)==protocol['source_assets'][name],name
    assert sha(paths['E7_splits'])=='48e631682448fca1cd6d53dd2c2c29dadda33cdbeb86d136ae3379a74a94f29a'
    return protocol,paths


def nested_masks(I,U,ids,task,sub):
    def ordered(rows):
        return sorted(map(int,rows),key=lambda index:hashlib.sha256(('E7-I-v1|'+ids[index]).encode()).hexdigest())
    gsm=ordered(I[task[I]=='gsm8k'])
    groups=[ordered(I[(task[I]=='bbh')&(sub[I]==label)]) for label in sorted(set(sub[I][task[I]=='bbh']))]
    bbh=[group[position] for position in range(max(map(len,groups))) for group in groups if position<len(group)]
    assert set(gsm[:120]+bbh[:120])==set(map(int,I))
    masks={60:np.array(gsm[:30]+bbh[:30]),120:np.array(gsm[:60]+bbh[:60]),240:I.copy(),
        300:np.r_[I,U[:60]],360:np.r_[I,U[:120]],480:np.sort(np.r_[I,U])}
    previous=set()
    for budget,rows in masks.items():
        assert len(rows)==len(set(rows))==budget
        assert previous<=set(rows)
        assert sum(task[rows]=='gsm8k')==sum(task[rows]=='bbh')==budget//2
        if budget in (60,120,240):
            support=sorted(int(sum(sub[rows[task[rows]=='bbh']]==label)) for label in sorted(set(sub[I][task[I]=='bbh'])))
            expected={60:[1]*24+[2]*3,120:[2]*21+[3]*6,240:[4]*15+[5]*12}[budget]
            assert support==expected,(budget,support)
        previous=set(rows)
    return masks


class HistoryFeedback:
    """Only the cache manager owns unrevealed H rewards."""
    def __init__(self,history,evaluation):
        self.__rewards=np.asarray(history['Y'],dtype=float)
        self.__positions={int(index):position for position,index in enumerate(history['global_indices'])}
        self.__evaluation=set(map(int,evaluation['global_indices']))
        self.access_log=[]

    def reveal(self,indices):
        indices=np.asarray(indices,dtype=int)
        assert len(indices)==len(set(indices))
        assert not set(map(int,indices))&self.__evaluation
        assert set(map(int,indices))<=set(self.__positions)
        positions=np.array([self.__positions[int(index)] for index in indices])
        self.access_log.append(dict(total_budget=len(indices),reward_values_revealed=60*len(indices),
            global_indices=indices.tolist(),reveal_scope='only this S_B, cumulative nested scenario'))
        return positions,self.__rewards[positions].copy()


def group_means(observed_mean,groups,group_count):
    means=np.empty((group_count,20));counts=np.bincount(groups,minlength=group_count)
    fallback=[]
    for group in range(group_count):
        selected=groups==group
        if selected.any():
            means[group]=observed_mean[selected].mean(axis=0)
        else:
            means[group]=observed_mean.mean(axis=0)
            fallback.append(group)
    return means,counts,fallback


def solve(matrix,rhs,kind,logs,fold,budget=None):
    cpu,wall=time.process_time(),time.perf_counter()
    value=np.linalg.solve(matrix,rhs)
    logs.append(dict(kind=kind,fold=fold,budget=budget,cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.perf_counter()-wall,rhs_columns=rhs.shape[1]))
    assert float(abs(matrix@value-rhs).max())<1e-8
    return value


def train():
    started=time.perf_counter();cpu_started=time.process_time()
    protocol,paths=verify_sources()
    roles=archive(paths['E7_splits'])
    features=archive(paths['prefill_features'],('full','global_indices'))
    assert np.array_equal(features['global_indices'],np.arange(600))
    # Inputs-only source supplies metadata for deterministic masks.
    input_path=ASSETS/'artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz'
    assert sha(input_path)=='57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3'
    inputs=archive(input_path,('question_ids','task_labels','subtask_labels'))
    ids=inputs['question_ids'].astype(str);task=inputs['task_labels'].astype(str);sub=inputs['subtask_labels'].astype(str)
    scores=np.empty((len(METHODS),len(BUDGETS),600,20));actions=np.empty((len(METHODS),len(BUDGETS),600),dtype=int)
    rbf_actions=np.empty((4,600),dtype=int);fold_ids=np.full(600,-1,dtype=int)
    all_logs=[];supports=[];endpoint_checks=[]
    old9=archive(paths['E9_background'],('actions__A','scores__A','fold'))
    old10=archive(paths['E10_predictions'],('actions__R_full','scores__R_full','fold'))
    old11=archive(paths['E11_predictions'],('actions__R_group','scores__R_group','fold'))
    for fold in range(5):
        history,evaluation,_,references=fold_packs(fold)
        validate_training_inputs(history,evaluation)
        H=history['global_indices'];E=evaluation['global_indices']
        I,U,E7_E=(roles[f'f{fold}_{name}'] for name in ('I','U','E'))
        assert set(H)==set(roles[f'f{fold}_H'])==set(I)|set(U)
        assert set(E)==set(E7_E) and not set(H)&set(E)
        assert np.array_equal(history['ids'],ids[H]) and np.array_equal(evaluation['ids'],ids[E])
        fold_ids[E]=fold
        masks=nested_masks(I,U,ids,task,sub)
        keys=sorted(set(zip(history['task'].tolist(),history['sub'].tolist())))
        assert len(keys)==28
        group_index=np.array([keys.index(pair) for pair in zip(history['task'],history['sub'])])
        Z=np.eye(28)[group_index]
        mstate=archive(paths[f'M_model{fold}'],STATE_KEYS)
        pstate=archive(paths[f'P_model{fold}'],STATE_KEYS)
        phiM_H=transform(history['E'],mstate);phiM_E=transform(evaluation['E'],mstate)
        phiP_H=transform(features['full'][H],pstate);phiP_E=transform(features['full'][E],pstate)
        matrixM=phiM_H.T@phiM_H+ETA*np.eye(17)
        matrixP=phiP_H.T@phiP_H+ETA*np.eye(17)
        TM=solve(matrixM,phiM_H.T@Z,'group_mapping_without_rewards',all_logs,fold)
        TP=solve(matrixP,phiP_H.T@Z,'group_mapping_without_rewards',all_logs,fold)
        env=HistoryFeedback(history,evaluation)
        folder=ROOT/f'private/fold{fold}';folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'GROUP_MAPPINGS.npz',T_M=TM,T_P=TP,group_keys=np.array(keys),H=H,E=E)
        support_fold=[];saved_masks={}
        for b_index,budget in enumerate(BUDGETS):
            S=masks[budget];positions,received=env.reveal(S)
            observed_mean=received.mean(axis=1)
            C,count,fallback=group_means(observed_mean,group_index[positions],28)
            assert not fallback,'Expected full28-group coverage, inspect mask reconstruction'
            XM,XP=phiM_H[positions],phiP_H[positions]
            current={'M_group_all':TM@C,'P_group_all':TP@C}
            if budget<480:
                current['M_raw']=solve(XM.T@XM+ETA*np.eye(17),XM.T@observed_mean,'M_raw',all_logs,fold,budget)
                current['P_raw']=solve(XP.T@XP+ETA*np.eye(17),XP.T@observed_mean,'P_raw',all_logs,fold,budget)
                current['P_group_revealed']=solve(XP.T@XP+ETA*np.eye(17),XP.T@C[group_index[positions]],
                    'P_group_revealed',all_logs,fold,budget)
            else:
                current['M_raw']=archive(paths[f'M_model{fold}'],('coefficient',))['coefficient']
                current['P_raw']=archive(paths[f'P_model{fold}'],('coefficient',))['coefficient']
                current['P_group_revealed']=archive(paths[f'Group_model{fold}'],('W_group',))['W_group']
                for name,phi,matrix,target in (
                    ('M_raw',phiM_H,matrixM,observed_mean),('P_raw',phiP_H,matrixP,observed_mean),
                    ('P_group_revealed',phiP_H,matrixP,C[group_index[positions]])):
                    # H and S480 are identical in the frozen fold order.
                    assert np.array_equal(S,H)
                    assert float(abs(matrix@current[name]-phi.T@target).max())<1e-8
            global_action=semantic_argmax(observed_mean.mean(axis=0,keepdims=True),ACTION_KEYS)[0]
            for name in METHODS:
                index=METHODS.index(name)
                if name=='Global_B':
                    q=np.tile(observed_mean.mean(axis=0),(120,1))
                else:
                    design=phiM_E if name.startswith('M_') else phiP_E
                    q=design@current[name]
                chosen=semantic_argmax(q,ACTION_KEYS)
                scores[index,b_index,E]=q;actions[index,b_index,E]=chosen
            assert np.all(actions[METHODS.index('Global_B'),b_index,E]==global_action)
            if budget==480:
                for name,expected in (('M_raw',old9['actions__A'][E]),('P_raw',old10['actions__R_full'][E]),
                    ('P_group_all',old11['actions__R_group'][E]),('P_group_revealed',old11['actions__R_group'][E])):
                    assert np.array_equal(actions[METHODS.index(name),b_index,E],expected),name
                score_errors={name:float(abs(scores[METHODS.index(name),b_index,E]-expected).max())
                    for name,expected in (('M_raw',old9['scores__A'][E]),('P_raw',old10['scores__R_full'][E]),
                        ('P_group_all',old11['scores__R_group'][E]),('P_group_revealed',old11['scores__R_group'][E]))}
                assert max(score_errors.values())<=1e-12
                group_score_error=float(abs(phiP_E@current['P_group_all']-phiP_E@current['P_group_revealed']).max())
                assert group_score_error<=1e-12
                endpoint_checks.append(dict(fold=fold,group_score_max_error=group_score_error,
                    original_score_max_errors=score_errors,all_endpoint_actions_equal=True))
            slot=folder/f'B{budget}';slot.mkdir()
            np.savez_compressed(slot/'MODELS_AND_GROUP_MEANS.npz',C=C,counts=count,observed_indices=S,
                **current)
            saved_masks[f'B{budget}']=S
            support_fold.append(dict(budget=budget,group_keys=keys,counts=count,fallback_groups=fallback))
        rbf=archive(paths[f'RBF_predictions{fold}'],('E','I','U','main_predictions'))
        assert np.array_equal(rbf['E'],E7_E) and np.array_equal(rbf['I'],I) and np.array_equal(rbf['U'],U)
        old_scores=rbf['main_predictions'][2]
        assert old_scores.shape==(4,120,20)
        old_actions=np.argmax(old_scores>=old_scores.max(axis=-1,keepdims=True)-1e-10,axis=-1)
        rbf_actions[:,E7_E]=old_actions
        np.savez_compressed(folder/'MASKS.npz',**saved_masks)
        put(folder/'SUPPORT_AND_ACCESS.json',dict(support=support_fold,access_log=env.access_log,
            history_source=references,unrevealed_rewards_in_model_arguments=False))
        supports.append(dict(fold=fold,budgets=support_fold))
        print('E12_FOLD_FROZEN',fold,flush=True)
    assert np.all(fold_ids>=0) and len(all_logs)==85
    output=ROOT/'private/ALL_PREDICTIONS_FROZEN.npz'
    assert not output.exists()
    np.savez_compressed(output,ids=ids,fold=fold_ids,scores=scores,actions=actions,
        methods=np.array(METHODS),budgets=np.array(BUDGETS),rbf_actions=rbf_actions,rbf_budgets=np.array((240,300,360,480)))
    put(ROOT/'private/PREDICTION_FREEZE.json',dict(frozen_utc=now(),protocol_sha256=sha(ROOT/'public/E12_PROTOCOL.json'),
        predictions_sha256=sha(output),solves=all_logs,solve_count=85,group_mapping_solves=10,
        supervised_solves=75,derived_group_all_heads=60,reused_full_budget_heads=15,
        endpoint_checks=endpoint_checks,group_support=supports,
        training_cpu_seconds=time.process_time()-cpu_started,training_wall_seconds=time.perf_counter()-started,
        all_new_policies_frozen=True,E_rewards_read_for_training=False))


def feedback_resources(cost,rows,actions=None):
    from itertools import combinations
    teams=tuple(combinations(range(6),3))
    total=dict(gold=len(rows) if actions is None else 0,member_calls=0,aggregator_calls=0,
        input_tokens=0,output_tokens=0,scoring_ops=0,source_model_wall_s=0.)
    for position,row in enumerate(rows):
        selected_actions=list(range(20)) if actions is None else [int(actions[position])]
        members=sorted(set(member for action in selected_actions for member in teams[action]))
        for seed in range(3):
            for stem,selected,call in (('member',members,'member_calls'),('aggregate',selected_actions,'aggregator_calls')):
                for suffix,key in (('execution_counts',call),('prompt_tokens','input_tokens'),
                    ('completion_tokens','output_tokens'),('wall_time_s','source_model_wall_s')):
                    value=cost[stem+'_'+suffix][seed,row,selected].sum()
                    total[key]+=float(value) if suffix=='wall_time_s' else int(value)
            if actions is None:
                total['scoring_ops']+=len(selected_actions)
    total['tokens']=total['input_tokens']+total['output_tokens']
    return total


def evaluate():
    started=time.perf_counter();cpu_started=time.process_time()
    protocol,_=verify_sources()
    freeze=read(ROOT/'private/PREDICTION_FREEZE.json')
    path=ROOT/'private/ALL_PREDICTIONS_FROZEN.npz'
    assert sha(path)==freeze['predictions_sha256'] and freeze['solve_count']==85
    saved=archive(path);opened=now();data,refs,_=load_historical(ASSETS)
    assert np.array_equal(saved['ids'],data['ids'])
    y=np.asarray(data['Y'],dtype=float);task=data['task'];sub=data['sub'];fold=saved['fold']
    # Explicit question/seed indexing keeps each generation inside its question.
    outcomes={name:np.stack([y[np.arange(600),:,saved['actions'][index,b]] for b in range(6)])
        for index,name in enumerate(METHODS)}
    rbf=np.stack([y[np.arange(600),:,saved['rbf_actions'][b]] for b in range(4)])

    def summary(event,action,full_action):
        return dict(accuracy=float(event.mean()),correct_events=int(event.sum()),independent_questions=600,event_denominator=1800,
            by_task={name:dict(accuracy=float(event[task==name].mean()),correct_events=int(event[task==name].sum()),
                questions=int(sum(task==name))) for name in ('gsm8k','bbh')},
            by_generation={str(seed):float(event[:,index].mean()) for index,seed in enumerate((42,137,2024))},
            by_outerfold={str(f):float(event[fold==f].mean()) for f in range(5)},
            bbh_subtasks={str(label):dict(questions=int(sum((task=='bbh')&(sub==label))),
                accuracy=float(event[(task=='bbh')&(sub==label)].mean())) for label in sorted(set(sub[task=='bbh']))},
            action_counts=np.bincount(action,minlength=20),changed_actions_from_own_full_budget=int(sum(action!=full_action)))

    def comparison(left,right):
        difference=left-right
        return dict(gain=float(difference.mean()),corrected_events=int((difference>0).sum()),
            destroyed_events=int((difference<0).sum()),net_events=int(difference.sum()),
            by_task={name:float(difference[task==name].mean()) for name in ('gsm8k','bbh')},
            by_generation={str(seed):float(difference[:,index].mean()) for index,seed in enumerate((42,137,2024))},
            by_outerfold={str(f):float(difference[fold==f].mean()) for f in range(5)})

    curves={name:{str(b):summary(outcomes[name][index],saved['actions'][METHODS.index(name),index],
        saved['actions'][METHODS.index(name),-1]) for index,b in enumerate(BUDGETS)} for name in METHODS}
    references={'RBF_E7':{str(b):summary(rbf[index],saved['rbf_actions'][index],saved['rbf_actions'][-1])
        for index,b in enumerate((240,300,360,480))}}
    assert abs(references['RBF_E7']['480']['accuracy']-.8677777777777778)<1e-12
    assert abs(curves['M_raw']['480']['accuracy']-.8527777777777777)<1e-12
    assert abs(curves['P_raw']['480']['accuracy']-.8633333333333333)<1e-12
    assert abs(curves['P_group_all']['480']['accuracy']-.87)<1e-12
    assert curves['P_group_revealed']['480']['accuracy']==curves['P_group_all']['480']['accuracy']
    pairs=(('P_group_all','P_raw'),('M_group_all','M_raw'),('P_group_all','M_group_all'),
        ('P_group_revealed','P_raw'),('P_group_all','P_group_revealed'))
    comparisons={f'{left}-{right}':{str(b):comparison(outcomes[left][index],outcomes[right][index])
        for index,b in enumerate(BUDGETS)} for left,right in pairs}
    versus={name:{str(b):dict(versus_P_raw=comparison(outcomes[name][index],outcomes['P_raw'][index]),
        versus_Global_B=comparison(outcomes[name][index],outcomes['Global_B'][index]))
        for index,b in enumerate(BUDGETS)} for name in METHODS}
    versus['RBF_E7']={str(b):dict(versus_P_raw=comparison(rbf[index],outcomes['P_raw'][index+2]),
        versus_Global_B=comparison(rbf[index],outcomes['Global_B'][index+2])) for index,b in enumerate(BUDGETS[2:])}
    aulc={name:dict(six_point=float(np.trapz([curves[name][str(b)]['accuracy'] for b in BUDGETS],BUDGETS)/(480-60)),
        common240_to480=float(np.trapz([curves[name][str(b)]['accuracy'] for b in BUDGETS[2:]],BUDGETS[2:])/(480-240))) for name in METHODS}
    aulc['RBF_E7']=dict(common240_to480=float(np.trapz([references['RBF_E7'][str(b)]['accuracy'] for b in BUDGETS[2:]],BUDGETS[2:])/(480-240)))
    targets={'R_full_at480':.8633333333333333,'RBF_at480':.8677777777777778}
    reaching={name:{label:next((b for b in BUDGETS if curves[name][str(b)]['accuracy']>=threshold-1e-12),None)
        for label,threshold in targets.items()} for name in METHODS}
    costs=[]
    for f in range(5):
        masks=archive(ROOT/f'private/fold{f}/MASKS.npz')
        E=np.flatnonzero(fold==f)
        for b_index,budget in enumerate(BUDGETS):
            feedback=feedback_resources(data['cost'],masks[f'B{budget}'])
            assert feedback['gold']==budget and feedback['member_calls']==18*budget
            assert feedback['aggregator_calls']==feedback['scoring_ops']==60*budget
            services={name:feedback_resources(data['cost'],E,saved['actions'][index,b_index,E]) for index,name in enumerate(METHODS)}
            if budget>=240:
                services['RBF_E7']=feedback_resources(data['cost'],E,saved['rbf_actions'][b_index-2,E])
            costs.append(dict(fold=f,budget=budget,feedback=feedback,counterfactual_E120_service=services))
    result=dict(status='EXPLORATORY_REUSED_TRAIN600',primary='P_group_all minus P_raw at all fixed budgets',
        budgets=list(BUDGETS),methods=curves,historical_references=references,comparisons=comparisons,
        versus_common_baselines=versus,AULC=aulc,first_observed_budget_reaching=reaching,reference_targets=targets,
        group_support=freeze['group_support'],action_keys=ACTION_KEYS,generation_seeds=[42,137,2024],
        task_weighting='GSM8K/BBH .5 each, original-question uniform within task',
        protocol_sha256=sha(ROOT/'public/E12_PROTOCOL.json'),prediction_sha256=freeze['predictions_sha256'],
        groups_use_current_budget_rewards_only=True,group_all_training_input_rows=480,
        evidence_unit='600 questions x3 existing generation outcomes, pseudo-target rows are unlabelled inputs',source_refs=refs)
    put(ROOT/'public/E12_RESULTS.json',result)
    np.savez_compressed(ROOT/'private/PAIRED_OUTCOMES.npz',task=task,sub=sub,fold=fold,
        **{'outcomes__'+name:value for name,value in outcomes.items()},RBF_E7=rbf)
    put(ROOT/'private/EVALUATION_ACCESS.json',dict(first_E_reward_access=opened,predictions_frozen_at=freeze['frozen_utc'],
        validation200_rewards=False,fresh400_rewards=False,sealed200_rewards=False))
    old_encoder_cost=read(E10/'public/COST.json')
    cost=dict(actual_linear_solves=85,reward_free_group_mapping_solves=10,supervised_solves=75,
        derived_group_all_heads=60,reused_full_budget_heads=15,reused_RBF_policies=20,
        new_preprocessing_fits=0,new_prefill=0,new_member_generations=0,new_aggregations=0,new_gold=0,
        cpu_seconds=sum(row['cpu_seconds'] for row in freeze['solves']),solve_wall_seconds=sum(row['wall_seconds'] for row in freeze['solves']),
        training_cpu_seconds=freeze['training_cpu_seconds'],training_wall_seconds=freeze['training_wall_seconds'],
        solve_log=freeze['solves'],per_fold_budget_accounts=costs,
        feedback_ledger_unit='one question,18 member calls,60 aggregate calls and60 final scores',
        feedback_shared_by_all_methods=True,budgets_nested=True,overlapping_folds_are_not_new_procurements=True,
        historical_cache_once=dict(gold=600,member_calls=10800,aggregate_calls=36000),
        historical_prefill=old_encoder_cost['online_increment'],
        historical_prefill_preparation=dict(full_question_features=old_encoder_cost['formal_groups']['full'],
            original_formal_events=606,original_all_input_tokens=49446,
            model_load_wall_seconds=old_encoder_cost['formal']['model_load_wall_seconds'],
            source_cost_sha256=sha(E10/'public/COST.json')),
        historical_MiniLM_encoding=read(E10/'public/CPU_EMBEDDING_REPLAY.json'),
        historical_prefill_head_cpu_profile=read(E10/'public/INFERENCE_CPU_PROFILE.json'),
        deployed_group_and_raw_prefill_encoder_identical=True,
        historical_wall_scope='source call wall sums, not currently measured GPU time',
        evaluation_cpu_seconds=time.process_time()-cpu_started,
        evaluation_wall_seconds=time.perf_counter()-started,
        evaluation_timing_scope='source checks, cached scoring, aggregation and account construction through cost record build')
    put(ROOT/'public/COST.json',cost)
    print(json.dumps(dict(status='COMPLETE',curves={name:{b:item['accuracy'] for b,item in values.items()} for name,values in curves.items()},
        RBF={b:item['accuracy'] for b,item in references['RBF_E7'].items()}),ensure_ascii=False),flush=True)


def verify_outputs():
    verify_sources();freeze=read(ROOT/'private/PREDICTION_FREEZE.json')
    assert sha(ROOT/'private/ALL_PREDICTIONS_FROZEN.npz')==freeze['predictions_sha256']
    result=read(ROOT/'public/E12_RESULTS.json');paired=archive(ROOT/'private/PAIRED_OUTCOMES.npz')
    for name in METHODS:
        for index,budget in enumerate(BUDGETS):
            value=paired['outcomes__'+name][index]
            assert float(value.mean())==result['methods'][name][str(budget)]['accuracy']
            assert int(value.sum())==result['methods'][name][str(budget)]['correct_events']
    print(json.dumps(dict(status='VERIFIED',new_solves=0,new_prefill=0,curves=6,budgets=6)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('run','verify'))
    args=parser.parse_args()
    if args.mode=='run':
        train();evaluate()
    else:
        verify_outputs()
