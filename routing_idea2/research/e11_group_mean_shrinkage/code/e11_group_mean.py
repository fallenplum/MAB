"""E11 fixed group-mean supervision, five solves and five derived heads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

ASSETS=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
ROOT=ASSETS/'research/e11_group_mean_shrinkage'
E10=ASSETS/'research/e10_input_prefill_matching_audit'
E9=ASSETS/'research/e9_consistent_cost_sensitive_learning'
sys.path.insert(0,str(E9/'public/code'))
from e9_common import ACTION_KEYS,fold_packs
from csce_operators import semantic_argmax,validate_training_inputs
from e8_assets import load_historical


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda item:item.tolist() if isinstance(item,np.ndarray) else item.item())+'\n',encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def transform(values,state):
    projected=(np.asarray(values,dtype=np.float64)-state['pca_mean'])@state['pca_components'].T
    scaled=(projected-state['scaler_mean'])/state['scaler_scale']
    return np.column_stack((np.ones(len(scaled)),scaled))


def grouped_targets(labels,task,sub):
    keys=sorted(set(zip(task.tolist(),sub.tolist())))
    group_index=np.empty(len(labels),dtype=int)
    means=np.empty((len(keys),labels.shape[1]),dtype=float)
    counts=[]
    for index,(task_name,sub_name) in enumerate(keys):
        rows=(task==task_name)&(sub==sub_name)
        group_index[rows]=index
        means[index]=labels[rows].mean(axis=0)
        counts.append(int(rows.sum()))
    return means[group_index],means,group_index,keys,counts


def tiny_algebra_check():
    phi=np.array([[1.,-1.,0.],[1.,0.,1.],[1.,1.,-1.],[1.,2.,1.]])
    labels=np.array([[0.,1.,.3],[1.,0.,.4],[.4,.6,.7],[.8,.2,.1]])
    grouped=np.vstack([np.tile(labels[:2].mean(axis=0),(2,1)),np.tile(labels[2:].mean(axis=0),(2,1))])
    matrix=phi.T@phi+(1000/3)*np.eye(3)
    # One tiny solve with three RHS blocks, separate from the five formal solves.
    solutions=np.linalg.solve(matrix,phi.T@np.hstack((labels,grouped,.5*(labels+grouped))))
    residual=float(abs(solutions[:,6:]-.5*(solutions[:,:3]+solutions[:,3:6])).max())
    assert residual<1e-12
    return dict(status='PASS',synthetic_questions=4,synthetic_solve_calls=1,
        RHS_target_blocks=3,coefficient_mixture_max_error=residual,tolerance=1e-12)


def source_paths():
    return {
        'features':E10/'private/features/PREFILL_FEATURES_FROZEN.npz',
        'features_receipt':E10/'private/features/FEATURE_FREEZE_RECEIPT.json',
        'predictions':E10/'private/ALL_E10_PREDICTIONS_FROZEN.npz',
        'prediction_receipt':E10/'private/ALL_PREDICTIONS_FREEZE_RECEIPT.json',
        'background':E9/'private/ALL_E9_PREDICTIONS_FROZEN.npz',
        **{f'R_full_model{fold}':E10/f'private/models/fold{fold}/R_full/MODEL.npz' for fold in range(5)}}


def verify_source():
    protocol=read(ROOT/'public/E11_PROTOCOL.json')
    if sha(Path(__file__))!=protocol['source_sha256']:
        raise AssertionError('E11 source differs from fixed protocol')
    paths=source_paths()
    for name,path in paths.items():
        if sha(path)!=protocol['source_assets'][name]:
            raise AssertionError('Reused source differs '+name)
    return protocol,paths


def train():
    started=time.perf_counter();cpu_started=time.process_time()
    protocol,paths=verify_source()
    if (ROOT/'private/ALL_PREDICTIONS_FROZEN.npz').exists():
        raise FileExistsError('Existing E11 predictions are immutable')
    old=arrays(paths['predictions']);features=arrays(paths['features'])
    assert sha(paths['features'])==read(paths['features_receipt'])['feature_sha256']
    assert sha(paths['predictions'])==read(paths['prediction_receipt'])['prediction_sha256']
    assert np.array_equal(features['global_indices'],np.arange(600))
    embeddings=features['full']
    scores={name:np.empty((600,20)) for name in ('R_half','R_group')}
    actions={name:np.empty(600,dtype=int) for name in scores}
    ids=old['ids'].copy();fold=old['fold'].copy();covered=[];logs=[]
    algebra=tiny_algebra_check()
    for outer in range(5):
        history,evaluation,_,refs=fold_packs(outer)
        validate_training_inputs(history,evaluation)
        fitting_rows=history['global_indices'];held_rows=evaluation['global_indices']
        assert np.array_equal(ids[held_rows],evaluation['ids'])
        assert np.array_equal(fold[held_rows],np.full(120,outer))
        assert not set(history['ids'])&set(evaluation['ids'])
        state=arrays(paths[f'R_full_model{outer}'])
        fit_phi=transform(embeddings[fitting_rows],state)
        eval_phi=transform(embeddings[held_rows],state)
        restored=eval_phi@state['coefficient']
        restore_error=float(abs(restored-old['scores__R_full'][held_rows]).max())
        assert restore_error<=1e-12
        assert np.array_equal(semantic_argmax(restored,ACTION_KEYS),old['actions__R_full'][held_rows])
        assert state['coefficient'].shape==(17,20)
        labels=np.asarray(history['Y'],dtype=float).mean(axis=1)
        target_start=time.perf_counter()
        group_labels,means,mapping,keys,counts=grouped_targets(labels,history['task'],history['sub'])
        target_wall=time.perf_counter()-target_start
        assert sum(counts)==480 and len(keys)==28
        matrix=fit_phi.T@fit_phi+(1000/3)*np.eye(17)
        original_residual=float(abs(matrix@state['coefficient']-fit_phi.T@labels).max())
        assert original_residual<1e-8
        before_cpu,before_wall=time.process_time(),time.perf_counter()
        group_coefficient=np.linalg.solve(matrix,fit_phi.T@group_labels)
        solve_wall=time.perf_counter()-before_wall;solve_cpu=time.process_time()-before_cpu
        before=time.perf_counter()
        half_coefficient=.5*state['coefficient']+.5*group_coefficient
        derivation_wall=time.perf_counter()-before
        group_equation_error=float(abs(matrix@group_coefficient-fit_phi.T@group_labels).max())
        half_equation_error=float(abs(matrix@half_coefficient-fit_phi.T@(.5*labels+.5*group_labels)).max())
        assert max(group_equation_error,half_equation_error)<1e-8
        folder=ROOT/f'private/fold{outer}';folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'GROUP_AND_HALF_MODEL.npz',W_group=group_coefficient,W_half=half_coefficient)
        np.savez_compressed(folder/'TRAIN_GROUP_MEANS.npz',means=means,mapping=mapping,
            group_keys=np.array(keys),counts=np.array(counts),fitting_ids=history['ids'],fitting_indices=fitting_rows)
        predicted={name:eval_phi@coefficient for name,coefficient in
            (('R_group',group_coefficient),('R_half',half_coefficient))}
        chosen={name:semantic_argmax(value,ACTION_KEYS) for name,value in predicted.items()}
        np.savez_compressed(folder/'PREDICTIONS.npz',evaluation_ids=evaluation['ids'],evaluation_indices=held_rows,
            **{'scores__'+name:value for name,value in predicted.items()},
            **{'actions__'+name:value for name,value in chosen.items()})
        log=dict(outer_fold=outer,fitting_questions=480,evaluation_questions=120,
            training_groups=len(keys),group_sizes=counts,reused_R_full_score_max_error=restore_error,
            original_normal_equation_error=original_residual,group_normal_equation_error=group_equation_error,
            half_normal_equation_error=half_equation_error,new_formal_solves=1,derived_half_heads=1,
            reused_R_full_heads=1,solve_cpu_seconds=solve_cpu,solve_wall_seconds=solve_wall,
            group_target_wall_seconds=target_wall,derivation_wall_seconds=derivation_wall,
            history_source=refs,reused_model_sha256=sha(paths[f'R_full_model{outer}']),
            model_sha256=sha(folder/'GROUP_AND_HALF_MODEL.npz'),group_means_sha256=sha(folder/'TRAIN_GROUP_MEANS.npz'),
            predictions_sha256=sha(folder/'PREDICTIONS.npz'),completed_utc=now(),evaluation_rewards_used=False)
        put(folder/'FIT_LOG.json',log);logs.append(log);covered.extend(held_rows.tolist())
        for name in scores:
            scores[name][held_rows]=predicted[name];actions[name][held_rows]=chosen[name]
        print('E11_FOLD_COMPLETE',outer,flush=True)
    assert sorted(covered)==list(range(600))
    output=ROOT/'private/ALL_PREDICTIONS_FROZEN.npz'
    np.savez_compressed(output,ids=ids,fold=fold,**{'scores__'+name:value for name,value in scores.items()},
        **{'actions__'+name:value for name,value in actions.items()})
    put(ROOT/'private/PREDICTION_FREEZE.json',dict(frozen_utc=now(),prediction_sha256=sha(output),
        protocol_sha256=sha(ROOT/'public/E11_PROTOCOL.json'),new_formal_solves=5,derived_half_heads=5,
        reused_R_full_heads=5,algebra_check=algebra,fold_logs=logs,all_new_actions_frozen=True,
        full_outer_reward_cube_read=False,training_cpu_seconds=time.process_time()-cpu_started,
        training_wall_seconds=time.perf_counter()-started))


def evaluate():
    protocol,paths=verify_source()
    freeze=read(ROOT/'private/PREDICTION_FREEZE.json')
    assert sha(ROOT/'private/ALL_PREDICTIONS_FROZEN.npz')==freeze['prediction_sha256']
    assert freeze['protocol_sha256']==sha(ROOT/'public/E11_PROTOCOL.json')
    new=arrays(ROOT/'private/ALL_PREDICTIONS_FROZEN.npz')
    old=arrays(paths['predictions']);background=arrays(paths['background'])
    assert np.array_equal(new['ids'],old['ids']) and np.array_equal(new['fold'],old['fold'])
    for outer in range(5):
        folder=ROOT/f'private/fold{outer}';row=freeze['fold_logs'][outer]
        assert sha(folder/'GROUP_AND_HALF_MODEL.npz')==row['model_sha256']
        assert sha(folder/'PREDICTIONS.npz')==row['predictions_sha256']
    opened=now();data,refs,_=load_historical(ASSETS)
    assert np.array_equal(data['ids'],new['ids'])
    y=np.asarray(data['Y'],dtype=float);task=data['task'];sub=data['sub'];fold=new['fold']
    assert y.shape==(600,3,20) and all(sum(task==name)==300 for name in ('gsm8k','bbh'))
    actions={name:new['actions__'+name] for name in ('R_half','R_group')}
    actions.update({name:old['actions__'+name] for name in ('R_full','R_match',
        'R_full_perm20261114091','R_full_perm20261114092','R_full_perm20261114093')})
    actions.update({name:background['actions__'+name] for name in ('A','NS','GlobalFixed')})
    outcomes={name:y[np.arange(600),:,action] for name,action in actions.items()}

    def point_comparison(left,right):
        difference=outcomes[left]-outcomes[right]
        return dict(gain=float(difference.mean()),corrected_events=int(sum((difference>0).ravel())),
            destroyed_events=int(sum((difference<0).ravel())),net_events=int(difference.sum()),
            questions_with_positive_mean_difference=int(sum(difference.mean(axis=1)>0)),
            questions_with_negative_mean_difference=int(sum(difference.mean(axis=1)<0)),
            by_task={name:float(difference[task==name].mean()) for name in ('gsm8k','bbh')},
            by_generation={str(seed):float(difference[:,index].mean()) for index,seed in enumerate((42,137,2024))},
            by_outerfold={str(index):float(difference[fold==index].mean()) for index in range(5)})

    methods={}
    for name,value in outcomes.items():
        methods[name]=dict(accuracy=float(value.mean()),correct_events=int(value.sum()),
            event_denominator=1800,independent_questions=600,
            by_task={label:dict(questions=int(sum(task==label)),correct_events=int(value[task==label].sum()),
                accuracy=float(value[task==label].mean())) for label in ('gsm8k','bbh')},
            by_generation={str(seed):dict(correct_events=int(value[:,index].sum()),accuracy=float(value[:,index].mean()))
                for index,seed in enumerate((42,137,2024))},
            by_outerfold={str(index):dict(questions=int(sum(fold==index)),correct_events=int(value[fold==index].sum()),
                accuracy=float(value[fold==index].mean())) for index in range(5)},
            bbh_subtasks={str(label):dict(questions=int(sum((task=='bbh')&(sub==label))),
                correct_events=int(value[(task=='bbh')&(sub==label)].sum()),
                accuracy=float(value[(task=='bbh')&(sub==label)].mean())) for label in sorted(set(sub[task=='bbh']))},
            action_counts=np.bincount(actions[name],minlength=20),
            changed_actions_from_R_full=int(sum(actions[name]!=actions['R_full'])),
            versus_R_full=point_comparison(name,'R_full'),versus_A=point_comparison(name,'A'))
    comparisons={f'{left}-{right}':point_comparison(left,right) for left in ('R_half','R_group')
        for right in ('R_full','A','NS','GlobalFixed')}
    # Reuse the existing pure decomposition only as a compact descriptive view.
    sys.path.insert(0,str(E10/'public/code'))
    from e10_matching import decompose_policy
    matching={}
    for name in ('R_full','R_half','R_group','A'):
        value=decompose_policy(y,actions[name],task,np.where(task=='gsm8k','gsm8k',sub),fold,
            generation_seeds=(42,137,2024),action_keys=ACTION_KEYS)
        matching[name]=value['overall']['metrics']
    old_metrics=read(E10/'public/E10_MAIN_RESULTS.json')['metrics']
    for name in old_metrics:
        assert abs(methods[name]['accuracy']-old_metrics[name]['accuracy'])<1e-12,name
    result=dict(status='EXPLORATORY_REUSED_TRAIN600',primary='R_half-R_full',
        methods=methods,comparisons=comparisons,descriptive_matching=matching,
        action_keys=ACTION_KEYS,generation_seeds=[42,137,2024],
        interpretation=dict(weights='GSM8K/BBH each .5, question-uniform within task',
            generation_events='600 original questions x3 already-recorded outcomes',
            group_labels_used_for_training_only=True,model_input='frozen complete-question prefill only',
            E10_permutation_accuracy_average='background average of three policy accuracies, separate from group-head accuracy'),
        protocol_sha256=sha(ROOT/'public/E11_PROTOCOL.json'),prediction_sha256=freeze['prediction_sha256'],
        source_refs=refs)
    put(ROOT/'public/E11_RESULTS.json',result)
    np.savez_compressed(ROOT/'private/PAIRED_OUTCOMES.npz',task=task,sub=sub,fold=fold,
        **{'outcome__'+name:value for name,value in outcomes.items()})
    put(ROOT/'private/EVALUATION_ACCESS.json',dict(opened_utc=opened,predictions_frozen_utc=freeze['frozen_utc'],
        validation200_rewards=False,fresh400_rewards=False,sealed200_rewards=False,source_refs=refs))
    cost=dict(new_formal_solves=5,derived_half_heads=5,reused_R_full_heads=5,
        reused_feature_artifacts=1,reused_PCA_Scaler_sets=5,reused_background_policy_vectors=7,
        synthetic_solve_calls=freeze['algebra_check']['synthetic_solve_calls'],new_prefill=0,
        new_answer_generations=0,new_aggregations=0,new_gold=0,new_teachers=0,
        new_inner_searches=0,new_permutation_seeds=0,
        solve_cpu_seconds_sum=sum(row['solve_cpu_seconds'] for row in freeze['fold_logs']),
        solve_wall_seconds_sum=sum(row['solve_wall_seconds'] for row in freeze['fold_logs']),
        group_target_wall_seconds_sum=sum(row['group_target_wall_seconds'] for row in freeze['fold_logs']),
        derivation_wall_seconds_sum=sum(row['derivation_wall_seconds'] for row in freeze['fold_logs']),
        training_cpu_seconds=freeze['training_cpu_seconds'],training_wall_seconds=freeze['training_wall_seconds'],
        fold_measurements=[{key:row[key] for key in ('outer_fold','solve_cpu_seconds','solve_wall_seconds',
            'group_target_wall_seconds','derivation_wall_seconds')} for row in freeze['fold_logs']],
        shared_history=dict(gold=600,member_calls=10800,aggregator_calls=36000,scope='previous complete feedback assets'),
        cached_prefill_source=read(paths['features_receipt'])['feature_sha256'],
        historical_prefill=dict(formal_question_events=606,input_tokens=49446,scope='E10 investment reused, incremental E11 zero'),
        source_sha256=sha(Path(__file__)))
    put(ROOT/'public/COST.json',cost)
    print(json.dumps(dict(status='COMPLETE',accuracies={name:methods[name]['accuracy'] for name in ('R_full','R_half','R_group')},
        primary=comparisons['R_half-R_full'],group=comparisons['R_group-R_full']),ensure_ascii=False),flush=True)


def verify_outputs():
    verify_source()
    receipt=read(ROOT/'private/PREDICTION_FREEZE.json')
    assert sha(ROOT/'private/ALL_PREDICTIONS_FROZEN.npz')==receipt['prediction_sha256']
    assert len(receipt['fold_logs'])==5
    for row in receipt['fold_logs']:
        folder=ROOT/f'private/fold{row["outer_fold"]}'
        for name,key in (('GROUP_AND_HALF_MODEL.npz','model_sha256'),
                         ('TRAIN_GROUP_MEANS.npz','group_means_sha256'),('PREDICTIONS.npz','predictions_sha256')):
            assert sha(folder/name)==row[key]
    result=read(ROOT/'public/E11_RESULTS.json')
    paired=arrays(ROOT/'private/PAIRED_OUTCOMES.npz')
    for name,item in result['methods'].items():
        value=paired['outcome__'+name]
        assert float(value.mean())==item['accuracy']
        assert int(value.sum())==item['correct_events']
    print(json.dumps(dict(status='VERIFIED',new_solves=0,new_prefill=0,methods=len(result['methods']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('run','verify'))
    arguments=parser.parse_args()
    if arguments.mode=='run':
        train();evaluate()
    else:
        verify_outputs()
