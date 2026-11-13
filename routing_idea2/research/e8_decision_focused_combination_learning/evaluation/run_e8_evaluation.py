"""Evaluate all sealed E8 predictions through an independent reward reader."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
BASE=ROOT/'research/e8_decision_focused_combination_learning'
sys.path[:0]=[str(BASE/'public/evaluation'),str(BASE/'public/qualification'),str(BASE/'public/training')]
from e8_assets import FeatureMap,load_historical
from run_e8_training_v1 import VARIANTS
from e8_metrics import affine_calibration,calibrate,frozen_metrics,summarize_metrics
from e8_metrics import bootstrap_matrix,paired_statistics,holm,PAIRS,I,J


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n',encoding='utf-8')


def archive(path):
    with np.load(path,allow_pickle=False) as source:return {k:source[k].copy() for k in source.files}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def restored_scores(state,embeddings):
    pca=PCA(16,svd_solver='full');pca.components_=state['pca_components'];pca.mean_=state['pca_mean']
    pca.n_features_in_=384;pca.n_components_=16
    # sklearn1.7 checks this attribute even though unwhitened PCA does not
    # use it in the numerical transform. The original feature map is unwhitened.
    pca.explained_variance_=np.ones(16);pca.whiten=False
    scaler=StandardScaler();scaler.mean_=state['scaler_mean'];scaler.scale_=state['scaler_scale']
    scaler.n_features_in_=16
    return FeatureMap(pca,scaler).phi(embeddings)@state['coefficient']


def collect_sealed_predictions():
    index=read(BASE/'private/training/E8_FROZEN_TRAINING_INDEX.json')
    if not index['complete'] or len(index['folds'])!=5 or len(index['variants'])!=14:
        raise AssertionError('All five folds and fourteen variants must be sealed first')
    if index['variants']!=VARIANTS or len(set(index['variants']))!=14:
        raise AssertionError('Variant list differs from the frozen training contract')
    if sha(BASE/'public/E8_PROTOCOL.json')!=index['protocol_sha256']:
        raise AssertionError('Protocol hash mismatch')
    protocol=read(BASE/'public/E8_PROTOCOL.json')
    if sha(BASE/'private/E8_SOURCE_LOCK.json')!=index['source_lock_sha256'] or index['source_lock_sha256']!=protocol['source_lock_sha256']:
        raise AssertionError('Source lock hash mismatch')
    manifest_path=BASE/'private/qualification/FOLD_PACKS_MANIFEST.json'
    if sha(manifest_path)!=protocol['data']['fold_pack_manifest_sha256']:
        raise AssertionError('Fold pack manifest changed')
    pack_specs={item['fold']:item for item in read(manifest_path)['folds']}
    if sorted(f['fold'] for f in index['folds'])!=list(range(5)):
        raise AssertionError('Frozen outer fold IDs must be unique and complete')
    for path,digest in index['source_lock']['files'].items():
        if sha(path)!=digest:raise AssertionError('Frozen source changed '+path)
    amendment_hashes=set()
    for amendment in index.get('execution_amendments',[]):
        if sha(amendment['path'])!=amendment['sha256']:
            raise AssertionError('Execution amendment hash mismatch')
        amendment_hashes.add(amendment['sha256'])
    if amendment_hashes:
        recovery_path=BASE/'private/E8_RECOVERY_SOURCE_LOCK.json'
        if sha(recovery_path)!=index['recovery_source_lock_sha256']:
            raise AssertionError('Numerical recovery source lock changed')
        if read(recovery_path)!=index['recovery_source_lock']:
            raise AssertionError('Embedded recovery source lock differs')
        for path,digest in index['recovery_source_lock']['files'].items():
            if sha(path)!=digest:raise AssertionError('Numerical recovery source changed '+path)
    variants=index['variants'];raw={name:np.empty((600,20)) for name in variants}
    calibrated={name:np.empty((600,20)) for name in variants}
    defaults=np.empty(600,dtype=int);fold_ids=np.empty(600,dtype=int)
    parameters={};training_summary={};restoration=[];covered=[]
    for fold in index['folds']:
        fold_id=fold['fold'];sealed=archive(fold['predictions_path'])
        if set(fold['variants'])!=set(VARIANTS):raise AssertionError('Incomplete per-fold model family')
        if sha(fold['predictions_path'])!=fold['predictions_sha256']:raise AssertionError('Prediction hash mismatch')
        spec=pack_specs[fold_id]
        if sha(fold['history_pack_path'])!=spec['history']['sha256'] or sha(fold['evaluation_input_pack_path'])!=spec['evaluation_inputs']['sha256']:
            raise AssertionError('Legal history or input-only pack hash mismatch')
        history=archive(fold['history_pack_path']);inputs=archive(fold['evaluation_input_pack_path'])
        if 'Y' in inputs:raise AssertionError('Evaluation input-only pack contains rewards')
        if not np.array_equal(sealed['eval_indices'],inputs['global_indices']) or not np.array_equal(sealed['eval_ids'],inputs['ids']):
            raise AssertionError('Frozen evaluation identity differs from the input pack')
        if not np.array_equal(sealed['history_indices'],history['global_indices']) or not np.array_equal(sealed['history_ids'],history['ids']):
            raise AssertionError('Frozen historical identity differs from the historical pack')
        if len(set(sealed['eval_indices'].tolist()))!=120 or len(set(sealed['history_indices'].tolist()))!=480:
            raise AssertionError('Unexpected original-question counts')
        if set(sealed['eval_indices'].tolist())&set(sealed['history_indices'].tolist()):
            raise AssertionError('Outer fitting and evaluation identities overlap')
        covered.extend(sealed['eval_indices'].tolist())
        rows=sealed['eval_indices'];defaults[rows]=int(history['Y'].mean(axis=(0,1)).argmax())
        fold_ids[rows]=fold_id
        for name in variants:
            receipt=fold['variants'][name];folder=BASE/'private/training'/('fold'+str(fold_id))/name
            if receipt['variant']!=name or receipt['fold']!=fold_id:
                raise AssertionError('Model receipt identity mismatch')
            if 'engineering_amendment_sha256' in receipt:
                if receipt['engineering_amendment_sha256'] not in amendment_hashes:
                    raise AssertionError('Recovered model has an unrecognized numerical amendment')
                if receipt['recovery_source_lock_sha256']!=index['recovery_source_lock_sha256']:
                    raise AssertionError('Recovered model source mismatch')
            for file,digest in receipt['artifact_hashes'].items():
                if sha(folder/file)!=digest:raise AssertionError('Sealed model file changed')
            q=sealed['raw_q__'+name]
            if not np.array_equal(np.argmax(q,axis=1),sealed['actions__'+name]):
                raise AssertionError('Saved decisions disagree with raw argmax')
            state=archive(folder/'model_state.npz');recovered=restored_scores(state,inputs['E'])
            if not np.allclose(recovered,q,rtol=0.,atol=1e-11):raise AssertionError('Saved model prediction cannot be restored')
            if not np.array_equal(np.argmax(recovered,axis=1),np.argmax(q,axis=1)):
                raise AssertionError('Exact saved-model action restoration failed')
            output=archive(folder/'train_outputs.npz')
            mapping=affine_calibration(sealed['inner_oof_q__'+name],output['original_train_reward_means'])
            raw[name][rows]=q;calibrated[name][rows]=calibrate(q,mapping)
            key='fold'+str(fold_id)+'__'+name;parameters[key]=mapping
            log=read(folder/'FIT_LOG.json')
            margins=np.sort(q,axis=1)[:,-1]-np.sort(q,axis=1)[:,-2]
            restoration.append(dict(fold=fold_id,variant=name,
                max_prediction_error=float(abs(recovered-q).max()),action_changes=0,
                minimum_top_score_gap=float(margins.min()),
                top_gap_below_1e8=int(np.sum(margins<1e-8)),top_gap_below_1e12=int(np.sum(margins<1e-12))))
            training_summary[key]=dict(selected_parameters=log['selected_parameters'],
                total_cpu_seconds=log['total_cpu_seconds'],total_wall_seconds=log['total_wall_seconds'],
                inference=log['inference'],final_diagnostics=log['final_diagnostics'],
                cv_candidates=[dict(parameters=x['parameters'],inner_balanced_accuracy=x['inner_balanced_accuracy']) for x in log['candidates']],
                final_tie_diagnostic=training_ties(output,receipt['method'],log['true_tie_rule']))
    if len(covered)!=600 or sorted(covered)!=list(range(600)):
        raise AssertionError('Sealed predictions must cover all600 original questions exactly once')
    destination=BASE/'private/evaluation';destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'PREDICTIONS_AND_CALIBRATION_FROZEN.npz',
        default_actions=defaults,fold=fold_ids,**{'raw__'+k:v for k,v in raw.items()},
        **{'calibrated__'+k:v for k,v in calibrated.items()})
    put(destination/'CALIBRATION_FROZEN_RECEIPT.json',dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),global_training_index_sha256=sha(BASE/'private/training/E8_FROZEN_TRAINING_INDEX.json'),
        prediction_and_calibration_sha256=sha(destination/'PREDICTIONS_AND_CALIBRATION_FROZEN.npz'),
        new_outer_reward_evaluation_started=False,parameters=parameters,
        calibration_uses_original_history_only=True,calibration_changes_actions=False))
    return index,raw,calibrated,defaults,fold_ids,training_summary,restoration,parameters


def training_ties(output,method,tie_rule):
    y=output['fitting_reward_means'];q=output['fitted_train_scores']
    maximum=y.max(axis=1,keepdims=True);best=y==maximum;counts=best.sum(axis=1)
    all_tied=counts==20;multiple=counts>1
    result=dict(questions=len(y),multiple_true_best=int(multiple.sum()),all20_tied=int(all_tied.sum()),
        unique_true_best=int((~multiple).sum()),mean_true_optimal_count=float(counts.mean()))
    if method=='M4':
        u=best/counts[:,None] if tie_rule=='uniform' else np.eye(20)[np.argmax(y,axis=1)]
        losses=np.max(2*q-y,axis=1)-2*np.sum(u*q,axis=1)+maximum[:,0]
        for label,mask in [('all20_tied',all_tied),('multiple_best',multiple),('unique_best',~multiple)]:
            result[label+'_mean_empirical_SPO_loss']=float(losses[mask].mean()) if mask.any() else None
            result[label+'_loss_total_share']=float(losses[mask].sum()/losses.sum()) if losses.sum()>1e-15 else None
    return result


def execution_costs(data,actions):
    from itertools import combinations
    teams=np.array(list(combinations(range(6),3)));out=data['cost'];n=len(actions)
    result={key:np.zeros(n) for key in ['member_calls','aggregator_calls','input_tokens','output_tokens','source_model_wall_s']}
    for row,action in enumerate(actions):
        members=teams[action]
        for stem,ids,call in [('member',members,'member_calls'),('aggregate',[action],'aggregator_calls')]:
            result[call][row]+=out[stem+'_execution_counts'][:,row,ids].sum()
            result['input_tokens'][row]+=out[stem+'_prompt_tokens'][:,row,ids].sum()
            result['output_tokens'][row]+=out[stem+'_completion_tokens'][:,row,ids].sum()
            result['source_model_wall_s'][row]+=out[stem+'_wall_time_s'][:,row,ids].sum()
    result['tokens']=result['input_tokens']+result['output_tokens']
    return result


def reward_noise(y,tasks):
    """Two-repeat pair directions followed by the held-out generation repeat."""
    summaries={}
    for group,mask in [('all',np.ones(len(y),bool)),('gsm8k',tasks=='gsm8k'),('bbh',tasks=='bbh')]:
        z=y[mask];records=[]
        for held in range(3):
            others=[s for s in range(3) if s!=held]
            prior=z[:,others,:].mean(axis=1);held_reward=z[:,held,:]
            gap=prior[:,I]-prior[:,J];held_gap=held_reward[:,I]-held_reward[:,J]
            for label,eligible in [('all_nonzero',gap!=0),('gap_half',np.abs(gap)==.5),('gap_one',np.abs(gap)==1.)]:
                total=int(eligible.sum());same=int(((gap*held_gap>0)&eligible).sum())
                opposite=int(((gap*held_gap<0)&eligible).sum());tie=int(((held_gap==0)&eligible).sum())
                records.append(dict(held_generation=[42,137,2024][held],gap_group=label,
                    original_questions=len(z),prior_strict_pair_events=total,same=same,opposite=opposite,held_tie=tie,
                    same_per_prior_strict=float(same/total) if total else None,
                    same_among_held_non_ties=float(same/(same+opposite)) if same+opposite else None,
                    held_tie_fraction=float(tie/total) if total else None))
        mean=z.mean(axis=1);counts=(mean==mean.max(axis=1,keepdims=True)).sum(axis=1)
        summaries[group]=dict(questions=len(z),multiple_empirical_best_questions=int((counts>1).sum()),
            all20_empirical_tie_questions=int((counts==20).sum()),
            empirical_optimal_action_count_distribution={str(k):int(np.sum(counts==k)) for k in sorted(set(counts.tolist()))},
            cross_generation_pair_direction=records)
    return summaries


def main():
    started=time.perf_counter()
    if (BASE/'public/E8_COMPLETE_RESULTS.json').exists():
        raise FileExistsError('Completed E8 evaluation requires a new reproduction output directory')
    index,raw,calibrated,defaults,fold,training,restoration,calibration=collect_sealed_predictions()
    # All training, preprocessing, choices and calibration are sealed before
    # this independent privilege opens the original train600 reward payload.
    data,refs,splits=load_historical(ROOT)
    put(BASE/'private/evaluation/REWARD_ACCESS_RECEIPT.json',dict(
        opened_at_utc=datetime.now(timezone.utc).isoformat(),role='independent outer-fold evaluator',
        complete_predictions_frozen=True,all_model_artifacts_verified=True,
        reward_payload='selector_train600 only',validation200_payloads_opened=0,
        fresh400_payloads_opened=0,sealed200_payloads_opened=0,
        training_uses_fold_scoped_history480=True,raw_file_is_shared_training_asset=True,
        scope='application-level role and file boundaries, no OS-isolation guarantee'))
    y=data['Y'];task=data['task'];sub=data['sub'];variants=index['variants']
    all_metrics={};summaries={};costs={}
    for name in variants:
        metrics=frozen_metrics(y,raw[name],calibrated[name],defaults,raw['M0_original'])
        all_metrics[name]=metrics
        groups={'all':np.ones(600,bool),'gsm8k':task=='gsm8k','bbh':task=='bbh'}
        summary={label:summarize_metrics(metrics,mask) for label,mask in groups.items()}
        summary['folds']={str(f):summarize_metrics(metrics,fold==f) for f in range(5)}
        summary['bbh_subtasks']={str(s):summarize_metrics(metrics,(task=='bbh')&(sub==s)) for s in sorted(set(sub[task=='bbh']))}
        summary['bbh_subtask_macro_accuracy']=float(np.mean([v['accuracy'] for v in summary['bbh_subtasks'].values()]))
        summaries[name]=summary
        cost=execution_costs(data,metrics['actions']);costs[name]={k:float(v.sum()) for k,v in cost.items()}
        np.savez_compressed(BASE/'private/evaluation'/(name+'_QUESTION_METRICS.npz'),**metrics,**{'service_cost_'+k:v for k,v in cost.items()})
    samples=bootstrap_matrix(task,fold,10000,20261113)
    primary=[]
    for name in ['M3','M4']:
        row=paired_statistics(all_metrics[name]['accuracy'],all_metrics['M0_original']['accuracy'],samples)
        row.update(method=name,baseline='M0_original')
        primary.append(row)
    holm(primary)
    for row in primary:
        row['scientific_positive_gate']=bool(row['passes_positive_primary'] and row['paired95'][0]>0)
    secondary={}
    for name in variants:
        secondary[name]=dict(
            versus_original=paired_statistics(all_metrics[name]['accuracy'],all_metrics['M0_original']['accuracy'],samples),
            versus_tuned=paired_statistics(all_metrics[name]['accuracy'],all_metrics['M0_tuned']['accuracy'],samples),
            versus_global_fixed=paired_statistics(all_metrics[name]['accuracy'],all_metrics[name]['default_accuracy'],samples))
        if not name.startswith('SHUFFLE_') and 'SHUFFLE_'+name in variants:
            secondary[name]['versus_matching_shuffle']=paired_statistics(all_metrics[name]['accuracy'],all_metrics['SHUFFLE_'+name]['accuracy'],samples)
        for metric in ['calibrated_centered_MSE','calibrated_competitor_gap_MAE','calibrated_unrelated_gap_MAE']:
            secondary[name][metric+'_difference_vs_original']=paired_statistics(all_metrics[name][metric],all_metrics['M0_original'][metric],samples)
    sensitivities=dict(
        Top5_minus_Top3=paired_statistics(all_metrics['M3_top5']['accuracy'],all_metrics['M3']['accuracy'],samples),
        uniform_true_ties_minus_lexicographic=paired_statistics(all_metrics['M4_uniformtie']['accuracy'],all_metrics['M4']['accuracy'],samples))
    statistics=dict(primary_comparisons=primary,secondary=secondary,sensitivities=sensitivities,
        bootstrap_repetitions=10000,original_question_clusters=600,generation_repeats_per_cluster=3,
        primary_weights='GSM8K50% BBH50%',BBH_question_weighting=True,
        primary_multiplicity='Holm2',other_comparisons='exploratory unadjusted paired-bootstrap intervals',
        scope='reused-development conditional cross-validation with overlapping training histories',
        primary_protocol_sha256=sha(BASE/'public/E8_PROTOCOL.json'))
    put(BASE/'public/E8_PRIMARY_STATISTICS.json',statistics)
    noise=reward_noise(y,task);put(BASE/'public/E8_NOISE_DIAGNOSTICS.json',noise)
    calibration_summary=dict(mappings=len(calibration),
        nonnegative_slopes=True,zero_slope_mappings=sum(x['slope']==0 for x in calibration.values()),
        exact_coefficients_retained_privately=True,
        private_calibration_receipt_sha256=sha(BASE/'private/evaluation/CALIBRATION_FROZEN_RECEIPT.json'))
    put(BASE/'public/E8_TRAINING_AND_RESTORATION_AUDIT.json',dict(training=training,
        restoration=restoration,calibration=calibration_summary,all70_saved_models_restore_exact_actions=True,
        all_raw_routing_scores_preserved=True,calibration_used_for_decisions=False))
    complete=dict(schema='MAB_E8_complete_results_v1',status='COMPLETE_FROZEN_CACHE_DEVELOPMENT_EVALUATION',
        variants=variants,questions=600,folds=5,models_frozen=70,generation_seeds=[42,137,2024],
        methods=summaries,service_cache_costs=costs,primary_comparisons=primary,
        next_stage='DRAFT_FEEDBACK_PROTOCOL_ONLY' if any(x['scientific_positive_gate'] for x in primary) else 'STOP_NO_FEEDBACK_EXPANSION',
        new_physical_LLM_calls=0,validation200_used=False,fresh400_used=False,sealed200_used=False,
        frozen_training_index_sha256=sha(BASE/'private/training/E8_FROZEN_TRAINING_INDEX.json'),
        protocol_sha256=sha(BASE/'public/E8_PROTOCOL.json'),evaluation_wall_seconds=time.perf_counter()-started,
        training_index_wall_seconds=index['total_wall_seconds'],
        training_index_wall_scope=index.get('total_wall_seconds_scope','Original complete training run'),
        successful_model_training_cpu_seconds=float(sum(v['total_cpu_seconds'] for v in training.values())),
        proof_scope='actual operators qualified; observed reward averages do not equal true conditional expectations',
        data_source_sha256={k:v['sha256'] for k,v in refs.items() if isinstance(v,dict) and 'sha256' in v})
    put(BASE/'public/E8_COMPLETE_RESULTS.json',complete)
    print(json.dumps(dict(status=complete['status'],primary=primary,next_stage=complete['next_stage']),ensure_ascii=False))


if __name__=='__main__':main()
