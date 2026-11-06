from common import *
import pickle,shutil

def raw_predict(saved,dd,base):
    x=saved['transform'].transform(dd);s=saved['state'];f=saved['config']['family']
    if f=='structured':return x@s['W']@B.T
    if f=='delta_mlp':p=s['model'].predict(x[:,1:])
    elif f=='delta_default_interaction':p=np.column_stack((x,(x[:,:,None]*np.eye(20)[base,None,:]).reshape(len(x),-1)))@s['W']
    else:p=x@s['W']
    return p-p[np.arange(len(p)),base,None] if f.startswith('delta_') else p

def main():
    d,refs=load();tr=d['selector_train'];va=d['selector_validation'];r=read(PUBLIC/'CPU_COMPLETE_RESULTS.json')
    backup=PRIVATE/'CPU_COMPLETE_RESULTS_BEFORE_VALUE_METRIC_AUDIT.json'
    if not backup.exists():shutil.copyfile(PUBLIC/'CPU_COMPLETE_RESULTS.json',backup)
    with (PRIVATE/'FITTED_CANDIDATE_STATES.pkl').open('rb') as f:pack=pickle.load(f)
    states=pack['states'];bt=defaults(tr,tr,10,4)['subtask_fixed'].argmax(1);bv=defaults(tr,va,10,4)['subtask_fixed'].argmax(1)
    with np.load(PRIVATE/'ALL_VALIDATION_PREDICTIONS.npz') as a:pred={k:a[k] for k in a.files}
    with np.load(PRIVATE/'BASELINE_OOF.npz') as a:bo=a['shrink_default'].argmax(1)
    max_error=0.;max_choice_changes=0
    for record in r['all_candidate_oof']:
        name=record['name'];saved=states[name];isdelta=record['family'].startswith('delta_')
        p=raw_predict(saved,va,bv);max_error=max(max_error,float(np.max(np.abs(p-pred[name]))));max_choice_changes+=int(np.sum(p.argmax(1)!=pred[name].argmax(1)))
        r['all_candidate_validation'][name]=metrics(p,va,bv if isdelta else None)
        r['final_train_metrics'][name]=metrics(raw_predict(saved,tr,bt),tr,bt if isdelta else None)
        with np.load(PRIVATE/'oof'/f'{name}.npz') as z:oo=z['pred']
        record['oof']=metrics(oo,tr,bo if isdelta else None)
        if isdelta:
            for fold in record['fold_fits']:
                # Fold fit scores were not stored; do not label absolute-target error as delta error.
                for k in ['mse','conditional_mean_mse']:fold['train_metrics'].pop(k,None)
                fold['train_metrics']['value_mse_scope']='not_retained_from_initial_fold_fit; final_train_and_oof_delta_errors_replayed_without_refit'
    assert max_choice_changes==0 and max_error<1e-9
    for k in ['mse','conditional_mean_mse','pair_value_difference_mae','non_tie_pair_ranking_accuracy']:
        r['original_validation_baselines']['independent_top3'][k]=None
    r['original_validation_baselines']['independent_top3']['value_prediction_error_scope']='choices_only; sentinel ranking/value errors not meaningful'
    for method in ['conditional_main_thresholded','empirically_calibrated_switch']:
        choice=pred[method].argmax(1);own=selected(va['Y'],choice)
        orig=selected(va['Y'],pred['subtask_fixed'].argmax(1));delta=own-orig
        r['conditional_switches'][method]['vs_original_subtask_detailed']={'question_any_wrong_to_right':int(np.any(delta>0,axis=1).sum()),'question_any_right_to_wrong':int(np.any(delta<0,axis=1).sum()),'unique_question_sets_can_overlap':True,'per_generation_net_corrected_cells':delta.sum(0).astype(int).tolist(),'per_generation_gain':[macro(delta[:,g],va['task']) for g in range(3)],'per_task_gain':{t:ci(delta[va['task']==t],va['task'][va['task']==t],va['sub'][va['task']==t]) for t in ['gsm8k','bbh']}}
    chosen=r['selected_conditional'];saved=states[chosen['candidate']]
    c=time.process_time();w=time.perf_counter()
    for _ in range(20):raw_predict(saved,va,bv)
    bench={'prediction_including_transform_cpu_s_per200':(time.process_time()-c)/20,'wall_s_per200':(time.perf_counter()-w)/20,'frozen_encoder_embeddings_reused':True,'MiniLM_encoding_seconds':'NOT_MEASURED_HERE','new_llm_calls':0}
    # Logical service resource projection uses actual recorded costs; precomputation is never free.
    cost={}
    for method in ['global_fixed','task_fixed','subtask_fixed','independent_top3','original_frozen_selector','conditional_main_thresholded']:
        ch=pred[method].argmax(1);members=np.array(TEAMS)[ch];a={}
        for suffix in ['execution_counts','prompt_tokens','completion_tokens','wall_time_s']:
            mc=va['cost']['member_'+suffix].transpose(1,0,2);ac=va['cost']['aggregate_'+suffix].transpose(1,0,2)
            mtotal=sum(float(mc[q,:,members[q]].sum()) for q in range(200));atotal=sum(float(ac[q,:,ch[q]].sum()) for q in range(200))
            a[suffix]={'member':mtotal,'aggregate':atotal,'total':mtotal+atotal}
        cost[method]=a
    put(PUBLIC/'ROUTER_AND_INFERENCE_COST.json',{'selector_cpu_benchmark':bench,'logical_service_projection_from_actual_cache':cost,'new_actual_llm_calls':0,'new_actual_gpu_seconds':0,'GPU_kernel_time':'NOT_RECORDED_IN_LEGACY_MATRIX','wall_time_s':'sum of model.generate wall times; not kernel time, occupancy time or calendar time','past_full_matrix_precomputation_costs_retained_in_original_receipts':True,'note':'Policy projections overlap and must not be summed as distinct physical spend.'})
    put(PUBLIC/'METRIC_AND_REPLAY_AUDIT.json',{'status':'PASS','prediction_max_abs_replay_error':max_error,'validation_choice_changes':0,'delta_value_mse_corrected_to_paired_target':True,'initial_pre_audit_results_sha256':sha(backup),'fold_conditional_training_mse_not_retained':True,'new_fits':0,'new_llm_calls':0})
    put(PUBLIC/'CPU_COMPLETE_RESULTS.json',r)
    # Freeze input-only fresh choices even while execution is held for scientific review.
    bundle=read(ROOT/'artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json');ref=bundle['inputs']['fresh_dev_test'];assert sha(ref['path'])==ref['sha256']
    with np.load(ref['path'],allow_pickle=False) as a:ip={k:a[k] for k in a.files}
    fresh={'E':ip['embeddings'],'task':ip['task_labels'],'sub':ip['subtask_labels']}
    base=defaults(tr,fresh,10,4)['subtask_fixed'].argmax(1);pp=raw_predict(saved,fresh,base)
    new=np.where(pp.max(1)>chosen['threshold'],pp.argmax(1),base)
    old=ROOT/'artifacts/private/routing_redesign_20261104/ff_v1'
    with np.load(old/'pred/predictions.npz') as a:policies={k:a[k].argmax(1) for k in ['selector','global_fixed','task_fixed','subtask_fixed']};assert np.array_equal(a['question_ids'],ip['question_ids'])
    with np.load(old/'it3_v1/fresh/predictions.npz') as a:policies['independent_top3']=a['selected_team_index'];assert np.array_equal(a['question_ids'],ip['question_ids'])
    policies['conditional_main']=new
    union=sum(len(set(int(v[q]) for v in policies.values())) for q in range(400))
    np.savez_compressed(PRIVATE/'FRESH_INPUT_ONLY_POLICIES_FROZEN.npz',**policies,question_ids=ip['question_ids'],canonical_hashes=ip['canonical_hashes'])
    put(PUBLIC/'FRESH_POLICY_FREEZE_INPUT_ONLY.json',{'status':'FROZEN_CHOICES_EXECUTION_HOLD_NO_FRESH_SCORES_OPENED','selected_main':chosen,'input_sha256':ref['sha256'],'private_frozen_choice_sha256':sha(PRIVATE/'FRESH_INPUT_ONLY_POLICIES_FROZEN.npz'),'policy_names':list(policies),'questions':400,'generation_seeds':SEEDS,'unique_question_team_calls_per_seed':union,'unique_question_team_calls_three_seeds':union*3,'control_calls_not_in_this_union':['strong_single','three_sample_vote_or_fusion','aggregator_alone'],'not_a_complete_nine_baseline_execution_budget':True,'fresh_gold_opened':False,'new_member_or_aggregate_calls':0,'reason_for_execution_hold':'Train-OOF selected candidate has no repeatable positive development validation gain; do not manufacture success or automatically expand compute.'})
    print(json.dumps({'replay_max_abs':max_error,'new_fits':0,'fresh_six_policy_union_per_seed':union,'actual_model_calls':0,'main_validation_accuracy':r['conditional_switches']['conditional_main_thresholded']['accuracy']}),flush=True)
if __name__=='__main__':main()
