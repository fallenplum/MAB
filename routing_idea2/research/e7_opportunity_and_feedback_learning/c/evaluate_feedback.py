"""Evaluate all prospectively frozen active-learning paths on outer E120."""
from active_feedback import *
from e7_common import paired_bootstrap,paired_interval
from e7_models import argmax as prediction_argmax


def error_components(prediction,truth):
    residual=prediction-truth
    offset=residual.mean(axis=-1)
    centered=residual-offset[...,None]
    return (centered**2).mean(axis=-1),offset**2


def main():
    guard=json.loads((PRIVATE/'c/ALL_PATHS_FROZEN.json').read_text())
    assert guard['path_count']==90 and guard['status']=='ALL90_FROZEN_BEFORE_EVALUATION'
    data,refs=load_train()
    kinds,methods,seeds=KINDS,METHODS,ALGSEEDS
    Q=[]
    tasks=[]
    subtasks=[]
    folds=[]
    scores=[]
    errors=[]
    offsets=[]
    actions=[]
    defaults=[]
    default_scores=[]
    costs=[]
    query_costs=[]
    route_counts=[]
    path_metadata=[]
    pool_records=[]
    cnames=None
    for f in range(5):
        r=splits(f)
        E,I=r['E'],r['I']
        initial_default=int(argmax(data['Y'][I].mean(axis=(0,1))))
        Y=data['Y'][E]
        Ym=Y.mean(axis=1)
        fscores=np.zeros((2,3,3,4,120,3))
        factions=np.zeros((2,3,3,4,120),int)
        ferr=np.zeros((2,3,3,4,120))
        foffset=np.zeros_like(ferr)
        fcost=[]
        fqcost=[]
        for ki,kind in enumerate(kinds):
            for mi,method in enumerate(methods):
                for si,seed in enumerate(seeds):
                    name=path_name(f,kind,method,seed)
                    folder=PRIVATE/'c'/name
                    receipt=json.loads((folder/'FROZEN_PATH.json').read_text())
                    assert receipt['status']=='FROZEN_BEFORE_E_REWARD_EVALUATION'
                    assert receipt['protocol_sha256']==sha(PUBLIC/'E7_PROTOCOL.json')
                    assert receipt['record_sha256']==sha(folder/'PATH_RECORDS.npz')
                    with np.load(folder/'PATH_RECORDS.npz') as a:
                        predictions=a['E_predictions']
                        purchased=a['purchased']
                        assert np.array_equal(a['E'],E)
                        assert np.array_equal(a['snapshot_steps'],CHECKPOINTS)
                        assert len(set(purchased))==60 and set(purchased)<=set(r['U'])
                        after=a['route_after']
                        before=a['route_before']
                    chosen=prediction_argmax(predictions)
                    factions[ki,mi,si]=chosen
                    actual=Y[np.arange(120)[None,:,None],np.arange(3)[None,None,:],chosen[:,:,None]]
                    assert actual.shape==(4,120,3)
                    fscores[ki,mi,si]=actual
                    centered,common=error_components(predictions,Ym)
                    ferr[ki,mi,si]=centered
                    foffset[ki,mi,si]=common
                    logs=json.loads((folder/'ACCESS_AND_COST_LOG.json').read_text())
                    cnames=list(logs['ledger'][0])
                    ledger=np.asarray([[row[key] for key in cnames] for row in logs['ledger']])
                    local_cost=np.zeros((4,len(cnames)))
                    local_query=np.zeros_like(local_cost)
                    for bi,B in enumerate(CHECKPOINTS):
                        local_cost[bi]=ledger[:B].sum(axis=0)
                        query=resources(data,E,[[int(q)] for q in chosen[bi]],gold=0)
                        local_query[bi]=[query[key] for key in cnames]
                        assert local_cost[bi,cnames.index('gold')]==B
                        assert local_cost[bi,cnames.index('member_calls')]==18*B
                        assert local_cost[bi,cnames.index('aggregator_calls')]==60*B
                    fcost.append(local_cost)
                    fqcost.append(local_query)
                    receipt['query_evaluation_after_all90_frozen']=True
                    path_metadata.append(receipt)
                    pool_records.append(dict(fold=f,learner=kind,method=method,algorithm_seed=seed,
                        purchased_ids_private=purchased.tolist(),initial_cost=logs['initial_shared_cost']))
                    put(folder/'EVALUATION_ACCESS.json',dict(role='outer_E_evaluator',questions=E.tolist(),
                        reward_values=120*3*20,all_paths_frozen_sha256=sha(PRIVATE/'c/ALL_PATHS_FROZEN.json'),
                        learner_feedback_access=False,score_source=refs['outcomes']))
                    route_counts.append(dict(path=name,changed_from_previous_step=int(np.sum(after!=before))))
        for ki in range(2):
            assert np.all(factions[ki,:,:,0]==factions[ki,0,0,0])
        scores.append(fscores)
        actions.append(factions)
        errors.append(ferr)
        offsets.append(foffset)
        costs.append(np.asarray(fcost).reshape(2,3,3,4,len(cnames)))
        query_costs.append(np.asarray(fqcost).reshape(2,3,3,4,len(cnames)))
        Q.extend(E.tolist())
        tasks.extend(data['task'][E].tolist())
        subtasks.extend(data['sub'][E].tolist())
        folds.extend([f]*120)
        defaults.extend([initial_default]*120)
        default_scores.append(Y[:,:,initial_default])
    score=np.concatenate(scores,axis=4)
    action=np.concatenate(actions,axis=4)
    centered=np.concatenate(errors,axis=4)
    offset=np.concatenate(offsets,axis=4)
    # Raw score dimensions learner,method,algorithm seed,feedback,question,LLM repeat.
    assert score.shape==(2,3,3,4,600,3)
    task=np.asarray(tasks)
    sub=np.asarray(subtasks)
    fold=np.asarray(folds)
    dscore=np.concatenate(default_scores,axis=0)
    perq=score.mean(axis=(2,5))
    boot=paired_bootstrap(perq,task,fold)
    point=perq.mean(axis=-1)
    delta_initial=perq-perq[:,:,:1]
    learn_boot=paired_bootstrap(delta_initial,task,fold)
    cost=np.mean(np.stack(costs),axis=(0,3))
    qcost=np.mean(np.stack(query_costs),axis=(0,3))
    result=dict(status='COMPLETE',original_E_questions=600,outer_folds=5,algorithm_seeds=seeds,
        LLM_generation_seeds=[42,137,2024],formal_paths=90,learners=kinds,
        learner_names={'P1':'E7-B P1, legacy direct20 PCA16 Ridge originally named P2','P3':'E7-B P3 RBF kernel Ridge'},
        methods=methods,checkpoints=CHECKPOINTS,primary_gold=60,new_llm_calls=0,
        table={},paired_comparisons={},per_fold=[],
        default_accuracy=float(dscore.mean()),default_source='I240-only best combination per fold, remains fixed',
        statistics='5000 original-question paired bootstrap stratified fold x broad task, all three generation repeats and algorithm seeds remain within question',
        CPU_scope='process_time measured each planning/update, Windows timer quantization can yield0 for short intervals; total wall saved. No original encoder time/GPU kernel time reconstructed.',
        cost_scope='mean per fold/path, initial I separately, feedbackGold per question with18 member and60 aggregator calls; actual Token reported, no equal-Token claim')
    for ki,kind in enumerate(kinds):
        result['table'][kind]={}
        for mi,method in enumerate(methods):
            result['table'][kind][method]={}
            selected_paths=[row for row in path_metadata if row['learner']==kind and row['method']==method]
            for bi,B in enumerate(CHECKPOINTS):
                values=perq[ki,mi,bi]
                initial=score[ki,mi,:,0]
                updated=score[ki,mi,:,bi]
                correction=(updated>initial).mean(axis=(0,2))
                harm=(updated<initial).mean(axis=(0,2))
                task_acc={name:float(values[task==name].mean()) for name in ['gsm8k','bbh']}
                sub_acc={name:float(values[(task=='bbh')&(sub==name)].mean()) for name in sorted(set(sub[task=='bbh']))}
                seed_acc=score[ki,mi,:,bi].mean(axis=(0,1))
                gain=float(delta_initial[ki,mi,bi].mean())
                row=dict(accuracy=float(point[ki,mi,bi]),ci95=np.quantile(boot[:,ki,mi,bi],[.025,.975]).tolist(),
                    gain_initial=gain,gain_initial_ci95=np.quantile(learn_boot[:,ki,mi,bi],[.025,.975]).tolist(),
                    gain_I_fixed=float(values.mean()-dscore.mean()),corrected_question_equivalents=float(correction.sum()),
                    harmed_question_equivalents=float(harm.sum()),net_correction=float((correction-harm).sum()),
                    changed_decisions=float((action[ki,mi,:,bi]!=action[ki,mi,:,0]).sum()/3),
                    centered_error=float(centered[ki,mi,:,bi].mean()),common_offset_error=float(offset[ki,mi,:,bi].mean()),
                    centered_error_change=float((centered[ki,mi,:,bi]-centered[ki,mi,:,0]).mean()),
                    GSM_BBH=task_acc,BBH_subtasks=sub_acc,
                    GSM_BBH_task_macro=float(.5*sum(task_acc.values())),
                    GSM_BBH27_macro=float(.5*task_acc['gsm8k']+.5*np.mean(list(sub_acc.values()))),
                    seed_accuracy=seed_acc.tolist(),seed_gain_initial=(updated-initial).mean(axis=(0,1)).tolist(),
                    algorithm_seed_accuracy=updated.mean(axis=(1,2)).tolist(),
                    feedback_cost=dict(zip(cnames,cost[ki,mi,bi].tolist())),
                    E_service_cost=dict(zip(cnames,qcost[ki,mi,bi].tolist())),
                    feedback_plus_E_service_cost=dict(zip(cnames,(cost[ki,mi,bi]+qcost[ki,mi,bi]).tolist())),
                    full_path_planning_CPU_mean=float(np.mean([p['planning_cpu_s'] for p in selected_paths])),
                    full_path_update_CPU_mean=float(np.mean([p['update_cpu_s'] for p in selected_paths])),
                    full_path_wall_mean=float(np.mean([p['wall_s'] for p in selected_paths])))
                result['table'][kind][method][str(B)]=row
            for comparison,other in [('DA-Random',0),('DA-IG',1)]:
                if mi!=2:
                    continue
                for bi,B in enumerate(CHECKPOINTS):
                    contribution=perq[ki,2,bi]-perq[ki,other,bi]
                    stat,draws=paired_interval(contribution,task,fold)
                    stat['seed_direction']=(score[ki,2,:,bi]-score[ki,other,:,bi]).mean(axis=(0,1)).tolist()
                    result['paired_comparisons'][f'{kind}:{comparison}@{B}']=stat
        for f in range(5):
            select=fold==f
            result['per_fold'].append(dict(fold=f,learner=kind,accuracy=perq[ki][...,select].mean(axis=-1).tolist(),
                shape_note='methods bycheckpoints; original120 questions, algorithm seeds averaged'))
    private=PRIVATE/'c/E7_C_PAIRED.npz'
    np.savez_compressed(private,Q=Q,task=task,sub=sub,fold=fold,score=score,actions=action,
        centered_error=centered,common_offset_error=offset,default_score=dscore,default_actions=defaults,
        perq=perq,bootstrap=boot,learn_bootstrap=learn_boot,feedback_cost=cost,E_service_cost=qcost,
        methods=np.asarray(methods),kinds=np.asarray(kinds),checkpoints=CHECKPOINTS)
    put(PRIVATE/'c/E7_C_PATH_COST_AND_ACCESS_INDEX.json',dict(path_metadata=path_metadata,private_purchase_index=pool_records,
        observed_E_reward_role='evaluator only after ALL90 frozen',model_snapshots=360))
    result['private_paired_sha256']=sha(private)
    put(PUBLIC/'c/E7_C_RESULTS.json',result)
    print(json.dumps(dict(status='COMPLETE',paths=90,primary=result['paired_comparisons']['P3:DA-Random@60'],
        accuracy={k:{m:result['table'][k][m]['60']['accuracy'] for m in methods} for k in kinds})),flush=True)


if __name__=='__main__':
    main()
