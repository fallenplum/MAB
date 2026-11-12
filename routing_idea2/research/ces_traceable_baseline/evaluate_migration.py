"""Question-paired evaluation of frozen author-code CES migration paths."""
from run_migration import *


def main():
    frozen=json.loads((PRIVATE/'ALL45_FROZEN.json').read_text())
    assert frozen['status']=='ALL45_FROZEN_BEFORE_EVALUATION' and len(frozen['paths'])==45
    data,source=load_train()
    score=[]
    predictions=[]
    actions=[]
    errors=[]
    offsets=[]
    baselines=[]
    ids=[]
    task=[]
    sub=[]
    folds=[]
    costs=[]
    initial=[]
    services=[]
    metadata=[]
    for f in range(5):
        E=splits(f)['E']
        I=splits(f)['I']
        Y=data['Y'][E]
        Ym=Y.mean(axis=1)
        default=int(np.argmax(data['Y'][I].mean(axis=(0,1))))
        fs=np.zeros((3,2,3,4,120,3))
        fa=np.zeros((3,2,3,4,120),int)
        fp=np.zeros((3,3,4,120,20))
        fe=np.zeros((3,3,4,120))
        fo=np.zeros_like(fe)
        fc=[]
        fi=[]
        fservice=[]
        for ti,track in enumerate(TRACKS):
            for si,seed in enumerate(SEEDS):
                folder=PRIVATE/'replay'/f'f{f}_{track}_s{seed}'
                receipt=json.loads((folder/'FROZEN_PATH.json').read_text())
                assert receipt['record_sha256']==sha(folder/'FROZEN_PATH.npz')
                assert receipt['protocol_sha256']==sha(PUBLIC/'CES_PROTOCOL.json')
                with np.load(folder/'FROZEN_PATH.npz') as a:
                    assert np.array_equal(a['E'],E)
                    chosen=np.asarray([a['E_actions'],a['E_mean_actions']])
                    pred=a['E_predicted_means']
                    purchase=a['purchased']
                assert chosen.shape==(2,4,120)
                for mode in range(2):
                    observed=Y[np.arange(120)[None,:,None],np.arange(3)[None,None,:],chosen[mode,:,:,None]]
                    fs[ti,mode,si]=observed
                    fa[ti,mode,si]=chosen[mode]
                fp[ti,si]=pred
                residual=pred-Ym
                common=residual.mean(axis=-1)
                fe[ti,si]=((residual-common[...,None])**2).mean(axis=-1)
                fo[ti,si]=common**2
                logs=json.loads((folder/'FEEDBACK_ACCESS_AND_COST.json').read_text())
                costnames=list(logs['cost'][0])
                rows=np.asarray([[r[k] for k in costnames] for r in logs['cost']])
                sums=np.asarray([rows[:n].sum(axis=0) for n in CHECKPOINTS])
                for bi,B in enumerate(CHECKPOINTS):
                    assert sums[bi,costnames.index('gold')]==B
                    assert sums[bi,costnames.index('member_calls')]==(18 if track=='HistoryFit-Full20' else 9)*B
                    assert sums[bi,costnames.index('aggregator_calls')]==(60 if track=='HistoryFit-Full20' else 3)*B
                fc.append(sums)
                fi.append([logs['initial_cost'][k] for k in costnames])
                qc=np.zeros((2,4,len(costnames)))
                for mode in range(2):
                    for bi,B in enumerate(CHECKPOINTS):
                        cost=resources(data,E,[[int(q)] for q in chosen[mode,bi]],gold=0)
                        qc[mode,bi]=[cost[k] for k in costnames]
                fservice.append(qc)
                metadata.append(receipt)
                put(folder/'EVALUATOR_ACCESS.json',dict(role='outer_E_only_evaluator',questions=E.tolist(),
                    frozen_all45_sha256=sha(PRIVATE/'ALL45_FROZEN.json'),feedback_returned_to_learner=False))
        score.append(fs)
        predictions.append(fp)
        actions.append(fa)
        errors.append(fe)
        offsets.append(fo)
        costs.append(np.asarray(fc).reshape(3,3,4,len(costnames)))
        initial.append(np.asarray(fi).reshape(3,3,len(costnames)))
        services.append(np.asarray(fservice).reshape(3,3,2,4,len(costnames)))
        baselines.append(Y[:,:,default])
        ids.extend(E.tolist())
        task.extend(data['task'][E].tolist())
        sub.extend(data['sub'][E].tolist())
        folds.extend([f]*120)
    scores=np.concatenate(score,axis=4)
    predicted=np.concatenate(predictions,axis=3)
    selected=np.concatenate(actions,axis=4)
    centered=np.concatenate(errors,axis=3)
    common=np.concatenate(offsets,axis=3)
    baseline=np.concatenate(baselines)
    task=np.asarray(task)
    sub=np.asarray(sub)
    fold=np.asarray(folds)
    perq=scores.mean(axis=(2,5))
    assert perq.shape==(3,2,4,600)
    with np.load(E7/'private/c/E7_C_PAIRED.npz') as a:
        assert np.array_equal(a['Q'],ids)
        previous_scores=a['score']
        previous_perq=a['perq']
    previous=np.asarray([previous_perq[0,0],previous_perq[1,0]])
    primary=[]
    primary_boot=[]
    for i,name in enumerate(['E7 Random PCA16 Ridge','E7 Random RBF Ridge']):
        contribution=perq[2,0,3]-previous[i,3]
        result,draw=paired_interval(contribution,task,fold,family=2)
        result['name']='HistoryFit-Full20 CES-UCB minus '+name+' atGold60'
        primary.append(result)
        primary_boot.append(draw)
    adjusted=holm([r['centered_bootstrap_p'] for r in primary])
    for r,p in zip(primary,adjusted):
        r['Holm_p']=float(p)
        r['positive_family_evidence']=bool(p<.05 and r['ci_family_bonferroni'][0]>0)
    from e7_common import paired_bootstrap
    boot=paired_bootstrap(perq,task,fold)
    relative=paired_bootstrap(perq-baseline.mean(axis=1),task,fold)
    cost=np.mean(np.stack(costs),axis=(0,2))
    init=np.mean(np.stack(initial),axis=(0,2))
    svc=np.mean(np.stack(services),axis=(0,2))
    # cost[track,B,key], svc[track,service,B,key], both mean per fold/seed.
    table={}
    for ti,track in enumerate(TRACKS):
        table[track]={}
        for mode,name in enumerate(['UCB','MeanOnlyDiagnostic']):
            table[track][name]={}
            for bi,B in enumerate(CHECKPOINTS):
                qscore=perq[ti,mode,bi]
                initial_score=scores[ti,mode,:,0]
                final_score=scores[ti,mode,:,bi]
                task_acc={t:float(qscore[task==t].mean()) for t in ['gsm8k','bbh']}
                sub_acc={s:float(qscore[(task=='bbh')&(sub==s)].mean()) for s in sorted(set(sub[task=='bbh']))}
                gain_initial,unused=paired_interval(qscore-perq[ti,mode,0],task,fold,family=2)
                row=dict(accuracy=float(qscore.mean()),ci95=np.quantile(boot[:,ti,mode,bi],[.025,.975]).tolist(),
                    gain_default=float(qscore.mean()-baseline.mean()),gain_default_ci95=np.quantile(relative[:,ti,mode,bi],[.025,.975]).tolist(),
                    gain_own_initial=gain_initial,
                    corrected_question_equivalents=float((final_score>initial_score).mean(axis=(0,2)).sum()),
                    harmed_question_equivalents=float((final_score<initial_score).mean(axis=(0,2)).sum()),
                    changed_from_initial=float(np.sum(selected[ti,mode,:,bi]!=selected[ti,mode,:,0])/3),
                    GSM_BBH=task_acc,BBH_subtasks=sub_acc,task_macro=float(.5*sum(task_acc.values())),
                    GSM_BBH27_macro=float(.5*task_acc['gsm8k']+.5*np.mean(list(sub_acc.values()))),
                    seed_accuracy=final_score.mean(axis=(0,1)).tolist(),algorithm_seed_accuracy=final_score.mean(axis=(1,2)).tolist(),
                    centered_MSE=float(centered[ti,:,bi].mean()),common_offset_MSE=float(common[ti,:,bi].mean()),
                    initial_cost=dict(zip(costnames,init[ti].tolist())),feedback_cost=dict(zip(costnames,cost[ti,bi].tolist())),
                    E_service_cost=dict(zip(costnames,svc[ti,mode,bi].tolist())),
                    complete_system_cost=dict(zip(costnames,(init[ti]+cost[ti,bi]+svc[ti,mode,bi]).tolist())))
                table[track][name][str(B)]=row
    contrasts={}
    for Bindex,B in enumerate(CHECKPOINTS):
        for ti,track in enumerate(TRACKS):
            for mode,name in enumerate(['UCB','MeanOnlyDiagnostic']):
                for baseline_idx,baseline_name in enumerate(['Ridge','KRR']):
                    r,unused=paired_interval(perq[ti,mode,Bindex]-previous[baseline_idx,Bindex],task,fold,family=2)
                    contrasts[f'{track}:{name}-{baseline_name}@{B}']=r
    result=dict(status='COMPLETE',scientific_scope='author-code controlled six-model migration; native paper performance not reproduced',
        original_questions=600,formal_paths=45,generation_seeds=[42,137,2024],algorithm_seeds=SEEDS,
        tracks=TRACKS,checkpoints=CHECKPOINTS,main_service='UCB withalpha3.5',secondary_service='alpha0 mean diagnostic frozen before evaluation',
        initial_fixed_accuracy=float(baseline.mean()),table=table,primary_comparisons=primary,exploratory_contrasts=contrasts,
        input_permissions='CES additionally uses I member-profile scores, paid and logged; E7 base learners did not receive that side information. Same I/U IDs and final-combination score packages do not imply identical total features.',
        feedback_source='E7 exact Random U schedule for everyfold/seed, no new active acquisition method',
        warmstart_units='Each delivered combination mean is one author update; I pretraining4800 labels, raw scoring14400.',
        CPU_scope='Windows process_time may round short planning/update steps to0, path wall time also saved',
        cost_scope='mean per fold/seed. Source-cache costs counterfactual and actual Token lengths known only after purchase. No equal-execution claim forSingle vsFull.',
        physical_new_LLM_calls=0,paths=[{k:p[k] for k in ['fold','seed','track','planning_CPU_s','update_CPU_s','total_CPU_s','wall_s']} for p in metadata])
    target=PRIVATE/'MIGRATION_PAIRED.npz'
    np.savez_compressed(target,Q=ids,task=task,sub=sub,fold=fold,score=scores,perq=perq,predicted_means=predicted,
        actions=selected,centered_error=centered,common_offset_error=common,baseline=baseline,previous=previous,
        bootstrap=boot,bootstrap_relative=relative,primary_boot=np.asarray(primary_boot),
        initial_cost=init,feedback_cost=cost,E_service_cost=svc,costnames=np.asarray(costnames))
    result['paired_sha256']=sha(target)
    put(PUBLIC/'MIGRATION_RESULTS.json',result)
    print(json.dumps(dict(status='COMPLETE',primary=primary,accuracy={t:{s:table[t][s]['60']['accuracy'] for s in table[t]} for t in TRACKS})),flush=True)


if __name__=='__main__':
    main()
