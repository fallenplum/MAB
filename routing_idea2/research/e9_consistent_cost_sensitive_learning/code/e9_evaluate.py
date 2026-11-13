"""Independent E9 scoring, only after every head and content control is sealed."""
from pathlib import Path
import sys
import time
from e9_common import (ACTION_KEYS, E8, NS, ROOT, ASSETS, archive, array_hash, fold_packs,
                      now, put, read, sha, verify_sources, PERMUTATION_SEEDS, permute_region)
import numpy as np
from scipy.stats import norm
from csce_operators import semantic_argmax
sys.path.insert(0,str(NS/'private/runtime/training'))
from ns_training import feature_map


def collect():
    head_lock=verify_sources('HEAD_SOURCE_LOCK.json')
    lock=read(ROOT/'private/EVALUATION_SOURCE_LOCK.json')
    if lock['source_sha256']!=sha(__file__):
        raise AssertionError('Frozen E9 evaluator changed')
    q={name:np.empty((600,20)) for name in ('B','C','D')}
    for family in ('B','C','D'):
        for seed in PERMUTATION_SEEDS:q[f'{family}_perm{seed}']=np.empty((600,20))
    fit_logs,restorations,selected,covered=[],[],{},[]
    expected_fold=np.empty(600,dtype=int)
    for outer in range(5):
        history,evaluation,splits,_=fold_packs(outer)
        folder=ROOT/f'private/models/fold{outer}'
        receipt=read(folder/'FOLD_FROZEN_RECEIPT.json')
        prediction_path=folder/'ALL_OUTER_PREDICTIONS_FROZEN.npz'
        if sha(prediction_path)!=receipt['predictions_sha256']:
            raise AssertionError('Fold score freeze changed')
        saved=archive(prediction_path)
        indices=evaluation['global_indices'];covered.extend(indices.tolist());expected_fold[indices]=outer
        if not np.array_equal(saved['evaluation_indices'],indices) or set(receipt['variants'])!=set(q):
            raise AssertionError('Missing predefined controls or invalid outer mapping')
        selected[str(outer)]=read(folder/'SELECTED_PARAMETERS_FROZEN.json')
        receipts=list(folder.rglob('FROZEN_RECEIPT.json'))
        if len(receipts)!=39:raise AssertionError('All39 predefined head slots required per fold')
        for path in receipts:
            spec=read(path);head=path.parent
            if spec['training_source_sha256']!=head_lock['sources']['e9_train.py'] or spec['head_source_sha256']!=head_lock['sources']['e9_heads.py']:
                raise AssertionError('Model source does not match the frozen training contract')
            for name,key in (('MODEL.npz','model_sha256'),('PREDICTIONS.npz','prediction_sha256'),('FIT_LOG.json','fit_log_sha256')):
                if sha(head/name)!=spec[key]:raise AssertionError('Head artifact changed')
            state,pred=archive(head/'MODEL.npz'),archive(head/'PREDICTIONS.npz')
            relative=head.relative_to(folder).parts
            if relative[0].startswith('inner'):
                inner=int(relative[0][5:]);rows=splits[inner][1]
                embeddings,ids=history['E'][rows],history['ids'][rows]
                fit_rows=splits[inner][0]
            else:embeddings,ids=evaluation['E'],evaluation['ids']
            if not relative[0].startswith('inner'):fit_rows=np.arange(480)
            if array_hash(history['ids'][fit_rows])!=spec['fit_ids_hash'] or array_hash(ids)!=spec['evaluation_ids_hash']:
                raise AssertionError('Fitting identities differ from the legal region')
            teacher=spec['teacher_reference']
            if teacher is not None:
                teacher_path=Path(teacher.get('path',teacher.get('source_path')))
                expected_hash=teacher.get('sha256',teacher.get('source_sha256'))
                if sha(teacher_path)!=expected_hash:raise AssertionError('Model teacher artifact changed')
                expected_labels=archive(teacher_path)['posterior_mean']
            elif relative[0]=='controls':
                seed=int(relative[1].split('_perm')[1])
                expected_labels=permute_region(history,seed)[0]['Y'].mean(axis=1)
            else:expected_labels=history['Y'][fit_rows].mean(axis=1)
            if array_hash(expected_labels)!=spec['labels_hash']:
                raise AssertionError('Model labels differ from their declared reward/teacher source')
            if not np.array_equal(ids,pred['evaluation_ids']):raise AssertionError('Head identity mapping differs')
            restored=feature_map(state).phi(embeddings)@state['coefficient']
            error=float(abs(restored-pred['scores']).max())
            if error>1e-12 or not np.array_equal(semantic_argmax(restored,ACTION_KEYS),pred['actions']):
                raise AssertionError('Saved predictor cannot restore its decisions')
            if relative[0] in ('outer','controls'):
                variant=relative[1]
                if not np.array_equal(saved['scores__'+variant],pred['scores']) or not np.array_equal(saved['actions__'+variant],pred['actions']):
                    raise AssertionError('Fold aggregate scores differ from the sealed individual head')
            diagnostic=read(head/'FIT_LOG.json')
            if not diagnostic['accepted']:raise AssertionError('Unqualified optimization entered scoring')
            fit_logs.append(dict(outer=outer,phase=relative[0],name='/'.join(relative),**diagnostic))
            restorations.append(dict(outer=outer,head='/'.join(relative),max_score_error=error))
        for name in q:
            q[name][indices]=saved['scores__'+name]
    if sorted(covered)!=list(range(600)) or len(fit_logs)!=195:
        raise AssertionError('All original outer questions and195slots must be frozen')
    old_path=E8/'private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz'
    old_receipt=read(E8/'private/evaluation/CALIBRATION_FROZEN_RECEIPT.json')
    if sha(old_path)!=old_receipt['prediction_and_calibration_sha256']:
        raise AssertionError('Old baseline score hash differs')
    old=archive(old_path)
    if not np.array_equal(old['fold'],expected_fold):raise AssertionError('E8 aggregate fold order differs')
    q.update(A=old['raw__M0_tuned'],M0_original=old['raw__M0_original'],M4=old['raw__M4'])
    if not np.array_equal(q['A'].argmax(1),q['M0_original'].argmax(1)):
        raise AssertionError('A and fixed anchor choices differ')
    prior_dir=ASSETS/'research/e8_m4_mechanism_audit/private/distribution'
    if sha(prior_dir/'INTERVENTIONS_FROZEN.npz')!=read(prior_dir/'INTERVENTIONS_FROZEN_RECEIPT.json')['prediction_sha256']:
        raise AssertionError('Frozen TSPO cache differs')
    prior=archive(prior_dir/'INTERVENTIONS_FROZEN.npz')
    if not np.array_equal(prior['fold'],expected_fold):raise AssertionError('TSPO fold order differs')
    q['TSPO']=prior['raw__full_spo_uniform']
    ns_path=NS/'private/runtime/training/private/ALL_PREDICTIONS_FROZEN.npz'
    ns_receipt=read(ns_path.parent/'ALL_PREDICTIONS_FROZEN_RECEIPT.json')
    if sha(ns_path)!=ns_receipt['predictions_sha256']:raise AssertionError('Frozen NS scores changed')
    ns=archive(ns_path)
    if not np.array_equal(ns['fold'],expected_fold):raise AssertionError('NS fold order differs')
    q['NS']=ns['scores__NS_main'];q['Davidson']=ns['scores__Davidson_main']
    return q,old,fit_logs,restorations,selected


def interval(values):return np.quantile(values,[.025,.975]).tolist()


def holm(pvalues):
    ordered=sorted(pvalues,key=pvalues.get);adjusted={};previous=0.
    for i,name in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*pvalues[name]));adjusted[name]=previous
    return adjusted


def score_distribution(scores,actions):
    counts=np.bincount(actions,minlength=20);p=counts[counts>0]/len(actions)
    result=dict(counts=counts,top_action=int(counts.argmax()),top_fraction=float(counts.max()/len(actions)),
        effective_combinations=float(np.exp(-np.sum(p*np.log(p)))))
    if scores is not None:
        gap=np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2]
        span=np.ptp(scores,axis=1)
        result.update(top_gap_min=float(gap.min()),top_gap_median=float(np.median(gap)),
            top_gap_max=float(gap.max()),score_span_mean=float(span.mean()),score_span_max=float(span.max()),
            near_ties_1e8=int((gap<1e-8).sum()),near_ties_1e12=int((gap<1e-12).sum()),
            exact_ties=int(((scores==scores.max(axis=1,keepdims=True)).sum(axis=1)>1).sum()),
            score_mean=float(scores.mean()),score_std=float(scores.std()),
            score_scale_scope='raw uncalibrated head score, CSCE softmax is not correctness probability')
    return result


def power_plan(question_difference,task):
    variance=.5*sum(float(np.var(question_difference[task==t],ddof=1)) for t in ('gsm8k','bbh'))
    sigma=np.sqrt(variance);critical=norm.ppf(.95);z80=norm.ppf(.8)
    if sigma<=1e-15:
        return dict(status='UNDEFINED_ZERO_DEVELOPMENT_PAIR_VARIANCE',observed_gain=float(question_difference.mean()),
            scope='no meaningful sample-size extrapolation from this degenerate empirical difference')
    return dict(question_pair_sigma=float(sigma),observed_gain=float(question_difference.mean()),
        scope='planning plug-in conditional OOF variance, future frozen model and population may differ',
        one_sided_alpha=.05,detect_positive_one_pp_n80=int(np.ceil(variance*(critical+z80)**2/.01**2)),
        for_200_and_400=[dict(n=n,power_for_one_pp=float(norm.sf(critical-.01*np.sqrt(n)/sigma)),
            minimum_gain_for_power80=float(sigma*(critical+z80)/np.sqrt(n))) for n in (200,400)],
        practical_margin=.01,lower_bound_above_one_pp='requires true effect above .01, power depends on effect-minus-margin')


def main(output):
    output=Path(output)
    if (output/'public/E9_MAIN_RESULTS.json').exists():raise FileExistsError('Scoring output is append-only')
    q,old,fit_logs,restorations,selected=collect()
    main_names=('A','B','C','D')
    background=('M0_original','M4','TSPO','NS','Davidson')
    actions={name:(scores.argmax(1) if name in background else semantic_argmax(scores,ACTION_KEYS)) for name,scores in q.items()}
    actions['GlobalFixed']=old['default_actions']
    seal=output/'private/ALL_E9_PREDICTIONS_FROZEN.npz'
    seal.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(seal,fold=old['fold'],**{'scores__'+k:v for k,v in q.items()},
        **{'actions__'+k:v for k,v in actions.items()})
    put(output/'private/EVALUATION_FREEZE_RECEIPT.json',dict(sealed_utc=now(),
        all_scores_sha256=sha(seal),heads=195,main_final_heads=15,control_heads=45,
        source_sha256=sha(__file__),all_controls_complete_before_scoring=True,outer_rewards_opened=False))
    sys.path.insert(0,str(E8/'public/qualification'))
    from e8_assets import load_historical
    opened=now();data,refs,_=load_historical(ASSETS)
    y,task,sub=data['Y'],data['task'],data['sub'];means=y.mean(axis=1)
    rewards={name:y[np.arange(600),:,action] for name,action in actions.items()}
    rng=np.random.default_rng(20261114091)
    samples=np.concatenate([rng.choice(rows,size=(10000,len(rows))) for f in range(5) for t in ('gsm8k','bbh')
        for rows in [np.flatnonzero((old['fold']==f)&(task==t))]],axis=1)
    def paired(left,right):
        d=rewards[left]-rewards[right];values=d.mean(axis=1);draws=values[samples].mean(axis=1);gain=float(values.mean())
        return dict(gain=gain,ordinary_paired95=interval(draws),
            unadjusted_two_sided_centered_bootstrap_p=float((1+np.sum(abs(draws-gain)>=abs(gain)))/10001),
            corrected_events=int((d>0).sum()),destroyed_events=int((d<0).sum()),net_events=int(d.sum()),
            independent_questions=600,recorded_generation_events=1800,
            seed_gain=d.mean(axis=0),task_gain={t:float(d[task==t].mean()) for t in ('gsm8k','bbh')},
            outer_gain=[float(d[old['fold']==f].mean()) for f in range(5)])
    metrics={}
    for name,values in rewards.items():
        score=q.get(name);action=actions[name]
        metrics[name]=dict(accuracy=float(values.mean()),seed_accuracy=values.mean(axis=0),
            task_accuracy={t:float(values[task==t].mean()) for t in ('gsm8k','bbh')},
            outer_accuracy=[float(values[old['fold']==f].mean()) for f in range(5)],
            bbh_subtasks={str(s):dict(n=int(((task=='bbh')&(sub==s)).sum()),
                accuracy=float(values[(task=='bbh')&(sub==s)].mean())) for s in sorted(set(sub[task=='bbh']))},
            distribution=score_distribution(score,action),
            empirical_mean_selection_loss=float((means.max(axis=1)-means[np.arange(600),action]).mean()),
            selection_loss_scope='three-realization empirical means, not true Bayes regret')
        metrics[name]['bbh_subtask_macro']=float(np.mean([v['accuracy'] for v in metrics[name]['bbh_subtasks'].values()]))
    comparisons={name:paired(left,right) for name,left,right in (
        ('C-A','C','A'),('B-A','B','A'),('D-C','D','C'),('D-B','D','B'),('D-A','D','A'))}
    adjusted=holm({name:comparisons[name]['unadjusted_two_sided_centered_bootstrap_p'] for name in ('C-A','B-A')})
    for name in ('C-A','B-A'):
        comparisons[name]['Holm_p']=adjusted[name]
        comparisons[name]['positive_Holm_support']=bool(comparisons[name]['gain']>0 and adjusted[name]<.05)
        comparisons[name]['practical_point_gain_at_least_one_pp']=bool(comparisons[name]['gain']>=.01)
        comparisons[name]['interval_scope']='ordinary95 paired interval, not Holm simultaneous interval'
    for name in main_names:
        comparisons[f'{name}-M0_original']=paired(name,'M0_original')
        comparisons[f'{name}-NS']=paired(name,'NS')
        comparisons[f'{name}-GlobalFixed']=paired(name,'GlobalFixed')
    controls={}
    for family in ('B','C','D'):
        members=[f'{family}_perm{seed}' for seed in PERMUTATION_SEEDS]
        controls[family]=dict(main_accuracy=metrics[family]['accuracy'],
            seeds=list(PERMUTATION_SEEDS),permuted_accuracy=[metrics[name]['accuracy'] for name in members],
            main_minus_permuted=[paired(family,name) for name in members],
            main_minus_mean_permutation_point=float(metrics[family]['accuracy']-np.mean([metrics[name]['accuracy'] for name in members])),
            minimum_possible_standard_MC_p=.25,
            inference='three fixed-configuration sensitivity runs, no formal content significance test or mean-control bootstrap claim')
    rename={}
    perm=np.random.default_rng(20261114096).permutation(20)
    keys=[ACTION_KEYS[i] for i in perm]
    for name in main_names:
        renamed=perm[semantic_argmax(q[name][:,perm],keys)]
        rename[name]=int((renamed!=actions[name]).sum())
        if rename[name]:raise AssertionError('Semantic relabeling changed main actions')
    cost=dict(planned_new_candidate_heads=195,saved_head_slots=len(fit_logs),
        actual_head_fits=sum(bool(v['actual_head_fit']) for v in fit_logs),
        derived_Ridge_heads=sum(not bool(v['actual_head_fit']) for v in fit_logs),
        head_cpu_seconds=sum(v['cpu_seconds'] for v in fit_logs),
        head_wall_seconds_sum=sum(v['wall_seconds'] for v in fit_logs),
        includes_teacher_or_qualification=False,fit_logs=fit_logs)
    result=dict(status='COMPLETE_EXPLORATORY_DEVELOPMENT',metrics=metrics,comparisons=comparisons,
        content_controls=controls,action_relabel_changes=rename,
        selected_hyperparameters={key:{family:{k:v for k,v in spec.items() if k!='inner_heads'}
            for family,spec in record['selected'].items()} for key,record in selected.items()},
        power_planning={name:power_plan((rewards[name]-rewards['A']).mean(axis=1),task) for name in ('B','C')},
        computational_cost=cost,model_restorations=restorations,
        statistics=dict(bootstrap=10000,seed=20261114091,unit='original question cluster',strata='outerfold x task',
            primary=['C-A','B-A'],Holm_family_size=2,ordinary_intervals=True,
            conditions='fixed fitted models, overlapping histories and reused development data'),
        physical_LLM_calls=0,retained_outcome_reads=0)
    put(output/'public/E9_MAIN_RESULTS.json',result)
    put(output/'public/E9_PRIMARY_STATISTICS.json',{name:comparisons[name] for name in ('C-A','B-A')})
    put(output/'public/E9_CONTENT_CONTROLS.json',controls)
    np.savez_compressed(output/'private/E9_PAIRED_OUTCOMES.npz',task=task,sub=sub,fold=old['fold'],
        **{'reward__'+name:value for name,value in rewards.items()})
    put(output/'private/REWARD_ACCESS_RECEIPT.json',dict(first_outer_evaluator_reward_access_utc=opened,
        frozen_prediction_sha256=sha(seal),decoded_split='selector_train600',source_refs=refs,
        source_sha256=sha(__file__),forbidden_outcomes_accessed=[],application_role_boundary=True))
    print(dict(status='COMPLETE',accuracy={k:metrics[k]['accuracy'] for k in main_names},
        primary={k:comparisons[k] for k in ('C-A','B-A')}),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT)
    args=parser.parse_args();main(args.out)
