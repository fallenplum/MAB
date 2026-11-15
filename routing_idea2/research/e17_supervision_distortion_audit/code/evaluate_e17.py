"""Grade immutable E17 predictions using only the reused train600 rewards."""
from __future__ import annotations

import time

from e17_common import (AUDIT, CODE, MC_COUNTS, METHODS, NS, NUMERIC_TOLERANCE,
    OUTCOMES, INPUTS, POLICIES, PRIVATE, PUBLIC, SEEDS, input_packet, load_npz,
    read_json, sha256, timing, verify_protocol, write_csv, write_json)
from objectives import select_actions
import numpy as np


def describe(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),sd=float(values.std(ddof=0)),
                variance=float(values.var(ddof=0)),
                minimum=float(values.min()),maximum=float(values.max()))


def frequency(actions):
    counts=np.bincount(actions,minlength=20)
    shares=counts/counts.sum()
    positive=shares[shares>0]
    return dict(counts=counts,max_share=float(shares.max()),
                effective_actions=float(np.exp(-np.sum(positive*np.log(positive)))))


def choose(scores,perm,policy):
    return select_actions(scores,perm,
        NUMERIC_TOLERANCE if policy.endswith("near") else 0,
        semantic=policy.startswith("semantic"))


def reward_for(rewards,actions):
    return rewards[:,np.arange(len(actions)),actions]


def paired(a,b):
    difference=a-b
    return dict(accuracy_difference_pp=100*float(difference.mean()),
                corrections_events=int(np.sum(difference>0)),
                corruptions_events=int(np.sum(difference<0)),
                net_correct_events=int(np.sum(difference)),
                improved_questions=int(np.sum(difference.mean(axis=0)>0)),
                harmed_questions=int(np.sum(difference.mean(axis=0)<0)))


def value_metrics(scores,rewards,calibrated):
    centered=scores-scores.mean(axis=1,keepdims=True)
    target=rewards.mean(axis=0)
    first,second=np.triu_indices(20,1)
    gap=target[:,first]-target[:,second]
    predicted=centered[:,first]-centered[:,second]
    valid=gap!=0
    centered_error=None;relative_error=None
    if calibrated is not None:
        residual=calibrated-(target-target.mean(axis=1,keepdims=True))
        centered_error=float(np.mean(residual**2))
        relative_error=float(np.mean(np.abs(residual[:,first]-residual[:,second])))
    return dict(centered_calibrated_MSE=centered_error,
                relative_gap_calibrated_MAE=relative_error,
                strict_pair_count=int(valid.sum()),
                strict_pair_rank_accuracy=float(np.mean(np.sign(predicted[valid])==np.sign(gap[valid]))) if valid.any() else None)


def score_ties(scores):
    top=np.sort(scores,axis=1)[:,-2:]
    gap=top[:,1]-top[:,0]
    return dict(exact_score_ties=int(np.sum(gap==0)),
                near_score_ties=int(np.sum(gap<=NUMERIC_TOLERANCE)),
                top_score_gap_summary=dict(**describe(gap),median=float(np.median(gap))))


def main():
    cpu,wall=time.process_time(),time.perf_counter()
    verify_protocol()
    qualification=read_json(PUBLIC/"FROZEN_ARTIFACT_QUALIFICATION.json")
    if qualification["status"]!="PASS":raise ValueError("Frozen artifact qualification failed")
    freeze=read_json(PRIVATE/"ALL_PREDICTIONS_FROZEN.json")
    if freeze["status"]!="FROZEN_BEFORE_OUTER_REWARD_EVALUATION":raise ValueError("Unfrozen predictions")
    protocol_hash=sha256(PUBLIC/"E17_PROTOCOL.json")
    prediction_hash=sha256(PRIVATE/"ALL_PREDICTIONS_FROZEN.json")
    if qualification["training_protocol_sha256"]!=protocol_hash or freeze["protocol_sha256"]!=protocol_hash:
        raise ValueError("Qualification protocol context differs")
    if qualification["frozen_predictions_sha256"]!=prediction_hash:
        raise ValueError("Qualification prediction context differs")
    if qualification["validator_sha256"]!=sha256(CODE/"validate_frozen.py"):
        raise ValueError("Qualified validator source changed")
    for item in freeze["files"]:
        if sha256(item["path"])!=item["sha256"]:raise ValueError("Frozen prediction changed")
    source_hashes={name:sha256(CODE/name) for name in
        ("evaluate_e17.py","objectives.py","e17_common.py","confirmation.py")}
    write_json(PRIVATE/"EVALUATOR_SOURCE_FREEZE.json",dict(status="FROZEN_BEFORE_REWARD_ACCESS",
        source_sha256=source_hashes["evaluate_e17.py"],source_hashes=source_hashes,
        protocol_sha256=protocol_hash,qualification_sha256=sha256(PUBLIC/"FROZEN_ARTIFACT_QUALIFICATION.json"),
        prediction_receipt_sha256=prediction_hash))
    inputs=load_npz(INPUTS)
    outcome=load_npz(OUTCOMES)
    if not np.array_equal(inputs["question_ids"],outcome["question_ids"]):raise ValueError("Question identity mismatch")
    if tuple(outcome["generation_seeds"])!=SEEDS:raise ValueError("Seed mismatch")
    rewards=outcome["aggregate_scores"]
    if rewards.shape!=(3,600,20) or not np.isfinite(rewards).all() or not np.isin(rewards,[0,1]).all():
        raise ValueError("Expected three generations by600 questions by20 binary aggregate rewards")
    target=rewards.mean(axis=0)
    task=inputs["task_labels"];subtask=inputs["subtask_labels"]
    if task.shape!=(600,) or subtask.shape!=(600,) or set(task)!={"gsm8k","bbh"}:
        raise ValueError("Task axes differ")
    if np.sum(task=="gsm8k")!=300 or np.sum(task=="bbh")!=300:
        raise ValueError("Equal task weights require the frozen300/300 question counts")
    if len(set(subtask[task=="bbh"]))!=27:
        raise ValueError("Expected27 distinct BBH subtasks")
    permutations=load_npz(PRIVATE/"PERMUTATIONS_FROZEN.npz")["permutations"]
    scores=np.empty((101,3,600,20));scales=np.empty((101,3,600));fold_ids=np.empty(600,dtype=int)
    default=np.empty(600,dtype=int);background={name:np.empty((600,20)) for name in ("SPO_reverse","NS005")}
    train_audit=[]
    prior_actions=np.empty((101,600),dtype=np.int16)
    for fold in range(5):
        packet=input_packet(fold);H,E=packet["H"],packet["E"];fold_ids[E]=fold
        default[E]=int(target[H].mean(axis=0).argmax())
        for index,perm in enumerate(permutations):
            champion=perm[target[H][:,perm].argmax(axis=1)]
            counts=np.bincount(champion,minlength=20)
            prior_actions[index,E]=int(np.flatnonzero(counts==counts.max())[0])
        for index in range(101):
            path=PRIVATE/f"permutations/fold{fold}/p{index:03d}/MODEL.npz"
            model=load_npz(path);receipt=read_json(path.with_name("FROZEN.json"))
            if not np.array_equal(model["evaluation"],E):raise ValueError("Fold rows differ")
            scores[index][:,E]=model["scores"]
            for j in range(3):scales[index,j,E]=receipt["calibrations"][j]["nonnegative_scale"]
        for name,path in (("SPO_reverse",AUDIT/f"private/distribution/models/fold{fold}_full_spo_reverselex.npz"),
                          ("NS005",NS/f"private/runtime/training/private/fold{fold}/NS_main/MODEL.npz")):
            background[name][E]=packet["phi_e"]@load_npz(path)["coefficient"]
        nh=(target[H]==target[H].max(axis=1,keepdims=True)).sum(axis=1)
        train_audit.append(dict(fold=fold,questions=len(H),multiple_best=int(np.sum(nh>1)),
            all_actions_tied=int(np.sum(nh==20)),unique_best=int(np.sum(nh==1))))
    choices=np.empty((101,3,4,600),dtype=np.int16)
    evaluated=np.empty((101,3,4,3,600))
    for index,perm in enumerate(permutations):
        for method in range(3):
            for policy,name in enumerate(POLICIES):
                actions=choose(scores[index,method],perm,name)
                choices[index,method,policy]=actions
                evaluated[index,method,policy]=reward_for(rewards,actions)
    np.savez_compressed(PRIVATE/"PERMUTATION_EVALUATION.npz",actions=choices,rewards=evaluated,
        task_labels=task,subtask_labels=subtask,fold_ids=fold_ids,
        question_ids=inputs["question_ids"],permutations=permutations)
    masks={"all":np.ones(600,dtype=bool),"GSM8K":task=="gsm8k","BBH":task=="bbh",
        "tie_free":(target==target.max(axis=1,keepdims=True)).sum(axis=1)==1,
        "tie_heavy":(target==target.max(axis=1,keepdims=True)).sum(axis=1)>1,
        "all20_tied":target.max(axis=1)==target.min(axis=1)}
    for name in sorted(set(subtask[task=="bbh"])):masks["BBH/"+str(name)]=(task=="bbh")&(subtask==name)
    default_reward=reward_for(rewards,default)
    rows=[];summaries=[]
    for method,name in enumerate(METHODS):
        for policy,policy_name in enumerate(POLICIES):
            for index in range(101):
                a=choices[index,method,policy];r=evaluated[index,method,policy]
                q=scores[index,method]
                centered=q-q.mean(axis=1,keepdims=True)
                centered0=scores[0,method]-scores[0,method].mean(axis=1,keepdims=True)
                top=np.sort(q,axis=1)[:,-2:];gap=top[:,1]-top[:,0]
                f=frequency(a)
                rows.append(dict(method=name,inference=policy_name,permutation=index,
                    accuracy=100*float(r.mean()),identity_consistency=float(np.mean(a==choices[0,method,policy])),
                    numeric_first_share=float(np.mean(a==permutations[index,0])),
                    maximum_action_share=f["max_share"],effective_actions=f["effective_actions"],
                    score_equivariance_max_error=float(np.max(np.abs(centered-centered0))),
                    exact_score_ties=int(np.sum(gap==0)),near_score_ties=int(np.sum(gap<=NUMERIC_TOLERANCE)),
                    **paired(r,default_reward)))
            for group,mask in masks.items():
                if not mask.any():continue
                accuracy=100*evaluated[1:,method,policy,:,:,][:,:,mask].mean(axis=(1,2))
                consistency=(choices[1:,method,policy][:,mask]==choices[0,method,policy,mask]).mean(axis=1)
                cumulative_counts=np.bincount(choices[1:,method,policy][:,mask].ravel(),minlength=20)
                summaries.append(dict(method=name,inference=policy_name,group=group,questions=int(mask.sum()),
                    identity_accuracy=100*float(evaluated[0,method,policy][:,mask].mean()),
                    accuracy_across100=describe(accuracy),identity_consistency_across100=describe(consistency),
                    accuracy_variance_unit="percentage_points_squared",identity_consistency_variance_unit="fraction_squared",
                    action_counts_across100=cumulative_counts,
                    seed_accuracy_across100=100*evaluated[1:,method,policy][:,:,mask].mean(axis=(0,2)),
                    identity_action_frequency=frequency(choices[0,method,policy,mask])))
    write_csv(PUBLIC/"PERMUTATION_RUNS.csv",rows)
    write_json(PUBLIC/"PERMUTATION_SUMMARIES.json",dict(training_ties=train_audit,groups=summaries,
        default_accuracy=100*float(default_reward.mean()),random_permutations=100,identity_controls=1,
        independence="100 numberings reuse the same600 questions and are not100 independent datasets"))
    inference_controls=[]
    for method,name in enumerate(METHODS):
        for i,perm in enumerate(permutations):
            # Canonical-label SPO/Ridge/TSPO solution is just re-expressed in renamed coordinates.
            # This fixes training and isolates inference priority without redundant refitting.
            for policy,policy_name in enumerate(POLICIES):
                a=choose(scores[0,method],perm,policy_name)
                inference_controls.append(dict(method=name,permutation=i,inference=policy_name,
                    changed_vs_canonical=int(np.sum(a!=choices[0,method,policy])),
                    accuracy=100*float(reward_for(rewards,a).mean())))
    write_csv(PUBLIC/"INFERENCE_ONLY_CONTROLS.csv",inference_controls)
    prior_rows=[]
    for i in range(101):
        r=reward_for(rewards,prior_actions[i])
        prior_rows.append(dict(permutation=i,accuracy=100*float(r.mean()),
            agreement_with_SPO=float(np.mean(prior_actions[i]==choices[i,1,0])),
            SPO_minus_no_content_champion_prior_pp=100*float((evaluated[i,1,0]-r).mean()),
            GSM8K_accuracy=100*float(r[:,task=="gsm8k"].mean()),
            BBH_accuracy=100*float(r[:,task=="bbh"].mean())))
    write_csv(PUBLIC/"NO_CONTENT_CHAMPION_PRIOR.csv",prior_rows)
    order_models={};order_rows=[];order_private={}
    ridge_actions=choices[0,0,0];ridge_reward=evaluated[0,0,0]
    for count in MC_COUNTS:
        for operator in ("LossThenMean","MeanThenLoss"):
            q=np.empty((600,20));cal=np.empty_like(q)
            for fold in range(5):
                packet=input_packet(fold);E=packet["E"]
                path=PRIVATE/f"loss_order/fold{fold}/B{count}/{operator}.npz"
                fitted=load_npz(path);record=read_json(path.with_suffix(".json"))
                q[E]=fitted["scores"]
                qc=q[E]-q[E].mean(axis=1,keepdims=True)
                cal[E]=qc*record["calibration"]["nonnegative_scale"]
            a=choose(q,np.arange(20),"semantic_exact");r=reward_for(rewards,a)
            name=f"{operator}_B{count}";order_models[name]=(q,a,r)
            order_private[name+"_actions"]=a;order_private[name+"_scores"]=q
            groups=[]
            for group,mask in masks.items():
                if not mask.any():continue
                groups.append(dict(group=group,questions=int(mask.sum()),accuracy=100*float(r[:,mask].mean()),
                    seed_accuracy=100*r[:,mask].mean(axis=1),versus_Ridge=paired(r[:,mask],ridge_reward[:,mask]),
                    versus_global_fixed=paired(r[:,mask],default_reward[:,mask])))
            row=dict(method=name,MC_samples=count,operator=operator,accuracy=100*float(r.mean()),
                actions_changed_vs_Ridge=int(np.sum(a!=ridge_actions)),
                action_frequency=frequency(a),versus_Ridge=paired(r,ridge_reward),
                empirical_three_repeat_selection_loss=float(np.mean(target.max(axis=1)-target[np.arange(600),a])),
                true_Bayes_regret=None,**value_metrics(q,rewards,cal),**score_ties(q),groups=groups)
            order_rows.append(row)
    order_pairs=[]
    for count in MC_COUNTS:
        qa,aa,ra=order_models[f"LossThenMean_B{count}"];qb,ab,rb=order_models[f"MeanThenLoss_B{count}"]
        order_pairs.append(dict(MC_samples=count,MeanThenLoss_minus_LossThenMean=paired(rb,ra),
            changed_questions=int(np.sum(aa!=ab)),seed_difference_pp=100*(rb-ra).mean(axis=1)))
    mc_stability=[]
    for operator in ("LossThenMean","MeanThenLoss"):
        reference=order_models[f"{operator}_B8"][1]
        reference_reward=order_models[f"{operator}_B8"][2]
        for count in MC_COUNTS:
            _,a,r=order_models[f"{operator}_B{count}"]
            mc_stability.append(dict(operator=operator,MC_samples=count,changed_vs_B8=int(np.sum(a!=reference)),
                empirical_three_repeat_vs_B8=paired(r,reference_reward),
                empirical_seed_difference_pp=100*(r-reference_reward).mean(axis=1),
                common_B8_mean_value_gap=float(np.mean((order_models[f"MeanThenLoss_B8"][0][np.arange(600),a]-
                    order_models[f"MeanThenLoss_B8"][0][np.arange(600),reference]))),
                diagnostic="common fitted mean-head score gap; not true posterior or population reward"))
    np.savez_compressed(PRIVATE/"LOSS_ORDER_EVALUATION.npz",**order_private)
    write_json(PUBLIC/"LOSS_ORDER_RESULTS.json",dict(models=order_rows,paired_operators=order_pairs,MC_stability=mc_stability,
        primary_MC=8,posterior_draw_counts=MC_COUNTS,
        limitation="Real conditional means and true Bayes regret are unobserved; finite3-seed rewards are empirical targets"))
    reverse_actions=choose(background["SPO_reverse"],np.arange(20),"semantic_exact")
    original_actions=choices[0,1,0]
    old_transition=dict(original_combo0=int(np.sum(original_actions==0)),reverse_combo19_total=int(np.sum(reverse_actions==19)),
        changed=int(np.sum(reverse_actions!=original_actions)),
        old_combo0_to_reverse_combo19=int(np.sum((original_actions==0)&(reverse_actions==19))))
    baseline_results=[]
    for method,q in [(name,scores[0,j]) for j,name in enumerate(METHODS)]+list(background.items()):
        a=choose(q,np.arange(20),"semantic_exact");r=reward_for(rewards,a)
        calibrated=None;calibration_scope="unavailable for historical background; no newly fitted scale"
        if method in METHODS:
            j=METHODS.index(method)
            calibrated=(q-q.mean(axis=1,keepdims=True))*scales[0,j,:,None]
            calibration_scope="frozen fold-specific training-only nonnegative relative-value scale"
        baseline_results.append(dict(method=method,accuracy=100*float(r.mean()),action_frequency=frequency(a),
            seed_accuracy=100*r.mean(axis=1),versus_Ridge=paired(r,ridge_reward),
            relative_metric_calibration=calibration_scope,**value_metrics(q,rewards,calibrated),**score_ties(q)))
    from confirmation import paired_power_plan
    power=dict(TSPO_vs_Ridge=paired_power_plan(evaluated[0,2,0].mean(axis=0)-ridge_reward.mean(axis=0),task),
        TSPO_vs_SPO=paired_power_plan(evaluated[0,2,0].mean(axis=0)-evaluated[0,1,0].mean(axis=0),task))
    write_json(PUBLIC/"CONFIRMATION_POWER.json",power)
    for name,digest in source_hashes.items():
        if sha256(CODE/name)!=digest:raise ValueError("Evaluator dependency changed during scoring")
    write_json(PUBLIC/"E17_COMPLETE_RESULTS.json",dict(status="CPU_MECHANISM_EXPERIMENTS_COMPLETE_INDEPENDENT_ACCURACY_PENDING",
        questions=600,generation_events=1800,old_transition_audit=old_transition,canonical_background=baseline_results,
        permutation_summary="PERMUTATION_SUMMARIES.json",loss_order="LOSS_ORDER_RESULTS.json",
        fresh_reward_reads=0,sealed_reward_reads=0,new_LLM_calls=0,confidence_intervals=0,bootstrap_runs=0,
        protocol_sha256=sha256(PUBLIC/"E17_PROTOCOL.json"),**timing(cpu,wall)))
    write_json(PRIVATE/"REWARD_ACCESS_LOG.json",dict(train600_rewards="post-prediction-freeze evaluation",
        manifest_sha256=sha256(PRIVATE/"ALL_PREDICTIONS_FROZEN.json"),fresh_reward_reads=0,sealed_reward_reads=0))
    print("E17_EVALUATION_COMPLETE",flush=True)


if __name__=="__main__":main()
