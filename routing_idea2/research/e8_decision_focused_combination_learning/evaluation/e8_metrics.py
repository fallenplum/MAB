"""Question-cluster metrics for frozen E8 predictions.

All functions are evaluator-only. Calibration is fitted on legal historical
OOF predictions and never changes the raw-score routing action.
"""
from __future__ import annotations

import itertools
import numpy as np

PAIRS=np.asarray(list(itertools.combinations(range(20),2)),dtype=int)
I,J=PAIRS.T


def affine_calibration(oof_scores,training_mean_rewards):
    scores=np.asarray(oof_scores,dtype=float).reshape(-1)
    rewards=np.asarray(training_mean_rewards,dtype=float).reshape(-1)
    if scores.shape!=rewards.shape or not np.isfinite(scores).all():
        raise ValueError('Historical OOF scores and rewards must align')
    centered=scores-scores.mean()
    denominator=float(centered@centered)
    slope=max(0.,float(centered@(rewards-rewards.mean())/denominator)) if denominator>1e-20 else 0.
    intercept=float(rewards.mean()-slope*scores.mean())
    return dict(slope=slope,intercept=intercept,
        source='historical question OOF scores only, one common nonnegative affine map over all20 actions',
        clipping=False,used_for_service_decisions=False)


def calibrate(scores,parameters):
    return parameters['slope']*np.asarray(scores)+parameters['intercept']


def _row_mean_with_support(values,mask):
    counts=mask.sum(axis=1)
    means=np.divide(np.sum(np.where(mask,values,0.),axis=1),counts,
        out=np.full(len(values),np.nan),where=counts>0)
    return means,counts


def frozen_metrics(rewards,raw_scores,calibrated_scores,default_actions,reference_scores):
    rewards=np.asarray(rewards,dtype=float)
    if rewards.ndim!=3 or rewards.shape[1:]!=(3,20):
        raise ValueError('Each original question must contain 3 repeats and20 actions')
    n=len(rewards);rows=np.arange(n)
    mean_rewards=rewards.mean(axis=1)
    q=np.asarray(raw_scores,dtype=float);cq=np.asarray(calibrated_scores,dtype=float)
    actions=np.argmax(q,axis=1)
    realized=rewards[rows,:,actions]
    baseline=rewards[rows,:,np.asarray(default_actions,dtype=int)]
    top=np.argsort(-np.asarray(reference_scores),axis=1,kind='stable')[:,:3]
    membership=np.zeros((n,20),dtype=bool)
    membership[rows[:,None],top]=True
    competition=membership[:,I]|membership[:,J]
    true_gap=mean_rewards[:,I]-mean_rewards[:,J]
    strict=true_gap!=0.
    raw_gap=q[:,I]-q[:,J]
    calibrated_gap=cq[:,I]-cq[:,J]
    prediction_tie=np.abs(raw_gap)<=1e-12
    correct_rank=np.where(prediction_tie,.5,(np.sign(raw_gap)==np.sign(true_gap)).astype(float))
    ranking,ranking_support=_row_mean_with_support(correct_rank,strict)
    competitor_ranking,competitor_support=_row_mean_with_support(correct_rank,strict&competition)
    all_pair_mae=np.mean(np.abs(calibrated_gap-true_gap),axis=1)
    competitor_mae,_=_row_mean_with_support(np.abs(calibrated_gap-true_gap),competition)
    unrelated_mae,_=_row_mean_with_support(np.abs(calibrated_gap-true_gap),~competition)
    raw_error=q-mean_rewards;calibrated_error=cq-mean_rewards
    result=dict(actions=actions,per_generation_rewards=realized,
        accuracy=realized.mean(axis=1),default_per_generation_rewards=baseline,
        default_accuracy=baseline.mean(axis=1),
        corrected=((baseline==0)&(realized==1)).sum(axis=1),
        destroyed=((baseline==1)&(realized==0)).sum(axis=1),
        net_corrections=((baseline==0)&(realized==1)).sum(axis=1)-((baseline==1)&(realized==0)).sum(axis=1),
        raw_absolute_MSE=np.mean(raw_error**2,axis=1),
        raw_centered_MSE=np.mean((raw_error-raw_error.mean(axis=1,keepdims=True))**2,axis=1),
        calibrated_absolute_MSE=np.mean(calibrated_error**2,axis=1),
        calibrated_centered_MSE=np.mean((calibrated_error-calibrated_error.mean(axis=1,keepdims=True))**2,axis=1),
        strict_pair_rank_accuracy=ranking,strict_pair_count=ranking_support,
        strict_competitor_rank_accuracy=competitor_ranking,strict_competitor_pair_count=competitor_support,
        calibrated_all_pair_gap_MAE=all_pair_mae,
        calibrated_competitor_gap_MAE=competitor_mae,
        calibrated_unrelated_gap_MAE=unrelated_mae,
        raw_competitor_gap_MAE=np.mean(np.abs(raw_gap-true_gap)*competition,axis=1)/(competition.mean(axis=1)),
        posthoc_three_repeat_selection_loss=mean_rewards.max(axis=1)-mean_rewards[rows,actions],
        competition_pair_mask=competition,actual_selected_combo_changed=actions!=default_actions)
    return result


def summarize_metrics(metrics,mask=None):
    n=len(metrics['accuracy']);mask=np.ones(n,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    result=dict(questions=int(mask.sum()))
    for key,values in metrics.items():
        values=np.asarray(values)
        if key in ['actions','competition_pair_mask']:continue
        selected=values[mask]
        if selected.ndim==1:
            finite=np.isfinite(selected)
            result[key]=float(selected[finite].mean()) if finite.any() else None
            if key.startswith('strict_') and 'accuracy' in key:
                result[key+'_supported_questions']=int(finite.sum())
        elif key in ['per_generation_rewards','default_per_generation_rewards']:
            result[key]=selected.mean(axis=0).tolist()
    for key in ['corrected','destroyed','net_corrections']:
        result[key+'_question_generation_events']=int(np.sum(metrics[key][mask]))
    result['net_accuracy_vs_global_fixed']=result['accuracy']-result['default_accuracy']
    return result


def bootstrap_matrix(task,fold,repetitions=10000,seed=20261113):
    """Stratified original-question resampling with all methods/repeats attached."""
    task=np.asarray(task);fold=np.asarray(fold);n=len(task)
    rng=np.random.default_rng(seed)
    samples=np.empty((repetitions,n),dtype=np.int32);offset=0
    for fold_id in sorted(set(fold.tolist())):
        for label in ['gsm8k','bbh']:
            rows=np.flatnonzero((fold==fold_id)&(task==label))
            samples[:,offset:offset+len(rows)]=rng.choice(rows,size=(repetitions,len(rows)),replace=True)
            offset+=len(rows)
    if offset!=n:raise ValueError('Unexpected task or fold values')
    return samples


def paired_statistics(candidate,baseline,samples):
    delta=np.asarray(candidate)-np.asarray(baseline)
    observed=float(delta.mean())
    means=delta[samples].mean(axis=1)
    lower,upper=np.quantile(means,[.025,.975])
    if np.all(delta==0):pvalue=1.
    else:pvalue=float((1+np.sum(np.abs(means-observed)>=abs(observed)-1e-15))/(len(means)+1))
    return dict(gain=observed,paired95=[float(lower),float(upper)],
        bootstrap_two_sided_centered_p=pvalue,questions=len(delta),bootstrap_repetitions=len(samples),
        interval_scope='conditional development cross-validation, overlapping training histories')


def holm(comparisons):
    order=np.argsort([x['bootstrap_two_sided_centered_p'] for x in comparisons])
    running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(comparisons)-rank)*comparisons[index]['bootstrap_two_sided_centered_p']))
        comparisons[index]['p_Holm']=running
        comparisons[index]['passes_positive_primary']=bool(comparisons[index]['gain']>0 and running<.05)
    return comparisons
