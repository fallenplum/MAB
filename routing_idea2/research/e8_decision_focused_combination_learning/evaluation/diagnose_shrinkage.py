"""Post-evaluation descriptive check of zero-gap shrinkage.

The zero-gap vector is a numerical comparator only. It is never deployed as a
new router or entered into the two frozen primary statistical comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

BASE=Path(r'E:\语言模型组合路由\research\e8_decision_focused_combination_learning')
sys.path[:0]=[str(BASE/'public/evaluation'),str(BASE/'public/qualification')]
from e8_assets import load_historical
from e8_metrics import I,J
from run_e8_evaluation import archive,put,sha


def conditional_question_error(error,mask,questions):
    count=mask.sum(axis=1)
    supported=questions&(count>0)
    row=np.divide(np.sum(error*mask,axis=1),count,out=np.zeros(len(error)),where=count>0)
    return dict(supported_questions=int(supported.sum()),
        pair_events=int(mask[questions].sum()),
        MAE=float(row[supported].mean()) if supported.any() else None)


def main():
    completed=BASE/'public/E8_COMPLETE_RESULTS.json'
    if not completed.exists():raise RuntimeError('The original frozen evaluation must finish first')
    pred=archive(BASE/'private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz')
    data,_,_=load_historical();y=data['Y'].mean(axis=1);true_gap=y[:,I]-y[:,J]
    baseline=pred['raw__M0_original'];top=np.argsort(-baseline,axis=1,kind='stable')[:,:3]
    member=np.zeros((600,20),bool);member[np.arange(600)[:,None],top]=True
    competition=member[:,I]|member[:,J]
    masks=dict(all_competitor=competition,strict_competitor=competition&(true_gap!=0),
        tied_competitor=competition&(true_gap==0),all_unrelated=~competition)
    variants=json.loads(completed.read_text(encoding='utf-8'))['variants']
    result={}
    additive_decomposition={}
    for name in variants+['ZERO_GAP_NUMERICAL_COMPARATOR']:
        if name=='ZERO_GAP_NUMERICAL_COMPARATOR':gap=np.zeros_like(true_gap)
        else:
            q=pred['calibrated__'+name];gap=q[:,I]-q[:,J]
        error=np.abs(gap-true_gap)
        denominator=float(competition.sum())
        strict_contribution=float(np.sum(error*competition*(true_gap!=0))/denominator)
        tied_contribution=float(np.sum(error*competition*(true_gap==0))/denominator)
        additive_decomposition[name]=dict(strict_contribution=strict_contribution,
            tied_contribution=tied_contribution,total=strict_contribution+tied_contribution,
            common_competitor_pair_denominator=int(denominator))
        result[name]={}
        for group,selected in [('all',np.ones(600,bool)),('gsm8k',data['task']=='gsm8k'),('bbh',data['task']=='bbh')]:
            result[name][group]={label:conditional_question_error(error,mask,selected) for label,mask in masks.items()}
        if name!='ZERO_GAP_NUMERICAL_COMPARATOR':
            original=archive(BASE/'private/evaluation'/(name+'_QUESTION_METRICS.npz'))
            base_actions=np.argmax(baseline,axis=1)
            result[name]['changed_actions_vs_M0']=int((original['actions']!=base_actions).sum())
    output=dict(status='POSTHOC_DESCRIPTIVE_DIAGNOSTIC',
        frozen_main_results_sha256=sha(completed),
        purpose='distinguish shrinking numerical differences from learning which competitor is better',
        extra_training_or_policy_selection=False,physical_LLM_calls=0,
        denominator='mean over supported original questions within each stated task; pair events are descriptive counts',
        metrics=result,additive_error_decomposition=additive_decomposition)
    put(BASE/'public/E8_SHRINKAGE_DIAGNOSTIC.json',output)
    print(json.dumps({name:result[name]['all'] for name in ['M0_original','M3','M4','ZERO_GAP_NUMERICAL_COMPARATOR']}))


if __name__=='__main__':main()
