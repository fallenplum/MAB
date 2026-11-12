"""E7-only paths, train600 inputs and original-question statistics."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import hashlib
import json
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
E5=ROOT/'research/e5_paired_gain_validation'
E6=ROOT/'research/e6_contrast_directed_feedback'
sys.path.insert(0,str(E5/'public'))
from e5_common import load_train,TEAMS,SEEDS,put,sha,argmax
from cost_and_parameters import resources


def splits(fold):
    with np.load(PRIVATE/'E7_SPLITS_FROZEN.npz',allow_pickle=False) as a:
        return {key:a[f'f{fold}_{key}'].copy() for key in ['H','I','U','E','refs','weights']}


def holm(p):
    values=np.asarray(p,float)
    result=np.zeros(len(values))
    last=0.
    for rank,index in enumerate(np.argsort(values)):
        last=max(last,min(1.,(len(values)-rank)*values[index]))
        result[index]=last
    return result


def paired_bootstrap(contributions,task,fold,reps=5000,seed=20261112):
    """Last axis is independent original question, all methods travel together."""
    values=np.asarray(contributions,float)
    task=np.asarray(task).astype(str)
    fold=np.asarray(fold,int)
    cells=[np.flatnonzero((fold==f)&(task==t)) for f in sorted(set(fold)) for t in ['gsm8k','bbh']]
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(reps):
        ids=np.concatenate([rng.choice(cell,len(cell),replace=True) for cell in cells if len(cell)])
        weights=np.where(task[ids]=='gsm8k',.5/np.sum(task[ids]=='gsm8k'),.5/np.sum(task[ids]=='bbh'))
        draws.append(values[...,ids]@weights)
    return np.asarray(draws)


def paired_interval(contribution,task,fold,reps=5000,seed=20261112,family=4):
    x=np.asarray(contribution,float)
    weights=np.where(task=='gsm8k',.5/np.sum(task=='gsm8k'),.5/np.sum(task=='bbh'))
    point=float(x@weights)
    draws=paired_bootstrap(x,task,fold,reps,seed)
    return dict(point=point,ci95=np.quantile(draws,[.025,.975]).tolist(),
        ci_family_bonferroni=np.quantile(draws,[.05/(2*family),1-.05/(2*family)]).tolist(),
        centered_bootstrap_p=float((1+np.sum(abs(draws-point)>=abs(point)-1e-15))/(len(draws)+1))),draws
