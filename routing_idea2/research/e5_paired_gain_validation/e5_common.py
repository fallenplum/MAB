"""E5-only work products, with immutable E3/E4 dependencies and train600 inputs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import hashlib
import json
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e5_paired_gain_validation'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
E4=ROOT/'research/e4_statistical_model_audit/public'
sys.path.insert(0,str(E4))
import e4_common as previous
from e4_gaussian import GaussianTemplate
from e4_binary import BinaryModel
from run_binary import fit_model,pred_and_variance
from e4_metrics import prediction_metrics,scope_means,paired_summary
TEAMS,SEEDS=previous.TEAMS,previous.SEEDS
TOLERANCE=1e-10


def sha(path):return previous.sha(path)
def put(path,value):return previous.put(path,value)
def load_train():return previous.load_train()
def semantic(data):return previous.semantic(data)


def split_roles(data):
    original=previous.e4_splits(data)
    out=[]
    for fold in original['folds']:
        F=fold['nested'][360]
        G=np.setdiff1d(fold['history'],F)
        E=fold['stream']
        assert (len(F),len(G),len(E))==(360,120,120)
        assert not set(F)&set(G) and not set(F)&set(E) and not set(G)&set(E)
        for x in (F,G,E):assert np.sum(data['task'][x]=='gsm8k')==len(x)//2
        rng=np.random.default_rng(np.random.SeedSequence([20261110,fold['fold'],520]))
        update=np.r_[rng.permutation(G[data['task'][G]=='gsm8k'])[:10],rng.permutation(G[data['task'][G]=='bbh'])[:10]]
        update=rng.permutation(update)
        out.append(dict(fold=fold['fold'],F=F,G=G,E=E,feedback_G20=update))
    return out


def argmax(values,excluded=None):
    values=np.asarray(values)
    single=values.ndim==1
    x=values[None].copy() if single else values.copy()
    if excluded is not None:
        forbidden=np.broadcast_to(np.asarray(excluded),(len(x),))
        x[np.arange(len(x)),forbidden]=-np.inf
    maxima=np.max(x,axis=1)
    chosen=np.argmax(x>=maxima[:,None]-TOLERANCE,axis=1)
    return int(chosen[0]) if single else chosen


def predict(model,rows):
    rows=np.asarray(rows,dtype=int)
    if isinstance(model,BinaryModel):return model.predict_many(rows)
    t=model.template
    return t.X[rows]@model.mean.reshape(t.question_dimension,t.action_dimension)@t.B.T


def fit_models(data,F):
    refs=F[:54]
    models=[GaussianTemplate(data,F,p,refs,variant='N2',alpha=100).clone() for p in ('P1','P2')]
    probit,_=fit_model(data,F,'P2','text-only','probit',10)
    models.append(probit)
    return models


def identity(files):
    return {name:sha(PUBLIC/name) for name in files}
