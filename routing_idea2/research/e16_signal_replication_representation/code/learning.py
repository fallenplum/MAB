"""One fixed linear value learner for repeat and representation controls."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_ridge(phi,y,alpha):
    phi=np.asarray(phi,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    if phi.ndim!=2 or y.ndim!=2 or len(phi)!=len(y):raise ValueError("Aligned history vectors required")
    if not np.isfinite(phi).all() or not np.isfinite(y).all() or alpha<=0:raise ValueError("Invalid finite ridge data")
    normal=phi.T@phi+alpha*np.eye(phi.shape[1])
    coefficient=np.linalg.solve(normal,phi.T@y)
    return coefficient,float(np.max(np.abs(normal@coefficient-phi.T@y)))


def pca_block(train,query,k):
    train=np.asarray(train,dtype=np.float64)
    query=np.asarray(query,dtype=np.float64)
    pca=PCA(n_components=k,svd_solver="full").fit(train)
    train_projected=pca.transform(train)
    scaler=StandardScaler().fit(train_projected)
    h=scaler.transform(train_projected)
    e=scaler.transform(pca.transform(query))
    state=dict(pca_mean=pca.mean_,components=pca.components_,scaler_mean=scaler.mean_,
               scaler_scale=scaler.scale_,explained_variance_ratio=pca.explained_variance_ratio_)
    return h,e,state


def feature_designs(ml,qmatched,qfull,history,evaluation):
    # Each branch is fitted on the same legitimate history. All individual PCs
    # have unit history variance. 32+32 concatenation therefore has variance64,
    # identical to a single64-coordinate branch, without an extra branch scale.
    mh,me,ms=pca_block(ml[history],ml[evaluation],64)
    qh,qe,qs=pca_block(qmatched[history],qmatched[evaluation],64)
    fh,fe,fs=pca_block(qfull[history],qfull[evaluation],32)
    ones_h=np.ones((len(history),1));ones_e=np.ones((len(evaluation),1))
    pairs={"MiniLM16":(mh[:,:16],me[:,:16]),"MiniLM64":(mh,me),"QwenMatched64":(qh,qe),
           "FusionMatched64":(np.c_[mh[:,:32],qh[:,:32]],np.c_[me[:,:32],qe[:,:32]]),
           "FusionFull64":(np.c_[mh[:,:32],fh],np.c_[me[:,:32],fe])}
    result={name:(np.c_[ones_h,h],np.c_[ones_e,e]) for name,(h,e) in pairs.items()}
    return result,{"minilm_"+k:v for k,v in ms.items()}|{"qmatched_"+k:v for k,v in qs.items()}|{"qfull_"+k:v for k,v in fs.items()}


def balanced_order(history,task,logger,fold):
    rng=np.random.default_rng(np.random.SeedSequence([20261115016,int(logger),int(fold)]))
    first=np.asarray(history)[task[history]=="gsm8k"].copy()
    second=np.asarray(history)[task[history]=="bbh"].copy()
    rng.shuffle(first);rng.shuffle(second)
    if len(first)!=len(second):raise ValueError("History must be task balanced")
    return np.column_stack((first,second)).ravel()


def seed_subset(length,rotation):
    return np.array(sorted({(rotation+j)%3 for j in range(length)}),dtype=int)
