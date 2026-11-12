"""Frozen text representations and coherent Gaussian mean-reward learners.

Each observation is one question's 20-combination three-repeat mean. The
original raw-repeat Ridge objective is preserved by kernel X X'/100 and
mean-observation noise1/3. P3 uses an RBF prior and its fixed lambda as noise.
No model constructor or update method receives any unpurchased reward array.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

TOL=1e-10
LAMBDAS=(.1,1.,10.)


def argmax(values):
    x=np.asarray(values)
    return np.argmax(x>=x.max(axis=-1,keepdims=True)-TOL,axis=-1)


def reward_means(rewards):
    y=np.asarray(rewards,dtype=float)
    if y.ndim==3:
        if y.shape[1:]!=(3,20):
            raise ValueError('A raw question package has three repeats and20 actions')
        y=y.mean(axis=1)
    if y.ndim!=2 or y.shape[1]!=20 or not np.isfinite(y).all():
        raise ValueError('Only finite authorized question-by20 rewards are accepted')
    return y


def bandwidth(embeddings):
    distances=pdist(np.asarray(embeddings),metric='euclidean')
    positive=distances[distances>0]
    if not len(positive):
        return 1.
    return float(np.median(positive))


def rbf(left,right,sigma):
    return np.exp(-cdist(left,right,metric='sqeuclidean')/(2*sigma**2))


def select_lambda(embeddings,initial,initial_rewards,tasks,seed):
    """Question-CV uses only I, including fold-local mean and median bandwidth."""
    I=np.asarray(initial,dtype=int)
    y=reward_means(initial_rewards)
    if len(y)!=len(I):
        raise ValueError('Initial rewards must align exactly with I')
    split=StratifiedKFold(3,shuffle=True,random_state=seed)
    scores=np.empty((len(LAMBDAS),len(I)))
    records=[]
    for inner,(fit,held) in enumerate(split.split(I,np.asarray(tasks)[I])):
        sigma=bandwidth(np.asarray(embeddings)[I[fit]])
        mu=y[fit].mean(axis=0)
        K=rbf(embeddings[I[fit]],embeddings[I[fit]],sigma)
        Ks=rbf(embeddings[I[held]],embeddings[I[fit]],sigma)
        row=dict(inner=inner,fit_positions=fit.tolist(),held_positions=held.tolist(),bandwidth=sigma,
                 fit_questions=len(fit),held_questions=len(held),lambda_scores=[])
        for j,lam in enumerate(LAMBDAS):
            inverse=cho_factor(K+lam*np.eye(len(fit)),lower=True,check_finite=False)
            predicted=mu+Ks@cho_solve(inverse,y[fit]-mu,check_finite=False)
            actions=argmax(predicted)
            scores[j,held]=y[held,actions]
            row['lambda_scores'].append(float(scores[j,held].mean()))
        records.append(row)
    means=scores.mean(axis=1)
    best=np.flatnonzero(means>=means.max()-1e-12)[-1]
    return float(LAMBDAS[best]),dict(seed=seed,lambdas=list(LAMBDAS),CV_service_accuracy=means.tolist(),
        chosen_lambda=float(LAMBDAS[best]),tie='stronger lambda, service-gain tolerance1e-12',
        folds=records,question_heldout_scores=scores)


@dataclass
class ModelSpec:
    kind:str
    K:np.ndarray
    noise:float
    offset:np.ndarray
    initial:np.ndarray
    metadata:dict
    parameters:dict

    @property
    def kernel(self):
        return self.K

    def posterior(self,rows,rewards):
        return ScalarPosterior(self,rows,rewards)

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        arrays=dict(K=self.K,noise=np.array(self.noise),offset=self.offset,initial=self.initial,
                    kind=np.array(self.kind),metadata=np.array(json.dumps(self.metadata,ensure_ascii=False)))
        arrays.update({'parameter_'+name:value for name,value in self.parameters.items()})
        np.savez_compressed(path,**arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            return cls(str(a['kind']),a['K'].copy(),float(a['noise']),a['offset'].copy(),a['initial'].copy(),
                       json.loads(str(a['metadata'])),{name[len('parameter_'):]:a[name].copy()
                                                     for name in a.files if name.startswith('parameter_')})


def build_spec(embeddings,initial,initial_rewards,kind,*,tasks=None,cv_seed=20261112,lam=None):
    x=np.asarray(embeddings,dtype=float)
    I=np.asarray(initial,dtype=int)
    y=reward_means(initial_rewards)
    if len(y)!=len(I) or len(set(I))!=len(I) or not np.isfinite(x).all():
        raise ValueError('I inputs and legal initial package rewards must align')
    metadata=dict(kind=kind,initial_questions=len(I),observation='one question, mean of3 raw rewards for each of20 combinations',
                  frozen_preprocessing=True,feature_inputs='MiniLM question vector only',automatic_task_input=False)
    if kind=='P1':
        pca=PCA(n_components=16,svd_solver='full').fit(x[I])
        scaler=StandardScaler().fit(pca.transform(x[I]))
        design=np.column_stack((np.ones(len(x)),scaler.transform(pca.transform(x))))
        K=design@design.T/100.
        noise=1/3
        offset=np.zeros(20)
        params=dict(pca_mean=pca.mean_,pca_components=pca.components_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,design=design)
        metadata.update(raw_alpha=100.,mean_equivalent_alpha=100/3,intercept='constant input, penalized identically to original P2',
                        prior='weights iidN(0,1/100)',mean_observation_noise=1/3,components=16)
    elif kind=='P2':
        design=np.column_stack((np.ones(len(x)),x))
        K=design@design.T/100.
        noise=1/3
        offset=np.zeros(20)
        params=dict(design=design)
        metadata.update(raw_alpha=100.,mean_equivalent_alpha=100/3,intercept='penalized constant input',
                        prior='weights iidN(0,1/100)',mean_observation_noise=1/3,full_vector_dimension=x.shape[1],
                        scaling='existing frozen normalized MiniLM vector, no added rescaling')
    elif kind=='P3':
        if lam is None:
            if tasks is None:
                raise ValueError('Broad task labels are used for internal CV stratification only')
            lam,cv=select_lambda(x,I,y,tasks,cv_seed)
            cv['question_heldout_scores']=cv['question_heldout_scores'].tolist()
            metadata['lambda_selection']=cv
        sigma=bandwidth(x[I])
        K=rbf(x,x,sigma)
        noise=float(lam)
        offset=y.mean(axis=0)
        params=dict(bandwidth=np.array(sigma))
        metadata.update(lambda_value=float(lam),bandwidth=sigma,bandwidth_rule='median positive I Euclidean pair distance',
                        formula='muI + k(x,H)^T (K_H + lambda I)^-1 (Y_H - 1 muI^T)',
                        prior='independent output RBF Gaussian working processes, amplitude1',
                        offset='I combination means fixed for all later supervision')
    else:
        raise ValueError('Unknown kind')
    metadata['noise']=noise
    return ModelSpec(kind,(K+K.T)/2,noise,offset,I,metadata,params)


def batch_predict(spec,rows,rewards,queries):
    """The same posterior mean without materializing unused covariance."""
    H=np.asarray(rows,dtype=int)
    y=reward_means(rewards)
    if len(y)!=len(H) or len(set(H))!=len(H):
        raise ValueError('Authorized rows and unique question packages must align')
    factor=cho_factor(spec.K[np.ix_(H,H)]+spec.noise*np.eye(len(H)),lower=True,check_finite=False)
    coefficient=cho_solve(factor,y-spec.offset,check_finite=False)
    return spec.offset+spec.K[np.ix_(np.asarray(queries,dtype=int),H)]@coefficient


class ScalarPosterior:
    """Independent20-output GP, with shared scalar input posterior geometry."""
    def __init__(self,spec,rows,rewards):
        self.spec=spec
        self.rows=list(map(int,rows))
        if len(set(self.rows))!=len(self.rows):
            raise ValueError('Repeated package')
        self.rewards=reward_means(rewards).copy()
        if len(self.rewards)!=len(self.rows):
            raise ValueError('Rows and received rewards do not align')
        self._batch()

    def _batch(self):
        H=np.asarray(self.rows,dtype=int)
        cross=self.spec.K[:,H]
        gram=self.spec.K[np.ix_(H,H)]+self.spec.noise*np.eye(len(H))
        factor=cho_factor(gram,lower=True,check_finite=False)
        self.mean=self.spec.offset+cross@cho_solve(factor,self.rewards-self.spec.offset,check_finite=False)
        self.cov=self.spec.K-cross@cho_solve(factor,cross.T,check_finite=False)
        self.cov=(self.cov+self.cov.T)/2

    def update(self,q,received):
        q=int(q)
        if q in self.rows:
            raise ValueError('Package already observed')
        value=np.asarray(received,dtype=float)
        if value.shape==(3,20):
            value=value.mean(axis=0)
        if value.shape!=(20,) or not np.isfinite(value).all():
            raise ValueError('Update accepts one received mean package')
        vector=self.cov[:,q].copy()
        denominator=float(self.cov[q,q]+self.spec.noise)
        innovation=value-self.mean[q]
        self.mean+=vector[:,None]*innovation[None]/denominator
        self.cov-=np.outer(vector,vector)/denominator
        self.cov=(self.cov+self.cov.T)/2
        self.rows.append(q)
        self.rewards=np.vstack((self.rewards,value))
        if float(np.diag(self.cov).min()) < -1e-8:
            raise ArithmeticError('Posterior lost positive-semidefinite numerical stability')

    def clone(self):
        result=object.__new__(ScalarPosterior)
        result.spec=self.spec
        result.rows=self.rows.copy()
        result.rewards=self.rewards.copy()
        result.mean=self.mean.copy()
        result.cov=self.cov.copy()
        return result

    def predict(self,rows):
        return self.mean[np.asarray(rows,dtype=int)]

    def predict_variance(self,rows):
        return np.maximum(0,np.diag(self.cov)[np.asarray(rows,dtype=int)])

    def save_state(self,path,*,include_covariance=False):
        """A replayable snapshot stores all received labels and frozen geometry.

        The shared ModelSpec is the immutable prior/preprocessing snapshot.
        Rows and received rewards reconstruct the complete posterior exactly.
        Full covariance can additionally be stored for direct numerical audits.
        """
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        arrays=dict(rows=np.asarray(self.rows,dtype=int),received_means=self.rewards,
                    mean=self.mean,variance=np.diag(self.cov),noise=np.array(self.spec.noise),
                    kind=np.array(self.spec.kind),offset=self.spec.offset)
        if include_covariance:
            arrays['covariance']=self.cov
        np.savez_compressed(path,**arrays)


def qualification():
    rng=np.random.default_rng(20261112)
    embeddings=rng.normal(size=(45,24))
    embeddings/=np.linalg.norm(embeddings,axis=1)[:,None]
    I=np.arange(24)
    y=rng.integers(0,2,size=(45,3,20)).astype(float)
    out=[]
    for kind in ['P1','P2','P3']:
        spec=build_spec(embeddings,I,y[I],kind,lam=.1 if kind=='P3' else None)
        sequential=spec.posterior(I,y[I])
        for q in range(24,34):
            sequential.update(q,y[q])
        batch=spec.posterior(np.arange(34),y[:34])
        mean_error=float(np.max(abs(batch.mean-sequential.mean)))
        covariance_error=float(np.max(abs(batch.cov-sequential.cov)))
        assert mean_error<1e-10 and covariance_error<1e-10
        out.append(dict(kind=kind,mean_max_error=mean_error,covariance_max_error=covariance_error))
    design=rng.normal(size=(24,17))
    repeated=np.repeat(design,3,axis=0)
    raw=y[I].reshape(72,20)
    original=np.linalg.solve(100*np.eye(17)+repeated.T@repeated,repeated.T@raw)
    averaged=np.linalg.solve(100/3*np.eye(17)+design.T@design,design.T@y[I].mean(1))
    error=float(np.max(abs(original-averaged)))
    assert error<1e-12
    return dict(status='PASS',sequential_batch=out,raw_three_repeat_vs_mean_regularization_max_error=error,
                raw_alpha=100,mean_alpha=100/3,working_variance_scope='Gaussian model internal geometry, not calibrated correctness confidence')


if __name__=='__main__':
    print(json.dumps(qualification(),indent=2))
