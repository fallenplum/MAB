"""Independent small-matrix tests for E7 posterior and observation units.

These tests qualify the implemented mathematical operators. They do not
calibrate the Gaussian working model on true LLM correctness rewards.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def solve(a,b):
    return cho_solve(cho_factor(a,lower=True),b)


def rbf(x,y,scale):
    squared=np.maximum(0,np.sum(x*x,axis=1)[:,None]+np.sum(y*y,axis=1)[None,:]-2*x@y.T)
    return np.exp(-squared/(2*scale**2))


def posterior(K,rows,Y,noise,mu):
    rows=np.asarray(rows,dtype=int)
    observed=K[np.ix_(rows,rows)]+noise*np.eye(len(rows))
    cross=K[:,rows]
    mean=mu+cross@solve(observed,Y-mu)
    covariance=K-cross@solve(observed,cross.T)
    return mean,(covariance+covariance.T)/2


def checks():
    rng=np.random.default_rng(20261112)
    errors={}
    x=np.column_stack((np.ones(30),rng.normal(size=(30,8))))
    raw=rng.integers(0,2,size=(30,3,20)).astype(float)
    alpha=100.
    allx=np.repeat(x,3,axis=0)
    rawbeta=solve(allx.T@allx+alpha*np.eye(x.shape[1]),allx.T@raw.reshape(90,20))
    meanbeta=solve(x.T@x+(alpha/3)*np.eye(x.shape[1]),x.T@raw.mean(axis=1))
    errors['raw_three_to_mean_regularization']=float(np.max(abs(rawbeta-meanbeta)))
    yy=raw.mean(axis=1)
    Klinear=x@x.T
    dualmean=Klinear@solve(Klinear+(alpha/3)*np.eye(len(x)),yy)
    errors['primal_dual_linear_mean']=float(np.max(abs(dualmean-x@meanbeta)))
    scaledK=Klinear/alpha
    _,linear_covariance=posterior(scaledK,np.arange(len(x)),yy,1/3,np.zeros(20))
    primal_covariance=x@solve(alpha*np.eye(x.shape[1])+3*x.T@x,x.T)
    errors['legacy_ridge_covariance_scale']=float(np.max(abs(linear_covariance-primal_covariance)))
    z=rng.normal(size=(35,5))
    K=rbf(z,z,1.3)
    H=np.arange(10)
    observations=rng.random((35,20))
    mu=observations[H].mean(axis=0)
    noise=.1
    mean,cov=posterior(K,H,observations[H],noise,mu)
    max_mean,max_cov=0.,0.
    purchased=[]
    for q in range(10,18):
        denominator=cov[q,q]+noise
        delta=cov[:,q].copy()/denominator
        mean=mean+delta[:,None]*(observations[q]-mean[q])[None]
        cov=cov-np.outer(delta,cov[q])
        cov=(cov+cov.T)/2
        purchased.append(q)
        rows=np.r_[H,purchased]
        batch_mean,batch_cov=posterior(K,rows,observations[rows],noise,mu)
        max_mean=max(max_mean,float(np.max(abs(mean-batch_mean))))
        max_cov=max(max_cov,float(np.max(abs(cov-batch_cov))))
    errors['sequential_batch_mean']=max_mean
    errors['sequential_batch_covariance']=max_cov
    errors['predictive_observation_variance']=float(abs((cov[20,20]+noise)-np.var(rng.normal(scale=np.sqrt(cov[20,20]+noise),size=500000))))
    n_actions=20
    latent=float(cov[20,20])
    ig=.5*n_actions*np.log1p(latent/noise)
    _,logdet=np.linalg.slogdet(np.eye(n_actions)+(latent/noise)*np.eye(n_actions))
    errors['full_packet_information_gain']=float(abs(ig-.5*logdet))
    refs=np.arange(20,35)
    point=18
    innovation=cov[refs,point]/np.sqrt(cov[point,point]+noise)
    normals=rng.normal(size=(2000,20))
    normals=np.r_[normals,-normals]
    updated=mean[refs][None]+normals[:,None,:]*innovation[None,:,None]
    kg=float(updated.max(axis=2).mean()-mean[refs].max(axis=1).mean())
    assert kg>=-1e-14
    # A shared package innovation creates correlated mean changes across
    # reference inputs, with exact covariance v_r*v_s for each output arm.
    shifts=normals[:,None,0]*innovation[None]
    empirical=np.cov(shifts,rowvar=False,ddof=0)
    exact=np.outer(innovation,innovation)
    empirical_error=float(np.max(abs(empirical-exact)))
    mc_relative=float(empirical_error/max(np.max(abs(exact)),1e-12))
    for name,value in errors.items():
        tolerance=.01 if name=='predictive_observation_variance' else 1e-11
        assert value<tolerance,(name,value)
    assert mc_relative<.1
    return dict(status='PASS',exact_unit_errors=errors,
                KG_antithetic_nonnegative=kg,shared_innovation_covariance_MC_relative_error=mc_relative,
                observation_unit='one question three-repeat mean, full20 output package',
                working_assumptions='independent output Gaussian prior with common scalar kernel and fixed noise',
                limitation='Numerical qualification only, no claim that posterior variance is calibrated for true rewards')


def main():
    result=checks()
    root=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))/'research/e7_opportunity_and_feedback_learning'
    target=root/'public/theory/WORKING_POSTERIOR_OPERATOR_CHECKS.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
