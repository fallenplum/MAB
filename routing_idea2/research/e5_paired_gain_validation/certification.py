"""Fixed-family, fixed-horizon paired certification at the question level.

The implementation follows manuscript Eq. (66), Eq. (96), and Appendix F.1.
Certification assumptions are caller responsibilities. OOF development outcomes
do not become independent calibration data by passing through these functions.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import math
import numpy as np
from scipy.stats import beta


def _arguments(n, J, delta):
    if isinstance(n,bool) or int(n)!=n or n<2:
        raise ValueError('At least two independent question clusters are required')
    if isinstance(J,bool) or int(J)!=J or J<1:
        raise ValueError('J must cover all nondefault rules participating in selection')
    if not 0<float(delta)<.5:
        raise ValueError('delta must be between zero and one half')


def empirical_bernstein_radius(variance,n,J,delta=.05):
    _arguments(n,J,delta)
    variance=np.asarray(variance,float)
    if not np.isfinite(variance).all() or np.any(variance<0):
        raise ValueError('Use finite nonnegative unbiased question variances')
    t=math.log(6*J/delta)
    return np.sqrt(2*variance*t/n)+14*t/(3*(n-1))


def certificate_from_moments(mean,variance,n,delta=.05):
    mean,variance=np.asarray(mean,float),np.asarray(variance,float)
    if mean.ndim<1 or mean.shape!=variance.shape or not np.isfinite(mean).all():
        raise ValueError('Means and variances require a shared last gate axis')
    J=mean.shape[-1]
    radius=empirical_bernstein_radius(variance,n,J,delta)
    lcb=mean-radius
    best=np.argmax(lcb,axis=-1)
    maximum=np.max(lcb,axis=-1)
    selected=np.where(maximum>0,best,-1)
    return dict(mean=mean,variance=variance,radius=radius,lcb=lcb,ucb=mean+radius,
                selected=selected,default_gain=0.,n=int(n),J=int(J),delta=float(delta),
                rule='largest strictly positive lower bound, otherwise exact zero default')


def full_feedback_certificate(z,delta=.05):
    z=np.asarray(z,float)
    if z.ndim!=2 or not np.isfinite(z).all() or np.any(np.abs(z)>1+1e-12):
        raise ValueError('z must be finite n-question by J-gate values within [-1,1]')
    _arguments(len(z),z.shape[1],delta)
    return certificate_from_moments(z.mean(0),z.var(0,ddof=1),len(z),delta)


def stratified_certificate(z,labels,target_weights,delta=.05):
    """Combine independent within-stratum bounds, allocating delta equally.

    Fixed target weights may differ from sampling proportions. Returned radii
    use independent-question counts in each stratum, not pooled repetitions.
    """
    z=np.asarray(z,float)
    labels=np.asarray(labels)
    if z.ndim!=2 or labels.shape!=(len(z),):
        raise ValueError('Labels must identify each question row')
    weights=dict(target_weights)
    if not weights or any(v<=0 for v in weights.values()) or not np.isclose(sum(weights.values()),1):
        raise ValueError('Positive fixed target weights must sum to one')
    if set(labels.tolist())-set(weights):
        raise ValueError('Every observed stratum must have a declared target weight')
    _arguments(len(z),z.shape[1],delta)
    total_mean=np.zeros(z.shape[1])
    total_radius=np.zeros(z.shape[1])
    strata={}
    for name,w in weights.items():
        subset=z[labels==name]
        if len(subset)<2:
            raise ValueError('Each fixed stratum needs at least two independent questions')
        result=full_feedback_certificate(subset,delta/len(weights))
        strata[str(name)]=result
        total_mean+=w*result['mean']
        total_radius+=w*result['radius']
    lcb=total_mean-total_radius
    return dict(mean=total_mean,radius=total_radius,lcb=lcb,ucb=total_mean+total_radius,
                selected=int(np.argmax(lcb)) if np.max(lcb)>0 else -1,strata=strata,
                n=len(z),J=z.shape[1],delta=delta,target_weights=weights,
                assumption='within-stratum iid questions, fixed family independent of calibration outcomes')


def ipw_contributions(dbar,gates,acquired,probabilities,p_min=.05):
    """Read rewards only at acquired rows. Inactive rows contribute known zero."""
    dbar=np.asarray(dbar,float)
    gates=np.asarray(gates,float)
    acquired=np.asarray(acquired,bool)
    p=np.asarray(probabilities,float)
    n=len(dbar)
    if gates.ndim!=2 or gates.shape[0]!=n or p.shape!=(n,) or acquired.shape!=(n,):
        raise ValueError('Inputs must share an incoming-question axis')
    if not np.isin(gates,(0,1)).all() or not 0<p_min<=1:
        raise ValueError('Binary frozen gates and positive p_min are required')
    if not np.isfinite(p).all():
        raise ValueError('Every recorded propensity must be finite')
    active=gates.any(1)
    if np.any((p[active]<p_min)|(p[active]>1)) or np.any(p[~active]!=0) or np.any(acquired&~active):
        raise ValueError('Recorded propensities violate active-region positivity')
    observed=active&acquired
    if not np.isfinite(dbar[observed]).all() or np.any(np.abs(dbar[observed])>1):
        raise ValueError('Acquired question averages must be finite and bounded')
    result=np.zeros_like(gates,float)
    result[observed]=gates[observed]*dbar[observed,None]/p[observed,None]
    return result


def ipw_certificate(contributions,p_min=.05,delta=.05):
    y=np.asarray(contributions,float)
    if y.ndim!=2 or not np.isfinite(y).all() or not 0<p_min<=1:
        raise ValueError('Finite T by J contributions and positive p_min are required')
    T,J=y.shape
    _arguments(T,J,delta)
    if np.any(abs(y)>1/p_min+1e-10):
        raise ValueError('Contribution exceeds the declared propensity bound')
    u=math.log(2*J/delta)
    radius=math.sqrt(2*u/(T*p_min))+4*u/(3*T*p_min)
    mean=y.mean(0)
    lcb=mean-radius
    return dict(mean=mean,radius=radius,lcb=lcb,ucb=mean+radius,
                selected=int(np.argmax(lcb)) if lcb.max()>0 else -1,T=T,J=J,p_min=p_min,
                delta=delta,rule='Appendix F fixed horizon, not iid empirical variance on weighted rows')


def sufficient_questions(rho,gain,J,delta=.05):
    if not 0<gain<=rho<=1:
        return None
    t=math.log(6*J/delta)
    return math.floor(max(72*rho*t/gain**2,24*t/gain,2))+1


def necessary_questions(rho,gain,delta=.05):
    if not 0<gain<=rho/2<=.5:
        return None
    kl=(1-delta)*math.log((1-delta)/delta)+delta*math.log(delta/(1-delta))
    return math.ceil(rho*kl/(4*gain**2))


def detectable_gain(n,variance,J,delta=.05):
    return float(empirical_bernstein_radius(float(variance),n,J,delta))


def binomial_interval(successes,trials,alpha=.05):
    k,n=int(successes),int(trials)
    if not 0<=k<=n or n<=0:raise ValueError('Invalid Monte Carlo count')
    return [0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]


def qualification():
    z=np.array([[0.,1.],[1.,0.],[-1.,1.],[0.,0.]])
    cert=full_feedback_certificate(z)
    t=math.log(12/.05)
    expected=np.sqrt(2*z.var(0,ddof=1)*t/4)+14*t/9
    assert np.max(abs(cert['radius']-expected))<1e-12
    assert cert['selected']==-1
    strat=stratified_certificate(np.vstack((z,z)),np.array(['a']*4+['b']*4),{'a':.5,'b':.5})
    assert np.allclose(strat['mean'],z.mean(0))
    assert np.allclose(strat['radius'],full_feedback_certificate(z,.025)['radius'])
    d=np.array([1.,np.nan,-1.,np.nan])
    g=np.array([[1,1],[1,0],[1,1],[0,0]])
    p=np.array([.5,.2,.5,0])
    y=ipw_contributions(d,g,np.array([1,0,1,0]),p)
    assert np.array_equal(y,np.array([[2,2],[0,0],[-2,-2],[0,0]]))
    c=ipw_certificate(y)
    assert c['T']==4 and c['selected']==-1
    assert sufficient_questions(.1,.01,20)>400
    assert necessary_questions(.1,.01)>0
    for bad in (np.array([np.nan,.2,.5,0]),np.array([.01,.2,.5,0])):
        try:
            ipw_contributions(d,g,np.array([1,0,1,0]),bad)
            raise AssertionError('Invalid propensity accepted')
        except ValueError:
            pass
    return dict(status='PASS',checks=['exact_Eq66','zero_default','stratified_delta_and_weight_combination',
                'inactive_and_unpurchased_reward_not_read','IPW_all_incoming_denominator','fixed_horizon_Eq96'],
                empirical_radius=cert['radius'].tolist(),ipw_radius=c['radius'])


if __name__=='__main__':
    import json
    print(json.dumps(qualification(),indent=2))
