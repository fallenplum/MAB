"""Two expected-resource minimax second-moment designs and numerical KKT checks."""
from certification import np
from scipy.optimize import minimize,nnls
import warnings


def _inputs(weights,moments,costs,gates,p_min,B_ann,B_exec):
    w,a,c,g=map(lambda x:np.asarray(x,float),(weights,moments,costs,gates))
    if w.ndim!=1 or a.shape!=w.shape or c.shape!=w.shape or g.ndim!=2 or g.shape[1]!=len(w):
        raise ValueError('Use K strata and J by K binary gates')
    if not all(np.isfinite(x).all() for x in (w,a,c,g)) or np.any(w<=0) or w.sum()>1+1e-12:
        raise ValueError('Finite positive full-flow weights may sum to at most one')
    if np.any((a<0)|(a>1)) or np.any(c<=0) or not np.isin(g,(0,1)).all() or not 0<p_min<1:
        raise ValueError('Invalid moments, costs, gates or p_min')
    active=g.any(0)
    minimum_ann=float(w[active].sum()*p_min)
    minimum_exec=float((w[active]*c[active]).sum()*p_min)
    if B_ann<minimum_ann-1e-12 or B_exec<minimum_exec-1e-12:
        raise ValueError('The positive-propensity design is infeasible at these budgets')
    return w,a,c,g,active


def second_moments(weights,moments,gates,probabilities):
    w,a,g,p=map(lambda x:np.asarray(x,float),(weights,moments,gates,probabilities))
    active=g.any(0)
    if np.any(p[active]<=0):raise ValueError('Active coordinates require positive propensities')
    return (g[:,active]*(w[active]*a[active]/p[active])).sum(1)


def gate_dominance(gates):
    g=np.asarray(gates,int)
    if g.ndim!=2 or not np.isin(g,(0,1)).all():raise ValueError('Binary J by K gates required')
    pairs=[]
    for j in range(len(g)):
        for k in range(len(g)):
            if j!=k and np.all(g[j]<=g[k]) and np.any(g[j]<g[k]):pairs.append([j,k])
    unique=[]
    for j in range(len(g)):
        if not any(np.array_equal(g[j],g[k]) for k in unique):unique.append(j)
    maximal=[j for j in unique if not any(a==j for a,b in pairs)]
    return dict(pointwise_dominated_pairs=pairs,maximal_gate_indices=maximal,
                unique_gate_count=len(unique),objective_reduces_to_single_gate=len(maximal)==1,
                interpretation='nonnegative a_h means every pointwise dominated gate has a smaller or equal second moment at every design')


def priority_design(weights,costs,gates,priority,B_ann,B_exec,p_min=.05):
    w,c,g,q=map(lambda x:np.asarray(x,float),(weights,costs,gates,priority))
    _,_,_,_,active=_inputs(w,np.zeros_like(w),c,g,p_min,B_ann,B_exec)
    q=np.maximum(q,1e-12)
    def at(scale):
        result=np.zeros_like(w)
        result[active]=np.clip(scale*q[active],p_min,1.)
        return result
    def feasible(p):return w@p<=B_ann+1e-12 and (w*c)@p<=B_exec+1e-12
    lower,upper=0.,1.
    while feasible(at(upper)) and np.any(at(upper)[active]<1):upper*=2
    for _ in range(100):
        middle=(lower+upper)/2
        if feasible(at(middle)):lower=middle
        else:upper=middle
    return at(lower)


def feasibility_backoff(probabilities,weights,costs,gates,B_ann,B_exec,p_min=.05):
    p,w,c,g=map(lambda x:np.asarray(x,float),(probabilities,weights,costs,gates))
    active=g.any(0)
    lower=np.where(active,p_min,0.)
    difference=p-lower
    scale=1.
    for coefficient,budget in ((w,B_ann),(w*c,B_exec)):
        denominator=float(coefficient@difference)
        if denominator>0:scale=min(scale,(budget-coefficient@lower)/denominator)
    return lower+np.clip(scale,0,1)*difference,float(np.clip(scale,0,1))


def _kkt(w,a,c,g,p,t,B_ann,B_exec,p_min):
    f=(g*(w*a/p)).sum(1)
    dimension=len(p)
    columns=[];names=[]
    tol=2e-6*max(1.,t,B_ann,B_exec)
    for j in range(len(g)):
        if t-f[j]<=tol:
            columns.append(np.r_[-w*a*g[j]/p**2,-1.]);names.append(('gate',j))
    if B_ann-w@p<=tol:columns.append(np.r_[w,0.]);names.append(('annotation',0))
    if B_exec-(w*c)@p<=tol:columns.append(np.r_[w*c,0.]);names.append(('execution',0))
    for h in range(dimension):
        if p[h]-p_min<=tol:
            v=np.zeros(dimension+1);v[h]=-1;columns.append(v);names.append(('lower',h))
        if 1-p[h]<=tol:
            v=np.zeros(dimension+1);v[h]=1;columns.append(v);names.append(('upper',h))
    target=np.r_[np.zeros(dimension),-1.]
    coefficients,residual=nnls(np.column_stack(columns),target,maxiter=10000)
    lam=np.zeros(len(g));eta=np.zeros(2)
    for value,(kind,index) in zip(coefficients,names):
        if kind=='gate':lam[index]=value
        elif kind=='annotation':eta[0]=value
        elif kind=='execution':eta[1]=value
    total=lam.sum()
    if total<=0:raise RuntimeError('No simplex dual witness was recovered')
    lam/=total;eta/=total
    N=a*(lam@g)
    d=eta[0]+eta[1]*c
    minimizer=np.full(dimension,p_min)
    positive=(N>0)&(d>0)
    minimizer[positive]=np.clip(np.sqrt(N[positive]/d[positive]),p_min,1)
    minimizer[(N>0)&(d==0)]=1.
    flat=(N==0)&(d==0)
    minimizer[flat]=p[flat]
    dual=float(np.sum(w*(N/minimizer+d*minimizer))-eta[0]*B_ann-eta[1]*B_exec)
    gap=float(t-dual)
    return dict(gate_dual_weights=lam.tolist(),resource_dual_multipliers=eta.tolist(),
                primal_value=float(t),dual_value=dual,duality_gap=gap,
                stationarity_nnls_residual=float(residual),
                max_pointwise_minimizer_difference=float(np.max(abs(p-minimizer))),
                gate_complementarity=float(np.max(abs(lam*(f-t)))),
                resource_complementarity=float(max(abs(eta[0]*(w@p-B_ann)),abs(eta[1]*((w*c)@p-B_exec)))),
                simplex_sum=float(lam.sum()),zero_numerator_count=int(np.sum(N==0)),flat_coordinate_count=int(flat.sum()))


def solve_design(weights,moments,costs,gates,B_ann,B_exec,p_min=.05):
    w0,a0,c0,g0,active=_inputs(weights,moments,costs,gates,p_min,B_ann,B_exec)
    result=np.zeros_like(w0)
    if not active.any():
        return dict(probabilities=result,objective=0.,status='NO_ACTIVE_REGION',dominance=gate_dominance(g0))
    w,a,c,g=w0[active],a0[active],c0[active],g0[:,active]
    p0=priority_design(w,c,g,np.ones_like(w),B_ann,B_exec,p_min)
    if np.all(a==0):
        result[active]=p_min
        return dict(probabilities=result,objective=0.,status='ALL_MOMENTS_ZERO',
                    annotation=float(w0@result),execution=float((w0*c0)@result),
                    dominance=gate_dominance(g0),KKT=dict(duality_gap=0.,flat_coordinate_count=len(w),gate_dual_weights=[1.]+[0.]*(len(g)-1),resource_dual_multipliers=[0.,0.]))
    def f(p):return (g*(w*a/p)).sum(1)
    def objective(z):return z[-1]
    def gradient(z):return np.r_[np.zeros(len(w)),1.]
    def epigraph(z):return z[-1]-f(z[:-1])
    def epigraph_jac(z):return np.column_stack((g*(w*a/z[:-1]**2),np.ones(len(g))))
    constraints=[dict(type='ineq',fun=epigraph,jac=epigraph_jac),
        dict(type='ineq',fun=lambda z:B_ann-w@z[:-1],jac=lambda z:np.r_[-w,0.]),
        dict(type='ineq',fun=lambda z:B_exec-(w*c)@z[:-1],jac=lambda z:np.r_[-w*c,0.])]
    initial=np.r_[p0,float(f(p0).max())+.01]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        fitted=minimize(objective,initial,jac=gradient,bounds=[(p_min,1.)]*len(w)+[(0.,None)],
                        constraints=constraints,method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
    p=fitted.x[:-1]
    objective_value=float(f(p).max())
    certificate=_kkt(w,a,c,g,p,objective_value,B_ann,B_exec,p_min)
    violation=max(0.,float(w@p-B_ann),float((w*c)@p-B_exec),float(p_min-p.min()),float(p.max()-1))
    if violation>1e-7 or abs(certificate['duality_gap'])>2e-6*max(1.,objective_value):
        raise RuntimeError('Convex design lacks a sufficiently accurate primal-dual certificate: '+str((fitted.message,violation,certificate)))
    result[active]=p
    return dict(probabilities=result,objective=objective_value,status='NUMERICALLY_CERTIFIED',
                optimizer_success=bool(fitted.success),optimizer_message=str(fitted.message),iterations=int(fitted.nit),
                optimizer_warnings=[str(warning.message) for warning in captured],
                annotation=float(w0@result),execution=float((w0*c0)@result),
                max_constraint_violation=violation,KKT=certificate,dominance=gate_dominance(g0),
                assumption='known full-flow weights and expected costs, two constraints in expectation')


def qualification():
    w=np.array([.2,.3,.4,.1]);a=np.array([.1,.3,.6,.8]);c=np.array([1.,3.,8.,10.])
    g=np.array([[1,0,1,0],[0,1,1,0]])
    solved=solve_design(w,a,c,g,.4,2.)
    p=solved['probabilities']
    assert p[-1]==0 and p[:3].min()>=.05
    assert w@p<=.4+1e-8 and (w*c)@p<=2+1e-8
    uniform=priority_design(w,c,g,np.ones(4),.4,2.)
    assert solved['objective']<=second_moments(w,a,g,uniform).max()+1e-8
    nested=np.array([[1,0,0,0],[1,1,0,0],[1,1,1,0]])
    check=gate_dominance(nested)
    assert check['objective_reduces_to_single_gate'] and check['maximal_gate_indices']==[2]
    with_all=np.vstack((g,np.array([1,1,1,0])))
    assert gate_dominance(with_all)['maximal_gate_indices']==[2]
    rng=np.random.default_rng(20261110)
    for _ in range(100):
        point=np.r_[rng.uniform(.05,1,3),0.]
        f=second_moments(w,a,nested,point)
        assert np.max(f)==f[-1]
    zero=solve_design(w,np.zeros(4),c,g,.4,2.)
    assert zero['objective']==0 and np.all(zero['probabilities'][:3]==.05)
    # Analytic one-gate, annotation-only optimum p proportional to sqrt(a).
    ww=np.array([.4,.6]);aa=np.array([.1,.3]);cc=np.ones(2);gg=np.ones((1,2))
    simple=solve_design(ww,aa,cc,gg,.3,10.)
    analytic=.3*np.sqrt(aa)/(ww@np.sqrt(aa))
    assert np.max(abs(simple['probabilities']-analytic))<1e-6
    return dict(status='PASS',checks=['active_union_positivity','inactive_probability_zero','both_expected_resource_constraints',
                'primal_dual_KKT_witness','oracle_not_worse_than_uniform','nested_objective_dominance',
                'all_one_gate_dominates_arbitrary_family','zero_moment_boundary','analytic_single_gate_square_root'],example_KKT=solved['KKT'])


if __name__=='__main__':
    import json
    print(json.dumps(qualification(),indent=2))
