"""Small Bayesian binary working models. Independent-arm likelihood is composite."""
from e4_common import *
from scipy.special import expit, ndtr, log_ndtr
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from time import perf_counter


def link(value, family):
    return expit(value) if family=='logistic' else ndtr(value)


def link_derivative(value,family):
    if family=='logistic':
        p=expit(value)
        return p*(1-p)
    return np.exp(-value**2/2)/np.sqrt(2*np.pi)


def nll_terms(score, successes, repeats, family):
    if family=='logistic':
        p=expit(score)
        return repeats*np.logaddexp(0,score)-successes*score,repeats*p-successes,repeats*p*(1-p)
    lp,ln=log_ndtr(score),log_ndtr(-score)
    logpdf=-score**2/2-.5*np.log(2*np.pi)
    rplus=np.exp(logpdf-lp)
    rminus=np.exp(logpdf-ln)
    failures=repeats-successes
    loss=-successes*lp-failures*ln
    gradient=-successes*rplus+failures*rminus
    curvature=successes*rplus*(score+rplus)+failures*rminus*(-score+rminus)
    return loss,gradient,np.maximum(curvature,1e-12)


class BinaryModel:
    def __init__(self,X,B,history,rewards,alpha,family='logistic',refs=None):
        self.X,self.B=np.asarray(X),np.asarray(B)
        self.history=np.asarray(history)
        self.alpha=float(alpha)
        self.family=family
        self.question_dimension=X.shape[1]
        self.action_dimension=B.shape[1]
        self.dimension=self.question_dimension*self.action_dimension
        self.rewards=np.asarray(rewards)
        self.refs=refs
        self.observed=set()
        self.online_observations=[]
        self.fit()

    def fit(self):
        start=perf_counter()
        Xh=self.X[self.history]
        count=self.rewards.sum(1)
        repeats=self.rewards.shape[1]
        shape=(self.question_dimension,self.action_dimension)
        def objective(theta):
            W=theta.reshape(shape)
            terms,grad,_=nll_terms(Xh@W@self.B.T,count,repeats,self.family)
            value=float(terms.sum()+.5*self.alpha*(theta@theta))
            gradient=(Xh.T@grad@self.B).reshape(-1)+self.alpha*theta
            return value,gradient
        solution=minimize(objective,np.zeros(self.dimension),method='L-BFGS-B',jac=True,
                          options={'maxiter':500,'gtol':1e-7,'ftol':1e-12,'maxls':30})
        self.mean=solution.x
        score=Xh@self.mean.reshape(shape)@self.B.T
        _,gradient,curvature=nll_terms(score,count,repeats,self.family)
        F=np.einsum('np,ar->napr',Xh,self.B).reshape(len(Xh),20,self.dimension)
        flat=F.reshape(-1,self.dimension)
        H=self.alpha*np.eye(self.dimension)+(flat.T*curvature.ravel())@flat
        self.cov=cho_solve(cho_factor(H),np.eye(self.dimension))
        self.cov=(self.cov+self.cov.T)/2
        # Each score block aggregates its 3 repeated generations and 20 correlated arms.
        question_score=np.einsum('nad,na->nd',F,gradient)
        centered=question_score-question_score.mean(0)
        self.cluster_sandwich=self.cov@(centered.T@centered)*len(Xh)/max(len(Xh)-1,1)@self.cov
        self.fit_info=dict(success=bool(solution.success),message=str(solution.message),iterations=int(solution.nit),
                           objective=float(solution.fun),gradient_infinity=float(np.max(abs(solution.jac))),
                           wall_s=perf_counter()-start,composite_likelihood=True,cluster_unit='original question')
        if not solution.success:
            raise RuntimeError('MAP convergence failed: '+str(self.fit_info))
        self.initial_mean=self.mean.copy()
        self.initial_cov=self.cov.copy()

    def design(self,rows):
        return np.einsum('np,ar->napr',self.X[np.asarray(rows)],self.B).reshape(len(rows),20,self.dimension)

    def latent(self,rows):
        return self.X[np.asarray(rows)]@self.mean.reshape(self.question_dimension,self.action_dimension)@self.B.T

    def predict_many(self,rows):
        # Serving uses link at MAP, not an unlabelled posterior-predictive integration.
        return link(self.latent(rows),self.family)

    def observation(self,q,actions):
        return np.einsum('p,ar->apr',self.X[q],self.B[np.asarray(actions)]).reshape(len(actions),self.dimension)

    def update(self,q,actions,values):
        if q in self.observed:
            raise ValueError('Same question feedback must be delivered once as a joint bundle')
        actions=np.asarray(actions,dtype=int)
        y=np.asarray(values,dtype=float)
        if not 1<=len(actions)<=2 or y.shape!=actions.shape:
            raise ValueError('Sequential approximation supports one legal main/pair bundle')
        if len(np.unique(actions))!=len(actions) or np.any((y!=0)&(y!=1)):
            raise ValueError('Invalid raw binary feedback')
        H=self.observation(q,actions)
        prior=H@self.mean
        HC=H@self.cov
        S=HC@H.T
        invS=np.linalg.inv(S)
        def objective(f):
            losses,grad,_=nll_terms(f,y,1,self.family)
            delta=f-prior
            return float(losses.sum()+.5*delta@invS@delta),grad+invS@delta
        result=minimize(objective,prior,method='BFGS',jac=True,options={'gtol':1e-9,'maxiter':100})
        if np.max(abs(result.jac))>1e-5:
            raise RuntimeError('Sequential latent MAP failed')
        _,_,weights=nll_terms(result.x,y,1,self.family)
        self.mean+=HC.T@np.linalg.solve(S,result.x-prior)
        self.cov-=HC.T@np.linalg.solve(np.diag(1/weights)+S,HC)
        self.cov=(self.cov+self.cov.T)/2
        self.observed.add(int(q))
        self.online_observations.append((int(q),actions.copy(),y.copy()))

    def full_refit_online(self):
        """Exact composite MAP/Laplace comparison for the accumulated legal raw scores."""
        Xh=self.X[self.history]
        F=self.design(self.history).reshape(-1,self.dimension)
        count=self.rewards.sum(1).ravel()
        L=self.rewards.shape[1]
        obsH=np.concatenate([self.observation(q,a) for q,a,y in self.online_observations],axis=0)
        obsY=np.concatenate([y for q,a,y in self.online_observations])
        def objective(theta):
            loss,g,_=nll_terms(F@theta,count,L,self.family)
            extra,eg,_=nll_terms(obsH@theta,obsY,1,self.family)
            return float(loss.sum()+extra.sum()+.5*self.alpha*(theta@theta)),F.T@g+obsH.T@eg+self.alpha*theta
        solution=minimize(objective,self.mean,method='L-BFGS-B',jac=True,options={'maxiter':500,'ftol':1e-12,'gtol':1e-7})
        _,_,weights=nll_terms(F@solution.x,count,L,self.family)
        _,_,extra_weights=nll_terms(obsH@solution.x,obsY,1,self.family)
        precision=self.alpha*np.eye(self.dimension)+(F.T*weights)@F+(obsH.T*extra_weights)@obsH
        solution.laplace_covariance=cho_solve(cho_factor(precision),np.eye(self.dimension))
        return solution


def qualification():
    from scipy.optimize._numdiff import approx_derivative
    rng=np.random.default_rng(20261107)
    X=np.column_stack((np.ones(40),rng.normal(size=(40,3))))
    B=np.eye(4)
    # BinaryModel intentionally supports an arbitrary action count in objectives only through B.
    # Formal models use the locked twenty actions. The numerical check therefore uses twenty.
    B=np.eye(20)
    Y=rng.integers(0,2,(30,3,20)).astype(float)
    results=[]
    for family in ('logistic','probit'):
        score=np.array([-10.,-2.,0.,2.,10.])
        count=np.array([0.,1.,2.,1.,3.])
        loss,gradient,curvature=nll_terms(score,count,3,family)
        numgrad=approx_derivative(lambda z:nll_terms(z,count,3,family)[0],score)
        numh=approx_derivative(lambda z:nll_terms(z,count,3,family)[1],score)
        assert np.max(abs(np.diag(numgrad)-gradient))<1e-6
        assert np.max(abs(np.diag(numh)-curvature))<1e-6
        model=BinaryModel(X,B,np.arange(30),Y,10,family)
        before=model.cov.copy()
        model.update(31,[2,5],[0.,1.])
        assert np.linalg.eigvalsh(before-model.cov).min()>-1e-10
        assert np.all((model.predict_many([31,32])>=0)&(model.predict_many([31,32])<=1))
        refit=model.full_refit_online()
        assert refit.success
        results.append(dict(family=family,fit=model.fit_info,gradient_check=float(np.max(abs(np.diag(numgrad)-gradient))),
                            hessian_check=float(np.max(abs(np.diag(numh)-curvature))),
                            sequential_batch_mean_l2=float(np.linalg.norm(refit.x-model.mean)),
                            sequential_batch_covariance_relative_frobenius=float(np.linalg.norm(model.cov-refit.laplace_covariance)/np.linalg.norm(refit.laplace_covariance))))
    return dict(status='PASS',results=results,new_llm_calls=0)


if __name__=='__main__':
    print(json.dumps(qualification(),indent=2))
