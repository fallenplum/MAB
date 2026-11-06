"""CPU-only fixed-N6/K3 research; private assets never enter git."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json, hashlib, time, itertools
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/conditional_gain_20261106'
PRIVATE=WORK/'private'
REPO=WORK/'repo'
PUBLIC=REPO/'routing_idea2/research/conditional_gain_20261106'
SEEDS=[42,137,2024]
TEAMS=list(itertools.combinations(range(6),3))
Z=np.array([[float(i in t) for i in range(6)] for t in TEAMS])
B=np.column_stack((np.ones(20),Z,*[Z[:,i]*Z[:,j] for i,j in itertools.combinations(range(6),2)]))
PAIRS=np.array(list(itertools.combinations(range(20),2)))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.pending')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(tmp,p)
def load():
    bpath=ROOT/'artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json'
    b=read(bpath); refs={'bundle':{'path':str(bpath),'sha256':sha(bpath)}};d={}
    assert b['N']==6 and b['K']==3 and b['generation_seeds']==SEEDS
    for key in ['member_pool']:
        ref=b[key];assert sha(ref['path'])==ref['sha256'];refs[key]=ref
    op=b['aggregator_operator'];assert op['fingerprint']=='9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e'
    f=op['freeze_evidence'];assert sha(f['path'])==f['sha256'];refs['operator']=f
    for split,n in [('selector_train',600),('selector_validation',200)]:
        vals=[]
        for group in ['inputs','outcomes','matrix_receipts']:
            r=b[group][split];assert sha(r['path'])==r['sha256'];refs[split+'/'+group]=r
            if group!='matrix_receipts':
                with np.load(r['path'],allow_pickle=False) as a:vals.append({k:a[k] for k in a.files})
            else:
                receipt=read(r['path']);assert receipt['status']=='REAL_FULL_INFORMATION_READY'
                assert receipt['aggregator_fingerprint']==op['fingerprint'] and receipt['all_combinations_executed']
        x,y=vals
        assert np.array_equal(x['question_ids'],y['question_ids']) and np.array_equal(x['canonical_hashes'],y['canonical_hashes'])
        assert np.array_equal(y['generation_seeds'],SEEDS)
        Y=y['aggregate_scores'].transpose(1,0,2).astype(float)
        assert Y.shape==(n,3,20) and np.isin(Y,[0,1]).all()
        d[split]={'E':x['embeddings'].astype(float),'Y':Y,'task':x['task_labels'],'sub':x['subtask_labels'],'ids':x['question_ids'],'canonical':x['canonical_hashes'],'cost':{k:v for k,v in y.items() if k.startswith(('member_','aggregate_'))}}
    assert not set(d['selector_train']['ids'])&set(d['selector_validation']['ids'])
    assert not set(d['selector_train']['canonical'])&set(d['selector_validation']['canonical'])
    return d,refs
def subset(d,idx):return {k:(v[idx] if k!='cost' else {}) for k,v in d.items()}
def macro(v,task):
    v=np.asarray(v);return float(np.mean([v[np.asarray(task)==t].mean() for t in ['gsm8k','bbh'] if np.any(np.asarray(task)==t)]))
def selected(Y,choice):return Y[np.arange(len(Y))[:,None],np.arange(Y.shape[1])[None,:],np.asarray(choice)[:,None]]
def ci(delta,task,sub,reps=2000,seed=20261106):
    # delta is n questions x L generation replicates; retain all repeats together.
    a=np.asarray(delta);a=a[:,None] if a.ndim==1 else a
    rng=np.random.default_rng(seed);draws=[];support={}
    for t in ['gsm8k','bbh']:
        mask=np.asarray(task)==t
        if not mask.any():continue
        total=np.zeros(reps);count=int(mask.sum())
        for s in sorted(set(np.asarray(sub)[mask])):
            ids=np.flatnonzero(mask&(np.asarray(sub)==s));support[t+'/'+str(s)]=len(ids)
            means=a[ids].mean(axis=1)
            total+=means[rng.integers(0,len(ids),(reps,len(ids)))].mean(axis=1)*len(ids)/count
        draws.append(total)
    lo,hi=np.quantile(np.mean(draws,axis=0),[.025,.975])
    return {'difference':macro(a.mean(axis=1),task),'ci95':[float(lo),float(hi)],'unique_questions':len(a),'strata_support':support,'bootstrap_replicates':reps,'unit':'question_cluster_with_all_generation_repeats','posthoc_descriptive':True}
def metrics(pred,d):
    Y=d['Y'];a=selected(Y,np.argmax(pred,axis=1));ym=Y.mean(axis=1)
    di=pred[:,PAIRS[:,0]]-pred[:,PAIRS[:,1]]
    dy=Y[:,:,PAIRS[:,0]]-Y[:,:,PAIRS[:,1]];mask=dy!=0
    # Strict predicted ties score0.5; no non-tie comparison cells are silently dropped.
    rank=np.where(di[:,None,:]*dy>0,1,np.where(di[:,None,:]==0,.5,0))
    valid=mask.sum(axis=(1,2))>0
    rq=np.divide((rank*mask).sum(axis=(1,2)),mask.sum(axis=(1,2)),out=np.zeros(len(Y)),where=valid)
    delta_mean=ym[:,PAIRS[:,0]]-ym[:,PAIRS[:,1]]
    return {'accuracy':macro(a.mean(axis=1),d['task']),'top1_regret':macro((Y.max(axis=2)-a).mean(axis=1),d['task']),
        'mse':float(np.mean((pred[:,None,:]-Y)**2)),'conditional_mean_mse':float(np.mean((pred-ym)**2)),
        'pair_value_difference_mae':float(np.abs(di-delta_mean).mean()),
        'non_tie_pair_ranking_accuracy':macro(rq[valid],d['task'][valid]) if valid.any() else None,
        'rankable_questions':int(valid.sum()),'per_generation_accuracy':[macro(a[:,g],d['task']) for g in range(3)],
        'per_task_accuracy':{t:float(a[d['task']==t].mean()) for t in sorted(set(d['task']))},
        'per_bbh_subtask':{s:{'questions':int(np.sum(d['sub']==s)),'accuracy':float(a[d['sub']==s].mean())} for s in sorted(set(d['sub'][d['task']=='bbh']))}}
def defaults(train,test,kappa=0,min_sub=1):
    y=train['Y'].mean(axis=1);gm=y.mean(axis=0)
    glob=np.tile(gm,(len(test['E']),1));task=glob.copy();sub=glob.copy()
    for j,(t,s) in enumerate(zip(test['task'],test['sub'])):
        ids=train['task']==t;tm=(y[ids].sum(axis=0)+kappa*gm)/(ids.sum()+kappa) if ids.any() else gm
        task[j]=tm;sub[j]=tm
        sm=ids&(train['sub']==s)
        if t=='bbh' and sm.sum()>=min_sub:sub[j]=(y[sm].sum(axis=0)+kappa*tm)/(sm.sum()+kappa)
    return {'global_fixed':glob,'task_fixed':task,'subtask_fixed':sub}
class Transform:
    def __init__(self,kind):self.kind=kind
    def meta(self,d):
        names=['gsm8k/']+['bbh/'+s for s in self.names]
        labels=[str(t)+'/'+str(s) for t,s in zip(d['task'],d['sub'])]
        # Vocabulary is train-fold-only; unseen subtask falls back to BBH task indicator.
        z=np.array([[float(v==name) for name in names] for v in labels])
        return np.column_stack((np.asarray(d['task']=='bbh',float),z))
    def fit(self,d):
        self.names=sorted(set(d['sub'][d['task']=='bbh']))
        if self.kind.startswith('pca'):
            self.pca=PCA(int(self.kind[3:]),svd_solver='full').fit(d['E']);a=self.pca.transform(d['E'])
        else:self.pca=None;a=d['E']
        if self.kind=='full_meta':a=np.column_stack((a,self.meta(d)))
        self.scaler=StandardScaler().fit(a);self.train_ids=d['ids'].copy();return self
    def transform(self,d):
        a=self.pca.transform(d['E']) if self.pca is not None else d['E']
        if self.kind=='full_meta':a=np.column_stack((a,self.meta(d)))
        return np.column_stack((np.ones(len(a)),self.scaler.transform(a)))
def ridge(X,Y,alpha):
    # Equivalent to three replicate squared-loss observations, not 3n independent questions.
    u,s,vt=np.linalg.svd(X,full_matrices=False)
    return (vt.T*(s/(s*s+alpha/3)))@(u.T@Y)
def structured(X,Y,alpha):
    # Exact Kronecker normal equations; preserves original nonorthogonal22 basis penalty.
    ux,sx,vx=np.linalg.svd(X,full_matrices=False);ub,sb,vb=np.linalg.svd(B,full_matrices=False)
    a=ux.T@Y@ub
    c=a*(sx[:,None]*sb[None,:])/(sx[:,None]**2*sb[None,:]**2+alpha/3)
    return vx.T@c@vb
def pair_rank(X,Y,alpha):
    i,j=PAIRS.T;pos=((Y[:,:,i]-Y[:,:,j])>0).sum(axis=1);neg=((Y[:,:,i]-Y[:,:,j])<0).sum(axis=1)
    def objective(w):
        W=w.reshape(X.shape[1],20);p=X@W;v=p[:,i]-p[:,j]
        loss=(pos*np.logaddexp(0,-v)+neg*np.logaddexp(0,v)).sum()+alpha*np.sum(W*W)/2
        dv=(pos+neg)*expit(v)-pos;dp=np.zeros_like(p)
        for k in range(190):dp[:,i[k]]+=dv[:,k];dp[:,j[k]]-=dv[:,k]
        return float(loss),(X.T@dp+alpha*W).ravel()
    out=minimize(objective,np.zeros(X.shape[1]*20),jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-9})
    return out.x.reshape(X.shape[1],20),{'converged':bool(out.success),'iterations':int(out.nit),'message':str(out.message)}
def fit_predict(family,X,Y,Xtest,alpha,base_train=None,base_test=None):
    mean=Y.mean(axis=1);meta={};n=len(X)
    if family=='structured':W=structured(X,mean,alpha);return Xtest@W@B.T,X@W@B.T,meta,{'W':W}
    if family in ['direct20','lowrank5']:
        W=ridge(X,mean,alpha)
        if family=='lowrank5':
            _,_,vt=np.linalg.svd(X@W,full_matrices=False);W=W@vt[:5].T@vt[:5]
        return Xtest@W,X@W,meta,{'W':W}
    if family=='ranknet':
        W,meta=pair_rank(X,Y,alpha);return Xtest@W,X@W,meta,{'W':W}
    target=mean-mean[np.arange(n),base_train,None]
    if family=='delta_linear':W=ridge(X,target,alpha);p=Xtest@W;tr=X@W;state={'W':W}
    elif family=='delta_default_interaction':
        def augment(x,b):return np.column_stack((x,(x[:,:,None]*np.eye(20)[b,None,:]).reshape(len(x),-1)))
        xx=augment(X,base_train);xt=augment(Xtest,base_test);W=ridge(xx,target,alpha);p=xt@W;tr=xx@W;state={'W':W}
    elif family=='delta_mlp':
        model=MLPRegressor(hidden_layer_sizes=(32,),activation='relu',alpha=alpha,random_state=20261106,max_iter=300,early_stopping=True,validation_fraction=.15,n_iter_no_change=25)
        model.fit(X[:,1:],target);p=model.predict(Xtest[:,1:]);tr=model.predict(X[:,1:]);meta={'iterations':int(model.n_iter_),'early_stopping_train_only':True};state={'model':model}
    else:raise ValueError(family)
    # Reanchor predicted differences at known default action; zero is an identity, not a measured label.
    p=p-p[np.arange(len(p)),base_test,None];tr=tr-tr[np.arange(n),base_train,None]
    return p,tr,meta,state
