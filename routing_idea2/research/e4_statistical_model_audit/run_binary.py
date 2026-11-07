"""History-selected mathematical predictors, unchanged question-held-out folds."""
from e4_common import *
from e4_gaussian import GaussianTemplate
from e4_binary import BinaryModel,link_derivative
from e4_metadata import make_features
from e4_metrics import prediction_metrics,scope_means
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import argparse


def fit_model(data,history,predictor,mode,family,alpha):
    X,metadata=make_features(data,history,mode)
    refs=np.asarray(history)[:min(54,len(history))]
    variant='N3' if family=='gaussian-correlated' else 'N1'
    template=GaussianTemplate(data,history,predictor,refs,variant=variant,alpha=alpha)
    if family.startswith('gaussian'):
        template.X=X
        template.question_dimension=X.shape[1]
        template.dimension=X.shape[1]*template.B.shape[1]
        template._gram=X[history].T@X[history]
        template.refF=np.einsum('mp,ar->mapr',X[refs],template.B).reshape(len(refs),20,template.dimension)
        template._initial=template._fit_initial()
        model=template.clone()
        model.fit_info={'family':family,'analytic':True}
    else:
        model=BinaryModel(X,template.B,history,data['Y'][history],alpha,family,refs=refs)
    return model,metadata


def pred_and_variance(model,query):
    if isinstance(model,BinaryModel):
        pred=model.predict_many(query)
        F=model.design(query)
        latent=model.latent(query)
        derivative=link_derivative(latent,model.family)
    else:
        t=model.template
        pred=t.X[query]@model.mean.reshape(t.question_dimension,t.action_dimension)@t.B.T
        F=np.einsum('np,ar->napr',t.X[query],t.B).reshape(len(query),20,t.dimension)
        derivative=np.ones_like(pred)
    flat=F.reshape(-1,len(model.mean))
    latent_variance=np.einsum('nd,nd->n',flat@model.cov,flat).reshape(len(query),20)
    return pred,latent_variance,latent_variance*derivative**2


def worker(job):
    f,predictor,mode=job
    path=PRIVATE/'binary'/f'f{f}_{predictor}_{mode}'
    start=perf_counter()
    data,_=load_train()
    fold=e4_splits(data)['folds'][f]
    h,q=fold['history'],fold['stream']
    code_files=['run_binary.py','e4_binary.py','e4_gaussian.py','e4_metadata.py','e4_common.py','e4_metrics.py']
    identity={name:sha(PUBLIC/name) for name in code_files}
    identity['history_rewards']=__import__('hashlib').sha256(data['Y'][h].tobytes()).hexdigest()
    if path.with_suffix('.json').exists():
        previous=json.loads(path.with_suffix('.json').read_text())
        if previous.get('identity')!=identity or not path.with_suffix('.npz').exists():
            raise ValueError('Cached binary case lacks matching code/input identity')
        with np.load(path.with_suffix('.npz')) as cached:
            assert cached['predictions'].shape==(4,120,20)
        return dict(fold=f,predictor=predictor,mode=mode,cached=True)
    labels=semantic(data)[h]
    inner=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=20261107+f).split(h,labels))
    outputs,variance,probvariance,sandwich_variance=[],[],[],[]
    details=[]
    for family in ('gaussian','gaussian-correlated','logistic','probit'):
        search=[]
        for alpha in (1.,10.,100.):
            oof=np.empty((len(h),20))
            fits=[]
            for it,iv in inner:
                model,meta=fit_model(data,h[it],predictor,mode,family,alpha)
                oof[iv]=pred_and_variance(model,h[iv])[0]
                fits.append(dict(fit_questions=h[it].tolist(),query_questions=h[iv].tolist(),fit_info=model.fit_info))
            perq=prediction_metrics(oof,data['Y'][h])['per_question']
            brier=scope_means(perq['brier'],data['task'][h],data['sub'][h])
            search.append(dict(alpha=alpha,brier=brier,inner_fits=fits))
        def scalar(item):
            value=item['brier']['balanced_v2']
            return value['mean'] if isinstance(value,dict) else value
        best=min(search,key=lambda item:(scalar(item),-item['alpha']))
        model,meta=fit_model(data,h,predictor,mode,family,best['alpha'])
        pred,latentvar,pvar=pred_and_variance(model,q)
        outputs.append(pred)
        variance.append(latentvar)
        probvariance.append(pvar)
        if isinstance(model,BinaryModel):
            flat=model.design(q).reshape(-1,model.dimension)
            sandwich_variance.append(np.einsum('nd,nd->n',flat@model.cluster_sandwich,flat).reshape(len(q),20))
        else:
            sandwich_variance.append(np.full_like(pred,np.nan))
        details.append(dict(family=family,alpha=best['alpha'],search=search,fit_info=model.fit_info,
                            metadata=meta,parameter_trace=float(np.trace(model.cov)),
                            parameter_dimension=len(model.mean),
                            cluster_sandwich_trace=float(np.trace(model.cluster_sandwich)) if isinstance(model,BinaryModel) else None))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending.npz')
    np.savez_compressed(temporary,query=q,predictions=np.asarray(outputs),
                        latent_variance=np.asarray(variance),probability_delta_variance=np.asarray(probvariance),
                        cluster_sandwich_latent_variance=np.asarray(sandwich_variance))
    os.replace(temporary,path.with_suffix('.npz'))
    put(path.with_suffix('.json'),dict(fold=f,predictor=predictor,mode=mode,models=details,
                                     wall_s=perf_counter()-start,new_llm_calls=0,identity=identity))
    return dict(fold=f,predictor=predictor,mode=mode,wall_s=perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--metadata-only',action='store_true')
    args=parser.parse_args()
    modes=('subtask-oracle','subtask-predicted') if args.metadata_only else ('text-only','task-known')
    jobs=[(f,p,m) for f in range(5) for p in ('P1','P2') for m in modes]
    if args.smoke: jobs=jobs[:1]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(worker,jobs): print(json.dumps(result),flush=True)
