"""Identical legal logged feedback, multiple Gaussian working posteriors."""
from e4_common import *
from e4_gaussian import GaussianTemplate, fit_noise
from e4_metrics import prediction_metrics
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import argparse


def predict_many(model, rows):
    t=model.template
    W=model.mean.reshape(t.question_dimension,t.action_dimension)
    return t.X[rows] @ W @ t.B.T


def design(template, rows):
    return np.einsum('np,ar->napr',template.X[rows],template.B).reshape(len(rows),20,template.dimension)


def evaluate(model, query, Y, frozen_pair):
    pred=predict_many(model,query)
    metrics=prediction_metrics(pred,Y)
    F=design(model.template,query)
    FC=F.reshape(-1,model.template.dimension)@model.cov
    var=np.einsum('na,na->n',FC,F.reshape(-1,model.template.dimension)).reshape(len(query),20)
    contrast=F[np.arange(len(query)),frozen_pair[:,0]]-F[np.arange(len(query)),frozen_pair[:,1]]
    pairvar=np.einsum('nd,nd->n',contrast@model.cov,contrast)
    return pred,var,pairvar,metrics['per_question']


def worker(job):
    f,predictor=job
    data,_=load_train()
    fold=e4_splits(data)['folds'][f]
    history=fold['history']
    query=fold['audit_query']
    refs=fold['references'][480]
    logger=GaussianTemplate(data,history,'P2',refs,variant='N2')
    logger_initial=logger.clone()
    logpred=predict_many(logger_initial,fold['audit_support'])
    actions={}
    for q,pred in zip(fold['audit_support'],logpred):
        main=int(np.argmax(pred))
        neighbors=[a for a in range(20) if len(set(TEAMS[a])^set(TEAMS[main]))==2]
        neighbor=max(neighbors,key=lambda a:(pred[a],-a))
        actions[int(q)]=(main,int(neighbor))
    noise=fit_noise(data,history,predictor,alpha=100,training_seeds=(0,1,2))
    put(PRIVATE/'noise'/f'f{f}_{predictor}.json',noise.summary())
    baseline=GaussianTemplate(data,history,predictor,refs,variant='N2').clone()
    basepred=predict_many(baseline,query)
    frozen_pair=np.argsort(-basepred,axis=1,kind='stable')[:,:2]
    for variant in ('N1','N2','N3','N4','N5'):
        template=GaussianTemplate(data,history,predictor,refs,variant=variant,noise_fit=noise)
        for g in range(3):
            for orderseed,order in fold['audit_orders'].items():
                for permission in ('main','pair'):
                    path=PRIVATE/'posterior'/f'f{f}_{predictor}_{variant}_g{g}_o{orderseed}_{permission}'
                    if path.with_suffix('.json').exists():
                        continue
                    start=perf_counter()
                    model=template.clone()
                    env=Env(data,history,order,g,20)
                    allmetrics, meanchanges, traces, maxeigs, pairvars, values, selected=[],[],[],[],[],[],[]
                    pred,var,pairvar,metrics=evaluate(model,query,data['Y'][query],frozen_pair)
                    initial_pred,initial_var=pred.copy(),var.copy()
                    baseline_choices=np.argmax(pred,axis=1)
                    for step in range(21):
                        allmetrics.append(metrics)
                        selected.append(np.argmax(pred,axis=1))
                        traces.append(float(np.trace(model.cov)))
                        maxeigs.append(float(np.linalg.eigvalsh(model.cov)[-1]))
                        pairvars.append(pairvar)
                        if step==20:
                            break
                        q=int(order[step])
                        main,neighbor=actions[q]
                        offered=[main] if permission=='main' else [main,neighbor]
                        env.service(q,main)
                        answer=env.buy(q,offered)
                        assert answer['accepted'] and len(answer['observations'])==len(offered)
                        obs=answer['observations']
                        before=model.mean.copy()
                        update_start=perf_counter()
                        model.update(q,[x[0] for x in obs],[x[1] for x in obs])
                        values.append(dict(q=q,actions=[x[0] for x in obs],raw_scores=[x[1] for x in obs],
                                           update_wall_s=perf_counter()-update_start,resources=answer['resources']))
                        meanchanges.append(float(np.linalg.norm(model.mean-before)))
                        pred,var,pairvar,metrics=evaluate(model,query,data['Y'][query],frozen_pair)
                    metric_arrays={key:np.asarray([m[key] for m in allmetrics]) for key in allmetrics[0]
                                   if allmetrics[0][key] is not None}
                    assert np.max(np.diff(traces))<1e-9
                    assert np.max(np.diff(np.asarray(pairvars),axis=0))<1e-9
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path.with_suffix('.npz'),query=query,initial_pred=initial_pred,
                                        final_pred=pred,initial_variance=initial_var,final_variance=var,
                                        chosen=np.asarray(selected),pair_variance=np.asarray(pairvars),
                                        trace=np.asarray(traces),maxeig=np.asarray(maxeigs),
                                        mean_change=np.asarray(meanchanges),**metric_arrays)
                    put(path.with_suffix('.json'),dict(fold=f,predictor=predictor,variant=variant,
                        generation_index=g,generation_seed=SEEDS[g],orderseed=orderseed,permission=permission,
                        feedback=values,cost=env.summary(),wall_s=perf_counter()-start,
                        negative_transfer_event_definition='query mean loss/regret increases between consecutive update snapshots',
                        role='fixed logging policy support, disjoint future query evaluator'))
    print(json.dumps(dict(fold=f,predictor=predictor,complete=True)),flush=True)
    return f,predictor


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    jobs=[(f,p) for f in range(5) for p in ('P1','P2')]
    if args.smoke: jobs=jobs[:1]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker,jobs))
    put(PRIVATE/'POSTERIOR_DONE.json',dict(groups=len(jobs),smoke=args.smoke,new_llm_calls=0))
