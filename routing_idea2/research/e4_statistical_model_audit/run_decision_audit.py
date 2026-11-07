"""E4-D small paired mechanism replay. Future rewards enter the evaluator only."""
from e4_common import *
from e4_binary import BinaryModel
from e4_gaussian import GaussianTemplate
from e4_decision import decision_proxies
from run_binary import fit_model,pred_and_variance
from e4_metrics import scope_means
sys.path.insert(0,str(ROOT/'src'))
from acquisition import BudgetPacing,fantasy_table
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.special import expit,logit,ndtr
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter,process_time
from copy import deepcopy
import argparse


def predict(model,rows):
    if isinstance(model,BinaryModel):
        return model.predict_many(np.asarray(rows))
    t=model.template
    return t.X[np.asarray(rows)]@model.mean.reshape(t.question_dimension,t.action_dimension)@t.B.T


def chosen_configuration():
    result=json.loads((PUBLIC/'BINARY_COMPARISON_RESULTS.json').read_text())
    return result['chosen_D_candidate'].split('_',2)


def setup(data,fold,which):
    history=fold['history']
    refs=fold['references'][480]
    if which=='E3':
        model=GaussianTemplate(data,history,'P2',refs,variant='N2').clone()
        predictor,family,mode,alpha='P2','gaussian','text-only',100.
    else:
        predictor,mode,family=chosen_configuration()
        selection=json.loads((PRIVATE/'binary'/f"f{fold['fold']}_{predictor}_{mode}.json").read_text())
        alpha=next(item['alpha'] for item in selection['models'] if item['family']==family)
        model,_=fit_model(data,history,predictor,mode,family,alpha)
    # A frozen history-OOF Platt map calibrates acquisition utilities only.
    start=perf_counter()
    oof=np.empty((len(history),20))
    strata=semantic(data)[history]
    for it,iv in StratifiedKFold(3,shuffle=True,random_state=20261107+fold['fold']).split(history,strata):
        if which=='E3':
            sub=GaussianTemplate(data,history[it],'P2',history[it][:54],variant='N2').clone()
        else:
            sub,_=fit_model(data,history[it],predictor,mode,family,alpha)
        oof[iv]=predict(sub,history[iv])
    scores=np.repeat(logit(np.clip(oof,1e-6,1-1e-6))[:,None,:],3,axis=1).reshape(-1,1)
    labels=data['Y'][history].reshape(-1)
    calibration=LogisticRegression(C=1.,max_iter=1000).fit(scores,labels)
    slope=float(calibration.coef_[0,0])
    intercept=float(calibration.intercept_[0])
    if slope<=0:
        slope,intercept=1.,0.
    info=dict(predictor=which,mode=mode,family=family,alpha=alpha,
              platt_slope=slope,platt_intercept=intercept,calibration_wall_s=perf_counter()-start,
              calibration_scope='history question-OOF predictions only, actor and update unchanged')
    return model,info


def transform_from(info):
    return lambda p:expit(info['platt_slope']*logit(np.clip(p,1e-6,1-1e-6))+info['platt_intercept'])


def base_latent(model,q):
    if isinstance(model,BinaryModel):
        latent=model.latent([q])[0]
    else:
        latent=model.predict(q)
    H=model.observation(q,np.arange(20))
    C=H@model.cov@H.T
    return latent,C


def boundary(model,q,main):
    latent,C=base_latent(model,q)
    alternatives=np.argsort(-latent,kind='stable')[1:]
    variance=np.maximum(C[main,main]+np.diag(C)[alternatives]-2*C[main,alternatives],1e-15)
    return float(np.max(ndtr(-(latent[main]-latent[alternatives])/np.sqrt(variance))))


def value_key(proxy,name):
    # Exact names are part of e4_decision's tested public API.
    return float(proxy[name])


def live(job):
    f,g,which=job
    data,_=load_train()
    fold=e4_splits(data)['folds'][f]
    order=fold['orders'][17]
    base,info=setup(data,fold,which)
    refs=fold['references'][480]
    weights=np.ones(len(refs))/len(refs)
    fantasies=fantasy_table(algorithm_seed(f,17,0),pairs=8)
    baseline=np.argmax(predict(base,order),axis=1)
    transform=transform_from(info)
    methods=['Random','Boundary','Parameter-IG','Decision-Aware','Calibrated-DA']
    if not isinstance(base,BinaryModel): methods.append('Probability-DA')
    for method in methods:
        path=PRIVATE/'decision_live'/f'f{f}_g{g}_{which}_{method}'
        if path.with_suffix('.json').exists(): continue
        start=perf_counter()
        actor=deepcopy(base)
        env=Env(data,fold['history'],order,g,20)
        pace=BudgetPacing(len(order),20,algorithm_seed(f,17,0),warmup=2)
        records=[]
        feedback=[]
        for tick,qvalue in enumerate(order):
            q=int(qvalue)
            pred=predict(actor,[q])[0]
            main=int(np.argmax(pred))
            env.service(q,main)
            plan=process_time()
            score=0.
            if pace.eligible(tick):
                if method=='Boundary':
                    score=boundary(actor,q,main)
                elif method!='Random':
                    proxy=decision_proxies(actor,q,[main],refs,weights,fantasies,incremental_cost=1.001,
                                            probability_transform=transform if method=='Calibrated-DA' else None)
                    if method=='Parameter-IG': score=value_key(proxy,'ig')
                    elif method in ('Calibrated-DA','Probability-DA'): score=value_key(proxy['per_cost'],'bernoulli_value_nonnegative')
                    elif isinstance(actor,BinaryModel): score=value_key(proxy['per_cost'],'bernoulli_value_nonnegative')
                    else: score=value_key(proxy['per_cost'],'decision_value_nonnegative')
                purchase=pace.decide(tick,score,random=method=='Random')
            else: purchase=False
            planning=process_time()-plan
            update=0.
            if purchase:
                receipt=env.buy(q,[main])
                assert receipt['accepted']
                observed=receipt['observations']
                then=process_time()
                actor.update(q,[a for a,y in observed],[y for a,y in observed])
                update=process_time()-then
                feedback.append(dict(tick=tick,q=q,actions=[a for a,y in observed],raw_scores=[y for a,y in observed],resources=receipt['resources']))
            # Scoring occurs strictly after the current service decision is frozen.
            records.append(dict(tick=tick,q=q,main=main,baseline_main=int(baseline[tick]),
                 reward=float(data['Y'][q,g,main]),no_feedback_reward=float(data['Y'][q,g,baseline[tick]]),
                 purchase=purchase,score=score,planning_cpu_s=planning,update_cpu_s=update,
                 cumulative=deepcopy(env.ledger['total'])))
        assert env.summary()['gold_revealed']==20
        put(path.with_suffix('.json'),dict(fold=f,generation_index=g,generation_seed=SEEDS[g],order=17,
            which=which,method=method,configuration=info,traces=records,feedback=feedback,cost=env.summary(),
            wall_s=perf_counter()-start,role='prequential120, 20gold, main-only legal feedback'))
    return dict(fold=f,g=g,predictor=which,complete=True)


def snapshots(job):
    f,g,which=job
    data,_=load_train()
    fold=e4_splits(data)['folds'][f]
    actor,info=setup(data,fold,which)
    transform=transform_from(info)
    refs=fold['references'][480]
    weights=np.ones(len(refs))/len(refs)
    fantasies=fantasy_table(algorithm_seed(f,17,0),pairs=8)
    logging=json.loads((PRIVATE/'posterior'/f'f{f}_P2_N2_g{g}_o17_pair.json').read_text())
    query=fold['audit_query']
    outputs=[]
    for step,event in enumerate(logging['feedback']):
        if step in (0,5,10,15):
            q=event['q']
            oldpred=predict(actor,query)
            before=np.argmax(oldpred,axis=1)
            main=int(np.argmax(predict(actor,[q])[0]))
            neighbors=[a for a in range(20) if len(set(TEAMS[a])^set(TEAMS[main]))==2]
            for actions in ([main],*[[main,a] for a in neighbors]):
                hypothetical=Env(data,fold['history'],np.asarray([q]),g,1)
                executed={'members':set(TEAMS[main]),'aggregators':{main}}
                expected=hypothetical.predict_incremental(actions,executed)
                # Prospective scores are computed before any counterfactual evaluator reward access.
                start=process_time()
                proxy=decision_proxies(actor,q,actions,refs,weights,fantasies,incremental_cost=expected['priced_units'])
                calibrated=decision_proxies(actor,q,actions,refs,weights,fantasies,incremental_cost=expected['priced_units'],probability_transform=transform)
                planning=process_time()-start
                counterfactual=deepcopy(actor)
                hypothetical.service(q,main)
                receipt=hypothetical.buy(q,actions)
                assert receipt['accepted']
                obs=receipt['observations']
                counterfactual.update(q,[a for a,y in obs],[y for a,y in obs])
                after=np.argmax(predict(counterfactual,query),axis=1)
                Y=data['Y'][query]
                b=Y[np.arange(len(query))[:,None],np.arange(3)[None,:],before[:,None]]
                a=Y[np.arange(len(query))[:,None],np.arange(3)[None,:],after[:,None]]
                delta=(a-b).mean(1)
                outputs.append(dict(step=step,q=q,actions=actions,proxy=proxy,calibrated=calibrated,
                    actual_gain=scope_means(delta,data['task'][query],data['sub'][query]),
                    correct=float(np.sum((b==0)&(a==1))/3),harm=float(np.sum((b==1)&(a==0))/3),
                    changed_questions=int(np.sum(before!=after)),planning_cpu_s=planning,
                    estimated_resources=expected,counterfactual_cost=hypothetical.summary(),
                    support_generation=g,actual_evaluation_role='auditor-only frozen disjoint query, no actor feedback'))
        actor.update(event['q'],event['actions'],event['raw_scores'])
    put(PRIVATE/'decision_snapshots'/f'f{f}_g{g}_{which}.json',dict(fold=f,generation_index=g,which=which,
        configuration=info,candidates=outputs,new_llm_calls=0))
    return dict(fold=f,g=g,predictor=which,snapshots_complete=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--snapshots',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    jobs=[(f,g,w) for f in range(5) for g in range(3) for w in ('E3','E4')]
    if args.smoke: jobs=jobs[:1]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(snapshots if args.snapshots else live,jobs): print(json.dumps(result),flush=True)
