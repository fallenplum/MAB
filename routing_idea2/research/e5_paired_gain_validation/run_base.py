from e5_common import *
from theory_checks import centering_check
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from time import perf_counter
import argparse


def worker(f):
    data,refs=load_train()
    roles=split_roles(data)[f]
    F,G,E=roles['F'],roles['G'],roles['E']
    started=perf_counter()
    models=fit_models(data,F)
    pred_F=np.asarray([predict(m,F) for m in models])
    pred_G=np.asarray([predict(m,G) for m in models])
    pred_E=np.asarray([predict(m,E) for m in models])
    base=int(argmax(data['Y'][F].mean((0,1))))
    candidate_G=argmax(pred_G[1],base)
    candidate_E=argmax(pred_E[1],base)
    source=identity(['e5_common.py','run_base.py','theory_checks.py','E5_EXPERIMENT_CONFIG.json'])
    source['E4_Gaussian']=sha(E4/'e4_gaussian.py')
    source['E4_Probit']=sha(E4/'e4_binary.py')
    source['inputs']=refs['inputs']['sha256']
    source['outcomes']=refs['outcomes']['sha256']
    check=centering_check(models[1].template.X[F],data['Y'][F].mean(1),models[1].template.X[E],100,
        question_weights=3*np.ones(len(F)),eval_rewards=data['Y'][E],eval_weights=np.ones(len(E))/len(E),case_id=f'fold{f}_real360_120')
    updates=[]
    receipts=[]
    for g in range(3):
        # The same legal G-only score packet is supplied to every base predictor.
        values=[]
        for q in roles['feedback_G20']:
            pos=int(np.flatnonzero(G==q)[0])
            actions=[base,int(candidate_G[pos])]
            values.append(dict(q=int(q),actions=actions,raw_scores=data['Y'][q,g,actions].tolist()))
        path_pred=[]
        for model in models:
            actor=deepcopy(model)
            for event in values:actor.update(event['q'],event['actions'],event['raw_scores'])
            path_pred.append(predict(actor,E))
        updates.append(path_pred)
        receipts.append(dict(generation_seed=SEEDS[g],feedback=values))
    output=PRIVATE/'base'/f'BASE_f{f}.npz'
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,F=F,G=G,E=E,pred_F=pred_F,pred_G=pred_G,pred_E=pred_E,
        base=np.asarray(base),candidate_G=candidate_G,candidate_E=candidate_E,
        updated_pred_E=np.asarray(updates),X_F=models[1].template.X[F],X_E=models[1].template.X[E],
        feedback_G20=roles['feedback_G20'])
    put(output.with_suffix('.json'),dict(fold=f,models=['P1','P2','P3_Probit'],alpha=[100,100,10],
        roles={name:len(roles[name]) for name in ('F','G','E')},base=base,source=source,
        centering_check=check,feedback=receipts,wall_s=perf_counter()-started,new_llm_calls=0,
        Probit_fit=models[2].fit_info))
    return dict(fold=f,status='COMPLETE',wall_s=perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(worker,range(5)):print(json.dumps(result),flush=True)
