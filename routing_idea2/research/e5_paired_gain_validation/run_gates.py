from e5_common import *
from gate_models import fit_gate_scores,freeze_thresholds,apply_thresholds,SCORE_NAMES
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import argparse


def native(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    return value


def worker(f):
    started=perf_counter()
    data,refs=load_train()
    with np.load(PRIVATE/'base'/f'BASE_f{f}.npz') as a:
        G,E=a['G'],a['E']
        predG,predE=a['pred_G'][1],a['pred_E'][1]
        base=int(a['base'])
        candidateG,candidateE=a['candidate_G'],a['candidate_E']
    Rc_G=data['Y'][G[:,None],np.arange(3)[None,:],candidateG[:,None]]
    assert Rc_G.shape==(len(G),3)
    assert all(np.array_equal(Rc_G[:,g],data['Y'][G,g,candidateG]) for g in range(3))
    Dg=Rc_G-data['Y'][G,:,base]
    scores=fit_gate_scores(predG,np.full(len(G),base,dtype=int),candidateG,Dg,
                          predE,np.full(len(E),base,dtype=int),candidateE,
                          strata_G=data['task'][G],seed=20261110+f)
    frozen=freeze_thresholds(scores['scores_G_oof'],Dg,data['task'][G])
    gates=apply_thresholds(scores['scores_E'],frozen)
    masks=np.asarray([gates[name] for name in SCORE_NAMES])
    main=np.asarray([frozen['main_indices'][name] for name in SCORE_NAMES])
    folder=PRIVATE/'gates'
    folder.mkdir(parents=True,exist_ok=True)
    # The actor's entire family and thresholds are durably frozen before evaluator outcomes.
    metadata=dict(fold=f,score_names=list(SCORE_NAMES),base=base,
        source=identity(['run_gates.py','gate_models.py','E5_EXPERIMENT_CONFIG.json']),
        base_receipt_sha256=sha(PRIVATE/'base'/f'BASE_f{f}.json'),
        frozen={k:v for k,v in frozen.items() if k!='gates_G'},
        fitted_heads={k:v.descriptor() for k,v in scores['model_heads'].items()},
        fit_metadata=scores['fit_metadata'],configuration=scores['configuration'],
        E_outcome_used_for_freeze=False,scope='conditional development evaluation')
    put(folder/f'FROZEN_GATE_f{f}.json',native(metadata))
    decision_path=folder/f'FROZEN_DECISIONS_f{f}.npz'
    np.savez_compressed(decision_path,E=E,gate_masks_E=masks,main_indices=main,base=base,candidate_E=candidateE)
    # Only now may the evaluator access the held-out scoring block.
    Rb=data['Y'][E,:,base]
    Rc=data['Y'][E[:,None],np.arange(3)[None,:],candidateE[:,None]]
    assert Rc.shape==(len(E),3)
    De=Rc-Rb
    np.savez_compressed(folder/f'GATES_f{f}.npz',G=G,E=E,gate_masks_G=np.asarray([frozen['gates_G'][n] for n in SCORE_NAMES]),
        gate_masks_E=masks,scores_G_oof=np.asarray([scores['scores_G_oof'][n] for n in SCORE_NAMES]),
        scores_E=np.asarray([scores['scores_E'][n] for n in SCORE_NAMES]),main_indices=main,
        r_G=scores['raw_G_oof']['r'],u_G=scores['raw_G_oof']['u'],r_E=scores['raw_E']['r'],u_E=scores['raw_E']['u'],
        D_G=Dg,D_E=De,Rb_E=Rb,Rc_E=Rc,base=base,candidate_G=candidateG,candidate_E=candidateE)
    put(folder/f'GATES_f{f}.json',dict(fold=f,status='COMPLETE',wall_s=perf_counter()-started,
        frozen_gate_sha256=sha(folder/f'FROZEN_GATE_f{f}.json'),decision_sha256=sha(decision_path),
        eval_after_freeze=True,new_llm_calls=0))
    return dict(fold=f,status='COMPLETE',wall_s=perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(worker,range(5)):print(json.dumps(result),flush=True)
