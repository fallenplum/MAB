"""Separate, equal-permission metadata diagnostics. Main text policy stays frozen."""
from e5_common import *
from gate_models import fit_gate_scores,freeze_thresholds,apply_thresholds,SCORE_NAMES
from run_gates import native
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter


def defaults(data,F,rows,mode):
    values=data['Y'][F].mean(1)
    global_action=argmax(values.mean(0))
    task_actions={t:argmax(values[data['task'][F]==t].mean(0)) for t in ('gsm8k','bbh')}
    output=np.asarray([task_actions[t] for t in data['task'][rows]],dtype=int)
    support={}
    if mode=='subtask-oracle':
        for s in sorted(set(data['sub'][F][data['task'][F]=='bbh'])):
            selected=(data['task'][F]=='bbh')&(data['sub'][F]==s)
            support[str(s)]=int(selected.sum())
            action=argmax(values[selected].mean(0))
            output[(data['task'][rows]=='bbh')&(data['sub'][rows]==s)]=action
    return output,dict(global_action=global_action,task_actions=task_actions,subtask_support=support,
                       fallback='historical subtask if present, otherwise task then global')


def worker(job):
    f,mode=job
    data,_=load_train()
    roles=split_roles(data)[f]
    F,G,E=roles['F'],roles['G'],roles['E']
    model,feature_info=fit_model(data,F,'P2',mode,'gaussian',100)
    qG,qE=predict(model,G),predict(model,E)
    bG,default_info=defaults(data,F,G,mode)
    bE,_=defaults(data,F,E,mode)
    cG,cE=argmax(qG,bG),argmax(qE,bE)
    DG=data['Y'][G[:,None],np.arange(3)[None,:],cG[:,None]]-data['Y'][G[:,None],np.arange(3)[None,:],bG[:,None]]
    score=fit_gate_scores(qG,bG,cG,DG,qE,bE,cE,strata_G=data['task'][G],seed=20261110+f)
    frozen=freeze_thresholds(score['scores_G_oof'],DG,data['task'][G])
    gates=np.asarray([apply_thresholds(score['scores_E'],frozen)[n] for n in SCORE_NAMES])
    main=np.asarray([frozen['main_indices'][n] for n in SCORE_NAMES])
    folder=PRIVATE/'metadata_gates'
    folder.mkdir(parents=True,exist_ok=True)
    put(folder/f'FROZEN_{mode}_f{f}.json',native(dict(fold=f,mode=mode,permission='same metadata visible to default and proposer',
        feature_info=feature_info,default_info=default_info,frozen={k:v for k,v in frozen.items() if k!='gates_G'},
        fitted_heads={k:v.descriptor() for k,v in score['model_heads'].items()},E_outcomes_used_for_freeze=False,
        source=identity(['run_metadata_controls.py','gate_models.py']))))
    Rb=data['Y'][E[:,None],np.arange(3)[None,:],bE[:,None]]
    Rc=data['Y'][E[:,None],np.arange(3)[None,:],cE[:,None]]
    np.savez_compressed(folder/f'{mode}_f{f}.npz',F=F,G=G,E=E,base_E=bE,candidate_E=cE,
        gate_masks_E=gates,main_indices=main,Rb_E=Rb,Rc_E=Rc,D_E=Rc-Rb)
    return dict(fold=f,mode=mode,status='COMPLETE')


if __name__=='__main__':
    jobs=[(f,mode) for f in range(5) for mode in ('task-known','subtask-oracle')]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for result in pool.map(worker,jobs):print(json.dumps(result),flush=True)
