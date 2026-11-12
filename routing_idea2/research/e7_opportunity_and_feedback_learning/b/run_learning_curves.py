"""E7-B fixed-learning-curve and within-semantic-group permutation diagnosis."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import json
from time import perf_counter,process_time
import argparse
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from e7_common import ROOT,WORK,PUBLIC as ALL_PUBLIC,PRIVATE as ALL_PRIVATE,load_train,splits,sha,put,resources,paired_bootstrap,paired_interval
from e7_models import ModelSpec,build_spec,argmax,reward_means,qualification,batch_predict

PUBLIC=ALL_PUBLIC/'b'
PRIVATE=ALL_PRIVATE/'b'
KINDS=('P1','P2','P3')
SIZES=(240,300,360,480)
PERMUTATIONS=tuple(range(100,120))


def _assert_freeze():
    protocol=json.loads((ALL_PUBLIC/'E7_PROTOCOL.json').read_text('utf-8'))
    assert protocol['split_sha256']==sha(ALL_PRIVATE/'E7_SPLITS_FROZEN.npz')
    return protocol


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    return value


def legacy_check(data,I,spec):
    sys.path.insert(0,str(ROOT/'research/decision_acquisition_e3/public'))
    from predictors import ModelTemplate
    safe={key:data[key] for key in ['E','task','sub','ids']}
    safe['Y']=np.zeros_like(data['Y'])
    safe['Y'][I]=data['Y'][I]
    legacy=ModelTemplate(safe,I,'P2',I[:1]).clone(independent_noise=True)
    current=spec.posterior(I,data['Y'][I])
    X=legacy.template.X
    predicted=X@legacy.mean.reshape(17,20)
    index=np.arange(17)*20
    qcov=legacy.cov[np.ix_(index,index)]
    variance=np.einsum('ni,ij,nj->n',X,qcov,X)
    mean_error=float(np.max(abs(predicted-current.mean)))
    variance_error=float(np.max(abs(variance-np.diag(current.cov))))
    assert mean_error<1e-10 and variance_error<1e-10
    return dict(status='PASS',legacy_P2_mean_max_error=mean_error,legacy_P2_variance_max_error=variance_error,
                feature_max_error=float(np.max(abs(X-spec.parameters['design']))),
                qualification_scope='Original E3 ModelTemplate P2, alpha100 raw3, authorized I-only raw labels')


def prepare():
    _assert_freeze()
    data,refs=load_train()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    results=[]
    for f in range(5):
        role=splits(f)
        I=role['I']
        yI=data['Y'][I].copy()
        row=dict(fold=f,default=int(argmax(yI.mean(axis=(0,1)))),models={})
        for kind in KINDS:
            start=perf_counter()
            spec=build_spec(data['E'],I,yI,kind,tasks=data['task'],cv_seed=20261112+f)
            destination=PRIVATE/f'SPEC_f{f}_{kind}.npz'
            spec.save(destination)
            row['models'][kind]=dict(spec_sha256=sha(destination),metadata=spec.metadata,wall_s=perf_counter()-start)
            if kind=='P1':
                row['legacy_equivalence']=legacy_check(data,I,spec)
        row.update(protocol_sha256=sha(ALL_PUBLIC/'E7_PROTOCOL.json'),
            split_sha256=sha(ALL_PRIVATE/'E7_SPLITS_FROZEN.npz'),source_sha256=sha(__file__),
            initial_reward_rows=I.tolist(),E_rewards_used=False)
        put(PRIVATE/f'PREPARED_f{f}.json',_json(row))
        results.append(row)
        print(json.dumps(dict(stage='prepare',fold=f,default=row['default'],P3_lambda=row['models']['P3']['metadata']['lambda_value'])),flush=True)
    checks=qualification()
    checks['actual_initial_history_equivalence']=[row['legacy_equivalence'] for row in results]
    put(PUBLIC/'E7_B_MODEL_QUALIFICATION.json',checks)
    put(PRIVATE/'PREPARED_ALL.json',_json(results))


def semantic(data,rows):
    return np.where(data['task'][rows]=='gsm8k','gsm8k',data['sub'][rows])


def permute_block(data,rows,seed,fold,role):
    rows=np.asarray(rows,dtype=int)
    result=rows.copy()
    groups=semantic(data,rows)
    rng=np.random.default_rng(np.random.SeedSequence([20261112,seed,fold,role,len(rows)]))
    for name in sorted(set(groups)):
        positions=np.flatnonzero(groups==name)
        result[positions]=rng.permutation(rows[positions])
    return result


def error_arrays(predicted,raw,default):
    predicted=np.asarray(predicted)
    raw=np.asarray(raw)
    actual=raw.mean(axis=1)
    n=len(actual)
    chosen=argmax(predicted)
    service=raw[np.arange(n),:,chosen]
    default_actions=np.broadcast_to(np.asarray(default,dtype=int),(n,))
    base=raw[np.arange(n),:,default_actions]
    residual=predicted-actual
    mean_residual=residual.mean(axis=1)
    centered=residual-mean_residual[:,None]
    a,b=np.triu_indices(20,1)
    truth=actual[:,a]-actual[:,b]
    predicted_diff=predicted[:,a]-predicted[:,b]
    nonzero=abs(truth)>1e-12
    signs=np.sign(truth)*np.sign(np.where(abs(predicted_diff)<=1e-10,0,predicted_diff))
    pair_score=np.where(signs>0,1.,np.where(signs<0,0.,.5))
    pair_correct=np.sum(pair_score*nonzero,axis=1)
    pair_count=nonzero.sum(axis=1)
    values=dict(actions=chosen,service=service,base=base,
        centered_mse=np.mean(centered**2,axis=1),offset_mse=mean_residual**2,
        absolute_mse=np.mean(residual**2,axis=1),contrast_mae=np.mean(abs(truth-predicted_diff),axis=1),
        pair_correct=pair_correct,pair_count=pair_count,
        regret=np.max(actual,axis=1)-actual[np.arange(n),chosen],
        correction=np.mean((base==0)&(service==1),axis=1),harm=np.mean((base==1)&(service==0),axis=1))
    assert np.max(abs(values['absolute_mse']-values['centered_mse']-values['offset_mse']))<1e-12
    return values


def run_fold(f):
    _assert_freeze()
    data,refs=load_train()
    roles=splits(f)
    I,U,E=roles['I'],roles['U'],roles['E']
    yI=data['Y'][I].copy()
    default=int(argmax(yI.mean(axis=(0,1))))
    start,cpu=perf_counter(),process_time()
    main=np.zeros((3,4,len(E),20))
    train_metrics=[]
    permission=[]
    specs={kind:ModelSpec.load(PRIVATE/f'SPEC_f{f}_{kind}.npz') for kind in KINDS}
    initial_fit=[]
    for j,kind in enumerate(KINDS):
        for k,n in enumerate(SIZES):
            H=np.r_[I,U[:n-240]]
            labels=data['Y'][H].copy()
            permission.append(dict(model=kind,n=n,role='train',authorized_rows=H.tolist()))
            fitted=specs[kind].posterior(H,labels)
            main[j,k]=fitted.predict(E)
            values=error_arrays(fitted.predict(H),labels,default)
            train_metrics.append(dict(model=kind,n=n,train_absolute_mse=float(values['absolute_mse'].mean()),
                train_centered_mse=float(values['centered_mse'].mean()),train_offset_mse=float(values['offset_mse'].mean()),
                train_accuracy=float(values['service'].mean()),
                independent_questions=n,
                direction_support=int(np.any(labels-labels[:,:,default,None]!=0,axis=(1,2)).sum()),
                positive_direction_support=int(np.any(labels-labels[:,:,default,None]>0,axis=(1,2)).sum()),
                negative_direction_support=int(np.any(labels-labels[:,:,default,None]<0,axis=(1,2)).sum())))
            fitted.save_state(PRIVATE/f'MODEL_f{f}_{kind}_n{n}.npz')
    shuffled=np.zeros((20,4,len(E),20))
    maps_I=[]
    maps_U=[]
    shuffled_specs=[]
    for p,seed in enumerate(PERMUTATIONS):
        initial_source=permute_block(data,I,seed,f,1)
        initial_labels=data['Y'][initial_source].copy()
        assert set(initial_source)==set(I)
        spec=build_spec(data['E'],I,initial_labels,'P3',tasks=data['task'],cv_seed=20261112+f)
        shuffled_specs.append(dict(seed=seed,lambda_value=spec.noise,bandwidth=float(spec.parameters['bandwidth']),
                                  initial_reward_source=initial_source.tolist(),CV=spec.metadata['lambda_selection']))
        maps_I.append(initial_source)
        map_this=[]
        for k,n in enumerate(SIZES):
            addition=U[:n-240]
            addition_source=permute_block(data,addition,seed,f,2)
            assert set(addition_source)==set(addition)
            H=np.r_[I,addition]
            sources=np.r_[initial_source,addition_source]
            labels=data['Y'][sources].copy()
            shuffled[p,k]=batch_predict(spec,H,labels,E)
            map_this.append(addition_source.tolist())
            permission.append(dict(model='P3-Permutation',seed=seed,n=n,role='train',authorized_rows=H.tolist(),label_sources=sources.tolist()))
        maps_U.append(map_this)
    prediction_path=PRIVATE/f'FROZEN_B_f{f}.npz'
    np.savez_compressed(prediction_path,E=E,I=I,U=U,default=default,main_predictions=main,
                        permutation_predictions=shuffled,permutation_I_sources=np.asarray(maps_I))
    put(PRIVATE/f'FROZEN_B_f{f}.json',_json(dict(status='PREDICTIONS_FROZEN_BEFORE_E_REWARD_INDEX',fold=f,
        predictions_sha256=sha(prediction_path),source_sha256=sha(__file__),models_source_sha256=sha(Path(__file__).with_name('e7_models.py')),
        protocol_sha256=sha(ALL_PUBLIC/'E7_PROTOCOL.json'),default=default,training=train_metrics,
        legal_label_accesses=permission,permutation_U_sources=maps_U,permutation_specs=shuffled_specs,
        permutation_rule='I source permutation only inside I, U_n only inside currently authorized U_n; task/subtask groups preserve whole20x3 blocks',
        permutation_curve_scope='U source mapping may change with n; this is a diagnostic control, not a sequential adaptation stream',
        fit_wall_s=perf_counter()-start,fit_cpu_s=process_time()-cpu,E_rewards_used_to_fit=False)))
    # First E reward indexing for this actor occurs only after the predictions
    # and their configuration have been durably frozen.
    evaluated=data['Y'][E].copy()
    np.savez_compressed(PRIVATE/f'EVALUATED_B_f{f}.npz',E=E,Y=evaluated,task=data['task'][E],sub=data['sub'][E],
                        default=default,main_predictions=main,permutation_predictions=shuffled)
    print(json.dumps(dict(stage='curves',fold=f,status='COMPLETE',wall_s=perf_counter()-start,
        main_accuracy=[[float(error_arrays(main[j,k],evaluated,default)['service'].mean()) for k in range(4)] for j in range(3)])),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare()
    elif args.fold is None:
        for f in range(5):run_fold(f)
    else:
        run_fold(args.fold)


if __name__=='__main__':
    main()
