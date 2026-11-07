"""Aggregate cached E4 fits without promoting repeated seeds/orders to new questions."""
from e4_common import *
from e4_metrics import prediction_metrics,scope_means,paired_summary
from collections import defaultdict
import argparse


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def holm(pvalues):
    names=list(pvalues)
    order=sorted(names,key=lambda k:pvalues[k])
    result={}
    previous=0.
    for i,name in enumerate(order):
        previous=max(previous,min(1.,pvalues[name]*(len(order)-i)))
        result[name]=previous
    return result


def learning():
    data,_=load_train()
    buffers={}
    settings={}
    for path in sorted((PRIVATE/'learning').glob('*.json')):
        meta=read(path)
        arrays=np.load(path.with_suffix('.npz'))
        q=arrays['query']
        for case,pred in zip(meta['cases'],arrays['predictions']):
            key=f"{meta['predictor']}_n{meta['n']}_L{case['L']}_{case['regularization']}"
            settings[key]=dict(predictor=meta['predictor'],n=meta['n'],L=case['L'],regularization=case['regularization'])
            metrics=prediction_metrics(pred,data['Y'][q])['per_question']
            if key not in buffers:
                buffers[key]=dict(count=np.zeros(600),metrics={k:np.zeros(600) for k in metrics},train=[],design=[])
            b=buffers[key]
            b['count'][q]+=1
            for name,val in metrics.items(): b['metrics'][name][q]+=val
            b['train'].append(case)
            b['design'].append({k:meta[k] for k in ('fold','design_rank','entropy_effective_rank','nonzero_design_condition')})
    table,raw={},{}
    for key,b in buffers.items():
        assert np.all(b['count']>0)
        raw[key]={name:val/b['count'] for name,val in b['metrics'].items()}
        table[key]=dict(config=settings[key],metrics={name:scope_means(val,data['task'],data['sub']) for name,val in raw[key].items()},
            train_mean_mse=float(np.mean([c['train_mean_mse'] for c in b['train']])),
            train_raw_mse=float(np.mean([c['train_raw_mse'] for c in b['train']])),
            effective_df=float(np.mean([c['effective_df'] for c in b['train']])),
            original_dimension=b['train'][0]['original_parameter_dimension'],
            observable_dimension=b['train'][0]['observable_parameter_dimension'],
            design=b['design'])
    primary={}
    for name,a,b in [('P2_n480_minus_n60_Brier','P2_n480_L3_fixed','P2_n60_L3_fixed'),
                     ('P2_n240_L3_minus_L1_Brier','P2_n240_L3_fixed','P2_n240_L1_fixed')]:
        primary[name]=paired_summary(raw[a]['brier']-raw[b]['brier'],data['task'],data['sub'])
    put(PUBLIC/'LEARNING_CURVE_RESULTS.json',dict(status='COMPLETE',configurations=table,primary=primary,
         complete_fit_groups=50,fit_cases=650,raw_primary_holm_family='A family completed with stability oracle comparison',new_llm_calls=0))
    np.savez_compressed(PRIVATE/'LEARNING_PAIRED.npz',**{key+'__'+m:v for key,mm in raw.items() for m,v in mm.items()})
    print(json.dumps({k:{'acc':v['metrics']['accuracy']['balanced_v2'],'brier':v['metrics']['brier']['balanced_v2']} for k,v in table.items() if '_L3_fixed' in k},indent=2))


def posterior():
    data,_=load_train()
    sums={}
    records=[]
    for path in sorted((PRIVATE/'posterior').glob('*.json')):
        meta=read(path)
        a=np.load(path.with_suffix('.npz'))
        q=a['query']
        key=f"{meta['predictor']}_{meta['variant']}_{meta['permission']}"
        if key not in sums:
            sums[key]=dict(count=np.zeros(600),before=defaultdict(lambda:np.zeros(600)),
                after=defaultdict(lambda:np.zeros(600)),curves=defaultdict(list),events=[],paths=[],
                gains_by_seed=defaultdict(list),correct=np.zeros(600),harm=np.zeros(600))
        b=sums[key]
        b['count'][q]+=1
        names=[n for n in ('raw_mse','mean_mse','brier','log_loss','pair_gap_mae','non_tie_rank_accuracy','accuracy','top1_regret','outside_rate') if n in a]
        for name in names:
            b['before'][name][q]+=a[name][0]
            b['after'][name][q]+=a[name][-1]
            b['curves'][name].append(np.nanmean(a[name],axis=1))
        chosen=a['chosen']
        Y=data['Y'][q]
        before_reward=Y[np.arange(len(q))[:,None],np.arange(3)[None,:],chosen[0,:,None]]
        after_reward=Y[np.arange(len(q))[:,None],np.arange(3)[None,:],chosen[-1,:,None]]
        b['correct'][q]+=np.mean((before_reward==0)&(after_reward==1),axis=1)
        b['harm'][q]+=np.mean((before_reward==1)&(after_reward==0),axis=1)
        g=meta['generation_index']
        b['gains_by_seed'][str(SEEDS[g])].append(float(np.mean(after_reward[:,g]-before_reward[:,g])))
        trace_decreased=np.diff(a['trace'])<0
        delta_regret=np.diff(a['top1_regret'].mean(axis=1))
        delta_mse=np.diff(a['mean_mse'].mean(axis=1))
        b['events'].append(dict(updates=20,cov_shrinks=int(trace_decreased.sum()),
            regret_worsens=int(np.sum(trace_decreased&(delta_regret>1e-12))),
            mse_worsens=int(np.sum(trace_decreased&(delta_mse>1e-12))),
            unchanged_route_events=int(np.sum(np.all(np.diff(chosen,axis=0)==0,axis=1))),
            trace_initial=float(a['trace'][0]),trace_final=float(a['trace'][-1]),
            fixed_contrast_var_initial=float(a['pair_variance'][0].mean()),fixed_contrast_var_final=float(a['pair_variance'][-1].mean())))
        b['paths'].append(dict(fold=meta['fold'],seed=meta['generation_seed'],order=meta['orderseed'],cost=meta['cost'],wall_s=meta['wall_s']))
        records.append(dict(predictor=meta['predictor'],variant=meta['variant'],permission=meta['permission'],
            fold=meta['fold'],seed=meta['generation_seed'],order=meta['orderseed'],cost=meta['cost'],wall_s=meta['wall_s']))
    table,raw={},{}
    for key,b in sums.items():
        valid=b['count']>0
        q=np.flatnonzero(valid)
        before={k:v[valid]/b['count'][valid] for k,v in b['before'].items()}
        after={k:v[valid]/b['count'][valid] for k,v in b['after'].items()}
        raw[key]=dict(q=q,before=before,after=after)
        gain=after['accuracy']-before['accuracy']
        assert np.max(abs(gain+after['top1_regret']-before['top1_regret']))<1e-12
        task,sub=data['task'][q],data['sub'][q]
        table[key]=dict(before={k:scope_means(v,task,sub) for k,v in before.items()},
            after={k:scope_means(v,task,sub) for k,v in after.items()},
            gain=paired_summary(gain,task,sub),question_support=len(q),path_count=len(b['events']),
            correct_expected=float(np.sum(b['correct'][valid]/b['count'][valid])),
            harm_expected=float(np.sum(b['harm'][valid]/b['count'][valid])),
            events={k:float(np.mean([e[k] for e in b['events']])) for k in b['events'][0]},
            quality_curves={k:np.mean(v,axis=0).tolist() for k,v in b['curves'].items()},
            same_generation_gain={s:float(np.mean(v)) for s,v in b['gains_by_seed'].items()},
            per_task_gain={name:float(np.mean(gain[semantic(data)[q]==name])) for name in sorted(set(semantic(data)[q]))})
    comparisons={}
    for p in ('P1','P2'):
        for perm in ('main','pair'):
            base=raw[f'{p}_N2_{perm}']
            for v in ('N1','N3','N4','N5'):
                alt=raw[f'{p}_{v}_{perm}']
                assert np.array_equal(base['q'],alt['q'])
                q=base['q']
                comparisons[f'{p}_{perm}_{v}_minus_N2']={metric:paired_summary(alt['after'][metric]-base['after'][metric],data['task'][q],data['sub'][q]) for metric in ('accuracy','top1_regret','brier','mean_mse')}
    primary_names=[f'P2_pair_{v}_minus_N2' for v in ('N3','N4','N5')]
    adjusted=holm({name:comparisons[name]['top1_regret']['balanced_v2']['p_value_descriptive_two_sided'] for name in primary_names})
    put(PUBLIC/'POSTERIOR_AUDIT_RESULTS.json',dict(status='COMPLETE',table=table,comparisons=comparisons,
        regret_holm_adjusted=adjusted,paths=len(records),new_llm_calls=0,
        event_warning='Each event evaluates overlapping future query questions. Event frequencies are descriptive, not independent tests.'))
    put(PUBLIC/'CONTROLLED_COST_LEDGER.json',dict(paths=records,new_physical_calls=0,source_wall_time_is_proxy=True))
    np.savez_compressed(PRIVATE/'POSTERIOR_PAIRED.npz',**{key+'__'+side+'__'+metric:val for key,r in raw.items() for side in ('before','after') for metric,val in r[side].items()},query=raw['P2_N2_pair']['q'])
    print(json.dumps({k:dict(acc=v['after']['accuracy']['balanced_v2'],brier=v['after']['brier']['balanced_v2'],gain=v['gain']['balanced_v2']) for k,v in table.items() if k.startswith('P2')},indent=2))


def binary():
    data,_=load_train()
    raw={}
    hyper=defaultdict(list)
    for path in sorted((PRIVATE/'binary').glob('*.json')):
        meta=read(path)
        arrays=np.load(path.with_suffix('.npz'))
        q=arrays['query']
        for index,detail in enumerate(meta['models']):
            key=f"{meta['predictor']}_{meta['mode']}_{detail['family']}"
            if key not in raw: raw[key]={}
            metrics=prediction_metrics(arrays['predictions'][index],data['Y'][q])
            for name,val in metrics['per_question'].items():
                if name not in raw[key]: raw[key][name]=np.full(600,np.nan)
                raw[key][name][q]=val
            hyper[key].append(dict(fold=meta['fold'],alpha=detail['alpha'],fit_info=detail['fit_info'],metadata=detail['metadata'],
                                   parameter_trace=detail['parameter_trace'],cluster_sandwich_trace=detail['cluster_sandwich_trace']))
    table={key:dict(metrics={m:scope_means(v,data['task'],data['sub']) for m,v in mm.items()},fits=hyper[key]) for key,mm in raw.items()}
    comparisons={}
    for key,mm in raw.items():
        p,mode,family=key.split('_',2)
        base=f'{p}_{mode}_gaussian'
        if base in raw and family!='gaussian':
            comparisons[key+'_minus_gaussian']={metric:paired_summary(mm[metric]-raw[base][metric],data['task'],data['sub']) for metric in ('accuracy','brier','mean_mse','pair_gap_mae')}
    eligible=[k for k in table if k.startswith('P2_') and ('_text-only_' in k or '_task-known_' in k)]
    # This is development selection. No held-out/fresh400 confirmation is implied.
    chosen=min(eligible,key=lambda k:(table[k]['metrics']['brier']['balanced_v2'],k))
    pvals={k:v['brier']['balanced_v2']['p_value_descriptive_two_sided'] for k,v in comparisons.items() if k.startswith('P2_text-only_') and ('logistic' in k or 'probit' in k)}
    put(PUBLIC/'BINARY_COMPARISON_RESULTS.json',dict(status='COMPLETE',table=table,comparisons=comparisons,
        chosen_D_candidate=chosen,selection='minimum development OOF Brier among deployable P2 candidates, routing performance not used',
        binary_primary_holm_adjusted=holm(pvals),new_llm_calls=0))
    np.savez_compressed(PRIVATE/'BINARY_PAIRED.npz',**{k+'__'+m:v for k,mm in raw.items() for m,v in mm.items()})
    print(json.dumps(dict(chosen=chosen,table={k:{'acc':v['metrics']['accuracy']['balanced_v2'],'brier':v['metrics']['brier']['balanced_v2']} for k,v in table.items()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['learning','posterior','binary'])
    args=parser.parse_args()
    globals()[args.stage]()
