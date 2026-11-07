"""Observation intervals, contrast calibration and separately charged service cost."""
from e4_common import *
from e4_gaussian import fit_noise,_e3_noise
from e4_metrics import prediction_metrics,scope_means,calibration_table
from scipy.stats import spearmanr
from collections import defaultdict


def correlation(x,y):
    x,y=np.asarray(x),np.asarray(y)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])==0 or np.std(y[ok])==0:return None
    return float(spearmanr(x[ok],y[ok]).statistic)


def direct_cost(data,rows,actions,g):
    value=dict(member_calls=0,aggregator_calls=0,input_tokens=0,output_tokens=0,tokens=0,source_model_wall_s=0.)
    for q,a in zip(rows,actions):
        for stem,selected,call in [('member',list(TEAMS[a]),'member_calls'),('aggregate',[a],'aggregator_calls')]:
            for suffix,name in [('execution_counts',call),('prompt_tokens','input_tokens'),('completion_tokens','output_tokens'),('wall_time_s','source_model_wall_s')]:
                x=data['cost'][stem+'_'+suffix][g,q,selected].sum()
                value[name]+=float(x) if suffix=='wall_time_s' else int(x)
    value['tokens']=value['input_tokens']+value['output_tokens']
    return value


def main():
    data,_=load_train()
    splits=e4_splits(data)
    complete_cache={k:0 for k in ('member_calls','aggregator_calls','input_tokens','output_tokens','tokens','gold','scoring_ops')}
    complete_cache['source_model_wall_s']=0.
    for stem,call in [('member','member_calls'),('aggregate','aggregator_calls')]:
        for suffix,name in [('execution_counts',call),('prompt_tokens','input_tokens'),('completion_tokens','output_tokens'),('wall_time_s','source_model_wall_s')]:
            value=data['cost'][stem+'_'+suffix].sum()
            complete_cache[name]+=float(value) if suffix=='wall_time_s' else int(value)
    complete_cache['tokens']=complete_cache['input_tokens']+complete_cache['output_tokens']
    complete_cache['gold']=600
    complete_cache['scoring_ops']=36000
    noise={}
    histories={}
    for fold in splits['folds']:
        h=fold['history']
        # Preparation and post-feedback service are different resource accounts.
        preparation={k:0 for k in ('member_calls','aggregator_calls','input_tokens','output_tokens','tokens','gold','scoring_ops')}
        preparation['source_model_wall_s']=0.
        for g in range(3):
            for stem,indices,call in [('member',list(range(6)),'member_calls'),('aggregate',list(range(20)),'aggregator_calls')]:
                for suffix,name in [('execution_counts',call),('prompt_tokens','input_tokens'),('completion_tokens','output_tokens'),('wall_time_s','source_model_wall_s')]:
                    v=data['cost'][stem+'_'+suffix][g,h][:,indices].sum()
                    preparation[name]+=float(v) if suffix=='wall_time_s' else int(v)
        preparation['tokens']=preparation['input_tokens']+preparation['output_tokens']
        preparation['gold']=len(h)
        preparation['scoring_ops']=len(h)*3*20
        histories[str(fold['fold'])]=preparation
        for p in ('P1','P2'):
            fitted=fit_noise(data,h,p)
            noise[(fold['fold'],p)]={
                'N1':np.eye(20),'N2':_e3_noise(data['Y'][h]),'N3':_e3_noise(data['Y'][h]),
                'N4':np.diag(fitted.diagonal),'N5':fitted.heteroscedastic_correlated}
    totals={}
    full_cost=[]
    for path in sorted((PRIVATE/'posterior').glob('*.json')):
        meta=json.loads(path.read_text())
        a=np.load(path.with_suffix('.npz'))
        q=a['query']
        key=f"{meta['predictor']}_{meta['variant']}_{meta['permission']}"
        if key not in totals: totals[key]=dict(count=np.zeros(600),before=defaultdict(lambda:np.zeros(600)),after=defaultdict(lambda:np.zeros(600)),corr=[],cal_before=[],cal_after=[],gap=[],question_rows=[])
        b=totals[key]
        b['count'][q]+=1
        b['question_rows'].append(q)
        R=noise[(meta['fold'],meta['predictor'])][meta['variant']]
        initial=np.load(path.with_name(path.name.replace('_'+meta['variant']+'_','_N2_')).with_suffix('.npz'))['initial_pred']
        pairs=np.argsort(-initial,axis=1,kind='stable')[:,:2]
        for stage,stem in [('before','initial'),('after','final')]:
            pred=a[stem+'_pred']
            variance=a[stem+'_variance']
            metrics=prediction_metrics(pred,data['Y'][q],latent_variance=variance,observation_variance=np.diag(R))
            for name,v in metrics['per_question'].items(): b[stage][name][q]+=v
            b['cal_before' if stage=='before' else 'cal_after'].append((pred,data['Y'][q]))
            error=(pred-data['Y'][q].mean(1))**2
            b['corr'].append(dict(stage=stage,spearman_variance_vs_squared_mean_error=correlation(variance.ravel(),error.ravel())))
            gap=pred[np.arange(len(q)),pairs[:,0]]-pred[np.arange(len(q)),pairs[:,1]]
            true=data['Y'][q].mean(1)
            truegap=true[np.arange(len(q)),pairs[:,0]]-true[np.arange(len(q)),pairs[:,1]]
            pv=a['pair_variance'][0 if stage=='before' else -1]
            b['gap'].append(dict(stage=stage,spearman_fixed_pair_variance_vs_squared_gap_error=correlation(pv,(gap-truegap)**2),
                                  mean_pair_variance=float(pv.mean()),pair_gap_mse=float(np.mean((gap-truegap)**2))))
        full_cost.append(dict(fold=meta['fold'],predictor=meta['predictor'],variant=meta['variant'],permission=meta['permission'],
            seed=meta['generation_seed'],order=meta['orderseed'],support=meta['cost'],
            future_query100_single_generation_service=direct_cost(data,q,a['chosen'][-1],meta['generation_index']),
            future_query100_three_generation_repeated_evaluation=[direct_cost(data,q,a['chosen'][-1],g) for g in range(3)],
            query_gold_for_evaluation_only=100,historical_preparation_account=str(meta['fold']),new_physical_calls=0))
    result={}
    for key,b in totals.items():
        keep=b['count']>0
        task,sub=data['task'][keep],data['sub'][keep]
        entry={stage:{n:scope_means(v[keep]/b['count'][keep],task,sub) for n,v in b[stage].items()} for stage in ('before','after')}
        for stage,cal in [('before','cal_before'),('after','cal_after')]:
            probability=np.concatenate([np.clip(x,0,1) for x,y in b[cal]])
            table=calibration_table(probability,np.concatenate([y for x,y in b[cal]]))
            question_rows=np.concatenate(b['question_rows'])
            table['prediction_rows']=table['independent_sample_size']
            table['independent_sample_size']=int(keep.sum())
            membership=np.minimum((probability*10).astype(int),9)
            for index,binrow in enumerate(table['bins']):
                binrow['prediction_row_support']=binrow['original_question_support']
                binrow['original_question_support']=len(np.unique(question_rows[(membership==index).any(1)]))
            table['weighting']='Repeated seed/order prediction rows and combination cells, original questions counted uniquely. ECE descriptive.'
            entry[stage+'_calibration']=table
        entry['variance_error_correlations']=b['corr']
        entry['fixed_pair_gap_calibration']=b['gap']
        entry['independent_query_questions']=int(keep.sum())
        result[key]=entry
    put(PUBLIC/'CALIBRATION_AND_INTERVALS.json',dict(table=result,
        latent_probability_true_coverage=None,
        interval_note='Latent mean inclusion uses noisy three-generation mean and is not true probability coverage. Future Gaussian binary coverage is reported with width and contains-both rate.',
        correlations_note='Cell/trajectory correlations are descriptive. Original questions, not cells, are independent units.'))
    put(PUBLIC/'FULL_SERVICE_COST_ACCOUNTS.json',dict(existing_train600_complete_cache_once=complete_cache,history_preparation_by_fold=histories,controlled_paths=full_cost,
        cost_roles='Historical preparation, logging Support, future Query deployment, and audit Query repeats reported separately; shared cache is charged once per hypothetical execution path.',
        diagnostic_precomputation='MSE/rank/Oracle audit reuses all20 combinations on three seeds; selected-combination Query service account is not that full diagnostic precomputation. Complete-cache-once source account is provided separately.',
        source_wall_time_is_not_gpu_kernel_time=True,new_physical_model_calls=0))
    print('CALIBRATION_AND_FULL_COST_COMPLETE',flush=True)


if __name__=='__main__': main()
