"""Conditional paired statistics for E4-D. Original questions remain the unit."""
from e4_common import *
from e4_metrics import scope_means,paired_summary
from scipy.stats import spearmanr,kendalltau
from collections import defaultdict


def corr(x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    return float(spearmanr(x,y).statistic)


def live():
    data,_=load_train()
    rows={}
    paths=[]
    curves=[]
    for path in sorted((PRIVATE/'decision_live').glob('*.json')):
        item=json.loads(path.read_text())
        key=item['which']+'_'+item['method']
        if key not in rows:rows[key]=dict(count=np.zeros(600),reward=np.zeros(600),baseline=np.zeros(600),correct=np.zeros(600),harm=np.zeros(600),changes=np.zeros(600),byseed=defaultdict(list))
        b=rows[key]
        for record in item['traces']:
            q=record['q']
            b['count'][q]+=1
            b['reward'][q]+=record['reward']
            b['baseline'][q]+=record['no_feedback_reward']
            b['correct'][q]+=record['reward']>record['no_feedback_reward']
            b['harm'][q]+=record['reward']<record['no_feedback_reward']
            b['changes'][q]+=record['main']!=record['baseline_main']
        r=np.asarray([t['reward'] for t in item['traces']])
        no=np.asarray([t['no_feedback_reward'] for t in item['traces']])
        b['byseed'][str(item['generation_seed'])].append(float(np.mean(r-no)))
        c=item['cost']
        paths.append(dict(which=item['which'],method=item['method'],fold=item['fold'],seed=item['generation_seed'],
            cost=c,accuracy=float(r.mean()),gain=float(np.mean(r-no)),planning_cpu_s=sum(t['planning_cpu_s'] for t in item['traces']),
            update_cpu_s=sum(t['update_cpu_s'] for t in item['traces']),configuration=item['configuration']))
        curves.append(dict(which=item['which'],method=item['method'],fold=item['fold'],seed=item['generation_seed'],
            cumulative_quality=(np.cumsum(r)/np.arange(1,121)).tolist(),
            cumulative_gold=[t['cumulative']['gold'] for t in item['traces']],
            cumulative_tokens=[t['cumulative']['tokens'] for t in item['traces']],
            cumulative_priced_cost=[t['cumulative']['gold']+t['cumulative']['member_calls']+t['cumulative']['aggregator_calls']+t['cumulative']['tokens']/1000+t['cumulative']['scoring_ops']*.001 for t in item['traces']]))
    result={}
    for key,b in rows.items():
        assert np.all(b['count']==3)
        reward=b['reward']/b['count']
        baseline=b['baseline']/b['count']
        b['average_reward']=reward
        result[key]=dict(accuracy=scope_means(reward,data['task'],data['sub']),
            no_feedback_accuracy=scope_means(baseline,data['task'],data['sub']),
            gain=paired_summary(reward-baseline,data['task'],data['sub']),
            correct_expected=float(np.sum(b['correct']/b['count'])),harm_expected=float(np.sum(b['harm']/b['count'])),
            changed_questions_expected=float(np.sum(b['changes']/b['count'])),
            same_generation_gain={s:float(np.mean(v)) for s,v in b['byseed'].items()},
            per_task_gain={name:float(np.mean((reward-baseline)[semantic(data)==name])) for name in sorted(set(semantic(data)))})
    comparisons={}
    for which in ('E3','E4'):
        for method in ('Decision-Aware','Calibrated-DA'):
            for base in ('Random','Boundary','Parameter-IG','Decision-Aware'):
                if base==method:continue
                a,b=which+'_'+method,which+'_'+base
                comparisons[a+'_minus_'+base]=paired_summary(rows[a]['average_reward']-rows[b]['average_reward'],data['task'],data['sub'])
    comparisons['E3_Calibrated-DA_minus_Probability-DA']=paired_summary(rows['E3_Calibrated-DA']['average_reward']-rows['E3_Probability-DA']['average_reward'],data['task'],data['sub'])
    from aggregate_e4 import holm
    pvalues={k:v['balanced_v2']['p_value_descriptive_two_sided'] for k,v in comparisons.items()}
    put(PUBLIC/'DECISION_LIVE_RESULTS.json',dict(status='COMPLETE',table=result,comparisons=comparisons,
        holm_adjusted_development_p=holm(pvalues),paths=paths,new_llm_calls=0,
        selection_warning='E4 candidate chosen on these development folds. This is a mechanism diagnostic, not independent confirmation.',
        inference_warning='Conditional question-level paired intervals retain fixed fitted histories. No randomization of new questions or model-selection uncertainty is integrated.'))
    put(PUBLIC/'DECISION_QUALITY_COST_CURVES.json',dict(curves=curves,
        note='Same gold20 ceiling and actual spend reported. Token and priced totals differ with selected service combinations. Cost curves retain their actual axes.'))
    np.savez_compressed(PRIVATE/'DECISION_PAIRED.npz',**{key:b['average_reward'] for key,b in rows.items()})
    print(json.dumps({k:dict(acc=v['accuracy']['balanced_v2'],gain=v['gain']['balanced_v2']) for k,v in result.items()},indent=2))


def snapshots():
    pools=defaultdict(list)
    for path in sorted((PRIVATE/'decision_snapshots').glob('*.json')):
        item=json.loads(path.read_text())
        for row in item['candidates']:
            pools[item['which']].append((item,row))
    output={}
    fields=['ig','parameter_trace_reduction','latent_variance_reduction','probability_delta_variance_reduction',
            'latent_contrast_variance_reduction','probability_contrast_variance_reduction','decision_value_raw','bernoulli_value_raw']
    for which,items in pools.items():
        actual=np.asarray([r['actual_gain']['balanced_v2'] for _,r in items])
        stats={}
        for field in fields+['calibrated_bernoulli','decision_per_cost','bernoulli_per_cost','calibrated_per_cost']:
            if field=='calibrated_bernoulli': values=[r['calibrated']['bernoulli_value_raw'] for _,r in items]
            elif field=='calibrated_per_cost':values=[r['calibrated']['per_cost']['bernoulli_value_raw'] for _,r in items]
            elif field.endswith('_per_cost'):
                name='decision_value_raw' if field=='decision_per_cost' else 'bernoulli_value_raw'
                values=[r['proxy']['per_cost'][name] for _,r in items]
            else:values=[r['proxy'][field] for _,r in items]
            values=np.asarray(values)
            # Tie groups stay intact in quantile diagnostics. No arbitrary tie ranking is promoted.
            bins=[]
            edges=np.unique(np.quantile(values,[0,.2,.4,.6,.8,1]))
            for j,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                keep=(values>=low)&(values<=high if j==len(edges)-2 else values<high)
                bins.append(dict(low=float(low),high=float(high),n=int(keep.sum()),actual_mean=float(actual[keep].mean()) if keep.any() else None))
            order=np.argsort(-values,kind='stable')
            cutoff=max(1,len(values)//10)
            threshold=values[order[cutoff-1]]
            tied=values>=threshold
            stats[field]=dict(spearman=corr(values,actual),positive_actual_rate=float(np.mean(actual>0)),
                top_decile_including_ties=int(tied.sum()),top_decile_positive_rate=float(np.mean(actual[tied]>0)),
                top_decile_actual_mean=float(actual[tied].mean()),quantiles=bins,
                raw_negative_proxy_count=int(np.sum(values<0)),nonzero_actual_count=int(np.sum(actual!=0)))
        # Source-question block bootstrap, retaining all generation/action repeats.
        groups=defaultdict(list)
        for idx,(item,row) in enumerate(items):groups[(item['fold'],row['q'])].append(idx)
        rng=np.random.default_rng(20261107)
        keys=list(groups)
        primary=['ig','decision_value_raw','bernoulli_value_raw','calibrated_bernoulli']
        for field in primary:
            values=np.asarray([r['calibrated']['bernoulli_value_raw'] if field=='calibrated_bernoulli' else r['proxy'][field] for _,r in items])
            draws=[]
            for _ in range(2000):
                sample=np.concatenate([groups[keys[j]] for j in rng.integers(len(keys),size=len(keys))])
                value=corr(values[sample],actual[sample])
                if value is not None:draws.append(value)
            stats[field]['source_question_block_ci95']=np.quantile(draws,[.025,.975]).tolist() if draws else None
        output[which]=dict(candidates=len(items),source_question_blocks=len(groups),proxies=stats,
            positive_actual_count=int(np.sum(actual>0)),negative_actual_count=int(np.sum(actual<0)),zero_actual_count=int(np.sum(actual==0)),
            planner_cpu_s=float(sum(r['planning_cpu_s'] for _,r in items)),
            counterfactual_cost_note='Each candidate is independently charged by its own hypothetical Env. Audit counterfactuals are not actor feedback and do not modify its logged policy.',
            actual_gain_by_support_generation_seed={str(SEEDS[g]):float(np.mean([r['actual_gain']['balanced_v2'] for i,r in items if i['generation_index']==g])) for g in range(3)},
            query_reward_repeats='All three generation seeds averaged within each original Query question')
    put(PUBLIC/'DECISION_PROXY_CALIBRATION.json',dict(table=output,new_llm_calls=0,
        scope='Frozen logging snapshots, descriptive counterfactual candidate diagnosis, not independent feedback policy trial.',
        bootstrap_scope='Source question blocks preserve action and generation repeats. Future Query overlap and fitted-history/model-selection dependence remain conditional.'))
    print('DECISION_PROXY_CALIBRATION_COMPLETE')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['live','snapshots'])
    args=parser.parse_args()
    globals()[args.stage]()
