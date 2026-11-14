"""Independent E10 evaluator, gated by complete pre-outcome prediction seals."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.stats import norm

ASSETS=Path(r'E:\语言模型组合路由')
ROOT=ASSETS/'research/e10_input_prefill_matching_audit'
E9=ASSETS/'research/e9_consistent_cost_sensitive_learning'
sys.path.insert(0,str(E9/'public/code'))
from e9_common import ACTION_KEYS
from e8_assets import load_historical
sys.path.insert(0,str(Path(__file__).parent))
from e10_matching import decompose_policies


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat()


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda value:value.tolist() if isinstance(value,np.ndarray) else value.item())+'\n',encoding='utf-8')


def power_plan(difference,task):
    sigma2=.5*sum(float(np.var(difference[task==name],ddof=1)) for name in ('gsm8k','bbh'))
    sigma=np.sqrt(sigma2)
    result=dict(question_level_paired_variance=sigma2, independent_candidate_count=1,
        null='future population gain <= 0', alternative_for_planning=.01,
        confirmation_test='fixed-time one-sided normal planning approximation, alpha .05',
        alpha=.05,power_target=.8,development_variance_is_plug_in=True)
    if sigma==0:
        result.update(status='DEGENERATE_DEVELOPMENT_VARIANCE',required_n=None)
        return result
    zsum=norm.ppf(.95)+norm.ppf(.8)
    result.update(status='PLANNING_ONLY', required_n_for_true_one_pp_and_null_zero=int(np.ceil(sigma2*zsum*zsum/.01**2)),
        existing_sizes=[dict(n=n,power_true_one_pp=float(norm.sf(norm.ppf(.95)-.01*np.sqrt(n)/sigma)),
            minimum_detectable_gain80=float(sigma*zsum/np.sqrt(n))) for n in (200,400)],
        lower_bound_above_one_pp=[dict(true_gain=effect,null_margin=.01,
            required_n=int(np.ceil(sigma2*zsum*zsum/(effect-.01)**2))) for effect in (.02,.03)])
    return result


def main():
    started=time.perf_counter()
    protocol_path=ROOT/'public/E10_PROTOCOL.json'
    freeze=read(ROOT/'private/PROTOCOL_FREEZE_RECEIPT.json')
    assert sha(protocol_path)==freeze['protocol_sha256']
    assert sha(Path(__file__))==freeze['sources']['e10_evaluate.py']
    predictions_path=ROOT/'private/ALL_E10_PREDICTIONS_FROZEN.npz'
    receipt=read(ROOT/'private/ALL_PREDICTIONS_FREEZE_RECEIPT.json')
    assert receipt['protocol_sha256']==freeze['protocol_sha256']
    assert receipt['prediction_sha256']==sha(predictions_path)
    assert receipt['formal_head_fits'] in (20,25)
    assert receipt['outer_reward_reads']==0
    predictions=np.load(predictions_path,allow_pickle=False)
    names=receipt['names']
    assert all(f'R_full_perm{seed}' in names for seed in (20261114091,20261114092,20261114093))
    for item in receipt['heads']:
        folder=ROOT/f'private/models/fold{item["outer_fold"]}/{item["name"]}'
        assert sha(folder/'MODEL.npz')==item['model_sha256']
        assert sha(folder/'PREDICTIONS.npz')==item['predictions_sha256']
        assert item['evaluation_rewards_used'] is False
    old_path=E9/'private/ALL_E9_PREDICTIONS_FROZEN.npz'
    parent=read(ROOT/'private/E9_PARENT_SNAPSHOT.json')
    assert sha(old_path)==parent['private/ALL_E9_PREDICTIONS_FROZEN.npz']
    old=np.load(old_path,allow_pickle=False)
    assert np.array_equal(predictions['fold'],old['fold'])
    opened=now()
    data,refs,_=load_historical(ASSETS)
    assert np.array_equal(predictions['ids'],data['ids'])
    assert {name:int(sum(data['task']==name)) for name in ('gsm8k','bbh')}=={'gsm8k':300,'bbh':300}
    rewards=np.asarray(data['Y'],dtype=np.float64);task=data['task'];sub=data['sub'];fold=predictions['fold']
    assert rewards.shape==(600,3,20)
    actions={name:predictions['actions__'+name] for name in names}
    actions.update({name:old['actions__'+name] for name in ('A','GlobalFixed','NS')})
    outcomes={name:rewards[np.arange(600),:,action] for name,action in actions.items()}
    rng=np.random.default_rng(20261114101)
    bootstrap=np.concatenate([rng.choice(rows,size=(10000,len(rows))) for f in range(5)
        for t in ('gsm8k','bbh') for rows in [np.flatnonzero((fold==f)&(task==t))]],axis=1)

    def paired(left,right,primary=False):
        events=outcomes[left]-outcomes[right]
        value=events.mean(axis=1)
        point=float(value.mean())
        draws=value[bootstrap].mean(axis=1)
        result=dict(gain=point,ordinary_paired95=np.quantile(draws,[.025,.975]).tolist(),
            corrected_events=int((events>0).sum()),destroyed_events=int((events<0).sum()),net_events=int(events.sum()),
            corrected_questions=int((value>0).sum()),destroyed_questions=int((value<0).sum()),
            independent_questions=600,recorded_generation_events=1800,
            seed_gain=value_from_events(events,axis=0),
            task_gain={t:float(value[task==t].mean()) for t in ('gsm8k','bbh')},
            fold_gain={str(f):float(value[fold==f].mean()) for f in range(5)})
        if primary:
            result['centered_bootstrap_two_sided_p']=float((1+sum(abs(draws-point)>=abs(point)))/10001)
        return result

    metrics={}
    for name,event in outcomes.items():
        counts=np.bincount(actions[name],minlength=20)
        p=counts[counts>0]/600
        metrics[name]=dict(accuracy=float(event.mean()),
            task_accuracy={t:float(event[task==t].mean()) for t in ('gsm8k','bbh')},
            generation_accuracy=event.mean(axis=0),fold_accuracy={str(f):float(event[fold==f].mean()) for f in range(5)},
            bbh_subtask={str(s):dict(n=int(sum((task=='bbh')&(sub==s))),
                accuracy=float(event[(task=='bbh')&(sub==s)].mean())) for s in sorted(set(sub[task=='bbh']))},
            action_counts=counts,maximum_action_fraction=float(counts.max()/600),
            effective_combinations=float(np.exp(-sum(p*np.log(p)))))
    comparisons={f'R_full-{name}':paired('R_full',name,primary=name=='A') for name in ('A','GlobalFixed','NS')}
    if 'R_match' in names:
        comparisons.update({'R_match-A':paired('R_match','A'),'R_full-R_match':paired('R_full','R_match')})
    controls={str(seed):paired('R_full',f'R_full_perm{seed}') for seed in (20261114091,20261114092,20261114093)}
    control_mean_gain=float(np.mean([item['gain'] for item in controls.values()]))
    decompositions=decompose_policies(rewards,actions,task,sub,fold,
        generation_seeds=(42,137,2024),action_keys=ACTION_KEYS)
    fine_gain=decompositions['R_full']['overall']['metrics']['G_match_fine']
    fine_baseline=decompositions['A']['overall']['metrics']['G_match_fine']
    primary=comparisons['R_full-A']
    go=dict(point_at_least_one_pp=primary['gain']>=.01,
        ordinary_ci_lower_positive=primary['ordinary_paired95'][0]>0,
        fine_matching_positive=fine_gain>0,
        fine_matching_above_A=fine_gain-fine_baseline>0,
        above_mean_three_content_controls=control_mean_gain>0,
        pending_final_input_source_and_cost_audit=True)
    provisional=all(value for key,value in go.items() if key!='pending_final_input_source_and_cost_audit')
    result=dict(status='EXPLORATORY_REUSED_TRAIN600',scope='conditional on fitted heads and repeatedly reused development questions',
        protocol_sha256=sha(protocol_path),predictions_sha256=sha(predictions_path),metrics=metrics,
        comparisons=comparisons,only_primary_comparison='R_full-A',matching=decompositions,
        mean_three_control_gain=control_mean_gain,go_conditions=go,provisional_scientific_go=provisional,
        confirmation_power_planning=power_plan((outcomes['R_full']-outcomes['A']).mean(axis=1),task),
        source_refs=refs,evaluation_wall_seconds=time.perf_counter()-started)
    truncated=np.load(ROOT/'private/features/PREFILL_FEATURES_FROZEN.npz',allow_pickle=False)['truncated']
    result['coverage_subsets']={label:dict(n=int(sum(mask)),
        methods={name:float(event[mask].mean()) if mask.any() else None for name,event in outcomes.items()})
        for label,mask in [('truncated',truncated),('not_truncated',~truncated)]}
    put(ROOT/'public/E10_MAIN_RESULTS.json',result)
    put(ROOT/'public/E10_PRIMARY_STATISTICS.json',dict(comparison='R_full-A',**primary,
        bootstrap_samples=10000,bootstrap_seed=20261114101,strata='outerfold x task',
        interval_scope='ordinary95 conditional fitted-model development evidence',primary_family_size=1))
    put(ROOT/'public/E10_CONTENT_CONTROLS.json',dict(status='SENSITIVITY_ONLY',comparisons=controls,
        mean_control_gain=control_mean_gain,permutations=3,minimum_standard_mc_p=.25,no_formal_content_significance=True))
    np.savez_compressed(ROOT/'private/E10_PAIRED_OUTCOMES.npz',task=task,sub=sub,fold=fold,
        **{name:event for name,event in outcomes.items()})
    put(ROOT/'private/EVALUATION_ACCESS_RECEIPT.json',dict(first_outer_reward_access=opened,
        all_predictions_frozen_at=receipt['frozen_utc'],prediction_sha256=sha(predictions_path),
        source_sha256=sha(Path(__file__)),source_refs=refs,
        validation200_reward_access=False,fresh400_reward_access=False,sealed200_reward_access=False))
    print(json.dumps(dict(status='COMPLETE',primary=primary,provisional_go=provisional),default=lambda x:x.tolist()),flush=True)


def value_from_events(events,axis):
    return events.mean(axis=axis).tolist()


if __name__=='__main__':
    main()
