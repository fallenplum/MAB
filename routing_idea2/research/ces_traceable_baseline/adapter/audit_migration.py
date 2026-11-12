"""Independent CES migration replay, state, permission and actual-cost audit.

This audit reads only train600 assets. It writes counterfactual poison replays
to the new experiment's private audit folder and never changes frozen actors.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import json
import hashlib
import importlib.util
import shutil
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/ces_traceable_baseline'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
sys.path.insert(0,str(PUBLIC))
import ces_common as common


def load_actor():
    path=PUBLIC/'run_migration.py'
    spec=importlib.util.spec_from_file_location('ces_independent_migration_actor',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def main():
    scalar_checks=0
    max_error=0.
    def equal(left,right,tol=1e-8,label='numeric'):
        nonlocal scalar_checks,max_error
        l=np.asarray(left);r=np.asarray(right)
        assert l.shape==r.shape,(label,l.shape,r.shape)
        scalar_checks+=l.size
        if l.dtype.kind in 'biufc' and r.dtype.kind in 'biufc':
            error=float(np.max(abs(l.astype(float)-r.astype(float)))) if l.size else 0.
            assert error<=tol,(label,error)
            max_error=max(max_error,error)
        else:
            assert np.array_equal(l,r),label
    manifest=json.loads((ROOT/'third_party/ces_source_audit/SOURCE_MANIFEST.json').read_text())
    source_entries={row['path']:row for row in manifest['files']}
    encoder=common.UPSTREAM/'src/embeddings/encoders.py'
    assert common.sha(encoder)==source_entries['src/embeddings/encoders.py']['sha256']
    protocol=json.loads((PUBLIC/'CES_PROTOCOL.json').read_text())
    assert protocol['manual_text_source']['encoder_class_sha256']==common.sha(encoder)
    data,refs=common.load_train()
    member,member_source=common.load_member_scores(data)
    with np.load(PRIVATE/'MANUAL_INPUTS.npz') as a:
        manual=a['features']
    Core,build_phi,initialize,Encoder=common.author_operators()
    assert Core.__module__.startswith('_ces_traceable_')
    prepared=json.loads((PRIVATE/'PREPARED_ALL.json').read_text())
    geometries=[]
    initial_states={}
    all_profiles={}
    all_phi={}
    for meta in prepared['models']:
        f,seed=meta['fold'],meta['seed']
        roles=common.splits(f)
        I=roles['I']
        path=PRIVATE/f'prepared/FEATURES_f{f}_s{seed}.npz'
        assert common.sha(path)==meta['feature_sha256']
        with np.load(path) as a:
            saved={k:a[k] for k in a.files}
        # Feature recomputation is given fully poisoned nonhistorical reward
        # and cost arrays. Geometry must depend only on inputs and I feedback.
        poisoned=dict(data)
        poisoned['Y']=data['Y'].copy()
        poisoned['Y'][np.setdiff1d(np.arange(600),I)]=np.nan
        poisoned['cost']={k:v.astype(float).copy() for k,v in data['cost'].items()}
        for array in poisoned['cost'].values():
            array[:,np.setdiff1d(np.arange(600),I)]=np.nan
        mp=member.copy()
        mp[np.setdiff1d(np.arange(600),I)]=np.nan
        phi,recomputed=common.make_features(poisoned,mp,I,seed,manual)
        equal(phi,saved['phi'],tol=0,label='I-only frozen features')
        for key in ['profile','W_x','W_m','x','v','historical_member_tokens']:
            equal(recomputed[key],saved[key],tol=0,label='feature parameter '+key)
        ordered=np.asarray(sorted(I,key=lambda q:str(data['ids'][q])),int)
        Z=np.asarray([phi[q,list(team)].sum(axis=0) for q in ordered for team in common.TEAMS])
        y=data['Y'][ordered].mean(axis=1).reshape(-1)
        V=.1*np.eye(96)+Z.T@Z
        b=Z.T@y
        dense_cov=np.linalg.inv(V)
        dense_mean=np.linalg.solve(V,b)
        warm_path=PRIVATE/f'prepared/WARM_f{f}_s{seed}.npz'
        assert common.sha(warm_path)==meta['warm_sha256']
        with np.load(warm_path) as a:
            equal(a['V_inv'],dense_cov,label='dense initial covariance')
            equal(a['b'],b,label='dense initial RHS')
            equal(a['w_hat'],dense_mean,label='dense initial mean')
            initial_states[(f,seed)]={k:a[k].copy() for k in ['V_inv','b','w_hat']}
        all_phi[(f,seed)]=phi
        all_profiles[(f,seed)]=saved['profile']
        geometries.append(dict(fold=f,seed=seed,I_only_geometry_exact=True,historical_combo_mean_updates=4800))
    paths=[]
    keys=['I','U','E','purchased','U_selected','E_actions','E_mean_actions','E_predicted_means','checkpoints']
    for receipt_path in sorted((PRIVATE/'replay').glob('f*_*/FROZEN_PATH.json')):
        folder=receipt_path.parent
        receipt=json.loads(receipt_path.read_text())
        f,seed,track=receipt['fold'],receipt['seed'],receipt['track']
        phi=all_phi[(f,seed)]
        roles=common.splits(f)
        E=roles['E']
        meta=json.loads((PRIVATE/f'prepared/PREPARED_f{f}_s{seed}.json').read_text())
        assert receipt['record_sha256']==common.sha(folder/'FROZEN_PATH.npz')
        assert receipt['protocol_sha256']==common.sha(PUBLIC/'CES_PROTOCOL.json')
        with np.load(folder/'FROZEN_PATH.npz') as a:
            raw={k:a[k] for k in a.files}
        equal(raw['purchased'],meta['question_schedule'],tol=0,label='common E7 random schedule')
        assert len(set(raw['purchased']))==60 and set(raw['purchased'])<=set(roles['U'])
        assert not set(raw['purchased'])&set(E)
        feedback=json.loads((folder/'FEEDBACK_ACCESS_AND_COST.json').read_text())
        prospective=json.loads((folder/'PROSPECTIVE_DECISIONS.json').read_text())
        agent=Core(96,alpha=3.5,lam=.1)
        if track!='ProfileOnly-SingleCombo':
            agent.load_state(**initial_states[(f,seed)])
        def snapshot(B):
            idx=[0,20,40,60].index(B)
            path=folder/f'MODEL_B{B}.npz'
            assert common.sha(path)==receipt['snapshot_sha256'][path.name]
            with np.load(path) as a:
                equal(a['V_inv'],agent.V_inv,tol=0,label='preserved actor covariance')
                equal(a['b'],agent.b,tol=0,label='preserved actor RHS')
                equal(a['w_hat'],agent.w_hat,tol=0,label='preserved actor mean')
                equal(a['alpha'],3.5,tol=0,label='main alpha')
                equal(a['ridge'],.1,tol=0,label='main ridge')
            main=[];mean=[];values=[]
            for q in E:
                current=tuple(sorted(map(int,agent.select_topk(list(phi[q]),3))))
                main.append(common.TEAMS.index(current))
                zero=Core(96,alpha=0.,lam=.1)
                zero.load_state(V_inv=agent.V_inv.copy(),b=agent.b.copy(),w_hat=agent.w_hat.copy())
                members=tuple(sorted(map(int,zero.select_topk(list(phi[q]),3))))
                mean.append(common.TEAMS.index(members))
                values.append([phi[q,list(team)].sum(axis=0)@agent.w_hat for team in common.TEAMS])
            equal(main,raw['E_actions'][idx],tol=0,label='original UCB selection')
            equal(mean,raw['E_mean_actions'][idx],tol=0,label='separate alpha0 diagnostic')
            equal(values,raw['E_predicted_means'][idx],tol=0,label='twenty means')
            assert agent.alpha==3.5
        snapshot(0)
        labels=0
        cost_totals={k:0. for k in feedback['cost'][0]}
        for tick,q in enumerate(raw['purchased']):
            q=int(q)
            members=tuple(sorted(map(int,agent.select_topk(list(phi[q]),3))))
            team=common.TEAMS.index(members)
            equal(raw['U_selected'][tick],team,tol=0,label='served-before-feedback combination')
            record=prospective[tick]
            assert record['step']==tick+1 and record['question']==q and record['served_combo']==team
            assert tuple(record['members'])==members
            expected_actions=tuple(range(20)) if track=='HistoryFit-Full20' else (team,)
            assert tuple(record['feedback_actions'])==expected_actions
            expected_reward=data['Y'][q][:,expected_actions]
            equal(feedback['labels'][tick]['raw'],expected_reward,tol=0,label='paid raw aggregate feedback')
            assert feedback['log'][tick]==dict(role='paid_U_feedback',question=q,actions=list(expected_actions),raw_reward_count=3*len(expected_actions),Gold=1)
            actual=common.resources(data,[q],[expected_actions],gold=1)
            for key,value in actual.items():
                equal(value,feedback['cost'][tick][key],tol=0,label='actual source cost')
                cost_totals[key]+=value
            assert actual['gold']==1 and actual['scoring_ops']==3*len(expected_actions)
            assert actual['member_calls']==(18 if track=='HistoryFit-Full20' else 9)
            assert actual['aggregator_calls']==(60 if track=='HistoryFit-Full20' else 3)
            assert actual['tokens']<=(602112 if track=='HistoryFit-Full20' else 79872)
            for combo,reward in zip(expected_actions,expected_reward.mean(axis=0)):
                agent.update(phi[q,list(common.TEAMS[combo])].sum(axis=0),float(reward))
                labels+=1
            if tick+1 in [20,40,60]:
                snapshot(tick+1)
        assert int(cost_totals['gold'])==60
        assert labels==(1200 if track=='HistoryFit-Full20' else 60)
        paths.append(dict(path=folder.name,observed_aggregate_means=labels,Gold=60,
            members=int(cost_totals['member_calls']),aggregators=int(cost_totals['aggregator_calls']),
            main_and_diagnostic_reconstructed=True,query_rewards_ever_passed_to_policy=False))
    assert len(paths)==45
    actor=load_actor()
    poison_results=[]
    for track in ['ProfileOnly-SingleCombo','HistoryFit-SingleCombo','HistoryFit-Full20']:
        f,seed=0,17
        name=f'f{f}_{track}_s{seed}'
        source=PRIVATE/'replay'/name
        feedback=json.loads((source/'FEEDBACK_ACCESS_AND_COST.json').read_text())
        with np.load(source/'FROZEN_PATH.npz') as a:
            original={key:a[key].copy() for key in keys}
        roles=common.splits(0)
        poisoned=dict(data)
        poisoned['Y']=np.full_like(data['Y'],np.nan)
        poisoned['Y'][roles['I']]=data['Y'][roles['I']]
        poisoned['cost']={k:np.full(v.shape,np.nan,dtype=float) for k,v in data['cost'].items()}
        for key in poisoned['cost']:
            poisoned['cost'][key][:,roles['I']]=data['cost'][key][:,roles['I']]
        for record in feedback['labels']:
            q=record['question']
            actions=record['actions']
            for action in actions:
                poisoned['Y'][q,:,action]=data['Y'][q,:,action]
            members=sorted(set().union(*(set(common.TEAMS[action]) for action in actions)))
            for key,value in poisoned['cost'].items():
                positions=members if key.startswith('member_') else actions
                for position in positions:
                    value[:,q,position]=data['cost'][key][:,q,position]
        target=PRIVATE/'audit_replays'/'CES_HIDDEN_REWARD_NAN'/track
        prep=target/'prepared'
        prep.mkdir(parents=True,exist_ok=True)
        for filename in ['FEATURES_f0_s17.npz','WARM_f0_s17.npz','PREPARED_f0_s17.json']:
            shutil.copyfile(PRIVATE/'prepared'/filename,prep/filename)
        actor.PRIVATE=target
        actor.load_train=lambda p=poisoned:(p,refs)
        actor.run_path((f,seed,track))
        generated=target/'replay'/name
        with np.load(generated/'FROZEN_PATH.npz') as a:
            for key in keys:
                equal(a[key],original[key],tol=0,label='hidden reward/cost poison '+key)
        cf=json.loads((generated/'FEEDBACK_ACCESS_AND_COST.json').read_text())
        assert cf==feedback
        poison_results.append(dict(track=track,fold=0,seed=17,steps=60,
            unpurchased_U_E_rewards_and_costs_NaN=True,
            nonexecuted_combinations_on_paid_single_questions_NaN=track!='HistoryFit-Full20',
            choices_predictions_mean_diagnostics_exact=True,paid_feedback_costs_exact=True))
    result=dict(status='PASS',claim_scope='Independent frozen CES shared-protocol migration operator/permissions/cost audit',
        scalar_checks=scalar_checks,maximum_error=max_error,
        feature_geometries=geometries,formal_paths=paths,snapshots_checked=180,
        hidden_reward_and_future_cost_poison=poison_results,
        prepared_feature_no_U_E_reward_or_cost_dependence=True,
        FeatEncoder_source_matches_upstream_manifest=True,
        original_source_hashes={k:common.sha(common.UPSTREAM/k) for k in common.SOURCE_PINS},
        extra_encoder_source_sha256=common.sha(encoder),new_physical_LLM_calls=0,
        interpretation='This establishes bounded adapter fidelity and permission isolation, not native benchmark performance or scientific superiority.')
    output=PUBLIC/'adapter/CES_MIGRATION_OPERATOR_ACCESS_AUDIT.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],scalar_checks=scalar_checks,max_error=max_error,
        formal_paths=len(paths),snapshots=180,poison_paths=len(poison_results))))


if __name__=='__main__':
    main()
