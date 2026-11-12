"""Independent checks of the pinned original CES set-selection/update core.

These are deterministic software and model-class qualifications. They neither
reproduce original benchmark service quality nor inspect held-out MAB rewards.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys
import importlib.util
import ast
import json
import hashlib
import itertools
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/ces_traceable_baseline'
UPSTREAM=ROOT/'third_party/ces_source_audit/upstream'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def extracted_initializer():
    p=UPSTREAM/'scripts/run_objective_mcq.py'
    tree=ast.parse(p.read_text('utf-8'))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
               and node.name in {'init_mats_pca_warmstart','init_mats_random'}]
    assert len(functions)==2
    namespace={'np':np,'List':list}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(p),'exec'),namespace)
    return namespace


def independent_select(phis,V,b,k,alpha,costs,penalty):
    inverse=np.linalg.inv(V)
    w=inverse@b
    chosen=[]
    z=np.zeros(len(w))
    old_std=0.
    for _ in range(k):
        best=None
        for i in range(len(phis)):
            if i in chosen:
                continue
            candidate=z+phis[i]
            new_std=float(np.sqrt(max(float(candidate@inverse@candidate),1e-12)))
            value=float(phis[i]@w)-penalty*float(costs[i])+alpha*(new_std-old_std)
            if best is None or value>best[0]:
                best=(value,i,new_std)
        chosen.append(best[1])
        z+=phis[best[1]]
        old_std=best[2]
    return np.asarray(chosen)


def main():
    manifest=json.loads((ROOT/'third_party/ces_source_audit/SOURCE_MANIFEST.json').read_text())
    entries={row['path']:row for row in manifest['files']}
    for name in ['src/bandits/linucb_combinatorial.py','src/embeddings/projectors.py',
                 'scripts/run_objective_mcq.py','LICENSE']:
        assert sha(UPSTREAM/name)==entries[name]['sha256']
    core=module(UPSTREAM/'src/bandits/linucb_combinatorial.py','ces_adapter_independent_original_core')
    projectors=module(UPSTREAM/'src/embeddings/projectors.py','ces_adapter_independent_original_projectors')
    init=extracted_initializer()
    rng=np.random.default_rng(20261112)
    max_mean=0.
    max_cov=0.
    maxfeature=0.
    updates=0
    selected=0
    per_setting=[]
    for n,k,dimension in [(15,2,32),(6,3,32),(6,3,16)]:
        for penalty in [0.,.2]:
            policy=core.LinUCBSharedCombinatorial(3*dimension,alpha=3.5,lam=.1)
            V=.1*np.eye(3*dimension)
            b=np.zeros(3*dimension)
            meta=rng.normal(size=(n,dimension))
            costs=rng.uniform(.1,3.,n)
            for tick in range(18):
                x=rng.normal(size=dimension)
                phis=np.asarray([projectors.build_phi(x,v) for v in meta])
                def l2(a):
                    return a/(np.linalg.norm(a)+1e-12)
                reference=np.asarray([np.r_[l2(x),3*l2(v),3*l2(x*v)] for v in meta])
                maxfeature=max(maxfeature,float(np.max(abs(reference-phis))))
                upstream=policy.select_topk(list(phis),k,costs=costs,lam_cost=penalty)
                expected=independent_select(phis,V,b,k,3.5,costs,penalty)
                assert np.array_equal(upstream,expected)
                selected+=1
                z=phis[upstream].sum(axis=0)
                quality=float(rng.integers(0,2))
                reward=quality-penalty*float(costs[upstream].sum())
                policy.update(z,reward)
                V+=np.outer(z,z)
                b+=reward*z
                inverse=np.linalg.inv(V)
                max_cov=max(max_cov,float(np.max(abs(inverse-policy.V_inv))))
                max_mean=max(max_mean,float(np.max(abs(inverse@b-policy.w_hat))))
                updates+=1
            per_setting.append(dict(members=n,K=k,d_x=dimension,cost_penalty=penalty,updates=18))
    assert max_mean<1e-8 and max_cov<1e-8 and maxfeature<1e-12
    rank=[]
    incidence=np.asarray([[float(i in team) for i in range(6)] for team in itertools.combinations(range(6),3)])
    for _ in range(50):
        phis=rng.normal(size=(6,96))
        design=incidence@phis
        centered=design-design.mean(axis=0)
        rawrank=int(np.linalg.matrix_rank(design))
        centeredrank=int(np.linalg.matrix_rank(centered))
        assert rawrank<=6 and centeredrank<=5
        rank.append((rawrank,centeredrank))
    # A question-level three-repeat mean and repeated raw updates have the
    # same fitted mean only after scaling regularization. UCB covariance also
    # changes by a factor of3, so unchanged alpha is not equivalent.
    p=24
    meanmodel=core.LinUCBSharedCombinatorial(p,alpha=3.5,lam=.1)
    repeated=core.LinUCBSharedCombinatorial(p,alpha=3.5*np.sqrt(3),lam=.3)
    for _ in range(30):
        z=rng.normal(size=p)
        rewards=rng.integers(0,2,3).astype(float)
        meanmodel.update(z,float(rewards.mean()))
        for r in rewards:
            repeated.update(z,float(r))
    repeat_mean_error=float(np.max(abs(meanmodel.w_hat-repeated.w_hat)))
    repeat_cov_error=float(np.max(abs(meanmodel.V_inv/3-repeated.V_inv)))
    query=rng.normal(size=p)
    bonus_error=abs(meanmodel._item_ucb(query)-repeated._item_ucb(query))
    assert repeat_mean_error<1e-10 and repeat_cov_error<1e-10 and bonus_error<1e-10
    base_profiles=rng.uniform(0,1,(6,3))
    w_x,w_m=init['init_mats_pca_warmstart'](32,392,list(base_profiles),seed=42)
    assert w_x.shape==(32,392) and w_m.shape==(32,3)
    native_rank_failure=False
    try:
        init['init_mats_pca_warmstart'](32,392,list(rng.uniform(0,1,(6,11))),seed=42)
    except ValueError:
        native_rank_failure=True
    assert native_rank_failure
    # Find a strict, valid build_phi witness, preserving the three-block CES
    # feature structure and an internally consistent offline-loaded state.
    witness_rng=np.random.default_rng(20261112732)
    witness=None
    for trial in range(1000):
        xx=witness_rng.normal(size=2)
        vv=witness_rng.normal(size=(6,2))
        phis=np.asarray([projectors.build_phi(xx,v) for v in vv])
        ww=witness_rng.normal(size=6)*.1
        policy=core.LinUCBSharedCombinatorial(6,alpha=3.5,lam=1.)
        policy.load_state(V_inv=np.eye(6),b=ww,w_hat=ww)
        set_choice=policy.select_topk(list(phis),3)
        item_scores=np.asarray([policy._item_ucb(phi) for phi in phis])
        item_choice=np.argsort(-item_scores)[:3]
        if set(set_choice)!=set(item_choice) and np.min(np.diff(np.sort(item_scores)))>1e-8:
            witness=dict(source='synthetic valid build_phi and consistent offline state',
                found_trial=trial,x=xx.tolist(),v=vv.tolist(),w_hat=ww.tolist(),
                V_inv=np.eye(6).tolist(),phis=phis.tolist(),item_scores=item_scores.tolist(),
                set_choice=set_choice.tolist(),independent_topK=item_choice.tolist())
            break
    assert witness is not None
    result=dict(status='PASS',claim_scope='Pinned pure-numpy selection/update operator and model-class software checks, not original benchmark reproduction',
        source_commit=manifest['commit'],source_hashes={name:sha(UPSTREAM/name) for name in ['src/bandits/linucb_combinatorial.py','src/embeddings/projectors.py','scripts/run_objective_mcq.py','LICENSE']},
        settings=per_setting,selected_sets_checked=selected,updates_checked=updates,
        maximum_dense_mean_error=max_mean,maximum_dense_covariance_error=max_cov,maximum_feature_error=maxfeature,
        fixed_question_rank=dict(checks=50,unrestricted_incidence_rank=int(np.linalg.matrix_rank(incidence)),
            maximum_design_rank=max(r[0] for r in rank),maximum_rowcentered_design_rank=max(r[1] for r in rank)),
        repeat_unit=dict(mean_error=repeat_mean_error,covariance_scale_error=repeat_cov_error,UCB_bonus_error=bonus_error,
            equivalence='lambda_raw=3*lambda_mean, alpha_raw=sqrt(3)*alpha_mean'),
        initializer=dict(six_members_three_profile_columns='PASS',six_members_eleven_profile_columns_native_dimension_failure=native_rank_failure,
            reason='upstream k=min(d_x,d_s) exceeds available SVD rows when number ofmembers<d_s'),
        set_UCB_counterexample=witness,
        physical_new_LLM_calls=0)
    out=WORK/'public/adapter'
    out.mkdir(parents=True,exist_ok=True)
    (out/'CES_ADAPTER_OPERATOR_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
