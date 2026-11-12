"""Read-only executable checks of the pinned public CES mathematical core."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import ast
import hashlib
import importlib.util
import json
import sys
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(r'E:\语言模型组合路由')
OLD=ROOT/'third_party/ces_source_audit/upstream'
WORK=ROOT/'research/ces_traceable_baseline'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private/native'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def import_exact(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def expert_specs(path):
    tree=ast.parse(path.read_text('utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expert_specs' for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node,ast.FunctionDef) and node.name=='default_expert_specs':
            for statement in node.body:
                if isinstance(statement,ast.Return):
                    return ast.literal_eval(statement.value)
    raise ValueError('No static expert specifications')


def main():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    core=import_exact('ces_author_core_native_audit',OLD/'src/bandits/linucb_combinatorial.py')
    features=import_exact('ces_author_phi_native_audit',OLD/'src/embeddings/projectors.py')
    rng=np.random.default_rng(20261112)
    errors=[]
    selection_errors=[]
    independent_topk_disagreements=0
    checks=0
    for trial in range(200):
        dimension=12
        model=core.LinUCBSharedCombinatorial(dimension,alpha=3.5,lam=.1)
        X=rng.normal(size=(20,dimension))
        rewards=rng.uniform(0,1,size=20)
        rows=[]
        labels=[]
        for x,y in zip(X,rewards):
            model.update(x,float(y))
            rows.append(x)
            labels.append(y)
            design=np.asarray(rows)
            direct_inverse=np.linalg.inv(.1*np.eye(dimension)+design.T@design)
            direct_b=design.T@np.asarray(labels)
            errors.append(float(np.max(abs(model.V_inv-direct_inverse))))
            errors.append(float(np.max(abs(model.w_hat-direct_inverse@direct_b))))
            checks+=2
        phis=rng.normal(size=(6,dimension))
        costs=rng.uniform(.1,2,size=6)
        selected=model.select_topk(list(phis),3,costs=costs,lam_cost=.1)
        selected_reference=[]
        z=np.zeros(dimension)
        for _ in range(3):
            scores=[]
            for j in range(6):
                if j in selected_reference:
                    scores.append(-np.inf)
                    continue
                before=np.sqrt(max(z@model.V_inv@z,1e-12)) if selected_reference else 0.
                after=np.sqrt(max((z+phis[j])@model.V_inv@(z+phis[j]),1e-12))
                scores.append(float(phis[j]@model.w_hat-.1*costs[j]+3.5*(after-before)))
            j=int(np.argmax(scores))
            selected_reference.append(j)
            z+=phis[j]
        selection_errors.append(int(np.sum(selected!=selected_reference)))
        checks+=3
        individual=phis@model.w_hat+3.5*np.sqrt(np.einsum('nd,df,nf->n',phis,model.V_inv,phis))-.1*costs
        independent=np.argsort(-individual)[:3]
        independent_topk_disagreements+=int(set(selected)!=set(independent))
    x=np.array([1.,2.,-1.,.5])
    v=np.array([2.,-.5,1.,3.])
    actual=features.build_phi(x,v)
    eps=1e-12
    expected=np.r_[x/(np.linalg.norm(x)+eps),3*v/(np.linalg.norm(v)+eps),3*(x*v)/(np.linalg.norm(x*v)+eps)]
    assert np.max(abs(actual-expected))<1e-14
    checks+=1
    default_dimension_errors=[]
    for _ in range(8):
        model=core.LinUCBSharedCombinatorial(96,alpha=3.5,lam=.1)
        history=[]
        targets=[]
        for step in range(10):
            query=rng.normal(size=32)
            phis=[features.build_phi(query,rng.normal(size=32)) for _ in range(15)]
            action=model.select_topk(phis,3)
            z=np.sum([phis[j] for j in action],axis=0)
            target=float(rng.uniform())
            model.update(z,target)
            history.append(z)
            targets.append(target)
            design=np.asarray(history)
            expected_weight=np.linalg.solve(.1*np.eye(96)+design.T@design,design.T@targets)
            default_dimension_errors.append(float(np.max(abs(model.w_hat-expected_weight))))
    assert max(default_dimension_errors)<1e-9
    profile=json.loads((OLD/'artifacts/mmlu_meta_10d.json').read_text())
    supported=set(profile['models'])
    runners={}
    for name in ['run_alpacaeval2.py','run_mt_bench.py','run_flask.py','run_objective_mcq.py']:
        path=OLD/'scripts'/name
        specifications=expert_specs(path)
        missing=[row['name'] for row in specifications if row['name'] not in supported]
        runners[name]=dict(expert_count=len(specifications),missing_profile_keys=missing,source_sha256=digest(path),
                           names=[row['name'] for row in specifications])
    status='PASS' if max(errors)<1e-9 and max(selection_errors)==0 else 'FAIL'
    result=dict(status=status,scope='Exact author mathematical modules imported, synthetic operator qualification only',
        physical_LLM_calls=0,API_clients_created=0,checks=checks,trials=200,
        sequential_dense_inverse_max_error=max(errors),selection_reconstruction_errors=max(selection_errors),
        shared_setUCB_vs_independentTopK_disagreements=independent_topk_disagreements,
        disagreement_denominator=200,feature_formula_max_error=float(np.max(abs(actual-expected))),
        original_default_dimension_check=dict(feature_dimension=96,projection_dimension=32,expert_pool=15,k=3,
            trials=8,updates_per_trial=10,max_dense_mean_error=max(default_dimension_errors)),
        source_sha256={str(path.relative_to(OLD)):digest(path) for path in [OLD/'src/bandits/linucb_combinatorial.py',OLD/'src/embeddings/projectors.py']},
        pinned_commit='9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7',runner_profiles=runners,
        original_paper_benchmark_reproduced=False,original_per_question_output_cache_available=False,
        claim='This validates only the official core numerical behavior. It cannot reproduce published benchmark scores without original outputs/judgments or new API generation.')
    (PUBLIC/'NATIVE_OPERATOR_QUALIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if status!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    main()
