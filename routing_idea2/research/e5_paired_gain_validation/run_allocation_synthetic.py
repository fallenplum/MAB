"""Shared-world dual-budget synthetic experiments. No train600 scores are read."""
from allocation import *
from certification import binomial_interval
from run_certification_synthetic import cluster_distribution
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import math
import time


SEED=20261110
WEIGHTS=.9*np.array([.10,.20,.15,.20,.15,.20])
WEIGHTS=np.r_[WEIGHTS,.1]
COSTS=np.array([.5,1.,2.,4.,8.,16.,2.])
RHO=np.array([.04,.12,.2,.4,.65,.9,0.])
MU=np.array([.015,.025,.03,-.04,-.03,.08,0.])
UNION_MEMBERS=np.array([4,4,5,5,6,6,0])
L=3
T=400
P_MIN=.05
PILOT_PER_ACTIVE_STRATUM=30
METHODS=['Uniform','Cost-only','Disagreement-based','Single-resource',
         'Estimated-two-resource','Oracle-two-resource','No-pilot-Uniform']
FAMILIES={
    'nested':np.array([[1,1,0,0,0,0,0],[1,1,1,1,0,0,0],[1,1,1,1,1,1,0]]),
    'nonnested':np.array([[1,1,0,0,0,0,0],[0,0,1,1,0,0,0],
                          [0,0,0,0,1,1,0],[1,0,1,0,1,0,0]])
}
CONFIG=dict(seed=SEED,L=L,questions=T,p_min=P_MIN,replicates=2000,budget_fractions=[.25,.5,.75],
            pilot_per_active_stratum=PILOT_PER_ACTIVE_STRATUM,pilot_independent_questions=180,
            pilot_sensitivity_repeats=50,weights=WEIGHTS.tolist(),costs=COSTS.tolist(),
            rho=RHO.tolist(),mu=MU.tolist(),union_members=UNION_MEMBERS.tolist(),
            methods=METHODS,gates={k:v.tolist() for k,v in FAMILIES.items()},
            independent_questions='All incoming questions iid, active mass .9, inactive .1 remains in denominator',
            repeat_law='Three conditionally independent ternary generation gains within each synthetic question',
            Monte_Carlo_scope='Common input/gain/acquisition-uniform world shared across all methods, conditional on one frozen independent pilot',
            primary_unknowns='Only second moments estimated, true weights and expected costs shared',
            pilot_sensitivity='50 additional iid pilots estimate weights/costs/moments; changed feasible set and true budget drift reported separately',
            selection='plugin argmax with zero-default and fixed-horizon IPW-certified selection both reported')


def native(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(x) for x in value]
    return value


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(native(value),indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def true_moments():
    return (RHO-MU**2)/L+MU**2


def conditional_pilot(family):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,1,list(FAMILIES).index(family)]))
    active=FAMILIES[family].any(0)
    a_hat=np.zeros(7);rho_hat=np.zeros(7)
    raw={}
    for h in np.flatnonzero(active):
        base=np.array([(RHO[h]-MU[h])/2,1-RHO[h],(RHO[h]+MU[h])/2])
        generated=rng.choice(np.array([-1.,0.,1.]),size=(PILOT_PER_ACTIVE_STRATUM,L),p=base)
        dbar=generated.mean(1)
        a_hat[h]=np.mean(dbar**2)
        rho_hat[h]=np.mean(generated**2)
        raw[str(h)]=dict(generation_gains=generated.tolist(),mean=dbar.tolist())
    mh=np.where(active,PILOT_PER_ACTIVE_STRATUM,0)
    cost=dict(gold=int(mh.sum()),execution_units=float(mh@COSTS),
              member_calls=int(L*(mh*UNION_MEMBERS).sum()),aggregator_calls=int(2*L*mh.sum()),
              scoring_ops=int(2*L*mh.sum()),real_tokens=None,new_physical_calls=0,
              conditional_input_sampling_assumed=True,
              expected_input_draws_if_separate_rejection_sampling=float(np.sum(mh[active]/WEIGHTS[active])),
              recruitment_note='Synthetic pilot samples each declared stratum conditionally. Rejection-input draws are reported; pair calls and gold charged only on accepted pilot questions.')
    return a_hat,rho_hat,cost,raw


def designs(family,f_ann,f_exec):
    gates=FAMILIES[family]
    a=true_moments()
    active=gates.any(0)
    B_ann=f_ann*float(WEIGHTS[active].sum())
    B_exec=f_exec*float((WEIGHTS[active]*COSTS[active]).sum())
    ah,rh,pilot,raw=conditional_pilot(family)
    probabilities={}
    detail={}
    probabilities['Uniform']=priority_design(WEIGHTS,COSTS,gates,np.ones(7),B_ann,B_exec)
    probabilities['Cost-only']=priority_design(WEIGHTS,COSTS,gates,1/COSTS,B_ann,B_exec)
    probabilities['Disagreement-based']=priority_design(WEIGHTS,COSTS,gates,np.sqrt(rh),B_ann,B_exec)
    # Annotation-only optimal shape, then common feasible backoff to the execution budget.
    single=solve_design(WEIGHTS,ah,COSTS,gates,B_ann,float(WEIGHTS@COSTS)+1)
    probabilities['Single-resource'],backoff=feasibility_backoff(single['probabilities'],WEIGHTS,COSTS,gates,B_ann,B_exec)
    detail['Single-resource']=dict(annotation_only_solution=native(single),execution_feasibility_backoff=backoff)
    estimated=solve_design(WEIGHTS,ah,COSTS,gates,B_ann,B_exec)
    oracle=solve_design(WEIGHTS,a,COSTS,gates,B_ann,B_exec)
    probabilities['Estimated-two-resource']=estimated['probabilities']
    probabilities['Oracle-two-resource']=oracle['probabilities']
    detail['Estimated-two-resource']=native(estimated)
    detail['Oracle-two-resource']=native(oracle)
    no_pilot_B_ann=B_ann+pilot['gold']/T
    no_pilot_B_exec=B_exec+pilot['execution_units']/T
    probabilities['No-pilot-Uniform']=priority_design(WEIGHTS,COSTS,gates,np.ones(7),no_pilot_B_ann,no_pilot_B_exec)
    true_F_est=float(second_moments(WEIGHTS,a,gates,estimated['probabilities']).max())
    epsilon=abs(ah-a)
    E_str=float((gates*(WEIGHTS*epsilon)).sum(1).max()/P_MIN)
    gap=true_F_est-oracle['objective']
    assert gap>=-1e-6 and gap<=2*E_str+2e-6
    assert abs(oracle['KKT']['duality_gap'])<2e-6
    for method,p in probabilities.items():
        b0=no_pilot_B_ann if method=='No-pilot-Uniform' else B_ann
        b1=no_pilot_B_exec if method=='No-pilot-Uniform' else B_exec
        assert np.all(p[active]>=P_MIN-1e-12) and np.all(p[~active]==0)
        assert WEIGHTS@p<=b0+1e-7 and (WEIGHTS*COSTS)@p<=b1+1e-7
    meta=dict(family=family,annotation_fraction=f_ann,execution_fraction=f_exec,
        phase_B_ann=B_ann,phase_B_exec=B_exec,total_gold_budget=T*B_ann+pilot['gold'],
        total_execution_budget=T*B_exec+pilot['execution_units'],pilot=pilot,
        pilot_estimated_second_moment=ah.tolist(),pilot_estimated_single_repeat_disagreement=rh.tolist(),
        true_cluster_second_moment=a.tolist(),true_single_repeat_disagreement=RHO.tolist(),
        moment_robustness=dict(true_objective_excess=gap,bound_2E_str=2*E_str,pointwise_abs_errors=epsilon.tolist(),
                               same_true_weights_costs_and_feasible_set=True),
        solver_details=detail,dominance=gate_dominance(gates),
        probabilities={name:p.tolist() for name,p in probabilities.items()})
    return probabilities,meta,raw


def shared_world_cells(gates,probabilities):
    """Partition one common acquisition U into intervals for every method."""
    M=len(METHODS);J=len(gates)
    p=np.asarray([probabilities[name] for name in METHODS])
    joint=[];values=[];costs=[]
    for h,w in enumerate(WEIGHTS):
        if not gates[:,h].any():
            joint.append(w);values.append(np.zeros((M,J)));costs.append(np.zeros((M,5)))
            continue
        support,law=cluster_distribution(RHO[h],MU[h],L,'conditional_independent')
        edges=np.unique(np.r_[0.,p[:,h],1.])
        for lo,hi in zip(edges[:-1],edges[1:]):
            acquire=(lo+hi)/2<p[:,h]
            for d,prob in zip(support,law):
                if prob==0 or hi==lo:continue
                joint.append(w*prob*(hi-lo))
                contribution=np.zeros((M,J))
                contribution[acquire]=d*gates[:,h][None,:]/p[acquire,h,None]
                values.append(contribution)
                resource=np.zeros((M,5))
                resource[acquire]=np.array([1.,COSTS[h],L*UNION_MEMBERS[h],2*L,2*L])
                costs.append(resource)
    joint=np.asarray(joint);joint/=joint.sum()
    return joint,np.asarray(values),np.asarray(costs)


def simulate(case,reps):
    family,f_ann,f_exec=case
    start=time.perf_counter()
    gates=FAMILIES[family]
    probabilities,meta,pilot_raw=designs(*case)
    joint,values,resources=shared_world_cells(gates,probabilities)
    J=len(gates)
    truth=(gates*(WEIGHTS*MU)).sum(1)
    true_best=float(max(0.,truth.max()))
    baseline_accuracy=float(1-(WEIGHTS*(RHO+MU)/2).sum())
    rng=np.random.default_rng(np.random.SeedSequence([SEED,2,list(FAMILIES).index(family),int(f_ann*100),int(f_exec*100)]))
    counts=rng.multinomial(T,joint,size=reps)
    estimates=(counts@values.reshape(len(joint),-1)/T).reshape(reps,len(METHODS),J)
    actual=(counts@resources.reshape(len(joint),-1)).reshape(reps,len(METHODS),5)
    summary={}
    all_squared=[]
    for m,method in enumerate(METHODS):
        p=probabilities[method]
        estimate=estimates[:,m]
        squares=(estimate-truth)**2
        all_squared.append(squares.max(1))
        selected=np.where(estimate.max(1)>0,np.argmax(estimate,axis=1),-1)
        chosen=np.where(selected>=0,truth[np.maximum(selected,0)],0.)
        u=math.log(2*J/.05)
        radius=math.sqrt(2*u/(T*P_MIN))+4*u/(3*T*P_MIN)
        certified=np.where((estimate-radius).max(1)>0,np.argmax(estimate-radius,axis=1),-1)
        certified_gain=np.where(certified>=0,truth[np.maximum(certified,0)],0.)
        pilot_gold=0 if method=='No-pilot-Uniform' else meta['pilot']['gold']
        pilot_exec=0 if method=='No-pilot-Uniform' else meta['pilot']['execution_units']
        pilot_mem=0 if method=='No-pilot-Uniform' else meta['pilot']['member_calls']
        pilot_agg=0 if method=='No-pilot-Uniform' else meta['pilot']['aggregator_calls']
        cost=actual[:,m]
        phase_ann=float(WEIGHTS@p);phase_exec=float((WEIGHTS*COSTS)@p)
        f=second_moments(WEIGHTS,true_moments(),gates,p)
        variance=(f-truth**2)/T
        expected_budget=dict(annotation_per_incoming=phase_ann,execution_per_incoming=phase_exec,
            gold_including_pilot=T*phase_ann+pilot_gold,execution_including_pilot=T*phase_exec+pilot_exec,
            annotation_unused=float(meta['total_gold_budget']-T*phase_ann-pilot_gold),
            execution_unused=float(meta['total_execution_budget']-T*phase_exec-pilot_exec),
            hypothetical_full_active_saturation=bool(np.all(p[gates.any(0)]>1-1e-10)))
        summary[method]=dict(probabilities=p.tolist(),true_gains=truth.tolist(),true_accuracy_default=baseline_accuracy,
            second_moments=f.tolist(),worst_second_moment=float(f.max()),
            exact_iid_estimator_variances=variance.tolist(),bias=(estimate.mean(0)-truth).tolist(),
            bias_MC_standard_errors=(estimate.std(0,ddof=1)/math.sqrt(reps)).tolist(),
            MSE=squares.mean(0).tolist(),MSE_MC_standard_errors=(squares.std(0,ddof=1)/math.sqrt(reps)).tolist(),
            worst_gate_MSE=float(squares.mean(0).max()),max_estimator_variance=float(variance.max()),
            plugin_policy_selection=dict(mean_value_loss=float(np.mean(true_best-chosen)),
                mean_true_gain=float(chosen.mean()),population_accuracy=float(baseline_accuracy+chosen.mean()),
                true_best_gain=true_best,harmful_selection_rate=float(np.mean(chosen<0)),default_rate=float(np.mean(selected<0)),
                selected_gate_counts=np.bincount(selected+1,minlength=J+1).tolist()),
            fixed_horizon_certified_selection=dict(radius=radius,mean_true_gain=float(certified_gain.mean()),
                positive_detection_rate=float(np.mean(certified_gain>0)),harmful_selection_rate=float(np.mean(certified_gain<0)),
                default_rate=float(np.mean(certified<0)),candidate_family_J=J,
                population_accuracy=float(baseline_accuracy+certified_gain.mean())),
            expected_resources=expected_budget,
            observed_resources=dict(mean_phase_gold=float(cost[:,0].mean()),mean_phase_execution=float(cost[:,1].mean()),
                mean_member_calls_including_pilot=float(cost[:,2].mean()+pilot_mem),
                mean_aggregator_calls_including_pilot=float(cost[:,3].mean()+pilot_agg),
                mean_scoring_ops_including_pilot=float(cost[:,4].mean()+(0 if method=='No-pilot-Uniform' else meta['pilot']['scoring_ops'])),
                total_gold_overflow_probability=float(np.mean(cost[:,0]+pilot_gold>meta['total_gold_budget']+1e-9)),
                total_execution_overflow_probability=float(np.mean(cost[:,1]+pilot_exec>meta['total_execution_budget']+1e-9)),
                mean_gold_including_pilot=float(cost[:,0].mean()+pilot_gold),
                mean_execution_including_pilot=float(cost[:,1].mean()+pilot_exec),
                real_tokens=None,resource_semantics='Expected constraints permit realized overflow; no hard-cap stopping applied'),
            reuse='one acquired pair shared by all J frozen gates, no J multiplier on gold/member/aggregator calls')
    # Paired MC errors concern simulation replication, not extra real questions.
    ref=METHODS.index('Uniform')
    for m,method in enumerate(METHODS):
        difference=((estimates[:,m]-truth)**2).mean(1)-((estimates[:,ref]-truth)**2).mean(1)
        se=float(difference.std(ddof=1)/math.sqrt(reps))
        summary[method]['mean_gate_MSE_minus_uniform_MC']=dict(difference=float(difference.mean()),
            normal_MC_ci95=[float(difference.mean()-1.96*se),float(difference.mean()+1.96*se)],
            note='paired common synthetic question and U worlds')
    meta.update(status='COMPLETE',methods=summary,replicates=reps,population='known iid synthetic mixture',
        incoming_question_count=T,active_incoming_probability=.9,unacquired_or_inactive_denominator=T,
        joint_shared_world_cells=len(joint),wall_s=time.perf_counter()-start,new_llm_calls=0)
    arrays=dict(estimates=estimates,realized_resources=actual,true_gains=truth,
                method_names=np.asarray(METHODS),joint_probability=joint,
                cells_contributions=values,cells_resources=resources)
    return meta,arrays,pilot_raw


def estimated_feasible_set_sensitivity(family,repeats=50):
    gates=FAMILIES[family]
    a=true_moments()
    rows=[]
    for seed in range(repeats):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,3,list(FAMILIES).index(family),seed]))
        h=rng.choice(len(WEIGHTS),size=180,p=WEIGHTS)
        counts=np.bincount(h,minlength=7)
        what=(counts+.5)/(len(h)+.5*len(WEIGHTS))
        ah=np.zeros(7);chat=np.ones(7)
        pilot_gold=0;pilot_exec=0.
        for stratum in range(7):
            count=counts[stratum]
            if not count or not gates[:,stratum].any():continue
            support,law=cluster_distribution(RHO[stratum],MU[stratum],L,'conditional_independent')
            dbar=rng.choice(support,size=count,p=law)
            ah[stratum]=np.mean(dbar**2)
            costs=rng.lognormal(math.log(COSTS[stratum])-.4**2/2,.4,size=count)
            chat[stratum]=costs.mean()
            pilot_gold+=count;pilot_exec+=costs.sum()
        for f_ann in (.25,.5,.75):
            for f_exec in (.25,.5,.75):
                B_ann=f_ann*WEIGHTS[gates.any(0)].sum()
                B_exec=f_exec*(WEIGHTS*COSTS)[gates.any(0)].sum()
                fitted=solve_design(what,ah,chat,gates,B_ann,B_exec)
                p=fitted['probabilities']
                actual_ann=float(WEIGHTS@p);actual_exec=float((WEIGHTS*COSTS)@p)
                rows.append(dict(seed=seed,annotation_fraction=f_ann,execution_fraction=f_exec,
                    true_expected_annotation=actual_ann,true_expected_execution=actual_exec,
                    annotation_budget=float(B_ann),execution_budget=float(B_exec),
                    true_annotation_violation=max(0.,float(actual_ann-B_ann)),
                    true_execution_violation=max(0.,float(actual_exec-B_exec)),
                    weight_max_error=float(abs(what-WEIGHTS).max()),moment_max_error=float(abs(ah-a).max()),
                    cost_max_error=float(abs(chat[gates.any(0)]-COSTS[gates.any(0)]).max()),pilot_gold=int(pilot_gold),pilot_execution=float(pilot_exec),
                    solver_duality_gap=float(fitted['KKT']['duality_gap']),
                    solver_warning_count=len(fitted.get('optimizer_warnings',[])),
                    solver_estimated_domain_constraint_violation=float(fitted.get('max_constraint_violation',0)),
                    assumption='estimated weights/costs change feasible set; same-feasible-set Theorem19 not asserted'))
    return dict(family=family,independent_pilot_repeats=repeats,rows=rows,
        true_annotation_violation_rate=float(np.mean([r['true_annotation_violation']>1e-7 for r in rows])),
        true_execution_violation_rate=float(np.mean([r['true_execution_violation']>1e-7 for r in rows])),
        no_automatic_feasibility_guarantee=True)


def worker(job):
    case,reps,output=job
    name=f"{case[0]}_ann{int(case[1]*100)}_exec{int(case[2]*100)}"
    result,arrays,pilot=simulate(case,reps)
    output=Path(output)
    save(output/'cases'/f'{name}.json',result)
    (output/'cases').mkdir(exist_ok=True)
    np.savez_compressed(output/'cases'/f'{name}.npz',**arrays)
    if case[1]==.25 and case[2]==.25:
        save(output/'private_pilots'/f'{case[0]}.json',pilot)
    print(json.dumps(dict(case=name,wall_s=result['wall_s'])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'allocation_synthetic')
    parser.add_argument('--replicates',type=int,default=2000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--real-gate-status',choices=['GATED_NOT_STARTED','NOT_RUN_GATE_NOT_PASSED'],default='GATED_NOT_STARTED')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    checked=qualification()
    cases=[(family,ann,exe) for family in FAMILIES for ann in (.25,.5,.75) for exe in (.25,.5,.75)]
    if args.smoke:cases=cases[:1]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(worker,[(case,args.replicates,str(args.output)) for case in cases]))
    sensitivity=[]
    if not args.smoke:
        for family in FAMILIES:
            sensitivity.append(estimated_feasible_set_sensitivity(family,50))
    report=dict(status='SMOKE' if args.smoke else 'COMPLETE',config={**CONFIG,'replicates':args.replicates},
        qualification=checked,cases=rows,estimated_weights_costs_sensitivity=sensitivity,
        wall_s=time.perf_counter()-start,manuscript_sha256='b2c9c7f9bf09a3183f4907eb7daedd3a7537dc9f6460f62ccfc246e525bb9e99',
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('allocation.py'),Path(__file__).with_name('certification.py')]},
        real_train600_allocation=args.real_gate_status,new_llm_calls=0)
    save(args.output/'ALLOCATION_SYNTHETIC_RESULTS.json',report)
    print(json.dumps(dict(status=report['status'],cases=len(rows),wall_s=report['wall_s'])),flush=True)


if __name__=='__main__':main()
