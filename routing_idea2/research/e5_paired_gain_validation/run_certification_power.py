"""Fold-specific E5-C certification and development-pool power diagnostics."""
from certification import *
from pathlib import Path
import argparse
import hashlib
import json
import time


ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e5_paired_gain_validation'
SEED=20261110
COUNTS=[200,400,1000,3000,10000]
TARGET_GAINS=[.005,.01,.02]


def native(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(x) for x in value]
    return value


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_sufficient_questions(rho_by_stratum,gain,J=20,delta=.05):
    """A plug-in F.1 corollary of manuscript Eq. (68), equal stratum counts.

    LCB_main >= G - 6 A sqrt(t/n) - 24 t/n,
    A = .5*(sqrt(rho_gsm)+sqrt(rho_bbh)), t=log(12J/delta).
    A future population bound requires population-valid rho upper bounds.
    """
    r=np.asarray(rho_by_stratum,float)
    if r.shape!=(2,) or np.any((r<0)|(r>1)) or not 0<gain<=r.mean()+1e-12:return None
    t=math.log(12*J/delta)
    A=float(np.sqrt(r).mean())
    q=6*A*math.sqrt(t)
    linear=24*t
    positive_root=2*gain/(q+math.sqrt(q*q+4*linear*gain))
    n=math.floor(1/positive_root**2)+1
    return max(4,2*math.ceil(n/2))


def metadata_only():
    bundle_path=ROOT/'artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json'
    bundle=json.loads(bundle_path.read_text(encoding='utf-8-sig'))
    ref=bundle['inputs']['selector_train']
    path=Path(ref['path'])
    if not path.is_absolute():path=ROOT/path
    path=path.resolve()
    if not path.is_relative_to(ROOT.resolve()) or sha(path)!=ref['sha256']:
        raise ValueError('Frozen input identity mismatch')
    with np.load(path,allow_pickle=False) as archive:
        tasks=archive['task_labels'].astype(str)
        subtasks=archive['subtask_labels'].astype(str)
    return tasks,subtasks,dict(bundle_sha256=sha(bundle_path),inputs_sha256=sha(path),
        accessed_arrays=['task_labels','subtask_labels'],outcome_bundle_accessed=False)


def weighted(value,task):
    return .5*(np.asarray(value)[task=='gsm8k'].mean(0)+np.asarray(value)[task=='bbh'].mean(0))


def bootstrap_fold(z,task,delta,reps,fold):
    J=z.shape[1]
    per=[z[task==name] for name in ('gsm8k','bbh')]
    empirical_truth=.5*(per[0].mean(0)+per[1].mean(0))
    output=[]
    for n in COUNTS:
        nh=n//2
        rng=np.random.default_rng(np.random.SeedSequence([SEED,4,fold,n]))
        mean=np.zeros((reps,J));radius=np.zeros((reps,J))
        for subset in per:
            counts=rng.multinomial(nh,np.ones(len(subset))/len(subset),size=reps)
            means=counts@subset/nh
            variances=np.maximum((counts@(subset**2)-nh*means**2)/(nh-1),0.)
            mean+=.5*means
            radius+=.5*empirical_bernstein_radius(variances,nh,J,delta/2)
        lcb=mean-radius
        chosen=np.where(lcb.max(1)>0,np.argmax(lcb,axis=1),-1)
        selected_gain=np.where(chosen>=0,empirical_truth[np.maximum(chosen,0)],0.)
        false=np.any((lcb>0)&(empirical_truth<=1e-15),axis=1)
        default=int(np.sum(chosen<0));positive=int(np.sum(selected_gain>1e-15));harmful=int(np.sum(selected_gain < -1e-15))
        output.append(dict(simulated_draws=n,observed_distinct_source_questions=len(z),
            replicates=reps,truth='fixed empirical within-dataset question distribution, not population truth',
            positive_selection_rate=positive/reps,positive_selection_MC_ci95=binomial_interval(positive,reps),
            empirical_harmful_selection_rate=harmful/reps,empirical_harmful_selection_MC_ci95=binomial_interval(harmful,reps),
            false_nonpositive_certificate_rate=float(false.mean()),
            false_nonpositive_certificate_MC_ci95=binomial_interval(int(false.sum()),reps),
            returned_default_rate=default/reps,returned_default_MC_ci95=binomial_interval(default,reps),
            mean_interval_width=float(2*radius.mean()),mean_selected_empirical_gain=float(selected_gain.mean()),
            no_new_independent_observations=True,certificate_delta=delta))
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'power_development')
    parser.add_argument('--replicates',type=int,default=2000)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    tasks,subtasks,input_identity=metadata_only()
    certs=[];pools=[];source={};rule_names=None
    means=[];radii=[];boot=[];planning=[]
    start=time.perf_counter()
    for fold in range(5):
        path=WORK/'private/gates'/f'GATES_f{fold}.npz'
        frozen=path.with_name(f'FROZEN_GATE_f{fold}.json')
        source[f'GATES_f{fold}.npz']=sha(path)
        source[f'FROZEN_GATE_f{fold}.json']=sha(frozen)
        with np.load(path,allow_pickle=False) as a:
            E=a['E'].astype(int)
            D=a['D_E'].astype(float)
            gates=a['gate_masks_E'].astype(int)
            main_indices=a['main_indices'].astype(int)
        assert D.shape==(120,3) and gates.shape==(4,5,120)
        masks=gates.reshape(20,120).T
        names=[score+'_'+threshold for score in ['gap','factorized','disagreement','direction'] for threshold in ['q10','q25','q50','q75','zero']]
        if rule_names is None:rule_names=names
        assert names==rule_names and np.isin(D,(-1,0,1)).all() and np.isin(masks,(0,1)).all()
        t=tasks[E]
        assert sum(t=='gsm8k')==sum(t=='bbh')==60
        Z=masks*D.mean(1)[:,None]
        U=masks*(D**2).mean(1)[:,None]
        cert=stratified_certificate(Z,t,{'gsm8k':.5,'bbh':.5},delta=.05/5)
        selected=int(cert['selected'])
        means.append(cert['mean']);radii.append(cert['radius'])
        certs.append(dict(fold=fold,n=120,J=20,delta_fold=.01,delta_each_dataset=.005,
            primary_factorized_rule=names[5+main_indices[1]],selected_rule=names[selected] if selected>=0 else 'default',
            certificate=native(cert),role='controlled existing-development diagnostic of the frozen fold-specific family',
            assumptions='a population certificate would additionally require independent within-stratum calibration questions and the target marginal law'))
        pools.append(dict(E=E,D=D,Z=Z,U=U,task=t,masks=masks))
        boot.append(dict(fold=fold,rows=bootstrap_fold(Z,t,.01,args.replicates,fold)))
        for j,name in enumerate(names):
            variances=np.asarray([Z[t==x,j].var(ddof=1) for x in ('gsm8k','bbh')])
            rho=np.asarray([U[t==x,j].mean() for x in ('gsm8k','bbh')])
            planning.append(dict(fold=fold,rule=name,empirical_gain=float(weighted(Z[:,j],t)),
                empirical_rho_by_dataset=rho.tolist(),empirical_question_variance_by_dataset=variances.tolist(),
                proposed_future_single_family_delta=.05,J=20,
                sample_size_quantity_type='certificate-specific plugin sufficient count, not a necessary count for all procedures',
                target_gains=[dict(gain=g,
                    hypothetical_iid_mixture_sufficient_n=sufficient_questions(float(rho.mean()),g,20,.05),
                    fixed_balanced_F1_plugin_sufficient_n=balanced_sufficient_questions(rho,g,20,.05)) for g in TARGET_GAINS],
                empirical_radius_planning=[dict(total_independent_questions=n,
                    F1_plugin_half_width=float(np.mean(empirical_bernstein_radius(variances,n//2,20,.05/2)))) for n in COUNTS],
                caveat='rho and variance estimated from development data, not validated population upper bounds'))
    mean=np.mean(means,axis=0);radius=np.mean(radii,axis=0)
    lcb=mean-radius
    aggregate=dict(mean=mean.tolist(),radius=radius.tolist(),lcb=lcb.tolist(),ucb=(mean+radius).tolist(),
        selected_rule=rule_names[int(np.argmax(lcb))] if lcb.max()>0 else 'default',delta_total=.05,
        target='average of five fold-specific frozen procedures; not one pooled iid 600-question policy',
        justification='union bound over five fold events; no independence between these events is required',
        observed_questions=600,question_generation_repeats=3)
    report=dict(status='COMPLETE',rule_names=rule_names,fold_certificates=certs,
        fivefold_family_average_diagnostic=aggregate,planning=planning,development_pool_resampling=boot,
        planning_derivation='F1 plus Eq68: balanced n/2 per dataset implies LCB>=G-6*mean(sqrt(rho_h))*sqrt(t/n)-24*t/n with t=log(12J/delta). Invert this strictly to plan even n.',
        inference_scope='Existing repeatedly inspected development questions, fixed finite pools and overlapping historical fits. This is neither fresh-data certification nor empirical proof of population power.',
        original_equation69_scope='iid target mixture assumption; shown separately from fixed-balanced F1 sample size',
        sample_size_quantity_scope=dict(
            necessary_counts='No information-theoretic necessary count is inferred from the development-data planning table.',
            equation69='Sufficient count for the specified empirical-Bernstein certificate, given population rho and iid incoming question clusters with arbitrary within-question repeat dependence.',
            fixed_balanced_F1='A certificate-specific sufficient planning corollary for independent questions within each fixed stratum, derived from Eq68 and F.1.',
            plugin_inputs='True population rho and variance are unknown. Values substituted here are estimates from repeatedly inspected development questions, not established population upper bounds.',
            interpretation='Numbers such as125000 describe the sufficient count of this conservative certificate with these plug-in assumptions. They are not a lower bound on the question needs of every possible algorithm.'),
        data_access=input_identity,source_sha256=source,
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('certification.py')]},
        n_Monte_Carlo=args.replicates,wall_s=time.perf_counter()-start,new_llm_calls=0,
        fresh400_scores_used=False,sealed200_scores_used=False)
    (args.output/'CERTIFICATION_DEVELOPMENT_POWER.json').write_text(json.dumps(native(report),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',folds=5,rules=20,
        selected_folds=[x['selected_rule'] for x in certs],average_selected=aggregate['selected_rule'],wall_s=report['wall_s'])),flush=True)


if __name__=='__main__':main()
