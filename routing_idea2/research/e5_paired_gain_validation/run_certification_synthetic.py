"""Exact multinomial question-cluster Monte Carlo for manuscript certification."""
from certification import *
from pathlib import Path
import argparse
import hashlib
import json
import time


SEED=20261110
CONFIG=dict(rho=[.02,.1,.3,.6],gain_ratios=[-.5,0.,.1,.5],n=[200,400,1000,3000,10000],
            L=[1,3],dependence=['conditional_independent','perfectly_correlated'],J=[1,5],
            delta=.05,replicates=10000,seed=SEED,default='known exact zero',
            generation_and_gate_repeats_are_new_questions=False,
            finite_family='four equiprobable input strata, common D per question; all/left/right/checker/complement gates')


def cluster_distribution(rho,mu,L,dependence):
    base=np.array([(rho-mu)/2,1-rho,(rho+mu)/2],float)
    if np.any(base<0) or not np.isclose(base.sum(),1):raise ValueError('Invalid ternary distribution')
    if dependence=='perfectly_correlated' or L==1:
        return np.array([-1.,0.,1.]),base
    p=np.array([1.])
    for _ in range(L):p=np.convolve(p,base)
    return np.arange(-L,L+1)/L,p


def joint_cells(rho,delta,L,dependence,J):
    if J==1:
        weights=np.array([1.]);gates=np.ones((1,1));mu=np.array([delta])
    else:
        weights=np.ones(4)/4
        gates=np.array([[1,1,0,1,0],[1,1,0,0,1],[1,0,1,1,0],[1,0,1,0,1]])
        mu=delta*np.array([1.,1.,-1.,-1.])
    probabilities=[];values=[]
    for h,w in enumerate(weights):
        d,p=cluster_distribution(rho,float(mu[h]),L,dependence)
        probabilities.extend(w*p)
        values.extend(d[:,None]*gates[h])
    probabilities=np.asarray(probabilities)
    probabilities/=probabilities.sum()
    values=np.asarray(values)
    truth=(weights[:,None]*gates*mu[:,None]).sum(0)
    second=probabilities@values**2
    return probabilities,values,truth,second


def experiment(rho,ratio,n,L,dependence,J,reps,batch=1000):
    p,values,truth,second=joint_cells(rho,rho*ratio,L,dependence,J)
    # L1 repeats have the same exact distribution under both dependence labels.
    key=[SEED,CONFIG['rho'].index(rho),CONFIG['gain_ratios'].index(ratio),n,L,J,
         0 if L==1 or dependence=='conditional_independent' else 1]
    rng=np.random.default_rng(np.random.SeedSequence(key))
    counts=dict(false_nonpositive_certificate=0,harmful_selected=0,positive_selected=0,
                returned_default=0,simultaneous_noncoverage=0)
    width=0.;selected_gain=0.;radii=0.
    for first in range(0,reps,batch):
        count=rng.multinomial(n,p,size=min(batch,reps-first))
        means=count@values/n
        variance=np.maximum((count@(values**2)-n*means**2)/(n-1),0.)
        cert=certificate_from_moments(means,variance,n,CONFIG['delta'])
        selected=cert['selected']
        gain=np.where(selected>=0,truth[np.maximum(selected,0)],0.)
        counts['false_nonpositive_certificate']+=int(np.sum(np.any((cert['lcb']>0)&(truth<=1e-15),axis=1)))
        counts['harmful_selected']+=int(np.sum(gain < -1e-15))
        counts['positive_selected']+=int(np.sum(gain > 1e-15))
        counts['returned_default']+=int(np.sum(selected<0))
        counts['simultaneous_noncoverage']+=int(np.sum(np.any((truth<cert['lcb'])|(truth>cert['ucb']),axis=1)))
        width+=float(np.sum(2*cert['radius'].mean(1)))
        radii+=float(np.sum(cert['radius'].mean(1)))
        selected_gain+=float(gain.sum())
    results={name:dict(count=k,rate=k/reps,mc_ci95=binomial_interval(k,reps)) for name,k in counts.items()}
    results['positive_detection_power']=results['positive_selected'] if np.any(truth>0) else None
    rho_gate=np.array([rho]) if J==1 else np.array([rho,rho/2,rho/2,rho/2,rho/2])
    return dict(rho=rho,nominal_delta=rho*ratio,gain_ratio=ratio,n=n,L=L,dependence=dependence,J=J,
                replicates=reps,true_gate_gains=truth.tolist(),true_cluster_second_moments=second.tolist(),
                true_gate_disagreement=rho_gate.tolist(),results=results,
                mean_interval_width=width/reps,mean_radius=radii/reps,mean_selected_true_gain=selected_gain/reps,
                sufficient_n=[sufficient_questions(float(r),float(g),J) for r,g in zip(rho_gate,truth)],
                necessary_single_gate_n=necessary_questions(rho,rho*ratio) if J==1 else None,
                necessary_single_gate_n_scope=('single_repetition_two_point_bound' if J==1 and L==1 else
                    'identical_repetition_two_point_bound' if J==1 and dependence=='perfectly_correlated' else
                    'L1_or_arbitrary_within_question_dependence_reference_only' if J==1 else
                    'no_multiple_gate_necessary_bound_asserted'),
                necessary_bound_applies_to_simulated_repetition_class=bool(J==1 and (L==1 or dependence=='perfectly_correlated') and ratio>0),
                always_default=dict(harmful_selection_rate=0.,positive_detection_power=0.,returned_default_rate=1.),
                fixed_n_no_optional_stopping=True,Monte_Carlo_population='iid incoming questions, all gates share each question and gain observation')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'cert_synthetic')
    parser.add_argument('--replicates',type=int,default=10000)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    qualification()
    jobs=[(rho,ratio,n,L,dep,J) for rho in CONFIG['rho'] for ratio in CONFIG['gain_ratios']
          for n in CONFIG['n'] for L in CONFIG['L'] for dep in CONFIG['dependence'] for J in CONFIG['J']]
    if args.smoke:jobs=jobs[:4]
    started=time.perf_counter()
    rows=[]
    for i,job in enumerate(jobs):
        row=experiment(*job,reps=args.replicates)
        rows.append(row)
        if (i+1)%64==0:print(json.dumps(dict(completed=i+1,total=len(jobs))),flush=True)
    report=dict(status='SMOKE' if args.smoke else 'COMPLETE',config={**CONFIG,'replicates':args.replicates},
                qualification=qualification(),rows=rows,wall_s=time.perf_counter()-started,
                necessary_bound_scope=dict(
                    source='Manuscript Theorem3 Eq13, AppendixD Theorem15 Eq70 and discussion on PDF page21',
                    reference_class='Single gate with L=1; the same question-count lower bound applies to a class allowing arbitrary within-question dependence through identical repetitions',
                    conditional_independent_L3='A repeated L=1 value in an L=3 conditional-independent row is only a reference to the larger arbitrary-dependence class. It is not asserted as a necessary bound for this stricter simulated subclass.',
                    prerequisites='Positive gain Delta in (0,rho/2], harmful-selection error at most delta under the negative distribution and detection probability at least1-delta under the positive distribution',
                    no_log_J_lower_bound=True),
                sufficient_bound_scope='Theorem3/15 gives sufficient question counts for its specified empirical-Bernstein certificate under the stated population assumptions. These are not universal necessary sample sizes for every procedure.',
                manuscript_sha256='b2c9c7f9bf09a3183f4907eb7daedd3a7537dc9f6460f62ccfc246e525bb9e99',
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('certification.py')]},
                new_llm_calls=0)
    (args.output/'CERTIFICATION_SYNTHETIC_RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],configurations=len(rows),wall_s=report['wall_s'])),flush=True)


if __name__=='__main__':main()
