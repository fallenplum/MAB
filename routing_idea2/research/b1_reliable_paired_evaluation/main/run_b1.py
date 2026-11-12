"""Freeze B1 designs, then independently evaluate the locked finite pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
from scipy.stats import norm

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
BASE = ROOT / 'research/b1_reliable_paired_evaluation'
sys.path[:0] = [str(BASE / 'public/main'), str(BASE / 'public/methods'),
                str(BASE / 'public/inference')]
from b1_design import optimal_probabilities, qualify_optimizer
from public_methods import asi_probabilities, robust_probabilities, scale_to_budgets
from public_methods import tune_asi_tau, tune_robust_radius
from b1_inference import finite_pool_many, exhaustive_design_check


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding='utf-8')
    temp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k].copy() for k in source.files}


def frozen_inputs():
    receipt = json.loads((BASE / 'private/assets/FREEZE_RECEIPT.json').read_text(encoding='utf-8'))
    for name, digest in receipt['files_sha256'].items():
        assert sha(BASE / 'private/assets' / name) == digest
    return load_npz(BASE / 'private/assets/FROZEN_INPUTS.npz'), receipt


def prepare():
    """This entry point never opens validation rewards or execution lengths."""
    started = time.perf_counter()
    a, receipt = frozen_inputs()
    train = load_npz(BASE / 'private/assets/TRAIN_OOF.npz')
    held = train['active']
    folds = []
    for fold in range(5):
        mask = held & (train['fold'] == fold)
        folds.append(dict(predicted_e2=np.clip(train['nested_moment_predictions'][mask, 0], 1e-6, 4),
                          residual=train['residual'][mask]))
    tuning = {}
    designs = {}
    arrays = {}
    active = a['active']
    v, raw, cost = a['vhat'][active], a['rawmom'][active], a['cost_prediction'][active]
    count = int(active.sum())
    budgets = {'primary': (0.25, 0.25), 'execution_tight': (0.50, 0.25),
               'answer_tight': (0.25, 0.50)}
    permutation = np.random.default_rng(2026111304).permutation(count)
    for budget, (answer_rate, execution_rate) in budgets.items():
        ba, be = answer_rate * count, execution_rate * cost.sum()
        tuning[budget] = dict(
            asi=tune_asi_tau(np.sqrt(np.clip(train['nested_moment_predictions'][held, 0], 1e-6, 4)),
                             train['residual'][held], answer_rate),
            robust=tune_robust_radius(folds, answer_rate))
        uniform = scale_to_budgets(np.ones(count), cost, ba, be)
        asi = asi_probabilities(v, cost, ba, be, tau=tuning[budget]['asi']['tau'])
        robust = robust_probabilities(v, cost, ba, be, c_radius=tuning[budget]['robust']['c_radius'])
        t1, t1diag = optimal_probabilities(v, cost, ba, be)
        t0, t0diag = optimal_probabilities(raw, cost, ba, be)
        noexec, noexecdiag = optimal_probabilities(v, cost, ba, cost.sum())
        permuted, permdiag = optimal_probabilities(v[permutation], cost, ba, be)
        candidates = {
            'U0': (uniform['p'], uniform, True), 'U1': (uniform['p'], uniform, False),
            'A1_ASI': (asi['p'], asi, False), 'A2_Robust_ASI': (robust['p'], robust, False),
            'T0_RawMoment': (t0, t0diag, False), 'T1': (t1, t1diag, False),
            'T1_no_execution_constraint': (noexec, noexecdiag, False),
            'T1_m0': (t0, t0diag, True), 'T1_permuted_moment': (permuted, permdiag, False),
        }
        for name, (p_active, diagnostic, zero_auxiliary) in candidates.items():
            key = budget + '__' + name
            p = np.zeros(len(active)); p[active] = p_active
            m = np.zeros(len(active)) if zero_auxiliary else a['m'].copy()
            m[~active] = 0
            arrays[key + '__p'] = p
            arrays[key + '__m'] = m
            diagnostic = {k: value for k, value in diagnostic.items() if k not in ('p', 'native_single_budget_p')}
            designs[key] = dict(method=name, budget=budget, expected_answers=float(p.sum()),
                expected_tokens=float(p @ a['cost_prediction']), answer_cap=float(ba),
                predicted_token_cap=float(be), floor=float(p_active.min()), ceiling=float(p_active.max()),
                zero_auxiliary=zero_auxiliary,
                cost_constraint_removed=name == 'T1_no_execution_constraint', diagnostic=diagnostic)
        for label, method in [('ASI', asi), ('Robust_ASI', robust)]:
            arrays[budget + '__native_' + label + '__p'] = method['native_single_budget_p']
    qualification = dict(optimizer=qualify_optimizer(), estimator=exhaustive_design_check())
    put(BASE / 'public/NUMERICAL_QUALIFICATION.json', qualification)
    put(BASE / 'public/HISTORICAL_METHOD_TUNING.json', tuning)
    probability_path = BASE / 'private/design/FROZEN_DESIGNS.npz'
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probability_path, **arrays)
    put(BASE / 'public/DESIGN_SUMMARY.json', designs)
    protocol = dict(
        schema='MAB_B1_frozen_paired_evaluation_v1', status='LOCKED_BEFORE_VALIDATION_REWARD_ACCESS',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        base_branch='codex/ces-traceable-baseline-20261112',
        base_commit='a701eabf0720b834d584c3f47ccd8e82fdcb4440',
        branch='codex/b1-reliable-paired-evaluation',
        target='one frozen candidate minus one frozen global default, question-mean of three repeats',
        finite_pool=dict(role='original V2 selector_validation', questions=200,
            target='unweighted fixed-pool mean across 200 original questions',
            repeats=[42,137,2024], prior_development_use=True, new_independent_test=False,
            active_questions=count, structural_zero_questions=int((~active).sum())),
        router=receipt['router'], default=receipt['default'],
        auxiliary='clipped original frozen direct20 score difference, zero on identical actions',
        nuisance='fixed alpha100 Ridge on history-only OOF paired moments; nested history-only diagnostic',
        sampling=dict(design='independent Bernoulli', pmin=0.05,
            budgets='expected answer and historical-predicted token constraints',
            trials=2000, seed=2026111305, common_random_numbers=True),
        budgets=budgets, methods=list(candidates),
        inference=dict(finite_primary='independent-design variance upper confidence bound plus Bernstein',
            finite_secondary='design-correct unbiased Vhat plus asymptotic Wald',
            population='iid question-cluster experiments with known synthetic truth only',
            adoption='single fixed-time one-sided alpha .05, thresholds 0 and .01',
            no_label='deterministic feasible wide interval; no zero-SE narrow fallback'),
        primary_comparisons=['T1 vs U1', 'T1 vs A1_ASI', 'T1 vs A2_Robust_ASI'],
        multiplicity='Holm on three paired Monte Carlo MSE-difference tests; MC precision only',
        inference_claim_scope='fixed-pool coverage is conditional sampling-design coverage, never future-population coverage',
        external_native='source-traced full method examples and clearly named adaptations, separate estimator/inference table',
        new_physical_llm_calls=0, fresh400_outcomes_permitted=False, sealed200_outcomes_permitted=False,
        freeze_hashes=dict(assets=sha(BASE / 'private/assets/FROZEN_INPUTS.npz'),
            probability_designs=sha(probability_path), method_tuning=sha(BASE / 'public/HISTORICAL_METHOD_TUNING.json')),
        code_hashes={str(path.relative_to(BASE)):sha(path) for path in sorted((BASE/'public').rglob('*.py'))},
        preparation_cpu_wall_seconds=time.perf_counter()-started,
        separation='design process exits before evaluation process opens validation rewards; application/file boundary, no OS security claim')
    put(BASE / 'public/B1_PROTOCOL.json', protocol)
    put(BASE / 'private/design/DESIGN_FREEZE_RECEIPT.json', dict(protocol_sha256=sha(BASE/'public/B1_PROTOCOL.json'),
        probability_sha256=sha(probability_path), validation_outcomes_opened=False,
        fresh400_or_sealed200_outcomes_opened=False, status='DESIGN_FROZEN'))
    print(json.dumps(dict(status=protocol['status'], active=count, designs=len(designs),
        protocol_sha256=sha(BASE/'public/B1_PROTOCOL.json'), seconds=time.perf_counter()-started)), flush=True)


def evaluator_truth(a):
    """Privileged truth/cost reader. No returned outcome enters a design function."""
    freeze = json.loads((BASE/'private/design/DESIGN_FREEZE_RECEIPT.json').read_text())
    assert sha(BASE/'public/B1_PROTOCOL.json') == freeze['protocol_sha256']
    assert sha(BASE/'private/design/FROZEN_DESIGNS.npz') == freeze['probability_sha256']
    receipt = json.loads((BASE/'private/assets/FREEZE_RECEIPT.json').read_text(encoding='utf-8'))
    record = receipt['data_lineage']['validation_outcomes_metadata_only']
    outcome_path = Path(record['path'])
    assert sha(outcome_path) == record['sha256']
    out = load_npz(outcome_path)
    assert np.array_equal(out['question_ids'], a['val_ids'])
    assert np.array_equal(out['canonical_hashes'], a['canonical_hashes'])
    assert np.array_equal(out['generation_seeds'], [42,137,2024])
    y = out['aggregate_scores']
    assert y.shape == (3, 200, 20) and np.isin(y, [0,1]).all()
    rows = np.arange(200)
    rb, rc = y[:, rows, a['b']], y[:, rows, a['c']]
    D, d = rc.astype(float)-rb, (rc.astype(float)-rb).mean(axis=0)
    assert np.all(d[~a['active']] == 0)
    ledgers = {name: np.zeros(200) for name in ('gold','member_calls','aggregator_calls','scoring_ops',
        'input_tokens','output_tokens','tokens','source_model_wall_s','no_reuse_tokens','no_reuse_member_calls')}
    for row in np.flatnonzero(a['active']):
        teams = a['teams']; actions = [int(a['b'][row]), int(a['c'][row])]
        members = np.unique(teams[actions].flatten())
        ledgers['gold'][row] = 1
        for prefix, indices, call_key in [('member',members,'member_calls'), ('aggregate',actions,'aggregator_calls')]:
            ledgers[call_key][row] += out[prefix+'_execution_counts'][:,row,indices].sum()
            ledgers['input_tokens'][row] += out[prefix+'_prompt_tokens'][:,row,indices].sum()
            ledgers['output_tokens'][row] += out[prefix+'_completion_tokens'][:,row,indices].sum()
            ledgers['source_model_wall_s'][row] += out[prefix+'_wall_time_s'][:,row,indices].sum()
        ledgers['scoring_ops'][row] = 6
        ledgers['tokens'][row] = ledgers['input_tokens'][row] + ledgers['output_tokens'][row]
        duplicate = set(teams[actions[0]]) & set(teams[actions[1]])
        ledgers['no_reuse_member_calls'][row] = 18
        ledgers['no_reuse_tokens'][row] = ledgers['tokens'][row] + sum(
            out['member_prompt_tokens'][:,row,j].sum()+out['member_completion_tokens'][:,row,j].sum() for j in duplicate)
        assert ledgers['member_calls'][row] == 3*len(members)
        assert ledgers['aggregator_calls'][row] == 6
    directory = BASE/'private/evaluator'; directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'ORACLE_PAIRED.npz', d=d, D=D, rb=rb, rc=rc, **ledgers)
    put(directory/'PRIVILEGED_ACCESS_RECEIPT.json', dict(opened_at_utc=datetime.now(timezone.utc).isoformat(),
        role='independent fixed-pool evaluation and retrospective cost accounting',
        protocol_sha256=freeze['protocol_sha256'], validation_outcome_sha256=record['sha256'],
        permitted_reward_payloads=['selector_validation'],
        fresh400_reward_payloads_opened=0, sealed200_reward_payloads_opened=0,
        design_process_receives_current_correctness_or_lengths=False,
        reward_use=['G_N ground truth','post-purchase feedback and resource summaries','estimator error and coverage'],
        source_model_wall_time_is_historical_wall_proxy_not_measured_new_gpu_time=True))
    return d,D,rb,rc,ledgers


def distribution(x):
    x=np.asarray(x)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)),
                minimum=float(x.min()), q05=float(np.quantile(x,.05)),
                median=float(np.median(x)), q95=float(np.quantile(x,.95)),maximum=float(x.max()))


def wilson(count,total):
    z=norm.ppf(.975); rate=count/total; denominator=1+z*z/total
    center=(rate+z*z/(2*total))/denominator
    half=z*np.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/denominator
    return [float(max(0,center-half)),float(min(1,center+half))]


def inference_summary(bounds,truth):
    result={}
    for label in ['finite_bernstein','asymptotic_wald']:
        lo,hi=bounds[label+'_lower'],bounds[label+'_upper']
        cover=(lo<=truth)&(hi>=truth)
        one_lo,one_hi=bounds[label+'_one_lower'],bounds[label+'_one_upper']
        result[label]=dict(coverage=float(cover.mean()),coverage_MC95=wilson(int(cover.sum()),len(cover)),
            width=distribution(hi-lo), thresholds={})
        for threshold in (0.,.01):
            adopt=one_lo>threshold; below=one_hi<=threshold
            result[label]['thresholds'][str(threshold)]=dict(adopt=float(adopt.mean()),
                below_threshold=float(below.mean()),insufficient=float((~adopt & ~below).mean()),
                false_adoption_when_null=float(adopt.mean()) if truth<=threshold else None)
    return result


def evaluate():
    started=time.perf_counter()
    a,receipt=frozen_inputs()
    designs=json.loads((BASE/'public/DESIGN_SUMMARY.json').read_text(encoding='utf-8'))
    probabilities=load_npz(BASE/'private/design/FROZEN_DESIGNS.npz')
    d,D,rb,rc,cost=evaluator_truth(a)
    truth=float(d.mean()); repeats=2000; n=len(d)
    rng=np.random.default_rng(2026111305)
    uniforms=rng.random((repeats,n))
    results={}; snapshots={}; squared_errors={}
    for key,design in designs.items():
        p=probabilities[key+'__p']; m=probabilities[key+'__m']
        A=uniforms<p; residual=d-m
        inv=np.divide(1.,p,out=np.zeros_like(p),where=p>0)
        estimate=(m.sum()+A@(residual*inv))/n
        variance=(A@((1-p)*inv**2*residual**2))/n**2
        count=A.sum(axis=1); observed=A@d
        bounds=finite_pool_many(estimate,variance,p,m,count,observed)
        err=estimate-truth
        exact_variance=float(np.sum((1-p)*inv*residual**2)/n**2)
        ledger={name:distribution(A@values) for name,values in cost.items()}
        ledger['predicted_tokens']=distribution(A@a['cost_prediction'])
        actual_token=A@cost['tokens']; predicted_token=A@a['cost_prediction']
        results[key]=dict(**design, truth=truth, bias=float(err.mean()),MSE=float(np.mean(err**2)),
            RMSE=float(np.sqrt(np.mean(err**2))), exact_design_variance=exact_variance,
            exact_design_RMSE=float(np.sqrt(exact_variance)), MC_standard_error_of_bias=float(err.std(ddof=1)/np.sqrt(repeats)),
            estimated_variance_average=float(variance.mean()), inference=inference_summary(bounds,truth),
            resources=ledger, zero_label_rate=float((count==0).mean()),
            expected_budget_exceedance=dict(answer=float((count>design['answer_cap']+1e-9).mean()),
                predicted_tokens=float((predicted_token>design['predicted_token_cap']+1e-9).mean()),
                actual_tokens_vs_predicted_budget=float((actual_token>design['predicted_token_cap']+1e-9).mean())))
        squared_errors[key]=err**2
        snapshot=dict(A=A,p=p,m=m,estimate=estimate,variance_estimate=variance,observed_gain_sum=observed,
            actual_tokens=actual_token,predicted_tokens=predicted_token,**bounds)
        target=BASE/'private/replays'/ (key+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,**snapshot)
        snapshots[key]=dict(path=str(target.relative_to(BASE)),sha256=sha(target),
            draws=repeats,question_clusters=n,authorized_reveal_count=int(A.sum()),
            outcome_payloads_available_to_design=0)
        print(json.dumps(dict(cell=key,RMSE=results[key]['RMSE'],
            wald_coverage=results[key]['inference']['asymptotic_wald']['coverage'],
            valid_coverage=results[key]['inference']['finite_bernstein']['coverage'])),flush=True)
    comparisons=[]
    for baseline in ['U1','A1_ASI','A2_Robust_ASI']:
        delta=squared_errors['primary__T1']-squared_errors['primary__'+baseline]
        se=float(delta.std(ddof=1)/np.sqrt(repeats));mean=float(delta.mean())
        pvalue=float(2*norm.sf(abs(mean/se))) if se>0 else float(mean==0)
        comparisons.append(dict(candidate='T1',baseline=baseline,target='paired Monte Carlo MSE difference at fixed pool',
            delta_MSE=mean,MC95=[mean-1.96*se,mean+1.96*se],p_raw=pvalue,
            exact_variance_difference=results['primary__T1']['exact_design_variance']-results['primary__'+baseline]['exact_design_variance']))
    order=np.argsort([x['p_raw'] for x in comparisons]);running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(order)-rank)*comparisons[index]['p_raw']))
        comparisons[index]['p_Holm']=running
    groups={}
    for label,mask in [('all',np.ones(n,dtype=bool)),('gsm8k',a['task']=='gsm8k'),('bbh',a['task']=='bbh')]:
        groups[label]=dict(questions=int(mask.sum()),default_accuracy=float(rb[:,mask].mean()),
            candidate_accuracy=float(rc[:,mask].mean()),gain=float(d[mask].mean()),
            per_generation_gain=D[:,mask].mean(axis=1).tolist(),
            changes_wrong_to_right=int(np.sum((rb[:,mask]==0)&(rc[:,mask]==1))),
            changes_right_to_wrong=int(np.sum((rb[:,mask]==1)&(rc[:,mask]==0))))
    subgroups={}
    for sub in sorted(set(a['sub'][a['task']=='bbh'])):
        mask=a['sub']==sub
        subgroups[str(sub)]=dict(questions=int(mask.sum()),gain=float(d[mask].mean()))
    complete=dict(status='COMPLETE_REAL_FIXED_POOL',target='finite pool of 200 old-development original questions',
        trials_per_cell=repeats,cells=len(results),unique_question_clusters=200,ground_truth=truth,
        structural_zeros=int((~a['active']).sum()),groups=groups,bbh_subtasks=subgroups,
        bbh_subtask_macro_gain=float(np.mean([x['gain'] for x in subgroups.values()])),
        methods=results,primary_comparisons=comparisons,
        actual_full_active_costs={name:float(values.sum()) for name,values in cost.items()},
        history_common_cost=dict(gold=600,member_calls=10800,aggregator_calls=36000,tokens=37274127,
            source='shared frozen full-information train600 cache; sunk acquisition, separately accounted'),
        zero_new_physical_LLM_calls=True,wall_seconds=time.perf_counter()-started)
    put(BASE/'public/REAL_FIXED_POOL_RESULTS.json',complete)
    put(BASE/'private/replays/REVEAL_AND_COST_MANIFEST.json',snapshots)
    put(BASE/'public/REPLAY_PERMISSION_AUDIT.json',dict(
        status='PASS',design_frozen_before_outcomes=True,probabilities_not_retrained_after_reveal=True,
        designs=len(results),trials_per_design=repeats,
        private_purchase_matrices=True,question_cluster_repeats=3,
        independent_truth_process_access_logged=True,fresh400_or_sealed200_opened=False,
        expected_budgets_not_hard_caps=True,cache_reuse_identical_for_all_methods=True))
    print(json.dumps(dict(status=complete['status'],truth=truth,seconds=complete['wall_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','evaluate'])
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:evaluate()
