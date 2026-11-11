"""Frozen-policy identification on train600 P60 pools and held-out Q60 blocks.

Only the environment and evaluator receive full rewards. The acquisition
planner gets input features, paid pilot rewards, frozen masks and F-only cost
estimates. The returned first-order inclusion probabilities belong to the
actual cost-capped conditional sampling law, not to its Bernoulli proposal.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import json
import hashlib
from time import perf_counter, process_time
import argparse
import numpy as np

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
WORK = ROOT / 'research/e6_contrast_directed_feedback'
OLD = ROOT / 'research/e5_paired_gain_validation'
for folder in [OLD / 'public', WORK / 'public/b', Path(__file__).parent.parent / 'b']:
    sys.path.insert(0, str(folder))
from e5_common import load_train, TEAMS, sha, put
from cost_and_parameters import resources
from contrast_design import optimize_design
from finite_pool_sampling import CappedSubsetDesign, qualification

METHODS = ['Uniform', 'Marginal', 'Contrast', 'GapContrast', 'NoPilotUniform']
BUDGETS = [24,36,48]
PILOTS = 50
DRAWS = 40


def choose(values):
    return np.argmax(values >= values.max(axis=-1, keepdims=True) - 1e-12, axis=-1)


def deduplicate(masks, names):
    kept = []
    mapping = {}
    for j, mask in enumerate(masks):
        match = next((k for k in kept if np.array_equal(mask, masks[k])), None)
        if match is None:
            kept.append(j)
            match = j
        mapping[str(names[j])] = str(names[match])
    return masks[kept].astype(float), [str(names[j]) for j in kept], mapping


def expected_tokens(data, F, base, candidates):
    member = (data['cost']['member_prompt_tokens'][:,F] + data['cost']['member_completion_tokens'][:,F]).mean(axis=(0,1))
    aggregate = (data['cost']['aggregate_prompt_tokens'][:,F] + data['cost']['aggregate_completion_tokens'][:,F]).mean(axis=(0,1))
    tokens = []
    calls = []
    for candidate in candidates:
        actions = sorted({int(base), int(candidate)})
        members = sorted(set().union(*(set(TEAMS[a]) for a in actions)))
        tokens.append(3 * (member[members].sum() + aggregate[actions].sum()))
        calls.append(3 * (len(members) + len(actions)))
    return np.asarray(tokens), np.asarray(calls, dtype=int)


def pooled_moments(masks, pilot, rewards, remaining):
    # Only purchased pilot feedback enters this estimator. Repeats are averaged
    # within questions before computing the finite-pool sampling second moment.
    squares = rewards ** 2
    pooled = float(squares.mean())
    signatures = masks.T.astype(np.int8)
    out = np.empty(len(remaining))
    for index, q in enumerate(remaining):
        same = np.all(signatures[pilot] == signatures[q], axis=1)
        out[index] = (float(squares[same].sum()) + 4 * pooled) / (int(same.sum()) + 4)
    return out


def cap_for(pilot_calls, remaining_calls, B):
    cap = 3 * int(np.floor(7.5 * B))
    m = B - 12
    remaining_cap = cap - int(pilot_calls.sum())
    # Ensure every residual pool question has positive inclusion probability.
    required = max(int(remaining_calls[i] + np.sort(np.delete(remaining_calls, i))[:m-1].sum())
                   for i in range(len(remaining_calls)))
    fallback = remaining_cap < required
    if fallback:
        cap = 24 * B
    return cap, fallback


def make_design(probabilities, calls, count, cap):
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    odds = clipped / (1 - clipped)
    # Every residual pair costs18+3*d calls. The fixed-size common term is
    # removed from the DP cost axis without altering the feasible subset law.
    residual_cost = ((calls - 18) // 3).astype(int)
    assert np.array_equal(calls, 18 + 3 * residual_cost)
    residual_cap = (int(cap) - 18 * count) // 3
    design = CappedSubsetDesign(odds, count, residual_cost, residual_cap)
    if np.any(design.pi <= 0):
        raise RuntimeError('Every remaining pool question must retain positive inclusion probability')
    return design


def plan(masks, pilot, feedback, tokens, calls, B):
    remaining = np.setdiff1d(np.arange(60), pilot)
    m = B - len(pilot)
    cap, fallback = cap_for(calls[pilot], calls[remaining], B)
    residual_cap = cap - int(calls[pilot].sum())
    estimated_a = pooled_moments(masks, pilot, feedback, remaining)
    expected_gains = (masks[:,pilot] * feedback).mean(axis=1)
    pair_indices = [(j,k) for j in range(len(masks)) for k in range(j)]
    difference = np.asarray([(masks[j,remaining] - masks[k,remaining]) ** 2 for j,k in pair_indices])
    C = masks[:,remaining] * estimated_a / 60**2
    pairC = difference * estimated_a / 60**2
    # Presence of exact default and all gates makes both objectives identical.
    assert any(np.all(row == 0) for row in masks)
    assert any(np.all(row == 1) for row in masks)
    assert np.array_equal(C.max(axis=0), pairC.max(axis=0))
    ann_weights = np.ones(len(remaining))
    exec_weights = tokens[remaining] / tokens[remaining].mean()
    p_marginal, diagnostic = optimize_design(C, ann_weights, exec_weights, m, m, offset=-C.sum(axis=1))
    gaps = np.maximum(np.asarray([abs(expected_gains[j] - expected_gains[k]) for j,k in pair_indices]), .01)
    Cgap = pairC / gaps[:,None] ** 2
    p_gap, gap_diagnostic = optimize_design(Cgap, ann_weights, exec_weights, m, m, offset=-Cgap.sum(axis=1))
    proposal = {'Uniform':np.full(len(remaining), m/len(remaining)),
                'Marginal':p_marginal, 'Contrast':p_marginal.copy(), 'GapContrast':p_gap}
    designs = {name:make_design(p, calls[remaining], m, residual_cap) for name,p in proposal.items()}
    metadata = dict(cap=cap, fallback=fallback, diagnostic=diagnostic, gap_diagnostic=gap_diagnostic,
                    conditional_expected_tokens={name:float(tokens[pilot].sum() + d.pi @ tokens[remaining]) for name,d in designs.items()},
                    predicted_token_budget=float(tokens[pilot].sum()+m*tokens[remaining].mean()),
                    estimated_a=estimated_a.tolist(), pilot_gain=expected_gains.tolist(),
                    conditional_expected_calls={name:float(calls[pilot].sum()+d.pi@calls[remaining]) for name,d in designs.items()},
                    proposal_constraint_scope='Expected Bernoulli design. Exact count/call-cap conditioning changes actual propensities; actual pi used for HT.',
                    objective_identity='default and all present => unnormalized marginal and contrast coincide for every p')
    return remaining, designs, proposal, metadata


class FeedbackEnvironment:
    def __init__(self, paired_rewards):
        self.__rewards = np.asarray(paired_rewards)
        self.observed = set()
    def buy(self, indices):
        indices = np.asarray(indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= 60):
            raise ValueError('Feedback purchase escaped the P pool')
        self.observed.update(map(int, indices))
        return self.__rewards[indices].mean(axis=1)


def run_fold(fold):
    start, cpu = perf_counter(), process_time()
    config = json.loads((WORK/'public/E6_RUN_CONFIG.json').read_text())
    freeze = json.loads((WORK/'private/PROTOCOL_FROZEN.json').read_text())
    assert freeze['config_sha256'] == sha(WORK/'public/E6_RUN_CONFIG.json')
    data, references = load_train()
    with np.load(WORK/'private/E6_FGE_PQ_FROZEN.npz') as a:
        F,P,Q,E = (a[f'f{fold}_{key}'] for key in ['F','P','Q','E'])
    policy_path = WORK/f'private/a/fold{fold}_policies.npz'
    with np.load(policy_path) as a:
        assert np.array_equal(E, a['E'])
        masks, names, mapping = deduplicate(a['masks'], a['names'])
        base, candidate = int(a['base']), a['candidate'].copy()
    plocal = np.asarray([int(np.flatnonzero(E==q)[0]) for q in P])
    qlocal = np.asarray([int(np.flatnonzero(E==q)[0]) for q in Q])
    masksP, masksQ = masks[:,plocal], masks[:,qlocal]
    candidatesP = candidate[plocal]
    tokens, calls = expected_tokens(data, F, base, candidatesP)
    costs = [resources(data,[q],[[base,int(c)]],gold=1) for q,c in zip(P,candidatesP)]
    costnames = list(costs[0])
    costmatrix = np.asarray([[cost[k] for k in costnames] for cost in costs])
    assert np.all(costmatrix[:,costnames.index('member_calls')]+costmatrix[:,costnames.index('aggregator_calls')]==calls)
    pool_folder = WORK / 'private/c'
    pool_folder.mkdir(parents=True, exist_ok=True)
    # Reward-independent identities and all P/Q masks are persisted first.
    put(pool_folder/f'FROZEN_C_f{fold}.json',dict(fold=fold,policy_sha256=sha(policy_path),
        protocol_sha256=sha(WORK/'private/PROTOCOL_FROZEN.json'),split_sha256=sha(WORK/'private/E6_FGE_PQ_FROZEN.npz'),
        source_sha256=sha(__file__),names=names,dedup_mapping=mapping,P=60,Q=60,
        outcome_roles='P feedback only through environment.buy, Q evaluation after all decisions',
        proposal_token_constraint='F-only expected residual tokens <= SRS expected tokens at same residual gold count',
        main_sampling_law='exact fixed-size/call-cap conditional Bernoulli; exact pi used for HT'))
    RbP = data['Y'][P,:,base]
    RcP = data['Y'][P[:,None],np.arange(3)[None,:],candidatesP[:,None]]
    DP = RcP - RbP
    nb,nm,nr = len(BUDGETS),len(METHODS),PILOTS*DRAWS
    selected = np.zeros((nb,nm,nr), dtype=np.int16)
    estimates = np.zeros((nb,nm,nr,len(masks)))
    purchases = np.zeros((nb,nm,nr,60),dtype=bool)
    inclusion = np.zeros((nb,nm,nr,60))
    pilot_ids = np.zeros((PILOTS,12),dtype=np.int16)
    proposed_probabilities = np.zeros((nb,nm,PILOTS,60))
    all_metadata = []
    ann_overrun = np.zeros((nb,4),dtype=int)
    exec_overrun = np.zeros((nb,4),dtype=int)
    bernoulli_var_estimates = []
    for pilot_index in range(PILOTS):
        rng = np.random.default_rng(np.random.SeedSequence([20261112,fold,pilot_index,611]))
        pilot = np.sort(rng.choice(60,12,replace=False))
        pilot_ids[pilot_index] = pilot
        environment = FeedbackEnvironment(DP)
        pilot_feedback = environment.buy(pilot)
        for bi,B in enumerate(BUDGETS):
            remaining, designs, proposals, meta = plan(masksP,pilot,pilot_feedback,tokens,calls,B)
            meta.update(fold=fold,pilot_index=pilot_index,gold_budget=B)
            all_metadata.append(meta)
            no_pilot_cap = 3 * int(np.floor(7.5*B))
            min_inclusion_cap = max(int(calls[i]+np.sort(np.delete(calls,i))[:B-1].sum()) for i in range(60))
            no_pilot_fallback = no_pilot_cap < min_inclusion_cap
            if no_pilot_fallback:
                no_pilot_cap = 24*B
            no_pilot_design = make_design(np.full(60,B/60),calls,B,no_pilot_cap)
            meta['no_pilot_cap'] = no_pilot_cap
            meta['no_pilot_fallback'] = no_pilot_fallback
            for mi,name in enumerate(METHODS[:4]):
                proposed_probabilities[bi,mi,pilot_index,pilot]=1
                proposed_probabilities[bi,mi,pilot_index,remaining]=proposals[name]
            proposed_probabilities[bi,4,pilot_index]=B/60
            for draw in range(DRAWS):
                rep = pilot_index*DRAWS+draw
                draw_rng=np.random.default_rng(np.random.SeedSequence([20261112,fold,pilot_index,bi,draw,612]))
                common=draw_rng.random(61)
                independent=draw_rng.random(48)
                for mi,name in enumerate(METHODS):
                    if name=='NoPilotUniform':
                        chosen=no_pilot_design.sample(common)
                        pi=no_pilot_design.pi
                        feedback=FeedbackEnvironment(DP).buy(np.flatnonzero(chosen))
                        ghat=(masksP[:,chosen]*(feedback/pi[chosen])).sum(axis=1)/60
                        inclusion[bi,mi,rep]=pi
                        cap=no_pilot_cap
                    else:
                        design=designs[name]
                        residual=design.sample(common[:49])
                        ids=remaining[residual]
                        local_env=FeedbackEnvironment(DP)
                        observed=local_env.buy(ids)
                        ghat=((masksP[:,pilot]*pilot_feedback).sum(axis=1)+(masksP[:,ids]*(observed/design.pi[residual])).sum(axis=1))/60
                        chosen=np.zeros(60,dtype=bool)
                        chosen[pilot]=True
                        chosen[ids]=True
                        inclusion[bi,mi,rep,pilot]=1
                        inclusion[bi,mi,rep,remaining]=design.pi
                        cap=meta['cap']
                        bernoulli=independent<proposals[name]
                        ann_overrun[bi,mi]+=int(len(pilot)+bernoulli.sum()>B)
                        exec_overrun[bi,mi]+=int(calls[pilot].sum()+calls[remaining]@bernoulli>cap)
                    estimates[bi,mi,rep]=ghat
                    selected[bi,mi,rep]=choose(ghat)
                    purchases[bi,mi,rep]=chosen
                    assert chosen.sum()==B
                    assert calls@chosen<=cap
                    assert costmatrix[chosen,costnames.index('tokens')].sum()<=159744*B
            if pilot_index==0:
                # Independent Bernoulli conditional variance, kept distinct from
                # the cost-capped subset design used above.
                p=proposals['GapContrast']
                diff=masksP[-1,remaining]-masksP[0,remaining]
                values=DP[remaining].mean(axis=1)*diff/60
                analytic=float(np.sum((1-p)/p*values**2))
                vrng=np.random.default_rng([20261112,fold,bi,619])
                sampled=vrng.random((20000,len(p)))<p
                sampled_est=(sampled/p)@values
                bernoulli_var_estimates.append(dict(budget=B,analytic=analytic,MC_variance=float(sampled_est.var(ddof=1)),
                    MC_reps=20000,scope='independent Bernoulli formula only, no cap conditioning'))
        if pilot_index%10==0:
            print(json.dumps(dict(fold=fold,pilots=pilot_index+1,total=PILOTS,wall_s=perf_counter()-start)),flush=True)
    assert np.array_equal(selected[:,1],selected[:,2])
    assert np.array_equal(estimates[:,1],estimates[:,2])
    # Persist all policy selections, observations and exact inclusion weights
    # before Q rewards are first indexed by the evaluator. The historical
    # loader materializes train600, while planner arguments exclude Q rewards.
    rawpath=pool_folder/f'C_f{fold}.npz'
    np.savez_compressed(rawpath,fold=fold,P=P,Q=Q,E=E,masksP=masksP,masksQ=masksQ,names=np.asarray(names),
        methods=np.asarray(METHODS),budgets=BUDGETS,selected=selected,estimates=estimates,
        purchases=purchases,inclusion=inclusion,pilot_ids=pilot_ids,proposal=proposed_probabilities,
        DP=DP,RbP=RbP,expected_tokens=tokens,calls=calls,costnames=np.asarray(costnames),costmatrix=costmatrix)
    put(pool_folder/f'DECISIONS_C_f{fold}.json',dict(status='FROZEN_BEFORE_Q_EVALUATION',decision_sha256=sha(rawpath),
        policy_sha256=sha(policy_path),shape=list(selected.shape),Q_rewards_used=False))
    RbQ=data['Y'][Q,:,base]
    RcQ=data['Y'][Q[:,None],np.arange(3)[None,:],candidate[qlocal,None]]
    DQ=RcQ-RbQ
    gainsP=(masksP*DP.mean(axis=1)).mean(axis=1)
    gainsQ=(masksQ*DQ.mean(axis=1)).mean(axis=1)
    result=dict(fold=fold,names=names,dedup=mapping,gainsP=gainsP.tolist(),gainsQ=gainsQ.tolist(),
                costs=costnames,methods={},wall_s=perf_counter()-start,cpu_s=process_time()-cpu,
                raw_sha256=sha(rawpath),new_llm_calls=0,Bernoulli_variance_diagnostic=bernoulli_var_estimates,
                Bernoulli_gold_overrun=ann_overrun.tolist(),Bernoulli_call_overrun=exec_overrun.tolist(),
                cap_fallbacks=sum(row['fallback'] for row in all_metadata),no_pilot_cap_fallbacks=sum(row['no_pilot_fallback'] for row in all_metadata))
    evaluated=pool_folder/f'EVALUATED_C_f{fold}.npz'
    np.savez_compressed(evaluated,selected=selected,DP=DP,DQ=DQ,RbQ=RbQ,gainsP=gainsP,gainsQ=gainsQ,Q=Q,
        masksQ=masksQ,masksP=masksP,task=data['task'][Q],sub=data['sub'][Q])
    for bi,B in enumerate(BUDGETS):
        result['methods'][str(B)]={}
        for mi,name in enumerate(METHODS):
            ids=selected[bi,mi]
            ledger=purchases[bi,mi].astype(float)@costmatrix
            result['methods'][str(B)][name]=dict(accuracy=float(RbQ.mean()+gainsQ[ids].mean()),
                gain_default=float(gainsQ[ids].mean()),pool_regret=float(gainsP.max()-gainsP[ids].mean()),
                Q_regret=float(gainsQ.max()-gainsQ[ids].mean()),pool_wrong_rate=float(np.mean(gainsP[ids]<gainsP.max()-1e-12)),
                Q_wrong_rate=float(np.mean(gainsQ[ids]<gainsQ.max()-1e-12)),
                cost_mean=dict(zip(costnames,ledger.mean(axis=0).tolist())),cost_min=dict(zip(costnames,ledger.min(axis=0).tolist())),
                cost_max=dict(zip(costnames,ledger.max(axis=0).tolist())),pilot_gold_fraction=0 if mi==4 else 12/B,
                selected_counts=np.bincount(ids,minlength=len(names)).tolist(),
                seed_gains=(masksQ[ids,:,None]*DQ[None]).mean(axis=(0,1)).tolist())
    put(pool_folder/f'PLANNING_C_f{fold}.json',all_metadata)
    put(pool_folder/f'SUMMARY_C_f{fold}.json',result)
    print(json.dumps(dict(fold=fold,status='COMPLETE',wall_s=perf_counter()-start,main=result['methods']['36'])),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(qualification()),flush=True)
    run_fold(args.fold)
