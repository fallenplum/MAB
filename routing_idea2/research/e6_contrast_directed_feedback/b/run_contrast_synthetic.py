"""E6-B paired policy-identification replay on the exact E5 synthetic worlds.

Original E5 multinomial question-world counts are regenerated and verified
against archived NPZ files. New propensities refine each existing common-U
cell by conditional binomial splits, preserving every original realization.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pathlib import Path
import sys
import json
import hashlib
import time
import argparse
import numpy as np

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
E5 = ROOT / 'research/e5_paired_gain_validation'
sys.path.append(str(E5 / 'public'))
import run_allocation_synthetic as old
from run_certification_synthetic import cluster_distribution
from contrast_design import optimize_design, qualification

OUTPUT = ROOT / 'research/e6_contrast_directed_feedback'
METHODS = ['Uniform', 'Estimated-Marginal', 'Estimated-Contrast', 'Estimated-GapContrast',
           'Oracle-Marginal', 'Oracle-Contrast', 'Oracle-GapContrast', 'NoPilot-Uniform']
PILOTS = [30, 5]
BUDGETS = [(.25, .5), (.5, .25), (.5, .5)]
SEED = 20261112
EPSILON = .01
TOLERANCE = 1e-12


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    return value


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(native(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def candidate_gates(family):
    original = old.FAMILIES[family]
    raw = np.vstack((np.zeros(7), original, np.ones(7)))
    unique = []
    mapping = []
    # The last inactive stratum has identically zero rewards. All-switch is
    # retained as a literal all-one policy, even though it needs no feedback there.
    for gate in raw:
        match = next((j for j, item in enumerate(unique) if np.array_equal(item[:6], gate[:6])), None)
        if match is None:
            match = len(unique)
            unique.append(gate)
        elif np.all(gate == 1):
            unique[match] = gate
        mapping.append(match)
    return np.asarray(unique), dict(raw_to_unique=mapping, original_gate_count=len(original),
                                   final_gate_count=len(unique), inactive_stratum_rewards_identically_zero=True,
                                   all_switch_was_added_to_original_family=family == 'nonnested')


def pilot(family, count):
    _, _, _, raw30 = old.conditional_pilot(family)
    a = np.zeros(7)
    mu = np.zeros(7)
    raw = {}
    for h in range(6):
        draws = np.asarray(raw30[str(h)]['generation_gains'])[:count]
        averages = draws.mean(1)
        a[h] = np.mean(averages ** 2)
        mu[h] = averages.mean()
        raw[str(h)] = draws
    mh = np.r_[np.full(6, count), 0]
    cost = dict(gold=int(mh.sum()), execution=float(mh @ old.COSTS),
                member_calls=int(old.L * (mh * old.UNION_MEMBERS).sum()),
                aggregate_calls=int(2 * old.L * mh.sum()),
                expected_input_screening_draws=float(np.sum(count / old.WEIGHTS[:6])),
                conditional_stratum_pilot=True)
    return a, mu, cost, raw


def policy_pairs(gates):
    return np.asarray([(j, k) for j in range(len(gates)) for k in range(j + 1, len(gates))], dtype=int)


def all_designs(family, annotation_fraction, execution_fraction, count):
    gates, mapping = candidate_gates(family)
    pairs = policy_pairs(gates)
    delta_gate = gates[pairs[:, 0]] - gates[pairs[:, 1]]
    ah, muh, pilot_cost, pilot_raw = pilot(family, count)
    truth = gates @ (old.WEIGHTS * old.MU)
    gainhat = gates @ (old.WEIGHTS * muh)
    ann = float(annotation_fraction * old.WEIGHTS[:6].sum())
    exe = float(execution_fraction * (old.WEIGHTS * old.COSTS)[:6].sum())
    probabilities = {}
    detail = {}
    original_gates = old.FAMILIES[family]
    probabilities['Uniform'] = old.priority_design(old.WEIGHTS, old.COSTS, original_gates,
                                                   np.ones(7), ann, exe)
    for label, a, gain in [('Estimated', ah, gainhat), ('Oracle', old.true_moments(), truth)]:
        # The default/all pair attains the pointwise bound. Therefore both
        # unnormalized objectives are the same single row, for every p.
        C = (old.WEIGHTS * a)[None, :6]
        p, diag = optimize_design(C, old.WEIGHTS[:6],
                                 (old.WEIGHTS * old.COSTS)[:6], ann, exe)
        fullp = np.r_[p, 0.]
        probabilities[label + '-Marginal'] = fullp
        probabilities[label + '-Contrast'] = fullp.copy()
        detail[label + '-Marginal'] = diag
        detail[label + '-Contrast'] = {**diag, 'same_solution_reused_due_to_exact_objective_identity': True}
        gaps = np.maximum(np.abs(gain[pairs[:, 0]] - gain[pairs[:, 1]]), EPSILON)
        Cgap = delta_gate[:, :6] ** 2 * (old.WEIGHTS * a)[None, :6] / gaps[:, None] ** 2
        pgap, dg = optimize_design(Cgap, old.WEIGHTS[:6],
                                  (old.WEIGHTS * old.COSTS)[:6], ann, exe)
        probabilities[label + '-GapContrast'] = np.r_[pgap, 0.]
        detail[label + '-GapContrast'] = {**dg, 'pair_gaps': gaps.tolist()}
    probabilities['NoPilot-Uniform'] = old.priority_design(
        old.WEIGHTS, old.COSTS, original_gates, np.ones(7),
        ann + pilot_cost['gold'] / old.T, exe + pilot_cost['execution'] / old.T)
    # Numerical pointwise identity qualification includes random feasible or
    # infeasible p, since the algebraic identity does not require a budget.
    rng = np.random.default_rng([SEED, count, list(old.FAMILIES).index(family), 31])
    identity_error = 0.
    for _ in range(200):
        p = rng.uniform(.05, 1., 6)
        moment = old.WEIGHTS[:6] * ah[:6] / p
        marginal = np.max(gates[:, :6] @ moment)
        contrast = np.max(delta_gate[:, :6] ** 2 @ moment)
        identity_error = max(identity_error, abs(marginal - contrast), abs(marginal - moment.sum()))
    return probabilities, dict(family=family, pilot_per_stratum=count,
        annotation_fraction=annotation_fraction, execution_fraction=execution_fraction,
        phase_annotation=ann, phase_execution=exe, pilot_cost=pilot_cost,
        phase_plus_pilot_budget_gold=old.T * ann + pilot_cost['gold'],
        phase_plus_pilot_budget_execution=old.T * exe + pilot_cost['execution'],
        estimated_moments=ah, estimated_conditional_gains=muh, estimated_policy_gains=gainhat,
        true_policy_gains=truth, gates=gates, mapping=mapping, pairs=pairs,
        solver=detail, exact_marginal_contrast_identity_max_error=identity_error,
        probabilities=probabilities), pilot_raw


def original_cells(family, probabilities):
    gates = old.FAMILIES[family]
    original_p = np.asarray([probabilities[name] for name in old.METHODS])
    cells = []
    for h, weight in enumerate(old.WEIGHTS):
        if h == 6:
            cells.append((h, 0., 0., 1., weight))
            continue
        support, law = cluster_distribution(old.RHO[h], old.MU[h], old.L, 'conditional_independent')
        edges = np.unique(np.r_[0., original_p[:, h], 1.])
        for lo, hi in zip(edges[:-1], edges[1:]):
            for d, probability in zip(support, law):
                if probability and hi > lo:
                    cells.append((h, d, lo, hi, weight * probability * (hi - lo)))
    return cells


def refine_original_world(family, ann, exe, all_probabilities, gates, reps):
    original_probabilities, original_meta, _ = old.designs(family, ann, exe)
    joint, old_values, old_resources = old.shared_world_cells(old.FAMILIES[family], original_probabilities)
    rng = np.random.default_rng(np.random.SeedSequence(
        [old.SEED, 2, list(old.FAMILIES).index(family), int(ann * 100), int(exe * 100)]))
    counts = rng.multinomial(old.T, joint, size=reps)
    legacy_estimates = (counts @ old_values.reshape(len(joint), -1) / old.T).reshape(reps, len(old.METHODS), -1)
    legacy_costs = (counts @ old_resources.reshape(len(joint), -1)).reshape(reps, len(old.METHODS), 5)
    name = f'{family}_ann{int(ann*100)}_exec{int(exe*100)}'
    archived = E5 / 'private/synthetic_allocation/cases' / (name + '.npz')
    with np.load(archived) as cached:
        estimate_error = float(np.max(np.abs(legacy_estimates - cached['estimates'][:reps])))
        cost_error = float(np.max(np.abs(legacy_costs - cached['realized_resources'][:reps])))
    assert estimate_error < 1e-12 and cost_error < 1e-10
    cells = original_cells(family, original_probabilities)
    probability_check = np.asarray([v[4] for v in cells])
    probability_check /= probability_check.sum()
    assert np.max(np.abs(probability_check - joint)) < 1e-15
    p = np.asarray(all_probabilities)
    M, J = len(p), len(gates)
    estimates = np.zeros((reps, M, J))
    resources = np.zeros((reps, M, 5))
    rng = np.random.default_rng([SEED, 6, list(old.FAMILIES).index(family), int(ann * 100), int(exe * 100)])
    for index, (h, d, lo, hi, _) in enumerate(cells):
        if h == 6:
            continue
        edges = np.unique(np.r_[lo, p[:, h][(p[:, h] > lo) & (p[:, h] < hi)], hi])
        remaining = counts[:, index].copy()
        for left, right in zip(edges[:-1], edges[1:]):
            if right == hi:
                draws = remaining
            else:
                draws = rng.binomial(remaining, (right - left) / (hi - left))
            remaining = remaining - draws
            acquire = (left + right) / 2 < p[:, h]
            contribution = np.zeros((M, J))
            contribution[acquire] = d * gates[:, h] / p[acquire, h, None]
            estimates += draws[:, None, None] * contribution[None] / old.T
            resource = np.zeros((M, 5))
            resource[acquire] = np.array([1., old.COSTS[h], old.L * old.UNION_MEMBERS[h],
                                         2 * old.L, 2 * old.L])
            resources += draws[:, None, None] * resource[None]
    return estimates, resources, dict(archive=str(archived.relative_to(ROOT)),
        archive_sha256=hashlib.sha256(archived.read_bytes()).hexdigest(),
        original_estimate_max_abs_error=estimate_error, original_resource_max_abs_error=cost_error,
        original_world_reused_exactly=True,
        method='conditional multinomial refinement of the archived E5 common-U cells',
        original_candidate_family_and_methods=old.METHODS, original_meta_budgets=[original_meta['total_gold_budget'], original_meta['total_execution_budget']])


def select(estimates):
    maximum = estimates.max(axis=-1, keepdims=True)
    # Default is first. The first policy within 1e-12 of the maximum wins.
    return np.argmax(estimates >= maximum - TOLERANCE, axis=-1)


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return dict(mean=mean, mc_standard_error=se, paired_mc_ci95=[mean - 1.96 * se, mean + 1.96 * se])


def summarize_case(meta, estimates, resources, reps):
    gates = np.asarray(meta['gates'])
    pairs = np.asarray(meta['pairs'])
    contrast = gates[pairs[:, 0]] - gates[pairs[:, 1]]
    truth = np.asarray(meta['true_policy_gains'])
    optimum = float(truth.max())
    true_a = old.true_moments()
    selected = select(estimates)
    chosen = truth[selected]
    regret = optimum - chosen
    mismatch = regret > TOLERANCE
    mistakes = regret > EPSILON + TOLERANCE
    squared = (estimates - truth) ** 2
    diffs = estimates[:, :, pairs[:, 0]] - estimates[:, :, pairs[:, 1]]
    exact_difference = truth[pairs[:, 0]] - truth[pairs[:, 1]]
    diff_sq = (diffs - exact_difference) ** 2
    result = {}
    for m, method in enumerate(METHODS):
        p = np.asarray(meta['probabilities'][method])
        fmarginal = gates[:, :6] @ (old.WEIGHTS[:6] * true_a[:6] / p[:6])
        fcontrast = contrast[:, :6] ** 2 @ (old.WEIGHTS[:6] * true_a[:6] / p[:6])
        variances = (fcontrast - exact_difference ** 2) / old.T
        phase_cov = ((gates[:, :6] * (old.WEIGHTS[:6] * true_a[:6] / p[:6])) @ gates[:, :6].T - np.outer(truth, truth)) / old.T
        covariance_empirical = np.cov(estimates[:, m], rowvar=False, ddof=1)
        pilot_cost = dict(gold=0, execution=0., member_calls=0, aggregate_calls=0) if method == 'NoPilot-Uniform' else meta['pilot_cost']
        actual = resources[:, m]
        phase_ann = float(old.WEIGHTS @ p)
        phase_exec = float((old.WEIGHTS * old.COSTS) @ p)
        result[method] = dict(
            oracle_diagnostic=method.startswith('Oracle'), probabilities=p,
            policy_selection= dict(nonoptimal_probability=float(mismatch[:, m].mean()),
                mean_value_regret=float(regret[:, m].mean()), probability_regret_above_epsilon=float(mistakes[:, m].mean()),
                mean_chosen_gain=float(chosen[:, m].mean()), oracle_gain=optimum,
                default_selection_rate=float(np.mean(selected[:, m] == 0)),
                harmful_selection_rate=float(np.mean(chosen[:, m] < -TOLERANCE)),
                selected_counts=np.bincount(selected[:, m], minlength=len(gates))),
            worst_marginal_second_moment=float(fmarginal.max()),
            worst_contrast_second_moment=float(fcontrast.max()),
            marginal_mse=squared[:, m].mean(0), contrast_mse=diff_sq[:, m].mean(0),
            worst_marginal_mse=float(squared[:, m].mean(0).max()),
            worst_contrast_mse=float(diff_sq[:, m].mean(0).max()),
            exact_contrast_variance=variances, empirical_covariance=covariance_empirical,
            theoretical_covariance=phase_cov,
            max_covariance_MC_error=float(np.max(np.abs(covariance_empirical - phase_cov))),
            expected_cost=dict(gold=old.T * phase_ann + pilot_cost['gold'],
                execution=old.T * phase_exec + pilot_cost['execution'],
                gold_budget=meta['phase_plus_pilot_budget_gold'], execution_budget=meta['phase_plus_pilot_budget_execution']),
            actual_cost=dict(mean_gold=float(actual[:, 0].mean() + pilot_cost['gold']),
                mean_execution=float(actual[:, 1].mean() + pilot_cost['execution']),
                mean_members=float(actual[:, 2].mean() + pilot_cost['member_calls']),
                mean_aggregates=float(actual[:, 3].mean() + pilot_cost['aggregate_calls']),
                gold_overflow_probability=float(np.mean(actual[:, 0] + pilot_cost['gold'] > meta['phase_plus_pilot_budget_gold'] + 1e-9)),
                execution_overflow_probability=float(np.mean(actual[:, 1] + pilot_cost['execution'] > meta['phase_plus_pilot_budget_execution'] + 1e-9)),
                expected_not_hard_caps=True, new_physical_llm_calls=0),
            paired_vs_uniform=dict(regret=paired_interval(regret[:, m] - regret[:, 0]),
                nonoptimal_probability=paired_interval(mismatch[:, m].astype(float) - mismatch[:, 0]),
                regret_above_epsilon=paired_interval(mistakes[:, m].astype(float) - mistakes[:, 0]),
                mean_contrast_mse=paired_interval(diff_sq[:, m].mean(1) - diff_sq[:, 0].mean(1))))
    contrasts = {}
    for left, right in [('Estimated-Contrast', 'Estimated-Marginal'), ('Estimated-Contrast', 'Uniform'),
                        ('Estimated-GapContrast', 'Estimated-Marginal'), ('Estimated-GapContrast', 'Uniform')]:
        i, j = METHODS.index(left), METHODS.index(right)
        contrasts[left + '_minus_' + right] = dict(
            regret=paired_interval(regret[:, i] - regret[:, j]),
            nonoptimal=paired_interval(mismatch[:, i].astype(float) - mismatch[:, j]),
            regret_above_epsilon=paired_interval(mistakes[:, i].astype(float) - mistakes[:, j]))
    assert np.array_equal(estimates[:, METHODS.index('Estimated-Marginal')], estimates[:, METHODS.index('Estimated-Contrast')])
    assert np.array_equal(estimates[:, METHODS.index('Oracle-Marginal')], estimates[:, METHODS.index('Oracle-Contrast')])
    report = dict(**meta, status='COMPLETE', methods=result, paired_comparisons=contrasts,
        replicates=reps, population_scope='Original E5 iid mixture, 400 incoming questions and conditional independent L=3 repeats',
        shared_feedback_covariance_retained=True, marginal_contrast_estimates_bitwise_equal=True,
        inference='Paired Monte Carlo standard errors conditional on the one frozen pilot; simulations are not new real questions',
        pilot_estimator_note='The independently stratum-conditioned pilot is used for design only, matching E5. Its full resource cost is charged.',
        latent_truth_for_selection='Only evaluator uses true gains; oracle methods separately identified')
    return report, dict(estimates=estimates, realized_resources=resources, selected=selected,
        regret=regret, nonoptimal=mismatch, regret_gt_epsilon=mistakes, true_gains=truth,
        method_names=np.asarray(METHODS), gate_masks=gates)


def estimated_domain_sensitivity(family, repeats=50):
    """Estimated feasible-set drift is separate from known-w/c main ranking."""
    gates, _ = candidate_gates(family)
    pairs = policy_pairs(gates)
    delta = gates[pairs[:, 0]] - gates[pairs[:, 1]]
    rows = []
    for count in PILOTS:
        n = 6 * count
        for repetition in range(repeats):
            rng = np.random.default_rng([SEED, 91, list(old.FAMILIES).index(family), count, repetition])
            strata = rng.choice(7, size=n, p=old.WEIGHTS)
            nh = np.bincount(strata, minlength=7)
            what = (nh + .5) / (n + 3.5)
            ah = np.zeros(7)
            muh = np.zeros(7)
            chat = np.ones(7)
            for h in range(6):
                if nh[h] == 0:
                    continue
                support, law = cluster_distribution(old.RHO[h], old.MU[h], old.L, 'conditional_independent')
                draws = rng.choice(support, size=nh[h], p=law)
                ah[h], muh[h] = np.mean(draws ** 2), draws.mean()
                costs = rng.lognormal(np.log(old.COSTS[h]) - .4 ** 2 / 2, .4, size=nh[h])
                chat[h] = costs.mean()
            gainhat = gates @ (what * muh)
            gaps = np.maximum(np.abs(gainhat[pairs[:, 0]] - gainhat[pairs[:, 1]]), EPSILON)
            matrices = {'Marginal': (what * ah)[None, :6],
                        'GapContrast': delta[:, :6] ** 2 * (what * ah)[None, :6] / gaps[:, None] ** 2}
            for ann_fraction, exe_fraction in BUDGETS:
                ann = ann_fraction * old.WEIGHTS[:6].sum()
                exe = exe_fraction * (old.WEIGHTS * old.COSTS)[:6].sum()
                for method, C in matrices.items():
                    p, detail = optimize_design(C, what[:6], (what * chat)[:6], ann, exe)
                    true_ann = float(old.WEIGHTS[:6] @ p)
                    true_exe = float((old.WEIGHTS * old.COSTS)[:6] @ p)
                    rows.append(dict(pilot_per_stratum_label=count, random_input_pilot_questions=n,
                        seed=repetition, method=method, annotation_fraction=ann_fraction, execution_fraction=exe_fraction,
                        annotation_violation=max(0., true_ann - ann), execution_violation=max(0., true_exe - exe),
                        estimated_domain_violation=detail['max_constraint_violation'],
                        scaled_duality_gap=detail['scaled_duality_gap'],
                        note='iid pilot for sensitivity only, not conditionally balanced main pilot'))
    return dict(family=family, repeats=repeats, rows=rows,
        guarantee_scope='Estimated weights/costs change the feasible set; true-resource feasibility is measured, not asserted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    started = time.perf_counter()
    q = qualification()
    output = OUTPUT / 'private/b'
    cases = []
    worlds = []
    plan = [(family, ann, exe) for family in old.FAMILIES for ann, exe in BUDGETS]
    if args.smoke:
        plan = plan[:1]
    for family, ann, exe in plan:
        designs = [all_designs(family, ann, exe, count) for count in PILOTS]
        fullprob = [probs[method] for probs, _, _ in designs for method in METHODS]
        gates, _ = candidate_gates(family)
        estimates, resources, world = refine_original_world(family, ann, exe, fullprob, gates, args.replicates)
        worlds.append(world)
        for index, count in enumerate(PILOTS):
            sl = slice(index * len(METHODS), (index + 1) * len(METHODS))
            report, arrays = summarize_case(designs[index][1], estimates[:, sl], resources[:, sl], args.replicates)
            name = f'{family}_ann{int(ann*100)}_exec{int(exe*100)}_pilot{count}'
            save(output / 'cases' / (name + '.json'), report)
            np.savez_compressed(output / 'cases' / (name + '.npz'), **arrays)
            save(output / 'pilots' / f'{family}_pilot{count}.json', designs[index][2])
            cases.append(report)
            print(json.dumps(dict(case=name, status='COMPLETE', elapsed_s=time.perf_counter() - started)), flush=True)
    sensitivity = [] if args.smoke else [estimated_domain_sensitivity(family) for family in old.FAMILIES]
    result = dict(status='SMOKE' if args.smoke else 'COMPLETE',
        config=dict(seed=SEED, epsilon=EPSILON, selection_tolerance=TOLERANCE, replicas=args.replicates,
            pilot_per_stratum=PILOTS, budget_fractions=BUDGETS, methods=METHODS,
            known_weights_costs_main=True, new_physical_llm_calls=0,
            synthetic_question_count=old.T, repeats_per_question=old.L),
        qualification=q, old_world_reproduction=worlds, cases=cases,
        estimated_feasible_set_sensitivity=sensitivity,
        structural_identity='With default and all-switch present, unnormalized marginal and contrast minimax objectives coincide pointwise.',
        elapsed_s=time.perf_counter() - started,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [Path(__file__), Path(__file__).with_name('contrast_design.py'),
                                    E5/'public/allocation.py', E5/'public/run_allocation_synthetic.py']})
    save(OUTPUT / 'public/b' / ('E6_B_SMOKE.json' if args.smoke else 'E6_B_RESULTS.json'), result)
    print(json.dumps(dict(status=result['status'], cases=len(cases), elapsed_s=result['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
