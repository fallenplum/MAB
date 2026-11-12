"""Independent-evaluator helpers for Active Testing and author CPS supplements.

Only the evaluation process supplies the complete d vector to this module.
Sampling designs are frozen from legal m, v, historical predicted costs and p.
Per-draw estimates index outcomes only at their purchased sample.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from public_methods import active_testing_lure


def _normal_interval(point, variance, alpha, active_count, denominator, no_labels=False):
    if no_labels or not np.isfinite(variance) or variance <= 0:
        return [-active_count / denominator, active_count / denominator]
    half = norm.ppf(1-alpha/2) * np.sqrt(variance)
    return [max(-1., point-half), min(1., point+half)]


def _fixed_n_target(probabilities, costs, answer_budget, execution_budget, pmin):
    p = np.asarray(probabilities, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n = int(np.floor(min(answer_budget, p.sum()) + 1e-9))
    while n >= 2:
        lo, hi = 0., 1.
        while np.clip(hi*p, pmin, 1.).sum() < n:
            hi *= 2.
        for _ in range(100):
            mid = (lo+hi)/2
            if np.clip(mid*p, pmin, 1.).sum() < n:
                lo = mid
            else:
                hi = mid
        target = np.clip(hi*p, pmin, 1.)
        if costs @ target <= execution_budget + 1e-7:
            return n, target
        n -= 1
    raise ValueError("At least two fixed-size labels required for SYG inference")


def fixed_size_asi_supplement(d, m, predicted_cost, active, bernoulli_p, answer_budget,
                              execution_budget, repeats=2000, seed=2026111307,
                              source_root=None, dependency_root=None, alpha=.05):
    """Official author's CPS package with actual pi/piij and exact SYG variance.

`bernoulli_p` is a legal ASI allocation already frozen under common resources.
The fixed n is rounded down, retaining its shape with a common p floor. This
is a B1 task adapter built on the author's package, not the inaccessible full
fixed-size-asi experiment repository.
"""
    if dependency_root is not None:
        sys.path.insert(0, str(dependency_root))
    if source_root is not None:
        sys.path.insert(0, str(Path(source_root) / "src"))
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    from cps_sampling import ConditionalPoisson, inclusion_from_matrix
    from cps_sampling.variance import syg_variance, ht_variance

    started = time.perf_counter()
    d = np.asarray(d, dtype=float)
    m = np.asarray(m, dtype=float)
    idx = np.flatnonzero(active)
    N, M = len(d), len(idx)
    costs = np.asarray(predicted_cost)[idx]
    p = np.asarray(bernoulli_p)[idx]
    n, target = _fixed_n_target(p, costs, answer_budget, execution_budget, .05)
    design = ConditionalPoisson(target, n=n)
    actual_pi = inclusion_from_matrix(design.q)
    joint_pi = design.joint_inclusion()
    if not np.allclose(actual_pi, np.diag(joint_pi), atol=2e-10):
        raise AssertionError("CPS first and second order inclusion disagree")
    if actual_pi @ costs > execution_budget + 1e-5:
        raise AssertionError("Actual CPS expected cost exceeds frozen resource budget")
    residual = d[idx]-m[idx]
    samples, estimates, variance_estimates, intervals = [], [], [], []
    for rep in range(repeats):
        chosen = design.sample(seed=seed+rep)
        global_chosen = idx[chosen]
        point = float(m.sum()/N + np.sum(residual[chosen]/actual_pi[chosen])/N)
        # Only chosen residuals are accessed in each design-unbiased estimator.
        variance = float(syg_variance(residual, actual_pi, joint_pi, chosen, N=N))
        samples.append(global_chosen)
        estimates.append(point)
        variance_estimates.append(variance)
        intervals.append(_normal_interval(point, variance, alpha, M, N))
    return {"method": "Fixed-Size ASI CPS package adapter", "samples": np.asarray(samples),
            "estimates": np.asarray(estimates), "variance_estimates": np.asarray(variance_estimates),
            "intervals": np.asarray(intervals), "actual_pi": actual_pi, "target_pi": target,
            "joint_pi": joint_pi, "active_indices": idx, "n": n,
            "diagnostics": {"target_actual_pi_max_error": float(abs(actual_pi-target).max()),
                            "fixed_n_identity_error": float(abs(joint_pi.sum(1)-n*actual_pi).max()),
                            "expected_answers": n, "expected_predicted_execution": float(actual_pi @ costs),
                            "exact_design_variance_evaluator_only": float(ht_variance(residual, actual_pi, joint_pi, N=M) * (M/N)**2),
                            "wall_time_s": time.perf_counter()-started,
                            "native_repo_status": "paper linked fixed-size-asi repo HTTP404, linked author cps-sampling package used",
                            "interval": "fixed-time asymptotic normal interval using exact design-unbiased SYG variance; no universal finite-sample coverage claim",
                            "budget": "fixed label count; predicted execution constrained in expectation; realized tokens vary"}}


def active_testing_supplement(d, m, predicted_cost, active, answer_budget, execution_budget,
                              repeats=2000, seed=2026111307, alpha=.05):
    """Sequential weighted-WoR LURE adaptation for paired bounded loss.

Uses ell=(1-d)/2 and its frozen prediction (1-m)/2 as expected-loss proposal.
The predictor is held fixed. This preserves original LURE estimator semantics
but omits native surrogate retraining, since paired response features differ.
The sample count is frozen using deterministic cost-only design Monte Carlo.
The interval is distribution-free Hoeffding-Azuma from known LURE ranges.
"""
    started = time.perf_counter()
    d, m, cost = map(np.asarray, [d, m, predicted_cost])
    idx = np.flatnonzero(active)
    N, M = len(d), len(idx)
    shape = np.clip((1-m[idx])/2., 1e-6, 1.)
    max_n = min(int(np.floor(answer_budget)), M)
    # Design simulations see only predicted loss and predicted execution cost.
    design_rng = np.random.default_rng(seed+991)
    design_costs = np.zeros((4096, max_n))
    for rep in range(len(design_costs)):
        remaining = np.arange(M)
        for t in range(max_n):
            q = shape[remaining]/shape[remaining].sum()
            # Native shared conf/config.yaml clip rule, value .5 of uniform.
            q = np.maximum(.5/len(remaining), q)
            q /= q.sum()
            position = design_rng.choice(len(remaining), p=q)
            selected = remaining[position]
            design_costs[rep, t] = cost[idx[selected]]
            remaining = np.delete(remaining, position)
    cumulative_means = np.cumsum(design_costs, axis=1).mean(0)
    feasible = np.flatnonzero(cumulative_means <= execution_budget)
    n = int(feasible[-1]+1) if len(feasible) else 0
    if n < 1:
        raise ValueError("No paid ActiveTesting observation fits expected resource budget")
    estimates, samples, intervals, conditional_qs, variance_proxy = [], [], [], [], []
    # LURE is a deterministic weighted average of the conditionally unbiased
    # PURE steps a_t=(sum_{j<t}ell_j + ell_t/q_t)/M. Their deterministic weights
    # are w_t=M*(M-n)/(n*(M-t)*(M-t+1)), summing to one. Native clipping gives
    # q_t >= 1/[3*(M-t+1)], hence the martingale increment range is bounded by
    # w_t/[M*q_min] = 3*(M-n)/[n*(M-t)]. Hoeffding-Azuma applies at fixed n.
    t = np.arange(1, n+1)
    increment_ranges = 3*(M-n)/(n*(M-t)) if n < M else np.zeros(n)
    loss_half = np.sqrt(np.log(2/alpha)*np.sum(increment_ranges**2)/2)
    gain_half = 2 * M/N * loss_half
    for rep in range(repeats):
        rng = np.random.default_rng(seed+rep)
        remaining = np.arange(M)
        chosen, qs = [], []
        for _ in range(n):
            q = shape[remaining]/shape[remaining].sum()
            q = np.maximum(.5/len(remaining), q)
            q /= q.sum()
            position = rng.choice(len(remaining), p=q)
            chosen.append(remaining[position])
            qs.append(q[position])
            remaining = np.delete(remaining, position)
        chosen = idx[np.asarray(chosen)]
        loss = (1-d[chosen])/2
        estimate = M/N*(1-2*active_testing_lure(loss, qs, M))
        estimates.append(estimate)
        samples.append(chosen)
        conditional_qs.append(qs)
        intervals.append([max(-M/N, estimate-gain_half), min(M/N, estimate+gain_half)])
        variance_proxy.append(np.nan)
    return {"method": "Active Testing LURE frozen-surrogate paired-loss adapter", "samples": np.asarray(samples),
            "conditional_q": np.asarray(conditional_qs), "estimates": np.asarray(estimates),
            "intervals": np.asarray(intervals), "variance_estimates": np.asarray(variance_proxy), "n": n,
            "diagnostics": {"expected_answers": n, "design_MC_repeats": len(design_costs),
                            "expected_predicted_execution_MC": float(cumulative_means[n-1]),
                            "design_MC_standard_error": float(np.cumsum(design_costs,axis=1)[:,n-1].std(ddof=1)/np.sqrt(len(design_costs))),
                            "interval": "conservative fixed-time LURE weighted-PURE martingale Hoeffding-Azuma interval",
                            "interval_halfwidth": float(gain_half),
                            "wall_time_s": time.perf_counter()-started,
                            "native_changes": ["paired loss=(1-d)/2", "frozen surrogate expected loss", "cost-only MC selects fixed label count", "conservative bounded-range interval added separately"]}}
