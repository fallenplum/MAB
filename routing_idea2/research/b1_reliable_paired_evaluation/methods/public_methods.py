"""Clean-room B1 adapters traced to public ASI and Robust-ASI formulas.

Native code remains in the private source snapshot. These adapters expose the
explicit B1 modifications, including a probability floor and two-budget scaling.
No evaluator labels or realized execution lengths are inputs to design functions.
"""

from __future__ import annotations

import numpy as np


def _weights(size, weights):
    w = np.ones(size, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (size,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid population weights")
    return w / w.sum()


def scale_to_budgets(shape, costs, answer_budget, execution_budget, pmin=0.05, weights=None):
    """Largest scalar along a fixed positive shape under two expected budgets.

With weights=None budgets are totals across the vector. With weights supplied,
budgets are per-population-unit expectations and weights sum to one internally.
"""
    shape = np.maximum(np.asarray(shape, dtype=float), 1e-15)
    costs = np.asarray(costs, dtype=float)
    if shape.shape != costs.shape or np.any(costs < 0):
        raise ValueError("Invalid shapes or costs")
    w = np.ones(len(shape)) if weights is None else _weights(len(shape), weights)
    if answer_budget < pmin * w.sum() - 1e-10 or execution_budget < pmin * (w @ costs) - 1e-10:
        raise ValueError("Probability floor makes expected budget infeasible")
    def feasible(p):
        return w @ p <= answer_budget + 1e-12 and w @ (costs * p) <= execution_budget + 1e-12
    lo, hi = 0.0, 1.0
    while feasible(np.clip(hi * shape, pmin, 1.0)) and hi < 1e18:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if feasible(np.clip(mid * shape, pmin, 1.0)):
            lo = mid
        else:
            hi = mid
    p = np.clip(lo * shape, pmin, 1.0)
    return {"p": p, "scale": lo, "expected_answers": float(w @ p),
            "expected_execution": float(w @ (costs * p)),
            "adaptation": "fixed-shape maximum-scalar two-expected-budget scaling with common floor"}


def tune_asi_tau(uncertainty, residual, rate, weights=None):
    """Author Pew 100-point tau search on legal held-out historical residuals."""
    u = np.maximum(np.asarray(uncertainty, dtype=float), 1e-12)
    residual = np.asarray(residual, dtype=float)
    w = _weights(len(u), weights)
    active = rate * u / (w @ u)
    taus = np.linspace(0.01, 0.99, 100)
    values = np.asarray([w @ (residual**2 / np.clip((1-t) * active + t * rate, 1e-12, 1)) for t in taus])
    index = int(np.argmin(values))
    return {"tau": float(taus[index]), "objective": float(values[index]), "grid": taus.tolist(),
            "historical_objectives": values.tolist()}


def asi_probabilities(v, c, Bann, Bexec, pmin=0.05, tau=0.01, weights=None):
    """ASI-inspired B1 allocation using shared RMS residual uncertainty.

The author binary-example min(f,1-f) is replaced by sqrt(v) for a paired
continuous reward. Tau must be frozen from historical feedback before use.
"""
    v = np.maximum(np.asarray(v, dtype=float), 1e-12)
    w = _weights(len(v), weights)
    rate = Bann / len(v) if weights is None else Bann
    u = np.sqrt(v)
    native_p = np.clip((1 - tau) * rate * u / (w @ u) + tau * rate, 1e-15, 1.0)
    result = scale_to_budgets(native_p, c, Bann, Bexec, pmin, weights)
    result.update(method="ASI sampling adapter", tau=float(tau), native_single_budget_p=native_p)
    return result


def robust_geometric_path(v, rate, c_radius, n_population=None, weights=None, rho_grid=None):
    """Author geometric path and l2 minimax objective, prior to B1 budget scaling.

Author MinMaxOptimizer_l2 uses C=c*sqrt(N), error=e_hat and objective
sum(e_hat**2 / q) + C*sqrt(sum(1/q**2)). With discrete population weights,
the same objective divided by N is E[v/q]+c*sqrt(E[1/q**2]).
"""
    v = np.maximum(np.asarray(v, dtype=float), 1e-12)
    w = _weights(len(v), weights)
    if not 0 < rate <= 1 or c_radius < 0:
        raise ValueError("Invalid rate or robustness radius")
    base = np.clip(rate * np.sqrt(v) / (w @ np.sqrt(v)), 1e-15, 1)
    grid = np.linspace(0, 1, 100) if rho_grid is None else np.asarray(rho_grid)
    objectives, probabilities = [], []
    for rho in grid:
        shape = base ** (1 - rho) * rate ** rho
        p = np.clip(rate * shape / (w @ shape), 1e-15, 1)
        objectives.append(float(w @ (v / p) + c_radius * np.sqrt(w @ (1 / p**2))))
        probabilities.append(p)
    index = int(np.argmin(objectives))
    n = len(v) if n_population is None else n_population
    return {"p": probabilities[index], "rho": float(grid[index]), "c_radius": float(c_radius),
            "author_C": float(c_radius * np.sqrt(n)), "objective": objectives[index],
            "rho_grid": grid.tolist(), "objectives": objectives,
            "formula_source": "Robust-ASI utils_optimization.py MinMaxOptimizer_l2"}


def robust_probabilities(v, c, Bann, Bexec, pmin=0.05, c_radius=0.02, weights=None, n_population=None):
    rate = Bann / len(v) if weights is None else Bann
    native = robust_geometric_path(v, rate, c_radius, n_population, weights)
    result = scale_to_budgets(native["p"], c, Bann, Bexec, pmin, weights)
    result.update({k: value for k, value in native.items() if k != "p"})
    result.update(method="Robust-ASI geometric minimax sampling adapter", native_single_budget_p=native["p"])
    return result


def tune_robust_radius(folds, rate, weights=None):
    """Historical held-out risk tuning for the author six-candidate c grid.

Each fold provides predicted_e2 and observed residual for held-out questions.
The exact expected conditional second-moment criterion replaces the author's
200-draw Monte Carlo mean-standard-deviation criterion. This is an explicit
B1 adaptation and must be reported beside a native-code operator check.
"""
    candidates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    objectives = []
    for radius in candidates:
        total = 0.0
        for fold in folds:
            v = np.asarray(fold["predicted_e2"], dtype=float)
            r = np.asarray(fold["residual"], dtype=float)
            w = _weights(len(v), fold.get("weights", weights))
            path = robust_geometric_path(v, rate, radius, weights=w)
            total += float(w @ (r**2 / path["p"]))
        objectives.append(total / len(folds))
    idx = int(np.argmin(objectives))
    return {"c_radius": candidates[idx], "c_candidates": candidates, "objectives": objectives,
            "adaptation": "historical held-out exact second-moment CV replacing native 200-draw pseudo-label SD"}


def native_population_interval(prediction, revealed_residual, selected, probabilities, alpha=0.05):
    """Native ASI sample-SD CLT interval, for population inference only."""
    from scipy.stats import norm
    m = np.asarray(prediction, dtype=float)
    a = np.asarray(selected, dtype=bool)
    p = np.asarray(probabilities, dtype=float)
    values = m.copy()
    values[a] += np.asarray(revealed_residual, dtype=float)[a] / p[a]
    point = float(values.mean())
    half = float(norm.ppf(1-alpha/2) * values.std(ddof=0) / np.sqrt(len(values)))
    return point, point-half, point+half


def active_testing_lure(losses, conditional_q, population_size):
    """LURE weights from upstream FancyUnbiasedRiskEstimator, no clipping."""
    loss = np.asarray(losses, dtype=float)
    q = np.asarray(conditional_q, dtype=float)
    M, N = len(loss), population_size
    if M == 0:
        raise ValueError("LURE requires an observation")
    if M == N:
        return float(loss.mean())
    m = np.arange(1, M+1)
    weights = 1 + (N-M)/(N-m) * (1/((N-m+1) * q) - 1)
    return float(weights @ loss / M)
