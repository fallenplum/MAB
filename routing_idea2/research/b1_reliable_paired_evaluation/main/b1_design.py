"""Two-expected-budget Bernoulli designs for a frozen paired estimator."""
from __future__ import annotations

import numpy as np


def optimal_probabilities(moment, costs, answer_budget, execution_budget, pmin=0.05):
    """Solve the separable convex program through its two resource multipliers.

    Inputs contain only nonstructural coordinates. Costs are scaled internally.
    The returned multipliers correspond to the original cost units.
    """
    v = np.asarray(moment, dtype=float)
    c = np.asarray(costs, dtype=float)
    if v.ndim != 1 or v.shape != c.shape or len(v) == 0:
        raise ValueError("Nonempty, equally sized moment and cost vectors required")
    if np.any(v <= 0) or np.any(c <= 0) or not 0 < pmin <= 1:
        raise ValueError("Strictly positive work moments, costs, and probability floor required")
    if answer_budget < pmin * len(v) - 1e-9 or execution_budget < pmin * c.sum() - 1e-6:
        raise ValueError("Expected budget infeasible at the common probability floor")
    scale = float(c.mean())
    c_scaled, budget_scaled = c / scale, execution_budget / scale

    def probabilities(alpha, beta):
        denominator = alpha + beta * c_scaled
        ratio = np.divide(v, denominator, out=np.full_like(v, np.inf), where=denominator > 0)
        return np.clip(np.sqrt(ratio), pmin, 1.0)

    def answer_multiplier(beta):
        if probabilities(0.0, beta).sum() <= answer_budget:
            return 0.0
        lo, hi = 0.0, max(1.0, float(v.max()))
        while probabilities(hi, beta).sum() > answer_budget:
            hi *= 2.0
        for _ in range(75):
            mid = (lo + hi) / 2.0
            if probabilities(mid, beta).sum() > answer_budget:
                lo = mid
            else:
                hi = mid
        return hi

    beta = 0.0
    alpha = answer_multiplier(beta)
    if c_scaled @ probabilities(alpha, beta) > budget_scaled + 1e-10:
        lo, hi = 0.0, max(1.0, float(v.max()))
        while c_scaled @ probabilities(answer_multiplier(hi), hi) > budget_scaled:
            hi *= 2.0
        for _ in range(75):
            mid = (lo + hi) / 2.0
            current = probabilities(answer_multiplier(mid), mid)
            if c_scaled @ current > budget_scaled:
                lo = mid
            else:
                hi = mid
        beta = hi
        alpha = answer_multiplier(beta)
    p = probabilities(alpha, beta)
    objective = float(np.sum(v / p))
    dual = float(np.sum(v / p + (alpha + beta * c_scaled) * p)
                 - alpha * answer_budget - beta * budget_scaled)
    diagnostics = dict(
        objective=objective, dual_lower_bound=dual,
        relative_primal_dual_gap=max(0.0, objective - dual) / max(1.0, abs(objective)),
        lambda_answer=float(alpha), lambda_execution=float(beta / scale),
        expected_answers=float(p.sum()), expected_execution=float(c @ p),
        answer_budget=float(answer_budget), execution_budget=float(execution_budget),
        answer_binding=bool(abs(p.sum() - answer_budget) <= 1e-6 * max(1, answer_budget)),
        execution_binding=bool(abs(c @ p - execution_budget) <= 1e-6 * execution_budget),
        pmin=float(pmin), floor_count=int(np.sum(p <= pmin + 1e-10)),
        ceiling_count=int(np.sum(p >= 1 - 1e-10)), cost_normalization=scale,
    )
    if diagnostics['relative_primal_dual_gap'] > 1e-7:
        raise AssertionError("Two-budget allocation failed its global dual certificate")
    return p, diagnostics


def qualify_optimizer():
    """Independent SLSQP checks, boundary cases, and KKT certificates."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(2026111301)
    checks = []
    for setting in range(8):
        v = rng.uniform(0.001, 0.4, 12)
        c = rng.uniform(0.2, 4.0, 12)
        rate_a, rate_e = ((0.25, 0.25), (0.5, 0.25), (0.25, 0.5), (1., 1.))[setting % 4]
        ba, be = rate_a * len(v), rate_e * c.sum()
        p, diag = optimal_probabilities(v, c, ba, be)
        independent = minimize(lambda x: np.sum(v / x), np.full(12, min(rate_a, rate_e)),
            jac=lambda x: -v / x**2, method='SLSQP', bounds=[(0.05, 1)] * 12,
            constraints=[{'type': 'ineq', 'fun': lambda x: ba - x.sum()},
                         {'type': 'ineq', 'fun': lambda x: be - c @ x}],
            options={'ftol': 1e-11, 'maxiter': 1000})
        objective_difference = float(abs(np.sum(v / p) - independent.fun))
        if objective_difference > 2e-6:
            raise AssertionError("Disagreement with independent constrained optimizer")
        checks.append(dict(setting=setting, independent_success=bool(independent.success),
                           objective_difference=objective_difference, **diag))
    return dict(status='PASS', cases=checks, numerical_tolerance=2e-6)
