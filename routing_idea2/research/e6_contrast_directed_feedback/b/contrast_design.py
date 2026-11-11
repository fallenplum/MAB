"""Convex two-resource designs for frozen-policy marginal or contrast loss.

The objective is max_r(sum_h C[r,h] / p[h] + offset[r]).  C must be
nonnegative.  Offsets permit exact finite-pool Bernoulli variance objectives.
The returned primal-dual witness concerns the supplied deterministic program.
"""
import warnings

import numpy as np
from scipy.optimize import minimize, nnls


def optimize_design(coefficient_matrix, annotation_weights, execution_weights,
                    ann_budget, exec_budget, pmin=.05, offset=None):
    """Return probabilities and a numerical primal-dual qualification record.

    Every supplied coordinate receives strictly positive probability.  Callers
    can remove structurally irrelevant coordinates before calling this routine.
    """
    C = np.asarray(coefficient_matrix, dtype=float)
    aw = np.asarray(annotation_weights, dtype=float)
    ew = np.asarray(execution_weights, dtype=float)
    if C.ndim != 2 or not len(C) or aw.ndim != 1 or C.shape[1] != len(aw) or ew.shape != aw.shape:
        raise ValueError('Expected a nonempty rows-by-coordinates objective and resource vectors')
    lower = np.broadcast_to(np.asarray(pmin, dtype=float), aw.shape).copy()
    offsets = np.zeros(len(C)) if offset is None else np.asarray(offset, dtype=float)
    arrays = (C, aw, ew, lower, offsets)
    if offsets.shape != (len(C),) or not all(np.isfinite(v).all() for v in arrays):
        raise ValueError('All arguments must be finite and have compatible shapes')
    if np.any(C < 0) or np.any(aw < 0) or np.any(ew < 0) or np.any(lower <= 0) or np.any(lower > 1):
        raise ValueError('Nonnegative coefficients/resources and positive probability bounds are required')
    budgets = np.array([ann_budget, exec_budget], dtype=float)
    resources = np.stack((aw, ew))
    minimum = resources @ lower
    if not np.isfinite(budgets).all() or np.any(budgets < minimum - 1e-10):
        raise ValueError('The positive-propensity resource constraints are infeasible')

    # Objective normalization keeps gap-scaled designs numerically comparable.
    scale = max(float(np.max(np.abs(C).sum(axis=1) + np.abs(offsets))), 1e-12)
    Cs = C / scale
    os = offsets / scale

    def values(p):
        return Cs @ (1 / p) + os

    def fit(initial):
        constraints = [
            dict(type='ineq', fun=lambda z: z[-1] - values(z[:-1]),
                 jac=lambda z: np.column_stack((Cs / z[:-1] ** 2, np.ones(len(Cs))))),
            dict(type='ineq', fun=lambda z: budgets - resources @ z[:-1],
                 jac=lambda z: np.column_stack((-resources, np.zeros(2))))
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = minimize(lambda z: z[-1], initial,
                              jac=lambda z: np.r_[np.zeros(len(aw)), 1.],
                              bounds=[(float(v), 1.) for v in lower] + [(None, None)],
                              constraints=constraints, method='SLSQP',
                              options={'ftol': 2e-12, 'maxiter': 2000})
        return result, [str(item.message) for item in caught]

    direction = 1 - lower
    ratios = np.divide(budgets - minimum, resources @ direction,
                       out=np.ones(2), where=resources @ direction > 0)
    fraction = min(1., float(ratios.min()))
    start = lower + max(0., fraction) * direction
    result, warning_text = fit(np.r_[start, values(start).max() + .01])
    p = np.maximum(lower, np.minimum(1., result.x[:-1]))
    fs = values(p)
    t = float(fs.max())

    # KKT nonnegative multiplier recovery, using scaled objective gradients.
    tolerance = 2e-8 * max(1., abs(t))
    columns = []
    names = []
    for row in range(len(Cs)):
        if t - fs[row] <= tolerance:
            columns.append(np.r_[-Cs[row] / p ** 2, -1.])
            names.append(('objective', row))
    for resource in range(2):
        if budgets[resource] - resources[resource] @ p <= tolerance:
            columns.append(np.r_[resources[resource], 0.])
            names.append(('resource', resource))
    for coordinate in range(len(p)):
        if p[coordinate] - lower[coordinate] <= tolerance:
            vector = np.zeros(len(p) + 1)
            vector[coordinate] = -1.
            columns.append(vector)
            names.append(('lower', coordinate))
        if 1 - p[coordinate] <= tolerance:
            vector = np.zeros(len(p) + 1)
            vector[coordinate] = 1.
            columns.append(vector)
            names.append(('upper', coordinate))
    multipliers, residual = nnls(np.column_stack(columns),
                                np.r_[np.zeros(len(p)), -1.], maxiter=20000)
    lam = np.zeros(len(Cs))
    eta = np.zeros(2)
    for value, (kind, index) in zip(multipliers, names):
        if kind == 'objective':
            lam[index] = value
        if kind == 'resource':
            eta[index] = value
    if lam.sum() <= 0:
        raise RuntimeError('No nonnegative simplex dual multiplier was recovered')
    eta /= lam.sum()
    lam /= lam.sum()
    numerator = lam @ Cs
    denominator = eta @ resources
    minimizer = lower.copy()
    positive = (numerator > 0) & (denominator > 0)
    minimizer[positive] = np.clip(np.sqrt(numerator[positive] / denominator[positive]),
                                  lower[positive], 1.)
    minimizer[(numerator > 0) & (denominator == 0)] = 1.
    flat = (numerator == 0) & (denominator == 0)
    minimizer[flat] = p[flat]
    dual_scaled = float(np.sum(numerator / minimizer + denominator * minimizer)
                        + lam @ os - eta @ budgets)
    gap_scaled = float(t - dual_scaled)
    violation = max(0., float(np.max(resources @ p - budgets)),
                    float(np.max(lower - p)), float(np.max(p - 1)))
    if violation > 1e-7 or abs(gap_scaled) > 3e-6 * max(1., abs(t)):
        raise RuntimeError('Design qualification failed: ' + str((result.message, violation, gap_scaled, residual)))
    diagnostics = dict(
        status='NUMERICALLY_QUALIFIED', objective=float(t * scale),
        objective_scale=scale, row_values=(fs * scale).tolist(),
        annotation=float(aw @ p), execution=float(ew @ p),
        max_constraint_violation=violation,
        dual_value=float(dual_scaled * scale), duality_gap=float(gap_scaled * scale),
        scaled_duality_gap=gap_scaled, dual_weights=lam.tolist(),
        resource_multipliers=(eta * scale).tolist(),
        stationarity_nnls_residual=float(residual),
        max_pointwise_minimizer_difference=float(np.max(np.abs(p - minimizer))),
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit), optimizer_warnings=warning_text,
        guarantee_scope='Deterministic convex program for the supplied coefficients and expected-resource constraints')
    return p, diagnostics


def qualification():
    C = np.array([[.08, .03, .1], [.02, .09, .04]])
    aw = np.array([.2, .3, .5])
    ew = np.array([.1, .6, 1.5])
    p, diagnostic = optimize_design(C, aw, ew, .4, .7)
    p2, finite = optimize_design(C, aw, ew, .4, .7, offset=-C.sum(1))
    assert aw @ p <= .4 + 1e-8 and ew @ p <= .7 + 1e-8
    assert np.all(p >= .05)
    assert diagnostic['objective'] == max(C @ (1 / p))
    assert np.isclose(finite['objective'], max(C @ ((1 - p2) / p2)))
    analytic, result = optimize_design(np.array([[.04, .18]]), np.array([.4, .6]),
                                      np.ones(2), .3, 10.)
    reference = .3 * np.sqrt([.1, .3]) / (np.array([.4, .6]) @ np.sqrt([.1, .3]))
    assert np.max(np.abs(analytic - reference)) < 2e-6
    return dict(status='PASS', checks=['two_resources', 'positive_propensity', 'finite_pool_offsets',
                                     'single_objective_analytic_square_root'], examples=[diagnostic, finite, result])


if __name__ == '__main__':
    import json
    print(json.dumps(qualification(), indent=2))
