"""Fixed-time inference for B1 paired evaluation.

The finite-pool and iid-population targets have different variances.  All
sampling probabilities and auxiliaries must be frozen before outcomes are read.
Three generation repeats are averaged within each original question.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np


def _upper_nonnegative_mean_sum(observed_sum, bound, failure_probability):
    """Upper bound on E sum(X_i), independent 0 <= X_i <= bound.

    Lower-tail Chernoff gives mu-s <= sqrt(2*bound*mu*log(1/delta)).
    Solving this quadratic produces the displayed upper confidence bound.
    This helper accepts arrays for Monte Carlo batches.
    """
    observed_sum = np.asarray(observed_sum, dtype=float)
    bound = np.asarray(bound, dtype=float)
    log_term = np.log(1.0 / failure_probability)
    correction = bound * log_term
    return observed_sum + correction + np.sqrt(
        2.0 * correction * observed_sum + correction * correction
    )


def _bernstein_radius(variance_upper, increment_bound, tail_failure):
    log_term = np.log(1.0 / tail_failure)
    linear = increment_bound * log_term / 3.0
    return linear + np.sqrt(2.0 * variance_upper * log_term + linear * linear)


def _intersect_or_widen(lower, upper, feasible_lower, feasible_upper):
    # Projection onto deterministic feasible bounds preserves coverage and
    # monotonicity of each one-sided bound when alpha varies.
    lower = np.clip(lower, feasible_lower, feasible_upper)
    upper = np.clip(upper, feasible_lower, feasible_upper)
    return lower, upper


def _validate_observations(d_observed, m, p, purchased):
    d_observed = np.asarray(d_observed, dtype=float)
    m = np.asarray(m, dtype=float)
    p = np.asarray(p, dtype=float)
    purchased = np.asarray(purchased, dtype=bool)
    if not (d_observed.shape == m.shape == p.shape == purchased.shape):
        raise ValueError("The four question arrays must have identical shapes")
    if d_observed.ndim != 1 or len(p) == 0:
        raise ValueError("A nonempty one-dimensional question pool is required")
    if np.any(~np.isfinite(m)) or np.any(np.abs(m) > 1.0 + 1e-12):
        raise ValueError("The frozen auxiliary must lie in [-1, 1]")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Inclusion probabilities must lie in [0, 1]")
    if np.any(purchased & (p == 0.0)) or np.any((p == 1.0) & ~purchased):
        raise ValueError("The purchase vector contradicts a deterministic probability")
    if np.any((p == 0.0) & (m != 0.0)):
        raise ValueError("A p=0 structural coordinate must have known gain and auxiliary 0")
    if np.any(~np.isfinite(d_observed[purchased])) or np.any(np.abs(d_observed[purchased]) > 1.0 + 1e-12):
        raise ValueError("Purchased question gains must lie in [-1, 1]")
    return d_observed, m, p, purchased


def finite_pool_many(estimate, variance_estimate, p, m, purchased_count,
                     observed_gain_sum, alpha=0.05):
    """Vectorized conditional-on-pool inference using independent Bernoulli buys.

    variance_estimate is Var(hat G_N)'s unbiased estimate, including 1/N**2.
    The caller supplies only observed summaries.  No hidden outcomes enter any
    confidence bound.  p=0 positions denote known structural zeros.
    """
    p = np.asarray(p, dtype=float)
    m = np.asarray(m, dtype=float)
    n = len(p)
    active = p > 0.0
    stochastic = active & (p < 1.0)
    active_count = int(active.sum())
    estimate = np.asarray(estimate, dtype=float)
    variance_estimate = np.asarray(variance_estimate, dtype=float)
    purchased_count = np.asarray(purchased_count)
    observed_gain_sum = np.asarray(observed_gain_sum, dtype=float)
    unseen_count = active_count - purchased_count
    feasible_lower = (observed_gain_sum - unseen_count) / n
    feasible_upper = (observed_gain_sum + unseen_count) / n
    residual_bound = 1.0 + np.abs(m[stochastic])
    probs = p[stochastic]
    increment_bound = float(np.max(
        residual_bound * np.maximum(1.0, (1.0 - probs) / probs), initial=0.0
    ))
    variance_observation_bound = float(np.max(
        (1.0 - probs) * residual_bound**2 / probs**2, initial=0.0
    ))
    deterministic_variance_upper = float(np.sum(
        (1.0 - probs) * residual_bound**2 / probs
    ))
    variance_upper = _upper_nonnegative_mean_sum(
        variance_estimate * n**2, variance_observation_bound, alpha / 2.0
    )
    variance_upper = np.minimum(variance_upper, deterministic_variance_upper)
    radius = _bernstein_radius(variance_upper, increment_bound, alpha / 4.0) / n
    lower, upper = _intersect_or_widen(
        estimate - radius, estimate + radius, feasible_lower, feasible_upper
    )
    one_sided_radius = _bernstein_radius(
        variance_upper, increment_bound, alpha / 2.0
    ) / n
    one_lower, one_upper = _intersect_or_widen(
        estimate - one_sided_radius, estimate + one_sided_radius,
        feasible_lower, feasible_upper
    )
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    standard_error = np.sqrt(np.maximum(variance_estimate, 0.0))
    wald_lower, wald_upper = _intersect_or_widen(
        estimate - z * standard_error, estimate + z * standard_error,
        feasible_lower, feasible_upper
    )
    z_one = NormalDist().inv_cdf(1.0 - alpha)
    wald_one_lower, wald_one_upper = _intersect_or_widen(
        estimate - z_one * standard_error, estimate + z_one * standard_error,
        feasible_lower, feasible_upper
    )
    # A zero estimated variance with unseen questions provides no variance
    # evidence.  The shared Wald adaptation returns its deterministic envelope.
    no_variance_evidence = (variance_estimate <= 0.0) & (unseen_count > 0)
    for name, value in (
        ("wald_lower", wald_lower), ("wald_one_lower", wald_one_lower)
    ):
        if name == "wald_lower":
            wald_lower = np.where(no_variance_evidence, feasible_lower, value)
        else:
            wald_one_lower = np.where(no_variance_evidence, feasible_lower, value)
    wald_upper = np.where(no_variance_evidence, feasible_upper, wald_upper)
    wald_one_upper = np.where(no_variance_evidence, feasible_upper, wald_one_upper)
    no_observations = purchased_count == 0
    lower = np.where(no_observations, feasible_lower, lower)
    upper = np.where(no_observations, feasible_upper, upper)
    one_lower = np.where(no_observations, feasible_lower, one_lower)
    one_upper = np.where(no_observations, feasible_upper, one_upper)
    return {
        "finite_bernstein_lower": lower,
        "finite_bernstein_upper": upper,
        "finite_bernstein_one_lower": one_lower,
        "finite_bernstein_one_upper": one_upper,
        "asymptotic_wald_lower": wald_lower,
        "asymptotic_wald_upper": wald_upper,
        "asymptotic_wald_one_lower": wald_one_lower,
        "asymptotic_wald_one_upper": wald_one_upper,
        "feasible_lower": feasible_lower,
        "feasible_upper": feasible_upper,
        "variance_upper": variance_upper / n**2,
        "no_observations": no_observations,
    }


def finite_pool(d_observed, m, p, purchased, alpha=0.05):
    """AIPW point estimate and fixed-pool confidence statements.

    d_observed may contain NaN on unpurchased questions.  Each row is one
    question cluster, after averaging its three generation repetitions.
    """
    d_observed, m, p, purchased = _validate_observations(d_observed, m, p, purchased)
    n = len(p)
    observed_residual = d_observed[purchased] - m[purchased]
    estimate = float((m.sum() + np.sum(observed_residual / p[purchased])) / n)
    variance_estimate = float(np.sum(
        (1.0 - p[purchased]) * observed_residual**2 / p[purchased]**2
    ) / n**2)
    intervals = finite_pool_many(
        estimate, variance_estimate, p, m, int(purchased.sum()),
        float(d_observed[purchased].sum()), alpha
    )
    return {
        "target": "fixed_pool_and_fixed_three_generation_repetitions",
        "estimate": estimate,
        "variance_estimate": variance_estimate,
        "purchased_questions": int(purchased.sum()),
        "population_denominator": n,
        "intervals": {key: float(value) if np.asarray(value).ndim == 0 else value
                      for key, value in intervals.items()},
        "assumptions": [
            "Fixed finite question pool and fixed auxiliaries/probabilities",
            "Independent Bernoulli inclusion of original question clusters",
            "Unpurchased outcomes are inaccessible to the design",
            "p=0 only for externally known structural zero gains",
            "All statements concern a single prespecified evaluation time",
        ],
    }


def population_many(estimate, sum_q_squared, purchased_count, n,
                    global_q_absolute_bound, alpha=0.05):
    """Fixed-time inference for iid question-cluster IPW/AIPW observations.

    q_i = m(X_i) + A_i/p(X_i)*(d_i-m(X_i)).  p(.) and m(.) are fixed
    independently of the evaluated iid questions.  The global bound must hold
    on the complete input support, not merely on the realized sample.
    """
    estimate = np.asarray(estimate, dtype=float)
    sum_q_squared = np.asarray(sum_q_squared, dtype=float)
    purchased_count = np.asarray(purchased_count)
    h = float(global_q_absolute_bound)
    if h < 0.0 or n <= 0:
        raise ValueError("A valid global q bound and positive question count are required")
    # E q^2 upper bounds Var(q).  This is an iid population variance bound,
    # not the finite-pool sampling variance.
    variance_sum_upper = _upper_nonnegative_mean_sum(
        sum_q_squared, h * h, alpha / 2.0
    )
    variance_sum_upper = np.minimum(variance_sum_upper, n * h * h)
    increment_bound = h + 1.0  # G lies in [-1, 1].
    radius = _bernstein_radius(
        variance_sum_upper, increment_bound, alpha / 4.0
    ) / n
    one_radius = _bernstein_radius(
        variance_sum_upper, increment_bound, alpha / 2.0
    ) / n
    lower = np.maximum(-1.0, estimate - radius)
    upper = np.minimum(1.0, estimate + radius)
    one_lower = np.maximum(-1.0, estimate - one_radius)
    one_upper = np.minimum(1.0, estimate + one_radius)
    sample_variance = np.maximum(0.0, (sum_q_squared - n * estimate**2) / max(1, n - 1))
    standard_error = np.sqrt(sample_variance / n)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_one = NormalDist().inv_cdf(1.0 - alpha)
    wald_lower = np.maximum(-1.0, estimate - z * standard_error)
    wald_upper = np.minimum(1.0, estimate + z * standard_error)
    wald_one_lower = np.maximum(-1.0, estimate - z_one * standard_error)
    wald_one_upper = np.minimum(1.0, estimate + z_one * standard_error)
    fallback = (purchased_count == 0) | (sample_variance <= 0.0)
    for name in ("lower", "one_lower", "wald_lower", "wald_one_lower"):
        if name == "lower":
            lower = np.where(purchased_count == 0, -1.0, lower)
        elif name == "one_lower":
            one_lower = np.where(purchased_count == 0, -1.0, one_lower)
        elif name == "wald_lower":
            wald_lower = np.where(fallback, -1.0, wald_lower)
        else:
            wald_one_lower = np.where(fallback, -1.0, wald_one_lower)
    upper = np.where(purchased_count == 0, 1.0, upper)
    one_upper = np.where(purchased_count == 0, 1.0, one_upper)
    wald_upper = np.where(fallback, 1.0, wald_upper)
    wald_one_upper = np.where(fallback, 1.0, wald_one_upper)
    return {
        "population_bernstein_lower": lower,
        "population_bernstein_upper": upper,
        "population_bernstein_one_lower": one_lower,
        "population_bernstein_one_upper": one_upper,
        "population_wald_lower": wald_lower,
        "population_wald_upper": wald_upper,
        "population_wald_one_lower": wald_one_lower,
        "population_wald_one_upper": wald_one_upper,
        "population_sample_variance": sample_variance,
        "population_variance_upper": variance_sum_upper / n**2,
        "no_observations": purchased_count == 0,
    }


def classify_gain(lower, upper, threshold=0.0):
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return np.where(lower > threshold, "adopt",
                    np.where(upper <= threshold, "below_threshold", "insufficient_evidence"))


def finite_pool_pvalue(d_observed, m, p, purchased, threshold=0.0,
                       direction="positive", iterations=48):
    """Invert the valid nested one-sided fixed-time Bernstein tests."""
    if direction not in ("positive", "negative"):
        raise ValueError("direction must be positive or negative")
    result = finite_pool(d_observed, m, p, purchased, alpha=0.5)
    interval = result["intervals"]
    if direction == "positive" and interval["feasible_lower"] > threshold:
        return 0.0
    if direction == "negative" and interval["feasible_upper"] <= threshold:
        return 0.0
    if result["purchased_questions"] == 0:
        return 1.0
    left, right = 0.0, 1.0
    for _ in range(iterations):
        alpha = (left + right) / 2.0
        interval = finite_pool(d_observed, m, p, purchased, alpha)["intervals"]
        rejected = (interval["finite_bernstein_one_lower"] > threshold
                    if direction == "positive"
                    else interval["finite_bernstein_one_upper"] <= threshold)
        if rejected:
            right = alpha
        else:
            left = alpha
    if right > 1.0 - 1e-10:
        return 1.0
    return right


def exhaustive_design_check():
    """Enumerate a small pool exactly; this does not read MAB outcomes."""
    from itertools import product

    d = np.array([1.0, -1.0 / 3.0, 0.0, 2.0 / 3.0])
    m = np.array([0.2, -0.1, 0.0, 0.15])
    p = np.array([0.35, 0.65, 0.0, 0.5])
    target = float(d.mean())
    analytic_variance = float(np.sum((1-p[p > 0]) * (d[p > 0]-m[p > 0])**2 / p[p > 0]) / len(p)**2)
    rows = []
    for pattern in product((False, True), repeat=3):
        a = np.zeros(4, dtype=bool)
        a[[0, 1, 3]] = pattern
        probability = float(np.prod(np.where(a[p > 0], p[p > 0], 1-p[p > 0])))
        observed = np.where(a, d, np.nan)
        result = finite_pool(observed, m, p, a)
        rows.append({
            "purchased": a.astype(int).tolist(), "probability": probability,
            "estimate": result["estimate"], "variance_estimate": result["variance_estimate"],
            "covered": bool(result["intervals"]["finite_bernstein_lower"] - 1e-12 <= target <= result["intervals"]["finite_bernstein_upper"] + 1e-12),
        })
    expectation = sum(row["probability"] * row["estimate"] for row in rows)
    variance = sum(row["probability"] * (row["estimate"]-target)**2 for row in rows)
    expected_vhat = sum(row["probability"] * row["variance_estimate"] for row in rows)
    coverage = sum(row["probability"] * row["covered"] for row in rows)
    no_labels = finite_pool(np.full(4, np.nan), m, p, np.zeros(4, dtype=bool))
    all_structural = finite_pool(np.full(3, np.nan), np.zeros(3), np.zeros(3), np.zeros(3, dtype=bool))
    assert abs(expectation-target) < 1e-12
    assert abs(variance-analytic_variance) < 1e-12
    assert abs(expected_vhat-analytic_variance) < 1e-12
    assert coverage >= 0.95 - 1e-12
    assert no_labels["intervals"]["finite_bernstein_lower"] == -0.75
    assert no_labels["intervals"]["finite_bernstein_upper"] == 0.75
    assert all_structural["intervals"]["finite_bernstein_lower"] == 0.0
    return {"target": target, "expected_estimate": expectation,
            "exact_design_variance": variance, "analytic_variance": analytic_variance,
            "expected_variance_estimate": expected_vhat,
            "finite_95_coverage": coverage, "enumerated_patterns": rows,
            "zero_observation_result": no_labels,
            "all_structural_result": all_structural}


if __name__ == "__main__":
    import json
    print(json.dumps(exhaustive_design_check(), ensure_ascii=False, indent=2))
