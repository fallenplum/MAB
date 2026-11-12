"""Exact small-case B1 inference and LTT native-operator qualification."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom

from b1_inference import exhaustive_design_check, finite_pool, population_many


def native_ltt_hb():
    source_path = Path(__file__).with_name("upstream") / "ltt_bounds.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "hb_p_value")
    # Execute the exact unmodified author function, avoiding imports needed
    # only for the repository's other optional betting functions.
    scope = {"np": np, "binom": binom}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), scope)
    return scope["hb_p_value"], hashlib.sha256(source.encode()).hexdigest()


def qualified_ltt_bounded_pvalue(mean_loss, n, risk_threshold):
    r = min(float(mean_loss), float(risk_threshold))
    divergence = xlogy(r, r/risk_threshold) + xlogy(1-r, (1-r)/(1-risk_threshold))
    hoeffding = np.exp(-n * divergence)
    bentkus = np.e * binom.cdf(np.ceil(n * mean_loss), n, risk_threshold)
    return float(min(1.0, hoeffding, bentkus))


def ltt_check():
    native, sha = native_ltt_hb()
    comparisons = []
    for n in (20, 50, 200):
        for risk_threshold in (0.2, 0.5, 0.7):
            for count in (1, max(1, n//5), n//2, n-1):
                mean_loss = count / n
                author = native(mean_loss, n, risk_threshold)
                adapted = qualified_ltt_bounded_pvalue(mean_loss, n, risk_threshold)
                assert abs(author-adapted) < 1e-12
                comparisons.append({"n": n, "risk_threshold": risk_threshold,
                                    "mean_loss": mean_loss, "native_p": float(author),
                                    "stable_operator_p": adapted})
    exact_null_checks = []
    for n in (20, 50, 200):
        null_risk = 0.5
        rejected = [qualified_ltt_bounded_pvalue(k/n, n, null_risk) <= 0.05 for k in range(n+1)]
        probability = float(np.sum(binom.pmf(np.arange(n+1), n, null_risk) * rejected))
        assert probability <= 0.05 + 1e-12
        exact_null_checks.append({"n": n, "iid_bernoulli_null_risk": null_risk,
                                  "exact_false_approval_probability": probability})
    rng = np.random.default_rng(2026111308)
    outcomes = []
    for gain in (0.0, -0.02, 0.01, 0.03):
        n = 5000
        # Legal iid bounded loss example for the native LTT operator.
        losses = rng.binomial(1, (1-gain)/2, size=n)
        outcomes.append({"true_gain": gain, "n": n,
                         "mean_loss": float(losses.mean()),
                         "native_hb_p": float(native(losses.mean(), n, 0.5))})
    return {"upstream_repository": "aangelopoulos/ltt",
            "upstream_commit": "3ad7a64adad9f356e29f40e73935a6114f896396",
            "source_path": "core/bounds.py", "source_sha256": sha,
            "reproduction_scope": "Exact native HB operator plus legal synthetic iid bounded-loss example; full image experiments were not executed",
            "comparisons": comparisons, "exact_null_checks": exact_null_checks,
            "single_candidate_reference_examples": outcomes,
            "active_ipw_input_to_native_hb": False,
            "optional_stopping_or_multiple_candidates": False}


def population_check():
    """Exact iid cluster enumeration with input-dependent purchase rates."""
    from itertools import product
    weights = np.array([.4, .6])
    gains = np.array([.2, -.1])
    disagreement = np.array([.4, .5])
    auxiliary = np.array([.1, -.05])
    probabilities = np.array([.3, .7])
    atoms = []
    for h, d, a in product(range(2), [-1., 0., 1.], [0, 1]):
        mass_d = ((disagreement[h]+gains[h])/2 if d == 1 else
                  (disagreement[h]-gains[h])/2 if d == -1 else 1-disagreement[h])
        mass = weights[h]*mass_d*(probabilities[h] if a else 1-probabilities[h])
        q = auxiliary[h]+a*(d-auxiliary[h])/probabilities[h]
        atoms.append((q, a, mass))
    atoms = np.array(atoms)
    n = 3
    indices = np.array(list(product(range(len(atoms)), repeat=n)))
    q = atoms[indices, 0]
    purchased = atoms[indices, 1]
    mass = np.prod(atoms[indices, 2], axis=1)
    estimate = q.mean(axis=1)
    truth = float(weights@gains)
    bound = float(np.max(np.abs(atoms[:, 0])))
    intervals = population_many(estimate, np.sum(q*q, axis=1), purchased.sum(axis=1), n, bound)
    expectation = float(mass@estimate)
    exact_variance = float(mass@((estimate-truth)**2))
    expected_variance_estimate = float(mass@(intervals['population_sample_variance']/n))
    covered = ((intervals['population_bernstein_lower'] <= truth+1e-12)
               & (intervals['population_bernstein_upper'] >= truth-1e-12))
    coverage = float(mass@covered)
    assert abs(expectation-truth) < 1e-12
    assert abs(exact_variance-expected_variance_estimate) < 1e-12
    assert coverage >= .95-1e-12
    return dict(question_clusters=n, atomic_states_per_question=len(atoms),
                enumerated_joint_states=len(indices), true_population_gain=truth,
                expected_estimate=expectation, exact_variance=exact_variance,
                expected_population_variance_estimate=expected_variance_estimate,
                exact_finite_bernstein_coverage=coverage,
                generation_repeats='Three identical copies within each iid cluster')


def main(output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    design = exhaustive_design_check()
    rng = np.random.default_rng(71)
    randomized = []
    for trial in range(30):
        d = rng.choice([-1., -1/3, 0., 1/3, 1.], size=5)
        m = rng.uniform(-0.9, 0.9, size=5)
        p = rng.uniform(0.05, 0.95, size=5)
        expectation, vhat_mean, variance, coverage, null_reject = 0., 0., 0., 0., 0.
        target = d.mean()
        from itertools import product
        for pattern in product((False, True), repeat=5):
            a = np.array(pattern)
            probability = float(np.prod(np.where(a, p, 1-p)))
            observed = np.where(a, d, np.nan)
            result = finite_pool(observed, m, p, a)
            expectation += probability * result["estimate"]
            variance += probability * (result["estimate"]-target)**2
            vhat_mean += probability * result["variance_estimate"]
            lo = result["intervals"]["finite_bernstein_lower"]
            hi = result["intervals"]["finite_bernstein_upper"]
            coverage += probability * (lo-1e-12 <= target <= hi+1e-12)
            null_reject += probability * (result["intervals"]["finite_bernstein_one_lower"] > target+1e-12)
        assert abs(expectation-target) < 1e-11
        assert abs(vhat_mean-variance) < 1e-11
        assert coverage >= 0.95 - 1e-12
        assert null_reject <= 0.05 + 1e-12
        randomized.append({"trial": trial, "target": float(target),
                           "bias": float(expectation-target), "variance_error": float(vhat_mean-variance),
                           "exact_two_sided_coverage": coverage,
                           "exact_one_sided_null_rejection": null_reject})
    result = {"status": "PASS", "primary_enumeration": design,
              "randomized_small_pool_exhaustive_checks": randomized,
              "ltt_native_qualification": ltt_check(),
              "population_exact_enumeration": population_check(),
              "interpretation": "Software and finite-case qualification; the general guarantee uses the stated analytic proof"}
    (output_directory / "INFERENCE_QUALIFICATION.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "random_cases": len(randomized),
                      "max_abs_bias": max(abs(x["bias"]) for x in randomized),
                      "min_exact_coverage": min(x["exact_two_sided_coverage"] for x in randomized)}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
