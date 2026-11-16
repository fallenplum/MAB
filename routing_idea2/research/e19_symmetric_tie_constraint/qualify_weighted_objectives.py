"""Small synthetic checks for the E19 weighted exact-QP operator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

import weighted_objectives as weighted


def run_checks(e17_objectives_path):
    phi = np.asarray([[1.0, -1.0], [1.0, 1.0]])
    rewards = np.asarray([[0.5, 0.5], [0.9, 0.1]])
    coefficient = np.asarray([[0.22, -0.13], [-0.07, 0.19]])
    regularization = 0.1
    epsilon = 1e-6
    records = []
    for beta in (0.0, 1.0):
        value, gradient = weighted.objective_value_and_gradient(
            coefficient, phi, rewards, regularization, beta
        )
        finite_difference = np.empty_like(coefficient)
        for index in np.ndindex(coefficient.shape):
            plus, minus = coefficient.copy(), coefficient.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            f_plus = weighted.objective_components(plus, phi, rewards, regularization, beta)["objective"]
            f_minus = weighted.objective_components(minus, phi, rewards, regularization, beta)["objective"]
            finite_difference[index] = (f_plus - f_minus) / (2.0 * epsilon)
        error = float(np.max(np.abs(finite_difference - gradient)))
        if error > 1e-7:
            raise AssertionError("Finite-difference gradient mismatch")
        records.append(dict(check="finite_difference", beta=beta, error=error, objective=value))
    components = weighted.objective_components(coefficient, phi, rewards, regularization, 0.0)
    if abs(components["identity_F1_minus_F0_minus_JT"]) > 1e-12:
        raise AssertionError("F1-F0=J_T failed")
    records.append(dict(check="exact_objective_decomposition", **components))

    fitted = {}
    for beta in (0.0, 1.0):
        original, diagnostics = weighted.fit_weighted_tspo(phi, rewards, regularization, beta)
        reverse, reverse_diagnostics = weighted.fit_weighted_tspo(
            phi, rewards, regularization, beta, permutation=np.asarray([1, 0])
        )
        centered = lambda w: phi @ w - (phi @ w).mean(axis=1, keepdims=True)
        error = float(np.max(np.abs(centered(original) - centered(reverse))))
        if error > 1e-7:
            raise AssertionError("Reverse physical-action mapping differs")
        if diagnostics["original_question_denominator"] != 2 or diagnostics["epigraph_rows_retained"] != 2:
            raise AssertionError("All original rows must remain in the QP")
        if diagnostics["regularization"] != regularization:
            raise AssertionError("The L2 coefficient changed across beta branches")
        fitted[beta] = original
        records.append(dict(check="exact_QP_and_reverse_mapping", beta=beta,
                            centered_score_error=error, diagnostics=diagnostics,
                            reverse_diagnostics=reverse_diagnostics))

    spec = importlib.util.spec_from_file_location("e19_frozen_e17_operator_check", e17_objectives_path)
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)
    old, diagnostics = original_module.fit_spo(phi, rewards, regularization, rule="uniform")
    error = float(np.max(np.abs(phi @ (fitted[1.0] - old))))
    if error > 1e-7:
        raise AssertionError("Beta1 does not recover the original E17 uniform objective")
    records.append(dict(check="beta1_original_E17_uniform_recovery", score_error=error,
                        original_source_sha256=hashlib.sha256(Path(e17_objectives_path).read_bytes()).hexdigest(),
                        original_diagnostics=diagnostics))
    for beta in (0.0, 1.0):
        zero, diagnostics = weighted.fit_weighted_tspo(
            phi, np.asarray([[0.5, 0.5], [0.0, 0.0]]), regularization, beta
        )
        if np.linalg.norm(zero) > 1e-7:
            raise AssertionError("Pure full ties do not recover the penalized zero model")
        records.append(dict(check="all_tie_zero_optimum", beta=beta,
                            coefficient_norm=float(np.linalg.norm(zero)), diagnostics=diagnostics))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e17-objectives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source = Path(weighted.__file__)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    records = run_checks(args.e17_objectives)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("The operator changed during the synthetic check")
    result = dict(schema="e19_weighted_objectives_synthetic_checks_v1", status="PASS",
                  source_sha256=source_sha, tests=len(records),
                  synthetic_tiny_QP_fits=7, real_question_reads=0,
                  real_predictor_fits=0, fresh_or_sealed_rewards_read=0,
                  new_llm_calls=0, checks=records,
                  wall_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], tests=result["tests"],
                          source_sha256=source_sha, output=str(args.output))))


if __name__ == "__main__":
    main()
