"""Synthetic checks for matched masks and row-weighted objective fidelity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from geometry import analyze
from row_weighted import fit
from run_audit import controls, source_binding, write

P3 = Path(__file__).resolve().parent.parent / "e21_p3_external_mechanism"
sys.path.insert(0, str(P3))
from endpoints import fit_endpoint_pair


def qualify():
    generator = np.random.default_rng(2026111808)
    phi = np.column_stack((np.ones(24), generator.normal(size=(24, 3))))
    rewards = generator.integers(0, 2, size=(24, 6)).astype(float)
    rewards[:13] = 1
    tied = np.all(rewards == rewards[:, :1], axis=1)
    labels = np.array(["gsm8k"]*12+["bbh"]*12)
    subtasks = np.array([""]*12+["logic"]*12)
    hashes = np.array([f"synthetic{index}" for index in range(24)])
    weights, _ = controls(0, phi, rewards, labels, subtasks, hashes)
    repeated, _ = controls(0, phi, rewards, labels, subtasks, hashes)
    checks = dict(
        deterministic_masks=all(np.array_equal(weights[key], repeated[key]) for key in weights),
        equal_effective_mass=all(abs(value.sum()-(~tied).sum()) < 1e-12
                                 for key, value in weights.items() if key != "B"),
        matched_masks_binary=all(np.isin(value, (0, 1)).all() for key, value in weights.items()
                                if key != "Uniform"),
        uniform_mass=np.allclose(weights["Uniform"], (~tied).mean()),
    )
    summaries = {}
    for family in ("tspo", "listnet"):
        originals, _ = fit_endpoint_pair(family, phi, rewards, regularization=0.01)
        recovered, gradients = [], []
        for index, name in enumerate(("B", "C")):
            coefficient, diag, gradient = fit(family, phi, rewards, 0.01, weights[name])
            checks[family+name+"_original_match"] = bool(np.allclose(coefficient, originals[index], rtol=0, atol=1e-8))
            recovered.append(coefficient)
            gradients.append(gradient)
        geometry, _ = analyze(family, phi, phi, rewards, 0.01, *recovered, *gradients)
        checks[family+"_geometry_qualified"] = geometry["kkt_residual_inf"] < 1e-7
        zero, _, _ = fit(family, phi, rewards, 0.01, np.zeros(24))
        checks[family+"_all_zero_weight_zero_model"] = bool(np.linalg.norm(zero) < 1e-7)
        fractional, diag, gradient = fit(family, phi, rewards, 0.01, weights["Uniform"])
        checks[family+"_fractional_stationarity"] = bool(np.abs(
            phi.T @ gradient/24 + 0.02*fractional).max() < 1e-7)
        checks[family+"_fractional_objective_qualified"] = diag["status"] == "Solved"
        summaries[family] = dict(kkt_residual_inf=geometry["kkt_residual_inf"],
                                score_identity_error=geometry["equilibrium_score_identity_max_error"])
    if not all(checks.values()):
        raise AssertionError("Synthetic audit qualification failed")
    return dict(schema="mab_specificity_qualification_v1", status="PASS", checks=checks,
                geometry=summaries, synthetic_head_fits=12, source_binding=source_binding(),
                real_reward_reads=0, new_llm_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify()
    write(args.output, result)
    print(json.dumps(dict(status=result["status"], checks=len(result["checks"]))))
