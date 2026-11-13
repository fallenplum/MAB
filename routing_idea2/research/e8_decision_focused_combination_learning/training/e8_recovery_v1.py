"""Versioned exact Clarabel backend for the frozen E8 SPO+ epigraph QP.

The frozen objective module is imported intact.  The only optional runtime
proxy replaces its fit_spo_qp solver binding.  No loss, regularization,
preprocessing, candidate set, selected configuration, or tie rule changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

import e8_training_v1 as frozen


CLARABEL_SETTINGS = dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9,
                         max_iter=1000, verbose=False, max_threads=1)
ORIGINAL_SOLVER = frozen.fit_spo_qp


def build_problem(phi, rewards, regularization, tie_rule):
    n, dimension = phi.shape
    weights = frozen.true_optimal_weights(rewards, tie_rule)
    coefficient_size = dimension*frozen.K
    quadratic = sparse.diags(np.r_[np.full(coefficient_size, 2*regularization), np.zeros(n)], format="csc")
    linear = np.r_[(-(2.0/n)*(phi.T@weights)).reshape(-1, order="F"), np.full(n, 1.0/n)]
    action_blocks = sparse.kron(sparse.eye(frozen.K, format="csc"), 2.0*sparse.csc_matrix(phi), format="csc")
    epigraph_blocks = sparse.vstack([-sparse.eye(n, format="csc")]*frozen.K, format="csc")
    constraint = sparse.hstack([action_blocks, epigraph_blocks], format="csc")
    upper = rewards.reshape(-1, order="F")
    return quadratic, linear, constraint, upper


def fit_spo_qp_clarabel(phi, rewards, regularization, tie_rule, settings):
    import clarabel
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    n, dimension = phi.shape
    quadratic, linear, constraint, upper = build_problem(phi, rewards, regularization, tie_rule)
    solver_settings = clarabel.DefaultSettings()
    for key, value in CLARABEL_SETTINGS.items():
        setattr(solver_settings, key, value)
    solver = clarabel.DefaultSolver(quadratic, linear, constraint, upper,
                                    [clarabel.NonnegativeConeT(n*frozen.K)], solver_settings)
    solved = solver.solve()
    primal, dual, slack = np.asarray(solved.x), np.asarray(solved.z), np.asarray(solved.s)
    coefficient = primal[:dimension*frozen.K].reshape(dimension, frozen.K, order="F")
    objective, _ = frozen.spo_plus_value_subgradient(coefficient, phi, rewards, regularization, tie_rule)
    epigraph_objective = float(solved.obj_val+rewards.max(axis=1).mean())
    constraint_violation = float(np.maximum(constraint@primal-upper, 0).max(initial=0.0))
    stationarity = quadratic@primal+linear+constraint.T@dual
    primal_equation = constraint@primal+slack-upper
    true_weights = frozen.true_optimal_weights(rewards, tie_rule)
    scores = phi@coefficient
    question_loss = (np.max(2*scores-rewards, axis=1)-2*np.sum(scores*true_weights, axis=1)
                     +rewards.max(axis=1))
    all_tied = np.all(rewards == rewards[:, :1], axis=1)
    success = solved.status == clarabel.SolverStatus.Solved
    diagnostics = dict(status=str(solved.status), success=bool(success), solver="Clarabel",
                       solver_version=clarabel.__version__, objective=objective,
                       epigraph_objective=epigraph_objective,
                       objective_discrepancy=abs(objective-epigraph_objective),
                       primal_residual=float(solved.r_prim), dual_residual=float(solved.r_dual),
                       constraint_violation=constraint_violation, iterations=int(solved.iterations),
                       primal_dual_objective_gap=float(abs(solved.obj_val-solved.obj_val_dual)),
                       stationarity_infinity_norm=float(np.max(np.abs(stationarity))),
                       primal_equation_infinity_norm=float(np.max(np.abs(primal_equation))),
                       complementarity_sum=float(np.dot(slack, dual)),
                       minimum_dual=float(np.min(dual)), minimum_slack=float(np.min(slack)),
                       true_tie_rule=tie_rule, surrogate_tie_rule="lexicographic",
                       all_twenty_tied_questions=int(all_tied.sum()),
                       all_twenty_tied_loss_contribution=float(np.sum(question_loss[all_tied])/n),
                       true_optimum_tie_fraction=float(np.mean((rewards == rewards.max(axis=1, keepdims=True)).sum(axis=1)>1)),
                       variables=dimension*frozen.K+n, constraints=n*frozen.K,
                       clarabel_settings=CLARABEL_SETTINGS,
                       numerical_backend_recovery=True, frozen_mathematical_objective_unchanged=True,
                       backend_cpu_seconds=time.process_time()-cpu_start,
                       backend_wall_seconds=time.perf_counter()-wall_start)
    return coefficient, diagnostics


def install_solver_proxy():
    if frozen.fit_spo_qp not in (ORIGINAL_SOLVER, fit_spo_qp_clarabel):
        raise RuntimeError("An unexpected E8 solver binding is already installed")
    frozen.fit_spo_qp = fit_spo_qp_clarabel
    return dict(proxy_binding="e8_training_v1.fit_spo_qp -> e8_recovery_v1.fit_spo_qp_clarabel",
                unchanged_functions=["train_fold", "fit_coefficients", "fit_gap_ridge", "ranknet_value_gradient",
                                     "listnet_value_gradient", "competition_weights", "legal_oof_m0"],
                mathematical_objective_unchanged=True)


def qualification(root, output):
    root, output = Path(root), Path(output)
    import osqp
    from e8_assets import fit_features, inner_splits, matched_mean_loss_lambda
    rng = np.random.default_rng(2026111320)
    comparisons = []
    for tie_rule in ("lexicographic", "uniform"):
        phi = np.column_stack((np.ones(24), rng.normal(size=(24, 3))))
        rewards = rng.integers(0, 2, size=(24, 3, 20)).mean(axis=1)
        regularization = .03
        before = osqp.OSQP
        captured = {}
        class CapturedSolver:
            def __init__(self):
                self.actual = before()
            def setup(self, **kwargs):
                captured.update(kwargs)
                self.actual.setup(**kwargs)
            def solve(self, **kwargs):
                return self.actual.solve(**kwargs)
        osqp.OSQP = CapturedSolver
        try:
            original_w, original = ORIGINAL_SOLVER(phi, rewards, regularization, tie_rule, frozen.DEFAULT_SETTINGS)
        finally:
            osqp.OSQP = before
        p, q, a, b = build_problem(phi, rewards, regularization, tie_rule)
        errors = {"P": float(abs(p-captured["P"]).max()),
                  "q": float(np.max(np.abs(q-captured["q"]))),
                  "A": float(abs(a-captured["A"]).max()),
                  "b": float(np.max(np.abs(b-captured["u"])))}
        recovered_w, recovered = fit_spo_qp_clarabel(phi, rewards, regularization, tie_rule, frozen.DEFAULT_SETTINGS)
        objective_error = abs(original["objective"]-recovered["objective"])
        assert max(errors.values()) == 0.0
        assert original["success"] and recovered["success"]
        assert objective_error < 1e-7
        assert recovered["primal_dual_objective_gap"] < 1e-7
        comparisons.append(dict(tie_rule=tie_rule, exact_matrix_errors=errors,
                                original=original, recovered=recovered, objective_error=objective_error,
                                max_coefficient_difference=float(np.max(np.abs(original_w-recovered_w)))))
    history_path = root / "private/qualification/fold_packs/fold0/history480.npz"
    with np.load(history_path, allow_pickle=False) as archive:
        history = {key: archive[key] for key in archive.files}
    candidates = []
    for fold, (fit_rows, _) in enumerate(inner_splits(history, np.arange(480), 20261113)):
        rewards = history["Y"][fit_rows].mean(axis=1)
        if int(np.all(rewards == rewards[:, :1], axis=1).sum()) == 166:
            candidates.append((fold, fit_rows, rewards))
    if len(candidates) != 1:
        raise RuntimeError("The logged failed inner region is not uniquely identified")
    fold, rows, rewards = candidates[0]
    feature_map = fit_features(history["E"][rows])
    phi = feature_map.phi(history["E"][rows])
    regularization = matched_mean_loss_lambda(len(rows))
    coefficient, diagnostic = fit_spo_qp_clarabel(phi, rewards, regularization, "uniform", frozen.DEFAULT_SETTINGS)
    assert diagnostic["success"]
    assert diagnostic["primal_dual_objective_gap"] < 1e-7
    assert diagnostic["constraint_violation"] < 1e-7
    result = dict(schema="E8_clarabel_recovery_qualification_v1", status="PASS",
                  original_source_sha256=hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
                  recovery_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  comparisons=comparisons,
                  actual_failure_region=dict(outer_fold=0, inner_fold=fold, n=320,
                                             rows_hash=frozen.array_hash(rows), phi_hash=frozen.array_hash(phi),
                                             reward_hash=frozen.array_hash(rewards),
                                             regularization=regularization, lambda_multiplier=1.0,
                                             true_tie_rule="uniform", diagnostic=diagnostic),
                  heldout_reward_access=False, physical_llm_calls=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["actual_failure_region"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--qualification-output", required=True)
    arguments = parser.parse_args()
    qualification(arguments.root, arguments.qualification_output)
