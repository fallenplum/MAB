"""Isolated Newton polishing of frozen E4 binary MAP references.

Original actors, selections and stored reference results are read-only. Only
historical labels and already-authorized B feedback enter the optimization.
Query inputs diagnose numerical policy changes without using Query rewards.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from e4_common import PRIVATE, PUBLIC, SEEDS, e4_splits, load_train, put, sha
from e4_binary import BinaryModel, link, nll_terms
from run_binary import fit_model


TOLERANCE = 1e-7
CHECKPOINTS = (1, 5, 20)


class CompositeObjective:
    """Exact penalized composite objective and Hessian, with shared history design."""

    def __init__(self, model, history_design=None):
        self.model = model
        self.x = model.X[model.history]
        self.basis = model.B
        self.count = model.rewards.sum(axis=1)
        self.repeats = model.rewards.shape[1]
        self.design = model.design(model.history).reshape(-1, model.dimension) if history_design is None else history_design
        if model.online_observations:
            self.extra_design = np.concatenate([model.observation(q, actions) for q, actions, _ in model.online_observations])
            self.extra_values = np.concatenate([values for _, _, values in model.online_observations])
        else:
            self.extra_design = np.empty((0, model.dimension))
            self.extra_values = np.empty(0)

    def evaluate(self, theta, hessian=False):
        coefficient = theta.reshape(self.model.question_dimension, self.model.action_dimension)
        score = self.x @ coefficient @ self.basis.T
        loss, gradient, curvature = nll_terms(score, self.count, self.repeats, self.model.family)
        value = float(loss.sum() + .5 * self.model.alpha * (theta @ theta))
        derivative = (self.x.T @ gradient @ self.basis).reshape(-1) + self.model.alpha * theta
        extra_curvature = np.empty(0)
        if len(self.extra_values):
            extra, extra_gradient, extra_curvature = nll_terms(
                self.extra_design @ theta, self.extra_values, 1, self.model.family)
            value += float(extra.sum())
            derivative += self.extra_design.T @ extra_gradient
        precision = None
        if hessian:
            precision = self.model.alpha * np.eye(self.model.dimension)
            precision += (self.design.T * curvature.ravel()) @ self.design
            if len(self.extra_values):
                precision += (self.extra_design.T * extra_curvature) @ self.extra_design
            precision = .5 * (precision + precision.T)
        return value, derivative, precision


def polish(objective, initial, tolerance=TOLERANCE, maxiter=12):
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    theta = np.asarray(initial, float).copy()
    before_value, before_gradient, _ = objective.evaluate(theta)
    iterations = []
    for iteration in range(maxiter + 1):
        value, gradient, precision = objective.evaluate(theta, hessian=True)
        infinity = float(np.max(np.abs(gradient)))
        if infinity < tolerance:
            covariance = cho_solve(cho_factor(precision), np.eye(len(theta)))
            covariance = .5 * (covariance + covariance.T)
            return theta, covariance, {"success": True, "iterations": len(iterations),
                "gradient_inf_before": float(np.max(np.abs(before_gradient))),
                "gradient_inf_after": infinity, "objective_before": before_value,
                "objective_after": value, "objective_reduction": before_value - value,
                "steps": iterations, "cpu_s": time.process_time() - started_cpu,
                "wall_s": time.perf_counter() - started_wall, "tolerance": tolerance}
        if iteration == maxiter:
            break
        direction = -cho_solve(cho_factor(precision), gradient)
        directional = float(gradient @ direction)
        if not directional < 0:
            raise RuntimeError("Newton direction is not a descent direction")
        step = 1.
        accepted = False
        roundoff_guard = False
        for backtrack in range(25):
            candidate = theta + step * direction
            trial_value, trial_gradient, _ = objective.evaluate(candidate)
            armijo = value + 1e-4 * step * directional
            if trial_value <= armijo:
                accepted = True
                break
            rounding = 16 * np.finfo(float).eps * max(1., abs(value))
            if (abs(step * directional) <= rounding and trial_value <= value + rounding
                    and np.max(np.abs(trial_gradient)) < infinity):
                accepted, roundoff_guard = True, True
                break
            step *= .5
        if not accepted:
            raise RuntimeError("Newton Armijo line search failed")
        iterations.append({"iteration": iteration + 1, "step": step, "backtracks": backtrack,
                           "gradient_inf_before": infinity,
                           "gradient_inf_after": float(np.max(np.abs(trial_gradient))),
                           "armijo_roundoff_guard": roundoff_guard})
        theta = candidate
    raise RuntimeError("Polishing did not attain gradient infinity below " + str(tolerance))


def _probability(model, rows, theta):
    coefficient = theta.reshape(model.question_dimension, model.action_dimension)
    return link(model.X[rows] @ coefficient @ model.B.T, model.family)


def _difference(model, rows, old_mean, old_cov, new_mean, new_cov):
    old_probability, new_probability = _probability(model, rows, old_mean), _probability(model, rows, new_mean)
    old_actions, new_actions = np.argmax(old_probability, axis=1), np.argmax(new_probability, axis=1)
    changed = np.flatnonzero(old_actions != new_actions)
    return {"mean_l2": float(np.linalg.norm(new_mean - old_mean)),
            "covariance_relative_frobenius": float(np.linalg.norm(new_cov - old_cov) / max(np.linalg.norm(new_cov), 1e-15)),
            "probability_max_absolute_difference": float(np.max(np.abs(new_probability - old_probability))),
            "probability_mean_absolute_difference": float(np.mean(np.abs(new_probability - old_probability))),
            "query_argmax_changes": len(changed), "query_questions": len(rows),
            "trace_before": float(np.trace(old_cov)), "trace_after": float(np.trace(new_cov))}, {
                "old_probability": old_probability, "refined_probability": new_probability,
                "changed_query_rows": np.asarray(rows)[changed],
                "old_changed_actions": old_actions[changed], "new_changed_actions": new_actions[changed]}


def _identity(selection_path, logger_path, original_online_path):
    source = Path(__file__).parent
    return {"selection_sha256": sha(selection_path), "logger_sha256": sha(logger_path),
            "original_online_sha256": sha(original_online_path),
            "code_sha256": {name: sha(source / name) for name in
                            ("audit_binary_optimization.py", "run_binary.py", "e4_binary.py", "e4_gaussian.py", "e4_metadata.py")}}


def worker(job):
    fold_index, predictor, mode, family = job
    selection_path = PRIVATE / "binary" / f"f{fold_index}_{predictor}_{mode}.json"
    logger_path = PRIVATE / "posterior" / f"f{fold_index}_{predictor}_N2_g0_o17_pair.json"
    original_online_path = PRIVATE / "binary_online" / f"f{fold_index}_{predictor}_{mode}_{family}_g0_o17_pair.json"
    identity = _identity(selection_path, logger_path, original_online_path)
    path = PRIVATE / "optimization" / f"f{fold_index}_{predictor}_{mode}_{family}.json"
    if path.exists() and path.with_suffix(".npz").exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity") == identity:
            return existing
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    detail = next(item for item in selection["models"] if item["family"] == family)
    old_online = json.loads(original_online_path.read_text(encoding="utf-8"))
    logger = json.loads(logger_path.read_text(encoding="utf-8"))
    data, _ = load_train()
    fold = e4_splits(data)["folds"][fold_index]
    base, metadata = fit_model(data, fold["history"], predictor, mode, family, detail["alpha"])
    if not isinstance(base, BinaryModel):
        raise TypeError("Optimization audit is restricted to binary models")
    evaluator = fold["stream"]
    index = selection["models"].index(detail)
    with np.load(selection_path.with_suffix(".npz"), allow_pickle=False) as cached:
        cached_pred = cached["predictions"][index]
    reproduction = float(np.max(np.abs(cached_pred - base.predict_many(evaluator))))
    if reproduction > 1e-10:
        raise ValueError("Reconstructed base model differs from frozen C predictions")
    objective = CompositeObjective(base)
    refined, covariance, initial_polish = polish(objective, base.mean)
    difference, arrays = _difference(base, evaluator, base.mean, base.cov, refined, covariance)
    group = {"fold": fold_index, "predictor": predictor, "mode": mode, "family": family,
             "alpha": detail["alpha"]}
    if difference["query_argmax_changes"]:
        print(json.dumps({"event": "ARGMAX_CHANGED", "role": "INITIAL_POLICY_QUERY120", **group,
                          "count": difference["query_argmax_changes"]}), flush=True)
    saved = {"initial_refined_mean": refined, "initial_refined_covariance": covariance,
             **{"initial_" + key: value for key, value in arrays.items()}}
    actor = deepcopy(base)
    checkpoints = []
    if len(logger["feedback"]) != 20 or [item["q"] for item in logger["feedback"]] != fold["audit_orders"][17].tolist():
        raise ValueError("Logger is not the original authorized twenty-Support path")
    for step, event in enumerate(logger["feedback"], 1):
        # Source scores are taken solely from the immutable paid logger receipt.
        actor.update(event["q"], event["actions"], event["raw_scores"])
        if step not in CHECKPOINTS:
            continue
        original = next(item for item in old_online["full_refit_comparisons"] if item["step"] == step)
        before_cpu, before_wall = time.process_time(), time.perf_counter()
        reference = actor.full_refit_online()
        if not reference.success:
            raise RuntimeError("Frozen L-BFGS reference could not be reproduced")
        reference_time = {"cpu_s": time.process_time() - before_cpu, "wall_s": time.perf_counter() - before_wall}
        current_objective = CompositeObjective(actor, objective.design)
        refined_mean, refined_cov, refined_info = polish(current_objective, reference.x)
        delta, detail_arrays = _difference(actor, fold["audit_query"], reference.x,
                                           reference.laplace_covariance, refined_mean, refined_cov)
        sequential, _ = _difference(actor, fold["audit_query"], actor.mean, actor.cov, refined_mean, refined_cov)
        if delta["query_argmax_changes"]:
            print(json.dumps({"event": "ARGMAX_CHANGED", "role": "REFERENCE_ONLY_QUERY100", **group,
                              "step": step, "count": delta["query_argmax_changes"]}), flush=True)
        checkpoints.append({"step": step, "polish": refined_info, "reference_change": delta,
                            "sequential_vs_refined": sequential, "frozen_original_comparison": original,
                            "reproduced_LBFGS_gradient_inf": float(np.max(np.abs(reference.jac))),
                            "LBFGS_reproduction": reference_time})
        saved[f"step{step}_refined_mean"] = refined_mean
        saved[f"step{step}_refined_covariance"] = refined_cov
        saved.update({f"step{step}_" + key: value for key, value in detail_arrays.items()})
    result = {"identity": identity, "meta": group, "base_prediction_reproduction_error": reproduction,
              "initial": {"polish": initial_polish, "change": difference}, "checkpoints": checkpoints,
              "metadata": metadata, "actor_changed": False, "original_files_overwritten": False,
              "query_rewards_used": False, "new_feedback_reveals": 0, "new_llm_calls": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.npz")
    np.savez_compressed(pending, **saved)
    os.replace(pending, path.with_suffix(".npz"))
    put(path, result)
    return result


def summarize(groups):
    initial = [group["initial"] for group in groups]
    checkpoints = [row for group in groups for row in group["checkpoints"]]
    if len(initial) != 40 or len(checkpoints) != 120:
        raise ValueError("Expected forty initial models and 120 prespecified references")
    def totals(rows, role):
        polish_rows = [row["polish"] for row in rows]
        changes = [row[role] for row in rows]
        return {"cases": len(rows), "gradient_inf_before_max": max(row["gradient_inf_before"] for row in polish_rows),
                "gradient_inf_after_max": max(row["gradient_inf_after"] for row in polish_rows),
                "polish_cpu_s_sum": sum(row["cpu_s"] for row in polish_rows),
                "polish_wall_s_sum": sum(row["wall_s"] for row in polish_rows),
                "newton_iterations_max": max(row["iterations"] for row in polish_rows),
                "query_argmax_changed_cases": sum(row["query_argmax_changes"] > 0 for row in changes),
                "query_argmax_changed_decisions": sum(row["query_argmax_changes"] for row in changes),
                "probability_max_absolute_difference": max(row["probability_max_absolute_difference"] for row in changes),
                "mean_l2_max": max(row["mean_l2"] for row in changes),
                "covariance_relative_frobenius_max": max(row["covariance_relative_frobenius"] for row in changes)}
    contexts = []
    for family in ("logistic", "probit"):
        for step in CHECKPOINTS:
            rows = [row for group in groups if group["meta"]["family"] == family
                    for row in group["checkpoints"] if row["step"] == step]
            contexts.append({"family": family, "step": step, **totals(rows, "reference_change"),
                             "sequential_vs_refined": {field: {"mean": float(np.mean([row["sequential_vs_refined"][field] for row in rows])),
                                                               "maximum": float(np.max([row["sequential_vs_refined"][field] for row in rows]))}
                                                          for field in ("mean_l2", "covariance_relative_frobenius",
                                                                        "probability_mean_absolute_difference", "query_argmax_changes")}})
    return {"status": "COMPLETE", "schema": "e4_binary_newton_optimization_sensitivity_v1",
            "gradient_tolerance": TOLERANCE, "initial_models": totals(initial, "change"),
            "checkpoint_references": totals(checkpoints, "reference_change"), "by_family_step": contexts,
            "original_C_selection_and_actor_results_unchanged": True,
            "query_rewards_used": False, "new_physical_model_calls": 0,
            "scope": "Independent numerical correction of insufficiently precise MAP references; finite Newton tolerance remains explicit",
            "policy_reexecution_performed": False,
            "private_derived_results": "private/optimization; original reference results retained separately"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    jobs = [(fold, predictor, mode, family) for fold in range(5) for predictor in ("P1", "P2")
            for mode in ("text-only", "task-known") for family in ("logistic", "probit")]
    started = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            group = future.result()
            groups.append(group)
            print(json.dumps({"event": "GROUP_COMPLETE", **group["meta"], "completed": len(groups)}), flush=True)
            put(PRIVATE / "optimization" / "PROGRESS.json", {"completed_groups": len(groups), "required_groups": 40})
    groups.sort(key=lambda group: tuple(group["meta"][key] for key in ("fold", "predictor", "mode", "family")))
    report = summarize(groups)
    report["wall_elapsed_s"] = time.perf_counter() - started
    report["audit_source_sha256"] = sha(Path(__file__))
    put(PUBLIC / "BINARY_OPTIMIZATION_ACCURACY.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
