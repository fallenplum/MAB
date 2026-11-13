"""Frozen E9 linear reward and cost-sensitive heads.

Training accepts one history row per original question.  The caller owns
history-only feature fitting and the task-balanced inner selection.  Routing
requires semantic action keys built from the frozen model member aliases.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from csce_operators import (
    csce_value_gradient,
    semantic_argmax,
    validate_training_inputs,
)


ACTION_COUNT = 20
FEATURE_COUNT = 17
RAW_ALPHA = 100.0
MAX_ITERATIONS = 2000
FUNCTION_TOLERANCE = 1e-12
GRADIENT_TOLERANCE = 1e-7


def _process_memory():
    """Return process RSS and the OS process-lifetime peak where available."""
    import psutil

    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", None)
    return int(memory.rss), None if peak is None else int(peak)


def _training_arrays(phi, labels):
    phi = np.asarray(phi, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != FEATURE_COUNT or len(phi) == 0:
        raise ValueError("Phi must have one nonempty original-question row and 17 columns")
    if labels.shape != (len(phi), ACTION_COUNT):
        raise ValueError("Labels must have one original-question row and 20 actions")
    if not np.isfinite(phi).all() or not np.isfinite(labels).all():
        raise ValueError("Features and labels must be finite float64 arrays")
    if np.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError("Historical action reward labels must lie in [0,1]")
    return phi, labels


def mse_value_gradient(coefficient, phi, labels, regularization):
    """Mean squared error over original questions and all twenty outputs."""
    phi, labels = _training_arrays(phi, labels)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.shape != (FEATURE_COUNT, ACTION_COUNT):
        raise ValueError("Coefficient must have shape (17,20)")
    if not np.isfinite(coefficient).all():
        raise ValueError("Coefficient must be finite")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("Regularization must be finite and nonnegative")
    error = phi @ coefficient - labels
    denominator = len(phi) * ACTION_COUNT
    data_loss = np.sum(error * error) / denominator
    gradient = 2.0 * (phi.T @ error) / denominator
    gradient += 2.0 * regularization * coefficient
    penalty = regularization * np.sum(coefficient * coefficient)
    return float(data_loss + penalty), gradient


def diagnose_head(family, coefficient, phi, labels, mult):
    """Recompute the frozen objective and stationarity for any supplied head."""
    phi, labels = _training_arrays(phi, labels)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.shape != (FEATURE_COUNT, ACTION_COUNT):
        raise ValueError("Coefficient must have shape (17,20)")
    family = str(family).upper()
    if family not in {"B", "C", "D"}:
        raise ValueError("Family must be B, C, or D")
    mult = float(mult)
    if not np.isfinite(mult) or mult <= 0.0:
        raise ValueError("The regularization multiplier must be finite and positive")
    raw_alpha = RAW_ALPHA * mult
    regularization = raw_alpha / (3.0 * len(phi) * ACTION_COUNT)
    objective = mse_value_gradient if family == "B" else csce_value_gradient
    value, gradient = objective(coefficient, phi, labels, regularization)
    coefficient_squared_norm = float(np.sum(coefficient * coefficient))
    penalty = regularization * coefficient_squared_norm
    gradient_infinity_norm = float(np.max(np.abs(gradient)))
    finite = bool(
        np.isfinite(value)
        and np.isfinite(coefficient).all()
        and np.isfinite(gradient).all()
    )
    gradient_tolerance_met = bool(finite and gradient_infinity_norm <= GRADIENT_TOLERANCE)
    return {
        "family": family,
        "objective_name": "MSE" if family == "B" else "CSCE",
        "finite": finite,
        "accepted": gradient_tolerance_met,
        "objective": float(value),
        "data_loss": float(value - penalty),
        "regularization_term": float(penalty),
        "regularization": float(regularization),
        "raw_alpha": float(raw_alpha),
        "multiplier": mult,
        "coefficient_norm": float(np.sqrt(coefficient_squared_norm)),
        "gradient_infinity_norm": gradient_infinity_norm,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "gradient_tolerance_met": gradient_tolerance_met,
        "normalization": "n_original_questions*20",
        "constant_penalized": True,
        "questions": int(len(phi)),
        "features": FEATURE_COUNT,
        "actions": ACTION_COUNT,
        "dtype": str(coefficient.dtype),
    }


def fit_head(family, phi, labels, mult):
    """Fit B/Ridge or C,D/CSCE under the frozen E9 objective.

    The regularization coefficient is 100*mult/(3*n*20).  Every coefficient,
    including the constant column, receives the same L2 penalty.  Diagnostics
    expose optimizer status and the independently recomputed gradient check.
    Callers must require ``accepted`` before using a fit in formal results.
    """
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    rss_before, peak_before = _process_memory()
    phi, labels = _training_arrays(phi, labels)
    family = str(family).upper()
    if family not in {"B", "C", "D"}:
        raise ValueError("Family must be B, C, or D")
    mult = float(mult)
    if not np.isfinite(mult) or mult <= 0.0:
        raise ValueError("The regularization multiplier must be finite and positive")
    raw_alpha = RAW_ALPHA * mult
    regularization = raw_alpha / (3.0 * len(phi) * ACTION_COUNT)

    if family == "B":
        normal = phi.T @ phi + (raw_alpha / 3.0) * np.eye(FEATURE_COUNT)
        coefficient = np.linalg.solve(normal, phi.T @ labels)
        optimizer_success = True
        status = 0
        message = "Closed-form penalized Ridge normal equations solved"
        iterations = 1
        evaluations = 1
        solver = "numpy.linalg.solve"
    else:
        shape = (FEATURE_COUNT, ACTION_COUNT)

        def objective(flat):
            value, gradient = csce_value_gradient(
                flat.reshape(shape), phi, labels, regularization
            )
            return value, gradient.ravel()

        result = minimize(
            objective,
            np.zeros(shape, dtype=np.float64).ravel(),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": MAX_ITERATIONS,
                "ftol": FUNCTION_TOLERANCE,
                "gtol": GRADIENT_TOLERANCE,
                "maxls": 50,
            },
        )
        coefficient = result.x.reshape(shape)
        optimizer_success = bool(result.success)
        status = int(result.status)
        message = str(result.message)
        iterations = int(result.nit)
        evaluations = int(result.nfev)
        solver = "scipy.optimize.minimize/L-BFGS-B"

    diagnostics = diagnose_head(family, coefficient, phi, labels, mult)
    rss_after, peak_after = _process_memory()
    recorded_peaks = [value for value in (peak_before, peak_after) if value is not None]
    peak_ram_bytes = max(recorded_peaks) if recorded_peaks else max(rss_before, rss_after)
    diagnostics.update({
        "solver": solver,
        "success": optimizer_success,
        "optimizer_success": optimizer_success,
        "accepted": bool(optimizer_success and diagnostics["gradient_tolerance_met"]),
        "status": status,
        "message": message,
        "iterations": iterations,
        "function_evaluations": evaluations,
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "peak_ram_bytes": int(peak_ram_bytes),
        "peak_ram_scope": (
            "OS process-lifetime peak working set"
            if recorded_peaks else "maximum of fit-entry and fit-exit process RSS"
        ),
        "lbfgs_options": None if family == "B" else {
            "maxiter": MAX_ITERATIONS,
            "ftol": FUNCTION_TOLERANCE,
            "gtol": GRADIENT_TOLERANCE,
            "maxls": 50,
        },
    })
    return coefficient, diagnostics


def routed_accuracy(scores, trueY, task, semantic_keys):
    """Evaluate selected actions with 50/50 GSM8K and BBH question means.

    ``trueY`` accepts [n,20] reward means or the frozen [n,3,20] repeat array.
    BBH is averaged over original questions.  Exact score ties use the given
    comparable semantic keys, so array relabeling preserves semantic choices.
    """
    scores = np.asarray(scores, dtype=np.float64)
    rewards = np.asarray(trueY, dtype=np.float64)
    task = np.asarray(task)
    if rewards.ndim == 3 and rewards.shape[1:] == (3, ACTION_COUNT):
        rewards = rewards.mean(axis=1)
    if scores.ndim != 2 or scores.shape[1] != ACTION_COUNT or len(scores) == 0:
        raise ValueError("Scores must have one nonempty original-question row and 20 actions")
    if rewards.shape != scores.shape or task.shape != (len(scores),):
        raise ValueError("Scores, original-question rewards, and task labels must align")
    if not np.isfinite(rewards).all() or np.any((rewards < 0.0) | (rewards > 1.0)):
        raise ValueError("Evaluation action rewards must be finite in [0,1]")
    if set(task.tolist()) != {"gsm8k", "bbh"}:
        raise ValueError("Task-balanced accuracy requires both frozen tasks gsm8k and bbh")
    selected = semantic_argmax(scores, semantic_keys)
    per_question = rewards[np.arange(len(scores)), selected]
    task_accuracy = {
        name: float(per_question[task == name].mean())
        for name in ("gsm8k", "bbh")
    }
    return {
        "balanced_accuracy": float(np.mean(list(task_accuracy.values()))),
        "overall_accuracy": float(per_question.mean()),
        "task_accuracy": task_accuracy,
        "selected_actions": selected.tolist(),
        "per_question_accuracy": per_question.tolist(),
        "task_question_counts": {
            name: int(np.sum(task == name)) for name in ("gsm8k", "bbh")
        },
        "question_count": int(len(scores)),
        "weighting": "0.5*GSM8K_question_mean+0.5*BBH_question_mean",
    }
