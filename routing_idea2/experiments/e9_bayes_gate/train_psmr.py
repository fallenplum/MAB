"""Independent Bernoulli heads for the frozen E9 proper-score experiment.

Only the caller's legal fitting questions are accepted.  The design already
contains its constant column, which receives the same L2 penalty as other rows.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def reward_mean(rewards):
    """Return n-by-action means and the three-repeat Brier constant."""
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim not in (2, 3) or not values.shape[0] or not values.shape[-1]:
        raise ValueError("rewards must be a nonempty n-by-action or n-by-repeat-by-action array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("rewards must be finite and lie in [0, 1]")
    if values.ndim == 2:
        return values, 0.0
    if values.shape[1] != 3:
        raise ValueError("the frozen experiment requires exactly three generation repeats")
    means = values.mean(axis=1)
    return means, float(np.mean((values - means[:, None, :]) ** 2))


def _validate_design(design, n_questions):
    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != n_questions or not design.shape[1]:
        raise ValueError("design must have one row per fitting question")
    if not np.isfinite(design).all():
        raise ValueError("design contains nonfinite values")
    return design


def objective_gradient(flat_weights, design, targets, l2, loss):
    """Data term divided by n * actions, plus l2 * squared weight norm."""
    n_questions, n_actions = targets.shape
    weights = np.asarray(flat_weights, dtype=np.float64).reshape(design.shape[1], n_actions)
    logits = design @ weights
    probabilities = expit(logits)
    difference = probabilities - targets
    if loss == "brier":
        data_loss = float(np.mean(difference ** 2))
        score_gradient = 2.0 * difference * probabilities * (1.0 - probabilities)
    elif loss == "bce":
        data_loss = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
        score_gradient = difference
    else:
        raise ValueError("loss must be brier or bce")
    regularization = float(l2 * np.sum(weights ** 2))
    gradient = design.T @ score_gradient / (n_questions * n_actions) + 2.0 * l2 * weights
    return data_loss + regularization, gradient.ravel()


def fit_psmr(design, rewards, l2, loss="brier"):
    """Fit one deterministic 17-by-20 head with stable float64 L-BFGS-B.

    The Brier objective on repeat means differs from the original repeated
    binary-reward objective by a parameter-independent constant.  Both are
    recorded. BCE is exactly equal after averaging the three observations.
    BCE is convex in the weights. Sigmoid-Brier is generally nonconvex.
    """
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    targets, repeat_constant = reward_mean(rewards)
    design = _validate_design(design, len(targets))
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative")
    if loss not in ("brier", "bce"):
        raise ValueError("loss must be brier or bce")
    initial = np.zeros((design.shape[1], targets.shape[1]), dtype=np.float64)
    initial_objective, _ = objective_gradient(initial.ravel(), design, targets, l2, loss)
    result = minimize(
        objective_gradient,
        initial.ravel(),
        args=(design, targets, float(l2), loss),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "gtol": 1e-8, "ftol": 1e-12, "maxls": 50},
    )
    weights = result.x.reshape(initial.shape)
    objective, gradient = objective_gradient(result.x, design, targets, l2, loss)
    penalty = float(l2 * np.sum(weights ** 2))
    data_loss = objective - penalty
    original_data_loss = data_loss + (repeat_constant if loss == "brier" else 0.0)
    diagnostics = {
        "family": "PSMR",
        "loss": loss,
        "n_questions": len(targets),
        "n_actions": targets.shape[1],
        "n_parameters": weights.size,
        "normalization": "data divided by n_questions * n_actions",
        "constant_column_penalized": True,
        "l2": float(l2),
        "initial_objective": float(initial_objective),
        "objective": float(objective),
        "data_loss_on_question_means": float(data_loss),
        "data_loss_on_three_repeated_rewards": float(original_data_loss),
        "brier_repeat_constant": repeat_constant,
        "regularization": penalty,
        "weight_norm": float(np.linalg.norm(weights)),
        "gradient_l2": float(np.linalg.norm(gradient)),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "optimizer": "float64 scipy L-BFGS-B, zero initialization",
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "convex_objective": loss == "bce",
        "cpu_seconds": float(time.process_time() - cpu_start),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    if not np.isfinite(weights).all() or not np.isfinite(objective):
        raise FloatingPointError("PSMR optimization produced nonfinite output")
    return {"weights": weights, "diagnostics": diagnostics}


def predict_psmr(design, weights):
    design = np.asarray(design, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if design.ndim != 2 or weights.ndim != 2 or design.shape[1] != weights.shape[0]:
        raise ValueError("prediction design and weight dimensions disagree")
    if not np.isfinite(design).all() or not np.isfinite(weights).all():
        raise ValueError("prediction input contains nonfinite values")
    return expit(design @ weights)
