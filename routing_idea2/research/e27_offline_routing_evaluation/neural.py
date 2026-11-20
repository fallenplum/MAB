"""Fit paired tanh ListNet reward heads using only each fold's history."""
from __future__ import annotations

from time import perf_counter, process_time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


WIDTH = 16
REGULARIZATION = 1e-4
SEEDS = (1701, 1702, 1703)
BETA_ORDER = (1, 0)
OPTIMIZER = {"gtol": 1e-9, "ftol": 1e-13, "maxiter": 2000, "maxls": 50, "maxcor": 20}
GRADIENT_TOLERANCE = 2e-6


def objective(flat, phi, target, target_log, row_weight):
    input_size = phi.shape[1]
    actions = target.shape[1]
    first_size = input_size * WIDTH
    hidden = flat[:first_size].reshape(input_size, WIDTH)
    output = flat[first_size:].reshape(WIDTH + 1, actions)
    activation = np.tanh(phi @ hidden)
    scores = activation @ output[1:] + output[0]
    prediction_log = scores - logsumexp(scores, axis=1, keepdims=True)
    losses = np.sum(target * (target_log - prediction_log), axis=1)
    objective_value = float(row_weight @ losses / len(phi) + REGULARIZATION * (flat @ flat))
    score_gradient = row_weight[:, None] * (np.exp(prediction_log) - target) / len(phi)
    grad_hidden = phi.T @ ((score_gradient @ output[1:].T) * (1 - activation**2))
    grad_output = np.vstack((score_gradient.sum(axis=0), activation.T @ score_gradient))
    gradient = np.concatenate((grad_hidden.ravel(), grad_output.ravel())) + 2 * REGULARIZATION * flat
    return objective_value, gradient, scores, losses


def initialize(seed, inputs, actions):
    rng = np.random.default_rng(seed)
    hidden = np.vstack((np.zeros(WIDTH), rng.normal(0, 0.25, (inputs - 1, WIDTH))))
    output = np.vstack((np.zeros(actions), rng.normal(0, 0.25, (WIDTH, actions))))
    return np.concatenate((hidden.ravel(), output.ravel()))


def target_distribution(rewards):
    target_log = rewards - logsumexp(rewards, axis=1, keepdims=True)
    return np.exp(target_log), target_log


def check_gradient():
    """Verify the gradient of both endpoints and uniform-target tie behavior."""
    rng = np.random.default_rng(2701)
    phi = np.column_stack((np.ones(8), rng.normal(size=(8, 3))))
    rewards = rng.integers(0, 4, size=(8, 5)) / 3
    rewards[0] = 1
    target, target_log = target_distribution(rewards)
    if not np.allclose(target[0], 1 / rewards.shape[1], atol=1e-15, rtol=0):
        raise AssertionError("A tied training row must have a uniform target")
    tied = np.all(rewards == rewards[:, :1], axis=1)
    flat = initialize(2701, phi.shape[1], rewards.shape[1])
    maximum_error = 0.0
    for beta in BETA_ORDER:
        weight = np.where(tied, beta, 1.0)
        _, gradient, _, _ = objective(flat, phi, target, target_log, weight)
        for index in range(flat.size):
            plus, minus = flat.copy(), flat.copy()
            step = 1e-6
            plus[index] += step
            minus[index] -= step
            difference = (
                objective(plus, phi, target, target_log, weight)[0]
                - objective(minus, phi, target, target_log, weight)[0]
            ) / (2 * step)
            maximum_error = max(maximum_error, abs(difference - gradient[index]))
    if maximum_error > 2e-8:
        raise AssertionError(f"Neural ListNet gradient error: {maximum_error}")
    return float(maximum_error)


def fit_neural_listnet(phi_train, rewards, phi_eval, *, seed, beta):
    """Return held-out scores and training diagnostics for a single endpoint."""
    phi, rewards, evaluation = [np.asarray(x, dtype=np.float64) for x in (phi_train, rewards, phi_eval)]
    if phi.ndim != 2 or phi.shape[1] != 17 or rewards.shape != (len(phi), 20) or not len(phi):
        raise ValueError("Neural ListNet requires history features and all twenty history rewards")
    if evaluation.ndim != 2 or evaluation.shape[1] != 17:
        raise ValueError("Expected intercept plus sixteen PCA features at evaluation")
    if not all(np.isfinite(x).all() for x in (phi, evaluation, rewards)):
        raise ValueError("Features and history rewards must be finite")
    if not (np.all(phi[:, 0] == 1) and np.all(evaluation[:, 0] == 1) and np.all((0 <= rewards) & (rewards <= 1))):
        raise ValueError("Invalid intercept or history reward range")
    if seed not in SEEDS or beta not in BETA_ORDER:
        raise ValueError("Select an optimizer seed and intervention endpoint from the frozen protocol")

    started_cpu, started_wall = process_time(), perf_counter()
    target, target_log = target_distribution(rewards)
    tied = np.all(rewards == rewards[:, :1], axis=1)
    row_weight = np.where(tied, beta, 1.0)
    initial = initialize(seed, phi.shape[1], rewards.shape[1])

    def optimize(flat):
        value, gradient, _, _ = objective(flat, phi, target, target_log, row_weight)
        return value, gradient

    initial_value = optimize(initial)[0]
    fitted = minimize(optimize, initial, jac=True, method="L-BFGS-B", options=OPTIMIZER)
    value, gradient, _, losses = objective(fitted.x, phi, target, target_log, row_weight)
    gradient_inf = float(np.max(np.abs(gradient)))
    if (not fitted.success or not np.isfinite(value) or not np.isfinite(gradient_inf)
            or gradient_inf > GRADIENT_TOLERANCE or value > initial_value + 1e-12):
        raise RuntimeError(
            f"Neural ListNet optimizer failed for seed {seed}, beta {beta}: "
            f"{fitted.message}, gradient_inf={gradient_inf:.3g}"
        )
    hidden = fitted.x[:phi.shape[1] * WIDTH].reshape(phi.shape[1], WIDTH)
    output = fitted.x[phi.shape[1] * WIDTH:].reshape(WIDTH + 1, rewards.shape[1])
    scores = np.tanh(evaluation @ hidden) @ output[1:] + output[0]
    if not np.isfinite(scores).all():
        raise FloatingPointError("Neural ListNet generated nonfinite held-out scores")
    profile = dict(
        seed=int(seed), beta=int(beta), width=WIDTH, regularization=REGULARIZATION,
        iterations=int(fitted.nit), objective_evaluations=int(fitted.nfev), optimizer_status=int(fitted.status),
        initial_train_objective=initial_value, final_train_objective=value, gradient_inf=gradient_inf,
        coefficient_dimension=int(fitted.x.size), history_questions=len(phi), full_tied_history_questions=int(tied.sum()),
        effective_question_weight_sum=float(row_weight.sum()),
        tied_train_kl_per_history_question=float(losses[tied].sum() / len(phi)),
        non_tied_train_kl_per_history_question=float(losses[~tied].sum() / len(phi)),
        cpu_seconds=process_time() - started_cpu, wall_seconds=perf_counter() - started_wall,
    )
    return scores, profile
