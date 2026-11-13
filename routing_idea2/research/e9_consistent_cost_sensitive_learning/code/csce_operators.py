"""Existing cost-sensitive cross-entropy for twenty-action reward routing.

The reward-to-cost mapping is c=1-r. Mozannar and Sontag (ICML2020),
Equation3 and Proposition1, use max(c)-c, which becomes r-min(r).
The numerical kernel permits smaller synthetic action sets, while retaining
the formal n*20 normalization. Production callers must enforce20 outputs.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

NORMALIZATION_ACTIONS = 20


def reward_weights(rewards):
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2 or len(rewards) == 0 or rewards.shape[1] < 2:
        raise ValueError('One finite reward vector per original question is required')
    if not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1)):
        raise ValueError('Rewards must be finite in[0,1]')
    weights = rewards - rewards.min(axis=1, keepdims=True)
    return weights, weights.sum(axis=1)


def csce_value_gradient(coefficient, phi, rewards, regularization):
    """Unnormalized per-row reward weights, then fixed n*20 normalization."""
    phi = np.asarray(phi, dtype=np.float64)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    weights, total = reward_weights(rewards)
    if phi.ndim != 2 or coefficient.shape != (phi.shape[1], weights.shape[1]):
        raise ValueError('Predictor and reward dimensions differ')
    if len(phi) != len(weights) or not np.isfinite(phi).all() or not np.isfinite(coefficient).all():
        raise ValueError('Finite features and coefficients with matching question rows required')
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError('Regularization must be finite and nonnegative')
    scores = phi @ coefficient
    # A shared row shift cancels from Z*logsumexp(q)-w*q because Z=sum(w).
    # Subtract it before evaluation to avoid cancellation of large logits.
    scores = scores - scores.max(axis=1, keepdims=True)
    scale = len(phi) * NORMALIZATION_ACTIONS
    # Constant-reward rows have weights=total=0. They remain in scale.
    data_loss = np.sum(total * logsumexp(scores, axis=1) - np.sum(weights * scores, axis=1)) / scale
    score_gradient = total[:, None] * softmax(scores, axis=1) - weights
    gradient = phi.T @ score_gradient / scale + 2 * regularization * coefficient
    return float(data_loss + regularization * np.sum(coefficient * coefficient)), gradient


def fit_csce(phi, rewards, regularization):
    """Convex float64 fit, including a penalized constant if present in phi."""
    phi, rewards = np.asarray(phi, dtype=np.float64), np.asarray(rewards, dtype=np.float64)
    shape = (phi.shape[1], rewards.shape[1])
    start_cpu, start_wall = time.process_time(), time.perf_counter()

    def objective(flat):
        value, gradient = csce_value_gradient(flat.reshape(shape), phi, rewards, regularization)
        return value, gradient.ravel()

    result = minimize(objective, np.zeros(shape, dtype=np.float64).ravel(), jac=True,
        method='L-BFGS-B', options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8, maxls=50))
    coefficient = result.x.reshape(shape)
    value, gradient = csce_value_gradient(coefficient, phi, rewards, regularization)
    diagnostics = dict(success=bool(result.success), status=int(result.status), message=str(result.message),
        iterations=int(result.nit), objective=value, gradient_infinity_norm=float(np.max(np.abs(gradient))),
        coefficient_norm=float(np.linalg.norm(coefficient)), regularization=float(regularization),
        regularization_term=float(regularization * np.sum(coefficient * coefficient)),
        data_loss=float(value - regularization * np.sum(coefficient * coefficient)),
        normalization='n_original_questions*20', constant_penalized=True,
        cpu_seconds=time.process_time() - start_cpu, wall_seconds=time.perf_counter() - start_wall)
    if not result.success or not np.isfinite(value) or not np.isfinite(gradient).all():
        raise RuntimeError('CSCE optimizer failed ' + str(diagnostics))
    return coefficient, diagnostics


def semantic_argmax(scores, action_keys):
    """Return exact argmax with fixed semantic identity keys for exact ties."""
    scores = np.asarray(scores, dtype=np.float64)
    keys = tuple(action_keys)
    if scores.ndim != 2 or scores.shape[1] != len(keys) or len(set(keys)) != len(keys):
        raise ValueError('Scores need unique, matching semantic action keys')
    if not np.isfinite(scores).all():
        raise ValueError('Finite scores required')
    semantic_order = np.array(sorted(range(len(keys)), key=lambda index: keys[index]), dtype=int)
    return semantic_order[np.argmax(scores[:, semantic_order], axis=1)]


def validate_training_inputs(history, evaluation):
    """Reject outcome-bearing evaluation packages and question identity overlap."""
    blocked = [str(key) for key in evaluation if str(key).lower() == 'y' or 'reward' in str(key).lower()]
    if blocked:
        raise ValueError('Evaluation package contains outcomes ' + ','.join(blocked))
    if 'ids' not in history or 'ids' not in evaluation:
        raise ValueError('Question identities are required')
    if set(np.asarray(history['ids']).tolist()) & set(np.asarray(evaluation['ids']).tolist()):
        raise ValueError('Fitting and evaluation question identities overlap')
    if 'global_indices' in history and 'global_indices' in evaluation:
        if set(np.asarray(history['global_indices']).tolist()) & set(np.asarray(evaluation['global_indices']).tolist()):
            raise ValueError('Fitting and evaluation question indices overlap')
    return True
