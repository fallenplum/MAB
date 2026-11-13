"""Validated set-valued SPO+ and Davidson objectives for frozen linear heads.

Posterior reward samples and set weights are held fixed while optimizing q.
None of these operators estimates or calibrates a reward posterior.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.special import logsumexp

ACTION_COUNT = 20
PAIR_I, PAIR_J = np.array(list(itertools.combinations(range(ACTION_COUNT), 2))).T
INCIDENCE = np.zeros((len(PAIR_I), ACTION_COUNT))
INCIDENCE[np.arange(len(PAIR_I)), PAIR_I] = 1.
INCIDENCE[np.arange(len(PAIR_I)), PAIR_J] = -1.


def set_weights(rewards, delta=0.):
    """One simplex vector per reward sample, exact practical delta threshold."""
    rewards = np.asarray(rewards, dtype=float)
    if delta < 0 or not np.isfinite(delta) or not np.isfinite(rewards).all():
        raise ValueError("Finite rewards and nonnegative finite delta required")
    included = rewards >= rewards.max(axis=-1, keepdims=True) - delta
    return included / included.sum(axis=-1, keepdims=True)


def set_spo_sample_loss(scores, rewards, weights):
    scores, rewards, weights = map(np.asarray, (scores, rewards, weights))
    if rewards.shape != weights.shape or scores.shape[-1] != rewards.shape[-1]:
        raise ValueError("Action dimensions differ")
    surrogate = 2 * scores - rewards
    return surrogate.max(axis=-1) - np.sum(weights * surrogate, axis=-1)


def ns_spo_value_subgradient(coefficient, phi, samples, weights, regularization):
    """Mean over original questions and fixed Monte Carlo samples plus L2.

    phi has shape(n,d), samples/weights(n,B,20), coefficient(d,20).
    At surrogate-max ties, the first maximizer gives a valid subgradient.
    """
    phi, samples, weights = map(np.asarray, (phi, samples, weights))
    if samples.ndim != 3 or samples.shape != weights.shape:
        raise ValueError("Fixed question by sample by action arrays required")
    n, draws, actions = samples.shape
    if actions != ACTION_COUNT or coefficient.shape != (phi.shape[1], actions) or len(phi) != n:
        raise ValueError("Fixed20-output linear capacity required")
    if not np.allclose(weights.sum(axis=-1), 1.) or np.any(weights < 0):
        raise ValueError("Weights must lie on the simplex")
    scores = phi @ coefficient
    shifted = 2 * scores[:, None, :] - samples
    selected = shifted.argmax(axis=-1)
    loss = np.mean(shifted.max(axis=-1) - np.sum(weights * shifted, axis=-1))
    active = np.zeros_like(weights)
    active[np.arange(n)[:, None], np.arange(draws)[None, :], selected] = 1.
    score_gradient = 2 * np.mean(active - weights, axis=1) / n
    gradient = phi.T @ score_gradient + 2 * regularization * coefficient
    return float(loss + regularization * np.sum(coefficient ** 2)), gradient


def posterior_pair_probabilities(samples, delta=0.):
    """Return P(Delta>delta), P(|Delta|<=delta), P(Delta<−delta)."""
    samples = np.asarray(samples)
    difference = samples[..., PAIR_I] - samples[..., PAIR_J]
    return np.stack(((difference > delta).mean(axis=-2),
                     (np.abs(difference) <= delta).mean(axis=-2),
                     (difference < -delta).mean(axis=-2)), axis=-1)


def davidson_value_gradient(coefficient, phi, targets, log_nu=0., regularization=0.):
    """Mean question and190 pair soft-label cross entropy.

    targets order is win_i, practical_tie, win_j. log_nu is the shared
    nuisance parameter, with the user's2*nu parameterization. Return loss,
    head gradient and the derivative with respect to log_nu.
    """
    phi, targets = np.asarray(phi), np.asarray(targets)
    if targets.shape != (len(phi), len(PAIR_I), 3):
        raise ValueError("One three-state simplex label per question/pair required")
    if np.any(targets < 0) or not np.allclose(targets.sum(axis=-1), 1.):
        raise ValueError("Targets must lie on the simplex")
    scores = phi @ coefficient
    margin = scores[:, PAIR_I] - scores[:, PAIR_J]
    logits = np.stack((margin / 2., np.full_like(margin, np.log(2.) + log_nu), -margin / 2.), axis=-1)
    log_probability = logits - logsumexp(logits, axis=-1, keepdims=True)
    probabilities = np.exp(log_probability)
    loss = -np.mean(np.sum(targets * log_probability, axis=-1))
    residual = probabilities - targets
    margin_gradient = .5 * (residual[..., 0] - residual[..., 2]) / (len(phi) * len(PAIR_I))
    gradient = phi.T @ (margin_gradient @ INCIDENCE) + 2 * regularization * coefficient
    eta_gradient = float(residual[..., 1].mean())
    return float(loss + regularization * np.sum(coefficient ** 2)), gradient, eta_gradient
