"""Bounded joint Gaussian working posterior for frozen combination rewards.

This is an empirical-Bayes Gaussian moment approximation. It is not an exact
Bernoulli likelihood or a proven calibrated posterior for conditional accuracy.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import truncnorm

K = 20
PAIR_I, PAIR_J = np.array(list(itertools.combinations(range(K), 2))).T
SHRINKAGE = .20
RIDGE = 1e-6
BURN = 128
THIN = 2
CHAIN_DRAWS = 256
TRAIN_DRAWS = 8
RHAT_LIMIT = 1.1
ESS_LIMIT = 32.


def symmetric(matrix):
    return (matrix + matrix.T) / 2


def diagonal_shrink(matrix):
    return symmetric((1 - SHRINKAGE) * matrix + SHRINKAGE * np.diag(np.diag(matrix))
                     + RIDGE * np.eye(K))


def fit_working_prior(rewards):
    """Fit only to a legal historical region, retaining within-question blocks."""
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 3 or rewards.shape[1:] != (3, K):
        raise ValueError('Expected independent questions x three generations x20')
    n, repeats, _ = rewards.shape
    means = rewards.mean(axis=1)
    residual = rewards - means[:, None, :]
    noise_raw = np.einsum('nsi,nsj->ij', residual, residual) / (n * (repeats - 1))
    noise = diagonal_shrink(noise_raw)
    mean_covariance = np.cov(means, rowvar=False, ddof=1)
    between_raw = symmetric(mean_covariance - noise / repeats)
    eigenvalues, vectors = np.linalg.eigh(between_raw)
    projected = (vectors * np.maximum(eigenvalues, 0.)) @ vectors.T
    prior_covariance = diagonal_shrink(projected)
    return dict(mean=means.mean(axis=0), noise=noise, noise_raw=noise_raw,
                prior_covariance=prior_covariance, between_raw=between_raw,
                mean_covariance=mean_covariance,
                between_raw_eigenvalues=eigenvalues, questions=n)


def conjugate_parameters(model, observed):
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 3 or observed.shape[2] != K:
        raise ValueError('Observations must preserve their question-level joint block')
    repeats = observed.shape[1]
    prior_precision = np.linalg.inv(model['prior_covariance'])
    noise_precision = np.linalg.inv(model['noise'])
    precision = symmetric(prior_precision + repeats * noise_precision)
    covariance = symmetric(np.linalg.inv(precision))
    linear = model['mean'] @ prior_precision + repeats * observed.mean(axis=1) @ noise_precision
    mean = linear @ covariance
    return mean, covariance, precision


def sample_bounded(model, observed, seed, *, burn=BURN, thin=THIN, draws=CHAIN_DRAWS):
    """Two joint Gibbs chains on [0,1]^20, no clipped-posterior substitution."""
    means, covariance, precision = conjugate_parameters(model, observed)
    n = len(means)
    target = np.repeat(means, 2, axis=0)
    # Interior overdispersed initial states are not posterior draws.
    state = np.tile(np.array([[.10] * K, [.90] * K]), (n, 1))
    residual_precision = (state - target) @ precision
    conditional_sd = np.sqrt(1 / np.diag(precision))
    rng = np.random.default_rng(seed)
    samples = np.empty((n * 2, draws, K))
    failed = np.zeros(n * 2, dtype=bool)
    slow_tail_updates = 0
    for sweep in range(burn + thin * draws):
        for coordinate in range(K):
            conditional_mean = state[:, coordinate] - residual_precision[:, coordinate] / precision[coordinate, coordinate]
            sd = conditional_sd[coordinate]
            lower, upper = -conditional_mean / sd, (1 - conditional_mean) / sd
            low_cdf, high_cdf = ndtr(lower), ndtr(upper)
            uniform = rng.uniform(np.finfo(float).eps, 1 - np.finfo(float).eps, len(state))
            stable = high_cdf - low_cdf > 1e-12
            value = np.empty(len(state))
            value[stable] = conditional_mean[stable] + sd * ndtri(
                low_cdf[stable] + (high_cdf[stable] - low_cdf[stable]) * uniform[stable])
            tail = ~stable
            if tail.any():
                value[tail] = truncnorm.ppf(uniform[tail], lower[tail], upper[tail],
                                           loc=conditional_mean[tail], scale=sd)
                slow_tail_updates += int(tail.sum())
            invalid = ~np.isfinite(value) | (value < 0) | (value > 1)
            failed |= invalid
            # Invalid rows are explicitly rejected after diagnostics.
            value[invalid] = .5
            change = value - state[:, coordinate]
            state[:, coordinate] = value
            residual_precision += change[:, None] * precision[coordinate]
        if sweep >= burn and (sweep - burn) % thin == 0:
            samples[:, (sweep - burn) // thin] = state
    chains = samples.reshape(n, 2, draws, K)
    rhat, ess = chain_diagnostics(chains)
    failed_question = failed.reshape(n, 2).any(axis=1)
    accepted = ~failed_question & (rhat.max(axis=1) <= RHAT_LIMIT) & (ess.min(axis=1) >= ESS_LIMIT)
    return dict(chains=chains, draws=chains.reshape(n, 2 * draws, K),
                rhat=rhat, ess=ess, accepted=accepted, numeric_failed=failed_question,
                zero_variance_coordinates=(chains.var(axis=2, ddof=1).mean(axis=1) <= 1e-20).sum(axis=1),
                unbounded_mean=means, unbounded_covariance=covariance,
                slow_tail_updates=slow_tail_updates)


def chain_diagnostics(chains):
    """Classical two-chain Rhat and positive-pair autocorrelation ESS diagnostic."""
    n, chain_count, draws, dimension = chains.shape
    chain_mean = chains.mean(axis=2)
    within = chains.var(axis=2, ddof=1).mean(axis=1)
    between = draws * chain_mean.var(axis=1, ddof=1)
    variance = ((draws - 1) * within + between) / draws
    rhat = np.sqrt(np.divide(variance, within, out=np.ones_like(variance), where=within > 1e-20))
    degenerate = within <= 1e-20
    rhat[degenerate & (between > 1e-20)] = np.inf
    centered = chains - chain_mean[:, :, None, :]
    correlations = []
    for lag in range(1, 33):
        covariance = np.mean(centered[:, :, :-lag] * centered[:, :, lag:], axis=(1, 2))
        rho = np.divide(covariance, within, out=np.zeros_like(covariance), where=within > 1e-20)
        correlations.append(rho)
    total = np.zeros((n, dimension))
    active = np.ones((n, dimension), dtype=bool)
    for lag in range(0, len(correlations), 2):
        pair = correlations[lag] + correlations[lag + 1]
        active &= pair > 0
        total += np.where(active, pair, 0.)
    ess = chain_count * draws / np.maximum(1 + 2 * total, 1.)
    ess[degenerate] = 0.
    return rhat, np.minimum(ess, chain_count * draws)


def summaries(samples, delta):
    maxima = samples.max(axis=2, keepdims=True)
    optimal = maxima - samples <= delta
    weights = optimal / optimal.sum(axis=2, keepdims=True)
    mean_weight = weights.mean(axis=1)
    entropy = -np.sum(np.where(mean_weight > 0, mean_weight * np.log(np.maximum(mean_weight, 1e-300)), 0), axis=1)
    difference = samples[:, :, PAIR_I] - samples[:, :, PAIR_J]
    probabilities = np.stack(((difference > delta).mean(axis=1),
                              (np.abs(difference) <= delta).mean(axis=1),
                              (difference < -delta).mean(axis=1)), axis=2)
    return dict(weights=mean_weight, effective_combinations=np.exp(entropy), entropy=entropy,
                pair_probability=probabilities,
                mean_set_size=optimal.sum(axis=2).mean(axis=1))


def training_payload(result, observed, delta):
    """Preserve failures with a declared empirical-set fallback."""
    empirical = observed.mean(axis=1)
    samples = result['draws'].copy()
    fallback = ~result['accepted']
    samples[fallback] = empirical[fallback, None, :]
    # Select well-separated positions from both chains for the small MC loss.
    count = result['chains'].shape[2]
    positions = np.r_[np.linspace(0, count - 1, 4, dtype=int),
                      count + np.linspace(0, count - 1, 4, dtype=int)]
    alternative_positions = np.r_[np.array([32, 96, 160, 224]),
                                  count + np.array([32, 96, 160, 224])]
    if len(set(positions.tolist()) & set(alternative_positions.tolist())):
        raise AssertionError('MC qualification batches must be disjoint')
    payload = dict(mu_samples=samples[:, positions], mu_samples_alt=samples[:, alternative_positions],
                   posterior_mean=samples.mean(axis=1),
                   fallback_mask=fallback, rhat_max=result['rhat'].max(axis=1),
                   ess_min=result['ess'].min(axis=1), numeric_failed=result['numeric_failed'])
    for value, suffix in ((delta, 'delta005'), (0., 'delta000')):
        stats = summaries(samples, value)
        for key, array in stats.items():
            payload[key + '_' + suffix] = array
    return payload


def predictive_diagnostics(samples, noise, held):
    """Observed-repeat checks only, no claims about latent-event calibration."""
    mean = samples.mean(axis=1)
    marginal_brier = (mean - held) ** 2
    pair_samples = samples[:, :, PAIR_I] - samples[:, :, PAIR_J]
    pair_mean = pair_samples.mean(axis=1)
    pair_variance = pair_samples.var(axis=1, ddof=1)
    repeat_variance = noise[PAIR_I, PAIR_I] + noise[PAIR_J, PAIR_J] - 2 * noise[PAIR_I, PAIR_J]
    predictive_variance = pair_variance + repeat_variance
    actual = held[:, PAIR_I] - held[:, PAIR_J]
    residual_square = (actual - pair_mean) ** 2
    interval_covered = np.abs(actual - pair_mean) <= 1.959963984540054 * np.sqrt(predictive_variance)
    # Gaussian-moment pair covariance need not be feasible for binary outcomes.
    posterior_product = np.mean(samples[:, :, PAIR_I] * samples[:, :, PAIR_J], axis=1)
    joint_success = posterior_product + noise[PAIR_I, PAIR_J]
    lower = np.maximum(0, mean[:, PAIR_I] + mean[:, PAIR_J] - 1)
    upper = np.minimum(mean[:, PAIR_I], mean[:, PAIR_J])
    violation = (joint_success < lower - 1e-10) | (joint_success > upper + 1e-10)
    marginal_variance = samples.var(axis=1, ddof=1) + np.diag(noise)
    bernoulli_variance = mean * (1 - mean)
    return dict(marginal_brier=marginal_brier.mean(axis=1),
                predicted_noise_variance=repeat_variance.mean(),
                pair_squared_prediction_error=residual_square.mean(axis=1),
                pair_predictive_variance=predictive_variance.mean(axis=1),
                gaussian_moment_interval_coverage=interval_covered.mean(axis=1),
                binary_Frechet_violation_fraction=violation.mean(axis=1),
                binary_marginal_variance_excess_fraction=(marginal_variance > bernoulli_variance + 1e-10).mean(axis=1),
                empirical_same_question_latent_events_observed=False)
