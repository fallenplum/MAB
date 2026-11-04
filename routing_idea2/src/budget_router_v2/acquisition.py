"""One-step KG-informed acquisition under a ridge Gaussian working posterior.

Inputs contain posterior parameters, candidate observation features and training
reference features only. Resource pricing and feedback permissions belong to
an independent ledger. Each explicit batch has one shared posterior snapshot.
"""
import time

import numpy as np


def _array(value, name, dimensions):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != dimensions or not np.isfinite(result).all():
        raise ValueError(name + " must have finite values and the documented dimensions")
    return result


def antithetic_fantasies(dimension, *, pairs=32, seed=0):
    """Return [Z, -Z] in shape (2*pairs, dimension), using a local seeded RNG."""
    for value, name, minimum in ((dimension, "dimension", 1), (pairs, "pairs", 1), (seed, "seed", 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(name + " must be an integer in its supported range")
    draws = np.random.default_rng(seed).standard_normal((pairs, dimension))
    return np.concatenate((draws, -draws), axis=0)


def _prepare_shared(mean, covariance, reference_features, reference_weights):
    mean = _array(mean, "mean", 1)
    covariance = _array(covariance, "covariance", 2)
    references = _array(reference_features, "reference_features", 3)
    dimension = len(mean)
    if dimension < 1 or covariance.shape != (dimension, dimension):
        raise ValueError("Posterior shapes do not agree")
    if references.shape[0] < 1 or references.shape[1] < 2 or references.shape[2] != dimension:
        raise ValueError("Reference features require M>=1, A>=2 and the posterior dimension")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Posterior covariance must be symmetric")
    covariance = covariance * 0.5 + covariance.T * 0.5
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -1e-10 * scale:
        raise ValueError("Posterior covariance must be positive semidefinite")
    weights = (np.ones(len(references), dtype=float) if reference_weights is None else
               _array(reference_weights, "reference_weights", 1))
    if weights.shape != (len(references),) or np.any(weights < 0) or weights.max() <= 0:
        raise ValueError("Reference weights must be aligned, nonnegative and have positive total")
    weights = weights / weights.max()
    weights = weights / weights.sum()
    flat_references = references.reshape(-1, dimension)
    reference_covariance = (flat_references @ covariance).reshape(references.shape)
    predictions = (flat_references @ mean).reshape(references.shape[:2])
    if not np.isfinite(reference_covariance).all() or not np.isfinite(predictions).all():
        raise ValueError("Reference predictions exceed finite numerical range")
    incumbent = np.argmax(predictions, axis=1)
    row = np.arange(len(references))
    contrasts = references[row, incumbent, None, :] - references
    covariance_contrasts = reference_covariance[row, incumbent, None, :] - reference_covariance
    variances = np.maximum(0.0, np.einsum("mad,mad->ma", contrasts, covariance_contrasts))
    maximum = predictions.max(axis=1)
    gaps = maximum[:, None] - predictions
    standardized = np.full_like(gaps, np.inf)
    np.divide(gaps, np.sqrt(variances), out=standardized, where=variances > 1e-14)
    standardized[row, incumbent] = np.inf
    minimum_standardized = standardized.min(axis=1)
    rankable = np.isfinite(minimum_standardized)
    rankable_weight = float(weights[rankable].sum())
    standardized_margin = (float(weights[rankable] @ minimum_standardized[rankable] / rankable_weight)
                           if rankable_weight > 0 else None)
    sorted_predictions = np.sort(predictions, axis=1)
    metrics = {
        "method_family": "KG-informed one-step Gaussian decision value",
        "reference_value_before": float(maximum @ weights),
        "weighted_top_two_margin": float(weights @ (sorted_predictions[:, -1] - sorted_predictions[:, -2])),
        "weighted_min_standardized_margin": standardized_margin,
        "standardized_margin_reference_weight": rankable_weight,
        "weighted_contrast_variance": float(weights @ variances.sum(axis=1) / (references.shape[1] - 1)),
        "reference_inputs": len(references), "reference_actions": references.shape[1],
    }
    return {"covariance": covariance, "reference_covariance": reference_covariance,
            "predictions": predictions, "weights": weights, "incumbent": incumbent,
            "dimension": dimension, "metrics": metrics}


def _score_candidate(shared, observations, noise_covariance, fantasies):
    observations = _array(observations, "observations", 2)
    noise = _array(noise_covariance, "noise_covariance", 2)
    fantasies = _array(fantasies, "fantasies", 2)
    count = len(observations)
    if count < 1 or observations.shape[1] != shared["dimension"]:
        raise ValueError("Posterior and observation shapes do not agree")
    if noise.shape != (count, count) or fantasies.shape[1] != count:
        raise ValueError("Noise and fantasy dimensions must match observation rows")
    if len(fantasies) < 2 or len(fantasies) % 2 or not np.allclose(
            fantasies[:len(fantasies)//2], -fantasies[len(fantasies)//2:], rtol=1e-12, atol=1e-12):
        raise ValueError("Fantasies must use the shared [Z, -Z] antithetic ordering")
    if not np.allclose(noise, noise.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Observation noise covariance must be symmetric")
    noise = noise * 0.5 + noise.T * 0.5
    cross_covariance = shared["covariance"] @ observations.T
    predictive_covariance = observations @ cross_covariance + noise
    predictive_covariance = predictive_covariance * 0.5 + predictive_covariance.T * 0.5
    if not np.isfinite(cross_covariance).all() or not np.isfinite(predictive_covariance).all():
        raise ValueError("Predictive covariance exceeds finite numerical range")
    try:
        noise_cholesky = np.linalg.cholesky(noise)
        predictive_cholesky = np.linalg.cholesky(predictive_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Noise and predictive covariance must be positive definite") from error
    mean_update_factor = np.linalg.solve(predictive_cholesky, cross_covariance.T).T
    reference_covariance = shared["reference_covariance"]
    reference_cross = reference_covariance.reshape(-1, shared["dimension"]) @ observations.T
    reference_update = np.linalg.solve(predictive_cholesky, reference_cross.T).T
    prediction_changes = (fantasies @ reference_update.T).reshape(
        len(fantasies), *shared["predictions"].shape)
    if not np.isfinite(prediction_changes).all():
        raise ValueError("Reference predictions exceed finite numerical range")
    fantasy_predictions = shared["predictions"][None, :, :] + prediction_changes
    if not np.isfinite(fantasy_predictions).all():
        raise ValueError("Fantasy predictions exceed finite numerical range")
    fantasy_values = fantasy_predictions.max(axis=2) @ shared["weights"]
    if not np.isfinite(fantasy_values).all():
        raise ValueError("Fantasy values exceed finite numerical range")
    baseline = shared["metrics"]["reference_value_before"]
    kg = float(np.mean(fantasy_values) - baseline)
    ig = float(np.log(np.diag(predictive_cholesky)).sum() - np.log(np.diag(noise_cholesky)).sum())
    if not np.isfinite(kg) or not np.isfinite(ig):
        raise ValueError("Acquisition values exceed finite numerical range")
    tolerance = 1e-9 * max(1.0, abs(baseline), float(np.max(np.abs(fantasy_values))))
    if kg < -tolerance or ig < -1e-9:
        raise ValueError("Materially negative Gaussian acquisition value")
    kg, ig = max(0.0, kg), max(0.0, ig)
    reference_update = reference_update.reshape(*shared["predictions"].shape, count)
    row = np.arange(len(reference_update))
    learned_contrasts = reference_update[row, shared["incumbent"], None, :] - reference_update
    variance_reduction = np.square(learned_contrasts).sum(axis=2)
    result = {
        **shared["metrics"], "decision_kg": kg, "ig_theta": ig,
        "expected_reference_value_after": baseline + kg,
        "weighted_contrast_variance_reduction": float(shared["weights"] @ variance_reduction.sum(axis=1) /
                                                       (shared["predictions"].shape[1] - 1)),
        "posterior_mean_update_trace": float(np.square(mean_update_factor).sum()),
        "observation_rows": count, "fantasy_count": len(fantasies),
    }
    if any(isinstance(value, float) and not np.isfinite(value) for value in result.values()):
        raise ValueError("Derived acquisition statistics exceed finite numerical range")
    return result


def decision_values(mean, covariance, candidates, reference_features, reference_weights=None):
    """Score one explicit round of candidates against shared training references.

    candidates is a nonempty iterable of dictionaries containing exactly
    observations [m,d], noise_covariance [m,m], and fantasies [2B,m]. Bundles
    may have different m. Each batch prepares and validates its mean [d],
    covariance [d,d], reference_features [M,A,d] and nonnegative reference
    weights once. Whole-batch clocks include that preparation and every
    candidate's validation and scoring. No posterior state survives the call.
    """
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    candidates = list(candidates)
    if not candidates:
        raise ValueError("At least one acquisition candidate is required")
    shared = _prepare_shared(mean, covariance, reference_features, reference_weights)
    scores = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"observations", "noise_covariance", "fantasies"}:
            raise ValueError("Each candidate must contain only observations, noise_covariance and fantasies")
        scores.append(_score_candidate(shared, **candidate))
    return {"candidates": scores, "candidate_count": len(scores),
            "planner_cpu_s": time.process_time() - started_cpu,
            "planner_wall_s": time.perf_counter() - started_wall}


def decision_value(mean, covariance, observations, reference_features, noise_covariance,
                   fantasies, reference_weights=None):
    """Compatibility entry for one Gaussian bundle, including whole-call clocks.

    With S=H covariance H.T+R=L L.T, each fantasy mean is
    mean + covariance H.T L^{-T} z. Reference weights are normalized to sum
    one, posterior covariance may be singular, and noise must be positive
    definite. References may contain the protocol's 20 teams or A>=2 actions
    for software audits. Raw score labels remain external to this operator.
    """
    batch = decision_values(mean, covariance,
        [{"observations": observations, "noise_covariance": noise_covariance, "fantasies": fantasies}],
        reference_features, reference_weights)
    return {**batch["candidates"][0], "planner_cpu_s": batch["planner_cpu_s"],
            "planner_wall_s": batch["planner_wall_s"]}
