"""Input-only decision proxies for the E4 Gaussian and binary working models.

Binary outcome weights use independent Bernoulli probabilities at the current
MAP. They are a composite approximation, not a fitted correlated pair law or an
integrated posterior predictive distribution. All binary fantasy updates use the
actor's unchanged sequential MAP/Laplace implementation on an isolated copy.
"""

from copy import deepcopy
from itertools import product
import time

import numpy as np

from e4_common import LegacyPosterior, LegacyTemplate, TEAMS
from e4_binary import BinaryModel, link_derivative, nll_terms


def _actor(model):
    if isinstance(model, (BinaryModel, LegacyPosterior)):
        return model
    if hasattr(model, "clone"):
        actor = model.clone()
        if isinstance(actor, LegacyPosterior):
            return actor
    raise TypeError("Expected BinaryModel, LegacyPosterior or a Gaussian template")


def _copy_actor(model):
    if isinstance(model, BinaryModel):
        shared = {id(array): array for array in (model.X, model.B, model.rewards,
                                                 model.initial_mean, model.initial_cov)}
    else:
        shared = {id(model.template): model.template, id(model.refF): model.refF}
    return deepcopy(model, shared)


def _transform(probability, callback):
    probability = np.asarray(probability, float)
    result = probability if callback is None else np.asarray(callback(probability.copy()), float)
    if result.shape != probability.shape or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Frozen probability transform must preserve shape and return finite probabilities")
    return result


def _inputs(model, q, actions, references, weights):
    binary = isinstance(model, BinaryModel)
    geometry = model if binary else model.template
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or not 0 <= q < len(geometry.X):
        raise ValueError("q must be a valid integer input row")
    actions, references = np.asarray(actions), np.asarray(references)
    if actions.ndim != 1 or actions.dtype.kind not in "iu" or not 1 <= len(actions) <= 2:
        raise ValueError("A proxy bundle must contain one main or a main/neighbor pair")
    actions = actions.astype(int)
    if len(set(actions)) != len(actions) or np.any(actions < 0) or np.any(actions >= 20):
        raise ValueError("Invalid or duplicate combination actions")
    if len(actions) == 2 and len(set(TEAMS[actions[0]]) ^ set(TEAMS[actions[1]])) != 2:
        raise ValueError("The pair must replace exactly one model")
    if references.ndim != 1 or references.dtype.kind not in "iu" or not len(references):
        raise ValueError("Historical reference input indices are required")
    references = references.astype(int)
    if len(set(references)) != len(references) or not np.isin(references, geometry.history).all():
        raise ValueError("Reference inputs must be distinct historical questions")
    weights = np.ones(len(references)) if weights is None else np.asarray(weights, float)
    if weights.shape != references.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.max() <= 0:
        raise ValueError("Reference weights must be aligned, finite and nonnegative")
    weights = weights / weights.max()
    weights /= weights.sum()
    features = np.einsum("np,ar->napr", geometry.X[references], geometry.B).reshape(len(references), 20, -1)
    latent = (features.reshape(-1, len(model.mean)) @ model.mean).reshape(len(references), 20)
    if binary:
        serving = model.predict_many(references)
        derivative = link_derivative(latent, model.family)
    else:
        serving = latent
        derivative = ((latent > 0) & (latent < 1)).astype(float)
    ranking = np.argsort(-serving, axis=1, kind="stable")[:, :2]
    return int(q), actions, references, weights, features, latent, serving, derivative, ranking


def _reduction(features, projected_update, derivative, ranking, weights):
    """Fixed-current-gradient reductions from a square-root covariance decrement."""
    variance = np.square(projected_update).sum(axis=2)
    rows = np.arange(len(features))
    best, runner = ranking[:, 0], ranking[:, 1]
    latent_contrast = projected_update[rows, best] - projected_update[rows, runner]
    probability_contrast = (derivative[rows, best, None] * projected_update[rows, best]
                            - derivative[rows, runner, None] * projected_update[rows, runner])
    return {"latent_variance_reduction": float(weights @ variance.mean(axis=1)),
            "probability_delta_variance_reduction": float(weights @ (variance * derivative ** 2).mean(axis=1)),
            "latent_contrast_variance_reduction": float(weights @ np.square(latent_contrast).sum(axis=1)),
            "probability_contrast_variance_reduction": float(weights @ np.square(probability_contrast).sum(axis=1))}


def _gaussian_enumeration(model, q, actions, references, weights, probability_transform):
    def probability(actor, rows):
        geometry = actor.template
        coefficient = actor.mean.reshape(geometry.question_dimension, geometry.action_dimension)
        return np.clip(geometry.X[rows] @ coefficient @ geometry.B.T, 0, 1)
    original = probability(model, [q])[0, actions]
    transformed = _transform(original, probability_transform)
    reference_before = probability(model, references)
    before = float(_transform(reference_before, probability_transform).max(axis=1) @ weights)
    uncalibrated_before = float(reference_before.max(axis=1) @ weights)
    expected, uncalibrated_expected, mass_sum = 0., 0., 0.
    outcomes = []
    for values in product((0., 1.), repeat=len(actions)):
        values = np.asarray(values)
        mass = float(np.prod(np.where(values == 1, transformed, 1 - transformed)))
        raw_mass = float(np.prod(np.where(values == 1, original, 1 - original)))
        mass_sum += mass
        if mass == 0 and raw_mass == 0:
            outcomes.append({"values": values.tolist(), "weight": 0., "uncalibrated_weight": 0., "zero_weight_skipped": True})
            continue
        child = _copy_actor(model)
        child.update(q, actions, values)
        after_probability = probability(child, references)
        after = float(_transform(after_probability, probability_transform).max(axis=1) @ weights)
        raw_after = float(after_probability.max(axis=1) @ weights)
        expected += mass * after
        uncalibrated_expected += raw_mass * raw_after
        outcomes.append({"values": values.tolist(), "weight": mass, "uncalibrated_weight": raw_mass,
                         "reference_value_after": after, "uncalibrated_reference_value_after": raw_after})
    if not np.isclose(mass_sum, 1., atol=1e-12):
        raise AssertionError("Gaussian-actor Bernoulli proxy weights do not sum to one")
    raw, uncalibrated = expected - before, uncalibrated_expected - uncalibrated_before
    return {"bernoulli_value_raw": raw, "bernoulli_value_nonnegative": max(0., raw),
            "bernoulli_value_uncalibrated_raw": uncalibrated,
            "bernoulli_value_uncalibrated_nonnegative": max(0., uncalibrated),
            "reference_probability_value_before": before, "expected_reference_probability_value_after": expected,
            "outcome_weight_sum": mass_sum, "outcomes": outcomes,
            "bernoulli_working_model": "Independent clip-MAP Bernoulli outcome weights, optional frozen calibration, unchanged Gaussian Kalman updates; this is distinct from linear Gaussian KG"}


def _gaussian(model, q, actions, references, weights, features, latent, derivative, ranking, fantasies, probability_transform):
    samples = fantasies[len(actions)] if isinstance(fantasies, dict) else fantasies
    samples = np.asarray(samples, float)
    if samples.ndim != 2 or samples.shape[1] != len(actions) or len(samples) < 2 or len(samples) % 2:
        raise ValueError("Gaussian fantasies must have shape (2B, bundle size)")
    half = len(samples) // 2
    if not np.isfinite(samples).all() or not np.allclose(samples[:half], -samples[half:], atol=1e-12, rtol=1e-12):
        raise ValueError("Gaussian fantasies must have the matched [Z,-Z] ordering")
    observation = model.observation(q, actions)
    cross = model.cross(q, actions)
    noise = model.noise[np.ix_(actions, actions)]
    predictive = observation @ cross + noise
    predictive = .5 * (predictive + predictive.T)
    chol = np.linalg.cholesky(predictive)
    white = np.linalg.solve(chol, cross.T).T
    projected = (features.reshape(-1, len(model.mean)) @ white).reshape(len(features), 20, len(actions))
    changes = (samples @ projected.reshape(-1, len(actions)).T).reshape(len(samples), *latent.shape)
    fantasy = latent[None] + changes
    linear_before = float(latent.max(axis=1) @ weights)
    clipped_before = float(np.clip(latent, 0, 1).max(axis=1) @ weights)
    linear_raw = float((fantasy.max(axis=2) @ weights).mean() - linear_before)
    clipped_raw = float((np.clip(fantasy, 0, 1).max(axis=2) @ weights).mean() - clipped_before)
    calibrated_before = float(_transform(np.clip(latent, 0, 1), probability_transform).max(axis=1) @ weights)
    calibrated_raw = float((_transform(np.clip(fantasy, 0, 1), probability_transform).max(axis=2) @ weights).mean() - calibrated_before)
    ig = float(np.log(np.diag(chol)).sum() - np.log(np.diag(np.linalg.cholesky(noise))).sum())
    parameter_before = float(np.trace(model.cov))
    parameter_reduction = float(np.square(white).sum())
    reductions = _reduction(features, projected, derivative, ranking, weights)
    enumeration = _gaussian_enumeration(model, q, actions, references, weights, probability_transform)
    return {**reductions, **enumeration, "ig": ig, "ig_kind": "Gaussian parameter mutual information under the specified working likelihood",
            "parameter_trace_before": parameter_before,
            "parameter_trace_reduction": parameter_reduction,
            "parameter_trace_expected_after": parameter_before - parameter_reduction,
            "gaussian_kg_linear_raw": linear_raw, "gaussian_kg_linear_nonnegative": max(0., linear_raw),
            "gaussian_kg_clipped_probability_raw": clipped_raw,
            "gaussian_kg_clipped_probability_nonnegative": max(0., clipped_raw),
            "gaussian_kg_calibrated_probability_raw": calibrated_raw,
            "gaussian_kg_calibrated_probability_nonnegative": max(0., calibrated_raw),
            "decision_value_raw": linear_raw, "decision_value_nonnegative": max(0., linear_raw),
            "fixed_contrast_reduction": reductions["latent_contrast_variance_reduction"],
            "fantasy_samples": len(samples), "decision_value_operator": "linear Gaussian KG",
            "probability_delta_note": "Fixed current clipping derivative, one inside (0,1) and zero outside and at endpoints",
            "working_model": "Gaussian continuous raw-score likelihood; linear KG and clipped-probability KG are distinct"}


def _binary(model, q, actions, references, weights, features, serving, derivative, ranking, probability_transform):
    observation = model.observation(q, actions)
    latent_covariance = observation @ model.cov @ observation.T
    original_probability = model.predict_many([q])[0, actions]
    current_probability = _transform(original_probability, probability_transform)
    before = float(_transform(serving, probability_transform).max(axis=1) @ weights)
    uncalibrated_before = float(serving.max(axis=1) @ weights)
    expected_after, expected_ig, total_mass, uncalibrated_expected, expected_trace = 0., 0., 0., 0., 0.
    names = ("latent_variance_reduction", "probability_delta_variance_reduction",
             "latent_contrast_variance_reduction", "probability_contrast_variance_reduction")
    expected_reduction = {name: 0. for name in names}
    outcomes = []
    for values in product((0., 1.), repeat=len(actions)):
        values = np.asarray(values)
        mass = float(np.prod(np.where(values == 1, current_probability, 1 - current_probability)))
        uncalibrated_mass = float(np.prod(np.where(values == 1, original_probability, 1 - original_probability)))
        total_mass += mass
        if mass == 0 and uncalibrated_mass == 0:
            outcomes.append({"values": values.tolist(), "weight": 0., "uncalibrated_weight": 0., "zero_weight_skipped": True})
            continue
        updated = _copy_actor(model)
        updated.update(q, actions, values)
        after_probability = updated.predict_many(references)
        after = float(_transform(after_probability, probability_transform).max(axis=1) @ weights)
        uncalibrated_after = float(after_probability.max(axis=1) @ weights)
        # This is exactly the actor's local-Laplace covariance decrement, expressed
        # as a low-rank factor. Derivatives remain fixed at the original MAP.
        _, _, curvature = nll_terms(observation @ updated.mean, values, 1, model.family)
        inverse_curvature = np.diag(1 / curvature)
        chol = np.linalg.cholesky(inverse_curvature + latent_covariance)
        cross = model.cov @ observation.T
        factor = np.linalg.solve(chol, cross.T).T
        projected = (features.reshape(-1, len(model.mean)) @ factor).reshape(len(features), 20, len(actions))
        reduction = _reduction(features, projected, derivative, ranking, weights)
        root_curvature = np.sqrt(curvature)
        determinant_matrix = np.eye(len(actions)) + root_curvature[:, None] * latent_covariance * root_curvature[None, :]
        sign, logdet = np.linalg.slogdet(determinant_matrix)
        if sign <= 0:
            raise ValueError("Laplace entropy decrement is undefined")
        ig = .5 * float(logdet)
        expected_after += mass * after
        expected_trace += mass * float(np.trace(updated.cov))
        uncalibrated_expected += uncalibrated_mass * uncalibrated_after
        expected_ig += mass * ig
        for name in names:
            expected_reduction[name] += mass * reduction[name]
        outcomes.append({"values": values.tolist(), "weight": mass, "uncalibrated_weight": uncalibrated_mass,
                         "reference_value_after": after, "laplace_entropy_reduction": ig,
                         "uncalibrated_reference_value_after": uncalibrated_after,
                         "mean_change_l2": float(np.linalg.norm(updated.mean - model.mean)), **reduction})
    if not np.isclose(total_mass, 1., atol=1e-12):
        raise AssertionError("Independent Bernoulli outcome weights do not sum to one")
    raw = expected_after - before
    uncalibrated_raw = uncalibrated_expected - uncalibrated_before
    return {**expected_reduction, "ig": expected_ig,
            "parameter_trace_before": float(np.trace(model.cov)),
            "parameter_trace_reduction": float(np.trace(model.cov)) - expected_trace,
            "parameter_trace_expected_after": expected_trace,
            "ig_kind": "Expected local-Laplace Gaussian entropy reduction; a composite proxy, not exact Bernoulli mutual information",
            "gaussian_kg_linear_raw": None, "gaussian_kg_linear_nonnegative": None,
            "gaussian_kg_clipped_probability_raw": None, "gaussian_kg_clipped_probability_nonnegative": None,
            "gaussian_kg_calibrated_probability_raw": None, "gaussian_kg_calibrated_probability_nonnegative": None,
            "bernoulli_value_raw": raw, "bernoulli_value_nonnegative": max(0., raw),
            "bernoulli_value_uncalibrated_raw": uncalibrated_raw,
            "bernoulli_value_uncalibrated_nonnegative": max(0., uncalibrated_raw),
            "decision_value_raw": raw, "decision_value_nonnegative": max(0., raw),
            "fixed_contrast_reduction": expected_reduction["probability_contrast_variance_reduction"],
            "reference_value_before": before, "expected_reference_value_after": expected_after,
            "outcome_weight_sum": total_mass, "outcomes": outcomes, "fantasy_samples": 0,
            "decision_value_operator": "enumerated composite Bernoulli value",
            "probability_delta_note": "Current MAP link derivative is fixed across outcomes; the delta-method reduction is not an exact posterior probability variance",
            "working_model": "Independent MAP Bernoulli outcome weights with sequential composite MAP/Laplace updates; correlated pair outcome law and posterior-predictive integration are absent",
            "negative_value_note": "Approximate outcome weights and approximate updates need not obey the posterior martingale property; the raw negative value is retained"}


def decision_proxies(model, q, actions, reference_indices, weights, fantasies, incremental_cost=1.0,
                     probability_transform=None):
    """Evaluate input-only proxies against a frozen actor snapshot without mutation."""
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    model = _actor(model)
    q, actions, references, weights, features, latent, serving, derivative, ranking = _inputs(
        model, q, actions, reference_indices, weights)
    if isinstance(incremental_cost, dict):
        incremental_cost = incremental_cost["priced_units"]
    cost = float(incremental_cost)
    if not np.isfinite(cost) or cost <= 0:
        raise ValueError("Expected complete incremental cost must be finite and positive")
    if isinstance(model, BinaryModel):
        result = _binary(model, q, actions, references, weights, features, serving, derivative, ranking, probability_transform)
    else:
        result = _gaussian(model, q, actions, references, weights, features, latent, derivative, ranking, fantasies, probability_transform)
    score_fields = ("ig", "latent_variance_reduction", "probability_delta_variance_reduction",
                    "parameter_trace_reduction", "parameter_trace_before", "parameter_trace_expected_after",
                    "latent_contrast_variance_reduction", "probability_contrast_variance_reduction",
                    "fixed_contrast_reduction", "decision_value_raw", "decision_value_nonnegative",
                    "gaussian_kg_linear_raw", "gaussian_kg_linear_nonnegative",
                    "gaussian_kg_clipped_probability_raw", "gaussian_kg_clipped_probability_nonnegative",
                    "gaussian_kg_calibrated_probability_raw", "gaussian_kg_calibrated_probability_nonnegative",
                    "bernoulli_value_raw", "bernoulli_value_nonnegative",
                    "bernoulli_value_uncalibrated_raw", "bernoulli_value_uncalibrated_nonnegative")
    result["per_cost"] = {name: result[name] / cost if result[name] is not None else None for name in score_fields}
    result.update({"incremental_cost": cost, "reference_inputs": len(references), "reference_actions": 20,
                   "reference_permission": "historical inputs only; no reference or query rewards read",
                   "fixed_contrast_rule": "Best and runner-up at the current serving prediction, frozen before every fantasy",
                   "probability_transform_applied": probability_transform is not None,
                   "probability_transform_scope": "Acquisition outcome weights and probability reference utilities only; actor service/update and fixed delta gradients are unchanged",
                   "actor_state_mutated": False, "new_feedback_reveals": 0, "new_llm_calls": 0,
                   "planner_cpu_s": time.process_time() - started_cpu,
                   "planner_wall_s": time.perf_counter() - started_wall})
    return result


def qualification():
    import sys
    from e4_common import ROOT
    sys.path.insert(0, str(ROOT / "src"))
    from e4_gaussian import GaussianTemplate
    from budget_router_v2.acquisition import antithetic_fantasies, decision_value
    rng = np.random.default_rng(20261109)
    size, history, refs, q = 56, np.arange(40), np.arange(5), 50
    data = {"E": rng.normal(size=(size, 24)), "Y": rng.integers(0, 2, (size, 3, 20)).astype(float),
            "task": np.array(["gsm8k" if i % 2 else "bbh" for i in range(size)]),
            "sub": np.array(["" if i % 2 else "logic" for i in range(size)]),
            "ids": np.array([str(i) for i in range(size)])}
    template = GaussianTemplate(data, history, "P2", refs, variant="N2")
    gaussian = template.clone()
    weights = np.arange(1., 6.)
    results = []
    def calibration(probability):
        probability = np.clip(probability, 1e-9, 1 - 1e-9)
        transformed_logit = .8 * np.log(probability / (1 - probability)) + .2
        return 1 / (1 + np.exp(-transformed_logit))
    for actions in ([0], [0, 1]):
        gaussian_mean_before, gaussian_cov_before = gaussian.mean.copy(), gaussian.cov.copy()
        samples = antithetic_fantasies(len(actions), pairs=8, seed=21)
        value = decision_proxies(gaussian, q, actions, refs, weights, samples, 2.)
        canonical = decision_value(gaussian.mean, gaussian.cov, gaussian.observation(q, actions),
                                   gaussian.refF, gaussian.noise[np.ix_(actions, actions)], samples, weights)
        assert abs(value["gaussian_kg_linear_nonnegative"] - canonical["decision_kg"]) < 1e-12
        assert abs(value["ig"] - canonical["ig_theta"]) < 1e-12
        h = gaussian.observation(q, actions)
        cross = gaussian.cross(q, actions)
        r = gaussian.noise[np.ix_(actions, actions)]
        predictive = h @ cross + r
        cloned = _copy_actor(gaussian)
        raw = np.array([0., 1.])[:len(actions)]
        expected_mean = gaussian.mean + cross @ np.linalg.solve(predictive, raw - h @ gaussian.mean)
        expected_cov = gaussian.cov - cross @ np.linalg.solve(predictive, cross.T)
        cloned.update(q, actions, raw)
        assert np.max(abs(expected_mean - cloned.mean)) < 1e-12
        assert np.max(abs(expected_cov - cloned.cov)) < 1e-12
        assert abs(value["parameter_trace_expected_after"] - np.trace(cloned.cov)) < 1e-12
        assert np.array_equal(gaussian.mean, gaussian_mean_before) and np.array_equal(gaussian.cov, gaussian_cov_before)
        assert value["per_cost"]["decision_value_raw"] == value["decision_value_raw"] / 2
        calibrated = decision_proxies(gaussian, q, actions, refs, weights, samples, 2., calibration)
        assert calibrated["gaussian_kg_linear_raw"] == value["gaussian_kg_linear_raw"]
        assert calibrated["bernoulli_value_uncalibrated_raw"] == value["bernoulli_value_raw"]
        for outcome in calibrated["outcomes"]:
            child = _copy_actor(gaussian)
            child.update(q, actions, outcome["values"])
            coefficient = child.mean.reshape(template.question_dimension, template.action_dimension)
            probabilities = np.clip(template.X[refs] @ coefficient @ template.B.T, 0, 1)
            expected = float(calibration(probabilities).max(axis=1) @ (weights / weights.sum()))
            assert abs(expected - outcome["reference_value_after"]) < 1e-12
        results.append({"family": "gaussian", "bundle": len(actions), "canonical_v2_kg_error":
                        abs(value["gaussian_kg_linear_nonnegative"] - canonical["decision_kg"])})
    for family in ("logistic", "probit"):
        binary = BinaryModel(template.X, template.B, history, data["Y"][history], 100., family)
        mean_before, cov_before = binary.mean.copy(), binary.cov.copy()
        for actions in ([0], [0, 1]):
            result = decision_proxies(binary, q, actions, refs, weights, None, 3.)
            assert len(result["outcomes"]) == 2 ** len(actions)
            assert abs(result["outcome_weight_sum"] - 1) < 1e-12
            for outcome in result["outcomes"]:
                child = _copy_actor(binary)
                child.update(q, actions, outcome["values"])
                normalized = weights / weights.sum()
                expected = float(child.predict_many(refs).max(axis=1) @ normalized)
                assert abs(expected - outcome["reference_value_after"]) < 1e-12
            expected_trace = 0.
            for outcome in result["outcomes"]:
                child = _copy_actor(binary)
                child.update(q, actions, outcome["values"])
                expected_trace += outcome["weight"] * np.trace(child.cov)
            assert abs(result["parameter_trace_expected_after"] - expected_trace) < 1e-12
            assert np.array_equal(mean_before, binary.mean) and np.array_equal(cov_before, binary.cov)
            calibrated = decision_proxies(binary, q, actions, refs, weights, None, 3., calibration)
            assert calibrated["bernoulli_value_uncalibrated_raw"] == result["bernoulli_value_raw"]
            for outcome in calibrated["outcomes"]:
                child = _copy_actor(binary)
                child.update(q, actions, outcome["values"])
                expected = float(calibration(child.predict_many(refs)).max(axis=1) @ (weights / weights.sum()))
                assert abs(expected - outcome["reference_value_after"]) < 1e-12
            results.append({"family": family, "bundle": len(actions),
                            "raw_value": result["bernoulli_value_raw"], "zero_clipped_value": result["bernoulli_value_nonnegative"]})
    # Frozen proxies have no data/Y input. Altering every external reward cannot
    # change the already-fitted actor or its input-only acquisition statistics.
    original = decision_proxies(template, q, [0, 1], refs, weights,
                                antithetic_fantasies(2, pairs=8, seed=21))
    data["Y"][40:] = np.nan
    alternative = GaussianTemplate(data, history, "P2", refs, variant="N2")
    same = decision_proxies(alternative, q, [0, 1], refs, weights,
                            antithetic_fantasies(2, pairs=8, seed=21))
    for field in original["per_cost"]:
        assert original[field] == same[field]
    return {"status": "PASS", "results": results,
            "checks": ["Gaussian_matched_sample_exact_V2", "Gaussian_actual_actor_block_update",
                       "Bernoulli_all_outcomes_match_actor_sequential_update", "actor_not_mutated",
                       "unseen_Y_perturbation_invariance", "raw_and_clipped_values_separate", "cost_normalization",
                       "frozen_calibration_weights_and_reference_utilities", "Gaussian_Bernoulli_actor_updates"],
            "scientific_success": False, "new_llm_calls": 0}


if __name__ == "__main__":
    import json
    print(json.dumps(qualification(), ensure_ascii=False, indent=2))
