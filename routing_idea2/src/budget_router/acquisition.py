"""Pre-output acquisition planning with no access to current scores or lengths.

`predicted_costs` maps each alternate subset (or its integer row index) to a
CostPrediction for its COMPLETE incremental purchase. With natural main feedback
this is one extra label; otherwise it is two. Key "main_only" prices buying only
its main label. All executions include model AND aggregator calls. A plan only
reserves/requests feedback; it never adds a hypothetical label to the learner.

Decision/no_cost/random/uncertainty/periodic share greedy main selection.
LinUCB/Thompson explore through the main service and request only its feedback.
max_uncertainty uses pure reward working variance; max_uncertainty_per_cost
divides it by predicted complete cost. Random and periodic choose uniformly
among affordable purchase actions. Non-decision baselines report zero unused
directional information and do not compute competitors or bootstrap directions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .features import canonical_subset, single_member_replacements
from .budget import CostPrediction, ResourceCost


def rank_one_inverse_update(inverse, observation, noise_variance=1.0):
    """Sherman-Morrison update, used for EXPECTED design only in planning."""
    inverse = np.asarray(inverse, dtype=float)
    x = np.asarray(observation, dtype=float)
    if inverse.ndim != 2 or inverse.shape[0] != inverse.shape[1] or x.shape != (len(inverse),):
        raise ValueError("Incompatible inverse/observation dimensions")
    if not np.isfinite(inverse).all() or not np.isfinite(x).all() or not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("Invalid design update")
    u = inverse @ x
    denominator = noise_variance + x @ u
    if denominator <= 0:
        raise ValueError("Design inverse must be positive definite")
    updated = inverse - np.outer(u, u) / denominator
    return (updated + updated.T) * 0.5


def directional_information_gain(inverse, directions, observation, noise_variance=1.0):
    """(d' U psi)^2 / (sigma2 + psi' U psi), one value per direction."""
    inverse = np.asarray(inverse, dtype=float)
    d = np.asarray(directions, dtype=float)
    x = np.asarray(observation, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if inverse.ndim != 2 or inverse.shape[0] != inverse.shape[1] or x.shape != (len(inverse),) or d.ndim != 2 or d.shape[1] != len(inverse):
        raise ValueError("Incompatible directional information dimensions")
    if not np.isfinite(d).all() or not np.isfinite(inverse).all() or not np.isfinite(x).all() or not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("Invalid directional information input")
    u = inverse @ x
    denominator = noise_variance + x @ u
    if denominator <= 0:
        raise ValueError("Design inverse must be positive definite")
    return np.square(d @ u) / denominator


def choose_main(learner, features, *, policy="greedy", alpha=1.0, rng=None,
                decision_statistics=None):
    """Select once; optionally copy the statistics already used by this decision."""
    if decision_statistics is not None and not isinstance(decision_statistics,dict):
        raise TypeError("decision_statistics must be a dict or None")
    features = np.asarray(features, dtype=float)
    predictions = learner.predict(features)
    policy = policy.lower()
    variances = None
    if policy == "greedy":
        values = predictions
    elif policy == "linucb":
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("LinUCB alpha must be nonnegative")
        variances = learner.working_variance(features)
        values = predictions + alpha * np.sqrt(variances)
    elif policy in ("thompson", "ts", "thompson_sampling"):
        if rng is None:
            raise ValueError("Stochastic policies require an explicit RNG")
        values = features @ learner.thompson_sample(rng=rng)
        policy = "thompson"
    else:
        raise ValueError(f"Unknown main selection policy: {policy}")
    if decision_statistics is not None:
        decision_statistics.update({
            "predicted_means":np.asarray(predictions,dtype=float).tolist(),
            "working_variances":None if variances is None else np.asarray(variances,dtype=float).tolist(),
            "selection_values":np.asarray(values,dtype=float).tolist(),
            "service_policy":policy,
        })
    return int(np.argmax(values))


@dataclass(frozen=True)
class AcquisitionPlan:
    main_index: int
    main_subset: tuple
    alternate_index: int | None
    alternate_subset: tuple | None
    action_kind: str
    feedback_subsets: tuple
    predicted_cost: float
    reservation: ResourceCost
    information_gain: float
    score: float
    competitor_indices: tuple
    strategy: str

    @property
    def audit(self):
        return self.alternate_index is not None

    @property
    def labels_to_request(self):
        return self.feedback_subsets


def plan_acquisition(learner, features, subsets, *, main_index=None,
                     strategy="decision", main_feedback_expected=False,
                     predicted_costs=None, kappa=1.0, threshold=0.0,
                     available=None, bootstrap_coefficients=None,
                     round_index=0, period=5, rng=None, n_models=6, alpha=1.0):
    """Choose at most one alternate before current outputs/labels are observed.

    Only decision/no_cost use competitor directions and optional question-block
    bootstrap coefficients. Other baselines ignore those inputs, set unused
    directional information to zero, and retain the same affordable actions.
    A pure uncertainty action maximizes its reward working variance; the
    explicitly named per-cost variant divides this by complete predicted cost.
    Natural main feedback conditions the alternate's expected design but never
    mutates the fitted learner or creates an observed label.
    """
    if not isinstance(main_feedback_expected, bool):
        raise TypeError("main_feedback_expected must be an explicit preauthorized boolean")
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != learner.feature_dim or not np.isfinite(features).all():
        raise ValueError("Expected a finite feasible-action feature matrix")
    subsets = tuple(canonical_subset(s, n_models) for s in subsets)
    if len(features) != len(subsets) or not subsets or len(set(subsets)) != len(subsets):
        raise ValueError("Subset rows must be nonempty, unique and aligned")
    if len({len(s) for s in subsets}) != 1:
        raise ValueError("The primary protocol requires fixed subset size")
    if not np.isfinite([kappa, threshold]).all() or kappa < 0 or threshold < 0:
        raise ValueError("kappa/threshold must be finite and nonnegative")
    strategy = strategy.lower()
    strategy = {"ts": "thompson", "thompson_sampling": "thompson", "decision_directed": "decision", "random_audit": "random", "max_variance": "max_uncertainty"}.get(strategy, strategy)
    if strategy not in {"decision", "no_cost", "random", "max_uncertainty", "max_uncertainty_per_cost", "periodic", "none", "linucb", "thompson"}:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")
    if strategy == "periodic" and (not isinstance(period, int) or period <= 0 or round_index < 0):
        raise ValueError("Periodic baseline requires positive period and nonnegative round")
    if main_index is None:
        main_policy = strategy if strategy in {"linucb", "thompson"} else "greedy"
        main_index = choose_main(learner, features, policy=main_policy, alpha=alpha, rng=rng)
    if not isinstance(main_index, (int, np.integer)) or not 0 <= main_index < len(subsets):
        raise ValueError("Invalid main subset index")
    main_index = int(main_index)
    base = subsets[main_index]
    competitors = ()

    def no_purchase():
        return AcquisitionPlan(main_index, base, None, None, "none", (), 0.0,
                               ResourceCost(), 0.0, 0.0, competitors, strategy)

    if strategy == "none" or (strategy == "periodic" and round_index % period):
        return no_purchase()

    decision_strategy = strategy in {"decision", "no_cost"}
    variance_strategy = strategy in {"max_uncertainty", "max_uncertainty_per_cost"}
    directions = None
    if decision_strategy:
        predictions = learner.predict(features)
        differences = features[main_index] - features
        draws = bootstrap_coefficients
        if learner.label_count < learner.bootstrap_min_labels or learner.question_count < learner.bootstrap_min_questions:
            draws = None
        if draws is None:
            sd = np.sqrt(learner.working_variance(differences))
        else:
            draws = np.asarray(draws, dtype=float)
            if draws.ndim != 2 or draws.shape[1] != learner.feature_dim or len(draws) < 2 or not np.isfinite(draws).all():
                raise ValueError("Invalid question-block bootstrap coefficients")
            sd = np.std(differences @ draws.T, axis=1, ddof=1)
        gaps = predictions[main_index] - predictions
        competitors = tuple(i for i in range(len(subsets)) if i != main_index and gaps[i] <= kappa * sd[i] + 1e-12)
        if not competitors:
            return no_purchase()
        directions = differences[list(competitors)]

    predicted_costs = {} if predicted_costs is None else predicted_costs
    inverse = learner.design_inverse if decision_strategy or variance_strategy else None
    main_feature = features[main_index]
    after_main = None
    if decision_strategy or (variance_strategy and main_feedback_expected):
        after_main = rank_one_inverse_update(inverse, main_feature, learner.noise_variance)
    main_gain = float(np.mean(directional_information_gain(inverse, directions, main_feature, learner.noise_variance))) if decision_strategy else 0.0
    candidates = []

    def append_action(kind, alt_index, feedback, prediction, gain, variance):
        if prediction is None:
            return
        if not isinstance(prediction, CostPrediction):
            raise TypeError("Acquisition costs must be CostPrediction, never realized output costs")
        if prediction.upper_bound.labels < len(feedback):
            raise ValueError("Every requested label must have a hard budget reservation")
        if alt_index is not None and prediction.upper_bound.executions < 2:
            raise ValueError("Alternate execution must reserve one new model and one aggregation")
        if available is not None and not prediction.upper_bound.fits_within(available):
            return
        if strategy == "no_cost":
            score = gain
        elif strategy in {"random", "periodic", "linucb", "thompson"}:
            score = 1.0
        elif strategy == "max_uncertainty":
            score = variance
        elif strategy == "max_uncertainty_per_cost":
            score = variance / prediction.scalar
        else:
            score = gain / prediction.scalar
        candidates.append(AcquisitionPlan(main_index, base, alt_index,
            subsets[alt_index] if alt_index is not None else None, kind, feedback,
            float(prediction.scalar), prediction.upper_bound, gain, float(score), competitors, strategy))

    if not main_feedback_expected:
        main_variance = float(main_feature @ inverse @ main_feature) if variance_strategy else 0.0
        append_action("main_only", None, (base,), predicted_costs.get("main_only"), main_gain, main_variance)
    if strategy in {"linucb", "thompson"}:
        return candidates[0] if candidates and candidates[0].score > threshold else no_purchase()
    allowed = set(single_member_replacements(base, n_models))
    for alt_index, alternate in enumerate(subsets):
        if alternate not in allowed:
            continue
        prediction = predicted_costs.get(alternate, predicted_costs.get(alt_index))
        extra_gain = float(np.mean(directional_information_gain(after_main, directions, features[alt_index], learner.noise_variance))) if decision_strategy else 0.0
        gain = extra_gain if main_feedback_expected else main_gain + extra_gain
        expected_inverse = after_main if main_feedback_expected else inverse
        variance = float(features[alt_index] @ expected_inverse @ features[alt_index]) if variance_strategy else 0.0
        feedback = (alternate,) if main_feedback_expected else (base, alternate)
        append_action("alternate" if main_feedback_expected else "paired", alt_index, feedback, prediction, gain, variance)
    if not candidates:
        return no_purchase()
    if strategy in {"random", "periodic"}:
        if rng is None:
            raise ValueError("Stochastic baselines require an explicit RNG")
        selected = candidates[int(rng.integers(len(candidates)))]
    else:
        selected = max(candidates, key=lambda item: item.score)
    return selected if selected.score > threshold else no_purchase()
