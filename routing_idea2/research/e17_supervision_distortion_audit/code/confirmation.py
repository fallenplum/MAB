"""Input-only confirmation planning and working-normal power calculations.

This module neither fits a predictor nor reads evaluation data. Question-level
paired differences supplied to ``paired_power_plan`` must already average the
three within-question generation repeats. Cache-key subtraction verifies logical
execution requirements only and grants no permission to reveal evaluation gold.
"""
from __future__ import annotations

from itertools import combinations
from math import ceil, sqrt
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np


def paired_power_plan(
    difference_by_question,
    task_labels,
    effect_pps=(0.5, 1),
    n_values=(400,),
    alpha=0.05,
    target_power=0.8,
) -> dict:
    """Plan a fixed one-sided test using equally weighted GSM8K/BBH variance.

    The normal approximation assumes an independent, task-balanced new question
    sample and transports the supplied development paired variance. It has no
    coverage or future-variance guarantee. A question is one observation even
    when its reward averages three correlated generation records.
    """
    differences = np.asarray(difference_by_question, dtype=np.float64)
    labels = np.asarray(task_labels)
    if differences.ndim != 1 or labels.ndim != 1:
        raise ValueError("differences and task_labels must be one-dimensional")
    if len(differences) != len(labels) or not np.all(np.isfinite(differences)):
        raise ValueError("finite differences and matching labels are required")
    if np.any(np.abs(differences) > 1 + 1e-12):
        raise ValueError("question-level correctness differences must lie in [-1, 1]")
    labels = np.array([str(label).strip().lower() for label in labels])
    if set(labels) != {"gsm8k", "bbh"}:
        raise ValueError("task labels must contain exactly GSM8K and BBH")
    if not 0 < alpha < 0.5 or not 0.5 < target_power < 1:
        raise ValueError("require 0 < alpha < .5 and .5 < target_power < 1")

    effects = tuple(float(value) for value in effect_pps)
    if not effects or any(not np.isfinite(value) or value <= 0 for value in effects):
        raise ValueError("effect_pps must contain finite positive percentage points")
    sample_sizes = _integer_array(n_values, "n_values", ndim=1)
    if not len(sample_sizes) or np.any(sample_sizes <= 0):
        raise ValueError("n_values must contain positive question counts")

    groups = {}
    for task in ("gsm8k", "bbh"):
        values = differences[labels == task]
        if len(values) < 2:
            raise ValueError("each task requires at least two independent questions")
        groups[task] = {
            "questions": int(len(values)),
            "mean_paired_difference": float(values.mean()),
            "sample_variance_ddof1": float(values.var(ddof=1)),
            "target_weight": 0.5,
        }
    variance = sum(0.5 * group["sample_variance_ddof1"] for group in groups.values())
    sigma = sqrt(variance)
    normal = NormalDist()
    critical_z = normal.inv_cdf(1 - alpha)
    target_z = normal.inv_cdf(target_power)
    zero_variance = sigma == 0.0
    effect_rows = []
    for effect_pp in effects:
        delta = effect_pp / 100
        required = None if zero_variance else max(1, ceil(((critical_z + target_z) * sigma / delta) ** 2))
        effect_rows.append({
            "effect_pp": effect_pp,
            "effect_fraction": delta,
            "required_n": required,
            "power_by_n": [
                {
                    "n_questions": int(n),
                    "normal_approximation_power": None if zero_variance else normal.cdf(sqrt(int(n)) * delta / sigma - critical_z),
                }
                for n in sample_sizes
            ],
        })
    observed = sum(0.5 * group["mean_paired_difference"] for group in groups.values())
    return {
        "schema": "e17_fixed_time_paired_normal_power_plan_v1",
        "question_unit": "one_original_question_with_generation_repeats_already_averaged",
        "independent_questions": int(len(differences)),
        "groups": groups,
        "observed_task_balanced_difference": observed,
        "observed_task_balanced_difference_pp": 100 * observed,
        "pooled_within_task_variance": variance,
        "sigma_question_mean": sigma,
        "alpha_one_sided": float(alpha),
        "target_power": float(target_power),
        "critical_z": critical_z,
        "target_power_z": target_z,
        "effects": effect_rows,
        "zero_development_variance": zero_variance,
        "positive_effect_zero_variance_limit": 1.0 if zero_variance else None,
        "zero_variance_note": (
            "Normal planning is undefined at estimated sigma=0; no zero-question confirmation claim is made. "
            "The positive-effect limit is 1 only under a truly zero-variance working model; development zero variance does not establish future zero variance."
            if zero_variance else None
        ),
        "assumptions": [
            "Fixed one-sided test at a predeclared sample size, with no optional stopping.",
            "New independent questions are sampled GSM8K/BBH equally; BBH is question-weighted.",
            "Development within-task paired variance transports to the new question population.",
            "Normal approximation and variance plug-in are planning assumptions, not guarantees.",
            "A positive-gain test and a lower-bound-above-one-percentage-point test require different effect definitions.",
        ],
        "confidence_intervals_computed": False,
        "bootstrap_repetitions": 0,
    }


def input_call_union(policy_actions, teams, seeds) -> dict:
    """Count semantic model calls required by frozen input-only policy actions.

    Key tuples are ``(input_position, generation_seed, action_or_member_index)``.
    They deliberately contain no original question text, gold, or private IDs.
    Every question requires at most one gold reveal shared by all policies and
    repeats. Cached outputs and already-revealed gold are separate permissions.
    """
    actions = _integer_array(policy_actions, "policy_actions", ndim=2)
    team_array = _integer_array(teams, "teams", ndim=2)
    generation_seeds = _integer_array(seeds, "seeds", ndim=1)
    if actions.shape[0] < 1:
        raise ValueError("at least one frozen policy is required")
    if team_array.shape != (20, 3):
        raise ValueError("teams must have exactly 20 rows of 3 members")
    semantic_teams = {tuple(sorted(map(int, team))) for team in team_array}
    if semantic_teams != set(combinations(range(6), 3)):
        raise ValueError("teams must be exactly the 20 unique six-model K=3 combinations")
    if not len(generation_seeds) or len(set(map(int, generation_seeds))) != len(generation_seeds):
        raise ValueError("generation seeds must be nonempty and unique")
    if np.any(actions < 0) or np.any(actions >= 20):
        raise ValueError("policy action indices must be in [0, 19]")

    aggregate_keys = set()
    member_keys = set()
    for question in range(actions.shape[1]):
        for action in set(map(int, actions[:, question])):
            for generation in map(int, generation_seeds):
                aggregate_keys.add((question, generation, action))
                member_keys.update((question, generation, int(member)) for member in team_array[action])
    return {
        "schema": "e17_input_only_execution_union_v1",
        "question_key_semantics": "input_array_position",
        "generation_key_semantics": "provided_generation_seed",
        "policies": int(actions.shape[0]),
        "questions": int(actions.shape[1]),
        "seeds": list(map(int, generation_seeds)),
        "teams": team_array.tolist(),
        "member_keys": sorted(member_keys),
        "aggregate_keys": sorted(aggregate_keys),
        "gold_question_indices": list(range(actions.shape[1])),
        "counts": {
            "member_calls": len(member_keys),
            "aggregate_calls": len(aggregate_keys),
            "scoring_ops": len(aggregate_keys),
            "gold_questions": int(actions.shape[1]),
        },
        "new_physical_calls_are_not_inferred_from_cache_filenames": True,
        "authorizes_evaluation": False,
    }


def subtract_verified_cache_keys(union: dict, member_keys: Iterable, aggregate_keys: Iterable) -> dict:
    """Subtract only positively verified output-cache keys from a frozen union.

    An existing aggregate output needs no member regeneration. It does not prove
    that any constituent member output is itself present. Gold revelation and
    deterministic grading remain separately budgeted unless the caller verifies
    those permissions outside this helper.
    """
    required_members = _key_set(union["member_keys"], "union member keys")
    required_aggregates = _key_set(union["aggregate_keys"], "union aggregate keys")
    verified_members = _key_set(member_keys, "verified member keys")
    verified_aggregates = _key_set(aggregate_keys, "verified aggregate keys")
    teams = _integer_array(union["teams"], "union teams", ndim=2)
    missing_aggregates = required_aggregates - verified_aggregates
    dependency_members = {
        (question, generation, int(member))
        for question, generation, action in missing_aggregates
        for member in teams[action]
    }
    if not dependency_members.issubset(required_members):
        raise ValueError("aggregate dependencies do not agree with the frozen member union")
    new_members = dependency_members - verified_members
    unverified_original_members = required_members - verified_members
    return {
        "schema": "e17_verified_execution_cache_subtraction_v1",
        "verified_required_member_keys": sorted(required_members & verified_members),
        "verified_required_aggregate_keys": sorted(required_aggregates & verified_aggregates),
        "unverified_original_union_member_keys": sorted(unverified_original_members),
        "new_member_keys": sorted(new_members),
        "new_aggregate_keys": sorted(missing_aggregates),
        "ignored_foreign_member_cache_keys": len(verified_members - required_members),
        "ignored_foreign_aggregate_cache_keys": len(verified_aggregates - required_aggregates),
        "counts": {
            "new_member_calls": len(new_members),
            "new_aggregate_calls": len(missing_aggregates),
            "scoring_ops_to_budget_separately": int(union["counts"]["scoring_ops"]),
            "gold_questions_to_authorize_separately": int(union["counts"]["gold_questions"]),
        },
        "zero_new_model_executions": not new_members and not missing_aggregates,
        "cache_key_verification_is_callers_responsibility": True,
        "authorizes_evaluation": False,
    }


def _integer_array(values, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a {ndim}-dimensional integer array")
    return array.astype(np.int64, copy=False)


def _key_set(values: Iterable[Sequence[int]], name: str) -> set[tuple[int, int, int]]:
    result = set()
    for key in values:
        array = _integer_array(key, name, ndim=1)
        if array.shape != (3,):
            raise ValueError(f"{name} must contain integer triples")
        result.add(tuple(map(int, array)))
    return result
