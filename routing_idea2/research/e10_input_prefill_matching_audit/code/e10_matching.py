"""Analytic within-group matching decomposition for already-frozen policies.

Input rewards have axes question, generation, action. This module receives
arrays from the independent evaluator and performs no file or model access.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _labels(values, n, name):
    values = np.asarray(values)
    if values.shape != (n,):
        raise ValueError(f"{name} must have one entry per question")
    labels = values.astype(str)
    if np.any(labels == ""):
        raise ValueError(f"{name} contains an empty label")
    return labels


def _task_weights(task):
    labels, counts = np.unique(task, return_counts=True)
    weights = np.empty(len(task), dtype=np.float64)
    for label, count in zip(labels, counts):
        weights[task == label] = 1.0 / (len(labels) * count)
    return weights


def _metric_values(real_by_generation, coarse_by_generation, fine_by_generation, seeds):
    coarse_gain = real_by_generation - coarse_by_generation
    fine_gain = real_by_generation - fine_by_generation
    between = fine_by_generation - coarse_by_generation
    residual = coarse_gain - between - fine_gain

    def record(real, coarse, fine, coarse_g, fine_g, diff, error):
        return {
            "V_real": float(real),
            "V_mix_coarse": float(coarse),
            "V_mix_fine": float(fine),
            "G_match_coarse": float(coarse_g),
            "G_match_fine": float(fine_g),
            "V_mix_fine_minus_V_mix_coarse": float(diff),
            "decomposition_identity_residual": float(error),
        }

    result = record(
        real_by_generation.mean(), coarse_by_generation.mean(),
        fine_by_generation.mean(), coarse_gain.mean(), fine_gain.mean(),
        between.mean(), residual.mean(),
    )
    result["by_generation"] = {
        str(seed): record(*values)
        for seed, values in zip(
            seeds,
            zip(real_by_generation, coarse_by_generation, fine_by_generation,
                coarse_gain, fine_gain, between, residual),
        )
    }
    result["maximum_absolute_identity_residual"] = float(np.max(np.abs(residual)))
    return result


def _grouping(rewards, actions, task, subtask, outerfold, weights, seeds, fine):
    columns = [outerfold, task]
    fields = ["outerfold", "task"]
    if fine:
        columns.append(subtask)
        fields.append("subtask")
    keys = sorted(set(zip(*[column.tolist() for column in columns])))
    n, generations, action_count = rewards.shape
    mixed = np.empty((n, generations), dtype=np.float64)
    records = []
    singleton = np.zeros(n, dtype=bool)
    single_action = np.zeros(n, dtype=bool)
    nondegenerate = np.zeros(n, dtype=bool)
    covariance_residuals = []

    for key in keys:
        mask = np.ones(n, dtype=bool)
        for column, label in zip(columns, key):
            mask &= column == label
        group_rewards = rewards[mask]
        group_actions = actions[mask]
        group_n = len(group_actions)
        counts = np.bincount(group_actions, minlength=action_count)
        frequencies = counts / group_n
        distinct = int(np.count_nonzero(counts))
        # Every question receives the same group action marginal. The formula
        # integrates all count-preserving assignments without sampling them.
        mixed[mask] = np.einsum("nsa,a->ns", group_rewards, frequencies)
        chosen = group_rewards[np.arange(group_n), :, group_actions]
        real_by_seed = chosen.mean(axis=0)
        mix_by_seed = mixed[mask].mean(axis=0)
        gain_by_seed = real_by_seed - mix_by_seed
        indicators = np.eye(action_count)[group_actions]
        centered_indicators = indicators - frequencies
        covariance = np.einsum("na,nsa->s", centered_indicators, group_rewards) / group_n
        covariance_residuals.append(float(np.max(np.abs(covariance - gain_by_seed))))
        singleton[mask] = group_n == 1
        single_action[mask] = distinct == 1
        nondegenerate[mask] = group_n >= 2 and distinct >= 2
        records.append({
            **dict(zip(fields, key)),
            "n": group_n,
            "distinct_actions": distinct,
            "action_counts": counts.tolist(),
            "action_frequencies": frequencies.tolist(),
            "question_weight_in_view": float(weights[mask].sum()),
            "is_singleton": group_n == 1,
            "is_single_action": distinct == 1,
            "has_non_degenerate_support": group_n >= 2 and distinct >= 2,
            "V_real": float(real_by_seed.mean()),
            "V_mix": float(mix_by_seed.mean()),
            "G_match": float(gain_by_seed.mean()),
            "by_generation": {
                str(seed): {"V_real": float(real), "V_mix": float(mix), "G_match": float(gain)}
                for seed, real, mix, gain in zip(seeds, real_by_seed, mix_by_seed, gain_by_seed)
            },
        })

    def coverage(mask, group_flag):
        return {
            "groups": sum(bool(record[group_flag]) for record in records),
            "questions": int(mask.sum()),
            "question_fraction": float(mask.mean()),
            "task_balanced_weight": float(weights[mask].sum()),
        }

    result = {
        "fields": fields,
        "groups": records,
        "support": {
            "total_groups": len(records),
            "total_questions": n,
            "n_equals_one": coverage(singleton, "is_singleton"),
            "one_selected_action": coverage(single_action, "is_single_action"),
            "n_at_least_two_and_actions_at_least_two": coverage(
                nondegenerate, "has_non_degenerate_support"),
            "singleton_and_single_action_categories_overlap": True,
        },
        "maximum_absolute_covariance_identity_residual": max(covariance_residuals),
    }
    return mixed, result


def _view(rewards, actions, task, subtask, outerfold, seeds):
    weights = _task_weights(task)
    coarse_mix, coarse = _grouping(
        rewards, actions, task, subtask, outerfold, weights, seeds, fine=False)
    fine_mix, fine = _grouping(
        rewards, actions, task, subtask, outerfold, weights, seeds, fine=True)
    selected = rewards[np.arange(len(actions)), :, actions]
    metrics = _metric_values(
        weights @ selected, weights @ coarse_mix, weights @ fine_mix, seeds)
    return {
        "n": len(actions),
        "task_counts": {str(label): int(np.sum(task == label)) for label in np.unique(task)},
        "task_mass": {str(label): float(weights[task == label].sum()) for label in np.unique(task)},
        "metrics": metrics,
        "coarse": coarse,
        "fine": fine,
    }


def decompose_policy(
    rewards, actions, task, subtask, outerfold, *, generation_seeds=None, action_keys=None,
):
    """Return exact expected mixtures and matching gains for one frozen policy.

    Each view assigns equal mass to its present tasks and equal mass to
    questions within each task. E10's overall view has gsm8k and bbh each at
    1/2. With the historical balanced train600 this is also question-uniform.
    Coarse and fine group frequencies are computed within the reported view.
    A generation uses the identical frozen actions and grouping frequencies.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 3 or min(rewards.shape) < 1:
        raise ValueError("rewards must have nonempty question, generation, action axes")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards contain nonfinite values")
    n, generations, action_count = rewards.shape
    actions = np.asarray(actions)
    if actions.shape != (n,) or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("actions must be an integer array with one action per question")
    if np.any(actions < 0) or np.any(actions >= action_count):
        raise ValueError("a frozen action is outside the reward action axis")
    task = _labels(task, n, "task")
    subtask = _labels(subtask, n, "subtask")
    outerfold = _labels(outerfold, n, "outerfold")
    subtask = np.where(task == "gsm8k", "gsm8k", subtask)
    seeds = list(range(generations)) if generation_seeds is None else list(generation_seeds)
    if len(seeds) != generations or len(set(map(str, seeds))) != generations:
        raise ValueError("generation_seeds must uniquely label the generation axis")
    keys = list(range(action_count)) if action_keys is None else list(action_keys)
    if len(keys) != action_count or len(set(map(str, keys))) != action_count:
        raise ValueError("action_keys must uniquely label the action axis")
    overall = _view(rewards, actions, task, subtask, outerfold, seeds)

    def restricted(mask):
        return _view(rewards[mask], actions[mask], task[mask], subtask[mask], outerfold[mask], seeds)

    return {
        "status": "EXPLORATORY_REUSED_TRAIN600",
        "method": "analytic expectation over within-group count-preserving assignments",
        "reward_axes": ["question", "generation", "action"],
        "generation_seeds": seeds,
        "action_keys": keys,
        "units": "reward proportion",
        "weighting": "equal present-task mass, uniform questions within each task in every view",
        "fine_gsm8k_subtask": "gsm8k",
        "overall": overall,
        "by_task": {str(label): restricted(task == label) for label in np.unique(task)},
        "by_outerfold": {
            str(label): restricted(outerfold == label) for label in np.unique(outerfold)
        },
        "by_bbh_subtask": {
            str(label): restricted((task == "bbh") & (subtask == label))
            for label in np.unique(subtask[task == "bbh"])
        },
    }


def decompose_policies(
    rewards, policies: Mapping[str, Sequence[int]], task, subtask, outerfold, *,
    generation_seeds=None, action_keys=None,
):
    """Reuse the same evaluation definition for E9 and frozen E10 policies."""
    if not policies:
        raise ValueError("at least one frozen policy is required")
    return {
        str(name): decompose_policy(
            rewards, actions, task, subtask, outerfold,
            generation_seeds=generation_seeds, action_keys=action_keys,
        )
        for name, actions in policies.items()
    }
