"""Leave-question-out hierarchical label smoothing and equal-capacity Ridge."""

from __future__ import annotations

import time

import numpy as np

from train_psmr import _validate_design, reward_mean


def _metadata(task, subtask, n_questions):
    tasks = np.asarray(task, dtype=str)
    subtasks = np.asarray(subtask, dtype=str)
    if tasks.shape != (n_questions,) or subtasks.shape != (n_questions,):
        raise ValueError("task and subtask must have one entry per fitting question")
    return tasks, subtasks


def _check_pseudocounts(task_pseudocount, subtask_pseudocount):
    if not np.isfinite(task_pseudocount) or task_pseudocount <= 0:
        raise ValueError("task pseudocount must be positive")
    if not np.isfinite(subtask_pseudocount) or subtask_pseudocount <= 0:
        raise ValueError("subtask pseudocount must be positive")


def _group_indices(tasks, subtasks):
    task_indices = {key: np.flatnonzero(tasks == key) for key in np.unique(tasks)}
    subgroup_indices = {}
    for key in sorted(set(zip(tasks.tolist(), subtasks.tolist()))):
        subgroup_indices[key] = np.flatnonzero((tasks == key[0]) & (subtasks == key[1]))
    return task_indices, subgroup_indices


def _uses_subtask(task, subtask):
    return task.lower() == "bbh" and subtask not in ("", "None", "nan")


def hierarchical_loo_prior(
    rewards, task, subtask, task_pseudocount=10.0, subtask_pseudocount=3.0
):
    """Exclude the entire current question from all three prior levels.

    Task means shrink toward the leave-question-out global mean. BBH subtask
    means shrink toward that task estimate. GSM uses only the task estimate.
    A single-question fitting set has a prespecified weak 0.5 prior.
    """
    targets, _ = reward_mean(rewards)
    n_questions = len(targets)
    tasks, subtasks = _metadata(task, subtask, n_questions)
    _check_pseudocounts(task_pseudocount, subtask_pseudocount)
    task_indices, subgroup_indices = _group_indices(tasks, subtasks)
    global_sum = targets.sum(axis=0)
    task_sums = {key: targets[index].sum(axis=0) for key, index in task_indices.items()}
    subgroup_sums = {key: targets[index].sum(axis=0) for key, index in subgroup_indices.items()}
    priors = np.empty_like(targets)
    for index in range(n_questions):
        global_prior = (
            (global_sum - targets[index]) / (n_questions - 1)
            if n_questions > 1
            else np.full(targets.shape[1], 0.5)
        )
        task_key = tasks[index]
        task_count = len(task_indices[task_key]) - 1
        task_prior = (
            task_sums[task_key] - targets[index] + task_pseudocount * global_prior
        ) / (task_count + task_pseudocount)
        priors[index] = task_prior
        if _uses_subtask(task_key, subtasks[index]):
            group_key = (task_key, subtasks[index])
            group_count = len(subgroup_indices[group_key]) - 1
            priors[index] = (
                subgroup_sums[group_key] - targets[index] + subtask_pseudocount * task_prior
            ) / (group_count + subtask_pseudocount)
    return priors


def predict_historical_prior(
    train_rewards,
    train_task,
    train_subtask,
    query_task,
    query_subtask,
    task_pseudocount=10.0,
    subtask_pseudocount=3.0,
):
    """Metadata-only D3 control, using fitting rewards and query metadata.

    These query task/subtask labels constitute an explicit metadata permission.
    No query rewards, answers or text embeddings are accepted by this function.
    """
    targets, _ = reward_mean(train_rewards)
    tasks, subtasks = _metadata(train_task, train_subtask, len(targets))
    query_tasks = np.asarray(query_task, dtype=str)
    query_subtasks = np.asarray(query_subtask, dtype=str)
    if query_tasks.ndim != 1 or query_subtasks.shape != query_tasks.shape:
        raise ValueError("query task and subtask shapes disagree")
    _check_pseudocounts(task_pseudocount, subtask_pseudocount)
    task_indices, subgroup_indices = _group_indices(tasks, subtasks)
    global_prior = targets.mean(axis=0)
    output = np.empty((len(query_tasks), targets.shape[1]), dtype=np.float64)
    for index, (task_key, subtask_key) in enumerate(zip(query_tasks, query_subtasks)):
        task_index = task_indices.get(task_key, np.empty(0, dtype=int))
        task_prior = (
            targets[task_index].sum(axis=0) + task_pseudocount * global_prior
        ) / (len(task_index) + task_pseudocount)
        output[index] = task_prior
        if _uses_subtask(task_key, subtask_key):
            group_index = subgroup_indices.get((task_key, subtask_key), np.empty(0, dtype=int))
            output[index] = (
                targets[group_index].sum(axis=0) + subtask_pseudocount * task_prior
            ) / (len(group_index) + subtask_pseudocount)
    return output


def shrink_targets(
    rewards, task, subtask, shrinkage, task_pseudocount=10.0, subtask_pseudocount=3.0
):
    targets, _ = reward_mean(rewards)
    if not np.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("shrinkage must be finite and nonnegative")
    priors = hierarchical_loo_prior(
        rewards, task, subtask, task_pseudocount, subtask_pseudocount
    )
    return (3.0 * targets + shrinkage * priors) / (3.0 + shrinkage)


def fit_hsmr(
    design,
    rewards,
    task,
    subtask,
    ridge_alpha,
    shrinkage,
    task_pseudocount=10.0,
    subtask_pseudocount=3.0,
):
    """Normal equations use +ridge_alpha * I, including the constant column."""
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    targets = shrink_targets(
        rewards, task, subtask, shrinkage, task_pseudocount, subtask_pseudocount
    )
    design = _validate_design(design, len(targets))
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("ridge_alpha must be finite and nonnegative")
    gram = design.T @ design + ridge_alpha * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ targets)
    errors = design @ weights - targets
    l2 = ridge_alpha / targets.size
    penalty = float(l2 * np.sum(weights ** 2))
    gradient = 2.0 * design.T @ errors / targets.size + 2.0 * l2 * weights
    diagnostics = {
        "family": "HSMR",
        "n_questions": len(targets),
        "n_actions": targets.shape[1],
        "n_parameters": weights.size,
        "shrinkage": float(shrinkage),
        "generation_repeats": 3,
        "task_prior_pseudocount": float(task_pseudocount),
        "subtask_prior_pseudocount": float(subtask_pseudocount),
        "prior_rule": "whole-question LOO at global, task and BBH-subtask levels",
        "ridge_alpha": float(ridge_alpha),
        "l2": float(l2),
        "normalization": "mean squared error divided by n_questions * n_actions",
        "constant_column_penalized": True,
        "data_loss": float(np.mean(errors ** 2)),
        "regularization": penalty,
        "objective": float(np.mean(errors ** 2) + penalty),
        "weight_norm": float(np.linalg.norm(weights)),
        "gradient_l2": float(np.linalg.norm(gradient)),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "optimizer": "float64 numpy.linalg.solve",
        "optimizer_success": bool(np.isfinite(weights).all()),
        "cpu_seconds": float(time.process_time() - cpu_start),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    return {"weights": weights, "diagnostics": diagnostics, "targets": targets}
