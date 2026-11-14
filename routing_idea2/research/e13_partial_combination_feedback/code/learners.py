"""Sparse-observation learners for E13's shared partial-feedback experiment.

Every reward argument is one actually purchased query/action/generation
observation. No dense counterfactual reward matrix is accepted. Query metadata
at prediction time is represented only by the caller's frozen predicted group.
"""
from __future__ import annotations

import time

import numpy as np


METHODS = (
    "Raw", "Group", "GroupResidual", "RawCentered",
    "GlobalFixed", "TaskFixed", "SubtaskFixed",
)
TASK_NAMES = ("gsm8k", "bbh")
ALPHA_OBSERVATION = 1000.0


def _integer_vector(values, length, upper, name):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != length:
        raise ValueError(name + " must be a one-dimensional aligned index vector")
    if values.size and values.dtype.kind not in "iu":
        raise ValueError(name + " must contain integer indices")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= upper):
        raise ValueError(name + " contains an index outside its permitted range")
    return values


def _task_codes(group_to_task):
    labels = np.asarray(group_to_task, dtype=str)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("group_to_task must label each training group")
    normalized = np.char.lower(labels).astype("U5")
    normalized[normalized == "gsm"] = "gsm8k"
    if not set(normalized.tolist()) <= set(TASK_NAMES):
        raise ValueError("group_to_task labels must be GSM8K/GSM or BBH")
    codes = np.array([TASK_NAMES.index(name) for name in normalized], dtype=np.int64)
    return normalized, codes


def _validate_training(phi_h, group_h, group_to_task, query_indices,
                       action_indices, rewards, n_actions):
    phi_h = np.asarray(phi_h, dtype=np.float64)
    if phi_h.ndim != 2 or not len(phi_h) or not phi_h.shape[1]:
        raise ValueError("phi_h must be a nonempty two-dimensional input-only design")
    if not np.isfinite(phi_h).all() or not np.all(phi_h[:, 0] == 1.0):
        raise ValueError("phi_h must be finite and retain its fixed constant column")
    if not isinstance(n_actions, (int, np.integer)) or n_actions < 1:
        raise ValueError("n_actions must be a positive integer")
    labels, task_codes = _task_codes(group_to_task)
    group_h = _integer_vector(group_h, len(phi_h), len(labels), "group_h")
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1:
        raise ValueError("Only a vector of purchased scalar rewards is accepted")
    if not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1)):
        raise ValueError("Purchased rewards must be finite in [0,1]")
    query = _integer_vector(query_indices, len(rewards), len(phi_h), "query_indices")
    action = _integer_vector(action_indices, len(rewards), n_actions, "action_indices")
    return phi_h, group_h, labels, task_codes, query, action, rewards


def _observed_statistics(query, action, rewards, group_h, task_codes, n_actions):
    n_groups = len(task_codes)
    group = group_h[query]
    task = task_codes[group]
    group_count = np.zeros((n_groups, n_actions), dtype=np.int64)
    group_sum = np.zeros((n_groups, n_actions), dtype=np.float64)
    np.add.at(group_count, (group, action), 1)
    np.add.at(group_sum, (group, action), rewards)
    task_count = np.zeros((len(TASK_NAMES), n_actions), dtype=np.int64)
    task_sum = np.zeros((len(TASK_NAMES), n_actions), dtype=np.float64)
    np.add.at(task_count, (task, action), 1)
    np.add.at(task_sum, (task, action), rewards)
    return dict(
        group_count=group_count, group_sum=group_sum,
        task_count=task_count, task_sum=task_sum,
        action_count=np.bincount(action, minlength=n_actions),
        action_sum=np.bincount(action, weights=rewards, minlength=n_actions),
        total_count=int(len(rewards)), total_sum=float(rewards.sum()),
    )


def _capabilities(statistics, task_codes, task_pseudocount, subgroup_pseudocount):
    grand = (statistics["total_sum"] / statistics["total_count"]
             if statistics["total_count"] else 0.5)
    action_count, action_sum = statistics["action_count"], statistics["action_sum"]
    global_a = (action_sum + task_pseudocount * grand) / (action_count + task_pseudocount)
    task_a = (statistics["task_sum"] + task_pseudocount * global_a[None, :]) / (
        statistics["task_count"] + task_pseudocount)
    group_a = task_a[task_codes].copy()
    bbh = task_codes == TASK_NAMES.index("bbh")
    group_a[bbh] = (
        statistics["group_sum"][bbh] + subgroup_pseudocount * task_a[TASK_NAMES.index("bbh")]
    ) / (statistics["group_count"][bbh] + subgroup_pseudocount)
    return grand, global_a, task_a, group_a


def _empirical_fixed_values(statistics, task_codes, grand):
    """Observed empirical means with explicit grand/global/task fallback."""
    action_count, action_sum = statistics["action_count"], statistics["action_sum"]
    global_values = np.full(len(action_count), grand, dtype=np.float64)
    np.divide(action_sum, action_count, out=global_values, where=action_count > 0)
    task_values = np.repeat(global_values[None, :], len(TASK_NAMES), axis=0)
    np.divide(statistics["task_sum"], statistics["task_count"], out=task_values,
              where=statistics["task_count"] > 0)
    group_values = task_values[task_codes].copy()
    np.divide(statistics["group_sum"], statistics["group_count"], out=group_values,
              where=statistics["group_count"] > 0)
    return global_values, task_values, group_values


def _without_whole_question(statistics, rows, action, rewards, group, task):
    count = np.bincount(action[rows], minlength=len(statistics["action_count"]))
    reward_sum = np.bincount(action[rows], weights=rewards[rows], minlength=len(count))
    reduced = {key: value.copy() if isinstance(value, np.ndarray) else value
               for key, value in statistics.items()}
    reduced["action_count"] -= count
    reduced["action_sum"] -= reward_sum
    reduced["task_count"][task] -= count
    reduced["task_sum"][task] -= reward_sum
    reduced["group_count"][group] -= count
    reduced["group_sum"][group] -= reward_sum
    reduced["total_count"] -= len(rows)
    reduced["total_sum"] -= float(rewards[rows].sum())
    return reduced


def fit_learners(phi_h, group_h, group_to_task, query_indices, action_indices,
                 rewards, alpha=ALPHA_OBSERVATION, n_actions=20,
                 task_pseudocount=10.0, subgroup_pseudocount=3.0):
    """Fit all seven models using only the supplied purchased observations.

    Ridge uses the sum of individual single-generation squared errors plus
    alpha times the entire coefficient norm, including the constant. All three
    Ridge variants share each action's normal matrix and one three-RHS solve.
    """
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    phi_h, group_h, task_labels, task_codes, query, action, rewards = _validate_training(
        phi_h, group_h, group_to_task, query_indices, action_indices, rewards, n_actions)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if (not np.isfinite(task_pseudocount) or task_pseudocount <= 0
            or not np.isfinite(subgroup_pseudocount) or subgroup_pseudocount <= 0):
        raise ValueError("Hierarchical pseudocounts must be finite and positive")
    statistics = _observed_statistics(query, action, rewards, group_h, task_codes, n_actions)
    grand, global_a, task_a, capabilities = _capabilities(
        statistics, task_codes, task_pseudocount, subgroup_pseudocount)
    global_fixed, task_fixed, subtask_fixed = _empirical_fixed_values(statistics, task_codes, grand)
    loo_baseline = np.empty(len(rewards), dtype=np.float64)
    excluded_counts = []
    loo_empty_observations = 0
    for question in np.unique(query):
        rows = np.flatnonzero(query == question)
        group = int(group_h[question])
        task = int(task_codes[group])
        reduced = _without_whole_question(statistics, rows, action, rewards, group, task)
        _, _, _, reduced_c = _capabilities(reduced, task_codes, task_pseudocount, subgroup_pseudocount)
        loo_baseline[rows] = reduced_c[group, action[rows]]
        excluded_counts.append(len(rows))
        if not reduced["total_count"]:
            loo_empty_observations += len(rows)
    residual = rewards - loo_baseline
    centered = rewards - global_a[action]
    raw_weights = np.zeros((phi_h.shape[1], n_actions), dtype=np.float64)
    residual_weights = np.zeros_like(raw_weights)
    centered_weights = np.zeros_like(raw_weights)
    solve_records = []
    for column in range(n_actions):
        rows = np.flatnonzero(action == column)
        if not len(rows):
            continue
        design = phi_h[query[rows]]
        normal = design.T @ design + alpha * np.eye(phi_h.shape[1])
        targets = np.column_stack((rewards[rows], residual[rows], centered[rows]))
        right = design.T @ targets
        solve_cpu, solve_wall = time.process_time(), time.perf_counter()
        solution = np.linalg.solve(normal, right)
        solve_records.append(dict(action=column, observations=len(rows), rhs_columns=3,
                                  cpu_s=time.process_time()-solve_cpu,
                                  wall_s=time.perf_counter()-solve_wall,
                                  normal_equation_max_residual=float(np.max(np.abs(normal @ solution-right)))))
        raw_weights[:, column] = solution[:, 0]
        residual_weights[:, column] = solution[:, 1]
        centered_weights[:, column] = solution[:, 2]
    losses = {}
    for name, weights, targets in (("Raw", raw_weights, rewards),
                                    ("GroupResidual", residual_weights, residual),
                                    ("RawCentered", centered_weights, centered)):
        fitted = np.einsum("nd,nd->n", phi_h[query], weights[:, action].T)
        losses[name] = dict(data_squared_error_sum=float(np.sum((fitted-targets)**2)),
                            penalty=float(alpha*np.sum(weights**2)),
                            weight_norm=float(np.linalg.norm(weights)))
    task_empty = statistics["task_count"] == 0
    group_empty = statistics["group_count"] == 0
    fallback = dict(
        grand_weak_prior_used=not bool(len(rewards)),
        unobserved_action_columns=np.flatnonzero(statistics["action_count"] == 0),
        empty_task_action_cells=np.argwhere(task_empty),
        empty_group_action_cells=np.argwhere(group_empty),
        raw_zero_weight_prior_actions=np.flatnonzero(statistics["action_count"] == 0),
        subtask_fixed_fallback_to_task_cells=int(np.sum(group_empty & ~task_empty[task_codes])),
        subtask_fixed_fallback_to_global_cells=int(np.sum(group_empty & task_empty[task_codes])),
        task_fixed_fallback_to_global_cells=int(np.sum(task_empty)),
        global_fixed_fallback_to_grand_actions=int(np.sum(statistics["action_count"] == 0)),
        loo_observations_with_no_other_question_feedback=loo_empty_observations,
    )
    return dict(
        raw_weights=raw_weights, residual_weights=residual_weights,
        centered_weights=centered_weights, C=capabilities, global_a=global_a, task_a=task_a,
        global_fixed=global_fixed, task_fixed=task_fixed, subtask_fixed=subtask_fixed,
        group_to_task=task_labels, task_codes=task_codes,
        observed_counts=statistics, grand=grand, loo_baseline=loo_baseline,
        residual_targets=residual, centered_targets=centered,
        fallback=fallback,
        diagnostics=dict(
            alpha=float(alpha), observation_unit="one purchased query/action/single-generation scalar reward",
            loss="sum of observed single-generation squared errors + alpha * entire weight norm squared",
            penalized_constant=True, n_parameters_per_ridge_head=int(raw_weights.size),
            n_observations=len(rewards), n_feedback_questions=len(np.unique(query)),
            n_input_questions=len(phi_h), n_actions=n_actions, n_groups=len(task_codes),
            task_pseudocount=float(task_pseudocount), subgroup_pseudocount=float(subgroup_pseudocount),
            group_prior="grand -> action -> task -> BBH subgroup; GSM uses task capability",
            group_residual_prior="exclude all purchased action observations for the current original question",
            raw_centering="global_a fitted from all purchased feedback, including current labels; not LOO",
            missing_reward_rows_created=0,
            actual_normal_matrix_solves=len(solve_records), model_rhs_fits=3*len(solve_records),
            solve_records=solve_records, losses=losses,
            maximum_excluded_observations_per_question=max(excluded_counts, default=0),
            cpu_s=time.process_time()-cpu_start, wall_s=time.perf_counter()-wall_start,
        ),
    )


def predict_learners(state, phi_query, predicted_group):
    """Return scores only, using caller-supplied input-predicted groups."""
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    phi_query = np.asarray(phi_query, dtype=np.float64)
    dimension, n_actions = state["raw_weights"].shape
    if (phi_query.ndim != 2 or phi_query.shape[1] != dimension
            or not np.isfinite(phi_query).all() or not np.all(phi_query[:, 0] == 1.0)):
        raise ValueError("Prediction design must match the frozen finite constant-containing feature design")
    predicted_group = _integer_vector(predicted_group, len(phi_query), len(state["C"]), "predicted_group")
    predicted_task = state["task_codes"][predicted_group]
    group_scores = state["C"][predicted_group]
    scores = dict(
        Raw=phi_query @ state["raw_weights"],
        Group=group_scores.copy(),
        GroupResidual=group_scores + phi_query @ state["residual_weights"],
        RawCentered=state["global_a"][None, :] + phi_query @ state["centered_weights"],
        GlobalFixed=np.repeat(state["global_fixed"][None, :], len(phi_query), axis=0),
        TaskFixed=state["task_fixed"][predicted_task],
        SubtaskFixed=state["subtask_fixed"][predicted_group],
    )
    if any(values.shape != (len(phi_query), n_actions) or not np.isfinite(values).all()
           for values in scores.values()):
        raise FloatingPointError("Learner produced invalid prediction scores")
    state["prediction_timing"] = dict(cpu_s=time.process_time()-cpu_start, wall_s=time.perf_counter()-wall_start,
                                     n_queries=len(phi_query))
    return scores


def fit_and_predict(phi_h, group_h, group_to_task, query_indices, action_indices,
                    rewards, phi_query, predicted_group, **settings):
    state = fit_learners(phi_h, group_h, group_to_task, query_indices,
                         action_indices, rewards, **settings)
    return dict(scores=predict_learners(state, phi_query, predicted_group), state=state)
