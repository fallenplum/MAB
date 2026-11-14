"""E14-B diagnostics for frozen, sparsely estimated group capabilities.

This module performs no I/O, fitting, sampling or confidence-interval work.
Both diagnostic policies use the original capability table with the true
heldout group. GroupResidual adds only the original frozen residual head.
True query groups and hindsight reward comparisons are diagnostic permissions.
"""
from __future__ import annotations

import numpy as np


SUPPORT_BINS = ("0", "1", "2", "3-4", "5-9", "10+")
METHODS = ("GroupOracle", "GroupResidualOracle")


def support_bin(count):
    """Prespecified bins for distinct purchased question/action observations."""
    count = int(count)
    if count < 0:
        raise ValueError("Support cannot be negative")
    if count <= 2:
        return str(count)
    if count <= 4:
        return "3-4"
    if count <= 9:
        return "5-9"
    return "10+"


def _indices(values, size, name):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError(name + " must be a one-dimensional integer index vector")
    result = values.astype(np.int64, copy=False)
    if np.any((result < 0) | (result >= size)):
        raise ValueError(name + " is outside the frozen group/action/question domain")
    return result


def _read_inputs(inputs, snapshot, rewards_e, rewards_mean_e):
    capability = np.asarray(snapshot["own__C"], dtype=np.float64)
    counts = np.asarray(snapshot["own_counts__group_count"])
    reward_sum = np.asarray(snapshot["own_counts__group_sum"], dtype=np.float64)
    if capability.ndim != 2 or not capability.size or not np.isfinite(capability).all():
        raise ValueError("The original frozen group capability table must be finite")
    if (counts.shape != capability.shape or counts.dtype.kind not in "iu"
            or np.any(counts < 0) or reward_sum.shape != capability.shape
            or not np.isfinite(reward_sum).all() or np.any(reward_sum < 0)
            or np.any(reward_sum > counts)):
        raise ValueError("Frozen observed group/action support statistics are invalid")
    n_groups, n_actions = capability.shape
    groups = _indices(inputs["oracle_group_e"], n_groups, "oracle_group_e")
    train_groups = _indices(inputs["group_h"], n_groups, "group_h")
    phi_e = np.asarray(inputs["phi_e"], dtype=np.float64)
    residual_weights = np.asarray(snapshot["own__residual_weights"], dtype=np.float64)
    if (phi_e.ndim != 2 or len(phi_e) != len(groups) or not np.isfinite(phi_e).all()
            or residual_weights.shape != (phi_e.shape[1], n_actions)
            or not np.isfinite(residual_weights).all()):
        raise ValueError("The original input-only design and frozen residual head must align")
    rewards = np.asarray(rewards_e, dtype=np.float64)
    rewards_mean = np.asarray(rewards_mean_e, dtype=np.float64)
    if rewards.shape != (len(groups), n_actions) or not np.isin(rewards, (0, 1)).all():
        raise ValueError("rewards_e must be matching-generation binary final scores")
    if (rewards_mean.shape != rewards.shape or not np.isfinite(rewards_mean).all()
            or np.any((rewards_mean < 0) | (rewards_mean > 1))):
        raise ValueError("rewards_mean_e must contain the separate three-repeat empirical means")
    query = _indices(snapshot["observed_local_queries"], len(train_groups), "observed_local_queries")
    action = _indices(snapshot["observed_actions"], n_actions, "observed_actions")
    observed = np.asarray(snapshot["observed_rewards"], dtype=np.float64)
    if len(action) != len(query) or observed.shape != query.shape or not np.isin(observed, (0, 1)).all():
        raise ValueError("Purchased sparse observations must be aligned binary scalar values")
    if len(np.unique(query*n_actions+action)) != len(query):
        raise ValueError("A single-generation snapshot must deduplicate each question/action feedback")
    rebuilt_counts = np.zeros_like(counts)
    rebuilt_sums = np.zeros_like(reward_sum)
    np.add.at(rebuilt_counts, (train_groups[query], action), 1)
    np.add.at(rebuilt_sums, (train_groups[query], action), observed)
    if not np.array_equal(rebuilt_counts, counts) or not np.array_equal(rebuilt_sums, reward_sum):
        raise ValueError("Support statistics differ from the frozen purchased tuples")
    keys = np.asarray(inputs["group_keys"], dtype=str)
    if keys.shape != (n_groups, 2):
        raise ValueError("Each frozen group needs a task and subtask key")
    global_e = _indices(inputs["E"], 2**63-1, "E")
    if len(global_e) != len(groups) or len(np.unique(global_e)) != len(global_e):
        raise ValueError("Heldout original question indices must align and remain distinct")
    return (capability, counts, reward_sum, groups, train_groups, phi_e,
            residual_weights, rewards, rewards_mean, query, action, keys, global_e)


def analyse_capabilities(inputs, snapshot, rewards_e, rewards_mean_e):
    """Describe support, heldout estimation errors and frozen oracle-group routes.

    ``inputs`` and ``snapshot`` are the original E13 dictionaries. The reward
    arguments are heldout evaluator-only arrays, with one matching generation
    and the separate three-generation mean. The capability table is read as-is.

    Return ``cell_rows`` with one row per original training group/action,
    including empty heldout groups whose target/error fields remain ``None``.
    Each ``query_rows`` entry is one original question under one of two frozen
    diagnostic policies. Repeated callers for seeds/loggers must retain the
    original question as their statistical unit.

    Individual binary reward MSE includes generation noise. Squared error of
    the heldout group mean is a different finite-sample diagnostic. Neither is
    an estimate of latent conditional-mean error without further assumptions.
    Hindsight best-action references and their support are descriptive only.
    Support/error associations do not identify a causal effect of annotation.
    """
    (capability, counts, reward_sum, groups, train_groups, phi_e,
     residual_weights, rewards, rewards_mean, query, action, keys, global_e) = _read_inputs(
        inputs, snapshot, rewards_e, rewards_mean_e)
    n_groups, n_actions = capability.shape
    n_e = len(groups)
    group_scores = capability[groups]
    residual_scores = group_scores + phi_e @ residual_weights
    scores = np.stack((group_scores, residual_scores))
    chosen = np.argmax(scores, axis=-1)
    rows = np.arange(n_e)
    selected_reward = rewards[rows[None, :], chosen]
    selected_mean_reward = rewards_mean[rows[None, :], chosen]
    selected_support = counts[groups[None, :], chosen]
    selected_support_bin = np.vectorize(support_bin, otypes=["U4"])(selected_support)
    seed_loss = rewards.max(axis=1)[None, :] - selected_reward
    mean_loss = rewards_mean.max(axis=1)[None, :] - selected_mean_reward
    corrections = (selected_reward == 1) & (selected_reward[0][None, :] == 0)
    destructions = (selected_reward == 0) & (selected_reward[0][None, :] == 1)
    net = selected_reward-selected_reward[0][None, :]
    evaluation_counts = np.bincount(groups, minlength=n_groups)
    train_distinct_questions = np.zeros(n_groups, dtype=np.int64)
    for group in range(n_groups):
        train_distinct_questions[group] = len(np.unique(query[train_groups[query] == group]))

    cell_rows, query_rows = [], []
    for group in range(n_groups):
        mask = groups == group
        q_rows = np.flatnonzero(mask)
        n_eval = int(len(q_rows))
        seed_target = rewards[mask].mean(axis=0) if n_eval else None
        mean_target = rewards_mean[mask].mean(axis=0) if n_eval else None
        group_action = int(np.argmax(capability[group]))
        best_seed = int(np.argmax(seed_target)) if n_eval else None
        best_mean = int(np.argmax(mean_target)) if n_eval else None
        for column in range(n_actions):
            support = int(counts[group, column])
            cell = dict(
                group_index=group, task=str(keys[group, 0]), subtask=str(keys[group, 1]), action=column,
                support=support, support_bin=support_bin(support),
                observed_reward_sum=float(reward_sum[group, column]),
                observed_reward_mean=(float(reward_sum[group, column]/support) if support else None),
                training_group_distinct_feedback_questions=int(train_distinct_questions[group]),
                training_group_scalar_feedback=int(counts[group].sum()),
                evaluation_original_questions=n_eval,
                capability_estimate=float(capability[group, column]),
                group_selected=column == group_action,
                heldout_seed_mean=(float(seed_target[column]) if n_eval else None),
                heldout_three_mean=(float(mean_target[column]) if n_eval else None),
                seed_empirical_group_best=(bool(column == best_seed) if n_eval else None),
                three_mean_empirical_group_best=(bool(column == best_mean) if n_eval else None),
            )
            for prefix, prediction in (("group", group_scores), ("group_residual", residual_scores)):
                prediction_mean = float(prediction[mask, column].mean()) if n_eval else None
                seed_error = prediction_mean-float(seed_target[column]) if n_eval else None
                mean_error = prediction_mean-float(mean_target[column]) if n_eval else None
                cell.update({
                    prefix+"_heldout_prediction_mean": prediction_mean,
                    prefix+"_seed_signed_mean_error": seed_error,
                    prefix+"_seed_abs_mean_error": abs(seed_error) if n_eval else None,
                    prefix+"_seed_squared_mean_error": seed_error**2 if n_eval else None,
                    prefix+"_individual_binary_reward_mse": (
                        float(np.mean((prediction[mask, column]-rewards[mask, column])**2)) if n_eval else None),
                    prefix+"_three_mean_signed_error": mean_error,
                    prefix+"_three_mean_abs_error": abs(mean_error) if n_eval else None,
                    prefix+"_three_mean_squared_error": mean_error**2 if n_eval else None,
                    prefix+"_individual_three_mean_mse": (
                        float(np.mean((prediction[mask, column]-rewards_mean[mask, column])**2)) if n_eval else None),
                })
            cell_rows.append(cell)
        if not n_eval:
            continue
        seed_tie_count = int(np.sum(seed_target == seed_target.max()))
        mean_tie_count = int(np.sum(mean_target == mean_target.max()))
        for q in q_rows:
            for method_index, method in enumerate(METHODS):
                selected = int(chosen[method_index, q])
                query_rows.append(dict(
                    evaluation_local_index=int(q), global_question_index=int(global_e[q]),
                    group_index=group, task=str(keys[group, 0]), subtask=str(keys[group, 1]),
                    method=method, input_permission="true heldout group, oracle mechanism diagnostic",
                    selected_action=selected, selected_support=int(selected_support[method_index, q]),
                    selected_support_bin=str(selected_support_bin[method_index, q]),
                    seed_reward=float(selected_reward[method_index, q]),
                    three_mean_reward=float(selected_mean_reward[method_index, q]),
                    seed_empirical_selection_loss=float(seed_loss[method_index, q]),
                    three_mean_empirical_selection_loss=float(mean_loss[method_index, q]),
                    correction_vs_group=bool(corrections[method_index, q]),
                    destruction_vs_group=bool(destructions[method_index, q]),
                    net_seed_reward_vs_group=float(net[method_index, q]),
                    changed_action_vs_group=bool(selected != chosen[0, q]),
                    group_seed_empirical_best_action=best_seed,
                    group_seed_empirical_best_support=int(counts[group, best_seed]),
                    group_seed_empirical_best_tie_count=seed_tie_count,
                    group_three_mean_empirical_best_action=best_mean,
                    group_three_mean_empirical_best_support=int(counts[group, best_mean]),
                    group_three_mean_empirical_best_tie_count=mean_tie_count,
                    group_seed_mean_selection_loss=float(seed_target.max()-seed_target[selected]),
                    group_three_mean_selection_loss=float(mean_target.max()-mean_target[selected]),
                    group_evaluation_original_questions=n_eval,
                ))
    return dict(
        cell_rows=cell_rows, query_rows=query_rows,
        array_diagnostics=dict(
            methods=np.array(METHODS), scores=scores, actions=chosen,
            selected_reward=selected_reward, selected_three_mean_reward=selected_mean_reward,
            selected_support=selected_support, selected_support_bin=selected_support_bin,
            seed_empirical_selection_loss=seed_loss, three_mean_empirical_selection_loss=mean_loss,
            corrections_vs_group=corrections, destructions_vs_group=destructions, net_seed_reward_vs_group=net,
            true_group=groups.copy(), group_evaluation_counts=evaluation_counts,
            training_group_distinct_feedback_questions=train_distinct_questions,
            support_matrix=counts.copy(), frozen_capability_table=capability.copy(),
        ),
    )
