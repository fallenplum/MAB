"""Repeated-generation variance moments for final combination rewards.

No data files are read, no predictor is fitted, and no uncertainty interval is
constructed. The estimated signal is an untruncated method-of-moments quantity.
It is not an identified per-question latent reward variance with three seeds.
"""
from __future__ import annotations

import numpy as np


def _validate_rewards(rewards):
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Rewards must have axes generation, original question, action")
    repeats, questions, actions = values.shape
    if repeats < 2 or questions < 2 or actions < 2:
        raise ValueError("Variance decomposition needs at least two repeats, questions and actions")
    if not np.isfinite(values).all():
        raise ValueError("Rewards must contain finite observed values, without missing-action imputation")
    return values


def _labels(values, n_questions, name):
    if values is None:
        return None
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != n_questions:
        raise ValueError(f"{name} must provide exactly one label per original question")
    return [str(value) for value in labels]


def _scope_summary(question_means, question_noise_variance, repeats, rows):
    means = question_means[rows]
    noise = question_noise_variance[rows]
    within = noise.mean(axis=0)
    correction = within / repeats
    observed = means.var(axis=0, ddof=1) if len(rows) >= 2 else None
    signal = observed - correction if observed is not None else None
    return {
        "n_original_questions": int(len(rows)),
        "question_degrees_of_freedom": max(int(len(rows)) - 1, 0),
        "mean_reward_or_difference": means.mean(axis=0).tolist(),
        "within_noise_variance_w": within.tolist(),
        "observed_between_question_variance": observed.tolist() if observed is not None else None,
        "noise_correction_w_over_L": correction.tolist(),
        "estimated_signal_variance": signal.tolist() if signal is not None else None,
        "negative_signal_estimate_count": int(np.sum(signal < 0)) if signal is not None else None,
    }


def _grouped_summary(question_means, question_noise_variance, repeats, rows, labels):
    groups = {}
    for row, label in zip(rows, labels):
        groups.setdefault(tuple(label), []).append(int(row))
    feature_count = question_means.shape[1]
    total_sse = np.zeros(feature_count, dtype=np.float64)
    weighted_noise = np.zeros(feature_count, dtype=np.float64)
    total_df = 0
    positive_df_questions = 0
    summaries = []
    for label in sorted(groups):
        index = np.asarray(groups[label], dtype=int)
        summary = _scope_summary(question_means, question_noise_variance, repeats, index)
        df = len(index) - 1
        if df > 0:
            centered_means = question_means[index] - question_means[index].mean(axis=0)
            total_sse += np.sum(centered_means ** 2, axis=0)
            weighted_noise += df * question_noise_variance[index].mean(axis=0)
            total_df += df
            positive_df_questions += len(index)
        summaries.append({"group_key": list(label), **summary})
    if total_df:
        observed = total_sse / total_df
        within = weighted_noise / total_df
        correction = within / repeats
        signal = observed - correction
    else:
        observed = within = correction = signal = None
    return {
        "groups": summaries,
        "pooled_within_groups": {
            "n_original_questions": int(len(rows)),
            "n_groups": len(groups),
            "n_singleton_groups": sum(len(index) == 1 for index in groups.values()),
            "questions_in_positive_df_groups": int(positive_df_questions),
            "pooled_degrees_of_freedom": int(total_df),
            "within_group_sum_of_squares": total_sse.tolist(),
            "df_weighted_within_noise_variance_w": within.tolist() if within is not None else None,
            "observed_between_question_variance": observed.tolist() if observed is not None else None,
            "noise_correction_w_over_L": correction.tolist() if correction is not None else None,
            "estimated_signal_variance": signal.tolist() if signal is not None else None,
            "negative_signal_estimate_count": int(np.sum(signal < 0)) if signal is not None else None,
            "noise_weighting": "sum_g (n_g-1)*mean_q_in_g within_variance / sum_g(n_g-1)",
            "singleton_rule": "df0 groups contribute neither SSE nor noise-correction weight",
        },
    }


def _representation(values, indices, repeats, tasks, subtasks):
    question_means = values.mean(axis=0)
    question_noise = values.var(axis=0, ddof=1)
    rows = np.arange(values.shape[1])
    result = {
        "indices": indices,
        "global": _scope_summary(question_means, question_noise, repeats, rows),
        "groupings": {},
    }
    if tasks is not None:
        result["groupings"]["task"] = _grouped_summary(
            question_means, question_noise, repeats, rows, [(task,) for task in tasks])
    if subtasks is not None:
        if tasks is None:
            result["groupings"]["subtask"] = _grouped_summary(
                question_means, question_noise, repeats, rows, [(subtask,) for subtask in subtasks])
        else:
            result["groupings"]["task_subtask"] = _grouped_summary(
                question_means, question_noise, repeats, rows,
                list(zip(tasks, subtasks)))
            bbh_rows = np.array([index for index, task in enumerate(tasks) if task.lower() == "bbh"], dtype=int)
            if len(bbh_rows):
                result["groupings"]["bbh_subtask"] = _grouped_summary(
                    question_means, question_noise, repeats, bbh_rows,
                    [(tasks[index], subtasks[index]) for index in bbh_rows])
    return result


def decompose_rewards(Y, task=None, subtask=None):
    """Decompose raw actions, row-centered actions and all unordered pair gaps.

    Parameters
    ----------
    Y : array, shape (L, n, A)
        Main experiment input is (3, 600, 20). Smaller finite arrays are accepted
        for software controls. The original question remains the sample unit.
    task, subtask : optional arrays, shape (n,)
        Metadata defines descriptive variance strata and never a predictor.

    Global moments use w=mean_q var_s(Y_sqa, ddof1), observed=var_q(mean_s Y_sqa,
    ddof1), estimated_signal=observed-w/L. For a pair, the same operations are
    applied directly to D=Y_i-Y_j, retaining same-generation covariance.

    Within-group observed variance is sum_g SSE_g / sum_g(n_g-1). Its noise
    correction is sum_g[(n_g-1)*w_g] / [L*sum_g(n_g-1)], not a question-count
    weighted grand w/L when group sizes differ. If all group df are zero, the
    pooled variance and signal are undefined and returned as null/None.

    The signal interpretation requires independent questions and conditionally
    independent (or sufficiently conditionally uncorrelated) generation repeats
    with a common conditional mean and comparable noise law. Exchangeability
    alone does not establish this. Correlated action errors within a generation
    are fully retained, with no action-independence assumption.
    """
    values = _validate_rewards(Y)
    repeats, n_questions, n_actions = values.shape
    tasks = _labels(task, n_questions, "task")
    subtasks = _labels(subtask, n_questions, "subtask")
    pair_i, pair_j = np.triu_indices(n_actions, 1)
    centered = values - values.mean(axis=2, keepdims=True)
    pairs = values[:, :, pair_i] - values[:, :, pair_j]
    return {
        "status": "COMPUTED_OBSERVED_MOMENTS",
        "input_shape": [repeats, n_questions, n_actions],
        "generation_repetitions": repeats,
        "independent_question_count": n_questions,
        "action_count": n_actions,
        "unordered_pair_count": len(pair_i),
        "generation_sample_variance_ddof": 1,
        "question_sample_variance_ddof": 1,
        "estimated_signal_is_truncated": False,
        "confidence_intervals": 0,
        "bootstrap_runs": 0,
        "new_model_fits": 0,
        "assumptions": [
            "Original questions are the independent units for population-moment interpretation.",
            "Repeats have a common conditional mean and conditionally independent or uncorrelated errors.",
            "Distinct seed names or exchangeability alone do not prove conditional independence.",
            "Same-repeat covariance across actions is retained by direct centering and pair differences.",
            "Only three repeats do not identify every latent per-question reward distribution.",
        ],
        "interpretation": "Untruncated finite-sample moment estimates; negative values are retained and are not negative population variances.",
        "centering": "subtract the20-action mean within each original-question/generation row; generic A uses its own action count",
        "correlated_repeat_caveat": "Under equal off-diagonal conditional repeat covariance kappa, observed-w/L can retain kappa in addition to signal.",
        "representations": {
            "raw_actions": _representation(values, list(range(n_actions)), repeats, tasks, subtasks),
            "centered_actions": _representation(centered, list(range(n_actions)), repeats, tasks, subtasks),
            "pairs": _representation(pairs, list(map(list, zip(pair_i.tolist(), pair_j.tolist()))), repeats, tasks, subtasks),
        },
    }
