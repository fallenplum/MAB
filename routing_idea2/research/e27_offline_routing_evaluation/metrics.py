"""Paired point metrics for frozen policies on the complete OOF score cache."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

E20 = Path(__file__).resolve().parents[1] / "e20_p2_execution"
sys.path.insert(0, str(E20))
from blind_evaluation import expected_corrections


def summarize(probabilities, rewards, task_labels, subtask_labels, generation_seeds, fold_ids, pairs):
    """Keep all generation observations for each question and report task-weighted points."""
    tasks = np.asarray(task_labels, dtype=str)
    subtasks = np.asarray(subtask_labels, dtype=str)
    folds = np.asarray(fold_ids, dtype=int)
    seeds = np.asarray(generation_seeds, dtype=int)
    rewards = np.asarray(rewards)
    size = len(tasks)
    if tasks.shape != (size,) or subtasks.shape != tasks.shape or folds.shape != tasks.shape:
        raise ValueError("Task, subtask and fold arrays must share one question axis")
    if set(tasks) != {"gsm8k", "bbh"} or np.any(subtasks[tasks == "bbh"] == ""):
        raise ValueError("Both tasks and all BBH subtask labels are required")
    if seeds.shape != (3,) or len(set(seeds)) != 3 or rewards.shape != (3, size, 20):
        raise ValueError("Exactly three generation seeds and 20 actions are required")
    if not np.isin(rewards, (0, 1)).all():
        raise ValueError("Every historical score must be present and binary")

    question_weights = np.where(tasks == "gsm8k", 0.5 / (tasks == "gsm8k").sum(),
                                0.5 / (tasks == "bbh").sum())
    all_questions = np.ones(size, dtype=bool)
    all_seeds = slice(None)
    task_masks = {name: tasks == name for name in ("gsm8k", "bbh")}
    subtask_masks = {name: task_masks["bbh"] & (subtasks == name)
                     for name in sorted(set(subtasks[task_masks["bbh"]]))}
    fold_masks = {str(fold): folds == fold for fold in sorted(set(folds))}
    seed_slices = {str(seed): slice(i, i + 1) for i, seed in enumerate(seeds)}

    def partition(statistic):
        return {
            "overall": statistic(all_questions, all_seeds),
            "by_task": {name: statistic(mask, all_seeds) for name, mask in task_masks.items()},
            "by_subtask": {name: statistic(mask, all_seeds) for name, mask in subtask_masks.items()},
            "by_generation_seed": {name: statistic(all_questions, selection)
                                   for name, selection in seed_slices.items()},
            "by_fold": {name: statistic(mask, all_seeds) for name, mask in fold_masks.items()},
        }

    utilities = {}
    for name, policy in probabilities.items():
        if (policy.shape != (size, 20) or not np.isfinite(policy).all() or np.any(policy < 0)
                or not np.allclose(policy.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
            raise ValueError(f"Invalid probability matrix for {name}")
        utilities[name] = np.einsum("qa,gqa->gq", policy, rewards)

    methods = {}
    for name, values in utilities.items():
        def accuracy(mask, selection):
            selected = values[selection][:, mask]
            return {
                "questions": int(mask.sum()),
                "generation_events": int(selected.size),
                "accuracy_pct": 100 * float(np.average(selected.mean(axis=0), weights=question_weights[mask])),
            }

        methods[name] = partition(accuracy)

    comparison_results = {}
    for candidate, baseline in pairs:
        candidate_policy = probabilities[candidate]
        baseline_policy = probabilities[baseline]
        corrected, damaged = expected_corrections(np.stack([baseline_policy, candidate_policy]), rewards)
        delta = utilities[candidate] - utilities[baseline]
        if not np.allclose(corrected - damaged, delta, atol=1e-12, rtol=0):
            raise AssertionError(f"Pairwise correction accounting differs for {candidate} and {baseline}")
        policy_tv = np.abs(candidate_policy - baseline_policy).sum(axis=1) / 2
        both_deterministic = ((candidate_policy > 0).sum(axis=1) == 1) & ((baseline_policy > 0).sum(axis=1) == 1)
        changed = policy_tv > 1e-12

        def paired(mask, selection):
            up = corrected[selection][:, mask]
            down = damaged[selection][:, mask]
            weights = question_weights[mask]
            return {
                "questions": int(mask.sum()),
                "generation_events": int(up.size),
                "delta_pp": 100 * float(np.average(delta[selection][:, mask].mean(axis=0), weights=weights)),
                "corrections": float(up.sum()),
                "damages": float(down.sum()),
                "net_corrected_events": float((up - down).sum()),
                "weighted_corrections_pp": 100 * float(np.average(up.mean(axis=0), weights=weights)),
                "weighted_damages_pp": 100 * float(np.average(down.mean(axis=0), weights=weights)),
                "probability_changed_questions": int(changed[mask].sum()),
                "different_one_hot_questions": int((mask & both_deterministic & changed).sum()),
                "weighted_policy_tv_pct": 100 * float(np.average(policy_tv[mask], weights=weights)),
            }

        comparison_results[f"{candidate}_vs_{baseline}"] = {
            "candidate": candidate,
            "baseline": baseline,
            **partition(paired),
        }

    return {
        "weighting": "GSM8K and BBH each receive half the overall weight, with equal question weights within each task",
        "pair_accounting": "Expected counts over three generation events per question, coupling maximal common action mass with independent residual actions",
        "methods": methods,
        "paired": comparison_results,
    }
