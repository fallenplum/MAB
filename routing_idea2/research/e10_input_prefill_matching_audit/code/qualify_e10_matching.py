"""Deterministic, synthetic-only E10 mathematical qualification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from e10_matching import decompose_policy


TOLERANCE = 5e-13


def report(rewards, actions, task, subtask, fold):
    return decompose_policy(rewards, actions, task, subtask, fold,
                            generation_seeds=[42, 137, 2024])


def metric_vector(value, keys):
    views = [value["overall"], *value["by_task"].values(),
             *value["by_outerfold"].values(), *value["by_bbh_subtask"].values()]
    return np.array([
        metric[key]
        for view in views
        for metric in [view["metrics"], *view["metrics"]["by_generation"].values()]
        for key in keys
    ])


def main(output):
    checks = []

    def add(name, residual, details):
        residual = float(residual)
        checks.append({"name": name, "passed": residual <= TOLERANCE,
                       "maximum_absolute_residual": residual, "details": details})
        if residual > TOLERANCE:
            raise AssertionError(f"{name} residual {residual} exceeds {TOLERANCE}")

    small = np.array([
        [[0, 1, 2], [2, 3, 4], [1, 0, 3]],
        [[3, 1, 0], [1, 2, 1], [2, 4, 0]],
        [[2, 4, 1], [3, 0, 2], [0, 2, 1]],
        [[1, 2, 5], [2, 1, 3], [4, 3, 1]],
    ], dtype=float) / 5
    actions = np.array([0, 0, 1, 2])
    task = np.repeat("bbh", 4)
    subtask = np.repeat("synthetic", 4)
    fold = np.zeros(4, dtype=int)
    base = report(small, actions, task, subtask, fold)
    permutation_scores = []
    for ordering in itertools.permutations(range(4)):
        assigned = actions[list(ordering)]
        score = sum(
            Fraction(int(round(small[i, generation, assigned[i]] * 5)), 5)
            for i in range(4) for generation in range(3)
        ) / 12
        permutation_scores.append(score)
    enumerated = sum(permutation_scores) / len(permutation_scores)
    add("enumerated_all_count_preserving_permutations_equal_analytic_mix",
        abs(float(enumerated) - base["overall"]["metrics"]["V_mix_fine"]),
        {"row_permutations": 24, "distinct_action_assignments": 12,
         "exact_rational_permutation_mean": str(enumerated),
         "analytic_value": base["overall"]["metrics"]["V_mix_fine"]})

    keys = ["G_match_coarse", "G_match_fine"]
    fixed = report(small, np.zeros(4, dtype=int), task, subtask, fold)
    singleton = report(small, actions, task, subtask, np.arange(4))
    add("single_action_groups_and_singleton_groups_have_zero_matching_gain",
        max(abs(metric_vector(fixed, keys)).max(), abs(metric_vector(singleton, keys)).max()),
        {"single_action_questions": 4, "singleton_groups": 4,
         "checked_views": "overall, task, generation, outerfold, BBH subtask"})

    # Unequal tasks and nested coarse/fine groups exercise task-balanced views.
    data = np.concatenate([small, small[::-1] + 0.2])
    chosen = np.array([0, 0, 1, 2, 2, 1, 0, 1])
    tasks = np.array(["bbh"] * 5 + ["gsm8k"] * 3)
    subtasks = np.array(["alpha", "alpha", "beta", "beta", "beta", "x", "y", "z"])
    folds = np.array([0, 0, 0, 0, 1, 0, 0, 1])
    full = report(data, chosen, tasks, subtasks, folds)
    all_metrics = ["V_real", "V_mix_coarse", "V_mix_fine", *keys,
                   "V_mix_fine_minus_V_mix_coarse"]
    permutation = np.array([2, 0, 1])
    renamed = report(data[:, :, permutation], np.argsort(permutation)[chosen], tasks, subtasks, folds)
    add("action_coordinate_relabeling_preserves_every_metric",
        abs(metric_vector(full, all_metrics) - metric_vector(renamed, all_metrics)).max(),
        {"new_column_old_indices": permutation.tolist(), "actions_remapped_by_inverse": True})

    common = np.array([-0.7, 0.4, 2.2, -1.9, 1.1, 0.0, 3.2, -4.1])
    shifted = report(data + common[:, None, None], chosen, tasks, subtasks, folds)
    add("arbitrary_question_common_offsets_preserve_matching_gain",
        abs(metric_vector(full, keys) - metric_vector(shifted, keys)).max(),
        {"question_offsets": common.tolist()})

    offset_residuals = []
    for fine in (False, True):
        normalized_subtask = np.where(tasks == "gsm8k", "gsm8k", subtasks)
        labels = list(zip(folds, tasks, normalized_subtask)) if fine else list(zip(folds, tasks))
        unique = sorted(set(labels))
        offsets = np.array([[0.7 * (j + 1), -0.3 * (j + 2), 1.1 * (j + 3)]
                            for j in range(len(unique))])
        shifted_rewards = data.copy()
        for i, label in enumerate(labels):
            shifted_rewards[i] += offsets[unique.index(label)]
        shifted = report(shifted_rewards, chosen, tasks, subtasks, folds)
        metric = "G_match_fine" if fine else "G_match_coarse"
        # BBH subtask restricted views recompute action marginals. A coarse
        # offset remains constant within their restricted coarse groups.
        offset_residuals.append(abs(metric_vector(full, [metric]) -
                                    metric_vector(shifted, [metric])).max())
    add("arbitrary_group_action_offsets_preserve_matching_gain",
        max(offset_residuals), {"separate_checks": ["coarse offsets/coarse gain", "fine offsets/fine gain"]})

    task_masses = full["overall"]["task_mass"]
    add("unequal_task_counts_still_assign_each_task_half_mass",
        max(abs(task_masses["bbh"] - 0.5), abs(task_masses["gsm8k"] - 0.5)),
        {"task_counts": {"bbh": 5, "gsm8k": 3}, "task_mass": task_masses})
    identity_error = max(
        value["metrics"]["maximum_absolute_identity_residual"]
        for value in [full["overall"], *full["by_task"].values(),
                      *full["by_outerfold"].values(), *full["by_bbh_subtask"].values()]
    )
    add("coarse_to_fine_decomposition_identity_all_views", identity_error,
        {"identity": "G_coarse = (V_mix_fine - V_mix_coarse) + G_fine"})
    covariance_error = max(
        view[grouping]["maximum_absolute_covariance_identity_residual"]
        for view in [full["overall"], *full["by_task"].values(), *full["by_outerfold"].values()]
        for grouping in ("coarse", "fine")
    )
    add("sum_of_within_group_covariances_equals_matching_gain", covariance_error,
        {"covariance_denominator": "n_g"})
    qualification = {
        "status": "PASS_SYNTHETIC_MATHEMATICAL_QUALIFICATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "small deterministic matrices only",
        "real_reward_values_decoded": False,
        "real_train600_decomposition_executed": False,
        "tolerance": TOLERANCE,
        "checks": checks,
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__), Path(__file__).with_name("e10_matching.py")]},
    }
    output = Path(output)
    output.write_text(json.dumps(qualification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": qualification["status"], "checks": len(checks),
                      "maximum_absolute_residual": max(item["maximum_absolute_residual"] for item in checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
