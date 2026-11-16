"""E18.1 question-cluster inference and frozen-action mechanism analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


POLICIES = ("Ridge", "SPO_coordinate", "TSPO_uniform", "GlobalFixed")
COMPARISONS = ((2, 1), (2, 0), (2, 3))
REWARDS_SHA = "5c8c8702ba1dd5b74474ec69475806a7325213e6b3ac7664e16f7a9de397e70a"
RESULTS_SHA = "86a91632c460085db016bf7718b56cd58fc40808f39a487d898c0dce080db41f"


def read_verified(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Completed E18 analysis input differs from the specified artifact")
    return raw


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_csv(path, rows):
    fields = list(rows[0])
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_means(values, repetitions, generator, batch_size):
    """Every row is one entire original question, with shared column indices."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    count, columns = values.shape
    if count < 2 or not np.isfinite(values).all():
        raise ValueError("Question bootstrap requires at least two finite question rows")
    result = np.empty((repetitions, columns), dtype=np.float64)
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        indices = generator.integers(0, count, size=(stop - start, count))
        result[start:stop] = values[indices].mean(axis=1)
    return result


def interval(replicates):
    lower, upper = np.quantile(replicates, (0.025, 0.975), method="linear")
    return dict(ci95_low_pp=float(100 * lower), ci95_high_pp=float(100 * upper))


def one_sample_test(differences):
    variance = float(np.var(differences, ddof=1))
    if variance == 0:
        return dict(t_p=None, t_statistic=None, t_df=len(differences) - 1,
                    empirical_degenerate=True)
    result = stats.ttest_1samp(differences, 0)
    return dict(t_p=float(result.pvalue), t_statistic=float(result.statistic),
                t_df=len(differences) - 1, empirical_degenerate=False)


def holm_adjust(p_values):
    values = np.asarray([1.0 if value is None else value for value in p_values])
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - position) * values[index]))
        adjusted[index] = running
    return [None if p is None else float(adjusted[i]) for i, p in enumerate(p_values)]


def paired_counts(rewards, candidate, baseline, mask):
    left = rewards[candidate][:, mask]
    right = rewards[baseline][:, mask]
    differences = (left - right).mean(axis=0)
    corrected = int(((left == 1) & (right == 0)).sum())
    damaged = int(((left == 0) & (right == 1)).sum())
    return dict(questions=int(mask.sum()), generation_events=int(left.size),
                corrected_events=corrected, damaged_events=damaged,
                net_corrected_events=corrected - damaged,
                improved_questions=int((differences > 0).sum()),
                harmed_questions=int((differences < 0).sum()),
                equal_questions=int((differences == 0).sum()))


def distribution(actions):
    counts = np.bincount(actions, minlength=20)
    probabilities = counts / len(actions)
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    hhi = float(np.square(probabilities).sum())
    return dict(used_combinations=int((counts > 0).sum()),
                dominant_combination=int(np.argmax(counts)),
                dominant_share=float(probabilities.max()),
                combination0_share=float(probabilities[0]),
                combination5_share=float(probabilities[5]),
                entropy_nats=entropy, normalized_entropy=entropy / math.log(20),
                entropy_effective_combinations=math.exp(entropy),
                hhi=hhi, inverse_hhi=1 / hhi, counts=counts.tolist())


def software_checks():
    synthetic = np.arange(2 * 3 * 7).reshape(2, 3, 7).astype(float)
    indices = np.asarray([1, 1, 4, 6, 0, 3, 3])
    explicit_blocks = np.take(synthetic, indices, axis=2).mean(axis=(1, 2))
    question_means = synthetic.mean(axis=1)[:, indices].mean(axis=1)
    if not np.allclose(explicit_blocks, question_means):
        raise AssertionError("Cluster resampling and question means differ")
    equal = np.column_stack((np.arange(7), np.arange(7)))
    boot = bootstrap_means(equal, 31, np.random.default_rng(91), 5)
    if not np.array_equal(boot[:, 0], boot[:, 1]):
        raise AssertionError("Paired columns were resampled separately")
    if 0.5 * 1.0 + 0.5 * 0.0 == (2 * 1.0 + 5 * 0.0) / 7:
        raise AssertionError("Task weighting fixture failed")
    constant = bootstrap_means(np.zeros(7), 31, np.random.default_rng(92), 5)
    if interval(constant[:, 0]) != dict(ci95_low_pp=0.0, ci95_high_pp=0.0):
        raise AssertionError("Degenerate empirical distribution was changed")
    return dict(status="PASS", checks=4, synthetic_only=True,
                purpose="cluster axes, paired columns, unequal task sizes and degeneracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewards", type=Path, required=True)
    parser.add_argument("--e18-results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    qualification = software_checks()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    read_verified(args.rewards, REWARDS_SHA)
    original = json.loads(read_verified(args.e18_results, RESULTS_SHA))
    with np.load(args.rewards, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    rewards = arrays["rewards"].astype(float)
    actions = arrays["frozen_policy_actions"]
    labels = arrays["task_labels"]
    subtasks = arrays["subtask_labels"]
    if rewards.shape != (4, 3, 2998) or tuple(arrays["policy_ids"]) != POLICIES:
        raise ValueError("Frozen reward axes or policies differ")
    if actions.shape != (4, 2998) or not np.isin(rewards, (0, 1)).all():
        raise ValueError("Actions or binary correctness records are invalid")
    if len(set(arrays["question_ids"])) != 2998 or int(arrays["valid_question_mask"].sum()) != 2998:
        raise ValueError("Original question identities or missingness mask differ")
    if np.any(actions[3] != 5):
        raise ValueError("Frozen GlobalFixed is not combination5")
    question_rewards = rewards.mean(axis=1)
    tasks = ("gsm8k", "bbh")
    masks = {task: labels == task for task in tasks}
    if tuple(int(masks[t].sum()) for t in tasks) != (1500, 1498):
        raise ValueError("Frozen task support differs")
    for left in range(4):
        for right in range(left):
            same = actions[left] == actions[right]
            if not np.array_equal(rewards[left][:, same], rewards[right][:, same]):
                raise ValueError("Shared frozen actions have inconsistent reused rewards")
    for index, policy in enumerate(POLICIES):
        observed = sum(question_rewards[index, masks[t]].mean() for t in tasks) / 2
        if abs(observed - original["results"]["quality"][policy]["task_macro_accuracy"]) > 1e-12:
            raise ValueError("E18 point estimate was not recovered")

    output, private = args.output_root, args.private_root
    output.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    repetitions = protocol["bootstrap"]["resamples"]
    batch = protocol["bootstrap"]["batch_size"]
    subtask_names = sorted(set(subtasks[masks["bbh"]]))
    streams = np.random.SeedSequence(protocol["bootstrap"]["seed"]).spawn(2 + len(subtask_names))
    task_bootstraps = {}
    for index, task in enumerate(tasks):
        # A row contains all four policies and all three cached generation outcomes.
        feature_rows = rewards[:, :, masks[task]].reshape(12, -1).T
        boot = bootstrap_means(feature_rows, repetitions, np.random.default_rng(streams[index]), batch)
        task_bootstraps[task] = boot.reshape(repetitions, 4, 3)
    task_policy_boot = {task: value.mean(axis=2) for task, value in task_bootstraps.items()}
    macro_boot = sum(task_policy_boot.values()) / 2
    all_questions = np.ones(2998, dtype=bool)
    overall_rows = []
    for candidate, baseline in COMPARISONS:
        name = POLICIES[candidate] + " minus " + POLICIES[baseline]
        existing = (original["results"]["primary"] if baseline == 1
                    else original["results"]["secondary"][name])
        overall_rows.append(dict(comparison=name, gain_pp=existing["estimate_pp"],
            **interval(macro_boot[:, candidate] - macro_boot[:, baseline]),
            original_t_p=existing["p_value"],
            original_secondary_holm_p=existing.get("holm_adjusted_p"),
            **paired_counts(rewards, candidate, baseline, all_questions)))

    task_rows, seed_rows = [], []
    for task in tasks:
        mask = masks[task]
        differences = (question_rewards[2] - question_rewards[1])[mask]
        task_rows.append(dict(task=task, gain_pp=float(100 * differences.mean()),
            **interval(task_policy_boot[task][:, 2] - task_policy_boot[task][:, 1]),
            **one_sample_test(differences), **paired_counts(rewards, 2, 1, mask)))
        for generation, seed in enumerate((42, 137, 2024)):
            delta = rewards[2, generation] - rewards[1, generation]
            seed_rows.append(dict(scope=task, generation_seed=seed,
                questions=int(mask.sum()), gain_pp=float(100 * delta[mask].mean()),
                corrected_events=int((delta[mask] > 0).sum()),
                damaged_events=int((delta[mask] < 0).sum())))
    for generation, seed in enumerate((42, 137, 2024)):
        delta = rewards[2, generation] - rewards[1, generation]
        seed_rows.append(dict(scope="task_macro_50_50", generation_seed=seed,
            questions=2998, gain_pp=float(50 * sum(delta[masks[t]].mean() for t in tasks)),
            corrected_events=int((delta > 0).sum()), damaged_events=int((delta < 0).sum())))

    delta_gsm = (question_rewards[2] - question_rewards[1])[masks["gsm8k"]]
    delta_bbh = (question_rewards[2] - question_rewards[1])[masks["bbh"]]
    welch = stats.ttest_ind(delta_bbh, delta_gsm, equal_var=False)
    interaction_boot = ((task_policy_boot["bbh"][:, 2] - task_policy_boot["bbh"][:, 1])
                        - (task_policy_boot["gsm8k"][:, 2] - task_policy_boot["gsm8k"][:, 1]))
    interaction = dict(effect="BBH TSPO-SPO gain minus GSM8K TSPO-SPO gain",
        estimate_pp=float(100 * (delta_bbh.mean() - delta_gsm.mean())),
        **interval(interaction_boot), two_sided_welch_p=float(welch.pvalue),
        t_statistic=float(welch.statistic), degrees_of_freedom=float(welch.df),
        gsm8k_questions=1500, bbh_questions=1498,
        role="Post-E18 exploratory task-by-policy heterogeneity; task assignment is observational")

    subtask_rows, subtask_replicates = [], []
    for index, subtask in enumerate(subtask_names):
        mask = masks["bbh"] & (subtasks == subtask)
        differences = (question_rewards[2] - question_rewards[1])[mask]
        boot = bootstrap_means(differences, repetitions,
                               np.random.default_rng(streams[index + 2]), batch)[:, 0]
        subtask_replicates.append(boot)
        subtask_rows.append(dict(subtask=subtask, gain_pp=float(100 * differences.mean()),
            **interval(boot), **one_sample_test(differences),
            **paired_counts(rewards, 2, 1, mask)))
    for row, adjusted in zip(subtask_rows, holm_adjust([r["t_p"] for r in subtask_rows])):
        row["diagnostic_holm27_p"] = adjusted

    distribution_rows = []
    for scope, mask in (("all_evaluable", all_questions), *masks.items()):
        for index, policy in enumerate(POLICIES):
            metrics = distribution(actions[index, mask])
            metrics.pop("counts")
            point = (sum(question_rewards[index, masks[t]].mean() for t in tasks) / 2
                     if scope == "all_evaluable" else question_rewards[index, mask].mean())
            distribution_rows.append(dict(scope=scope, policy=policy, questions=int(mask.sum()),
                                          accuracy_pct=float(100 * point), **metrics))
    transitions = []
    for left_action in range(20):
        for right_action in range(20):
            mask = (actions[1] == left_action) & (actions[2] == right_action)
            transitions.append(dict(spo_action=left_action, tspo_action=right_action,
                **paired_counts(rewards, 2, 1, mask)))

    departure_rows = []
    groups = [("all_evaluable", all_questions), *masks.items()]
    groups += [("bbh/" + subtask, masks["bbh"] & (subtasks == subtask))
               for subtask in subtask_names]
    for name, group in groups:
        for relation, mask in (("stay_combination5", group & (actions[2] == 5)),
                               ("leave_combination5", group & (actions[2] != 5))):
            count = int(mask.sum())
            delta = question_rewards[2] - question_rewards[3]
            departure_rows.append(dict(group=name, relation=relation,
                group_questions=int(group.sum()), subset_fraction=count / int(group.sum()),
                tspo_accuracy_pct=float(100 * question_rewards[2, mask].mean()) if count else None,
                global_accuracy_pct=float(100 * question_rewards[3, mask].mean()) if count else None,
                conditional_gain_pp=float(100 * delta[mask].mean()) if count else None,
                **paired_counts(rewards, 2, 3, mask)))
    action_changes = {scope: int(((actions[1] != actions[2]) & mask).sum())
                      for scope, mask in groups[:3]}
    selected_counts = {policy: distribution(actions[index]) for index, policy in enumerate(POLICIES)}
    inference = dict(
        unit="original question cluster with three recorded seeds retained together",
        weights="GSM8K0.5 and BBH0.5; BBH questions weighted equally inside task",
        method="stratified paired percentile bootstrap",
        scope="Approximate question-sampling generalization conditional on frozen models, present source mixture and recorded generations",
        finite_pool="Complete evaluable finite-pool averages are observed values; bootstrap addresses question-sampling uncertainty",
        marginal_intervals=True, simultaneous_interval_guarantee=False,
        question_representativeness_required=True,
        degeneracy="An all-zero empirical subgroup yields [0,0] under resampling; this does not establish zero population effect",
        missing_bounds="E18 deterministic missing-outcome ranges are retained separately and have no confidence level")
    result = dict(schema="e18_1_statistics_and_mechanisms_results_v1", status="COMPLETE",
        base_commit=protocol["base_commit"], requested_questions=3000,
        scorable_questions=2998, excluded_questions=2, generation_events_per_policy=8994,
        bootstrap_resamples=repetitions, bootstrap_seed=protocol["bootstrap"]["seed"],
        inference=inference, overall=overall_rows, task_effects=task_rows,
        interaction=interaction, generation_seed_effects=seed_rows,
        bbh_subtask_effects=subtask_rows, distributions=distribution_rows,
        selection_counts=selected_counts, spo_tspo_changed_actions=action_changes,
        tspo_global_fixed_decomposition=departure_rows,
        original_e18_statistics=original["results"],
        deterministic_missing_gain_bounds=original["results"]["full_requested_gain_bounds"],
        source_sha256s=dict(rewards=REWARDS_SHA, e18_results=RESULTS_SHA,
                           analysis=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           protocol=hashlib.sha256(args.protocol.read_bytes()).hexdigest()),
        software_checks=qualification, actual_new_llm_calls=0, actual_refits=0,
        sealed_data_accesses=0, bootstrap_unit_is_not_generation=True)
    elapsed = dict(wall_seconds=time.perf_counter() - started_wall,
                   cpu_seconds=time.process_time() - started_cpu,
                   python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    result["analysis_execution"] = elapsed
    save_json(output / "E18_1_COMPLETE_RESULTS.json", result)
    save_json(output / "TASK_POLICY_INTERACTION.json", interaction)
    save_json(output / "SOFTWARE_CHECKS.json", qualification)
    save_json(output / "ORIGINAL_E18_STATISTICS_UNCHANGED.json", original["results"])
    for name, rows in (("BOOTSTRAP_COMPARISONS.csv", overall_rows), ("TASK_EFFECTS.csv", task_rows),
                       ("GENERATION_SEED_EFFECTS.csv", seed_rows), ("BBH_SUBTASK_EFFECTS.csv", subtask_rows),
                       ("POLICY_DISTRIBUTIONS.csv", distribution_rows),
                       ("SPO_TSPO_ACTION_TRANSITIONS.csv", transitions),
                       ("TSPO_GLOBAL_FIXED_DECOMPOSITION.csv", departure_rows)):
        save_csv(output / name, rows)
    np.savez_compressed(private / "BOOTSTRAP_REPLICATES.npz",
        gsm8k_policy_seed=task_bootstraps["gsm8k"], bbh_policy_seed=task_bootstraps["bbh"],
        macro_policy=macro_boot, interaction=interaction_boot,
        subtask_names=np.asarray(subtask_names), subtask_differences=np.asarray(subtask_replicates))
    save_json(private / "INPUT_AND_EXECUTION_MANIFEST.json", dict(
        rewards_path=str(args.rewards), e18_results_path=str(args.e18_results),
        protocol_path=str(args.protocol), **elapsed,
        source_sha256s=result["source_sha256s"], bootstrap_child_streams=2 + len(subtask_names),
        original_inputs_opened_read_only=True))
    print(json.dumps(dict(status=result["status"], overall=overall_rows,
                          interaction=interaction, execution=elapsed), ensure_ascii=False))


if __name__ == "__main__":
    main()
