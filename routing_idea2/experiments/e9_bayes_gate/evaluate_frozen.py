"""Evaluate only fully frozen E9 policies on original train600 outcomes.

This script performs no fitting. Its reward-free qualification seals every
prescribed main/control prediction file and this evaluator's source before
opening the original outcome archive. Public outputs contain aggregates only.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import time

from common import (
    ASSETS, CODE, E8, INPUTS, OUTCOMES, PERMUTATION_SEEDS, PRIVATE,
    RESULTS, array_hash, choice, legal_metadata, load_npz, process_state,
    read_json, sha256, write_json,
)
import numpy as np


BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 202611160001
MAIN_NAMES = (
    "PSMR", "HSMR", "GlobalFixed", "TaskFixed", "SubtaskFixed", "PriorShrink", "E9Rule"
)
HISTORICAL_NAMES = ("M0_original", "M0_tuned", "M4", "NS_main")
PRIMARY_COMPARISONS = ("E9Rule-SubtaskFixed", "E9Rule-M0_original")
PROBABILITY_NAMES = {"PSMR"}
UNCALIBRATED_NAMES = {"M4", "NS_main"}
PERMISSIONS = {
    "PSMR": "text-only frozen MiniLM features",
    "HSMR": "text-only deployment; training task/subtask label smoothing",
    "GlobalFixed": "no query metadata; history-only global fixed action",
    "TaskFixed": "query task metadata",
    "SubtaskFixed": "query task and true BBH subtask metadata",
    "PriorShrink": "query task and true BBH subtask metadata",
    "E9Rule": "fold-frozen choice of text-only head or metadata-based SubtaskFixed fallback",
    "M0_original": "text-only frozen MiniLM features, original fixed regularization",
    "M0_tuned": "text-only frozen MiniLM features, historical nested-grid regularization",
    "M4": "text-only frozen MiniLM features, historical exact SPO+",
    "NS_main": "text-only frozen MiniLM features, historical NS-SPO+ delta=.05",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def asset_hashes():
    manifest = read_json(RESULTS / "00_asset_manifest.json")
    if manifest.get("status") != "PASS":
        raise ValueError("Original asset qualification must pass before evaluation")
    return {
        str(Path(record["root"]) / record["relative_path"]): record["sha256"]
        for record in manifest["assets"] if record.get("status") == "PRESENT"
    }


def check_historical_hash(path, expected):
    key = str(Path(path))
    if key not in expected or sha256(path) != expected[key]:
        raise ValueError("Historical source is absent from or differs from the asset manifest: " + key)


def validate_prediction_packet(packet, expected_indices, expected_names):
    if set(packet) != {"methods", "scores", "actions", "evaluation_indices"}:
        raise ValueError("Unexpected new prediction packet schema")
    names = packet["methods"].astype(str).tolist()
    if len(names) != len(set(names)) or set(names) != set(expected_names):
        raise ValueError("Prediction method set differs from the fixed matrix")
    if not np.array_equal(packet["evaluation_indices"], expected_indices):
        raise ValueError("Prediction rows differ from original evaluation question indices")
    if packet["scores"].shape != (len(names), 120, 20):
        raise ValueError("Unexpected frozen score shape")
    if packet["actions"].shape != (len(names), 120):
        raise ValueError("Unexpected frozen action shape")
    if not np.array_equal(packet["actions"], choice(packet["scores"])):
        raise ValueError("Frozen actions disagree with their exact semantic-order argmax")
    if not np.issubdtype(packet["actions"].dtype, np.integer):
        raise ValueError("Actions must be integer column indices")
    return names


def load_and_seal_predictions():
    """Read predictions and reward-free metadata, then create an append-only seal."""
    from run_nested_cv import verify_sources

    protocol = verify_sources()
    if protocol["primary_comparisons"] != list(PRIMARY_COMPARISONS):
        raise ValueError("Primary comparisons differ from the frozen protocol")
    frozen_path = PRIVATE / "ALL_PREDICTIONS_FROZEN.json"
    all_frozen = read_json(frozen_path)
    if all_frozen.get("status") != "FROZEN":
        raise ValueError("Every prescribed prediction must be frozen before outcome access")
    if all_frozen.get("protocol_sha256") != sha256(CODE / "protocol_frozen.json"):
        raise ValueError("Global prediction freeze points to a different protocol")
    expected_runs = {(fold, seed) for seed in (None, *PERMUTATION_SEEDS) for fold in range(5)}
    records = all_frozen["paths"]
    observed_runs = {(row["fold"], row["permutation_seed"]) for row in records}
    if observed_runs != expected_runs or len(records) != len(expected_runs):
        raise ValueError("Incomplete or duplicate main/control fold freezes")
    records_by_key = {(row["fold"], row["permutation_seed"]): row for row in records}
    software = read_json(RESULTS / "01_software_qualification.json")
    mathematics = read_json(RESULTS / "01_math_qualification.json")
    if software.get("status") != "PASS" or mathematics.get("status") != "PASS":
        raise ValueError("Software and mathematical qualifications must pass before outcome access")

    methods = list(MAIN_NAMES + HISTORICAL_NAMES)
    scores = np.full((len(methods), 600, 20), np.nan)
    actions = np.full((len(methods), 600), -1, dtype=np.int64)
    control_actions = np.full((len(PERMUTATION_SEEDS), len(MAIN_NAMES), 600), -1, dtype=np.int64)
    folds = np.full(600, -1, dtype=int)
    task = np.empty(600, dtype="U32")
    subtask = np.empty(600, dtype="U128")
    task[:] = ""
    subtask[:] = ""
    expected_assets = asset_hashes()
    sealed_files = {}
    selected_rule = {}
    for fold in range(5):
        evaluation_path = E8 / f"private/qualification/fold_packs/fold{fold}/evaluation_inputs_only120.npz"
        check_historical_hash(evaluation_path, expected_assets)
        indices = load_npz(evaluation_path)["global_indices"]
        if len(np.unique(indices)) != 120 or (folds[indices] >= 0).any():
            raise ValueError("Outer evaluation questions overlap or repeat")
        metadata = legal_metadata(fold, indices)
        folds[indices] = fold
        task[indices], subtask[indices] = metadata["task"], metadata["sub"]
        if not all(int(np.sum(metadata["task"] == key)) == 60 for key in ("gsm8k", "bbh")):
            raise ValueError("Original outer evaluation task balance changed")
        for seed_index, seed in enumerate((None, *PERMUTATION_SEEDS)):
            region = "main" if seed is None else f"controls/seed{seed}"
            root = PRIVATE / region / f"fold{fold}"
            prediction_path = root / "PREDICTIONS.npz"
            receipt_path = root / "PREDICTIONS_FROZEN.json"
            receipt = read_json(receipt_path)
            global_record = records_by_key[(fold, seed)]
            if receipt != global_record or receipt.get("status") != "FROZEN":
                raise ValueError("Fold freeze differs from the global freeze record")
            if sha256(prediction_path) != receipt["prediction_sha256"]:
                raise ValueError("A frozen prediction changed")
            if sha256(root / "FITTING_LOG.json") != receipt["fitting_log_sha256"]:
                raise ValueError("A frozen fitting log changed")
            packet = load_npz(prediction_path)
            names = validate_prediction_packet(packet, indices, MAIN_NAMES)
            sealed_files[str(prediction_path)] = receipt["prediction_sha256"]
            sealed_files[str(receipt_path)] = sha256(receipt_path)
            if seed is None:
                selected_rule[str(fold)] = receipt["selected_rule"]
                for local, name in enumerate(names):
                    target = methods.index(name)
                    scores[target, indices] = packet["scores"][local]
                    actions[target, indices] = packet["actions"][local]
            else:
                for local, name in enumerate(names):
                    control_actions[seed_index - 1, MAIN_NAMES.index(name), indices] = packet["actions"][local]
        for name in HISTORICAL_NAMES:
            if name == "NS_main":
                path = ASSETS / f"research/m4_noise_aware_set_learning/private/runtime/training/private/fold{fold}/NS_main/PREDICTIONS_FROZEN.npz"
                score_key, index_key = "scores", "evaluation_indices"
            else:
                path = E8 / f"private/training/fold{fold}/{name}/eval_predictions.npz"
                score_key, index_key = "raw_q", "eval_indices"
            check_historical_hash(path, expected_assets)
            packet = load_npz(path)
            if not np.array_equal(packet[index_key], indices):
                raise ValueError("Historical policy evaluation identity changed")
            if not np.array_equal(packet["actions"], choice(packet[score_key])):
                raise ValueError("Historical score-to-action mapping changed")
            target = methods.index(name)
            scores[target, indices] = packet[score_key]
            actions[target, indices] = packet["actions"]
            sealed_files[str(path)] = sha256(path)
    if not np.isfinite(scores).all() or (actions < 0).any() or (control_actions < 0).any() or (folds < 0).any():
        raise ValueError("Frozen policies do not cover all original questions")
    if not np.array_equal(actions[methods.index("M0_original")], actions[methods.index("M0_tuned")]):
        raise ValueError("Historical M0 original/tuned anchor action equality no longer holds")
    seal = dict(
        status="FROZEN_BEFORE_OUTCOME_ACCESS", created_utc=utc_now(),
        evaluator_sha256=sha256(Path(__file__)),
        global_freeze_sha256=sha256(frozen_path),
        protocol_sha256=sha256(CODE / "protocol_frozen.json"),
        software_receipt_sha256=sha256(RESULTS / "01_software_qualification.json"),
        math_receipt_sha256=sha256(RESULTS / "01_math_qualification.json"),
        new_prediction_runs=len(records), method_order=methods, selected_rule=selected_rule,
        prediction_files=sealed_files,
        all_main_scores_sha256=array_hash(scores), all_main_actions_sha256=array_hash(actions),
        all_control_actions_sha256=array_hash(control_actions),
        bootstrap=dict(repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED,
                       strata="outerfold x task", cluster="original question retaining three generations",
                       ordinary_ci=0.95, primary_family_individual_ci=0.975,
                       primary_p="two-sided centered paired bootstrap, plus-one correction", adjustment="Holm"),
        score_diagnostics="raw MSE is uncalibrated for M4/NS; only PSMR probabilities receive probability Brier/calibration",
        new_llm_calls=0, retained_reward_payload_reads_so_far=0,
    )
    seal_path = PRIVATE / "EVALUATOR_SOURCE_FROZEN.json"
    if seal_path.exists():
        previous = read_json(seal_path)
        for key in ("evaluator_sha256", "global_freeze_sha256", "protocol_sha256",
                    "all_main_scores_sha256", "all_main_actions_sha256", "all_control_actions_sha256",
                    "prediction_files", "bootstrap"):
            if previous[key] != seal[key]:
                raise ValueError("Existing evaluator seal differs in " + key)
        seal = previous
    else:
        write_json(seal_path, seal)
    return methods, scores, actions, control_actions, task, subtask, folds, seal, expected_assets


def bootstrap_question_means(values, task, folds):
    """Resample questions within all ten outerfold-by-task strata."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for fold in range(5):
        for key in ("gsm8k", "bbh"):
            rows = np.flatnonzero((folds == fold) & (task == key))
            samples.append(rows[rng.integers(0, len(rows), size=(BOOTSTRAP_REPEATS, len(rows)))])
    indices = np.concatenate(samples, axis=1)
    result = np.empty((BOOTSTRAP_REPEATS, values.shape[0]), dtype=np.float64)
    for method in range(values.shape[0]):
        result[:, method] = values[method, indices].mean(axis=1)
    return result


def holm_adjust(p_values):
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, previous)
    return adjusted


def grouped_accuracy(selected, rows):
    return dict(questions=int(np.sum(rows)), accuracy_pct=float(100.0 * selected[rows].mean()))


def action_distribution(actions):
    counts = np.bincount(actions, minlength=20)
    probabilities = counts[counts > 0] / len(actions)
    return dict(counts=counts.tolist(), effective_actions=float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
                used_actions=int(np.sum(counts > 0)), max_action=int(np.argmax(counts)),
                max_count=int(counts.max()), max_fraction=float(counts.max() / len(actions)))


def calibration_bins(probabilities, rewards):
    means = rewards.mean(axis=1)
    bins = []
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = lower + 0.1
        rows = (probabilities >= lower) & (probabilities < upper if upper < 0.999999 else probabilities <= 1.0)
        count = int(rows.sum())
        bins.append(dict(lower=float(lower), upper=float(upper), question_action_pairs=count,
                         generation_events=3 * count,
                         predicted_probability=float(probabilities[rows].mean()) if count else None,
                         observed_repeat_mean=float(means[rows].mean()) if count else None))
    return bins


def score_diagnostics(name, scores, rewards, competition_mask):
    mean_rewards = rewards.mean(axis=1)
    error = scores - mean_rewards
    centered_error = error - error.mean(axis=1, keepdims=True)
    repeat_error = scores[:, None, :] - rewards
    first, second = np.triu_indices(20, 1)
    true_gap = mean_rewards[:, first] - mean_rewards[:, second]
    predicted_gap = scores[:, first] - scores[:, second]
    nonzero = true_gap != 0.0
    competitive_nonzero = nonzero & competition_mask
    sorted_scores = np.sort(scores, axis=1)
    top_gap = sorted_scores[:, -1] - sorted_scores[:, -2]
    strict_correct = np.sign(predicted_gap) == np.sign(true_gap)
    numerical_ties = predicted_gap == 0.0
    pair_credits = strict_correct.astype(float) + 0.5 * numerical_ties
    calibrated_probability = name in PROBABILITY_NAMES
    result = dict(
        interpretation=("independent Bernoulli probabilities" if calibrated_probability else
                        "uncalibrated decision scores" if name in UNCALIBRATED_NAMES else
                        "unbounded reward-regression or historical prior scores; no clipping"),
        raw_mean_mse=float(np.mean(error ** 2)), centered_mean_mse=float(np.mean(centered_error ** 2)),
        raw_three_repeat_squared_error=float(np.mean(repeat_error ** 2)),
        raw_mse_cross_method_caveat="M4/NS raw-score scales are uncalibrated and are not fairly ranked by probability MSE",
        strict_nonzero_pair_count=int(nonzero.sum()),
        strict_nonzero_pair_rank_accuracy=float(pair_credits[nonzero].mean()) if nonzero.any() else None,
        predicted_pair_tie_credit=0.5,
        competition_nonzero_pair_count=int(competitive_nonzero.sum()),
        competition_definition="pairs touching original frozen M0 top-three score actions; nonzero true three-repeat mean gap",
        competition_subset_role="posthoc empirical reward diagnostic, never a deployable query filter",
        nonzero_competition_gap_mae=(float(np.abs(predicted_gap - true_gap)[competitive_nonzero].mean())
                                     if competitive_nonzero.any() and name not in UNCALIBRATED_NAMES else None),
        uncalibrated_raw_competition_gap_mae=(float(np.abs(predicted_gap - true_gap)[competitive_nonzero].mean())
                                             if competitive_nonzero.any() else None),
        score_min=float(scores.min()), score_max=float(scores.max()),
        maximum_top_score_gap=float(top_gap.max()), mean_top_score_gap=float(top_gap.mean()),
        exact_top_tie_questions=int(np.sum(top_gap == 0.0)),
        near_top_tie_questions_1e_10=int(np.sum(top_gap <= 1e-10)),
        near_top_tie_questions_1e_8=int(np.sum(top_gap <= 1e-8)),
        probability_brier=(float(np.mean(repeat_error ** 2)) if calibrated_probability else None),
        probability_calibration_bins=(calibration_bins(scores, rewards) if calibrated_probability else None),
    )
    return result


def paired_comparison(candidate, reference, candidate_actions, reference_actions, bootstrap_difference, task):
    difference = candidate.mean(axis=1) - reference.mean(axis=1)
    estimate = float(difference.mean())
    centered = bootstrap_difference - estimate
    p_value = float((1 + np.sum(np.abs(centered) >= abs(estimate) - 1e-15)) / (len(centered) + 1))
    return dict(
        difference_pp=100.0 * estimate,
        ci95_pp=(100.0 * np.quantile(bootstrap_difference, [0.025, 0.975])).tolist(),
        corrections=int(np.sum((candidate == 1) & (reference == 0))),
        destructions=int(np.sum((candidate == 0) & (reference == 1))),
        net_events=int(np.sum(candidate - reference)),
        independent_questions=len(candidate), recorded_generation_events=int(candidate.size),
        improved_questions=int(np.sum(difference > 0)), harmed_questions=int(np.sum(difference < 0)),
        unchanged_mean_reward_questions=int(np.sum(difference == 0)),
        changed_action_questions=int(np.sum(candidate_actions != reference_actions)),
        generation_difference_pp={str(seed): float(100 * (candidate[:, i] - reference[:, i]).mean())
                                  for i, seed in enumerate((42, 137, 2024))},
        task_difference_pp={key: float(100 * difference[task == key].mean()) for key in ("gsm8k", "bbh")},
        p_unadjusted=p_value,
    )


def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    public_outputs = (
        "03_main_results.csv", "03_main_results.json", "03_question_cluster_ci.json",
        "04_ablation_and_negative_controls.csv", "06_gate_decision.json",
    )
    private_outputs = ("PAIRED_OUTCOMES.npz", "EVALUATION_RECEIPT.json")
    if any((RESULTS / name).exists() for name in public_outputs) or any((PRIVATE / name).exists() for name in private_outputs):
        raise FileExistsError("Final evaluation outputs already exist; no overwrite is permitted")
    methods, scores, actions, control_actions, task, subtask, folds, seal, expected_assets = load_and_seal_predictions()

    # First access to the full evaluation outcomes in this script occurs only
    # after the evaluator source and all 505 new prediction files were sealed.
    outcome_read_started_utc = utc_now()
    check_historical_hash(OUTCOMES, expected_assets)
    check_historical_hash(INPUTS, expected_assets)
    outcomes = load_npz(OUTCOMES)
    inputs = load_npz(INPUTS)
    rewards = outcomes["aggregate_scores"].transpose(1, 0, 2).astype(np.float64, copy=False)
    if rewards.shape != (600, 3, 20) or not np.isin(rewards, (0, 1)).all():
        raise ValueError("Original final aggregate reward matrix is missing or malformed")
    if outcomes["generation_seeds"].tolist() != [42, 137, 2024]:
        raise ValueError("Original generation mapping changed")
    for key in ("question_ids", "canonical_hashes"):
        if not np.array_equal(inputs[key], outcomes[key]) or len(np.unique(inputs[key])) != 600:
            raise ValueError("Original input/outcome identities disagree")
    if not np.array_equal(task, inputs["task_labels"]) or not np.array_equal(subtask, inputs["subtask_labels"]):
        raise ValueError("Reward-free fold metadata differs from original train600 metadata")
    selected = np.stack([rewards[np.arange(600)[:, None], np.arange(3)[None, :], a[:, None]] for a in actions])
    values = selected.mean(axis=2)
    bootstrap = bootstrap_question_means(values, task, folds)
    first, second = np.triu_indices(20, 1)
    top_three = np.argsort(-scores[methods.index("M0_original")], axis=1, kind="stable")[:, :3]
    top_mask = np.zeros((600, 20), dtype=bool)
    np.put_along_axis(top_mask, top_three, True, axis=1)
    competition_mask = top_mask[:, first] | top_mask[:, second]
    empirical_best = rewards.mean(axis=1).max(axis=1)
    results = {}
    all_pairs = {}
    for index, name in enumerate(methods):
        by_task = {key: grouped_accuracy(selected[index], task == key) for key in ("gsm8k", "bbh")}
        by_subtask = {key: grouped_accuracy(selected[index], (task == "bbh") & (subtask == key))
                      for key in sorted(set(subtask[task == "bbh"]))}
        record = dict(
            accuracy=float(values[index].mean()), accuracy_pct=float(100 * values[index].mean()),
            accuracy_ci95_pct=(100 * np.quantile(bootstrap[:, index], [0.025, 0.975])).tolist(),
            gsm8k_pct=by_task["gsm8k"]["accuracy_pct"], bbh_pct=by_task["bbh"]["accuracy_pct"],
            bbh_subtask_macro_pct=float(np.mean([row["accuracy_pct"] for row in by_subtask.values()])),
            independent_questions=600, recorded_generation_events=1800,
            input_permission=PERMISSIONS[name], by_task=by_task,
            by_generation_seed={str(seed): dict(accuracy_pct=float(100 * selected[index, :, j].mean()))
                                for j, seed in enumerate((42, 137, 2024))},
            by_outerfold={str(fold): grouped_accuracy(selected[index], folds == fold) for fold in range(5)},
            by_bbh_subtask=by_subtask, action_distribution=action_distribution(actions[index]),
            score_diagnostics=score_diagnostics(name, scores[index], rewards, competition_mask),
            empirical_regret_mean=float(np.mean(empirical_best - values[index])),
            regret_interpretation="posthoc loss relative to best of three-repeat empirical means, not true Bayes regret",
            versus={},
        )
        for reference in ("SubtaskFixed", "M0_original", "GlobalFixed"):
            baseline = methods.index(reference)
            paired = paired_comparison(selected[index], selected[baseline], actions[index], actions[baseline],
                                       bootstrap[:, index] - bootstrap[:, baseline], task)
            record["versus"][reference] = paired
            all_pairs[name + "-" + reference] = paired
        results[name] = record
    primary = {}
    p_values = []
    for label in PRIMARY_COMPARISONS:
        candidate, reference = label.split("-", 1)
        candidate_index, reference_index = methods.index(candidate), methods.index(reference)
        record = dict(all_pairs[label])
        record["family_ci975_pp"] = (100 * np.quantile(
            bootstrap[:, candidate_index] - bootstrap[:, reference_index], [0.0125, 0.9875]
        )).tolist()
        record["positive_seed_count"] = sum(value > 0 for value in record["generation_difference_pp"].values())
        record["task_directions_nonnegative"] = all(value >= -1e-12 for value in record["task_difference_pp"].values())
        p_values.append(record["p_unadjusted"])
        primary[label] = record
    for label, adjusted in zip(PRIMARY_COMPARISONS, holm_adjust(p_values)):
        primary[label]["p_holm"] = float(adjusted)

    control_values = np.empty((len(PERMUTATION_SEEDS), len(MAIN_NAMES)), dtype=float)
    control_rows = []
    for seed_index, seed in enumerate(PERMUTATION_SEEDS):
        for method_index, name in enumerate(MAIN_NAMES):
            a = control_actions[seed_index, method_index]
            reward = rewards[np.arange(600)[:, None], np.arange(3)[None, :], a[:, None]]
            accuracy = float(reward.mean())
            control_values[seed_index, method_index] = accuracy
            distribution = action_distribution(a)
            control_rows.append(dict(
                permutation_seed=seed, method=name, input_permission=PERMISSIONS[name],
                independent_questions=600, recorded_generation_events=1800,
                accuracy_pct=100 * accuracy,
                gsm8k_pct=float(100 * reward[task == "gsm8k"].mean()),
                bbh_pct=float(100 * reward[task == "bbh"].mean()),
                seed42_pct=float(100 * reward[:, 0].mean()),
                seed137_pct=float(100 * reward[:, 1].mean()),
                seed2024_pct=float(100 * reward[:, 2].mean()),
                main_accuracy_pct=results[name]["accuracy_pct"],
                main_minus_control_pp=results[name]["accuracy_pct"] - 100 * accuracy,
                max_action=distribution["max_action"], max_action_count=distribution["max_count"],
                effective_actions=distribution["effective_actions"],
            ))
    negative_controls = dict(
        repeats=len(PERMUTATION_SEEDS), seeds=list(PERMUTATION_SEEDS),
        operation="within fitting task/BBH-subtask whole-question reward-block permutation; nested training rerun",
        interpretation="100 diagnostic training permutations on the same 600 questions, not 100 independent test datasets",
        monte_carlo_minimum_p=1 / (len(PERMUTATION_SEEDS) + 1), per_method={},
    )
    for index, name in enumerate(MAIN_NAMES):
        control_accuracy = control_values[:, index]
        main_accuracy = results[name]["accuracy"]
        negative_controls["per_method"][name] = dict(
            main_accuracy_pct=100 * main_accuracy,
            control_mean_pct=float(100 * control_accuracy.mean()),
            control_min_pct=float(100 * control_accuracy.min()),
            control_max_pct=float(100 * control_accuracy.max()),
            control_quantile95_pct=float(100 * np.quantile(control_accuracy, 0.95)),
            monte_carlo_p=float((1 + np.sum(control_accuracy >= main_accuracy - 1e-12)) / (len(control_accuracy) + 1)),
            main_minus_control_mean_pp=float(100 * (main_accuracy - control_accuracy.mean())),
        )

    software = read_json(RESULTS / "01_software_qualification.json")
    conditions = dict(
        software_pass=software.get("status") == "PASS",
        primary_positive_family_ci=all(row["family_ci975_pp"][0] > 0 for row in primary.values()),
        primary_holm_pass=all(row["p_holm"] < 0.05 for row in primary.values()),
        seed_consistency=all(row["positive_seed_count"] >= 2 for row in primary.values()),
        task_guard=all(row["task_directions_nonnegative"] for row in primary.values()),
        content_guard=results["E9Rule"]["accuracy_pct"] > negative_controls["per_method"]["E9Rule"]["control_quantile95_pct"] + 1e-12,
    )
    passed = all(conditions.values())
    gate = dict(
        decision="GO_FRESH_REQUEST" if passed else "NO_GO_INPUT_ROUTER",
        conditions=conditions, failed_conditions=[key for key, value in conditions.items() if not value],
        primary_comparisons=primary,
        content_guard=dict(main_accuracy_pct=results["E9Rule"]["accuracy_pct"],
                           control_quantile95_pct=negative_controls["per_method"]["E9Rule"]["control_quantile95_pct"]),
        next_priority=("prepare a separately authorized frozen independent-confirmation design" if passed else
                       "task/subtask-conditioned combination feedback efficiency under a separately frozen protocol"),
        interpretation=("Development evidence permits requesting an independent confirmation plan" if passed else
                        "The prescribed development gate did not establish generalizable conditional routing improvement"),
        no_fresh_or_sealed_access=True, no_automatic_llm_or_feedback_experiment=True,
    )
    evaluation_cost = dict(cpu_s=time.process_time() - start_cpu, wall_s=time.perf_counter() - start_wall,
                           scope="evaluator through aggregate statistics, before final artifact writes", **process_state())
    complete = dict(
        status="COMPLETE", evidence="reused_train600_development_cross_validation",
        method_order=methods, methods=results, comparisons=primary,
        negative_controls=negative_controls, evaluation_cost=evaluation_cost,
        selected_rule_by_fold=seal["selected_rule"],
        limitations=[
            "All 600 development questions have been reused in prior research.",
            "Bootstrap conditions on the frozen fitted models; overlapping fold training histories are not resampled.",
            "Three generations and 20 actions within a question are dependent repeated observations.",
            "Metadata defaults and E9Rule fallback have explicitly broader query metadata permission than text-only heads.",
            "M4/NS raw scores are not calibrated probabilities and cannot be fairly ranked by probability Brier.",
            "Empirical regret and nonzero-gap subsets use the three recorded generations, not true latent conditional means.",
        ],
    )
    ci_output = dict(
        status="COMPLETE", bootstrap_repeats=BOOTSTRAP_REPEATS, bootstrap_seed=BOOTSTRAP_SEED,
        unit="original question with all methods/actions/generations kept together", strata="outerfold x GSM/BBH",
        ordinary_intervals="95% paired percentile intervals",
        primary_family_intervals="individual 97.5% two-sided Bonferroni intervals for two comparisons, family coverage at least95% under bootstrap approximation",
        p_value="two-sided centered paired bootstrap with plus-one correction; Holm correction for the two primary comparisons",
        bootstrap_is_approximate=True, fitted_model_conditional=True,
        primary_comparisons=primary, all_paired_comparisons=all_pairs,
        accuracy_ci95_pct={name: results[name]["accuracy_ci95_pct"] for name in methods},
    )
    rows = []
    for name in methods:
        record = results[name]
        base = record["versus"]["SubtaskFixed"]
        anchor = record["versus"]["M0_original"]
        rows.append(dict(
            method=name, input_permission=PERMISSIONS[name], independent_questions=600, recorded_generation_events=1800,
            accuracy_pct=record["accuracy_pct"], ci95_low_pct=record["accuracy_ci95_pct"][0],
            ci95_high_pct=record["accuracy_ci95_pct"][1], gsm8k_pct=record["gsm8k_pct"], bbh_pct=record["bbh_pct"],
            bbh_subtask_macro_pct=record["bbh_subtask_macro_pct"],
            seed42_pct=record["by_generation_seed"]["42"]["accuracy_pct"],
            seed137_pct=record["by_generation_seed"]["137"]["accuracy_pct"],
            seed2024_pct=record["by_generation_seed"]["2024"]["accuracy_pct"],
            versus_subtask_pp=base["difference_pp"], versus_m0_pp=anchor["difference_pp"],
            corrections_vs_subtask=base["corrections"], destructions_vs_subtask=base["destructions"],
            net_events_vs_subtask=base["net_events"], empirical_three_repeat_selection_loss=record["empirical_regret_mean"],
            effective_actions=record["action_distribution"]["effective_actions"],
            most_frequent_action=record["action_distribution"]["max_action"],
            most_frequent_action_questions=record["action_distribution"]["max_count"],
            probability_brier=record["score_diagnostics"]["probability_brier"],
            raw_uncalibrated_mean_mse=record["score_diagnostics"]["raw_mean_mse"],
            nonzero_competition_gap_mae=record["score_diagnostics"]["nonzero_competition_gap_mae"],
        ))
    write_csv(RESULTS / "03_main_results.csv", rows, list(rows[0]))
    write_json(RESULTS / "03_main_results.json", complete)
    write_json(RESULTS / "03_question_cluster_ci.json", ci_output)
    write_csv(RESULTS / "04_ablation_and_negative_controls.csv", control_rows, list(control_rows[0]))
    write_json(RESULTS / "06_gate_decision.json", gate)
    np.savez_compressed(
        PRIVATE / "PAIRED_OUTCOMES.npz", methods=np.asarray(methods), selected_rewards=selected,
        question_mean_rewards=values, actions=actions, task=task, subtask=subtask, outerfold=folds,
        generation_seeds=outcomes["generation_seeds"], global_indices=np.arange(600),
        question_ids=inputs["question_ids"], canonical_hashes=inputs["canonical_hashes"],
        control_seeds=np.asarray(PERMUTATION_SEEDS), control_methods=np.asarray(MAIN_NAMES),
        control_actions=control_actions, control_accuracies=control_values,
    )
    public_hashes = {name: sha256(RESULTS / name) for name in public_outputs}
    receipt = dict(
        status="PASS", created_utc=utc_now(), evaluator_sha256=sha256(Path(__file__)),
        source_freeze_sha256=sha256(PRIVATE / "EVALUATOR_SOURCE_FROZEN.json"),
        source_freeze_created_utc=seal["created_utc"], outcome_read_started_utc=outcome_read_started_utc,
        original_outcome_sha256=sha256(OUTCOMES), original_input_sha256=sha256(INPUTS),
        public_outputs=public_hashes, paired_outcomes_sha256=sha256(PRIVATE / "PAIRED_OUTCOMES.npz"),
        new_fits=0, new_llm_calls=0, new_gold=0, validation_fresh_sealed_payload_reads=0,
        independent_questions=600, recorded_generation_events=1800,
        new_prediction_runs_verified=len(PERMUTATION_SEEDS) * 5 + 5,
        evaluation_cost=evaluation_cost, decision=gate["decision"],
    )
    write_json(PRIVATE / "EVALUATION_RECEIPT.json", receipt)
    print("EVALUATION_COMPLETE", gate["decision"], flush=True)


if __name__ == "__main__":
    main()
