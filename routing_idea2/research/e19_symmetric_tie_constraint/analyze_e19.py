"""Frozen E19 OOF utilities, strict-decision mechanisms and question bootstrap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE.parent / "e18_1_statistics_and_mechanisms"))
from analyze_e18_1 import bootstrap_means, interval


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_rows(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro(values, labels):
    return float(sum(values[labels == task].mean() for task in ("gsm8k", "bbh")) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    public, private = args.study_root / "public", args.study_root / "private"
    protocol = read(args.protocol)
    point = read(public / "P1_POINT_RESULTS.json")
    if point["status"] != "P1_ALL_FIVE_FOLDS_POINTS_COMPLETE":
        raise ValueError("All five frozen and evaluated folds must be complete")
    with np.load(private / "OOF_EVALUATION.npz", allow_pickle=False) as file:
        arrays = {name: file[name] for name in file.files}
    utilities, raw = arrays["utilities"], arrays["raw_exact_utilities"]
    labels, subtasks = arrays["task_labels"], arrays["subtask_labels"]
    if utilities.shape != (2, 3, 600) or not np.array_equal(arrays["E"], np.arange(600)):
        raise ValueError("OOF question and generation axes differ")
    if tuple(int((labels == task).sum()) for task in ("gsm8k", "bbh")) != (300, 300):
        raise ValueError("The original task-balanced600-question frame differs")
    if not np.allclose(arrays["probabilities"].sum(axis=2), 1.0, atol=1e-14):
        raise ValueError("Uniform routing probabilities are not normalized")
    difference = utilities[1] - utilities[0]
    question_difference = difference.mean(axis=0)
    if abs(100 * macro(question_difference, labels) - point["C_minus_B_pp"]) > 1e-10:
        raise ValueError("Completed OOF point estimate was not recovered")
    coupling_residual = arrays["expected_corrections"] - arrays["expected_damages"] - difference
    if np.max(np.abs(coupling_residual)) > 1e-12:
        raise ValueError("Expected correction coupling does not recover service utility")

    bootstrap = protocol["bootstrap"]
    streams = np.random.SeedSequence(bootstrap["seed"]).spawn(2)
    task_boot = {}
    task_rows = []
    for index, task in enumerate(("gsm8k", "bbh")):
        mask = labels == task
        feature_rows = np.concatenate((utilities[:, :, mask].reshape(6, -1).T,
                                       raw[:, :, mask].reshape(6, -1).T), axis=1)
        sampled = bootstrap_means(feature_rows, bootstrap["repetitions"],
                                  np.random.default_rng(streams[index]), 256)
        task_boot[task] = sampled.reshape(bootstrap["repetitions"], 2, 2, 3)
        delta_boot = (task_boot[task][:, 0, 1] - task_boot[task][:, 0, 0]).mean(axis=1)
        task_rows.append(dict(task=task, questions=int(mask.sum()),
            B_accuracy_pct=float(100 * utilities[0][:, mask].mean()),
            C_accuracy_pct=float(100 * utilities[1][:, mask].mean()),
            gain_pp=float(100 * question_difference[mask].mean()), **interval(delta_boot)))
    macro_boot = sum(task_boot.values()) / 2
    primary_boot = (macro_boot[:, 0, 1] - macro_boot[:, 0, 0]).mean(axis=1)
    raw_boot = (macro_boot[:, 1, 1] - macro_boot[:, 1, 0]).mean(axis=1)
    primary = dict(B_accuracy_pct=100 * macro(utilities[0].mean(axis=0), labels),
        C_accuracy_pct=100 * macro(utilities[1].mean(axis=0), labels),
        gain_pp=100 * macro(question_difference, labels), **interval(primary_boot),
        expected_corrections=float(arrays["expected_corrections"].sum()),
        expected_damages=float(arrays["expected_damages"].sum()),
        expected_net_corrections=float(difference.sum()))
    raw_exact = dict(B_accuracy_pct=100 * macro(raw[0].mean(axis=0), labels),
        C_accuracy_pct=100 * macro(raw[1].mean(axis=0), labels),
        gain_pp=100 * macro((raw[1] - raw[0]).mean(axis=0), labels),
        **interval(raw_boot), role="predeclared floating-point exact-argmax uniform sensitivity")

    strict = arrays["strict_changed"]
    strict_both = arrays["strict_both_unique"]
    subgroups = []
    masks = dict(all_questions=np.ones(600, bool), strict_both_unique=strict_both,
                 strict_changed=strict, numerical_tie_in_at_least_one=~strict_both)
    for name, mask in masks.items():
        contribution = 0.0
        for task in ("gsm8k", "bbh"):
            contribution += 50 * question_difference[mask & (labels == task)].sum() / int((labels == task).sum())
        subgroups.append(dict(subset=name, questions=int(mask.sum()),
            conditional_gain_pp=float(100 * question_difference[mask].mean()) if mask.any() else None,
            primary_macro_gain_contribution_pp=float(contribution),
            expected_corrections=float(arrays["expected_corrections"][:, mask].sum()),
            expected_damages=float(arrays["expected_damages"][:, mask].sum()),
            expected_net_corrections=float(difference[:, mask].sum())))
    strict_metrics = dict(questions=int(strict.sum()),
        corrected_generation_events=int(((utilities[1][:, strict] == 1) & (utilities[0][:, strict] == 0)).sum()),
        damaged_generation_events=int(((utilities[1][:, strict] == 0) & (utilities[0][:, strict] == 1)).sum()))

    fold_rows, qualifications, score_rows = [], [], []
    score_b, score_c = np.empty((600, 20)), np.empty((600, 20))
    fit_logs = []
    for fold in range(5):
        folder = private / f"fold{fold}"
        qualification = read(folder / "NUMERICAL_QUALIFICATION.json")
        if qualification["status"] != "PASS":
            raise ValueError("A fold qualification failed")
        qualifications.append(qualification)
        log_b, log_c = (read(folder / f"{name}_FIT_LOG.json")["diagnostics"] for name in ("B", "C"))
        point_fold = read(public / f"FOLD{fold}_POINT_RESULTS.json")
        fold_rows.append(dict(fold=fold, history_questions=480,
            training_all_tied_questions=log_b["all_action_tied_questions"],
            B_effective_weight_sum=log_b["effective_question_weight_sum"],
            C_effective_weight_sum=log_c["effective_question_weight_sum"],
            regularization=log_b["regularization"], F0_B=log_b["F0"], F0_C=log_c["F0"],
            F0_B_minus_F0_C=qualification["F0_B_minus_F0_C"],
            JT_B=log_b["J_T"], JT_C=log_c["J_T"], JT_C_minus_JT_B=qualification["JT_C_minus_JT_B"],
            penalty_B=log_b["regularization_penalty"], penalty_C=log_c["regularization_penalty"],
            nonT_loss_B=log_b["F0"] - log_b["regularization_penalty"],
            nonT_loss_C=log_c["F0"] - log_c["regularization_penalty"],
            B_accuracy_pct=100 * point_fold["B_task_macro_accuracy"],
            C_accuracy_pct=100 * point_fold["C_task_macro_accuracy"],
            gain_pp=point_fold["C_minus_B_pp"],
            strict_changed_questions=point_fold["strict_changed_questions"]))
        with np.load(folder / "B_PREDICTIONS.npz", allow_pickle=False) as b, np.load(folder / "C_PREDICTIONS.npz", allow_pickle=False) as c:
            score_b[b["E"]] = b["scores_e"]
            score_c[c["E"]] = c["scores_e"]
        fit_logs.extend(read(path)["diagnostics"] for path in sorted(folder.glob("*_FIT_LOG.json")))
    if len(fit_logs) != 12:
        raise ValueError("Real-data fits exceed or do not complete the planned10+2")
    centered_b = score_b - score_b.mean(axis=1, keepdims=True)
    centered_c = score_c - score_c.mean(axis=1, keepdims=True)
    centered_delta = centered_c - centered_b
    maximum_change = np.abs(centered_delta).max(axis=1)
    total_variation = 0.5 * np.abs(arrays["probabilities"][1] - arrays["probabilities"][0]).sum(axis=1)
    mechanism = dict(centered_score_changed_questions_threshold_1e_minus7=int((maximum_change > 1e-7).sum()),
        centered_score_max_abs_change=float(maximum_change.max()),
        centered_score_mean_row_l2_change=float(np.linalg.norm(centered_delta, axis=1).mean()),
        common_score_shift_max_abs=float(np.abs((score_c - score_b).mean(axis=1)).max()),
        strict_both_unique_questions=int(strict_both.sum()), strict_changed_questions=int(strict.sum()),
        strict_generation_counts=strict_metrics,
        B_numerical_tied_questions=int((~(arrays["margins"][0] > 1e-8)).sum()),
        C_numerical_tied_questions=int((~(arrays["margins"][1] > 1e-8)).sum()),
        mean_total_variation=float(macro(total_variation, labels)),
        fixed_pool_utility_bound_pp=float(100 * macro(total_variation, labels)),
        bound_scope="deterministic bounded-reward influence bound on this fixed OOF question pool")
    for index in range(600):
        score_rows.append(dict(question_index=index,
            centered_score_max_change=float(maximum_change[index]),
            centered_score_l2_change=float(np.linalg.norm(centered_delta[index])),
            common_shift=float((score_c[index] - score_b[index]).mean()),
            B_margin=float(arrays["margins"][0, index]), C_margin=float(arrays["margins"][1, index]),
            strict_both_unique=bool(strict_both[index]), strict_changed=bool(strict[index]),
            total_variation=float(total_variation[index])))
    subtask_rows = []
    for subtask in sorted(set(subtasks[labels == "bbh"])):
        mask = (labels == "bbh") & (subtasks == subtask)
        subtask_rows.append(dict(subtask=subtask, questions=int(mask.sum()),
            B_accuracy_pct=float(100 * utilities[0][:, mask].mean()),
            C_accuracy_pct=float(100 * utilities[1][:, mask].mean()),
            gain_pp=float(100 * question_difference[mask].mean()),
            strict_changed_questions=int((strict & mask).sum())))
    result = dict(schema="e19_complete_results_v1", status="P0_P1_COMPLETE", primary=primary,
        task_effects=task_rows, raw_exact_uniform_diagnostic=raw_exact, subgroup_decomposition=subgroups,
        mechanism=mechanism, folds=fold_rows, bbh_subtask_results=subtask_rows,
        bbh_subtask_macro=dict(B_accuracy_pct=float(np.mean([r["B_accuracy_pct"] for r in subtask_rows])),
                              C_accuracy_pct=float(np.mean([r["C_accuracy_pct"] for r in subtask_rows]))),
        seed_effects=point["by_generation_seed"], P0_qualification=read(public / "P0_QUALIFICATION.json"),
        real_main_fits=10, real_qualification_fits=2,
        real_fit_cpu_seconds=sum(r["cpu_seconds"] for r in fit_logs),
        real_fit_wall_seconds_sum=sum(r["wall_seconds"] for r in fit_logs),
        max_KKT_stationarity=max(r["stationarity_inf"] for r in fit_logs),
        max_primal_dual_gap=max(r["primal_dual_objective_gap"] for r in fit_logs),
        bootstrap_repetitions=bootstrap["repetitions"], bootstrap_seed=bootstrap["seed"],
        inference_scope=protocol["uncertainty_scope"],
        correction_scope=protocol["correction_accounting"],
        probability_policy=protocol["deployment"],
        original_questions=600, generation_records_per_policy=1800,
        new_llm_calls=0, fresh400_reward_reads=0, sealed200_reads=0, P2_P3_P4_runs=0,
        source_hashes=dict(analysis=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           protocol=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                           oof_evaluation=hashlib.sha256((private / "OOF_EVALUATION.npz").read_bytes()).hexdigest()),
        analysis_wall_seconds=time.perf_counter() - start_wall,
        analysis_cpu_seconds=time.process_time() - start_cpu)
    save(public / "E19_COMPLETE_RESULTS.json", result)
    for name, rows in (("FIVE_FOLD_RESULTS.csv", fold_rows), ("TASK_EFFECTS.csv", task_rows),
                       ("MECHANISM_SUBGROUPS.csv", subgroups), ("BBH_SUBTASK_RESULTS.csv", subtask_rows)):
        csv_rows(public / name, rows)
    csv_rows(private / "PER_QUESTION_SCORE_DIAGNOSTICS.csv", score_rows)
    np.savez_compressed(private / "OOF_SCORES_AND_PROBABILITY_CHANGES.npz",
        E=np.arange(600), scores_B=score_b, scores_C=score_c,
        centered_scores_B=centered_b, centered_scores_C=centered_c,
        full_probability_difference=arrays["probabilities"][1] - arrays["probabilities"][0],
        strict_both_unique=strict_both, strict_changed=strict, margins=arrays["margins"])
    np.savez_compressed(private / "BOOTSTRAP_REPLICATES.npz",
        gsm8k=task_boot["gsm8k"], bbh=task_boot["bbh"],
        primary_macro_gain=primary_boot, raw_exact_macro_gain=raw_boot)
    print(json.dumps(dict(status=result["status"], primary=primary, raw=raw_exact,
                          mechanism=mechanism, task_effects=task_rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
