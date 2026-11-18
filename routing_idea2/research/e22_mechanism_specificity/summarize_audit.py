"""Compact tables and matched-policy contrasts from the frozen audit outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from run_audit import CONDITIONS, digest, read, write

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "e19_symmetric_tie_constraint"))
import run_e19 as old


def contrast(first, second, rewards, tasks, subtasks):
    corrected, damaged, baseline, candidate, error = old.expected_coupled_changes(first, second, rewards)
    differences = candidate-baseline
    return dict(
        baseline_accuracy_pct=100*old.task_macro(baseline, tasks),
        candidate_accuracy_pct=100*old.task_macro(candidate, tasks),
        difference_pp=100*old.task_macro(differences, tasks),
        expected_corrected_events=float(corrected.sum()),
        expected_damaged_events=float(damaged.sum()), expected_net_corrections=float(differences.sum()),
        coupling_net_identity_error=error,
        by_task={task: dict(
            questions=int((tasks == task).sum()),
            difference_pp=float(100*differences[:, tasks == task].mean()))
            for task in ("gsm8k", "bbh")},
        by_seed={str(seed): 100*old.task_macro(differences[index:index+1], tasks)
                 for index, seed in enumerate((42, 137, 2024))},
        by_bbh_subtask={key: dict(questions=int((subtasks == key).sum()),
                                 difference_pp=float(100*differences[:, subtasks == key].mean()))
                        for key in sorted(set(subtasks[tasks == "bbh"]))},
    )


def summarize(args):
    results = read(args.results)
    receipt = read(args.root / "private/FROZEN.json")
    if receipt["source_binding"] != results["source_binding"]:
        raise ValueError("Result summary and frozen fit identities differ")
    records, outcomes, tables, geometry = {}, {}, [], []
    all_indices = []
    for family in ("tspo", "listnet"):
        blocks = []
        for fold in receipt["folds"]:
            artifact = next(item for item in fold["artifacts"] if item["family"] == family)
            path = args.root / "private" / f"fold{fold['fold']}" / artifact["filename"]
            if digest(path) != artifact["sha256"]:
                raise ValueError("Frozen prediction hash differs at derived summary")
            with np.load(path, allow_pickle=False) as packet:
                blocks.append(packet["scores"].copy())
                if family == "tspo":
                    all_indices.append(packet["E"].copy())
            diag = fold["families"][family]["geometry"]
            geometry.append(dict(
                family=family, fold=fold["fold"], lambda_value=fold["lambda_value"],
                J_T_B=diag["B"]["J_T"], J_T_C=diag["C"]["J_T"],
                parameter_norm_B=diag["B"]["raw_parameter_frobenius"],
                parameter_norm_C=diag["C"]["raw_parameter_frobenius"],
                tie_gradient_norm=diag["tie_gradient_frobenius"],
                non_tie_response_norm=diag["non_tie_feedback_gradient_change_frobenius"],
                parameter_delta_norm=diag["parameter_change_frobenius"],
                best_positive_scale=diag["best_positive_rescaling"],
                nonradial_fraction=diag["nonradial_fraction_of_C_norm"],
                margin_median_B=diag["margin_B"]["median"], margin_median_C=diag["margin_C"]["median"],
                strict_switches=diag["strict_switch_questions"],
                tie_winner_contrast_mean=diag["strict_switched_tie_component"]["mean"],
                nontie_winner_contrast_mean=diag["strict_switched_non_tie_response_component"]["mean"],
                actual_winner_contrast_mean=diag["strict_switched_winner_contrast_change"]["mean"],
                tie_response_cosine=diag["tie_non_tie_change_cosine"],
                kkt_inf=diag["kkt_residual_inf"], score_identity_error=diag["equilibrium_score_identity_max_error"],
            ))
        records[family] = np.concatenate(blocks, axis=1)
    indices = np.concatenate(all_indices)
    if not np.array_equal(np.sort(indices), np.arange(600)):
        raise ValueError("Question partition mismatch")
    rewards, _ = old.read_reward_rows(old.DEFAULT_REWARDS, indices)
    with np.load(old.DEFAULT_INPUTS, allow_pickle=False) as packet:
        tasks, subtasks = packet["task_labels"][indices], packet["subtask_labels"][indices]
    for family, scores in records.items():
        probabilities = np.array([old.uniform_policy(condition, 1e-8)["probabilities"] for condition in scores])
        outcomes[family] = dict(
            C_minus_Uniform=contrast(probabilities[2], probabilities[1], rewards, tasks, subtasks),
            C_minus_MatchedMean=contrast(probabilities[3:].mean(axis=0), probabilities[1], rewards, tasks, subtasks),
        )
        for name in CONDITIONS:
            if name == "B":
                tables.append(dict(family=family, condition=name, accuracy_pct=results["outcome_results"][family]["baseline_accuracy_pct"],
                                   gain_over_B_pp=0, corrected=0, damaged=0, strict_switches=0))
            else:
                primary = results["outcome_results"][family]["control_metrics"][name]["primary"]
                tables.append(dict(family=family, condition=name,
                    accuracy_pct=100*primary["C_task_macro_accuracy"], gain_over_B_pp=primary["C_minus_B_pp"],
                    corrected=primary["expected_corrections"], damaged=primary["expected_damages"],
                    strict_switches=primary["strict_changed_questions"]))
    summary = dict(schema="mab_specificity_summary_v1", primary_controls=outcomes,
                   training_questions=600, source_results_sha256=digest(args.results),
                   condition_table=tables, geometry_table=geometry,
                   new_llm_calls=0, new_fits=0, bootstrap_repetitions=0)
    write(args.output / "SUMMARY.json", summary)
    for name, rows in (("controls.csv", tables), ("geometry.csv", geometry)):
        with (args.output / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(status="COMPLETE", comparisons=4, tables=2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("results", "root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    summarize(parser.parse_args())
