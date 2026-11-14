"""Held-out-only evaluator for E14; no fitting or confidence intervals."""
from __future__ import annotations

from collections import defaultdict
import time

from e14_common import (
    A_METHODS, ASSETS, BUDGETS, CODE, E13, LOGGERS, METHODS, PRIVATE, PUBLIC,
    SEEDS, check_protocol, load_npz, measured, read_json, sha256, world_path,
    write_csv, write_json,
)
from capability_diagnostic import analyse_capabilities
import numpy as np


def mean_records(records, keys, fields):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for group, rows in sorted(groups.items()):
        result = dict(zip(keys, group))
        result["measurement_rows"] = len(rows)
        for field in fields:
            values = [r[field] for r in rows if r[field] is not None]
            result[field] = float(np.mean(values)) if values else None
        out.append(result)
    return out


def accuracy_summary(pred, oracle, rewards, mask):
    a, b = rewards[np.arange(len(rewards)), pred], rewards[np.arange(len(rewards)), oracle]
    a, b = a[mask], b[mask]
    return dict(questions=int(mask.sum()), predicted_accuracy_pct=float(100*a.mean()),
        oracle_accuracy_pct=float(100*b.mean()), oracle_minus_predicted_pp=float(100*(b-a).mean()),
        action_changed=int(np.sum(pred[mask] != oracle[mask])),
        corrected=int(np.sum((a == 0) & (b == 1))), damaged=int(np.sum((a == 1) & (b == 0))),
        net_corrected=int(np.sum(b-a)))


def evaluate():
    check_protocol()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    freeze_path = PRIVATE / "ALL_DIAGNOSTIC_PREDICTIONS_FROZEN.json"
    freeze = read_json(freeze_path)
    if freeze["status"] != "FROZEN_BEFORE_E14_EVALUATION":
        raise ValueError("Prediction freeze missing")
    for item in freeze["files"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("E14 prediction changed after freeze")
    evaluator_source = {str(p.relative_to(CODE)): sha256(p) for p in CODE.glob("*.py")}
    write_json(PRIVATE / "EVALUATION_SOURCE_FROZEN.json", dict(source_hashes=evaluator_source,
        prediction_receipt_sha256=sha256(freeze_path), no_new_training_after_this_point=True))
    original_protocol = read_json(E13 / "public/E13_PROTOCOL.json")
    outcome_path = ASSETS / "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
    if sha256(outcome_path) != original_protocol["source_assets"][1]["sha256"]:
        raise ValueError("Original reward cache identity changed")
    with np.load(outcome_path, allow_pickle=False) as saved:
        reward = saved["aggregate_scores"].copy()
        if not np.array_equal(saved["generation_seeds"], SEEDS):
            raise ValueError("Generation axis changed")
    input_path = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz"
    if sha256(input_path) != original_protocol["source_assets"][0]["sha256"]:
        raise ValueError("Input identity changed")
    with np.load(input_path, allow_pickle=False) as saved:
        task, subtask = saved["task_labels"].copy(), saved["subtask_labels"].copy()
    reward_mean = reward.mean(axis=0)
    oracle_actions = load_npz(PRIVATE / "A_ORACLE_ACTIONS_FROZEN.npz")["actions"]
    c_actions = load_npz(PRIVATE / "C_MATCHED_ACTIONS_FROZEN.npz")["actions"]
    a_rows, a_slices, b_cells, b_queries = [], [], [], []
    confusion = np.zeros((28, 28), dtype=int)
    grouping_rows = []
    for fold in range(5):
        inputs = load_npz(E13 / f"private/inputs/fold{fold}/INPUTS_ONLY.npz")
        e = inputs["E"]
        pred_g, true_g = inputs["predicted_group_e"], inputs["oracle_group_e"]
        np.add.at(confusion, (true_g, pred_g), 1)
        for group in range(28):
            use = true_g == group
            grouping_rows.append(dict(fold=fold, task=str(inputs["group_keys"][group, 0]),
                subtask=str(inputs["group_keys"][group, 1]), questions=int(use.sum()),
                group_correct=int(np.sum(pred_g[use] == group))))
        for si, scenario in enumerate(("Main", "Pair")):
            for bi, budget in enumerate(BUDGETS):
                for li, logger in enumerate(LOGGERS):
                    for gi, seed in enumerate(SEEDS):
                        snap = load_npz(world_path(fold, logger, seed, scenario) / f"B{budget}/SNAPSHOT.npz")
                        identity = dict(scenario=scenario, budget=budget, fold=fold,
                                        logger_seed=logger, generation_seed=seed)
                        for mi, method in enumerate(A_METHODS):
                            pred = snap["actions"][snap["methods"].tolist().index(method)]
                            oracle = oracle_actions[si, bi, li, gi, mi, e]
                            for label, mask in (("all", np.ones(len(e), dtype=bool)),
                                                ("gsm8k", task[e] == "gsm8k"), ("bbh", task[e] == "bbh")):
                                a_rows.append(dict(**identity, method=method, task=label,
                                    **accuracy_summary(pred, oracle, reward[gi, e], mask)))
                            for label in np.unique(subtask[e]):
                                mask = subtask[e] == label
                                a_slices.append(dict(**identity, method=method, subtask=str(label),
                                    **accuracy_summary(pred, oracle, reward[gi, e], mask)))
                        capability = analyse_capabilities(inputs, snap, reward[gi, e], reward_mean[e])
                        b_cells.extend(dict(**identity, **row) for row in capability["cell_rows"])
                        b_queries.extend(dict(**identity, **row) for row in capability["query_rows"])
        print(f"E14_EVALUATED fold={fold}", flush=True)
    write_csv(PRIVATE / "B_GROUP_ACTION_CELLS.csv", b_cells)
    write_csv(PRIVATE / "B_QUERY_DIAGNOSTICS.csv", b_queries)
    write_csv(PRIVATE / "A_WORLD_RESULTS.csv", a_rows)
    write_csv(PRIVATE / "A_SUBTASK_RESULTS.csv", a_slices)
    write_csv(PUBLIC / "A_GROUP_CLASSIFICATION.csv", grouping_rows)
    a_fields = ("predicted_accuracy_pct", "oracle_accuracy_pct", "oracle_minus_predicted_pp", "action_changed", "corrected", "damaged", "net_corrected")
    a_summary = mean_records(a_rows, ("scenario", "budget", "method", "task"), a_fields)
    write_csv(PUBLIC / "A_SUMMARY.csv", a_summary)
    write_csv(PUBLIC / "A_SEED_RESULTS.csv", mean_records(a_rows,
        ("scenario", "budget", "method", "task", "generation_seed"), a_fields))
    write_csv(PUBLIC / "A_FOLD_RESULTS.csv", mean_records(a_rows,
        ("scenario", "budget", "method", "task", "fold"), a_fields))
    subtask_groups = defaultdict(list)
    for row in a_slices:
        subtask_groups[(row["scenario"], row["budget"], row["method"], row["subtask"])].append(row)
    subtask_summary = []
    for key, rows in sorted(subtask_groups.items()):
        total = sum(row["questions"] for row in rows)
        out = dict(zip(("scenario", "budget", "method", "subtask"), key))
        out["measurement_events"] = total
        for field in ("predicted_accuracy_pct", "oracle_accuracy_pct", "oracle_minus_predicted_pp"):
            out[field] = sum(row[field]*row["questions"] for row in rows)/total
        subtask_summary.append(out)
    write_csv(PUBLIC / "A_SUBTASK_SUMMARY.csv", subtask_summary)
    b_results = summarize_b(b_cells, b_queries)
    matches = read_json(PRIVATE / "TOKEN_MATCHES_FROZEN.json")
    write_csv(PUBLIC / "C_TOKEN_MATCHES.csv", flatten_matches(matches))
    c_rows, c_slices = [], []
    for bi, budget in enumerate(BUDGETS):
        for li, logger in enumerate(LOGGERS):
            for gi, seed in enumerate(SEEDS):
                for mi, method in enumerate(METHODS):
                    main, pair = c_actions[bi, li, gi, :, mi]
                    h_main = reward[gi, np.arange(600), main]
                    h_pair = reward[gi, np.arange(600), pair]
                    for label, use in (("all", np.ones(600, dtype=bool)),
                                       ("gsm8k", task == "gsm8k"), ("bbh", task == "bbh")):
                        c_rows.append(dict(anchor_main_gold=budget, logger_seed=logger,
                            generation_seed=seed, method=method, task=label, independent_questions=int(use.sum()),
                            main_accuracy_pct=float(100*h_main[use].mean()), pair_accuracy_pct=float(100*h_pair[use].mean()),
                            pair_minus_main_pp=float(100*(h_pair[use]-h_main[use]).mean()),
                            corrected=int(np.sum((h_main[use] == 0) & (h_pair[use] == 1))),
                            damaged=int(np.sum((h_main[use] == 1) & (h_pair[use] == 0))),
                            action_changed=int(np.sum(main[use] != pair[use]))))
                    for fold in range(5):
                        e = load_npz(E13 / f"private/inputs/fold{fold}/INPUTS_ONLY.npz")["E"]
                        c_slices.append(dict(anchor_main_gold=budget, logger_seed=logger,
                            generation_seed=seed, method=method, slice_type="fold", slice=str(fold),
                            questions=len(e), pair_minus_main_pp=float(100*(h_pair[e]-h_main[e]).mean())))
                    for label in np.unique(subtask):
                        use = subtask == label
                        c_slices.append(dict(anchor_main_gold=budget, logger_seed=logger,
                            generation_seed=seed, method=method, slice_type="subtask", slice=str(label),
                            questions=int(use.sum()), pair_minus_main_pp=float(100*(h_pair[use]-h_main[use]).mean())))
    c_summary = mean_records(c_rows, ("anchor_main_gold", "method", "task"),
        ("main_accuracy_pct", "pair_accuracy_pct", "pair_minus_main_pp", "corrected", "damaged", "action_changed"))
    write_csv(PRIVATE / "C_WORLD_RESULTS.csv", c_rows)
    write_csv(PRIVATE / "C_SUBTASK_AND_FOLD.csv", c_slices)
    write_csv(PUBLIC / "C_SEED_RESULTS.csv", mean_records(c_rows,
        ("anchor_main_gold", "method", "task", "generation_seed"),
        ("main_accuracy_pct", "pair_accuracy_pct", "pair_minus_main_pp", "corrected", "damaged")))
    write_csv(PUBLIC / "C_SUBTASK_AND_FOLD.csv", mean_records(c_slices,
        ("anchor_main_gold", "method", "slice_type", "slice"), ("pair_minus_main_pp", "questions")))
    write_csv(PUBLIC / "C_SUMMARY.csv", c_summary)
    result = dict(status="COMPLETE", evidence="REUSED_TRAIN600_EXPLORATORY_DEVELOPMENT",
        independent_questions=600, llm_generation_repetitions=3, logger_repetitions=3,
        confidence_intervals_computed=False, bootstrap_runs=0, new_physical_llm_calls=0,
        original_snapshots_reused=540, existing_learner_prefix_fits=freeze["unique_existing_learner_prefix_fits"],
        A=dict(summary=a_summary, confusion_matrix=confusion,
               group_accuracy_pct=float(100*np.trace(confusion)/confusion.sum()), oracle_only=True),
        B=b_results, C=dict(summary=c_summary, matching="same actual-token upper limit; report integer-package residual",
                         cost_summary=cost_summary(matches)), evaluation=measured(start_cpu, start_wall))
    write_json(PUBLIC / "E14_COMPLETE_RESULTS.json", result)
    write_json(PRIVATE / "EVALUATION_RECEIPT.json", dict(status="COMPLETE",
        result_sha256=sha256(PUBLIC / "E14_COMPLETE_RESULTS.json"), dense_reward_sha256=sha256(outcome_path),
        predictions_sha256=sha256(freeze_path), fresh400_reads=0, sealed200_reads=0,
        confidence_intervals_computed=False, bootstrap_runs=0, **measured(start_cpu, start_wall)))
    print("E14_EVALUATION_COMPLETE", flush=True)


def flatten_matches(matches):
    rows = []
    for match in matches:
        row = {}
        def flatten(values, prefix=""):
            for key, value in values.items():
                name = prefix + key
                if isinstance(value, dict):
                    flatten(value, name + "__")
                elif not isinstance(value, list):
                    row[name] = value
        flatten(match)
        rows.append(row)
    return rows


def cost_summary(matches):
    rows = flatten_matches(matches)
    fields = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ("fold", "logger_seed", "generation_seed", "generation_index", "anchor_main_gold")]
    return mean_records(rows, ("anchor_main_gold",), fields)


def summarize_b(cells, queries):
    # The pure-array module defines the exact row names. Aggregate them without
    # adding weights to empty held-out groups or expanding repeated seeds into n.
    valid_cells = [row for row in cells if row["evaluation_original_questions"] > 0]
    numeric = [k for k, v in valid_cells[0].items() if isinstance(v, (int, float, np.integer, np.floating))
               and k not in ("fold", "logger_seed", "generation_seed", "budget", "group_index", "action")]
    keys = ("scenario", "budget", "task", "support_bin")
    summary = mean_records(valid_cells, keys, numeric)
    write_csv(PUBLIC / "B_SUPPORT_CELL_SUMMARY.csv", summary)
    q_numeric = [k for k, v in queries[0].items() if isinstance(v, (int, float, np.integer, np.floating))
                 and k not in ("fold", "logger_seed", "generation_seed", "budget", "evaluation_local_index", "global_question_index", "selected_action", "group_index")]
    query_summary = mean_records(queries, ("scenario", "budget", "task", "method", "selected_support_bin"), q_numeric)
    write_csv(PUBLIC / "B_SUPPORT_QUERY_SUMMARY.csv", query_summary)
    return dict(cell_support_summary=summary, selected_support_summary=query_summary,
        cell_rows=len(cells), query_rows=len(queries), evaluated_cell_rows=len(valid_cells),
        interpretation="held-out descriptive association; support is not randomized; three-repeat means remain noisy")


if __name__ == "__main__":
    evaluate()
