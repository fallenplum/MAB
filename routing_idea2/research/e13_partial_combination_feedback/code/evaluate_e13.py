"""Evaluate frozen E13 sparse-feedback policies without any model fitting.

All worlds, snapshots and source locks are checked before the evaluator seals
its own source and decodes the original train600 reward/cost archive. Each
prediction is evaluated only against its matching generation trajectory.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import itertools
from pathlib import Path
import time

from common import (
    BUDGETS, CODE, GENERATION_SEEDS, INPUTS, LOGGER_SEEDS, METHODS, OUTCOMES,
    PRIVATE, PUBLIC, SCENARIOS, SOURCE_HASHES, array_hash, choice, load_npz,
    process_state, protocol, read_json, sha256, verify_source_lock, write_json,
)
import numpy as np


MAIN_METHODS = tuple(name for name in METHODS if not name.endswith("Oracle"))
ORACLE_METHODS = tuple(name for name in METHODS if name.endswith("Oracle"))
COMPARISON_PAIRS = (
    ("Group", "Raw"), ("Group", "RawCentered"),
    ("GroupResidual", "Group"), ("GroupResidual", "Raw"),
    ("GroupResidual", "SaveRouterFixedMask"),
    ("Group", "SaveRouterFixedMask"), ("RawCentered", "Raw"),
    ("Group", "TaskFixed"), ("Group", "SubtaskFixed"),
    ("GroupResidual", "TaskFixed"), ("GroupResidual", "SubtaskFixed"),
    ("SaveRouterFixedMask", "SubtaskFixed"),
)
SERVICE_RESOURCE_KEYS = (
    "member_calls", "aggregator_calls", "gold", "scoring_ops",
    "member_input_tokens", "member_output_tokens", "aggregator_input_tokens",
    "aggregator_output_tokens", "input_tokens", "output_tokens", "tokens",
    "member_source_wall_s", "aggregator_source_wall_s", "source_model_wall_s",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _validate_snapshot(packet, indices):
    names = packet["methods"].astype(str).tolist()
    if names != list(METHODS):
        raise ValueError("Snapshot method order differs from the frozen protocol")
    if packet["scores"].shape != (len(METHODS), 120, 20):
        raise ValueError("Invalid E13 frozen score dimensions")
    if packet["actions"].shape != (len(METHODS), 120):
        raise ValueError("Invalid E13 frozen action dimensions")
    if not np.array_equal(packet["evaluation_indices"], indices):
        raise ValueError("Snapshot evaluation identity differs from original input-only pack")
    if not np.array_equal(packet["actions"], choice(packet["scores"])):
        raise ValueError("Snapshot action differs from its frozen score argmax")
    if not np.issubdtype(packet["actions"].dtype, np.integer):
        raise ValueError("Snapshot action columns must be integers")


def _validate_observed_packet(packet, receipt, history):
    budget, generation = int(receipt["budget"]), int(receipt["generation_index"])
    multiplicity = 2 if receipt["scenario"] == "Pair" else 1
    expected = budget * multiplicity
    q, a, r = (packet[name] for name in ("observed_global_queries", "observed_actions", "observed_rewards"))
    if q.shape != (expected,) or a.shape != q.shape or r.shape != q.shape:
        raise ValueError("Snapshot does not contain the prescribed sparse feedback packet")
    if not set(q.tolist()) <= set(history.tolist()) or not np.isin(r, (0, 1)).all():
        raise ValueError("Snapshot feedback escaped its historical domain or binary generation unit")
    if len(np.unique(q)) != budget or len(np.unique(q * 20 + a)) != expected:
        raise ValueError("Feedback count or duplicate query/action tuple mismatch")
    sparse = dict(query_indices=q, action_indices=a, rewards=r,
                  standalone_tokens=packet["observed_standalone_tokens"],
                  generation_index=np.full(expected, generation, dtype=int))
    if {name: array_hash(value) for name, value in sparse.items()} != receipt["observed_packet_hashes"]:
        raise ValueError("Observed packet differs from its frozen receipt")
    local = packet["observed_local_queries"]
    if not np.array_equal(history[local], q):
        raise ValueError("Sparse feedback local/global question mapping differs")
    totals = receipt["ledger"]["total"]
    if (totals["gold"] != budget
            or totals["member_calls"] != (4 if multiplicity == 2 else 3) * budget
            or totals["aggregator_calls"] != multiplicity * budget
            or totals["scoring_ops"] != multiplicity * budget):
        raise ValueError("Frozen feedback ledger differs from the complete fixed package")


def load_and_seal_predictions():
    frozen_protocol = verify_source_lock()
    if frozen_protocol["methods"] != list(METHODS):
        raise ValueError("Evaluator method order differs from the protocol")
    if frozen_protocol["primary"]["comparisons"] != [
        "Group-Raw", "Group-RawCentered", "GroupResidual-SaveRouterFixedMask"
    ]:
        raise ValueError("Evaluator primary comparison family changed")
    if frozen_protocol["primary"]["scenario"] != "Pair" or frozen_protocol["primary"]["gold_budget"] != 120:
        raise ValueError("Evaluator primary scenario or budget changed")
    if read_json(PUBLIC / "QUALIFICATION.json").get("status") != "PASS":
        raise ValueError("E13 software qualification must pass before scoring")
    global_path = PRIVATE / "ALL_PREDICTIONS_FROZEN.json"
    global_receipt = read_json(global_path)
    if (global_receipt.get("status") != "FROZEN" or global_receipt.get("snapshots") != 540
            or global_receipt.get("worlds_count") != 90):
        raise ValueError("All 90 worlds and 540 snapshots must be frozen before evaluation")
    if global_receipt["protocol_sha256"] != sha256(PUBLIC / "E13_PROTOCOL.json"):
        raise ValueError("Global prediction receipt points to a different protocol")
    expected = {(f, logger, gen, scenario) for f in range(5) for logger in LOGGER_SEEDS
                for gen in range(3) for scenario in SCENARIOS}
    worlds = global_receipt["worlds"]
    found = {(r["fold"], r["logger_seed"], r["generation_index"], r["scenario"]) for r in worlds}
    if len(worlds) != 90 or found != expected:
        raise ValueError("A prescribed world is missing or duplicated")
    source_hash = sha256(INPUTS)
    if source_hash != SOURCE_HASHES["inputs"]:
        raise ValueError("Original reward-free input identity changed")
    inputs = load_npz(INPUTS)
    if len(set(inputs["question_ids"].tolist())) != 600 or len(set(inputs["canonical_hashes"].tolist())) != 600:
        raise ValueError("Original input questions are not distinct")
    task, subtask = inputs["task_labels"], inputs["subtask_labels"]
    if set(task.tolist()) != {"gsm8k", "bbh"} or any(np.sum(task == key) != 300 for key in ("gsm8k", "bbh")):
        raise ValueError("Original balanced task identity changed")
    shape = (len(SCENARIOS), len(BUDGETS), len(LOGGER_SEEDS), len(GENERATION_SEEDS), len(METHODS), 600)
    actions = np.full(shape, -1, dtype=np.int16)
    outerfold = np.full(600, -1, dtype=np.int8)
    input_packs = {}
    sealed_files, snapshot_count = {}, 0
    for fold in range(5):
        path = PRIVATE / f"inputs/fold{fold}/INPUTS_ONLY.npz"
        if sha256(path) != frozen_protocol["input_packets"][fold]["sha256"]:
            raise ValueError("Reward-free input packet changed")
        packet = load_npz(path)
        indices = packet["E"]
        if (len(indices) != 120 or len(set(indices.tolist())) != 120
                or np.any(outerfold[indices] >= 0) or set(indices) & set(packet["H"])):
            raise ValueError("Original outer input scope overlaps or repeats")
        if any(np.sum(task[indices] == key) != 60 for key in ("gsm8k", "bbh")):
            raise ValueError("Original evaluation task balance changed")
        outerfold[indices] = fold
        input_packs[fold] = packet
        sealed_files[str(path)] = sha256(path)
    for world in worlds:
        fold, logger, generation, scenario = (world[k] for k in ("fold", "logger_seed", "generation_index", "scenario"))
        if world["generation_seed"] != GENERATION_SEEDS[generation]:
            raise ValueError("World generation seed/index mapping changed")
        root = PRIVATE / f"worlds/fold{fold}/logger{logger}/seed{GENERATION_SEEDS[generation]}/{scenario}"
        world_path = root / "WORLD_FROZEN.json"
        if read_json(world_path) != world or world["status"] != "FROZEN":
            raise ValueError("World receipt differs from the global prediction freeze")
        access = root / "EXECUTION_AND_ACCESS.json"
        if sha256(access) != world["execution_and_access_sha256"]:
            raise ValueError("Execution/access log changed after freezing")
        sealed_files[str(world_path)] = sha256(world_path)
        sealed_files[str(access)] = world["execution_and_access_sha256"]
        if len(world["snapshots"]) != len(BUDGETS) or [r["budget"] for r in world["snapshots"]] != list(BUDGETS):
            raise ValueError("World budget checkpoint sequence changed")
        indices = input_packs[fold]["E"]
        for checkpoint in world["snapshots"]:
            budget = checkpoint["budget"]
            receipt_path = root / f"B{budget}/FROZEN.json"
            snapshot_path = root / f"B{budget}/SNAPSHOT.npz"
            if read_json(receipt_path) != checkpoint or checkpoint["status"] != "FROZEN":
                raise ValueError("Checkpoint receipt differs from its world freeze")
            for name, expected_value in (("fold", fold), ("logger_seed", logger), ("generation_index", generation),
                                         ("generation_seed", GENERATION_SEEDS[generation]), ("scenario", scenario)):
                if checkpoint[name] != expected_value:
                    raise ValueError("Checkpoint world identity mismatch")
            if sha256(snapshot_path) != checkpoint["snapshot_sha256"]:
                raise ValueError("A frozen E13 snapshot changed")
            packet = load_npz(snapshot_path)
            _validate_snapshot(packet, indices)
            _validate_observed_packet(packet, checkpoint, input_packs[fold]["H"])
            s, b = SCENARIOS.index(scenario), BUDGETS.index(budget)
            l = LOGGER_SEEDS.index(logger)
            actions[s, b, l, generation][:, indices] = packet["actions"]
            sealed_files[str(snapshot_path)] = checkpoint["snapshot_sha256"]
            sealed_files[str(receipt_path)] = sha256(receipt_path)
            snapshot_count += 1
    if snapshot_count != 540 or np.any(actions < 0) or np.any(outerfold < 0):
        raise ValueError("Frozen E13 policies do not cover all question/trajectory cells")
    seal = dict(
        status="FROZEN_BEFORE_FULL_REWARD_ACCESS", created_utc=utc_now(),
        evaluator_sha256=sha256(__file__), protocol_sha256=sha256(PUBLIC / "E13_PROTOCOL.json"),
        global_freeze_sha256=sha256(global_path), input_sha256=source_hash,
        source_files={str(p.relative_to(CODE)): sha256(p) for p in CODE.glob("*.py")},
        all_actions_sha256=array_hash(actions), verified_worlds=90, verified_snapshots=540,
        frozen_files=sealed_files, primary=frozen_protocol["primary"],
        dense_reward_decoded=False, service_costs_not_available_to_learners=True,
    )
    seal_path = PRIVATE / "EVALUATOR_SOURCE_FROZEN.json"
    if seal_path.exists():
        old = read_json(seal_path)
        for key in ("evaluator_sha256", "protocol_sha256", "global_freeze_sha256", "input_sha256",
                    "all_actions_sha256", "verified_worlds", "verified_snapshots", "frozen_files", "primary"):
            if old[key] != seal[key]:
                raise ValueError("Existing evaluator seal differs in " + key)
        seal = old
    else:
        write_json(seal_path, seal)
    return actions, outerfold, task, subtask, inputs, seal, frozen_protocol


def bootstrap_indices(task, outerfold, repeats, seed):
    rng = np.random.default_rng(seed)
    pieces = []
    for fold in range(5):
        for key in ("gsm8k", "bbh"):
            rows = np.flatnonzero((outerfold == fold) & (task == key))
            if len(rows) != 60:
                raise ValueError("Bootstrap strata must preserve original 60-question fold/task cells")
            pieces.append(rows[rng.integers(0, len(rows), size=(repeats, len(rows)))])
    return np.concatenate(pieces, axis=1)


def bootstrap_means(question_values, indices):
    values = np.asarray(question_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != indices.shape[1]:
        raise ValueError("Bootstrap requires methods by independent original questions")
    result = np.empty((len(indices), len(values)), dtype=np.float64)
    for method in range(len(values)):
        result[:, method] = values[method, indices].mean(axis=1)
    return result


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    adjusted, previous = np.empty(len(p_values)), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(p_values)-rank)*p_values[index])
        adjusted[index] = min(previous, 1.0)
    return adjusted


def action_distribution(actions):
    counts = np.bincount(np.asarray(actions).ravel(), minlength=20)
    positive = counts[counts > 0] / counts.sum()
    return dict(counts=counts, total_service_decisions=int(counts.sum()), used_actions=int(np.sum(counts > 0)),
                effective_actions=float(np.exp(-np.sum(positive*np.log(positive)))),
                max_action=int(np.argmax(counts)), max_count=int(counts.max()),
                max_fraction=float(counts.max()/counts.sum()),
                count_unit="recorded logger x generation x original-question decisions")


def paired_comparison(candidate, reference, candidate_actions, reference_actions, bootstrap_difference,
                      task, outerfold):
    delta = np.asarray(candidate, dtype=float)-np.asarray(reference, dtype=float)
    question_delta = delta.mean(axis=(0, 1))
    estimate = float(question_delta.mean())
    centered = bootstrap_difference-estimate
    p = float((1+np.sum(np.abs(centered) >= abs(estimate)-1e-15))/(len(centered)+1))
    return dict(
        difference_pp=100*estimate, ci95_pp=100*np.quantile(bootstrap_difference, [.025, .975]),
        p_unadjusted=p, corrections=int(np.sum((candidate == 1) & (reference == 0))),
        destructions=int(np.sum((candidate == 0) & (reference == 1))), net_events=int(delta.sum()),
        independent_questions=len(question_delta), recorded_generation_logger_events=int(delta.size),
        improved_question_means=int(np.sum(question_delta > 0)), harmed_question_means=int(np.sum(question_delta < 0)),
        equal_question_means=int(np.sum(question_delta == 0)),
        changed_action_events=int(np.sum(candidate_actions != reference_actions)),
        changed_original_questions_any_trajectory=int(np.any(candidate_actions != reference_actions, axis=(0, 1)).sum()),
        generation_difference_pp={str(seed): float(100*delta[:, index].mean()) for index, seed in enumerate(GENERATION_SEEDS)},
        logger_difference_pp={str(seed): float(100*delta[index].mean()) for index, seed in enumerate(LOGGER_SEEDS)},
        fold_difference_pp={str(fold): float(100*delta[:, :, outerfold == fold].mean()) for fold in range(5)},
        task_difference_pp={key: float(100*delta[:, :, task == key].mean()) for key in ("gsm8k", "bbh")},
    )


def _one_service_cost(outcomes, generation, actions):
    teams = np.array(list(itertools.combinations(range(6), 3)), dtype=int)
    q = np.arange(len(actions))
    result = {key: 0 for key in SERVICE_RESOURCE_KEYS}
    for source, columns, prefix, count_key in (
        ("member", teams[actions], "member", "member_calls"),
        ("aggregate", actions, "aggregator", "aggregator_calls"),
    ):
        for suffix, key in (("execution_counts", count_key), ("prompt_tokens", prefix+"_input_tokens"),
                            ("completion_tokens", prefix+"_output_tokens"), ("wall_time_s", prefix+"_source_wall_s")):
            matrix = outcomes[source+"_"+suffix]
            selected = matrix[generation, q[:, None], columns] if source == "member" else matrix[generation, q, columns]
            result[key] = float(selected.sum()) if suffix == "wall_time_s" else int(selected.sum())
    result["input_tokens"] = result["member_input_tokens"]+result["aggregator_input_tokens"]
    result["output_tokens"] = result["member_output_tokens"]+result["aggregator_output_tokens"]
    result["tokens"] = result["input_tokens"]+result["output_tokens"]
    result["source_model_wall_s"] = result["member_source_wall_s"]+result["aggregator_source_wall_s"]
    return result


def service_cost_summary(outcomes, actions):
    trajectories = []
    for logger, logger_seed in enumerate(LOGGER_SEEDS):
        for generation, generation_seed in enumerate(GENERATION_SEEDS):
            values = _one_service_cost(outcomes, generation, actions[logger, generation])
            if values["member_calls"] != 1800 or values["aggregator_calls"] != 600:
                raise ValueError("Each service trajectory must execute three members and one aggregator per question")
            trajectories.append(dict(logger_seed=logger_seed, generation_seed=generation_seed, resources=values))
    total = {key: sum(row["resources"][key] for row in trajectories) for key in SERVICE_RESOURCE_KEYS}
    return dict(
        total_across_nine_replays=total,
        mean_per_600_question_generation_trajectory={key: value/9 for key, value in total.items()},
        mean_tokens_per_service_event=total["tokens"]/5400,
        trajectory_token_distribution=dict(minimum=min(row["resources"]["tokens"] for row in trajectories),
                                           maximum=max(row["resources"]["tokens"] for row in trajectories),
                                           mean=float(np.mean([row["resources"]["tokens"] for row in trajectories]))),
        trajectories=trajectories,
        scope="policy-specific heldout service inference only, separate from shared historical feedback procurement",
        source_wall_scope="sum of original cached call walls, not newly measured GPU inference time",
        new_physical_calls=0,
    )


def grouped_accuracy(y, mask):
    return dict(questions=int(np.sum(mask)), recorded_events=int(9*np.sum(mask)),
                accuracy_pct=float(100*y[:, :, mask].mean()))


def select_matching_generation_rewards(actions, original_rewards):
    selected = np.empty(actions.shape, dtype=np.int8)
    for scenario in range(actions.shape[0]):
        for budget in range(actions.shape[1]):
            for logger in range(actions.shape[2]):
                for generation in range(actions.shape[3]):
                    for method in range(actions.shape[4]):
                        a = actions[scenario, budget, logger, generation, method]
                        selected[scenario, budget, logger, generation, method] = original_rewards[generation, np.arange(actions.shape[-1]), a]
    return selected


def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    public_names = ("E13_RESULTS.json", "E13_PRIMARY_STATISTICS.json", "E13_SUMMARY.csv")
    if (any((PUBLIC / name).exists() for name in public_names)
            or any((PRIVATE / name).exists() for name in ("PAIRED_OUTCOMES.npz", "EVALUATION_RECEIPT.json"))):
        raise FileExistsError("Final E13 scoring outputs already exist")
    actions, outerfold, task, subtask, inputs, seal, frozen_protocol = load_and_seal_predictions()
    reward_read_utc = utc_now()
    if sha256(OUTCOMES) != SOURCE_HASHES["outcomes"]:
        raise ValueError("Original V2 reward/cost archive changed")
    outcomes = load_npz(OUTCOMES)
    if outcomes["generation_seeds"].tolist() != list(GENERATION_SEEDS):
        raise ValueError("Original generation seed order changed")
    for key in ("question_ids", "canonical_hashes"):
        if not np.array_equal(outcomes[key], inputs[key]):
            raise ValueError("Input/outcome question identity mismatch")
    original_rewards = outcomes["aggregate_scores"].astype(np.float64, copy=False)
    if original_rewards.shape != (3, 600, 20) or not np.isin(original_rewards, (0, 1)).all():
        raise ValueError("Complete original executed final aggregate rewards are required")
    selected = select_matching_generation_rewards(actions, original_rewards)
    primary_config = frozen_protocol["primary"]
    indices = bootstrap_indices(task, outerfold, int(primary_config["bootstrap"]), int(primary_config["seed"]))
    scenario_results, csv_rows = {}, []
    for scenario_index, scenario in enumerate(SCENARIOS):
        budget_results = {}
        for budget_index, budget in enumerate(BUDGETS):
            cell = selected[scenario_index, budget_index].astype(float)
            cell_actions = actions[scenario_index, budget_index]
            question_values = cell.mean(axis=(0, 1))
            bootstrap = bootstrap_means(question_values, indices)
            method_results, comparisons = {}, {}
            for method_index, name in enumerate(METHODS):
                y = cell[:, :, method_index]
                by_task = {key: grouped_accuracy(y, task == key) for key in ("gsm8k", "bbh")}
                by_subtask = {key: grouped_accuracy(y, (task == "bbh") & (subtask == key))
                              for key in sorted(set(subtask[task == "bbh"]))}
                method_results[name] = dict(
                    accuracy=float(question_values[method_index].mean()),
                    accuracy_pct=float(100*question_values[method_index].mean()),
                    accuracy_ci95_pct=100*np.quantile(bootstrap[:, method_index], [.025, .975]),
                    independent_questions=600, recorded_generation_logger_events=5400,
                    gsm8k_pct=by_task["gsm8k"]["accuracy_pct"], bbh_pct=by_task["bbh"]["accuracy_pct"],
                    bbh_subtask_macro_pct=float(np.mean([r["accuracy_pct"] for r in by_subtask.values()])),
                    input_permission=("true evaluation task/subtask metadata diagnostic" if name in ORACLE_METHODS else
                                      "text-only evaluation inputs with common frozen predicted-group mapping"),
                    equal_permission_main=name in MAIN_METHODS, by_task=by_task, by_bbh_subtask=by_subtask,
                    by_generation_seed={str(seed): dict(accuracy_pct=float(100*y[:, g].mean()))
                                        for g, seed in enumerate(GENERATION_SEEDS)},
                    by_logger_seed={str(seed): dict(accuracy_pct=float(100*y[l].mean()))
                                    for l, seed in enumerate(LOGGER_SEEDS)},
                    by_outerfold={str(fold): grouped_accuracy(y, outerfold == fold) for fold in range(5)},
                    action_distribution=action_distribution(cell_actions[:, :, method_index]),
                    service_execution_cost=service_cost_summary(outcomes, cell_actions[:, :, method_index]),
                    versus={},
                )
            for candidate, reference in COMPARISON_PAIRS:
                c, r = METHODS.index(candidate), METHODS.index(reference)
                comparisons[candidate+"-"+reference] = paired_comparison(
                    cell[:, :, c], cell[:, :, r], cell_actions[:, :, c], cell_actions[:, :, r],
                    bootstrap[:, c]-bootstrap[:, r], task, outerfold,
                )
            for method_index, name in enumerate(METHODS):
                record = method_results[name]
                for reference in ("Raw", "RawCentered", "Group", "SaveRouterFixedMask", "GlobalFixed", "TaskFixed", "SubtaskFixed"):
                    r = METHODS.index(reference)
                    record["versus"][reference] = paired_comparison(
                        cell[:, :, method_index], cell[:, :, r], cell_actions[:, :, method_index], cell_actions[:, :, r],
                        bootstrap[:, method_index]-bootstrap[:, r], task, outerfold,
                    )
                accuracy_ci = record["accuracy_ci95_pct"]
                raw = record["versus"]["Raw"]
                sr = record["versus"]["SaveRouterFixedMask"]
                service = record["service_execution_cost"]
                csv_rows.append(dict(
                    scenario=scenario, gold_per_trajectory=budget, feedback_actions_per_question=2 if scenario == "Pair" else 1,
                    method=name, equal_permission_main=record["equal_permission_main"],
                    independent_questions=600, recorded_generation_logger_events=5400,
                    accuracy_pct=record["accuracy_pct"], ci95_low_pct=float(accuracy_ci[0]), ci95_high_pct=float(accuracy_ci[1]),
                    gsm8k_pct=record["gsm8k_pct"], bbh_pct=record["bbh_pct"],
                    bbh_subtask_macro_pct=record["bbh_subtask_macro_pct"],
                    seed42_pct=record["by_generation_seed"]["42"]["accuracy_pct"],
                    seed137_pct=record["by_generation_seed"]["137"]["accuracy_pct"],
                    seed2024_pct=record["by_generation_seed"]["2024"]["accuracy_pct"],
                    versus_raw_pp=raw["difference_pp"], versus_saverouter_pp=sr["difference_pp"],
                    corrections_vs_raw=raw["corrections"], destructions_vs_raw=raw["destructions"],
                    net_events_vs_raw=raw["net_events"], effective_actions=record["action_distribution"]["effective_actions"],
                    max_action=record["action_distribution"]["max_action"],
                    max_action_events=record["action_distribution"]["max_count"],
                    service_tokens_across_nine_replays=service["total_across_nine_replays"]["tokens"],
                    mean_service_tokens_per_event=service["mean_tokens_per_service_event"],
                ))
            budget_results[str(budget)] = dict(methods=method_results, comparisons=comparisons,
                                               shared_feedback_ledger="one FROZEN.json ledger per fold/logger/generation, common to all methods")
        scenario_results[scenario] = budget_results
    primary_cell = scenario_results[primary_config["scenario"]][str(primary_config["gold_budget"])]
    primary = {key: dict(primary_cell["comparisons"][key]) for key in primary_config["comparisons"]}
    for key, adjusted in zip(primary_config["comparisons"], holm_adjust([r["p_unadjusted"] for r in primary.values()])):
        primary[key]["p_holm"] = float(adjusted)
    evaluation_cost = dict(cpu_s=time.process_time()-start_cpu, wall_s=time.perf_counter()-start_wall,
                           **process_state(), scope="frozen-policy verification, scoring and aggregate statistics before output serialization")
    axes = dict(names=["scenario", "budget", "logger_seed", "generation_seed", "method", "original_question"],
                shape=list(selected.shape), scenarios=list(SCENARIOS), budgets=list(BUDGETS),
                logger_seeds=list(LOGGER_SEEDS), generation_seeds=list(GENERATION_SEEDS), methods=list(METHODS),
                original_questions=600, repeated_events_per_question=9,
                question_unit="average the three generation and three logger values within each original question")
    result = dict(
        status="COMPLETE", evidence="REUSED_TRAIN600_EXPLORATORY_DEVELOPMENT", axes=axes,
        main_methods=list(MAIN_METHODS), oracle_diagnostic_methods=list(ORACLE_METHODS),
        method_permissions={name: ("oracle task/subtask metadata diagnostic" if name in ORACLE_METHODS else
                                   "common text-only service input permission") for name in METHODS},
        scenarios=scenario_results, primary_comparisons=primary, evaluation_cost=evaluation_cost,
        feedback_cost_permission="All methods share each actually purchased sparse tuple and its single execution ledger",
        inference_cost_permission="Actual source tokens for frozen E120 choices enter only this evaluator, never training",
        limitations=[
            "The original 600 questions have been repeatedly used for research development.",
            "Outer training pools overlap; bootstrap conditions on fitted models and fixed logger/generation trajectories.",
            "The 5400 recorded events per scenario/budget/method do not increase the 600 independent original questions.",
            "Oracle task/subtask rows have extra metadata permission and are diagnostic only.",
            "Shared partial-feedback tokens can match while final policy-specific service tokens differ.",
            "Source-call wall sums are cached historical proxies, not fresh GPU measurements.",
        ],
    )
    statistics = dict(
        status="COMPLETE", scenario=primary_config["scenario"], gold_budget=primary_config["gold_budget"],
        primary_comparisons=primary, bootstrap_repeats=int(primary_config["bootstrap"]),
        bootstrap_seed=int(primary_config["seed"]), strata="outerfold x GSM/BBH", independent_questions=600,
        cluster="Original question retaining all three generations, three logger seeds and paired policies",
        ci_definition="ordinary95% paired percentile intervals, not Holm simultaneous confidence intervals",
        p_definition="two-sided centered paired bootstrap with plus-one correction, followed by Holm3",
        bootstrap_is_approximate=True, conditional_on_fitted_models=True,
    )
    _write_csv(PUBLIC / "E13_SUMMARY.csv", csv_rows)
    write_json(PUBLIC / "E13_RESULTS.json", result)
    write_json(PUBLIC / "E13_PRIMARY_STATISTICS.json", statistics)
    np.savez_compressed(PRIVATE / "PAIRED_OUTCOMES.npz", selected_rewards=selected, actions=actions,
                        methods=np.array(METHODS), scenarios=np.array(SCENARIOS), budgets=np.array(BUDGETS),
                        logger_seeds=np.array(LOGGER_SEEDS), generation_seeds=np.array(GENERATION_SEEDS),
                        task=task, subtask=subtask, outerfold=outerfold, global_indices=np.arange(600),
                        question_ids=inputs["question_ids"], canonical_hashes=inputs["canonical_hashes"])
    receipt = dict(
        status="PASS", created_utc=utc_now(), source_seal_sha256=sha256(PRIVATE / "EVALUATOR_SOURCE_FROZEN.json"),
        evaluator_sha256=sha256(__file__), seal_created_utc=seal["created_utc"], full_reward_read_utc=reward_read_utc,
        original_outcome_sha256=sha256(OUTCOMES), original_input_sha256=sha256(INPUTS),
        public_outputs={name: sha256(PUBLIC / name) for name in public_names},
        paired_outcomes_sha256=sha256(PRIVATE / "PAIRED_OUTCOMES.npz"), evaluation_cost=evaluation_cost,
        independent_questions=600, recorded_events_per_cell=5400,
        verified_worlds=90, verified_snapshots=540, new_fits=0, new_physical_llm_calls=0,
        new_gold_procurement=0, validation_fresh_sealed_payload_reads=0,
    )
    write_json(PRIVATE / "EVALUATION_RECEIPT.json", receipt)
    print("E13_EVALUATION_COMPLETE", len(csv_rows), "summary rows", flush=True)


if __name__ == "__main__":
    main()
