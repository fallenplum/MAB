"""Aggregate frozen E13 feedback ledgers without reading dense reward caches.

The only decoded array from a model snapshot is observed_standalone_tokens.
Its sum is a no-member-reuse accounting counterfactual, never procurement cost.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import shutil
import sys
import time

from common import (
    BUDGETS, CODE, GENERATION_SEEDS, LOGGER_SEEDS, METHODS, PRIVATE, PUBLIC,
    SCENARIOS, Timer, read_json, sha256, verify_source_lock, write_json,
)
import numpy as np


def resource_distribution(records, expected):
    if len(records) != expected:
        raise ValueError("Unexpected number of matched feedback cost records")
    keys = tuple(records[0])
    if any(set(row) != set(keys) for row in records):
        raise ValueError("Inconsistent frozen resource dictionary keys")
    values = np.asarray([[row[key] for key in keys] for row in records], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Frozen resource values must be finite and nonnegative")
    return {"paths": expected,
            "mean": {key: float(value) for key, value in zip(keys, values.mean(axis=0))},
            "min": {key: float(value) for key, value in zip(keys, values.min(axis=0))},
            "max": {key: float(value) for key, value in zip(keys, values.max(axis=0))}}


def add_resource_rows(rows):
    keys = tuple(rows[0])
    return {key: sum(row[key] for row in rows) for key in keys}


def validate_ledger(snapshot):
    scenario, budget = snapshot["scenario"], int(snapshot["budget"])
    ledger = snapshot["ledger"]
    if tuple(snapshot["methods"]) != METHODS:
        raise ValueError("A feedback ledger no longer covers the common method set")
    expected = {"gold": budget, "member_calls": (4 if scenario == "Pair" else 3)*budget,
                "aggregator_calls": (2 if scenario == "Pair" else 1)*budget,
                "scoring_ops": (2 if scenario == "Pair" else 1)*budget}
    if any(ledger["total"][key] != value for key, value in expected.items()):
        raise ValueError("Frozen single-generation package accounting changed")
    summed = add_resource_rows([ledger["logger_service"], ledger["feedback"]])
    for key, value in summed.items():
        if not np.isclose(value, ledger["total"][key], atol=1e-8, rtol=1e-12):
            raise ValueError("Logger service plus incremental feedback does not equal total")
    if ledger["total"]["input_tokens"] + ledger["total"]["output_tokens"] != ledger["total"]["tokens"]:
        raise ValueError("Frozen token components do not add up")
    keys = set(tuple(value) for value in snapshot["gold_keys"])
    if len(keys) != budget:
        raise ValueError("Canonical/reference-version gold deduplication changed")
    return keys


def counterfactual_row(snapshot, standalone_sum):
    total = snapshot["ledger"]["total"]
    scenario, budget = snapshot["scenario"], int(snapshot["budget"])
    expected_tuples = budget * (2 if scenario == "Pair" else 1)
    saving = int(standalone_sum) - int(total["tokens"])
    if saving < 0 or (scenario == "Main" and saving != 0):
        raise ValueError("Observed standalone tokens are inconsistent with shared execution")
    return {
        "gold": int(total["gold"]), "member_calls": 3*expected_tuples,
        "aggregator_calls": expected_tuples, "scoring_ops": expected_tuples,
        "tokens": int(standalone_sum), "tokens_saved_by_shared_members": saving,
        "members_saved_by_shared_members": 3*expected_tuples-int(total["member_calls"]),
    }


def checkpoint_source(root, snapshot):
    budget = int(snapshot["budget"])
    path = root / f"B{budget}/SNAPSHOT.npz"
    receipt = read_json(root / f"B{budget}/FROZEN.json")
    if receipt != snapshot or sha256(path) != snapshot["snapshot_sha256"]:
        raise ValueError("Frozen cost checkpoint no longer matches its receipt")
    with np.load(path, allow_pickle=False) as archive:
        standalone = archive["observed_standalone_tokens"].copy()
    expected = budget * (2 if snapshot["scenario"] == "Pair" else 1)
    if standalone.shape != (expected,) or not np.isfinite(standalone).all() or (standalone < 0).any():
        raise ValueError("Executed standalone token vector is invalid")
    if not np.equal(standalone, np.round(standalone)).all():
        raise ValueError("Executed token values must be integer counts")
    return int(standalone.sum())


def timing_sum(rows, cpu="cpu_s", wall="wall_s"):
    return {"cpu_s": float(sum(row[cpu] for row in rows)),
            "wall_s_sum": float(sum(row[wall] for row in rows))}


def fitting_costs(global_freeze, snapshots, protocol, qualification):
    own = [row["own_diagnostics"] for row in snapshots]
    sr = [row["saverouter_diagnostics"] for row in snapshots]
    fits = [row["fitting_cost"] for row in snapshots]
    solves = [item for row in own for item in row["solve_records"]]
    actual_solves = sum(row["actual_normal_matrix_solves"] for row in own)
    rhs_fits = sum(row["model_rhs_fits"] for row in own)
    if actual_solves != len(solves) or rhs_fits != 3*actual_solves:
        raise ValueError("Three-RHS sparse Ridge computation count changed")
    residual_count = sum(row["native_training_summary"]["residual_models"] for row in sr)
    if residual_count != sum(len(row["residual_action_models"]) for row in sr):
        raise ValueError("Native residual quality model counts disagree")
    new_formal = sum(world["actual_new_checkpoint_fits"] for world in global_freeze["worlds"])
    reused = sum(world["reused_checkpoints"] for world in global_freeze["worlds"])
    if new_formal != 539 or reused != 1 or len(snapshots) != 540:
        raise ValueError("Expected one fully reused pilot plus 539 formal checkpoints")
    pilot = read_json(PRIVATE / "RESOURCE_PILOT.json")["snapshot"]
    matching_pilot = [row for row in snapshots if all(row[key] == pilot[key] for key in
                      ("fold", "logger_seed", "generation_index", "scenario", "budget"))]
    if len(matching_pilot) != 1 or matching_pilot[0]["snapshot_sha256"] != pilot["snapshot_sha256"]:
        raise ValueError("Pilot checkpoint was not reused byte-for-byte")
    workers = global_freeze["worker_costs"]
    return {
        "actual_unique_checkpoint_fits": 540, "formal_new_checkpoint_fits": 539,
        "resource_pilot_checkpoint_fits": 1, "pilot_reused_in_formal": 1,
        "own_actual_normal_matrix_solves": int(actual_solves),
        "own_model_right_hand_side_fits": int(rhs_fits),
        "own_rhs_per_normal_matrix": 3,
        "own_normal_matrix_solve_timing": timing_sum(solves),
        "own_all_sparse_capability_and_ridge_timing": timing_sum(own),
        "own_prediction_timing": timing_sum([row["own_prediction_timing"] for row in snapshots]),
        "native_saverouter_prior_quality_fits": len(sr),
        "native_saverouter_residual_quality_models": int(residual_count),
        "native_saverouter_observed_cost_table_builds": len(sr),
        "native_saverouter_fit_timing": timing_sum(sr),
        "all_unique_checkpoint_fitting_timing": timing_sum(fits),
        "formal_orchestrator_wall_s": global_freeze["orchestrator_wall_s"],
        "formal_worker_process_timing": timing_sum([row["time"] for row in workers]),
        "formal_worker_cache_load_timing": timing_sum([row["cache_load_cost"] for row in workers]),
        "maximum_recorded_worker_peak_rss_bytes": max(row["time"]["peak_rss_bytes"] for row in workers),
        "preparation_input_only_mapping_verification": protocol["preparation_cost"],
        "group_mapping_verification_small_solves": 5,
        "new_encoder_or_PCA_fits": 0,
        "software_qualification": qualification["timing"],
        "world_timing_sum": timing_sum([row["time"] for row in global_freeze["worlds"]]),
        "total_end_to_end_cpu_s": None,
        "timing_caveats": [
            "Formal worker process timing includes their cache loading, fitting, logger replay and serialization",
            "The pilot is outside formal worker timing and its fitting component is included once in unique checkpoint totals",
            "Fitting, solve, prediction, worker and preparation timers are nested or overlapping components, not additive totals",
            "Wall sums overlap across three workers and are not the orchestrator elapsed wall time",
            "Tiny per-solve CPU values can be zero at Windows timer granularity",
            "Maximum worker peak RSS is per process, not an observed simultaneous system memory peak",
        ],
    }


def main():
    outputs = [PUBLIC / "COST.json", PUBLIC / "FEEDBACK_COSTS.csv", PRIVATE / "BUILD_COST_RUN.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("E13 cost output already exists")
    source_receipt = PRIVATE / "COST_SOURCE_FROZEN_v2.json"
    source = {"status": "FROZEN_BEFORE_COST_AGGREGATION", "source_sha256": sha256(__file__),
              "protocol_sha256": sha256(PUBLIC / "E13_PROTOCOL.json"),
              "global_prediction_freeze_sha256": sha256(PRIVATE / "ALL_PREDICTIONS_FROZEN.json"),
              "decoded_snapshot_fields": ["observed_standalone_tokens"],
              "dense_outcome_cache_opened": False,
              "revision": "v2 clarifies manual-check actual/shared and no-reuse token field names; v1 retained in private/cost_versions/v1",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if source_receipt.exists():
        previous = read_json(source_receipt)
        for key in ("source_sha256", "protocol_sha256", "global_prediction_freeze_sha256"):
            if previous[key] != source[key]:
                raise ValueError("Cost source freeze changed")
    else:
        write_json(source_receipt, source)
    with Timer() as timer:
        protocol = verify_source_lock()
        qualification = read_json(PUBLIC / "QUALIFICATION.json")
        all_frozen = read_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json")
        if all_frozen["status"] != "FROZEN" or all_frozen["worlds_count"] != 90 or all_frozen["snapshots"] != 540:
            raise ValueError("All prescribed E13 worlds must be complete")
        if all_frozen["protocol_sha256"] != source["protocol_sha256"]:
            raise ValueError("Cost aggregation protocol identity changed")
        seen_worlds, records, csv_rows, all_snapshots = set(), defaultdict(list), [], []
        for world in all_frozen["worlds"]:
            fold, logger, generation, scenario = (world[key] for key in ("fold", "logger_seed", "generation_index", "scenario"))
            world_key = (fold, logger, generation, scenario)
            if world_key in seen_worlds:
                raise ValueError("Duplicate frozen world")
            seen_worlds.add(world_key)
            root = PRIVATE / f"worlds/fold{fold}/logger{logger}/seed{GENERATION_SEEDS[generation]}/{scenario}"
            if read_json(root / "WORLD_FROZEN.json") != world:
                raise ValueError("World receipt differs from all-prediction freeze")
            if [row["budget"] for row in world["snapshots"]] != list(BUDGETS):
                raise ValueError("A world lacks the six cumulative checkpoints")
            for snapshot in world["snapshots"]:
                gold_keys = validate_ledger(snapshot)
                standalone_sum = checkpoint_source(root, snapshot)
                counterfactual = counterfactual_row(snapshot, standalone_sum)
                record = {"fold": fold, "logger_seed": logger, "generation_index": generation,
                          "snapshot": snapshot, "gold_keys": gold_keys, "counterfactual": counterfactual}
                records[(scenario, snapshot["budget"])].append(record)
                all_snapshots.append(snapshot)
                row = {"scenario": scenario, "budget_gold_questions": snapshot["budget"],
                       "fold": fold, "logger_seed": logger, "generation_index": generation,
                       "generation_seed": GENERATION_SEEDS[generation],
                       "observed_single_generation_tuples": snapshot["own_diagnostics"]["n_observations"],
                       "standalone_sum_tokens_counterfactual_only": standalone_sum,
                       "tokens_saved_by_member_reuse": counterfactual["tokens_saved_by_shared_members"]}
                for phase, resources in snapshot["ledger"].items():
                    row.update({phase + "__" + name: value for name, value in resources.items()})
                csv_rows.append(row)
        expected_worlds = {(f, logger, g, scenario) for f in range(5) for logger in LOGGER_SEEDS
                           for g in range(3) for scenario in SCENARIOS}
        if seen_worlds != expected_worlds:
            raise ValueError("Frozen world grid differs from the protocol")
        conditions = {scenario: {} for scenario in SCENARIOS}
        hand_checks = []
        for scenario in SCENARIOS:
            for budget in BUDGETS:
                rows = records[(scenario, budget)]
                phases = {phase: resource_distribution([row["snapshot"]["ledger"][phase] for row in rows], 45)
                          for phase in ("logger_service", "feedback", "total")}
                grouped = defaultdict(list)
                for row in rows:
                    grouped[(row["fold"], row["logger_seed"])].append(row)
                joint, joint_phase_rows, joint_counterfactual = [], defaultdict(list), []
                joint_details = []
                for (fold, logger), generation_rows in sorted(grouped.items()):
                    if {row["generation_index"] for row in generation_rows} != {0, 1, 2}:
                        raise ValueError("Three-generation joint account is incomplete")
                    gold_union = set().union(*(row["gold_keys"] for row in generation_rows))
                    if len(gold_union) != budget or any(row["gold_keys"] != gold_union for row in generation_rows):
                        raise ValueError("Generation paths do not share the same question/reference mask")
                    joint_resource = add_resource_rows([row["snapshot"]["ledger"]["total"] for row in generation_rows])
                    joint_resource["gold"] = len(gold_union)
                    if joint_resource["member_calls"] != (12 if scenario == "Pair" else 9)*budget:
                        raise ValueError("Joint member-call count is inconsistent")
                    if joint_resource["aggregator_calls"] != (6 if scenario == "Pair" else 3)*budget:
                        raise ValueError("Joint aggregation-call count is inconsistent")
                    joint.append(joint_resource)
                    for phase in ("logger_service", "feedback", "total"):
                        value = add_resource_rows([row["snapshot"]["ledger"][phase] for row in generation_rows])
                        if phase in ("feedback", "total"):
                            value["gold"] = len(gold_union)
                        joint_phase_rows[phase].append(value)
                    joint_cf = add_resource_rows([row["counterfactual"] for row in generation_rows])
                    joint_cf["gold"] = len(gold_union)
                    joint_counterfactual.append(joint_cf)
                    joint_details.append({"fold": fold, "logger_seed": logger, "resources": joint_resource})
                condition = {"gold_budget_original_questions": budget,
                             "single_generation_path": phases["total"],
                             "single_generation_phases": phases,
                             "joint_three_generations": {**resource_distribution(joint, 15),
                                                          "per_fold_logger": joint_details,
                                                          "gold_rule": "union of canonical question/reference-version keys, not sum of three path gold values"},
                             "joint_three_generation_phases": {phase: resource_distribution(values, 15)
                                                                 for phase, values in joint_phase_rows.items()},
                             "no_member_reuse_counterfactual": {
                                 "single_generation_path": resource_distribution([row["counterfactual"] for row in rows], 45),
                                 "joint_three_generations": resource_distribution(joint_counterfactual, 15),
                                 "role": "accounting counterfactual only; all learners already share the same lawful reuse permission",
                                 "input_output_and_wall_split": None,
                             }}
                conditions[scenario][str(budget)] = condition
                if budget in (60, 120, 480):
                    first = rows[0]
                    hand_checks.append({"scenario": scenario, "budget": budget, "fold": first["fold"],
                                        "logger_seed": first["logger_seed"], "generation_index": first["generation_index"],
                                        "gold": first["snapshot"]["ledger"]["total"]["gold"],
                                        "member_calls": first["snapshot"]["ledger"]["total"]["member_calls"],
                                        "aggregator_calls": first["snapshot"]["ledger"]["total"]["aggregator_calls"],
                                        "actual_shared_execution_tokens": first["snapshot"]["ledger"]["total"]["tokens"],
                                        "no_reuse_counterfactual_tokens": first["counterfactual"]["tokens"],
                                        "token_savings_from_member_reuse": first["counterfactual"]["tokens_saved_by_shared_members"],
                                        "actual_resources": first["snapshot"]["ledger"]["total"],
                                        "no_member_reuse_counterfactual": first["counterfactual"]})
        costs = {"status": "COMPLETE", "schema": "E13_frozen_shared_partial_feedback_costs_v2",
                 "conditions": conditions,
                 "computation": fitting_costs(all_frozen, all_snapshots, protocol, qualification),
                 "physical_new": {"llm_calls": 0, "member_generation_calls": 0, "aggregation_calls": 0,
                                  "gold_procurement": 0, "encoder_calls": 0},
                 "gpu_elapsed_s": None,
                 "source_wall_caveat": "Existing per-call source wall sums are execution proxies, not measured new GPU elapsed time",
                 "historical_full_train600_investment": {"questions": 600, "gold": 600,
                                                         "member_calls": 10800, "aggregator_calls": 36000,
                                                         "aggregate_scoring_ops": 36000,
                                                         "role": "previously paid common dense cache, not charged anew per learner or replay"},
                 "heldout_policy_service_cost": {
                     "location": "E13_RESULTS.json/scenarios/{Main|Pair}/{B}/methods/{method}/service_execution_cost",
                     "source_questions_per_world": 120,
                     "role": "policy-dependent E120 service costs computed after prediction freeze; distinct from H feedback procurement",
                     "rule": "Do not add per-world H480 feedback means to service aggregates with a different fold/logger/generation denominator",
                 },
                 "counting_rules": [
                     "Each single-generation distribution contains 45 paths: five folds, three logger seeds, three generation seeds",
                     "Each joint-three-generation distribution contains 15 fold/logger groups and deduplicates gold by canonical/reference version",
                     "All ten methods receive exactly the same feedback ledger; feedback procurement cost is not multiplied by methods",
                     "Six budgets are cumulative nested scenarios and must not be added as one procurement",
                     "Overlapping five training folds and three logger repetitions are development conditions, not independent data purchases",
                     "Standalone tuple tokens describe independently executing each three-member combination and are not the shared procurement total",
                 ],
                 "manual_scope_checks": hand_checks,
                 "sources": {"cost_generator_sha256": source["source_sha256"],
                             "protocol_sha256": source["protocol_sha256"],
                             "all_predictions_frozen_sha256": source["global_prediction_freeze_sha256"]}}
        csv_rows.sort(key=lambda row: (row["scenario"], row["budget_gold_questions"], row["fold"], row["logger_seed"], row["generation_index"]))
        with (PUBLIC / "FEEDBACK_COSTS.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        write_json(PUBLIC / "COST.json", costs)
        code_target = PUBLIC / "code/build_costs.py"
        if code_target.exists():
            if sha256(code_target) != sha256(__file__):
                raise FileExistsError("A different cost generator already exists in public/code")
        else:
            shutil.copy2(__file__, code_target)
    run = {"status": "PASS", "source_freeze_sha256": sha256(source_receipt),
           "source_sha256": sha256(__file__), "timing": timer.record,
           "verified_worlds": 90, "verified_snapshots": 540,
           "decoded_npz_fields": ["observed_standalone_tokens"],
           "dense_reward_or_evaluation_payload_reads": 0,
           "csv_rows": len(csv_rows), "manual_scope_checks": len(hand_checks),
           "new_fits": 0, "new_llm_calls": 0,
           "output_sha256": {path.name: sha256(path) for path in outputs[:2]}}
    write_json(PRIVATE / "BUILD_COST_RUN.json", run)
    print("E13_COST_COMPLETE", len(csv_rows), "paths", costs["computation"]["own_actual_normal_matrix_solves"], "own solves")


if __name__ == "__main__":
    main()
