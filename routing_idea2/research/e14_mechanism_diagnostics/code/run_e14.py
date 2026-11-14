"""Freeze group interventions and cost-matched existing-learner predictions.

This entry point never loads held-out rewards. Prefix targets are determined
solely by complete, already executed E13 feedback packages and their ledger.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import time
import unittest

from e14_common import (
    A_METHODS, ASSETS, BUDGETS, CODE, E13, E13_CODE, LOGGERS, METHODS, PRIVATE,
    PUBLIC, SEEDS, array_hash, check_protocol, load_npz, measured, read_json,
    sha256, sources, world_path, write_json,
)
import numpy as np

sys.path.insert(0, str(E13_CODE))
from common import verify_source_lock
from learners import fit_learners, predict_learners
from saverouter_adapter import fit_saverouter, predict_from_state
from token_matching import execution_packages, match_pair_prefix


def prepare():
    verify_source_lock()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    tests = unittest.defaultTestLoader.discover(str(CODE / "tests"))
    tested = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (PRIVATE / "QUALIFICATION.log").write_text(stream.getvalue(), encoding="utf-8")
    if not tested.wasSuccessful():
        raise RuntimeError(stream.getvalue())
    restoration = []
    qualification_inputs = load_npz(E13 / "private/inputs/fold0/INPUTS_ONLY.npz")
    for scenario in ("Main", "Pair"):
        packages = execution_packages(read_json(world_path(0, 17, 42, scenario) / "EXECUTION_AND_ACCESS.json"))
        fitted, diagnostics = fit_existing_prefix(qualification_inputs, packages[:120])
        original = load_npz(world_path(0, 17, 42, scenario) / "B120/SNAPSHOT.npz")
        maximum_error = float(np.max(np.abs(fitted["scores"] - original["scores"][:8])))
        if maximum_error > 1e-10 or not np.array_equal(fitted["actions"], original["actions"][:8]):
            raise ValueError("Real fixed-learner restoration failed for " + scenario)
        restoration.append(dict(scenario=scenario, gold=120, max_score_error=maximum_error,
                                actions_identical=True, diagnostics=diagnostics))
    write_json(PUBLIC / "QUALIFICATION.json", dict(status="PASS", tests=tested.testsRun,
               real_restoration_prefix_fits=2, restoration=restoration, **measured(start_cpu, start_wall)))
    files = []
    def add(path):
        files.append(dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size))
    add(E13 / "public/E13_PROTOCOL.json")
    add(E13 / "private/ALL_PREDICTIONS_FROZEN.json")
    add(E13 / "private/FINAL_VERIFICATION.json")
    add(E13 / "private/REMOTE_DELIVERY_VERIFIED.json")
    frozen = read_json(E13 / "private/ALL_PREDICTIONS_FROZEN.json")
    if frozen["snapshots"] != 540 or frozen["worlds_count"] != 90:
        raise ValueError("Incomplete E13 study")
    original = read_json(E13 / "public/E13_PROTOCOL.json")
    for fold in range(5):
        path = E13 / f"private/inputs/fold{fold}/INPUTS_ONLY.npz"
        if sha256(path) != original["input_packets"][fold]["sha256"]:
            raise ValueError("E13 input packet differs")
        add(path)
    for world in frozen["worlds"]:
        root = world_path(world["fold"], world["logger_seed"], world["generation_seed"], world["scenario"])
        if read_json(root / "WORLD_FROZEN.json") != world:
            raise ValueError("World manifest mismatch")
        add(root / "WORLD_FROZEN.json")
        execution = root / "EXECUTION_AND_ACCESS.json"
        if sha256(execution) != world["execution_and_access_sha256"]:
            raise ValueError("Execution ledger mismatch")
        add(execution)
        for receipt in world["snapshots"]:
            checkpoint = root / f"B{receipt['budget']}"
            if read_json(checkpoint / "FROZEN.json") != receipt:
                raise ValueError("Checkpoint receipt mismatch")
            if sha256(checkpoint / "SNAPSHOT.npz") != receipt["snapshot_sha256"]:
                raise ValueError("Checkpoint weights changed")
            add(checkpoint / "FROZEN.json")
            add(checkpoint / "SNAPSHOT.npz")
    for path in sorted(E13_CODE.glob("*.py")):
        add(path)
    write_json(PRIVATE / "INPUT_MANIFEST.json", dict(files=files, protected_historical_files=len(files)))
    protocol = dict(
        schema="E14_three_controlled_mechanism_diagnostics_v1",
        evidence="REUSED_TRAIN600_EXPLORATORY_DEVELOPMENT",
        base_branch="codex/e13-partial-combination-feedback",
        base_commit="0114a7f744eb39aff24e7e2eb2dbb82af66cdd82",
        branch="codex/e14-controlled-mechanism-diagnostics",
        original_questions=600, outer_folds=5, history=480, evaluation=120,
        logger_seeds=LOGGERS, generation_seeds=SEEDS, gold_checkpoints=BUDGETS,
        primary_view=dict(scenario="Pair", gold=120, token_anchor_main_gold=120),
        diagnosis_A=dict(intervention="replace predicted group by true group in the same frozen capability table",
            methods=A_METHODS, retrained_models=0, oracle_only=True,
            group_residual="hold residual coefficients and all preprocessing fixed",
            prior_fixed_baselines="same original empirical tables; never substitute these for Group C"),
        diagnosis_B=dict(learner_updates=0, target="E120 only; no history self-evaluation",
            support_unit="distinct purchased history question per group/action within one generation",
            support_bins=["0", "1", "2", "3-4", "5-9", "10+"],
            group_intervention="true group isolates capability estimation; oracle diagnostic only",
            heldout_targets="matching-generation binary reward and three-generation empirical mean separately",
            empty_heldout_group="undefined/null", causality="descriptive support-error association"),
        diagnosis_C=dict(token_source="sum complete executed logger_service + feedback resource ledgers",
            target="actual Main tokens at six original checkpoints",
            pair_matching="longest complete original-order executed Pair prefix at or below target",
            indivisible_feedback_packages=True, ranking_by_rewards=False,
            actual_token_equality="discrete residual spend gap always reported; no accuracy interpolation",
            existing_learners=METHODS, own_alpha=1000.0,
            task_pseudocount=10.0, subgroup_pseudocount=3.0,
            saverouter="same pinned E13 native estimator/profile",
            retraining_scope="same fixed learners on already executed selected prefixes; no new algorithm",
            maximum_new_prefix_fits=270, main_snapshots="reuse existing E13 actions",
            cost_scope="historical executed cost replay; new physical calls zero",
            prospective_budget_guarantee=False),
        statistics=dict(primary_weights="GSM8K/BBH each 50%; BBH question weighted",
            repeated_measurements="logger and generation repetitions within the same original question",
            independent_questions=600, confidence_intervals=False, bootstrap=False,
            hypothesis_tests=False, task_and_seed_directions=True),
        new_physical_llm_calls=0, fresh400_reads=0, sealed200_reads=0,
        source_hashes={str(p.relative_to(CODE)): sha256(p) for p in sources()},
        protected_input_manifest_sha256=sha256(PRIVATE / "INPUT_MANIFEST.json"),
        limitations=["repeatedly exposed development assets", "three generation repetitions",
            "true group is privileged", "cost matching is retrospective and discrete"],
    )
    write_json(PUBLIC / "E14_PROTOCOL.json", protocol)
    write_json(PRIVATE / "PROTOCOL_FREEZE_RECEIPT.json", dict(status="FROZEN",
               protocol_sha256=sha256(PUBLIC / "E14_PROTOCOL.json"),
               source_hashes=protocol["source_hashes"], historical_input_files=len(files)))
    print("E14_PROTOCOL_FROZEN", len(files), flush=True)


def interventions(inputs, snapshot):
    pred, true = inputs["predicted_group_e"], inputs["oracle_group_e"]
    c = snapshot["own__C"]
    if not np.array_equal(snapshot["scores"][1], c[pred]):
        raise ValueError("Group scores do not restore from frozen capability table")
    residual = inputs["phi_e"] @ snapshot["own__residual_weights"]
    native = {key[4:]: value for key, value in snapshot.items() if key.startswith("sr__")}
    oracle_scores = dict(Group=c[true], GroupResidual=c[true] + residual,
        SaveRouterFixedMask=predict_from_state(native, inputs["phi_e"][:, 1:], true),
        TaskFixed=snapshot["own__task_fixed"][snapshot["own__task_codes"][true]],
        SubtaskFixed=snapshot["own__subtask_fixed"][true])
    for name in A_METHODS:
        old_index = snapshot["methods"].tolist().index(name)
        if not np.array_equal(snapshot["actions"][old_index], np.argmax(snapshot["scores"][old_index], axis=1)):
            raise ValueError("Original selection does not match saved scores")
    return np.stack([np.argmax(oracle_scores[name], axis=1) for name in A_METHODS])


def fit_existing_prefix(inputs, packages):
    if not packages:
        raise ValueError("Selected E14 real prefixes must have observations")
    observations = [o for package in packages for o in package["observations"]]
    lookup = {int(q): j for j, q in enumerate(inputs["H"])}
    q = np.array([lookup[int(o["question_index"])] for o in observations], dtype=np.int64)
    a = np.array([o["action"] for o in observations], dtype=np.int64)
    y = np.array([o["reward"] for o in observations], dtype=float)
    costs = np.array([o["standalone_tokens"] for o in observations], dtype=float)
    state = fit_learners(inputs["phi_h"], inputs["group_h"], inputs["group_to_task"], q, a, y)
    scores = predict_learners(state, inputs["phi_e"], inputs["predicted_group_e"])
    native = fit_saverouter(features_h=inputs["phi_h"][:, 1:], groups_h=inputs["group_h"],
        query_indices=q, action_indices=a, rewards=y, costs=costs,
        query_features=inputs["phi_e"][:, 1:], query_groups=inputs["predicted_group_e"])
    scores["SaveRouterFixedMask"] = native["scores"]
    arrays = dict(scores=np.stack([scores[name] for name in METHODS]),
        actions=np.stack([np.argmax(scores[name], axis=1) for name in METHODS]),
        observed_local_queries=q, observed_actions=a, observed_rewards=y)
    arrays.update({"own__" + key: value for key, value in state.items() if isinstance(value, np.ndarray)})
    arrays.update({"sr__" + key: value for key, value in native["state"].items()})
    return arrays, dict(own=state["diagnostics"], saverouter=native["diagnostics"])


def freeze_predictions():
    check_protocol()
    if (PRIVATE / "ALL_DIAGNOSTIC_PREDICTIONS_FROZEN.json").exists() or (PRIVATE / "EVALUATION_RECEIPT.json").exists():
        raise FileExistsError("E14 prediction freeze already completed")
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    a_oracle = np.full((2, 6, 3, 3, len(A_METHODS), 600), -1, dtype=np.int16)
    c_actions = np.full((6, 3, 3, 2, len(METHODS), 600), -1, dtype=np.int16)
    matches, fit_records, sealed = [], [], []
    for fold in range(5):
        inputs = load_npz(E13 / f"private/inputs/fold{fold}/INPUTS_ONLY.npz")
        evaluation = inputs["E"]
        if set(evaluation) & set(inputs["H"]):
            raise ValueError("Evaluation/history overlap")
        for li, logger in enumerate(LOGGERS):
            for gi, seed in enumerate(SEEDS):
                for si, scenario in enumerate(("Main", "Pair")):
                    for bi, budget in enumerate(BUDGETS):
                        path = world_path(fold, logger, seed, scenario) / f"B{budget}/SNAPSHOT.npz"
                        snap = load_npz(path)
                        a_oracle[si, bi, li, gi][:, evaluation] = interventions(inputs, snap)
                main_log = read_json(world_path(fold, logger, seed, "Main") / "EXECUTION_AND_ACCESS.json")
                pair_log = read_json(world_path(fold, logger, seed, "Pair") / "EXECUTION_AND_ACCESS.json")
                main, pair = execution_packages(main_log), execution_packages(pair_log)
                seen_prefix = {}
                for bi, budget in enumerate(BUDGETS):
                    match = match_pair_prefix(main, pair, budget)
                    count = match["pair_questions"]
                    dest = PRIVATE / f"prefixes/fold{fold}/logger{logger}/seed{seed}/N{count}"
                    dest.mkdir(parents=True, exist_ok=True)
                    snapshot_path, receipt_path = dest / "PREDICTIONS.npz", dest / "FROZEN.json"
                    if count not in seen_prefix:
                        if receipt_path.exists():
                            receipt = read_json(receipt_path)
                            if receipt["sha256"] != sha256(snapshot_path):
                                raise ValueError("E14 prefix cache changed")
                            arrays = load_npz(snapshot_path)
                        else:
                            cpu, wall = time.process_time(), time.perf_counter()
                            arrays, diagnostics = fit_existing_prefix(inputs, pair[:count])
                            # If a prefix hits an E13 checkpoint, verify exact reuse equivalence.
                            verification = None
                            if count in BUDGETS:
                                original = load_npz(world_path(fold, logger, seed, "Pair") / f"B{count}/SNAPSHOT.npz")
                                delta = float(np.max(np.abs(arrays["scores"] - original["scores"][:8])))
                                if delta > 1e-10 or not np.array_equal(arrays["actions"], original["actions"][:8]):
                                    raise ValueError("Same fixed learner does not restore original checkpoint")
                                verification = delta
                            arrays["evaluation_indices"] = evaluation
                            np.savez_compressed(snapshot_path, **arrays)
                            receipt = dict(status="FROZEN", sha256=sha256(snapshot_path),
                                protocol_sha256=sha256(PUBLIC / "E14_PROTOCOL.json"),
                                selected_questions=count, selected_observations=2*count,
                                evaluation_hash=array_hash(evaluation), diagnostics=diagnostics,
                                matching_existing_checkpoint_max_score_error=verification,
                                **measured(cpu, wall))
                            write_json(receipt_path, receipt)
                        seen_prefix[count] = arrays
                        fit_records.append(dict(fold=fold, logger_seed=logger, generation_seed=seed,
                                                pair_questions=count, receipt=str(receipt_path), **receipt))
                        sealed.append(dict(path=str(snapshot_path), sha256=sha256(snapshot_path)))
                    arrays = seen_prefix[count]
                    original_main = load_npz(world_path(fold, logger, seed, "Main") / f"B{budget}/SNAPSHOT.npz")
                    c_actions[bi, li, gi, 0][:, evaluation] = original_main["actions"][:8]
                    c_actions[bi, li, gi, 1][:, evaluation] = arrays["actions"]
                    matches.append(dict(fold=fold, logger_seed=logger, generation_seed=seed,
                                        generation_index=gi, anchor_main_gold=budget, **match))
            print(f"E14_FROZEN fold={fold} logger={logger}", flush=True)
    if np.any(a_oracle < 0) or np.any(c_actions < 0):
        raise ValueError("Missing question coverage")
    np.savez_compressed(PRIVATE / "A_ORACLE_ACTIONS_FROZEN.npz", actions=a_oracle)
    np.savez_compressed(PRIVATE / "C_MATCHED_ACTIONS_FROZEN.npz", actions=c_actions)
    write_json(PRIVATE / "TOKEN_MATCHES_FROZEN.json", matches)
    write_json(PRIVATE / "FIT_LEDGER.json", fit_records)
    for name in ("A_ORACLE_ACTIONS_FROZEN.npz", "C_MATCHED_ACTIONS_FROZEN.npz", "TOKEN_MATCHES_FROZEN.json", "FIT_LEDGER.json"):
        path = PRIVATE / name
        sealed.append(dict(path=str(path), sha256=sha256(path)))
    write_json(PRIVATE / "ALL_DIAGNOSTIC_PREDICTIONS_FROZEN.json", dict(status="FROZEN_BEFORE_E14_EVALUATION",
        protocol_sha256=sha256(PUBLIC / "E14_PROTOCOL.json"), files=sealed,
        unique_existing_learner_prefix_fits=len(fit_records), original_E13_snapshots_reused=540,
        heldout_rewards_loaded_by_runner=False, new_physical_llm_calls=0, **measured(start_cpu, start_wall)))
    print("E14_PREDICTIONS_FROZEN", len(fit_records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "freeze"))
    args = parser.parse_args()
    prepare() if args.command == "prepare" else freeze_predictions()
