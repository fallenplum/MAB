"""Fixed-cost feedback-structure replay with three unchanged learners."""
from __future__ import annotations

import argparse
import io
import time
import unittest

from e15_common import (
    ASSETS, CAPS, CODE, DESIGNS, E13, E14, E14_CODE, INPUTS, LOGGERS,
    METHODS, PRIMARY_CAP, PRIVATE, PUBLIC, SEEDS, STRUCTURES, array_hash,
    fit_existing_prefix, load_npz, measured, new_world_path, read_json,
    sha256, sources, support_statistics, verify_protocol, verify_source_lock,
    world_path, write_csv, write_json,
)
from trajectory import REFERENCE_VERSION, build_random_world, choose_random_second, execution_packages, select_prefix
from feedback_environment import load_train600
import numpy as np


def selected_arrays(original):
    # Original eight-method order is hash-locked by the E13/E14 protocols.
    index = np.array([3, 1, 4])
    arrays = {k: v.copy() for k, v in original.items() if k.startswith(("own__", "sr__"))}
    arrays.update(scores=original["scores"][index].copy(), actions=original["actions"][index].copy())
    for name in ("observed_local_queries", "observed_actions", "observed_rewards"):
        arrays[name] = original[name].copy()
    return arrays


def prepare():
    verify_source_lock()
    if REFERENCE_VERSION != read_json(E13 / "public/E13_PROTOCOL.json")["reference_version"]:
        raise ValueError("RandomPair gold identity must match original E13")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    tested = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.discover(str(CODE / "tests")))
    (PRIVATE / "QUALIFICATION.log").write_text(stream.getvalue(), encoding="utf-8")
    if not tested.wasSuccessful():
        raise RuntimeError(stream.getvalue())
    write_json(PUBLIC / "QUALIFICATION.json", dict(status="PASS", tests=tested.testsRun))
    previous=ASSETS/"research/e15_feedback_structure_interaction_preflight_v1_reference_fix"
    if previous.exists():
        old_pilot=read_json(previous/"private/RESOURCE_PILOT.json")
        write_json(PRIVATE/"PREFLIGHT_AMENDMENT.json",dict(
            reason="RandomPair gold reference tag corrected from scorer identity to original V2 bundle",
            previous_protocol_sha256=sha256(previous/"public/E15_PROTOCOL.json"),
            previous_pilot=old_pilot,previous_model_packets=old_pilot["unique_model_packets"],
            previous_pilot_used_for_final_results=False,quality_results_read_before_amendment=0,
            reward_and_token_arrays_changed=False,full_v2_rebuild=True))
    old_manifest = read_json(E14 / "private/INPUT_MANIFEST.json")
    files = old_manifest["files"].copy()
    for p in (INPUTS, E14 / "public/E14_PROTOCOL.json", E14 / "private/FINAL_VERIFICATION.json",
              E14 / "private/REMOTE_DELIVERY_VERIFIED.json"):
        files.append(dict(path=str(p), sha256=sha256(p), bytes=p.stat().st_size))
    for path in sorted(E14_CODE.glob("*.py")):
        files.append(dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size))
    for row in read_json(E14 / "private/ALL_DIAGNOSTIC_PREDICTIONS_FROZEN.json")["files"]:
        # Only prefix weights can be reused; never ingest old evaluation tables.
        if "prefixes" in row["path"]:
            files.append(dict(path=row["path"], sha256=row["sha256"]))
    # The random second-action plans use inputs/identities only and are frozen
    # before opening the cache manager with reward/cost arrays.
    ids = load_npz(INPUTS)["question_ids"]
    plans = {}
    for fold in range(5):
        for logger in LOGGERS:
            log = read_json(world_path(fold, logger, SEEDS[0], "Main") / "EXECUTION_AND_ACCESS.json")
            packages = execution_packages(log, scenario="Main")
            q = np.array([p["question_index"] for p in packages], dtype=np.int64)
            main = np.array([p["main_action"] for p in packages], dtype=np.int64)
            second = np.array([choose_random_second(str(ids[i]), fold, logger, int(a)) for i, a in zip(q, main)], dtype=np.int64)
            plans[f"f{fold}_l{logger}_q"] = q
            plans[f"f{fold}_l{logger}_main"] = main
            plans[f"f{fold}_l{logger}_second"] = second
    np.savez_compressed(PRIVATE / "RANDOM_ACTION_PLANS_FROZEN.npz", **plans)
    write_json(PRIVATE / "INPUT_MANIFEST.json", dict(files=files))
    protocol = dict(
        schema="E15_fixed_actual_token_feedback_structure_interaction_v2",
        amendment="pre-quality v1 pilot archived; RandomPair gold identity corrected to original V2 bundle",
        reference_version=REFERENCE_VERSION,
        evidence="REPEATEDLY_EXPOSED_TRAIN600_DEVELOPMENT_REPLAY",
        parent_branch="codex/e14-controlled-mechanism-diagnostics",
        parent_commit="9f84007bf4d5ebf516f20a8e0fc74b5e721c4526",
        branch="codex/e15-feedback-structure-interaction",
        original_questions=600, history=480, evaluation=120, outerfolds=5,
        members=6, combination_size=3, combinations=20, aggregator="frozen V2 Qwen3",
        logger_seeds=LOGGERS, generation_seeds=SEEDS, features="original fold-frozen MiniLM PCA16",
        absolute_token_caps=CAPS, unique_primary_cap=PRIMARY_CAP,
        token_unit="one generation trajectory, input plus output, logger service plus feedback",
        cap_origin="250000 is a rounded fixed resource scale near prior E14 Main120; no E15 quality inspected",
        structures=dict(Main="one original uniform main combination",
            NeighborPair="same main plus frozen uniform member-replacement neighbor",
            RandomPair="same main plus uniform choice from remaining 19, including nine neighbors"),
        random_action_rule="independent input/identity/fold/logger hash; shared across three generation seeds",
        cost_rules="actual union of executed members, each distinct aggregate once, gold once, scoring per aggregate",
        matching="longest original-order whole-package prefix <= absolute cap; actual spend and residual reported",
        budget_scope="retrospective frozen-cache resource comparison; full plan pool and selected prefix separated",
        methods=METHODS, history_permissions="all historical input/group metadata; purchased sparse rewards only",
        query_permissions="text-only predicted group; true-group capability metrics are oracle diagnostics",
        learner_parameters="all E13/E14 own alpha=1000, task prior10, subgroup prior3 and pinned native SaveRouter unchanged",
        common_coverage="N=min three token-matched question counts; same N original question prefix in every structure",
        decomposition="Nfull-Mfull=(Ncommon-Mcommon)+(Nfull-Ncommon)-(Mfull-Mcommon)",
        maximum_unique_learner_packets=900,
        primary_comparisons=["Group NeighborPair-Main", "Group NeighborPair-RandomPair",
                             "(NeighborPair-Main)Group-(NeighborPair-Main)RawCentered"],
        descriptive_go_gate=dict(primary_group_neighbor_minus_main_pp_at_least=1.0,
            primary_group_neighbor_minus_random_positive=True, primary_interaction_positive=True,
            group_neighbor_minus_main_positive_generation_count_at_least=2,
            group_neighbor_minus_main_positive_fold_count_at_least=3,
            other_caps_group_neighbor_minus_main_positive_count_at_least=2),
        stopping="complete all predeclared cells; failed gate stops automatic expansion; passed gate permits frozen confirmation planning only",
        statistics=dict(primary_weights="GSM8K and BBH each 50%, BBH question-weighted",
            independent_unit="600 original questions", repetitions="generation and logger within original question",
            confidence_intervals=False, bootstrap=False, hypothesis_tests=False,
            decision_gate="descriptive practical screen, no error-rate guarantee"),
        source_hashes={str(p.relative_to(CODE)): sha256(p) for p in sources()},
        random_plan_sha256=sha256(PRIVATE / "RANDOM_ACTION_PLANS_FROZEN.npz"),
        historical_manifest_sha256=sha256(PRIVATE / "INPUT_MANIFEST.json"),
        new_physical_llm_calls=0, fresh400_reads=0, sealed200_reads=0,
    )
    write_json(PUBLIC / "E15_PROTOCOL.json", protocol)
    write_json(PRIVATE / "PROTOCOL_FREEZE_RECEIPT.json", dict(status="FROZEN", protocol_sha256=sha256(PUBLIC / "E15_PROTOCOL.json")))
    print("E15_PROTOCOL_FROZEN", flush=True)


def legacy_cache(fold, logger, seed, structure, count):
    if structure == "RandomPair":
        return None
    old = world_path(fold, logger, seed, structure) / f"B{count}/SNAPSHOT.npz"
    if old.exists():
        return old
    if structure == "NeighborPair":
        old = E14 / f"private/prefixes/fold{fold}/logger{logger}/seed{seed}/N{count}/PREDICTIONS.npz"
        if old.exists():
            return old
    return None


def fit_or_reuse(inputs, packages, fold, logger, seed, structure, count):
    root = new_world_path(fold, logger, seed) / f"{structure}/N{count}"
    root.mkdir(parents=True, exist_ok=True)
    out = root / "PREDICTIONS.npz"
    receipt_path = root / "FROZEN.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if sha256(out) != receipt["sha256"]:
            raise ValueError("E15 cached packet changed")
        return load_npz(out), receipt
    cpu, wall = time.process_time(), time.perf_counter()
    original = legacy_cache(fold, logger, seed, structure, count)
    if original:
        arrays = selected_arrays(load_npz(original))
        diagnostics = dict(reused_path=str(original), original_sha256=sha256(original))
    else:
        fitted, diagnostics = fit_existing_prefix(inputs, packages[:count])
        arrays = selected_arrays(fitted)
    # Exact same observations, order and costs underlie all three learners.
    obs = [o for p in packages[:count] for o in p["observations"]]
    lookup = {int(q): j for j, q in enumerate(inputs["H"])}
    expected_q = np.array([lookup[o["question_index"]] for o in obs], dtype=int)
    expected_a = np.array([o["action"] for o in obs], dtype=int)
    expected_y = np.array([o["reward"] for o in obs], dtype=float)
    if not all(np.array_equal(arrays[k], v) for k, v in
        (("observed_local_queries", expected_q), ("observed_actions", expected_a), ("observed_rewards", expected_y))):
        raise ValueError("Reused or fitted model observed packet mismatch")
    if inputs["group_h"][expected_q].max() != 27:
        raise ValueError("Native SaveRouter group domain does not cover all 28 fixed groups")
    count_matrix, reward_sum = support_statistics(inputs, arrays)
    arrays.update(group_count=count_matrix, group_sum=reward_sum, evaluation_indices=inputs["E"], methods=np.array(METHODS))
    np.savez_compressed(out, **arrays)
    receipt = dict(status="FROZEN", sha256=sha256(out), original_reused=original is not None,
        protocol_sha256=sha256(PUBLIC / "E15_PROTOCOL.json"),
        fold=fold, logger_seed=logger, generation_seed=seed, structure=structure, original_questions=count,
        observed_scalar_rewards=len(obs), observation_hashes=dict(query=array_hash(expected_q), action=array_hash(expected_a), reward=array_hash(expected_y)),
        unobserved_groups=int(np.sum(np.bincount(inputs["group_h"][np.unique(expected_q)], minlength=28)==0)),
        missing_group_action_cells=int(np.sum(count_matrix==0)), diagnostics=diagnostics, **measured(cpu, wall))
    write_json(receipt_path, receipt)
    return arrays, receipt


def freeze(pilot=False):
    verify_protocol()
    if (PRIVATE / "ALL_PREDICTIONS_FROZEN.json").exists():
        raise FileExistsError("E15 is already frozen")
    cpu, wall = time.process_time(), time.perf_counter()
    train, refs = load_train600()
    plans = load_npz(PRIVATE / "RANDOM_ACTION_PLANS_FROZEN.npz")
    all_actions = np.full((2, 4, 3, 3, 3, 3, 600), -1, dtype=np.int16)
    records, fit_receipts, all_files, world_receipts = [], {}, [], []
    for fold in range(5):
        inputs = load_npz(E13 / f"private/inputs/fold{fold}/INPUTS_ONLY.npz")
        e = inputs["E"]
        for li, logger in enumerate(LOGGERS):
            for gi, seed in enumerate(SEEDS):
                main_log=read_json(world_path(fold, logger, seed, "Main") / "EXECUTION_AND_ACCESS.json")
                neighbor_log=read_json(world_path(fold, logger, seed, "NeighborPair") / "EXECUTION_AND_ACCESS.json")
                main = execution_packages(main_log, scenario="Main")
                neighbor = execution_packages(neighbor_log, scenario="NeighborPair")
                random = build_random_world(train, inputs, main, fold, logger, gi, train["ids"])
                for log in (main_log,neighbor_log,random["log"]):
                    if log["final_summary"]["reference_version"] != REFERENCE_VERSION or any(
                        key[1] != REFERENCE_VERSION for key in log["final_summary"]["gold_keys"]):
                        raise ValueError("Feedback structures use different canonical/reference identities")
                expected_second = plans[f"f{fold}_l{logger}_second"]
                if not np.array_equal(np.array([next(a for a in p["actions"] if a != p["main_action"]) for p in random["packages"]]), expected_second):
                    raise ValueError("Random actions differ from pre-cost/reward plan")
                root = new_world_path(fold, logger, seed)
                log_path = root / "RANDOM_PAIR_EXECUTION.json"
                if not log_path.exists():
                    write_json(log_path, random["log"])
                world_receipts.append(dict(fold=fold, logger_seed=logger, generation_seed=seed,
                                           path=str(log_path), sha256=sha256(log_path)))
                packages = dict(Main=main, NeighborPair=neighbor, RandomPair=random["packages"])
                for bi, cap in enumerate(CAPS):
                    chosen = {name: select_prefix(packages[name], cap) for name in STRUCTURES}
                    common_n = min(v["n_questions"] for v in chosen.values())
                    for di, design in enumerate(DESIGNS):
                        for si, structure in enumerate(STRUCTURES):
                            selected = chosen[structure]
                            n = selected["n_questions"] if di == 0 else common_n
                            actual = select_prefix(packages[structure], sum(p["resources"]["tokens"] for p in packages[structure][:n]))
                            if actual["n_questions"] != n:
                                raise ValueError("Common question prefix changed")
                            arrays, receipt = fit_or_reuse(inputs, packages[structure], fold, logger, seed, structure, n)
                            all_actions[di, bi, li, gi, si][:, e] = arrays["actions"]
                            model_path = root / f"{structure}/N{n}/PREDICTIONS.npz"
                            fit_receipts[str(model_path)] = receipt
                            records.append(dict(fold=fold, logger_seed=logger, generation_index=gi,
                                generation_seed=seed, design=design, token_cap=cap, structure=structure,
                                questions=n, common_questions=common_n, model_path=str(model_path),
                                model_sha256=receipt["sha256"], ledger=actual["ledger"],
                                actual_tokens=actual["actual_tokens"], remaining_tokens=cap-actual["actual_tokens"],
                                next_package_tokens=actual["next_package_tokens"],
                                complete_pool_exhausted=n==len(packages[structure]),
                                full_token_matched_questions=selected["n_questions"],
                                full_plan_ledger=selected["full_plan_ledger"]))
                print(f"E15_FROZEN fold={fold} logger={logger} seed={seed}", flush=True)
                if pilot:
                    write_json(PRIVATE / "RESOURCE_PILOT.json", dict(status="PASS", fold=fold,
                        logger_seed=logger,generation_seed=seed,diagnostic_design_records=len(records),
                        unique_model_packets=len(fit_receipts), outer_rewards_evaluated=0,
                        reused_by_formal_run=True, **measured(cpu,wall)))
                    return
    if np.any(all_actions < 0):
        raise ValueError("E15 prediction coverage incomplete")
    np.savez_compressed(PRIVATE / "ACTIONS_FROZEN.npz", actions=all_actions)
    write_json(PRIVATE / "DESIGN_RECORDS_FROZEN.json", records)
    write_json(PRIVATE / "FIT_LEDGER.json", list(fit_receipts.values()))
    for path, receipt in fit_receipts.items():
        all_files.append(dict(path=path, sha256=receipt["sha256"]))
    for name in ("ACTIONS_FROZEN.npz", "DESIGN_RECORDS_FROZEN.json", "FIT_LEDGER.json"):
        path = PRIVATE / name
        all_files.append(dict(path=str(path), sha256=sha256(path)))
    write_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json", dict(status="FROZEN_BEFORE_E15_EVALUATION",
        files=all_files, random_worlds=world_receipts, unique_learner_packets=len(fit_receipts),
        newly_fitted_packets=sum(not r["original_reused"] for r in fit_receipts.values()),
        legacy_reused_packets=sum(r["original_reused"] for r in fit_receipts.values()),
        protocol_sha256=sha256(PUBLIC / "E15_PROTOCOL.json"), new_physical_llm_calls=0,
        learner_read_only_purchased_historical_rewards=True, cache_source_refs=refs, **measured(cpu, wall)))
    print("E15_ALL_PREDICTIONS_FROZEN", len(fit_receipts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "freeze"))
    args = parser.parse_args()
    prepare() if args.command == "prepare" else freeze(pilot=args.command=="smoke")
