"""Fixed-mask partial-feedback training with isolated sparse learner inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
from pathlib import Path
import shutil
import time
import unittest

from common import (
    ASSETS, BUDGETS, CODE, GENERATION_SEEDS, INPUTS, LOGGER_SEEDS, METHODS,
    PRIVATE, PUBLIC, REFERENCE_VERSION, SCENARIOS, SOURCE_HASHES, STUDY,
    Timer, array_hash, choice, input_features, jsonable, load_npz, read_json,
    sha256, source_manifest, verify_source_lock, write_json,
)
from feedback_environment import PartialFeedbackEnv, load_train600, make_logger_plan, source_identity
from learners import fit_learners, predict_learners
from saverouter_adapter import UPSTREAM_COMMIT, UPSTREAM_HASHES, fit_saverouter, load_upstream
import numpy as np


OWN_ARRAYS = ("raw_weights", "residual_weights", "centered_weights", "C", "global_a", "task_a",
              "global_fixed", "task_fixed", "subtask_fixed", "group_to_task", "task_codes")


def prepare():
    if (PUBLIC / "E13_PROTOCOL.json").exists():
        raise FileExistsError("E13 protocol is already frozen")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    if sha256(INPUTS) != SOURCE_HASHES["inputs"]:
        raise ValueError("Original input identity changed")
    inputs = load_npz(INPUTS)
    stream = io.StringIO()
    with Timer() as qualification:
        suite = unittest.defaultTestLoader.discover(str(CODE / "tests"), pattern="test_*.py")
        tested = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (PRIVATE / "QUALIFICATION.log").write_text(stream.getvalue(), encoding="utf-8")
    if not tested.wasSuccessful():
        raise RuntimeError("E13 software qualification failed")
    write_json(PUBLIC / "QUALIFICATION.json", dict(status="PASS", tests=tested.testsRun,
               failures=len(tested.failures), errors=len(tested.errors), timing=qualification.record))
    with Timer() as preparation:
        inputs_receipts = []
        all_e = []
        for fold in range(5):
            packet = input_features(fold, inputs)
            root = PRIVATE / f"inputs/fold{fold}"
            root.mkdir(parents=True, exist_ok=True)
            path = root / "INPUTS_ONLY.npz"
            np.savez_compressed(path, **packet)
            inputs_receipts.append(dict(fold=fold, sha256=sha256(path), history=480, evaluation=120,
                                        input_only=True, old_reward_coefficient_loaded=False))
            all_e.extend(packet["E"].tolist())
        if len(set(all_e)) != 600:
            raise ValueError("Original five evaluation folds do not partition train600")
    _, _, sr_profile = load_upstream()
    core_names = ("common.py", "feedback_environment.py", "learners.py", "saverouter_adapter.py", "run_e13.py")
    core_files = [CODE / name for name in core_names] + sorted((CODE / "tests").glob("test_*.py"))
    protocol = dict(
        schema="E13_real_partial_combination_feedback_v1", evidence="REUSED_TRAIN600_EXPLORATORY_DEVELOPMENT",
        hypothesis="Task-group statistical sharing uses identical genuinely sparse combination feedback more effectively",
        members=6, members_per_service=3, actions=20, aggregator="original frozen V2 Qwen3-8B",
        original_questions=600, outerfolds=5, history_pool=480, evaluation=120,
        budgets=list(BUDGETS), logger_seeds=list(LOGGER_SEEDS), generation_seeds=list(GENERATION_SEEDS),
        feedback_scenarios={"Main": "one frozen uniform main combination per revealed question",
                            "Pair": "same main plus one frozen uniform single-member-replacement neighbor"},
        feedback_unit="one question, one generation, one actually executed combination score",
        masks="E12 nested question IDs, uniform hash-based reward-blind logger actions, common to every learner and generation",
        feedback_worlds=90, snapshots=540,
        input="frozen H480 MiniLM PCA16/Scaler; ours penalized constant, native SaveRouter implicit unpenalized intercept",
        grouping="g_hat(x)=exact argmax(Phi(x) T_M), using the existing reward-free E12 text/metadata group map",
        query_permissions="text-only for all eight main methods; two true task/subtask oracle diagnostics separately marked",
        history_permissions="all H480 input vectors and task/subtask metadata; only purchased sparse reward tuples",
        methods=list(METHODS),
        raw=dict(alpha_per_single_generation_observation=1000.0, penalized_constant=True,
                 no_observation_action="zero coefficient prior, never a zero-filled training label"),
        group=dict(grand="mean of purchased rewards; empty prior .5",
                   global_action="(sum_a+10*grand)/(count_a+10)",
                   task_action="(sum_ta+10*global_a)/(count_ta+10)",
                   bbh_subtask_action="(sum_ga+3*task_bbh_a)/(count_ga+3)",
                   gsm="task capability, no duplicated subtask level", prediction="C[g_hat(x)]"),
        residual="one sparse ridge residual per action, r-C_LOO(group,a), exclude all purchased actions for the current whole question",
        raw_centered_control="same observed targets centered by legal global_action mean; isolates zero-prior/count shrinkage from task sharing",
        fixed_baselines="observed empirical global/task/subtask means with observed-only hierarchical fallback; main uses predicted groups",
        saverouter=dict(commit=UPSTREAM_COMMIT, profile="llmrouterbench", parameters=sr_profile,
                        upstream_source_hashes=UPSTREAM_HASHES, acquisition="replaced by common fixed logger",
                        estimator="original FixedKSparseRouter unchanged", decision="quality-only score argmax",
                        cost_input="standalone executed combination tokens, not shared incremental acquisition costs"),
        reference_version=REFERENCE_VERSION,
        reference_version_caveat="fixed reference-cache identity tag bound to original V2 bundle, not a hash of gold content",
        resource_rules=dict(gold="once per canonical question/reference version within a trajectory; union across three generations reported",
                            main="3 members+1 aggregate+1 score+1 gold per question", pair="4 members+2 aggregates+2 scores+1 gold",
                            neighbor_increment="1 member+1 aggregate+1 score, same gold",
                            actual_tokens="read only after the action execution is accepted; prompt/completion and member/aggregate separated",
                            equal_cost="every learner shares exactly the same full feedback execution ledger",
                            final_service="policy-dependent E120 service cost separately accounted after prediction freeze",
                            hard_caps="fixed counts plus public conservative token ceilings; no hidden-cost selection"),
        primary=dict(scenario="Pair", gold_budget=120,
                     comparisons=["Group-Raw", "Group-RawCentered", "GroupResidual-SaveRouterFixedMask"],
                     bootstrap=5000, seed=202611150013, strata="outerfold x task", independent_unit="original question",
                     repeated_unit="three generation trajectories x three logger seeds averaged within each question",
                     multiplicity="Holm across the three predeclared primary contrasts; ordinary95% paired bootstrap intervals labeled separately"),
        no_new_loss=True, no_hyperparameter_search=True, no_new_encoder=True,
        new_physical_llm_calls=0, new_gold_procurement=0, validation_fresh_sealed_payload_reads=0,
        model_selection="all settings fixed now; no evaluation-driven refitting or winner replacement",
        scientific_scope="pool-based sparse-feedback offline replay, heldout E120 receives no feedback",
        source_hashes={str(p.relative_to(CODE)): sha256(p) for p in core_files},
        source_assets=sources, input_packets=inputs_receipts, preparation_cost=preparation.record,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(PUBLIC / "E13_PROTOCOL.json", protocol)
    write_json(PRIVATE / "PROTOCOL_FREEZE_RECEIPT.json", dict(status="FROZEN", protocol_sha256=sha256(PUBLIC / "E13_PROTOCOL.json"),
                actual_real_feedback_model_fits_before_freeze=0))
    for path in core_files:
        target = PUBLIC / "code" / path.relative_to(CODE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print("E13_PROTOCOL_FROZEN", sha256(PUBLIC / "E13_PROTOCOL.json"), tested.testsRun, "tests", flush=True)


def _observed_packet(env, history, generation, budget, scenario):
    sparse = env.training_packet()
    reverse = np.full(600, -1, dtype=int)
    reverse[history] = np.arange(len(history))
    local_query = reverse[sparse["query_indices"]]
    expected = budget * (2 if scenario == "Pair" else 1)
    if (len(local_query) != expected or np.any(local_query < 0)
            or not np.all(sparse["generation_index"] == generation)
            or not np.isin(sparse["rewards"], (0, 1)).all()):
        raise ValueError("Only purchased binary single-generation feedback may enter learners")
    pairs = local_query*20+sparse["action_indices"]
    if len(set(pairs.tolist())) != expected:
        raise ValueError("Duplicate query/action feedback")
    packet_hash = {name: array_hash(values) for name, values in sparse.items()}
    return sparse, local_query, packet_hash


def run_world(train, fold, logger_seed, generation, scenario, smoke=False):
    frozen_protocol = verify_source_lock()
    input_path = PRIVATE / f"inputs/fold{fold}/INPUTS_ONLY.npz"
    if sha256(input_path) != frozen_protocol["input_packets"][fold]["sha256"]:
        raise ValueError("Frozen input-only preprocessing packet changed")
    inputs = load_npz(input_path)
    history, evaluation = inputs["H"], inputs["E"]
    root = PRIVATE / f"worlds/fold{fold}/logger{logger_seed}/seed{GENERATION_SEEDS[generation]}/{scenario}"
    done_path = root / "WORLD_FROZEN.json"
    if done_path.exists() and not smoke:
        done = read_json(done_path)
        if done["protocol_sha256"] != sha256(PUBLIC / "E13_PROTOCOL.json"):
            raise ValueError("Completed world refers to a different protocol")
        for receipt in done["snapshots"]:
            if sha256(root / f"B{receipt['budget']}/SNAPSHOT.npz") != receipt["snapshot_sha256"]:
                raise ValueError("Completed world snapshot changed")
        if sha256(root / "EXECUTION_AND_ACCESS.json") != done["execution_and_access_sha256"]:
            raise ValueError("Completed world feedback access history changed")
        return done
    root.mkdir(parents=True, exist_ok=True)
    plan = make_logger_plan(history, train["ids"], logger_seed, fold)
    plan_by_q = {int(q): (int(a), int(b)) for q, a, b in zip(plan["question_indices"], plan["main_actions"], plan["neighbor_actions"])}
    main_multiplier = 4 if scenario == "Pair" else 3
    aggregate_multiplier = 2 if scenario == "Pair" else 1
    limit = dict(gold=480, member_calls=main_multiplier*480,
                 aggregator_calls=aggregate_multiplier*480, scoring_ops=aggregate_multiplier*480,
                 tokens=(main_multiplier*6144+aggregate_multiplier*8192)*480)
    env = PartialFeedbackEnv(train, history, generation, forbidden_questions=evaluation,
                             reference_version=REFERENCE_VERSION, limits=limit)
    visited, snapshots = set(), []
    reused, actual_fits = 0, 0
    with Timer() as world_timer:
        for budget in BUDGETS:
            for q in inputs[f"B{budget}"]:
                q = int(q)
                if q in visited:
                    continue
                main, neighbor = plan_by_q[q]
                env.serve(q, main)
                env.buy(q, [main, neighbor] if scenario == "Pair" else [main])
                visited.add(q)
            sparse, local_query, packet_hash = _observed_packet(env, history, generation, budget, scenario)
            checkpoint = root / f"B{budget}"
            receipt_path = checkpoint / "FROZEN.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if receipt["observed_packet_hashes"] != packet_hash or sha256(checkpoint / "SNAPSHOT.npz") != receipt["snapshot_sha256"]:
                    raise ValueError("Cached checkpoint lineage mismatch")
                snapshots.append(receipt)
                reused += 1
            else:
                with Timer() as fit_timer:
                    state = fit_learners(inputs["phi_h"], inputs["group_h"], inputs["group_to_task"],
                                         local_query, sparse["action_indices"], sparse["rewards"])
                    scores = predict_learners(state, inputs["phi_e"], inputs["predicted_group_e"])
                    native = fit_saverouter(features_h=inputs["phi_h"][:, 1:], groups_h=inputs["group_h"],
                              query_indices=local_query, action_indices=sparse["action_indices"], rewards=sparse["rewards"],
                              costs=sparse["standalone_tokens"], query_features=inputs["phi_e"][:, 1:],
                              query_groups=inputs["predicted_group_e"])
                    scores["SaveRouterFixedMask"] = native["scores"]
                    true_group = inputs["oracle_group_e"]
                    scores["TaskFixedOracle"] = state["task_fixed"][state["task_codes"][true_group]]
                    scores["SubtaskFixedOracle"] = state["subtask_fixed"][true_group]
                    scores = np.stack([scores[name] for name in METHODS])
                    actions = choice(scores)
                checkpoint.mkdir(parents=True, exist_ok=True)
                summary = env.summary()
                totals = summary["ledger"]["total"]
                if (totals["gold"] != budget or totals["member_calls"] != main_multiplier*budget
                        or totals["aggregator_calls"] != aggregate_multiplier*budget
                        or totals["scoring_ops"] != aggregate_multiplier*budget):
                    raise ValueError("Complete fixed-package execution accounting failed")
                payload = dict(methods=np.array(METHODS), scores=scores, actions=actions, evaluation_indices=evaluation,
                               observed_global_queries=sparse["query_indices"], observed_local_queries=local_query,
                               observed_actions=sparse["action_indices"], observed_rewards=sparse["rewards"],
                               observed_standalone_tokens=sparse["standalone_tokens"],
                               predicted_service_costs=native["predicted_service_costs"])
                payload.update({"own__"+key: state[key] for key in OWN_ARRAYS})
                payload.update({"own_counts__"+key: np.asarray(value) for key, value in state["observed_counts"].items()})
                payload.update({"sr__"+key: value for key, value in native["state"].items()})
                np.savez_compressed(checkpoint / "SNAPSHOT.npz", **payload)
                receipt = dict(status="FROZEN", fold=fold, logger_seed=logger_seed,
                               generation_index=generation, generation_seed=GENERATION_SEEDS[generation],
                               scenario=scenario, budget=budget, methods=list(METHODS),
                               observed_packet_hashes=packet_hash, snapshot_sha256=sha256(checkpoint / "SNAPSHOT.npz"),
                               own_diagnostics=state["diagnostics"], own_prediction_timing=state["prediction_timing"],
                               saverouter_diagnostics=native["diagnostics"], fitting_cost=fit_timer.record,
                               observed_group_action_cells=int(np.sum(state["observed_counts"]["group_count"] > 0)),
                               total_group_action_cells=560, fallback=state["fallback"], ledger=summary["ledger"],
                               gold_keys=summary["gold_keys"], feedback_permissions="one identical sparse packet for every method")
                write_json(receipt_path, receipt)
                snapshots.append(receipt)
                actual_fits += 1
            if smoke:
                write_json(PRIVATE / "RESOURCE_PILOT.json", dict(status="PASS", snapshot=snapshots[-1],
                           real_outer_rewards_evaluated=0, reuse="Formal run reuses this complete frozen B60 checkpoint"))
                return dict(status="PILOT_PASS", folds=1, budgets=1)
        write_json(root / "EXECUTION_AND_ACCESS.json", dict(plan=plan, access_log=env.access_log,
                   execution_log=env.execution_log, final_summary=env.summary()))
    done = dict(status="FROZEN", fold=fold, logger_seed=logger_seed, generation_index=generation,
                generation_seed=GENERATION_SEEDS[generation], scenario=scenario, snapshots=snapshots,
                protocol_sha256=sha256(PUBLIC / "E13_PROTOCOL.json"),
                time=world_timer.record, reused_checkpoints=reused, actual_new_checkpoint_fits=actual_fits,
                execution_and_access_sha256=sha256(root / "EXECUTION_AND_ACCESS.json"))
    write_json(done_path, done)
    return done


def run_fold_logger(fold, logger_seed):
    with Timer() as whole_task:
        verify_source_lock()
        with Timer() as load_timer:
            train, _ = load_train600()
        records = []
        for scenario in SCENARIOS:
            for generation in range(3):
                records.append(run_world(train, fold, logger_seed, generation, scenario))
    return dict(worlds=records, time=whole_task.record, cache_load_cost=load_timer.record)


def train_all(workers):
    verify_source_lock()
    if not (PRIVATE / "RESOURCE_PILOT.json").exists():
        raise ValueError("A full mechanism resource pilot is required")
    if (PRIVATE / "ALL_PREDICTIONS_FROZEN.json").exists():
        raise FileExistsError("E13 training already completed")
    records, worker_costs, start = [], [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        tasks = {executor.submit(run_fold_logger, f, seed): (f, seed) for f in range(5) for seed in LOGGER_SEEDS}
        for future in as_completed(tasks):
            current = future.result()
            records.extend(current["worlds"])
            worker_costs.append({"fold": tasks[future][0], "logger_seed": tasks[future][1],
                                 "time": current["time"], "cache_load_cost": current["cache_load_cost"]})
            print(f"FROZEN worlds={len(records)}/90 snapshots={6*len(records)}/540 elapsed={time.perf_counter()-start:.2f}s", flush=True)
    write_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json", dict(status="FROZEN", worlds=records,
               worker_costs=worker_costs,
               protocol_sha256=sha256(PUBLIC / "E13_PROTOCOL.json"), orchestrator_wall_s=time.perf_counter()-start,
               snapshots=540, worlds_count=90, new_physical_llm_calls=0, retained_payload_reads=0))
    print("ALL_E13_PREDICTIONS_FROZEN", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "train"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "smoke":
        verify_source_lock()
        data, _ = load_train600()
        print(run_world(data, 0, 17, 0, "Pair", smoke=True), flush=True)
    else:
        train_all(args.workers)


if __name__ == "__main__":
    main()
