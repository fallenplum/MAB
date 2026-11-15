"""Freeze full-train600 heads and fresh400 input-only execution requirements.

This is a preparation entry point. It reads only historical train600 rewards,
the already cached fresh400 input vectors, and importer metadata. It never opens
fresh/sealed gold, model answers, aggregate answers, scores, or actual output
lengths. Running it creates three CPU fits and immutable preparation artifacts;
it does not execute or score the independent evaluation.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import time

from e17_common import (
    ASSETS, CODE, E8, E13, INPUTS, OUTCOMES, PRIVATE, PUBLIC, SEEDS,
    array_hash, read_json, ridge, sha256, timing, verify_protocol,
    write_json,
)
from confirmation import input_call_union, subtract_verified_cache_keys
from objectives import fit_spo, select_actions

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


INPUT_FOLDER = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
FRESH_INPUTS = INPUT_FOLDER / "fresh_dev_test_inputs.npz"
INPUT_RECEIPT = INPUT_FOLDER / "INPUT_ASSETS_RECEIPT.json"
SELECTION = ASSETS / "artifacts/private/routing_redesign_20261104/fresh_dev400/SELECTION_MANIFEST_v1.json"
POOL = ASSETS / "configs/pool_v1.json"
STAGE2 = ASSETS / "artifacts/private/routing_redesign_20261104/stage2_generation"
TEAMS = np.array(list(combinations(range(6), 3)), dtype=np.int64)
HEADS = ("Ridge", "SPO_coordinate", "TSPO_uniform")
POLICIES = HEADS + ("GlobalFixed", "TaskFixedOracle", "SubtaskFixedOracle")
RECEIPT_DIRECTORIES = (
    "e2b9ac8c63b86d87", "bf05154826ddc626", "a6d972a0ae5a8296", "77bf182d449b5456",
    "bc4bddccfa0085fa", "fd593f2d6693a32d", "ec08a78049776ade", "639b843a06246b22",
)
SAFE_INPUT_KEYS = {
    "canonical_hashes", "embeddings", "question_ids", "subtask_labels", "task_labels",
}
AGGREGATOR_FINGERPRINT = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
STATUS = "FROZEN_INPUT_ONLY_EVALUATION_NOT_EXECUTED"


def _record(path, role):
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size, role=role)


def _validate_inputs(packet, questions, task_count):
    if set(packet) != SAFE_INPUT_KEYS:
        raise ValueError("Input packet must contain exactly the five label-free feature/identity keys")
    if packet["embeddings"].shape != (questions, 384) or not np.isfinite(packet["embeddings"]).all():
        raise ValueError("Original finite MiniLM384 input shape differs")
    for name in SAFE_INPUT_KEYS - {"embeddings"}:
        if packet[name].shape != (questions,):
            raise ValueError("Input identity/metadata length differs")
    for name in ("question_ids", "canonical_hashes"):
        if len(set(packet[name].tolist())) != questions:
            raise ValueError("Duplicate input identity")
    if any(int(np.sum(packet["task_labels"] == task)) != task_count for task in ("gsm8k", "bbh")):
        raise ValueError("Input task balance differs")


def _load_inputs_only(path):
    # Inspect member names before decoding any value, so a contaminated input
    # archive fails closed without materializing a reward/answer member.
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != SAFE_INPUT_KEYS:
            raise ValueError("An input-only archive contains forbidden or missing members")
        return {name: archive[name].copy() for name in sorted(SAFE_INPUT_KEYS)}


def _training_targets(train):
    # Only these historical NPZ members are decoded. No resource-length array is
    # needed for the input-only plan.
    with np.load(OUTCOMES, allow_pickle=False) as archive:
        required = ("question_ids", "canonical_hashes", "generation_seeds", "aggregate_scores")
        packet = {name: archive[name].copy() for name in required}
    for name in ("question_ids", "canonical_hashes"):
        if not np.array_equal(packet[name], train[name]):
            raise ValueError("Historical training scores and input identities differ")
    if not np.array_equal(packet["generation_seeds"], SEEDS):
        raise ValueError("Historical generation seed order differs")
    scores = packet["aggregate_scores"]
    if scores.shape != (3, 600, 20) or not np.isin(scores, (0, 1)).all():
        raise ValueError("Expected original three-repeat binary final aggregation scores")
    return scores.mean(axis=0)


def _frozen_parameters():
    records = []
    regularizations, alphas = [], []
    for fold in range(5):
        spo_path = E8 / f"private/training/fold{fold}/M4/FIT_LOG.json"
        ridge_path = E8 / f"private/training/fold{fold}/M0_tuned/FIT_LOG.json"
        spo = read_json(spo_path)
        mse = read_json(ridge_path)
        regularization = float(spo["final_diagnostics"]["regularization"])
        alpha = 100 * float(mse["selected_parameters"]["lambda_multiplier"]) / 3
        if regularization <= 0 or alpha <= 0:
            raise ValueError("Original positive regularization required")
        regularizations.append(regularization)
        alphas.append(alpha)
        records.append(dict(
            fold=fold, spo_regularization=regularization, ridge_mean_target_alpha=alpha,
            spo_source=_record(spo_path, "frozen M4 regularization"),
            ridge_source=_record(ridge_path, "frozen M0_tuned alpha"),
        ))
    return dict(
        spo_regularization=float(np.median(regularizations)),
        ridge_mean_target_alpha=float(np.median(alphas)),
        source_folds=records,
        rule="Median of five frozen final parameter values; no n600 rescaling or retuning.",
        equal_numerical_values_do_not_imply_equal_effective_complexity=True,
    )


def _feature_design(train_embeddings, fresh_embeddings):
    # Literal E8 recipe, fitted anew on the full historical600 inputs only.
    pca = PCA(n_components=16, svd_solver="full").fit(np.asarray(train_embeddings, dtype=np.float64))
    transformed = pca.transform(np.asarray(train_embeddings, dtype=np.float64))
    scaler = StandardScaler().fit(transformed)
    history = np.column_stack((np.ones(len(train_embeddings)), scaler.transform(transformed)))
    evaluation = np.column_stack((
        np.ones(len(fresh_embeddings)),
        scaler.transform(pca.transform(np.asarray(fresh_embeddings, dtype=np.float64))),
    ))
    state = dict(
        pca_mean=pca.mean_, pca_components=pca.components_,
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
    )
    return history, evaluation, state


def _fixed_actions(reward, train, fresh):
    # The canonical action row is the immutable physical team identity. Exact
    # ties use its fixed semantic order, unchanged by any E17 numeric renaming.
    global_action = int(select_actions(reward.mean(axis=0, keepdims=True))[0])
    task_actions, subtask_actions = {}, {}
    for task in sorted(set(map(str, train["task_labels"]))):
        selected = train["task_labels"] == task
        task_actions[task] = int(select_actions(reward[selected].mean(axis=0, keepdims=True))[0])
        for subtask in sorted(set(map(str, train["subtask_labels"][selected]))):
            within = selected & (train["subtask_labels"] == subtask)
            subtask_actions[(task, subtask)] = int(select_actions(reward[within].mean(axis=0, keepdims=True))[0])
    task_result = np.array([task_actions.get(str(task), global_action) for task in fresh["task_labels"]], dtype=int)
    subtask_result = np.array([
        subtask_actions.get((str(task), str(subtask)), task_actions.get(str(task), global_action))
        for task, subtask in zip(fresh["task_labels"], fresh["subtask_labels"])
    ], dtype=int)
    return np.stack((np.full(len(task_result), global_action), task_result, subtask_result)), dict(
        global_action=global_action, task_actions=task_actions,
        subtask_actions=[dict(task=task, subtask=subtask, action=action) for (task, subtask), action in subtask_actions.items()],
        fallback="subtask to task to global, all chosen from historical600 rewards",
        input_permissions=dict(GlobalFixed="no task labels", TaskFixedOracle="true task label", SubtaskFixedOracle="true task and BBH subtask labels"),
    )


def _receipt_verified_members(pool, fresh_questions):
    aliases = {str(model["alias"]).casefold(): model for model in pool["models"]}
    keys, records = set(), []
    for directory in RECEIPT_DIRECTORIES:
        path = STAGE2 / "phase_returns" / directory / "LOCAL_INGEST_RECEIPT_v1.json"
        receipt = read_json(path)
        if receipt["schema"] != "v2_stage2_member_closed_phase_local_ingest_v1":
            raise ValueError("Unexpected importer receipt schema")
        if receipt["status"] != "CLOSED_MEMBER_PHASE_IMPORTED_VERIFIED" or receipt["software_test_only"] is not False:
            raise ValueError("Unverified or software-only member cache")
        alias = str(receipt["alias"]).casefold()
        model = aliases[alias]
        member = int(model["index"])
        generation = int(receipt["member_generation_seed"])
        if receipt["role"] != "member" or receipt["phase"] != f"fresh_g{generation}" or generation not in SEEDS:
            raise ValueError("Cache phase, seed or role differs")
        if receipt["logical_model_fingerprint"] != model["fingerprint"] or receipt["aggregator_fingerprint"] != AGGREGATOR_FINGERPRINT:
            raise ValueError("Cache model/operator identity differs")
        resources, claim = receipt["resources"], receipt["assignment_claim"]
        if any(int(resources[key]) != fresh_questions for key in ("successful_jobs", "assigned_jobs")):
            raise ValueError("The receipt does not certify the complete fresh400 phase")
        if not resources["complete"] or int(resources["unknown_cost_rows"]) != 0 or int(claim["assigned_job_count"]) != fresh_questions:
            raise ValueError("Incomplete or unknown importer assignment")
        if claim["plan_sha256"] != receipt["plan_sha256"] or str(claim["alias"]).casefold() != alias or claim["phase"] != receipt["phase"]:
            raise ValueError("Importer assignment claim differs")
        if any(int(receipt[key]) != 0 for key in ("gold_scoring_ops", "fresh_gold_decoded", "sealed_payloads_decoded")):
            raise ValueError("Importer metadata declares prior scoring or gold decoding")
        output = receipt["canonical_output"]
        if not Path(output["path"]).is_file():
            raise FileNotFoundError("Receipt-backed member cache is no longer present")
        new_keys = {(question, generation, member) for question in range(fresh_questions)}
        if keys & new_keys:
            raise ValueError("Duplicate imported physical member phase")
        keys.update(new_keys)
        records.append(dict(
            receipt=_record(path, "metadata-only verified complete fresh member importer receipt"),
            alias=receipt["alias"], member_index=member, generation_seed=generation,
            complete_questions=fresh_questions, source_plan_sha256=receipt["plan_sha256"],
            canonical_output_path=output["path"], canonical_output_declared_sha256=output["sha256"],
            payload_reopened=False, payload_sha_recomputed=False, actual_token_lengths_read=False,
        ))
    return keys, records


def _budget_summary(union, member_cache):
    remainder = subtract_verified_cache_keys(union, member_cache, [])
    counts = union["counts"]
    new_counts = remainder["counts"]
    # Input/output caps are frozen E3 operator limits, not lengths observed on
    # fresh answers. Actual Token cost remains unknown until legal execution.
    return dict(
        from_zero=counts,
        receipt_backed_cached_member_calls=len(remainder["verified_required_member_keys"]),
        receipt_backed_cached_aggregate_calls=0,
        unverified_or_missing_local_execution=dict(
            member_calls=new_counts["new_member_calls"],
            aggregate_calls=new_counts["new_aggregate_calls"],
        ),
        conservative_token_upper_bound_from_zero=6144*counts["member_calls"]+8192*counts["aggregate_calls"],
        conservative_token_upper_bound_missing_local=6144*new_counts["new_member_calls"]+8192*new_counts["new_aggregate_calls"],
        actual_fresh_tokens=None, remaining_gold_questions=counts["gold_questions"],
        deterministic_scoring_ops=counts["scoring_ops"],
        new_physical_generation_required=None,
        missing_local_is_not_proof_of_missing_remote_cache=True,
    ), remainder


def freeze_confirmation():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    verify_protocol()
    private = PRIVATE / "confirmation_input_only"
    public_plan = PUBLIC / "CONFIRMATION_INPUT_ONLY_PLAN.json"
    prefreeze = PUBLIC / "CONFIRMATION_INPUT_ONLY_CONFIG_FROZEN.json"
    if private.exists() or public_plan.exists() or prefreeze.exists():
        raise FileExistsError("Confirmation preparation is immutable; inspect any previous partial run before proceeding")

    receipt = read_json(INPUT_RECEIPT)
    historical = read_json(E13 / "public/E13_PROTOCOL.json")
    for path, digest in ((INPUTS, receipt["inputs"]["selector_train"]["sha256"]),
                         (FRESH_INPUTS, receipt["inputs"]["fresh_dev_test"]["sha256"]),
                         (SELECTION, receipt["fresh_selection"]["sha256"]),
                         (OUTCOMES, historical["source_assets"][1]["sha256"])):
        if sha256(path) != digest:
            raise ValueError("Input/training source no longer matches its frozen receipt")
    selection = read_json(SELECTION)
    if selection["status"] != "SELECTED_INPUTS_ONLY_NOT_EVALUATED":
        raise ValueError("Fresh400 selection metadata no longer declares input-only status")
    train, fresh = _load_inputs_only(INPUTS), _load_inputs_only(FRESH_INPUTS)
    _validate_inputs(train, 600, 300)
    _validate_inputs(fresh, 400, 200)
    if set(train["canonical_hashes"]) & set(fresh["canonical_hashes"]) or set(train["question_ids"]) & set(fresh["question_ids"]):
        raise ValueError("Historical/fresh original question identities overlap")
    parameters = _frozen_parameters()
    pool = read_json(POOL)
    if sorted(int(model["index"]) for model in pool["models"]) != list(range(6)):
        raise ValueError("Original six-model identity map differs")
    member_cache, cache_receipts = _receipt_verified_members(pool, 400)
    if len(member_cache) != 3200:
        raise ValueError("Expected eight distinct complete400 member phases")
    manifest = [
        _record(path, role) for path, role in (
            (INPUTS, "historical600 input-only features"),
            (OUTCOMES, "historical600 rewards; only four identity/reward NPZ members decoded"),
            (FRESH_INPUTS, "fresh400 cached input-only features"),
            (INPUT_RECEIPT, "feature encoder and input identity receipt"),
            (SELECTION, "fresh400 score-blind selection metadata"),
            (POOL, "original physical member identity map"),
        )
    ]
    source_hashes = {name: sha256(CODE/name) for name in ("freeze_confirmation.py", "confirmation.py", "objectives.py", "e17_common.py")}
    config = dict(
        schema="e17_fresh400_confirmation_input_only_prefreeze_v1", status="CONFIG_FROZEN_BEFORE_FULL600_FITS",
        parameters=parameters, source_hashes=source_hashes,
        training_questions=600, fresh_input_questions=400, generation_seeds=SEEDS,
        feature_recipe="full600-only MiniLM384 PCA16(full SVD), StandardScaler, penalized constant plus16 PC coordinates",
        heads=HEADS, input_only_policies=POLICIES, fitted_parameter_count_per_head=340,
        semantic_action_order="immutable original physical six-model identity order, then lexicographic K=3 combinations; no alias reordering",
        training_tie_rules=dict(SPO_coordinate="original canonical first exact true maximizer", TSPO_uniform="uniform over exact true maximizers"),
        service_tie_rule="semantic_exact in immutable physical team order; no near-optimal threshold",
        no_model_selection_on_fresh=True, fresh_rewards_read=0, sealed_payloads_read=0,
        new_llm_calls=0, new_encoder_forwards=0, confidence_intervals=0, bootstrap_runs=0,
    )
    PUBLIC.mkdir(parents=True, exist_ok=True)
    write_json(prefreeze, config)
    private.mkdir(parents=True)
    write_json(private/"INPUT_AND_CACHE_MANIFEST.json", dict(files=manifest, cache_importer_receipts=cache_receipts))

    reward = _training_targets(train)
    phi_h, phi_f, feature_state = _feature_design(train["embeddings"], fresh["embeddings"])
    weights, scores, fit_logs = [], [], []
    for method in HEADS:
        cpu, wall = time.process_time(), time.perf_counter()
        if method == "Ridge":
            coefficient, diagnostics = ridge(phi_h, reward, parameters["ridge_mean_target_alpha"])
        else:
            coefficient, diagnostics = fit_spo(
                phi_h, reward, parameters["spo_regularization"],
                rule="lexicographic" if method == "SPO_coordinate" else "uniform",
            )
        predicted = phi_f @ coefficient
        weights.append(coefficient)
        scores.append(predicted)
        fit_logs.append(dict(method=method, diagnostics=diagnostics, **timing(cpu, wall)))
    scores = np.stack(scores)
    head_actions = np.stack([select_actions(value, tolerance=0, semantic=True) for value in scores])
    fixed_actions, fixed_log = _fixed_actions(reward, train, fresh)
    actions = np.vstack((head_actions, fixed_actions))

    model_path = private / "FULL600_MODELS.npz"
    action_path = private / "FRESH_POLICY_ACTIONS_FROZEN.npz"
    np.savez_compressed(model_path, weights=np.stack(weights), heads=np.array(HEADS), history=np.arange(600), **feature_state)
    np.savez_compressed(action_path, policies=np.array(POLICIES), actions=actions, head_scores=scores,
                        question_ids=fresh["question_ids"], canonical_hashes=fresh["canonical_hashes"],
                        generation_seeds=np.array(SEEDS), teams=TEAMS)
    write_json(private/"FULL600_FIT_LOG.json", dict(fits=fit_logs, fixed_policies=fixed_log, new_prediction_head_fits=3))

    unions, summaries = {}, {}
    def add(name, selected, permission):
        union = input_call_union(actions[np.asarray(selected, dtype=int)], TEAMS, SEEDS)
        summary, remainder = _budget_summary(union, member_cache)
        unions[name] = dict(required=union, remainder=remainder)
        summaries[name] = dict(policies=[POLICIES[index] for index in selected], input_permission=permission, **summary)
    add("all_text_only_policies", [0,1,2,3], "embeddings plus fixed policy; no true task/subtask labels")
    add("all_policies_with_oracle_metadata", list(range(6)), "includes extra true task/subtask-label oracle diagnostics")
    for index, method in enumerate(HEADS):
        add(f"{method}_versus_GlobalFixed", [index,3], "text-only head versus no-input global default")
    add("SPO_coordinate_versus_Ridge", [1,0], "same text-only input permission")
    add("TSPO_uniform_versus_Ridge", [2,0], "same text-only input permission")
    all_twenty = np.repeat(np.arange(20, dtype=int)[:, None], 400, axis=1)
    oracle_union = input_call_union(all_twenty, TEAMS, SEEDS)
    oracle_summary, oracle_remainder = _budget_summary(oracle_union, member_cache)
    unions["all20_hindsight_diagnostic_separate"] = dict(required=oracle_union, remainder=oracle_remainder)
    summaries["all20_hindsight_diagnostic_separate"] = dict(
        input_permission="all20 reward hindsight diagnostic; not an input-only deployable routing policy",
        included_in_main_policy_union=False, **oracle_summary,
    )
    write_json(private/"CALL_UNIONS_AND_VERIFIED_REMAINDERS.json", unions)
    artifacts = [_record(path, "new private frozen confirmation preparation") for path in (
        model_path, action_path, private/"FULL600_FIT_LOG.json", private/"INPUT_AND_CACHE_MANIFEST.json",
        private/"CALL_UNIONS_AND_VERIFIED_REMAINDERS.json",
    )]
    write_json(public_plan, dict(
        schema="e17_fresh400_input_only_budget_plan_v1", status=STATUS,
        config_sha256=sha256(prefreeze), source_hashes=source_hashes,
        source_input_hashes=dict(train=sha256(INPUTS), fresh=sha256(FRESH_INPUTS)),
        private_artifact_hashes={Path(item["path"]).name:item["sha256"] for item in artifacts},
        policies=[dict(name=name, input_permission="true task/subtask metadata oracle" if "Oracle" in name else "text-only or global constant") for name in POLICIES],
        training=dict(questions=600, reward_repeats=3, unique_actions=20, new_cpu_head_fits=3,
                      full600_only_preprocessing=True, parameters=parameters),
        fresh=dict(questions=400, generation_repeats=3, rewards_read=0, gold_decoded=0,
                   score_operations_executed=0, new_model_calls=0, new_encoder_forwards=0),
        cache=dict(verified_importer_receipts=8, complete_member_outputs_declared=3200,
                   verified_aggregate_outputs=0, cache_payloads_reopened=False,
                   actual_output_lengths_read=False, remote_inventory_verified=False,
                   declaration_scope="Receipt-backed complete fresh400 assignments. Payloads are not reopened or rehashed here; later execution must validate original output identities before reuse.",
                   local_absence_does_not_establish_new_physical_call_requirement=True),
        budget_plans=summaries,
        token_bound=dict(member_input_cap=4096, member_output_cap=2048, aggregate_input_cap=6144,
                         aggregate_output_cap=2048, type="conservative per-call caps, not actual observed lengths"),
        evaluation_execution="PENDING; no gold or output read authorized by this preparation artifact",
        power_planning="Use final frozen candidates' legal original-question OOF paired differences; these full600 fitted training rewards do not provide independent confirmation power evidence.",
        independent_role="fresh400 is a once-only frozen-policy development confirmation pool; sealed200 retains its original final role",
        confidence_intervals=0, bootstrap_runs=0, **timing(start_cpu, start_wall),
    ))
    write_json(private/"CONFIRMATION_PREPARATION_FROZEN.json", dict(
        status=STATUS, config_sha256=sha256(prefreeze), public_plan_sha256=sha256(public_plan),
        private_artifacts=artifacts, actions_sha256=array_hash(actions),
        fresh_rewards_read=0, sealed_payloads_read=0, independent_evaluation_executed=False,
    ))
    print(STATUS, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-inputs-only",))
    parser.parse_args()
    freeze_confirmation()
