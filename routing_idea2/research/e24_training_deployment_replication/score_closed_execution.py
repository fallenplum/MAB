"""Verify the full E24 native output closure before revealing local references."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from prepare_reference_locators import LOCATOR_SCHEMA, REFERENCE_VERSION, read_json, sha, write_json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e20_p2_execution"))
import p2_execution_adapter as original
import score_paid_outputs as paid
from collect_native_exports import load_auditor


COHORT = "e24_training_deployment_replication"
SCOPE_SCHEMA = "e24_independent_execution_scope_v1"
APPROVAL_SCHEMA = "e24_independent_execution_approval_v1"
CLOSED_SCHEMA = "e24_native_outputs_frozen_v1"
LOCAL_APPROVAL_SCHEMA = "e24_local_scoring_approval_v1"
NAMES = ("B_U", "B_P", "C_U", "C_P")
EXPECTED_COUNTS = {"member_generation": 20661, "aggregation": 53661}
INPUT_SHA = "16c65d5f97c4bca3f3db66b3ac47944632034d84c699d9f6486bcee31bd5b435"
PREDICTION_SHA = "9595ee1ddf9e30e85a0c95e51af2155e6b3de4cbd36265114341f32f9c324cef"
PROTOCOL_SHA = "6c30f10b6ac417e04b33170d8088230dc07d3c4893b65f2a51283ab216dca8f1"


def require_fields(record, expected, description):
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(description)


def policy_support(predictions, scope):
    if sha(predictions) != PREDICTION_SHA:
        raise ValueError("Predictions differ from the published E24 freeze")
    with np.load(predictions, allow_pickle=False) as packet:
        if tuple(packet["names"]) != NAMES or packet["probabilities"].shape != (4, 1196, 20):
            raise ValueError("Expected exactly the four deployed policies over 1,196 original questions")
        question_ids = packet["question_ids"].tolist()
        identities = list(zip(question_ids, packet["canonical_question_sha256"].tolist(),
                              packet["task_labels"].tolist(), packet["subtask_labels"].tolist()))
        metadata = [(row["question_id"], row["canonical_question_sha256"], row["task"], row["subtask"] or "")
                    for row in scope["question_keys"]]
        if metadata != identities or len(set(question_ids)) != len(question_ids):
            raise ValueError("Execution question identities differ from the frozen prediction frame")
        packet_probabilities = packet["probabilities"]
        for index, name in enumerate(NAMES):
            if not np.array_equal(np.array(scope["frozen_policy_probabilities"][name]), packet_probabilities[index]):
                raise ValueError("Execution policy probabilities differ from the frozen policy")
        support = np.any(packet_probabilities > 0, axis=0)
    aggregate = {(question, seed, int(action)) for question, row in zip(question_ids, support)
                 for seed in original.GENERATION_SEEDS for action in np.flatnonzero(row)}
    member = {(question, seed, model) for question, seed, action in aggregate for model in original.TEAMS[action]}
    return {"member_generation": member, "aggregation": aggregate}


def evidence_gate(args):
    scope = read_json(args.scope)
    scope_sha = original.canonical_digest(scope)
    bindings = scope["bindings"]
    require_fields(scope, dict(schema=SCOPE_SCHEMA, role="execution", cohort=COHORT,
                   transport_split=original.TRANSPORT_SPLIT, operator_fingerprint=original.OPERATOR_FP,
                   aggregation_seed=42, generation_seeds=list(original.GENERATION_SEEDS),
                   execution_limits=EXPECTED_COUNTS, cached_member_dependency_keys=[]),
                   "The closed scorer requires the independent E24 execution scope")
    require_fields(bindings, dict(protocol_sha256=PROTOCOL_SHA, target_inputs_sha256=INPUT_SHA,
                   frozen_actions_sha256=PREDICTION_SHA, reference_plan_sha256=sha(args.reference_plan)),
                   "Execution scope is disconnected from E24 protocol, input or reference freeze")
    expected = policy_support(args.predictions, scope)
    for stage, required, callable_keys in (("member_generation", "required_member_keys", "member_execution_keys"),
                                          ("aggregation", "required_aggregate_keys", "aggregate_execution_keys")):
        rows = list(map(tuple, scope[required]))
        if rows != list(map(tuple, scope[callable_keys])) or len(rows) != EXPECTED_COUNTS[stage] or set(rows) != expected[stage]:
            raise ValueError("Closed scoring supports only the exact authorized execution union")

    approval = read_json(args.approval)
    approval_sha = sha(args.approval)
    require_fields(approval, dict(schema=APPROVAL_SCHEMA, status="APPROVED", cohort=COHORT, bindings=bindings,
                   software_test_only=False, execution_limits=EXPECTED_COUNTS),
                   "Independent E24 production execution approval is required")
    if (approval.get("software_test_only") is not False or approval.get("sealed200_answer_access") is not False
            or approval.get("scope_sha256_by_role", {}).get("execution") != scope_sha
            or not set(EXPECTED_COUNTS) <= set(approval.get("operations", []))
            or not approval.get("approved_output_root")):
        raise PermissionError("Execution approval did not bind this production scope")
    frozen = read_json(args.outputs_frozen)
    require_fields(frozen, dict(schema=CLOSED_SCHEMA, status="ALL_REQUIRED_OUTPUTS_FROZEN", cohort=COHORT,
                   scope_sha256=sha(args.scope), scope_canonical_sha256=scope_sha, approval_sha256=approval_sha,
                   scope_bindings=bindings, completed_member_calls=EXPECTED_COUNTS["member_generation"],
                   completed_aggregate_calls=EXPECTED_COUNTS["aggregation"], unknown_or_failed_attempts=0,
                   new_gold_values_read=0), "All 74,322 native tasks must be closed before scoring")
    if not original.valid_sha(frozen.get("execution_plan_sha256")):
        raise ValueError("Closed native outputs must bind the execution plan")
    paths = {}
    filenames = set()
    for stage, shards in frozen["records"].items():
        if stage not in expected or not isinstance(shards, list) or not shards:
            raise ValueError("Complete per-stage native receipt export shards required")
        paths[stage] = []
        for shard in shards:
            name = shard["file"]
            if (Path(name).name != name or not name.endswith(".jsonl.gz") or name in filenames
                    or type(shard.get("records")) is not int or shard["records"] <= 0):
                raise ValueError("Expected unique gzip JSONL shard basenames and recorded row counts")
            filenames.add(name)
            path = args.outputs_frozen.parent / name
            if path.is_symlink() or path.stat().st_size != shard["bytes"] or sha(path) != shard["sha256"]:
                raise ValueError("Native receipt shard bytes differ from the closed export")
            paths[stage].append((path, shard["records"]))
        if sum(count for _, count in paths[stage]) != EXPECTED_COUNTS[stage]:
            raise ValueError("Native receipt shard counts differ from the exact E24 union")
    if set(paths) != set(expected):
        raise ValueError("Both member and aggregation export stages are required")
    return scope, approval, frozen, paths, expected


def local_reference_gate(args, scope, plan):
    local = read_json(args.local_scoring_approval)
    require_fields(local, dict(schema=LOCAL_APPROVAL_SCHEMA, status="APPROVED", cohort=COHORT,
                   scope_canonical_sha256=original.canonical_digest(scope),
                   outputs_frozen_sha256=sha(args.outputs_frozen),
                   reference_plan_sha256=sha(args.reference_plan),
                   prediction_sha256=PREDICTION_SHA, approved_reference_questions=len(scope["question_keys"]),
                   operations=["read_reference", "score_outputs"], sealed_reference_allowed=False),
                   "Local authorization must identify this exact E24 closure and unsealed references")
    require_fields(plan, dict(schema=LOCATOR_SCHEMA, status="INPUT_LOCATORS_ONLY_ANSWERS_NOT_READ",
                   protocol_sha256=PROTOCOL_SHA, input_sha256=INPUT_SHA, reference_version=REFERENCE_VERSION,
                   input_source_positions_verified=len(scope["question_keys"]),
                   answer_values_decoded=0, sealed_answer_values_decoded=0),
                   "Reference positions must be frozen input-only official source metadata")


def profiles(assets):
    audit, _ = load_auditor()
    members = {}
    for row in read_json(assets / "configs/pool_v1.json")["models"]:
        profile, fingerprint = audit.model_fingerprint(read_json(assets / row["config"]))
        if fingerprint != row["fingerprint"]:
            raise ValueError("Original frozen member profile differs")
        members[row["index"]] = profile, fingerprint, row["alias"]
    if set(members) != set(range(6)):
        raise ValueError("Six frozen member profiles required")
    config = assets / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"
    aggregator, fingerprint = audit.model_fingerprint(read_json(config))
    if fingerprint != original.AGGREGATOR_MODEL_FP:
        raise ValueError("Original V2 aggregator profile differs")
    return members, aggregator


def native_rows(path, expected_rows):
    rows = 0
    with gzip.open(path, mode="rb") as source:
        for raw in source:
            if not raw.endswith(b"\n") or not raw.strip():
                raise ValueError("Native JSONL rows must be nonempty and newline terminated")
            rows += 1
            yield json.loads(raw)
    if rows != expected_rows:
        raise ValueError("Native shard row count differs from export closure")


def verify_identity(receipt, stage, namespace, expected, metadata, seen, attempts):
    require_fields(receipt, dict(schema="p2_execution_task_receipt_v1", cohort=COHORT, stage=stage, status="SUCCESS",
                   software_test_only=False, from_cache=False, aggregator_fingerprint=original.OPERATOR_FP,
                   transport_split=original.TRANSPORT_SPLIT, aggregation_seed=42, **namespace),
                   "Native record lies outside the exact approved production execution scope")
    task_key = receipt["task_key"]
    index_field = "member_index" if stage == "member_generation" else "combination_index"
    key = (task_key["question_id"], task_key["generation_seed"], task_key[index_field])
    if type(key[1]) is not int or type(key[2]) is not int or key not in expected or key in seen:
        raise ValueError("Unexpected or duplicate native task key")
    native = receipt["native_record"]
    question = metadata[key[0]]
    digest = original.canonical_digest({name: value for name, value in native.items() if name != "record_sha256"})
    attempt = native.get("attempt_id")
    if (not isinstance(attempt, str) or not attempt or attempt in attempts
            or task_key.get("canonical_id") != question["canonical_question_sha256"]
            or receipt.get("native_record_sha256") != digest or receipt.get("physical_attempt_id") != attempt):
        raise ValueError("Native record digest, canonical question or physical attempt differs")
    require_fields(native, dict(record_sha256=digest, status="success", cost_status="recorded_actual_usage",
                   job_id=receipt["native_job_id"], question_id=key[0], task=question["task"], split=original.TRANSPORT_SPLIT,
                   prompt_version=receipt["prompt_version"]), "Native row identity or cost settlement differs")
    if (native.get("subtask") or "") != (question.get("subtask") or ""):
        raise ValueError("Native row subtask differs from the frozen input")
    seen.add(key)
    attempts.add(attempt)
    return key, native


def verify_result(result, profile, seed, totals):
    require_fields(result, dict(seed=seed, model_id=profile["model_id"], model_revision=profile["revision"],
                   backend=profile["backend"], model_identity_verified=True), "Served model or generation seed differs")
    if result.get("model_identity_verified") is not True or not isinstance(result.get("text"), str) or not result["text"].strip():
        raise ValueError("Native result lacks its verified nonempty completion")
    for field, cap in (("prompt_tokens", "max_input_tokens"), ("completion_tokens", "max_new_tokens")):
        value = result.get(field)
        if type(value) is not int or not 0 <= value <= profile[cap]:
            raise ValueError("Measured native token count is missing or outside the frozen operator bound")
        totals[field] += value
    wall = result.get("wall_time_s")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Measured native call wall time is missing or invalid")
    totals["call_wall_seconds_sum"] += wall


def verify_receipts(args, scope, approval, frozen, paths, expected, native_api, destination):
    member_profiles, aggregate_profile = profiles(args.assets)
    metadata = {row["question_id"]: row for row in scope["question_keys"]}
    namespace = dict(scope_sha256=original.canonical_digest(scope), approval_sha256=sha(args.approval),
                     bindings=scope["bindings"], approved_output_root=approval["approved_output_root"])
    dependencies, attempts, costs = {}, set(), {}
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        for stage in EXPECTED_COUNTS:
            seen = set()
            totals = dict(prompt_tokens=0, completion_tokens=0, call_wall_seconds_sum=0.0)
            for path, count in paths[stage]:
                for receipt in native_rows(path, count):
                    key, native = verify_identity(receipt, stage, namespace, expected[stage], metadata, seen, attempts)
                    question, seed, index = key
                    result = native["result"]
                    input_digest = metadata[question]["question_sha256"]
                    if stage == "member_generation":
                        profile, fingerprint, alias = member_profiles[index]
                        require_fields(native, dict(seed=seed, model_fingerprint=fingerprint, model_alias=alias,
                            source_question_sha256=input_digest, sample_index=0, sampling_control=False,
                            prompt_version="independent_user_only_brief_final_v2"), "Original member inference contract differs")
                        verify_result(result, profile, seed, totals)
                        dependencies[key] = dict(job_id=native["job_id"], model_fingerprint=fingerprint,
                                                 model_id=profile["model_id"], revision=profile["revision"],
                                                 record_sha256=native["record_sha256"])
                    else:
                        verify_result(result, aggregate_profile, 42, totals)
                        evidence = native_api.verified_aggregation_source(dict(receipt,
                            schema="e18_execution_task_receipt_v2", cohort="e18_confirm"))
                        provenance = native["member_provenance"]
                        if (native["source_hashes"]["question_sha256"] != input_digest
                                or any(provenance[member_profiles[member][2]] != dependencies[question, seed, member]
                                       for member in original.TEAMS[index])):
                            raise ValueError("Native aggregation does not use its exact same-seed member outputs")
                        output.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            if seen != expected[stage]:
                raise ValueError("The exact closed E24 task union must be complete")
            costs[stage] = dict(calls=len(seen), **totals, total_tokens=totals["prompt_tokens"] + totals["completion_tokens"])
    tokens = {name: sum(stage[name] for stage in costs.values())
              for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    stage_tokens = {stage: {name: item[name] for name in token_fields} for stage, item in costs.items()}
    if (len(attempts) != sum(EXPECTED_COUNTS.values()) or tokens != frozen["actual_tokens"]
            or stage_tokens != frozen["stage_tokens"]):
        raise ValueError("Native unique attempts and measured token sums disagree with the closed export")
    return dict(schema="e24_native_verified_before_gold_v1", status="ALL_REQUIRED_NATIVE_OUTPUTS_VERIFIED",
                outputs_frozen_sha256=sha(args.outputs_frozen), approved_execution_scope_sha256=namespace["scope_sha256"],
                native_attempts=len(attempts), actual_tokens=tokens, by_stage=costs, reference_values_decoded=0,
                sealed_answer_values_decoded=0, compact_verified_outputs_sha256=sha(destination))


def score_references(plan, scope, verified, output_dir, reference_api, projection, scorer):
    score_path = output_dir / "PAID_SUPPORT_SCORES.jsonl"
    normalized, unscorable, score_count = {}, {}, 0
    start = time.process_time()
    with (output_dir / "REFERENCE_ACCESS_EVENTS.jsonl").open("x", encoding="utf-8", newline="\n") as events, \
            verified.open("r", encoding="utf-8") as native_rows_stream, \
            score_path.open("x", encoding="utf-8", newline="\n") as output:
        provider = paid.ApprovedReferences(plan, scope, reference_api, projection, events)
        for raw in native_rows_stream:
            row = json.loads(raw)
            question = row["question_id"]
            if question not in normalized and question not in unscorable:
                try:
                    normalized[question] = paid.canonical_reference(provider.get(question), scorer)
                except paid.UnscorableReference as error:
                    metadata = provider.records[question]
                    unscorable[question] = dict(task=metadata["task"], subtask=metadata["subtask"], reason=str(error))
            if question in unscorable:
                continue
            reference = normalized[question]
            scored = scorer.score_answer(reference["task"], row["prediction"], reference["reference"],
                                         subtask=reference.get("subtask"), choices=reference.get("choices"))
            if scored.scorer_version != REFERENCE_VERSION or scored.score not in (0, 1):
                raise ValueError("Original frozen scorer must produce binary correctness under the pinned version")
            output.write(json.dumps(dict(question_id=question, generation_seed=row["generation_seed"],
                combination_index=row["combination_index"], score=float(scored.score), parsed=bool(scored.parsed),
                reason=scored.reason, reference_version=REFERENCE_VERSION,
                native_record_sha256=row["aggregation_record_sha256"]), ensure_ascii=False) + "\n")
            score_count += 1
    if len(normalized) + len(unscorable) != len(scope["question_keys"]) or len(provider.accounts) != provider.references_decoded:
        raise ValueError("Reference access accounting differs from the exact independent cohort")
    if unscorable:
        write_json(output_dir / "UNSCORABLE_APPROVED_QUESTIONS_PRIVATE.json", unscorable)
    return score_path, dict(reference_values_decoded=provider.references_decoded, unique_gold_accounts=len(provider.accounts),
                           reference_questions_scored=len(normalized), unscorable_questions=len(unscorable),
                           rows_scored=score_count, new_llm_calls=0, sealed_answer_values_decoded=0,
                           scoring_cpu_seconds=time.process_time()-start)


def run(args):
    scope, approval, frozen, paths, expected = evidence_gate(args)
    plan = read_json(args.reference_plan)
    if args.execute_scoring:
        if args.local_scoring_approval is None:
            raise PermissionError("Local reference access requires its hash-bound approval receipt")
        local_reference_gate(args, scope, plan)
    reference_api, projection, native_api, scorer = paid.load_frozen_apis()
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "VERIFIED_NATIVE_AGGREGATIONS.jsonl"
    evidence = verify_receipts(args, scope, approval, frozen, paths, expected, native_api, destination)
    native_receipt = args.output / "NATIVE_OUTPUTS_VERIFIED_BEFORE_GOLD.json"
    write_json(native_receipt, evidence)
    if not args.execute_scoring:
        return evidence
    score_path, accounting = score_references(plan, scope, destination, args.output, reference_api, projection, scorer)
    write_json(args.output / "REFERENCE_AND_SCORING_COSTS.json", accounting)
    if accounting["unscorable_questions"] or accounting["rows_scored"] != EXPECTED_COUNTS["aggregation"]:
        return dict(status="PRIMARY_CLOSURE_BLOCKED_BY_UNSCORABLE_REFERENCE", **accounting)
    receipt = dict(schema="e24_closed_native_scoring_v1", status="ALL_FROZEN_POLICIES_FULLY_SCORED",
                   prediction_sha256=PREDICTION_SHA, score_sha256=sha(score_path), aggregation_fingerprint=original.OPERATOR_FP,
                   native_verification_sha256=sha(native_receipt), reference_plan_sha256=sha(args.reference_plan),
                   scoring_approval_sha256=sha(args.local_scoring_approval), scorer_version=REFERENCE_VERSION, **accounting)
    write_json(args.output / "CLOSED_NATIVE_SCORING_RECEIPT.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "scope", "approval", "predictions", "reference-plan", "outputs-frozen", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--execute-scoring", action="store_true")
    parser.add_argument("--local-scoring-approval", type=Path)
    args = parser.parse_args()
    import os
    os.environ["MAB_ASSET_ROOT"] = str(args.assets)
    paid.ASSETS = args.assets
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
