"""Check a single real P2 aggregate receipt without opening reference answers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import blind_evaluation as blind
import p2_execution_adapter as execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = blind.read_json(args.scope)
    receipt = blind.read_json(args.receipt)
    approval_ref = dict(path=str(args.approval),sha256=blind.file_sha(args.approval))
    execution.validate_scope_approval(approval_ref,scope,"aggregation")
    task = receipt["task_key"]
    key = (task["question_id"],task["generation_seed"],task["combination_index"])
    expected = {row["question_id"]:row["canonical_question_sha256"] for row in scope["question_keys"]}
    checks = dict(
        p2_production_receipt=receipt.get("schema")=="p2_execution_task_receipt_v1" and receipt.get("cohort")==execution.COHORT and receipt.get("software_test_only") is False,
        approved_runtime_scope=receipt.get("scope_sha256")==execution.canonical_digest(scope),
        approved_source_bindings=receipt.get("bindings")==scope["bindings"],
        approval_identity=receipt.get("approval_sha256")==approval_ref["sha256"],
        action_in_required_union=key in set(map(tuple,scope["required_aggregate_keys"])),
        action_in_paid_execution_scope=key in set(map(tuple,scope["aggregate_execution_keys"])),
        canonical_question_identity=expected.get(key[0])==task["canonical_id"],
    )
    if not all(checks.values()):
        raise AssertionError(checks)
    prep = Path(__file__).resolve().parent.parent/"e18_mechanism_and_confirmation_preparation"
    sys.path.insert(0,str(prep))
    from e18_scoring_adapter import verified_aggregation_source
    legacy_view = dict(receipt,schema="e18_execution_task_receipt_v2",cohort="e18_confirm")
    source = verified_aggregation_source(legacy_view)
    checks.update(original_native_payload_and_hash_verified=True,
        operator_model_prompt_members_and_seed_verified=True,
        source_model_identity_verified=True,measured_prompt_and_completion_tokens=True,
        nonempty_aggregate_output=True)
    report = dict(schema="p2_first_real_aggregation_semantic_qualification_v1",status="PASS",
        checks=checks,receipt_file_sha256=blind.file_sha(args.receipt),
        runtime_scope_file_sha256=blind.file_sha(args.scope),
        runtime_scope_canonical_sha256=execution.canonical_digest(scope),
        approval_file_sha256=approval_ref["sha256"],
        verification_source_sha256=blind.file_sha(prep/"e18_scoring_adapter.py"),
        semantic_namespace_adapter="Only outer receipt schema and cohort aliases; original paid native record unchanged",
        aggregation_operator_fingerprint=blind.OPERATOR_SHA256,
        member_generation_seed=source["generation_seed"],aggregation_seed=42,
        prompt_tokens=source["prompt_tokens"],completion_tokens=source["completion_tokens"],
        total_tokens=source["tokens"],source_wall_seconds=source["source_wall_time_s"],
        private_question_or_answer_text_published=False,reference_values_read=0,
        deterministic_scores_computed=0,new_llm_calls=0,
        interpretation="Single paid execution compatibility check; complete-output scoring gate remains required")
    blind.write_json(args.output,report)
    print(json.dumps({"status":"PASS","checks":len(checks),"prompt_tokens":source["prompt_tokens"],
        "completion_tokens":source["completion_tokens"],"reference_values_read":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
