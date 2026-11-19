"""Exercise E24's closure and scoring on synthetic official-format records only."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import score_closed_execution as scoring


def receipt(stage, key, metadata, namespace, profile, fingerprint, alias, provenance=None):
    qid, seed, index = key
    member = stage == "member_generation"
    task_key = dict(question_id=qid, generation_seed=seed, canonical_id=metadata["canonical_question_sha256"])
    if member:
        task_key["member_index"] = index
    else:
        task_key.update(combination_index=index, team=list(scoring.original.TEAMS[index]))
    slot = f"{stage}_{seed}_{index}"
    prompt = "independent_user_only_brief_final_v2" if member else "anonymous_final_answer_typed_v3"
    native_seed = seed if member else 42
    result = dict(text="Final answer: 7", seed=native_seed, model_id=profile["model_id"],
                  model_revision=profile["revision"], backend=profile["backend"], model_identity_verified=True,
                  prompt_tokens=13, completion_tokens=5, wall_time_s=0.1)
    native = dict(question_id=qid, task=metadata["task"], subtask=None, seed=native_seed,
                  split=scoring.original.TRANSPORT_SPLIT, prompt_version=prompt, cost_status="recorded_actual_usage",
                  job_id="job_" + slot, attempt_id="paid_" + slot, status="success", result=result,
                  model_fingerprint=fingerprint)
    if member:
        native.update(model_alias=alias, source_question_sha256=metadata["question_sha256"], sample_index=0, sampling_control=False)
    else:
        native.update(members=sorted(provenance), member_provenance=provenance,
                      source_hashes=dict(question_sha256=metadata["question_sha256"]))
    native["record_sha256"] = scoring.original.canonical_digest(native)
    return dict(schema="p2_execution_task_receipt_v1", status="SUCCESS", cohort=scoring.COHORT,
                stage=stage, task_key=task_key, aggregator_fingerprint=scoring.original.OPERATOR_FP,
                aggregation_seed=42, transport_split=scoring.original.TRANSPORT_SPLIT, from_cache=False,
                software_test_only=False, prompt_version=prompt, native_record_sha256=native["record_sha256"],
                native_job_id=native["job_id"], physical_attempt_id=native["attempt_id"], native_record=native, **namespace)


def write_shard(path, receipts):
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for item in receipts:
            stream.write(json.dumps(item) + "\n")
    return dict(file=path.name, sha256=scoring.sha(path), bytes=path.stat().st_size, records=len(receipts))


def qualified(args, native_api, reference_api, projection, scorer):
    text = "Synthetic E24 closure qualification question."
    canonical = scoring.paid.canonical_input(text)
    metadata = dict(question_id="synthetic-approved", canonical_question_sha256=canonical,
                    question_sha256=hashlib.sha256(text.encode()).hexdigest(), task="gsm8k", subtask="")
    qid = metadata["question_id"]
    with tempfile.TemporaryDirectory(prefix="e24-closure-toy-") as temporary:
        root = Path(temporary)
        scope_path = root / "SCOPE.json"
        approval_path = root / "APPROVAL.json"
        reference_path = root / "PLAN.json"
        frozen_path = root / "OUTPUTS_FROZEN.json"
        local_path = root / "LOCAL_APPROVAL.json"
        source = root / "official_fixture.jsonl"
        source.write_text(json.dumps({"question":text,"answer":"fixture #### 7"})+"\n", encoding="utf-8")
        loc = dict(question_id=qid, canonical_question_sha256=canonical, task="gsm8k", subtask=None,
                   source_split="train", source_id="synthetic", row_index=0)
        plan = dict(schema=scoring.LOCATOR_SCHEMA, status="INPUT_LOCATORS_ONLY_ANSWERS_NOT_READ",
                    protocol_sha256=scoring.PROTOCOL_SHA, input_sha256=scoring.INPUT_SHA, reference_version=scoring.REFERENCE_VERSION,
                    input_source_positions_verified=1, answer_values_decoded=0, sealed_answer_values_decoded=0,
                    sources=[dict(source_id="synthetic", path=str(source), sha256=scoring.sha(source),
                        format="gsm8k_official_jsonl_v1", task="gsm8k", subtask=None, source_split="train")],
                    records=[loc], sealed_exclusions=[dict(question_id="synthetic-excluded",canonical_question_sha256="f"*64)])
        scoring.write_json(reference_path, plan)
        bindings = dict(protocol_sha256=scoring.PROTOCOL_SHA, target_inputs_sha256=scoring.INPUT_SHA,
                        frozen_actions_sha256=scoring.PREDICTION_SHA, reference_plan_sha256=scoring.sha(reference_path))
        expected = {"member_generation": {(qid, 42, member) for member in (0, 1, 2)}, "aggregation": {(qid, 42, 0)}}
        counts = {stage: len(items) for stage, items in expected.items()}
        member_keys, aggregate_keys = sorted(expected["member_generation"]), sorted(expected["aggregation"])
        scope = dict(schema=scoring.SCOPE_SCHEMA, role="execution", cohort=scoring.COHORT, bindings=bindings,
                     question_keys=[metadata], transport_split=scoring.original.TRANSPORT_SPLIT,
                     operator_fingerprint=scoring.original.OPERATOR_FP, generation_seeds=list(scoring.original.GENERATION_SEEDS),
                     aggregation_seed=42, execution_limits=counts, cached_member_dependency_keys=[], required_member_keys=member_keys,
                     member_execution_keys=member_keys, required_aggregate_keys=aggregate_keys, aggregate_execution_keys=aggregate_keys)
        scoring.write_json(scope_path, scope)
        scope_sha = scoring.original.canonical_digest(scope)
        approval = dict(schema=scoring.APPROVAL_SCHEMA, status="APPROVED", cohort=scoring.COHORT, bindings=bindings,
                        software_test_only=False, execution_limits=counts, operations=list(counts), sealed200_answer_access=False,
                        scope_sha256_by_role=dict(execution=scope_sha), approved_output_root=str(root / "owned"))
        scoring.write_json(approval_path, approval)
        approval_sha = scoring.sha(approval_path)
        namespace = dict(scope_sha256=scope_sha, approval_sha256=approval_sha, bindings=bindings,
                         approved_output_root=approval["approved_output_root"])
        profiles, aggregate_profile = scoring.profiles(args.assets)
        members = [receipt("member_generation", key, metadata, namespace, *profiles[key[2]]) for key in member_keys]
        provenance = {}
        for item in members:
            native = item["native_record"]
            provenance[native["model_alias"]] = dict(job_id=native["job_id"], record_sha256=native["record_sha256"],
                model_id=native["result"]["model_id"], revision=native["result"]["model_revision"], model_fingerprint=native["model_fingerprint"])
        aggregate = receipt("aggregation", aggregate_keys[0], metadata, namespace, aggregate_profile,
                            scoring.original.AGGREGATOR_MODEL_FP, "", provenance)
        stages = {"member_generation": members, "aggregation": [aggregate]}
        shards = {stage: [write_shard(root / f"{stage}.jsonl.gz", rows)] for stage, rows in stages.items()}
        closed = dict(schema=scoring.CLOSED_SCHEMA,status="ALL_REQUIRED_OUTPUTS_FROZEN",cohort=scoring.COHORT,
                      scope_sha256=scoring.sha(scope_path),scope_canonical_sha256=scope_sha,approval_sha256=approval_sha,
                      scope_bindings=bindings,execution_plan_sha256="e"*64,completed_member_calls=3,completed_aggregate_calls=1,
                      actual_tokens=dict(prompt_tokens=52,completion_tokens=20,total_tokens=72),
                      stage_tokens={"member_generation":dict(prompt_tokens=39,completion_tokens=15,total_tokens=54),
                                    "aggregation":dict(prompt_tokens=13,completion_tokens=5,total_tokens=18)},
                      unknown_or_failed_attempts=0,
                      new_gold_values_read=0,records=shards)
        scoring.write_json(frozen_path, closed)
        qualification = SimpleNamespace(assets=args.assets, predictions=root/"unused.npz", scope=scope_path, approval=approval_path,
            reference_plan=reference_path, outputs_frozen=frozen_path, local_scoring_approval=local_path)

        def rejects(call):
            try:
                call()
            except (ValueError, PermissionError, KeyError, FileNotFoundError):
                return True
            return False

        checks = {}
        with patch.object(scoring,"EXPECTED_COUNTS",counts), patch.object(scoring,"policy_support",return_value=expected):
            loaded_scope, loaded_approval, freeze, paths, required = scoring.evidence_gate(qualification)
            checks["missing_local_reference_authorization_rejected"] = rejects(lambda: scoring.local_reference_gate(qualification,scope,plan))
            local = dict(schema=scoring.LOCAL_APPROVAL_SCHEMA,status="APPROVED",cohort=scoring.COHORT,
                         scope_canonical_sha256=scope_sha,outputs_frozen_sha256=scoring.sha(frozen_path),
                         reference_plan_sha256=scoring.sha(reference_path),prediction_sha256=scoring.PREDICTION_SHA,
                         approved_reference_questions=1,operations=["read_reference","score_outputs"],sealed_reference_allowed=False)
            scoring.write_json(local_path,local)
            scoring.local_reference_gate(qualification,loaded_scope,plan)
            verified = root / "VERIFIED.jsonl"
            native = scoring.verify_receipts(qualification, loaded_scope, loaded_approval, freeze, paths, required, native_api, verified)
            checks["all_four_native_tries_and_exact_measured_tokens_verified"] = native["native_attempts"] == 4 and native["actual_tokens"]["total_tokens"] == 72
            checks["no_reference_decoded_by_native_gate"] = native["reference_values_decoded"] == 0
            inconsistent = {**freeze, "stage_tokens": {**freeze["stage_tokens"],
                            "aggregation": dict(prompt_tokens=12, completion_tokens=6, total_tokens=18)}}
            checks["inconsistent_per_stage_token_cost_rejected"] = rejects(lambda: scoring.verify_receipts(
                qualification, loaded_scope, loaded_approval, inconsistent, paths, required, native_api, root / "bad_cost.jsonl"))
            score_path, costs = scoring.score_references(plan, scope, verified,root,reference_api,projection,scorer)
            checks["pinned_binary_scorer_and_single_reference_account"] = json.loads(score_path.read_text(encoding="utf-8"))["score"] == 1 and costs["unique_gold_accounts"] == 1
            excluded = {**plan,"sealed_exclusions":[dict(question_id=qid,canonical_question_sha256=canonical)]}
            subfolder = root / "sealed_attempt"
            subfolder.mkdir()
            checks["sealed_overlap_rejected"] = rejects(lambda: scoring.score_references(excluded,scope,verified,subfolder,reference_api,projection,scorer))
            broken = root / "aggregation_bad_provenance.jsonl.gz"
            incorrect_provenance = {name: dict(row) for name,row in provenance.items()}
            incorrect_provenance[next(iter(incorrect_provenance))]["record_sha256"] = "0"*64
            bad_aggregate = receipt("aggregation", aggregate_keys[0], metadata, namespace, aggregate_profile,
                                    scoring.original.AGGREGATOR_MODEL_FP,"",incorrect_provenance)
            write_shard(broken, [bad_aggregate])
            altered_paths = {**paths,"aggregation":[(broken,1)]}
            checks["wrong_same_seed_member_provenance_rejected"] = rejects(lambda: scoring.verify_receipts(
                qualification, loaded_scope, loaded_approval, freeze, altered_paths, required, native_api, root/"bad_output.jsonl"))
            with (root/shards["aggregation"][0]["file"]).open("ab") as stream:
                stream.write(b"changed")
            checks["altered_gzip_export_rejected_before_reference_access"] = rejects(lambda: scoring.evidence_gate(qualification))
        if not all(checks.values()):
            raise AssertionError(checks)
    return dict(schema="e24_closed_scoring_qualification_v1",status="PASS",checks=checks,
                frozen_scorer_version=scorer.SCORER_VERSION,synthetic_references_decoded=1,real_reference_values_decoded=0,
                sealed_answer_values_decoded=0,new_llm_calls=0,scoring_source_sha256=scoring.sha(Path(scoring.__file__)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    scoring.paid.ASSETS=args.assets
    reference_api, projection, native_api, scorer = scoring.paid.load_frozen_apis()
    report=qualified(args,native_api,reference_api,projection,scorer)
    scoring.write_json(args.output,report)
    print(json.dumps({"status":report["status"],"checks":len(report["checks"]),"real_reference_values_decoded":0}))


if __name__ == "__main__":
    main()
