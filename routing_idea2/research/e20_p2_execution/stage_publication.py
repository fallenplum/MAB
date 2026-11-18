"""Stage P2 source, aggregate receipts and live activation status for publication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone
import uuid


CODE = Path(__file__).resolve().parent
WORKSPACE = CODE.parents[1]
STUDY = Path(r"E:\语言模型组合路由\research\e20_p2_execution")
STAGE = WORKSPACE / ".safe_e20_p2_delivery"
PREFIX = "routing_idea2/research/e20_p2_execution/"
DEFAULT_PARENT = "f8661bba7cba6e9a93d6bafcaef15c77404e8140"
SOURCE_ALLOWLIST = {
    "prepare_independent_pool.py", "p2_scope_freeze.py", "p2_execution_adapter.py",
    "p2_member_worker.py", "p2_aggregation_worker.py", "prepare_runtime_sources.py",
    "qualify_blind_evaluation.py", "prepare_reference_locators.py", "blind_evaluation.py",
    "prepare_launch.py", "score_paid_outputs.py", "qualify_paid_scorer.py",
    "launch_native_profile.py", "p2_aggregation_follower.py", "monitor_native_status.py",
    "qualify_first_aggregate.py", "record_activation_status.py", "p2_export_native.py",
    "stage_publication.py", "migrate_aggregation_once.py", "collect_native_exports.py",
    "qualify_native_collector.py", "ENVIRONMENT_CLARIFICATION.md",
    "P2_DATA_AND_FROZEN_SCOPE.md", "P2_PREPARATION_SUMMARY.md",
    "P2_BLIND_EVALUATION_README.md", "README.md", "REPRODUCE.md",
    "RUNTIME_EXECUTION_README.md", "P2_RUNTIME_SCALING_REVIEW.md",
    "P2_NATIVE_COLLECTION_README.md", "requirements_features.txt",
    "p2_manifest_registry.py", "p2_aggregation_follower_v3.py", "p2_export_native_v3.py",
    "monitor_native_status_v3.py", "prepare_aggregation_shards_v3.py",
    "launch_registered_worker_v3.py", "qualify_registry_consumers.py",
    "qualify_registry_dispatch.py", "switch_aggregation_to_three_shards.py",
    "P2_V3_REGISTERED_DISPATCH_HANDOFF.md", "record_three_shard_activation.py",
    "prepare_aggregation_shards_v4.py", "launch_registered_worker_v4.py",
    "qualify_five_shard_dispatch.py", "scale_aggregation_to_five.py",
    "P2_FIVE_SHARD_HANDOFF.md", "record_five_shard_activation.py",
    "prepare_member_shards.py", "launch_registered_member.py", "stop_registered_workers.py",
    "qualify_member_parallel.py", "P2_MEMBER_PARALLEL_HANDOFF.md",
    "record_runtime_fill_activation.py", "P2_FINAL_RESULTS.md",
    "summarize_final_diagnostics.py", "audit_final_execution.py",
}
JSON_ALLOWLIST = {
    "P2_PRESELECTION_PROTOCOL.json", "P2_INPUT_SELECTION_FROZEN.json",
    "P2_INPUT_ADMISSION_REPORT.json", "P2_ENCODER_ENVIRONMENT.json",
    "P2_ENCODER_RECEIPT.json", "P2_FROZEN_EXECUTION_UNION.json",
    "P2_REAL_SCOPE_QUALIFICATION.json", "P2_REFERENCE_LOCATOR_QUALIFICATION.json",
    "P2_REAL_REFERENCE_PLAN_QUALIFICATION.json", "P2_REAL_RUNTIME_SCOPE_QUALIFICATION.json",
    "P2_BLIND_AND_SCOPE_QUALIFICATION.json", "P2_PAID_SCORER_QUALIFICATION.json",
    "P2_RUNTIME_SOURCE_ADAPTATION.json", "P2_FIRST_REAL_AGGREGATION_QUALIFICATION.json",
    "P2_ACTIVATION_STATUS.json",
    "P2_FIRST_NATIVE_EXPORT_QUALIFICATION.json",
    "P2_NATIVE_COLLECTOR_SOFTWARE_QUALIFICATION.json",
    "P2_NATIVE_COLLECTION_REAL_PARTIAL.json",
    "P2_SINGLE_AGGREGATION_MIGRATION.json", "P2_RUNTIME_SCALING_REVIEW.json",
    "P2_REGISTRY_DISPATCH_QUALIFICATION.json", "P2_REGISTRY_CONSUMERS_QUALIFICATION.json",
    "P2_V3_TOOLING_QUALIFICATION.json", "P2_THREE_SHARD_ACTIVATION.json",
    "P2_FIVE_SHARD_TOOLING_QUALIFICATION.json", "P2_FIVE_SHARD_ACTIVATION.json",
    "P2_FIVE_SHARD_THROUGHPUT.json",
    "P2_MEMBER_PARALLEL_QUALIFICATION.json", "P2_RUNTIME_FILL_ACTIVATION.json",
    "P2_POINT_RESULTS.json", "P2_FINAL_DIAGNOSTICS.json", "P2_FINAL_EXECUTION_LEDGER.json",
}
NEW_JSON_FIELDS = {
    "P2_FINAL_EXECUTION_LEDGER.json": {
        "schema", "status", "scope_canonical_sha256", "stages",
        "all_required_successful", "closed_exporter_coverage_complete", "exporter_closure",
        "journal_failure_events", "unresolved_failure_events", "partial_journals",
        "unregistered_dispatcher_directories", "unregistered_dispatcher_classifications",
        "unresolved_unregistered_dispatcher_directories",
        "extra_export_index_keys", "duplicate_paid_success_keys",
        "successful_keys_without_original_paid_journal_evidence", "unique_paid_successful_attempts",
        "successful_call_tokens", "failure_closure_pending", "failed_or_unknown_native_attempts_audited",
        "all_attempt_total_tokens", "model_loading_planning_transfer_cpu_and_gpu_overhead",
        "native_output_payloads_read", "gold_values_read", "new_llm_calls", "audit_boundary",
    },
    "P2_FINAL_DIAGNOSTICS.json": {
        "schema", "requested_questions", "scope_sha256", "prediction_sha256",
        "by_task", "by_source_frame", "policy_summaries", "action_probability_summary",
        "mean_total_variation", "mean_question_centered_score_l2_change", "new_llm_calls",
    },
    "P2_NATIVE_COLLECTOR_SOFTWARE_QUALIFICATION.json": {
        "schema", "status", "checks", "collector_sha256", "synthetic_fixture_only",
        "real_native_archives_modified", "new_llm_calls", "reference_values_read",
        "deterministic_scores_computed",
    },
    "P2_NATIVE_COLLECTION_REAL_PARTIAL.json": {
        "schema", "status", "source_closed_archive_task_counts", "unique_successful_outputs",
        "member_successes", "aggregate_successes", "actual_success_prompt_tokens",
        "actual_success_completion_tokens", "actual_success_total_tokens",
        "repeated_import_new_outputs", "native_files_match_committed_offsets",
        "final_output_freeze_present", "failure_closure_pending", "all_attempt_total_tokens",
        "new_llm_calls", "reference_values_read", "deterministic_scores_computed", "collector_sha256",
    },
    "P2_SINGLE_AGGREGATION_MIGRATION.json": {
        "schema", "status", "snapshot_utc", "source_gpu", "target_gpu",
        "previous_completed", "new_unique_successes", "total_completed_at_verification",
        "first_new_actual_prompt_tokens", "first_new_actual_completion_tokens",
        "old_process_exited", "graceful_stop", "old_success_receipt_unchanged",
        "repeated_success_semantic_keys", "scope_or_approval_or_model_or_policy_changes",
        "frozen_scope_canonical_sha256", "manifest_sha256",
        "shared_utilization_override_explicit", "target_gpu_admission_free_mib",
        "target_gpu_admission_utilization_percent", "new_member_profiles_started",
        "new_aggregation_shards_created", "signals_to_foreign_processes",
        "reference_values_decoded", "receipts",
    },
    "P2_RUNTIME_SCALING_REVIEW.json": {
        "scope", "profile_rates", "aggregation", "global_member_increment",
        "planned_aggregate_to_member_ratio", "live_telemetry_obtained_by_this_review",
        "process_changes", "new_llm_calls", "confidence_intervals",
    },
    "P2_REGISTRY_DISPATCH_QUALIFICATION.json": {
        "schema", "status", "checks", "checks_passed", "software_test_only",
        "new_llm_calls", "reference_values_decoded", "ssh_operations",
        "production_process_changes", "production_deployed", "source_sha256",
    },
    "P2_REGISTRY_CONSUMERS_QUALIFICATION.json": {
        "schema", "status", "checks", "checks_passed", "software_test_only",
        "gold_reads", "new_llm_calls", "ssh_or_process_changes", "production_deployed", "source_sha256",
    },
    "P2_V3_TOOLING_QUALIFICATION.json": {
        "schema", "status", "registry_dispatch_checks_passed", "monitor_export_checks_passed",
        "total_checks_passed", "default_aggregate_shards", "explicit_supported_aggregate_shards",
        "original_scope_canonical_sha256", "final_model_sha256", "required_member_keys",
        "required_aggregate_keys", "new_scientific_execution_keys", "scope_model_or_generation_changes",
        "new_llm_calls_in_this_tooling_task", "reference_values_decoded", "ssh_operations",
        "production_process_changes", "production_deployed", "original_pinned_sources_match",
        "new_source_sha256", "qualification_reports", "handoff", "parent_production_checks_remaining",
    },
    "P2_THREE_SHARD_ACTIVATION.json": {
        "schema", "status", "snapshot_utc", "old_actor_stop_utc", "old_actor_pid",
        "old_actor_exited", "old_unique_completed_aggregations", "old_repeated_success_keys",
        "original_registered_manifests", "registered_manifests_after_add", "dispatch_plan_sha256",
        "disjoint_remaining_partition_verified", "remaining_assigned_aggregations",
        "required_aggregations", "aggregation_shards", "member_profiles", "active_actor_count_at_snapshot",
        "duplicate_success_rows_at_snapshot", "extra_paid_attempts_for_successful_keys_at_snapshot",
        "remote_successful_outputs_at_snapshot", "local_collection_at_its_own_snapshot",
        "software_checks_at_preparation", "preparation_qualification_reports_preserved",
        "preparation_qualification_sha256", "control_evidence_sha256",
        "science_scope_model_policy_or_generation_changes", "reference_values_read",
        "deterministic_scores_computed", "local_publication_new_llm_calls", "claims_scope",
    },
    "P2_FIVE_SHARD_TOOLING_QUALIFICATION.json": {
        "schema", "status", "checks", "checks_passed", "software_test_only",
        "production_deployed", "old_source_sha256", "source_sha256",
        "reference_values_decoded", "new_llm_calls", "ssh_operations", "production_process_changes",
    },
    "P2_FIVE_SHARD_ACTIVATION.json": {
        "schema", "status", "snapshot_utc", "user_requested_additional_gpu_indices",
        "old_actor_stop_utc", "retired_three_actor_count", "old_unique_completed_aggregations",
        "duplicate_paid_success_keys_at_checkpoint", "registered_manifests_before_add",
        "registered_manifests_after_add", "dispatch_plan_sha256", "disjoint_remaining_partition_verified",
        "remaining_assigned_aggregations", "required_aggregations", "aggregation_shards", "member_profiles",
        "active_actor_count_at_snapshot", "first_five_shard_successes", "duplicate_success_rows_at_snapshot",
        "extra_paid_attempts_for_successful_keys_at_snapshot", "remote_successful_outputs_at_snapshot",
        "software_checks_at_preparation", "preparation_qualification_report_preserved",
        "preparation_qualification_sha256", "control_evidence_sha256",
        "science_scope_model_policy_or_generation_changes", "reference_values_read",
        "deterministic_scores_computed", "local_publication_new_llm_calls", "failure_closure_pending",
        "all_attempt_total_tokens", "claims_scope",
    },
    "P2_FIVE_SHARD_THROUGHPUT.json": {
        "schema", "status", "before", "after", "observed_rate_ratio",
        "loading_and_graceful_transition_excluded_from_windows",
        "transition_stop_checkpoint_successful_aggregations",
        "startup_and_loading_seconds_retained_in_private_actor_logs",
        "all_five_new_workers_alive_at_after_end", "new_worker_stop_events_at_after_end",
        "additional_paid_attempts_for_successful_keys", "gold_values_read",
        "confidence_interval_computations", "interpretation",
    },
    "P2_MEMBER_PARALLEL_QUALIFICATION.json": {
        "schema", "status", "checks", "checks_passed", "software_test_only",
        "production_deployed", "frozen_source_sha256", "source_sha256",
        "ssh_operations", "production_process_changes", "new_llm_calls", "reference_values_decoded",
    },
    "P2_RUNTIME_FILL_ACTIVATION.json": {
        "schema", "status", "snapshot_utc", "member_index", "member_stop_utc", "aggregation_stop_utc",
        "historical_member4_successes", "historical_aggregation_successes", "member4_required_outputs",
        "required_aggregations", "member_plan_sha256", "aggregation_plan_sha256", "member4_shards",
        "aggregation_shards", "disjoint_remaining_partitions_verified", "registered_manifests_after_add",
        "active_actor_count_at_snapshot", "remote_successful_outputs_at_snapshot",
        "duplicate_success_rows_at_snapshot", "extra_paid_attempts_for_successful_keys_at_snapshot",
        "software_checks_at_preparation", "preparation_qualification_report_preserved",
        "preparation_qualification_sha256", "control_evidence_sha256",
        "science_scope_model_policy_or_generation_changes", "reference_values_read",
        "deterministic_scores_computed", "local_publication_new_llm_calls",
        "failure_closure_pending", "all_attempt_total_tokens", "claims_scope",
    },
    "P2_POINT_RESULTS.json": {
        "schema", "primary", "raw_exact_sensitivity", "primary_task_weights",
        "requested_questions", "valid_questions", "excluded_questions",
        "expected_corrected_generation_events", "expected_damaged_generation_events",
        "expected_net_corrections", "strict_both_unique_questions", "strict_changed_questions",
        "strict_valid_questions", "strict_corrected_events", "strict_damaged_events",
        "centered_score_changed_questions_gt_1e_minus7", "bbh_subtasks",
        "bbh_subtask_macro_difference_pp", "by_generation_seed",
        "deterministic_missing_outcome_bounds_pp", "missing_bound_scope",
        "new_llm_calls", "new_model_fits", "confidence_intervals_computed", "bootstrap_repetitions",
        "scope_sha256", "prediction_sha256", "final_model_sha256", "reference_and_scoring_costs",
    },
}
EXPECTED_REDACTIONS = {"P2_NATIVE_COLLECTOR_SOFTWARE_QUALIFICATION.json": {"private_test_artifacts"}}
ARTIFACT_ROLES = {
    "collect_native_exports.py": "LOCAL_REAL_PARTIAL_QUALIFIED_COLLECTOR",
    "qualify_native_collector.py": "LOCAL_SOFTWARE_QUALIFICATION",
    "P2_NATIVE_COLLECTOR_SOFTWARE_QUALIFICATION.json": "SYNTHETIC_SOFTWARE_QUALIFICATION",
    "P2_NATIVE_COLLECTION_REAL_PARTIAL.json": "TWO_ARCHIVE_REAL_PARTIAL_QUALIFICATION",
    "migrate_aggregation_once.py": "SINGLE_MIGRATION_OPERATOR_WITH_VERIFIED_RECEIPT",
    "P2_SINGLE_AGGREGATION_MIGRATION.json": "VERIFIED_MIGRATION_AT_RECORDED_SNAPSHOT",
    "P2_RUNTIME_SCALING_REVIEW.json": "LOCAL_ANALYSIS_ONLY",
    "P2_RUNTIME_SCALING_REVIEW.md": "LOCAL_ANALYSIS_ONLY",
    "p2_manifest_registry.py": "DEPLOYED_V3_SOURCE_REGISTRATION_CONTROL",
    "p2_aggregation_follower_v3.py": "DEPLOYED_THREE_SHARD_WORKER_AT_ACTIVATION",
    "p2_export_native_v3.py": "DEPLOYED_AND_EXERCISED_REGISTERED_EXPORT",
    "monitor_native_status_v3.py": "DEPLOYED_AND_EXERCISED_REGISTERED_MONITOR",
    "prepare_aggregation_shards_v3.py": "EXECUTED_DISJOINT_REMAINING_PARTITION_PREPARER",
    "launch_registered_worker_v3.py": "EXECUTED_REGISTERED_THREE_SHARD_LAUNCHER",
    "qualify_registry_consumers.py": "LOCAL_PREDEPLOYMENT_SYNTHETIC_QUALIFICATION",
    "qualify_registry_dispatch.py": "LOCAL_PREDEPLOYMENT_SYNTHETIC_QUALIFICATION",
    "P2_REGISTRY_DISPATCH_QUALIFICATION.json": "PRESERVED_PREDEPLOYMENT_QUALIFICATION_SNAPSHOT",
    "P2_REGISTRY_CONSUMERS_QUALIFICATION.json": "PRESERVED_PREDEPLOYMENT_QUALIFICATION_SNAPSHOT",
    "P2_V3_TOOLING_QUALIFICATION.json": "PRESERVED_PREDEPLOYMENT_QUALIFICATION_SNAPSHOT",
    "switch_aggregation_to_three_shards.py": "EXECUTED_GRACEFUL_THREE_SHARD_SWITCH",
    "P2_V3_REGISTERED_DISPATCH_HANDOFF.md": "PREPARATION_HANDOFF_WITH_ACTIVATION_ADDENDUM",
    "record_three_shard_activation.py": "LOCAL_DEIDENTIFIED_ACTIVATION_RECORDER",
    "P2_THREE_SHARD_ACTIVATION.json": "ACTUAL_THREE_SHARD_FIRST_SUCCESS_SNAPSHOT",
    "prepare_aggregation_shards_v4.py": "EXECUTED_FIVE_WAY_REMAINING_PARTITION_PREPARER",
    "launch_registered_worker_v4.py": "EXECUTED_FIVE_SHARD_LAUNCHER",
    "qualify_five_shard_dispatch.py": "LOCAL_PREDEPLOYMENT_SYNTHETIC_QUALIFICATION",
    "scale_aggregation_to_five.py": "EXECUTED_GRACEFUL_FIVE_SHARD_SWITCH",
    "P2_FIVE_SHARD_HANDOFF.md": "FIVE_SHARD_HANDOFF_WITH_ACTIVATION_ADDENDUM",
    "record_five_shard_activation.py": "LOCAL_DEIDENTIFIED_ACTIVATION_RECORDER",
    "P2_FIVE_SHARD_TOOLING_QUALIFICATION.json": "PRESERVED_PREDEPLOYMENT_QUALIFICATION_SNAPSHOT",
    "P2_FIVE_SHARD_ACTIVATION.json": "ACTUAL_FIVE_SHARD_FIRST_SUCCESS_SNAPSHOT",
    "P2_FIVE_SHARD_THROUGHPUT.json": "OBSERVED_SHARED_LOAD_THROUGHPUT_WINDOWS",
    "prepare_member_shards.py": "QUALIFIED_DISJOINT_MEMBER_CONTINUATION_PREPARER",
    "launch_registered_member.py": "QUALIFIED_REGISTERED_MEMBER_LAUNCHER",
    "stop_registered_workers.py": "QUALIFIED_EXPLICIT_OWNED_WORKER_GRACEFUL_STOP",
    "qualify_member_parallel.py": "LOCAL_PREDEPLOYMENT_SYNTHETIC_QUALIFICATION",
    "P2_MEMBER_PARALLEL_QUALIFICATION.json": "PRESERVED_PREDEPLOYMENT_QUALIFICATION_SNAPSHOT",
    "P2_MEMBER_PARALLEL_HANDOFF.md": "MEMBER_CONTINUATION_INTERFACE_HANDOFF",
    "record_runtime_fill_activation.py": "LOCAL_DEIDENTIFIED_ACTIVATION_RECORDER",
    "P2_RUNTIME_FILL_ACTIVATION.json": "ACTUAL_FOUR_MEMBER_SIX_AGGREGATION_FIRST_SUCCESS_SNAPSHOT",
    "P2_POINT_RESULTS.json": "FINAL_FROZEN_POLICY_PAIRED_EVALUATION_SUMMARY",
    "P2_FINAL_RESULTS.md": "FINAL_FROZEN_POLICY_EVALUATION_REPORT",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def write_atomic(path,data):
    temporary = path.with_name("."+path.name+"-"+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def private_path_fields(value, prefix=""):
    if isinstance(value,dict):
        return [field for key,item in value.items() for field in private_path_fields(item,prefix+"."+key)]
    if isinstance(value,list):
        return [field for item in value for field in private_path_fields(item,prefix+"[]")]
    if isinstance(value,str):
        normalized=value.replace("\\","/").casefold()
        if (re.match(r"^[a-z]:/",normalized) or normalized.startswith("/")
                or "/private/" in normalized or "/.e20_p2_" in normalized):
            return [prefix]
    return []


def publication_bytes(path,source):
    redactions=[]
    data=source
    if path.name in NEW_JSON_FIELDS:
        value=json.loads(source.decode("utf-8-sig"))
        permitted=NEW_JSON_FIELDS[path.name]
        if path.name == "P2_POINT_RESULTS.json" and (
                set(value) != permitted
                or value.get("schema") != "e20_p2_frozen_probability_evaluation_v1"):
            raise ValueError("Final result fields differ from the frozen projection/scoring output")
        redactions=sorted(set(value)-permitted)
        if set(redactions) != EXPECTED_REDACTIONS.get(path.name,set()):
            raise ValueError("New qualification fields require publication review " + path.name)
        value={key:item for key,item in value.items() if key in permitted}
        if private_path_fields(value):
            raise ValueError("Private runtime paths in new public JSON " + path.name)
        if redactions:
            data=(json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    return data,redactions


def stage(parent_commit=DEFAULT_PARENT):
    import numpy as np

    if not re.fullmatch(r"[0-9a-f]{40}",parent_commit):
        raise ValueError("A full parent commit SHA is required")
    paths=sorted(path for path in CODE.iterdir() if path.is_file()
        and path.name in SOURCE_ALLOWLIST | JSON_ALLOWLIST)
    captured={}
    signatures={}
    for path in paths:
        before=path.stat()
        source=path.read_bytes()
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError("Ready source changed while being captured " + path.name)
        data,redactions=publication_bytes(path,source)
        captured[path.name]=(data,digest(source),redactions)
        signatures[path.name]=(after.st_size,after.st_mtime_ns)
    final_results_present = "P2_POINT_RESULTS.json" in captured
    publication_stage = "FROZEN_POLICY_EVALUATION_COMPLETE" if final_results_present else "PRODUCTION_EXECUTION_STARTED"
    snapshot_name=next(name for name in ("P2_POINT_RESULTS.json","P2_RUNTIME_FILL_ACTIVATION.json","P2_FIVE_SHARD_ACTIVATION.json","P2_THREE_SHARD_ACTIVATION.json","P2_ACTIVATION_STATUS.json") if name in captured)
    status=json.loads(captured[snapshot_name][0].decode("utf-8-sig"))
    forbidden = set()
    with np.load(STUDY / "private/P2_PREDICTIONS_FROZEN.npz", allow_pickle=False) as archive:
        for key in ("question_ids", "canonical_hashes"):
            forbidden.update(map(str, archive[key].tolist()))
    for line in (STUDY / "private/ADMITTED2400_INPUTS_ONLY.jsonl").read_bytes().splitlines():
        question = json.loads(line)["question"]
        if len(question) >= 60:
            forbidden.add(question)
    for path in paths:
        content = captured[path.name][0].decode("utf-8-sig")
        if any(value in content for value in forbidden):
            raise ValueError("Private input content in public artifact " + path.name)
        if path.suffix == ".json":
            json.loads(content)
        if path.suffix == ".py":
            compile(content, str(path), "exec")
    STAGE.mkdir(parents=True, exist_ok=True)
    public = STUDY / "public"
    public.mkdir(parents=True, exist_ok=True)
    files = []
    for path in paths:
        data,source_sha,redactions=captured[path.name]
        write_atomic(STAGE/path.name,data)
        write_atomic(public/path.name,data)
        files.append(dict(package_path=path.name, publication_path=PREFIX + path.name,
                          bytes=len(data), sha256=digest(data), git_sha1=git_sha(data),
                          source_sha256=source_sha,removed_private_fields=redactions,
                          artifact_role=ARTIFACT_ROLES.get(path.name,"READY_SOURCE_OR_RECORDED_SNAPSHOT")))
    for path in paths:
        current=path.stat()
        if (current.st_size,current.st_mtime_ns)!=signatures[path.name]:
            raise ValueError("Ready source changed during staging, repeat final staging " + path.name)
    pending=sorted(path.name for path in CODE.iterdir() if path.is_file()
        and path.suffix in {".py",".md",".txt"} and path.name not in SOURCE_ALLOWLIST
        and "PRIVATE" not in path.name.upper())
    index = dict(schema="p2_activation_public_file_index_v1",
                 branch="codex/e20-p2-independent-confirmation",
                 parent_commit=parent_commit,
                 stage=publication_stage, files=files,
                 private_identity_or_text_matches=0, private_arrays_or_weights_uploaded=0,
                 activation_snapshot_sha256=digest(captured["P2_ACTIVATION_STATUS.json"][0]),
                 current_runtime_snapshot=snapshot_name,
                 current_runtime_snapshot_sha256=digest(captured[snapshot_name][0]),
                 staging_not_a_deployment_event=True,unselected_pending_sources=pending,
                 new_json_private_path_matches=0,
                 staged_utc=datetime.now(timezone.utc).isoformat())
    data = (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode()
    write_atomic(STAGE/"FILE_INDEX.json",data)
    write_atomic(public/"FILE_INDEX.json",data)
    print(json.dumps({"status":"STAGED", "files":len(files)+1,
                      "bytes":sum(row["bytes"] for row in files)+len(data),
                      "source_status":publication_stage if final_results_present else status.get("status"), "private_matches":0,
                      "unselected_pending_sources":pending,
                      "redacted_fields":sum(len(row["removed_private_fields"]) for row in files)}, ensure_ascii=False))


def chunks():
    result, current, length = [], [], 0
    index_bytes=(STAGE/"FILE_INDEX.json").read_bytes()
    index=json.loads(index_bytes)
    selected={row["package_path"]:row for row in index["files"]}
    for name in sorted([*selected,"FILE_INDEX.json"]):
        path=STAGE/name
        raw=path.read_bytes()
        if name in selected and digest(raw)!=selected[name]["sha256"]:
            raise ValueError("Staging changed after its file index was written " + name)
        text = raw.decode("utf-8")
        for offset in range(0, len(text), 18000):
            part = text[offset:offset + 18000]
            if current and length + len(part) > 30000:
                result.append(current)
                current, length = [], 0
            current.append(dict(path=PREFIX + path.name, offset=offset, text=part))
            length += len(part)
    if current:
        result.append(current)
    if (STAGE/"FILE_INDEX.json").read_bytes()!=index_bytes:
        raise ValueError("Publication index changed while chunks were prepared")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--parent-commit",default=DEFAULT_PARENT)
    args = parser.parse_args()
    if args.stage:
        stage(args.parent_commit)
    elif args.chunk is not None:
        print(json.dumps(chunks()[args.chunk], ensure_ascii=False))
    else:
        print(json.dumps({"chunks":len(chunks())}))
