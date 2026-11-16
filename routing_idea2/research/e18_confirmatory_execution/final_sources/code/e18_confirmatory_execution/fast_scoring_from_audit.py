"""Load closed native receipts under the completed audit and run frozen scoring.

The user's later explicit authorization permits reuse of the completed full
native audit. This entry does not rerun the original9b execution_gate. It loads
only archived LAST_TASK_RECEIPT JSON and the already audited historical member
cache, checks exact selected sets and measured Token totals, and calls the
unchanged original9b execute_scoring under the frozen7fd ScopeMemo context.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
import zipfile

import cpu_scope_memo as memo_api


FINAL_AUDIT_SHA = "8367d50ef3138c4c0d7327fe18ebc883a93b709d28bdfcdcb59f6c19ef8f8655"
FROZEN_MEMO_SHA = "7fd45e003f09c8fc342e5b90f72b0e96f18e08bca63045c90fe13a424d15ed75"
STAGES = ("member_generation", "aggregation")


def require(condition, description):
    if not condition:
        raise ValueError(description)


def native_cost(record, key, stage, *, historical=False):
    """Read already audited identity and actual usage without rehashing it."""
    require(record.get("status") == "success" and record.get("cost_status") == "recorded_actual_usage",
            "Only already successful actual-usage native outputs are admitted")
    usage = record.get("result", {})
    require(all(type(usage.get(field)) is int and usage[field] >= 0
                for field in ("prompt_tokens", "completion_tokens")), "Actual Token fields are missing")
    require(isinstance(record.get("attempt_id"), str) and bool(record["attempt_id"]),
            "An already audited physical attempt identity is required")
    require(isinstance(record.get("record_sha256"), str) and bool(record["record_sha256"]),
            "An already audited native source identity is required")
    require(record.get("question_id") == key[0], "Native question and selected key differ")
    return dict(stage=stage, task_key=list(key), native_record_sha256=record["record_sha256"],
                physical_attempt_id=record["attempt_id"], status=record["status"],
                prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
                reused_historical_source=historical)


def assemble_archived_receipts(archive_paths, scope_sha, required_members,
                               required_aggregates, *, expected_stage_totals=None,
                               progress=None):
    """Stream receipt JSON from audited tar files; leave records and fences unread.

    required_members is the owned/callable member set. The historical cache is
    supplied separately. Any repeated logical key or physical attempt rejects.
    This is a loader under an existing audit, with no native rehash/revalidation.
    """
    required_members, required_aggregates = set(required_members), set(required_aggregates)
    members, by_key, costs = {}, {}, {}
    totals = {stage: dict(tasks=0, prompt_tokens=0, completion_tokens=0) for stage in STAGES}
    seen_archives = set()
    for archive_index, supplied in enumerate(archive_paths, 1):
        path = Path(supplied)
        require(path.name not in seen_archives, "Repeated closed archive filename")
        seen_archives.add(path.name)
        before = path.stat()
        with tarfile.open(path, "r|gz") as archive:
            for item in archive:
                if PurePosixPath(item.name).name != "LAST_TASK_RECEIPT.json":
                    continue
                parts = PurePosixPath(item.name).parts
                require(item.isfile() and len(parts) == 4 and parts[0] == scope_sha
                        and parts[1] in STAGES and re.fullmatch(r"[0-9a-f]{64}", parts[2]),
                        "An audited receipt has an unexpected archive path or type")
                receipt = json.load(archive.extractfile(item))
                stage = receipt.get("stage")
                require(receipt.get("schema") == "e18_execution_task_receipt_v2"
                        and receipt.get("status") == "SUCCESS"
                        and receipt.get("scope_sha256") == scope_sha
                        and receipt.get("software_test_only") is False
                        and stage == parts[1], "Receipt mode, scope, stage or success differs")
                task = receipt["task_key"]
                action_field = "member_index" if stage == "member_generation" else "combination_index"
                key = (task["question_id"], task["generation_seed"], task[action_field])
                required = required_members if stage == "member_generation" else required_aggregates
                destination = members if stage == "member_generation" else by_key
                require(key in required and key not in destination, "Repeated or out-of-union receipt key")
                record = receipt["native_record"]
                require(receipt.get("physical_attempt_id") == record.get("attempt_id")
                        and receipt.get("native_record_sha256") == record.get("record_sha256"),
                        "Receipt and native source identity differ")
                cost = native_cost(record, key, stage)
                require(cost["physical_attempt_id"] not in costs, "A physical attempt is repeated")
                costs[cost["physical_attempt_id"]] = cost
                destination[key] = record if stage == "member_generation" else receipt
                for field in ("prompt_tokens", "completion_tokens"):
                    totals[stage][field] += cost[field]
                totals[stage]["tasks"] += 1
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "A closed archive changed during its receipt load")
        if progress is not None:
            progress(dict(archives_loaded=archive_index, member_keys_loaded=len(members),
                          aggregation_keys_loaded=len(by_key)))
    require(set(members) == required_members and set(by_key) == required_aggregates,
            "Audited archive receipts do not cover the exact frozen native sets")
    if expected_stage_totals is not None:
        for stage in STAGES:
            require(all(totals[stage][field] == expected_stage_totals[stage][field]
                        for field in ("tasks", "prompt_tokens", "completion_tokens")),
                    "Loaded native counts or actual Token totals differ from the completed audit")
    return dict(members=members, by_key=by_key,
                receipts=[by_key[key] for key in sorted(by_key)], costs=costs,
                stage_totals=totals, archives=len(seen_archives))


def append_historical_members(loaded, historical_by_key, required_all_members):
    """Add the separately audited historical cache and its measured resources."""
    require(not set(loaded["members"]).intersection(historical_by_key),
            "Historical and newly executed member keys overlap")
    require(set(loaded["members"]) | set(historical_by_key) == set(required_all_members),
            "Historical and new member keys do not form the frozen complete union")
    for key, record in historical_by_key.items():
        require(record.get("seed") == key[1], "Historical member generation seed differs")
        cost = native_cost(record, key, "member_generation", historical=True)
        require(cost["physical_attempt_id"] not in loaded["costs"],
                "Historical physical attempt overlaps another selected execution")
        loaded["costs"][cost["physical_attempt_id"]] = cost
        loaded["members"][key] = record
    return loaded


def historical_records(external, historical_zip=None):
    """Read the existing native cache bytes using its already approved index."""
    records = {}
    if historical_zip is not None:
        with zipfile.ZipFile(historical_zip) as archive:
            names = set(archive.namelist())
            for key, reference in external.items():
                name = "cache/member_records/" + PurePosixPath(reference["path"].replace("\\", "/")).name
                require(name in names, "The historical native ZIP is missing an indexed source")
                records[key] = json.loads(archive.read(name))
    else:
        for key, reference in external.items():
            with Path(reference["path"]).open(encoding="utf-8-sig") as stream:
                records[key] = json.load(stream)
    return records


def context_from_completed_audit(final, manifest, audit_ref, archive_directory,
                                *, historical_zip=None, progress=None):
    """Build the original scorer context under the explicit later authorization."""
    require(audit_ref["sha256"] == FINAL_AUDIT_SHA
            and memo_api.file_sha(audit_ref["path"]) == FINAL_AUDIT_SHA,
            "The approved completed native audit differs")
    audit = final.worker.read_json(audit_ref["path"])
    require(audit["status"] == "PASS_INDEPENDENT_STREAMING_NATIVE_AUDIT"
            and audit["archives_checked"] == 155 and audit["native_tasks_checked"] == 73490
            and audit["unique_physical_attempts"] == 73490
            and audit["gold_or_reward_sources_read"] == audit["evaluation_results_read"] == 0,
            "Completed independent native audit scope differs")
    execution = final.worker.read_json(manifest["execution_manifest_ref"]["path"])
    module, adapter, _, _, _, external = final.worker.load_worker(execution)
    adapter._worker_software_mode = False
    scope = adapter.scope
    scope_sha = audit["execution_scope_sha256"]
    require(module.canonical_digest(scope) == scope_sha
            and audit["execution_scope_file_sha256"] == execution["scope_ref"]["sha256"],
            "Frozen execution scope differs from the completed native audit")
    metadata, required = final.question_union(scope)
    metadata = [dict(row, source_frame=adapter.questions[row["question_id"]].get("source_frame"),
                     cohort_role=adapter.questions[row["question_id"]].get("cohort_role"))
                for row in metadata]
    require(all(row["source_frame"] and row["cohort_role"] for row in metadata),
            "Frozen input source roles or frames are absent")
    reference_scope = final.worker.read_json(manifest["reference_scope_ref"]["path"])
    scoring_scope = final.worker.read_json(manifest["scoring_scope_ref"]["path"])
    validator = final.audit_startup16.load_source_validator(manifest["source_validator_ref"])
    for component, operation in ((reference_scope, "read_reference"), (scoring_scope, "score_outputs")):
        module.validate_scope_approval(execution["approval_ref"], component, operation)
    require(validator.build_scoring_scope(scope, reference_scope) == scoring_scope,
            "Scoring/reference bindings differ from the frozen execution scope")
    archive_directory = Path(archive_directory)
    archives = []
    for row in audit["archives"]:
        path = archive_directory / row["filename"]
        require(path.name == row["filename"] and path.is_file()
                and path.stat().st_size == row["compressed_bytes"],
                "A completed audited archive is missing or its size changed")
        archives.append(path)
    loaded = assemble_archived_receipts(archives, scope_sha, adapter.callable_members, required,
                                       expected_stage_totals=audit["stage_costs"], progress=progress)
    require(len(external) == 4717, "The completed historical cache count differs")
    append_historical_members(loaded, historical_records(external, historical_zip), adapter.required_members)
    gate = dict(
        schema="e18_complete_execution_admission_v1", status="PASS",
        admission_basis="REUSED_COMPLETED_INDEPENDENT_NATIVE_AUDIT_AUTHORIZED_BY_LATEST_USER",
        original9b_execution_gate_rerun=False,
        authorized_by_latest_user="Direct formal reference scoring and results analysis after completed execution/audit",
        completed_native_audit_ref=dict(audit_ref),
        completed_native_archives=155, completed_new_native_tasks=73490,
        loaded_native_receipt_counts=loaded["stage_totals"],
        original_questions=len(metadata), unique_aggregate_keys=len(required),
        unique_member_keys=len(loaded["members"]), policy_ids=list(final.POLICIES),
        generation_seeds=[42, 137, 2024], execution_scope_sha256=scope_sha,
        approval_sha256=execution["approval_ref"]["sha256"], physical_attempts=len(loaded["costs"]),
        failed_known_cost_attempts=0, reference_values_read=0, scorer_calls=0,
        sealed_payload_reads=0, source_frame_counts=dict(Counter(row["source_frame"] for row in metadata)),
        cohort_role_counts=dict(Counter(row["cohort_role"] for row in metadata)),
        historical_cached_member_records=len(external),
        native_rehash_or_original_full_gate_repeated=False,
        loader_source_sha256=memo_api.file_sha(__file__),
    )
    return dict(gate=gate, module=module, adapter=adapter, validator=validator,
                reference_scope=reference_scope, scoring_scope=scoring_scope,
                approval_ref=execution["approval_ref"], execution=execution,
                receipts=loaded["receipts"], by_key=loaded["by_key"],
                members=loaded["members"], costs=loaded["costs"], metadata=metadata,
                runtime_io=dict(logical_owned_root=execution["owned_output_root"],
                                physical_native_read_root=str(archive_directory),
                                member_reference_mappings=manifest.get("runtime_reference_path_mappings", []),
                                closed_native_archive_receipt_loader=True,
                                original_native_bytes_and_scope_hashes_preserved=True))


def main():
    import e18_final_scoring as final

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--completed-audit", type=Path, required=True)
    parser.add_argument("--completed-audit-sha256", default=FINAL_AUDIT_SHA)
    parser.add_argument("--archive-directory", type=Path, required=True)
    parser.add_argument("--historical-zip", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--performance-receipt", type=Path, required=True)
    parser.add_argument("--execute-scoring", action="store_true", required=True)
    args = parser.parse_args()
    require(not args.performance_receipt.exists(), "Performance receipt must use a new filename")
    require(memo_api.file_sha(args.manifest) == args.manifest_sha256, "Final manifest bytes differ")
    require(memo_api.file_sha(memo_api.__file__) == FROZEN_MEMO_SHA, "Frozen scope memo source differs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manifest = final.worker.read_json(args.manifest)
    memo = memo_api.ScopeMemo(final, manifest)
    memo._pin_ref(dict(path=str(args.manifest), sha256=args.manifest_sha256))
    audit_ref = dict(path=str(args.completed_audit), sha256=args.completed_audit_sha256)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    status, error_type = "FAILED", None
    try:
        with memo:
            context = context_from_completed_audit(final, manifest, audit_ref, args.archive_directory,
                                                  historical_zip=args.historical_zip,
                                                  progress=lambda row: print(json.dumps(dict(event="AUDITED_RECEIPTS_LOADED", **row)), flush=True))
            private = Path(args.runtime_root or manifest.get("cpu_output_root") or manifest["private_output_root"])
            final.write_new_json(private / ("PRE_SCORING_ADMISSION_AUDIT_REUSE_" + str(time.time_ns()) + ".json"), context["gate"])
            print(json.dumps(dict(event="COMPLETED_NATIVE_AUDIT_REUSED_ENTERING_FROZEN_SCORING",
                                  original_questions=len(context["metadata"]),
                                  unique_member_keys=len(context["members"]),
                                  unique_aggregate_keys=len(context["receipts"]))), flush=True)
            result = final.execute_scoring(manifest, context, runtime_root=args.runtime_root)
            status = result["status"]
    except BaseException as error:
        status = "FAILED_INTEGRITY_OR_EXECUTION"
        error_type = type(error).__name__
        raise
    finally:
        record = dict(schema="e18_audit_reuse_fast_scoring_performance_v1", status=status,
                      error_type=error_type, entry_source_sha256=memo_api.file_sha(__file__),
                      original_finalizer_sha256=memo_api.SOURCE_PINS["e18_final_scoring.py"],
                      scope_memo_source_sha256=FROZEN_MEMO_SHA,
                      completed_native_audit_ref=audit_ref,
                      original9b_execution_gate_rerun=False,
                      wall_seconds=time.perf_counter()-started_wall,
                      cpu_seconds=time.process_time()-started_cpu, scope_memo=memo.stats())
        args.performance_receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.performance_receipt.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
