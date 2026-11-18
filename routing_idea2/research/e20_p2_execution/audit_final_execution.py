"""Audit final P2 execution metadata without opening answers or native outputs.

Committed native records are trusted only through the existing closed-export
index. This audit enumerates owner names and reads fences for exceptional owners
and logged failures, leaving previously closed successful payloads untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sqlite3

from p2_manifest_registry import digest, iter_complete_events, load_registry, read_json
from monitor_native_status_v3 import actual_usage, semantic_key, unique_entries


STAGES = {"member_generation": "required_member_keys", "aggregation": "required_aggregate_keys"}
KNOWN_PREOWNER_DISPATCHERS = {
    "28d45cb9cfed19e7d308e624468099662e69ebcd3a7335ebf470db4a3426f4aa": (0, 6717),
    "f519cb85f3446692b1d1389817e2aa834e88efd87e81f52cff858b7391a9baff": (2, 6867),
}


def time_labels(value):
    if value is None:
        return None
    return dict(unix=value, utc=datetime.fromtimestamp(value, timezone.utc).isoformat(),
                asia_shanghai=datetime.fromtimestamp(value, timezone(timedelta(hours=8))).isoformat())


def inspect_fence(owner, scope_sha, stage, expected_key=None):
    path = owner / "ATTEMPT_STATE.json"
    if owner.is_symlink() or path.is_symlink():
        return dict(status="SYMLINK_REQUIRES_REVIEW", identity_valid=False)
    if not path.is_file():
        return dict(status="NO_FENCE", identity_valid=False)
    value = read_json(path)
    key = value.get("task_key")
    valid = (value.get("scope_sha256") == scope_sha and value.get("stage") == stage
             and isinstance(key, list) and len(key) == 3 and digest(key) == owner.name)
    if expected_key is not None:
        valid = valid and tuple(key or ()) == expected_key
    return dict(status=value.get("status", "MISSING_STATUS"), identity_valid=bool(valid))


def closed_export_index(root, scope_sha):
    exports = root / "exports"
    database = exports / "exported.sqlite"
    if not database.is_file():
        return set(), dict(status="INDEX_NOT_PRESENT", verified_closed_batches=0, issues=1)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        rows = list(connection.execute("SELECT stage,digest,batch FROM exported"))
    finally:
        connection.close()
    by_batch = defaultdict(set)
    for stage, identity, batch in rows:
        by_batch[batch].add((stage, identity))
    closed, issues, legacy, legacy_sidecars = set(), 0, 0, 0
    for batch, identities in by_batch.items():
        if Path(batch).name != batch:
            issues += 1
            continue
        archive = exports / (batch if batch.endswith(".tar.gz") else batch + ".tar.gz")
        if archive.is_symlink() or not archive.is_file() or archive.stat().st_size == 0:
            issues += 1
            continue
        sidecar = exports / (batch + ".json")
        if sidecar.is_file():
            receipt = read_json(sidecar)
            if (receipt.get("status") != "CLOSED_NATIVE_BATCH"
                    or receipt.get("scope_sha256") != scope_sha
                    or Path(receipt["path"]).resolve() != archive.resolve()
                    or receipt.get("bytes") != archive.stat().st_size):
                issues += 1
                continue
            if receipt.get("schema") == "p2_native_export_receipt_v3":
                indexed = {(row["stage"], row["digest"]) for row in receipt.get("index_rows", ())}
                if indexed != identities:
                    issues += 1
                    continue
            elif receipt.get("schema") is None and re.fullmatch(r"p2_native_\d+", batch):
                fields = {"status", "path", "sha256", "bytes", "tasks", "prompt_tokens", "completion_tokens",
                          "scope_sha256", "gold_reads", "new_llm_calls"}
                if (set(receipt) != fields or receipt.get("tasks") != len(identities)
                        or not isinstance(receipt.get("sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])
                        or any(type(receipt.get(name)) is not int or receipt[name] < 0
                               for name in ("prompt_tokens", "completion_tokens"))
                        or receipt.get("gold_reads") != 0 or receipt.get("new_llm_calls") != 0):
                    issues += 1
                    continue
                legacy += 1
                legacy_sidecars += 1
            else:
                issues += 1
                continue
        else:
            if not re.fullmatch(r"p2_native_\d+", batch):
                issues += 1
                continue
            legacy += 1
        closed.update(identities)
    return closed, dict(status="EXISTING_CLOSED_EXPORT_INDEX_REUSED", verified_closed_batches=len(by_batch)-issues,
                        legacy_closed_batches_without_v3_sidecar=legacy, issues=issues,
                        legacy_original_sidecars_verified=legacy_sidecars,
                        archive_payloads_opened=0, archive_hashes_recomputed=0,
                        trust_scope="Existing successful exporter closure and immutable native-owner contract")


def tuple_preowner_proof(event, entry, starts, adapter_source):
    error = str(event.get("error", ""))
    tuple_error = "tuple" in error.lower() and "encode" in error.lower()
    body = adapter_source.split("def run_approved_task(", 1)[-1]
    call = body.find("job = adapter.aggregation_job(")
    owner = body.find("owner = owned_root / scope_sha")
    invocation = body.find("row, reused = store.run(backend, job)")
    source_order = 0 <= call < owner < invocation
    matched_log = False
    at = event.get("at_unix", 0)
    candidates = sorted((row for row in starts if row.get("manifest_sha256") == entry["manifest_ref"]["sha256"]
                         and row.get("at_unix", 0) <= at), key=lambda row: row.get("at_unix", 0), reverse=True)
    if candidates:
        path = Path(candidates[0].get("log_path", ""))
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 262144))
                tail = stream.read().decode("utf-8", errors="replace")
            matched_log = ("tuple" in tail.lower() and "encode" in tail.lower()
                           and "job = adapter.aggregation_job(" in tail
                           and "run_approved_task" in tail)
    return dict(category="TOKENIZER_TUPLE_ENCODE" if tuple_error else "UNCLASSIFIED_FAILURE",
                original_log_preowner_stack_verified=bool(matched_log),
                bound_adapter_job_before_owner_and_store_run=bool(source_order),
                no_native_invocation_proven=bool(tuple_error and matched_log and source_order))


def classify_unregistered_dispatcher(folder, expected_members, adapter_source):
    result = dict(dispatcher_manifest_sha256=folder.name,
                  category="UNREGISTERED_DISPATCHER_REQUIRES_REVIEW", no_native_invocation_proven=False)
    if folder.name not in KNOWN_PREOWNER_DISPATCHERS or folder.is_symlink():
        return result
    profile, planned = KNOWN_PREOWNER_DISPATCHERS[folder.name]
    files = list(folder.iterdir())
    if ({path.name for path in files} != {"EVENTS.jsonl", "SUMMARY.json"}
            or any(path.is_symlink() or not path.is_file() for path in files)):
        return result
    journal = folder / "EVENTS.jsonl"
    raw = journal.read_bytes()
    if not raw.endswith(b"\n"):
        return result
    events = list(iter_complete_events(journal))
    summary = read_json(folder / "SUMMARY.json")
    body = adapter_source.split("def run_approved_task(", 1)[-1]
    rejection = body.find('"Approval does not cover the frozen stage execution budget"')
    owner = body.find("owner = owned_root / scope_sha")
    backend = body.find("backend = backend_factory(spec)")
    source_preowner = 0 <= rejection < owner < backend
    sequence = len(events) == 2 and [row.get("event") for row in events] == [
        "TASK_ENTER", "STOP_REQUIRES_RECONCILIATION"]
    stop = events[-1] if events else {}
    task = events[0].get("task", {}) if events else {}
    key = (task.get("question_id"), task.get("generation_seed"), task.get("member_index"))
    task_valid = (set(task) == {"stage", "question_id", "generation_seed", "member_index"}
                  and task.get("stage") == "member_generation" and task.get("member_index") == profile
                  and expected_members.get(digest(list(key))) == key and stop.get("task") == task)
    cost_fields = {"physical_attempt_id", "actual_usage", "native_record_sha256", "native_job_id"}
    no_native_fields = all(not cost_fields.intersection(row) for row in events)
    summary_valid = (summary.get("schema") == "e18_gpu_worker_summary_v1"
        and summary.get("manifest_sha256") == folder.name and summary.get("stage") == "member_generation"
        and summary.get("profile") == profile and summary.get("planned") == planned
        and summary.get("completed") == 0 and summary.get("cached") == 0
        and summary.get("new_successes") == 0 and summary.get("backend_loads") == 0
        and summary.get("software_test_only") is False and summary.get("stop_requested") is False
        and len(summary.get("failures", ())) == 1
        and summary["failures"][0].get("error_type") == "PermissionError"
        and summary["failures"][0].get("task") == task)
    exact_rejection = (stop.get("error_type") == "PermissionError"
                      and stop.get("error") == "Approval does not cover the frozen stage execution budget")
    checks = dict(exact_two_event_startup_sequence=sequence, approved_task_identity=task_valid,
                  zero_completed_zero_backend_summary=summary_valid, no_native_cost_fields=no_native_fields,
                  exact_budget_rejection=exact_rejection, bound_source_gate_before_owner_and_backend=source_preowner)
    proven = all(checks.values())
    return dict(result, category="PRE_OWNER_EXECUTION_BUDGET_REJECTION" if proven else result["category"],
                no_native_invocation_proven=proven, member_index=profile, checks=checks,
                event_counts=dict(Counter(row.get("event") for row in events)),
                error_type=stop.get("error_type", "UNSPECIFIED"), at=time_labels(stop.get("at_unix")))


def audit(registry_path):
    registry = load_registry(registry_path)
    if registry["software_test_only"]:
        raise PermissionError("Final execution audit requires the production registry")
    owned = Path(registry["common"]["owned_output_root"])
    root, scope_sha = owned.parent, registry["scope_canonical_sha256"]
    scope = read_json(registry["common"]["scope_ref"]["path"])
    expected = {stage: {digest(list(key)): tuple(key) for key in scope[field]} for stage, field in STAGES.items()}
    closed, export = closed_export_index(root, scope_sha)
    starts = [read_json(path) for path in (root / "control").glob("*.start.json")]
    adapter_source = Path(registry["common"]["adapter_ref"]["path"]).read_text(encoding="utf-8-sig")
    successes, attempts, tokens, last = defaultdict(set), {}, Counter(), {}
    failed_events, partial_journals, registered_dirs = [], 0, set()
    for entry in unique_entries(registry):
        journal = Path(entry["journal_path"])
        registered_dirs.add(journal.parent.name)
        if journal.is_file() and journal.stat().st_size:
            with journal.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                partial_journals += int(stream.read(1) != b"\n")
        allowed = set(entry["task_keys"])
        for event in iter_complete_events(journal):
            if event.get("event") == "TASK_RESULT" and event.get("status") == "SUCCESS":
                stage, key = semantic_key(event)
                if stage != entry["stage"] or key not in allowed:
                    raise ValueError("A journal result escaped its registered manifest")
                successes[stage].add(digest(list(key)))
                last[stage] = max(last.get(stage, 0), event.get("at_unix", 0))
                if type(event.get("from_cache")) is not bool:
                    raise ValueError("A native success must declare whether it reused a cache")
                if event.get("from_cache") is False:
                    usage = actual_usage(event)
                    if not isinstance(event.get("physical_attempt_id"), str) or not event["physical_attempt_id"]:
                        raise ValueError("A paid success requires its physical attempt identity")
                    identity = (stage, event["physical_attempt_id"])
                    signature = (key, usage["prompt_tokens"], usage["completion_tokens"])
                    if identity in attempts and attempts[identity] != signature:
                        raise ValueError("One physical attempt has inconsistent journal metadata")
                    if identity not in attempts:
                        attempts[identity] = signature
                        for kind, count in usage.items():
                            tokens[(stage, kind)] += count
            elif event.get("event") == "STOP_REQUIRES_RECONCILIATION" or (
                    event.get("event") == "TASK_RESULT" and event.get("status") != "SUCCESS"):
                stage, key = semantic_key(event)
                proof = tuple_preowner_proof(event, entry, starts, adapter_source)
                fence = inspect_fence(owned / scope_sha / stage / digest(list(key)), scope_sha, stage, key)
                failed_events.append(dict(worker_id=entry["worker_id"], stage=stage,
                    error_type=event.get("error_type", "UNSPECIFIED"), at=time_labels(event.get("at_unix")),
                    **proof, owner_fence_now=fence))
    stages, owner_issues = {}, 0
    for stage, keys in expected.items():
        folder = owned / scope_sha / stage
        actual, extra_fences, non_directories = set(), Counter(), 0
        if folder.is_dir() and not folder.is_symlink():
            with os.scandir(folder) as entries:
                for item in entries:
                    if item.is_dir(follow_symlinks=False):
                        actual.add(item.name)
                        if item.name not in keys:
                            fence = inspect_fence(Path(item.path), scope_sha, stage)
                            extra_fences[fence["status"]] += 1
                    else:
                        non_directories += 1
        missing, extra = set(keys)-actual, actual-set(keys)
        owner_issues += len(missing)+len(extra)+non_directories
        stages[stage] = dict(expected_keys=len(keys), unique_successful_keys=len(successes[stage]),
            missing_success_keys=len(set(keys)-successes[stage]), extra_success_keys=len(successes[stage]-set(keys)),
            owner_directories=len(actual), missing_owner_directories=len(missing), extra_owner_directories=len(extra),
            other_stage_entries=non_directories, extra_owner_fence_status_counts=dict(extra_fences),
            closed_exported_keys=sum((stage, identity) in closed for identity in keys),
            last_success=time_labels(last.get(stage)),
            successful_prompt_tokens=tokens[(stage, "prompt_tokens")],
            successful_completion_tokens=tokens[(stage, "completion_tokens")])
    dispatcher = owned / "_dispatcher"
    unregistered = [path for path in dispatcher.iterdir() if path.is_dir() and path.name not in registered_dirs] if dispatcher.is_dir() else []
    unregistered_audits = [classify_unregistered_dispatcher(path, expected["member_generation"], adapter_source)
                          for path in sorted(unregistered)]
    unresolved_dispatchers = sum(not row["no_native_invocation_proven"] for row in unregistered_audits)
    extra_closed = len(closed-{(stage, identity) for stage, keys in expected.items() for identity in keys})
    all_success = all(row["missing_success_keys"] == row["extra_success_keys"] == 0 for row in stages.values())
    closure_complete = all(row["closed_exported_keys"] == row["expected_keys"] for row in stages.values())
    unknown = sum(not row["no_native_invocation_proven"] for row in failed_events)
    paid_keys = defaultdict(set)
    for (stage, attempt), (key, _, _) in attempts.items():
        paid_keys[(stage, key)].add(attempt)
    duplicate_paid = sum(max(0, len(values)-1) for values in paid_keys.values())
    missing_paid_keys = sum((stage, key) not in paid_keys for stage, keys in expected.items() for key in keys.values())
    cost_closed = (all_success and closure_complete and not any((owner_issues, extra_closed, export["issues"],
                   unknown, partial_journals, unresolved_dispatchers, duplicate_paid, missing_paid_keys)))
    successful_tokens = sum(tokens.values())
    return dict(schema="p2_final_metadata_execution_audit_v1",
        status="COMPLETE_SUCCESS_AND_NATIVE_COST_CLOSED" if cost_closed else "SUCCESS_OR_COST_AUDIT_PENDING",
        scope_canonical_sha256=scope_sha, stages=stages, all_required_successful=all_success,
        closed_exporter_coverage_complete=closure_complete, exporter_closure=export,
        journal_failure_events=failed_events, unresolved_failure_events=unknown,
        partial_journals=partial_journals, unregistered_dispatcher_directories=len(unregistered),
        unregistered_dispatcher_classifications=unregistered_audits,
        unresolved_unregistered_dispatcher_directories=unresolved_dispatchers,
        extra_export_index_keys=extra_closed, duplicate_paid_success_keys=duplicate_paid,
        successful_keys_without_original_paid_journal_evidence=missing_paid_keys,
        unique_paid_successful_attempts=len(attempts), successful_call_tokens=successful_tokens,
        failure_closure_pending=not cost_closed, failed_or_unknown_native_attempts_audited=cost_closed,
        all_attempt_total_tokens=successful_tokens if cost_closed else None,
        model_loading_planning_transfer_cpu_and_gpu_overhead="Separate runtime overhead, excluded from native Token counts",
        native_output_payloads_read=0, gold_values_read=0, new_llm_calls=0,
        audit_boundary="Exact approved scope owner directories, registered journals and existing successful export closure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.registry), ensure_ascii=False, allow_nan=False))
