"""Synthetic retirement, cost-deduplication and export checks for monitor v2."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import execution_monitor as base
import execution_monitor_v2 as monitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n", encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value)+"\n")


def fixture(root):
    question_metadata = {f"q{index}": dict(question_id=f"q{index}", canonical_question_sha256=f"canonical{index}")
                         for index in range(6)}
    scope = dict(bindings={"synthetic": "synthetic"}, operator_fingerprint="operator",
                 question_keys=list(question_metadata.values()))
    scope_sha = base.canonical(scope)
    required = dict(member_generation={(f"q{index}", 42, 4) for index in range(6)},
                    aggregation={("q0", 42, 0)})
    state = dict(
        root=root, scope=scope, scope_sha=scope_sha, owned=root/"owned",
        required=required, plan_sha256="plan", question_metadata=question_metadata,
        plan=dict(cohort="e23_reference_supplement", scope_ref={"sha256": "scope"},
                  approval_ref={"sha256": "approval"}),
        manifests=[], retired_worker_ids={"old0", "old1"}, dispatch_sha256="dispatch",
    )
    assignments = [
        ("old0", "member_generation", {("q0", 42, 4), ("q1", 42, 4), ("q2", 42, 4)}, {("q0", 42, 4)}),
        ("old1", "member_generation", {("q3", 42, 4), ("q4", 42, 4), ("q5", 42, 4)}, {("q3", 42, 4)}),
    ]
    assignments.extend((f"new{index}", "member_generation", {key}, {key})
                       for index, key in enumerate(sorted(required["member_generation"]-{("q0", 42, 4), ("q3", 42, 4)})))
    assignments.append(("aggregation", "aggregation", required["aggregation"], required["aggregation"]))
    for worker_id, stage, keys, completed in assignments:
        manifest_sha = base.canonical(worker_id)
        state["manifests"].append((
            dict(worker_id=worker_id, stage=stage, profile=4 if stage == "member_generation" else 0,
                 manifest_ref={"sha256": manifest_sha}), {}, keys))
        dispatcher = state["owned"] / "_dispatcher" / manifest_sha
        dispatcher.mkdir(parents=True)
        journal = dispatcher/"EVENTS.jsonl"
        for key in sorted(completed):
            attempt = stage+"-"+key[0]
            native = dict(status="success", split="online_stream", cost_status="recorded_actual_usage",
                          attempt_id=attempt, result=dict(prompt_tokens=3, completion_tokens=2))
            native["record_sha256"] = base.canonical(native)
            field = "member_index" if stage == "member_generation" else "combination_index"
            task_key = dict(question_id=key[0], generation_seed=42,
                            canonical_id=question_metadata[key[0]]["canonical_question_sha256"])
            task_key[field] = key[2]
            receipt = dict(
                schema="p2_execution_task_receipt_v1", cohort="e23_reference_supplement",
                scope_sha256=scope_sha, approval_sha256="approval", bindings=scope["bindings"],
                status="SUCCESS", stage=stage, software_test_only=False, transport_split="online_stream",
                aggregator_fingerprint="operator", task_key=task_key, native_record=native,
                native_record_sha256=native["record_sha256"], physical_attempt_id=attempt)
            folder = state["owned"] / scope_sha / stage / base.canonical(list(key))
            write(folder/"LAST_TASK_RECEIPT.json", receipt)
            write(folder/"ATTEMPT_STATE.json", dict(
                status="SUCCESS_COMMITTED", scope_sha256=scope_sha, stage=stage,
                task_key=list(key), native_record_sha256=native["record_sha256"]))
            write(folder/"records.jsonl", native)
            task = dict(stage=stage, question_id=key[0], generation_seed=42)
            task[field] = key[2]
            append(journal, dict(event="TASK_RESULT", task=task, status="SUCCESS",
                                 physical_attempt_id=attempt, native_record_sha256=native["record_sha256"],
                                 from_cache=False, actual_usage=dict(prompt_tokens=3, completion_tokens=2),
                                 at_unix=time.time()))
        write(dispatcher/"SUMMARY.json", dict(
            status="SAFELY_STOPPED" if worker_id.startswith("old") else "COMPLETE",
            planned=len(keys), completed=len(completed)))
    return state


def qualify():
    checks = {}
    old_counts, old_actor = base.COUNTS, base.actor_state
    base.COUNTS = dict(member_generation=6, aggregation=1)
    try:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            state = fixture(Path(directory))
            report, successes, fences = monitor.monitor(state)
            checks["retired_incomplete_subsets_do_not_block_completion"] = (
                report["ready_for_final_export"] and report["retired_source_workers"] == 2
                and report["current_assignment_workers"] == 5)
            checks["global_semantic_and_token_accounting_exact"] = (
                report["unique_success_native_attempts"] == 7
                and report["actual_success_native_tokens"]["total_tokens"] == 35)
            original_snapshot, snapshots = base.monitor, []
            def advancing_snapshot(value):
                result, successful, owned_fences = original_snapshot(value)
                snapshots.append(1)
                if len(snapshots) == 1:
                    result["unique_success_native_attempts"] += 1
                return result, successful, owned_fences
            base.monitor = advancing_snapshot
            try:
                advanced, _, _ = monitor.monitor(state)
                checks["success_count_snapshot_change_retried_once"] = (
                    len(snapshots) == 2 and advanced["success_snapshot_retries"] == 1)
            finally:
                base.monitor = original_snapshot
            snapshots.clear()
            def failed_advancing_snapshot(value):
                result, successful, owned_fences = original_snapshot(value)
                snapshots.append(1)
                result["unique_success_native_attempts"] += 1
                result["failure_events"] = [{"error_type": "SYNTHETIC_FAILURE"}]
                return result, successful, owned_fences
            base.monitor = failed_advancing_snapshot
            try:
                monitor.monitor(state)
            except RuntimeError:
                checks["failure_not_hidden_by_snapshot_retry"] = len(snapshots) == 1
            else:
                checks["failure_not_hidden_by_snapshot_retry"] = False
            finally:
                base.monitor = original_snapshot
            checks["retired_tasks_not_counted_as_current_remaining"] = (
                report["current_assignment_remaining"] == 0
                and all(row["remaining"] == 0 for row in report["workers"] if row["retired_source"]))
            closed = monitor.export_final(state, report, successes, fences)
            frozen = base.read(Path(closed["freeze_path"]))
            checks["original_export_schema_and_scope_preserved"] = (
                frozen["schema"] == "e23_supplement_outputs_frozen_v1"
                and frozen["completed_member_calls"] == 6 and frozen["completed_aggregate_calls"] == 1
                and frozen["execution_plan_sha256"] == "plan")
            checks["additive_dispatch_export_evidence"] = (
                base.read(Path(closed["dispatch_evidence_path"]))["dispatch_sha256"] == "dispatch")
            checks["final_export_idempotent"] = (
                monitor.export_final(state, report, successes, fences)["status"]
                == "EXISTING_SUPPLEMENT_OUTPUT_FREEZE_REUSED")
            active_entry = next(entry for entry, _, _ in state["manifests"] if entry["worker_id"] == "new0")
            journal = state["owned"]/"_dispatcher"/active_entry["manifest_ref"]["sha256"]/"EVENTS.jsonl"
            original_rows = journal.read_bytes()
            event = base.events(journal)[0][0]
            append(journal, dict(event, from_cache=True))
            replay, _, _ = monitor.monitor(state)
            checks["same_physical_replay_deduplicated"] = (
                replay["native_attempt_audit"]["replay_success_rows_deduplicated"] == 1
                and replay["actual_success_native_tokens"]["total_tokens"] == 35)
            journal.write_bytes(original_rows)
            append(journal, dict(event, physical_attempt_id="different-paid-attempt"))
            try:
                monitor.monitor(state)
            except RuntimeError:
                checks["duplicate_paid_semantic_key_rejected"] = True
            else:
                checks["duplicate_paid_semantic_key_rejected"] = False
            journal.write_bytes(original_rows)
            append(journal, dict(event="STOP_REQUIRES_RECONCILIATION", error_type="SyntheticFailure"))
            try:
                monitor.monitor(state)
            except RuntimeError:
                checks["unknown_failure_rejected"] = True
            else:
                checks["unknown_failure_rejected"] = False
            journal.write_bytes(original_rows)
            folder = state["owned"] / state["scope_sha"] / "member_generation" / base.canonical(["q1", 42, 4])
            fence = base.read(folder/"ATTEMPT_STATE.json")
            write(folder/"ATTEMPT_STATE.json", dict(fence, status="STOP_REQUIRES_RECONCILIATION"))
            try:
                monitor.monitor(state)
            except RuntimeError:
                checks["failed_owner_rejected"] = True
            else:
                checks["failed_owner_rejected"] = False
            write(folder/"ATTEMPT_STATE.json", fence)
            original_manifest = next(entry for entry, _, _ in state["manifests"] if entry["worker_id"] == "old0")
            write(state["root"]/"control"/"old0.start.json", dict(
                manifest_sha256=original_manifest["manifest_ref"]["sha256"], pid=1234))
            base.actor_state = lambda receipt: dict(pid=receipt["pid"], alive=True, state="SYNTHETIC_LIVE")
            try:
                monitor.monitor(state)
            except RuntimeError:
                checks["live_retired_actor_rejected"] = True
            else:
                checks["live_retired_actor_rejected"] = False
            base.actor_state = old_actor
    finally:
        base.COUNTS, base.actor_state = old_counts, old_actor
    if not all(checks.values()):
        raise AssertionError("Monitor v2 qualification failed " + repr(checks))
    return dict(
        status="PASS", checks=checks, checks_passed=len(checks),
        monitor_v2_sha256=base.sha(Path(monitor.__file__)),
        original_monitor_sha256=base.sha(Path(base.__file__)),
        new_llm_calls=0, gold_values_read=0,
    )


if __name__ == "__main__":
    print(json.dumps(qualify()))
