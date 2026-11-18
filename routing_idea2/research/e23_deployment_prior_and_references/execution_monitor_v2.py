"""Monitor original and replacement E23 actors under one unchanged paid scope."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
import os

import execution_monitor as original


def context(root, dispatch_path=None):
    from prepare_member_dispatch_v2 import validate_dispatch

    root = Path(root)
    dispatch_path = (Path(dispatch_path) if dispatch_path is not None else
                     root / "control/member4_dispatch_v2/DISPATCH_V2.json")
    state, dispatch, replacements = validate_dispatch(root, dispatch_path)
    retired_ids = {row["worker_id"] for row in dispatch["retired"]}
    if (dispatch.get("schema") != "e23_member_dispatch_v2"
            or len(retired_ids) != 2 or len(replacements) != 4
            or any(entry["stage"] != "member_generation" or entry["profile"] != 4
                   for entry, _, _ in replacements)):
        raise ValueError("An exact two-source to four-actor member4 dispatch is required")
    state = dict(state)
    state["manifests"] = list(state["manifests"]) + list(replacements)
    state["dispatch"] = dispatch
    state["dispatch_path"] = dispatch_path
    state["dispatch_sha256"] = original.sha(dispatch_path)
    state["retired_worker_ids"] = retired_ids
    worker_ids = [entry["worker_id"] for entry, _, _ in state["manifests"]]
    if len(worker_ids) != len(set(worker_ids)):
        raise ValueError("The source registry repeats a worker identity")
    return state


def audit_journal_attempts(state):
    physical, semantic = {}, {}
    sources, repeated = Counter(), 0
    for entry, _, keys in state["manifests"]:
        stage = entry["stage"]
        folder = state["owned"] / "_dispatcher" / entry["manifest_ref"]["sha256"]
        journal, partial = original.events(folder / "EVENTS.jsonl")
        if entry["worker_id"] in state["retired_worker_ids"] and partial:
            raise ValueError("A retired member journal is not closed")
        for event in journal:
            if event.get("event") == "STOP_REQUIRES_RECONCILIATION" or (
                    event.get("event") == "TASK_RESULT" and event.get("status") != "SUCCESS"):
                raise RuntimeError("A native failure requires cost reconciliation before continuation export")
            if event.get("event") != "TASK_RESULT":
                continue
            key = original.task_key(event, stage)
            if key not in keys:
                raise ValueError("A native success exceeds its exact source task subset")
            attempt = event.get("physical_attempt_id")
            usage = original.actual_usage(event.get("actual_usage"))
            if (not isinstance(attempt, str) or not attempt
                    or type(event.get("from_cache")) is not bool):
                raise ValueError("Explicit physical-attempt and cache evidence is required")
            signature = (
                stage, key, event.get("native_record_sha256"),
                usage["prompt_tokens"], usage["completion_tokens"],
            )
            previous = physical.setdefault(attempt, signature)
            if previous != signature:
                raise ValueError("A physical native attempt has conflicting task or cost evidence")
            identity = (stage, key)
            previous_attempt = semantic.setdefault(identity, attempt)
            if previous_attempt != attempt:
                raise RuntimeError("A semantic native key has multiple paid physical attempts")
            repeated += int(identity in sources)
            sources[identity] += 1
    return dict(unique_physical_attempts=len(physical), unique_semantic_success_keys=len(semantic),
                replay_success_rows_deduplicated=repeated)


def monitor(state):
    for snapshot_attempt in range(2):
        attempt_audit = audit_journal_attempts(state)
        report, successes, fences = original.monitor(state)
        if report["failure_events"] or report["additional_paid_success_attempts"]:
            raise RuntimeError("Failed or repeated paid native outputs require reconciliation")
        failed_fences = {key: count for key, count in report["unresolved_owner_states"].items()
                         if key not in {"RESERVED_BEFORE_BACKEND", "OWNER_WITHOUT_FENCE"}}
        if failed_fences:
            raise RuntimeError("Failed or unknown native owner states require cost reconciliation")
        audited_count = attempt_audit["unique_physical_attempts"]
        current_count = report["unique_success_native_attempts"]
        if audited_count == current_count:
            break
        if current_count < audited_count or snapshot_attempt:
            raise ValueError("Physical-attempt deduplication disagrees with the original cost ledger")
    retired = state["retired_worker_ids"]
    unknown = set()
    dispatcher = state["owned"] / "_dispatcher"
    expected = {entry["manifest_ref"]["sha256"] for entry, _, _ in state["manifests"]}
    if dispatcher.is_dir():
        unknown = {path.name for path in dispatcher.iterdir()
                   if path.is_dir() and path.name not in expected}
    if unknown:
        raise RuntimeError("An unregistered native dispatcher requires reconciliation")
    retired_rows, active_rows = [], []
    for worker in report["workers"]:
        worker["retired_source"] = worker["worker_id"] in retired
        if worker["retired_source"]:
            if any(actor["alive"] is not False for actor in worker["actors"]):
                raise RuntimeError("A retired original member actor is still alive or cannot be verified")
            worker["historical_unfinished_tasks_reassigned"] = worker["remaining"]
            worker["remaining"] = 0
            retired_rows.append(worker)
        else:
            active_rows.append(worker)
    report.update(
        schema="e23_native_execution_status_v2",
        dispatch_sha256=state["dispatch_sha256"],
        original_execution_plan_sha256=state["plan_sha256"],
        retired_source_workers=len(retired_rows),
        current_assignment_workers=len(active_rows),
        source_manifest_count=len(state["manifests"]),
        current_assignment_remaining=sum(worker["remaining"] for worker in active_rows),
        live_actor_count=sum(actor["alive"] is True for worker in report["workers"]
                             for actor in worker["actors"]),
        native_attempt_audit=attempt_audit,
        success_snapshot_retries=snapshot_attempt,
        original_scope_changed=False,
    )
    return report, successes, fences


def export_final(state, report, successes, fences):
    result = original.export_final(state, report, successes, fences)
    evidence = dict(
        schema="e23_dispatch_v2_export_evidence_v1",
        scope_canonical_sha256=state["scope_sha"],
        original_execution_plan_sha256=state["plan_sha256"],
        dispatch_sha256=state["dispatch_sha256"],
        native_output_freeze_sha256=result["freeze_sha256"],
        original_exporter_sha256=original.sha(Path(original.__file__)),
        monitor_v2_sha256=original.sha(Path(__file__)),
        registered_sources=len(state["manifests"]),
        retired_sources=len(state["retired_worker_ids"]),
        native_attempt_audit=report["native_attempt_audit"],
        original_scope_changed=False,
    )
    path = state["root"] / "exports/DISPATCH_V2_EXPORT_EVIDENCE.json"
    if path.exists():
        if original.read(path) != evidence:
            raise ValueError("The closed dispatch export evidence changed")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return dict(result, dispatch_evidence_path=str(path), dispatch_evidence_sha256=original.sha(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path)
    parser.add_argument("--export-final", action="store_true")
    args = parser.parse_args()
    state = context(args.root, args.dispatch)
    report, successes, fences = monitor(state)
    if args.export_final:
        report["final_export"] = export_final(state, report, successes, fences)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
