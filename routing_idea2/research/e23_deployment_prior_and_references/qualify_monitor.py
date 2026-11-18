"""Synthetic native-output monitor and export qualification."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import execution_monitor as monitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False)+"\n", encoding="utf-8")


def fixture(root):
    scope = dict(bindings={"synthetic": "synthetic"}, operator_fingerprint="operator",
                 question_keys=[dict(question_id="q", canonical_question_sha256="canonical")])
    scope_sha = monitor.canonical(scope)
    required = {stage: {("q", 42, 0)} for stage in monitor.STAGES}
    state = dict(root=root, scope=scope, scope_sha=scope_sha, owned=root/"owned",
                 required=required, plan_sha256="plan", question_metadata={"q": scope["question_keys"][0]},
                 plan=dict(cohort="e23_reference_supplement", scope_ref={"sha256": "scope"},
                           approval_ref={"sha256": "approval"}), manifests=[])
    for stage in monitor.STAGES:
        key = ("q", 42, 0)
        field = "member_index" if stage == "member_generation" else "combination_index"
        folder = state["owned"] / scope_sha / stage / monitor.canonical(list(key))
        native = dict(status="success", split="online_stream", cost_status="recorded_actual_usage",
                      attempt_id=stage, result=dict(prompt_tokens=3, completion_tokens=2))
        native["record_sha256"] = monitor.canonical(native)
        fence = dict(status="SUCCESS_COMMITTED", scope_sha256=scope_sha, stage=stage,
                     task_key=list(key), native_record_sha256=native["record_sha256"])
        task = dict(question_id="q", generation_seed=42, canonical_id="canonical")
        task[field] = 0
        receipt = dict(
            schema="p2_execution_task_receipt_v1", cohort="e23_reference_supplement",
            scope_sha256=scope_sha, approval_sha256="approval", bindings=scope["bindings"],
            status="SUCCESS", stage=stage, software_test_only=False, transport_split="online_stream",
            aggregator_fingerprint="operator", task_key=task, native_record=native,
            native_record_sha256=native["record_sha256"], physical_attempt_id=stage)
        write(folder/"ATTEMPT_STATE.json", fence)
        write(folder/"LAST_TASK_RECEIPT.json", receipt)
        write(folder/"records.jsonl", native)
        event_task = dict(stage=stage, question_id="q", generation_seed=42)
        event_task[field] = 0
        write(state["owned"]/"_dispatcher"/stage/"EVENTS.jsonl", dict(
            event="TASK_RESULT", task=event_task, status="SUCCESS", physical_attempt_id=stage,
            native_record_sha256=native["record_sha256"],
            actual_usage=dict(prompt_tokens=3, completion_tokens=2), at_unix=time.time()))
        state["manifests"].append((
            dict(stage=stage, profile=0, worker_id=stage, manifest_ref={"sha256": stage}),
            {}, required[stage]))
    return state


def qualify():
    original_counts = monitor.COUNTS
    monitor.COUNTS = {stage: 1 for stage in monitor.STAGES}
    checks = {}
    try:
        with TemporaryDirectory(dir=Path.cwd()) as name:
            state = fixture(Path(name))
            report, successes, fences = monitor.monitor(state)
            checks["complete_counts_and_tokens"] = (
                report["ready_for_final_export"] and report["actual_success_native_tokens"]["total_tokens"] == 10)
            closed = monitor.export_final(state, report, successes, fences)
            checks["exact_export_counts"] = all(row["records"] == 1 for row in closed["records"].values())
            checks["freeze_binding"] = monitor.read(Path(closed["freeze_path"]))["completed_member_calls"] == 1
            checks["closed_export_idempotent"] = (
                monitor.export_final(state, report, successes, fences)["status"] ==
                "EXISTING_SUPPLEMENT_OUTPUT_FREEZE_REUSED")
            write(state["owned"]/"_dispatcher/aggregation/EVENTS.jsonl",
                  dict(event="STOP_REQUIRES_RECONCILIATION", error_type="SyntheticError"))
            failed, _, _ = monitor.monitor(state)
            checks["failure_prevents_ready"] = not failed["ready_for_final_export"]
            try:
                monitor.export_final(state, failed, successes, fences)
            except PermissionError:
                checks["failed_export_refused"] = True
            else:
                checks["failed_export_refused"] = False
            owner = state["owned"] / state["scope_sha"] / "member_generation" / monitor.canonical(["q", 42, 0])
            receipt = monitor.read(owner/"LAST_TASK_RECEIPT.json")
            receipt["native_record"]["result"]["prompt_tokens"] = 4
            write(owner/"LAST_TASK_RECEIPT.json", receipt)
            try:
                monitor.verified_owner(state, "member_generation", ("q", 42, 0),
                                       successes["member_generation"][("q", 42, 0)],
                                       fences[("member_generation", ("q", 42, 0))])
            except ValueError:
                checks["mutated_native_refused"] = True
            else:
                checks["mutated_native_refused"] = False
    finally:
        monitor.COUNTS = original_counts
    if not all(checks.values()):
        raise AssertionError("Monitor qualification failed")
    return dict(status="PASS", checks=checks, checks_passed=len(checks),
                monitor_sha256=monitor.sha(Path(monitor.__file__)), gold_reads=0, new_llm_calls=0)


if __name__ == "__main__":
    print(json.dumps(qualify()))
