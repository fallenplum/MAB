"""Exercise the real dispatch validator and monitor together on synthetic files."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import execution_monitor as original
import execution_monitor_v2 as monitor
import prepare_member_dispatch_v2 as dispatch
from qualify_member_dispatch_v2 import fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def qualify():
    checks = {}
    old_loader, old_counts = dispatch.load_parent, original.COUNTS
    try:
        with TemporaryDirectory(dir=Path.cwd()) as name:
            root = Path(name)
            state = fixture(root)
            state["plan_sha256"] = dispatch.sha(root/"EXECUTION_PLAN.json")
            for entry, _, _ in state["manifests"]:
                path = state["owned"]/"_dispatcher"/entry["manifest_ref"]["sha256"]/"EVENTS.jsonl"
                if not path.is_file():
                    continue
                rows, _ = original.events(path)
                path.write_text("".join(json.dumps(dict(row, from_cache=False))+"\n"
                                        for row in rows), encoding="utf-8")
            dispatch.load_parent = lambda _: state
            original.COUNTS = {stage: len(keys) for stage, keys in state["required"].items()}
            output = root/"control/member4_dispatch_v2"
            dispatch.prepare(root, output, 4)
            dispatch_path = output/"DISPATCH_V2.json"
            base_context = monitor.context(root, dispatch_path)
            checks["real_validator_context_contains_two_retired_four_successors"] = (
                len(base_context["manifests"]) == len(state["manifests"])+4
                and len(base_context["retired_worker_ids"]) == 2)
            entry, manifest, keys = base_context["manifests"][len(state["manifests"])]
            key = sorted(keys)[0]
            owner = state["owned"]/state["scope_sha"]/"member_generation"/dispatch.canonical(list(key))
            reservation = dict(
                status="RESERVED_BEFORE_BACKEND", stage="member_generation",
                scope_sha256=state["scope_sha"], task_key=list(key),
                approval_sha256=state["plan"]["approval_ref"]["sha256"])
            write(owner/"ATTEMPT_STATE.json", reservation)
            reserved_context = monitor.context(root, dispatch_path)
            reserved_report, _, _ = monitor.monitor(reserved_context)
            checks["first_successor_reservation_allows_all_sibling_validation"] = (
                len(reserved_context["manifests"]) == len(state["manifests"])+4
                and reserved_report["stages"]["member_generation"]["completed"] == 8
                and reserved_report["unresolved_owner_states"] == {"RESERVED_BEFORE_BACKEND": 1})
            native_sha = "1"*64
            write(owner/"ATTEMPT_STATE.json", dict(
                reservation, status="SUCCESS_COMMITTED", native_record_sha256=native_sha))
            write(owner/"LAST_TASK_RECEIPT.json", {})
            write(owner/"records.jsonl", {})
            path = state["owned"]/"_dispatcher"/entry["manifest_ref"]["sha256"]/"EVENTS.jsonl"
            write(path, dict(
                event="TASK_RESULT", task=dict(stage="member_generation", question_id=key[0],
                                              generation_seed=key[1], member_index=key[2]),
                status="SUCCESS", physical_attempt_id="new-physical-attempt",
                native_record_sha256=native_sha, from_cache=False,
                actual_usage=dict(prompt_tokens=1, completion_tokens=1)))
            committed_context = monitor.context(root, dispatch_path)
            committed_report, _, _ = monitor.monitor(committed_context)
            checks["first_successor_commit_allows_all_sibling_validation"] = (
                len(committed_context["manifests"]) == len(state["manifests"])+4
                and committed_report["stages"]["member_generation"]["completed"] == 9
                and committed_report["actual_success_native_tokens"]["total_tokens"] == 18
                and committed_report["current_assignment_remaining"] == sum(original.COUNTS.values())-9)
            checks["retired_success_snapshot_is_unchanged_after_successor_commit"] = (
                committed_context["dispatch"]["completed_at_preparation"] == 8
                and sum(row["completed"] for row in committed_report["workers"] if row["retired_source"]) == 8)
            write(owner/"ATTEMPT_STATE.json", dict(reservation, approval_sha256="foreign-approval"))
            try:
                monitor.context(root, dispatch_path)
            except RuntimeError:
                checks["foreign_successor_approval_rejected_by_real_validator"] = True
            else:
                checks["foreign_successor_approval_rejected_by_real_validator"] = False
            write(owner/"ATTEMPT_STATE.json", dict(reservation, status="STOP_REQUIRES_RECONCILIATION"))
            try:
                monitor.context(root, dispatch_path)
            except RuntimeError:
                checks["failed_successor_rejected_by_real_validator"] = True
            else:
                checks["failed_successor_rejected_by_real_validator"] = False
    finally:
        dispatch.load_parent, original.COUNTS = old_loader, old_counts
    if not all(checks.values()):
        raise AssertionError("Dispatch/monitor integration qualification failed " + repr(checks))
    return dict(status="PASS", checks=checks, checks_passed=len(checks),
                source_sha256={name: dispatch.sha(Path(__file__).with_name(name)) for name in (
                    "prepare_member_dispatch_v2.py", "execution_monitor_v2.py",
                    "qualify_dispatch_monitor_integration.py")},
                validator_function_replaced=False, only_parent_fixture_loader_replaced=True,
                new_llm_calls=0, reference_values_read=0)


if __name__ == "__main__":
    print(json.dumps(qualify()))
