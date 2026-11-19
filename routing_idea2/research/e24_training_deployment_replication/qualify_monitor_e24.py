"""Qualify E24 native progress and closure against synthetic signed task evidence."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import monitor_e24 as monitor


COUNTS = {"member_generation": 2, "aggregation": 2}
QUESTION = "toy"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False)+"\n", encoding="utf-8")
    return dict(path=str(path), sha256=monitor.sha(path))


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value)+"\n")


def context_fixture(root):
    scope = dict(schema=monitor.SCOPE_SCHEMA, cohort=monitor.COHORT, transport_split="online_stream",
                 bindings={"input_sha256": "toy"}, operator_fingerprint="toy-operator",
                 question_keys=[dict(question_id=QUESTION, canonical_question_sha256="toy-canonical")],
                 member_execution_keys=[[QUESTION, 42, 0], [QUESTION, 137, 0]],
                 aggregate_execution_keys=[[QUESTION, 42, 1], [QUESTION, 137, 1]], execution_limits=COUNTS)
    scope_ref = write(root/"inputs/EXECUTION_SCOPE.json", scope)
    scope_sha = monitor.canonical(scope)
    approval = dict(schema=monitor.APPROVAL_SCHEMA, status="APPROVED", cohort=monitor.COHORT,
                    approved_output_root=str(root/"owned"), scope_sha256_by_role={"execution":scope_sha},
                    bindings=scope["bindings"], software_test_only=False, execution_limits=COUNTS,
                    sealed200_answer_access=False, operations=list(monitor.STAGES))
    approval_ref = write(root/"inputs/AUTHORIZATION.json", approval)
    plan = dict(schema=monitor.PLAN_SCHEMA, cohort=monitor.COHORT, root=str(root), scope_ref=scope_ref,
                scope_canonical_sha256=scope_sha, approval_ref=approval_ref, input_questions=1,
                required_new_member_calls=COUNTS["member_generation"], required_new_aggregate_calls=COUNTS["aggregation"],
                manifests=[])
    for stage in monitor.STAGES:
        member = stage == "member_generation"
        key_ref = write(root/"inputs"/(stage+"_keys.json"), scope["member_execution_keys" if member else "aggregate_execution_keys"])
        ref = write(root/"manifests"/(stage+".json"), dict(worker_id=stage, stage=stage, scope_ref=scope_ref,
            approval_ref=approval_ref, task_keys_ref=key_ref, owned_output_root=approval["approved_output_root"],
            software_test_only=False))
        plan["manifests"].append(dict(worker_id=stage, stage=stage, profile=0, task_count=COUNTS[stage], manifest_ref=ref))
    write(root/"EXECUTION_PLAN.json", plan)
    return monitor.context(root, counts=COUNTS, question_count=1)


def outputs_fixture(state):
    for entry, keys in state["manifests"]:
        stage = entry["stage"]
        folder = state["owned"] / "_dispatcher" / entry["manifest_ref"]["sha256"]
        folder.mkdir(parents=True)
        for question, seed, action in sorted(keys):
            owner = owner_directory(state, stage, (question, seed, action))
            physical = f"toy:{question}:{seed}:{stage}:{action}"
            job = "job:" + physical
            usage = dict(prompt_tokens=3+int(seed==137), completion_tokens=2+int(stage=="aggregation"))
            native = dict(status="success", split="online_stream", cost_status="recorded_actual_usage",
                          attempt_id=physical, result=usage, artificial_payload="纯合成非生产回执")
            native["record_sha256"] = monitor.canonical(native)
            approval_sha = state["plan"]["approval_ref"]["sha256"]
            field = "member_index" if stage=="member_generation" else "combination_index"
            task = dict(question_id=question, generation_seed=seed, **{field:action})
            receipt = dict(schema="p2_execution_task_receipt_v1", cohort=monitor.COHORT, scope_sha256=state["scope_sha"],
                           status="SUCCESS", stage=stage, approval_sha256=approval_sha, bindings=state["scope"]["bindings"],
                           aggregator_fingerprint=state["scope"]["operator_fingerprint"], transport_split="online_stream",
                           approved_output_root=str(state["owned"]), aggregation_seed=42, native_job_id=job,
                           task_key=dict(task, canonical_id="toy-canonical"), software_test_only=False,
                           native_record=native, native_record_sha256=native["record_sha256"], physical_attempt_id=physical)
            fence = dict(scope_sha256=state["scope_sha"], stage=stage, approval_sha256=approval_sha,
                         status="SUCCESS_COMMITTED", task_key=[question, seed, action], software_test_only=False,
                         native_job_id=job, native_record_sha256=native["record_sha256"])
            write(owner/"ATTEMPT_STATE.json", fence)
            write(owner/"LAST_TASK_RECEIPT.json", receipt)
            write(owner/"records.jsonl", native)
            event_task = dict(task, stage=stage)
            append(folder/"EVENTS.jsonl", dict(event="TASK_ENTER", task=event_task, at_unix=time.time()))
            append(folder/"EVENTS.jsonl", dict(event="TASK_RESULT", task=event_task, status="SUCCESS", from_cache=False,
                  physical_attempt_id=physical, native_record_sha256=native["record_sha256"], actual_usage=usage,
                  at_unix=time.time()))


def owner_directory(state, stage, key):
    return state["owned"] / state["scope_sha"] / stage / monitor.canonical(list(key))


def rejected(action, exception):
    try:
        action()
    except exception:
        return True
    return False


def qualify():
    checks = {}
    with TemporaryDirectory(dir=Path.cwd()) as path:
        state = context_fixture(Path(path))
        report, _ = monitor.journal_progress(state)
        checks["empty_scope_has_zero_successes"] = report["unique_success_native_attempts"] == 0
        checks["progress_does_not_probe_fences"] = not report["owner_state_checked"]
        outputs_fixture(state)
        report, successes = monitor.journal_progress(state)
        checks["exact_cost_and_keys_closed_in_journal"] = (report["ready_for_owner_check"] and
            report["unique_success_native_attempts"] == 4 and report["actual_success_native_tokens"] ==
            dict(prompt_tokens=14,completion_tokens=10,total_tokens=24) and report["stage_tokens"]["aggregation"]["total_tokens"]==13)
        checks["journal_alone_cannot_export"] = rejected(lambda: monitor.export_final(state, report, successes, {}), PermissionError)
        observer = state["owned"] / "_dispatcher" / state["manifests"][0][0]["manifest_ref"]["sha256"] / "EVENTS.jsonl"
        clean = observer.read_bytes()
        event = json.loads(clean.splitlines()[-1])
        append(observer, dict(event, from_cache=True))
        replay, _ = monitor.journal_progress(state)
        checks["one_cached_replay_does_not_rebill"] = (replay["repeated_success_rows"]==1
            and replay["cached_success_journal_rows"]==1 and replay["actual_success_native_tokens"]["total_tokens"]==24)
        append(observer, dict(event, physical_attempt_id="new-attempt"))
        checks["replayed_semantic_key_with_fresh_physical_call_rejected"] = rejected(lambda: monitor.journal_progress(state), ValueError)
        observer.write_bytes(clean+b'{"partial')
        partial, _ = monitor.journal_progress(state)
        checks["partial_unsettled_journal_blocks_closure"] = (partial["partial_journals"]==1 and not partial["ready_for_owner_check"])
        observer.write_bytes(clean)
        append(observer, dict(event="TASK_ENTER", task=event["task"]))
        unsettled, _ = monitor.journal_progress(state)
        checks["unclosed_native_entry_blocks_export"] = unsettled["unresolved_enter_events"] == 1 and not unsettled["ready_for_owner_check"]
        observer.write_bytes(clean)
        append(observer, dict(event="STOP_REQUIRES_RECONCILIATION", task=event["task"], error_type="ToyError"))
        failed, failures = monitor.journal_progress(state)
        checks["unknown_cost_failure_blocks_closure"] = failed["failure_events_without_known_cost"]==1 and not failed["ready_for_owner_check"]
        observer.write_bytes(clean)
        report, successes = monitor.journal_progress(state)
        first = owner_directory(state, "member_generation", (QUESTION,42,0)) / "ATTEMPT_STATE.json"
        correct_fence = monitor.read(first)
        write(first, dict(correct_fence,status="RESERVED_BEFORE_BACKEND"))
        fences = monitor.owner_fences(state, report)
        checks["open_fence_blocks_closure"] = (report["unresolved_owner_states"] == {"RESERVED_BEFORE_BACKEND":1}
            and rejected(lambda: monitor.export_final(state,report,successes,fences), PermissionError))
        write(first, correct_fence)
        report, successes = monitor.journal_progress(state)
        fences = monitor.owner_fences(state, report)
        checks["exhaustive_fences_settle_approved_keys"] = report["ready_for_final_export"] and report["settled_owner_entries"]==4
        extra_scope = state["owned"] / "unexpected-scope"
        extra_scope.mkdir()
        monitor.owner_fences(state, report)
        checks["out_of_scope_native_owner_blocks_export"] = report["extra_owner_entries"] == 1 and not report["ready_for_final_export"]
        extra_scope.rmdir()
        report, successes = monitor.journal_progress(state)
        fences = monitor.owner_fences(state, report)
        payment = first.parent/"LAST_TASK_RECEIPT.json"
        correct_receipt = monitor.read(payment)
        write(payment, dict(correct_receipt, native_job_id="wrong"))
        checks["tampered_native_job_binding_refused"] = rejected(lambda: monitor.export_final(state,report,successes,fences), ValueError)
        write(payment, correct_receipt)
        prior_size = monitor.SHARD_SIZE
        monitor.SHARD_SIZE = 1
        try:
            closed = monitor.export_final(state, report, successes, fences)
            freeze_path = Path(closed["freeze_path"])
            frozen = monitor.read(freeze_path)
            checks["exact_shard_total_count_and_freeze_binding"] = (frozen["completed_member_calls"]==2 and frozen["completed_aggregate_calls"]==2
                and frozen["scope_sha256"]==state["plan"]["scope_ref"]["sha256"] and all(len(entries)==2 for entries in frozen["records"].values()))
            actual = []
            for stage, entries in frozen["records"].items():
                for entry in entries:
                    monitor.safe_export_file(freeze_path.parent, entry)
                    with gzip.open(freeze_path.parent/entry["file"], "rt", encoding="utf-8") as stream:
                        actual.extend(json.loads(line) for line in stream)
            checks["scores_can_consume_full_raw_native_receipts"] = (len(actual)==4 and all(row.get("native_record",{}).get("artificial_payload") for row in actual))
            second = monitor.export_final(state, report, successes, fences)
            checks["validated_freeze_is_reusable"] = second["status"] == "EXISTING_E24_OUTPUT_FREEZE_REUSED"
            damaged = freeze_path.parent / frozen["records"]["member_generation"][0]["file"]
            damaged.write_bytes(damaged.read_bytes()+b"invalid")
            checks["damaged_frozen_gzip_rejected"] = rejected(lambda:monitor.export_final(state,report,successes,fences), ValueError)
        finally:
            monitor.SHARD_SIZE = prior_size
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(status="PASS", checks=checks, checks_passed=len(checks), monitor_sha256=monitor.sha(Path(monitor.__file__)),
                qualification_sha256=monitor.sha(Path(__file__)), gold_reads=0, new_llm_calls=0)


if __name__ == "__main__":
    print(json.dumps(qualify(), indent=2))
