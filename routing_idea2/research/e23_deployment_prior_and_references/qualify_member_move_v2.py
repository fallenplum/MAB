"""Pure synthetic qualification of one-time same-manifest GPU migration."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import move_member_dispatch_v2 as move
import prepare_member_dispatch_v2 as dispatch
from qualify_member_dispatch_v2 import fixture
from supplement_worker import restored_successes


def must_reject(operation, exception):
    try:
        operation()
    except exception:
        return True
    raise AssertionError("An unsafe synthetic migration was accepted")


def actor_fixture(state, entry, manifest, keys):
    stem = Path(entry["manifest_ref"]["path"]).stem
    pid = 1000000020+entry["shard"]
    argv = [dispatch.PYTHON, "-B", "-u", str(state["root"]/"code/supplement_worker.py"),
            "--manifest", entry["manifest_ref"]["path"],
            "--manifest-sha256", entry["manifest_ref"]["sha256"], "--gpu", "5"]
    dispatch.freeze(state["root"]/"control"/f"{stem}.start.json", dict(
        worker_id=entry["worker_id"], stage="member_generation", gpu=5, pid=pid,
        manifest_sha256=entry["manifest_ref"]["sha256"], argv=argv, at_unix=3))
    key = sorted(keys)[0]
    native_sha = "1"*64
    journal = state["owned"]/"_dispatcher"/entry["manifest_ref"]["sha256"]/"EVENTS.jsonl"
    journal.parent.mkdir(parents=True)
    row = dict(event="TASK_RESULT", task=dict(stage="member_generation", question_id=key[0],
        generation_seed=key[1], member_index=key[2]), status="SUCCESS",
        actual_usage={"prompt_tokens": 3, "completion_tokens": 4},
        physical_attempt_id="synthetic-only", native_record_sha256=native_sha)
    journal.write_text(json.dumps(row)+"\n", encoding="utf-8")
    owner = state["owned"]/state["scope_sha"]/"member_generation"/dispatch.canonical(list(key))
    dispatch.freeze(owner/"ATTEMPT_STATE.json", dict(status="SUCCESS_COMMITTED",
        scope_sha256=state["scope_sha"], approval_sha256=state["plan"]["approval_ref"]["sha256"],
        stage="member_generation", task_key=list(key), native_record_sha256=native_sha))
    dispatch.freeze(owner/"LAST_TASK_RECEIPT.json", {})
    dispatch.freeze(owner/"records.jsonl", {})
    summary = dict(schema="e23_native_worker_summary_v1", status="SAFELY_STOPPED", stopped=True,
        pid=pid, scope_sha256=state["scope_sha"], manifest_sha256=entry["manifest_ref"]["sha256"],
        stage="member_generation", profile=4, planned=len(keys), completed=1, failures=[], at_unix=4)
    dispatch.freeze(journal.with_name("SUMMARY.json"), summary)
    return pid, key, journal


def qualify():
    checks = {}
    with TemporaryDirectory(dir=Path.cwd()) as name:
        root = Path(name)
        state = fixture(root)
        live = set()
        def probe(receipt):
            return dict(alive=receipt["pid"] in live, reason="synthetic exact argv ownership")
        with patch.object(dispatch, "load_parent", lambda _: state), patch.object(
                dispatch.original_monitor, "actor_state", probe):
            folder = root/"control/member4_dispatch_v2"
            dispatch.prepare(root, folder, 4)
            dispatch_path = folder/"DISPATCH_V2.json"
            _, frozen, new = dispatch.validate_dispatch(root, dispatch_path)
            entry, manifest, keys = new[2]
            pid, success, journal = actor_fixture(state, entry, manifest, keys)
            journal_before = journal.read_bytes()
            immutable_refs = [dispatch.reference(root/"EXECUTION_PLAN.json"),
                              dispatch.reference(dispatch_path), frozen["scope_ref"], frozen["approval_ref"],
                              entry["manifest_ref"], manifest["task_keys_ref"]]
            live.add(pid)
            signals = []
            def signal_actor(target, signum):
                assert target == pid and signum == move.signal.SIGTERM
                signals.append(target)
                live.remove(target)
            with patch.object(move.os, "kill", signal_actor):
                stopped = move.stop(root, dispatch_path, 2)
                again = move.stop(root, dispatch_path, 2)
            checks["stop_signals_only_exact_original_pid_once"] = signals == [pid]
            checks["repeated_stop_reuses_request_and_verifies_safe_exit"] = (
                stopped["stop_request_ref"] == again["stop_request_ref"]
                and again["status"] == "ORIGINAL_ACTOR_SAFELY_EXITED")
            live.add(pid)
            checks["active_old_actor_blocks_resume"] = must_reject(
                lambda: move.resume(root, dispatch_path, 2, 1), RuntimeError)
            live.remove(pid)
            checks["wrong_gpu_blocks_resume"] = must_reject(
                lambda: move.resume(root, dispatch_path, 2, 3), ValueError)
            checks["unapproved_shard_blocks_migration"] = must_reject(
                lambda: move.stop(root, dispatch_path, 1), ValueError)
            summary_path = journal.with_name("SUMMARY.json")
            correct_summary = summary_path.read_bytes()
            bad = json.loads(correct_summary)
            bad["failures"] = [{"synthetic": True}]
            summary_path.write_text(json.dumps(bad), encoding="utf-8")
            checks["failure_summary_blocks_resume"] = must_reject(
                lambda: move.resume(root, dispatch_path, 2, 1), RuntimeError)
            summary_path.write_bytes(correct_summary)
            launch_calls = []
            def launch(argv, **kwargs):
                launch_calls.append((argv, kwargs))
                return SimpleNamespace(pid=1000000100)
            with patch.object(move, "admission", lambda gpu: [dict(gpu=gpu, free_mib=39000)]), \
                    patch.object(move.subprocess, "Popen", launch):
                resumed = move.resume(root, dispatch_path, 2, 1)
            receipt = dispatch.read(resumed["resume_start_ref"]["path"])
            checks["resume_launches_once_with_same_manifest_and_new_gpu"] = (
                len(launch_calls) == 1 and receipt["manifest_sha256"] == entry["manifest_ref"]["sha256"]
                and receipt["argv"][-1] == "1" and receipt["argv"][5] == entry["manifest_ref"]["path"])
            checks["resume_receipt_top_level_control_visible_to_monitor"] = (
                Path(resumed["resume_start_ref"]["path"]).parent == root/"control"
                and len([item for item in dispatch.original_monitor.start_receipts(root)
                         if item.get("manifest_sha256") == entry["manifest_ref"]["sha256"]]) == 2)
            checks["same_journal_restores_exact_success_without_duplicate_call"] = (
                journal.read_bytes() == journal_before
                and restored_successes(journal, "member_generation", keys) == {success}
                and resumed["restored_successes"] == 1 and resumed["remaining"] == len(keys)-1)
            safe = dispatch.checked(receipt["safe_checkpoint_ref"])
            checks["safe_checkpoint_preserves_original_pid_and_success_digest"] = (
                safe["original_summary"]["pid"] == pid
                and safe["checkpoint"]["successful_count"] == 1)
            checks["duplicate_resume_rejected"] = must_reject(
                lambda: move.resume(root, dispatch_path, 2, 1), FileExistsError)
            checks["stop_after_resume_rejected"] = must_reject(
                lambda: move.stop(root, dispatch_path, 2), FileExistsError)
            checks["parent_scope_approval_manifest_tasks_unchanged"] = all(
                dispatch.reference(item["path"]) == item for item in immutable_refs)
            _, _, candidates = dispatch.validate_dispatch(root, dispatch_path)
            checks["dispatch_validation_still_accepts_same_journal_continuation"] = len(candidates) == 4
    with patch.object(move.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="1, 39000, 0\n")), \
            patch.object(move.time, "sleep", lambda _: None):
        checks["three_gpu_admission_observations"] = len(move.admission(1)) == 3
    with patch.object(move.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="1, 1000, 0\n")):
        checks["insufficient_memory_blocks_admission"] = must_reject(lambda: move.admission(1), RuntimeError)
    if not all(checks.values()):
        raise AssertionError("Synthetic member migration qualification failed")
    return dict(schema="e23_member_move_qualification_v1", status="PASS", checks=checks,
        checks_passed=len(checks), source_sha256={name: dispatch.sha(Path(__file__).with_name(name))
            for name in ("move_member_dispatch_v2.py", "qualify_member_move_v2.py")},
        physical_llm_calls=0, gold_reads=0, production_processes_signaled=0,
        production_processes_started=0, synthetic_only=True)


if __name__ == "__main__":
    print(json.dumps(qualify()))
