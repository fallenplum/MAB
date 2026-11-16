"""Run an unused original shard or explicitly resume its settled stopped prefix.

The caller admits the physical GPU slot and stops the former process externally.
This module sends no signals and never takes over a RUNNING shard. The original
manifest, worker, Store, plan and per-manifest journal remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import e18_gpu_dispatcher as dispatcher
import e18_gpu_worker as worker
import run_original_plan_shard as original


ORIGINAL_RUNNER_SHA = "67daa49781cd9bf2849874e95f1aa784d0ce149a1b5003ee4e5e106cf1aa3dee"


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def process_is_alive(pid):
    """Inspect process metadata without sending even a zero-numbered signal."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("A positive former process ID is required")
    if os.name != "nt":
        proc = Path("/proc")
        if not proc.is_dir():
            raise RuntimeError("Process-exit inspection requires Linux proc metadata")
        return (proc / str(pid)).exists()
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:
            return False
        raise PermissionError("Former-process exit cannot be established")
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise PermissionError("Former-process exit cannot be established")
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def stopped_state(status_path, summary_path, manifest_sha, planned, software_test_only):
    state = worker.read_json(status_path)
    summary = state.get("summary")
    if state.get("status") != "WAITING_FOR_DEPENDENCIES":
        raise RuntimeError("Only a safely stopped WAITING shard can be resumed")
    if state.get("manifest_sha256") != manifest_sha or not isinstance(summary, dict):
        raise ValueError("Stopped shard identity or worker summary differs")
    if (
        summary.get("manifest_sha256") != manifest_sha
        or summary.get("schema") != "e18_gpu_worker_summary_v1"
        or summary.get("stage") != "member_generation"
        or summary.get("software_test_only") is not software_test_only
        or summary.get("stop_requested") is not True
        or summary.get("failures") != []
        or type(summary.get("completed")) is not int
        or type(summary.get("planned")) is not int
        or summary["planned"] != planned
        or not 0 < summary["completed"] < planned
    ):
        raise RuntimeError("Stopped summary does not establish a clean partial checkpoint")
    if not summary_path.is_file() or worker.read_json(summary_path) != summary:
        raise ValueError("Original journal SUMMARY differs from the stopped plan status")
    return state, summary


def audit_committed_prefix(manifest, manifest_sha, completed, *, software_test_only=False):
    """Check every original native owner before any worker can resume its Store."""
    module, adapter, keys, _, _, _ = worker.load_worker(
        manifest, software_test_only=software_test_only
    )
    scope_sha = module.canonical_digest(adapter.scope)
    approval_sha = manifest["approval_ref"]["sha256"]
    output = Path(manifest["owned_output_root"]).resolve()
    committed_hashes = []
    for position, key in enumerate(keys):
        owner = output / scope_sha / "member_generation" / module.canonical_digest(list(key))
        if not owner.exists():
            if position < completed:
                raise RuntimeError("Stopped summary counts a missing native commit")
            continue
        fence_path = owner / "ATTEMPT_STATE.json"
        receipt_path = owner / "LAST_TASK_RECEIPT.json"
        records_path = owner / "records.jsonl"
        if not fence_path.exists() and not receipt_path.exists():
            children = list(owner.iterdir())
            if all(path.name == "records.jsonl" and path.stat().st_size == 0 for path in children):
                if position < completed:
                    raise RuntimeError("Stopped summary counts an unexecuted native owner")
                continue
        if not all(path.is_file() for path in (fence_path, receipt_path, records_path)):
            raise RuntimeError("An existing native owner has incomplete commit evidence")
        fence, receipt = worker.read_json(fence_path), worker.read_json(receipt_path)
        if fence.get("status") != "SUCCESS_COMMITTED" or receipt.get("status") != "SUCCESS":
            raise RuntimeError("Reserved, failed or unknown native attempts require reconciliation")
        expected = dict(
            scope_sha256=scope_sha,
            approval_sha256=approval_sha,
            stage="member_generation",
            software_test_only=software_test_only,
        )
        if (
            any(fence.get(name) != value for name, value in expected.items())
            or fence.get("task_key") != list(key)
            or any(receipt.get(name) != value for name, value in expected.items())
            or receipt.get("schema") != "e18_execution_task_receipt_v2"
        ):
            raise ValueError("Native checkpoint scope, mode or task fence differs")
        task = receipt.get("task_key", {})
        if (task.get("question_id"), task.get("generation_seed"), task.get("member_index")) != key:
            raise ValueError("Native receipt semantic task differs")
        native = receipt.get("native_record", {})
        native_sha = module.canonical_digest(
            {name: value for name, value in native.items() if name != "record_sha256"}
        )
        if (
            native.get("record_sha256") != native_sha
            or fence.get("native_record_sha256") != native_sha
            or receipt.get("native_record_sha256") != native_sha
            or native.get("status") != "success"
            or native.get("cost_status") != "recorded_actual_usage"
            or fence.get("native_job_id") != native.get("job_id")
            or receipt.get("native_job_id") != native.get("job_id")
            or receipt.get("physical_attempt_id") != native.get("attempt_id")
        ):
            raise ValueError("Native checkpoint SHA, usage or physical attempt differs")
        with records_path.open(encoding="utf-8-sig", newline="") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if records != [native]:
            raise RuntimeError("Native Store contains another or unsettled physical attempt")
        adapter.checked_member_dependency(*key, native)
        if position >= completed:
            raise RuntimeError("Native commits do not form the stopped summary's exact prefix")
        committed_hashes.append(native_sha)
    if len(committed_hashes) != completed:
        raise RuntimeError("The exact native committed prefix is incomplete")
    return dict(
        manifest_sha256=manifest_sha,
        execution_scope_sha256=scope_sha,
        approval_sha256=approval_sha,
        original_tasks=len(keys),
        checkpoint_successes=completed,
        remaining_tasks=len(keys) - completed,
        pending_or_unknown_attempts=0,
        committed_native_hashes_sha256=canonical_digest(committed_hashes),
    )


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_one(
    plan,
    plan_sha,
    *,
    manifest_sha=None,
    member_index=2,
    gpu="0",
    gpu_lock_root=None,
    resume_stopped=False,
    previous_pid=None,
    software_test_only=False,
    execute=None,
):
    if worker.file_sha(original.__file__) != ORIGINAL_RUNNER_SHA:
        raise ValueError("The original unused-shard runner source differs")
    if not resume_stopped:
        if previous_pid is not None:
            raise ValueError("Previous PID is meaningful only for explicit stopped resume")
        return original.run_one(
            plan,
            plan_sha,
            manifest_sha=manifest_sha,
            member_index=member_index,
            gpu=gpu,
            gpu_lock_root=gpu_lock_root,
            software_test_only=software_test_only,
            execute=execute,
        )
    if (
        not str(gpu).isdigit()
        or type(member_index) is not int
        or member_index not in range(6)
        or not isinstance(manifest_sha, str)
        or type(previous_pid) is not int
        or previous_pid <= 0
        or previous_pid == os.getpid()
    ):
        raise ValueError("Explicit stopped resume requires the exact manifest and former PID")
    if execute is not None and not software_test_only:
        raise PermissionError("Production must call the original worker entry point")
    admitted = dispatcher.admit_plan(plan)
    if any(
        manifest.get("software_test_only") is not software_test_only for _, manifest in admitted
    ):
        raise PermissionError("Plan execution mode differs")
    selected = [
        (reference, manifest)
        for reference, manifest in admitted
        if reference["sha256"] == manifest_sha
        and manifest.get("stage") == "member_generation"
        and manifest.get("member_index") == member_index
    ]
    if len(selected) != 1:
        raise ValueError("Selected manifest is not the requested original member shard")
    reference, manifest = selected[0]
    keys = worker.read_json(worker.checked_ref(manifest["task_keys_ref"]))
    output_root = Path(manifest["owned_output_root"]).resolve()
    state_root = output_root / "_dispatcher/plans" / plan_sha
    state_root.mkdir(parents=True, exist_ok=True)
    stop_path = state_root / "STOP_REQUIRES_RECONCILIATION.json"
    status_path = state_root / (manifest_sha + ".status.json")
    summary_path = output_root / "_dispatcher" / manifest_sha / "SUMMARY.json"
    base = dict(
        plan_sha256=plan_sha,
        member_index=member_index,
        physical_gpu=str(gpu),
        manifest_sha256=manifest_sha,
        original_plan_and_manifest_unchanged=True,
        software_test_only=software_test_only,
        resume_stopped=True,
        previous_pid=previous_pid,
    )
    if stop_path.exists() or any(
        (path := state_root / (ref["sha256"] + ".status.json")).exists()
        and worker.read_json(path).get("status") == "FAILED"
        for ref, _ in admitted
    ):
        return dict(base, status="STOP_REQUIRES_RECONCILIATION", executed_shards=0)
    if not status_path.is_file():
        raise RuntimeError("Stopped resume requires an existing original shard status")
    stopped_state(status_path, summary_path, manifest_sha, len(keys), software_test_only)
    if process_is_alive(previous_pid):
        raise RuntimeError("The former process is still present")
    claim = dispatcher.ShardClaim(state_root / (manifest_sha + ".claim"))
    if not claim.acquire():
        return dict(base, status="CLAIM_BUSY", executed_shards=0)
    try:
        if stop_path.exists():
            return dict(base, status="STOP_REQUIRES_RECONCILIATION", executed_shards=0)
        state, summary = stopped_state(
            status_path, summary_path, manifest_sha, len(keys), software_test_only
        )
        if process_is_alive(previous_pid):
            raise RuntimeError("Former-process exit changed after acquiring the original claim")
        status_sha, summary_sha = worker.file_sha(status_path), worker.file_sha(summary_path)
        checkpoint = audit_committed_prefix(
            manifest, manifest_sha, summary["completed"], software_test_only=software_test_only
        )
        lock_root = (
            Path(gpu_lock_root) if gpu_lock_root is not None else Path(manifest["gpu_lock_root"])
        )
        ownership = worker.GPUOwnership(lock_root, str(gpu), plan_sha)
        try:
            ownership.__enter__()
        except RuntimeError:
            return dict(base, status="GPU_BUSY", executed_shards=0)
        try:
            if (
                stop_path.exists()
                or worker.file_sha(status_path) != status_sha
                or worker.file_sha(summary_path) != summary_sha
                or process_is_alive(previous_pid)
            ):
                raise RuntimeError("Stopped checkpoint controls changed before execution")
            receipt_path = state_root / (manifest_sha + ".resume-" + str(time.time_ns()) + ".json")
            admission = dict(
                schema="e18_original_stopped_shard_resume_admission_v1",
                status="ADMITTED_AFTER_ORIGINAL_FLOCK_AND_NATIVE_AUDIT",
                **base,
                checkpoint=checkpoint,
                previous_status=state,
                previous_status_file_sha256=status_sha,
                original_summary_file_sha256=summary_sha,
                previous_pid_confirmed_exited=True,
                previous_pid_identity_source="Caller supplied former PID from its controlled checkpoint",
                new_owner_pid=os.getpid(),
                admitted_gpu_lock_root=str(lock_root.resolve()),
                runner_source_sha256=worker.file_sha(__file__),
                original_runner_source_sha256=ORIGINAL_RUNNER_SHA,
                original_worker_source_sha256=worker.file_sha(worker.__file__),
                at_unix=time.time(),
            )
            write_once(receipt_path, admission)
            worker.save_json(
                status_path,
                dict(
                    status="RUNNING",
                    manifest_sha256=manifest_sha,
                    pid=os.getpid(),
                    started_at_unix=time.time(),
                    software_test_only=software_test_only,
                    stopped_resume_admission_sha256=worker.file_sha(receipt_path),
                ),
            )
            try:
                result = (execute or worker.execute_manifest)(
                    manifest, manifest_sha, software_test_only=software_test_only
                )
                if (
                    not isinstance(result, dict)
                    or any(type(result.get(name)) is not int for name in ("completed", "planned"))
                    or result["planned"] != len(keys)
                    or not 0 <= result["completed"] <= result["planned"]
                    or result.get("failures") != []
                ):
                    raise RuntimeError("The original worker returned a failed or invalid summary")
            except BaseException as error:
                failure = dict(
                    status="FAILED",
                    manifest_sha256=manifest_sha,
                    error_type=type(error).__name__,
                    at_unix=time.time(),
                )
                worker.save_json(status_path, failure)
                worker.save_json(stop_path, failure)
                raise
            status = (
                "DONE" if result["completed"] == result["planned"] else "WAITING_FOR_DEPENDENCIES"
            )
            worker.save_json(
                status_path,
                dict(
                    status=status,
                    manifest_sha256=manifest_sha,
                    summary=result,
                    finished_at_unix=time.time(),
                ),
            )
            return dict(
                base,
                status=status,
                executed_shards=1,
                manifest_ref=reference,
                state_root=str(state_root),
                admitted_gpu_lock_root=str(lock_root.resolve()),
                resume_admission_ref=dict(
                    path=str(receipt_path), sha256=worker.file_sha(receipt_path)
                ),
                checkpoint=checkpoint,
                summary=result,
            )
        finally:
            ownership.__exit__(None, None, None)
    finally:
        claim.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--member-index", type=int, default=2)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-lock-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--unused", action="store_true")
    mode.add_argument("--resume-stopped", action="store_true")
    parser.add_argument("--previous-pid", type=int)
    args = parser.parse_args()
    if worker.file_sha(args.plan) != args.plan_sha256:
        raise ValueError("Original dispatch plan bytes differ")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result = run_one(
        worker.read_json(args.plan),
        args.plan_sha256,
        manifest_sha=args.manifest_sha256,
        member_index=args.member_index,
        gpu=args.gpu,
        gpu_lock_root=args.gpu_lock_root,
        resume_stopped=args.resume_stopped,
        previous_pid=args.previous_pid,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
