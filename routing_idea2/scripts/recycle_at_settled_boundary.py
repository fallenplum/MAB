"""Acquire only a naturally vacant member ledger lock; never stop an active call.

This narrowly scoped operator helper has no terminate/suspend/set-priority API.
The original runner rejects its next pre-call lock acquisition and exits through
its own finally block. Only the helper's own lock may be released, after both
original process handles prove exit and the new completed ledger is unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from budget_router.execution import _digest
from budget_router.inference import ModelSpec, configuration_fingerprint
from budget_router.stage_validation import atomic_json, read_json

ALLOWED_OUTPUT = ROOT/"results/raw/e1/members_qwen3-8b.jsonl"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_identity(pid):
    import psutil
    process = psutil.Process(int(pid))
    return {"pid": int(pid), "create_time": process.create_time(),
            "command_line": process.cmdline(), "executable": process.exe()}


def assert_identity(expected):
    actual = process_identity(expected["pid"])
    if actual != expected: raise RuntimeError("Original process PID/creation/command identity changed")
    return actual


def _usage(result):
    values = {name: result.get(name) for name in ("prompt_tokens", "completion_tokens")}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise ValueError("No trustworthy original token usage")
    wall = result.get("wall_time_s")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
        raise ValueError("No trustworthy original wall-time usage")
    return {**values, "wall_time_s": wall}


def inspect_ledger(path, *, baseline_count, baseline_prefix_sha256=None, expected_model=None):
    """Validate hashes/usage without exporting any response or question content."""
    payload = Path(path).read_bytes(); lines = payload.splitlines(keepends=True)
    if any(not line.strip() for line in lines): raise ValueError("Unexpected blank ledger row")
    if len(lines) < baseline_count or not payload.endswith(b"\n"):
        raise ValueError("Truncated/incomplete original ledger")
    prefix = b"".join(lines[:baseline_count]); prefix_sha = hashlib.sha256(prefix).hexdigest()
    if baseline_prefix_sha256 is not None and prefix_sha != baseline_prefix_sha256:
        raise ValueError("Previously paid ledger prefix changed")
    seen = set(); success = 0; last = None
    for number, line in enumerate(lines, 1):
        value = json.loads(line.decode("utf-8-sig")); checksum = value.pop("record_sha256", None)
        if not isinstance(checksum, str) or checksum != _digest(value): raise ValueError("Execution ledger checksum mismatch")
        if value.get("status") != "success" or value.get("cost_status") != "recorded_actual_usage":
            raise ValueError("Unreconciled or non-success execution; operator review required")
        result = value.get("result", {}); usage = _usage(result)
        if result.get("model_identity_verified") is not True: raise ValueError("Unverified actual model identity")
        model = {"model_id": result.get("model_id"), "model_revision": result.get("model_revision"),
                 "model_fingerprint": value.get("model_fingerprint")}
        if expected_model is not None and model != expected_model: raise ValueError("Member model/source protocol changed")
        if value.get("job_id") in seen: raise ValueError("Duplicate successful paid job")
        seen.add(value.get("job_id")); success += 1
        last = {"row_number": number, "job_id": value.get("job_id"), "record_sha256": checksum,
                "status": "success", "cost_status": "recorded_actual_usage", **usage, "model": model}
    return {"rows": len(lines), "success_count": success, "baseline_count": baseline_count,
            "baseline_prefix_sha256": prefix_sha, "file_sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "last": last, "unreconciled_count": 0}


def prepare_plan(output, *, worker_pid, supervisor_pid, baseline_count, target_min_count):
    output = Path(output).resolve(strict=True)
    if output != ALLOWED_OUTPUT.resolve(strict=True): raise ValueError("This operator authorization is confined to the named Qwen3 E1 output")
    if baseline_count != 107 or target_min_count < 108: raise ValueError("Authorized baseline is107, target must be at least108")
    worker, supervisor = process_identity(worker_pid), process_identity(supervisor_pid)
    command = worker["command_line"]
    if str(ROOT/"scripts/run_members.py") not in command or "--output" not in command or Path(command[command.index("--output")+1]).resolve() != output:
        raise ValueError("Worker is not the exact named member output command")
    if str(ROOT/"scripts/run_local_e1.py") not in supervisor["command_line"]:
        raise ValueError("Supervisor command differs from the authorized controller")
    if command.count("--model-config") != 1: raise ValueError("Recycle targets exactly one qualified member spec")
    config_path = Path(command[command.index("--model-config")+1]).resolve(strict=True)
    if config_path != (ROOT/"configs/member_qwen3-8b.json").resolve(strict=True): raise ValueError("Unexpected qualified model config")
    spec = ModelSpec(**read_json(config_path)); spec.validate()
    expected_model = {"model_id": spec.model_id, "model_revision": spec.revision,
        "model_fingerprint": configuration_fingerprint(spec)}
    baseline = inspect_ledger(output, baseline_count=baseline_count, expected_model=expected_model)
    if baseline["success_count"] < 107: raise ValueError("The107 paid successful baseline is absent")
    return {"schema": "settled_boundary_recycle_plan_v1", "created_at_unix": time.time(), "output": str(output),
        "worker": worker, "supervisor": supervisor, "baseline_count": baseline_count,
        "target_min_count": target_min_count, "baseline_prefix_sha256": baseline["baseline_prefix_sha256"],
        "expected_model": expected_model, "model_config_path": str(config_path), "model_config_sha256": file_sha(config_path), "baseline_ledger": baseline,
        "coordination_sources": {name: file_sha(ROOT/name) for name in ("src/budget_router/execution.py", "src/budget_router/member_execution.py")},
        "helper_source_sha256": file_sha(__file__), "no_kill_suspend_or_generation": True}


class Audit:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def emit(self, event, **values):
        record = {"event": event, "at_unix": time.time(), **values}
        with self.path.open("a", encoding="utf8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n"); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


class MonitoredProcess:
    """Read/synchronize handle only; keeps original process identity after PID reuse."""
    def __init__(self, identity):
        from ctypes import wintypes
        assert_identity(identity)
        self.identity = identity; self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.OpenProcess(0x100000|0x1000, False, identity["pid"])
        if not self.handle: raise OSError("Cannot open original process read/synchronize handle")
        assert_identity(identity)
    def exited(self):
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result not in (0, 258): raise OSError("Cannot verify original process state")
        return result == 0
    def exit_code(self):
        from ctypes import wintypes
        if not self.exited(): return None
        value = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(value)): raise OSError("Cannot read exit code")
        return int(value.value)
    def close(self):
        if self.handle: self.kernel.CloseHandle(self.handle); self.handle = None


class BoundaryWatcher:
    """Read-only directory notification; wake promptly at natural lock removal."""
    def __init__(self, directory):
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.FindFirstChangeNotificationW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL, wintypes.DWORD]
        self.kernel.FindFirstChangeNotificationW.restype = wintypes.HANDLE
        self.kernel.FindNextChangeNotification.argtypes = [wintypes.HANDLE]
        self.kernel.FindCloseChangeNotification.argtypes = [wintypes.HANDLE]
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.handle = self.kernel.FindFirstChangeNotificationW(str(directory), False, 0x1|0x10)
        if self.handle in (None, ctypes.c_void_p(-1).value): raise OSError("Cannot observe natural ledger boundary")
    def wait(self):
        result = self.kernel.WaitForSingleObject(self.handle, 50)
        if result == 0:
            if not self.kernel.FindNextChangeNotification(self.handle): raise OSError("Cannot rearm boundary observer")
            return True
        if result != 258: raise OSError("Boundary observation failed")
        return False
    def close(self):
        if self.handle: self.kernel.FindCloseChangeNotification(self.handle); self.handle = None


def gpu_snapshot(pid):
    """Read-only NVML CLI plus actual target WDDM local budget if accessible."""
    result = {"at_unix": time.time()}
    try:
        data = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.free,power.draw",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        result["nvidia_smi_returncode"] = data.returncode
        if data.returncode == 0: result["nvidia_smi_csv"] = data.stdout.strip()
    except Exception as error: result["nvidia_smi_error_type"] = type(error).__name__
    try:
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True); gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        class Luid(ctypes.Structure): _fields_ = [("Low", wintypes.DWORD), ("High", wintypes.LONG)]
        class Open(ctypes.Structure): _fields_ = [("Luid", Luid), ("Adapter", wintypes.UINT)]
        class Memory(ctypes.Structure): _fields_ = [("Process", wintypes.HANDLE), ("Adapter", wintypes.UINT), ("Group", ctypes.c_int),
            ("Budget", ctypes.c_ulonglong), ("Usage", ctypes.c_ulonglong), ("Reservation", ctypes.c_ulonglong),
            ("Available", ctypes.c_ulonglong), ("Physical", wintypes.UINT)]
        class Close(ctypes.Structure): _fields_ = [("Adapter", wintypes.UINT)]
        gdi.D3DKMTOpenAdapterFromLuid.argtypes = [ctypes.POINTER(Open)]; gdi.D3DKMTQueryVideoMemoryInfo.argtypes = [ctypes.POINTER(Memory)]
        gdi.D3DKMTCloseAdapter.argtypes = [ctypes.POINTER(Close)]
        process = kernel.OpenProcess(0x400, False, pid); adapter = Open(Luid(0x11885, 0), 0)
        try:
            if process and gdi.D3DKMTOpenAdapterFromLuid(ctypes.byref(adapter)) == 0:
                try:
                    value = Memory(process, adapter.Adapter, 0, 0, 0, 0, 0, 0)
                    status = gdi.D3DKMTQueryVideoMemoryInfo(ctypes.byref(value))
                    result["wddm_status"] = hex(status & 0xffffffff)
                    if status == 0: result.update(local_budget_bytes=value.Budget, local_usage_bytes=value.Usage,
                        local_overbudget_bytes=max(0, value.Usage-value.Budget))
                finally: gdi.D3DKMTCloseAdapter(ctypes.byref(Close(adapter.Adapter)))
            else: result["wddm_target_unavailable"] = True
        finally:
            if process: kernel.CloseHandle(process)
    except Exception as error: result["wddm_error_type"] = type(error).__name__
    return result


@dataclass
class OperatorLease:
    path: Path
    descriptor: int
    owner: dict

    def verify_owned(self):
        own_stat, current = os.fstat(self.descriptor), self.path.stat()
        if (own_stat.st_dev, own_stat.st_ino) != (current.st_dev, current.st_ino): raise RuntimeError("Operator lock file identity changed")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        payload = os.read(self.descriptor, 65536)
        if json.loads(payload.decode("utf8")) != self.owner or json.loads(self.path.read_text(encoding="utf8")) != self.owner:
            raise RuntimeError("Operator ownership token changed; never remove another lock")

    def release_after_exit(self, worker, supervisor):
        if not worker.exited() or not supervisor.exited(): raise RuntimeError("Original processes are not both proven exited")
        self.verify_owned()
        os.close(self.descriptor); self.descriptor = -1
        # Only this exact still-owned pathname is eligible. No worker stale-lock
        # recovery, force delete or recursive file operation exists here.
        if json.loads(self.path.read_text(encoding="utf8")) != self.owner: raise RuntimeError("Lock ownership changed at release")
        self.path.unlink()


def try_acquire_boundary(plan, owner, *, identity_check=assert_identity, fast_after_observed_worker_lock=False):
    output = Path(plan["output"])
    if output.with_name(output.name+".lock").exists(): return None
    if not fast_after_observed_worker_lock:
        ledger = inspect_ledger(output, baseline_count=plan["baseline_count"],
            baseline_prefix_sha256=plan["baseline_prefix_sha256"], expected_model=plan["expected_model"])
        if ledger["success_count"] < plan["target_min_count"]: return None
        identity_check(plan["worker"]); identity_check(plan["supervisor"])
    lock = output.with_name(output.name+".lock")
    try: descriptor = os.open(lock, os.O_RDWR|os.O_CREAT|os.O_EXCL, 0o600)
    except FileExistsError: return None
    lease = OperatorLease(lock, descriptor, owner)
    try:
        os.write(descriptor, json.dumps(owner, sort_keys=True, allow_nan=False).encode("utf8")); os.fsync(descriptor)
        lease.verify_owned()
        identity_check(plan["worker"]); identity_check(plan["supervisor"])
        # If a prior writer won between our precheck and exclusive creation, its
        # completed result must still be valid; never infer safety from linecount.
        settled = inspect_ledger(output, baseline_count=plan["baseline_count"],
            baseline_prefix_sha256=plan["baseline_prefix_sha256"], expected_model=plan["expected_model"])
        if settled["success_count"] < plan["target_min_count"]: raise RuntimeError("Target boundary is not settled")
        return lease, settled
    except BaseException:
        os.close(descriptor); lease.descriptor = -1
        # Preserve the operator lock for manual ownership reconciliation.
        raise


def run_waiter(plan, audit, *, poll_seconds=.05):
    output = Path(plan["output"]).resolve(strict=True)
    if output != ALLOWED_OUTPUT.resolve(strict=True): raise ValueError("Unauthorized recycle output")
    if plan.get("schema") != "settled_boundary_recycle_plan_v1" or plan["baseline_count"] != 107 or plan["target_min_count"] < 108:
        raise ValueError("Unsupported narrow operator plan")
    if file_sha(__file__) != plan["helper_source_sha256"]: raise ValueError("Helper changed since plan preparation")
    if file_sha(plan["model_config_path"]) != plan["model_config_sha256"]: raise ValueError("Qualified model configuration changed")
    for name, sha in plan["coordination_sources"].items():
        if file_sha(ROOT/name) != sha: raise ValueError("Qualified ledger/lock source changed")
    worker, supervisor = MonitoredProcess(plan["worker"]), MonitoredProcess(plan["supervisor"])
    owner = {"schema": "operator_settled_boundary_lock_v1", "pid": os.getpid(),
        "helper_create_time": process_identity(os.getpid())["create_time"], "owner_token": uuid.uuid4().hex,
        "created_at_unix": time.time(), "worker_pid": plan["worker"]["pid"], "supervisor_pid": plan["supervisor"]["pid"],
        "target_min_count": plan["target_min_count"], "baseline_prefix_sha256": plan["baseline_prefix_sha256"]}
    lease = None
    watcher = BoundaryWatcher(output.parent)
    try:
        audit.emit("WAITING_FOR_NATURAL_SETTLED_BOUNDARY", helper_identity=process_identity(os.getpid()), owner_token=owner["owner_token"],
            worker=plan["worker"], supervisor=plan["supervisor"], baseline=plan["baseline_ledger"],
            gpu_before_wait=gpu_snapshot(plan["worker"]["pid"]))
        heartbeat = time.monotonic()
        observed_worker_lock = False
        while lease is None:
            if worker.exited() or supervisor.exited(): raise RuntimeError("Original process exited before operator boundary acquisition; reconcile without touching any lock")
            lock_path = output.with_name(output.name+".lock")
            if lock_path.exists():
                try:
                    existing = json.loads(lock_path.read_text(encoding="utf8"))
                    if existing.get("pid") == plan["worker"]["pid"]: observed_worker_lock = True
                except (FileNotFoundError, json.JSONDecodeError): pass
            acquired = try_acquire_boundary(plan, owner, fast_after_observed_worker_lock=observed_worker_lock)
            if acquired is not None: lease, settled = acquired; break
            if time.monotonic()-heartbeat >= 30:
                audit.emit("STILL_WAITING_NO_PROCESS_INTERVENTION"); heartbeat = time.monotonic()
            if watcher.wait():
                # Ledger fsync precedes lock unlink. Briefly watch this exact
                # natural transition, rather than relying on a coarse50ms race.
                until = time.monotonic()+.25
                while observed_worker_lock and time.monotonic() < until:
                    if worker.exited() or supervisor.exited(): break
                    acquired = try_acquire_boundary(plan, owner, fast_after_observed_worker_lock=True)
                    if acquired is not None: lease, settled = acquired; break
                    time.sleep(.001)
        audit.emit("OWN_OPERATOR_LOCK_ACQUIRED_AFTER_SETTLEMENT", lock_path=str(lease.path), owner=owner, settled=settled,
            gpu_at_boundary=gpu_snapshot(plan["worker"]["pid"]))
        reported = set()
        while not (worker.exited() and supervisor.exited()):
            lease.verify_owned()
            for name, process in (("worker", worker), ("supervisor", supervisor)):
                if name not in reported and process.exited():
                    audit.emit("ORIGINAL_PROCESS_NATURALLY_EXITED", role=name, identity=process.identity, exit_code=process.exit_code()); reported.add(name)
            time.sleep(.2)
        for name, process in (("worker", worker), ("supervisor", supervisor)):
            if name not in reported: audit.emit("ORIGINAL_PROCESS_NATURALLY_EXITED", role=name, identity=process.identity, exit_code=process.exit_code())
        last = inspect_ledger(output, baseline_count=107, baseline_prefix_sha256=plan["baseline_prefix_sha256"], expected_model=plan["expected_model"])
        if last != settled: raise RuntimeError("Ledger changed while operator owned the exclusive boundary")
        after = gpu_snapshot(plan["worker"]["pid"])
        audit.emit("SAFE_RESOURCE_BOUNDARY_VERIFIED", settled=last, gpu_after_original_exit=after,
            worker_exit_code=worker.exit_code(), supervisor_exit_code=supervisor.exit_code(), no_generation_started_by_helper=True)
        lease.release_after_exit(worker, supervisor)
        audit.emit("OWN_OPERATOR_LOCK_RELEASED", lock_path=str(lease.path), owner_token=owner["owner_token"],
            ledger_sha256=last["file_sha256"], restart_authority="root operator only; helper never starts a GPU runner")
        return 0
    except BaseException as error:
        owned = lease is not None
        if not owned:
            try: owned = json.loads(output.with_name(output.name+".lock").read_text(encoding="utf8")) == owner
            except (FileNotFoundError, json.JSONDecodeError): pass
        audit.emit("STOPPED_REQUIRES_OPERATOR_RECONCILIATION", error_type=type(error).__name__,
            operator_lock_owned=owned, lock_removed=False)
        raise
    finally:
        if lease is not None and lease.descriptor >= 0: os.close(lease.descriptor)
        watcher.close()
        worker.close(); supervisor.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-plan", type=Path)
    parser.add_argument("--run-plan", type=Path)
    parser.add_argument("--output", type=Path, default=ALLOWED_OUTPUT)
    parser.add_argument("--worker-pid", type=int, default=53204)
    parser.add_argument("--supervisor-pid", type=int, default=76332)
    parser.add_argument("--baseline-count", type=int, default=107)
    parser.add_argument("--target-min-count", type=int, default=108)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    if bool(args.prepare_plan) == bool(args.run_plan): parser.error("Choose prepare-plan or run-plan")
    if args.prepare_plan:
        plan = prepare_plan(args.output, worker_pid=args.worker_pid, supervisor_pid=args.supervisor_pid,
            baseline_count=args.baseline_count, target_min_count=args.target_min_count)
        if args.prepare_plan.exists(): raise FileExistsError("Operator plan cannot be overwritten")
        atomic_json(args.prepare_plan, plan)
        print(json.dumps({"prepared_plan": str(args.prepare_plan), "worker": plan["worker"], "supervisor": plan["supervisor"],
            "baseline_successes": plan["baseline_ledger"]["success_count"], "target_min_count": plan["target_min_count"]}, ensure_ascii=False))
        return 0
    if args.audit is None: parser.error("run-plan requires private audit path")
    return run_waiter(read_json(args.run_plan), Audit(args.audit))


if __name__ == "__main__": raise SystemExit(main())
