"""PRIVATE guarded E1 operator adapter; default dry-run, never starts a model.

The state machine and Windows lease are independently testable. Production
execution remains disabled until a complete source/profile/helper/GPU adapter
has passed its capability checks AND a new exact authorization packet exists.
No PID, job, authorization, or historical snapshot is hardcoded here.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from budget_router.execution import _digest
from budget_router.member_execution import MemberStore
from budget_router.inference import _validated_usage

ACCOUNTING_METHOD = "operator_observed_committed_prefix_and_host_span_v1"
ACTIONS = {"freeze_inspect_current_e1_call", "hold_and_restore_waiting_helper",
           "terminate_this_attempt", "record_exact_failure_only"}


class ProofUnavailable(RuntimeError): pass
class ReconciliationRequired(RuntimeError): pass


def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
def finite(value): return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_authorization(plan, packet, *, plan_sha256, now=None, software_test_only=False):
    """Declaration binding; actual new user-message provenance is operator-owned."""
    now = time.time() if now is None else now
    if plan.get("schema") != "guarded_e1_abort_plan_v1" or not isinstance(plan.get("nonce"), str) or len(plan["nonce"]) < 16:
        raise ValueError("Exact operator plan and nonce required")
    if packet is None or packet.get("schema") != "explicit_e1_abort_authorization_v1":
        raise PermissionError("A NEW explicit abort authorization packet is required")
    if bool(packet.get("software_test_only")) != software_test_only:
        raise PermissionError("Software authorization is never production authorization")
    if bool(plan.get("software_test_only")) != software_test_only:
        raise PermissionError("Plan software/production scope mismatch")
    if packet.get("explicit_user_approval") is not True or packet.get("approval_scope") != "abort_this_current_e1_attempt_once":
        raise PermissionError("Prior success-boundary cleanup is not abort authorization")
    if packet.get("plan_sha256") != plan_sha256 or packet.get("plan_nonce") != plan["nonce"]:
        raise PermissionError("Authorization plan hash/nonce mismatch")
    for key in ("worker", "helper", "job_id", "model_fingerprint"):
        if packet.get(key) != plan.get(key): raise PermissionError("Authorization exact target binding mismatch: " + key)
    for role in ("worker", "helper"):
        target = plan.get(role, {})
        if (not isinstance(target.get("pid"), int) or isinstance(target["pid"], bool) or target["pid"] <= 0
                or not finite(target.get("create_time")) or not target.get("command_line")):
            raise ValueError("Exact PID/create_time/command identity required")
    if set(packet.get("actions", [])) != ACTIONS:
        raise PermissionError("Explicit narrowly scoped actions required")
    if not (finite(packet.get("issued_at_unix")) and finite(packet.get("expires_at_unix"))
            and packet["issued_at_unix"] <= now < packet["expires_at_unix"]
            and packet["expires_at_unix"] - packet["issued_at_unix"] <= 1800):
        raise PermissionError("Missing/expired authorization time window")
    evidence = packet.get("user_approval_evidence", {})
    if not isinstance(evidence.get("sha256"), str) or len(evidence["sha256"]) != 64 or not evidence.get("path"):
        raise PermissionError("New user-message evidence reference required")
    if not software_test_only:
        path = Path(evidence["path"])
        if sha(path.read_bytes()) != evidence["sha256"]: raise PermissionError("User-message evidence changed")
    return {"plan_sha256": plan_sha256, "nonce": plan["nonce"], "software_test_only": software_test_only}


class WindowsFreezeLease:
    """Bounded suspension of one exact process; no unverified atomic-API claim.

After NtSuspendProcess, two stable thread enumerations and balanced
SuspendThread/ResumeThread probes establish that every observed user thread
already had a positive suspend count. Once that closed set is frozen it cannot
create another user thread itself. External injectors/debuggers are excluded by
operator ownership; a changing thread set or pre-suspended target is refused.
The timer restores this lease's process suspension on any pre-kill timeout.
"""
    def __init__(self, identity, *, timeout_s=15., software_test_only=False):
        if sys.platform != "win32": raise ProofUnavailable("Windows-only process lease")
        if not finite(timeout_s) or not 0 < timeout_s <= 30: raise ValueError("Bounded freeze timeout required")
        self.identity, self.timeout_s = dict(identity), timeout_s
        self.software_test_only = software_test_only
        self.nonce, self.handle, self.thread_ids = uuid.uuid4().hex, None, []
        self.raw_dump_sha256 = None
        self._guard = threading.RLock(); self._suspended = False
        self._expired = False; self._terminated = False; self._entered = False
        self.events = []
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenThread.restype = wintypes.HANDLE
        for name in ("SuspendThread", "ResumeThread", "WaitForSingleObject"):
            getattr(self.kernel, name).restype = wintypes.DWORD
        self.kernel.SuspendThread.argtypes = self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateProcess.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        for name in ("NtSuspendProcess", "NtResumeProcess"):
            getattr(self.ntdll, name).argtypes = [wintypes.HANDLE]
            getattr(self.ntdll, name).restype = ctypes.c_long

    def _identity(self):
        import psutil
        process = psutil.Process(self.identity["pid"])
        actual = {"pid": process.pid, "create_time": process.create_time(), "command_line": process.cmdline()}
        if actual != self.identity: raise ProofUnavailable("PID/create_time/command identity changed")
        return process

    def _probe(self):
        process = self._identity()
        before = sorted(t.id for t in process.threads())
        if not before: raise ProofUnavailable("Empty process thread set")
        counts = {}
        for tid in before:
            handle = self.kernel.OpenThread(0x2, False, tid)
            if not handle: raise ProofUnavailable("Cannot open exact suspended thread")
            added = False
            try:
                previous = self.kernel.SuspendThread(handle)
                if previous == 0xffffffff: raise ProofUnavailable("Thread suspend-count probe failed")
                added = True
                counts[tid] = int(previous)
                if previous < 1: raise ProofUnavailable("Observed user thread was not already suspended")
            finally:
                if added and self.kernel.ResumeThread(handle) == 0xffffffff:
                    self.events.append({"event": "probe_balance_failed", "thread_id": tid})
                    self.kernel.CloseHandle(handle)
                    raise ReconciliationRequired("Cannot balance thread-count probe")
                self.kernel.CloseHandle(handle)
        after = sorted(t.id for t in process.threads())
        if before != after: raise ProofUnavailable("Thread set changed during frozen proof")
        self.thread_ids = before
        return counts

    def __enter__(self):
        with self._guard:
            process = self._identity()
            import psutil
            if process.status() == psutil.STATUS_STOPPED:
                raise ProofUnavailable("Target was already suspended; cannot own its suspension")
            self.handle = self.kernel.OpenProcess(0x0800 | 0x1000 | 0x100000 | 0x1, False, process.pid)
            if not self.handle: raise ProofUnavailable("Cannot open identity-bound process handle")
            try:
                if self.ntdll.NtSuspendProcess(self.handle) < 0: raise ProofUnavailable("NtSuspendProcess failed")
                self._suspended = True; self.deadline = time.monotonic() + self.timeout_s
                self.timer = threading.Timer(self.timeout_s, self._watchdog); self.timer.daemon = True; self.timer.start()
                counts = self._probe()
                self._entered = True
                self.events.append({"event": "closed_thread_set_all_user_suspended", "counts": counts})
                return self
            except BaseException:
                if hasattr(self, "timer"): self.timer.cancel()
                self._resume(); self.kernel.CloseHandle(self.handle); self.handle = None
                raise

    def _resume(self):
        if self._suspended and self.handle:
            if self.kernel.WaitForSingleObject(self.handle, 0) == 0:
                self._suspended = False
            else:
                status = self.ntdll.NtResumeProcess(self.handle)
                self.events.append({"event": "resume_own_process_suspension", "status": int(status)})
                if status >= 0:
                    self._suspended = False
                elif not self._terminated:
                    raise ReconciliationRequired("Cannot confirm restoration of owned suspension")

    def _watchdog(self):
        with self._guard:
            self._expired = True
            self.events.append({"event": "freeze_watchdog_expired"})
            self._resume()

    def require_active(self):
        with self._guard:
            if (not self._entered or not self._suspended or self._expired or time.monotonic() >= self.deadline
                    or self.kernel.WaitForSingleObject(self.handle, 0) == 0):
                raise ProofUnavailable("Frozen lease expired or process exited")
            self._probe()

    def capture_pyspy(self, executable, expected_sha256, *, timeout_s=4.):
        self.require_active()
        executable = Path(executable)
        if sha(executable.read_bytes()) != expected_sha256: raise ProofUnavailable("py-spy executable changed")
        if timeout_s >= max(0, self.deadline - time.monotonic()): raise ProofUnavailable("Capture exceeds remaining freeze lease")
        result = subprocess.run([str(executable), "dump", "--pid", str(self.identity["pid"]),
                                 "--json", "--locals", "--full-filenames", "--nonblocking"],
                                capture_output=True, timeout=timeout_s, creationflags=0x08000000)
        self.require_active()
        if result.returncode != 0: raise ProofUnavailable("Nonblocking py-spy capture failed")
        self.raw_dump_sha256 = sha(result.stdout)
        return result.stdout

    def ticket_state(self):
        self.require_active()
        return {"status": "verified_closed_thread_set_all_user_suspended", "active": True,
                "pid": self.identity["pid"], "create_time_ns": round(self.identity["create_time"] * 1e9),
                "nonce": self.nonce, "thread_ids": list(self.thread_ids), "raw_dump_sha256": self.raw_dump_sha256}

    def terminate_and_wait(self, *, timeout_s=5.):
        if not finite(timeout_s) or not 0 < timeout_s <= 15: raise ValueError("Bounded process-exit timeout required")
        with self._guard:
            self.require_active()
            if not self.kernel.TerminateProcess(self.handle, 129): raise ProofUnavailable("Target termination request failed")
            self._terminated = True
            self.timer.cancel()
        if self.kernel.WaitForSingleObject(self.handle, int(timeout_s * 1000)) != 0:
            raise ReconciliationRequired("Process exit not confirmed; keep call unreconciled")
        self._suspended = False
        return {"process_exit_confirmed": True, "observed_at_qpc": time.perf_counter(),
                "identity": dict(self.identity)}

    def __exit__(self, typ, value, traceback):
        with self._guard:
            if hasattr(self, "timer"): self.timer.cancel()
            try: self._resume()
            finally:
                if self.handle: self.kernel.CloseHandle(self.handle); self.handle = None
        return False


def make_failure_row(job, proof, *, end_qpc, wall_evidence, attempt_id=None, started_at_unix=None):
    if proof.job_id != job["job_id"] or proof.model_fingerprint != job["model_fingerprint"] or proof.seed != job["seed"]:
        raise ProofUnavailable("Settled prefix belongs to another job/spec/seed")
    usage = _validated_usage({"prompt_tokens": proof.prompt_tokens, "completion_tokens": proof.completion_tokens})
    if usage is None or not finite(proof.backend_start_qpc) or not finite(end_qpc) or end_qpc < proof.backend_start_qpc:
        raise ProofUnavailable("Exact finite token/clock observations required")
    if wall_evidence.get("process_exit_confirmed") is not True or wall_evidence.get("gpu_context_absent_verified") is not True:
        raise ProofUnavailable("Process exit and GPU context retirement evidence required")
    key = sha(canonical({"job_id": job["job_id"], "start_qpc": proof.backend_start_qpc,
                         "capture_sha256": proof.evidence["raw_dump_sha256"]}))
    row = {name: job.get(name) for name in ("job_id", "question_id", "task", "subtask", "split", "model_alias",
        "sample_index", "seed", "model_fingerprint", "member_mode", "sampling_control", "prompt_version", "source_question_sha256")}
    row.update(attempt_id=attempt_id or "operator_assigned_" + key,
        attempt_id_origin="captured_original_common" if attempt_id else "operator_assigned_bound_to_call_evidence",
        started_at_unix=started_at_unix, status="failure", error_type="OperatorAbortedKnownUsage",
        reason_code="operator_aborted_settled_prefix", error="Interrupted attempt; no answer or reward recorded",
        cost_status="recorded_actual_usage", actual_usage=usage,
        failure_wall_time_s=float(end_qpc - proof.backend_start_qpc), accounting_method=ACCOUNTING_METHOD,
        failure_wall_time_scope="backend_start_to_operator_observed_process_exit_and_context_absence_including_pause_and_drain",
        operator_accounting={"backend_start_qpc": proof.backend_start_qpc, "observed_end_qpc": end_qpc,
           "clock": vars(time.get_clock_info("perf_counter")), "wall_evidence": wall_evidence,
           "prefix_evidence": proof.evidence, "software_test_only": proof.software_test_only})
    if "result" in row or "reward" in row or "text" in row: raise AssertionError("No interrupted answer allowed")
    canonical(row)
    return row


def reconcile_failure_once(ledger, *, baseline_bytes, old_lock_bytes, old_lock_identity, row,
                           journal_dir, worker_exit_confirmed, helper_exit_confirmed):
    """Owned dead-worker lock handoff without a vacant lock-path race.

Hardlink archives the original lock inode, then atomic replacement installs our
exclusive claim. No model is run. A failed append leaves the owned claim for
manual reconciliation; never delete another owner's lock.
"""
    if worker_exit_confirmed is not True or helper_exit_confirmed is not True:
        raise ProofUnavailable("Dead worker and retired waiting helper required")
    ledger, folder = Path(ledger).resolve(), Path(journal_dir).resolve()
    if not ledger.is_relative_to(ROOT) or not folder.is_relative_to(ROOT): raise ValueError("Reconciliation escaped project root")
    folder.mkdir(parents=True, exist_ok=True)
    row_path = folder / "accounted_failure_row.json"
    expected = canonical(row) + b"\n"
    if row_path.exists():
        if row_path.read_bytes() != expected: raise ReconciliationRequired("Existing measured accounting differs")
    else:
        with row_path.open("xb") as stream: stream.write(expected); stream.flush(); os.fsync(stream.fileno())
    payload = ledger.read_bytes()
    if not payload.startswith(baseline_bytes): raise ReconciliationRequired("Previously paid ledger prefix changed")
    tail = payload[len(baseline_bytes):]
    if tail:
        lines = tail.splitlines()
        if len(lines) != 1: raise ReconciliationRequired("New ledger rows require independent reconciliation")
        existing = json.loads(lines[0]); checksum = existing.pop("record_sha256", None)
        if existing == row and checksum == _digest(row): return {"status": "ALREADY_RECORDED", "record_sha256": checksum}
        raise ReconciliationRequired("Ledger has another new success/failure; do not append")
    lock = ledger.with_name(ledger.name + ".lock")
    stat = lock.stat()
    if (stat.st_dev, stat.st_ino) != tuple(old_lock_identity) or lock.read_bytes() != old_lock_bytes:
        raise ReconciliationRequired("Original dead-worker lock identity changed")
    archived = folder / "archived_worker_call.lock"
    if archived.exists():
        if archived.read_bytes() != old_lock_bytes:
            raise ReconciliationRequired("Original lock archive differs")
    else:
        with archived.open("xb") as stream:
            stream.write(old_lock_bytes); stream.flush(); os.fsync(stream.fileno())
    current = lock.stat()
    if ((current.st_dev, current.st_ino) != tuple(old_lock_identity)
            or lock.read_bytes() != old_lock_bytes or archived.read_bytes() != old_lock_bytes):
        raise ReconciliationRequired("Original lock changed during exact-byte archival")
    claim = folder / "operator_claim.pending"
    owner = {"schema": "operator_abort_reconciliation_lock_v1", "pid": os.getpid(), "nonce": uuid.uuid4().hex}
    owner_bytes = canonical(owner)
    with claim.open("xb") as stream: stream.write(owner_bytes); stream.flush(); os.fsync(stream.fileno())
    # Original path still belongs to the dead worker until this atomic handoff.
    final_old_stat = lock.stat()
    if (lock.read_bytes() != old_lock_bytes or (final_old_stat.st_dev, final_old_stat.st_ino) != tuple(old_lock_identity)
            or archived.read_bytes() != old_lock_bytes):
        raise ReconciliationRequired("Original lock changed before atomic ownership handoff")
    os.replace(claim, lock)
    own_stat = lock.stat()
    def owned():
        current = lock.stat()
        return (current.st_dev, current.st_ino) == (own_stat.st_dev, own_stat.st_ino) and lock.read_bytes() == owner_bytes
    if not owned() or ledger.read_bytes() != baseline_bytes: raise ReconciliationRequired("Ledger/claim changed before append")
    stored = MemberStore(ledger)._append_unlocked(row)
    if not owned(): raise ReconciliationRequired("Operator ownership changed after append")
    final = ledger.read_bytes()
    if not final.startswith(baseline_bytes) or len(final[len(baseline_bytes):].splitlines()) != 1:
        raise ReconciliationRequired("Failure append verification failed")
    lock.unlink()
    return {"status": "RECORDED_ONCE", "record_sha256": stored["record_sha256"], "archived_lock_sha256": sha(old_lock_bytes)}


def run_guarded_abort(plan, packet, *, plan_sha256, adapter, software_test_only=False):
    validate_authorization(plan, packet, plan_sha256=plan_sha256, software_test_only=software_test_only)
    adapter.require_capabilities(plan, software_test_only=software_test_only)
    with adapter.operator_guard(plan):
        current = adapter.inspect_current(plan)
        if current.get("already_successful"):
            return {"status": "NOOP_ALREADY_SETTLED", "abort_started": False}
        with adapter.freeze_worker(current) as frozen:
            proof = adapter.prove_prefix(current, frozen)
            if bool(proof.software_test_only) != software_test_only:
                raise ProofUnavailable("Software prefix proof is never production usage")
            adapter.verify_unchanged(current, frozen)
            # The helper is touched only after the worker's safe state is proved.
            with adapter.hold_waiting_helper(current, frozen) as helper:
                frozen.require_active()
                adapter.verify_unchanged(current, frozen)
                exit_proof = frozen.terminate_and_wait()
                retirement = adapter.wait_gpu_context_absent(current, exit_proof)
                end_qpc = adapter.qpc()
                wall = {**exit_proof, **retirement}
                row = make_failure_row(current["job"], proof, end_qpc=end_qpc, wall_evidence=wall,
                    attempt_id=getattr(proof, "original_attempt_id", None),
                    started_at_unix=getattr(proof, "original_started_at_unix", None))
                adapter.persist_accounting(current, row)
                # Keep old worker lock in place until the WAITING helper sees its
                # original worker exited and itself exits through its own code.
                helper.restore_and_wait_for_natural_exit()
                adapter.verify_dead_lock_and_unchanged_prefix(current)
                result = adapter.reconcile_once(current, row)
                return {"status": "ABORT_RECONCILED_NO_RETRY_STARTED", "accounting_method": ACCOUNTING_METHOD,
                        "actual_usage": row["actual_usage"], "failure_wall_time_s": row["failure_wall_time_s"], **result}


class UnqualifiedProductionAdapter:
    """No actual target is opened before a complete adapter is qualified."""
    def require_capabilities(self, plan, *, software_test_only=False):
        raise ProofUnavailable("Production profile/helper/GPU retirement adapters not yet qualified; no target accessed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    args = parser.parse_args(argv)
    raw = Path(args.plan).read_bytes(); plan = json.loads(raw.decode("utf-8-sig"))
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN_ONLY", "plan_sha256": sha(raw), "actions_performed": [],
                          "live_process_accessed": False, "production_ready": False,
                          "new_abort_authorization_required": True}))
        return 0
    try:
        packet = json.loads(Path(args.authorization).read_text("utf-8-sig")) if args.authorization else None
        result = run_guarded_abort(plan, packet, plan_sha256=sha(raw), adapter=UnqualifiedProductionAdapter())
    except (PermissionError, ValueError, ProofUnavailable, ReconciliationRequired) as error:
        print(json.dumps({"status": "REFUSED_NO_RETRY", "reason": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result)); return 0


if __name__ == "__main__": raise SystemExit(main())
