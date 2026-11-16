"""Retire one bound E18 follower only after its original claim FDs are absent.

The original follower, dispatcher and worker stay byte-identical. Production
control is Linux-only and opt-in. A selected but not yet claimed batch may run
to completion after the outer actor receives TERM, so zero worker calls after
the request are never asserted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


FOLLOWER_SHA = "4870cfa665b5c3c289c1b71dd310c56061fab31bc4341b68c6ee5bea8fc83f3f"
FOLLOWER_NAME = "e18_auxiliary_aggregation_follower_v2.py"
PRODUCTION_ROOT = "/data1/data2/chendong/routing/research/e18_confirmatory_execution"
QUEUES = {"gpu5_slot0", "gpu5_slot1", "gpu6_slot0"}
REQUEST_FIELDS = {
    "schema", "server_root", "target_pid", "start_ticks", "command",
    "follower_source_ref", "queue_name", "gpu", "software_test_only",
    "timeout_seconds", "poll_seconds",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def argument(command, name):
    if command.count(name) != 1:
        return None
    index = command.index(name) + 1
    return command[index] if index < len(command) else None


class LinuxProcesses:
    """Only inspect one process and send STOP, TERM and CONT to that PID."""

    def __init__(self):
        if sys.platform != "linux":
            raise RuntimeError("Production process control requires Linux")

    def start_ticks(self, pid):
        try:
            text = (Path("/proc") / str(pid) / "stat").read_text()
        except FileNotFoundError:
            return None
        fields = text[text.rfind(")") + 2:].split()
        return None if fields[0] == "Z" else fields[19]

    def identity(self, pid):
        root = Path("/proc") / str(pid)
        before = self.start_ticks(pid)
        if before is None:
            return None
        try:
            command = [item.decode() for item in (root / "cmdline").read_bytes().split(b"\0") if item]
            fields = dict(line.split(":", 1) for line in (root / "status").read_text().splitlines() if ":" in line)
        except FileNotFoundError:
            return None
        after = self.start_ticks(pid)
        if after is None:
            return None
        if before != after:
            raise PermissionError("Process birth changed during identity observation")
        caught = int(fields["SigCgt"].strip(), 16)
        return dict(
            pid=pid, start_ticks=before, command=command,
            state=fields["State"].strip().split()[0],
            sigterm_caught=bool(caught & (1 << (signal.SIGTERM - 1))),
        )

    def claim_fds(self, pid, claim_root):
        found = []
        root = Path(claim_root).resolve()
        for descriptor in (Path("/proc") / str(pid) / "fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            clean = target.removesuffix(" (deleted)")
            path = Path(clean)
            if path.is_absolute() and path.resolve().is_relative_to(root) and path.name.endswith(".claim"):
                found.append(str(descriptor) + " -> " + target)
        return sorted(found)

    def stop(self, pid):
        os.kill(pid, signal.SIGSTOP)

    def terminate(self, pid):
        os.kill(pid, signal.SIGTERM)

    def continue_process(self, pid):
        os.kill(pid, signal.SIGCONT)


class SafeIdleRetirement:
    def __init__(self, request, processes=None, *, clock=time.monotonic, sleeper=time.sleep):
        if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
            raise ValueError("An exact bound E18 retirement request is required")
        if request["schema"] != "e18_safe_idle_retirement_request_v1" or type(request["software_test_only"]) is not bool:
            raise ValueError("Unknown request schema or execution mode")
        if request["software_test_only"] is False and processes is not None:
            raise PermissionError("Production accepts Linux process observation only")
        self.request = request
        self.root = Path(request["server_root"]).resolve(strict=True)
        if not request["software_test_only"] and self.root != Path(PRODUCTION_ROOT).resolve():
            raise PermissionError("Production target must stay inside the E18 private research root")
        self.pid = request["target_pid"]
        self.start = request["start_ticks"]
        self.command = request["command"]
        if type(self.pid) is not int or self.pid <= 0 or not isinstance(self.start, str) or not self.start.isdigit():
            raise ValueError("An explicit positive PID and process birth are required")
        if not isinstance(self.command, list) or not self.command or any(not isinstance(item, str) or not item for item in self.command):
            raise ValueError("The exact nonempty original argv is required")
        if request["queue_name"] not in QUEUES or not str(request["gpu"]).isdigit():
            raise ValueError("An original queue alias and explicit physical GPU are required")
        source_ref = request["follower_source_ref"]
        if not isinstance(source_ref, dict) or set(source_ref) != {"path", "sha256"}:
            raise ValueError("The follower source requires an exact path/SHA binding")
        source = Path(source_ref["path"]).resolve(strict=True)
        if source != self.root / "code" / FOLLOWER_NAME or not source.is_file() or file_sha(source) != source_ref["sha256"]:
            raise PermissionError("Follower source path or byte identity differs")
        if not request["software_test_only"] and source_ref["sha256"] != FOLLOWER_SHA:
            raise PermissionError("Production requires the qualified4870 follower")
        scripts = [Path(item).resolve() for item in self.command if item.endswith(".py")]
        if scripts != [source] or argument(self.command, "--gpu") != str(request["gpu"]):
            raise PermissionError("Original follower argv/source/GPU differs")
        queue_path = self.root / "aggregation_queues" / request["queue_name"]
        lock_path = self.root / "gpu-locks/aggregation" / request["queue_name"]
        if Path(argument(self.command, "--queue-root") or "").resolve() != queue_path or Path(argument(self.command, "--gpu-lock-root") or "").resolve() != lock_path:
            raise PermissionError("The original queue and lock-root argv differ")
        self.claim_root = self.root / "owned/_dispatcher/plans"
        self.timeout = request["timeout_seconds"]
        self.poll = request["poll_seconds"]
        if type(self.timeout) not in (int, float) or type(self.poll) not in (int, float) or not math.isfinite(self.timeout) or not math.isfinite(self.poll) or self.timeout <= 0 or self.poll <= 0:
            raise ValueError("Positive wait bounds are required")
        self.processes = processes if processes is not None else LinuxProcesses()
        self.clock, self.sleeper = clock, sleeper
        self.events = []
        self.signals_sent = []

    def event(self, name, **fields):
        self.events.append(dict(event=name, at_unix=time.time(), **fields))

    def current(self):
        value = self.processes.identity(self.pid)
        if value is not None and (
            value.get("pid") != self.pid or value.get("start_ticks") != self.start or value.get("command") != self.command
        ):
            raise PermissionError("Bound PID/start/complete argv identity changed")
        return value

    def signal(self, name):
        self.signals_sent.append(name)
        self.event("SIGNAL_REQUESTED", signal=name)
        {"STOP": self.processes.stop, "TERM": self.processes.terminate, "CONT": self.processes.continue_process}[name](self.pid)

    def release_stop(self):
        """Release our STOP for the same process birth, fencing PID reuse."""
        try:
            identity = self.processes.identity(self.pid)
        except BaseException as error:
            self.event("FULL_IDENTITY_UNAVAILABLE_DURING_CONT", error_type=type(error).__name__)
            birth = self.processes.start_ticks(self.pid)
            if birth is None:
                return
            if birth != self.start:
                self.event("CONT_REFUSED_PID_REUSED")
                return
            self.signal("CONT")
            return
        if identity is None:
            return
        if identity.get("start_ticks") != self.start or identity.get("command") != self.command:
            self.event("CONT_REFUSED_PID_REUSED_OR_ARGV_CHANGED")
            return
        self.signal("CONT")

    def result(self, status):
        return dict(
            schema="e18_safe_idle_retirement_result_v1", status=status,
            target_pid=self.pid, start_ticks=self.start,
            queue_name=self.request["queue_name"], gpu=str(self.request["gpu"]),
            software_test_only=self.request["software_test_only"],
            signals_sent=list(self.signals_sent), events=list(self.events),
            max_additional_complete_batches_after_term=1,
            worker_calls_after_request="NOT_MEASURED_MAY_INCLUDE_ONE_COMPLETE_BATCH",
            model_calls_by_controller=0, gold_reads=0,
        )

    def run(self, execute=False):
        deadline = self.clock() + self.timeout
        while True:
            identity = self.current()
            if identity is None:
                return self.result("EXITED")
            descriptors = self.processes.claim_fds(self.pid, self.claim_root)
            if not execute:
                return self.result("DRY_RUN_CLAIM_BUSY" if descriptors else "DRY_RUN_ELIGIBLE")
            if not descriptors and identity.get("sigterm_caught"):
                possibly_stopped = False
                term_requested = False
                try:
                    possibly_stopped = True
                    self.signal("STOP")
                    while True:
                        frozen = self.current()
                        if frozen is None:
                            return self.result("EXITED")
                        if frozen.get("state") == "T":
                            break
                        if self.clock() >= deadline:
                            return self.result("TIMEOUT_WAITING_IDLE")
                        self.sleeper(self.poll)
                    # No original claim FD proves that this fixed follower is
                    # outside execute's inner-handler interval. Recheck while
                    # the process is frozen to remove the observation race.
                    descriptors = self.processes.claim_fds(self.pid, self.claim_root)
                    if not descriptors:
                        frozen = self.current()
                        if frozen is None:
                            return self.result("EXITED")
                        if frozen.get("state") != "T":
                            raise RuntimeError("The frozen process resumed before the no-claim proof closed")
                        if not frozen.get("sigterm_caught"):
                            raise PermissionError("The outer caught TERM handler is unavailable")
                        self.event("FROZEN_NO_ORIGINAL_CLAIM_FDS_VERIFIED")
                        self.signal("TERM")
                        term_requested = True
                finally:
                    if possibly_stopped:
                        self.release_stop()
                if term_requested:
                    while self.current() is not None:
                        if self.clock() >= deadline:
                            return self.result("TIMEOUT_AFTER_TERM")
                        self.sleeper(self.poll)
                    return self.result("EXITED")
            if self.clock() >= deadline:
                return self.result("TIMEOUT_WAITING_IDLE")
            self.sleeper(self.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if file_sha(args.request) != args.request_sha256:
        raise ValueError("The reviewed retirement request bytes differ")
    request = read_json(args.request)
    if request.get("software_test_only") is not False:
        raise PermissionError("Production CLI accepts production requests only")
    root = Path(request["server_root"]).resolve()
    private = root / "resource_scheduling"
    if not args.request.resolve().is_relative_to(private) or not args.receipt.resolve().is_relative_to(private):
        raise PermissionError("Request and receipt must stay in private resource_scheduling")
    if args.receipt.exists():
        raise FileExistsError("Operation receipts are append-only")
    actor = SafeIdleRetirement(request)
    error = None
    previous_handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError("Retirement controller received signal " + str(signum))

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    try:
        result = actor.run(execute=args.execute)
    except BaseException as caught:
        result = actor.result("ERROR_REQUIRES_REVIEW")
        result.update(error_type=type(caught).__name__, error=str(caught))
        error = caught
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    result.update(controller_source_sha256=file_sha(__file__), request_sha256=args.request_sha256)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], receipt=str(args.receipt)), ensure_ascii=False))
    if error is not None:
        raise error
    if result["status"].startswith("TIMEOUT"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
