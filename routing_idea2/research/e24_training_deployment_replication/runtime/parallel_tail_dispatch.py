"""Drain an E24 aggregate actor and assign only its outstanding keys to two slots."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from deferred_launcher import gpu_snapshot
from monitor_e24 import actor_state, canonical, context, read, sha, start_receipts, task_key


ROOT = Path("/data2/chendong/e24_training_deployment_replication")
PLAN_SHA = "c6ee8de42f5b72f3cff8c6591b267eb0098b6e5bf743efb4486af5f2766bc208"
HEADROOM_MIB = 14336
STAGE = "aggregation"


def emit(status, **fields):
    print(json.dumps(dict(status=status, at_unix=time.time(), **fields)), flush=True)


def freeze(path, record):
    pending = path.with_suffix(f"{path.suffix}.pending-{os.getpid()}")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()
    return dict(path=str(path), sha256=sha(path))


def actors(root):
    result = {}
    for receipt in start_receipts(root):
        state = actor_state(receipt)
        if state["alive"] is None:
            raise RuntimeError("A registered E24 process has an unknown identity")
        if state["alive"]:
            result[receipt["pid"]] = receipt
    return result


def admission(root, gpus, retire_pid=None, samples=6, utilization_limit=83, retained_origin=None):
    measured = {gpu: [] for gpu in gpus}
    for _ in range(samples):
        live = actors(root)
        devices = gpu_snapshot()
        for gpu in gpus:
            observed = devices[gpu]
            current = observed["pids"] | {pid for pid, row in live.items() if row["gpu"] == gpu}
            projected = (current - {retire_pid}) | {-1}
            pending = {pid for pid in current - observed["pids"] - {retire_pid} if pid in live}
            if len(projected) > 2 or observed["free_mib"] < HEADROOM_MIB * (1 + len(pending)):
                raise RuntimeError(f"GPU {gpu} no longer has a safe second slot")
            measured[gpu].append(dict(utilization=observed["utilization"], free_mib=observed["free_mib"],
                                      current_compute_processes=len(observed["pids"]), planned_processes=len(projected)))
        if samples > 1:
            time.sleep(0.65)
    for gpu, readings in measured.items():
        # Replacing the retired slot on its original card adds no contention there.
        if gpu != retained_origin and any(value["planned_processes"] == 2 for value in readings):
            if statistics.mean(value["utilization"] for value in readings) > utilization_limit:
                raise RuntimeError(f"GPU {gpu} is too busy for a second slot")
    return measured


def process_start(pid):
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def drain(start, request_path, request):
    pid = start["pid"]
    start_time = process_start(pid)
    descriptor = (os.pidfd_open(pid) if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal") else None)
    try:
        if actor_state(start)["alive"] is not True or process_start(pid) != start_time:
            raise RuntimeError("The expected aggregate actor no longer matches its start receipt")
        proof = freeze(request_path, dict(request, pid=pid, proc_start_time=start_time,
                                          signal="SIGTERM", at_unix=time.time()))
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        else:
            if actor_state(start)["alive"] is not True or process_start(pid) != start_time:
                raise RuntimeError("The expected aggregate actor changed before the graceful signal")
            os.kill(pid, signal.SIGTERM)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    emit("DRAIN_REQUESTED", pid=pid, parent=request["parent"], receipt=proof["sha256"])
    deadline = time.monotonic() + 240
    while actor_state(start)["alive"] is not False:
        if time.monotonic() >= deadline:
            raise RuntimeError("The retiring process has not yet closed its current call")
        time.sleep(1)


def validate_journal(raw, keys):
    if not raw or not raw.endswith(b"\n"):
        raise RuntimeError("The primary journal lacks a complete checkpoint")
    pending, completed = set(), {}
    for line in raw.split(b"\n"):
        if not line:
            continue
        event = json.loads(line)
        key = task_key(event, STAGE)
        if key not in keys or key in completed:
            raise RuntimeError("The retiring journal contains an out-of-scope or repeated key")
        if event["event"] == "TASK_ENTER":
            if key in pending:
                raise RuntimeError("The retiring journal repeats an in-flight key")
            pending.add(key)
        elif event["event"] == "TASK_RESULT" and event["status"] == "SUCCESS" and key in pending:
            pending.remove(key)
            completed[key] = event
        else:
            raise RuntimeError("The retiring journal has a failed or unaccounted native attempt")
    if pending:
        raise RuntimeError("The retiring journal still has open native attempts")
    return completed


def checkpoint(state, entry, keys, start):
    digest = entry["manifest_ref"]["sha256"]
    folder = state["owned"] / "_dispatcher" / digest
    summary = read(folder / "SUMMARY.json")
    raw = (folder / "EVENTS.jsonl").read_bytes()
    successes = validate_journal(raw, keys)
    safe_stop = summary["status"] == "SAFELY_STOPPED" and summary["stopped"] is True
    complete = summary["status"] == "COMPLETE" and len(successes) == len(keys)
    if (not (safe_stop or complete) or summary["pid"] != start["pid"] or summary["failures"]
            or summary["scope_sha256"] != state["scope_sha"] or summary["manifest_sha256"] != digest
            or summary["worker_id"] != entry["worker_id"] or summary["completed"] != len(successes)
            or summary["planned"] != len(keys) or summary["stage"] != STAGE
            or actor_state(start)["alive"] is not False):
        raise RuntimeError("The original actor has not published a clean terminal checkpoint")
    folder = state["owned"] / state["scope_sha"] / STAGE
    for key in keys:
        owner = folder / canonical(list(key))
        event = successes.get(key)
        if event is None:
            if owner.exists():
                raise RuntimeError("Unjournaled native attempt requires reconciliation")
            continue
        fence = read(owner / "ATTEMPT_STATE.json")
        native = read(owner / "LAST_TASK_RECEIPT.json")
        if (fence["status"] != "SUCCESS_COMMITTED" or fence["scope_sha256"] != state["scope_sha"]
                or tuple(fence["task_key"]) != key or fence["stage"] != STAGE
                or fence["approval_sha256"] != state["plan"]["approval_ref"]["sha256"]
                or fence["native_record_sha256"] != event["native_record_sha256"]
                or native["status"] != "SUCCESS" or native["from_cache"] is not False
                or native["physical_attempt_id"] != event["physical_attempt_id"]
                or native["native_record_sha256"] != event["native_record_sha256"]):
            raise RuntimeError("A committed owner does not match the retiring journal")
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), keys - successes.keys()


def divide(remaining, gpus):
    grouped = defaultdict(list)
    for key in remaining:
        grouped[key[:2]].append(key)
    result = [dict(slot=slot, gpu=gpu, keys=[]) for slot, gpu in enumerate(gpus)]
    for group in sorted(grouped, key=canonical):
        target = min(result, key=lambda slot: len(slot["keys"]))
        target["keys"].extend(sorted(grouped[group]))
    if not all(part["keys"] for part in result):
        raise ValueError("Cannot divide the remaining keys into two nonempty slots")
    return result


def spawn(state, entry, name, gpu, script, args, placement, **extra):
    control = state["root"] / "control"
    filename = control / f"{name}.start.json"
    if filename.exists():
        raise FileExistsError("A start receipt for the exact slot already exists")
    digest = entry["manifest_ref"]["sha256"]
    argv = [sys.executable, "-B", "-u", str(script), "--manifest", entry["manifest_ref"]["path"],
            "--manifest-sha256", digest, "--gpu", str(gpu), *args]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false")
    log = control / f"{name}.log"
    with log.open("xb") as stream:
        proc = subprocess.Popen(argv, cwd=state["root"], env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    freeze(filename, dict(status="LAUNCHED", pid=proc.pid, gpu=gpu, argv=argv,
                          worker_id=entry["worker_id"], stage=STAGE, manifest_sha256=digest,
                          at_unix=time.time(), log_path=str(log), admission=placement, **extra))
    return proc.pid


def dispatch(args):
    state = context(ROOT)
    if state["plan_sha256"] != PLAN_SHA:
        raise ValueError("The original E24 plan has changed")
    control = ROOT / "control"
    entry, keys = next((e, k) for e, k in state["manifests"] if Path(e["manifest_ref"]["path"]).stem == args.parent)
    reference = entry["manifest_ref"]
    wrapper, child_script = (ROOT / "code" / name for name in ("supplement_worker.py", "parallel_tail_worker.py"))
    if (entry["stage"] != STAGE or sha(wrapper) != state["plan"]["source_sha256"]["supplement_worker.py"]
            or sha(child_script) != args.worker_sha256):
        raise ValueError("The two required worker sources do not match the admitted code")
    starts = [row for row in actors(ROOT).values() if row.get("manifest_sha256") == reference["sha256"]]
    if len(starts) != 1 or starts[0]["pid"] != args.pid or starts[0]["gpu"] != args.gpus[0]:
        raise RuntimeError("The retiring E24 actor does not match the exact original placement")
    start = starts[0]
    expected = [sys.executable, "-B", "-u", str(wrapper), "--manifest", reference["path"],
                "--manifest-sha256", reference["sha256"], "--gpu", str(args.gpus[0])]
    if start["argv"] != expected:
        raise RuntimeError("The retiring actor does not use the original frozen worker")
    assignment_path = control / f"{args.parent}.parallel_tail.assignment.json"
    stop_path = control / f"{args.parent}.parallel_tail.stop.json"
    affected_paths = [assignment_path, stop_path]
    for slot in (0, 1):
        affected_paths.extend(control / f"{args.parent}.tail{slot}.{suffix}" for suffix in ("log", "start.json"))
    if any(path.exists() for path in affected_paths):
        raise FileExistsError("This aggregate parent already has a tail control artifact")
    summary = read(state["owned"] / "_dispatcher" / reference["sha256"] / "SUMMARY.json")
    pending = len(keys) - summary["completed"]
    if (summary["pid"] != args.pid or summary["status"] != "RUNNING" or summary["failures"]
            or pending < 160 or summary["ready"] < 100):
        raise RuntimeError("The aggregate parent lacks sufficient ready work for a split")
    placement = admission(ROOT, args.gpus, args.pid, retained_origin=args.gpus[0])
    emit("ADMITTED", parent=args.parent, gpus=args.gpus, pending=pending,
         min_free_mib={gpu: min(x["free_mib"] for x in rows) for gpu, rows in placement.items()})
    if not args.apply:
        return
    drain(start, stop_path, dict(parent=args.parent, plan_sha256=PLAN_SHA,
                                scope_sha256=state["scope_sha"], manifest_ref=reference, planned_gpus=args.gpus))
    journal, remaining = checkpoint(state, entry, keys, start)
    if not remaining:
        emit("COMPLETED_WHILE_DRAINING", parent=args.parent)
        return
    try:
        placement = admission(ROOT, args.gpus, samples=3, retained_origin=args.gpus[0])
    except RuntimeError:
        gpu = start["gpu"]
        placement = admission(ROOT, (gpu,), samples=2, utilization_limit=100)
        pid = spawn(state, entry, f"{args.parent}.tail_fallback", gpu, wrapper, (), placement,
                    original_pid=start["pid"], restored_successes=len(keys)-len(remaining))
        emit("RESUMED_ORIGINAL_PLACEMENT", parent=args.parent, pid=pid, gpu=gpu, remaining=len(remaining))
        return
    children = divide(remaining, args.gpus)
    packet = dict(schema="e24_parallel_tail_assignment_v1", plan_sha256=PLAN_SHA, scope_sha256=state["scope_sha"],
                  manifest_ref=reference, journal_checkpoint=journal, children=children, retired_pid=start["pid"])
    ref = freeze(assignment_path, packet)
    for child in children:
        slot, gpu = child["slot"], child["gpu"]
        pid = spawn(state, entry, f"{args.parent}.tail{slot}", gpu, child_script,
                    ("--assignment", str(assignment_path), "--assignment-sha256", ref["sha256"], "--slot", str(slot)),
                    placement[gpu], parallel_slot=slot, assignment_ref=ref, assigned_keys=len(child["keys"]),
                    original_pid=start["pid"], original_plan_sha256=PLAN_SHA)
        emit("SLOT_LAUNCHED", parent=args.parent, slot=slot, gpu=gpu, pid=pid, keys=len(child["keys"]),
             restored_parent_keys=len(keys)-len(remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", choices=tuple(f"aggregation{i}" for i in range(8)), required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.gpus[0] == args.gpus[1] or any(not 0 <= gpu <= 7 for gpu in args.gpus):
        parser.error("Choose two distinct GPU indices")
    if args.apply:
        with (ROOT / "control" / "parallel_tail_dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            dispatch(args)
    else:
        dispatch(args)


if __name__ == "__main__":
    main()
