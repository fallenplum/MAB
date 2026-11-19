"""Dispatch only named E24 manifests as their approved GPU placements become available."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from execution_monitor import actor_state, read, sha, start_receipts
import launch_supplement


HEADROOM_MIB = 14336  # Original launcher preflight reserve for one unloaded worker.


def query(fields, kind):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = [[value.strip() for value in row] for row in csv.reader(result.stdout.splitlines())]
    return [row for row in rows if any(row) and row != ["No running processes found"]]


def gpu_snapshot():
    devices, by_uuid = {}, {}
    for index, uuid, free, util in query("index,uuid,memory.free,utilization.gpu", "gpu"):
        idx = int(index)
        devices[idx] = dict(free_mib=int(free), utilization=int(util), pids=set())
        by_uuid[uuid] = idx
    for uuid, pid in query("gpu_uuid,pid", "compute-apps"):
        # An unavailable PID still counts as foreign activity for admission.
        devices[by_uuid[uuid]]["pids"].add(int(pid) if pid.isdecimal() else None)
    return devices


def read_manifests(root):
    plan = read(root / "EXECUTION_PLAN.json")
    if plan.get("cohort") != "e24_training_deployment_replication" or Path(plan["root"]).resolve() != root:
        raise ValueError("E24 dispatch requires the exact registered execution plan")
    if plan["source_sha256"].get("launch_supplement.py") != sha(Path(launch_supplement.__file__)):
        raise ValueError("Original launcher differs from the registered execution plan")
    manifests = {}
    for row in plan["manifests"]:
        path = Path(row["manifest_ref"]["path"]).resolve(strict=True)
        if path.parent != root / "manifests" or path.stem in manifests:
            raise ValueError("Unexpected manifest location or duplicate manifest name")
        manifests[path.stem] = row
    return manifests


def placements_from(path, manifests, gpu_indices):
    raw = read(path)
    if not isinstance(raw, dict) or not raw or not raw.keys() <= manifests.keys():
        raise ValueError("Placements must name only workers in the registered E24 plan")
    placements = {}
    for name, value in raw.items():
        candidates = [value] if type(value) is int else value
        if (not isinstance(candidates, list) or not candidates
                or any(type(idx) is not int or idx not in gpu_indices for idx in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError(f"Invalid GPU placements for {name}")
        placements[name] = candidates
    return placements


def active_workers(root, manifests):
    registered = {row["manifest_ref"]["sha256"]: name for name, row in manifests.items()}
    active = {}
    started = set()
    for receipt in start_receipts(root):
        manifest_hash = receipt.get("manifest_sha256")
        if manifest_hash not in registered:
            raise ValueError("Unexpected start receipt in the E24 control directory")
        started.add(registered[manifest_hash])
        state = actor_state(receipt)
        # Unknown process state must hold its reservation as long as its PID is observed.
        if not state["alive"] and not Path(f"/proc/{receipt['pid']}").is_dir():
            continue
        if not state["alive"] and state["state"] in ("EXITED_ZOMBIE", "PID_REUSED_OR_COMMAND_DIFFERS"):
            continue
        if receipt["pid"] in active:
            continue
        summary_path = root / "owned" / "_dispatcher" / manifest_hash / "SUMMARY.json"
        summary = read(summary_path) if summary_path.is_file() else {}
        loaded = summary.get("pid") == receipt["pid"] and summary.get("backend_loads", 0) >= 1
        active[receipt["pid"]] = (receipt["gpu"], loaded)
    return active, started


def choose_gpu(candidates, devices, active, max_exclusive, max_shared, max_shared_util):
    available = []
    for order, idx in enumerate(candidates):
        device = devices.get(idx)
        if device is None:
            continue
        own = {pid: loaded for pid, (gpu, loaded) in active.items() if gpu == idx}
        foreign = bool(device["pids"] - own.keys())
        capacity = max_shared if foreign else max_exclusive
        pending_reservations = sum(not loaded for loaded in own.values())
        if (len(own) >= capacity or device["free_mib"] < HEADROOM_MIB * (1 + pending_reservations)
                or foreign and device["utilization"] > max_shared_util):
            continue
        available.append((foreign, len(own), device["utilization"], -device["free_mib"], order, idx))
    return min(available)[-1] if available else None


def report(status, **details):
    print(json.dumps(dict(status=status, at_unix=time.time(), **details), ensure_ascii=False), flush=True)


def dispatch(args, root, manifests, placements):
    pending = dict(placements)
    failed = []
    while pending:
        active, started = active_workers(root, manifests)
        devices = None
        for name in tuple(pending):
            if name in started:
                pending.pop(name)
                continue
            if (root / "control" / f"{name}.log").exists():
                # A process might have been forked before its launcher wrote the receipt.
                pending.pop(name)
                failed.append(name)
                report("UNRESOLVED_EXISTING_WORKER_LOG", worker=name)
                continue
            if devices is None:
                devices = gpu_snapshot()
            gpu = choose_gpu(pending[name], devices, active, args.max_exclusive,
                             args.max_shared, args.max_shared_util)
            if gpu is None:
                continue
            try:
                record = launch_supplement.launch(root, name, gpu)
            except RuntimeError as error:
                if str(error) == "The frozen NF4 profile lacks its required loading and KV headroom":
                    devices = None
                    report("ADMISSION_CHANGED_BEFORE_LAUNCH", worker=name, gpu=gpu)
                    continue
                raise
            except FileExistsError:
                active, started = active_workers(root, manifests)
                devices = None
                continue
            pending.pop(name)
            report("LAUNCHED", worker=name, gpu=gpu, pid=record["pid"], tasks=record["tasks"])
            active, started = active_workers(root, manifests)
            devices = None
        if pending:
            report("WAITING_FOR_GPU_ADMISSION", pending=list(pending))
            time.sleep(args.interval)
    report("DEFERRED_DISPATCH_COMPLETE", unresolved_workers=failed)
    return int(bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--max-exclusive", type=int, default=3)
    parser.add_argument("--max-shared", type=int, default=1)
    parser.add_argument("--max-shared-util", type=int, default=40)
    args = parser.parse_args()
    if (args.interval <= 0 or not 1 <= args.max_exclusive <= 8
            or not 0 <= args.max_shared <= args.max_exclusive or not 0 <= args.max_shared_util <= 100):
        parser.error("Invalid dispatch interval or GPU capacities")
    root = args.root.resolve(strict=True)
    manifests = read_manifests(root)
    placements = placements_from(args.placements, manifests, gpu_snapshot().keys())
    control = root / "control"
    control.mkdir(exist_ok=True)
    with (control / "deferred_dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return dispatch(args, root, manifests, placements)


if __name__ == "__main__":
    sys.exit(main())
