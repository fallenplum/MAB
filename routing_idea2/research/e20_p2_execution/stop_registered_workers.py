"""Gracefully stop only explicit, command-matched registered P2 worker PIDs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal

from launch_registered_member import checked_member_checkpoint
from launch_registered_worker_v3 import matching_process_alive, start_receipts
from p2_manifest_registry import iter_successes, load_registry, read_json, reference


def write_once(folder, name, value):
    Path(folder).mkdir(parents=True, exist_ok=True)
    with (Path(folder) / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def request_stop(registry_path, pids, record_dir):
    registry = load_registry(registry_path)
    if registry["software_test_only"]:
        raise PermissionError("A production registry is required")
    if not pids or len(pids) != len(set(pids)) or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ValueError("Explicit unique positive owned worker PIDs are required")
    root = Path(registry["common"]["owned_output_root"]).parent
    starts = start_receipts(root)
    by_manifest = {entry["manifest_ref"]["sha256"]: entry for entry in registry["entries"]}
    selected = []
    for pid in pids:
        candidates = [start for start in starts if start.get("pid") == pid and matching_process_alive(start)]
        if len(candidates) != 1:
            raise RuntimeError("A requested PID no longer matches one owned live invocation")
        start = candidates[0]
        entry = by_manifest.get(start.get("manifest_sha256"))
        if (entry is None or reference(entry["worker_source_ref"]["path"]) != entry["worker_source_ref"]
                or start.get("worker_source_sha256") != entry["worker_source_ref"]["sha256"]
                or (start.get("scope_sha256") is not None
                    and start["scope_sha256"] != registry["scope_canonical_sha256"])):
            raise RuntimeError("A requested actor differs from its registered source")
        selected.append(dict(start=start, worker_id=entry["worker_id"], stage=entry["stage"],
                             member_index=entry["member_index"], manifest_ref=entry["manifest_ref"]))
    stages = {row["stage"] for row in selected}
    profiles = {row["member_index"] for row in selected}
    if len(stages) != 1 or (stages == {"member_generation"} and len(profiles) != 1):
        raise ValueError("One stop group must contain one stage and one member profile")
    receipt = dict(status="GRACEFUL_STOP_REQUESTED", utc=datetime.now(timezone.utc).isoformat(),
                   registry_path=str(Path(registry_path).resolve()), registry_head=registry["last_event_sha256"],
                   scope_canonical_sha256=registry["scope_canonical_sha256"], selected=selected,
                   foreign_process_signals=0, reference_values_decoded=0)
    write_once(record_dir, "01_STOP_REQUEST.json", receipt)
    for row in selected:
        if not matching_process_alive(row["start"]):
            raise RuntimeError("An owned actor changed immediately before SIGTERM")
        os.kill(row["start"]["pid"], signal.SIGTERM)
    return dict(status=receipt["status"], pids=pids, stage=next(iter(stages)))


def verify_stopped(registry_path, record_dir):
    request = read_json(Path(record_dir) / "01_STOP_REQUEST.json")
    registry = load_registry(registry_path)
    if request["scope_canonical_sha256"] != registry["scope_canonical_sha256"]:
        raise ValueError("Stop request scope differs")
    alive = [row["start"]["pid"] for row in request["selected"] if matching_process_alive(row["start"])]
    if alive:
        return dict(status="WAITING_FOR_CURRENT_NATIVE_CALLS_TO_COMMIT", pids=alive)
    by_id = {entry["worker_id"]: entry for entry in registry["entries"]}
    checks = []
    for row in request["selected"]:
        entry, start = by_id[row["worker_id"]], row["start"]
        if entry["manifest_ref"] != row["manifest_ref"]:
            raise ValueError("Registered predecessor identity changed")
        if row["stage"] == "member_generation":
            checks.append(checked_member_checkpoint(registry, entry, start))
        else:
            summary = read_json(Path(entry["journal_path"]).with_name("SUMMARY.json"))
            if (summary.get("status"), summary.get("stopped"), summary.get("pid"), summary.get("scope_sha256")) != (
                    "SAFELY_STOPPED", True, start["pid"], registry["scope_canonical_sha256"]):
                raise RuntimeError("An aggregate predecessor lacks its successful graceful checkpoint")
            successes = [key for _, _, key in iter_successes(dict(registry, entries=[entry]), "aggregation")]
            if len(set(successes)) != summary["completed"]:
                raise RuntimeError("Aggregate checkpoint differs from committed journal success keys")
            checks.append(dict(worker_id=entry["worker_id"], pid=start["pid"], status="SAFELY_STOPPED",
                               completed=summary["completed"]))
    stage = request["selected"][0]["stage"]
    member_index = request["selected"][0]["member_index"]
    done = {key for _, _, key in iter_successes(registry, stage)
            if stage == "aggregation" or key[2] == member_index}
    receipt = dict(status="OWNED_WORKERS_SAFELY_STOPPED", utc=datetime.now(timezone.utc).isoformat(),
                   stage=stage, member_index=member_index, scope_canonical_sha256=registry["scope_canonical_sha256"],
                   completed_unique_keys=len(done), predecessors=checks, reference_values_decoded=0)
    write_once(record_dir, "02_STOPPED_CHECKPOINT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pids", type=int, nargs="+")
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("request", "verify"), required=True)
    args = parser.parse_args()
    if args.phase == "request":
        result = request_stop(args.registry, args.pids, args.record_dir)
    else:
        result = verify_stopped(args.registry, args.record_dir)
    print(json.dumps(result))
