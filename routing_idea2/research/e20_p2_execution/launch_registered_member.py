"""Launch a frozen member continuation using the existing ownership primitives."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from launch_registered_worker_v3 import (
    DEFAULT_PYTHON, admit_gpu, assert_no_live_overlap, build_argv,
    matching_process_alive, start_receipts,
)
from p2_manifest_registry import append_lock, iter_complete_events, load_registry, read_json, reference
from prepare_member_shards import validate_dispatch_plan


def checked_member_checkpoint(registry, entry, start):
    summary_path = Path(entry["journal_path"]).with_name("SUMMARY.json")
    summary = read_json(summary_path)
    if (summary.get("schema") != "e18_gpu_worker_summary_v1"
            or summary.get("manifest_sha256") != entry["manifest_ref"]["sha256"]
            or summary.get("stage") != "member_generation"
            or summary.get("profile") != entry["member_index"]
            or summary.get("software_test_only") is not registry["software_test_only"]
            or summary.get("planned") != len(entry["task_keys"])
            or summary.get("failures") != [] or summary_path.stat().st_mtime < start.get("at_unix", 0)):
        raise RuntimeError("The member checkpoint differs from the exact registered invocation")
    completed = summary.get("completed")
    if type(completed) is not int or not 0 <= completed <= summary["planned"]:
        raise RuntimeError("Invalid member completion count")
    if summary.get("stop_requested") is not True and not (
            summary.get("stop_requested") is False and completed == summary["planned"]):
        raise RuntimeError("A member predecessor lacks a successful stop or completion")
    events = [event for event in iter_complete_events(entry["journal_path"])
              if event.get("event") == "TASK_RESULT" and event.get("status") == "SUCCESS"
              and event.get("at_unix", 0) >= start.get("at_unix", 0)]
    keys = [(row["task"]["question_id"], row["task"]["generation_seed"], row["task"]["member_index"])
            for row in events]
    if len(events) != completed or len(keys) != len(set(keys)) or not set(keys) <= set(entry["task_keys"]):
        raise RuntimeError("Member checkpoint does not recover its committed invocation journal")
    return dict(worker_id=entry["worker_id"], pid=start["pid"], completed=completed,
                status="SAFELY_STOPPED" if summary["stop_requested"] else "COMPLETE",
                member_summary_schema=summary["schema"])


def stopped_member_predecessors(registry, entry, starts, alive=matching_process_alive):
    latest = {}
    for start in starts:
        key = start.get("manifest_sha256")
        if key not in latest or start.get("at_unix", 0) > latest[key].get("at_unix", 0):
            latest[key] = start
    checks = []
    own_keys = set(entry["task_keys"])
    for previous in registry["entries"]:
        start = latest.get(previous["manifest_ref"]["sha256"])
        if (start is None or previous["stage"] != "member_generation"
                or not own_keys.intersection(previous["task_keys"])):
            continue
        if alive(start):
            raise RuntimeError("An overlapping member predecessor remains active")
        checks.append(checked_member_checkpoint(registry, previous, start))
    return checks


def launch(registry_path, worker_id, gpu, dispatch_plan, required_free_mib,
           shared_authorized, python=DEFAULT_PYTHON):
    registry = load_registry(registry_path)
    if registry["software_test_only"]:
        raise PermissionError("The native launcher rejects synthetic registries")
    root = Path(registry["common"]["owned_output_root"]).parent
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    with append_lock(control / "registered-launch"):
        registry = load_registry(registry_path)
        entries = [row for row in registry["entries"] if row["worker_id"] == worker_id]
        if len(entries) != 1 or entries[0]["stage"] != "member_generation" or not entries[0]["task_keys"]:
            raise ValueError("One explicit nonempty registered member subset is required")
        entry = entries[0]
        plan = read_json(dispatch_plan)
        plan_ref = reference(dispatch_plan)
        qualification = validate_dispatch_plan(registry, plan)
        planned = [row for row in plan["shards"] if row["worker_id"] == worker_id]
        if len(planned) != 1 or any(planned[0][name] != entry[name] for name in ("manifest_ref", "worker_source_ref")):
            raise ValueError("Selected member identity differs from the dispatch plan")
        starts = start_receipts(root)
        active = assert_no_live_overlap(registry, entry, starts)
        barriers = stopped_member_predecessors(registry, entry, starts)
        snapshots = admit_gpu(gpu, active, required_free_mib, shared_authorized)
        argv = build_argv(registry_path, entry, gpu, python)
        log = control / (worker_id + "-" + str(time.time_ns()) + ".log")
        environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        with log.open("xb") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=root, env=environment, start_new_session=True)
        receipt = dict(status="LAUNCHED", worker_id=worker_id, stage="member_generation",
                       profile="member" + str(entry["member_index"]), member_index=entry["member_index"],
                       gpu=gpu, pid=process.pid, manifest_sha256=entry["manifest_ref"]["sha256"],
                       worker_source_sha256=entry["worker_source_ref"]["sha256"],
                       registry_path=str(Path(registry_path).resolve()), registry_head=registry["last_event_sha256"],
                       scope_sha256=registry["scope_canonical_sha256"], approval_ref=registry["common"]["approval_ref"],
                       owned_output_root=registry["common"]["owned_output_root"], task_keys=len(entry["task_keys"]),
                       dispatch_plan_ref=plan_ref, dispatch_plan_check=qualification,
                       predecessor_stop_checks=barriers, admitted_gpu_snapshots=snapshots,
                       required_free_mib=required_free_mib, shared_authorization_used=bool(shared_authorized),
                       argv=argv, log_path=str(log), at_unix=time.time(), reference_access=False,
                       scope_model_or_generation_changes=0)
        with (control / (worker_id + "-" + str(process.pid) + ".start.json")).open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dispatch-plan", type=Path, required=True)
    parser.add_argument("--required-free-mib", type=int, default=14336)
    parser.add_argument("--shared-authorized", action="store_true")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    args = parser.parse_args()
    print(json.dumps(launch(args.registry, args.worker_id, args.gpu, args.dispatch_plan,
                            args.required_free_mib, args.shared_authorized, args.python)))
