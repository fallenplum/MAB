"""Launch one explicitly registered worker after scope and live-key admission.

This file provides an execution tool. Importing it or preparing dispatch plans
does not launch processes, probe a remote server, or load a language model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from launch_native_profile import gpu_state
from p2_manifest_registry import append_lock, load_registry, read_json, reference
from prepare_aggregation_shards_v3 import validate_dispatch_plan


DEFAULT_PYTHON = "/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python"


def matching_process_alive(start):
    """A reused PID or a zombie is not the actor described by this receipt."""
    process = Path("/proc") / str(start["pid"])
    try:
        command = (process / "cmdline").read_bytes().split(b"\0")
        state = (process / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    expected = [str(value).encode() for value in start.get("argv", ())]
    return bool(expected) and command[:-1] == expected and state != "Z"


def start_receipts(root):
    return [read_json(path) for path in sorted((Path(root) / "control").glob("*.start.json"))]


def assert_no_live_overlap(registry, entry, starts, alive=matching_process_alive):
    """Historical coverage may overlap. Live actors' remaining jobs may not."""
    own_keys = set(entry["task_keys"])
    by_manifest = {item["manifest_ref"]["sha256"]: item for item in registry["entries"]}
    active = []
    for start in starts:
        if not alive(start):
            continue
        previous = by_manifest.get(start.get("manifest_sha256"))
        if previous is None:
            raise RuntimeError("A live actor is missing from the explicit evidence registry")
        if previous["stage"] == entry["stage"] and own_keys.intersection(previous["task_keys"]):
            raise RuntimeError("A live actor still owns overlapping stage/task keys")
        active.append(start)
    return active


def stopped_predecessor_barrier(registry, entry, starts, alive=matching_process_alive):
    """A planned replacement must follow a completed or gracefully stopped actor."""
    own_keys = set(entry["task_keys"])
    latest = {}
    for start in starts:
        manifest_sha = start.get("manifest_sha256")
        previous = latest.get(manifest_sha)
        if previous is None or start.get("at_unix", 0) > previous.get("at_unix", 0):
            latest[manifest_sha] = start
    checked = []
    for previous in registry["entries"]:
        old = latest.get(previous["manifest_ref"]["sha256"])
        if old is None or previous["stage"] != entry["stage"] or not own_keys.intersection(previous["task_keys"]):
            continue
        if alive(old):
            raise RuntimeError("An overlapping predecessor has not exited")
        summary_path = Path(previous["journal_path"]).with_name("SUMMARY.json")
        summary = read_json(summary_path)
        stopped = summary.get("status") == "SAFELY_STOPPED" and summary.get("stopped") is True
        complete = summary.get("status") == "COMPLETE"
        if not (stopped or complete) or summary.get("scope_sha256") != registry["scope_canonical_sha256"]:
            raise RuntimeError("An overlapping predecessor lacks its successful stop/complete checkpoint")
        if summary.get("pid") != old["pid"]:
            raise RuntimeError("The stop checkpoint belongs to another actor")
        checked.append(dict(worker_id=previous["worker_id"], pid=old["pid"], status=summary["status"]))
    return checked


def build_argv(registry_path, entry, gpu, python=DEFAULT_PYTHON):
    argv = [python, "-B", entry["worker_source_ref"]["path"], "--manifest", entry["manifest_ref"]["path"],
            "--manifest-sha256", entry["manifest_ref"]["sha256"], "--gpu", str(gpu)]
    if entry["stage"] == "aggregation":
        argv.extend(["--registry", str(Path(registry_path).resolve()), "--worker-id", entry["worker_id"]])
    return argv


def admit_gpu(gpu, active, required_free_mib, shared_authorized):
    if type(required_free_mib) is not int or required_free_mib <= 0:
        raise ValueError("An explicit positive profile loading/KV headroom requirement is needed")
    snapshots = []
    owned_pids = {row["pid"] for row in active}
    for index in range(3):
        state = gpu_state(gpu)
        if state["free_mib"] < required_free_mib:
            raise RuntimeError("Live free VRAM fell below the declared frozen-profile headroom")
        foreign = [row for row in state["compute_processes"] if row["pid"] not in owned_pids]
        if foreign and not shared_authorized:
            raise PermissionError("This shared GPU requires the existing user's shared-resource authorization flag")
        snapshots.append(state)
        if index < 2:
            time.sleep(0.5)
    return snapshots


def launch(registry_path, worker_id, gpu, dispatch_plan, required_free_mib, shared_authorized, python=DEFAULT_PYTHON):
    registry = load_registry(registry_path)
    if registry["software_test_only"]:
        raise PermissionError("The launcher rejects software-only registries")
    root = Path(registry["common"]["owned_output_root"]).parent
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    with append_lock(control / "registered-launch"):
        registry = load_registry(registry_path)
        selected = [entry for entry in registry["entries"] if entry["worker_id"] == worker_id]
        if len(selected) != 1:
            raise ValueError("The explicit worker ID must resolve to one registered manifest")
        entry = selected[0]
        if not entry["task_keys"]:
            raise ValueError("No remaining task keys need a worker")
        plan_ref = None
        plan_check = None
        if entry["stage"] == "aggregation":
            if dispatch_plan is None:
                raise ValueError("A prepared disjoint aggregation dispatch plan is required")
            plan = read_json(dispatch_plan)
            plan_ref = reference(dispatch_plan)
            plan_check = validate_dispatch_plan(registry, plan)
            planned = [item for item in plan["shards"] if item["worker_id"] == worker_id]
            if len(planned) != 1 or any(planned[0][name] != entry[name] for name in ("manifest_ref", "worker_source_ref")):
                raise ValueError("The selected registry entry differs from this plan's exact shard identity")
        starts = start_receipts(root)
        active = assert_no_live_overlap(registry, entry, starts)
        barriers = stopped_predecessor_barrier(registry, entry, starts)
        snapshots = admit_gpu(gpu, active, required_free_mib, shared_authorized)
        argv = build_argv(registry_path, entry, gpu, python)
        log = control / (worker_id + "-" + str(time.time_ns()) + ".log")
        environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           TOKENIZERS_PARALLELISM="false")
        with log.open("xb") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=root,
                                       env=environment, start_new_session=True)
        receipt = dict(status="LAUNCHED", worker_id=worker_id, stage=entry["stage"],
                       profile="aggregation" if entry["member_index"] is None else "member" + str(entry["member_index"]),
                       member_index=entry["member_index"], gpu=gpu, pid=process.pid,
                       manifest_sha256=entry["manifest_ref"]["sha256"],
                       worker_source_sha256=entry["worker_source_ref"]["sha256"],
                       registry_path=str(Path(registry_path).resolve()), registry_head=registry["last_event_sha256"],
                       scope_sha256=registry["scope_canonical_sha256"], approval_ref=registry["common"]["approval_ref"],
                       owned_output_root=registry["common"]["owned_output_root"],
                       task_keys=len(entry["task_keys"]), dispatch_plan_ref=plan_ref, dispatch_plan_check=plan_check,
                       predecessor_stop_checks=barriers, admitted_gpu_snapshots=snapshots,
                       required_free_mib=required_free_mib, shared_authorization_used=bool(shared_authorized),
                       argv=argv, log_path=str(log), at_unix=time.time(), reference_access=False,
                       scope_model_or_generation_changes=0)
        target = control / (worker_id + "-" + str(process.pid) + ".start.json")
        with target.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dispatch-plan", type=Path)
    parser.add_argument("--required-free-mib", type=int, default=12288)
    parser.add_argument("--shared-authorized", action="store_true")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    args = parser.parse_args()
    print(json.dumps(launch(args.registry, args.worker_id, args.gpu, args.dispatch_plan,
                            args.required_free_mib, args.shared_authorized, args.python)))
