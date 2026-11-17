"""Launch a disjoint P2 aggregation shard with the existing admission primitives."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from launch_registered_worker_v3 import (
    DEFAULT_PYTHON, admit_gpu, assert_no_live_overlap, build_argv,
    start_receipts, stopped_predecessor_barrier,
)
from p2_manifest_registry import append_lock, load_registry, read_json, reference
from prepare_aggregation_shards_v4 import validate_dispatch_plan


def launch(registry_path, worker_id, gpu, dispatch_plan, required_free_mib,
           shared_authorized, python=DEFAULT_PYTHON):
    registry = load_registry(registry_path)
    if registry["software_test_only"]:
        raise PermissionError("The launcher rejects software-only registries")
    root = Path(registry["common"]["owned_output_root"]).parent
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    with append_lock(control / "registered-launch"):
        registry = load_registry(registry_path)
        entries = [entry for entry in registry["entries"] if entry["worker_id"] == worker_id]
        if len(entries) != 1:
            raise ValueError("The explicit worker ID must resolve to one registered manifest")
        entry = entries[0]
        if entry["stage"] != "aggregation" or not entry["task_keys"]:
            raise ValueError("A nonempty registered aggregation shard is required")
        plan = read_json(dispatch_plan)
        plan_ref = reference(dispatch_plan)
        qualification = validate_dispatch_plan(registry, plan)
        planned = [item for item in plan["shards"] if item["worker_id"] == worker_id]
        if len(planned) != 1 or any(planned[0][name] != entry[name]
                                  for name in ("manifest_ref", "worker_source_ref")):
            raise ValueError("The registered identity differs from the prepared dispatch shard")
        starts = start_receipts(root)
        active = assert_no_live_overlap(registry, entry, starts)
        barriers = stopped_predecessor_barrier(registry, entry, starts)
        snapshots = admit_gpu(gpu, active, required_free_mib, shared_authorized)
        argv = build_argv(registry_path, entry, gpu, python)
        log = control / (worker_id + "-" + str(time.time_ns()) + ".log")
        environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        with log.open("xb") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=root, env=environment, start_new_session=True)
        receipt = dict(status="LAUNCHED", worker_id=worker_id, stage="aggregation",
                       profile="aggregation", member_index=None, gpu=gpu, pid=process.pid,
                       manifest_sha256=entry["manifest_ref"]["sha256"],
                       worker_source_sha256=entry["worker_source_ref"]["sha256"],
                       registry_path=str(Path(registry_path).resolve()),
                       registry_head=registry["last_event_sha256"],
                       scope_sha256=registry["scope_canonical_sha256"],
                       approval_ref=registry["common"]["approval_ref"],
                       owned_output_root=registry["common"]["owned_output_root"],
                       task_keys=len(entry["task_keys"]), dispatch_plan_ref=plan_ref,
                       dispatch_plan_check=qualification, predecessor_stop_checks=barriers,
                       admitted_gpu_snapshots=snapshots, required_free_mib=required_free_mib,
                       shared_authorization_used=bool(shared_authorized), argv=argv,
                       log_path=str(log), at_unix=time.time(), reference_access=False,
                       scope_model_or_generation_changes=0, launcher_version=4)
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
    parser.add_argument("--dispatch-plan", type=Path, required=True)
    parser.add_argument("--required-free-mib", type=int, default=14336)
    parser.add_argument("--shared-authorized", action="store_true")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    args = parser.parse_args()
    print(json.dumps(launch(args.registry, args.worker_id, args.gpu, args.dispatch_plan,
                            args.required_free_mib, args.shared_authorized, args.python)))
