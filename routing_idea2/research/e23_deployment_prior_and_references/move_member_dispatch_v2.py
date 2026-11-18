"""Move one exact member4 successor once, preserving its manifest and journal."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from prepare_member_dispatch_v2 import (
    PYTHON, actor_state, checkpoint, freeze, read, reference, sha,
    start_receipt, validate_dispatch,
)


TARGET_GPU = {2: 1, 3: 3}


def selected(root, dispatch_path, shard):
    if shard not in TARGET_GPU:
        raise ValueError("Only the two frozen GPU5 successor shards may move")
    state, dispatch, candidates = validate_dispatch(root, dispatch_path)
    matches = [item for item in candidates if item[0]["shard"] == shard]
    if len(matches) != 1:
        raise ValueError("An exact existing member4 successor is required")
    entry, manifest, keys = matches[0]
    start_path, start = start_receipt(state, entry)
    name = Path(entry["manifest_ref"]["path"]).stem
    return state, dispatch, entry, manifest, keys, start_path, start, name


def paths(state, name):
    control = state["root"]/"control"
    return (control/f"{name}.move_stop.json", control/f"{name}.move_checkpoint.json",
            control/f"{name}.resume1.start.json", control/f"{name}.resume1.log")


def request_identity(state, dispatch_path, entry, start_path, shard):
    return dict(schema="e23_member_dispatch_move_stop_v1", shard=shard,
                target_gpu=TARGET_GPU[shard], dispatch_ref=reference(dispatch_path),
                manifest_ref=entry["manifest_ref"], original_start_ref=reference(start_path),
                scope_canonical_sha256=state["scope_sha"])


def verify_request(path, expected):
    request = read(path)
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("The migration request binds a different actor or dispatch")
    return request


def stop(root, dispatch_path, shard):
    state, dispatch, entry, manifest, keys, start_path, start, name = selected(root, dispatch_path, shard)
    request_path, checkpoint_path, resume_path, log_path = paths(state, name)
    if resume_path.exists() or log_path.exists():
        raise FileExistsError("The one-time successor migration already has a resume artifact")
    expected = request_identity(state, dispatch_path, entry, start_path, shard)
    current = actor_state(start)
    if current.get("alive") is None:
        raise RuntimeError("The exact original PID and argv cannot be checked")
    if current.get("alive") is False:
        checkpoint(state, entry, manifest, keys)
    if request_path.exists():
        verify_request(request_path, expected)
    else:
        freeze(request_path, dict(expected, requested_at_unix=time.time(),
                                 signal="SIGTERM", foreign_processes_signaled=0))
    current = actor_state(start)
    if current.get("alive") is True:
        os.kill(start["pid"], signal.SIGTERM)
        status = "GRACEFUL_STOP_REQUESTED"
    elif current.get("alive") is False:
        checkpoint(state, entry, manifest, keys)
        status = "ORIGINAL_ACTOR_SAFELY_EXITED"
    else:
        raise RuntimeError("Actor ownership changed before SIGTERM")
    return dict(status=status, shard=shard, original_pid=start["pid"],
                stop_request_ref=reference(request_path), manifest_changed=False)


def admission(gpu):
    snapshots = []
    for index in range(3):
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=10)
        rows = [[int(value.strip()) for value in line.split(",")]
                for line in result.stdout.splitlines()]
        row = next(item for item in rows if item[0] == gpu)
        if row[1] < 14336:
            raise RuntimeError("Frozen NF4 loading and KV headroom fell below 14336 MiB")
        snapshots.append(dict(gpu=row[0], free_mib=row[1], utilization_percent=row[2]))
        if index < 2:
            time.sleep(0.5)
    return snapshots


def resume(root, dispatch_path, shard, gpu):
    if TARGET_GPU.get(shard) != gpu:
        raise ValueError("Frozen migration targets are shard2/GPU1 and shard3/GPU3")
    state, dispatch, entry, manifest, keys, start_path, start, name = selected(root, dispatch_path, shard)
    request_path, checkpoint_path, resume_path, log_path = paths(state, name)
    if resume_path.exists() or log_path.exists():
        raise FileExistsError("This actor already has its one permitted resume artifact")
    verify_request(request_path, request_identity(state, dispatch_path, entry, start_path, shard))
    safe, complete = checkpoint(state, entry, manifest, keys)
    snapshot = dict(schema="e23_member_move_safe_checkpoint_v1", shard=shard,
                    dispatch_ref=reference(dispatch_path), manifest_ref=entry["manifest_ref"],
                    original_start_ref=reference(start_path), checkpoint=safe,
                    original_summary=read(safe["stop_summary_ref"]["path"]),
                    completed=len(complete), remaining=len(keys)-len(complete))
    checkpoint_ref = freeze(checkpoint_path, snapshot)
    if complete == keys:
        return dict(status="COMPLETE_NO_RESUME_NEEDED", shard=shard,
                    completed=len(complete), checkpoint_ref=checkpoint_ref)
    snapshots = admission(gpu)
    worker = state["root"]/"code/supplement_worker.py"
    if sha(worker) != dispatch["worker_source_sha256"]:
        raise ValueError("The frozen worker wrapper changed before migration")
    # Recheck the exact stopped actor after the bounded GPU admission checks.
    checkpoint(state, entry, manifest, keys)
    argv = [PYTHON, "-B", "-u", str(worker), "--manifest", entry["manifest_ref"]["path"],
            "--manifest-sha256", entry["manifest_ref"]["sha256"], "--gpu", str(gpu)]
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    with log_path.open("xb") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=state["root"], env=environment, start_new_session=True)
    receipt = dict(schema="e23_member_move_resume_v1", status="LAUNCHED", shard=shard,
                   worker_id=entry["worker_id"], stage=entry["stage"], gpu=gpu, pid=process.pid,
                   argv=argv, manifest_sha256=entry["manifest_ref"]["sha256"],
                   dispatch_ref=reference(dispatch_path), scope_canonical_sha256=state["scope_sha"],
                   original_start_ref=reference(start_path), stop_request_ref=reference(request_path),
                   safe_checkpoint_ref=checkpoint_ref, task_count=len(keys), restored_successes=len(complete),
                   admission_snapshots=snapshots, log_path=str(log_path), at_unix=time.time(),
                   source_sha256=sha(Path(__file__)), manifest_and_journal_reused=True)
    freeze(resume_path, receipt)
    return dict(status="RESUMED_SAME_MANIFEST", shard=shard, gpu=gpu, pid=process.pid,
                restored_successes=len(complete), remaining=len(keys)-len(complete),
                resume_start_ref=reference(resume_path), manifest_changed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("stop", "resume"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(2, 3), required=True)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.action == "resume" and args.gpu is None:
        parser.error("--gpu is required for resume")
    result = (stop(args.root, args.dispatch, args.shard) if args.action == "stop"
              else resume(args.root, args.dispatch, args.shard, args.gpu))
    print(json.dumps(result))
