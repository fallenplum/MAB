"""Start one frozen disjoint member4 successor after predecessor and GPU checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_member_dispatch_v2 import PYTHON, reference, sha, validate_dispatch


def launch(root, dispatch_path, shard, gpu):
    state, dispatch, candidates = validate_dispatch(root, dispatch_path)
    selected = [item for item in candidates if item[0]["shard"] == shard]
    if len(selected) != 1:
        raise ValueError("An exact new member4 shard is required")
    entry, manifest, keys = selected[0]
    control = state["root"]/"control"
    control.mkdir(exist_ok=True)
    name = Path(entry["manifest_ref"]["path"]).stem
    receipt_path = control/f"{name}.start.json"
    log_path = control/f"{name}.log"
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError("This successor already has a launch or log artifact")
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
    worker = state["root"]/"code/supplement_worker.py"
    if sha(worker) != dispatch["worker_source_sha256"]:
        raise ValueError("The immutable worker wrapper changed before launch")
    argv = [PYTHON, "-B", "-u", str(worker), "--manifest", entry["manifest_ref"]["path"],
            "--manifest-sha256", entry["manifest_ref"]["sha256"], "--gpu", str(gpu)]
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    with log_path.open("xb") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=state["root"], env=environment, start_new_session=True)
    receipt = dict(status="LAUNCHED", worker_id=entry["worker_id"], stage="member_generation",
                   manifest_sha256=entry["manifest_ref"]["sha256"], argv=argv, pid=process.pid,
                   gpu=gpu, shared_gpu_authorized=True, admission_snapshots=snapshots,
                   at_unix=time.time(), log_path=str(log_path), dispatch_ref=reference(dispatch_path),
                   predecessor_checkpoint_refs=[item["stop_summary_ref"] for item in dispatch["retired"]],
                   scope_canonical_sha256=state["scope_sha"], task_count=len(keys))
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(status="LAUNCHED", worker_id=entry["worker_id"], shard=shard,
                gpu=gpu, pid=process.pid, tasks=len(keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(launch(args.root, args.dispatch, args.shard, args.gpu)))
