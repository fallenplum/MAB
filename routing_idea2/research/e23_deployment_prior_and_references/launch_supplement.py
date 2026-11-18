"""Launch one explicitly named frozen supplementary worker after GPU admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PYTHON = "/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def launch(root, name, gpu):
    plan = read(root/"EXECUTION_PLAN.json")
    rows = [row for row in plan["manifests"] if Path(row["manifest_ref"]["path"]).stem == name]
    if len(rows) != 1:
        raise ValueError("One exact prepared worker manifest is required")
    row = rows[0]
    path = Path(row["manifest_ref"]["path"])
    if sha(path) != row["manifest_ref"]["sha256"]:
        raise ValueError("Worker manifest bytes changed")
    worker = root/"code/supplement_worker.py"
    if sha(worker) != plan["source_sha256"]["supplement_worker.py"]:
        raise ValueError("Prepared original worker wrapper changed")
    snapshots = []
    for _ in range(3):
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
        state = next([int(item.strip()) for item in line.split(",")] for line in result.stdout.splitlines()
                     if int(line.split(",")[0]) == gpu)
        if state[1] < 14336:
            raise RuntimeError("The frozen NF4 profile lacks its required loading and KV headroom")
        snapshots.append(dict(gpu=state[0], free_mib=state[1], utilization_percent=state[2]))
        time.sleep(0.5)
    control = root/"control"
    control.mkdir(exist_ok=True)
    receipt = control/f"{name}.start.json"
    if receipt.exists():
        raise FileExistsError("This worker already has its original launch receipt")
    argv = [PYTHON, "-B", "-u", str(worker), "--manifest", str(path),
            "--manifest-sha256", row["manifest_ref"]["sha256"], "--gpu", str(gpu)]
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    log = control/f"{name}.log"
    with log.open("xb") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=root, env=environment, start_new_session=True)
    record = dict(status="LAUNCHED", worker_id=row["worker_id"], stage=row["stage"],
                  manifest_sha256=row["manifest_ref"]["sha256"], argv=argv, pid=process.pid,
                  gpu=gpu, shared_gpu_authorized=True, admission_snapshots=snapshots,
                  at_unix=time.time(), log_path=str(log))
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return dict(status="LAUNCHED", worker_id=row["worker_id"], gpu=gpu, pid=process.pid, tasks=row["task_count"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(launch(args.root, args.worker, args.gpu)))
