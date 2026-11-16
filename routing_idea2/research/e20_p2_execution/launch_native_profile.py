"""Launch one approved persistent P2 native profile after live GPU admission."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/data2/chendong/e20_p2_execution")
PYTHON = "/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_state(gpu):
    result = subprocess.run(["nvidia-smi", "--id="+str(gpu),
                             "--query-gpu=index,uuid,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    row = next(csv.reader(result.stdout.splitlines()))
    index, uuid, free, utilization = [value.strip() for value in row]
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                "--format=csv,noheader,nounits"], capture_output=True,text=True,check=True)
    active = [dict(pid=int(row[1]),used_mib=int(row[2])) for row in csv.reader(processes.stdout.splitlines()) if row[0].strip()==uuid]
    return dict(index=int(index),uuid=uuid,free_mib=int(free),utilization=int(utilization),compute_processes=active)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["member"+str(i) for i in range(6)]+["aggregation"],required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args()
    control=ROOT/"control"
    control.mkdir(exist_ok=True)
    owned_pids=set()
    for path in control.glob("*.start.json"):
        previous=json.loads(path.read_text())
        pid=previous["pid"]
        if Path("/proc") .joinpath(str(pid)).exists():
            owned_pids.add(pid)
            if previous["profile"]==args.profile:
                raise RuntimeError("Profile already running "+str(pid))
    state=gpu_state(args.gpu)
    if state["free_mib"] < 12288:
        raise RuntimeError("Insufficient live free VRAM for a new frozen8B profile")
    foreign=[row for row in state["compute_processes"] if row["pid"] not in owned_pids]
    if foreign and (state["utilization"]>20 or state["free_mib"]<14336):
        raise RuntimeError("Shared GPU lacks current utilization and memory headroom")
    if sum(row["pid"] in owned_pids for row in state["compute_processes"])>=4:
        raise RuntimeError("Four owned profile contexts already occupy this GPU")
    manifest=ROOT/"manifests"/(args.profile+"_v2.json")
    worker="p2_aggregation_follower.py" if args.profile=="aggregation" else "p2_member_worker.py"
    argv=[PYTHON,"-B",str(ROOT/"code"/worker),"--manifest",str(manifest),
          "--manifest-sha256",sha(manifest),"--gpu",str(args.gpu)]
    log=control/(args.profile+"-"+str(time.time_ns())+".log")
    environment=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false")
    with log.open("xb") as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,
                                 env=environment,start_new_session=True)
    receipt=dict(status="LAUNCHED",profile=args.profile,gpu=args.gpu,pid=process.pid,
                 manifest_sha256=sha(manifest),worker_source_sha256=sha(ROOT/"code"/worker),
                 admitted_gpu_state=state,argv=argv,log_path=str(log),at_unix=time.time(),
                 reference_access=False,owned_output_root=str(ROOT/"owned"))
    path=control/(args.profile+"-"+str(process.pid)+".start.json")
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    main()
