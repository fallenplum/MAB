"""Gracefully relocate the one frozen P2 aggregation process from GPU7 to4."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from launch_native_profile import gpu_state, sha


ROOT = Path("/data2/chendong/e20_p2_execution")
RECORDS = ROOT / "control/aggregation_gpu7_to_gpu4_20261117"
SCOPE = "b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9"
MANIFEST_SHA = "2b1f634e7c8775a0222bb65284182d7db8fcc372bdcdb8d2b22231e21c3dffc2"
FOLLOWER_SHA = "95f6eb7c1502c98ececfd8f13f69a05220b7f03fabfa93ac24b79f534b374c67"
JOURNAL_ROOT = ROOT / "owned/_dispatcher" / MANIFEST_SHA


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def checkpoint():
    rows = []
    for line in (JOURNAL_ROOT / "EVENTS.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "TASK_RESULT" and row.get("status") == "SUCCESS":
            rows.append(row)
    keys = [(row["task"]["question_id"], row["task"]["generation_seed"], row["task"]["combination_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Aggregation journal already contains duplicate successful semantic keys")
    return rows, keys


def latest_aggregation():
    rows = [read(path) for path in (ROOT / "control").glob("*.start.json")]
    return max((row for row in rows if row["profile"] == "aggregation"), key=lambda row: row["at_unix"])


def request_stop():
    previous = latest_aggregation()
    if previous["gpu"] != 7 or previous["manifest_sha256"] != MANIFEST_SHA or previous["worker_source_sha256"] != FOLLOWER_SHA:
        raise RuntimeError("Current aggregation process differs from the exact authorized migration")
    pid = previous["pid"]
    proc = Path("/proc") / str(pid)
    command = (proc / "cmdline").read_bytes().split(b"\0")
    expected = [value.encode() for value in previous["argv"]]
    if command[:-1] != expected:
        raise RuntimeError("PID command no longer matches the owned aggregation actor")
    if sha(ROOT / "code/p2_aggregation_follower.py") != FOLLOWER_SHA:
        raise RuntimeError("The frozen operational follower source changed")
    telemetry = gpu_state(4)
    if telemetry["free_mib"] < 14336:
        raise RuntimeError("GPU4 lacks the explicit NF4 loading and KV headroom")
    summary = read(JOURNAL_ROOT / "SUMMARY.json")
    receipt = dict(status="GRACEFUL_STOP_REQUESTED", utc=now(), previous_start=previous,
                   before_summary=summary, target_gpu_state=telemetry,
                   source_code_sha256=sha(__file__),
                   authorization="Root delegated the single aggregation7-to4 migration under the user's existing shared-resource execution approval",
                   shared_utilization_override_explicit=True,
                   frozen_manifest_sha256=MANIFEST_SHA, frozen_scope_canonical_sha256=SCOPE,
                   model_policy_or_task_key_changes=0, signals_to_other_processes=0)
    write_once(RECORDS / "01_STOP_REQUEST.json", receipt)
    os.kill(pid, signal.SIGTERM)
    print(json.dumps(dict(status=receipt["status"],pid=pid,target_gpu=4,utc=receipt["utc"])))


def resume():
    request = read(RECORDS / "01_STOP_REQUEST.json")
    old = request["previous_start"]
    if (Path("/proc") / str(old["pid"])).exists():
        raise RuntimeError("Original aggregation process has not exited")
    summary = read(JOURNAL_ROOT / "SUMMARY.json")
    if summary.get("status") != "SAFELY_STOPPED" or summary.get("stopped") is not True or summary.get("scope_sha256") != SCOPE:
        raise RuntimeError("An exact successful graceful-stop summary is required")
    manifest_path = ROOT / "manifests/aggregation_v2.json"
    manifest = read(manifest_path)
    if sha(manifest_path) != MANIFEST_SHA or sha(ROOT / "code/p2_aggregation_follower.py") != FOLLOWER_SHA:
        raise RuntimeError("Manifest or operational inference source changed during migration")
    if sha(manifest["scope_ref"]["path"]) != manifest["scope_ref"]["sha256"] or sha(manifest["approval_ref"]["path"]) != manifest["approval_ref"]["sha256"]:
        raise RuntimeError("Scope or approval file changed")
    rows, keys = checkpoint()
    if len(keys) != summary["completed"]:
        raise RuntimeError("Successful native journal does not recover the stopped count")
    snapshot = dict(status="STOPPED_CHECKPOINT_VERIFIED", utc=now(), completed=len(keys),
                    successful_semantic_keys=[list(key) for key in keys],
                    native_hashes=[row["native_record_sha256"] for row in rows],
                    actual_prompt_tokens=sum(row["actual_usage"]["prompt_tokens"] for row in rows),
                    actual_completion_tokens=sum(row["actual_usage"]["completion_tokens"] for row in rows),
                    stopped_summary=summary, source_scope_sha256=SCOPE,
                    duplicate_success_keys=0, pending_current_task=False)
    if keys:
        digest=hashlib.sha256(json.dumps(list(keys[0]),sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        sentinel=ROOT / "owned" / SCOPE / "aggregation" / digest / "LAST_TASK_RECEIPT.json"
        snapshot["old_success_sentinel"] = dict(path=str(sentinel),sha256=sha(sentinel))
    write_once(RECORDS / "02_STOPPED_CHECKPOINT_PRIVATE.json", snapshot)
    telemetry = gpu_state(4)
    if telemetry["free_mib"] < 14336:
        raise RuntimeError("GPU4 headroom changed after the graceful stop")
    argv = list(old["argv"])
    argv[argv.index("--gpu")+1] = "4"
    log = ROOT / "control" / ("aggregation-migrated-"+str(time.time_ns())+".log")
    environment=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false")
    with log.open("xb") as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=environment,start_new_session=True)
    started=dict(status="LAUNCHED", profile="aggregation",gpu=4,pid=process.pid,
                 manifest_sha256=MANIFEST_SHA,worker_source_sha256=FOLLOWER_SHA,
                 admitted_gpu_state=telemetry,argv=argv,log_path=str(log),at_unix=time.time(),
                 reference_access=False,owned_output_root=str(ROOT/"owned"),
                 migration_from_pid=old["pid"],migration_source_gpu=7,
                 shared_utilization_override_explicit=True,
                 migration_authorization=request["authorization"],previous_completed_calls=len(keys))
    write_once(ROOT / "control" / ("aggregation-"+str(process.pid)+".start.json"),started)
    write_once(RECORDS / "03_RESUMED_ON_GPU4.json",started)
    print(json.dumps(dict(status="RESUMED_ON_GPU4",pid=process.pid,previous_completed=len(keys),log_path=str(log),gpu4_free_mib=telemetry["free_mib"],gpu4_utilization=telemetry["utilization"])))


def verify():
    snapshot=read(RECORDS / "02_STOPPED_CHECKPOINT_PRIVATE.json")
    resumed=read(RECORDS / "03_RESUMED_ON_GPU4.json")
    rows,keys=checkpoint()
    previous=set(map(tuple,snapshot["successful_semantic_keys"]))
    new=[row for row,key in zip(rows,keys) if key not in previous]
    if not new:
        print(json.dumps(dict(status="AWAITING_FIRST_MIGRATED_SUCCESS",pid=resumed["pid"],alive=(Path('/proc')/str(resumed['pid'])).exists())))
        return
    if any(row["from_cache"] for row in new):
        raise RuntimeError("New journal rows unexpectedly replay a cached key")
    sentinel=snapshot.get("old_success_sentinel")
    if sentinel is not None and sha(sentinel["path"]) != sentinel["sha256"]:
        raise RuntimeError("An already successful native receipt changed")
    if any((Path('/proc')/str(pid)).exists() for pid in [read(RECORDS/'01_STOP_REQUEST.json')['previous_start']['pid']]):
        raise RuntimeError("Old and resumed aggregation processes overlap")
    result=dict(status="MIGRATION_VERIFIED_RUNNING",utc=now(),previous_completed=snapshot["completed"],
                total_completed=len(keys),new_unique_successes=len(new),first_new_actual_usage=new[0]["actual_usage"],
                new_actual_prompt_tokens=sum(row["actual_usage"]["prompt_tokens"] for row in new),
                new_actual_completion_tokens=sum(row["actual_usage"]["completion_tokens"] for row in new),
                repeated_success_semantic_keys=0,old_success_receipt_unchanged=True,
                source_gpu=7,target_gpu=4,new_pid=resumed["pid"],old_process_exited=True,
                scope_or_approval_or_model_or_policy_changes=0,reference_values_decoded=0)
    write_once(RECORDS/'04_MIGRATION_VERIFIED.json',result)
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("request-stop","resume","verify"),required=True)
    args=parser.parse_args()
    {"request-stop":request_stop,"resume":resume,"verify":verify}[args.phase]()
