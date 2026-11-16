"""Gracefully checkpoint the original P2 aggregate actor before disjoint dispatch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal

from launch_native_profile import gpu_state
from launch_registered_worker_v3 import matching_process_alive
from p2_manifest_registry import file_sha, iter_complete_events, read_json


ROOT = Path("/data2/chendong/e20_p2_execution")
RECEIPTS = ROOT / "control/aggregation_three_shards_20261117"
MANIFEST_SHA = "2b1f634e7c8775a0222bb65284182d7db8fcc372bdcdb8d2b22231e21c3dffc2"
SOURCE_SHA = "95f6eb7c1502c98ececfd8f13f69a05220b7f03fabfa93ac24b79f534b374c67"
SCOPE_SHA = "b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9"
EXPECTED_PID = 2637354


def write_once(name, value):
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    with (RECEIPTS / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stop_request():
    starts = [read_json(path) for path in (ROOT / "control").glob("*.start.json")]
    previous = max((row for row in starts if row.get("profile") == "aggregation"),
                   key=lambda row: row["at_unix"])
    expected = (EXPECTED_PID, 4, MANIFEST_SHA, SOURCE_SHA)
    observed = tuple(previous[name] for name in ("pid", "gpu", "manifest_sha256", "worker_source_sha256"))
    if observed != expected or not matching_process_alive(previous):
        raise RuntimeError("The exact owned predecessor no longer matches this one-time operation")
    if file_sha(ROOT / "code/p2_aggregation_follower.py") != SOURCE_SHA:
        raise RuntimeError("The original follower source changed")
    snapshots = [gpu_state(index) for index in (0, 1, 3)]
    if any(row["free_mib"] < 14336 for row in snapshots):
        raise RuntimeError("A target GPU lacks measured NF4 loading and KV headroom")
    receipt = dict(status="GRACEFUL_STOP_REQUESTED", utc=datetime.now(timezone.utc).isoformat(),
                   previous_start=previous, target_gpu_snapshots=snapshots,
                   source_sha256=file_sha(__file__), existing_shared_resource_authorization=True,
                   task_scope_model_approval_or_generation_changes=0, foreign_process_signals=0)
    write_once("01_STOP_REQUEST.json", receipt)
    os.kill(previous["pid"], signal.SIGTERM)
    print(json.dumps(dict(status=receipt["status"], pid=previous["pid"], target_gpus=[0, 1, 3])))


def verify_stop():
    request = read_json(RECEIPTS / "01_STOP_REQUEST.json")
    previous = request["previous_start"]
    if matching_process_alive(previous):
        print(json.dumps(dict(status="WAITING_FOR_CURRENT_NATIVE_CALL_TO_COMMIT", pid=previous["pid"])))
        return
    journal_root = ROOT / "owned/_dispatcher" / MANIFEST_SHA
    summary = read_json(journal_root / "SUMMARY.json")
    if (summary.get("status"), summary.get("stopped"), summary.get("pid"), summary.get("scope_sha256")) != (
            "SAFELY_STOPPED", True, EXPECTED_PID, SCOPE_SHA):
        raise RuntimeError("The predecessor lacks its exact successful graceful checkpoint")
    successes = [row for row in iter_complete_events(journal_root / "EVENTS.jsonl")
                 if row.get("event") == "TASK_RESULT" and row.get("status") == "SUCCESS"]
    keys = [(row["task"]["question_id"], row["task"]["generation_seed"],
             row["task"]["combination_index"]) for row in successes]
    if len(keys) != len(set(keys)) or len(keys) != summary["completed"]:
        raise RuntimeError("Completed journal keys fail uniqueness or checkpoint recovery")
    result = dict(status="OLD_FULL_ACTOR_SAFELY_STOPPED", utc=datetime.now(timezone.utc).isoformat(),
                  completed_unique_aggregations=len(keys), checkpoint=summary,
                  old_actor_exited=True, repeated_success_keys=0, scope_sha256=SCOPE_SHA,
                  successor_processes_started=0, reference_values_decoded=0)
    write_once("02_STOPPED_CHECKPOINT.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "checkpoint"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("request-stop", "verify-stop"), required=True)
    args = parser.parse_args()
    {"request-stop": stop_request, "verify-stop": verify_stop}[args.phase]()
