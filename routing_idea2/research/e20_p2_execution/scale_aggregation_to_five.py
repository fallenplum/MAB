"""Checkpoint the three owned P2 aggregate actors before a five-worker split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal

from launch_native_profile import gpu_state
from launch_registered_worker_v3 import matching_process_alive, start_receipts
from p2_manifest_registry import file_sha, iter_successes, load_registry, read_json


ROOT = Path("/data2/chendong/e20_p2_execution")
REGISTRY = ROOT / "control/manifest_registry_v3.jsonl"
RECORDS = ROOT / "control/aggregation_five_workers_20261117"
EXPECTED_PIDS = {2884571, 2885196, 2885924}
FOLLOWER_SHA = "094617587d90a6459eba2f46ffd2c55da77dc97d497742d83bfd86b250de2e98"


def write_once(name, value):
    RECORDS.mkdir(parents=True, exist_ok=True)
    with (RECORDS / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def request_stop():
    registry = load_registry(REGISTRY)
    if registry["software_test_only"]:
        raise PermissionError("A production registry is required")
    if file_sha(ROOT / "code/p2_aggregation_follower_v3.py") != FOLLOWER_SHA:
        raise RuntimeError("Frozen follower changed")
    selected = [row for row in start_receipts(ROOT)
                if row.get("profile") == "aggregation" and matching_process_alive(row)]
    if {row["pid"] for row in selected} != EXPECTED_PIDS or len(selected) != 3:
        raise RuntimeError("Live owned aggregate actors differ from the exact three-worker baseline")
    by_manifest = {row["manifest_ref"]["sha256"]: row for row in registry["entries"]}
    for start in selected:
        entry = by_manifest.get(start["manifest_sha256"])
        if (entry is None or entry["stage"] != "aggregation"
                or entry["worker_source_ref"]["sha256"] != FOLLOWER_SHA
                or start["scope_sha256"] != registry["scope_canonical_sha256"]):
            raise RuntimeError("Owned predecessor identity differs from its registered scope")
    snapshots = [gpu_state(index) for index in (1, 6)]
    if any(row["free_mib"] < 14336 for row in snapshots):
        raise RuntimeError("The explicitly requested GPU lacks measured loading and KV headroom")
    receipt = dict(status="GRACEFUL_STOP_REQUESTED", utc=datetime.now(timezone.utc).isoformat(),
                   source_sha256=file_sha(__file__), previous_starts=selected,
                   requested_additional_gpus=[1, 6], proposed_five_worker_gpus=[0, 1, 3, 1, 6],
                   admitted_extra_gpu_snapshots=snapshots,
                   registry_head=registry["last_event_sha256"],
                   scope_canonical_sha256=registry["scope_canonical_sha256"],
                   user_authorization="Add aggregate workers to GPU1 and GPU6 under the existing frozen P2 execution scope",
                   scientific_parameter_changes=0, foreign_process_signals=0)
    write_once("01_STOP_REQUEST.json", receipt)
    for start in selected:
        if not matching_process_alive(start):
            raise RuntimeError("A predecessor changed before its graceful stop signal")
        os.kill(start["pid"], signal.SIGTERM)
    print(json.dumps(dict(status=receipt["status"], pids=sorted(EXPECTED_PIDS),
                          proposed_five_worker_gpus=receipt["proposed_five_worker_gpus"])))


def verify_stopped():
    request = read_json(RECORDS / "01_STOP_REQUEST.json")
    alive = [row["pid"] for row in request["previous_starts"] if matching_process_alive(row)]
    if alive:
        print(json.dumps(dict(status="WAITING_FOR_CURRENT_CALLS_TO_COMMIT", pids=alive)))
        return
    registry = load_registry(REGISTRY)
    checks = []
    for start in request["previous_starts"]:
        summary = read_json(ROOT / "owned/_dispatcher" / start["manifest_sha256"] / "SUMMARY.json")
        if (summary.get("status"), summary.get("stopped"), summary.get("pid"), summary.get("scope_sha256")) != (
                "SAFELY_STOPPED", True, start["pid"], registry["scope_canonical_sha256"]):
            raise RuntimeError("A predecessor lacks its exact successful graceful checkpoint")
        checks.append(dict(pid=start["pid"], gpu=start["gpu"], completed=summary["completed"],
                           status=summary["status"], manifest_sha256=start["manifest_sha256"]))
    successes = list(iter_successes(registry, "aggregation"))
    unique = {key for _, _, key in successes}
    paid = [(key, row["physical_attempt_id"]) for _, row, key in successes if not row["from_cache"]]
    if len(paid) != len({key for key, _ in paid}):
        raise RuntimeError("A successful aggregate task has multiple paid journal attempts")
    receipt = dict(status="THREE_OWNED_ACTORS_SAFELY_STOPPED", utc=datetime.now(timezone.utc).isoformat(),
                   predecessors=checks, completed_unique_aggregations=len(unique),
                   duplicate_paid_success_keys=0, scope_canonical_sha256=registry["scope_canonical_sha256"],
                   source_registry_head=registry["last_event_sha256"],
                   new_workers_started=0, reference_values_decoded=0)
    write_once("02_STOPPED_CHECKPOINT.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("request-stop", "verify-stopped"), required=True)
    args = parser.parse_args()
    {"request-stop": request_stop, "verify-stopped": verify_stopped}[args.phase]()
