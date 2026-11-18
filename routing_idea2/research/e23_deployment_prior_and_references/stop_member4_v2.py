"""Request graceful exit only for the two exact original member4 invocations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from prepare_member_dispatch_v2 import (
    actor_state, checkpoint, freeze, load_parent, read, reference,
    selected_original, start_receipt,
)


def stop(root, record_dir=None):
    state = load_parent(root)
    selected = selected_original(state)
    folder = Path(record_dir) if record_dir is not None else state["root"]/"control/member4_stop_v2"
    if not folder.resolve().is_relative_to(state["root"].resolve()):
        raise ValueError("The stop record directory escaped the experiment root")
    entries = []
    for entry, manifest, keys in selected:
        start_path, receipt = start_receipt(state, entry)
        status = actor_state(receipt)
        if status.get("alive") is None:
            raise RuntimeError("Original actor ownership cannot be checked")
        if status.get("alive") is not True:
            checkpoint(state, entry, manifest, keys)
        entries.append((entry, receipt, reference(start_path), status))
    request_path = folder/"STOP_REQUEST.json"
    identities = [dict(worker_id=entry["worker_id"], manifest_ref=entry["manifest_ref"], start_ref=start_ref)
                  for entry, receipt, start_ref, status in entries]
    if request_path.exists():
        prior = read(request_path)
        if (prior.get("parent_plan_ref") != reference(state["root"]/"EXECUTION_PLAN.json")
                or prior.get("scope_canonical_sha256") != state["scope_sha"]
                or prior.get("selected") != identities):
            raise ValueError("An existing graceful stop request binds other actors")
    else:
        freeze(request_path, dict(schema="e23_member4_graceful_stop_v2",
               parent_plan_ref=reference(state["root"]/"EXECUTION_PLAN.json"),
               scope_canonical_sha256=state["scope_sha"], selected=identities,
               signals="SIGTERM only after exact PID and argv match",
               requested_at_unix=time.time(), foreign_processes_signaled=0))
    signaled, exited = [], []
    for entry, receipt, start_ref, initial in entries:
        current = actor_state(receipt)
        if current.get("alive") is True:
            os.kill(receipt["pid"], signal.SIGTERM)
            signaled.append(receipt["pid"])
        elif current.get("alive") is False:
            exited.append(receipt["pid"])
        else:
            raise RuntimeError("Actor ownership changed before SIGTERM")
    return dict(status="GRACEFUL_STOP_REQUESTED" if signaled else "ORIGINAL_MEMBER4_ACTORS_EXITED",
                signaled_pids=signaled, already_exited_pids=exited,
                stop_request_ref=reference(request_path), scope_or_approval_changes=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(stop(args.root, args.record_dir)))
