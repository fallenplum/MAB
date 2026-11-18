"""Freeze disjoint member4 successor tasks after both original actors exit safely."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import execution_monitor as original_monitor


PYTHON = "/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python"
SCHEMA = "e23_member_dispatch_v2"
sha = original_monitor.sha
read = original_monitor.read
canonical = original_monitor.canonical
checked = original_monitor.checked


def reference(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def freeze(path, value):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("An existing immutable dispatch artifact differs")
    else:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return reference(path)


def load_parent(root):
    return original_monitor.context(Path(root))


def selected_original(state):
    selected = [row for row in state["manifests"]
                if row[0]["stage"] == "member_generation" and row[0]["profile"] == 4]
    if len(selected) != 2:
        raise ValueError("The original plan must contain exactly two member4 actors")
    return selected


def start_receipt(state, entry):
    stem = Path(entry["manifest_ref"]["path"]).stem
    path = state["root"] / "control" / f"{stem}.start.json"
    receipt = read(path)
    gpu, pid = receipt.get("gpu"), receipt.get("pid")
    argv = [PYTHON, "-B", "-u", str(state["root"]/"code/supplement_worker.py"),
            "--manifest", entry["manifest_ref"]["path"],
            "--manifest-sha256", entry["manifest_ref"]["sha256"], "--gpu", str(gpu)]
    if (type(gpu) is not int or type(pid) is not int or pid <= 0
            or receipt.get("worker_id") != entry["worker_id"]
            or receipt.get("stage") != entry["stage"]
            or receipt.get("manifest_sha256") != entry["manifest_ref"]["sha256"]
            or receipt.get("argv") != argv):
        raise ValueError("An original actor start receipt differs from its exact invocation")
    return path, receipt


def actor_state(receipt):
    return original_monitor.actor_state(receipt)


def ensure_exited(receipt, probe=actor_state):
    state = probe(receipt)
    if state.get("alive") is not False:
        raise RuntimeError("An original overlapping actor has not exited with proven identity")
    return state


def successful_keys(entry, keys, journal):
    rows, partial = original_monitor.events(journal)
    if partial:
        raise ValueError("A retired native journal has a partial line")
    completed = set()
    for event in rows:
        if event.get("event") == "STOP_REQUIRES_RECONCILIATION" or (
                event.get("event") == "TASK_RESULT" and event.get("status") != "SUCCESS"):
            raise RuntimeError("A retired actor has an unresolved failure event")
        if event.get("event") != "TASK_RESULT":
            continue
        key = original_monitor.task_key(event, entry["stage"])
        original_monitor.actual_usage(event.get("actual_usage"))
        if key not in keys or key in completed or not event.get("physical_attempt_id"):
            raise ValueError("A retired journal success is duplicated or outside its owned tasks")
        completed.add(key)
    return completed


def checkpoint(state, entry, manifest, keys, probe=actor_state, successor_keys=frozenset()):
    start_path, start = start_receipt(state, entry)
    exited = ensure_exited(start, probe)
    folder = state["owned"] / "_dispatcher" / entry["manifest_ref"]["sha256"]
    summary_path, journal = folder/"SUMMARY.json", folder/"EVENTS.jsonl"
    summary = read(summary_path)
    complete = successful_keys(entry, keys, journal)
    journal_rows, _ = original_monitor.events(journal)
    native_hashes = {
        original_monitor.task_key(event, entry["stage"]): event.get("native_record_sha256")
        for event in journal_rows if event.get("event") == "TASK_RESULT" and event.get("status") == "SUCCESS"
    }
    safe = ((summary.get("status") == "SAFELY_STOPPED" and summary.get("stopped") is True)
            or (summary.get("status") == "COMPLETE" and len(complete) == len(keys)))
    if (not safe or summary.get("schema") != "e23_native_worker_summary_v1"
            or summary.get("pid") != start["pid"]
            or summary.get("scope_sha256") != state["scope_sha"]
            or summary.get("manifest_sha256") != entry["manifest_ref"]["sha256"]
            or summary.get("stage") != "member_generation" or summary.get("profile") != 4
            or summary.get("planned") != len(keys) or summary.get("completed") != len(complete)
            or summary.get("failures") or summary.get("at_unix", 0) < start.get("at_unix", 0)):
        raise RuntimeError("A retired actor lacks its exact successful final checkpoint")
    for key in keys:
        owner = state["owned"] / state["scope_sha"] / "member_generation" / canonical(list(key))
        if not owner.exists():
            if key in complete:
                raise ValueError("A committed predecessor task has no native owner")
            continue
        fence_path = owner / "ATTEMPT_STATE.json"
        if not fence_path.is_file():
            raise RuntimeError("A predecessor task has an unknown owner without a fence")
        fence = read(fence_path)
        if (fence.get("scope_sha256") != state["scope_sha"]
                or fence.get("approval_sha256") != state["plan"]["approval_ref"]["sha256"]
                or fence.get("stage") != "member_generation"
                or tuple(fence.get("task_key", ())) != key):
            raise RuntimeError("A prior reserved, failed or unjournaled owner requires reconciliation")
        if key in complete:
            if (fence.get("status") != "SUCCESS_COMMITTED"
                    or fence.get("native_record_sha256") != native_hashes.get(key)
                    or not (owner/"LAST_TASK_RECEIPT.json").is_file()
                    or not (owner/"records.jsonl").is_file()):
                raise RuntimeError("A retired successful key lacks its matching committed native owner")
        elif key in successor_keys:
            if fence.get("status") not in {"RESERVED_BEFORE_BACKEND", "SUCCESS_COMMITTED"}:
                raise RuntimeError("An assigned successor owner has a failed or unknown execution fence")
        else:
            raise RuntimeError("An unassigned prior owner requires reconciliation before dispatch")
    item = dict(worker_id=entry["worker_id"], manifest_ref=entry["manifest_ref"],
                task_keys_ref=manifest["task_keys_ref"], start_ref=reference(start_path),
                stop_summary_ref=reference(summary_path), journal_ref=reference(journal),
                successful_keys_sha256=canonical([list(key) for key in sorted(complete)]),
                successful_count=len(complete), exited_identity=exited)
    return item, complete


def partition(required, completed, shards):
    required, completed = set(required), set(completed)
    if shards != 4 or not completed <= required:
        raise ValueError("This dispatch is the frozen four-way member4 continuation")
    pieces = [[] for _ in range(shards)]
    for key in sorted(required-completed):
        pieces[int(canonical([key[0], key[1]]), 16) % shards].append(key)
    union = set(completed)
    for piece in pieces:
        if union.intersection(piece):
            raise AssertionError("Successors overlap each other or completed tasks")
        union.update(piece)
    if union != required or any(not piece for piece in pieces):
        raise ValueError("Four nonempty disjoint remaining-task shards are required")
    return pieces


def global_snapshot(state):
    result = {stage: set() for stage in original_monitor.STAGES}
    for entry, manifest, keys in state["manifests"]:
        journal = state["owned"]/"_dispatcher"/entry["manifest_ref"]["sha256"]/"EVENTS.jsonl"
        rows, _ = original_monitor.events(journal)
        for event in rows:
            if event.get("event") == "TASK_RESULT" and event.get("status") == "SUCCESS":
                key = original_monitor.task_key(event, entry["stage"])
                if key not in keys:
                    raise ValueError("Global snapshot success escaped its original manifest")
                result[entry["stage"]].add(key)
    return {stage: [list(key) for key in sorted(keys)] for stage, keys in result.items()}


def prepare(root, output, shards=4):
    existing = Path(output)/"DISPATCH_V2.json"
    if existing.is_file():
        state, dispatch, new = validate_dispatch(root, existing)
        return dict(status="EXISTING_FOUR_WAY_DISPATCH_REUSED", dispatch_ref=reference(existing),
                    completed=dispatch["completed_at_preparation"], remaining=dispatch["remaining_member4_keys"],
                    shard_counts=[len(item[2]) for item in new], production_workers_started=0)
    state = load_parent(root)
    selected = selected_original(state)
    required = set().union(*(row[2] for row in selected))
    retired, completed = [], set()
    for entry, manifest, keys in selected:
        record, done = checkpoint(state, entry, manifest, keys)
        retired.append(record)
        completed.update(done)
    pieces = partition(required, completed, shards)
    output = Path(output).resolve()
    if output != state["root"].resolve() and not output.is_relative_to(state["root"].resolve()):
        raise ValueError("Dispatch output must stay inside the approved experiment root")
    parent_ref = reference(state["root"]/"EXECUTION_PLAN.json")
    plan_id = canonical(dict(parent=parent_ref["sha256"], scope=state["scope_sha"],
                             completed=canonical([list(key) for key in sorted(completed)]), shards=shards))[:20]
    completed_ref = freeze(output/"COMPLETED_MEMBER4_KEYS.json", [list(key) for key in sorted(completed)])
    snapshot_ref = freeze(output/"GLOBAL_SUCCESS_SNAPSHOT.json", global_snapshot(state))
    worker = state["root"]/"code/supplement_worker.py"
    if sha(worker) != state["plan"]["source_sha256"]["supplement_worker.py"]:
        raise ValueError("The prepared original worker wrapper changed")
    template = selected[0][1]
    relevant = lambda value: {key: item for key, item in value.items()
                              if key not in {"worker_id", "task_keys_ref", "gpu_lock_root"}}
    if relevant(template) != relevant(selected[1][1]):
        raise ValueError("The two predecessor scientific and approval profiles differ")
    new_manifests = []
    for shard, keys in enumerate(pieces):
        worker_id = f"e23-member4-v2-{plan_id}-shard{shard}"
        task_ref = freeze(output/f"member4_v2_shard{shard}_tasks.json", [list(key) for key in keys])
        manifest = dict(template, worker_id=worker_id, task_keys_ref=task_ref,
                        gpu_lock_root=str(state["root"]/"gpu-locks"/worker_id))
        manifest_ref = freeze(output/f"member4_v2_shard{shard}.json", manifest)
        new_manifests.append(dict(worker_id=worker_id, stage="member_generation", profile=4,
                                  shard=shard, task_count=len(keys), manifest_ref=manifest_ref))
    dispatch = dict(
        schema=SCHEMA, root=str(state["root"]), parent_plan_ref=parent_ref,
        scope_ref=state["plan"]["scope_ref"], scope_canonical_sha256=state["scope_sha"],
        approval_ref=state["plan"]["approval_ref"], stage="member_generation", member_index=4,
        retired=retired, completed_member4_keys_ref=completed_ref,
        completed_at_preparation=len(completed), required_member4_keys=len(required),
        remaining_member4_keys=len(required)-len(completed), new_manifests=new_manifests,
        global_success_snapshot_ref=snapshot_ref, shards=shards, plan_id=plan_id,
        worker_source_sha256=sha(worker), scientific_profile_or_scope_changes=0,
        checks=dict(disjoint_success_and_remaining=True, exact_original_member4_union=True,
                    original_actors_safely_exited=True, unknown_or_failed_native_owners=0),
        prepared_at_unix=time.time())
    path = output/"DISPATCH_V2.json"
    dispatch_ref = freeze(path, dispatch)
    return dict(status="FOUR_WAY_MEMBER4_DISPATCH_PREPARED", dispatch_ref=dispatch_ref,
                completed=len(completed), remaining=len(required)-len(completed),
                shard_counts=[len(piece) for piece in pieces], production_workers_started=0)


def validate_dispatch(root, dispatch_path):
    state = load_parent(root)
    dispatch = read(dispatch_path)
    if (dispatch.get("schema") != SCHEMA or Path(dispatch["root"]).resolve() != state["root"].resolve()
            or dispatch.get("scope_ref") != state["plan"]["scope_ref"]
            or dispatch.get("scope_canonical_sha256") != state["scope_sha"]
            or dispatch.get("approval_ref") != state["plan"]["approval_ref"]
            or dispatch.get("stage") != "member_generation" or dispatch.get("member_index") != 4
            or checked(dispatch["parent_plan_ref"]) != state["plan"]):
        raise ValueError("A dispatch changed the immutable parent scope, plan or approval")
    worker = state["root"]/"code/supplement_worker.py"
    if sha(worker) != dispatch["worker_source_sha256"] or sha(worker) != state["plan"]["source_sha256"]["supplement_worker.py"]:
        raise ValueError("The dispatch worker source differs from the original plan")
    selected = selected_original(state)
    required = set().union(*(row[2] for row in selected))
    completed = set(map(tuple, checked(dispatch["completed_member4_keys_ref"])))
    pieces = partition(required, completed, dispatch["shards"])
    if (dispatch["completed_at_preparation"] != len(completed)
            or dispatch["required_member4_keys"] != len(required)
            or dispatch["remaining_member4_keys"] != len(required)-len(completed)
            or len(dispatch["new_manifests"]) != 4):
        raise ValueError("The dispatch count identity differs")
    template = selected[0][1]
    snapshot = checked(dispatch["global_success_snapshot_ref"])
    if any(not set(map(tuple, snapshot[stage])) <= state["required"][stage] for stage in original_monitor.STAGES):
        raise ValueError("The global preparation snapshot escaped the original scope")
    new = []
    for index, entry in enumerate(dispatch["new_manifests"]):
        manifest = checked(entry["manifest_ref"])
        raw_keys = checked(manifest["task_keys_ref"])
        keys = set(map(tuple, raw_keys))
        expected = dict(template, worker_id=entry["worker_id"], task_keys_ref=manifest["task_keys_ref"],
                        gpu_lock_root=str(state["root"]/"gpu-locks"/entry["worker_id"]))
        if (manifest != expected or len(keys) != len(raw_keys) or keys != set(pieces[index])
                or entry["stage"] != "member_generation" or entry["profile"] != 4
                or entry["shard"] != index or entry["task_count"] != len(keys)):
            raise ValueError("A successor changed its exact frozen remaining-key subset")
        new.append((entry, manifest, keys))
    successor_keys = set().union(*(item[2] for item in new))
    actual_retired, journal_completed = [], set()
    for entry, manifest, keys in selected:
        record, done = checkpoint(state, entry, manifest, keys, successor_keys=successor_keys)
        actual_retired.append(record)
        journal_completed.update(done)
    if journal_completed != completed:
        raise ValueError("Retired member4 successes changed after dispatch preparation")
    for actual, expected in zip(actual_retired, dispatch["retired"]):
        for name in ("worker_id", "manifest_ref", "task_keys_ref", "start_ref", "stop_summary_ref",
                     "journal_ref", "successful_keys_sha256", "successful_count"):
            if actual[name] != expected[name]:
                raise ValueError("A stopped predecessor changed after dispatch preparation")
    return state, dispatch, new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, choices=(4,), default=4)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output, args.shards)))
