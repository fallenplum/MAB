"""Execute one immutable tail partition under its original E24 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


SCHEMA = "e24_parallel_tail_assignment_v1"
TASK_FIELDS = {"member_generation": "member_index", "aggregation": "combination_index"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bound(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"Frozen bytes differ for {path.name}")
    return json.loads(data)


def keys_from(rows):
    keys = set()
    for row in rows:
        if (not isinstance(row, list) or len(row) != 3 or not isinstance(row[0], str)
                or type(row[1]) is not int or type(row[2]) is not int):
            raise ValueError("Malformed tail task key")
        keys.add(tuple(row))
    if len(keys) != len(rows):
        raise ValueError("Duplicate tail task keys")
    return keys


def event_key(event, stage):
    task = event["task"]
    if task.get("stage") != stage:
        raise ValueError("Tail journal contains a task from another stage")
    row = [task["question_id"], task["generation_seed"], task[TASK_FIELDS[stage]]]
    return next(iter(keys_from([row])))


def scan_events(data, stage, allowed, assignments=None):
    """Validate a journal segment without treating any open call as a success."""
    pending, successes = set(), set()
    if data and not data.endswith(b"\n"):
        raise ValueError("Tail journal ends with a partial event")
    for line in data.splitlines():
        event = json.loads(line)
        key = event_key(event, stage)
        if key not in allowed:
            raise ValueError("Tail journal contains an out-of-partition task")
        if assignments is not None:
            writer = event.get("parallel_slot")
            if (type(writer) is not int or key not in assignments.get(writer, ())
                    or type(event.get("parallel_pid")) is not int or event["parallel_pid"] <= 0):
                raise ValueError("Tail journal row has an unbound writer")
        kind = event["event"]
        if kind == "TASK_ENTER" and key not in pending and key not in successes:
            pending.add(key)
        elif kind == "TASK_RESULT" and event.get("status") == "SUCCESS" and key in pending:
            pending.remove(key)
            successes.add(key)
        else:
            raise ValueError("Tail journal contains a failed, repeated or unmatched event")
    return successes, pending


def audit_journal(data, stage, parent, assignment, partitions, slot):
    checkpoint = assignment["journal_checkpoint"]
    size = checkpoint["bytes"]
    if type(size) is not int or not 0 <= size <= len(data):
        raise ValueError("Tail checkpoint length is invalid")
    prefix, suffix = data[:size], data[size:]
    if digest(prefix) != checkpoint["sha256"]:
        raise ValueError("Original journal prefix differs from the frozen checkpoint")
    successful, pending = scan_events(prefix, stage, parent)
    tail = set().union(*partitions.values())
    if pending or tail != parent - successful:
        raise ValueError("Tail partitions must exactly cover the clean checkpoint remainder")
    completed, inflight = scan_events(suffix, stage, tail, partitions)
    if inflight & partitions[slot]:
        raise ValueError("This tail slot still has an unresolved native task")
    return completed & partitions[slot]


def shared_journal(path):
    import fcntl

    with path.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        return stream.read()


def append_locked(path, event):
    import fcntl

    remaining = memoryview((json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    with path.open("ab", buffering=0) as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        while remaining:
            count = os.write(stream.fileno(), remaining)
            if count <= 0:
                raise OSError("Incomplete tail journal write")
            remaining = remaining[count:]
        os.fsync(stream.fileno())


def load_assignment(args):
    manifest_path = args.manifest.resolve(strict=True)
    manifest = read_bound(manifest_path, args.manifest_sha256)
    manifest_ref = {"path": str(manifest_path), "sha256": args.manifest_sha256}
    root = manifest_path.parent.parent
    assignment_path = args.assignment.resolve(strict=True)
    if not assignment_path.is_relative_to(root):
        raise ValueError("Tail assignment must be inside its owned E24 run")
    assignment = read_bound(assignment_path, args.assignment_sha256)
    plan = read_bound(root / "EXECUTION_PLAN.json", assignment["plan_sha256"])
    scope = assignment["scope_sha256"]
    if (assignment.get("schema") != SCHEMA or assignment.get("manifest_ref") != manifest_ref
            or type(assignment.get("retired_pid")) is not int or assignment["retired_pid"] <= 0
            or plan.get("cohort") != "e24_training_deployment_replication" or Path(plan["root"]).resolve() != root
            or plan["scope_canonical_sha256"] != scope or manifest_path.parent != root / "manifests"
            or manifest["scope_ref"] != plan["scope_ref"] or manifest["approval_ref"] != plan["approval_ref"]
            or manifest["software_test_only"] is not False
            or Path(manifest["owned_output_root"]).resolve() != root / "owned"):
        raise ValueError("Tail assignment does not identify the frozen E24 run")
    entries = [row for row in plan["manifests"] if row["manifest_ref"] == manifest_ref]
    if len(entries) != 1:
        raise ValueError("Tail parent manifest is not uniquely registered")
    entry = entries[0]
    if entry["stage"] != manifest["stage"] or entry["worker_id"] != manifest["worker_id"]:
        raise ValueError("Tail parent manifest identity differs from the plan")
    task_ref = manifest["task_keys_ref"]
    parent = keys_from(read_bound(Path(task_ref["path"]), task_ref["sha256"]))
    if len(parent) != entry["task_count"]:
        raise ValueError("Tail parent task count differs from the plan")
    partitions, devices, assigned = {}, set(), set()
    selected = None
    for child in assignment["children"]:
        slot, gpu, keys = child["slot"], child["gpu"], keys_from(child["keys"])
        if (type(slot) is not int or slot < 0 or slot in partitions or type(gpu) is not int or gpu < 0
                or gpu in devices or not keys or not keys <= parent or assigned & keys):
            raise ValueError("Tail partition or same-parent GPU allocation overlaps")
        partitions[slot] = keys
        devices.add(gpu)
        assigned.update(keys)
        if args.slot == slot and args.gpu == gpu:
            selected = keys
    if selected is None:
        raise ValueError("Tail child slot and GPU are not assigned")
    return manifest, assignment, parent, partitions, selected


def run_slot(args, manifest, assignment, parent, partitions, selected, stop):
    import fcntl
    from supplement_worker import reference

    stage = manifest["stage"]
    journal_root = Path(manifest["owned_output_root"]).resolve() / "_dispatcher" / args.manifest_sha256
    journal = journal_root / "EVENTS.jsonl"
    slot_root = journal_root / "parallel" / str(args.slot)
    slot_root.mkdir(parents=True, exist_ok=True)
    summary = slot_root / "SUMMARY.json"
    worker = reference.load_original_worker(stage, manifest["adapter_ref"])
    original_save = worker.save_json
    metadata = dict(assignment_sha256=args.assignment_sha256, parallel_slot=args.slot, parallel_gpu=args.gpu)

    def restore(path, event_stage, assigned):
        if Path(path) != journal or event_stage != stage or set(assigned) != selected:
            raise ValueError("Tail worker requested an unexpected restore")
        return audit_journal(shared_journal(path), stage, parent, assignment, partitions, args.slot)

    def append(path, event):
        if Path(path) != journal or event_key(event, stage) not in selected:
            raise ValueError("Tail worker tried to append outside its assignment")
        append_locked(path, dict(event, parallel_slot=args.slot, parallel_pid=os.getpid()))

    def save(path, value):
        path = Path(path)
        if path.parent == journal_root:
            if path.name not in {"SUMMARY.json", "TOKENIZER_ALIAS_RECEIPT.json"}:
                raise ValueError("Tail child cannot change shared parent metadata")
            value = dict(value, **metadata)
            path = slot_root / path.name
        original_save(path, value)

    with (slot_root / "SLOT.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if summary.exists():
            prior = reference.read(summary)
            if (prior.get("status") != "SAFELY_STOPPED"
                    or any(prior.get(key) != value for key, value in metadata.items())):
                raise RuntimeError("Tail slot cannot restart without a matching clean stop")
        reference.restored_successes = restore
        worker.append_event = append
        worker.save_json = save
        restored = 0
        try:
            restored = len(audit_journal(shared_journal(journal), stage, parent, assignment, partitions, args.slot))
            with worker.GPUOwnership(manifest["gpu_lock_root"], args.gpu, args.manifest_sha256):
                loaded = worker.load_worker(manifest)
                module, adapter, full_keys, profile, backend, external = loaded
                if set(full_keys) != parent or module.canonical_digest(adapter.scope) != assignment["scope_sha256"]:
                    raise ValueError("Native tail loader differs from the registered parent")
                original_save(slot_root / f"WRAPPER_RECEIPT.{os.getpid()}.json", dict(
                    schema="e24_parallel_tail_worker_receipt_v1", source_sha256=digest(Path(__file__).read_bytes()),
                    manifest_sha256=args.manifest_sha256, scope_sha256=assignment["scope_sha256"], pid=os.getpid(),
                    reference_source_sha256=digest(Path(reference.__file__).read_bytes()),
                    original_worker_sha256=reference.ORIGINAL_WORKERS[stage][1], adapter_ref=manifest["adapter_ref"],
                    native_backend_implementation_changed=False, scientific_generation_parameters_changed=False,
                    at_unix=time.time(), **metadata))
                subset = [key for key in full_keys if key in selected]
                result = reference.run_loaded(worker, (module, adapter, subset, profile, backend, external),
                                              manifest, args.manifest_sha256, stop=stop)
        except BaseException as error:
            previous = reference.read(summary) if summary.is_file() else {}
            if previous.get("pid") != os.getpid() or previous.get("status") != "STOP_REQUIRES_RECONCILIATION":
                original_save(summary, dict(previous, schema="e23_native_worker_summary_v1",
                    status="STOP_REQUIRES_RECONCILIATION", pid=os.getpid(), stage=stage, planned=len(selected),
                    completed=restored, scope_sha256=assignment["scope_sha256"], manifest_sha256=args.manifest_sha256,
                    worker_id=manifest["worker_id"], failures=[{"error_type": type(error).__name__}],
                    gold_reads=0, at_unix=time.time(), **metadata))
            raise
    return dict(result, **metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--slot", required=True, type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    stop = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))
    print(json.dumps(run_slot(args, *load_assignment(args), stop), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
