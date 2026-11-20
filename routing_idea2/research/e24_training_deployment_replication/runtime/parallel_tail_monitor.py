"""Read E24 shared journals and qualify parallel tail slots before native export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    import monitor_e24 as original
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import monitor_e24 as original


WINDOWS = (5, 15, 60)
EXIT_STATES = {"EXITED", "EXITED_ZOMBIE", "PID_REUSED_OR_COMMAND_DIFFERS"}


def counts_and_rates(recent):
    counts = {str(window): recent.get(str(window), 0) for window in WINDOWS}
    return counts, {str(window): round(counts[str(window)] / window, 4) for window in WINDOWS}


def checkpoint_successes(entry, keys, folder, assignment):
    checkpoint = assignment["journal_checkpoint"]
    length = checkpoint["bytes"]
    if type(length) is not int or length < 0:
        raise ValueError("Invalid parallel-tail checkpoint length")
    with (folder / "EVENTS.jsonl").open("rb") as source:
        prefix = source.read(length)
    if (len(prefix) != length or (prefix and not prefix.endswith(b"\n"))
            or hashlib.sha256(prefix).hexdigest() != checkpoint.get("sha256")):
        raise ValueError("The retired journal checkpoint changed")
    completed, pending = set(), set()
    for line in prefix.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = original.task_key(row, entry["stage"])
        if key not in keys or row.get("parallel_slot") is not None:
            raise ValueError("The original checkpoint contains a foreign or child task")
        if row["event"] == "TASK_ENTER" and key not in pending and key not in completed:
            pending.add(key)
        elif row["event"] == "TASK_RESULT" and row.get("status") == "SUCCESS" and key in pending:
            pending.remove(key)
            completed.add(key)
        else:
            raise ValueError("The retired journal checkpoint has unresolved or repeated work")
    if pending:
        raise ValueError("The retired journal checkpoint ends with open work")
    return completed


def inspect_assignment(state, path, entry, keys, receipts):
    assignment = original.read(path)
    reference = entry["manifest_ref"]
    manifest_hash = reference["sha256"]
    source = state["owned"] / "_dispatcher" / manifest_hash
    if (path.is_symlink() or assignment.get("schema") != "e24_parallel_tail_assignment_v1"
            or assignment.get("plan_sha256") != state["plan_sha256"]
            or assignment.get("scope_sha256") != state["scope_sha"]
            or assignment.get("manifest_ref") != reference or entry["stage"] != "aggregation"):
        raise ValueError("Parallel-tail assignment is outside the original aggregation plan")
    completed = checkpoint_successes(entry, keys, source, assignment)
    summary = original.read(source / "SUMMARY.json")
    retired_pid = assignment.get("retired_pid")
    retired = [row for row in receipts if row.get("pid") == retired_pid and row.get("parallel_slot") is None]
    if (len(retired) != 1 or original.actor_state(retired[0])["state"] not in EXIT_STATES
            or summary.get("status") != "SAFELY_STOPPED" or summary.get("stopped") is not True
            or summary.get("pid") != retired_pid or summary.get("failures")
            or summary.get("manifest_sha256") != manifest_hash or summary.get("worker_id") != entry["worker_id"]
            or summary.get("scope_sha256") != state["scope_sha"] or summary.get("completed") != len(completed)):
        raise ValueError("The original aggregation actor lacks a safe exit checkpoint")

    children, partition, gpus = {}, set(completed), set()
    for child in assignment["children"]:
        slot, gpu = child["slot"], child["gpu"]
        assigned = set(map(tuple, child["keys"]))
        if (type(slot) is not int or slot < 0 or slot in children
                or type(gpu) is not int or gpu < 0 or gpu in gpus
                or not assigned or len(assigned) != len(child["keys"])
                or not assigned <= keys or assigned & partition):
            raise ValueError("Parallel-tail slot assignments overlap or leave the original scope")
        children[slot] = (gpu, assigned)
        partition.update(assigned)
        gpus.add(gpu)
    if len(children) < 2 or partition != keys:
        raise ValueError("The checkpoint and child slots must exactly cover the original manifest")
    return source, children, {"path": str(path), "sha256": original.sha(path)}, retired_pid


def inspect_children(state, entry, keys, path, receipts, successes, now):
    source, children, assignment_ref, retired_pid = inspect_assignment(state, path, entry, keys, receipts)
    starts = {}
    for receipt in receipts:
        slot = receipt.get("parallel_slot")
        if slot is None:
            continue
        if (type(slot) is not int or slot not in children or slot in starts
                or receipt.get("assignment_ref") != assignment_ref
                or receipt.get("worker_id") != entry["worker_id"] or receipt.get("gpu") != children[slot][0]
                or type(receipt.get("pid")) is not int or receipt["pid"] <= 0
                or receipt["pid"] == retired_pid or not receipt.get("argv")):
            raise ValueError("A child start receipt differs from its assigned aggregation slot")
        starts[slot] = receipt
    if len({receipt["pid"] for receipt in starts.values()}) != len(starts):
        raise ValueError("Parallel-tail slots have duplicate physical process identities")

    rows = []
    for slot, (gpu, assigned) in sorted(children.items()):
        start = starts.get(slot)
        actor = original.actor_state(start) if start else None
        completed, recent, last = 0, {str(window): 0 for window in WINDOWS}, None
        for key in assigned:
            event = successes.get(key)
            if event is None:
                continue
            if (event.get("parallel_slot") != slot
                    or start is not None and event.get("parallel_pid") != start["pid"]):
                raise ValueError("A shared journal success differs from its assigned child slot")
            completed += 1
            stamp = event.get("at_unix")
            if type(stamp) in (int, float):
                last = max(last or stamp, stamp)
                for window in WINDOWS:
                    recent[str(window)] += int(now - stamp <= 60 * window)
        summary_path = source / "parallel" / str(slot) / "SUMMARY.json"
        summary = original.read(summary_path) if summary_path.is_file() else {}
        if summary and (summary.get("scope_sha256") != state["scope_sha"]
                        or summary.get("manifest_sha256") != entry["manifest_ref"]["sha256"]
                        or summary.get("worker_id") != entry["worker_id"]
                        or summary.get("parallel_slot") != slot or summary.get("parallel_gpu") != gpu
                        or summary.get("assignment_sha256") != assignment_ref["sha256"]
                        or summary.get("planned") != len(assigned)
                        or start is not None and summary.get("pid") != start["pid"]):
            raise ValueError("A child summary differs from its frozen slot or startup identity")
        counts, rates = counts_and_rates(recent)
        terminal_actor = actor is not None and actor["state"] in EXIT_STATES
        status = summary.get("status")
        failed = (status == "STOP_REQUIRES_RECONCILIATION" or bool(summary.get("failures"))
                  or terminal_actor and status not in {"SAFELY_STOPPED", "COMPLETE"})
        complete = (not failed and len(assigned) == completed == summary.get("completed")
                    and status == "COMPLETE" and terminal_actor)
        rows.append(dict(slot=slot, gpu=gpu, planned=len(assigned), completed=completed,
                         remaining=len(assigned)-completed, actor=actor, summary_status=summary.get("status"),
                         ready=summary.get("ready"), waiting_members=summary.get("waiting_members"),
                         failed=failed, complete_and_exited=complete, last_success_unix=last,
                         recent_successes=counts, successes_per_minute=rates))
    return rows


def monitor(root, *, verify_owners=False, export_final=False):
    state = original.context(root)
    report, successes = original.journal_progress(state)
    control = state["root"] / "control"
    assignments = {path.name.removesuffix(".parallel_tail.assignment.json"): path
                   for path in control.glob("*.parallel_tail.assignment.json")}
    entries = {Path(entry["manifest_ref"]["path"]).stem: (entry, keys) for entry, keys in state["manifests"]}
    if assignments.keys() - entries.keys():
        raise ValueError("An unregistered aggregation tail assignment is present")
    names = {entry["worker_id"]: name for name, (entry, _) in entries.items()}
    receipts = {}
    for receipt in original.start_receipts(state["root"]):
        receipts.setdefault(receipt.get("manifest_sha256"), []).append(receipt)
    children_complete, failure_count = True, 0
    for row in report["workers"]:
        name = names[row["worker_id"]]
        entry, keys = entries[name]
        starts = receipts.get(entry["manifest_ref"]["sha256"], [])
        counts, rates = counts_and_rates(row["recent_successes"])
        row.update(name=name, recent_successes=counts, successes_per_minute=rates)
        if name not in assignments:
            if any(receipt.get("parallel_slot") is not None for receipt in starts):
                raise ValueError("A child start receipt has no matching tail assignment")
            continue
        children = inspect_children(state, entry, keys, assignments[name], starts,
                                    successes[entry["stage"]], report["at_unix"])
        row["parallel_children"] = children
        row["parallel_ready"] = sum(child["ready"] or 0 for child in children)
        row["parallel_waiting_members_sum"] = sum(child["waiting_members"] or 0 for child in children)
        children_complete = children_complete and all(child["complete_and_exited"] for child in children)
        failure_count += sum(child["failed"] for child in children)
    report.update(parallel_assignments=len(assignments), all_parallel_children_complete_and_exited=children_complete,
                  parallel_worker_failures=failure_count)
    if not children_complete:
        report.update(ready_for_owner_check=False, ready_for_final_export=False,
                      status="WAITING_FOR_PARALLEL_CHILD_CLOSURE")
    if verify_owners or export_final:
        fences = original.owner_fences(state, report)
    if export_final:
        if not children_complete:
            raise PermissionError("Every registered parallel child must close and exit before E24 native export")
        report["final_export"] = original.export_final(state, report, successes, fences)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-owners", action="store_true")
    parser.add_argument("--export-final", action="store_true")
    args = parser.parse_args()
    result = monitor(args.root, verify_owners=args.verify_owners, export_final=args.export_final)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
