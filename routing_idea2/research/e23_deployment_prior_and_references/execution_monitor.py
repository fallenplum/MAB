"""Monitor and close only the approved E23 supplementary native outputs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


STAGES = ("member_generation", "aggregation")
COUNTS = {"member_generation": 1770, "aggregation": 4263}
FILES = {"member_generation": "MEMBER_NATIVE_RECEIPTS.jsonl",
         "aggregation": "AGGREGATION_NATIVE_RECEIPTS.jsonl"}


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked(reference):
    if set(reference) != {"path", "sha256"}:
        raise ValueError("An exact file source binding is required")
    path = Path(reference["path"]).resolve(strict=True)
    if sha(path) != reference["sha256"]:
        raise ValueError("A frozen supplementary source changed")
    return read(path)


def task_key(event, stage):
    task = event["task"]
    field = "member_index" if stage == "member_generation" else "combination_index"
    if (task.get("stage") != stage or not isinstance(task.get("question_id"), str)
            or type(task.get("generation_seed")) is not int or type(task.get(field)) is not int):
        raise ValueError("A native journal task identity differs")
    return task["question_id"], task["generation_seed"], task[field]


def actual_usage(row):
    if not isinstance(row, dict) or any(type(row.get(key)) is not int or row[key] < 0
                                        for key in ("prompt_tokens", "completion_tokens")):
        raise ValueError("A successful native attempt requires exact actual Token counts")
    return {key: row[key] for key in ("prompt_tokens", "completion_tokens")}


def events(path):
    if not path.is_file():
        return [], False
    result, partial = [], False
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                partial = True
                break
            if line.strip():
                result.append(json.loads(line))
    return result, partial


def start_receipts(root):
    result = []
    for path in sorted((root / "control").glob("*.start.json")):
        result.append(read(path))
    for path in (root / "START_RECEIPTS.json", root / "control/START_RECEIPTS.json"):
        if path.is_file():
            packet = read(path)
            result.extend(packet if isinstance(packet, list) else packet.get("receipts", []))
    return result


def actor_state(receipt):
    pid = receipt.get("pid")
    report = dict(pid=pid, gpu=receipt.get("gpu"), alive=False)
    if type(pid) is not int or pid <= 0:
        return dict(report, state="INVALID_PID")
    path = Path("/proc") / str(pid)
    try:
        status = (path / "status").read_text()
        if "\nState:\tZ" in "\n"+status:
            return dict(report, state="EXITED_ZOMBIE")
        if isinstance(receipt.get("argv"), list):
            actual = (path / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            if actual != [item.encode() for item in receipt["argv"]]:
                return dict(report, state="PID_REUSED_OR_COMMAND_DIFFERS")
        return dict(report, alive=True, state="OWNED_PROCESS_MATCH")
    except FileNotFoundError:
        return dict(report, state="EXITED")
    except (OSError, PermissionError):
        return dict(report, alive=None, state="PROCESS_PROBE_UNAVAILABLE")


def context(root):
    plan_path = root / "EXECUTION_PLAN.json"
    plan = read(plan_path)
    if (plan.get("schema") != "e23_supplement_execution_plan_v1"
            or plan.get("cohort") != "e23_reference_supplement"
            or Path(plan["root"]).resolve() != root.resolve()):
        raise ValueError("An exact supplementary execution plan is required")
    scope, approval = checked(plan["scope_ref"]), checked(plan["approval_ref"])
    scope_sha = canonical(scope)
    if (scope_sha != plan["scope_canonical_sha256"]
            or scope.get("schema") != "e23_supplement_scope_v1"
            or scope.get("cohort") != plan["cohort"]
            or scope.get("transport_split") != "online_stream"
            or approval.get("schema") != "e23_supplement_approval_v1"
            or approval.get("status") != "APPROVED"
            or approval.get("software_test_only") is not False
            or approval.get("cohort") != plan["cohort"]
            or approval.get("scope_sha256_by_role", {}).get("execution") != scope_sha
            or approval.get("bindings") != scope["bindings"]):
        raise ValueError("Supplementary scope or approval identity differs")
    owned = Path(approval["approved_output_root"])
    if owned.resolve() != (root / "owned").resolve():
        raise ValueError("Supplementary output owner escaped the approved root")
    required = {
        "member_generation": set(map(tuple, scope["member_execution_keys"])),
        "aggregation": set(map(tuple, scope["aggregate_execution_keys"])),
    }
    if ({stage: len(keys) for stage, keys in required.items()} != COUNTS
            or plan["required_new_member_calls"] != COUNTS["member_generation"]
            or plan["required_new_aggregate_calls"] != COUNTS["aggregation"]
            or scope["execution_limits"] != COUNTS or approval["execution_limits"] != COUNTS):
        raise ValueError("Supplementary execution counts differ from the approved union")
    planned = {stage: set() for stage in STAGES}
    manifests = []
    for entry in plan["manifests"]:
        manifest = checked(entry["manifest_ref"])
        stage = entry["stage"]
        keys = set(map(tuple, checked(manifest["task_keys_ref"])))
        if (stage not in STAGES or manifest["stage"] != stage
                or manifest["worker_id"] != entry["worker_id"]
                or manifest["scope_ref"] != plan["scope_ref"]
                or manifest["approval_ref"] != plan["approval_ref"]
                or Path(manifest["owned_output_root"]).resolve() != owned.resolve()
                or len(keys) != entry["task_count"] or not keys <= required[stage]
                or keys & planned[stage]):
            raise ValueError("Supplementary worker manifests overlap or differ from the scope")
        planned[stage].update(keys)
        manifests.append((entry, manifest, keys))
    if planned != required:
        raise ValueError("Supplementary worker manifests do not cover exactly the approved calls")
    return dict(root=root, plan=plan, scope=scope, approval=approval,
                scope_sha=scope_sha, owned=owned, required=required,
                manifests=manifests, plan_sha256=sha(plan_path),
                question_metadata={row["question_id"]: row for row in scope["question_keys"]})


def monitor(state):
    root, owned, now = state["root"], state["owned"], time.time()
    starts = start_receipts(root)
    successes = {stage: {} for stage in STAGES}
    attempts, totals = {}, Counter()
    workers, errors, partials, duplicate_rows, duplicate_paid = [], [], 0, 0, 0
    for entry, manifest, keys in state["manifests"]:
        stage = entry["stage"]
        manifest_sha = entry["manifest_ref"]["sha256"]
        folder = owned / "_dispatcher" / manifest_sha
        journal, partial = events(folder / "EVENTS.jsonl")
        partials += int(partial)
        local = set()
        latest = None
        recent = Counter()
        for event in journal:
            if event.get("event") == "STOP_REQUIRES_RECONCILIATION" or (
                    event.get("event") == "TASK_RESULT" and event.get("status") != "SUCCESS"):
                errors.append(dict(worker_id=entry["worker_id"], error_type=event.get("error_type", "NATIVE_FAILURE"),
                                   at_unix=event.get("at_unix")))
                continue
            if event.get("event") != "TASK_RESULT":
                continue
            key = task_key(event, stage)
            usage = actual_usage(event.get("actual_usage"))
            physical = event.get("physical_attempt_id")
            if key not in keys or not isinstance(physical, str) or not physical:
                raise ValueError("A successful supplementary journal key or attempt is invalid")
            signature = (key, event.get("native_record_sha256"), usage["prompt_tokens"], usage["completion_tokens"])
            identity = (stage, physical)
            if identity in attempts and attempts[identity] != signature:
                raise ValueError("An attempt has inconsistent native cost or task evidence")
            existing = successes[stage].get(key)
            if existing is not None:
                duplicate_rows += 1
                if existing["physical_attempt_id"] != physical:
                    duplicate_paid += 1
            if identity not in attempts:
                attempts[identity] = signature
                totals.update(usage)
            successes[stage][key] = event
            local.add(key)
            timestamp = event.get("at_unix")
            if isinstance(timestamp, (int, float)):
                latest = max(latest or timestamp, timestamp)
                for window in (15, 60):
                    if now-timestamp <= 60*window:
                        recent[str(window)] += 1
        summary = read(folder / "SUMMARY.json") if (folder / "SUMMARY.json").is_file() else {}
        matching = [item for item in starts if item.get("manifest_sha256") == manifest_sha]
        workers.append(dict(
            worker_id=entry["worker_id"], stage=stage, profile=entry["profile"],
            completed=len(local), planned=len(keys), remaining=len(keys)-len(local),
            last_success_unix=latest, recent_successes=dict(recent),
            summary_status=summary.get("status"), backend_loads=summary.get("backend_loads"),
            ready=summary.get("ready"), waiting_members=summary.get("waiting_members"),
            actors=[actor_state(item) for item in matching]))
    fences, unresolved, extra = {}, Counter(), 0
    for stage, keys in state["required"].items():
        expected = {canonical(list(key)): key for key in keys}
        folder = owned / state["scope_sha"] / stage
        actual = list(folder.iterdir()) if folder.is_dir() else []
        for path in actual:
            if not path.is_dir() or path.is_symlink() or path.name not in expected:
                extra += 1
                continue
            fence_path = path / "ATTEMPT_STATE.json"
            if not fence_path.is_file():
                unresolved["OWNER_WITHOUT_FENCE"] += 1
                continue
            fence = read(fence_path)
            key = expected[path.name]
            if (fence.get("scope_sha256") != state["scope_sha"] or fence.get("stage") != stage
                    or tuple(fence.get("task_key", ())) != key):
                raise ValueError("A supplementary native owner fence has another task identity")
            fences[(stage, key)] = fence
            if fence.get("status") != "SUCCESS_COMMITTED":
                unresolved[str(fence.get("status"))] += 1
    all_complete = all(set(successes[stage]) == state["required"][stage] for stage in STAGES)
    ready_export = (all_complete and len(fences) == sum(COUNTS.values())
                    and not any((len(errors), partials, duplicate_paid, sum(unresolved.values()), extra)))
    report = dict(
        schema="e23_native_execution_status_v1", status="COMPLETE_READY_FOR_EXPORT" if ready_export else "IN_PROGRESS_OR_RECONCILIATION_REQUIRED",
        at_unix=now, utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        scope_canonical_sha256=state["scope_sha"],
        stages={stage: dict(completed=len(successes[stage]), required=COUNTS[stage],
                            remaining=COUNTS[stage]-len(successes[stage])) for stage in STAGES},
        actual_success_native_tokens=dict(totals, total_tokens=sum(totals.values())),
        unique_success_native_attempts=len(attempts), workers=workers,
        failure_events=errors, partial_journals=partials,
        repeated_success_rows=duplicate_rows, additional_paid_success_attempts=duplicate_paid,
        unresolved_owner_states=dict(unresolved), extra_owner_entries=extra,
        all_required_successful=all_complete, ready_for_final_export=ready_export,
        reference_values_read=0, new_llm_calls=0)
    return report, successes, fences


def verified_owner(state, stage, key, event, fence):
    folder = state["owned"] / state["scope_sha"] / stage / canonical(list(key))
    receipt = read(folder / "LAST_TASK_RECEIPT.json")
    raw = (folder / "records.jsonl").read_bytes()
    native_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    native = receipt.get("native_record", {})
    native_sha = canonical({name: value for name, value in native.items() if name != "record_sha256"})
    task = receipt.get("task_key", {})
    field = "member_index" if stage == "member_generation" else "combination_index"
    metadata = state["question_metadata"][key[0]]
    usage = actual_usage(native.get("result") or native.get("actual_usage"))
    if (receipt.get("schema") != "p2_execution_task_receipt_v1"
            or receipt.get("cohort") != state["plan"]["cohort"]
            or receipt.get("scope_sha256") != state["scope_sha"]
            or receipt.get("approval_sha256") != state["plan"]["approval_ref"]["sha256"]
            or receipt.get("bindings") != state["scope"]["bindings"]
            or receipt.get("status") != "SUCCESS" or receipt.get("stage") != stage
            or receipt.get("software_test_only") is not False
            or receipt.get("transport_split") != "online_stream"
            or receipt.get("aggregator_fingerprint") != state["scope"]["operator_fingerprint"]
            or (task.get("question_id"), task.get("generation_seed"), task.get(field)) != key
            or task.get("canonical_id") != metadata["canonical_question_sha256"]
            or native_rows != [native] or native.get("record_sha256") != native_sha
            or native.get("status") != "success" or native.get("cost_status") != "recorded_actual_usage"
            or native.get("split") != "online_stream"
            or receipt.get("native_record_sha256") != native_sha
            or fence.get("native_record_sha256") != native_sha
            or event.get("native_record_sha256") != native_sha
            or event.get("physical_attempt_id") != receipt.get("physical_attempt_id")
            or native.get("attempt_id") != receipt.get("physical_attempt_id")
            or event.get("actual_usage") != usage):
        raise ValueError("Supplementary native receipt, record or fence identity differs")
    return receipt, usage


def export_final(state, report, successes, fences):
    if not report["ready_for_final_export"]:
        raise PermissionError("All exact successful outputs and settled fences are required before export")
    exports = state["root"] / "exports"
    exports.mkdir(exist_ok=True)
    freeze_path = exports / "OUTPUTS_FROZEN.json"
    if freeze_path.is_file():
        frozen = read(freeze_path)
        if (frozen.get("scope_canonical_sha256") != state["scope_sha"]
                or frozen.get("approval_sha256") != state["plan"]["approval_ref"]["sha256"]):
            raise ValueError("Existing supplementary export freeze differs")
        for entry in frozen["records"].values():
            if sha(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("Closed supplementary export bytes changed")
        return dict(status="EXISTING_SUPPLEMENT_OUTPUT_FREEZE_REUSED", freeze_path=str(freeze_path),
                    freeze_sha256=sha(freeze_path), records=frozen["records"])
    if any((exports / name).exists() for name in FILES.values()):
        raise FileExistsError("Partial final export exists without a freeze; retain it for reconciliation")
    records, totals = {}, Counter()
    for stage in STAGES:
        destination = exports / FILES[stage]
        temporary = destination.with_suffix(".jsonl.part")
        with temporary.open("xb") as stream:
            for key in sorted(state["required"][stage]):
                receipt, usage = verified_owner(state, stage, key, successes[stage][key], fences[(stage, key)])
                stream.write((json.dumps(receipt, ensure_ascii=False, allow_nan=False)+"\n").encode())
                totals.update(usage)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(destination)
        records[stage] = dict(path=str(destination), sha256=sha(destination),
                              bytes=destination.stat().st_size, records=COUNTS[stage])
    actual_tokens = dict(totals, total_tokens=sum(totals.values()))
    if actual_tokens != report["actual_success_native_tokens"]:
        raise ValueError("Exported original native usage differs from the complete journals")
    freeze = dict(
        schema="e23_supplement_outputs_frozen_v1", status="ALL_REQUIRED_SUPPLEMENT_OUTPUTS_FROZEN",
        cohort=state["plan"]["cohort"], scope_sha256=state["plan"]["scope_ref"]["sha256"],
        scope_canonical_sha256=state["scope_sha"], approval_sha256=state["plan"]["approval_ref"]["sha256"],
        scope_bindings=state["scope"]["bindings"], execution_plan_sha256=state["plan_sha256"],
        completed_member_calls=COUNTS["member_generation"], completed_aggregate_calls=COUNTS["aggregation"],
        records=records, actual_tokens=actual_tokens, unknown_or_failed_attempts=0, new_gold_values_read=0,
        frozen_utc=datetime.now(timezone.utc).isoformat(), exporter_sha256=sha(__file__))
    with freeze_path.open("x", encoding="utf-8") as stream:
        json.dump(freeze, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(status=freeze["status"], freeze_path=str(freeze_path),
                freeze_sha256=sha(freeze_path), records=records, actual_tokens=actual_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--export-final", action="store_true")
    args = parser.parse_args()
    state = context(args.root)
    report, successes, fences = monitor(state)
    if args.export_final:
        report["final_export"] = export_final(state, report, successes, fences)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
