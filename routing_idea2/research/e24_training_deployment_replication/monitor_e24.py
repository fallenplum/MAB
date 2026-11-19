"""Read E24 native journals and export complete verified receipts in bounded gzip shards."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

CODE = Path(__file__).resolve().parent
if not (CODE / "execution_monitor.py").is_file():
    sys.path.insert(0, str(CODE.parent / "e23_deployment_prior_and_references"))

from execution_monitor import (STAGES, actor_state, actual_usage, canonical, checked, read,
                               sha, start_receipts, task_key, verified_owner)


COHORT = "e24_training_deployment_replication"
COUNTS = {"member_generation": 20661, "aggregation": 53661}
TOKEN_CAP = {"member_generation": 6144, "aggregation": 8192}
PLAN_SCHEMA = "e24_independent_execution_plan_v1"
SCOPE_SCHEMA = "e24_independent_execution_scope_v1"
APPROVAL_SCHEMA = "e24_independent_execution_approval_v1"
FREEZE_SCHEMA = "e24_native_outputs_frozen_v1"
SHARD_SIZE = 2048
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def context(root, counts=COUNTS, question_count=1196):
    root = root.resolve(strict=True)
    plan_path = root / "EXECUTION_PLAN.json"
    plan = read(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("cohort") != COHORT or Path(plan["root"]).resolve() != root:
        raise ValueError("An exact E24 execution plan is required")
    scope, approval = checked(plan["scope_ref"]), checked(plan["approval_ref"])
    scope_sha = canonical(scope)
    owned = Path(approval["approved_output_root"]).resolve()
    if (scope_sha != plan["scope_canonical_sha256"] or scope.get("schema") != SCOPE_SCHEMA
            or scope.get("cohort") != COHORT or scope.get("transport_split") != "online_stream"
            or approval.get("schema") != APPROVAL_SCHEMA or approval.get("status") != "APPROVED"
            or approval.get("cohort") != COHORT or approval.get("software_test_only") is not False
            or set(approval["operations"]) != set(STAGES) or approval.get("sealed200_answer_access") is not False
            or approval["scope_sha256_by_role"]["execution"] != scope_sha
            or approval["bindings"] != scope["bindings"] or owned != (root / "owned").resolve()):
        raise ValueError("E24 scope, approval or output owner binding differs")
    key_rows = {"member_generation": scope["member_execution_keys"], "aggregation": scope["aggregate_execution_keys"]}
    required = {stage: set(map(tuple, rows)) for stage, rows in key_rows.items()}
    if (any(len(required[stage]) != len(key_rows[stage]) or len(required[stage]) != counts[stage] for stage in STAGES)
            or plan["required_new_member_calls"] != counts["member_generation"]
            or plan["required_new_aggregate_calls"] != counts["aggregation"]
            or scope["execution_limits"] != counts or approval["execution_limits"] != counts):
        raise ValueError("E24 execution limits differ from the exact frozen calls")
    questions = {row["question_id"]: row for row in scope["question_keys"]}
    if len(questions) != question_count or len(scope["question_keys"]) != question_count or plan["input_questions"] != question_count:
        raise ValueError("E24 question registry differs from its declared size")
    for stage, limit in (("member_generation", 6), ("aggregation", 20)):
        if any(len(key) != 3 or key[0] not in questions or type(key[1]) is not int or key[1] not in {42, 137, 2024}
               or type(key[2]) is not int or not 0 <= key[2] < limit for key in required[stage]):
            raise ValueError("An E24 call key has invalid question, seed or action identity")

    planned = {stage: set() for stage in STAGES}
    worker_ids, manifest_hashes, manifests = set(), set(), []
    for entry in plan["manifests"]:
        manifest = checked(entry["manifest_ref"])
        stage, worker_id, manifest_hash = entry["stage"], entry["worker_id"], entry["manifest_ref"]["sha256"]
        rows = checked(manifest["task_keys_ref"])
        keys = set(map(tuple, rows))
        if (stage not in STAGES or stage != manifest["stage"] or worker_id != manifest["worker_id"]
                or worker_id in worker_ids or manifest_hash in manifest_hashes
                or manifest["scope_ref"] != plan["scope_ref"] or manifest["approval_ref"] != plan["approval_ref"]
                or Path(manifest["owned_output_root"]).resolve() != owned or manifest.get("software_test_only") is not False
                or len(keys) != len(rows) or len(keys) != entry["task_count"] or not keys <= required[stage]
                or keys & planned[stage]):
            raise ValueError("E24 worker manifests overlap or differ from the approved plan")
        if stage == "member_generation" and any(key[2] != entry["profile"] for key in keys):
            raise ValueError("E24 member manifest mixes member profiles")
        planned[stage].update(keys)
        worker_ids.add(worker_id)
        manifest_hashes.add(manifest_hash)
        manifests.append((entry, keys))
    if planned != required:
        raise ValueError("E24 manifests must cover every approved call exactly once")
    return dict(root=root, plan=plan, scope=scope, scope_sha=scope_sha, owned=owned, required=required,
                question_metadata=questions, manifests=manifests, plan_sha256=sha(plan_path), counts=counts)


def token_report(counter):
    return dict(prompt_tokens=counter["prompt_tokens"], completion_tokens=counter["completion_tokens"],
                total_tokens=counter["prompt_tokens"] + counter["completion_tokens"])


def journal_progress(state):
    started, now = time.monotonic(), time.time()
    starts = defaultdict(list)
    for row in start_receipts(state["root"]):
        starts[row.get("manifest_sha256")].append(row)
    dispatcher = state["owned"] / "_dispatcher"
    registered = {entry["manifest_ref"]["sha256"] for entry, _ in state["manifests"]}
    unknown_dispatchers = sum(path.name not in registered for path in dispatcher.iterdir() if path.is_dir()) if dispatcher.is_dir() else 0
    successes = {stage: {} for stage in STAGES}
    stage_tokens = {stage: Counter() for stage in STAGES}
    attempts, workers, failures = {}, [], []
    duplicates, cache_rows, partials, token_cap_violations, unresolved_enters = 0, 0, 0, 0, 0
    unknown_cost_failures, stopped_worker_failures = 0, 0
    failed_usage = Counter()
    for entry, keys in state["manifests"]:
        stage, manifest_hash = entry["stage"], entry["manifest_ref"]["sha256"]
        folder = dispatcher / manifest_hash
        journal = folder / "EVENTS.jsonl"
        local, pending, repeated_enters = set(), set(), 0
        recent, latest = Counter(), None
        if journal.is_file():
            with journal.open("rb") as stream:
                for raw_line in stream:
                    if not raw_line.endswith(b"\n"):
                        partials += 1
                        break
                    if not raw_line.strip():
                        continue
                    event = json.loads(raw_line)
                    kind = event.get("event")
                    if kind not in {"TASK_ENTER", "TASK_RESULT", "STOP_REQUIRES_RECONCILIATION"}:
                        raise ValueError("Unexpected event in an E24 registered journal")
                    key = task_key(event, stage)
                    if key not in keys:
                        raise ValueError("E24 journal work exceeds its allocated manifest")
                    if kind == "TASK_ENTER":
                        repeated_enters += int(key in pending)
                        pending.add(key)
                        continue
                    pending.discard(key)
                    if kind == "STOP_REQUIRES_RECONCILIATION" or event.get("status") != "SUCCESS":
                        failures.append(dict(worker_id=entry["worker_id"], stage=stage,
                                             error_type=event.get("error_type", "NATIVE_FAILURE"),
                                             at_unix=event.get("at_unix")))
                        if event.get("actual_usage") is None:
                            unknown_cost_failures += 1
                        else:
                            failed_usage.update(actual_usage(event["actual_usage"]))
                        continue
                    usage, physical = actual_usage(event.get("actual_usage")), event.get("physical_attempt_id")
                    native_hash = event.get("native_record_sha256")
                    if (not isinstance(physical, str) or not physical or not isinstance(native_hash, str)
                            or not SHA256.fullmatch(native_hash) or type(event.get("from_cache")) is not bool):
                        raise ValueError("E24 successful journals must name a physical attempt and native hash")
                    signature = (stage, key, native_hash, usage["prompt_tokens"], usage["completion_tokens"])
                    previous_attempt = attempts.get(physical)
                    previous_key = successes[stage].get(key)
                    if previous_attempt is not None and previous_attempt != signature:
                        raise ValueError("One physical attempt has conflicting native identity or Token evidence")
                    if previous_key is not None and previous_key["physical_attempt_id"] != physical:
                        raise ValueError("E24 semantic call has multiple successful physical attempts")
                    if previous_key is None:
                        successes[stage][key] = event
                        stage_tokens[stage].update(usage)
                        attempts[physical] = signature
                        token_cap_violations += int(sum(usage.values()) > TOKEN_CAP[stage])
                    else:
                        duplicates += 1
                    cache_rows += int(event["from_cache"])
                    local.add(key)
                    timestamp = event.get("at_unix")
                    if type(timestamp) in (int, float):
                        latest = max(latest or timestamp, timestamp)
                        for window in (5, 15, 60):
                            if now-timestamp <= 60*window and previous_key is None:
                                recent[str(window)] += 1
        summary_path = folder / "SUMMARY.json"
        summary = read(summary_path) if summary_path.is_file() else {}
        if summary and (summary.get("scope_sha256") != state["scope_sha"]
                        or summary.get("manifest_sha256") != manifest_hash or summary.get("worker_id") != entry["worker_id"]):
            raise ValueError("An E24 worker summary belongs to another run")
        stopped_worker_failures += int(summary.get("status") == "STOP_REQUIRES_RECONCILIATION")
        actors = [actor_state(row) for row in starts.get(manifest_hash, [])]
        unresolved_enters += len(pending)
        workers.append(dict(worker_id=entry["worker_id"], stage=stage, profile=entry.get("profile"),
                            planned=len(keys), completed=len(local), remaining=len(keys)-len(local),
                            summary_status=summary.get("status"), backend_loads=summary.get("backend_loads"),
                            last_process_wall_seconds=summary.get("wall_seconds"), last_process_cpu_seconds=summary.get("cpu_seconds"),
                            ready=summary.get("ready"), waiting_members=summary.get("waiting_members"), actors=actors,
                            open_enters=len(pending), repeated_enters=repeated_enters, last_success_unix=latest,
                            recent_successes=dict(recent)))
    counts = state["counts"]
    all_complete = all(set(successes[stage]) == state["required"][stage] for stage in STAGES)
    ready_to_verify = all_complete and not any((unresolved_enters, partials, failures, unknown_dispatchers,
                                                token_cap_violations, stopped_worker_failures, set(starts)-registered))
    total = sum(stage_tokens.values(), Counter())
    report = dict(schema="e24_native_execution_status_v1", status="JOURNALS_COMPLETE_FOR_OWNER_CHECK"
                  if ready_to_verify else "EXECUTION_IN_PROGRESS_OR_RECONCILIATION_REQUIRED",
                  utc=datetime.fromtimestamp(now, timezone.utc).isoformat(), at_unix=now,
                  cohort=COHORT, scope_canonical_sha256=state["scope_sha"], execution_plan_sha256=state["plan_sha256"],
                  stages={stage: dict(completed=len(successes[stage]), required=counts[stage],
                                      remaining=counts[stage]-len(successes[stage])) for stage in STAGES},
                  actual_success_native_tokens=token_report(total), stage_tokens={stage: token_report(stage_tokens[stage]) for stage in STAGES},
                  unique_success_native_attempts=len(attempts), repeated_success_rows=duplicates,
                  cached_success_journal_rows=cache_rows, failure_events=failures,
                  failed_event_tokens_with_recorded_usage=token_report(failed_usage),
                  failure_events_without_known_cost=unknown_cost_failures, worker_summaries_requiring_reconciliation=stopped_worker_failures,
                  partial_journals=partials,
                  token_cap_violations=token_cap_violations, unresolved_enter_events=unresolved_enters,
                  unknown_dispatchers=unknown_dispatchers, unregistered_start_manifest_count=len(set(starts)-registered),
                  all_required_successful=all_complete, ready_for_owner_check=ready_to_verify,
                  owner_state_checked=False, ready_for_final_export=False, workers=workers,
                  journal_scan_seconds=time.monotonic()-started, reference_values_read=0, new_llm_calls=0)
    return report, successes


def owner_fences(state, report):
    started = time.monotonic()
    fences, unresolved = {}, Counter()
    scope_root = state["owned"] / state["scope_sha"]
    extra = sum(path.name not in {"_dispatcher", state["scope_sha"]} for path in state["owned"].iterdir()) if state["owned"].is_dir() else 0
    if scope_root.is_dir():
        extra += sum(path.name not in STAGES for path in scope_root.iterdir())
    settled = 0
    for stage, keys in state["required"].items():
        expected = {canonical(list(key)): key for key in keys}
        folder = scope_root / stage
        if not folder.is_dir():
            continue
        with os.scandir(folder) as entries:
            for path in entries:
                if not path.is_dir(follow_symlinks=False) or path.name not in expected:
                    extra += 1
                    continue
                fence_path = Path(path.path) / "ATTEMPT_STATE.json"
                if not fence_path.is_file():
                    unresolved["OWNER_WITHOUT_FENCE"] += 1
                    continue
                try:
                    fence = read(fence_path)
                except json.JSONDecodeError:
                    unresolved["FENCE_NOT_READABLE_AS_JSON"] += 1
                    continue
                key = expected[path.name]
                if (fence.get("scope_sha256") != state["scope_sha"] or fence.get("stage") != stage
                        or tuple(fence.get("task_key", ())) != key or fence.get("software_test_only") is not False
                        or fence.get("approval_sha256") != state["plan"]["approval_ref"]["sha256"]):
                    raise ValueError("An E24 native fence differs from the exact approved key or mode")
                if fence.get("status") != "SUCCESS_COMMITTED":
                    unresolved[str(fence.get("status"))] += 1
                else:
                    settled += 1
                fences[(stage, key)] = fence
    expected_count = sum(state["counts"].values())
    ready = report["ready_for_owner_check"] and len(fences) == expected_count and not unresolved and not extra
    report.update(owner_state_checked=True, settled_owner_entries=settled,
                  unresolved_owner_states={key: value for key, value in unresolved.items() if value},
                  extra_owner_entries=extra, ready_for_final_export=ready, owner_scan_seconds=time.monotonic()-started)
    if ready:
        report["status"] = "COMPLETE_READY_FOR_EXPORT"
    return fences


def safe_export_file(folder, entry):
    leaf = entry["file"]
    if not isinstance(leaf, str) or Path(leaf).name != leaf or not leaf.endswith(".jsonl.gz"):
        raise ValueError("Frozen export entries must be adjacent gzip basenames")
    path = folder / leaf
    if path.is_symlink() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
        raise ValueError("Frozen E24 native export bytes changed")


def gzip_shard(path, state, stage, keys, successes, fences):
    tokens = Counter()
    stream = tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".", suffix=".part", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=3, mtime=0) as zipped:
                for key in keys:
                    fence = fences[(stage, key)]
                    receipt, usage = verified_owner(state, stage, key, successes[key], fence)
                    if (Path(receipt["approved_output_root"]).resolve() != state["owned"]
                            or receipt.get("native_job_id") != fence.get("native_job_id") or receipt.get("aggregation_seed") != 42):
                        raise ValueError("E24 native receipt owner or job identity differs")
                    zipped.write((json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                    tokens.update(usage)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size, records=len(keys)), tokens


def export_final(state, report, successes, fences):
    if not report["ready_for_final_export"]:
        raise PermissionError("Every approved E24 result, associated cost and native fence must close before export")
    exports = state["root"] / "exports"
    exports.mkdir(exist_ok=True)
    freeze_path = exports / "OUTPUTS_FROZEN.json"
    identity = dict(schema=FREEZE_SCHEMA, status="ALL_REQUIRED_OUTPUTS_FROZEN", cohort=COHORT,
                    scope_sha256=state["plan"]["scope_ref"]["sha256"], scope_canonical_sha256=state["scope_sha"],
                    approval_sha256=state["plan"]["approval_ref"]["sha256"], scope_bindings=state["scope"]["bindings"],
                    execution_plan_sha256=state["plan_sha256"], completed_member_calls=state["counts"]["member_generation"],
                    completed_aggregate_calls=state["counts"]["aggregation"],
                    actual_tokens=report["actual_success_native_tokens"], stage_tokens=report["stage_tokens"],
                    unknown_or_failed_attempts=0, new_gold_values_read=0)
    if freeze_path.is_file():
        frozen = read(freeze_path)
        if any(frozen.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing E24 output freeze identifies a different cohort, scope or paid cost")
        for stage, entries in frozen["records"].items():
            if sum(entry["records"] for entry in entries) != state["counts"][stage]:
                raise ValueError("Frozen E24 output shard count differs")
            for entry in entries:
                safe_export_file(exports, entry)
        if set(frozen["records"]) != set(STAGES):
            raise ValueError("Frozen E24 output stages differ")
        return dict(status="EXISTING_E24_OUTPUT_FREEZE_REUSED", freeze_path=str(freeze_path), freeze_sha256=sha(freeze_path),
                    records=frozen["records"], actual_tokens=frozen["actual_tokens"])

    records = {stage: [] for stage in STAGES}
    for stage in STAGES:
        keys = sorted(state["required"][stage])
        totals = Counter()
        for start in range(0, len(keys), SHARD_SIZE):
            basename = f"{stage.upper()}_{start//SHARD_SIZE:03d}.jsonl.gz"
            entry, usage = gzip_shard(exports / basename, state, stage, keys[start:start+SHARD_SIZE], successes[stage], fences)
            records[stage].append(entry)
            totals.update(usage)
        if token_report(totals) != report["stage_tokens"][stage]:
            raise ValueError("Verified E24 native costs differ from the corresponding journal totals")
    frozen = dict(identity, records=records, frozen_utc=datetime.now(timezone.utc).isoformat(), exporter_sha256=sha(Path(__file__)))
    with tempfile.NamedTemporaryFile(dir=exports, prefix="OUTPUTS_FROZEN.", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, freeze_path)
    return dict(status=frozen["status"], freeze_path=str(freeze_path), freeze_sha256=sha(freeze_path),
                records=records, actual_tokens=frozen["actual_tokens"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-owners", action="store_true")
    parser.add_argument("--export-final", action="store_true")
    args = parser.parse_args()
    state = context(args.root)
    report, successes = journal_progress(state)
    if args.verify_owners or args.export_final:
        fences = owner_fences(state, report)
    if args.export_final:
        report["final_export"] = export_final(state, report, successes, fences)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
