"""Read registered P2 journals with semantic progress and physical-cost deduplication."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import time

from p2_manifest_registry import load_registry


def complete_events(path):
    """Yield only newline-committed JSON events from a concurrently appended journal."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                break
            if raw.strip():
                yield json.loads(raw)


def semantic_key(event):
    task = event["task"]
    stage = task["stage"]
    if stage not in {"member_generation", "aggregation"}:
        raise ValueError("Unknown native task stage")
    index_name = "member_index" if stage == "member_generation" else "combination_index"
    key = (task["question_id"], task["generation_seed"], task[index_name])
    if (not isinstance(key[0], str) or type(key[1]) is not int
            or type(key[2]) is not int):
        raise ValueError("Unexpected native semantic task identity")
    return stage, key


def actual_usage(event):
    usage = event.get("actual_usage")
    if not isinstance(usage, dict) or any(type(usage.get(name)) is not int or usage[name] < 0
            for name in ("prompt_tokens", "completion_tokens")):
        raise ValueError("A paid native success requires actual integer Token counts")
    return {name: usage[name] for name in ("prompt_tokens", "completion_tokens")}


def unique_entries(registry):
    """Preserve append order while refusing inconsistent manifest/journal aliases."""
    seen_manifests, seen_journals = {}, {}
    for entry in registry["entries"]:
        manifest_sha = entry["manifest_ref"]["sha256"]
        journal = str(Path(entry["journal_path"]).resolve())
        previous = seen_manifests.get(manifest_sha)
        if previous is not None:
            if previous != journal:
                raise ValueError("One registered manifest has inconsistent journal paths")
            continue
        if journal in seen_journals and seen_journals[journal] != manifest_sha:
            raise ValueError("One native journal has inconsistent manifest identities")
        seen_manifests[manifest_sha] = journal
        seen_journals[journal] = manifest_sha
        yield entry


def _actor_state(receipt, probe):
    result = {name: receipt.get(name) for name in
              ("pid", "gpu", "at_unix", "manifest_sha256", "log_path")}
    if not probe:
        return dict(result, alive=None, identity_status="NOT_PROBED")
    pid = receipt.get("pid")
    if type(pid) is not int or pid <= 0:
        return dict(result, alive=False, identity_status="INVALID_PID")
    proc = Path("/proc") / str(pid)
    try:
        state = next(line for line in (proc / "status").read_text().splitlines()
                     if line.startswith("State:"))
        result["process_state"] = state
        if "\tZ" in state:
            return dict(result, alive=False, identity_status="EXITED_ZOMBIE")
        expected = receipt.get("argv")
        if isinstance(expected, list):
            actual = (proc / "cmdline").read_bytes().split(b"\0")
            if actual[-1:] == [b""]:
                actual = actual[:-1]
            if actual != [value.encode() for value in expected]:
                return dict(result, alive=False, identity_status="PID_REUSED_OR_COMMAND_DIFFERS")
        result["process_io"] = {name: int(value) for name, value in
                                (line.split(": ", 1) for line in (proc / "io").read_text().splitlines())}
        return dict(result, alive=True, identity_status="OWNED_PROCESS_MATCH")
    except (FileNotFoundError, ProcessLookupError):
        return dict(result, alive=False, identity_status="EXITED")
    except (PermissionError, OSError) as error:
        return dict(result, alive=None, identity_status=type(error).__name__)


def collect_status(registry_path, root=None, *, probe_processes=True, probe_gpu=True,
                   software_test_only=False):
    registry = load_registry(Path(registry_path))
    if bool(registry.get("software_test_only")) != software_test_only:
        raise PermissionError("Registry and monitor execution modes differ")
    root = Path(root) if root is not None else Path(registry["common"]["owned_output_root"]).parent
    starts = []
    for path in sorted((root / "control").glob("*.start.json")):
        starts.append(json.loads(path.read_text(encoding="utf-8")))
    service_keys = defaultdict(set)
    charged_attempts = {}
    paid_by_key = defaultdict(set)
    stage_tokens = defaultdict(Counter)
    stage_cached = defaultdict(set)
    stage_events = defaultdict(Counter)
    worker_rows = []
    global_duplicate_rows = 0
    for entry in unique_entries(registry):
        manifest_sha = entry["manifest_ref"]["sha256"]
        allowed = set(map(tuple, entry["task_keys"]))
        local_keys, local_paid = set(), set()
        counts, charged_here = Counter(), Counter()
        last_success = None
        for event in complete_events(entry["journal_path"]):
            counts[event["event"]] += 1
            if event.get("event") != "TASK_RESULT" or event.get("status") != "SUCCESS":
                continue
            stage, key = semantic_key(event)
            if stage != entry["stage"] or key not in allowed:
                raise ValueError("Registered journal success escaped its manifest task subset")
            cached = event.get("from_cache")
            if type(cached) is not bool:
                raise ValueError("Native success must explicitly distinguish cache use")
            stage_events[stage]["success_rows"] += 1
            if key in service_keys[stage]:
                global_duplicate_rows += 1
            service_keys[stage].add(key)
            local_keys.add(key)
            last_success = event.get("at_unix")
            if cached:
                counts["cached_success_rows"] += 1
                stage_events[stage]["cached_success_rows"] += 1
                stage_cached[stage].add(key)
                continue
            attempt = event.get("physical_attempt_id")
            if not isinstance(attempt, str) or not attempt:
                raise ValueError("A paid native success requires a physical attempt identity")
            usage = actual_usage(event)
            identity = (stage, attempt)
            signature = (key, usage["prompt_tokens"], usage["completion_tokens"])
            if identity in charged_attempts and charged_attempts[identity] != signature:
                raise ValueError("One physical attempt has contradictory semantic identity or usage")
            paid_by_key[(stage, key)].add(attempt)
            local_paid.add(identity)
            if identity not in charged_attempts:
                charged_attempts[identity] = signature
                charged_here.update(usage)
                stage_tokens[stage].update(usage)
            counts["paid_success_rows"] += 1
            stage_events[stage]["paid_success_rows"] += 1
        matching = [row for row in starts if row.get("manifest_sha256") == manifest_sha]
        actors = [_actor_state(row, probe_processes) for row in matching]
        worker_rows.append(dict(worker_id=entry["worker_id"], stage=entry["stage"],
                                member_index=entry.get("member_index"), manifest_sha256=manifest_sha,
                                planned_subset_keys=len(allowed), unique_successful_keys=len(local_keys),
                                unique_paid_attempts_seen=len(local_paid), event_counts=dict(counts),
                                actual_tokens_attributed=dict(charged_here),
                                last_success_unix=last_success, actors=actors,
                                live_actor_count=sum(row.get("alive") is True for row in actors)))
    stages = {}
    for stage in ("member_generation", "aggregation"):
        usage = dict(stage_tokens[stage])
        usage.setdefault("prompt_tokens", 0)
        usage.setdefault("completion_tokens", 0)
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        stages[stage] = dict(unique_successful_keys=len(service_keys[stage]),
                             unique_paid_success_attempts=sum(identity[0] == stage for identity in charged_attempts),
                             unique_cached_success_keys=len(stage_cached[stage]),
                             event_counts=dict(stage_events[stage]), actual_tokens=usage)
    gpu = None
    if probe_gpu:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        gpu = dict(exit_code=result.returncode, rows=result.stdout.splitlines())
    return dict(schema="p2_registry_live_native_status_v3", at_unix=time.time(),
                stages=stages, workers=worker_rows, gpu=gpu,
                registered_manifest_journals_read=len(worker_rows),
                duplicate_success_rows_deduplicated=global_duplicate_rows,
                additional_paid_attempts_for_already_successful_keys=sum(max(0, len(attempts)-1)
                    for attempts in paid_by_key.values()),
                cost_attribution="Charge each noncached physical attempt once, attributed to its first registered journal occurrence",
                registry_trailing_partial_line_ignored=bool(registry.get("trailing_partial_line")),
                software_test_only=software_test_only, reference_access=False,
                quality_or_scores_read=0, new_llm_calls=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--software-test-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect_status(args.registry, args.root, probe_gpu=not args.no_gpu,
                                   software_test_only=args.software_test_only)))


if __name__ == "__main__":
    main()
