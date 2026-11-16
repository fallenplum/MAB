"""Close P2 outputs from an append-only journal registry with semantic deduplication."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import tarfile
import time
import uuid

import p2_execution_adapter as adapter
from p2_manifest_registry import load_registry
from monitor_native_status_v3 import actual_usage, complete_events, semantic_key, unique_entries


OWNER_FILES = ("ATTEMPT_STATE.json", "LAST_TASK_RECEIPT.json", "records.jsonl")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def stable_bytes(path, owned_root):
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(owned_root.resolve(strict=True)):
        raise ValueError("Native source escaped its registered owned root")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 64 * 1024**2:
        raise ValueError("Only bounded regular native output files are exportable")
    data = path.read_bytes()
    after = path.stat()
    signature = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if signature(before) != signature(after) or len(data) != before.st_size:
        raise RuntimeError("Native output changed during closure")
    return data


def successful_candidates(registry):
    """Prefer the original paid evidence over cached replays of the same native owner."""
    found = {}
    sources = {}
    for entry in unique_entries(registry):
        allowed = set(map(tuple, entry["task_keys"]))
        for event in complete_events(entry["journal_path"]):
            if event.get("event") != "TASK_RESULT" or event.get("status") != "SUCCESS":
                continue
            stage, key = semantic_key(event)
            if stage != entry["stage"] or key not in allowed:
                raise ValueError("Journal success escaped its registered manifest")
            cached = event.get("from_cache")
            if type(cached) is not bool:
                raise ValueError("Native success cache state must be explicit")
            identity = (stage, key)
            if not cached:
                attempt = event.get("physical_attempt_id")
                if not isinstance(attempt, str) or not attempt:
                    raise ValueError("Paid output evidence requires a physical attempt ID")
                actual_usage(event)
            previous = found.get(identity)
            if previous is not None and not previous["from_cache"] and not cached:
                if (previous.get("physical_attempt_id") != event.get("physical_attempt_id")
                        or actual_usage(previous) != actual_usage(event)
                        or previous.get("native_record_sha256") != event.get("native_record_sha256")):
                    raise ValueError("One semantic output has inconsistent paid success evidence")
            if previous is None or (previous["from_cache"] and not cached):
                found[identity] = event
                sources[identity] = entry["worker_id"]
    return [(stage, key, event, sources[(stage, key)]) for (stage, key), event in found.items()]


def _recover_closed_batches(connection, exports, scope_sha):
    recovered = 0
    indexed = {row[0] for row in connection.execute("SELECT DISTINCT batch FROM exported")}
    for path in sorted(exports.glob("p2_native_v3_*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "p2_native_export_receipt_v3" or receipt.get("batch") in indexed:
            continue
        if receipt.get("scope_sha256") != scope_sha:
            raise ValueError("An unindexed closed native batch belongs to another scope")
        archive = Path(receipt["path"])
        if archive.parent.resolve() != exports.resolve() or hashlib.sha256(archive.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("Unindexed closed batch bytes differ")
        rows = [(row["stage"], row["digest"], receipt["batch"]) for row in receipt["index_rows"]]
        before = connection.total_changes
        connection.executemany("INSERT OR IGNORE INTO exported(stage,digest,batch) VALUES(?,?,?)", rows)
        recovered += connection.total_changes - before
    return recovered


def export_batch(registry_path, root=None, max_tasks=500, *, software_test_only=False):
    if type(max_tasks) is not int or max_tasks < 1:
        raise ValueError("A positive bounded native batch size is required")
    registry = load_registry(Path(registry_path))
    if bool(registry.get("software_test_only")) != software_test_only:
        raise PermissionError("Registry and native exporter execution modes differ")
    owned = Path(registry["common"]["owned_output_root"])
    root = Path(root) if root is not None else owned.parent
    scope_sha = registry["scope_canonical_sha256"]
    approval_sha = registry["common"]["approval_ref"]["sha256"]
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(exports / "exported.sqlite", timeout=30)
    connection.execute("CREATE TABLE IF NOT EXISTS exported (stage TEXT,digest TEXT,batch TEXT,PRIMARY KEY(stage,digest))")
    connection.commit()
    temporary = None
    try:
        # A single transaction serializes registry-based exporters. The legacy
        # exporter must be retired before switching the scheduled export entry.
        connection.execute("BEGIN IMMEDIATE")
        recovered = _recover_closed_batches(connection, exports, scope_sha)
        exported = {(stage, digest) for stage, digest in connection.execute("SELECT stage,digest FROM exported")}
        selected = []
        for stage, key, event, source_worker in successful_candidates(registry):
            digest = adapter.canonical_digest(list(key))
            if (stage, digest) not in exported:
                selected.append((stage, key, digest, event, source_worker))
            if len(selected) == max_tasks:
                break
        if not selected:
            connection.commit()
            return dict(schema="p2_native_export_receipt_v3", status="NO_NEW_COMMITTED_TASKS",
                        new_tasks=0, recovered_index_rows=recovered, gold_reads=0, new_llm_calls=0)
        batch = "p2_native_v3_" + str(time.time_ns()) + "_" + uuid.uuid4().hex[:8]
        target = exports / (batch + ".tar.gz")
        temporary = exports / (batch + ".part")
        task_rows = []
        prompt = completion = native_prompt = native_completion = 0
        with tarfile.open(temporary, "x:gz") as archive:
            for stage, key, digest, evidence, source_worker in selected:
                owner = owned / scope_sha / stage / digest
                payload = {name: stable_bytes(owner / name, owned) for name in OWNER_FILES}
                fence = json.loads(payload["ATTEMPT_STATE.json"])
                receipt = json.loads(payload["LAST_TASK_RECEIPT.json"])
                native = receipt["native_record"]
                native_sha = adapter.canonical_digest({name: value for name, value in native.items() if name != "record_sha256"})
                task = receipt.get("task_key", {})
                field = "member_index" if stage == "member_generation" else "combination_index"
                received_key = (task.get("question_id"), task.get("generation_seed"), task.get(field))
                if (receipt.get("schema") != "p2_execution_task_receipt_v1" or receipt.get("cohort") != adapter.COHORT
                        or receipt.get("scope_sha256") != scope_sha or receipt.get("approval_sha256") != approval_sha
                        or receipt.get("status") != "SUCCESS" or receipt.get("software_test_only") is not software_test_only
                        or receipt.get("stage") != stage or received_key != key
                        or fence.get("status") != "SUCCESS_COMMITTED" or fence.get("scope_sha256") != scope_sha
                        or fence.get("stage") != stage or tuple(fence.get("task_key", ())) != key
                        or native.get("record_sha256") != native_sha or receipt.get("native_record_sha256") != native_sha
                        or fence.get("native_record_sha256") != native_sha
                        or native.get("status") != "success" or native.get("cost_status") != "recorded_actual_usage"
                        or evidence.get("native_record_sha256") != native_sha):
                    raise ValueError("Native owner semantic identity, scope or actual-cost qualification differs")
                if [json.loads(line) for line in payload["records.jsonl"].splitlines() if line.strip()] != [native]:
                    raise ValueError("Native Store does not contain exactly the committed record")
                usage = actual_usage({"actual_usage": native.get("result") or native.get("actual_usage")})
                native_prompt += usage["prompt_tokens"]
                native_completion += usage["completion_tokens"]
                if not evidence["from_cache"]:
                    if evidence["physical_attempt_id"] != receipt.get("physical_attempt_id") or actual_usage(evidence) != usage:
                        raise ValueError("Paid journal and committed native usage disagree")
                    prompt += usage["prompt_tokens"]
                    completion += usage["completion_tokens"]
                file_rows = []
                for name, data in payload.items():
                    relative = (owner / name).relative_to(root).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
                    file_rows.append(dict(path=relative, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
                task_rows.append(dict(stage=stage, task_key=list(key), digest=digest, native_record_sha256=native_sha,
                                      source_worker_id=source_worker, from_cache=evidence["from_cache"],
                                      physical_attempt_id=receipt.get("physical_attempt_id"), files=file_rows))
            manifest = dict(schema="p2_closed_native_export_v1", exporter_version=3, batch=batch,
                            scope_sha256=scope_sha, registry_sha256=hashlib.sha256(Path(registry_path).read_bytes()).hexdigest(),
                            tasks=task_rows, prompt_tokens=prompt, completion_tokens=completion,
                            native_output_prompt_tokens=native_prompt, native_output_completion_tokens=native_completion,
                            software_test_only=software_test_only, gold_reads=0)
            data = encoded(manifest)
            info = tarfile.TarInfo("P2_NATIVE_EXPORT_MANIFEST.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if os.name == "nt":
            # Windows rename refuses an existing target, including in the app sandbox.
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        temporary = None
        index_rows = [dict(stage=stage, digest=digest) for stage, _, digest, _, _ in selected]
        receipt = dict(schema="p2_native_export_receipt_v3", status="CLOSED_NATIVE_BATCH", batch=batch,
                       path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size,
                       tasks=len(task_rows), prompt_tokens=prompt, completion_tokens=completion,
                       native_output_prompt_tokens=native_prompt, native_output_completion_tokens=native_completion,
                       recovered_index_rows=recovered, scope_sha256=scope_sha, index_rows=index_rows,
                       software_test_only=software_test_only, gold_reads=0, new_llm_calls=0)
        with (exports / (batch + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        connection.executemany("INSERT INTO exported(stage,digest,batch) VALUES(?,?,?)",
                               [(row["stage"], row["digest"], batch) for row in index_rows])
        connection.commit()
        return receipt
    except BaseException:
        connection.rollback()
        # An incomplete .part stays distinguishable from closed transfer batches.
        # A closed sidecar permits deterministic index recovery on the next call.
        raise
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--max-tasks", type=int, default=500)
    parser.add_argument("--software-test-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_batch(args.registry, args.root, args.max_tasks,
                                 software_test_only=args.software_test_only)))


if __name__ == "__main__":
    main()
