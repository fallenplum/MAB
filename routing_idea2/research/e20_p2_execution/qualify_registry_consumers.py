"""Small local synthetic qualification for registry progress and native export."""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile

import monitor_native_status_v3 as monitor
import p2_execution_adapter as adapter
import p2_export_native_v3 as exporter
import p2_manifest_registry as registry


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return registry.reference(path)


def append_event(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(json.dumps(event).encode() + b"\n")


def fixture(root):
    owned = root / "owned"
    members = [["synthetic-a", 42, index] for index in range(6)]
    aggregates = [["synthetic-" + label, 42, 0] for label in ("a", "b", "c")]
    scope = dict(schema="p2_scope_contract_v1", role="execution", cohort=adapter.COHORT,
                 bindings={"final_model_sha256": "synthetic-only"}, required_member_keys=members,
                 required_aggregate_keys=aggregates,
                 execution_limits={"member_generation": len(members), "aggregation": len(aggregates)})
    scope_ref = write_json(root / "scope.json", scope)
    approval = dict(schema="p2_scope_approval_v1", status="APPROVED", cohort=adapter.COHORT,
                    software_test_only=True, scope_sha256_by_role={"execution": registry.digest(scope)},
                    approved_output_root=str(owned), bindings=scope["bindings"],
                    execution_limits=scope["execution_limits"])
    approval_ref = write_json(root / "approval.json", approval)
    common = dict(scope_ref=scope_ref, approval_ref=approval_ref, owned_output_root=str(owned),
                  question_records_ref=write_json(root / "input.json", {"input_only": True}),
                  adapter_ref=registry.reference(Path(adapter.__file__)), asset_root=str(root))
    worker_source = root / "worker.py"
    worker_source.write_text("# synthetic source, never executed\n", encoding="utf-8")
    source_ref = registry.reference(worker_source)

    def spec(worker_id, stage, keys, member_index=None):
        manifest = dict(common, stage=stage, software_test_only=True, backend={"profile": member_index},
                        task_keys_ref=write_json(root / (worker_id + "_keys.json"), keys))
        if member_index is not None:
            manifest["member_index"] = member_index
        return dict(worker_id=worker_id, worker_source_ref=source_ref,
                    manifest_ref=write_json(root / (worker_id + ".json"), manifest))

    specs = [spec("original-member" + str(index), "member_generation", [members[index]], index)
             for index in range(6)]
    specs.append(spec("original-aggregation", "aggregation", aggregates))
    path = root / "registry.jsonl"
    registry.create_registry(path, specs)
    registry.add_manifest(path, spec("successor-aggregation", "aggregation", aggregates))
    registry.add_manifest(path, spec("successor-member0", "member_generation", [members[0]], 0))
    return path, common, scope, spec


def committed_event(root, common, scope, stage, key, prompt, completion, attempt, cached=False):
    scope_sha = registry.digest(scope)
    task = dict(stage=stage, question_id=key[0], generation_seed=key[1])
    task["member_index" if stage == "member_generation" else "combination_index"] = key[2]
    native = dict(status="success", cost_status="recorded_actual_usage", attempt_id=attempt,
                  result={"prompt_tokens": prompt, "completion_tokens": completion,
                          "text": "synthetic-output-not-a-model-answer"})
    native["record_sha256"] = adapter.canonical_digest(native)
    owner = root / "owned" / scope_sha / stage / adapter.canonical_digest(list(key))
    # The production receipt keeps stage at top level and a stage-free task_key.
    receipt = dict(schema="p2_execution_task_receipt_v1", cohort=adapter.COHORT, status="SUCCESS",
                   scope_sha256=scope_sha, approval_sha256=common["approval_ref"]["sha256"],
                   stage=stage, task_key={name: value for name, value in task.items() if name != "stage"},
                   native_record=native, native_record_sha256=native["record_sha256"],
                   physical_attempt_id=attempt, software_test_only=True, from_cache=cached)
    fence = dict(status="SUCCESS_COMMITTED", scope_sha256=scope_sha, stage=stage,
                 task_key=list(key), native_record_sha256=native["record_sha256"])
    write_json(owner / "ATTEMPT_STATE.json", fence)
    write_json(owner / "LAST_TASK_RECEIPT.json", receipt)
    (owner / "records.jsonl").write_text(json.dumps(native) + "\n", encoding="utf-8")
    return dict(event="TASK_RESULT", status="SUCCESS", task=task, from_cache=cached,
                physical_attempt_id=attempt, native_record_sha256=native["record_sha256"],
                actual_usage={"prompt_tokens": prompt, "completion_tokens": completion}, at_unix=1)


def expect_raises(callback, error=ValueError):
    try:
        callback()
    except error:
        return True
    raise AssertionError("Expected rejection was not raised")


def qualify():
    checks = {}
    # A short workspace-local path also exercises native owners on Windows.
    with tempfile.TemporaryDirectory(prefix="_q3_", dir=Path.cwd()) as folder:
        root = Path(folder)
        path, common, scope, spec = fixture(root)
        state = registry.load_registry(path)
        by_id = {row["worker_id"]: row for row in state["entries"]}
        journals = {key: Path(value["journal_path"]) for key, value in by_id.items()}
        member = committed_event(root, common, scope, "member_generation", ("synthetic-a", 42, 0), 5, 1, "m0")
        first = committed_event(root, common, scope, "aggregation", ("synthetic-a", 42, 0), 10, 2, "a0")
        second = committed_event(root, common, scope, "aggregation", ("synthetic-b", 42, 0), 20, 3, "a1")
        cached_only = committed_event(root, common, scope, "aggregation", ("synthetic-c", 42, 0), 30, 4, "a2", True)
        append_event(journals["original-member0"], member)
        append_event(journals["successor-member0"], dict(member, from_cache=True))
        append_event(journals["original-aggregation"], first)
        append_event(journals["successor-aggregation"], dict(first, from_cache=True))
        append_event(journals["successor-aggregation"], first)
        append_event(journals["successor-aggregation"], second)
        append_event(journals["successor-aggregation"], cached_only)
        with journals["successor-aggregation"].open("ab") as stream:
            stream.write(b'{"event":"INCOMPLETE')
        for worker_id in ("original-member0", "successor-member0"):
            write_json(root / "control" / (worker_id + ".start.json"),
                       dict(worker_id=worker_id, manifest_sha256=by_id[worker_id]["manifest_ref"]["sha256"], pid=1))
        status = monitor.collect_status(path, probe_processes=False, probe_gpu=False, software_test_only=True)
        aggregation = status["stages"]["aggregation"]
        assert aggregation["unique_successful_keys"] == 3
        checks["old_and_new_journals_visible"] = True
        assert aggregation["unique_paid_success_attempts"] == 2 and aggregation["actual_tokens"]["total_tokens"] == 35
        checks["semantic_and_physical_attempt_deduplication"] = True
        assert aggregation["unique_cached_success_keys"] == 2
        checks["cached_rows_do_not_charge_tokens"] = True
        assert len(status["workers"]) == 9 and sum(row["member_index"] == 0 for row in status["workers"]) == 2
        assert sum(len(row["actors"]) for row in status["workers"]) == 2
        checks["all_same_profile_manifest_actors_visible"] = True
        checks["partial_journal_line_ignored"] = True
        checks["production_consumer_rejects_fixture_mode"] = expect_raises(
            lambda: monitor.collect_status(path, probe_processes=False, probe_gpu=False), PermissionError)
        checks["duplicate_manifest_semantic_keys_rejected"] = expect_raises(lambda: registry.add_manifest(
            path, spec("bad-duplicate", "aggregation", [scope["required_aggregate_keys"][0]] * 2)))

        # A legacy already-closed key remains in the exact same SQLite index.
        exports = root / "exports"
        exports.mkdir()
        with closing(sqlite3.connect(exports / "exported.sqlite")) as connection:
            connection.execute("CREATE TABLE exported (stage TEXT,digest TEXT,batch TEXT,PRIMARY KEY(stage,digest))")
            connection.execute("INSERT INTO exported VALUES(?,?,?)", ("member_generation",
                               adapter.canonical_digest(["synthetic-a", 42, 0]), "p2_native_legacy"))
            connection.commit()
        closed = exporter.export_batch(path, max_tasks=10, software_test_only=True)
        assert closed["tasks"] == 3 and closed["prompt_tokens"] + closed["completion_tokens"] == 35
        assert closed["native_output_prompt_tokens"] + closed["native_output_completion_tokens"] == 69
        checks["export_native_identity_and_paid_only_cost"] = True
        with tarfile.open(closed["path"], "r:gz") as archive:
            exported = json.load(archive.extractfile("P2_NATIVE_EXPORT_MANIFEST.json"))
            assert len(exported["tasks"]) == 3 and exported["exporter_version"] == 3
        checks["old_sqlite_index_retained_and_archive_semantic_union"] = True
        again = exporter.export_batch(path, max_tasks=10, software_test_only=True)
        assert again["status"] == "NO_NEW_COMMITTED_TASKS"
        checks["second_export_is_noop"] = True
        with closing(sqlite3.connect(exports / "exported.sqlite")) as connection:
            connection.execute("DELETE FROM exported WHERE batch=?", (closed["batch"],))
            connection.commit()
        recovered = exporter.export_batch(path, max_tasks=10, software_test_only=True)
        assert recovered["status"] == "NO_NEW_COMMITTED_TASKS" and recovered["recovered_index_rows"] == 3
        checks["closed_archive_index_recovers_without_reexport"] = True
        assert len(list(exports.glob("*.tar.gz"))) == 1
        checks["no_duplicate_closed_archive"] = True

        corrupt = root / "complete-corrupt.jsonl"
        corrupt.write_bytes(b'{"event":bad}\n')
        checks["complete_bad_json_is_rejected"] = expect_raises(lambda: list(monitor.complete_events(corrupt)))
        conflict = deepcopy(first)
        conflict["actual_usage"]["prompt_tokens"] = 999
        append_event(journals["original-aggregation"], conflict)
        checks["contradictory_attempt_usage_rejected"] = expect_raises(lambda: monitor.collect_status(
            path, probe_processes=False, probe_gpu=False, software_test_only=True))
    return dict(schema="p2_registry_consumers_qualification_v1", status="PASS", checks=checks,
                checks_passed=len(checks), software_test_only=True, gold_reads=0, new_llm_calls=0,
                ssh_or_process_changes=0, production_deployed=False)


if __name__ == "__main__":
    result = qualify()
    files = (Path(monitor.__file__), Path(exporter.__file__), Path(__file__))
    result["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    target = Path(__file__).with_name("P2_REGISTRY_CONSUMERS_QUALIFICATION.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
