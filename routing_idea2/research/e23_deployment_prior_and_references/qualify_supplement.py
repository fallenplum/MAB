"""Synthetic lifecycle checks for the E23 persistent missing-call worker."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from supplement_worker import (
    CODE, ORIGINAL_WORKERS, P2_CODE, DependencyQueue, complete_events,
    restored_successes, run_loaded, sha,
)


sys.path.insert(0, str(P2_CODE))
import p2_member_worker as original_worker


def fixture(root, stage):
    spec = SimpleNamespace(model_id="synthetic")
    backend_instances = []

    def make_backend(value):
        instance = SimpleNamespace(spec=value, tokenizer=object())
        backend_instances.append(instance)
        return instance

    fingerprint_api = SimpleNamespace(configuration_fingerprint=lambda value: value.model_id)
    backend = original_worker.PersistentBackend(fingerprint_api, make_backend)
    keys = [("q", 42, 0), ("q", 42, 1)]
    members = {("q", 42, index) for index in range(4)}
    adapter = SimpleNamespace(
        scope={"synthetic": True}, required_members=members,
        callable_members=set(keys), callable_aggregates=set(keys), aggregator_spec=spec)
    available, reads, dispatched = set(), Counter(), []
    tokenizer_calls, sleeps = [], []

    def dependency_reader(module, actual_adapter, output_root, external, key):
        reads[key] += 1
        if key not in available:
            raise FileNotFoundError("Synthetic member not committed yet")
        return {"synthetic_member": key[2]}

    def native_task(actual_adapter, task, approval, output, factory, *, tokenizer=None,
                    member_records=None, software_test_only=False):
        if not software_test_only:
            raise AssertionError("A synthetic task must not enter production mode")
        instance = factory(spec)
        if stage == "aggregation":
            if tokenizer is not instance.tokenizer or len(member_records) != 3:
                raise AssertionError("Native invocation did not retain its tokenizer and triple")
            available.add(("q", 42, 3))
        dispatched.append(task)
        return dict(status="SUCCESS", physical_attempt_id=f"synthetic-{len(dispatched)}",
                    native_record_sha256=hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
                    from_cache=False,
                    native_record={"result": {"prompt_tokens": 3, "completion_tokens": 2}})

    module = SimpleNamespace(
        canonical_digest=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
        TEAMS=((0, 1, 2), (0, 1, 3)), run_approved_task=native_task)

    def tokenizer_loader(actual_module, actual_adapter, factory):
        tokenizer_calls.append(1)
        return factory(spec).tokenizer, {"software_test_only": True}

    worker = SimpleNamespace(
        append_event=original_worker.append_event, save_json=original_worker.save_json,
        native_usage=original_worker.native_usage, member_dependency=dependency_reader,
        context_tokenizer=tokenizer_loader)

    def sleeper(duration):
        sleeps.append(duration)
        available.update(("q", 42, index) for index in range(3))

    manifest = dict(stage=stage, approval_ref={"synthetic": True},
                    owned_output_root=str(root), worker_id="synthetic-owned")
    loaded = (module, adapter, keys, "aggregation" if stage == "aggregation" else 0, backend, {})
    return SimpleNamespace(worker=worker, loaded=loaded, manifest=manifest, module=module,
                           backend_instances=backend_instances, dispatched=dispatched,
                           reads=reads, sleeps=sleeps, tokenizer_calls=tokenizer_calls,
                           sleeper=sleeper, available=available)


def qualify():
    checks = {}
    for stage in ("member_generation", "aggregation"):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = fixture(root, stage)
            summary = run_loaded(data.worker, data.loaded, data.manifest, "synthetic",
                                 software_test_only=True, sleeper=data.sleeper)
            checks[stage+"_complete"] = summary["status"] == "COMPLETE" and summary["completed"] == 2
            checks[stage+"_one_backend"] = len(data.backend_instances) == summary["backend_loads"] == 1
            checks[stage+"_exact_two_calls"] = len(data.dispatched) == summary["new_successes"] == 2
            if stage == "aggregation":
                checks["aggregation_wait_then_resume"] = len(data.sleeps) == 1
                checks["aggregation_one_tokenizer"] = len(data.tokenizer_calls) == 1
                checks["aggregation_validated_dependency_cached"] = all(
                    count <= 2 for key, count in data.reads.items() if key[2] != 3)
                checks["aggregation_late_member_waited"] = data.reads[("q", 42, 3)] == 3
            repeated = run_loaded(data.worker, data.loaded, data.manifest, "synthetic",
                                  software_test_only=True, sleeper=data.sleeper)
            checks[stage+"_resume_no_backend_calls"] = (
                repeated["restored_successes"] == 2 and repeated["new_successes"] == 0
                and len(data.dispatched) == 2 and len(data.backend_instances) == 1)
            journal = root / "_dispatcher/synthetic/EVENTS.jsonl"
            expected_events = [row["event"] for row in complete_events(journal)]
            checks[stage+"_native_journal_pairing"] = expected_events == [
                "TASK_ENTER", "TASK_RESULT", "TASK_ENTER", "TASK_RESULT"]
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = fixture(root, "member_generation")
        stop = [False]
        original = data.module.run_approved_task

        def native_then_stop(*args, **kwargs):
            receipt = original(*args, **kwargs)
            stop[0] = True
            return receipt

        data.module.run_approved_task = native_then_stop
        summary = run_loaded(data.worker, data.loaded, data.manifest, "stopping",
                             software_test_only=True, stop=stop)
        checks["stop_commits_current_call"] = (
            summary["status"] == "SAFELY_STOPPED" and summary["completed"] == 1
            and len(data.dispatched) == 1)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = fixture(root, "member_generation")
        data.loaded[1].callable_members = {("q", 42, 0)}
        try:
            run_loaded(data.worker, data.loaded, data.manifest, "invalid", software_test_only=True)
        except ValueError:
            checks["cached_or_unapproved_key_rejected_before_backend"] = not data.dispatched
        else:
            checks["cached_or_unapproved_key_rejected_before_backend"] = False
        journal = root / "partial.jsonl"
        journal.write_bytes(b'{"event":"TASK_ENTER"}')
        try:
            list(complete_events(journal))
        except ValueError:
            checks["partial_journal_refused"] = True
        else:
            checks["partial_journal_refused"] = False
    for stage, (module_name, expected) in ORIGINAL_WORKERS.items():
        checks[stage+"_original_source_retained"] = sha(P2_CODE / f"{module_name}.py") == expected
    if not all(checks.values()):
        raise AssertionError("Worker qualification failed " + repr([name for name, value in checks.items() if not value]))
    return dict(
        schema="e23_supplement_worker_qualification_v1", status="PASS", checks=checks,
        checks_passed=len(checks), software_test_only=True,
        worker_sha256=sha(CODE / "supplement_worker.py"),
        original_worker_sha256={stage: value[1] for stage, value in ORIGINAL_WORKERS.items()},
        original_sources_modified=0, new_llm_calls=0, gold_values_read=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], checks_passed=report["checks_passed"])))
