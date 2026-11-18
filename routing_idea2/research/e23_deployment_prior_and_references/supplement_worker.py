"""Persistent native inference for the approved E23 missing-call union."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time


CODE = Path(__file__).resolve().parent
P2_CODE = CODE if (CODE / "p2_member_worker.py").is_file() else CODE.parent / "e20_p2_execution"
ORIGINAL_WORKERS = {
    "member_generation": ("p2_member_worker", "58839460471e33536cd2b00274ede029b98355d355df2919b8bf3790f16b6c81"),
    "aggregation": ("p2_aggregation_worker", "9b85c1bb42864c8d5845b2d60cab0b6caf59f820564f12b84168d28fa2610a70"),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def complete_events(path):
    if not path.is_file():
        return
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                raise ValueError("A prior partial journal line requires reconciliation")
            yield json.loads(raw)


def semantic_key(event, stage):
    task = event["task"]
    field = "member_index" if stage == "member_generation" else "combination_index"
    if task.get("stage") != stage:
        raise ValueError("Worker journal event belongs to another stage")
    return str(task["question_id"]), int(task["generation_seed"]), int(task[field])


def restored_successes(path, stage, keys):
    allowed, completed = set(keys), set()
    for event in complete_events(path):
        if event.get("event") != "TASK_RESULT" or event.get("status") != "SUCCESS":
            continue
        key = semantic_key(event, stage)
        if key not in allowed or key in completed:
            raise ValueError("Journal success is repeated or outside this worker's task set")
        completed.add(key)
    return completed


def load_original_worker(stage, adapter_reference):
    if stage not in ORIGINAL_WORKERS:
        raise ValueError("The supplement worker requires a native inference stage")
    name, expected_sha = ORIGINAL_WORKERS[stage]
    path = P2_CODE / f"{name}.py"
    if sha(path) != expected_sha:
        raise ValueError("Original native worker source differs")
    sys.path.insert(0, str(P2_CODE))
    original = importlib.import_module(name)
    adapter_path = original.checked_ref(adapter_reference)
    if adapter_path.name != "supplement_adapter.py":
        raise ValueError("The E23 worker requires its explicitly approved supplement adapter")
    module_name = "e23_supplement_adapter"
    existing = sys.modules.get(module_name)
    if existing is not None:
        if Path(existing.__file__).resolve() != adapter_path:
            raise ValueError("Another E23 adapter path is already active")
        adapter_module = existing
    else:
        specification = importlib.util.spec_from_file_location(module_name, adapter_path)
        adapter_module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = adapter_module
        specification.loader.exec_module(adapter_module)

    def bound_adapter(reference):
        if reference != adapter_reference or sha(adapter_path) != adapter_reference["sha256"]:
            raise ValueError("The E23 adapter binding changed")
        return adapter_module

    original.load_adapter = bound_adapter
    return original


class DependencyQueue:
    """Check each successful member once and keep its validated native record."""

    def __init__(self, module, adapter, keys, output_root, external, dependency_reader):
        self.module = module
        self.adapter = adapter
        self.output_root = output_root
        self.external = external
        self.dependency_reader = dependency_reader
        self.members = {}
        self.pending_counts = {}
        self.dependants = defaultdict(list)
        self.waiting_members = set()
        self.ready = deque()
        for key in keys:
            self.pending_counts[key] = 3
            for member in module.TEAMS[key[2]]:
                member_key = (key[0], key[1], member)
                if member_key not in adapter.required_members:
                    raise ValueError("Aggregate dependency is outside the supplement scope")
                self.dependants[member_key].append(key)
                self.waiting_members.add(member_key)

    def refresh(self):
        for member_key in sorted(self.waiting_members):
            try:
                record = self.dependency_reader(
                    self.module, self.adapter, self.output_root, self.external, member_key)
            except FileNotFoundError:
                continue
            self.members[member_key] = record
            self.waiting_members.remove(member_key)
            for key in self.dependants[member_key]:
                self.pending_counts[key] -= 1
                if self.pending_counts[key] == 0:
                    self.ready.append(key)

    def records(self, key):
        return {member: self.members[(key[0], key[1], member)]
                for member in self.module.TEAMS[key[2]]}


def run_loaded(worker, loaded, manifest, manifest_sha, *, software_test_only=False,
               stop=None, sleeper=time.sleep, injected_tokenizer=None):
    module, adapter, keys, profile, backend, external = loaded
    stage = manifest["stage"]
    callable_keys = adapter.callable_members if stage == "member_generation" else adapter.callable_aggregates
    if not set(keys) <= callable_keys or len(keys) != len(set(keys)):
        raise ValueError("A supplement actor may execute only unique new callable task keys")
    adapter._worker_software_mode = software_test_only
    output_root = Path(manifest["owned_output_root"]).resolve()
    scope_sha = module.canonical_digest(adapter.scope)
    journal_root = output_root / "_dispatcher" / manifest_sha
    journal_root.mkdir(parents=True, exist_ok=True)
    journal = journal_root / "EVENTS.jsonl"
    completed = restored_successes(journal, stage, keys)
    restored = len(completed)
    remaining = [key for key in keys if key not in completed]
    ready = deque(remaining) if stage == "member_generation" else None
    dependency_queue = None
    if stage == "aggregation":
        dependency_queue = DependencyQueue(
            module, adapter, remaining, output_root, external, worker.member_dependency)
        ready = dependency_queue.ready
    stop = [False] if stop is None else stop
    tokenizer, cached, failures = injected_tokenizer, 0, []
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    last_status = 0.0

    def status(state):
        return dict(
            schema="e23_native_worker_summary_v1", status=state,
            stage=stage, profile=profile, scope_sha256=scope_sha,
            manifest_sha256=manifest_sha, worker_id=manifest.get("worker_id"),
            planned=len(keys), completed=len(completed), restored_successes=restored,
            new_successes=len(completed)-restored-cached, cached=cached,
            backend_loads=backend.loads, pid=os.getpid(), stopped=stop[0],
            ready=len(ready), waiting_members=len(dependency_queue.waiting_members) if dependency_queue else 0,
            failures=failures, at_unix=time.time(), gold_reads=0,
            cpu_seconds=time.process_time()-started_cpu,
            wall_seconds=time.perf_counter()-started_wall,
        )

    try:
        while len(completed) < len(keys) and not stop[0]:
            if dependency_queue is not None:
                dependency_queue.refresh()
            if not ready:
                if time.monotonic()-last_status >= 30:
                    worker.save_json(journal_root / "SUMMARY.json", status("WAITING_FOR_COMMITTED_MEMBERS"))
                    last_status = time.monotonic()
                sleeper(2)
                continue
            key = ready.popleft()
            if key in completed:
                continue
            records = dependency_queue.records(key) if dependency_queue is not None else None
            if stage == "aggregation" and tokenizer is None:
                tokenizer, tokenizer_receipt = worker.context_tokenizer(module, adapter, backend)
                worker.save_json(journal_root / "TOKENIZER_ALIAS_RECEIPT.json", tokenizer_receipt)
            task = dict(stage=stage, question_id=key[0], generation_seed=key[1])
            task["member_index" if stage == "member_generation" else "combination_index"] = key[2]
            worker.append_event(journal, dict(event="TASK_ENTER", task=task, at_unix=time.time()))
            try:
                receipt = module.run_approved_task(
                    adapter, task, manifest["approval_ref"], manifest["owned_output_root"],
                    backend, tokenizer=tokenizer, member_records=records,
                    software_test_only=software_test_only)
            except BaseException as error:
                failure = dict(event="STOP_REQUIRES_RECONCILIATION", task=task,
                               error_type=type(error).__name__, error=str(error), at_unix=time.time())
                worker.append_event(journal, failure)
                failures.append(dict(task=task, error_type=type(error).__name__))
                raise
            usage = worker.native_usage(receipt["native_record"])
            if usage is None:
                raise ValueError("A successful native result requires recorded actual usage")
            worker.append_event(journal, dict(
                event="TASK_RESULT", task=task, status=receipt["status"],
                physical_attempt_id=receipt["physical_attempt_id"],
                native_record_sha256=receipt["native_record_sha256"],
                from_cache=receipt["from_cache"], actual_usage=usage, at_unix=time.time()))
            if receipt["status"] != "SUCCESS":
                failures.append(dict(task=task, physical_attempt_id=receipt["physical_attempt_id"]))
                raise RuntimeError("Supplement native output requires reconciliation")
            completed.add(key)
            cached += int(receipt["from_cache"])
            worker.save_json(journal_root / "SUMMARY.json", status("RUNNING"))
        state = "COMPLETE" if len(completed) == len(keys) else "SAFELY_STOPPED"
    except BaseException:
        worker.save_json(journal_root / "SUMMARY.json", status("STOP_REQUIRES_RECONCILIATION"))
        raise
    summary = status(state)
    worker.save_json(journal_root / "SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    if sha(args.manifest) != args.manifest_sha256:
        raise ValueError("Supplement manifest bytes changed")
    manifest = read(args.manifest)
    if manifest.get("software_test_only") is not False:
        raise PermissionError("The CLI admits production manifests only")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    worker = load_original_worker(manifest["stage"], manifest["adapter_ref"])
    stop, prior_handlers = [False], {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        prior_handlers[signum] = signal.signal(signum, lambda *_: stop.__setitem__(0, True))
    try:
        with worker.GPUOwnership(manifest["gpu_lock_root"], args.gpu, args.manifest_sha256):
            loaded = worker.load_worker(manifest)
            journal_root = Path(manifest["owned_output_root"]) / "_dispatcher" / args.manifest_sha256
            journal_root.mkdir(parents=True, exist_ok=True)
            worker.save_json(journal_root / "WRAPPER_RECEIPT.json", dict(
                schema="e23_worker_source_wrapper_v1", wrapper_sha256=sha(__file__),
                original_worker_sha256=ORIGINAL_WORKERS[manifest["stage"]][1],
                adapter_ref=manifest["adapter_ref"], manifest_sha256=args.manifest_sha256,
                original_load_worker_calls=1, native_backend_implementation_changed=False,
                scientific_generation_parameters_changed=False, at_unix=time.time()))
            result = run_loaded(worker, loaded, manifest, args.manifest_sha256, stop=stop)
            print(json.dumps(result), flush=True)
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
