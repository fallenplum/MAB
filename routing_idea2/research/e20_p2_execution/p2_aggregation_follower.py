"""Aggregate only completed member triples with one persistent original backend."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import p2_aggregation_worker as worker


ROOT = Path("/data2/chendong/e20_p2_execution")


def publish(path, value):
    worker.save_json(path, value)


class MemberJournalTail:
    def __init__(self, paths):
        self.positions = {path: 0 for path in paths}

    def completed(self):
        for path in self.positions:
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                stream.seek(self.positions[path])
                while True:
                    start = stream.tell()
                    raw = stream.readline()
                    if not raw or not raw.endswith(b"\n"):
                        self.positions[path] = start
                        break
                    row = json.loads(raw)
                    self.positions[path] = stream.tell()
                    if row.get("event") == "TASK_RESULT" and row.get("status") == "SUCCESS":
                        task = row["task"]
                        yield (task["question_id"], task["generation_seed"], task["member_index"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    if worker.file_sha(args.manifest) != args.manifest_sha256:
        raise ValueError("Aggregation manifest changed")
    manifest = worker.read_json(args.manifest)
    if manifest.get("software_test_only") is not False:
        raise PermissionError("Production manifest required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    stop = [False]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))
    with worker.GPUOwnership(manifest["gpu_lock_root"], args.gpu, args.manifest_sha256):
        module, adapter, keys, _, backend, external = worker.load_worker(manifest)
        adapter._worker_software_mode = False
        scope_sha = module.canonical_digest(adapter.scope)
        journal_root = Path(manifest["owned_output_root"]) / "_dispatcher" / args.manifest_sha256
        journal_root.mkdir(parents=True, exist_ok=True)
        journal = journal_root / "EVENTS.jsonl"
        summary_path = journal_root / "SUMMARY.json"
        completed = set()
        if journal.is_file():
            for line in journal.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("event") == "TASK_RESULT" and row.get("status") == "SUCCESS":
                    task = row["task"]
                    completed.add((task["question_id"], task["generation_seed"], task["combination_index"]))
        pending = {}
        dependants = defaultdict(list)
        for key in keys:
            if key in completed:
                continue
            pending[key] = 3
            for member in module.TEAMS[key[2]]:
                dependants[(key[0], key[1], member)].append(key)
        tails = []
        for index in range(6):
            path = ROOT / "manifests" / f"member{index}_v2.json"
            tails.append(Path(manifest["owned_output_root"]) / "_dispatcher" / worker.file_sha(path) / "EVENTS.jsonl")
        tail = MemberJournalTail(tails)
        seen_members = set()
        ready = deque()
        tokenizer = None
        started_cpu, started_wall = time.process_time(), time.perf_counter()
        newly_completed = 0
        last_status = 0.0
        while len(completed) < len(keys) and not stop[0]:
            for member_key in tail.completed():
                if member_key in seen_members:
                    continue
                if member_key not in adapter.required_members:
                    raise ValueError("Member journal escaped the frozen P2 support scope")
                seen_members.add(member_key)
                for key in dependants.get(member_key, ()):
                    pending[key] -= 1
                    if pending[key] == 0:
                        ready.append(key)
            if not ready:
                now = time.monotonic()
                if now - last_status > 30:
                    publish(summary_path, dict(status="WAITING_FOR_COMMITTED_MEMBERS", stage="aggregation",
                            scope_sha256=scope_sha, planned=len(keys), completed=len(completed),
                            observed_member_successes=len(seen_members), ready=0, backend_loads=backend.loads,
                            pid=os.getpid(), at_unix=time.time()))
                    last_status = now
                time.sleep(2)
                continue
            key = ready.popleft()
            if key in completed:
                continue
            records = {member: worker.member_dependency(module, adapter, Path(manifest["owned_output_root"]),
                        external, (key[0], key[1], member)) for member in module.TEAMS[key[2]]}
            if tokenizer is None:
                tokenizer, alias_receipt = worker.context_tokenizer(module, adapter, backend)
                publish(journal_root / "TOKENIZER_ALIAS_RECEIPT.json", alias_receipt)
            task = dict(stage="aggregation", question_id=key[0], generation_seed=key[1], combination_index=key[2])
            worker.append_event(journal, dict(event="TASK_ENTER",task=task,at_unix=time.time()))
            try:
                receipt = module.run_approved_task(adapter, task, manifest["approval_ref"], manifest["owned_output_root"],
                                                   backend, tokenizer=tokenizer, member_records=records)
            except BaseException as error:
                worker.append_event(journal, dict(event="STOP_REQUIRES_RECONCILIATION",task=task,
                        error_type=type(error).__name__, error=str(error), at_unix=time.time()))
                raise
            worker.append_event(journal, dict(event="TASK_RESULT",task=task,status=receipt["status"],
                    physical_attempt_id=receipt["physical_attempt_id"], native_record_sha256=receipt["native_record_sha256"],
                    from_cache=receipt["from_cache"], actual_usage=worker.native_usage(receipt["native_record"]),at_unix=time.time()))
            if receipt["status"] != "SUCCESS":
                raise RuntimeError("Native aggregation failed and requires owner reconciliation")
            completed.add(key)
            newly_completed += 1
            publish(summary_path, dict(status="RUNNING",stage="aggregation",scope_sha256=scope_sha,
                    planned=len(keys),completed=len(completed),observed_member_successes=len(seen_members),ready=len(ready),
                    backend_loads=backend.context.constructions,pid=os.getpid(),at_unix=time.time()))
        publish(summary_path, dict(status="COMPLETE" if len(completed)==len(keys) else "SAFELY_STOPPED",stage="aggregation",
                scope_sha256=scope_sha,planned=len(keys),completed=len(completed),newly_completed=newly_completed,
                observed_member_successes=len(seen_members),cpu_seconds=time.process_time()-started_cpu,
                wall_seconds=time.perf_counter()-started_wall,backend_loads=backend.context.constructions,
                stopped=stop[0],at_unix=time.time(),pid=os.getpid(),gold_reads=0))


if __name__ == "__main__":
    main()
