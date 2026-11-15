"""Consume immutable E18 ready plans with one persistent aggregation process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import e18_aggregation_dispatcher_v3 as dispatcher
import e18_aggregation_worker_v3 as worker


ENTRY_SCHEMA = "e18_aggregation_queue_entry_v1"
EXPECTED_WORKER_SHA = "de5c3cc8a509be10e66b3290899862bab73a2784a62aa743e7c48d41a7842495"
EXPECTED_DISPATCHER_SHA = (
    "715fb1515968717941695224cf5f9a7e60e4fdaa174e18330bf12a9b72911ec6"
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def publish_once(path, value):
    """Atomically publish complete JSON bytes using a non-replacing hard link."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        "." + path.name + ".tmp-" + str(os.getpid()) + "-" + str(time.time_ns())
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def read_stable_entry(path):
    before = path.stat()
    reference = dict(path=str(path.resolve()), sha256=worker.file_sha(path))
    entry = worker.read_json(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Ready entry changed while being read")
    if (
        not isinstance(entry, dict)
        or set(entry) != {"schema", "plan_ref"}
        or entry["schema"] != ENTRY_SCHEMA
    ):
        raise ValueError("Unknown or malformed aggregation queue entry")
    plan = worker.read_json(worker.checked_ref(entry["plan_ref"]))
    return reference, entry, plan


def shared_identity(manifest):
    return dict(
        asset_root=str(Path(manifest["asset_root"]).resolve()),
        adapter_ref=manifest["adapter_ref"],
        scope_ref=manifest["scope_ref"],
        approval_ref=manifest["approval_ref"],
        inputs_ref=manifest["question_records_ref"],
        owned_output_root=str(Path(manifest["owned_output_root"]).resolve()),
        gpu_lock_root=str(Path(manifest["gpu_lock_root"]).resolve()),
        backend_loading=manifest["backend"],
        software_test_only=manifest["software_test_only"],
        worker_source_sha256=EXPECTED_WORKER_SHA,
        dispatcher_source_sha256=EXPECTED_DISPATCHER_SHA,
    )


class AggregationQueue:
    def __init__(
        self,
        queue_root,
        gpu_lock_root,
        *,
        metadata_only=False,
        software_test_only=False,
        fixture_factory=None,
        fixture_runner=None,
    ):
        if not software_test_only and (
            fixture_factory is not None or fixture_runner is not None
        ):
            raise PermissionError(
                "Production accepts frozen original factories and run_plan only"
            )
        if (
            worker.file_sha(worker.__file__) != EXPECTED_WORKER_SHA
            or worker.file_sha(dispatcher.__file__) != EXPECTED_DISPATCHER_SHA
            or dispatcher.worker is not worker
        ):
            raise ValueError("Frozen aggregation core identity differs")
        self.root = Path(queue_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_root = Path(gpu_lock_root).resolve()
        self.metadata_only = metadata_only
        self.software_test_only = software_test_only
        self.fixture_factory = fixture_factory
        self.runner = fixture_runner or dispatcher.run_plan
        self.state = self.root / ("metadata_receipts" if metadata_only else "receipts")
        self.state.mkdir(exist_ok=True)
        self.identity = None
        self.key_owners = {}
        self.accepted = {}
        self.completed = set()
        self._restore()

    def _restore(self):
        for path in sorted(self.state.glob("*.accepted.json")):
            record = worker.read_json(path)
            if (
                record.get("schema") != "e18_aggregation_queue_acceptance_v1"
                or record.get("metadata_only") is not self.metadata_only
                or record.get("software_test_only") is not self.software_test_only
            ):
                raise ValueError(
                    "Stored queue acceptance has an incompatible mode or schema"
                )
            entry_ref, entry, _ = read_stable_entry(
                worker.checked_ref(record["entry_ref"])
            )
            if (
                entry_ref != record["entry_ref"]
                or entry["plan_ref"] != record["plan_ref"]
            ):
                raise ValueError("Stored queue entry or plan binding differs")
            self._register(record)
            done_path = self._completion_path(record["entry_ref"]["sha256"])
            if done_path.exists():
                done = worker.read_json(done_path)
                expected_status = "METADATA_ONLY" if self.metadata_only else "DONE"
                if (
                    done.get("schema") != "e18_aggregation_queue_completion_v1"
                    or done.get("status") != expected_status
                    or done.get("acceptance_sha256") != worker.file_sha(path)
                    or done.get("plan_ref") != record["plan_ref"]
                ):
                    raise ValueError(
                        "Stored completion does not bind its immutable acceptance"
                    )
                self.completed.add(record["entry_ref"]["sha256"])

    def _completion_path(self, entry_sha):
        return self.state / (
            entry_sha + (".metadata.json" if self.metadata_only else ".done.json")
        )

    def _register(self, record):
        entry_sha = record["entry_ref"]["sha256"]
        if (
            record["identity"].get("worker_source_sha256") != EXPECTED_WORKER_SHA
            or record["identity"].get("dispatcher_source_sha256")
            != EXPECTED_DISPATCHER_SHA
            or record.get("task_count") != len(record["task_keys"])
            or len(record["task_keys"]) != len(set(map(tuple, record["task_keys"])))
        ):
            raise ValueError("Stored queue core or semantic key identity differs")
        if record.get("identity_sha256") != digest(record["identity"]):
            raise ValueError("Queue acceptance identity hash differs")
        if self.identity is None:
            self.identity = record["identity"]
        elif self.identity != record["identity"]:
            raise ValueError(
                "Queue lifetime scope, approval, inputs or loading paths changed"
            )
        for key in map(tuple, record["task_keys"]):
            owner = self.key_owners.get(key)
            if owner is not None and owner != entry_sha:
                raise ValueError(
                    "A task key was already accepted by another queue batch"
                )
            self.key_owners[key] = entry_sha
        existing = self.accepted.get(entry_sha)
        if existing is not None and existing != record:
            raise ValueError("One entry identity has conflicting immutable acceptances")
        self.accepted[entry_sha] = record

    def admit(self, path):
        entry_ref, entry, plan = read_stable_entry(path)
        manifests = dispatcher.admit_plan(plan)
        identities = [shared_identity(manifest) for _, manifest in manifests]
        identity = identities[0]
        if any(value != identity for value in identities):
            raise ValueError(
                "One batch contains different scope or backend loading paths"
            )
        if identity["software_test_only"] is not self.software_test_only:
            raise PermissionError("Queue and task execution modes differ")
        if Path(identity["gpu_lock_root"]).resolve() != self.lock_root:
            raise PermissionError(
                "Queue GPU owner lock root differs from the task plan"
            )
        if self.identity is not None and self.identity != identity:
            raise ValueError(
                "Queue lifetime scope, approval, inputs or loading paths changed"
            )
        keys = []
        for _, manifest in manifests:
            if (
                manifest.get("schema") != worker.MANIFEST_SCHEMA
                or manifest.get("stage") != "aggregation"
            ):
                raise ValueError("Only frozen aggregation-v3 shards are accepted")
            worker.load_worker(
                manifest,
                software_test_only=self.software_test_only,
                injected_factory=self.fixture_factory,
            )
            keys.extend(
                map(
                    tuple,
                    worker.read_json(worker.checked_ref(manifest["task_keys_ref"])),
                )
            )
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate semantic aggregation keys in one batch")
        entry_sha = entry_ref["sha256"]
        for key in keys:
            owner = self.key_owners.get(key)
            if owner is not None and owner != entry_sha:
                raise ValueError(
                    "A task key was already accepted by another queue batch"
                )
        existing = self.accepted.get(entry_sha)
        if existing is not None:
            if (
                existing["entry_ref"] != entry_ref
                or existing["plan_ref"] != entry["plan_ref"]
                or existing["task_keys"] != list(map(list, keys))
                or existing["identity"] != identity
            ):
                raise ValueError("Published entry or plan changed after acceptance")
            return entry_sha, entry, plan
        record = dict(
            schema="e18_aggregation_queue_acceptance_v1",
            entry_ref=entry_ref,
            plan_ref=entry["plan_ref"],
            identity=identity,
            identity_sha256=digest(identity),
            task_keys=list(map(list, keys)),
            task_count=len(keys),
            metadata_only=self.metadata_only,
            software_test_only=self.software_test_only,
            accepted_at_unix=time.time(),
            daemon_source_sha256=worker.file_sha(__file__),
        )
        publish_once(self.state / (entry_sha + ".accepted.json"), record)
        self._register(record)
        return entry_sha, entry, plan

    def run(self, *, max_batches=None, sleep_seconds=15, max_idle_rounds=None):
        count = 0
        idle = 0
        while True:
            changed = False
            for path in sorted(
                self.root.glob("*.ready.json"), key=lambda value: value.name
            ):
                entry_ref, _, _ = read_stable_entry(path)
                entry_sha = entry_ref["sha256"]
                old = [
                    record
                    for record in self.accepted.values()
                    if record["entry_ref"]["path"] == entry_ref["path"]
                ]
                if old and old[0]["entry_ref"] != entry_ref:
                    raise ValueError("A published ready file was overwritten")
                if entry_sha in self.completed:
                    if self.accepted[entry_sha]["entry_ref"] != entry_ref:
                        raise ValueError(
                            "One immutable entry was republished under another filename"
                        )
                    continue
                entry_sha, entry, plan = self.admit(path)
                if self.metadata_only:
                    result = dict(status="METADATA_ONLY", actual_model_calls=0)
                else:
                    result = self.runner(
                        plan,
                        entry["plan_ref"]["sha256"],
                        software_test_only=self.software_test_only,
                    )
                    if result.get("status") != "DONE":
                        return dict(
                            status=result.get("status", "INCOMPLETE"),
                            completed_batches=count,
                            task_keys_reserved=len(self.key_owners),
                            batch_result=result,
                        )
                accepted_path = self.state / (entry_sha + ".accepted.json")
                completion = dict(
                    schema="e18_aggregation_queue_completion_v1",
                    status=result["status"],
                    plan_ref=entry["plan_ref"],
                    entry_sha256=entry_sha,
                    acceptance_sha256=worker.file_sha(accepted_path),
                    result=result,
                    metadata_only=self.metadata_only,
                    software_test_only=self.software_test_only,
                    finished_at_unix=time.time(),
                    daemon_source_sha256=worker.file_sha(__file__),
                )
                publish_once(self._completion_path(entry_sha), completion)
                self.completed.add(entry_sha)
                count += 1
                changed = True
                print(
                    json.dumps(
                        dict(
                            event="BATCH_" + result["status"],
                            entry=path.name,
                            batch_tasks=self.accepted[entry_sha]["task_count"],
                            completed_lifetime_batches=len(self.completed),
                        )
                    ),
                    flush=True,
                )
                if max_batches is not None and count >= max_batches:
                    return dict(
                        status="METADATA_ONLY"
                        if self.metadata_only
                        else "BATCH_LIMIT_REACHED",
                        completed_batches=count,
                        task_keys_reserved=len(self.key_owners),
                    )
            marker = self.root / "ALL_DONE"
            if marker.exists():
                if not marker.is_file():
                    raise ValueError(
                        "ALL_DONE must be an explicit coordinator marker file"
                    )
                return dict(
                    status="METADATA_ONLY" if self.metadata_only else "DONE",
                    completion_basis="ROOT_ALL_DONE_AND_ALL_PUBLISHED_BATCHES_COMPLETED",
                    all_done_marker_sha256=worker.file_sha(marker),
                    completed_batches=count,
                    completed_lifetime_batches=len(self.completed),
                    task_keys_reserved=len(self.key_owners),
                )
            idle = 0 if changed else idle + 1
            if max_idle_rounds is not None and idle >= max_idle_rounds:
                return dict(
                    status="WAITING",
                    completed_batches=count,
                    task_keys_reserved=len(self.key_owners),
                )
            time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-lock-root", type=Path, required=True)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if not args.gpu.isdigit() or (
        args.max_batches is not None and args.max_batches < 1
    ):
        raise ValueError("Explicit GPU and positive optional batch limit required")
    queue = AggregationQueue(
        args.queue_root, args.gpu_lock_root, metadata_only=args.metadata_only
    )
    queue_lock = dispatcher.ShardClaim(queue.root / "QUEUE_CONSUMER.lock")
    if not queue_lock.acquire():
        raise RuntimeError("Another consumer already owns this queue")
    try:
        if args.metadata_only:
            result = queue.run(max_batches=args.max_batches)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            owner_sha = digest(
                dict(
                    queue_root=str(queue.root),
                    gpu=args.gpu,
                    daemon_source_sha256=worker.file_sha(__file__),
                )
            )
            with worker.GPUOwnership(args.gpu_lock_root, args.gpu, owner_sha):
                result = queue.run(max_batches=args.max_batches)
        print(json.dumps(result), flush=True)
    finally:
        queue_lock.release()


if __name__ == "__main__":
    main()
