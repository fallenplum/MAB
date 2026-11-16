"""Claim-fenced subdivision of one paused E18 member4 execution shard.

This operational layer retains the original 5ac9 worker, adapter, upstream
checkpoint lease, native owners and scientific requests. It never sends process
signals, constructs a model itself, reads Gold, or changes the evaluated action
set. A worker may finish its already-entered call after a broker dies, exactly as
with the upstream lease. Every subsequent call checks both lease levels.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

import member4_checkpoint_broker as base


WORKER_SHA = "5ac9a17dd07d282bfa2cceaff3dd062ce837e68885b4086816ef509974af5786"
DISPATCHER_SHA = "b097a47f67c3116f5635b16d26dcb8db366aac24fde06afb925d362c45c97327"
BROKER_SOURCE_SHA = "b2dc379cb05cb10bd66979fbca38a6920f2af70ba3f8f4a944d343228b7a304b"
PARENT_PLAN_SHA = "a627f3094c742ec0d352582a9abe8b6338af6eaf0967333ebf30f45727c97166"
TARGET_MANIFEST_SHA = "826e72af7b84e48d0421417eab3c1ed9c8907633fa88fdd756229c3bdc7fe323"
LEASE_SCHEMA = "e18_member4_tail_partition_lease_v1"
ADDED_MANIFEST_FIELDS = {"tail_parent_lease_ref", "tail_partition_index"}


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=base.file_sha(path))


def load_runtime(path):
    """Load a single original dispatcher/worker module pair."""
    path = Path(path).resolve(strict=True)
    if (
        base.file_sha(path) != DISPATCHER_SHA
        or base.file_sha(path.with_name("e18_member4_worker_v3.py")) != WORKER_SHA
        or base.file_sha(base.__file__) != BROKER_SOURCE_SHA
    ):
        raise ValueError("Frozen operational dependency bytes differ")
    sys.path.insert(0, str(path.parent))
    name = "e18_member4_dispatcher_v3"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ValueError("Another dispatcher module is already imported")
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _keys(reference):
    rows = base.read_json(base.checked_ref(reference))
    keys = [tuple(row) for row in rows]
    if (
        not keys
        or len(keys) != len(set(keys))
        or any(len(key) != 3 or key[2] != 4 for key in keys)
    ):
        raise ValueError("Unique nonempty member4 execution keys are required")
    return keys


def assert_tail_active(lease, worker, *, software_test_only=False, processes=None):
    """Validate the additional parent claim without weakening the upstream one."""
    if lease.get("schema") != LEASE_SCHEMA or lease.get("status") != "LEASED":
        raise PermissionError("A live tail-partition lease is required")
    if lease.get("software_test_only") is not software_test_only:
        raise PermissionError("Tail lease execution mode differs")
    processes = processes or base.LinuxProcesses()
    identity = processes.identity(lease["owner_pid"])
    if identity is None or str(identity["start_ticks"]) != str(lease["owner_start_ticks"]):
        raise PermissionError("Tail broker died or its PID was reused")
    if processes.claim_owner(lease["parent_claim_path"]) != lease["owner_pid"]:
        raise PermissionError("Tail broker lost the original child claim")
    if processes.identity(lease["former_pid"]) is not None:
        raise PermissionError("The original shard process is still active")
    if Path(lease["stop_path"]).exists():
        raise RuntimeError("Tail partition requires explicit reconciliation")
    upstream = base.read_json(base.checked_ref(lease["upstream_checkpoint_lease_ref"]))
    worker.assert_lease_active(upstream, software_test_only=software_test_only)


def admit_fine_plan(
    plan,
    plan_sha,
    dispatcher,
    *,
    expected_lease_ref=None,
    software_test_only=False,
    processes=None,
    require_live_parent=True,
):
    """Validate exact remaining-key coverage and unchanged request inputs."""
    if not isinstance(plan_sha, str) or len(plan_sha) != 64:
        raise ValueError("An explicit fine-plan file SHA is required")
    manifests = dispatcher.admit_plan(plan)
    lease_ref = plan.get("tail_parent_lease_ref")
    if expected_lease_ref is not None and lease_ref != expected_lease_ref:
        raise ValueError("Fine plan binds a different tail lease")
    lease = base.read_json(base.checked_ref(lease_ref))
    parent = base.read_json(base.checked_ref(lease["parent_manifest_ref"]))
    parent_keys = _keys(parent["task_keys_ref"])
    remaining = _keys(lease["remaining_task_keys_ref"])
    expected_claim = (
        Path(parent["owned_output_root"]).resolve()
        / "_dispatcher/plans"
        / lease["parent_plan_ref"]["sha256"]
        / (lease["parent_manifest_ref"]["sha256"] + ".claim")
    )
    if (
        lease.get("parent_plan_ref") != plan.get("tail_parent_plan_ref")
        or lease.get("parent_manifest_ref") != plan.get("tail_parent_manifest_ref")
        or plan.get("checkpoint_lease_ref") != parent.get("checkpoint_lease_ref")
        or lease.get("upstream_checkpoint_lease_ref") != parent.get("checkpoint_lease_ref")
        or len(parent_keys) != lease.get("original_parent_tasks")
        or len(remaining) != lease.get("remaining_count")
        or lease.get("checkpoint_successes", -1) + len(remaining) != len(parent_keys)
        or not set(remaining) <= set(parent_keys)
        or plan.get("worker_source_sha256") != base.file_sha(dispatcher.worker.__file__)
        or Path(lease.get("parent_claim_path", "")).resolve() != expected_claim
        or type(lease.get("owner_pid")) is not int
        or lease["owner_pid"] <= 0
        or type(lease.get("former_pid")) is not int
        or lease["former_pid"] <= 0
        or lease["former_pid"] == lease["owner_pid"]
    ):
        raise ValueError("Tail plan, parent and exact checkpoint identity differ")
    parent_plan = base.read_json(base.checked_ref(lease["parent_plan_ref"]))
    if lease["parent_manifest_ref"] not in parent_plan.get("manifest_refs", []):
        raise ValueError("Parent manifest is outside the original member4 plan")
    dispatcher.admit_plan(parent_plan)
    scope = base.read_json(base.checked_ref(parent["scope_ref"]))
    upstream = base.read_json(base.checked_ref(parent["checkpoint_lease_ref"]))
    upstream_remaining = set(_keys(upstream["remaining_task_keys_ref"]))
    if (
        lease.get("execution_scope_sha256") != base.digest(scope)
        or lease.get("approval_sha256") != parent["approval_ref"]["sha256"]
        or lease.get("pending_or_unknown_attempts") != 0
        or not 1 <= lease.get("maximum_child_keys", 0) <= 64
        or not set(parent_keys) <= upstream_remaining
    ):
        raise ValueError("Tail scope or upstream remaining-key membership differs")
    base.checked_ref(lease["controller_source_ref"])
    audit = base.read_json(base.checked_ref(lease["checkpoint_source_audit_ref"]))
    if (
        audit.get("original_tasks") != len(parent_keys)
        or audit.get("checkpoint_successes") != lease["checkpoint_successes"]
        or audit.get("pending_or_unknown_attempts") != 0
    ):
        raise ValueError("Tail native checkpoint audit differs")
    audit_rows = audit.get("tasks", [])
    by_hash = {row["task_key_sha256"]: row for row in audit_rows}
    parent_hashes = {base.digest(list(key)) for key in parent_keys}
    remaining_hashes = {base.digest(list(key)) for key in remaining}
    if (
        len(by_hash) != len(audit_rows)
        or set(by_hash) != parent_hashes
        or {key for key, value in by_hash.items() if value.get("status") == "UNEXECUTED"} != remaining_hashes
        or any(value.get("status") not in {"SUCCESS", "UNEXECUTED"} for value in by_hash.values())
        or sum(value["status"] == "SUCCESS" for value in by_hash.values()) != lease["checkpoint_successes"]
    ):
        raise ValueError("Checkpoint audit is not the exact parent SUCCESS/unexecuted partition")
    seen = set()
    for index, (_, manifest) in enumerate(manifests):
        if (
            manifest.get("tail_parent_lease_ref") != lease_ref
            or manifest.get("tail_partition_index") != index
            or manifest.get("software_test_only") is not software_test_only
            or set(manifest) != set(parent) | ADDED_MANIFEST_FIELDS
            or any(manifest[name] != value for name, value in parent.items() if name != "task_keys_ref")
        ):
            raise ValueError("A fine manifest changes the original scientific request")
        keys = _keys(manifest["task_keys_ref"])
        if len(keys) > lease["maximum_child_keys"] or seen.intersection(keys):
            raise ValueError("Fine child size or disjointness differs")
        seen.update(keys)
    if seen != set(remaining):
        raise ValueError("Fine plan is not the exact unexecuted-key partition")
    if require_live_parent:
        assert_tail_active(lease, dispatcher.worker, software_test_only=software_test_only, processes=processes)
    return manifests, lease


@contextmanager
def nested_call_guard(worker, lease, *, lease_ref=None, software_test_only=False, processes=None):
    """Add parent liveness to the original per-task guard, then restore it."""
    original = worker.assert_lease_active
    lease_ref = lease_ref or ref(Path(lease["stop_path"]).parent / "LEASE.json")
    if base.read_json(base.checked_ref(lease_ref)) != lease:
        raise ValueError("Tail lease mapping differs from its frozen file")

    def guard(upstream, *, software_test_only=False):
        base.checked_ref(lease_ref)
        base.checked_ref(lease["controller_source_ref"])
        original(upstream, software_test_only=software_test_only)
        # assert_tail_active also checks upstream, using the original function.
        worker.assert_lease_active = original
        try:
            assert_tail_active(lease, worker, software_test_only=software_test_only, processes=processes)
        finally:
            worker.assert_lease_active = guard

    worker.assert_lease_active = guard
    try:
        yield
    finally:
        worker.assert_lease_active = original


def execute_fine_plan(
    plan,
    plan_sha,
    dispatcher,
    *,
    software_test_only=False,
    max_idle_rounds=None,
    processes=None,
    execute=None,
):
    manifests, lease = admit_fine_plan(
        plan, plan_sha, dispatcher, software_test_only=software_test_only, processes=processes
    )
    allowed = {reference["sha256"] for reference, _ in manifests}
    original_execute = dispatcher.worker.execute_manifest if execute is None else execute
    if not software_test_only and execute is not None:
        raise PermissionError("Production cannot inject an execution callback")

    def guarded_execute(manifest, checksum, *, software_test_only=False):
        if checksum not in allowed:
            raise ValueError("Original dispatcher requested an unadmitted fine child")
        with nested_call_guard(
            dispatcher.worker,
            lease,
            lease_ref=manifest["tail_parent_lease_ref"],
            software_test_only=software_test_only,
            processes=processes,
        ):
            return original_execute(manifest, checksum, software_test_only=software_test_only)

    return dispatcher.run_plan(
        plan,
        plan_sha,
        software_test_only=software_test_only,
        execute=guarded_execute,
        max_idle_rounds=max_idle_rounds,
    )


class TailPartitionBroker:
    """Hold one original child claim until all its native commits are audited."""

    def __init__(
        self,
        parent_plan_path,
        parent_plan_sha,
        parent_manifest_sha,
        dispatcher,
        processes,
        lease_root,
        fine_root,
        fine_plan_path,
        *,
        former_pid,
        former_gpu,
        validator,
        scope_sha,
        chunk_size=64,
        expected_tasks=512,
        owner_pid=None,
        software_test_only=False,
        clock=time.monotonic,
        sleeper=time.sleep,
        poll_seconds=0.2,
        timeout_seconds=1800,
        completion_timeout_seconds=172800,
    ):
        if not 1 <= chunk_size <= 64:
            raise ValueError("Fine execution chunks must contain at most64 keys")
        self.plan_path = Path(parent_plan_path).resolve()
        if base.file_sha(self.plan_path) != parent_plan_sha:
            raise ValueError("Original member4 plan bytes differ")
        self.plan_sha = parent_plan_sha
        self.dispatcher, self.worker, self.processes = dispatcher, dispatcher.worker, processes
        matches = [item for item in dispatcher.admit_plan(base.read_json(self.plan_path)) if item[0]["sha256"] == parent_manifest_sha]
        if len(matches) != 1:
            raise ValueError("Exactly one original child must be subdivided")
        self.parent_ref, self.parent = matches[0]
        self.keys = _keys(self.parent["task_keys_ref"])
        if len(self.keys) != expected_tasks or self.parent.get("member_index") != 4:
            raise ValueError("Original parent key count or member profile differs")
        self.output_root = Path(self.parent["owned_output_root"]).resolve()
        self.state_root = self.output_root / "_dispatcher/plans" / parent_plan_sha
        self.status_path = self.state_root / (parent_manifest_sha + ".status.json")
        self.claim_path = self.state_root / (parent_manifest_sha + ".claim")
        self.root, self.fine_root, self.fine_plan_path = map(lambda p: Path(p).resolve(), (lease_root, fine_root, fine_plan_path))
        if not self.root.is_relative_to(self.output_root):
            raise ValueError("Tail control records must remain inside approved owned output")
        if not all(path.is_relative_to(self.output_root.parent) for path in (self.fine_root, self.fine_plan_path)):
            raise ValueError("Fine plan and keys must remain inside the experiment directory")
        self.former_pid, self.former_gpu = former_pid, str(former_gpu)
        self.owner_pid = os.getpid() if owner_pid is None else owner_pid
        self.validator, self.scope_sha = validator, scope_sha
        if scope_sha != base.digest(base.read_json(base.checked_ref(self.parent["scope_ref"]))):
            raise ValueError("Broker execution-scope identity differs")
        self.approval_sha = self.parent["approval_ref"]["sha256"]
        self.chunk_size, self.software_test_only = chunk_size, software_test_only
        self.clock, self.sleep, self.poll = clock, sleeper, poll_seconds
        self.timeout, self.completion_timeout = timeout_seconds, completion_timeout_seconds
        self.claim = dispatcher.ShardClaim(self.claim_path)
        self.held = False
        self.stage = "INITIALIZED"
        self.stop_requested = False
        self.former_identity = None

    def upstream_active(self):
        upstream = base.read_json(base.checked_ref(self.parent["checkpoint_lease_ref"]))
        self.worker.assert_lease_active(upstream, software_test_only=self.software_test_only)
        if self.held and self.processes.claim_owner(self.claim_path) != self.owner_pid:
            raise PermissionError("Tail broker lost its parent claim")

    def inspect(self):
        self.upstream_active()
        identity = self.processes.identity(self.former_pid)
        status = base.read_json(self.status_path)
        command = identity["command"] if identity else []
        def argument(name):
            return command[command.index(name) + 1] if name in command else None
        if (
            identity is None
            or str(Path(self.dispatcher.__file__).resolve()) not in command
            or argument("--plan") != str(self.plan_path)
            or argument("--plan-sha256") != self.plan_sha
            or argument("--gpu") != self.former_gpu
            or status.get("status") != "RUNNING"
            or status.get("pid") != self.former_pid
            or self.processes.claim_owner(self.claim_path) != self.former_pid
        ):
            raise PermissionError("Former child process/claim identity differs")
        self.former_identity = identity
        return dict(
            status="TAIL_PARENT_METADATA_ADMITTED",
            parent_tasks=len(self.keys),
            parent_manifest_ref=self.parent_ref,
            process_signals_sent=0,
            gold_reads=0,
            model_calls=0,
        )

    def heartbeat(self):
        base.write_atomic(
            self.root / "HEARTBEAT.json",
            dict(
                schema="e18_member4_tail_partition_heartbeat_v1",
                owner_pid=self.owner_pid,
                stage=self.stage,
                parent_claim_held=self.held,
                at_unix=time.time(),
            ),
        )

    def wait(self, predicate, message, timeout):
        deadline = self.clock() + timeout
        while True:
            if self.stop_requested:
                raise RuntimeError("Tail broker stop requested")
            self.upstream_active()
            if predicate():
                return
            if self.clock() >= deadline:
                raise TimeoutError(message)
            self.heartbeat()
            self.sleep(self.poll)

    def acquire_checkpoint(self):
        self.stage = "WAITING_FOR_ROOT_SAFE_PAUSE_AND_PARENT_CLAIM"
        def acquire():
            self.held = self.claim.acquire()
            return self.held
        self.wait(acquire, "The original child claim was not released", self.timeout)
        def former_gone():
            identity = self.processes.identity(self.former_pid)
            if identity is not None and identity["start_ticks"] != self.former_identity["start_ticks"]:
                raise PermissionError("Former PID was reused")
            return identity is None and not self.processes.gpu_context_present(self.former_pid)
        self.wait(former_gone, "Former child process/GPU context remains", self.timeout)
        status = base.read_json(self.status_path)
        if status.get("status") not in {"PAUSED", "DONE"}:
            raise RuntimeError("Original child has no confirmed paused/completed state")
        if self.processes.claim_owner(self.claim_path) != self.owner_pid:
            raise PermissionError("Tail broker does not own its parent FLOCK")
        self.stage = "PARENT_CLAIM_HELD_FORMER_EXITED"

    def audit(self, require_all=False):
        audit, remaining = [], []
        physical_attempts = set()
        for key in self.keys:
            owner = self.output_root / self.scope_sha / "member_generation" / base.digest(list(key))
            row = base.validate_checkpoint_owner(owner, key, self.scope_sha, self.approval_sha, self.validator)
            if row["status"] == "SUCCESS":
                attempt = base.read_json(owner / "LAST_TASK_RECEIPT.json")["physical_attempt_id"]
                if attempt in physical_attempts:
                    raise ValueError("One physical attempt is assigned to multiple original keys")
                physical_attempts.add(attempt)
            audit.append(row)
            if row["status"] == "UNEXECUTED":
                remaining.append(list(key))
        if require_all and remaining:
            raise RuntimeError("Fine DONE flags do not prove all original native commits")
        return audit, remaining

    def partition(self):
        audit, remaining = self.audit()
        base.write_once(self.root / "remaining_task_keys.json", remaining)
        audit_path = self.root / "CHECKPOINT_SOURCE_AUDIT.json"
        base.write_once(
            audit_path,
            dict(
                schema="e18_tail_native_checkpoint_audit_v1",
                tasks=audit,
                original_tasks=len(self.keys),
                checkpoint_successes=len(self.keys) - len(remaining),
                pending_or_unknown_attempts=0,
            ),
        )
        identity = self.processes.identity(self.owner_pid)
        if identity is None:
            raise PermissionError("Tail broker process is absent")
        lease = dict(
            schema=LEASE_SCHEMA,
            status="LEASED",
            controller_source_ref=ref(__file__),
            owner_pid=self.owner_pid,
            owner_start_ticks=identity["start_ticks"],
            former_pid=self.former_pid,
            former_start_ticks=self.former_identity["start_ticks"],
            parent_plan_ref=ref(self.plan_path),
            parent_manifest_ref=self.parent_ref,
            parent_claim_path=str(self.claim_path),
            upstream_checkpoint_lease_ref=self.parent["checkpoint_lease_ref"],
            remaining_task_keys_ref=ref(self.root / "remaining_task_keys.json"),
            checkpoint_source_audit_ref=ref(audit_path),
            original_parent_tasks=len(self.keys),
            checkpoint_successes=len(self.keys) - len(remaining),
            remaining_count=len(remaining),
            pending_or_unknown_attempts=0,
            maximum_child_keys=self.chunk_size,
            execution_scope_sha256=self.scope_sha,
            approval_sha256=self.approval_sha,
            stop_path=str(self.root / "STOP_REQUIRES_RECONCILIATION.json"),
            software_test_only=self.software_test_only,
        )
        base.write_once(self.root / "LEASE.json", lease)
        lease_ref = ref(self.root / "LEASE.json")
        manifests = []
        self.fine_root.mkdir(parents=True, exist_ok=False)
        for index, offset in enumerate(range(0, len(remaining), self.chunk_size)):
            key_path = self.fine_root / f"member4_tail_{index:03d}.keys.json"
            base.write_once(key_path, remaining[offset:offset + self.chunk_size])
            manifest = copy.deepcopy(self.parent)
            manifest.update(task_keys_ref=ref(key_path), tail_parent_lease_ref=lease_ref, tail_partition_index=index)
            manifest_path = self.fine_root / f"member4_tail_{index:03d}.manifest.json"
            base.write_once(manifest_path, manifest)
            manifests.append(ref(manifest_path))
        plan = dict(
            schema=self.dispatcher.PLAN_SCHEMA,
            worker_source_sha256=base.file_sha(self.worker.__file__),
            checkpoint_lease_ref=self.parent["checkpoint_lease_ref"],
            tail_parent_lease_ref=lease_ref,
            tail_parent_plan_ref=ref(self.plan_path),
            tail_parent_manifest_ref=self.parent_ref,
            manifest_refs=manifests,
        )
        base.write_once(self.fine_plan_path, plan)
        if remaining:
            admit_fine_plan(
                plan,
                base.file_sha(self.fine_plan_path),
                self.dispatcher,
                expected_lease_ref=lease_ref,
                software_test_only=self.software_test_only,
                processes=self.processes,
            )
        base.write_once(
            self.root / "FINE_PLAN_ADMISSION.json",
            dict(
                schema="e18_tail_fine_plan_admission_v1",
                plan_ref=ref(self.fine_plan_path),
                parent_tasks=len(self.keys),
                checkpoint_successes=lease["checkpoint_successes"],
                remaining_count=len(remaining),
                children=len(manifests),
                added_scientific_keys=0,
                worker_source_sha256=base.file_sha(self.worker.__file__),
                gold_reads=0,
                model_calls=0,
            ),
        )
        self.stage = "FINE_PLAN_RUNNING_PARENT_CLAIM_HELD"
        return plan, lease

    def finish(self, plan, lease):
        plan_ref = ref(self.fine_plan_path)
        child_state = self.output_root / "_dispatcher/plans" / plan_ref["sha256"]
        def done():
            base.checked_ref(plan_ref)
            if (child_state / "STOP_REQUIRES_RECONCILIATION.json").exists():
                raise RuntimeError("A fine execution plan stopped")
            values = []
            for reference in plan["manifest_refs"]:
                path = child_state / (reference["sha256"] + ".status.json")
                value = base.read_json(path) if path.exists() else {}
                if value.get("status") == "FAILED":
                    raise RuntimeError("A fine execution child failed")
                values.append(value.get("status") == "DONE" and value.get("manifest_sha256") == reference["sha256"])
            return all(values)
        self.wait(done, "Fine native execution did not complete", self.completion_timeout)
        audit, _ = self.audit(require_all=True)
        complete = dict(
            schema="e18_member4_tail_partition_completion_v1",
            status="COMPLETED",
            parent_manifest_ref=self.parent_ref,
            fine_plan_ref=plan_ref,
            tail_lease_ref=ref(self.root / "LEASE.json"),
            original_native_tasks_verified=len(audit),
            native_audit_sha256=base.digest(audit),
            at_unix=time.time(),
        )
        base.write_once(self.root / "COMPLETED.json", complete)
        base.write_atomic(
            self.status_path,
            dict(
                status="DONE",
                manifest_sha256=self.parent_ref["sha256"],
                completion_kind="FINE_PARTITION_AND_ALL_ORIGINAL_NATIVE_COMMITS_VERIFIED",
                tail_completion_ref=ref(self.root / "COMPLETED.json"),
                summary=dict(planned=len(self.keys), completed=len(self.keys), failures=[], stop_requested=False),
                finished_at_unix=time.time(),
            ),
        )
        self.stage = "COMPLETED"
        self.heartbeat()
        return complete

    def run(self):
        self.inspect()
        self.root.mkdir(parents=True, exist_ok=False)
        base.write_once(
            self.root / "START_RECEIPT.json",
            dict(
                schema="e18_tail_partition_start_v1",
                parent_plan_ref=ref(self.plan_path),
                parent_manifest_ref=self.parent_ref,
                former_pid=self.former_pid,
                former_start_ticks=self.former_identity["start_ticks"],
                owner_pid=self.owner_pid,
                source_sha256=base.file_sha(__file__),
                signals_sent=0,
                model_calls=0,
                gold_reads=0,
            ),
        )
        try:
            self.acquire_checkpoint()
            plan, lease = self.partition()
            return self.finish(plan, lease)
        except BaseException as error:
            self.stage = "STOP_REQUIRES_RECONCILIATION"
            base.write_atomic(
                self.root / "STOP_REQUIRES_RECONCILIATION.json",
                dict(status=self.stage, error_type=type(error).__name__, error=str(error), at_unix=time.time(), owner_pid=self.owner_pid),
            )
            if self.held:
                base.write_atomic(
                    self.status_path,
                    dict(status="FAILED", manifest_sha256=self.parent_ref["sha256"], error_type=type(error).__name__, tail_reconciliation_root=str(self.root), at_unix=time.time()),
                )
            raise
        finally:
            if self.held:
                self.claim.release()
                self.held = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    broker = sub.add_parser("broker")
    broker.add_argument("--parent-plan", type=Path, required=True)
    broker.add_argument("--parent-plan-sha256", default=PARENT_PLAN_SHA)
    broker.add_argument("--parent-manifest-sha256", default=TARGET_MANIFEST_SHA)
    broker.add_argument("--former-pid", type=int, required=True)
    broker.add_argument("--former-gpu", required=True)
    broker.add_argument("--lease-root", type=Path, required=True)
    broker.add_argument("--fine-root", type=Path, required=True)
    broker.add_argument("--fine-plan", type=Path, required=True)
    broker.add_argument("--chunk-size", type=int, default=64)
    broker.add_argument("--execute", action="store_true")
    fine = sub.add_parser("fine")
    fine.add_argument("--plan", type=Path, required=True)
    fine.add_argument("--plan-sha256", required=True)
    fine.add_argument("--gpu", required=True)
    fine.add_argument("--gpu-slot", type=int, choices=(0, 1), default=0)
    fine.add_argument("--resource-policy", type=Path, required=True)
    fine.add_argument("--resource-policy-sha256", required=True)
    parser.add_argument("--dispatcher", type=Path, default=Path(__file__).with_name("e18_member4_dispatcher_v3.py"))
    args = parser.parse_args()
    dispatcher = load_runtime(args.dispatcher)
    if args.operation == "fine":
        if base.file_sha(args.plan) != args.plan_sha256:
            raise ValueError("Fine plan bytes differ")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        plan = base.read_json(args.plan)
        manifests, _ = admit_fine_plan(plan, args.plan_sha256, dispatcher)
        with dispatcher.worker.SlotAdmission(
            manifests[0][1],
            args.plan_sha256,
            args.gpu,
            args.gpu_slot,
            dict(path=str(args.resource_policy), sha256=args.resource_policy_sha256),
        ):
            result = execute_fine_plan(plan, args.plan_sha256, dispatcher)
    else:
        if args.parent_plan_sha256 != PARENT_PLAN_SHA or args.parent_manifest_sha256 != TARGET_MANIFEST_SHA:
            raise PermissionError("Production subdivides only the approved original long child")
        manifests = dispatcher.admit_plan(base.read_json(args.parent_plan))
        parent = next(value for reference, value in manifests if reference["sha256"] == args.parent_manifest_sha256)
        module, adapter, _, _, _, _ = dispatcher.worker.load_worker(parent, software_test_only=False)
        item = TailPartitionBroker(
            args.parent_plan,
            args.parent_plan_sha256,
            args.parent_manifest_sha256,
            dispatcher,
            base.LinuxProcesses(),
            args.lease_root,
            args.fine_root,
            args.fine_plan,
            former_pid=args.former_pid,
            former_gpu=args.former_gpu,
            validator=adapter.checked_member_dependency,
            scope_sha=module.canonical_digest(adapter.scope),
            chunk_size=args.chunk_size,
        )
        if not args.execute:
            result = item.inspect()
        else:
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, lambda *_: setattr(item, "stop_requested", True))
            result = item.run()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
