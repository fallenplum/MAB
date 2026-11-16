"""Two-owner graceful checkpoint broker for the original E18 Granite tasks.

The default CLI is metadata-only. Only --execute can signal the two explicitly
authorized former processes. Native validation and storage helpers are reused
from the unchanged member4 broker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import member4_checkpoint_broker as original


BASE_PLAN_SHA = original.BASE_PLAN_SHA
ORIGINAL_BROKER_SHA = "b2dc379cb05cb10bd66979fbca38a6920f2af70ba3f8f4a944d343228b7a304b"
FORMER_METADATA_SHA = "8266b25b8c8ee56f54635c2d41ffc8a418b0f337164883842063cd8288fe6237"
RUNNER_SHA = "2650c784e848886276608893ecbc412e4e67e9d336520238d20bb1c078393783"
CHILD_WORKER_SHA = "2f2f6424e65e39f6a42dc0ba8e2268fd0b63b6b1b20d5fab501c1da53e9cdb40"
TARGETS = ((3274323, "0"), (3934277, "3"))
ORIGINAL_TASKS = 8005
PARENT_COUNTS = (3909, 4096)
MAX_CHILD_TASKS = 512

read_json = original.read_json
file_sha = original.file_sha
digest = original.digest
write_once = original.write_once
write_atomic = original.write_atomic
checked_ref = original.checked_ref
validate_checkpoint_owner = original.validate_checkpoint_owner
LinuxProcesses = original.LinuxProcesses


def command_sha(command):
    return hashlib.sha256(b"\0".join(item.encode() for item in command) + b"\0").hexdigest()


def argument(command, name):
    if command.count(name) != 1:
        return None
    index = command.index(name)
    return command[index + 1] if index + 1 < len(command) else None


class CheckpointBroker(original.CheckpointBroker):
    def __init__(
        self, plan_path, plan_sha, dispatcher, processes, lease_root, child_plan_path,
        *, former_metadata_ref, targets=TARGETS, owner_pid=None,
        expected_tasks=ORIGINAL_TASKS, expected_parent_counts=PARENT_COUNTS,
        expected_runner_sha=RUNNER_SHA, expected_child_worker_sha=CHILD_WORKER_SHA,
        clock=time.monotonic, sleeper=time.sleep, poll_seconds=2,
        timeout_seconds=1800, child_timeout_seconds=172800, validator=None, scope_sha=None,
    ):
        if file_sha(original.__file__) != ORIGINAL_BROKER_SHA:
            raise ValueError("Original checkpoint validation source changed")
        self.plan_path = Path(plan_path).resolve(strict=True)
        self.plan_sha = plan_sha
        if file_sha(self.plan_path) != plan_sha:
            raise ValueError("Original dispatch-plan SHA differs")
        self.dispatcher, self.worker = dispatcher, dispatcher.worker
        self.manifests = dispatcher.admit_plan(read_json(self.plan_path))
        self.parents = [item for item in self.manifests if item[1].get("member_index") == 2]
        if len(self.manifests) != 12 or len(self.parents) != 2 or any(
            item[1].get("stage") != "member_generation" for item in self.manifests
        ):
            raise ValueError("Twelve original manifests and two Granite parents are required")
        base = self.parents[0][1]
        self.parent_keys = []
        counts = []
        for _, manifest in self.parents:
            if any(manifest.get(key) != base.get(key) for key in (
                "scope_ref", "approval_ref", "question_records_ref", "owned_output_root", "adapter_ref", "asset_root", "backend",
            )):
                raise ValueError("Original parent scientific/runtime sources differ")
            keys = read_json(checked_ref(manifest["task_keys_ref"]))
            counts.append(len(keys))
            self.parent_keys.extend(tuple(key) for key in keys)
        if sorted(counts) != sorted(expected_parent_counts) or len(self.parent_keys) != expected_tasks or (
            len(set(self.parent_keys)) != expected_tasks
        ) or any(len(key) != 3 or key[2] != 2 for key in self.parent_keys):
            raise ValueError("Original Granite task-key union differs")
        self.output_root = Path(base["owned_output_root"]).resolve()
        self.state_root = self.output_root / "_dispatcher/plans" / plan_sha
        self.root = Path(lease_root).resolve()
        if self.root != self.output_root / "_dispatcher/member2_checkpoint":
            raise ValueError("The member2_checkpoint ownership namespace is required")
        self.child_plan_path = Path(child_plan_path).resolve()
        self.expected_child_worker_sha = expected_child_worker_sha
        self.processes = processes
        self.targets = tuple((int(pid), str(gpu)) for pid, gpu in targets)
        if len(self.targets) != 2 or len({pid for pid, _ in self.targets}) != 2:
            raise ValueError("Two distinct former process identities are required")
        self.owner_pid = os.getpid() if owner_pid is None else owner_pid
        if type(self.owner_pid) is not int or self.owner_pid <= 0 or self.owner_pid in dict(self.targets):
            raise ValueError("Broker and former process identities must be distinct")
        self.clock, self.sleeper = clock, sleeper
        self.poll_seconds, self.timeout = poll_seconds, timeout_seconds
        self.child_timeout = child_timeout_seconds
        if min(poll_seconds, timeout_seconds, child_timeout_seconds) <= 0:
            raise ValueError("Positive checkpoint wait bounds are required")
        self.validator, self.scope_sha = validator, scope_sha
        self.approval_sha = base["approval_ref"]["sha256"]
        self.held = {}
        self.stop_requested = False
        self.stage = "INITIALIZED"
        self.signalled = set()
        self.former_identities = {}
        self.active_parents = {}
        self.metadata_ref = dict(former_metadata_ref)
        metadata = read_json(checked_ref(self.metadata_ref))
        if metadata.get("schema") != "e18_member2_former_workers_live_metadata_v1" or (
            metadata.get("original_plan_sha256") != plan_sha
        ):
            raise ValueError("Original two-former metadata identity differs")
        rows = metadata.get("rows", [])
        self.snapshots = {row["pid"]: row for row in rows}
        if len(rows) != 2 or set(self.snapshots) != set(dict(self.targets)):
            raise ValueError("Former metadata must identify exactly the authorized pair")
        parent_refs = {ref["sha256"]: ref for ref, _ in self.parents}
        snapshot_parents = set()
        for index, (pid, gpu) in enumerate(self.targets):
            row = self.snapshots[pid]
            command = row["cmdline"]
            if command_sha(command) != row["cmdline_sha256"] or row["verified_claim_lock_owner"] != pid:
                raise ValueError("Former metadata command or claim digest differs")
            if argument(command, "--plan-sha256") != plan_sha or argument(command, "--gpu") != gpu:
                raise ValueError("Former metadata plan/GPU differs")
            if Path(argument(command, "--plan") or "").resolve() != self.plan_path:
                raise ValueError("Former metadata plan path differs")
            reference = row["parent_manifest_ref"]
            if parent_refs.get(reference["sha256"]) != reference or (
                Path(row["claim_path"]).resolve() != self.claim_path(reference)
            ):
                raise ValueError("Former metadata parent claim differs")
            snapshot_parents.add(reference["sha256"])
            scripts = [Path(value).resolve() for value in command if value.endswith(".py")]
            if index == 0:
                if scripts != [Path(dispatcher.__file__).resolve()]:
                    raise ValueError("First former must run the original dispatcher")
            elif len(scripts) != 1 or scripts[0].name != "run_original_plan_shard_v2.py" or (
                file_sha(scripts[0]) != expected_runner_sha
            ) or argument(command, "--manifest-sha256") != reference["sha256"] or (
                argument(command, "--member-index") != "2"
            ) or command.count("--resume-stopped") != 1 or argument(command, "--previous-pid") != "3667022":
                raise ValueError("Second former must match the pinned stopped-prefix runner")
        if len(snapshot_parents) != 2:
            raise ValueError("Both original Granite parents must be separately owned")

    def heartbeat(self):
        if self.root.exists():
            write_atomic(self.root / "HEARTBEAT.json", dict(
                schema="e18_member2_checkpoint_heartbeat_v1", owner_pid=self.owner_pid,
                former_pids=[pid for pid, _ in self.targets], state=self.stage,
                at_unix=time.time(), held_parent_claims=sorted(self.held),
            ))

    def same_former_identity(self, pid):
        observed = self.processes.identity(pid)
        if observed is not None:
            snapshot = self.snapshots[pid]
            if observed["start_ticks"] != snapshot["proc_start_ticks"] or (
                observed["command"] != snapshot["cmdline"]
            ) or command_sha(observed["command"]) != snapshot["cmdline_sha256"]:
                raise PermissionError("Former PID/start/complete command identity changed")
        return observed

    def inspect_target(self):
        if (self.state_root / "STOP_REQUIRES_RECONCILIATION.json").exists():
            raise RuntimeError("The original plan requires reconciliation")
        for pid, _ in self.targets:
            identity = self.same_former_identity(pid)
            if identity is None:
                raise PermissionError("An authorized former process has already exited")
            active = []
            for reference, manifest in self.manifests:
                path = self.status_path(reference)
                status = read_json(path) if path.exists() else {}
                if status.get("status") == "FAILED":
                    raise RuntimeError("An original parent is FAILED")
                if status.get("status") == "RUNNING" and status.get("pid") == pid:
                    if status.get("manifest_sha256") != reference["sha256"] or manifest.get("member_index") != 2:
                        raise PermissionError("Former RUNNING status identity differs")
                    active.append((reference, manifest))
            expected = self.snapshots[pid]["parent_manifest_ref"]["sha256"]
            if len(active) != 1 or active[0][0]["sha256"] != expected or (
                self.processes.claim_owner(self.claim_path(active[0][0])) != pid
            ):
                raise PermissionError("Former process lacks its exact RUNNING parent FLOCK")
            self.former_identities[pid] = identity
            self.active_parents[pid] = active[0]
        return dict(
            status="DRY_RUN_METADATA_VALID", base_member_plan_sha256=self.plan_sha,
            former_pids=[pid for pid, _ in self.targets], target_member_index=2,
            former_metadata_ref=self.metadata_ref, original_member2_tasks=len(self.parent_keys),
            process_signals_sent=0, model_calls=0, gold_reads=0,
            parent_shards=[dict(manifest_sha256=ref["sha256"], claim_path=str(self.claim_path(ref))) for ref, _ in self.parents],
        )

    def assert_parent_claims(self):
        for reference, _ in self.parents:
            if reference["sha256"] not in self.held or (
                self.processes.claim_owner(self.claim_path(reference)) != self.owner_pid
            ):
                raise RuntimeError("Broker must retain both original Granite parent claims")

    def checkpoint_stop(self):
        self.hold_available()
        before = {}
        for pid, _ in self.targets:
            reference, _ = self.active_parents[pid]
            identity = self.same_former_identity(pid)
            if identity is None or not identity["sigterm_caught"] or (
                self.processes.claim_owner(self.claim_path(reference)) != pid
            ):
                raise PermissionError("Both original workers must still own caught-SIGTERM checkpoints")
            before[pid] = original.summary_signature(self.output_root / "_dispatcher" / reference["sha256"] / "SUMMARY.json")
        evidence = {}
        # The targeted stopped-prefix runner goes first. Its parent must be
        # leased before signalling the broad dispatcher, which could otherwise
        # claim that newly released Granite shard.
        for pid, _ in reversed(self.targets):
            reference, _ = self.active_parents[pid]
            active_sha = reference["sha256"]
            summary_path = self.output_root / "_dispatcher" / active_sha / "SUMMARY.json"
            if pid == self.targets[0][0]:
                other_ref = self.active_parents[self.targets[1][0]][0]
                if other_ref["sha256"] not in self.held or self.processes.claim_owner(self.claim_path(other_ref)) != self.owner_pid:
                    raise RuntimeError("Targeted runner parent must be leased before the broad dispatcher stop")
            identity = self.same_former_identity(pid)
            if identity is None:
                self.hold_available()
                if active_sha not in self.held:
                    raise RuntimeError("Exited former parent has another owner")
                evidence[pid] = dict(former_pid=pid, kind="PROCESS_EXIT_AND_NATIVE_COMMIT_CONFIRMED", summary_file_sha256=None)
            else:
                if not identity["sigterm_caught"] or self.processes.claim_owner(self.claim_path(reference)) != pid:
                    raise PermissionError("Former ownership changed before graceful signal")
                self.stage = "CHECKPOINT_REQUESTED"
                self.event("FIRST_SIGTERM_REQUESTED", target_pid=pid)
                self.signalled.add(pid)
                try:
                    self.processes.terminate(pid)
                except ProcessLookupError:
                    self.event("TARGET_EXITED_AT_FIRST_SIGNAL_WINDOW", target_pid=pid)
            second_signalled = [False]

            def ready():
                self.hold_available()
                observed = self.same_former_identity(pid)
                if active_sha not in self.held:
                    return False
                fresh = original.summary_signature(summary_path)
                summary = read_json(summary_path) if fresh is not None and fresh != before[pid] else {}
                valid = summary.get("stop_requested") is True and summary.get("manifest_sha256") == active_sha and (
                    summary.get("stage") == "member_generation" and summary.get("profile") == 2
                    and summary.get("software_test_only") is False and summary.get("failures") == []
                )
                if observed is None:
                    evidence[pid] = dict(
                        former_pid=pid, kind="SUMMARY_CONFIRMED" if valid else "PROCESS_EXIT_AND_NATIVE_COMMIT_CONFIRMED",
                        summary_file_sha256=file_sha(summary_path) if valid else None,
                    )
                    return True
                if not valid or observed["sigterm_caught"] or second_signalled[0]:
                    return False
                if any(self.processes.claim_owner(self.claim_path(ref)) == pid for ref, _ in self.manifests):
                    raise RuntimeError("Former acquired another parent; no blind second signal")
                self.event("SECOND_SIGTERM_AT_QUIESCENT_OWNER", target_pid=pid)
                second_signalled[0] = True
                try:
                    self.processes.terminate(pid)
                except ProcessLookupError:
                    pass
                return False

            self.deadline_wait(ready, "Former worker has not reached a safe exited checkpoint")
            self.deadline_wait(lambda pid=pid: not self.processes.gpu_context_present(pid), "Former GPU context remains")
        self.assert_parent_claims()
        parent_shas = {ref["sha256"] for ref, _ in self.parents}
        for key in list(self.held):
            if key not in parent_shas:
                self.held.pop(key).release()
        self.stage = "FORMERS_EXITED_PARENT_CLAIMS_HELD"
        result = [evidence[pid] for pid, _ in self.targets]
        self.event("BOTH_FORMERS_AND_GPU_CONTEXTS_RELEASED", evidence=result)
        return result

    def write_lease(self, evidence):
        self.assert_parent_claims()
        audit, remaining = self.audit_tasks()
        remaining_path = self.root / "remaining_task_keys.json"
        write_once(remaining_path, remaining)
        audit_path = self.root / "CHECKPOINT_SOURCE_AUDIT.json"
        write_once(audit_path, dict(
            schema="e18_member2_native_checkpoint_audit_v1", tasks=audit, original_tasks=len(audit),
            checkpoint_successes=len(audit) - len(remaining), pending_or_unknown_attempts=0,
        ))
        lease = dict(
            schema="e18_member2_checkpoint_lease_v1", status="LEASED", base_member_plan_sha256=self.plan_sha,
            execution_scope_sha256=self.scope_sha, approval_sha256=self.approval_sha, target_member_index=2,
            owner_pid=self.owner_pid, former_pids=[pid for pid, _ in self.targets], former_metadata_ref=self.metadata_ref,
            parent_shards=[dict(manifest_sha256=ref["sha256"], claim_path=str(self.claim_path(ref)),
                                task_keys_sha256=manifest["task_keys_ref"]["sha256"]) for ref, manifest in self.parents],
            remaining_task_keys_ref=dict(path=str(remaining_path), sha256=file_sha(remaining_path)),
            original_member2_tasks=len(self.parent_keys), checkpoint_successes=len(audit) - len(remaining),
            remaining_count=len(remaining), checkpoint_stop_verified=True, pending_or_unknown_attempts=0,
            checkpoint_evidence=evidence, checkpoint_source_audit_sha256=file_sha(audit_path), software_test_only=False,
        )
        write_once(self.root / "LEASE.json", lease)
        self.stage = "LEASED_WAITING_FOR_CHILD_PLAN"
        self.heartbeat()
        self.event("STATIC_LEASE_COMMITTED", lease_sha256=file_sha(self.root / "LEASE.json"), remaining=len(remaining))
        return lease, remaining

    def admit_child_plan(self, remaining):
        self.assert_parent_claims()
        plan_sha = file_sha(self.child_plan_path)
        plan = read_json(self.child_plan_path)
        lease_ref = dict(path=str(self.root / "LEASE.json"), sha256=file_sha(self.root / "LEASE.json"))
        if plan.get("schema") != "e18_member2_dispatch_plan_v1" or plan.get("checkpoint_lease_ref") != lease_ref or (
            plan.get("worker_source_sha256") != self.expected_child_worker_sha or not plan.get("manifest_refs")
        ):
            raise ValueError("Child plan must bind the exact lease and frozen member2 worker")
        base = self.parents[0][1]
        seen, refs = set(), []
        for reference in plan["manifest_refs"]:
            manifest = read_json(checked_ref(reference))
            if manifest.get("schema") != "e18_member2_worker_manifest_v1" or manifest.get("checkpoint_lease_ref") != lease_ref or (
                manifest.get("member_index") != 2 or manifest.get("stage") != "member_generation" or manifest.get("software_test_only") is not False
            ):
                raise ValueError("Child manifest checkpoint/profile/mode differs")
            if any(manifest.get(key) != base.get(key) for key in (
                "scope_ref", "approval_ref", "question_records_ref", "owned_output_root", "adapter_ref", "asset_root", "backend",
            )):
                raise ValueError("Child manifest changes an original scientific/backend source or owner")
            rows = read_json(checked_ref(manifest["task_keys_ref"]))
            keys = {tuple(key) for key in rows}
            if not 1 <= len(rows) <= MAX_CHILD_TASKS or len(keys) != len(rows) or seen & keys:
                raise ValueError("Child chunks must be disjoint nonempty sets of at most512 tasks")
            seen.update(keys)
            refs.append(reference)
        if seen != {tuple(key) for key in remaining}:
            raise ValueError("Child plan is not the exact remaining task union")
        binding = dict(schema="e18_member2_checkpoint_child_plan_admission_v1", checkpoint_lease_ref=lease_ref,
                       child_plan_ref=dict(path=str(self.child_plan_path), sha256=plan_sha),
                       worker_source_sha256=self.expected_child_worker_sha, manifest_refs=refs, remaining_count=len(remaining))
        path = self.root / "CHILD_PLAN_ADMISSION.json"
        if path.exists():
            if read_json(path) != binding:
                raise ValueError("Checkpoint child plan binding changed")
        else:
            write_once(path, binding)
        return plan_sha, refs

    def wait_for_completion(self, lease, remaining, completion_delay):
        child_sha = None
        if remaining:
            self.deadline_wait(lambda: self.child_plan_path.is_file(), "Root has not supplied the child plan", timeout_seconds=self.child_timeout)
            child_sha, refs = self.admit_child_plan(remaining)
            state = self.output_root / "_dispatcher/plans" / child_sha
            self.stage = "LEASED_CHILD_PLAN_RUNNING"

            def children_done():
                self.assert_parent_claims()
                if file_sha(self.child_plan_path) != child_sha:
                    raise ValueError("Child plan changed after lease admission")
                if (state / "STOP_REQUIRES_RECONCILIATION.json").exists():
                    raise RuntimeError("Child plan requires reconciliation")
                statuses = []
                for reference in refs:
                    checked_ref(reference)
                    path = state / (reference["sha256"] + ".status.json")
                    status = read_json(path) if path.exists() else {}
                    summary = status.get("summary", {})
                    if status.get("status") == "FAILED" or summary.get("failures") or summary.get("stop_requested"):
                        raise RuntimeError("A child shard failed or requested a stop")
                    statuses.append(status.get("status") == "DONE" and status.get("manifest_sha256") == reference["sha256"])
                return all(statuses)

            self.deadline_wait(children_done, "Child plan has not completed", timeout_seconds=self.child_timeout)
        self.assert_parent_claims()
        if (self.state_root / "STOP_REQUIRES_RECONCILIATION.json").exists():
            raise RuntimeError("The original plan requires reconciliation before parent completion")
        audit, _ = self.audit_tasks(require_all=True)
        if child_sha is not None:
            if file_sha(self.child_plan_path) != child_sha:
                raise ValueError("Child plan changed during the final native audit")
            for reference in refs:
                checked_ref(reference)
        complete = dict(schema="e18_member2_checkpoint_completion_v1", status="COMPLETED", owner_pid=self.owner_pid,
                        lease_sha256=file_sha(self.root / "LEASE.json"), child_plan_sha256=child_sha,
                        original_tasks_verified=len(audit), checkpoint_successes=lease["checkpoint_successes"],
                        new_remaining_tasks=lease["remaining_count"], native_source_audit_sha256=digest(audit), at_unix=time.time())
        for reference, manifest in self.parents:
            write_atomic(self.status_path(reference), dict(
                status="DONE", manifest_sha256=reference["sha256"],
                completion_kind="CHECKPOINT_CHILD_PLAN_AND_ALL_NATIVE_COMMITS_VERIFIED",
                checkpoint_lease_sha256=complete["lease_sha256"], child_plan_sha256=child_sha,
                verified_native_tasks=len(read_json(checked_ref(manifest["task_keys_ref"])))),
            )
        write_once(self.root / "LEASE_COMPLETED.json", complete)
        self.stage = "COMPLETED_HOLDING_PARENT_CLAIMS"
        self.event("ALL_ORIGINAL_MEMBER2_NATIVE_COMMITS_VERIFIED", tasks=len(audit))
        deadline = self.clock() + completion_delay
        while self.clock() < deadline and not self.stop_requested:
            self.heartbeat()
            self.sleeper(min(self.poll_seconds, deadline - self.clock()))
        return complete

    def run(self, *, completion_delay=60):
        if self.validator is None or not isinstance(self.scope_sha, str) or len(self.scope_sha) != 64:
            raise ValueError("Native dependency validator and canonical execution scope are required")
        self.inspect_target()
        self.root.mkdir(parents=True, exist_ok=False)
        self.event("BROKER_STARTED", owner_pid=self.owner_pid, former_pids=[pid for pid, _ in self.targets])
        try:
            evidence = self.checkpoint_stop()
            lease, remaining = self.write_lease(evidence)
            return self.wait_for_completion(lease, remaining, completion_delay)
        except BaseException as error:
            self.stage = "STOP_REQUIRES_RECONCILIATION"
            failure = dict(schema="e18_member2_checkpoint_stop_v1", status=self.stage,
                           error_type=type(error).__name__, error=str(error), owner_pid=self.owner_pid,
                           former_pids=[pid for pid, _ in self.targets], at_unix=time.time())
            write_atomic(self.root / "STOP_REQUIRES_RECONCILIATION.json", failure)
            if self.signalled:
                stop = self.state_root / "STOP_REQUIRES_RECONCILIATION.json"
                if not stop.exists():
                    write_once(stop, failure)
            self.heartbeat()
            raise
        finally:
            for claim in self.held.values():
                claim.release()
            self.held.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", default=BASE_PLAN_SHA)
    parser.add_argument("--dispatcher", type=Path, default=Path(__file__).with_name("e18_gpu_dispatcher.py"))
    parser.add_argument("--former-metadata", type=Path, required=True)
    parser.add_argument("--former-metadata-sha256", default=FORMER_METADATA_SHA)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--child-plan", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--child-timeout-seconds", type=float, default=172800)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.plan_sha256 != BASE_PLAN_SHA or args.former_metadata_sha256 != FORMER_METADATA_SHA:
        raise PermissionError("The original plan and authorized former-process snapshot are required")
    dispatcher = original.load_dispatcher(args.dispatcher)
    broker = CheckpointBroker(
        args.plan, args.plan_sha256, dispatcher, LinuxProcesses(), args.lease_root, args.child_plan,
        former_metadata_ref=dict(path=str(args.former_metadata), sha256=args.former_metadata_sha256),
        poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds, child_timeout_seconds=args.child_timeout_seconds,
    )
    if not args.execute:
        print(json.dumps(broker.inspect_target(), ensure_ascii=False, indent=2))
        return
    module, adapter, _, _, _, _ = dispatcher.worker.load_worker(broker.parents[0][1], software_test_only=False)
    broker.scope_sha = module.canonical_digest(adapter.scope)
    broker.validator = adapter.checked_member_dependency
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: setattr(broker, "stop_requested", True))
    try:
        result = broker.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
