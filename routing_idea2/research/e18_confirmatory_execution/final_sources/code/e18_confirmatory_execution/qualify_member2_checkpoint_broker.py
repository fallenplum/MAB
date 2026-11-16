"""Synthetic process/claim qualifications for the two-former Granite broker."""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import member2_checkpoint_broker as broker
from qualify_member4_checkpoint_broker import MockClock


class Fixture:
    def __init__(self, root, counts=(3, 2)):
        self.root = Path(root)
        self.output = self.root / "owned"
        self.owner_pid = 9001
        self.targets = broker.TARGETS
        self.clock = MockClock()
        self.scope_sha, self.approval_sha = "c" * 64, "a" * 64
        self.locks = {}
        self.validator_calls = []
        self.code = self.root / "e18_gpu_dispatcher.py"
        self.runner = self.root / "run_original_plan_shard_v2.py"
        self.code.write_text("# synthetic dispatcher\n", encoding="utf-8")
        self.runner.write_text("# synthetic stopped-prefix runner\n", encoding="utf-8")
        self.manifests, self.keys = [], []
        for index, profile in enumerate([4, 3, 2, 0, 1, 5, 4, 3, 2, 0, 1, 5]):
            count = {2: counts[0], 8: counts[1]}.get(index, 1)
            keys = [[f"synthetic-{index}-{j}", 42, profile] for j in range(count)]
            task_path = self.root / f"tasks-{index}.json"
            broker.write_once(task_path, keys)
            manifest = dict(
                member_index=profile, stage="member_generation", software_test_only=False,
                scope_ref=dict(path=str(self.root / "scope.json"), sha256=self.scope_sha),
                approval_ref=dict(path=str(self.root / "approval.json"), sha256=self.approval_sha),
                question_records_ref=dict(path=str(self.root / "inputs.jsonl"), sha256="b" * 64),
                adapter_ref=dict(path=str(self.root / "adapter.py"), sha256="d" * 64),
                asset_root=str(self.root / "assets"), backend=dict(mode="original"),
                owned_output_root=str(self.output),
                task_keys_ref=dict(path=str(task_path), sha256=broker.file_sha(task_path)),
            )
            path = self.root / f"manifest-{index}.json"
            broker.write_once(path, manifest)
            self.manifests.append((dict(path=str(path), sha256=broker.file_sha(path)), manifest))
            if profile == 2:
                self.keys.extend(map(tuple, keys))
        self.plan_path = self.root / "base_plan.json"
        broker.write_once(self.plan_path, dict(manifest_refs=[ref for ref, _ in self.manifests]))
        self.plan_sha = broker.file_sha(self.plan_path)
        self.state = self.output / "_dispatcher/plans" / self.plan_sha
        self.state.mkdir(parents=True)
        self.parent_items = [self.manifests[2], self.manifests[8]]
        self.metadata = dict(schema="e18_member2_former_workers_live_metadata_v1", original_plan_sha256=self.plan_sha, rows=[])
        for index, ((pid, gpu), (reference, _)) in enumerate(zip(self.targets, self.parent_items)):
            command = ["python", "-B", "-u", str(self.code if index == 0 else self.runner),
                       "--plan", str(self.plan_path), "--plan-sha256", self.plan_sha, "--gpu", gpu]
            if index == 1:
                command += ["--manifest-sha256", reference["sha256"], "--member-index", "2",
                            "--gpu-lock-root", str(self.root / "gpu-locks/member2_resume_gpu3"),
                            "--resume-stopped", "--previous-pid", "3667022"]
            claim = self.state / (reference["sha256"] + ".claim")
            status = dict(status="RUNNING", manifest_sha256=reference["sha256"], pid=pid, software_test_only=False)
            broker.write_once(self.state / (reference["sha256"] + ".status.json"), status)
            self.locks[str(claim)] = pid
            self.metadata["rows"].append(dict(
                pid=pid, cmdline=command, cmdline_sha256=broker.command_sha(command), proc_start_ticks=str(1000 + index),
                parent_manifest_ref=reference, original_status=status, claim_path=str(claim), verified_claim_lock_owner=pid,
            ))
        self.metadata_path = self.root / "former_metadata.json"
        broker.write_once(self.metadata_path, self.metadata)
        self.metadata_ref = dict(path=str(self.metadata_path), sha256=broker.file_sha(self.metadata_path))
        self.foreign_claim = self.state / (self.manifests[1][0]["sha256"] + ".claim")
        self.locks[str(self.foreign_claim)] = 8001
        fixture = self

        class Claim:
            def __init__(self, path):
                self.path, self.held = Path(path), False

            def acquire(self):
                if str(self.path) in fixture.locks:
                    return False
                fixture.locks[str(self.path)] = fixture.owner_pid
                self.held = True
                return True

            def release(self):
                if self.held:
                    fixture.locks.pop(str(self.path), None)
                    self.held = False

        module = SimpleNamespace(canonical_digest=lambda _: self.scope_sha)
        adapter = SimpleNamespace(scope={}, checked_member_dependency=self.validate)
        self.dispatcher = SimpleNamespace(
            __file__=str(self.code), ShardClaim=Claim, admit_plan=lambda _: self.manifests,
            worker=SimpleNamespace(load_worker=lambda *_, **__: (module, adapter, None, None, None, None)),
        )
        self.processes = MockProcesses(self)
        self.instance = broker.CheckpointBroker(
            self.plan_path, self.plan_sha, self.dispatcher, self.processes,
            self.output / "_dispatcher/member2_checkpoint", self.root / "child_plan.json",
            former_metadata_ref=self.metadata_ref, owner_pid=self.owner_pid,
            expected_tasks=sum(counts), expected_parent_counts=counts, expected_runner_sha=broker.file_sha(self.runner),
            clock=self.clock.now, sleeper=self.clock.sleep, poll_seconds=1, timeout_seconds=3, child_timeout_seconds=4,
            validator=self.validate, scope_sha=self.scope_sha,
        )
        self.make_success(self.keys[0])

    def validate(self, *values):
        self.validator_calls.append(values[:3])
        return values[3]

    def owner(self, key):
        return self.output / self.scope_sha / "member_generation" / broker.digest(list(key))

    def make_success(self, key):
        owner = self.owner(key)
        owner.mkdir(parents=True, exist_ok=True)
        native = dict(status="success", cost_status="recorded_actual_usage", job_id=broker.digest(["job", *key]),
                      attempt_id="synthetic-" + broker.digest(list(key)), result=dict(text="synthetic", prompt_tokens=1, completion_tokens=2))
        native["record_sha256"] = broker.digest(native)
        receipt = dict(schema="e18_execution_task_receipt_v2", status="SUCCESS", scope_sha256=self.scope_sha,
                       approval_sha256=self.approval_sha, stage="member_generation", software_test_only=False,
                       task_key=dict(question_id=key[0], generation_seed=key[1], member_index=key[2]), native_record=native,
                       native_record_sha256=native["record_sha256"], native_job_id=native["job_id"], physical_attempt_id=native["attempt_id"])
        fence = dict(status="SUCCESS_COMMITTED", scope_sha256=self.scope_sha, approval_sha256=self.approval_sha,
                     task_key=list(key), stage="member_generation", software_test_only=False,
                     native_job_id=native["job_id"], native_record_sha256=native["record_sha256"])
        broker.write_atomic(owner / "ATTEMPT_STATE.json", fence)
        broker.write_atomic(owner / "LAST_TASK_RECEIPT.json", receipt)
        (owner / "records.jsonl").write_text(json.dumps(native) + "\n", encoding="utf-8")

    def checkpoint(self):
        self.instance.inspect_target()
        self.instance.root.mkdir(parents=True)
        evidence = self.instance.checkpoint_stop()
        return self.instance.write_lease(evidence)

    def release(self):
        for claim in self.instance.held.values():
            claim.release()

    def create_children(self, *, populate=True, extra=False, duplicate=False, chunk_size=512, changed=None):
        item = self.instance
        lease_ref = dict(path=str(item.root / "LEASE.json"), sha256=broker.file_sha(item.root / "LEASE.json"))
        remaining = broker.read_json(item.root / "remaining_task_keys.json")
        keys = copy.deepcopy(remaining)
        if extra:
            keys.append(["synthetic-outside", 42, 2])
        if duplicate:
            keys.append(keys[0])
        references = []
        for index, start in enumerate(range(0, len(keys), chunk_size)):
            rows = keys[start:start + chunk_size]
            path = self.root / f"child_keys-{index}.json"
            broker.write_once(path, rows)
            manifest = dict(self.parent_items[0][1], schema="e18_member2_worker_manifest_v1", checkpoint_lease_ref=lease_ref,
                            gpu_lock_root=str(self.root / "gpu-locks/member2_checkpoint"),
                            task_keys_ref=dict(path=str(path), sha256=broker.file_sha(path)))
            if changed:
                manifest.update(changed)
            path = self.root / f"child_manifest-{index}.json"
            broker.write_once(path, manifest)
            references.append(dict(path=str(path), sha256=broker.file_sha(path)))
        broker.write_once(item.child_plan_path, dict(schema="e18_member2_dispatch_plan_v1", checkpoint_lease_ref=lease_ref,
                                                   worker_source_sha256=broker.CHILD_WORKER_SHA, manifest_refs=references))
        state = self.output / "_dispatcher/plans" / broker.file_sha(item.child_plan_path)
        state.mkdir(parents=True)
        for reference in references:
            broker.write_once(state / (reference["sha256"] + ".status.json"), dict(status="DONE", manifest_sha256=reference["sha256"]))
        if populate:
            for key in remaining:
                self.make_success(tuple(key))
        return references


class MockProcesses:
    def __init__(self, fixture):
        self.fixture = fixture
        self.signals = []
        self.states = {pid: dict(alive=True, caught=True, gpu=True, reused=False, mode="graceful", keep_gpu=False)
                       for pid, _ in fixture.targets}

    def identity(self, pid):
        state = self.states[pid]
        if not state["alive"]:
            return None
        row = next(row for row in self.fixture.metadata["rows"] if row["pid"] == pid)
        return dict(pid=pid, start_ticks="reused" if state["reused"] else row["proc_start_ticks"],
                    command=copy.deepcopy(row["cmdline"]), sigterm_caught=state["caught"])

    def claim_owner(self, path):
        return self.fixture.locks.get(str(path))

    def terminate(self, pid):
        state = self.states[pid]
        f = self.fixture
        self.signals.append(pid)
        first = self.signals.count(pid) == 1
        reference = next(row["parent_manifest_ref"] for row in f.metadata["rows"] if row["pid"] == pid)
        claim = f.state / (reference["sha256"] + ".claim")
        if first:
            if state["mode"] == "stuck":
                return
            f.locks.pop(str(claim), None)
            state["caught"] = False
            if state["mode"] == "exit_no_summary":
                state.update(alive=False, gpu=state["keep_gpu"])
                return
            path = f.output / "_dispatcher" / reference["sha256"] / "SUMMARY.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            broker.write_atomic(path, dict(stop_requested=True, manifest_sha256=reference["sha256"], stage="member_generation",
                                           profile=2, software_test_only=False, failures=[]))
            if state["mode"] == "pid_reused":
                state["reused"] = True
            if state["mode"] == "foreign_claim":
                f.locks[str(f.foreign_claim)] = pid
        else:
            if state["caught"] or pid in f.locks.values():
                raise AssertionError("Second signal would interrupt a live paid owner")
            state.update(alive=False, gpu=state["keep_gpu"])

    def gpu_context_present(self, pid):
        return self.states[pid]["gpu"]


class BrokerQualification(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2] / ".broker_toy"
        parent.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="m2_", dir=parent)
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(temporary.name)
        self.addCleanup(self.fixture.release)

    def test_dryrun_checks_both_without_signal_or_lease(self):
        f = self.fixture
        result = f.instance.inspect_target()
        self.assertEqual(result["former_pids"], [pid for pid, _ in f.targets])
        self.assertEqual(f.processes.signals, [])
        self.assertFalse(f.instance.root.exists())

    def test_second_runner_complete_command_change_rejected(self):
        f = self.fixture
        f.metadata["rows"][1]["cmdline"][-1] = "wrong-prior-pid"
        with self.assertRaises(PermissionError):
            f.instance.inspect_target()
        self.assertEqual(f.processes.signals, [])

    def test_each_start_tick_is_checked(self):
        f = self.fixture
        f.processes.states[f.targets[1][0]]["reused"] = True
        with self.assertRaises(PermissionError):
            f.instance.inspect_target()

    def test_exact_manifest_status_and_claim_are_required(self):
        f = self.fixture
        reference = f.parent_items[1][0]
        value = broker.read_json(f.instance.status_path(reference))
        value["manifest_sha256"] = "f" * 64
        broker.write_atomic(f.instance.status_path(reference), value)
        with self.assertRaises(PermissionError):
            f.instance.inspect_target()

    def test_both_first_signals_require_installed_graceful_handlers(self):
        f = self.fixture
        f.processes.states[f.targets[1][0]]["caught"] = False
        with self.assertRaises(PermissionError):
            f.instance.run(completion_delay=0)
        self.assertEqual(f.processes.signals, [])

    def test_two_graceful_workers_exit_and_two_claims_stay_held(self):
        f = self.fixture
        lease, remaining = f.checkpoint()
        self.assertEqual(sorted(f.processes.signals), sorted([pid for pid, _ in f.targets] * 2))
        self.assertEqual(len(f.instance.held), 2)
        self.assertEqual(f.locks[str(f.foreign_claim)], 8001)
        self.assertEqual((lease["target_member_index"], lease["original_member2_tasks"], len(remaining)), (2, 5, 4))
        self.assertEqual(lease["schema"], "e18_member2_checkpoint_lease_v1")

    def test_targeted_runner_exits_and_parent_is_leased_before_broad_signal(self):
        f = self.fixture
        broad, targeted = [pid for pid, _ in f.targets]
        original_terminate = f.processes.terminate

        def checked_terminate(pid):
            if pid == broad:
                self.assertFalse(f.processes.states[targeted]["alive"])
                self.assertFalse(f.processes.states[targeted]["gpu"])
                reference = f.parent_items[1][0]
                self.assertEqual(f.locks[str(f.instance.claim_path(reference))], f.owner_pid)
            original_terminate(pid)

        f.processes.terminate = checked_terminate
        f.checkpoint()
        self.assertEqual(f.processes.signals, [targeted, targeted, broad, broad])

    def test_broad_dispatcher_cannot_steal_targeted_released_parent(self):
        f = self.fixture
        broad, targeted = [pid for pid, _ in f.targets]
        original_terminate = f.processes.terminate
        steals = []

        def simulate_dispatcher_race(pid):
            original_terminate(pid)
            if pid == targeted and f.processes.signals.count(targeted) == 1 and not f.processes.states[broad]["caught"]:
                path = f.instance.claim_path(f.parent_items[1][0])
                if str(path) not in f.locks:
                    f.locks[str(path)] = broad
                    steals.append(str(path))

        f.processes.terminate = simulate_dispatcher_race
        f.checkpoint()
        self.assertEqual(steals, [])

    def test_process_exit_race_still_audits_native_commits(self):
        f = self.fixture
        for state in f.processes.states.values():
            state["mode"] = "exit_no_summary"
        lease, _ = f.checkpoint()
        self.assertEqual(len(f.processes.signals), 2)
        self.assertTrue(all(x["kind"] == "PROCESS_EXIT_AND_NATIVE_COMMIT_CONFIRMED" for x in lease["checkpoint_evidence"]))

    def test_stale_summary_never_authorizes_second_signal(self):
        f = self.fixture
        pid = f.targets[0][0]
        f.processes.states[pid]["mode"] = "stuck"
        reference = f.parent_items[0][0]
        path = f.output / "_dispatcher" / reference["sha256"] / "SUMMARY.json"
        path.parent.mkdir(parents=True)
        broker.write_once(path, dict(stop_requested=True, manifest_sha256=reference["sha256"], stage="member_generation",
                                    profile=2, software_test_only=False, failures=[]))
        with self.assertRaises(TimeoutError):
            f.instance.run(completion_delay=0)
        self.assertEqual(f.processes.signals.count(pid), 1)

    def test_pid_reuse_after_checkpoint_blocks_second_signal(self):
        f = self.fixture
        pid = f.targets[1][0]
        f.processes.states[pid]["mode"] = "pid_reused"
        with self.assertRaises(PermissionError):
            f.instance.run(completion_delay=0)
        self.assertEqual(f.processes.signals.count(pid), 1)

    def test_former_cannot_hold_another_claim_before_second_signal(self):
        f = self.fixture
        pid = f.targets[1][0]
        f.processes.states[pid]["mode"] = "foreign_claim"
        with self.assertRaises(RuntimeError):
            f.instance.run(completion_delay=0)
        self.assertEqual(f.processes.signals.count(pid), 1)

    def test_both_gpu_contexts_must_disappear(self):
        f = self.fixture
        f.processes.states[f.targets[1][0]].update(mode="exit_no_summary", keep_gpu=True)
        with self.assertRaises(TimeoutError):
            f.instance.run(completion_delay=0)
        self.assertFalse((f.instance.root / "LEASE.json").exists())

    def test_reserved_native_owner_blocks_lease_and_fences_original_plan(self):
        f = self.fixture
        owner = f.owner(f.keys[1])
        owner.mkdir(parents=True)
        broker.write_once(owner / "ATTEMPT_STATE.json", dict(status="RESERVED_BEFORE_BACKEND"))
        with self.assertRaises(RuntimeError):
            f.instance.run(completion_delay=0)
        self.assertFalse((f.instance.root / "LEASE.json").exists())
        self.assertTrue((f.state / "STOP_REQUIRES_RECONCILIATION.json").is_file())

    def test_failed_native_owner_blocks_lease(self):
        f = self.fixture
        path = f.owner(f.keys[0]) / "ATTEMPT_STATE.json"
        value = broker.read_json(path)
        value["status"] = "STOP_REQUIRES_RECONCILIATION"
        broker.write_atomic(path, value)
        with self.assertRaises(RuntimeError):
            f.instance.run(completion_delay=0)

    def test_unknown_native_usage_blocks_lease(self):
        f = self.fixture
        path = f.owner(f.keys[0]) / "LAST_TASK_RECEIPT.json"
        value = broker.read_json(path)
        value["native_record"]["cost_status"] = "unknown_must_reconcile"
        broker.write_atomic(path, value)
        with self.assertRaises(ValueError):
            f.instance.run(completion_delay=0)

    def test_extra_store_attempt_blocks_lease(self):
        f = self.fixture
        with (f.owner(f.keys[0]) / "records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"status":"failure"}\n')
        with self.assertRaises(RuntimeError):
            f.instance.run(completion_delay=0)

    def test_empty_owner_is_unexecuted_and_all_successes_use_validator(self):
        f = self.fixture
        owner = f.owner(f.keys[1])
        owner.mkdir(parents=True)
        (owner / "records.jsonl").touch()
        lease, remaining = f.checkpoint()
        self.assertEqual((lease["checkpoint_successes"], len(remaining)), (1, 4))
        self.assertEqual(f.validator_calls, [f.keys[0]])

    def test_static_lease_does_not_change_with_heartbeat(self):
        f = self.fixture
        f.checkpoint()
        before = broker.file_sha(f.instance.root / "LEASE.json")
        f.instance.heartbeat()
        self.assertEqual(before, broker.file_sha(f.instance.root / "LEASE.json"))

    def test_child_union_and_worker_sha_are_bound(self):
        f = self.fixture
        _, remaining = f.checkpoint()
        f.create_children(extra=True, populate=False)
        with self.assertRaises(ValueError):
            f.instance.admit_child_plan(remaining)

    def test_duplicate_child_keys_are_rejected(self):
        f = self.fixture
        _, remaining = f.checkpoint()
        f.create_children(duplicate=True, populate=False)
        with self.assertRaises(ValueError):
            f.instance.admit_child_plan(remaining)

    def test_child_backend_change_is_rejected(self):
        f = self.fixture
        _, remaining = f.checkpoint()
        f.create_children(changed=dict(backend=dict(mode="changed")), populate=False)
        with self.assertRaises(ValueError):
            f.instance.admit_child_plan(remaining)

    def test_child_asset_root_change_is_rejected(self):
        f = self.fixture
        _, remaining = f.checkpoint()
        f.create_children(changed=dict(asset_root="changed"), populate=False)
        with self.assertRaises(ValueError):
            f.instance.admit_child_plan(remaining)

    def test_child_plan_worker_sha_is_rejected_when_changed(self):
        f = self.fixture
        _, remaining = f.checkpoint()
        f.create_children(populate=False)
        value = broker.read_json(f.instance.child_plan_path)
        value["worker_source_sha256"] = "f" * 64
        broker.write_atomic(f.instance.child_plan_path, value)
        with self.assertRaises(ValueError):
            f.instance.admit_child_plan(remaining)

    def test_child_done_cannot_replace_all_native_successes(self):
        f = self.fixture
        lease, remaining = f.checkpoint()
        f.create_children(populate=False)
        with self.assertRaises(RuntimeError):
            f.instance.wait_for_completion(lease, remaining, 0)
        self.assertFalse((f.instance.root / "LEASE_COMPLETED.json").exists())
        self.assertTrue(all(broker.read_json(f.instance.status_path(ref))["status"] == "RUNNING" for ref, _ in f.parent_items))

    def test_native_completion_marks_both_original_parents_done(self):
        f = self.fixture
        lease, remaining = f.checkpoint()
        f.create_children()
        before = broker.file_sha(f.instance.root / "LEASE.json")
        result = f.instance.wait_for_completion(lease, remaining, 0)
        self.assertEqual(result["original_tasks_verified"], 5)
        self.assertTrue(all(broker.read_json(f.instance.status_path(ref))["status"] == "DONE" for ref, _ in f.parent_items))
        self.assertTrue((f.instance.root / "CHILD_PLAN_ADMISSION.json").is_file())
        self.assertEqual(before, broker.file_sha(f.instance.root / "LEASE.json"))

    def test_512_boundary_and_remaining_partition(self):
        with tempfile.TemporaryDirectory(prefix="m2_big_", dir=self.fixture.root.parent) as directory:
            f = Fixture(directory, counts=(514, 1))
            try:
                _, remaining = f.checkpoint()
                references = f.create_children(populate=False)
                self.assertEqual(len(references), 2)
                f.instance.admit_child_plan(remaining)
                manifest = broker.read_json(broker.checked_ref(references[0]))
                chunk = broker.read_json(broker.checked_ref(manifest["task_keys_ref"]))
                self.assertEqual(len(chunk), 512)
            finally:
                f.release()

    def test_513_chunk_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m2_big_", dir=self.fixture.root.parent) as directory:
            f = Fixture(directory, counts=(514, 1))
            try:
                _, remaining = f.checkpoint()
                f.create_children(populate=False, chunk_size=513)
                with self.assertRaises(ValueError):
                    f.instance.admit_child_plan(remaining)
            finally:
                f.release()

    def test_original_stop_prevents_all_signals(self):
        f = self.fixture
        broker.write_once(f.state / "STOP_REQUIRES_RECONCILIATION.json", dict(status="STOP"))
        with self.assertRaises(RuntimeError):
            f.instance.run(completion_delay=0)
        self.assertEqual(f.processes.signals, [])

    def test_full_run_waits_for_actual_child_refs(self):
        f = self.fixture
        def publish():
            if (f.instance.root / "LEASE.json").is_file() and not f.instance.child_plan_path.exists():
                f.create_children()
        f.clock.callback = publish
        result = f.instance.run(completion_delay=0)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(f.instance.held, {})

    def invoke_cli(self, execute):
        f = self.fixture
        args = ["member2_checkpoint_broker.py", "--plan", str(f.plan_path), "--plan-sha256", f.plan_sha,
                "--former-metadata", str(f.metadata_path), "--former-metadata-sha256", f.metadata_ref["sha256"],
                "--lease-root", str(f.instance.root), "--child-plan", str(f.instance.child_plan_path)]
        if execute:
            args.append("--execute")
        output = io.StringIO()
        with patch.object(broker, "BASE_PLAN_SHA", f.plan_sha), patch.object(broker, "FORMER_METADATA_SHA", f.metadata_ref["sha256"]), (
            patch.object(broker.original, "load_dispatcher", return_value=f.dispatcher)
        ), patch.object(broker, "LinuxProcesses", return_value=f.processes), (
            patch.object(broker, "CheckpointBroker", return_value=f.instance)
        ), patch.object(broker.os.sys, "argv", args), redirect_stdout(output):
            broker.main()
        return json.loads(output.getvalue())

    def test_cli_default_is_metadata_only(self):
        result = self.invoke_cli(False)
        self.assertEqual(result["status"], "DRY_RUN_METADATA_VALID")
        self.assertEqual(self.fixture.processes.signals, [])

    def test_cli_execute_is_qualified_with_mock_processes_only(self):
        f = self.fixture
        def publish():
            if (f.instance.root / "LEASE.json").is_file() and not f.instance.child_plan_path.exists():
                f.create_children()
        f.clock.callback = publish
        self.assertEqual(self.invoke_cli(True)["status"], "COMPLETED")
        self.assertEqual(len(f.processes.signals), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BrokerQualification))
    receipt = dict(
        schema="e18_member2_checkpoint_broker_mock_qualification_v1", status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, broker_source_sha256=broker.file_sha(broker.__file__), test_source_sha256=broker.file_sha(__file__),
        base_member_plan_sha256=broker.BASE_PLAN_SHA, authorized_former_metadata_sha256=broker.FORMER_METADATA_SHA,
        actual_process_signals=0, actual_model_calls=0, real_gold_reads=0, production_state_or_lease_files_written=0,
        software_test_only=True, control_backend="MockProcesses and MockClaims in temporary synthetic directories",
        wall_seconds=time.perf_counter() - started, test_output=output.getvalue(),
    )
    broker.write_once(args.output, receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "test_output"}, ensure_ascii=False, indent=2))
    if not result.wasSuccessful():
        print(output.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
