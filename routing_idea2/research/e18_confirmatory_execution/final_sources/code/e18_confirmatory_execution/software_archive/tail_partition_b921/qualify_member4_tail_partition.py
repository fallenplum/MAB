"""Pure synthetic qualifications for the E18 tail execution partition."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import member4_tail_partition as tail


class MockProcesses:
    def __init__(self, fixture):
        self.fixture = fixture
        self.former_alive = True
        self.owner_alive = True
        self.owner_reused = False
        self.claim_lost = False
        self.signals = []

    def identity(self, pid):
        f = self.fixture
        if pid == f.owner_pid:
            return dict(pid=pid, start_ticks="reused" if self.owner_reused else "owner-birth", command=[]) if self.owner_alive else None
        if pid == f.former_pid and self.former_alive:
            return dict(pid=pid, start_ticks="former-birth", command=["python", str(f.dispatcher.__file__), "--plan", str(f.plan_path), "--plan-sha256", tail.base.file_sha(f.plan_path), "--gpu", "7"])
        return None

    def claim_owner(self, path):
        f = self.fixture
        if Path(path) != f.broker.claim_path or self.claim_lost:
            return None
        if f.broker.held:
            return f.owner_pid
        return f.former_pid if self.former_alive else None

    def gpu_context_present(self, pid):
        return pid == self.fixture.former_pid and self.former_alive


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.output = self.root / "owned"
        self.owner_pid, self.former_pid = 9001, 7001
        self.keys = [[f"synthetic-question-{i}", 42, 4] for i in range(5)]
        self.dispatcher = tail.load_runtime(Path(tail.__file__).with_name("e18_member4_dispatcher_v3.py"))
        self.worker = self.dispatcher.worker
        self.upstream_alive = True
        self.upstream_checks = 0
        self.charges = 0
        self.patch = patch.object(self.worker, "assert_lease_active", self.upstream_guard)
        self.patch.start()
        self.scope_path = self.root / "scope.json"
        self.approval_path = self.root / "approval.json"
        self.upstream_path = self.root / "UPSTREAM_LEASE.json"
        self.parent_keys_path = self.root / "parent.keys.json"
        for path, value in ((self.scope_path, dict(required_members=self.keys)), (self.approval_path, dict(status="synthetic")), (self.parent_keys_path, self.keys)):
            tail.base.write_once(path, value)
        self.scope_sha = tail.base.digest(tail.base.read_json(self.scope_path))
        self.approval_sha = tail.base.file_sha(self.approval_path)
        tail.base.write_once(self.upstream_path, dict(schema="e18_member4_checkpoint_lease_v1", remaining_task_keys_ref=tail.ref(self.parent_keys_path)))
        self.parent = dict(schema=self.worker.MANIFEST_SCHEMA, stage="member_generation", member_index=4, software_test_only=True, scope_ref=tail.ref(self.scope_path), approval_ref=tail.ref(self.approval_path), checkpoint_lease_ref=tail.ref(self.upstream_path), task_keys_ref=tail.ref(self.parent_keys_path), question_records_ref=dict(path=str(self.root / "unused-inputs.jsonl"), sha256="b" * 64), adapter_ref=dict(path=str(self.root / "unused-adapter.py"), sha256="d" * 64), owned_output_root=str(self.output), gpu_lock_root=str(self.root / "gpu-locks"), backend=dict(mode="synthetic-test-only"))
        self.parent_path = self.root / "parent.manifest.json"
        tail.base.write_once(self.parent_path, self.parent)
        self.parent_ref = tail.ref(self.parent_path)
        self.plan_path = self.root / "ORIGINAL_PLAN.json"
        self.plan = dict(schema=self.dispatcher.PLAN_SCHEMA, worker_source_sha256=tail.base.file_sha(self.worker.__file__), checkpoint_lease_ref=tail.ref(self.upstream_path), manifest_refs=[self.parent_ref])
        tail.base.write_once(self.plan_path, self.plan)
        self.plan_sha = tail.base.file_sha(self.plan_path)
        self.processes = MockProcesses(self)
        self.broker = tail.TailPartitionBroker(self.plan_path, self.plan_sha, self.parent_ref["sha256"], self.dispatcher, self.processes, self.output / "_dispatcher/tail", self.root / "fine", self.root / "FINE_PLAN.json", former_pid=self.former_pid, former_gpu="7", validator=self.validate, scope_sha=self.scope_sha, chunk_size=2, expected_tasks=5, owner_pid=self.owner_pid, software_test_only=True, poll_seconds=0, timeout_seconds=0.001)
        self.broker.state_root.mkdir(parents=True)
        tail.base.write_once(self.broker.status_path, dict(status="RUNNING", pid=self.former_pid, manifest_sha256=self.parent_ref["sha256"]))
        self.make_success(self.keys[0])
        self.make_success(self.keys[1])

    def close(self):
        if self.broker.held:
            self.broker.claim.release()
            self.broker.held = False
        self.patch.stop()

    def upstream_guard(self, lease, *, software_test_only=False):
        self.upstream_checks += 1
        if not self.upstream_alive:
            raise PermissionError("Synthetic upstream broker died")

    def validate(self, question, seed, member, native):
        if member != 4 or native["job_id"] != tail.base.digest(["synthetic-job", question, seed, member]):
            raise ValueError("Synthetic job identity differs")
        return native

    def owner(self, key):
        return self.output / self.scope_sha / "member_generation" / tail.base.digest(key)

    def make_success(self, key):
        owner = self.owner(key)
        owner.mkdir(parents=True, exist_ok=True)
        native = dict(status="success", cost_status="recorded_actual_usage", job_id=tail.base.digest(["synthetic-job", *key]), attempt_id="synthetic-attempt-" + tail.base.digest(key), result=dict(text="synthetic output", prompt_tokens=2, completion_tokens=3))
        native["record_sha256"] = tail.base.digest(native)
        receipt = dict(schema="e18_execution_task_receipt_v2", status="SUCCESS", scope_sha256=self.scope_sha, approval_sha256=self.approval_sha, stage="member_generation", software_test_only=False, task_key=dict(question_id=key[0], generation_seed=key[1], member_index=4), native_record=native, native_record_sha256=native["record_sha256"], native_job_id=native["job_id"], physical_attempt_id=native["attempt_id"])
        fence = dict(status="SUCCESS_COMMITTED", scope_sha256=self.scope_sha, approval_sha256=self.approval_sha, task_key=key, stage="member_generation", software_test_only=False, native_job_id=native["job_id"], native_record_sha256=native["record_sha256"])
        tail.base.write_atomic(owner / "ATTEMPT_STATE.json", fence)
        tail.base.write_atomic(owner / "LAST_TASK_RECEIPT.json", receipt)
        (owner / "records.jsonl").write_text(json.dumps(native) + "\n", encoding="utf-8")

    def checkpoint(self):
        self.broker.inspect()
        self.broker.root.mkdir(parents=True)
        self.processes.former_alive = False
        tail.base.write_atomic(self.broker.status_path, dict(status="PAUSED", manifest_sha256=self.parent_ref["sha256"]))
        self.broker.acquire_checkpoint()
        self.fine_plan, self.lease = self.broker.partition()
        self.fine_sha = tail.base.file_sha(self.broker.fine_plan_path)
        return self.fine_plan, self.lease

    def fake_execute(self, manifest, checksum, *, software_test_only=False):
        keys = tail._keys(manifest["task_keys_ref"])
        cached = 0
        for key in keys:
            self.worker.assert_lease_active(tail.base.read_json(self.upstream_path), software_test_only=True)
            row = tail.base.validate_checkpoint_owner(self.owner(list(key)), key, self.scope_sha, self.approval_sha, self.validate)
            if row["status"] == "SUCCESS":
                cached += 1
            else:
                self.make_success(list(key))
                self.charges += 1
        return dict(planned=len(keys), completed=len(keys), cached=cached, failures=[], stop_requested=False)

    def execute(self):
        return tail.execute_fine_plan(self.fine_plan, self.fine_sha, self.dispatcher, software_test_only=True, processes=self.processes, execute=self.fake_execute)


class Qualification(unittest.TestCase):
    def setUp(self):
        simulation_root = Path.cwd() / ".e18_tail_toy"
        simulation_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=simulation_root)
        self.f = Fixture(self.directory.name)

    def tearDown(self):
        self.f.close()
        self.directory.cleanup()

    def test_exact_disjoint_partition(self):
        plan, lease = self.f.checkpoint()
        self.assertEqual((lease["original_parent_tasks"], lease["checkpoint_successes"], lease["remaining_count"]), (5, 2, 3))
        self.assertEqual([len(tail._keys(tail.base.read_json(tail.base.checked_ref(x))["task_keys_ref"])) for x in plan["manifest_refs"]], [2, 1])

    def test_prefix_preserved_and_retry_charges_once(self):
        paths = [self.f.owner(key) / "records.jsonl" for key in self.f.keys[:2]]
        hashes = [tail.base.file_sha(path) for path in paths]
        self.f.checkpoint()
        self.assertEqual(self.f.execute()["status"], "DONE")
        self.assertEqual(self.f.execute()["status"], "DONE")
        self.assertEqual(self.f.charges, 3)
        self.assertEqual(hashes, [tail.base.file_sha(path) for path in paths])

    def test_all_fine_done_is_not_native_completion(self):
        self.f.checkpoint()
        state = self.f.output / "_dispatcher/plans" / self.f.fine_sha
        state.mkdir(parents=True)
        for reference in self.f.fine_plan["manifest_refs"]:
            tail.base.write_once(state / (reference["sha256"] + ".status.json"), dict(status="DONE", manifest_sha256=reference["sha256"]))
        with self.assertRaisesRegex(RuntimeError, "all original native"):
            self.f.broker.finish(self.f.fine_plan, self.f.lease)
        self.assertEqual(tail.base.read_json(self.f.broker.status_path)["status"], "PAUSED")

    def test_parent_done_only_after_all_native_successes(self):
        self.f.checkpoint()
        self.f.execute()
        complete = self.f.broker.finish(self.f.fine_plan, self.f.lease)
        self.assertEqual(complete["original_native_tasks_verified"], 5)
        self.assertEqual(tail.base.read_json(self.f.broker.status_path)["status"], "DONE")
        self.assertEqual(self.f.processes.signals, [])

    def test_unknown_paid_owner_rejected(self):
        owner = self.f.owner(self.f.keys[2])
        owner.mkdir(parents=True)
        tail.base.write_once(owner / "ATTEMPT_STATE.json", dict(status="RESERVED"))
        with self.assertRaises(RuntimeError):
            self.f.checkpoint()

    def test_failed_native_rejected(self):
        self.f.make_success(self.f.keys[2])
        path = self.f.owner(self.f.keys[2]) / "ATTEMPT_STATE.json"
        value = tail.base.read_json(path)
        value["status"] = "FAILED_KNOWN_COST"
        tail.base.write_atomic(path, value)
        with self.assertRaises(RuntimeError):
            self.f.checkpoint()

    def test_changed_native_record_rejected(self):
        path = self.f.owner(self.f.keys[0]) / "records.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.f.checkpoint()

    def test_broker_death_rejects_new_execution(self):
        self.f.checkpoint()
        self.f.processes.owner_alive = False
        with self.assertRaises(PermissionError):
            self.f.execute()
        self.assertEqual(self.f.charges, 0)

    def test_broker_pid_reuse_rejected(self):
        self.f.checkpoint()
        self.f.processes.owner_reused = True
        with self.assertRaises(PermissionError):
            self.f.execute()

    def test_parent_claim_loss_rejected(self):
        self.f.checkpoint()
        self.f.processes.claim_lost = True
        with self.assertRaises(PermissionError):
            self.f.execute()

    def test_upstream_broker_death_rejected(self):
        self.f.checkpoint()
        self.f.upstream_alive = False
        with self.assertRaises(PermissionError):
            self.f.execute()

    def test_former_process_return_rejected(self):
        self.f.checkpoint()
        self.f.processes.former_alive = True
        with self.assertRaises(PermissionError):
            self.f.execute()

    def test_static_history_registry_after_broker_exit(self):
        self.f.checkpoint()
        self.f.processes.owner_alive = False
        manifests, _ = tail.admit_fine_plan(self.f.fine_plan, self.f.fine_sha, self.f.dispatcher, software_test_only=True, processes=self.f.processes, require_live_parent=False)
        self.assertEqual(len(manifests), 2)
        with self.assertRaises(PermissionError):
            self.f.execute()

    def mutate_first_manifest(self, transform):
        plan = copy.deepcopy(self.f.fine_plan)
        reference = plan["manifest_refs"][0]
        manifest = tail.base.read_json(tail.base.checked_ref(reference))
        transform(manifest)
        path = self.f.root / "changed.manifest.json"
        tail.base.write_once(path, manifest)
        plan["manifest_refs"][0] = tail.ref(path)
        return plan

    def test_backend_change_rejected(self):
        self.f.checkpoint()
        plan = self.mutate_first_manifest(lambda m: m.update(backend=dict(mode="changed")))
        with self.assertRaises(ValueError):
            tail.admit_fine_plan(plan, self.f.fine_sha, self.f.dispatcher, software_test_only=True, processes=self.f.processes)

    def test_foreign_key_rejected(self):
        self.f.checkpoint()
        path = self.f.root / "foreign.keys.json"
        tail.base.write_once(path, [["synthetic-foreign", 42, 4]])
        plan = self.mutate_first_manifest(lambda m: m.update(task_keys_ref=tail.ref(path)))
        with self.assertRaises(ValueError):
            tail.admit_fine_plan(plan, self.f.fine_sha, self.f.dispatcher, software_test_only=True, processes=self.f.processes)

    def test_duplicate_partition_rejected(self):
        self.f.checkpoint()
        plan = copy.deepcopy(self.f.fine_plan)
        plan["manifest_refs"].append(plan["manifest_refs"][0])
        with self.assertRaises(ValueError):
            tail.admit_fine_plan(plan, self.f.fine_sha, self.f.dispatcher, software_test_only=True, processes=self.f.processes)

    def test_modified_scope_source_rejected(self):
        self.f.checkpoint()
        self.f.scope_path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.f.execute()

    def test_nested_guard_restored_after_exception(self):
        self.f.checkpoint()
        original = self.f.worker.assert_lease_active
        with self.assertRaisesRegex(RuntimeError, "synthetic exception"):
            with tail.nested_call_guard(self.f.worker, self.f.lease, software_test_only=True, processes=self.f.processes):
                self.f.worker.assert_lease_active({}, software_test_only=True)
                raise RuntimeError("synthetic exception")
        self.assertIs(self.f.worker.assert_lease_active, original)

    def test_each_paid_call_rechecks_parent(self):
        self.f.checkpoint()
        original = self.f.worker.assert_lease_active
        with tail.nested_call_guard(self.f.worker, self.f.lease, software_test_only=True, processes=self.f.processes):
            self.f.worker.assert_lease_active({}, software_test_only=True)
            self.f.processes.owner_alive = False
            with self.assertRaises(PermissionError):
                self.f.worker.assert_lease_active({}, software_test_only=True)
        self.assertIs(self.f.worker.assert_lease_active, original)

    def test_each_paid_call_rechecks_lease_bytes(self):
        self.f.checkpoint()
        with tail.nested_call_guard(self.f.worker, self.f.lease, software_test_only=True, processes=self.f.processes):
            self.f.worker.assert_lease_active({}, software_test_only=True)
            path = self.f.broker.root / "LEASE.json"
            changed = tail.base.read_json(path)
            changed["remaining_count"] = 1000
            tail.base.write_atomic(path, changed)
            with self.assertRaises(ValueError):
                self.f.worker.assert_lease_active({}, software_test_only=True)

    def test_wrong_parent_claim_path_rejected(self):
        self.f.checkpoint()
        plan = copy.deepcopy(self.f.fine_plan)
        lease = copy.deepcopy(self.f.lease)
        lease["parent_claim_path"] = str(self.f.root / "unrelated.claim")
        path = self.f.root / "changed-lease.json"
        tail.base.write_once(path, lease)
        plan["tail_parent_lease_ref"] = tail.ref(path)
        with self.assertRaises(ValueError):
            tail.admit_fine_plan(plan, self.f.fine_sha, self.f.dispatcher, software_test_only=True, processes=self.f.processes)

    def test_no_signal_api_used_for_checkpoint(self):
        self.f.checkpoint()
        self.assertEqual(self.f.processes.signals, [])

    def test_large_chunk_configuration_rejected(self):
        with self.assertRaises(ValueError):
            tail.TailPartitionBroker(self.f.plan_path, self.f.plan_sha, self.f.parent_ref["sha256"], self.f.dispatcher, self.f.processes, self.f.output / "x", self.f.root / "y", self.f.root / "z", former_pid=7001, former_gpu="7", validator=self.f.validate, scope_sha=self.f.scope_sha, chunk_size=65, expected_tasks=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualifications are append-only")
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Qualification))
    report = dict(schema="e18_member4_tail_partition_qualification_v1", status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), software_test_only=True, physical_llm_calls=0, gold_reads=0, process_signals_sent=0, original_worker_source_sha256=tail.base.file_sha(Path(tail.__file__).with_name("e18_member4_worker_v3.py")), controller_source_sha256=tail.base.file_sha(tail.__file__), qualification_source_sha256=tail.base.file_sha(__file__), wall_seconds=time.perf_counter() - started, limitations=["Mock process identities and parent-lock observations", "No live Linux checkpoint or physical model replay performed by this qualification"])
    tail.base.write_once(args.output, report)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
