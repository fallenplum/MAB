"""Software-only member4 persistent worker, lease and slot qualification."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import signal
import time
import unittest

import e18_member4_worker_v3 as worker
import e18_member4_dispatcher_v3 as dispatcher
import qualify_gpu_worker as fixtures


class Member4Qualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GPUWorkerQualification.setUpClass()

    def setUp(self):
        worker._PROCESS_BACKEND_CONTEXT = None
        worker._ACTIVE_GPU_SLOT_CONTEXT = None
        self.fixture = fixtures.GPUWorkerQualification(
            "test_persistent_model_for_multiple_generation_seeds"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.manifest = copy.deepcopy(self.fixture.manifest)
        self.manifest.update(schema=worker.MANIFEST_SCHEMA, member_index=4)
        self.manifest["task_keys_ref"] = self.fixture.artifact(
            "member4-tasks.json", [["synthetic-gsm", 42, 4], ["synthetic-gsm", 137, 4]]
        )
        remaining = sorted(
            key for key in self.fixture.fixture.adapter.callable_members if key[2] == 4
        )
        self.lease = dict(
            schema="e18_member4_checkpoint_lease_v1",
            status="LEASED",
            base_member_plan_sha256=fixtures.fixtures.canonical_digest("base"),
            execution_scope_sha256=fixtures.fixtures.canonical_digest(
                self.fixture.fixture.scope
            ),
            approval_sha256=self.manifest["approval_ref"]["sha256"],
            target_member_index=4,
            owner_pid=os.getpid(),
            former_pid=999999999,
            checkpoint_stop_verified=True,
            pending_or_unknown_attempts=0,
            original_member4_tasks=len(remaining),
            checkpoint_successes=0,
            remaining_count=len(remaining),
            remaining_task_keys_ref=self.fixture.artifact("remaining.json", remaining),
            parent_shards=[
                dict(
                    manifest_sha256=fixtures.fixtures.canonical_digest(
                        "parent" + str(index)
                    ),
                    task_keys_sha256=fixtures.fixtures.canonical_digest(
                        "parent-keys" + str(index)
                    ),
                    claim_path=str(self.root / ("parent" + str(index) + ".claim")),
                )
                for index in range(2)
            ],
        )
        self.manifest["checkpoint_lease_ref"] = self.fixture.artifact(
            "lease.json", self.lease
        )
        self.backend = fixtures.fixtures.FakeBackend(
            self.fixture.fixture.member_specs[4], self.fixture.fixture.apis.inference
        )
        self.policy = dict(
            schema="e18_member4_gpu_slot_policy_v1",
            cohort="e18_confirm",
            software_test_only=True,
            scope_sha256=self.lease["execution_scope_sha256"],
            approval_sha256=self.lease["approval_sha256"],
            max_slots=2,
            minimum_free_memory_bytes=worker.MINIMUM_EXTRA_SLOT_BYTES,
            gpu_lock_root=self.manifest["gpu_lock_root"],
            gpus=[dict(index=0, uuid="GPU-SYNTHETIC", known_e18_pids=[111])],
        )
        self.policy_ref = self.fixture.artifact("policy.json", self.policy)
        self.state = dict(
            index=0,
            uuid="GPU-SYNTHETIC",
            free_memory_bytes=16 * 1024**3,
            compute_pids=[111],
        )

    def run_worker(self, manifest=None, factory=None):
        manifest = manifest or self.manifest
        return worker.execute_manifest(
            manifest,
            fixtures.fixtures.canonical_digest(manifest),
            software_test_only=True,
            injected_factory=factory or (lambda _: self.backend),
        )

    def resources(self, slot=1, reference=None):
        return worker.validate_slot_resources(
            self.manifest,
            "0",
            slot,
            resource_policy_ref=reference or self.policy_ref,
            software_test_only=True,
            probe=lambda _: self.state,
        )

    def test_two_shards_load_once(self):
        first = copy.deepcopy(self.manifest)
        first["task_keys_ref"] = self.fixture.artifact(
            "first.json", [["synthetic-gsm", 42, 4]]
        )
        second = copy.deepcopy(self.manifest)
        second["task_keys_ref"] = self.fixture.artifact(
            "second.json", [["synthetic-gsm", 137, 4]]
        )
        one = self.run_worker(first)
        two = self.run_worker(
            second, factory=lambda _: self.fail("Cross-shard model reloaded")
        )
        self.assertEqual(
            (one["backend_loads"], two["backend_loads"], self.backend.calls), (1, 0, 2)
        )

    def test_success_cache_adds_no_generation(self):
        self.run_worker()
        reused = self.run_worker(
            factory=lambda _: self.fail("Cache loaded another backend")
        )
        self.assertEqual(
            (reused["cached"], reused["backend_loads"], self.backend.calls), (2, 0, 2)
        )

    def test_other_profile_rejected(self):
        self.manifest["member_index"] = 3
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_changed_backend_path_rejected(self):
        self.run_worker()
        self.manifest["backend"]["physical_root"] = "/different"
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_production_injected_backend_rejected(self):
        self.manifest["software_test_only"] = False
        with self.assertRaises(PermissionError):
            worker.load_worker(self.manifest, injected_factory=lambda _: self.backend)

    def test_slot0_uses_legacy_lock(self):
        with fixtures.worker.GPUOwnership(self.root / "locks", "0", "a" * 64):
            with self.assertRaises(RuntimeError):
                with worker.GPUOwnership(self.root / "locks", "0", "b" * 64, 0):
                    self.fail("Legacy GPU lock bypassed")

    def test_two_slots_have_distinct_locks(self):
        with worker.GPUOwnership(self.root / "locks", "0", "a" * 64, 0):
            with worker.GPUOwnership(self.root / "locks", "0", "b" * 64, 1):
                pass

    def test_slot1_is_exclusive(self):
        with worker.GPUOwnership(self.root / "locks", "0", "a" * 64, 1):
            with self.assertRaises(RuntimeError):
                with worker.GPUOwnership(self.root / "locks", "0", "b" * 64, 1):
                    pass

    def test_third_slot_rejected(self):
        with self.assertRaises(ValueError):
            worker.GPUOwnership(self.root / "locks", "0", "a" * 64, 2)

    def test_known_pids_and_sufficient_memory_pass(self):
        self.assertEqual(self.resources()["status"], "PASS")

    def test_unknown_pid_rejected(self):
        self.state["compute_pids"] = [222]
        with self.assertRaises(PermissionError):
            self.resources()

    def test_own_pid_is_implicitly_allowed(self):
        self.state["compute_pids"] = [111, os.getpid()]
        self.assertEqual(self.resources()["status"], "PASS")

    def test_extra_slot_requires_12gib(self):
        self.state["free_memory_bytes"] = worker.MINIMUM_EXTRA_SLOT_BYTES - 1
        with self.assertRaises(PermissionError):
            self.resources()

    def test_slot0_keeps_legacy_memory_admission(self):
        self.state["free_memory_bytes"] = 1
        self.assertEqual(self.resources(slot=0)["status"], "PASS")

    def test_two_existing_processes_rejected(self):
        changed = copy.deepcopy(self.policy)
        changed["gpus"][0]["known_e18_pids"].append(222)
        self.state["compute_pids"] = [111, 222]
        reference = self.fixture.artifact("two-process-policy.json", changed)
        with self.assertRaises(PermissionError):
            self.resources(reference=reference)

    def test_wrong_gpu_uuid_rejected(self):
        self.state["uuid"] = "GPU-OTHER"
        with self.assertRaises(PermissionError):
            self.resources()

    def test_policy_cannot_allow_three_slots(self):
        changed = copy.deepcopy(self.policy)
        changed["max_slots"] = 3
        with self.assertRaises(PermissionError):
            self.resources(
                reference=self.fixture.artifact("three-slot-policy.json", changed)
            )

    def test_policy_cannot_lower_memory_minimum(self):
        changed = copy.deepcopy(self.policy)
        changed["minimum_free_memory_bytes"] = 1
        with self.assertRaises(ValueError):
            self.resources(
                reference=self.fixture.artifact("low-memory-policy.json", changed)
            )

    def test_production_disallows_injected_telemetry(self):
        with self.assertRaises(PermissionError):
            worker.validate_slot_resources(
                self.manifest,
                "0",
                1,
                resource_policy_ref=self.policy_ref,
                probe=lambda _: self.state,
            )

    def test_dead_broker_rejected(self):
        self.lease["owner_pid"] = 999999998
        self.manifest["checkpoint_lease_ref"] = self.fixture.artifact(
            "dead-broker.json", self.lease
        )
        with self.assertRaises(PermissionError):
            self.run_worker()

    def test_live_former_worker_rejected(self):
        self.lease["former_pid"] = os.getpid()
        self.manifest["checkpoint_lease_ref"] = self.fixture.artifact(
            "live-former.json", self.lease
        )
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_keys_outside_remaining_rejected(self):
        self.lease["remaining_task_keys_ref"] = self.fixture.artifact(
            "smaller-remaining.json", [["synthetic-gsm", 42, 4]]
        )
        self.lease["remaining_count"] = 1
        self.lease["checkpoint_successes"] = self.lease["original_member4_tasks"] - 1
        self.manifest["checkpoint_lease_ref"] = self.fixture.artifact(
            "smaller-lease.json", self.lease
        )
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_signal_stops_after_current_result(self):
        self.manifest["install_signal_handlers"] = True
        original = self.backend.generate

        def generate(messages, seed):
            signal.raise_signal(signal.SIGTERM)
            return original(messages, seed)

        self.backend.generate = generate
        result = self.run_worker()
        self.assertTrue(result["stop_requested"])
        self.assertEqual((result["completed"], self.backend.calls), (1, 1))

    def test_dispatcher_returns_paused_without_next_claim(self):
        first = copy.deepcopy(self.manifest)
        first["task_keys_ref"] = self.fixture.artifact(
            "pause-first-keys.json", [["synthetic-gsm", 42, 4]]
        )
        second = copy.deepcopy(self.manifest)
        second["task_keys_ref"] = self.fixture.artifact(
            "pause-second-keys.json", [["synthetic-gsm", 137, 4]]
        )
        plan = dict(
            schema=dispatcher.PLAN_SCHEMA,
            worker_source_sha256=worker.file_sha(worker.__file__),
            checkpoint_lease_ref=self.manifest["checkpoint_lease_ref"],
            manifest_refs=[
                self.fixture.artifact("pause-first.json", first),
                self.fixture.artifact("pause-second.json", second),
            ],
        )
        calls = []

        def paused(manifest, digest, software_test_only=False):
            calls.append(digest)
            return dict(planned=1, completed=1, failures=[], stop_requested=True)

        result = dispatcher.run_plan(
            plan,
            fixtures.fixtures.canonical_digest(plan),
            software_test_only=True,
            execute=paused,
            idle_seconds=0,
        )
        self.assertEqual((result["status"], len(calls)), ("PAUSED", 1))

    def test_original_fast_worker_bytes_unchanged(self):
        directory = Path(__file__).parent
        self.assertEqual(
            worker.file_sha(directory / "e18_gpu_worker.py"),
            "17f74192c3ccf10930633f3f1dac48487ce9f62b25c848e2703f03fc6bc9c4f0",
        )
        self.assertEqual(
            worker.file_sha(directory / "e18_gpu_dispatcher.py"),
            "5d273799e2c74f0b5cbd40aa6a6c6ae9555fa9892672bc47f3cba174b9c481a2",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification outputs are append-only")
    wall = time.perf_counter()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Member4Qualification)
    )
    record = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        worker_source_sha256=worker.file_sha(worker.__file__),
        dispatcher_source_sha256=worker.file_sha(dispatcher.__file__),
        test_source_sha256=worker.file_sha(__file__),
        actual_model_calls=0,
        actual_gpu_queries=0,
        real_gold_reads=0,
        actual_jobs_submitted=0,
        linux_proc_lock_test_status="ROOT_PENDING",
        wall_seconds=time.perf_counter() - wall,
        test_output=stream.getvalue(),
    )
    worker.save_json(args.output, record)
    print(json.dumps(dict(status=record["status"], tests=result.testsRun)))
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
