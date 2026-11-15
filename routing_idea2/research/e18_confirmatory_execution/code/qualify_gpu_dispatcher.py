"""Software-only qualification for dynamic, disjoint E18 shard dispatch."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import time
import unittest

import e18_gpu_dispatcher as dispatcher
import e18_gpu_worker as worker
import qualify_gpu_worker as fixtures


class DispatcherQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GPUWorkerQualification.setUpClass()

    def setUp(self):
        self.fixture = fixtures.GPUWorkerQualification(
            "test_persistent_model_for_multiple_generation_seeds"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.manifest = self.fixture.manifest
        self.reference = self.fixture.artifact("worker-manifest.json", self.manifest)
        self.plan = dict(
            schema=dispatcher.PLAN_SCHEMA,
            worker_source_sha256=worker.file_sha(worker.__file__),
            manifest_refs=[self.reference],
        )

    def execute(self, manifest, digest, software_test_only=False):
        return worker.execute_manifest(
            manifest,
            digest,
            software_test_only=software_test_only,
            injected_factory=lambda _: self.fixture.backend,
        )

    def run_plan(self):
        return dispatcher.run_plan(
            self.plan,
            fixtures.fixtures.canonical_digest(self.plan),
            software_test_only=True,
            execute=self.execute,
            idle_seconds=0,
            max_idle_rounds=1,
        )

    def test_shared_plan_executes_frozen_keys_once(self):
        self.assertEqual(self.run_plan()["status"], "DONE")
        self.assertEqual(self.fixture.backend.calls, 2)
        self.assertEqual(self.run_plan()["locally_completed_shards"], 0)
        self.assertEqual(self.fixture.backend.calls, 2)

    def test_duplicate_manifest_rejected(self):
        self.plan["manifest_refs"].append(self.reference)
        with self.assertRaises(ValueError):
            dispatcher.admit_plan(self.plan)

    def test_overlapping_shards_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed["task_keys_ref"] = self.fixture.artifact(
            "overlap.json", [["synthetic-gsm", 42, 0]]
        )
        self.plan["manifest_refs"].append(
            self.fixture.artifact("overlap-manifest.json", changed)
        )
        with self.assertRaises(ValueError):
            dispatcher.admit_plan(self.plan)

    def test_wrong_source_hash_rejected(self):
        self.plan["worker_source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            dispatcher.admit_plan(self.plan)

    def test_shard_claim_is_exclusive_and_released(self):
        first = dispatcher.ShardClaim(self.root / "claim")
        second = dispatcher.ShardClaim(self.root / "claim")
        self.assertTrue(first.acquire())
        self.assertFalse(second.acquire())
        first.release()
        self.assertTrue(second.acquire())
        second.release()

    def test_disjoint_shards_finish_dynamically(self):
        first = copy.deepcopy(self.manifest)
        first["task_keys_ref"] = self.fixture.artifact(
            "part1.json", [["synthetic-gsm", 42, 0]]
        )
        second = copy.deepcopy(self.manifest)
        second["task_keys_ref"] = self.fixture.artifact(
            "part2.json", [["synthetic-gsm", 137, 0]]
        )
        self.plan["manifest_refs"] = [
            self.fixture.artifact("manifest1.json", first),
            self.fixture.artifact("manifest2.json", second),
        ]
        self.assertEqual(self.run_plan()["locally_completed_shards"], 2)
        self.assertEqual(self.fixture.backend.calls, 2)

    def test_changed_scope_between_shards_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed["scope_ref"]["sha256"] = "0" * 64
        self.plan["manifest_refs"].append(
            self.fixture.artifact("changed-scope.json", changed)
        )
        with self.assertRaises(ValueError):
            dispatcher.admit_plan(self.plan)

    def test_dependency_waiting_retains_pending_state(self):
        def waiting(manifest, digest, software_test_only=False):
            return dict(planned=1, completed=0, failures=[])

        result = dispatcher.run_plan(
            self.plan,
            fixtures.fixtures.canonical_digest(self.plan),
            software_test_only=True,
            execute=waiting,
            idle_seconds=0,
            max_idle_rounds=1,
        )
        self.assertEqual(result["status"], "WAITING")

    def test_failed_shard_stops_plan_and_resume(self):
        calls = []

        def failed(manifest, digest, software_test_only=False):
            calls.append(digest)
            return dict(planned=2, completed=0, failures=[dict(status="unknown")])

        with self.assertRaises(RuntimeError):
            dispatcher.run_plan(
                self.plan,
                fixtures.fixtures.canonical_digest(self.plan),
                software_test_only=True,
                execute=failed,
                idle_seconds=0,
            )
        with self.assertRaises(RuntimeError):
            dispatcher.run_plan(
                self.plan,
                fixtures.fixtures.canonical_digest(self.plan),
                software_test_only=True,
                execute=failed,
                idle_seconds=0,
            )
        self.assertEqual(len(calls), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification output is append-only")
    wall = time.perf_counter()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DispatcherQualification)
    )
    record = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        dispatcher_source_sha256=worker.file_sha(dispatcher.__file__),
        worker_source_sha256=worker.file_sha(worker.__file__),
        test_source_sha256=worker.file_sha(__file__),
        actual_model_calls=0,
        real_gold_reads=0,
        actual_jobs_submitted=0,
        native_gpu_runtime_status="NOT_RUNTIME_TESTED",
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
