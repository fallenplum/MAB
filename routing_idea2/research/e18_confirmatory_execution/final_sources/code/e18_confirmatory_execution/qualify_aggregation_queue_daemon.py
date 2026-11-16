"""Pure FakeBackend qualification for the persistent aggregation ready queue."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import e18_aggregation_queue_daemon as daemon
import e18_aggregation_worker_v3 as worker
import e18_aggregation_dispatcher_v3 as dispatcher
import qualify_aggregation_worker_v3 as fixtures


class QueueQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.AggregationQualification.setUpClass()

    def setUp(self):
        self.fixture = fixtures.AggregationQualification(
            "test_two_shards_construct_one_backend"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.queue_root = self.root / "queue"
        self.queue_root.mkdir()
        self.first, self.second = self.fixture.two_manifests()
        self.backend = self.fixture.fake_backend()
        self.constructions = 0
        self.calls = []
        self.tokenizer_patch = patch.object(
            worker, "frozen_tokenizer_files", return_value=self.fixture.files
        )
        self.tokenizer_patch.start()
        self.addCleanup(self.tokenizer_patch.stop)

    def factory(self, spec):
        self.constructions += 1
        return self.backend

    def runner(self, plan, digest, software_test_only=False):
        self.calls.append(digest)

        def execute(manifest, sha, software_test_only=False):
            return worker.execute_manifest(
                manifest,
                sha,
                software_test_only=software_test_only,
                injected_factory=self.factory,
            )

        return dispatcher.run_plan(
            plan,
            digest,
            software_test_only=software_test_only,
            execute=execute,
            idle_seconds=0,
            max_idle_rounds=1,
        )

    def entry(self, name, manifest):
        manifest_ref = self.fixture.fixture.artifact(name + "-manifest.json", manifest)
        plan = dict(
            schema=dispatcher.PLAN_SCHEMA,
            worker_source_sha256=daemon.EXPECTED_WORKER_SHA,
            manifest_refs=[manifest_ref],
        )
        plan_ref = self.fixture.fixture.artifact(name + "-plan.json", plan)
        path = self.queue_root / (name + ".ready.json")
        daemon.publish_once(path, dict(schema=daemon.ENTRY_SCHEMA, plan_ref=plan_ref))
        return path, plan_ref

    def actor(self, metadata_only=False):
        return daemon.AggregationQueue(
            self.queue_root,
            self.first["gpu_lock_root"],
            metadata_only=metadata_only,
            software_test_only=True,
            fixture_factory=self.factory,
            fixture_runner=self.runner,
        )

    def run_actor(self, actor=None, max_batches=None):
        return (actor or self.actor()).run(
            max_batches=max_batches, sleep_seconds=0, max_idle_rounds=1
        )

    def test_two_plans_reuse_one_actual_fake_backend_construction(self):
        self.entry("001", self.first)
        self.entry("002", self.second)
        result = self.run_actor()
        self.assertEqual(
            (result["completed_batches"], self.constructions, self.backend.calls),
            (2, 1, 2),
        )
        self.assertEqual(worker._PROCESS_BACKEND_CONTEXT.constructions, 1)

    def test_completed_entry_resume_adds_no_calls(self):
        self.entry("001", self.first)
        self.run_actor()
        self.run_actor()
        self.assertEqual(
            (len(self.calls), self.constructions, self.backend.calls), (1, 1, 1)
        )

    def test_duplicate_key_in_independent_batch_rejected(self):
        self.entry("001", self.first)
        self.run_actor(max_batches=1)
        self.entry("002", self.first)
        with self.assertRaises(ValueError):
            self.run_actor()
        self.assertEqual(self.backend.calls, 1)

    def test_changed_scope_path_rejected(self):
        self.entry("001", self.first)
        self.run_actor(max_batches=1)
        altered = copy.deepcopy(self.second)
        altered["scope_ref"] = self.fixture.fixture.artifact(
            "copied-scope.json", self.fixture.fixture.fixture.scope
        )
        self.entry("002", altered)
        with self.assertRaises(ValueError):
            self.run_actor()

    def test_changed_approval_path_rejected(self):
        self.entry("001", self.first)
        self.run_actor(max_batches=1)
        altered = copy.deepcopy(self.second)
        altered["approval_ref"] = self.fixture.fixture.artifact(
            "copied-approval.json", self.fixture.fixture.fixture.approval
        )
        self.entry("002", altered)
        with self.assertRaises(ValueError):
            self.run_actor()

    def test_changed_physical_loading_path_rejected(self):
        self.entry("001", self.first)
        self.run_actor(max_batches=1)
        altered = copy.deepcopy(self.second)
        altered["backend"]["physical_root"] = "/different"
        self.entry("002", altered)
        with self.assertRaises(ValueError):
            self.run_actor()

    def test_unknown_entry_schema_rejected(self):
        path, _ = self.entry("001", self.first)
        value = worker.read_json(path)
        value["schema"] = "unknown"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_actor()

    def test_wrong_worker_source_rejected(self):
        path, plan_ref = self.entry("001", self.first)
        plan = worker.read_json(plan_ref["path"])
        plan["worker_source_sha256"] = "0" * 64
        changed_ref = self.fixture.fixture.artifact("wrong-worker-plan.json", plan)
        path.write_text(
            json.dumps(dict(schema=daemon.ENTRY_SCHEMA, plan_ref=changed_ref)),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            self.run_actor()

    def test_published_ready_entry_cannot_be_overwritten(self):
        path, _ = self.entry("001", self.first)
        self.run_actor()
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.actor()

    def test_publish_once_rejects_existing_target(self):
        path = self.queue_root / "immutable.json"
        daemon.publish_once(path, dict(value=1))
        with self.assertRaises(FileExistsError):
            daemon.publish_once(path, dict(value=2))
        self.assertEqual(worker.read_json(path), dict(value=1))

    def test_metadata_only_writes_no_execution_done(self):
        self.entry("001", self.first)
        result = self.run_actor(self.actor(metadata_only=True), max_batches=1)
        self.assertEqual(result["status"], "METADATA_ONLY")
        self.assertEqual(
            (self.backend.calls, self.constructions, self.calls), (0, 0, [])
        )
        self.assertEqual(list(self.queue_root.glob("receipts/*.done.json")), [])
        self.assertEqual(
            len(list(self.queue_root.glob("metadata_receipts/*.metadata.json"))), 1
        )

    def test_max_one_batch_leaves_second_pending(self):
        self.entry("001", self.first)
        self.entry("002", self.second)
        result = self.run_actor(max_batches=1)
        self.assertEqual(
            (result["status"], result["completed_batches"], self.backend.calls),
            ("BATCH_LIMIT_REACHED", 1, 1),
        )

    def test_root_all_done_exits_after_all_published_plans(self):
        self.entry("001", self.first)
        self.entry("002", self.second)
        (self.queue_root / "ALL_DONE").write_text(
            "Coordinator confirmed all required native tasks.", encoding="utf-8"
        )
        result = self.run_actor()
        self.assertEqual(
            (result["status"], result["completed_batches"], self.backend.calls),
            ("DONE", 2, 2),
        )

    def test_no_ready_waits_without_model_construction(self):
        result = self.run_actor()
        self.assertEqual(
            (result["status"], self.constructions, self.backend.calls),
            ("WAITING", 0, 0),
        )

    def test_production_rejects_injected_runner_or_factory(self):
        with self.assertRaises(PermissionError):
            daemon.AggregationQueue(
                self.queue_root,
                self.first["gpu_lock_root"],
                fixture_factory=self.factory,
            )
        with self.assertRaises(PermissionError):
            daemon.AggregationQueue(
                self.queue_root, self.first["gpu_lock_root"], fixture_runner=self.runner
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification output is append-only")
    wall = time.perf_counter()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(QueueQualification)
    )
    record = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        daemon_source_sha256=worker.file_sha(daemon.__file__),
        test_source_sha256=worker.file_sha(__file__),
        worker_source_sha256=worker.file_sha(worker.__file__),
        dispatcher_source_sha256=worker.file_sha(dispatcher.__file__),
        actual_model_calls=0,
        actual_gpu_queries=0,
        real_gold_reads=0,
        actual_jobs_submitted=0,
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
