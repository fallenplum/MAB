"""Toy-only qualification of original-lock persistent auxiliary aggregation."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import e18_auxiliary_aggregation_follower_v2 as follower
import e18_aggregation_dispatcher_v3 as dispatcher
import e18_aggregation_queue_daemon as queue
import e18_aggregation_worker_v3 as worker
import qualify_aggregation_worker_v3 as fixtures


class AuxiliaryQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.AggregationQualification.setUpClass()

    def setUp(self):
        self.fixture = fixtures.AggregationQualification("test_two_shards_construct_one_backend")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.queue_root = self.root / "queue"
        self.queue_root.mkdir()
        self.first, self.second = self.fixture.two_manifests()
        self.backend = self.fixture.fake_backend()
        self.constructions = 0
        self.config = dict(
            schema="e18_incremental_aggregation_planner_config_v1",
            server_root=str(self.root), scope_ref=self.first["scope_ref"],
            database_path=str(self.root / "planner/state.sqlite3"),
            startup_plan_ref=dict(path=str(self.root / "startup.json"), sha256="a" * 64),
            queues=[dict(name="synthetic", queue_root=str(self.queue_root), gpu_lock_root=self.first["gpu_lock_root"])],
        )
        self.config_ref = self.fixture.fixture.artifact("planner-config.json", self.config)
        tokenizer = patch.object(worker, "frozen_tokenizer_files", return_value=self.fixture.files)
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def factory(self, spec):
        self.constructions += 1
        return self.backend

    def actor(self, gpu="2"):
        return follower.AuxiliaryAggregationFollower(
            self.queue_root, self.first["gpu_lock_root"], gpu, self.config_ref,
            software_test_only=True, fixture_factory=self.factory,
        )

    def entry(self, name, manifest):
        reference = self.fixture.fixture.artifact(name + "-manifest.json", manifest)
        plan = dict(schema=dispatcher.PLAN_SCHEMA, worker_source_sha256=queue.EXPECTED_WORKER_SHA, manifest_refs=[reference])
        plan_ref = self.fixture.fixture.artifact(name + "-plan.json", plan)
        path = self.queue_root / (name + ".ready.json")
        queue.publish_once(path, dict(schema=queue.ENTRY_SCHEMA, plan_ref=plan_ref))
        state = self.root / "owned/_dispatcher/plans" / plan_ref["sha256"]
        return path, reference, plan_ref, state

    def run_actor(self, actor=None, **kwargs):
        return (actor or self.actor()).run(sleep_seconds=0, max_idle_rounds=1, **kwargs)

    def write_status(self, item, status, **extra):
        _, reference, _, state = item
        state.mkdir(parents=True, exist_ok=True)
        worker.save_json(state / (reference["sha256"] + ".status.json"), dict(status=status, manifest_sha256=reference["sha256"], **extra))

    def test_two_original_plans_reverse_order_construct_backend_once(self):
        one = self.entry("00000001", self.first)
        two = self.entry("00000002", self.second)
        seen = []
        original = dispatcher.run_plan

        def record(plan, sha, **kwargs):
            seen.append(sha)
            return original(plan, sha, **kwargs)

        with patch.object(dispatcher, "run_plan", side_effect=record):
            result = self.run_actor()
        self.assertEqual(seen, [two[2]["sha256"], one[2]["sha256"]])
        self.assertEqual((result["completed_batches"], self.constructions, self.backend.calls), (2, 1, 2))
        self.assertEqual(worker._PROCESS_BACKEND_CONTEXT.constructions, 1)
        self.assertFalse((self.queue_root / "QUEUE_CONSUMER.lock").exists())
        self.assertFalse((self.queue_root / "receipts").exists())
        logs = list((self.root / "auxiliary_aggregation_logs").glob("*.jsonl"))
        self.assertEqual(len(logs), 1)
        self.assertNotEqual(logs[0].name, "EVENTS.jsonl")

    def test_original_claim_blocks_aux_then_is_reusable(self):
        item = self.entry("00000001", self.first)
        item[3].mkdir(parents=True)
        claim = dispatcher.ShardClaim(item[3] / (item[1]["sha256"] + ".claim"))
        self.assertTrue(claim.acquire())
        try:
            self.assertEqual(self.run_actor()["status"], "WAITING")
            self.assertEqual((self.constructions, self.backend.calls), (0, 0))
        finally:
            claim.release()
        self.assertEqual(self.run_actor()["completed_batches"], 1)
        self.assertEqual(self.backend.calls, 1)

    def test_multi_manifest_plan_rejected_before_backend_or_native_status(self):
        first_ref = self.fixture.fixture.artifact("combined-first-manifest.json", self.first)
        second_ref = self.fixture.fixture.artifact("combined-second-manifest.json", self.second)
        plan = dict(
            schema=dispatcher.PLAN_SCHEMA, worker_source_sha256=queue.EXPECTED_WORKER_SHA,
            manifest_refs=[first_ref, second_ref],
        )
        reference = self.fixture.fixture.artifact("combined-plan.json", plan)
        queue.publish_once(
            self.queue_root / "00000001.ready.json", dict(schema=queue.ENTRY_SCHEMA, plan_ref=reference)
        )
        with self.assertRaisesRegex(ValueError, "exactly one manifest"):
            self.run_actor()
        self.assertEqual((self.backend.calls, self.constructions), (0, 0))
        self.assertFalse((self.root / "owned/_dispatcher/plans" / reference["sha256"]).exists())

    def test_skip_running_and_done_original_states(self):
        one = self.entry("00000001", self.first)
        two = self.entry("00000002", self.second)
        self.write_status(one, "RUNNING", pid=os.getpid())
        self.write_status(two, "DONE")
        result = self.run_actor()
        self.assertEqual((result["status"], self.constructions, self.backend.calls), ("WAITING", 0, 0))

    def test_foreign_claim_race_returns_waiting_without_duplicate(self):
        item = self.entry("00000001", self.first)
        actor = self.actor()
        original_execute = actor.execute
        claims = []

        def race(plan, sha):
            claim = dispatcher.ShardClaim(item[3] / (item[1]["sha256"] + ".claim"))
            self.assertTrue(claim.acquire())
            claims.append(claim)
            try:
                return original_execute(plan, sha)
            finally:
                claim.release()

        actor.execute = race
        result = self.run_actor(actor, max_batches=1)
        self.assertEqual(result["completed_batches"], 0)
        self.assertEqual((self.constructions, self.backend.calls), (0, 0))

    def test_original_stop_prevents_other_future_work(self):
        one = self.entry("00000001", self.first)
        self.entry("00000002", self.second)
        one[3].mkdir(parents=True)
        worker.save_json(one[3] / "STOP_REQUIRES_RECONCILIATION.json", dict(status="STOP"))
        with self.assertRaises(RuntimeError):
            self.run_actor()
        self.assertEqual(self.backend.calls, 0)

    def test_failed_or_unknown_cost_status_is_not_skipped(self):
        one = self.entry("00000001", self.first)
        self.entry("00000002", self.second)
        self.write_status(one, "FAILED", summary=dict(failures=[dict(usage_status="UNKNOWN")]))
        with self.assertRaises(RuntimeError):
            self.run_actor()
        self.assertEqual(self.backend.calls, 0)

    def test_worker_stop_requested_preserves_evidence(self):
        one = self.entry("00000001", self.first)
        self.write_status(one, "WAITING_FOR_DEPENDENCIES", summary=dict(stop_requested=True, failures=[]))
        with self.assertRaises(RuntimeError):
            self.run_actor()
        self.assertEqual(self.backend.calls, 0)

    def test_planner_stop_prevents_backend(self):
        self.entry("00000001", self.first)
        path = self.root / "planner/STOP_REQUIRES_RECONCILIATION.json"
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.run_actor()
        self.assertEqual(self.backend.calls, 0)

    def test_different_scope_is_rejected_before_any_backend(self):
        self.entry("00000001", self.first)
        altered = copy.deepcopy(self.second)
        altered["scope_ref"] = self.fixture.fixture.artifact("foreign-scope.json", self.fixture.fixture.fixture.scope)
        self.entry("00000002", altered)
        with self.assertRaises(ValueError):
            self.run_actor()
        self.assertEqual(self.backend.calls, 0)

    def test_empty_queue_backend_zero_and_no_false_done(self):
        result = self.run_actor()
        self.assertEqual((result["status"], self.constructions, self.backend.calls), ("WAITING", 0, 0))

    def test_arbitrary_all_done_marker_rejected(self):
        (self.queue_root / "ALL_DONE").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_actor()
        self.assertEqual(self.backend.calls, 0)

    def test_bound_original_full_target_marker_accepted(self):
        actor = self.actor()
        marker = dict(
            schema="e18_incremental_aggregation_all_done_v1", status="ALL_FROZEN_NATIVE_COMMITS_SUCCESS",
            config_sha256=self.config_ref["sha256"], execution_scope_sha256=queue.digest(actor.scope),
            total_frozen_aggregate_keys=actor.target_count,
            startup_plan_sha256=self.config["startup_plan_ref"]["sha256"],
            native_success_metadata_digest="b" * 64, gold_or_native_answer_payloads_read=0,
        )
        queue.publish_once(self.queue_root / "ALL_DONE", marker)
        self.assertEqual(self.run_actor(actor)["status"], "ALL_DONE")
        self.assertEqual(self.backend.calls, 0)

    def test_aux_lock_busy_does_not_take_main_queue_lock(self):
        claim = dispatcher.ShardClaim(self.queue_root / "AUX_CONSUMER_GPU2.lock")
        self.assertTrue(claim.acquire())
        try:
            self.assertEqual(self.run_actor()["status"], "AUX_CONSUMER_BUSY")
        finally:
            claim.release()
        self.assertFalse((self.queue_root / "QUEUE_CONSUMER.lock").exists())

    def test_existing_original_gpu_lock_prevents_model(self):
        item = self.entry("00000001", self.first)
        with worker.GPUOwnership(self.first["gpu_lock_root"], "2", item[2]["sha256"]):
            with self.assertRaises(RuntimeError):
                self.run_actor()
        self.assertEqual(self.backend.calls, 0)

    def test_production_rejects_fixture_backend(self):
        with self.assertRaises(PermissionError):
            follower.AuxiliaryAggregationFollower(
                self.queue_root, self.first["gpu_lock_root"], "2", self.config_ref,
                fixture_factory=self.factory,
            )

    def test_stop_before_work_does_not_construct_backend(self):
        self.entry("00000001", self.first)
        actor = self.actor()
        actor.stop_requested = True
        self.assertEqual(self.run_actor(actor)["status"], "STOP_REQUESTED")
        self.assertEqual(self.backend.calls, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification results are append-only")
    started = time.perf_counter()
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AuxiliaryQualification)
    )
    report = dict(
        schema="e18_auxiliary_aggregation_follower_qualification_v1",
        status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors),
        wall_seconds=time.perf_counter() - started,
        follower_source_sha256=worker.file_sha(follower.__file__),
        qualification_source_sha256=worker.file_sha(__file__),
        original_worker_sha256=worker.file_sha(worker.__file__),
        original_dispatcher_sha256=worker.file_sha(dispatcher.__file__),
        original_queue_sha256=worker.file_sha(queue.__file__),
        actual_gpu_queries=0, actual_model_calls=0, real_gold_reads=0,
        server_logins=0, production_launch_performed=False,
        test_output=capture.getvalue(),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "tests", "failures", "errors", "wall_seconds")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
