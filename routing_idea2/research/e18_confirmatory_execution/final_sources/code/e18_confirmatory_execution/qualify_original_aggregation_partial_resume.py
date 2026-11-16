"""Software-only partial-batch resume checks for unchanged original sources.

All task outputs and signal deliveries below are synthetic. The SIGTERM handler
registered by the original worker is invoked directly during FakeBackend output
generation. No operating-system signal, GPU query or production task is issued.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import signal
import time
import unittest
from unittest.mock import patch

import e18_aggregation_dispatcher_v3 as dispatcher
import e18_aggregation_worker_v3 as worker
import qualify_aggregation_worker_v3 as fixtures


WORKER_SHA = "de5c3cc8a509be10e66b3290899862bab73a2784a62aa743e7c48d41a7842495"
DISPATCHER_SHA = "715fb1515968717941695224cf5f9a7e60e4fdaa174e18330bf12a9b72911ec6"
EVIDENCE = []


class OriginalPartialResumeQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.AggregationQualification.setUpClass()

    def setUp(self):
        self.fixture = fixtures.AggregationQualification("test_two_shards_construct_one_backend")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.fixture.fixture.fixture.dependencies(seed=42)
        self.fixture.fixture.fixture.dependencies(seed=137)
        self.fixture.fixture.fixture.dependencies(seed=2024)
        self.manifest = self.fixture.fixture.aggregation_manifest((42, 137, 2024))
        self.manifest["schema"] = worker.MANIFEST_SCHEMA
        self.manifest["install_signal_handlers"] = True
        self.manifest_ref = self.fixture.fixture.artifact("partial-manifest.json", self.manifest)
        self.plan = dict(
            schema=dispatcher.PLAN_SCHEMA, worker_source_sha256=WORKER_SHA,
            manifest_refs=[self.manifest_ref],
        )
        self.plan_ref = self.fixture.fixture.artifact("partial-plan.json", self.plan)
        self.state_root = self.root / "owned/_dispatcher/plans" / self.plan_ref["sha256"]
        self.status_path = self.state_root / (self.manifest_ref["sha256"] + ".status.json")
        self.backend = self.fixture.fake_backend()
        self.constructions = 0
        tokenizer = patch.object(worker, "frozen_tokenizer_files", return_value=self.fixture.files)
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def factory(self, spec):
        self.constructions += 1
        return self.backend

    def execute(self, manifest, digest, *, software_test_only):
        return worker.execute_manifest(
            manifest, digest, software_test_only=software_test_only,
            injected_factory=self.factory,
        )

    def run_original(self):
        self.assertEqual(worker.file_sha(self.manifest_ref["path"]), self.manifest_ref["sha256"])
        self.assertEqual(worker.file_sha(self.plan_ref["path"]), self.plan_ref["sha256"])
        return dispatcher.run_plan(
            self.plan, self.plan_ref["sha256"], software_test_only=True,
            execute=self.execute, idle_seconds=0, max_idle_rounds=1,
        )

    def owner(self, seed=42):
        digest = fixtures.fixtures.fixtures.canonical_digest
        scope = self.fixture.fixture.fixture.scope
        return self.root / "owned" / digest(scope) / "aggregation" / digest(["synthetic-gsm", seed, 0])

    def stop_after_first_committed_output(self):
        generate = self.backend.generate

        def signal_after_result(messages, seed):
            result = generate(messages, seed)
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            return result

        self.backend.generate = signal_after_result
        try:
            result = self.run_original()
        finally:
            self.backend.generate = generate
        status = worker.read_json(self.status_path)
        self.assertEqual(result["status"], "WAITING")
        self.assertEqual(status["status"], "WAITING_FOR_DEPENDENCIES")
        summary = status["summary"]
        self.assertEqual((summary["planned"], summary["completed"], summary["cached"]), (3, 1, 0))
        self.assertTrue(summary["stop_requested"])
        self.assertEqual(self.backend.calls, 1)
        self.assertFalse((self.state_root / "STOP_REQUIRES_RECONCILIATION.json").exists())
        self.assertEqual(worker.read_json(self.owner() / "ATTEMPT_STATE.json")["status"], "SUCCESS_COMMITTED")
        return result, summary

    def test_same_plan_manifest_reuses_prefix_then_finishes_remaining(self):
        _, stopped = self.stop_after_first_committed_output()
        owner = self.owner()
        prefix = {name: (owner / name).read_bytes() for name in ("ATTEMPT_STATE.json", "records.jsonl")}
        first_receipt = worker.read_json(owner / "LAST_TASK_RECEIPT.json")
        resumed = self.run_original()
        self.assertEqual(resumed["status"], "DONE")
        summary = worker.read_json(self.status_path)["summary"]
        self.assertEqual((summary["completed"], summary["cached"], summary["new_successes"]), (3, 1, 2))
        self.assertFalse(summary["stop_requested"])
        self.assertEqual((self.backend.calls, self.constructions), (3, 1))
        for name, data in prefix.items():
            self.assertEqual((owner / name).read_bytes(), data)
        cached = worker.read_json(owner / "LAST_TASK_RECEIPT.json")
        self.assertTrue(cached["from_cache"])
        for name in ("physical_attempt_id", "native_record_sha256", "native_job_id"):
            self.assertEqual(cached[name], first_receipt[name])
        EVIDENCE.append(dict(
            case="same_process_same_original_plan_manifest", stopped_summary=stopped,
            resumed_summary=summary, synthetic_backend_calls=self.backend.calls,
            backend_constructions=self.constructions, prefix_fence_and_record_bytes_unchanged=True,
            prefix_original_attempt_job_and_record_identity_preserved=True,
            original_plan_and_manifest_sha_preserved=True,
        ))

    def test_process_context_reset_reuses_prefix_without_new_prefix_call(self):
        self.stop_after_first_committed_output()
        previous = worker.read_json(self.owner() / "LAST_TASK_RECEIPT.json")
        first_backend_calls = self.backend.calls
        # Reset only the synthetic module state to model a new process. Original
        # source files, manifests, scope and committed owner bytes remain intact.
        worker._PROCESS_BACKEND_CONTEXT = None
        self.fixture.tokenizer.name_or_path = str(self.fixture.physical)
        self.backend = self.fixture.fake_backend()
        result = self.run_original()
        current = worker.read_json(self.owner() / "LAST_TASK_RECEIPT.json")
        self.assertEqual(result["status"], "DONE")
        self.assertEqual((first_backend_calls, self.backend.calls, self.constructions), (1, 2, 2))
        self.assertTrue(current["from_cache"])
        self.assertEqual(current["physical_attempt_id"], previous["physical_attempt_id"])
        EVIDENCE.append(dict(
            case="simulated_new_process_original_owner_resume", stopped_prefix_calls=first_backend_calls,
            resumed_backend_calls=self.backend.calls, cached_prefix_calls=0,
            total_synthetic_generation_calls=first_backend_calls + self.backend.calls,
            prefix_original_attempt_preserved=True,
        ))

    def test_repeated_completed_plan_adds_no_calls(self):
        self.stop_after_first_committed_output()
        self.run_original()
        before = (self.backend.calls, self.constructions)
        self.assertEqual(self.run_original()["status"], "DONE")
        self.assertEqual((self.backend.calls, self.constructions), before)

    def test_incompatible_committed_fence_rejects_resume_without_extra_call(self):
        self.stop_after_first_committed_output()
        path = self.owner() / "ATTEMPT_STATE.json"
        fence = worker.read_json(path)
        fence["status"] = "RESERVED_BEFORE_BACKEND"
        worker.save_json(path, fence)
        with self.assertRaises(RuntimeError):
            self.run_original()
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(worker.read_json(self.status_path)["status"], "FAILED")
        self.assertTrue((self.state_root / "STOP_REQUIRES_RECONCILIATION.json").exists())

    def test_missing_committed_fence_rejects_resume_without_extra_call(self):
        self.stop_after_first_committed_output()
        (self.owner() / "ATTEMPT_STATE.json").unlink()
        with self.assertRaises(RuntimeError):
            self.run_original()
        self.assertEqual(self.backend.calls, 1)

    def test_unknown_cost_failure_blocks_same_plan_retry(self):
        def unknown_cost(messages, seed):
            self.backend.calls += 1
            raise self.fixture.fixture.fixture.apis.inference.GenerationFailure("synthetic_unknown_cost")

        self.backend.generate = unknown_cost
        with self.assertRaises(RuntimeError):
            self.run_original()
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(worker.read_json(self.owner() / "ATTEMPT_STATE.json")["status"], "STOP_REQUIRES_RECONCILIATION")
        with self.assertRaises(RuntimeError):
            self.run_original()
        self.assertEqual(self.backend.calls, 1)

    def test_known_cost_failed_native_attempt_also_blocks_retry(self):
        self.backend.failure = dict(prompt_tokens=2, completion_tokens=1)
        with self.assertRaises(RuntimeError):
            self.run_original()
        self.assertEqual(worker.read_json(self.status_path)["status"], "FAILED")
        with self.assertRaises(RuntimeError):
            self.run_original()
        self.assertEqual(self.backend.calls, 1)

    def test_original_core_source_hashes_match_frozen_contract(self):
        self.assertEqual(worker.file_sha(worker.__file__), WORKER_SHA)
        self.assertEqual(worker.file_sha(dispatcher.__file__), DISPATCHER_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification results are append-only")
    started = time.perf_counter()
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(OriginalPartialResumeQualification)
    )
    record = dict(
        schema="e18_original_aggregation_partial_resume_qualification_v1",
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        wall_seconds=time.perf_counter() - started,
        qualification_source_sha256=worker.file_sha(__file__),
        original_worker_sha256=worker.file_sha(worker.__file__),
        original_dispatcher_sha256=worker.file_sha(dispatcher.__file__),
        case_evidence=EVIDENCE, test_output=capture.getvalue(),
        actual_gpu_queries=0, actual_model_calls=0, real_gold_reads=0,
        server_logins=0, production_jobs_submitted=0,
        real_os_signals_sent=0, simulated_signal_handler_only=True,
        production_resume_performed=False, original_core_source_files_modified=False,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: record[key] for key in ("status", "tests", "failures", "errors", "wall_seconds")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
