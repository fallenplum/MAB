"""Synthetic qualification of unused and safely stopped original-shard modes."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import e18_gpu_dispatcher as dispatcher
import e18_gpu_worker as worker
import qualify_original_plan_shard as original_tests
import run_original_plan_shard_v2 as runner


class UnusedModeQualification(original_tests.OriginalPlanShardQualification):
    """Run all eleven original tests through the new default-mode wrapper."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(original_tests, "runner", runner)
        patcher.start()
        self.addCleanup(patcher.stop)


class StoppedModeQualification(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e18-stopped-shard-toy-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "owned"
        self.keys = [("synthetic-granite-" + str(index), 42, 2) for index in range(3)]
        self.scope = dict(cohort="synthetic_stopped_resume")
        self.scope_sha = runner.canonical_digest(self.scope)
        self.manifest = dict(
            schema=worker.MANIFEST_SCHEMA,
            stage="member_generation",
            member_index=2,
            scope_ref=dict(path="synthetic-scope", sha256="1" * 64),
            approval_ref=dict(path="synthetic-approval", sha256="2" * 64),
            question_records_ref=dict(path="synthetic-inputs", sha256="3" * 64),
            adapter_ref=dict(path="synthetic-adapter", sha256="4" * 64),
            owned_output_root=str(self.output),
            gpu_lock_root=str(self.root / "original-gpu-locks"),
            software_test_only=True,
            task_keys_ref=self.write("keys.json", self.keys),
        )
        self.reference = self.write("manifest.json", self.manifest)
        self.manifest_sha = self.reference["sha256"]
        self.plan = dict(
            schema=dispatcher.PLAN_SCHEMA,
            worker_source_sha256=worker.file_sha(worker.__file__),
            manifest_refs=[self.reference],
        )
        self.plan_ref = self.write("plan.json", self.plan)
        self.plan_sha = self.plan_ref["sha256"]
        self.state_root = self.output / "_dispatcher/plans" / self.plan_sha
        self.state_root.mkdir(parents=True)
        self.status_path = self.state_root / (self.manifest_sha + ".status.json")
        self.summary_path = self.output / "_dispatcher" / self.manifest_sha / "SUMMARY.json"
        self.summary_path.parent.mkdir(parents=True)
        self.previous_pid = 987654321
        self.calls = []
        self.validator_calls = []
        self.set_stopped()
        self.commit(0)
        adapter = SimpleNamespace(scope=self.scope, checked_member_dependency=self.validate_native)
        module = SimpleNamespace(canonical_digest=runner.canonical_digest)
        loader = mock.patch.object(
            worker, "load_worker", return_value=(module, adapter, self.keys, 2, None, {})
        )
        self.load_worker = loader.start()
        self.addCleanup(loader.stop)
        process = mock.patch.object(runner, "process_is_alive", return_value=False)
        self.process = process.start()
        self.addCleanup(process.stop)

    def write(self, name, value):
        path = self.root / name
        worker.save_json(path, value)
        return dict(path=str(path), sha256=worker.file_sha(path))

    def owner(self, index):
        return (
            self.output
            / self.scope_sha
            / "member_generation"
            / runner.canonical_digest(list(self.keys[index]))
        )

    def validate_native(self, qid, seed, member, native):
        self.assertEqual(native["synthetic_key"], [qid, seed, member])
        self.validator_calls.append((qid, seed, member))

    def commit(self, index):
        owner = self.owner(index)
        owner.mkdir(parents=True, exist_ok=True)
        key = list(self.keys[index])
        native = dict(
            synthetic_key=key,
            status="success",
            cost_status="recorded_actual_usage",
            job_id=runner.canonical_digest(dict(synthetic_key=key)),
            attempt_id="synthetic-attempt-" + str(index),
        )
        native["record_sha256"] = runner.canonical_digest(native)
        shared = dict(
            stage="member_generation",
            scope_sha256=self.scope_sha,
            approval_sha256="2" * 64,
            native_job_id=native["job_id"],
            native_record_sha256=native["record_sha256"],
            software_test_only=True,
        )
        worker.save_json(
            owner / "ATTEMPT_STATE.json",
            dict(shared, status="SUCCESS_COMMITTED", task_key=key),
        )
        worker.save_json(
            owner / "LAST_TASK_RECEIPT.json",
            dict(
                shared,
                schema="e18_execution_task_receipt_v2",
                status="SUCCESS",
                task_key=dict(question_id=key[0], generation_seed=key[1], member_index=key[2]),
                native_record=native,
                physical_attempt_id=native["attempt_id"],
            ),
        )
        (owner / "records.jsonl").write_text(json.dumps(native) + "\n", encoding="utf-8")

    def set_stopped(self, **changes):
        summary = dict(
            schema="e18_gpu_worker_summary_v1",
            manifest_sha256=self.manifest_sha,
            stage="member_generation",
            profile=2,
            completed=1,
            planned=3,
            failures=[],
            stop_requested=True,
            software_test_only=True,
        )
        summary.update(changes)
        worker.save_json(self.summary_path, summary)
        worker.save_json(
            self.status_path,
            dict(
                status="WAITING_FOR_DEPENDENCIES",
                manifest_sha256=self.manifest_sha,
                summary=summary,
            ),
        )

    def execute(self, manifest, manifest_sha, *, software_test_only):
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(manifest_sha, self.manifest_sha)
        self.assertTrue(software_test_only)
        self.calls.append(manifest_sha)
        return dict(completed=3, planned=3, failures=[])

    def run_resume(self, **changes):
        arguments = dict(
            manifest_sha=self.manifest_sha,
            member_index=2,
            gpu="3",
            gpu_lock_root=self.root / "admitted-slot",
            resume_stopped=True,
            previous_pid=self.previous_pid,
            software_test_only=True,
            execute=self.execute,
        )
        arguments.update(changes)
        return runner.run_one(self.plan, self.plan_sha, **arguments)

    def test_clean_stopped_prefix_resumes_same_manifest_and_original_state_root(self):
        original_bytes = worker.file_sha(self.reference["path"])
        result = self.run_resume()
        self.assertEqual(result["status"], "DONE")
        self.assertEqual(result["checkpoint"]["checkpoint_successes"], 1)
        self.assertEqual(result["checkpoint"]["remaining_tasks"], 2)
        self.assertEqual(worker.file_sha(self.reference["path"]), original_bytes)
        self.assertEqual(worker.file_sha(self.plan_ref["path"]), self.plan_sha)
        self.assertEqual(self.validator_calls, self.keys[:1])
        self.assertEqual(self.calls, [self.manifest_sha])
        self.assertEqual(worker.read_json(self.status_path)["status"], "DONE")

    def test_live_previous_process_is_rejected(self):
        self.process.return_value = True
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])
        self.load_worker.assert_not_called()

    def test_running_status_is_never_taken_over(self):
        state = worker.read_json(self.status_path)
        state["status"] = "RUNNING"
        worker.save_json(self.status_path, state)
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_failed_status_stops_plan(self):
        worker.save_json(self.status_path, dict(status="FAILED"))
        self.assertEqual(self.run_resume()["status"], "STOP_REQUIRES_RECONCILIATION")
        self.assertEqual(self.calls, [])

    def test_reserved_owner_is_not_retried(self):
        owner = self.owner(1)
        owner.mkdir(parents=True)
        worker.save_json(owner / "ATTEMPT_STATE.json", dict(status="RESERVED"))
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_failed_or_unknown_complete_owner_is_not_retried(self):
        for bad_status in ("FAILED", "UNKNOWN"):
            self.commit(1)
            path = self.owner(1) / "ATTEMPT_STATE.json"
            fence = worker.read_json(path)
            fence["status"] = bad_status
            worker.save_json(path, fence)
            with self.assertRaises(RuntimeError):
                self.run_resume()
        self.assertEqual(self.calls, [])

    def test_unknown_store_attempt_beside_success_is_rejected(self):
        with (self.owner(0) / "records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(status="unknown")) + "\n")
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_native_hash_tamper_is_rejected(self):
        path = self.owner(0) / "LAST_TASK_RECEIPT.json"
        receipt = worker.read_json(path)
        receipt["native_record"]["job_id"] = "tampered"
        worker.save_json(path, receipt)
        with self.assertRaises(ValueError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_non_stopping_summary_is_rejected(self):
        self.set_stopped(stop_requested=False)
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_summary_with_failures_is_rejected(self):
        self.set_stopped(failures=[dict(error_type="synthetic")])
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_summary_must_be_partial_and_match_journal(self):
        for completed in (0, 3):
            self.set_stopped(completed=completed)
            with self.assertRaises(RuntimeError):
                self.run_resume()
        self.set_stopped()
        summary = worker.read_json(self.summary_path)
        summary["completed"] = 2
        worker.save_json(self.summary_path, summary)
        with self.assertRaises(ValueError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_required_explicit_pid_and_manifest(self):
        for changes in (
            dict(previous_pid=None),
            dict(manifest_sha=None),
            dict(previous_pid=os.getpid()),
        ):
            with self.assertRaises(ValueError):
                self.run_resume(**changes)
        self.assertEqual(self.calls, [])

    def test_original_claim_is_exclusive(self):
        claim = dispatcher.ShardClaim(self.state_root / (self.manifest_sha + ".claim"))
        self.assertTrue(claim.acquire())
        self.addCleanup(claim.release)
        self.assertEqual(self.run_resume()["status"], "CLAIM_BUSY")
        self.assertEqual(self.calls, [])

    def test_previous_process_is_rechecked_after_claim(self):
        self.process.side_effect = [False, True]
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])
        self.load_worker.assert_not_called()

    def test_missing_committed_prefix_is_rejected(self):
        (self.owner(0) / "ATTEMPT_STATE.json").unlink()
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_commit_beyond_stopped_prefix_is_rejected(self):
        self.commit(2)
        with self.assertRaises(RuntimeError):
            self.run_resume()
        self.assertEqual(self.calls, [])

    def test_admitted_busy_gpu_preserves_stopped_status(self):
        before = worker.file_sha(self.status_path)
        with worker.GPUOwnership(self.root / "admitted-slot", "3", self.plan_sha):
            self.assertEqual(self.run_resume()["status"], "GPU_BUSY")
        self.assertEqual(worker.file_sha(self.status_path), before)
        self.assertEqual(self.calls, [])

    def test_production_disallows_custom_execute(self):
        with self.assertRaises(PermissionError):
            self.run_resume(software_test_only=False)

    def test_no_signal_function_is_called_or_present(self):
        import ast

        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        call_names = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(call_names & {"kill", "killpg", "send_signal", "terminate"})


class NativeStoreCacheQualification(unittest.TestCase):
    def test_original_store_reuses_paid_success_without_additional_backend_call(self):
        import qualify_gpu_worker as native_tests

        native_tests.GPUWorkerQualification.setUpClass()
        fixture = native_tests.GPUWorkerQualification(
            "test_resume_reuses_success_before_model_load"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest = copy.deepcopy(fixture.manifest)
        full_reference = fixture.artifact("full-original-manifest.json", manifest)
        manifest_sha = full_reference["sha256"]
        first = copy.deepcopy(manifest)
        first["task_keys_ref"] = fixture.artifact("first-paid-key.json", [["synthetic-gsm", 42, 0]])
        first_summary = fixture.run_worker(first)
        self.assertEqual(fixture.backend.calls, 1)
        plan = dict(
            schema=dispatcher.PLAN_SCHEMA,
            worker_source_sha256=worker.file_sha(worker.__file__),
            manifest_refs=[full_reference],
        )
        plan_sha = fixture.artifact("original-plan.json", plan)["sha256"]
        state_root = fixture.root / "owned/_dispatcher/plans" / plan_sha
        state_root.mkdir(parents=True)
        summary = dict(first_summary, planned=2, manifest_sha256=manifest_sha, stop_requested=True)
        journal_root = fixture.root / "owned/_dispatcher" / manifest_sha
        journal_root.mkdir(parents=True)
        worker.save_json(journal_root / "SUMMARY.json", summary)
        worker.save_json(
            state_root / (manifest_sha + ".status.json"),
            dict(status="WAITING_FOR_DEPENDENCIES", manifest_sha256=manifest_sha, summary=summary),
        )

        def execute(original_manifest, checksum, *, software_test_only):
            return worker.execute_manifest(
                original_manifest,
                checksum,
                software_test_only=software_test_only,
                injected_factory=lambda _: fixture.backend,
            )

        with mock.patch.object(runner, "process_is_alive", return_value=False):
            result = runner.run_one(
                plan,
                plan_sha,
                manifest_sha=manifest_sha,
                member_index=0,
                gpu="3",
                gpu_lock_root=fixture.root / "admitted-extra-slot",
                resume_stopped=True,
                previous_pid=987654321,
                software_test_only=True,
                execute=execute,
            )
        self.assertEqual(result["status"], "DONE")
        self.assertEqual(result["summary"]["cached"], 1)
        self.assertEqual(result["summary"]["new_successes"], 1)
        self.assertEqual(fixture.backend.calls, 2)
        self.assertEqual(worker.file_sha(full_reference["path"]), manifest_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification outputs are append-only")
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        for test_case in (
            UnusedModeQualification,
            StoppedModeQualification,
            NativeStoreCacheQualification,
        )
    )
    capture = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
    report = dict(
        schema="e18_original_plan_targeted_shard_qualification_v2",
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        wall_s=time.perf_counter() - started,
        source_sha256=worker.file_sha(runner.__file__),
        qualification_source_sha256=worker.file_sha(__file__),
        original_runner_sha256=worker.file_sha(runner.original.__file__),
        original_dispatcher_sha256=worker.file_sha(dispatcher.__file__),
        original_worker_sha256=worker.file_sha(worker.__file__),
        test_log=capture.getvalue(),
        software_fixture_only=True,
        actual_gpu_queries=0,
        signals_sent=0,
        physical_model_calls=0,
        gold_reads=0,
        server_logins=0,
        production_launch_performed=False,
    )
    runner.write_once(args.output, report)
    print(
        json.dumps(
            {key: report[key] for key in ("status", "tests", "failures", "errors", "wall_s")}
        )
    )
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
