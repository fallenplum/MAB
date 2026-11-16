"""Mock-only entry-order qualification with zero production resource access."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_member4_fine_cooperative as runner


class RunnerQualification(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e18_cooperative_runner_qualification_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = self.root / "fine.json"
        self.plan.write_text(json.dumps({"schema": "synthetic fine plan"}), encoding="utf-8")
        self.args = SimpleNamespace(
            plan=self.plan, plan_sha256=runner.tail.base.file_sha(self.plan),
            dispatcher=self.root / "e18_member4_dispatcher_v3.py", gpu="0", gpu_slot=1,
            resource_policy=self.root / "synthetic_policy.json", resource_policy_sha256="a" * 64,
        )
        self.worker = SimpleNamespace(_ACTIVE_GPU_SLOT_CONTEXT=None)
        self.dispatcher = SimpleNamespace(worker=self.worker)
        self.manifest = {"source": "synthetic manifest identity"}
        self.calls = []
        self.execute_error = None

    def patches(self):
        fixture = self

        def load(path):
            fixture.calls.append("load_runtime")
            return fixture.dispatcher

        def admit(plan, sha, dispatcher):
            fixture.calls.append("admit_metadata")
            fixture.assertIs(dispatcher, fixture.dispatcher)
            fixture.assertIsNone(dispatcher.worker._ACTIVE_GPU_SLOT_CONTEXT)
            return [(dict(sha256="b" * 64), fixture.manifest)], {"synthetic": True}

        class MockContext:
            def __init__(self, worker, manifest, checksum, gpu, slot, policy):
                fixture.assertIs(worker, fixture.worker)
                fixture.assertIs(manifest, fixture.manifest)
                fixture.assertEqual((checksum, gpu, slot), (fixture.args.plan_sha256, "0", 1))

            def __enter__(self):
                fixture.calls.append("context_enter")
                fixture.worker._ACTIVE_GPU_SLOT_CONTEXT = {"synthetic": True}

            def __exit__(self, *args):
                fixture.calls.append("context_exit")
                fixture.worker._ACTIVE_GPU_SLOT_CONTEXT = None

        def execute(plan, sha, dispatcher):
            fixture.calls.append("execute_fine_plan")
            fixture.assertIs(dispatcher, fixture.dispatcher)
            fixture.assertIsNotNone(dispatcher.worker._ACTIVE_GPU_SLOT_CONTEXT)
            if fixture.execute_error is not None:
                raise fixture.execute_error
            return dict(status="DONE", original_worker_object_preserved=True)

        return (
            patch.object(runner.tail, "load_runtime", side_effect=load),
            patch.object(runner.tail, "admit_fine_plan", side_effect=admit),
            patch.object(runner.resources, "CooperativeSlotAdmission", MockContext),
            patch.object(runner.tail, "execute_fine_plan", side_effect=execute),
        )

    def run_mocked(self):
        first, second, third, fourth = self.patches()
        with first, second, third, fourth:
            return runner.run_cooperative(self.args)

    def test_metadata_precedes_context_and_execution_uses_same_worker(self):
        self.assertEqual(self.run_mocked()["status"], "DONE")
        self.assertEqual(self.calls, ["load_runtime", "admit_metadata", "context_enter", "execute_fine_plan", "context_exit"])
        self.assertIsNone(self.worker._ACTIVE_GPU_SLOT_CONTEXT)

    def test_execute_exception_releases_original_context(self):
        self.execute_error = RuntimeError("synthetic execution failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic execution failure"):
            self.run_mocked()
        self.assertEqual(self.calls[-1], "context_exit")
        self.assertIsNone(self.worker._ACTIVE_GPU_SLOT_CONTEXT)

    def test_changed_plan_rejected_before_resource_context(self):
        self.args.plan_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            self.run_mocked()
        self.assertEqual(self.calls, [])

    def test_control_source_mismatch_rejected(self):
        with patch.object(runner, "CONTROL_SOURCE_SHA", "0" * 64):
            with self.assertRaises(ValueError):
                self.run_mocked()
        self.assertEqual(self.calls, [])

    def test_resource_source_mismatch_rejected(self):
        with patch.object(runner, "COOPERATIVE_SOURCE_SHA", "0" * 64):
            with self.assertRaises(ValueError):
                self.run_mocked()
        self.assertEqual(self.calls, [])

    def test_invalid_gpu_rejected_before_admission(self):
        self.args.gpu = "automatic"
        with self.assertRaises(ValueError):
            self.run_mocked()
        self.assertEqual(self.calls, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification receipts are append-only")
    started = time.perf_counter()
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RunnerQualification))
    record = dict(
        schema="e18_cooperative_fine_runner_qualification_v1", status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures), wall_seconds=time.perf_counter() - started,
        runner_source_sha256=runner.tail.base.file_sha(runner.__file__),
        qualification_source_sha256=runner.tail.base.file_sha(__file__),
        frozen_control_source_sha256=runner.tail.base.file_sha(runner.tail.__file__),
        cooperative_source_sha256=runner.tail.base.file_sha(runner.resources.__file__),
        actual_gpu_queries=0, actual_process_signals=0, actual_model_calls=0,
        real_gold_reads=0, production_admissions=0, test_output=capture.getvalue(),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: record[key] for key in ("status", "tests", "errors", "failures", "wall_seconds")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
