"""Pure FakeBackend qualification of the native E18 sparse worker."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import sys
import time
import unittest

import e18_gpu_worker as worker


PREPARATION = (
    Path(__file__).resolve().parent.parent
    / "e18_mechanism_and_confirmation_preparation"
)
sys.path.insert(0, str(PREPARATION))
import qualify_execution_adapter as fixtures


class GPUWorkerQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ExecutionAdapterQualification.setUpClass()

    def setUp(self):
        self.fixture = fixtures.ExecutionAdapterQualification(
            "test_approval_binds_scope_and_operation"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.backend = fixtures.FakeBackend(
            self.fixture.member_specs[0], self.fixture.apis.inference
        )
        self.manifest = dict(
            schema=worker.MANIFEST_SCHEMA,
            software_test_only=True,
            asset_root=str(fixtures.ASSETS),
            adapter_ref=self.reference(PREPARATION / "e18_execution_adapter.py"),
            scope_ref=self.artifact("scope.json", self.fixture.scope),
            question_records_ref=self.input_records(),
            approval_ref=self.fixture.approval_ref,
            owned_output_root=str(self.root / "owned"),
            stage="member_generation",
            member_index=0,
            task_keys_ref=self.artifact(
                "tasks.json", [["synthetic-gsm", 42, 0], ["synthetic-gsm", 137, 0]]
            ),
            backend=dict(mode="original"),
            gpu_lock_root=str(self.root / "gpu-locks"),
            install_signal_handlers=False,
        )

    def reference(self, path):
        return dict(path=str(path), sha256=worker.file_sha(path))

    def artifact(self, name, value):
        path = self.root / name
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream)
        return self.reference(path)

    def input_records(self):
        path = self.root / "questions.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in self.fixture.records:
                stream.write(json.dumps(row) + "\n")
        return self.reference(path)

    def run_worker(self, manifest=None, factory=None, tokenizer=None):
        manifest = manifest or self.manifest
        return worker.execute_manifest(
            manifest,
            fixtures.canonical_digest(manifest),
            software_test_only=True,
            injected_factory=factory or (lambda spec: self.backend),
            injected_tokenizer=tokenizer,
        )

    def aggregation_manifest(self, seeds=(42,)):
        manifest = copy.deepcopy(self.manifest)
        manifest["stage"] = "aggregation"
        manifest.pop("member_index")
        manifest["task_keys_ref"] = self.artifact(
            "aggregate-tasks.json", [["synthetic-gsm", seed, 0] for seed in seeds]
        )
        return manifest

    def test_persistent_model_for_multiple_generation_seeds(self):
        result = self.run_worker()
        self.assertEqual(
            (result["completed"], result["backend_loads"], self.backend.calls),
            (2, 1, 2),
        )

    def test_resume_reuses_success_before_model_load(self):
        self.run_worker()
        result = self.run_worker(
            factory=lambda _: self.fail("Cached resume loaded a model")
        )
        self.assertEqual(
            (result["cached"], result["new_successes"], result["backend_loads"]),
            (2, 0, 0),
        )

    def test_duplicate_keys_rejected(self):
        self.manifest["task_keys_ref"] = self.artifact(
            "duplicates.json", [["synthetic-gsm", 42, 0]] * 2
        )
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_foreign_key_rejected(self):
        self.manifest["task_keys_ref"] = self.artifact(
            "foreign.json", [["foreign", 42, 0]]
        )
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_mixed_profile_rejected(self):
        self.manifest["task_keys_ref"] = self.artifact(
            "mixed.json", [["synthetic-gsm", 42, 1]]
        )
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_changed_question_bytes_rejected(self):
        path = Path(self.manifest["question_records_ref"]["path"])
        path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_changed_approval_rejected_before_backend(self):
        self.manifest["approval_ref"] = dict(
            path=str(self.root / "missing.json"), sha256="0" * 64
        )
        with self.assertRaises(FileNotFoundError):
            self.run_worker()
        self.assertEqual(self.backend.calls, 0)

    def test_production_rejects_injected_backend(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["software_test_only"] = False
        with self.assertRaises(PermissionError):
            worker.load_worker(manifest, injected_factory=lambda _: self.backend)

    def test_gpu_lock_is_exclusive(self):
        with worker.GPUOwnership(self.root / "locks", "0", "a" * 64):
            with self.assertRaises(RuntimeError):
                with worker.GPUOwnership(self.root / "locks", "0", "b" * 64):
                    self.fail("Concurrent GPU owner was admitted")

    def test_gpu_lock_releases_after_process_context(self):
        with worker.GPUOwnership(self.root / "locks", "0", "a" * 64):
            pass
        with worker.GPUOwnership(self.root / "locks", "0", "b" * 64):
            pass

    def test_unknown_cost_failure_stops_resume(self):
        backend = fixtures.UnknownCostBackend(
            self.fixture.member_specs[0], self.fixture.apis.inference
        )
        failed = self.run_worker(factory=lambda _: backend)
        self.assertEqual(len(failed["failures"]), 1)
        with self.assertRaises(RuntimeError):
            self.run_worker(factory=lambda _: self.fail("Unknown-cost owner retried"))
        self.assertEqual(backend.calls, 1)

    def test_profile_shards_are_disjoint_and_same_owner_reuses(self):
        first = copy.deepcopy(self.manifest)
        first["task_keys_ref"] = self.artifact(
            "shard1.json", [["synthetic-gsm", 42, 0]]
        )
        second = copy.deepcopy(self.manifest)
        second["task_keys_ref"] = self.artifact(
            "shard2.json", [["synthetic-gsm", 137, 0]]
        )
        self.assertEqual(self.run_worker(first)["completed"], 1)
        self.assertEqual(self.run_worker(second)["completed"], 1)
        self.assertEqual(self.run_worker()["cached"], 2)
        self.assertEqual(self.backend.calls, 2)

    def test_journal_records_original_actual_usage(self):
        result = self.run_worker()
        path = (
            self.root
            / "owned"
            / "_dispatcher"
            / result["manifest_sha256"]
            / "EVENTS.jsonl"
        )
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        records = [row for row in rows if row["event"] == "TASK_RESULT"]
        self.assertEqual(len(records), 2)
        self.assertEqual(
            records[0]["actual_usage"], dict(prompt_tokens=8, completion_tokens=4)
        )
        self.assertEqual(len({row["physical_attempt_id"] for row in records}), 2)

    def test_aggregate_preserves_generation_dependencies_and_seed42(self):
        for seed in (42, 137, 2024):
            self.fixture.dependencies(seed=seed)
        backend = fixtures.FakeBackend(
            self.fixture.aggregator_spec, self.fixture.apis.inference
        )
        result = self.run_worker(
            self.aggregation_manifest((42, 137, 2024)),
            factory=lambda spec: backend,
            tokenizer=self.fixture.tokenizer,
        )
        self.assertEqual(
            (result["completed"], result["backend_loads"], backend.calls), (3, 1, 3)
        )

    def test_aggregate_waits_for_three_member_dependencies(self):
        result = self.run_worker(
            self.aggregation_manifest(),
            factory=lambda _: self.fail("Incomplete dependency loaded a model"),
            tokenizer=self.fixture.tokenizer,
        )
        self.assertEqual((result["completed"], result["backend_loads"]), (0, 0))

    def test_aggregate_external_member_cache_index(self):
        dependencies = self.fixture.dependencies(seed=137)
        manifest = self.aggregation_manifest((137,))
        index = [
            dict(
                task_key=["synthetic-gsm", 137, member],
                record_ref=self.artifact(
                    "external-member-" + str(member) + ".json", row
                ),
            )
            for member, row in dependencies.items()
        ]
        manifest["member_dependencies_ref"] = self.artifact(
            "external-index.json", index
        )
        backend = fixtures.FakeBackend(
            self.fixture.aggregator_spec, self.fixture.apis.inference
        )
        result = self.run_worker(
            manifest, factory=lambda _: backend, tokenizer=self.fixture.tokenizer
        )
        self.assertEqual(result["completed"], 1)

    def test_aggregate_rejects_wrong_generation_external_record(self):
        dependencies = self.fixture.dependencies(seed=42)
        manifest = self.aggregation_manifest((137,))
        index = [
            dict(
                task_key=["synthetic-gsm", 137, member],
                record_ref=self.artifact("wrong-seed-" + str(member) + ".json", row),
            )
            for member, row in dependencies.items()
        ]
        manifest["member_dependencies_ref"] = self.artifact(
            "wrong-generation-index.json", index
        )
        with self.assertRaises(ValueError):
            self.run_worker(
                manifest,
                factory=lambda _: self.fail("Wrong seed loaded a model"),
                tokenizer=self.fixture.tokenizer,
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
        unittest.defaultTestLoader.loadTestsFromTestCase(GPUWorkerQualification)
    )
    record = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        worker_source_sha256=worker.file_sha(worker.__file__),
        test_source_sha256=worker.file_sha(__file__),
        adapter_sha256=worker.ADAPTER_SHA,
        actual_model_calls=0,
        real_gold_reads=0,
        actual_jobs_submitted=0,
        native_gpu_runtime_status="NOT_RUNTIME_TESTED",
        wall_seconds=time.perf_counter() - wall,
        test_output=stream.getvalue(),
    )
    worker.save_json(args.output, record)
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("status", "tests", "native_gpu_runtime_status")
            }
        )
    )
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
