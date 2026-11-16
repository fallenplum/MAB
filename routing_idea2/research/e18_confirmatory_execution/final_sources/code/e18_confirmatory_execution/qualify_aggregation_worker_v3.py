"""Synthetic qualification of the isolated tokenizer loading-path alias."""

from __future__ import annotations

import argparse
import copy
import io
import inspect
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import e18_aggregation_dispatcher_v3 as dispatcher
import e18_aggregation_worker_v3 as worker
import e18_aggregation_worker_v2 as previous_worker
import qualify_gpu_worker as fixtures


class AggregationQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GPUWorkerQualification.setUpClass()

    def setUp(self):
        worker._PROCESS_BACKEND_CONTEXT = None
        self.fixture = fixtures.GPUWorkerQualification(
            "test_aggregate_waits_for_three_member_dependencies"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.spec = self.fixture.fixture.aggregator_spec
        self.physical = self.root / "physical-tokenizer"
        self.physical.mkdir()
        self.files = {}
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ):
            path = self.physical / name
            path.write_text("Synthetic tokenizer identity " + name, encoding="utf-8")
            self.files[name] = worker.file_sha(path)
        self.tokenizer = fixtures.fixtures.FakeTokenizer(self.spec)
        self.tokenizer.name_or_path = str(self.physical)
        self.receipt = dict(
            path=str(self.physical),
            repo_id=self.spec.model_id,
            revision=self.spec.revision,
            completed_at_unix=1,
        )
        self.execution = self.fixture.fixture.apis.execution

    def alias(self):
        return worker.restore_logical_tokenizer_path(
            self.tokenizer, self.spec, self.receipt, self.files, self.execution
        )

    def test_alias_keeps_same_class_methods_revision_and_behavior(self):
        original_class = type(self.tokenizer)
        original_init = copy.deepcopy(self.tokenizer.init_kwargs)
        encode = self.tokenizer.encode.__func__
        result = self.alias()
        self.assertIs(type(self.tokenizer), original_class)
        self.assertIs(self.tokenizer.encode.__func__, encode)
        self.assertEqual(self.tokenizer.init_kwargs, original_init)
        self.assertEqual(self.tokenizer.name_or_path, self.spec.model_path)
        self.assertEqual(result["encode_decode_head_tail_probes"], 4)
        self.assertGreater(max(result["probe_source_token_counts"]), 512)

    def test_changed_tokenizer_bytes_fail_before_alias(self):
        (self.physical / "vocab.json").write_text("Changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.alias()
        self.assertEqual(self.tokenizer.name_or_path, str(self.physical))

    def test_receipt_repository_must_match(self):
        self.receipt["repo_id"] = "foreign"
        with self.assertRaises(ValueError):
            self.alias()

    def test_receipt_revision_must_match(self):
        self.receipt["revision"] = "0" * 40
        with self.assertRaises(ValueError):
            self.alias()

    def test_instance_revision_must_match(self):
        self.tokenizer.init_kwargs["revision"] = "0" * 40
        with self.assertRaises(ValueError):
            self.alias()

    def test_physical_tokenizer_name_must_match_receipt(self):
        self.tokenizer.name_or_path = str(self.root)
        with self.assertRaises(ValueError):
            self.alias()

    def test_exact_four_file_contract_required(self):
        self.files.pop("merges.txt")
        with self.assertRaises(ValueError):
            self.alias()

    def test_completed_source_receipt_required(self):
        self.receipt["completed_at_unix"] = None
        with self.assertRaises(ValueError):
            self.alias()

    def test_original_factory_rejects_physical_then_accepts_alias(self):
        dependencies = self.fixture.fixture.dependencies(seed=137)
        adapter = self.fixture.fixture.adapter
        with self.assertRaises(ValueError):
            adapter.aggregation_job(
                "synthetic-gsm", 137, 0, dependencies, self.tokenizer
            )
        original = adapter.aggregation_job(
            "synthetic-gsm", 137, 0, dependencies, self.fixture.fixture.tokenizer
        )
        self.alias()
        restored = adapter.aggregation_job(
            "synthetic-gsm", 137, 0, dependencies, self.tokenizer
        )
        self.assertEqual(original, restored)

    def test_aggregation_worker_rejects_member_stage(self):
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["schema"] = worker.MANIFEST_SCHEMA
        with self.assertRaises(ValueError):
            worker.load_worker(manifest, software_test_only=True)

    def test_complete_worker_aliases_before_original_job_validation(self):
        self.fixture.fixture.dependencies(seed=2024)
        manifest = self.fixture.aggregation_manifest((2024,))
        manifest["schema"] = worker.MANIFEST_SCHEMA
        backend = fixtures.fixtures.FakeBackend(
            self.spec, self.fixture.fixture.apis.inference
        )
        backend.tokenizer = self.tokenizer
        backend.source_receipt = self.receipt
        with patch.object(worker, "frozen_tokenizer_files", return_value=self.files):
            result = worker.execute_manifest(
                manifest,
                fixtures.fixtures.canonical_digest(manifest),
                software_test_only=True,
                injected_factory=lambda _: backend,
            )
        self.assertEqual(
            (result["completed"], backend.calls, result["backend_loads"]), (1, 1, 1)
        )
        receipt_path = (
            self.root
            / "owned"
            / "_dispatcher"
            / result["manifest_sha256"]
            / "TOKENIZER_LOADING_PATH_RECEIPT.json"
        )
        self.assertEqual(worker.read_json(receipt_path)["status"], "PASS")

    def test_aggregation_dispatcher_binds_new_worker(self):
        manifest = self.fixture.aggregation_manifest()
        manifest["schema"] = worker.MANIFEST_SCHEMA
        plan = dict(
            schema=dispatcher.PLAN_SCHEMA,
            worker_source_sha256=worker.file_sha(worker.__file__),
            manifest_refs=[self.fixture.artifact("agg-manifest.json", manifest)],
        )
        self.assertEqual(len(dispatcher.admit_plan(plan)), 1)

    def test_active_member_sources_unchanged(self):
        directory = Path(__file__).parent
        self.assertEqual(
            worker.file_sha(directory / "e18_gpu_worker.py"),
            "17f74192c3ccf10930633f3f1dac48487ce9f62b25c848e2703f03fc6bc9c4f0",
        )
        self.assertEqual(
            worker.file_sha(directory / "e18_gpu_dispatcher.py"),
            "5d273799e2c74f0b5cbd40aa6a6c6ae9555fa9892672bc47f3cba174b9c481a2",
        )

    def two_manifests(self):
        self.fixture.fixture.dependencies(seed=42)
        self.fixture.fixture.dependencies(seed=137)
        first = self.fixture.aggregation_manifest((42,))
        first["schema"] = worker.MANIFEST_SCHEMA
        second = copy.deepcopy(first)
        second["task_keys_ref"] = self.fixture.artifact(
            "second-shard-tasks.json", [["synthetic-gsm", 137, 0]]
        )
        return first, second

    def execute(self, manifest, backend):
        with patch.object(worker, "frozen_tokenizer_files", return_value=self.files):
            return worker.execute_manifest(
                manifest,
                fixtures.fixtures.canonical_digest(manifest),
                software_test_only=True,
                injected_factory=lambda _: backend,
            )

    def fake_backend(self):
        backend = fixtures.fixtures.FakeBackend(
            self.spec, self.fixture.fixture.apis.inference
        )
        backend.tokenizer = self.tokenizer
        backend.source_receipt = self.receipt
        return backend

    def test_two_shards_construct_one_backend(self):
        first, second = self.two_manifests()
        backend = self.fake_backend()
        one = self.execute(first, backend)
        with patch.object(worker, "frozen_tokenizer_files", return_value=self.files):
            two = worker.execute_manifest(
                second,
                fixtures.fixtures.canonical_digest(second),
                software_test_only=True,
                injected_factory=lambda _: self.fail(
                    "Second shard reconstructed the backend"
                ),
            )
        self.assertEqual(
            (one["backend_loads"], two["backend_loads"], backend.calls), (1, 0, 2)
        )
        self.assertTrue(two["cross_shard_backend_reused"])
        self.assertEqual(two["process_backend_constructions"], 1)

    def test_successful_cached_shard_makes_no_additional_generation(self):
        first, second = self.two_manifests()
        backend = self.fake_backend()
        self.execute(first, backend)
        self.execute(second, backend)
        cached = self.execute(second, backend)
        self.assertEqual(
            (
                cached["cached"],
                cached["new_successes"],
                cached["backend_loads"],
                backend.calls,
            ),
            (1, 0, 0, 2),
        )

    def test_changed_scope_reference_requires_new_process(self):
        first, second = self.two_manifests()
        backend = self.fake_backend()
        self.execute(first, backend)
        second["scope_ref"] = self.fixture.artifact(
            "copied-scope.json", self.fixture.fixture.scope
        )
        with self.assertRaises(ValueError):
            self.execute(second, backend)
        self.assertEqual(backend.calls, 1)

    def test_changed_backend_loading_path_requires_new_process(self):
        first, second = self.two_manifests()
        backend = self.fake_backend()
        self.execute(first, backend)
        second["backend"]["physical_root"] = "/different-loading-root"
        with self.assertRaises(ValueError):
            self.execute(second, backend)
        self.assertEqual(backend.calls, 1)

    def test_changed_tokenizer_bytes_between_shards_rejected(self):
        first, second = self.two_manifests()
        backend = self.fake_backend()
        self.execute(first, backend)
        (self.physical / "merges.txt").write_text(
            "changed between shards", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            self.execute(second, backend)
        self.assertEqual(backend.calls, 1)

    def test_foreign_profile_rejected_before_backend_creation(self):
        first, _ = self.two_manifests()
        module, adapter, _, _, persistent, _ = worker.load_worker(
            first,
            software_test_only=True,
            injected_factory=lambda _: self.fail("Foreign profile loaded"),
        )
        with self.assertRaises(ValueError):
            persistent(adapter.member_specs[0])

    def test_tokenizer_alias_function_unchanged_from_v2(self):
        self.assertEqual(
            inspect.getsource(worker.restore_logical_tokenizer_path),
            inspect.getsource(previous_worker.restore_logical_tokenizer_path),
        )

    def test_production_still_rejects_injected_factory(self):
        manifest = self.fixture.aggregation_manifest()
        manifest["schema"] = worker.MANIFEST_SCHEMA
        manifest["software_test_only"] = False
        with self.assertRaises(PermissionError):
            worker.load_worker(manifest, injected_factory=lambda _: self.fake_backend())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification outputs are append-only")
    wall = time.perf_counter()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AggregationQualification)
    )
    record = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        worker_source_sha256=worker.file_sha(worker.__file__),
        dispatcher_source_sha256=worker.file_sha(dispatcher.__file__),
        test_source_sha256=worker.file_sha(__file__),
        adapter_sha256=worker.ADAPTER_SHA,
        actual_model_calls=0,
        real_gold_reads=0,
        actual_jobs_submitted=0,
        real_hf_tokenizer_test_status="ROOT_PENDING",
        fake_tokenizer_only=True,
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
