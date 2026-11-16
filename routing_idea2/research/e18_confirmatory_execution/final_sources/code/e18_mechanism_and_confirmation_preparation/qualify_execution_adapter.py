"""Runnable synthetic single-task demo and qualification for the E18 adapter.

The test backend returns in-memory fabricated text and usage. No real model is
constructed, no executable queue is saved, and no real Gold/output is opened.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import e18_execution_adapter as adapter_module
from e18_execution_adapter import (
    E18ExecutionAdapter,
    build_scope_contract,
    canonical_digest,
    canonical_question,
    load_original_apis,
    run_approved_task,
    sha256,
    validate_scope_approval,
)


ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))


class FakeBackend:
    """Software fixture with zero Transformers, CUDA or remote model access."""

    def __init__(self, spec, inference, text=None, failure=None):
        self.spec = spec
        self.inference = inference
        self.text = text
        self.failure = failure
        self.calls = 0

    def generate(self, messages, seed):
        self.calls += 1
        if self.failure:
            raise self.inference.GenerationFailure(
                "synthetic_failure", usage=self.failure
            )
        text = (
            self.text
            if self.text is not None
            else "Synthetic seed " + str(seed) + "\nFinal answer: 1"
        )
        return self.inference.GenerationResult(
            text=text,
            prompt_tokens=8,
            completion_tokens=4,
            wall_time_s=0.001,
            model_id=self.spec.model_id,
            model_revision=self.spec.revision,
            seed=seed,
            backend=self.spec.backend,
            model_identity_verified=True,
        )


class UnknownCostBackend(FakeBackend):
    def generate(self, messages, seed):
        self.calls += 1
        raise self.inference.GenerationFailure("synthetic_unknown_cost")


class FakeTokenizer:
    def __init__(self, spec):
        self.name_or_path = spec.model_path
        self.init_kwargs = {"revision": spec.revision}

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(map(chr, ids))


class ExecutionAdapterQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.apis = load_original_apis(ASSETS)
        pool = adapter_module.read_json(ASSETS / "configs/pool_v1.json")["models"]
        cls.member_specs = {
            row["index"]: cls.apis.inference.ModelSpec(
                **adapter_module.read_json(ASSETS / row["config"])
            )
            for row in pool
        }
        cls.aggregator_spec = cls.apis.inference.ModelSpec(
            **adapter_module.read_json(
                ASSETS
                / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"
            )
        )
        cls.original_splits = set(cls.apis.execution.ALLOWED_SPLITS)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e18-execution-toy-")
        self.root = Path(self.temporary.name).resolve()
        if not self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise AssertionError("Temporary fixture escaped its designated directory")
        self.addCleanup(self.temporary.cleanup)
        self.records = [
            dict(
                question_id="synthetic-gsm",
                task="gsm8k",
                subtask="",
                question="Synthetic arithmetic input.",
                source_frame="synthetic",
            ),
            dict(
                question_id="synthetic-bbh",
                task="bbh",
                subtask="boolean_expressions",
                question="Synthetic boolean input.",
                source_frame="synthetic",
            ),
        ]
        for record in self.records:
            record["canonical_question_sha256"] = canonical_question(record["question"])
        self.actions = [[0, 0], [0, 1], [1, 1], [19, 19]]
        self.bindings = {
            key: canonical_digest(key) for key in adapter_module.REQUIRED_BINDINGS
        }
        self.bindings["target_ids_sha256"] = canonical_digest(
            [[r["question_id"], r["canonical_question_sha256"]] for r in self.records]
        )
        member_keys = [
            (record["question_id"], seed, index)
            for record in self.records
            for seed in adapter_module.GENERATION_SEEDS
            for index in range(6)
        ]
        aggregate_keys = [
            (record["question_id"], seed, team)
            for record in self.records
            for seed in adapter_module.GENERATION_SEEDS
            for team in (0, 1, 19)
        ]
        self.scope = build_scope_contract(
            self.records, self.actions, self.bindings, member_keys, aggregate_keys
        )
        self.adapter = E18ExecutionAdapter(
            self.apis, self.scope, self.records, self.member_specs, self.aggregator_spec
        )
        self.approval = dict(
            schema="e18_scope_approval_v2",
            status="APPROVED",
            cohort="e18_confirm",
            bindings=self.bindings,
            scope_sha256_by_role={"execution": canonical_digest(self.scope)},
            operations=["member_generation", "aggregation"],
            software_test_only=True,
            execution_limits=self.scope["execution_limits"],
            approved_output_root=str((self.root / "owned").resolve()),
        )
        self.approval_ref = self.write_approval(self.approval, "approval.json")
        self.tokenizer = FakeTokenizer(self.aggregator_spec)

    def write_approval(self, value, name):
        path = self.root / name
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream)
        return {"path": str(path), "sha256": sha256(path)}

    def run_member(self, index=0, seed=42, backend=None):
        task = dict(
            stage="member_generation",
            question_id="synthetic-gsm",
            generation_seed=seed,
            member_index=index,
        )
        backend = backend or FakeBackend(self.member_specs[index], self.apis.inference)
        result = run_approved_task(
            self.adapter,
            task,
            self.approval_ref,
            self.root / "owned",
            lambda spec: backend,
            software_test_only=True,
        )
        return result, backend

    def dependencies(self, seed=42, text=None):
        result = {}
        for index in (0, 1, 2):
            backend = FakeBackend(
                self.member_specs[index], self.apis.inference, text=text
            )
            receipt, _ = self.run_member(index, seed, backend)
            result[index] = receipt["native_record"]
        return result

    def test_approval_binds_scope_and_operation(self):
        self.assertEqual(
            validate_scope_approval(
                self.approval_ref,
                self.scope,
                "member_generation",
                software_test_only=True,
            ),
            self.approval,
        )

    def test_missing_approval_prevents_backend(self):
        self.assertRaises(
            FileNotFoundError,
            run_approved_task,
            self.adapter,
            dict(
                stage="member_generation",
                question_id="synthetic-gsm",
                generation_seed=42,
                member_index=0,
            ),
            {"path": str(self.root / "missing.json"), "sha256": "0" * 64},
            self.root / "owned",
            lambda spec: self.fail("Backend was constructed"),
            software_test_only=True,
        )

    def test_synthetic_approval_cannot_authorize_production(self):
        with self.assertRaises(PermissionError):
            validate_scope_approval(self.approval_ref, self.scope, "member_generation")

    def test_unapproved_operation_rejected(self):
        with self.assertRaises(PermissionError):
            validate_scope_approval(
                self.approval_ref, self.scope, "read_reference", software_test_only=True
            )

    def test_changed_binding_rejected(self):
        changed = copy.deepcopy(self.approval)
        changed["bindings"]["budget_sha256"] = "0" * 64
        with self.assertRaises(PermissionError):
            validate_scope_approval(
                self.write_approval(changed, "changed.json"),
                self.scope,
                "aggregation",
                software_test_only=True,
            )

    def test_changed_scope_hash_rejected(self):
        changed = copy.deepcopy(self.approval)
        changed["scope_sha256_by_role"]["execution"] = "0" * 64
        with self.assertRaises(PermissionError):
            validate_scope_approval(
                self.write_approval(changed, "changed.json"),
                self.scope,
                "aggregation",
                software_test_only=True,
            )

    def test_foreign_question_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.member_job("outside", 42, 0)

    def test_unselected_aggregate_key_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.aggregation_job("synthetic-gsm", 42, 2, {}, self.tokenizer)

    def test_native_member_factory_identical(self):
        job = self.adapter.member_job("synthetic-gsm", 137, 0)
        original = self.apis.member.make_member_job(
            question_id="synthetic-gsm",
            task="gsm8k",
            question=self.records[0]["question"],
            subtask=None,
            model_alias=self.adapter.members[0]["alias"],
            seed=137,
            model_spec=self.member_specs[0],
            split="online_stream",
            sample_index=0,
            sampling_control=False,
        )
        self.assertEqual(job, original)

    def test_member_generation_seeds_are_distinct_jobs(self):
        jobs = [
            self.adapter.member_job("synthetic-gsm", seed, 0)
            for seed in adapter_module.GENERATION_SEEDS
        ]
        self.assertEqual(len({job["job_id"] for job in jobs}), 3)
        self.assertEqual([job["seed"] for job in jobs], [42, 137, 2024])

    def test_member_single_task_runs_original_store(self):
        receipt, backend = self.run_member()
        self.assertEqual((receipt["status"], backend.calls), ("SUCCESS", 1))
        self.assertEqual(
            receipt["native_record_sha256"],
            canonical_digest(
                {
                    k: v
                    for k, v in receipt["native_record"].items()
                    if k != "record_sha256"
                }
            ),
        )
        self.assertEqual(
            receipt["physical_attempt_id"], receipt["native_record"]["attempt_id"]
        )

    def test_existing_success_reused_before_backend_construction(self):
        first, _ = self.run_member()
        task = dict(
            stage="member_generation",
            question_id="synthetic-gsm",
            generation_seed=42,
            member_index=0,
        )
        second = run_approved_task(
            self.adapter,
            task,
            self.approval_ref,
            self.root / "owned",
            lambda spec: self.fail("Cache loaded another backend"),
            software_test_only=True,
        )
        self.assertTrue(second["from_cache"])
        self.assertEqual(first["physical_attempt_id"], second["physical_attempt_id"])

    def test_unknown_cost_failure_cannot_retry(self):
        backend = UnknownCostBackend(self.member_specs[0], self.apis.inference)
        result, _ = self.run_member(backend=backend)
        self.assertEqual(result["status"], "FAILED_REQUIRES_RECONCILIATION")
        with self.assertRaises(RuntimeError):
            self.run_member(backend=backend)
        self.assertEqual(backend.calls, 1)

    def test_known_cost_failure_cannot_retry(self):
        backend = FakeBackend(
            self.member_specs[0],
            self.apis.inference,
            failure={"prompt_tokens": 2, "completion_tokens": 1},
        )
        self.run_member(backend=backend)
        with self.assertRaises(RuntimeError):
            self.run_member(backend=backend)
        self.assertEqual(backend.calls, 1)

    def test_backend_initialization_failure_leaves_stop_fence(self):
        task = dict(
            stage="member_generation",
            question_id="synthetic-gsm",
            generation_seed=42,
            member_index=0,
        )

        def fail(spec):
            raise RuntimeError("synthetic initialization failure")

        with self.assertRaises(RuntimeError):
            run_approved_task(
                self.adapter,
                task,
                self.approval_ref,
                self.root / "owned",
                fail,
                software_test_only=True,
            )
        with self.assertRaises(RuntimeError):
            run_approved_task(
                self.adapter,
                task,
                self.approval_ref,
                self.root / "owned",
                lambda spec: self.fail("Backend retried"),
                software_test_only=True,
            )

    def test_aggregate_uses_exactly_selected_three_dependencies(self):
        dependencies = self.dependencies()
        job = self.adapter.aggregation_job(
            "synthetic-gsm", 42, 0, dependencies, self.tokenizer
        )
        self.assertEqual(len(job["members"]), 3)
        self.assertEqual(job["seed"], 42)
        self.assertEqual(job["prompt_version"], "anonymous_final_answer_typed_v3")
        self.apis.aggregation.validate_job(
            job, self.aggregator_spec, require_token_policy=True
        )

    def test_generation137_2024_preserve_original_aggregation_order_seed42(self):
        jobs = [
            self.adapter.aggregation_job(
                "synthetic-gsm", seed, 0, self.dependencies(seed), self.tokenizer
            )
            for seed in (42, 137, 2024)
        ]
        self.assertEqual({job["seed"] for job in jobs}, {42})
        self.assertEqual(jobs[0]["candidate_order"], jobs[1]["candidate_order"])
        self.assertEqual(jobs[1]["candidate_order"], jobs[2]["candidate_order"])
        for job, seed in zip(jobs, (42, 137, 2024)):
            expected = {
                self.adapter.member_job("synthetic-gsm", seed, index)["job_id"]
                for index in (0, 1, 2)
            }
            actual = {
                value["job_id"]
                for key, value in job["member_provenance"].items()
                if key != "output_contract"
            }
            self.assertEqual(actual, expected)

    def test_mixed_generation_dependencies_rejected(self):
        dependencies = self.dependencies(42)
        dependencies[0] = self.dependencies(137)[0]
        with self.assertRaises(ValueError):
            self.adapter.aggregation_job(
                "synthetic-gsm", 42, 0, dependencies, self.tokenizer
            )

    def test_aggregate_single_task_runs_typed_original_store(self):
        dependencies = self.dependencies()
        backend = FakeBackend(self.aggregator_spec, self.apis.inference)
        task = dict(
            stage="aggregation",
            question_id="synthetic-gsm",
            generation_seed=42,
            combination_index=0,
        )
        first = run_approved_task(
            self.adapter,
            task,
            self.approval_ref,
            self.root / "owned",
            lambda spec: backend,
            member_records=dependencies,
            tokenizer=self.tokenizer,
            software_test_only=True,
        )
        second = run_approved_task(
            self.adapter,
            task,
            self.approval_ref,
            self.root / "owned",
            lambda spec: self.fail("Aggregate cache regenerated"),
            member_records=dependencies,
            tokenizer=self.tokenizer,
            software_test_only=True,
        )
        self.assertEqual(first["status"], "SUCCESS")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(first["physical_attempt_id"], second["physical_attempt_id"])

    def test_changed_model_profiles_rejected(self):
        changed = dict(self.member_specs)
        changed[0] = replace(changed[0], temperature=0.0)
        with self.assertRaises(ValueError):
            E18ExecutionAdapter(
                self.apis, self.scope, self.records, changed, self.aggregator_spec
            )
        with self.assertRaises(ValueError):
            E18ExecutionAdapter(
                self.apis,
                self.scope,
                self.records,
                self.member_specs,
                replace(self.aggregator_spec, temperature=0.7),
            )

    def test_raw_input_mutation_rejected(self):
        self.adapter.questions["synthetic-gsm"]["question"] += " "
        with self.assertRaises(ValueError):
            self.adapter.member_job("synthetic-gsm", 42, 0)

    def test_original_global_allowed_splits_unchanged(self):
        self.dependencies()
        self.assertEqual(set(self.apis.execution.ALLOWED_SPLITS), self.original_splits)
        self.assertNotIn("e18_confirm", self.apis.execution.ALLOWED_SPLITS)

    def test_prior_fresh400_member_receipt_bridge(self):
        record = self.run_member()[0]["native_record"]
        source_job = self.adapter.member_job("synthetic-gsm", 42, 0)
        source_job["split"] = "fresh_dev_test"
        source_job["job_id"] = canonical_digest(
            {k: v for k, v in source_job.items() if k != "job_id"}
        )
        changed = copy.deepcopy(record)
        changed["split"] = "fresh_dev_test"
        changed["job_id"] = source_job["job_id"]
        changed["record_sha256"] = canonical_digest(
            {k: v for k, v in changed.items() if k != "record_sha256"}
        )
        self.assertEqual(
            self.adapter.checked_member_dependency("synthetic-gsm", 42, 0, changed),
            changed,
        )

    def test_cache_only_key_is_not_admitted_for_new_call(self):
        remaining = [
            key
            for key in self.scope["member_execution_keys"]
            if key != ["synthetic-gsm", 42, 0]
        ]
        scope = build_scope_contract(
            self.records,
            self.actions,
            self.bindings,
            remaining,
            self.scope["aggregate_execution_keys"],
        )
        instance = E18ExecutionAdapter(
            self.apis, scope, self.records, self.member_specs, self.aggregator_spec
        )
        approval = copy.deepcopy(self.approval)
        approval["scope_sha256_by_role"]["execution"] = canonical_digest(scope)
        ref = self.write_approval(approval, "cache_only.json")
        with self.assertRaises(PermissionError):
            run_approved_task(
                instance,
                dict(
                    stage="member_generation",
                    question_id="synthetic-gsm",
                    generation_seed=42,
                    member_index=0,
                ),
                ref,
                self.root / "owned",
                lambda spec: self.fail("Cache-only key generated"),
                software_test_only=True,
            )

    def test_tokenizer_head_tail_protocol_retained(self):
        dependencies = self.dependencies(
            text="synthetic reasoning " * 100 + "Final answer: 1"
        )
        job = self.adapter.aggregation_job(
            "synthetic-gsm", 42, 0, dependencies, self.tokenizer
        )
        self.assertEqual(job["candidate_token_policy"]["source_token_cap"], 512)
        self.assertTrue(
            all(
                item["retained_source_tokens"] == 512 and item["truncated"]
                for item in job["candidate_statistics"].values()
            )
        )

    def test_insufficient_execution_budget_rejected(self):
        changed = copy.deepcopy(self.approval)
        changed["execution_limits"]["member_generation"] = 0
        ref = self.write_approval(changed, "small_budget.json")
        with self.assertRaises(PermissionError):
            run_approved_task(
                self.adapter,
                dict(
                    stage="member_generation",
                    question_id="synthetic-gsm",
                    generation_seed=42,
                    member_index=0,
                ),
                ref,
                self.root / "owned",
                lambda spec: self.fail("Insufficient budget executed"),
                software_test_only=True,
            )

    def test_corrupt_member_record_rejected(self):
        dependencies = self.dependencies()
        dependencies[0]["result"]["text"] = "changed"
        with self.assertRaises(ValueError):
            self.adapter.aggregation_job(
                "synthetic-gsm", 42, 0, dependencies, self.tokenizer
            )

    def test_duplicate_execution_ownership_rejected(self):
        members = self.scope["member_execution_keys"]
        with self.assertRaises(ValueError):
            build_scope_contract(
                self.records,
                self.actions,
                self.bindings,
                members + [members[0]],
                self.scope["aggregate_execution_keys"],
            )

    def test_runtime_api_load_preserves_cuda_visibility(self):
        before = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "5"
            load_original_apis(ASSETS)
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "5")
        finally:
            if before is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = before

    def test_production_rejects_injected_fake_backend(self):
        approved = copy.deepcopy(self.approval)
        approved["software_test_only"] = False
        reference = self.write_approval(approved, "production_approval.json")
        backend = FakeBackend(self.member_specs[0], self.apis.inference)
        with self.assertRaises(ValueError):
            run_approved_task(
                self.adapter,
                dict(
                    stage="member_generation",
                    question_id="synthetic-gsm",
                    generation_seed=42,
                    member_index=0,
                ),
                reference,
                self.root / "owned",
                lambda spec: backend,
            )
        self.assertEqual(backend.calls, 0)

    def test_native_task_request_rejects_extra_label_field(self):
        task = dict(
            stage="member_generation",
            question_id="synthetic-gsm",
            generation_seed=42,
            member_index=0,
            reference="synthetic-only",
        )
        with self.assertRaises(ValueError):
            run_approved_task(
                self.adapter,
                task,
                self.approval_ref,
                self.root / "owned",
                lambda spec: self.fail("Label-bearing request executed"),
                software_test_only=True,
            )

    def test_same_task_cannot_change_approved_output_root(self):
        self.run_member()
        with self.assertRaises(PermissionError):
            run_approved_task(
                self.adapter,
                dict(
                    stage="member_generation",
                    question_id="synthetic-gsm",
                    generation_seed=42,
                    member_index=0,
                ),
                self.approval_ref,
                self.root / "different-owner",
                lambda spec: self.fail("A second owner constructed a backend"),
                software_test_only=True,
            )

    def test_software_success_cannot_be_reused_as_production(self):
        self.run_member()
        approved = copy.deepcopy(self.approval)
        approved["software_test_only"] = False
        reference = self.write_approval(approved, "production_cache_approval.json")
        with self.assertRaises(RuntimeError):
            run_approved_task(
                self.adapter,
                dict(
                    stage="member_generation",
                    question_id="synthetic-gsm",
                    generation_seed=42,
                    member_index=0,
                ),
                reference,
                self.root / "owned",
                lambda spec: self.fail("Synthetic cache reached production backend"),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("EXECUTION_ADAPTER_QUALIFICATION_v2.json"),
    )
    args = parser.parse_args()
    cpu, wall = time.process_time(), time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        ExecutionAdapterQualification
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise RuntimeError("E18 synthetic execution adapter qualification failed")
    record = dict(
        status="PASS",
        tests=result.testsRun,
        source_sha256=sha256(adapter_module.__file__),
        test_source_sha256=sha256(__file__),
        fake_backend_only=True,
        fake_tokenizer_only=True,
        original_member_store_used=True,
        original_typed_aggregation_store_used=True,
        approval_before_backend_verified=True,
        cache_before_backend_verified=True,
        failed_or_unknown_retry_blocked=True,
        global_dataset_or_split_mutations=0,
        member_generation_seeds=[42, 137, 2024],
        aggregator_decode_and_anonymous_order_seed=42,
        scope_and_single_task_worker_status="SYNTHETICALLY_EXECUTABLE_QUALIFIED",
        native_gpu_runtime_status="NOT_RUNTIME_TESTED",
        full_cohort_native_dispatcher_status="NOT_IMPLEMENTED",
        actual_transformers_backends_constructed=0,
        real_model_calls=0,
        real_gold_reads=0,
        fresh_output_reads=0,
        sealed_payload_reads=0,
        real_jobs_submitted=0,
        cpu_s=time.process_time() - cpu,
        wall_s=time.perf_counter() - wall,
        test_names=[
            test.id().split(".")[-1]
            for test in unittest.defaultTestLoader.loadTestsFromTestCase(
                ExecutionAdapterQualification
            )
        ],
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            dict(
                status="PASS",
                tests=record["tests"],
                native_gpu_runtime_status=record["native_gpu_runtime_status"],
            )
        )
    )


if __name__ == "__main__":
    main()
