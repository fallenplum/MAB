"""Synthetic identity controls for the read-only startup audit."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import time
import unittest

import audit_startup16 as audit
import e18_aggregation_worker_v3 as worker
import e18_aggregation_dispatcher_v3 as dispatcher
import qualify_gpu_worker as fixtures


class StartupAuditQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GPUWorkerQualification.setUpClass()

    def setUp(self):
        worker._PROCESS_BACKEND_CONTEXT = None
        self.fixture = fixtures.GPUWorkerQualification(
            "test_persistent_model_for_multiple_generation_seeds"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.case = self.fixture.fixture
        self.module = fixtures.fixtures.adapter_module
        self.keys = sorted(self.case.adapter.required_aggregates)[:16]
        members = set()
        for qid, seed, combination in self.keys:
            members.update(
                (qid, seed, member) for member in self.module.TEAMS[combination]
            )
        native = {}
        for qid, seed, index in sorted(members):
            receipt = self.module.run_approved_task(
                self.case.adapter,
                dict(
                    stage="member_generation",
                    question_id=qid,
                    generation_seed=seed,
                    member_index=index,
                ),
                self.case.approval_ref,
                self.case.root / "owned",
                lambda spec: fixtures.fixtures.FakeBackend(
                    spec, self.case.apis.inference
                ),
                software_test_only=True,
            )
            native[(qid, seed, index)] = receipt["native_record"]
        self.receipt_paths = []
        for qid, seed, combination in self.keys:
            self.module.run_approved_task(
                self.case.adapter,
                dict(
                    stage="aggregation",
                    question_id=qid,
                    generation_seed=seed,
                    combination_index=combination,
                ),
                self.case.approval_ref,
                self.case.root / "owned",
                lambda spec: fixtures.fixtures.FakeBackend(
                    spec, self.case.apis.inference
                ),
                tokenizer=self.case.tokenizer,
                member_records={
                    index: native[(qid, seed, index)]
                    for index in self.module.TEAMS[combination]
                },
                software_test_only=True,
            )
            owner = (
                self.case.root
                / "owned"
                / self.module.canonical_digest(self.case.scope)
                / "aggregation"
                / self.module.canonical_digest([qid, seed, combination])
            )
            self.receipt_paths.append(owner / "LAST_TASK_RECEIPT.json")
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest.update(schema=worker.MANIFEST_SCHEMA, stage="aggregation")
        manifest.pop("member_index")
        manifest["task_keys_ref"] = self.fixture.artifact(
            "startup16-keys.json", self.keys
        )
        plan = dict(
            schema=dispatcher.PLAN_SCHEMA,
            worker_source_sha256=worker.file_sha(worker.__file__),
            manifest_refs=[self.fixture.artifact("startup16-manifest.json", manifest)],
        )
        self.plan_ref = self.fixture.artifact("startup16-plan.json", plan)
        path = fixtures.PREPARATION / "e18_scoring_adapter.py"
        self.validator_ref = dict(path=str(path), sha256=worker.file_sha(path))

    def run_audit(self, expected_tasks=16):
        return audit.audit_plan(
            self.plan_ref,
            self.validator_ref,
            expected_tasks=expected_tasks,
            software_test_only=True,
            fixture_tokenizer=self.case.tokenizer,
            fixture_factory=lambda _: self.fail("Audit tried to construct a model"),
        )

    def mutate_native(self, change):
        path = self.receipt_paths[0]
        receipt = worker.read_json(path)
        change(receipt)
        row = receipt["native_record"]
        row["record_sha256"] = self.module.canonical_digest(
            {key: value for key, value in row.items() if key != "record_sha256"}
        )
        receipt["native_record_sha256"] = row["record_sha256"]
        path.write_text(json.dumps(receipt), encoding="utf-8")

    def test_valid16_and_exact_typed_prompt_rebuild(self):
        result = self.run_audit()
        self.assertEqual(
            (
                result["status"],
                result["audited_tasks"],
                result["aggregate_physical_attempts"],
            ),
            ("PASS", 16, 16),
        )
        self.assertEqual(result["actual_aggregate_prompt_tokens"], 128)
        self.assertEqual(result["actual_aggregate_completion_tokens"], 64)
        self.assertEqual(result["full_scope_aggregate_keys"], 18)
        self.assertEqual(result["scorer_calls"], 0)
        self.assertNotIn('"prediction":', json.dumps(result))

    def test_outer_operator_mismatch_rejected(self):
        self.mutate_native(lambda row: row.update(aggregator_fingerprint="0" * 64))
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_verified_true_foreign_model_result_rejected(self):
        self.mutate_native(
            lambda row: row["native_record"]["result"].update(model_id="foreign")
        )
        with self.assertRaises(Exception):
            self.run_audit()

    def test_candidate_provenance_mismatch_rejected(self):
        def change(receipt):
            provenance = receipt["native_record"]["member_provenance"]
            provenance[next(iter(provenance))]["record_sha256"] = "0" * 64

        self.mutate_native(change)
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_paid_prompt_job_identity_mismatch_rejected(self):
        def change(receipt):
            receipt["native_job_id"] = "0" * 64
            receipt["native_record"]["job_id"] = "0" * 64

        self.mutate_native(change)
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_completion_budget_overflow_rejected(self):
        self.mutate_native(
            lambda row: row["native_record"]["result"].update(completion_tokens=2049)
        )
        with self.assertRaises(Exception):
            self.run_audit()

    def test_aggregate_decode_seed_mismatch_rejected(self):
        self.mutate_native(lambda row: row["native_record"].update(seed=137))
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_expected_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.run_audit(expected_tasks=15)

    def test_production_rejects_fixture_injection(self):
        with self.assertRaises(PermissionError):
            audit.audit_plan(
                self.plan_ref, self.validator_ref, fixture_tokenizer=self.case.tokenizer
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
        unittest.defaultTestLoader.loadTestsFromTestCase(StartupAuditQualification)
    )
    record = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        audit_source_sha256=worker.file_sha(audit.__file__),
        test_source_sha256=worker.file_sha(__file__),
        actual_model_calls=0,
        actual_gpu_queries=0,
        formal_gold_reads=0,
        scorer_calls=0,
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
