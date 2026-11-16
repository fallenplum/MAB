"""Toy-only end-to-end qualification of E18 gated sparse scoring and costs."""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

import e18_execution_adapter as execution
from e18_scoring_adapter import (
    SparseScoringAdapter, build_scoring_scope, canonical_bytes, digest,
    resource_ledger_from_execution_receipts,
)
from qualify_execution_adapter import FakeBackend, FakeTokenizer, UnknownCostBackend
from sparse_policy_projection import project_frozen_policy_scores


ASSET_ROOT = Path("E:/语言模型组合路由")


class ScoringAdapterQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from e18_reference_adapter import E18ReferenceProvider, build_reference_scope
        cls.reference_type = E18ReferenceProvider
        cls.build_reference_scope = staticmethod(build_reference_scope)
        cls.apis = execution.load_original_apis(ASSET_ROOT)
        pool = execution.read_json(ASSET_ROOT / "configs/pool_v1.json")["models"]
        cls.member_specs = {row["index"]: cls.apis.inference.ModelSpec(**execution.read_json(ASSET_ROOT / row["config"])) for row in pool}
        cls.aggregator_spec = cls.apis.inference.ModelSpec(**execution.read_json(
            ASSET_ROOT / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e18-scoring-toy-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = [
            dict(question_id="synthetic-score-gsm", task="gsm8k", subtask=None,
                question="Synthetic arithmetic input for scoring only.", source_split="train"),
            dict(question_id="synthetic-score-bbh", task="bbh", subtask="boolean_expressions",
                question="Synthetic boolean input for scoring only.", source_split="test"),
        ]
        for row in self.records:
            row["canonical_question_sha256"] = execution.canonical_question(row["question"])
        self.actions = [[0, 0], [1, 1], [0, 1], [0, 0]]
        bindings = {key: digest(key) for key in execution.REQUIRED_BINDINGS}
        bindings["target_ids_sha256"] = digest([[row["question_id"], row["canonical_question_sha256"]] for row in self.records])
        member_keys = [(row["question_id"], seed, member) for row in self.records for seed in execution.GENERATION_SEEDS for member in (0, 1, 2, 3)]
        aggregate_keys = [(row["question_id"], seed, action) for row in self.records for seed in execution.GENERATION_SEEDS for action in (0, 1)]
        self.execution_scope = execution.build_scope_contract(self.records, self.actions, bindings, member_keys, aggregate_keys)
        reference_records, sources = [], []
        self.reference_version = "synthetic_reference_v2"
        for index, row in enumerate(self.records):
            reference = "4" if row["task"] == "gsm8k" else "True"
            payload = {"question_id": row["question_id"], "task": row["task"], "subtask": row["subtask"],
                "source_split": row["source_split"], "question": row["question"], "reference": reference}
            path = self.root / ("toy-reference-" + str(index) + ".jsonl")
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            source_id = "synthetic-source-" + str(index)
            sources.append(dict(source_id=source_id, format="normalized_qid_jsonl_v1", path=str(path),
                sha256=execution.sha256(path), task=row["task"], source_split=row["source_split"], subtask=row["subtask"]))
            reference_records.append(dict(question_id=row["question_id"], canonical_question_sha256=row["canonical_question_sha256"],
                task=row["task"], subtask=row["subtask"], source_split=row["source_split"], source_id=source_id, row_index=0))
        exclusions = [dict(question_id="synthetic-sealed-exclusion", canonical_question_sha256=digest("unrelated synthetic sealed input"))]
        self.reference_scope = self.build_reference_scope(reference_records, sources, bindings, self.reference_version,
            exclusions, allow_structural_byte_scan=True)
        self.scoring_scope = build_scoring_scope(self.execution_scope, self.reference_scope)
        self.approval = dict(schema="e18_scope_approval_v2", status="APPROVED", cohort="e18_confirm",
            bindings=self.scoring_scope["bindings"], scope_sha256_by_role={
                "execution": digest(self.execution_scope), "reference": digest(self.reference_scope), "scoring": digest(self.scoring_scope)},
            operations=sorted(execution.OPERATIONS), software_test_only=True,
            execution_limits=self.execution_scope["execution_limits"],
            approved_output_root=str(self.root / "executions"))
        self.approval_ref = self.write_approval(self.approval)
        self.provider = self.reference_type(self.reference_scope, self.approval_ref, software_test_only=True)
        self.scoring = self.scoring_adapter()
        self.addCleanup(self.scoring.close)
        self.executor = execution.E18ExecutionAdapter(self.apis, self.execution_scope, self.records, self.member_specs, self.aggregator_spec)
        self.tokenizer = FakeTokenizer(self.aggregator_spec)
        self.execution_receipts = []

    def write_approval(self, approval, name="approval.json"):
        path = self.root / name
        path.write_bytes(canonical_bytes(approval))
        return dict(path=str(path), sha256=execution.sha256(path))

    def scoring_adapter(self, *, approval_ref=None, scope=None, provider=None, name="main", software_test_only=True):
        return SparseScoringAdapter(approval_ref or self.approval_ref, scope or self.scoring_scope,
            provider or self.provider, self.root / (name + ".sqlite3"), self.root / (name + "-receipts"),
            software_test_only=software_test_only, scorer_source_root=ASSET_ROOT / "src")

    def receipt(self, question=0, seed=42, action=0):
        row = self.records[question]
        dependencies = {}
        for member in execution.TEAMS[action]:
            task = dict(stage="member_generation", question_id=row["question_id"], generation_seed=seed, member_index=member)
            backend = FakeBackend(self.member_specs[member], self.apis.inference)
            receipt = execution.run_approved_task(self.executor, task, self.approval_ref, self.root / "executions",
                lambda spec, backend=backend: backend, software_test_only=True)
            self.execution_receipts.append(receipt)
            dependencies[member] = receipt["native_record"]
        if row["task"] == "gsm8k":
            answer = "Final answer: 4" if action == 0 else "Final answer: 5"
        else:
            answer = "#### Reasoning\nFinal answer: True" if action == 1 else "Final answer: False"
        backend = FakeBackend(self.aggregator_spec, self.apis.inference, text=answer)
        task = dict(stage="aggregation", question_id=row["question_id"], generation_seed=seed, combination_index=action)
        receipt = execution.run_approved_task(self.executor, task, self.approval_ref, self.root / "executions",
            lambda spec: backend, member_records=dependencies, tokenizer=self.tokenizer, software_test_only=True)
        self.execution_receipts.append(receipt)
        return receipt

    def test_native_fakebackend_to_original_scorer_to_sparse_projector(self):
        sparse = [self.scoring.score_one(self.receipt(question, seed, action), self.reference_version)["sparse_row"]
            for question in range(2) for seed in execution.GENERATION_SEEDS for action in (0, 1)]
        output = project_frozen_policy_scores(["Ridge", "SPO", "TSPO", "GlobalFixed"], np.asarray(self.actions),
            [row["question_id"] for row in self.records], [row["canonical_question_sha256"] for row in self.records], sparse)
        expected = np.asarray([[[1, 0]] * 3, [[0, 1]] * 3, [[1, 1]] * 3, [[1, 0]] * 3], dtype=float)
        np.testing.assert_array_equal(output["rewards"], expected)
        self.assertEqual(output["unique_required_scored_keys"], 12)
        self.assertEqual(self.scoring.ledger()["confirmed_reference_keys_in_scoring_journal"], 2)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 2)

    def test_duplicate_service_policy_reuses_scored_key_without_reference_or_tokens(self):
        receipt = self.receipt()
        first = self.scoring.score_one(receipt, self.reference_version)
        before = self.provider.stats()
        second = self.scoring.score_one(receipt, self.reference_version)
        self.assertFalse(first["from_scoring_cache"])
        self.assertTrue(second["from_scoring_cache"])
        self.assertEqual(first["sparse_row"], second["sparse_row"])
        self.assertEqual(self.provider.stats(), before)
        ledger = self.scoring.ledger()
        self.assertEqual(ledger["physical_aggregation_sources"], 1)
        self.assertEqual(ledger["actual_source_input_tokens"], 8)
        self.assertEqual(ledger["actual_source_output_tokens"], 4)
        self.assertEqual(ledger["successful_sparse_keys"], 1)

    def test_same_question_different_actions_reuse_one_reference(self):
        self.scoring.score_one(self.receipt(action=0), self.reference_version)
        self.scoring.score_one(self.receipt(action=1), self.reference_version)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 1)
        self.assertEqual(self.scoring.ledger()["confirmed_reference_keys_in_scoring_journal"], 1)
        self.assertEqual(self.scoring.ledger()["physical_aggregation_sources"], 2)

    def test_synthetic_approval_is_rejected_for_production(self):
        receipt = self.receipt()
        scoring = self.scoring_adapter(name="production-denied", software_test_only=False)
        self.addCleanup(scoring.close)
        with self.assertRaises(PermissionError):
            scoring.score_one(receipt, self.reference_version)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 0)

    def test_missing_approval_denies_reference_access(self):
        receipt = self.receipt()
        scoring = self.scoring_adapter(name="missing-approval", approval_ref={"path": str(self.root / "absent.json"), "sha256": "0" * 64})
        self.addCleanup(scoring.close)
        with self.assertRaises(FileNotFoundError):
            scoring.score_one(receipt, self.reference_version)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 0)

    def test_unapproved_scoring_operation_denies_reference_access(self):
        receipt = self.receipt()
        approval = copy.deepcopy(self.approval)
        approval["operations"].remove("score_outputs")
        scoring = self.scoring_adapter(name="operation-denied", approval_ref=self.write_approval(approval, "operation-denied.json"))
        self.addCleanup(scoring.close)
        with self.assertRaises(PermissionError):
            scoring.score_one(receipt, self.reference_version)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 0)

    def test_unselected_combination_is_denied_before_reference(self):
        receipt = self.receipt()
        receipt["task_key"]["combination_index"] = 2
        receipt["task_key"]["team"] = list(execution.TEAMS[2])
        with self.assertRaises((PermissionError, ValueError)):
            self.scoring.score_one(receipt, self.reference_version)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 0)

    def test_wrong_canonical_identity_is_denied_before_reference(self):
        receipt = self.receipt()
        receipt["task_key"]["canonical_id"] = "0" * 64
        with self.assertRaises(PermissionError):
            self.scoring.score_one(receipt, self.reference_version)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 0)

    def test_wrong_execution_scope_is_denied_before_reference(self):
        receipt = self.receipt()
        receipt["scope_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.scoring.score_one(receipt, self.reference_version)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 0)

    def test_reference_version_mismatch_is_journaled_without_decoding(self):
        receipt = self.receipt()
        with self.assertRaises((PermissionError, ValueError)):
            self.scoring.score_one(receipt, "unapproved_reference_version")
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 0)
        self.assertEqual(self.scoring.ledger()["attempts"][-1]["status"], "FAILED_REQUIRES_REVIEW")

    def test_tampered_native_record_is_denied_before_reference(self):
        receipt = self.receipt()
        receipt["native_record"]["result"]["completion_tokens"] += 1
        with self.assertRaises(ValueError):
            self.scoring.score_one(receipt, self.reference_version)
        self.assertEqual(self.provider.stats()["reference_values_decoded"], 0)

    def test_immutable_scoring_receipt_and_resume(self):
        receipt = self.receipt()
        output = self.scoring.score_one(receipt, self.reference_version)
        path = self.root / "main-receipts" / (output["sparse_row"]["scoring_receipt_sha256"] + ".json")
        self.assertEqual(execution.sha256(path), output["sparse_row"]["scoring_receipt_sha256"])
        resumed = self.scoring_adapter()
        self.addCleanup(resumed.close)
        self.assertTrue(resumed.score_one(receipt, self.reference_version)["from_scoring_cache"])
        path.write_bytes(b"synthetic corruption")
        with self.assertRaises(ValueError):
            resumed.score_one(receipt, self.reference_version)

    def test_failed_reference_delivery_preserves_failure_and_provider_cost(self):
        receipt = self.receipt()
        provider = self.provider
        class DeliveryFailure:
            scope = provider.scope
            def get_reference(self, *args):
                provider.get_reference(*args)
                raise OSError("Synthetic failure after toy reference decoding")
            def stats(self):
                return provider.stats()
        failed = self.scoring_adapter(provider=DeliveryFailure(), name="failure")
        self.addCleanup(failed.close)
        with self.assertRaises(OSError):
            failed.score_one(receipt, self.reference_version)
        ledger = failed.ledger()
        self.assertEqual(ledger["attempts"][-1]["status"], "FAILED_REQUIRES_REVIEW")
        self.assertTrue(ledger["attempts"][-1]["gold_access_may_have_occurred"])
        self.assertEqual(ledger["reference_provider_stats"]["reference_values_decoded"], 1)
        with self.assertRaises(RuntimeError):
            failed.score_one(receipt, self.reference_version)

    def test_resource_ledger_deduplicates_member_and_aggregate_actuals(self):
        self.receipt(action=0)
        self.receipt(action=1)
        ledger = resource_ledger_from_execution_receipts(self.execution_receipts + self.execution_receipts)
        self.assertEqual(ledger["member_attempts"], 4)
        self.assertEqual(ledger["aggregation_attempts"], 2)
        self.assertEqual(ledger["unique_physical_attempts"], 6)
        self.assertEqual(ledger["total_actual_tokens"], 6 * 12)

    def test_failed_unknown_execution_usage_remains_null(self):
        task = dict(stage="member_generation", question_id=self.records[0]["question_id"], generation_seed=137, member_index=0)
        backend = UnknownCostBackend(self.member_specs[0], self.apis.inference)
        failure = execution.run_approved_task(self.executor, task, self.approval_ref, self.root / "executions",
            lambda spec: backend, software_test_only=True)
        ledger = resource_ledger_from_execution_receipts([failure])
        self.assertEqual(ledger["failed_attempts"], 1)
        self.assertEqual(ledger["unknown_usage_attempts"], 1)
        self.assertIsNone(ledger["total_actual_tokens"])


def main():
    global ASSET_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ASSET_ROOT = args.asset_root
    if args.output.exists():
        raise FileExistsError("Qualification output refuses overwrite")
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScoringAdapterQualification)
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
    report = dict(schema="e18_scoring_adapter_qualification_v2", status="PASS" if result.wasSuccessful() else "FAIL",
        checks_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        elapsed_s=time.perf_counter() - started,
        adapter_source_sha256=execution.sha256(Path(__file__).with_name("e18_scoring_adapter.py")),
        qualification_source_sha256=execution.sha256(__file__),
        execution_adapter_source_sha256=execution.sha256(Path(__file__).with_name("e18_execution_adapter.py")),
        reference_adapter_source_sha256=execution.sha256(Path(__file__).with_name("e18_reference_adapter.py")),
        original_scorer_sha256="6fc1d2f3cc81b3e954350143abd1071d6556e1cfe4f321d9333f2016baf2150d",
        qualification_data="synthetic question/reference files and original Stores with FakeBackend/FakeTokenizer",
        checks_log=capture.getvalue(), real_fresh_reference_reads=0, real_sealed_reference_reads=0,
        real_model_inference_calls=0, gpu_runtime_status="NOT_RUNTIME_TESTED",
        real_dataset_runtime_status="NOT_RUNTIME_TESTED", confidence_intervals=0, bootstrap_runs=0)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "checks_run", "failures", "errors", "elapsed_s")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
