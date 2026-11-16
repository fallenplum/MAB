"""Three synthetic parity controls for bulk deterministic scoring IO."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

TOOL_PATH = Path(__file__).with_name("bulk_scoring_from_audit.py")
TOOL_IMPORT_SHA256 = hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest()

import bulk_scoring_from_audit as tool

if hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() != TOOL_IMPORT_SHA256:
    raise RuntimeError("The bulk source changed while it was imported")


class BulkScoringQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import qualify_final_scoring as fixture_api

        fixture = fixture_api.FinalScoringQualification(
            "test_toy_end_to_end_uses_frozen_scorer_and_sparse_projection"
        )
        cls.addClassCleanup(fixture.doCleanups)
        fixture.setUpClass()
        fixture.setUp()
        cls.fixture = fixture
        cls.final = fixture_api.final
        execution = json.loads(
            Path(fixture.manifest["execution_manifest_ref"]["path"]).read_text(encoding="utf-8")
        )
        scope_sha = fixture.module.canonical_digest(fixture.scope)
        receipts = []
        for key in sorted(fixture.scope["required_aggregate_keys"]):
            owner = (
                fixture.root / "owned" / scope_sha / "aggregation"
                / fixture.module.canonical_digest(list(key))
            )
            receipts.append(json.loads((owner / "LAST_TASK_RECEIPT.json").read_text(encoding="utf-8")))
        cls.context = dict(
            validator=cls.final.audit_startup16.load_source_validator(
                fixture.manifest["source_validator_ref"]
            ),
            reference_scope=json.loads(
                Path(fixture.manifest["reference_scope_ref"]["path"]).read_text(encoding="utf-8")
            ),
            scoring_scope=json.loads(
                Path(fixture.manifest["scoring_scope_ref"]["path"]).read_text(encoding="utf-8")
            ),
            approval_ref=execution["approval_ref"],
            adapter=fixture.module.E18ExecutionAdapter(
                fixture.apis, fixture.scope, fixture.records, fixture.members, fixture.aggregator
            ),
            module=fixture.module,
            receipts=receipts,
        )
        cls.plain_results, cls.plain_ledger, cls.plain_stats, _ = cls.run_scores(
            cls.context["validator"].SparseScoringAdapter, "bulk_parity_plain"
        )
        tool.install_bulk_scorer(cls.context, fixture.root / "bulk_parity_wrapped")
        cls.bulk_results, cls.bulk_ledger, cls.bulk_stats, cls.bulk_calls = cls.run_scores(
            cls.context["validator"].SparseScoringAdapter, "bulk_parity_wrapped"
        )

    @classmethod
    def run_scores(cls, constructor, name):
        provider_api = cls.final.load_reference_api(
            cls.fixture.manifest["reference_adapter_ref"]
        )
        provider = provider_api.E18ReferenceProvider(
            cls.context["reference_scope"],
            cls.context["approval_ref"],
            software_test_only=True,
        )
        runtime = cls.fixture.root / name
        scorer = constructor(
            cls.context["approval_ref"],
            cls.context["scoring_scope"],
            provider,
            runtime / "journal.sqlite3",
            runtime / "receipts",
            software_test_only=True,
            scorer_source_root=cls.context["adapter"].apis.root / "src",
        )
        try:
            with patch.object(provider, "get_reference", wraps=provider.get_reference) as requests:
                results = [
                    scorer.score_one(receipt, cls.context["reference_scope"]["reference_version"])
                    for receipt in cls.context["receipts"]
                ]
                calls = [call.args[0] for call in requests.call_args_list]
                ledger = scorer.ledger()
                stats = provider.stats()
        finally:
            scorer.close()
        if name == "bulk_parity_wrapped":
            cls.bulk_provider = provider
        return results, ledger, stats, calls

    def test_each_sparse_score_and_parser_outcome_matches_original(self):
        self.assertEqual(len(self.plain_results), 24)
        self.assertEqual(len(self.bulk_results), 24)
        receipt_fields = (
            "score", "parsed", "reason", "question_id", "canonical_id",
            "generation_seed", "aggregation_seed", "combination_index", "team",
            "scope_sha256", "approval_sha256", "bindings", "reference_version",
            "reference_source_identity_hash", "gold_new_in_scoring_journal",
            "scorer_version", "scorer_source_sha256", "aggregator_fingerprint",
            "aggregation_record_sha256",
            "physical_attempt_id", "actual_source_input_tokens", "actual_source_output_tokens",
        )
        for index, (plain, bulk) in enumerate(zip(self.plain_results, self.bulk_results)):
            with self.subTest(response=index):
                self.assertEqual(
                    {key: value for key, value in plain["sparse_row"].items() if key != "scoring_receipt_sha256"},
                    {key: value for key, value in bulk["sparse_row"].items() if key != "scoring_receipt_sha256"},
                )
                for field in receipt_fields:
                    self.assertEqual(plain["scoring_receipt"][field], bulk["scoring_receipt"][field], field)
                self.assertIsNone(bulk["scoring_receipt"]["source_execution_receipt_sha256"])

    def test_gold_and_physical_resource_ledgers_match_with_one_provider_call_per_question(self):
        fields = (
            "schema", "physical_aggregation_sources", "actual_source_input_tokens",
            "actual_source_output_tokens", "confirmed_reference_keys_in_scoring_journal",
            "successful_sparse_keys", "deterministic_scoring_model_calls",
            "incremental_member_calls", "incremental_aggregation_calls",
            "execution_resources_already_charged_upstream",
        )
        for field in fields:
            self.assertEqual(self.plain_ledger[field], self.bulk_ledger[field], field)
        order = lambda row: row["physical_attempt_id"]
        self.assertEqual(
            sorted(self.plain_ledger["source_execution_resources"], key=order),
            sorted(self.bulk_ledger["source_execution_resources"], key=order),
        )
        self.assertEqual(self.bulk_ledger["confirmed_reference_keys_in_scoring_journal"], 4)
        self.assertEqual(self.bulk_ledger["physical_aggregation_sources"], 24)
        expected = {row["question_id"]: 1 for row in self.fixture.records}
        self.assertEqual(dict(Counter(self.bulk_calls)), expected)
        for key, value in self.plain_stats.items():
            if key not in {"requests", "reference_cache_hits"}:
                self.assertEqual(value, self.bulk_stats[key], key)

    def test_sealed_identity_is_denied_without_another_reveal(self):
        denied = self.context["reference_scope"]["sealed_exclusions"][0]
        before = self.bulk_provider.stats()
        with self.assertRaises(PermissionError):
            self.bulk_provider.get_reference(
                denied["question_id"],
                denied["canonical_question_sha256"],
                self.context["reference_scope"]["reference_version"],
            )
        after = self.bulk_provider.stats()
        self.assertEqual(after["gold_reveal_accounts_consumed"], before["gold_reveal_accounts_consumed"])
        self.assertEqual(after["reference_values_decoded"], before["reference_values_decoded"])

    def test_point_only_statistics_skips_interval_and_original_paired_helper(self):
        import numpy as np

        rewards = np.asarray(
            [
                [[1, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
                [[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                [[1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 1]],
                [[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]],
            ],
            dtype=np.int8,
        )
        forbidden = Mock(side_effect=AssertionError("Original interval helper was reached"))
        with patch("scipy.stats.t.ppf", side_effect=AssertionError("Interval quantile was reached")) as quantile:
            with patch.object(self.final, "holm_five", wraps=self.final.holm_five) as holm:
                result = tool.point_only_statistics(self.final.final_statistics)(
                    rewards, self.fixture.records,
                    SimpleNamespace(paired_question_test=forbidden),
                )
                holm.assert_called_once()
        forbidden.assert_not_called()
        quantile.assert_not_called()
        self.assertEqual(set(result["quality"]), set(self.final.POLICIES))
        self.assertEqual(result["secondary_family_size"], 5)
        self.assertEqual(len(result["secondary"]), 5)
        self.assertTrue(result["user_reporting_override"])
        for comparison in [result["primary"], *result["secondary"].values()]:
            self.assertTrue(np.isfinite(comparison["p_value"]))
            self.assertGreaterEqual(comparison["p_value"], 0)
            self.assertLessEqual(comparison["p_value"], 1)
            self.assertIsNone(comparison["confidence_interval"])
            self.assertIsNone(comparison["confidence_level"])
            self.assertTrue(comparison["user_reporting_override"])
        for comparison in result["secondary"].values():
            self.assertGreaterEqual(comparison["holm_adjusted_p"], comparison["p_value"])
            self.assertLessEqual(comparison["holm_adjusted_p"], 1)
        self.assertEqual(
            result["primary"]["corrected_generation_outcomes"],
            int(((rewards[2] == 1) & (rewards[1] == 0)).sum()),
        )
        self.assertEqual(
            result["primary"]["damaged_generation_outcomes"],
            int(((rewards[2] == 0) & (rewards[1] == 1)).sum()),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Software qualification records are append-only")
    if args.asset_root is not None:
        os.environ["MAB_ASSET_ROOT"] = str(args.asset_root)
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BulkScoringQualification)
    )
    unchanged = hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == TOOL_IMPORT_SHA256
    record = dict(
        schema="e18_bulk_scoring_io_qualification_v1",
        status="PASS" if result.wasSuccessful() and unchanged else "FAIL",
        tests_run=result.testsRun,
        errors=len(result.errors),
        failures=len(result.failures),
        tool_source_sha256=TOOL_IMPORT_SHA256,
        tool_source_unchanged_during_qualification=unchanged,
        qualification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        software_test_only=True,
        synthetic_questions=4,
        synthetic_responses_per_path=24,
        real_gold_or_question_reads=0,
        actual_model_calls=0,
        actual_jobs_submitted=0,
        actual_process_signals=0,
        execution_gate_reruns=0,
        whole_execution_receipt_rehash=False,
        t_ppf_calls=0,
        original_interval_paired_helper_calls=0,
        wall_seconds=time.perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(record, ensure_ascii=False))
    if record["status"] != "PASS":
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
