"""Two synthetic sanity controls for reference-body mapping and score reuse."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

TOOL_PATH = Path(__file__).with_name("resume_bulk_choice_reference.py")
TOOL_IMPORT_SHA256 = hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest()

import resume_bulk_choice_reference as tool

if hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() != TOOL_IMPORT_SHA256:
    raise RuntimeError("The resume source changed while it was imported")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ResumeChoiceSanity(unittest.TestCase):
    def test_reference_body_maps_uniquely_without_expanding_prediction_parser(self):
        def canonical(subtask, candidate, choices):
            match = re.fullmatch(r"\(([A-Z])\)", candidate)
            return match[1] if match else None

        original = SimpleNamespace(
            CHOICE_TASKS={"ruin_names"},
            _candidate=lambda value: (value.strip(), None),
            _canonical=canonical,
            _normal_text=lambda value: " ".join(value.split()).casefold(),
        )
        reference = dict(task="bbh", subtask="ruin_names", reference="beta", choices=None)
        unchanged = copy.deepcopy(reference)
        question = "Synthetic name question\n(A) alpha\n(B) beta"
        adapted, event = tool.adapt_choice_reference(reference, question, original)
        self.assertEqual(adapted["reference"], "(B)")
        self.assertEqual(adapted["choices"], reference["choices"])
        self.assertEqual(reference, unchanged)
        self.assertEqual(event["chosen_label"], "B")
        self.assertEqual(event["visible_options"], 2)
        label = dict(reference, reference="(A)")
        kept, event = tool.adapt_choice_reference(label, question, original)
        self.assertIs(kept, label)
        self.assertIsNone(event)
        with self.assertRaises(ValueError):
            tool.adapt_choice_reference(reference, "(A) beta\n(B) beta", original)
        with self.assertRaises(ValueError):
            tool.adapt_choice_reference(dict(reference, reference="gamma"), question, original)
        self.assertIs(original._canonical, canonical)
        self.assertIsNone(original._canonical("ruin_names", "beta", None))
        self.assertEqual(original._canonical("ruin_names", "(B)", None), "B")

    def test_cached_receipt_reuses_exact_bytes_without_provider_or_scoring(self):
        execution_scope = digest("synthetic execution scope")
        scoring_scope = digest("synthetic scoring scope")
        approval = digest("synthetic approval")
        canonical = digest("synthetic canonical input")
        native_sha = digest("synthetic native record")
        key = ("synthetic-resume", 42, 0)
        version = "synthetic-reference-v1"
        receipt = dict(
            question_id=key[0], canonical_id=canonical, generation_seed=key[1],
            combination_index=key[2], aggregation_seed=42, team=[0, 1, 2],
            aggregator_fingerprint=digest("synthetic operator"),
            scorer_version="synthetic-scorer-v1", score=1.0, parsed=True,
            reason="synthetic exact match", reference_version=version,
            scope_sha256=scoring_scope, approval_sha256=approval,
            aggregation_record_sha256=native_sha, physical_attempt_id="synthetic-attempt",
            actual_source_input_tokens=7, actual_source_output_tokens=3,
            reference_source_identity_hash=digest("synthetic reference identity"),
        )
        encoded = json.dumps(receipt, sort_keys=True).encode()
        execution = dict(
            scope_sha256=execution_scope, approval_sha256=approval,
            task_key=dict(question_id=key[0], canonical_id=canonical,
                          generation_seed=key[1], combination_index=key[2], team=[0, 1, 2]),
            native_record_sha256=native_sha, physical_attempt_id="synthetic-attempt",
            native_record=dict(result=dict(prompt_tokens=7, completion_tokens=3, wall_time_s=0.125)),
        )

        class FakeBase:
            def __init__(self):
                self.scope_contract = {"execution_scope_sha256": execution_scope}
                self.approval_ref = {"sha256": approval}
                self._scope_sha = scoring_scope
                self._scored, self._resources, self._gold = {}, {}, {}
                self._attempts = []
                self._receipt_stream, self._event_stream = io.StringIO(), io.StringIO()
                self.reference_provider = SimpleNamespace(
                    get_reference=Mock(side_effect=AssertionError("Provider was called"))
                )
                self.scorer = SimpleNamespace(
                    score_answer=Mock(side_effect=AssertionError("Scorer was called"))
                )
                self.upstream_checks = 0

            def _validate_upstream(self, value):
                self.upstream_checks += 1
                if value["scope_sha256"] != execution_scope:
                    raise ValueError("Synthetic upstream scope mismatch")

            def score_one(self, *args):
                raise AssertionError("Cache reuse fell through to the fresh path")

            def ledger(self):
                return dict(
                    source_execution_resources=list(self._resources.values()),
                    physical_aggregation_sources=len(self._resources),
                    actual_source_input_tokens=sum(row["prompt_tokens"] for row in self._resources.values()),
                    actual_source_output_tokens=sum(row["completion_tokens"] for row in self._resources.values()),
                    confirmed_reference_keys_in_scoring_journal=len(self._gold),
                    successful_sparse_keys=len(self._scored),
                )

        fields = {
            "question_id", "canonical_id", "generation_seed", "aggregation_seed",
            "combination_index", "team", "aggregator_fingerprint", "scorer_version",
            "score", "aggregation_record_sha256", "scoring_receipt_sha256",
        }
        module = SimpleNamespace(FIELDS=fields, SparseScoringAdapter=FakeBase)
        context = {"validator": module}
        with patch.object(tool.bulk, "install_bulk_scorer", return_value=FakeBase):
            tool.install_resuming_scorer(context, Path("synthetic-unused-output"), {key: (receipt, encoded)})
        scorer = module.SparseScoringAdapter()
        result = scorer.score_one(execution, version)
        self.assertTrue(result["from_scoring_cache"])
        self.assertIs(result["scoring_receipt"], receipt)
        self.assertEqual(result["sparse_row"]["score"], 1.0)
        self.assertEqual(result["sparse_row"]["scoring_receipt_sha256"], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(scorer._receipt_stream.getvalue(), encoded.decode() + "\n")
        self.assertEqual(scorer.upstream_checks, 1)
        scorer.reference_provider.get_reference.assert_not_called()
        scorer.scorer.score_answer.assert_not_called()
        ledger = scorer.ledger()
        self.assertEqual(ledger["confirmed_reference_keys_in_scoring_journal"], 1)
        self.assertEqual(ledger["physical_aggregation_sources"], 1)
        self.assertEqual(ledger["successful_sparse_keys"], 1)
        self.assertEqual(ledger["actual_source_input_tokens"], 7)
        self.assertEqual(ledger["actual_source_output_tokens"], 3)
        self.assertEqual(ledger["previous_committed_scores_reused"], 1)
        self.assertEqual(ledger["new_deterministic_scores"], 0)
        self.assertEqual(ledger["source_execution_resources"], [dict(
            physical_attempt_id="synthetic-attempt", aggregation_record_sha256=native_sha,
            prompt_tokens=7, completion_tokens=3, tokens=10, source_wall_time_s=0.125,
            resource_kind="aggregation_source_reference", charged_by_execution_adapter=True,
            incremental_member_calls_here=0, incremental_aggregation_calls_here=0,
        )])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Synthetic sanity receipts are append-only")
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ResumeChoiceSanity)
    )
    unchanged = hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == TOOL_IMPORT_SHA256
    record = dict(
        schema="e18_resume_bulk_choice_reference_synthetic_sanity_v1",
        status="PASS" if result.wasSuccessful() and unchanged else "FAIL",
        tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
        tool_source_sha256=TOOL_IMPORT_SHA256,
        tool_source_unchanged_during_qualification=unchanged,
        qualification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        software_test_only=True,
        fixture_mode="synthetic choice helpers, fake base and StringIO",
        real_gold_or_question_reads=0, real_prior_score_receipts_read=0,
        model_or_scoring_calls=0, execution_gate_reruns=0,
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
