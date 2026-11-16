"""Toy-only final gate, frozen scorer, sparse projection and paired statistics."""

from __future__ import annotations

import argparse
import copy
import io
import json
import shutil
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import e18_final_scoring as final
import e18_aggregation_worker_v3 as worker
import qualify_gpu_worker as fixtures


class FinalScoringQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GPUWorkerQualification.setUpClass()
        cls.apis = fixtures.fixtures.ExecutionAdapterQualification.apis
        cls.members = fixtures.fixtures.ExecutionAdapterQualification.member_specs
        cls.aggregator = fixtures.fixtures.ExecutionAdapterQualification.aggregator_spec
        cls.module = fixtures.fixtures.adapter_module
        cls.prep = fixtures.PREPARATION

    def reference(self, path):
        return dict(path=str(path), sha256=worker.file_sha(path))

    def artifact(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return self.reference(path)

    def setUp(self):
        worker._PROCESS_BACKEND_CONTEXT = None
        self.temporary = tempfile.TemporaryDirectory(prefix="e18-final-scoring-toy-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = [
            dict(
                question_id="synthetic-final-" + str(index),
                task="gsm8k" if index < 2 else "bbh",
                subtask=None if index < 2 else "boolean_expressions",
                source_split="train" if index < 2 else "test",
                question="Synthetic independent final question "
                + str(index)
                + ("\u2028legitimate separator" if index == 0 else ""),
                source_frame="toy_gsm" if index < 2 else "toy_bbh",
                cohort_role="synthetic",
            )
            for index in range(4)
        ]
        for row in self.records:
            row["canonical_question_sha256"] = self.module.canonical_question(
                row["question"]
            )
        path = self.root / "inputs.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.records),
            encoding="utf-8",
        )
        bindings = {
            key: self.module.canonical_digest(key)
            for key in self.module.REQUIRED_BINDINGS
        }
        bindings["target_ids_sha256"] = self.module.canonical_digest(
            [
                [row["question_id"], row["canonical_question_sha256"]]
                for row in self.records
            ]
        )
        bindings["target_inputs_sha256"] = worker.file_sha(path)
        self.actions = [[0] * 4, [1] * 4, [0, 0, 1, 1], [0] * 4]
        member_keys = [
            (row["question_id"], seed, member)
            for row in self.records
            for seed in (42, 137, 2024)
            for member in (0, 1, 2, 3)
        ]
        aggregate_keys = [
            (row["question_id"], seed, action)
            for row in self.records
            for seed in (42, 137, 2024)
            for action in (0, 1)
        ]
        self.scope = self.module.build_scope_contract(
            self.records, self.actions, bindings, member_keys, aggregate_keys
        )
        reference_api = final.load_reference_api(
            self.reference(self.prep / "e18_reference_adapter.py")
        )
        validator = final.audit_startup16.load_source_validator(
            self.reference(self.prep / "e18_scoring_adapter.py")
        )
        sources, locators = [], []
        for index, row in enumerate(self.records):
            source = self.root / ("reference" + str(index) + ".jsonl")
            payload = {
                key: row[key]
                for key in (
                    "question_id",
                    "task",
                    "subtask",
                    "source_split",
                    "question",
                )
            }
            payload["reference"] = "4" if row["task"] == "gsm8k" else "True"
            source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            source_id = "synthetic-source-" + str(index)
            sources.append(
                dict(
                    source_id=source_id,
                    format="normalized_qid_jsonl_v1",
                    path=str(source),
                    sha256=worker.file_sha(source),
                    task=row["task"],
                    subtask=row["subtask"],
                    source_split=row["source_split"],
                )
            )
            locators.append(
                dict(
                    question_id=row["question_id"],
                    canonical_question_sha256=row["canonical_question_sha256"],
                    task=row["task"],
                    subtask=row["subtask"],
                    source_split=row["source_split"],
                    source_id=source_id,
                    row_index=0,
                )
            )
        reference_scope = reference_api.build_reference_scope(
            locators,
            sources,
            bindings,
            "synthetic_final_v1",
            [
                dict(
                    question_id="synthetic-sealed-denied",
                    canonical_question_sha256=self.module.canonical_digest(
                        "synthetic sealed exclusion"
                    ),
                )
            ],
            allow_structural_byte_scan=True,
        )
        scoring_scope = validator.build_scoring_scope(self.scope, reference_scope)
        approval = dict(
            schema="e18_scope_approval_v2",
            status="APPROVED",
            cohort="e18_confirm",
            bindings=scoring_scope["bindings"],
            scope_sha256_by_role=dict(
                execution=self.module.canonical_digest(self.scope),
                reference=self.module.canonical_digest(reference_scope),
                scoring=self.module.canonical_digest(scoring_scope),
            ),
            operations=sorted(self.module.OPERATIONS),
            software_test_only=True,
            execution_limits=self.scope["execution_limits"],
            approved_output_root=str(self.root / "owned"),
        )
        approval_ref = self.artifact("approval.json", approval)
        executor = self.module.E18ExecutionAdapter(
            self.apis, self.scope, self.records, self.members, self.aggregator
        )
        native = {}
        for qid, seed, member in member_keys:
            receipt = self.module.run_approved_task(
                executor,
                dict(
                    stage="member_generation",
                    question_id=qid,
                    generation_seed=seed,
                    member_index=member,
                ),
                approval_ref,
                self.root / "owned",
                lambda spec: fixtures.fixtures.FakeBackend(spec, self.apis.inference),
                software_test_only=True,
            )
            native[(qid, seed, member)] = receipt["native_record"]
        tokenizer = fixtures.fixtures.FakeTokenizer(self.aggregator)
        for qid, seed, action in aggregate_keys:
            question = next(
                index
                for index, row in enumerate(self.records)
                if row["question_id"] == qid
            )
            correct = action == (0 if question < 2 else 1)
            if question == 1 and action == 1 and seed == 137:
                correct = True
            answer = (
                ("4" if correct else "5")
                if question < 2
                else ("True" if correct else "False")
            )
            self.module.run_approved_task(
                executor,
                dict(
                    stage="aggregation",
                    question_id=qid,
                    generation_seed=seed,
                    combination_index=action,
                ),
                approval_ref,
                self.root / "owned",
                lambda spec, text="Final answer: " + answer: (
                    fixtures.fixtures.FakeBackend(spec, self.apis.inference, text=text)
                ),
                tokenizer=tokenizer,
                member_records={
                    member: native[(qid, seed, member)]
                    for member in self.module.TEAMS[action]
                },
                software_test_only=True,
            )
        execution = dict(
            schema=worker.MANIFEST_SCHEMA,
            software_test_only=True,
            asset_root=str(fixtures.fixtures.ASSETS),
            adapter_ref=self.reference(self.prep / "e18_execution_adapter.py"),
            scope_ref=self.artifact("scope.json", self.scope),
            question_records_ref=self.reference(path),
            approval_ref=approval_ref,
            owned_output_root=str(self.root / "owned"),
            stage="aggregation",
            task_keys_ref=self.artifact("aggregate-keys.json", aggregate_keys),
            backend=dict(mode="original"),
            gpu_lock_root=str(self.root / "locks"),
            install_signal_handlers=False,
        )
        self.manifest = dict(
            schema="e18_final_scoring_manifest_v1",
            software_test_only=True,
            policy_ids=list(final.POLICIES),
            execution_manifest_ref=self.artifact("execution.json", execution),
            reference_scope_ref=self.artifact("reference-scope.json", reference_scope),
            scoring_scope_ref=self.artifact("scoring-scope.json", scoring_scope),
            source_validator_ref=self.reference(self.prep / "e18_scoring_adapter.py"),
            reference_adapter_ref=self.reference(
                self.prep / "e18_reference_adapter.py"
            ),
            power_source_ref=self.reference(self.prep / "power_and_statistics.py"),
            private_output_root=str(self.root / "final-private"),
            public_output_root=str(self.root / "final-public"),
        )
        startup_plan = dict(
            schema="e18_aggregation_dispatch_plan_v3",
            worker_source_sha256=worker.file_sha(worker.__file__),
            manifest_refs=[self.manifest["execution_manifest_ref"]],
        )
        startup_ref = self.artifact("toy-startup-plan.json", startup_plan)
        startup = final.audit_startup16.audit_plan(
            startup_ref,
            self.manifest["source_validator_ref"],
            expected_tasks=24,
            software_test_only=True,
            fixture_tokenizer=tokenizer,
            fixture_factory=lambda _: self.fail("Audit loaded a model"),
        )
        startup["audit_source_sha256"] = worker.file_sha(final.audit_startup16.__file__)
        self.manifest["startup_audit_ref"] = self.artifact(
            "toy-startup-audit.json", startup
        )

    def gate(self):
        return final.execution_gate(
            self.manifest,
            software_test_only=True,
            fixture_factory=lambda _: self.fail("Finalization tried loading a model"),
        )

    def test_complete_gate_is_reference_and_scorer_free(self):
        with patch.object(
            final,
            "load_reference_api",
            side_effect=AssertionError("Provider was reached before gate"),
        ):
            context = self.gate()
        self.assertEqual(
            (
                context["gate"]["unique_aggregate_keys"],
                context["gate"]["unique_member_keys"],
            ),
            (24, 48),
        )
        self.assertEqual(
            (context["gate"]["reference_values_read"], context["gate"]["scorer_calls"]),
            (0, 0),
        )

    def test_missing_aggregate_blocks_reference_phase(self):
        owner = (
            self.root
            / "owned"
            / self.module.canonical_digest(self.scope)
            / "aggregation"
            / self.module.canonical_digest([self.records[0]["question_id"], 42, 0])
        )
        (owner / "LAST_TASK_RECEIPT.json").unlink()
        with patch.object(
            final, "load_reference_api", side_effect=AssertionError("Provider reached")
        ):
            with self.assertRaises(FileNotFoundError):
                self.gate()

    def test_requiring_full20_scores_is_rejected(self):
        changed = copy.deepcopy(self.scope)
        changed["required_aggregate_keys"].append(
            [self.records[0]["question_id"], 42, 19]
        )
        with self.assertRaises(ValueError):
            final.question_union(changed, software_test_only=True)

    def test_incomplete_gate_cannot_open_reference(self):
        with patch.object(
            final, "load_reference_api", side_effect=AssertionError("Provider reached")
        ):
            with self.assertRaises(PermissionError):
                final.execute_scoring(
                    self.manifest,
                    dict(gate=dict(status="INCOMPLETE")),
                    software_test_only=True,
                )

    def test_toy_end_to_end_uses_frozen_scorer_and_sparse_projection(self):
        context = self.gate()
        result = final.execute_scoring(self.manifest, context, software_test_only=True)
        self.assertEqual(result["distinct_gold_accounts"], 4)
        self.assertEqual(
            result["results"]["quality"]["TSPO_uniform"]["task_macro_accuracy"], 1
        )
        self.assertEqual(result["results"]["primary"]["original_questions"], 4)
        self.assertEqual(len(result["results"]["secondary"]), 5)
        text = json.dumps(result)
        self.assertNotIn("synthetic-final-", text)
        with np.load(
            self.root / "final-private/FINAL_POLICY_REWARDS.npz", allow_pickle=False
        ) as arrays:
            self.assertEqual(arrays["rewards"].shape, (4, 3, 4))
            self.assertTrue(all(arrays[key].dtype.kind != "O" for key in arrays.files))

    def test_scoring_resume_preserves_immutable_publication(self):
        context = self.gate()
        first = final.execute_scoring(self.manifest, context, software_test_only=True)
        second = final.execute_scoring(self.manifest, context, software_test_only=True)
        self.assertEqual(first, second)

    def test_unknown_physical_usage_blocks_gold(self):
        key = [self.records[0]["question_id"], 42, 0]
        owner = (
            self.root
            / "owned"
            / self.module.canonical_digest(self.scope)
            / "member_generation"
            / self.module.canonical_digest(key)
        )
        source = json.loads(
            (owner / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        source.pop("result")
        source.update(
            status="failure",
            actual_usage=None,
            cost_status="unknown_must_reconcile",
            attempt_id="synthetic-extra-unknown",
        )
        source["record_sha256"] = self.module.canonical_digest(
            {key: value for key, value in source.items() if key != "record_sha256"}
        )
        with (owner / "records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(source) + "\n")
        with self.assertRaises(ValueError):
            self.gate()

    def test_primary_statistics_is_original_frozen_function(self):
        power = final.load_statistics(self.manifest["power_source_ref"])
        rewards = np.asarray(
            [
                [[1, 1, 0, 0]] * 3,
                [[0, 1, 1, 1]] * 3,
                [[1, 1, 1, 1]] * 3,
                [[1, 1, 0, 0]] * 3,
            ],
            dtype=float,
        )
        metadata = [
            dict(
                task=row["task"],
                subtask=row["subtask"],
                source_frame=row["source_frame"],
            )
            for row in self.records
        ]
        actual = final.final_statistics(rewards, metadata, power)["primary"]
        expected = power.paired_question_test(
            rewards[2], rewards[1], np.asarray([row["task"] for row in self.records])
        )
        self.assertEqual(actual["estimate"], expected["estimate"])
        self.assertEqual(actual["confidence_interval"], expected["confidence_interval"])
        self.assertEqual(actual["p_value"], expected["p_value"])

    def test_production_rejects_fixture_factory(self):
        changed = copy.deepcopy(self.manifest)
        changed["software_test_only"] = False
        with self.assertRaises(PermissionError):
            final.execution_gate(changed, fixture_factory=lambda _: None)

    def test_explicit_native_and_cpu_roots_preserve_logical_scope(self):
        native_view = self.root / "physical-native-view"
        shutil.move(str(self.root / "owned"), str(native_view))
        context = final.execution_gate(
            self.manifest,
            software_test_only=True,
            fixture_factory=lambda _: self.fail("Transport mapping loaded a model"),
            native_read_root=native_view,
        )
        cpu = self.root / "cpu-small-files"
        result = final.execute_scoring(
            self.manifest, context, software_test_only=True, runtime_root=cpu
        )
        self.assertEqual(
            result["execution_gate"]["execution_scope_sha256"],
            self.module.canonical_digest(self.scope),
        )
        self.assertTrue((cpu / "SCORING_JOURNAL.sqlite3").exists())
        self.assertFalse(Path(self.manifest["private_output_root"]).exists())
        self.assertEqual(
            worker.read_json(cpu / "RUNTIME_IO_METADATA.json")[
                "physical_native_read_root"
            ],
            str(native_view),
        )

    def test_runtime_reference_mapping_keeps_exact_original_sha(self):
        physical = self.root / "cache-records"
        physical.mkdir()
        path = physical / "record.json"
        path.write_text('{"synthetic": true}', encoding="utf-8")
        reference = dict(
            path="/remote/native-cache/record.json", sha256=worker.file_sha(path)
        )
        mapped = final.runtime_reference(
            reference,
            [dict(logical_prefix="/remote/native-cache", physical_root=str(physical))],
        )
        self.assertEqual(mapped["sha256"], reference["sha256"])
        self.assertEqual(worker.checked_ref(mapped), path)

    def test_holm_preserves_five_comparison_family(self):
        values = {
            str(index): dict(p_value=p)
            for index, p in enumerate((0.01, 0.04, 0.03, 0.2, None))
        }
        result = final.holm_five(values)
        self.assertAlmostEqual(result["0"]["holm_adjusted_p"], 0.05)
        self.assertAlmostEqual(result["1"]["holm_adjusted_p"], 0.12)
        self.assertAlmostEqual(result["2"]["holm_adjusted_p"], 0.12)
        self.assertAlmostEqual(result["3"]["holm_adjusted_p"], 0.4)
        self.assertIsNone(result["4"]["holm_adjusted_p"])

    def test_input_jsonl_keeps_legal_unicode_separator_inside_one_record(self):
        execution = worker.read_json(self.manifest["execution_manifest_ref"]["path"])
        path = Path(execution["question_records_ref"]["path"])
        with path.open(encoding="utf-8", newline="") as stream:
            physical = list(stream)
        self.assertEqual(len(physical), 4)
        self.assertIn("\u2028", json.loads(physical[0])["question"])
        self.assertEqual(self.gate()["gate"]["original_questions"], 4)

    def test_production_member_gate47285_stops_before_provider(self):
        context = dict(
            gate=dict(
                status="PASS",
                original_questions=3000,
                unique_aggregate_keys=30921,
                unique_member_keys=47285,
            ),
            metadata=[None] * 3000,
            receipts=[None] * 30921,
            members={index: None for index in range(47285)},
        )
        with patch.object(
            final, "load_reference_api", side_effect=AssertionError("Provider reached")
        ):
            with self.assertRaises(PermissionError):
                final.execute_scoring(self.manifest, context)

    def test_declared_full_member_gate_with_short_container_stops_before_provider(self):
        context = dict(
            gate=dict(
                status="PASS",
                original_questions=3000,
                unique_aggregate_keys=30921,
                unique_member_keys=47286,
            ),
            metadata=[None] * 3000,
            receipts=[None] * 30921,
            members={index: None for index in range(47285)},
        )
        with patch.object(
            final, "load_reference_api", side_effect=AssertionError("Provider reached")
        ):
            with self.assertRaises(PermissionError):
                final.execute_scoring(self.manifest, context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification output is append-only")
    wall = time.perf_counter()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FinalScoringQualification)
    )
    record = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        finalizer_source_sha256=worker.file_sha(final.__file__),
        test_source_sha256=worker.file_sha(__file__),
        formal_gold_reads=0,
        formal_scorer_calls=0,
        toy_only=True,
        actual_model_calls=0,
        actual_jobs_submitted=0,
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
