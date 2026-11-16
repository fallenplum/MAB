"""Small synthetic controls for final de-identified report generation."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

import generate_final_report as report
import e18_final_scoring as frozen_final


PREP = (
    Path(__file__).resolve().parent.parent
    / "e18_mechanism_and_confirmation_preparation"
)


class ReportQualification(unittest.TestCase):
    def ref(self, path):
        return dict(path=str(path), sha256=report.file_sha(path))

    def artifact(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return self.ref(path)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e18-report-toy-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.n = 8
        labels = np.asarray(
            ["gsm8k", "bbh", "gsm8k", "gsm8k", "gsm8k", "bbh", "bbh", "bbh"]
        )
        frames = np.asarray(
            [
                "gsm8k_official_train_original_primary_frame",
                report.OLD_BBH,
                "gsm8k_official_train_original_primary_frame",
                "gsm8k_official_train_original_primary_frame",
                "gsm8k_official_train_original_primary_frame",
                report.OLD_BBH,
                report.EXT_BBH,
                report.EXT_BBH,
            ]
        )
        self.inputs = []
        for index in range(self.n):
            question = (
                "SYNTHETIC_SECRET_QUESTION "
                + str(index)
                + ("\u2028inside same record" if index == 0 else "")
            )
            canonical = hashlib.sha256(" ".join(question.split()).encode()).hexdigest()
            self.inputs.append(
                dict(
                    question_id="toy-report-q" + str(index),
                    canonical_question_sha256=canonical,
                    question=question,
                    task=str(labels[index]),
                    source_frame=str(frames[index]),
                    subtask=""
                    if labels[index] == "gsm8k"
                    else "boolean_expressions"
                    if index in (1, 5)
                    else "word_sorting",
                    cohort_role="original_fresh400"
                    if index in (2, 6)
                    else "approved_new2600",
                    input_position=index,
                )
            )
        inputs = self.root / "inputs.jsonl"
        inputs.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.inputs),
            encoding="utf-8",
        )
        self.inputs_ref = self.ref(inputs)
        actions = np.asarray(
            [[0] * 8, [1] * 8, [0, 1, 1, 0, 1, 1, 0, 1], [0] * 8], dtype=np.int64
        )
        rng = np.random.default_rng(20261116033)
        action0 = rng.binomial(1, 0.75, size=(3, 8))
        action1 = rng.binomial(1, 0.55, size=(3, 8))
        rewards = np.asarray(
            [
                np.where(actions[index][None, :] == 0, action0, action1)
                for index in range(4)
            ],
            dtype=np.int8,
        )
        self.arrays = dict(
            rewards=rewards,
            frozen_policy_actions=actions,
            policy_ids=np.asarray(report.POLICIES, dtype="U"),
            question_ids=np.asarray(
                [row["question_id"] for row in self.inputs], dtype="U"
            ),
            canonical_hashes=np.asarray(
                [row["canonical_question_sha256"] for row in self.inputs], dtype="U"
            ),
            task_labels=labels,
            subtask_labels=np.asarray(
                [row["subtask"] for row in self.inputs], dtype="U"
            ),
            source_frames=frames,
        )
        self.rewards_path = self.root / "rewards.npz"
        np.savez_compressed(self.rewards_path, **self.arrays)
        self.rewards_ref = self.ref(self.rewards_path)
        aggregate = {
            (row["question_id"], seed, int(action))
            for index, row in enumerate(self.inputs)
            for action in set(actions[:, index])
            for seed in report.SEEDS
        }
        members = {
            (qid, seed, member)
            for qid, seed, action in aggregate
            for member in report.TEAMS[action]
        }
        rows = []
        for stage, keys in (("member_generation", members), ("aggregation", aggregate)):
            for index, key in enumerate(sorted(keys)):
                identity = hashlib.sha256(repr((stage, key)).encode()).hexdigest()
                rows.append(
                    dict(
                        stage=stage,
                        task_key=list(key),
                        physical_attempt_id=identity[:32],
                        native_record_sha256=identity,
                        status="success",
                        prompt_tokens=10 if stage == "member_generation" else 20,
                        completion_tokens=3 if stage == "member_generation" else 4,
                        reused_historical_source=stage == "member_generation"
                        and index < 3,
                    )
                )
        extra = copy.deepcopy(
            next(
                row
                for row in rows
                if row["stage"] == "member_generation"
                and not row["reused_historical_source"]
            )
        )
        extra.update(
            physical_attempt_id="synthetic-failed-physical",
            native_record_sha256="a" * 64,
            status="failure",
            prompt_tokens=2,
            completion_tokens=0,
        )
        rows.append(extra)
        totals = {
            stage: dict(
                physical_attempts=sum(row["stage"] == stage for row in rows),
                prompt_tokens=sum(
                    row["prompt_tokens"] for row in rows if row["stage"] == stage
                ),
                completion_tokens=sum(
                    row["completion_tokens"] for row in rows if row["stage"] == stage
                ),
            )
            for stage in ("member_generation", "aggregation")
        }
        self.costs = dict(rows=rows, totals=totals)
        self.costs_ref = self.artifact("costs.json", self.costs)
        power = frozen_final.load_statistics(self.ref(PREP / "power_and_statistics.py"))
        statistics = frozen_final.final_statistics(rewards, self.inputs, power)
        service = {}
        bykey = {
            (row["stage"], *row["task_key"]): row
            for row in rows
            if row["status"] == "success"
        }
        for policy, name in enumerate(report.POLICIES):
            selected = []
            for question, metadata in enumerate(self.inputs):
                action = int(actions[policy, question])
                for seed in report.SEEDS:
                    selected.append(
                        bykey[("aggregation", metadata["question_id"], seed, action)]
                    )
                    selected.extend(
                        bykey[
                            ("member_generation", metadata["question_id"], seed, member)
                        ]
                        for member in report.TEAMS[action]
                    )
            service[name] = dict(
                member_calls=9 * self.n,
                aggregation_calls=3 * self.n,
                measured_prompt_tokens=sum(row["prompt_tokens"] for row in selected),
                measured_completion_tokens=sum(
                    row["completion_tokens"] for row in selected
                ),
            )
        self.result = dict(
            schema="e18_final_confirmation_results_v1",
            status="COMPLETE",
            finalizer_source_sha256=report.FINALIZER_SHA,
            original_questions=self.n,
            unique_aggregate_keys=len(aggregate),
            unique_member_keys=len(members),
            generation_seeds=list(report.SEEDS),
            execution_gate=dict(
                status="PASS",
                original_questions=self.n,
                unique_aggregate_keys=len(aggregate),
                unique_member_keys=len(members),
                cohort_role_counts=dict(
                    Counter(row["cohort_role"] for row in self.inputs)
                ),
                source_frame_counts=dict(Counter(frames.tolist())),
                physical_attempts=len(rows),
            ),
            distinct_gold_accounts=self.n,
            results=statistics,
            physical_cost_totals=totals,
            original_source_cached_member_records=3,
            selected_policy_service_costs=service,
            sealed_reference_values_decoded=0,
            sealed_input_values_decoded=0,
            reference_provider_stats=dict(
                software_test_only=True,
                protected_input_values_decoded=0,
                unapproved_reference_values_decoded=0,
            ),
        )
        self.result_ref = self.artifact("result.json", self.result)

    def generate(self, output="report", **kwargs):
        return report.generate_report(
            self.result_ref,
            self.rewards_ref,
            self.costs_ref,
            self.inputs_ref,
            self.root / output,
            software_test_only=True,
            **kwargs,
        )

    def test_complete_toy_report_has_four_deidentified_artifacts(self):
        result = self.generate()
        self.assertEqual(result["status"], "TOY_REPORT_COMPLETE")
        self.assertEqual(len(result["artifacts"]), 4)
        for path in (self.root / "report").iterdir():
            text = path.read_text(encoding="utf-8-sig")
            self.assertNotIn("toy-report-q", text)
            self.assertNotIn("SYNTHETIC_SECRET_QUESTION", text)
        text = (self.root / "report/E18_CONFIRMATION_REPORT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("合成软件资格样例", text)
        self.assertIn("8道原题", text)
        self.assertIn("24条生成事件", text)
        self.assertIn("独占GPU核时间未计量", text)

    def test_noncomplete_status_rejected_before_private_load(self):
        changed = copy.deepcopy(self.result)
        changed["status"] = "RUNNING"
        self.result_ref = self.artifact("running.json", changed)
        with patch.object(
            np, "load", side_effect=AssertionError("Private reward opened early")
        ):
            with self.assertRaises(PermissionError):
                self.generate()

    def test_toy_cannot_enter_formal_report_mode(self):
        with self.assertRaises(ValueError):
            report.generate_report(
                self.result_ref,
                self.rewards_ref,
                self.costs_ref,
                self.inputs_ref,
                self.root / "formal",
            )

    def test_input_metadata_roles_are_used_instead_of_prefix(self):
        result, arrays, roles, costs = report.validate_inputs(
            self.result_ref,
            self.rewards_ref,
            self.costs_ref,
            self.inputs_ref,
            software_test_only=True,
        )
        np.testing.assert_array_equal(
            np.flatnonzero(roles == "original_fresh400"), [2, 6]
        )
        self.assertNotEqual(roles[0], "original_fresh400")

    def test_legitimate_unicode_separator_is_one_physical_record(self):
        with Path(self.inputs_ref["path"]).open(encoding="utf-8", newline="") as stream:
            lines = list(stream)
        self.assertEqual(len(lines), self.n)
        self.assertIn("\u2028", json.loads(lines[0])["question"])
        self.generate()

    def test_same_action_reward_conflict_rejected(self):
        altered = {key: value.copy() for key, value in self.arrays.items()}
        altered["rewards"][3, 0, 0] = 1 - altered["rewards"][0, 0, 0]
        path = self.root / "bad-share.npz"
        np.savez_compressed(path, **altered)
        self.rewards_ref = self.ref(path)
        with self.assertRaises(ValueError):
            self.generate()

    def test_new_and_historical_cost_partition_has_known_failure(self):
        rows = report.cost_rows(self.result, self.costs)
        self.assertEqual(rows[0]["已知失败数"], 1)
        self.assertEqual(rows[1]["成员调用数"], 3)
        self.assertEqual(rows[1]["已知失败数"], 0)
        self.assertEqual(rows[2]["总Token"], rows[0]["总Token"] + rows[1]["总Token"])

    def test_duplicate_physical_attempt_rejected(self):
        changed = copy.deepcopy(self.costs)
        changed["rows"].append(changed["rows"][0])
        self.costs_ref = self.artifact("duplicate-cost.json", changed)
        with self.assertRaises(ValueError):
            self.generate()

    def test_logical_service_cost_mismatch_rejected(self):
        changed = copy.deepcopy(self.result)
        changed["selected_policy_service_costs"]["Ridge"]["measured_prompt_tokens"] += 1
        self.result_ref = self.artifact("bad-service.json", changed)
        with self.assertRaises(ValueError):
            self.generate()

    def test_optional_native_wall_is_separate_from_kernel_time(self):
        payload = dict(
            schema="e18_native_call_walltime_metadata_v1",
            rows=[
                dict(
                    physical_attempt_id=row["physical_attempt_id"],
                    native_record_sha256=row["native_record_sha256"],
                    wall_time_s=0.25,
                )
                for row in self.costs["rows"]
            ],
        )
        self.generate(walltime_ref=self.artifact("wall.json", payload))
        text = (self.root / "report/E18_CONFIRMATION_REPORT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Native调用wall", text)
        self.assertIn("历史所需缓存成功记录覆盖3次", text)
        self.assertIn("独占GPU核时间未计量", text)

    def test_stored_p_and_holm_are_not_recomputed(self):
        before = copy.deepcopy(self.result["results"])
        self.generate()
        self.assertEqual(self.result["results"], before)
        self.assertNotIn("scipy", report.__dict__)

    def test_zero_and_one_pp_interpretations_are_distinct(self):
        zero, minimum = report.inference_text(
            dict(estimate=0.005, confidence_interval=[-0.002, 0.012], p_value=0.2)
        )
        self.assertIn("证据不足", zero)
        self.assertIn("等效性需要", zero)
        self.assertIn("跨越", minimum)

    def test_mutated_input_order_rejected(self):
        path = self.root / "bad-order.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in reversed(self.inputs)),
            encoding="utf-8",
        )
        self.inputs_ref = self.ref(path)
        with self.assertRaises(ValueError):
            self.generate()

    def test_frozen_mechanism_registry_remains_distinct(self):
        self.generate(mechanism_registry_ref=self.ref(PREP / "A_SOURCE_MANIFEST.json"))
        text = (self.root / "report/E18_CONFIRMATION_REPORT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("旧600题的100次", text)
        self.assertIn("6.135%", text)
        self.assertIn("独立质量结论", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification output is append-only")
    wall = time.perf_counter()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ReportQualification)
    )
    payload = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        generator_source_sha256=report.file_sha(report.__file__),
        test_source_sha256=report.file_sha(__file__),
        formal_e18_result_reads=0,
        formal_gold_reads=0,
        formal_scoring_calls=0,
        model_calls=0,
        github_sends=0,
        wall_seconds=time.perf_counter() - wall,
        test_output=stream.getvalue(),
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(dict(status=payload["status"], tests=result.testsRun)))
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
