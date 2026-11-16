"""Independent report-unit and disclosure controls over synthetic fixtures only."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import io
import json
from pathlib import Path
import time
import unittest

import numpy as np

import generate_final_report as report
import qualify_final_report as fixtures


GENERATOR_SHA = "fa5a93ad7cad1820a3836a30aa643a773c24569d13be7c7080d2172bf1cbbfbe"
QUALIFIER_SHA = "9070d8b4fff1b68ab92852ebc09efbad8068344a91e06ee3df04d44c3079a580"


class IndependentReportReview(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReportQualification(
            "test_complete_toy_report_has_four_deidentified_artifacts"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def artifact_text(self, name):
        return (self.fixture.root / "report" / name).read_text(encoding="utf-8-sig")

    def test_task_weights_and_percent_units_from_raw_toy_events(self):
        self.fixture.generate()
        table = list(csv.DictReader(io.StringIO(self.artifact_text("MAIN_TABLE.csv"))))
        raw = self.fixture.arrays["rewards"]
        tasks = self.fixture.arrays["task_labels"]
        for index, row in enumerate(table):
            by_task = {
                task: raw[index][:, np.flatnonzero(tasks == task)].sum()
                / (3 * np.sum(tasks == task))
                for task in ("gsm8k", "bbh")
            }
            self.assertAlmostEqual(float(row["GSM8K准确率_pct"]), 100 * by_task["gsm8k"])
            self.assertAlmostEqual(float(row["BBH准确率_pct"]), 100 * by_task["bbh"])
            self.assertAlmostEqual(float(row["任务宏平均_pct"]), 50 * sum(by_task.values()))
            self.assertEqual(int(row["原题数"]), self.fixture.n)
            self.assertEqual(int(row["生成事件数"]), 3 * self.fixture.n)

    def test_stored_primary_and_holm_values_are_rendered_without_replacement(self):
        changed = copy.deepcopy(self.fixture.result)
        primary = changed["results"]["primary"]
        primary["p_value"] = 0.12345678
        primary["confidence_interval"] = [
            max(-1.0, primary["estimate"] - 0.2),
            min(1.0, primary["estimate"] + 0.2),
        ]
        for index, secondary in enumerate(changed["results"]["secondary"].values()):
            secondary["p_value"] = 0.201 + index * 0.01
            secondary["holm_adjusted_p"] = 0.901 + index * 0.01
        self.fixture.result_ref = self.fixture.artifact("stored-sentinels.json", changed)
        before = copy.deepcopy(changed)
        self.fixture.generate()
        text = self.artifact_text("E18_CONFIRMATION_REPORT.md")
        self.assertIn("0.12345678", text)
        lo, hi = primary["confidence_interval"]
        self.assertIn(f"[{100 * lo:+.3f}, {100 * hi:+.3f}]pp", text)
        for secondary in changed["results"]["secondary"].values():
            self.assertIn(format(secondary["holm_adjusted_p"], ".8g"), text)
        self.assertEqual(changed, before)

    def test_logical_service_total_spans_three_seeds_and_mean_denominator_is_three_n(self):
        self.fixture.generate()
        table = list(csv.DictReader(io.StringIO(self.artifact_text("COST_TABLE.csv"))))
        service = [row for row in table if row["口径"] == "每策略逻辑服务投影"]
        self.assertEqual(len(service), 4)
        for row in service:
            self.assertEqual(int(row["成员调用数"]), 9 * self.fixture.n)
            self.assertEqual(int(row["聚合调用数"]), 3 * self.fixture.n)
            # Toy source records cost 3*(10+3)+(20+4)=63 tokens per single-seed service.
            self.assertEqual(int(row["总Token"]), 63 * 3 * self.fixture.n)
            self.assertEqual(int(row["总Token"]) / (3 * self.fixture.n), 63)
            self.assertEqual(row["独占GPU核秒"], "未计量")

    def test_shared_action_uses_one_physical_record_in_union_but_each_service_projection(self):
        self.fixture.generate()
        table = list(csv.DictReader(io.StringIO(self.artifact_text("COST_TABLE.csv"))))
        joint = next(row for row in table if row["口径"] == "已知物理执行并集")
        logical = [row for row in table if row["口径"] == "每策略逻辑服务投影"]
        success_rows = [row for row in self.fixture.costs["rows"] if row["status"] == "success"]
        semantic = {(row["stage"], *row["task_key"]) for row in success_rows}
        self.assertEqual(len(semantic), len(success_rows))
        self.assertLess(int(joint["总Token"]), sum(int(row["总Token"]) for row in logical))

    def test_historical_failed_record_is_rejected(self):
        changed = copy.deepcopy(self.fixture.costs)
        historical = next(row for row in changed["rows"] if row["reused_historical_source"])
        historical["status"] = "failure"
        self.fixture.costs_ref = self.fixture.artifact("historical-failure.json", changed)
        with self.assertRaises(ValueError):
            self.fixture.generate()

    def test_public_artifacts_omit_toy_ids_text_hashes_native_attempts_and_private_paths(self):
        self.fixture.generate()
        text = "\n".join(
            path.read_text(encoding="utf-8-sig")
            for path in (self.fixture.root / "report").iterdir()
        )
        forbidden = [str(self.fixture.root)]
        forbidden.extend(self.fixture.arrays["question_ids"].tolist())
        forbidden.extend(self.fixture.arrays["canonical_hashes"].tolist())
        forbidden.extend(row["question"] for row in self.fixture.inputs)
        forbidden.extend(row["physical_attempt_id"] for row in self.fixture.costs["rows"])
        for value in forbidden:
            self.assertNotIn(value, text)
        self.assertIn("准备期曾获批准的一次sealed输入指纹审计单独留档", text)
        self.assertIn("protected input值解码为0", text)
        self.assertIn("优势包括", text)
        self.assertIn("风险包括", text)

    def test_zero_and_one_pp_require_separate_interpretation(self):
        zero, minimum = report.inference_text(
            dict(estimate=0.005, confidence_interval=[0.001, 0.008], p_value=0.01)
        )
        self.assertIn("正向", zero)
        self.assertIn("整体低于", minimum)
        zero, minimum = report.inference_text(
            dict(estimate=0.0, confidence_interval=[-0.01, 0.01], p_value=0.9)
        )
        self.assertIn("证据不足", zero)
        self.assertIn("等效性需要独立", zero)
        self.assertIn("跨越", minimum)

    def test_report_module_imports_no_inference_scoring_reference_or_model_library(self):
        tree = ast.parse(Path(report.__file__).read_text(encoding="utf-8"))
        imported = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        self.assertFalse(
            any(
                name.startswith(("scipy", "torch", "transformers", "statsmodels"))
                for name in imported
            )
        )
        self.assertFalse(
            calls
            & {
                "ttest_rel",
                "ttest_1samp",
                "paired_t",
                "bootstrap",
                "get_reference",
                "score_one",
                "generate",
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Independent review receipts are append-only")
    if (
        report.file_sha(report.__file__) != GENERATOR_SHA
        or report.file_sha(fixtures.__file__) != QUALIFIER_SHA
    ):
        raise ValueError("Requested frozen generator or qualifier bytes differ")
    started = time.perf_counter()
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(IndependentReportReview)
    )
    unchanged = (
        report.file_sha(report.__file__) == GENERATOR_SHA
        and report.file_sha(fixtures.__file__) == QUALIFIER_SHA
    )
    payload = dict(
        schema="e18_independent_final_report_review_v1",
        status="PASS" if result.wasSuccessful() and unchanged else "FAIL",
        tests=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        generator_source_sha256=GENERATOR_SHA,
        qualifier_source_sha256=QUALIFIER_SHA,
        independent_review_source_sha256=report.file_sha(__file__),
        frozen_sources_unchanged=unchanged,
        formal_e18_result_reads=0,
        formal_gold_reads=0,
        physical_model_calls=0,
        formal_scoring_calls=0,
        new_formal_inference_tests=0,
        wall_seconds=time.perf_counter() - started,
        test_output=capture.getvalue(),
        cost_unit_note=(
            "Logical policy service rows are totals over three seeds and N questions, "
            "with 9N member and 3N aggregate calls. A mean per single-question single-seed "
            "service is total/(3N). The current report emits totals, without claiming an average."
        ),
        review_boundary="Only source code, prior aggregate mechanism registry and disposable toy fixtures were read.",
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: payload[key] for key in ("status", "tests", "failures", "errors", "wall_seconds")}
        )
    )
    if payload["status"] != "PASS":
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
