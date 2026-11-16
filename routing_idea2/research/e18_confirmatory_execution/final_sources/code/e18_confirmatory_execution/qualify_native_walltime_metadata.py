"""Pure software controls for answer-blind native call-wall metadata extraction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest import mock

import generate_final_report as report
import native_walltime_metadata as tool


SENTINEL = "SYNTHETIC_UNSELECTED_ANSWER_VALUE"


def native_row(attempt="toy-attempt-1", wall=1.25, include_wall=True):
    result = dict(
        text=SENTINEL + ' \\"wall_time_s\\":99999\n\u2028 same physical JSONL record',
        backend="transformers",
        raw_usage=dict(unselected=SENTINEL),
    )
    if include_wall:
        result["wall_time_s"] = wall
    value = dict(
        attempt_id=attempt,
        status="success",
        cost_status="recorded_actual_usage",
        extra_unselected=dict(answer=SENTINEL, nested=[1, True, None, {"text": SENTINEL}]),
        result=result,
    )
    value["record_sha256"] = tool.canonical_digest(value)
    return value


def raw_row(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=True).encode() + b"\n"


class MetadataQualification(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e18-native-walltime-toy-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def archive(self, row=None, *, bad_file_sha=False, bad_task_identity=False, trailing=False):
        row = row or native_row()
        prefix = "a" * 64 + "/member_generation/" + "b" * 64
        files = {
            prefix + "/ATTEMPT_STATE.json": b'{"status":"SUCCESS_COMMITTED"}',
            prefix + "/LAST_TASK_RECEIPT.json": raw_row(dict(text=SENTINEL)),
            prefix + "/records.jsonl": raw_row(row),
        }
        declared = [
            dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            for name, data in files.items()
        ]
        if bad_file_sha:
            declared[-1]["sha256"] = "0" * 64
        task = dict(
            physical_attempt_id=row["attempt_id"], native_record_sha256=row["record_sha256"]
        )
        if bad_task_identity:
            task["physical_attempt_id"] = "unexecuted-other-attempt"
        manifest = dict(schema="e18_closed_native_export_manifest_v1", files=declared, tasks=[task])
        files["EXPORT_MANIFEST.json"] = json.dumps(manifest).encode()
        if trailing:
            files["after_manifest.json"] = b"{}"
        path = self.root / "toy-native.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        expected = dict(
            bytes=path.stat().st_size,
            sha256=tool.file_sha(path),
            tasks=1,
            manifest_sha256=tool.canonical_digest(manifest),
        )
        return path, expected

    def test_metadata_keys_and_seconds_are_projected(self):
        row = native_row()
        projected = tool.project_native(raw_row(row))
        self.assertEqual(projected["physical_attempt_id"], row["attempt_id"])
        self.assertEqual(projected["native_record_sha256"], row["record_sha256"])
        self.assertEqual(projected["wall_time_s"], 1.25)
        self.assertNotIn("text", projected)

    def test_generated_text_and_nested_answer_values_are_never_json_decoded(self):
        original = json.loads

        def guarded(value, *args, **kwargs):
            data = value.encode() if isinstance(value, str) else value
            self.assertNotIn(SENTINEL.encode(), data)
            return original(value, *args, **kwargs)

        with mock.patch.object(tool.json, "loads", side_effect=guarded):
            self.assertEqual(tool.project_native(raw_row(native_row()))["wall_time_s"], 1.25)

    def test_json_lookalike_answer_string_cannot_change_metadata(self):
        row = native_row()
        row["result"]["text"] = '{"attempt_id":"fake","wall_time_s":88888}'
        self.assertEqual(tool.project_native(raw_row(row))["wall_time_s"], 1.25)

    def test_missing_and_null_are_unmeasured_without_scheduler_fallback(self):
        missing = native_row(include_wall=False)
        missing["wall_time_s"] = 999
        for row in (missing, native_row(wall=None)):
            result = tool.ledger_from_metadata([tool.project_native(raw_row(row))])
            self.assertEqual(result["rows"], [])
            self.assertEqual(result["measurement_status"], "UNMEASURED")
            self.assertIsNone(result["summary"]["recorded_call_wall_seconds"])
            self.assertEqual(result["summary"]["unmeasured_attempts"], 1)

    def test_zero_is_a_recorded_measurement(self):
        result = tool.ledger_from_metadata([tool.project_native(raw_row(native_row(wall=0)))])
        self.assertEqual(result["summary"]["measured_attempts"], 1)
        self.assertEqual(result["rows"][0]["wall_time_s"], 0)

    def test_invalid_wall_types_or_values_fail_closed(self):
        for value in (-1, True, "1.0", float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    tool.project_native(raw_row(native_row(wall=value)))

    def test_selected_container_is_rejected_before_decoding(self):
        row = native_row()
        row["result"]["wall_time_s"] = dict(answer=SENTINEL)
        with self.assertRaises(ValueError):
            tool.project_native(raw_row(row))

    def test_duplicate_native_or_result_key_is_rejected(self):
        for blob in (
            b'{"attempt_id":"a","attempt_id":"b"}',
            b'{"result":{"wall_time_s":1,"wall_time_s":2}}',
        ):
            with self.assertRaises(ValueError):
                tool.project_native(blob)

    def test_exact_attempt_duplicates_are_counted_once(self):
        row = tool.project_native(raw_row(native_row()))
        result = tool.ledger_from_metadata([row, copy.deepcopy(row)])
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["summary"]["duplicate_occurrences_removed"], 1)
        self.assertEqual(result["summary"]["recorded_call_wall_seconds"], 1.25)

    def test_conflicting_attempt_or_native_identity_is_rejected(self):
        row = tool.project_native(raw_row(native_row()))
        changed = dict(row, wall_time_s=3)
        with self.assertRaises(ValueError):
            tool.ledger_from_metadata([row, changed])
        changed = dict(row, physical_attempt_id="other-attempt")
        with self.assertRaises(ValueError):
            tool.ledger_from_metadata([row, changed])

    def test_streamed_record_file_and_manifest_binding(self):
        path, expected = self.archive()
        rows = tool.read_archive_metadata(path, expected)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["wall_time_s"], 1.25)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_archive_external_sha_or_record_file_sha_rejected(self):
        path, expected = self.archive(bad_file_sha=True)
        with self.assertRaises(ValueError):
            tool.read_archive_metadata(path, expected)
        with self.assertRaises(ValueError):
            tool.read_archive_metadata(path, dict(expected, sha256="0" * 64))

    def test_task_identity_mismatch_and_after_manifest_entry_rejected(self):
        path, expected = self.archive(bad_task_identity=True)
        with self.assertRaises(ValueError):
            tool.read_archive_metadata(path, expected)
        path.unlink()
        path, expected = self.archive(trailing=True)
        with self.assertRaises(ValueError):
            tool.read_archive_metadata(path, expected)

    def test_single_inspection_stops_at_one_record(self):
        path, expected = self.archive()
        result = tool.inspect_one(path, expected["sha256"], expected["bytes"])
        self.assertEqual(result["native_records_projected"], 1)
        self.assertEqual(result["generated_text_values_decoded"], 0)

    def test_frozen_report_schema_accepts_partial_known_wall_without_gpu_kernel_claim(self):
        first = tool.project_native(raw_row(native_row("new-attempt", wall=1.25)))
        second = tool.project_native(raw_row(native_row("historical-attempt", wall=2.5)))
        missing = tool.project_native(raw_row(native_row("unmeasured-attempt", include_wall=False)))
        ledger = tool.ledger_from_metadata([first, second, missing])
        path = self.root / "private-wall.json"
        tool.write_new_json(path, ledger)
        costs = dict(
            rows=[
                dict(
                    physical_attempt_id=row["physical_attempt_id"],
                    native_record_sha256=row["native_record_sha256"],
                    reused_historical_source=row is second,
                )
                for row in (first, second, missing)
            ]
        )
        actual = report.wall_measurements(dict(path=str(path), sha256=tool.file_sha(path)), costs)
        self.assertEqual((actual["measured_attempts"], actual["total_attempts"]), (2, 3))
        self.assertEqual(actual["seconds"], 3.75)
        self.assertEqual(actual["origins"]["required_historical_success"]["seconds"], 2.5)
        self.assertFalse(actual["full_coverage"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Software qualification receipts are append-only")
    report_sha = tool.file_sha(report.__file__)
    start = time.perf_counter()
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(MetadataQualification)
    )
    payload = dict(
        schema="e18_native_walltime_metadata_qualification_v1",
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        tool_source_sha256=tool.file_sha(tool.__file__),
        qualification_source_sha256=tool.file_sha(__file__),
        frozen_report_source_sha256=report_sha,
        frozen_report_source_unchanged=tool.file_sha(report.__file__) == report_sha,
        software_fixtures_only=True,
        production_archives_scanned=0,
        generated_answer_values_json_decoded=0,
        model_or_scoring_calls=0,
        gold_reads=0,
        ssh_logins=0,
        files_extracted=0,
        wall_seconds=time.perf_counter() - start,
        test_output=capture.getvalue(),
    )
    tool.write_new_json(args.output, payload)
    print(json.dumps({name: payload[name] for name in ("status", "tests", "failures", "errors")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
