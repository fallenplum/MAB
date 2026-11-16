"""Eight artificial-TAR controls for audited-receipt scoring assembly."""

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
from unittest.mock import patch

WRAPPER_PATH = Path(__file__).with_name("fast_scoring_from_audit.py")
WRAPPER_IMPORT_SHA256 = hashlib.sha256(WRAPPER_PATH.read_bytes()).hexdigest()

import fast_scoring_from_audit as tool

if hashlib.sha256(WRAPPER_PATH.read_bytes()).hexdigest() != WRAPPER_IMPORT_SHA256:
    raise RuntimeError("The assembly source changed while it was imported")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


class ReceiptAssemblyQualification(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e18-fast-receipt-toy-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scope_sha = digest("synthetic fast scoring scope")
        self.member_key = ("synthetic-fast", 42, 0)
        self.aggregate_key = ("synthetic-fast", 42, 0)
        self.member = self.receipt("member_generation", self.member_key, 8, 4)
        self.aggregate = self.receipt("aggregation", self.aggregate_key, 16, 5)
        self.archive_number = 0
        self.expected = {
            "member_generation": dict(tasks=1, prompt_tokens=8, completion_tokens=4),
            "aggregation": dict(tasks=1, prompt_tokens=16, completion_tokens=5),
        }

    def receipt(self, stage, key, prompt, completion):
        qid, seed, index = key
        native = dict(
            question_id=qid,
            seed=seed,
            task="gsm8k",
            subtask=None,
            status="success",
            cost_status="recorded_actual_usage",
            attempt_id="synthetic-" + stage + "-" + str(index),
            job_id="synthetic-job-" + stage,
            result=dict(
                text="SYNTHETIC_UNSCORED_OUTPUT",
                prompt_tokens=prompt,
                completion_tokens=completion,
                wall_time_s=0.125,
            ),
        )
        native["record_sha256"] = digest(native)
        task_key = dict(
            question_id=qid,
            canonical_id=digest(["synthetic canonical", qid]),
            generation_seed=seed,
        )
        if stage == "member_generation":
            task_key["member_index"] = index
        else:
            task_key.update(combination_index=index, team=[0, 1, 2])
        return dict(
            schema="e18_execution_task_receipt_v2",
            status="SUCCESS",
            stage=stage,
            cohort="e18_confirm",
            scope_sha256=self.scope_sha,
            task_key=task_key,
            native_record=native,
            native_record_sha256=native["record_sha256"],
            native_job_id=native["job_id"],
            physical_attempt_id=native["attempt_id"],
            # Artificial bytes emulate the production-only loader's mode field.
            software_test_only=False,
            from_cache=False,
        )

    def archive(self, receipts=None, *, forbidden_payloads=False):
        if receipts is None:
            receipts = [self.member, self.aggregate]
        path = self.root / ("synthetic-" + str(self.archive_number) + ".tar.gz")
        self.archive_number += 1
        with tarfile.open(path, "w:gz") as stream:
            for receipt in receipts:
                key = receipt["task_key"]
                ordinal = key.get("member_index", key.get("combination_index"))
                prefix = "/".join(
                    [self.scope_sha, receipt["stage"], digest([key["question_id"], key["generation_seed"], ordinal])]
                )
                name = prefix + "/LAST_TASK_RECEIPT.json"
                self.add_entry(stream, name, json.dumps(receipt).encode())
                if forbidden_payloads:
                    for basename in ("records.jsonl", "ATTEMPT_STATE.json", "INFLIGHT.json"):
                        self.add_entry(
                            stream,
                            prefix + "/" + basename,
                            b"FORBIDDEN_SYNTHETIC_NON_JSON_PAYLOAD",
                        )
        return path

    def add_entry(self, stream, name, data):
        member = tarfile.TarInfo(name)
        member.size = len(data)
        stream.addfile(member, io.BytesIO(data))

    def assemble(self, paths=None, **kwargs):
        return tool.assemble_archived_receipts(
            [self.archive()] if paths is None else paths,
            self.scope_sha,
            {self.member_key},
            {self.aggregate_key},
            **kwargs,
        )

    def test_complete_two_task_archive_has_exact_sets_and_measured_totals(self):
        loaded = self.assemble(expected_stage_totals=self.expected)
        self.assertEqual(set(loaded["members"]), {self.member_key})
        self.assertEqual(set(loaded["by_key"]), {self.aggregate_key})
        self.assertEqual(loaded["receipts"], [self.aggregate])
        self.assertEqual(loaded["stage_totals"], self.expected)
        self.assertEqual(len(loaded["costs"]), 2)

    def test_duplicate_logical_key_and_physical_attempt_are_rejected(self):
        with self.assertRaises(ValueError):
            self.assemble([self.archive([self.member, self.aggregate, self.member])])
        duplicate = copy.deepcopy(self.aggregate)
        duplicate["physical_attempt_id"] = self.member["physical_attempt_id"]
        duplicate["native_record"]["attempt_id"] = duplicate["physical_attempt_id"]
        duplicate["native_record"]["record_sha256"] = digest(
            {key: value for key, value in duplicate["native_record"].items() if key != "record_sha256"}
        )
        duplicate["native_record_sha256"] = duplicate["native_record"]["record_sha256"]
        with self.assertRaises(ValueError):
            self.assemble([self.archive([self.member, duplicate])])

    def test_historical_members_require_disjoint_exact_union(self):
        external_key = ("synthetic-external", 42, 1)
        external = self.receipt("member_generation", external_key, 7, 3)["native_record"]
        loaded = self.assemble()
        result = tool.append_historical_members(
            loaded, {external_key: external}, {self.member_key, external_key}
        )
        self.assertIs(result, loaded)
        self.assertEqual(set(loaded["members"]), {self.member_key, external_key})
        self.assertTrue(loaded["costs"][external["attempt_id"]]["reused_historical_source"])
        with self.assertRaises(ValueError):
            tool.append_historical_members(
                self.assemble(), {self.member_key: self.member["native_record"]}, {self.member_key}
            )
        with self.assertRaises(ValueError):
            tool.append_historical_members(
                self.assemble(), {}, {self.member_key, external_key}
            )

    def test_missing_required_task_and_wrong_scope_are_rejected(self):
        with self.assertRaises(ValueError):
            self.assemble([self.archive([self.aggregate])])
        changed = copy.deepcopy(self.member)
        changed["scope_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.assemble([self.archive([changed, self.aggregate])])

    def test_wrong_receipt_stage_is_rejected(self):
        changed = copy.deepcopy(self.member)
        changed["stage"] = "unexpected_stage"
        with self.assertRaises(ValueError):
            self.assemble([self.archive([changed, self.aggregate])])

    def test_native_records_and_fences_are_never_extracted_or_json_decoded(self):
        path = self.archive(forbidden_payloads=True)
        original = tarfile.TarFile.extractfile
        extracted = []

        def receipt_only(stream, member, *args, **kwargs):
            name = member if isinstance(member, str) else member.name
            self.assertEqual(Path(name).name, "LAST_TASK_RECEIPT.json")
            extracted.append(name)
            return original(stream, member, *args, **kwargs)

        with patch.object(tarfile.TarFile, "extractfile", receipt_only):
            loaded = self.assemble([path])
        self.assertEqual(len(extracted), 2)
        self.assertEqual(len(loaded["costs"]), 2)

    def test_expected_audit_token_mismatch_is_rejected(self):
        expected = copy.deepcopy(self.expected)
        expected["aggregation"]["prompt_tokens"] += 1
        with self.assertRaises(ValueError):
            self.assemble(expected_stage_totals=expected)

    def test_invalid_measured_usage_is_rejected(self):
        for value in (True, -1, None, "8"):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.member)
                changed["native_record"]["result"]["prompt_tokens"] = value
                changed["native_record"]["record_sha256"] = digest(
                    {key: item for key, item in changed["native_record"].items() if key != "record_sha256"}
                )
                changed["native_record_sha256"] = changed["native_record"]["record_sha256"]
                with self.assertRaises(ValueError):
                    self.assemble([self.archive([changed, self.aggregate])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Software qualification records are append-only")
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ReceiptAssemblyQualification)
    )
    unchanged = hashlib.sha256(WRAPPER_PATH.read_bytes()).hexdigest() == WRAPPER_IMPORT_SHA256
    record = dict(
        schema="e18_fast_scoring_from_audit_qualification_v1",
        status="PASS" if result.wasSuccessful() and unchanged else "FAIL",
        tests_run=result.testsRun,
        errors=len(result.errors),
        failures=len(result.failures),
        tool_source_sha256=WRAPPER_IMPORT_SHA256,
        tool_source_unchanged_during_qualification=unchanged,
        qualification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        software_test_only=True,
        synthetic_archive_tasks=2,
        real_receipts_or_native_records_read=0,
        real_gold_or_question_reads=0,
        model_or_scoring_calls=0,
        execution_gate_reruns=0,
        actual_process_signals=0,
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
