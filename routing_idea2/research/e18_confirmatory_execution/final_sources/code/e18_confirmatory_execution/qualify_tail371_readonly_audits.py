"""Pure-software qualification of the new read-only371 provenance helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unittest

import audit_tail371_export_lineage as export
import audit_tail371_static_sources as static


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


class Qualification(unittest.TestCase):
    def test_frozen_helpers(self):
        self.assertEqual(export.old.file_sha(export.old.__file__), static.LINEAGE_SHA)
        self.assertEqual(export.old.file_sha(static.__file__), export.STATIC_SOURCE_SHA)

    def test_exact(self):
        raw = b'{"from_cache":false,"native_record":{"answer":"toy"}}'
        self.assertEqual(export.cache_flag_relation(raw, checksum(raw)),
                         "EXACT_CHECKPOINT_FILE_SHA")

    def test_archive_true_checkpoint_false(self):
        raw = b'{"from_cache": true,"native_record":{"answer":"toy"}}'
        previous = raw.replace(b'"from_cache": true', b'"from_cache": false')
        self.assertEqual(export.cache_flag_relation(raw, checksum(previous)),
                         "ARCHIVE_CACHE_TRUE_CHECKPOINT_FALSE_ONLY")

    def test_archive_false_checkpoint_true(self):
        raw = b'{"from_cache": false,"native_record":{"answer":"toy"}}'
        later = raw.replace(b'"from_cache": false', b'"from_cache": true')
        self.assertEqual(export.cache_flag_relation(raw, checksum(later)),
                         "ARCHIVE_CACHE_FALSE_CHECKPOINT_TRUE_ONLY")

    def test_nested_change_refused(self):
        raw = b'{"from_cache": false,"nested":{"from_cache":false}}'
        wrong = b'{"from_cache": false,"nested":{"from_cache":true}}'
        with self.assertRaises(ValueError):
            export.cache_flag_relation(raw, checksum(wrong))

    def test_answer_change_refused(self):
        raw = b'{"from_cache":false,"native_record":{"answer":"toy"}}'
        wrong = raw.replace(b'"toy"', b'"changed"')
        with self.assertRaises(ValueError):
            export.cache_flag_relation(raw, checksum(wrong))

    def test_cache_and_answer_change_refused(self):
        raw = b'{"from_cache":false,"native_record":{"answer":"toy"}}'
        wrong = raw.replace(b'false', b'true').replace(b'"toy"', b'"changed"')
        with self.assertRaises(ValueError):
            export.cache_flag_relation(raw, checksum(wrong))

    def test_cache_looking_string_skipped(self):
        raw = b'{"from_cache":false,"answer":"\\\"from_cache\\\":false"}'
        later = raw.replace(b'"from_cache":false,', b'"from_cache":true,')
        self.assertEqual(export.cache_flag_relation(raw, checksum(later)),
                         "ARCHIVE_CACHE_FALSE_CHECKPOINT_TRUE_ONLY")

    def test_multiple_top_level_flags_refused(self):
        raw = b'{"from_cache":false,"from_cache":false}'
        wrong = b'{"from_cache":true,"from_cache":false}'
        with self.assertRaises(ValueError):
            export.cache_flag_relation(raw, checksum(wrong))

    def test_missing_flag_refused(self):
        raw = b'{"native_record":{"answer":"toy"}}'
        with self.assertRaises(ValueError):
            export.cache_flag_relation(raw, "0" * 64)

    def test_metadata_protected_value_refused(self):
        with self.assertRaises(ValueError):
            export.old.metadata_json(b'{"answer":"toy"}')

    def test_metadata_receipt_hash_allowed(self):
        raw = b'{"native_receipt_file_sha256":"' + b'0' * 64 + b'"}'
        self.assertEqual(export.old.metadata_json(raw)["native_receipt_file_sha256"], "0" * 64)

    def test_explicit_parent_identity(self):
        self.assertEqual(static.PARENT_SHA,
                         "5fc23696b12239804d4e22eb7fa4a7c15c0c3882e8b7ebc2ec2b0c56ee762995")
        self.assertEqual(static.PLAN_SHA,
                         "af3fddaa5ec89e4724d3dbcc840c2b20abc239b8bb0192c4a60628fe075d66b3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Qualification receipts are append-only")
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Qualification))
    report = dict(
        schema="e18_tail371_readonly_audit_qualification_v1",
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        static_audit_source_sha256=export.old.file_sha(static.__file__),
        export_audit_source_sha256=export.old.file_sha(export.__file__),
        qualification_source_sha256=export.old.file_sha(__file__),
        metadata_helper_sha256=static.LINEAGE_SHA,
        software_test_only=True, model_calls=0, gold_reads=0,
        real_native_record_reads=0, real_question_or_reward_reads=0, signals_sent=0,
    )
    with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(report))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
