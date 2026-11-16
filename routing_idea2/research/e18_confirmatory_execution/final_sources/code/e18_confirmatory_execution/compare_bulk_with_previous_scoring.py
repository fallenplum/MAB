"""Read-only comparison of committed old scores and completed bulk receipts.

No standard-answer source, native answer, reward matrix or model is opened.
Only previously produced sparse scoring receipts and completion metadata are
read. The input SQLite is opened with mode=ro, and both scoring runs are kept.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote


STABLE_FIELDS = (
    "question_id", "canonical_id", "generation_seed", "aggregation_seed",
    "combination_index", "team", "reference_version", "reference_source_identity_hash",
    "scorer_version", "scorer_source_sha256", "aggregator_fingerprint",
    "approval_sha256", "scope_sha256", "bindings", "software_test_only",
    "aggregation_record_sha256", "physical_attempt_id", "actual_source_input_tokens",
    "actual_source_output_tokens", "execution_resources_already_charged_upstream",
    "incremental_member_calls", "incremental_aggregation_calls", "scoring_model_calls",
    "score", "parsed", "reason",
)


def logical_key(receipt):
    return (receipt["canonical_id"], receipt["reference_version"],
            receipt["generation_seed"], receipt["combination_index"])


def compare(previous_database, bulk_receipts, completion_result, *, expected_previous=5896,
            expected_bulk=30921):
    """Return aggregate consistency findings after both runs have stopped writing."""
    previous_database, bulk_receipts = Path(previous_database), Path(bulk_receipts)
    with Path(completion_result).open(encoding="utf-8-sig") as stream:
        completion = json.load(stream)
    if completion.get("status") != "COMPLETE" or completion.get("unique_aggregate_keys") != expected_bulk:
        raise ValueError("The bulk scorer has not published the expected completed cohort")
    bulk_before = bulk_receipts.stat()
    uri = "file:" + quote(previous_database.resolve().as_posix(), safe="/:") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        previous = {}
        for source_sha, sparse_json, receipt_json in connection.execute(
            "SELECT source_record_sha256,sparse_json,receipt_json FROM scores"
        ):
            receipt, sparse = json.loads(receipt_json), json.loads(sparse_json)
            key = logical_key(receipt)
            if key in previous or receipt["aggregation_record_sha256"] != source_sha:
                raise ValueError("The old committed scoring journal has repeated or conflicting sources")
            if receipt["score"] != sparse["score"]:
                raise ValueError("Old sparse and committed receipt scores disagree")
            previous[key] = receipt
        old_attempt_states = dict(connection.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status"))
    if len(previous) != expected_previous:
        raise ValueError("The old successful-score count differs from the frozen transition record")
    found, seen_bulk = set(), set()
    mismatch_fields = Counter()
    mismatched_keys = 0
    old_input = old_output = bulk_input = bulk_output = 0
    total_bulk = 0
    for_line_errors = 0
    with bulk_receipts.open(encoding="utf-8-sig") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            current = json.loads(raw)
            key = logical_key(current)
            total_bulk += 1
            if key in seen_bulk:
                raise ValueError("Bulk receipt stream repeats a logical scored key")
            seen_bulk.add(key)
            if current.get("schema") != "e18_sparse_scoring_receipt_v2":
                for_line_errors += 1
            if key not in previous:
                continue
            found.add(key)
            old = previous[key]
            differences = [field for field in STABLE_FIELDS if old.get(field) != current.get(field)]
            mismatch_fields.update(differences)
            mismatched_keys += bool(differences)
            old_input += old["actual_source_input_tokens"]
            old_output += old["actual_source_output_tokens"]
            bulk_input += current["actual_source_input_tokens"]
            bulk_output += current["actual_source_output_tokens"]
    bulk_after = bulk_receipts.stat()
    if (bulk_before.st_size, bulk_before.st_mtime_ns) != (bulk_after.st_size, bulk_after.st_mtime_ns):
        raise ValueError("Bulk receipts changed while the read-only comparison was running")
    if total_bulk != expected_bulk or for_line_errors:
        raise ValueError("The complete bulk scoring stream count or schema differs")
    missing = len(previous)-len(found)
    passed = missing == mismatched_keys == 0
    return dict(
        schema="e18_previous_vs_bulk_scoring_readonly_comparison_v1",
        status="PASS_ALL_PREVIOUS_COMMITTED_SCORES_MATCH_BULK" if passed else "SCORING_COMPARISON_FAILED",
        previous_committed_scores=len(previous), complete_bulk_scores=total_bulk,
        matched_previous_keys=len(found), missing_previous_keys=missing,
        mismatched_keys=mismatched_keys, mismatch_field_counts=dict(mismatch_fields),
        compared_stable_fields=list(STABLE_FIELDS),
        matching_subset_tokens=dict(previous_input_tokens=old_input, bulk_input_tokens=bulk_input,
                                   previous_output_tokens=old_output, bulk_output_tokens=bulk_output),
        previous_attempt_status_counts=old_attempt_states,
        excluded_operational_fields=["scoring_attempt_id", "scoring_receipt_sha256", "CPU/wall timings",
                                     "run-local reference cache/reveal flags", "source_execution_receipt_sha256"],
        excluded_field_reason="Bulk batches I/O and explicitly reuses the already completed native audit; scores, frozen scorer identity, selected native sources and actual Token fields remain comparable",
        completed_result_status=completion["status"],
        standard_answer_sources_read=0, native_answer_sources_read=0,
        real_reward_matrices_read=0, model_calls=0, scorer_calls=0,
        source_database_mode="read_only", input_files_modified=0,
        private_question_identifiers_in_this_report=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-database", type=Path, required=True)
    parser.add_argument("--bulk-receipts", type=Path, required=True)
    parser.add_argument("--completion-result", type=Path, required=True)
    parser.add_argument("--expected-previous-count", type=int, default=5896)
    parser.add_argument("--expected-bulk-count", type=int, default=30921)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Comparison results use new append-only filenames")
    result = compare(args.previous_database, args.bulk_receipts, args.completion_result,
                     expected_previous=args.expected_previous_count, expected_bulk=args.expected_bulk_count)
    result["comparison_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "previous_committed_scores",
                                                 "complete_bulk_scores", "mismatched_keys",
                                                 "missing_previous_keys")}, ensure_ascii=False))
    if result["status"] != "PASS_ALL_PREVIOUS_COMMITTED_SCORES_MATCH_BULK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
