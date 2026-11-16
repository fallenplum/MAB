"""Read-only semantic comparison for a completed evaluable-only scoring run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

from compare_bulk_with_previous_scoring import STABLE_FIELDS, logical_key


def compare(previous_database, bulk_receipts, completion_result, *, expected_previous=5896):
    with Path(completion_result).open(encoding="utf-8-sig") as stream:
        complete = json.load(stream)
    if complete.get("status") not in {"COMPLETE", "COMPLETE_WITH_UNSCORABLE_INPUTS"}:
        raise ValueError("The evaluable-only run has not completed")
    expected_bulk = complete["scored_aggregate_keys"]
    uri = "file:"+quote(Path(previous_database).resolve().as_posix(), safe="/:")+"?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        old = {logical_key(receipt): receipt for (raw,) in connection.execute("SELECT receipt_json FROM scores")
               for receipt in (json.loads(raw),)}
        old_states = dict(connection.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status"))
    if len(old) != expected_previous:
        raise ValueError("The old committed count differs from the frozen transition")
    seen, found, mismatch_fields = set(), set(), Counter()
    mismatched, previous_input, previous_output, current_input, current_output = 0, 0, 0, 0, 0
    path = Path(bulk_receipts)
    before = path.stat()
    with path.open(encoding="utf-8-sig") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            receipt = json.loads(raw)
            key = logical_key(receipt)
            if key in seen:
                raise ValueError("The completed evaluable stream repeats a scored key")
            seen.add(key)
            if key not in old:
                continue
            previous = old[key]
            found.add(key)
            differing = [field for field in STABLE_FIELDS if previous.get(field) != receipt.get(field)]
            mismatch_fields.update(differing)
            mismatched += bool(differing)
            previous_input += previous["actual_source_input_tokens"]
            previous_output += previous["actual_source_output_tokens"]
            current_input += receipt["actual_source_input_tokens"]
            current_output += receipt["actual_source_output_tokens"]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(seen) != expected_bulk:
        raise ValueError("The completed receipt stream changed or differs from its valid-key count")
    missing = len(old)-len(found)
    return dict(schema="e18_previous_vs_evaluable_scoring_comparison_v1",
        status="PASS_ALL_PREVIOUS_COMMITTED_SCORES_MATCH" if missing == mismatched == 0 else "MISMATCH_REQUIRES_REVIEW",
        previous_committed_scores=len(old), matched_previous_keys=len(found), missing_previous_keys=missing,
        mismatched_keys=mismatched, mismatch_field_counts=dict(mismatch_fields),
        complete_evaluable_scored_keys=len(seen), requested_questions=complete["requested_questions"],
        valid_questions=complete["valid_questions"], excluded_questions=complete["excluded_questions"],
        compared_stable_fields=list(STABLE_FIELDS), previous_attempt_status_counts=old_states,
        matched_source_tokens=dict(previous_prompt_tokens=previous_input, new_prompt_tokens=current_input,
                                   previous_completion_tokens=previous_output, new_completion_tokens=current_output),
        omitted_operational_fields="run IDs, timings, local cache flags, and explicitly skipped whole-execution-receipt digest",
        standard_answer_reads=0, original_question_or_native_answer_reads=0,
        model_calls=0, scorer_calls=0, input_files_modified=0, private_question_identifiers_in_report=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-database", "bulk-receipts", "completion-result", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-previous-count", type=int, default=5896)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Comparison results are append-only")
    result = compare(args.previous_database, args.bulk_receipts, args.completion_result,
                     expected_previous=args.expected_previous_count)
    result["comparison_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "previous_committed_scores", "matched_previous_keys", "mismatched_keys", "missing_previous_keys")}))
    if result["status"] != "PASS_ALL_PREVIOUS_COMMITTED_SCORES_MATCH":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
