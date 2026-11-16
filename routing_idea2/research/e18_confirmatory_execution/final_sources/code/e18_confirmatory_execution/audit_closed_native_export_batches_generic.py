"""Audit any nonempty set of closed E18 export batches, without extraction.

External inventories supply checksums and declarations. File, task and token
totals in the final receipt are accumulated by the independent native tar reader.
No reference source, Gold value or correctness result is consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import audit_closed_native_export_batches as base


BASE_WRAPPER_SHA = "d8eb782c9f9dce78953735dbd13f1294d7661b1a7ebbe039c8364c6e7c252172"


def review(args):
    auditor = base.auditor
    if (
        auditor.file_sha(base.__file__) != BASE_WRAPPER_SHA
        or auditor.file_sha(auditor.__file__) != base.FROZEN_AUDITOR_SHA
    ):
        raise ValueError("Frozen independent tar-reader source or existing helper differs")
    if not args.source_result:
        raise ValueError("At least one closed-export inventory is required")
    context = auditor.FrozenContext(args.asset_root, args.scope, args.approval)
    batches, entries = [], []
    for path in args.source_result:
        source = auditor.read_json(path)
        rows = source["new_archives"]
        if (
            not isinstance(rows, list)
            or not rows
            or source["scope_sha256"] != context.scope_sha
            or source["software_test_only"] is not False
            or any(
                source.get(name) != 0 for name in ("new_model_calls", "scorer_calls", "gold_reads")
            )
        ):
            raise ValueError("A batch changes the approved closed-export scope or mode")
        batches.append(
            dict(
                path=str(path),
                sha256=auditor.file_sha(path),
                bytes=path.stat().st_size,
                archive_count=len(rows),
                native_tasks=sum(row["tasks"] for row in rows),
                compressed_archive_bytes=sum(row["bytes"] for row in rows),
            )
        )
        entries.extend(rows)
    base.write_new_json(
        args.sources_output,
        dict(
            schema="e18_independent_combined_external_inventory_v1",
            scope_sha256=context.scope_sha,
            source_batches=batches,
            new_archives=entries,
        ),
    )
    result = auditor.audit_all(
        context,
        args.sources_output,
        args.archive_directory,
        expected_archives=len(entries),
        expected_tasks=sum(row["tasks"] for row in entries),
    )
    if result["compressed_bytes"] != sum(row["bytes"] for row in entries):
        raise ValueError(
            "Actual compressed-file bytes differ from the external checksums inventory"
        )
    if args.expected_archives is not None and result["archives_checked"] != args.expected_archives:
        raise ValueError("Actual audited archive count differs from the caller's cohort check")
    if args.expected_tasks is not None and result["native_tasks_checked"] != args.expected_tasks:
        raise ValueError("Actual audited task count differs from the caller's cohort check")
    if any(auditor.file_sha(row["path"]) != row["sha256"] for row in batches):
        raise ValueError("A source inventory changed during its audit")
    if (
        auditor.file_sha(base.__file__) != BASE_WRAPPER_SHA
        or auditor.file_sha(auditor.__file__) != base.FROZEN_AUDITOR_SHA
    ):
        raise ValueError("An audit source changed during execution")
    result.update(
        source_batches=batches,
        combined_external_inventory_ref=dict(
            path=str(args.sources_output), sha256=auditor.file_sha(args.sources_output)
        ),
        independent_tar_auditor_sha256=base.FROZEN_AUDITOR_SHA,
        reused_helper_source_sha256=BASE_WRAPPER_SHA,
        cross_batch_native_task_and_physical_attempt_uniqueness=True,
        source_inventories_unchanged=True,
        caller_expected_archives=args.expected_archives,
        caller_expected_tasks=args.expected_tasks,
        totals_basis="Actual streamed native files and independently validated task records",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "asset-root",
        "scope",
        "approval",
        "archive-directory",
        "sources-output",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-result", type=Path, action="append", required=True)
    parser.add_argument("--expected-archives", type=int)
    parser.add_argument("--expected-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists() or args.sources_output.exists():
        raise FileExistsError("Independent audit inputs and receipts are append-only")
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    result = review(args)
    result.update(
        review_source_sha256=base.auditor.file_sha(__file__),
        audit_cpu_seconds=time.process_time() - started_cpu,
        audit_wall_seconds=time.perf_counter() - started_wall,
    )
    base.write_new_json(args.output, result)
    print(
        json.dumps(
            {
                name: result[name]
                for name in (
                    "status",
                    "archives_checked",
                    "native_tasks_checked",
                    "unique_physical_attempts",
                    "native_files_checked",
                    "compressed_bytes",
                    "by_stage",
                    "distinct_question_clusters",
                    "actual_prompt_tokens",
                    "actual_completion_tokens",
                    "actual_total_tokens",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
