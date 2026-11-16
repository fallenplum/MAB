"""Audit multiple closed E18 export inventories through the frozen tar auditor.

Only native generated-output archives and their execution metadata are read.
Reference sources, correctness scores and evaluation results are outside this API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import audit_closed_native_archives as auditor


FROZEN_AUDITOR_SHA = "c506dff8698666fa1928ed922c6d0df746e7a7ca7a0bada0f9d37541d6433948"
EXPECTED_BATCHES = ((12, 5814, 6680984), (15, 7275, 9055254))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def review(asset_root, scope, approval, source_paths, archive_directory, sources_output):
    auditor_path = Path(auditor.__file__)
    if auditor.file_sha(auditor_path) != FROZEN_AUDITOR_SHA:
        raise ValueError("The independent tar auditor is no longer the frozen source")
    if len(source_paths) != len(EXPECTED_BATCHES):
        raise ValueError("Exactly the first two completed E18 export batches are required")
    context = auditor.FrozenContext(asset_root, scope, approval)
    batches, entries = [], []
    for path, expected in zip(source_paths, EXPECTED_BATCHES, strict=True):
        source = auditor.read_json(path)
        rows = source["new_archives"]
        counts = (len(rows), sum(row["tasks"] for row in rows), sum(row["bytes"] for row in rows))
        if counts != expected:
            raise ValueError("A completed export batch differs from its declared external cohort")
        if (
            source["scope_sha256"] != context.scope_sha
            or source["software_test_only"] is not False
            or any(
                source.get(field) != 0
                for field in ("new_model_calls", "scorer_calls", "gold_reads")
            )
        ):
            raise ValueError("An external batch changes scope, mode or execution-side declarations")
        batches.append(
            dict(
                path=str(path),
                sha256=auditor.file_sha(path),
                bytes=path.stat().st_size,
                archive_count=counts[0],
                native_tasks=counts[1],
                compressed_archive_bytes=counts[2],
            )
        )
        entries.extend(rows)
    combined = dict(
        schema="e18_independent_combined_external_inventory_v1",
        scope_sha256=context.scope_sha,
        source_batches=batches,
        new_archives=entries,
    )
    write_new_json(sources_output, combined)
    result = auditor.audit_all(
        context,
        sources_output,
        archive_directory,
        expected_archives=sum(row[0] for row in EXPECTED_BATCHES),
        expected_tasks=sum(row[1] for row in EXPECTED_BATCHES),
    )
    if result["compressed_bytes"] != sum(row[2] for row in EXPECTED_BATCHES):
        raise ValueError("The audited compressed-file total differs from the external batch total")
    if any(auditor.file_sha(row["path"]) != row["sha256"] for row in batches):
        raise ValueError("A source export inventory changed during its read-only audit")
    if auditor.file_sha(auditor_path) != FROZEN_AUDITOR_SHA:
        raise ValueError("The independent auditor changed during execution")
    result.update(
        source_batches=batches,
        combined_external_inventory_ref=dict(
            path=str(sources_output), sha256=auditor.file_sha(sources_output)
        ),
        independent_tar_auditor_sha256=FROZEN_AUDITOR_SHA,
        cross_batch_native_task_and_physical_attempt_uniqueness=True,
        source_inventories_unchanged=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, action="append", required=True)
    parser.add_argument("--archive-directory", type=Path, required=True)
    parser.add_argument("--sources-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.sources_output.exists():
        raise FileExistsError("Independent audit inputs and receipts are append-only")
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    result = review(
        args.asset_root,
        args.scope,
        args.approval,
        args.source_result,
        args.archive_directory,
        args.sources_output,
    )
    result.update(
        review_source_sha256=auditor.file_sha(Path(__file__)),
        audit_cpu_seconds=time.process_time() - started_cpu,
        audit_wall_seconds=time.perf_counter() - started_wall,
    )
    write_new_json(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
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
