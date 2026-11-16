"""Bind the frozen 371-task tail checkpoint to independently audited exports.

Only EXPORT_MANIFEST metadata is JSON-decoded. Receipt payloads may be hashed
and subjected to a byte-only cache-flag proof. Generated answers are never
decoded, exposed or used for a decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

import audit_tail_export_lineage as old
import audit_tail371_static_sources as static


STATIC_SOURCE_SHA = "90b5551718845b27632b341c24a131fa664fffbdac82758cb4801377ef63e60f"


def cache_flag_relation(raw, checkpoint_sha):
    """Return an exact-byte proof, permitting only one top-level flag change."""
    raw_sha = hashlib.sha256(raw).hexdigest()
    if raw_sha == checkpoint_sha:
        return "EXACT_CHECKPOINT_FILE_SHA"
    try:
        if old.cache_flag_predecessor_sha(raw) == checkpoint_sha:
            return "ARCHIVE_CACHE_TRUE_CHECKPOINT_FALSE_ONLY"
    except ValueError:
        pass
    for match in re.finditer(rb'"from_cache"\s*:\s*false', raw):
        start = match.end() - len(b"false")
        candidate = raw[:start] + b"true" + raw[match.end():]
        if hashlib.sha256(candidate).hexdigest() != checkpoint_sha:
            continue
        if old.cache_flag_predecessor_sha(candidate) == raw_sha:
            return "ARCHIVE_CACHE_FALSE_CHECKPOINT_TRUE_ONLY"
    raise ValueError("Checkpoint receipt differs beyond the sole top-level cache flag")


def receipt_relation(private_root, row, checkpoint_sha):
    if row["_receipt_sha256"] == checkpoint_sha:
        return "EXACT_CHECKPOINT_FILE_SHA"
    with tarfile.open(private_root / "native_exports" / row["_archive"], "r|gz") as archive:
        for member in archive:
            if member.name == row["_receipt_name"]:
                raw = archive.extractfile(member).read()
                static.require(hashlib.sha256(raw).hexdigest() == row["_receipt_sha256"],
                               "Native-audited receipt file SHA differs")
                return cache_flag_relation(raw, checkpoint_sha)
    raise ValueError("Checkpoint receipt is absent from its closed archive")


def audit(private_root, bundle_ref, native_audit_ref, scope_ref, gate_source_ref,
          *, require_complete=False):
    private_root = Path(private_root).resolve()
    static.require(old.file_sha(old.__file__) == static.LINEAGE_SHA
                   and old.file_sha(static.__file__) == STATIC_SOURCE_SHA,
                   "Frozen read-only helper source changed")
    scope = old.checked_file(scope_ref)
    native = old.checked_file(native_audit_ref)
    gate_raw = old.checked_file(gate_source_ref, decode=False)
    static.require(
        hashlib.sha256(gate_raw).hexdigest() == old.GATE_SHA
        and native["status"] == "PASS_INDEPENDENT_STREAMING_NATIVE_AUDIT"
        and native["independent_tar_auditor_sha256"] == old.NATIVE_AUDITOR_SHA
        and native["execution_scope_sha256"] == old.digest(scope) == static.SCOPE_SHA
        and native["execution_scope_file_sha256"] == scope_ref["sha256"] == static.SCOPE_FILE_SHA
        and all(native[name] == 0 for name in (
            "evaluation_results_read", "gold_or_reward_sources_read", "model_or_scorer_calls")),
        "Trusted native audit, scope or final CPU gate differs",
    )
    static.require(bundle_ref["sha256"] == static.BUNDLE_SHA, "Explicit371 bundle differs")
    bundle = old.MetadataBundle(bundle_ref)
    provenance = static.inspect(bundle)
    plan = bundle.read(provenance["fine_plan_ref"])
    parent = bundle.read(provenance["original_parent_ref"])
    lease = bundle.read(provenance["nested_lease_ref"])
    checkpoint = bundle.read(provenance["checkpoint_source_audit_ref"])
    checkpoint_rows = {row["task_key_sha256"]: row for row in checkpoint["tasks"]}
    parent_keys = old.unique_keys(bundle.read(parent["task_keys_ref"]))
    remaining = old.unique_keys(bundle.read(lease["remaining_task_keys_ref"]))
    static.require(parent_keys <= set(map(tuple, scope["member_execution_keys"])),
                   "Parent escapes the frozen scientific execution union")
    producer_for_key, children = {}, {}
    registered_refs = [provenance["fine_plan_ref"]]
    for source in plan["manifest_refs"]:
        child = bundle.read(source)
        children[source["sha256"]] = child
        registered_refs.extend([source, child["task_keys_ref"]])
        for key in old.unique_keys(bundle.read(child["task_keys_ref"])):
            producer_for_key[key] = source["sha256"]
    audited_archives = {row["sha256"]: row for row in native["archives"]}
    seen_archives, seen_tasks, seen_attempts = set(), set(), set()
    parent_rows, archive_reports, catalogues = {}, [], []
    for batch in native["source_batches"]:
        path = Path(batch["path"]).resolve()
        static.require(path.parent == private_root
                       and re.fullmatch(r"NATIVE_EXPORT_RESULT_v\d+\.json", path.name),
                       "Only canonical closed-export catalogues are accepted")
        catalogue_ref = dict(path=str(path), sha256=batch["sha256"])
        catalogue = old.checked_file(catalogue_ref)
        catalogues.append(catalogue_ref)
        for entry in catalogue["new_archives"]:
            checksum = entry["sha256"]
            static.require(checksum not in seen_archives and checksum in audited_archives,
                           "Duplicate or unaudited closed archive")
            name = PurePosixPath(entry["path"]).name
            static.require(re.fullmatch(r"native-[0-9a-f]{32}\.tar\.gz", name),
                           "Noncanonical native archive filename")
            metadata = old.export_metadata(private_root / "native_exports" / name,
                                           entry, audited_archives[checksum])
            static.require(metadata["scope_sha256"] == static.SCOPE_SHA
                           and metadata["software_test_only"] is False,
                           "Export scope differs")
            registered = {row["sha256"] for row in metadata["metadata_sources"]}
            workers = {row["manifest_sha256"]: row for row in metadata["worker_sources"]}
            files = {row["path"]: row for row in metadata["files"]}
            count = 0
            for row in metadata["tasks"]:
                task_id, attempt = row["task_id"], row["physical_attempt_id"]
                static.require(task_id not in seen_tasks and attempt not in seen_attempts,
                               "Repeated native task or physical attempt")
                seen_tasks.add(task_id)
                seen_attempts.add(attempt)
                key = tuple(row["task_key"])
                if row["stage"] != "member_generation" or key not in parent_keys:
                    continue
                static.require(key not in parent_rows
                               and static.PARENT_SHA in row["manifest_sha256s"],
                               "Original parent producer missing or repeated")
                key_hash = old.digest(list(key))
                row["_archive"] = name
                row["_archive_sha256"] = checksum
                row["_receipt_name"] = static.SCOPE_SHA + "/member_generation/" + key_hash + "/LAST_TASK_RECEIPT.json"
                row["_receipt_sha256"] = files[row["_receipt_name"]]["sha256"]
                parent_rows[key] = row
                if key not in remaining:
                    continue
                count += 1
                static.require(producer_for_key[key] in row["manifest_sha256s"]
                               and all(ref["sha256"] in registered for ref in registered_refs),
                               "Actual fine task lacks complete fine-source registration")
                for child_sha, child in children.items():
                    static.require(child_sha in workers and all(
                        workers[child_sha][field] == child[field]
                        for field in ("adapter_ref", "backend", "task_keys_ref")),
                        "Native-audited fine producer request fields differ")
            seen_archives.add(checksum)
            archive_reports.append(dict(archive_sha256=checksum,
                                        native_tasks=len(metadata["tasks"]), fine_native_tasks=count))
    static.require(seen_archives == set(audited_archives)
                   and len(seen_tasks) == native["native_tasks_checked"],
                   "Catalogue coverage differs from trusted completed native audit")
    prefix_rows, fine_rows = [], []
    fingerprints = set()
    for key, row in sorted(parent_rows.items()):
        key_hash = old.digest(list(key))
        item = dict(task_key_sha256=key_hash, native_record_sha256=row["native_record_sha256"],
                    physical_attempt_id=row["physical_attempt_id"],
                    source_archive_sha256=row["_archive_sha256"])
        fingerprints.add(row["model_fingerprint"])
        if key in remaining:
            item["fine_manifest_sha256"] = producer_for_key[key]
            fine_rows.append(item)
        else:
            saved = checkpoint_rows[key_hash]
            static.require(saved["status"] == "SUCCESS"
                           and saved["native_record_sha256"] == row["native_record_sha256"],
                           "Successful checkpoint native result changed")
            item["receipt_snapshot_comparison"] = receipt_relation(
                private_root, row, saved["native_receipt_file_sha256"])
            prefix_rows.append(item)
    static.require(len(fingerprints) <= 1, "Prefix/fine physical ModelSpec fingerprints differ")
    complete = len(prefix_rows) == 348 and len(fine_rows) == 23
    static.require(not require_complete or complete,
                   "Complete371 requires all348 checkpoint and23 fine native tasks")
    for source in [bundle_ref, native_audit_ref, scope_ref, gate_source_ref, *catalogues]:
        static.require(old.file_sha(source["path"]) == source["sha256"],
                       "An explicit input changed during read-only audit")
    return dict(
        schema="e18_tail371_export_lineage_audit_v1",
        status="PASS_COMPLETE371_SOURCE_AND_NATIVE_EXPORT_BINDING" if complete else "PASS_PARTIAL371_SOURCE_AND_NATIVE_EXPORT_BINDING",
        static_source_bundle_ref=bundle_ref, source_chain=provenance,
        completed_native_audit_ref=native_audit_ref, scope_ref=scope_ref,
        frozen_cpu_gate_ref=gate_source_ref, expected_parent_tasks=371,
        archived_checkpoint_successes=len(prefix_rows), expected_checkpoint_successes=348,
        archived_fine_native_tasks=len(fine_rows), expected_fine_native_tasks=23,
        missing_parent_tasks=371-len(parent_rows),
        checkpoint_receipt_cache_only_transitions=sum(
            row["receipt_snapshot_comparison"] != "EXACT_CHECKPOINT_FILE_SHA" for row in prefix_rows),
        checkpoint_task_metadata=prefix_rows, fine_task_metadata=fine_rows,
        archives_checked=len(seen_archives), all_native_tasks_cross_checked=len(seen_tasks),
        unique_physical_attempts_cross_checked=len(seen_attempts), archive_reports=archive_reports,
        review_source_sha256=old.file_sha(__file__),
        frozen_metadata_helper_sha256=static.LINEAGE_SHA,
        frozen_static_helper_sha256=STATIC_SOURCE_SHA,
        model_calls=0, scorer_calls=0, gold_reads=0, reward_reads=0,
        question_text_reads=0, native_answer_json_decodes=0, signals_sent=0,
        files_extracted=0, additional_scientific_keys=0,
        final_global_cpu_gate="UNCHANGED_9b_MUST_SEPARATELY_ADMIT_ALL_NATIVE_OUTPUTS",
        interpretation="Source-only proof over independently native-audited closed outputs, with no quality conclusion.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", required=True)
    for name in ("bundle", "native-audit", "scope", "gate-source"):
        parser.add_argument("--" + name + "-ref", nargs=2, required=True)
    parser.add_argument("--require-complete-fine", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    static.require(not Path(args.output).exists(), "Receipts are append-only")
    started = time.perf_counter()
    ref = lambda pair: dict(path=pair[0], sha256=pair[1])
    result = audit(args.private_root, ref(args.bundle_ref), ref(args.native_audit_ref),
                   ref(args.scope_ref), ref(args.gate_source_ref),
                   require_complete=args.require_complete_fine)
    result["wall_time_s"] = time.perf_counter() - started
    with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({name: result[name] for name in (
        "status", "archives_checked", "all_native_tasks_cross_checked",
        "archived_checkpoint_successes", "archived_fine_native_tasks", "missing_parent_tasks")}))
    print(json.dumps(dict(output=args.output, sha256=old.file_sha(args.output))))


if __name__ == "__main__":
    main()
