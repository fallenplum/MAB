"""Rebuild the E24 unselected frame from input fields and prior exclusion records."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
E18_CODE = HERE.parent / "e18_mechanism_and_confirmation_preparation"
sys.path.insert(0, str(E18_CODE))

from audit_sources import (  # noqa: E402
    INPUT_FIELDS,
    NEAR_RULES,
    _rank,
    _screen_near_duplicates,
    canonical_hash,
    project,
    read_json,
    sha,
    write_json,
    write_jsonl,
)
from candidate_proposal import protected_screen, safe_source_rows  # noqa: E402


def corpus_identities(records):
    """Check uniqueness within a scope and return its canonical identity map."""
    identities = {row["question_id"]: row["canonical_question_sha256"] for row in records}
    if len(identities) != len(records) or len(set(identities.values())) != len(records):
        raise ValueError("Historical scope repeats an ID or canonical question")
    return identities


def inputs_only(path):
    with path.open("rb") as stream:
        for raw in stream:
            if raw.strip():
                yield project(raw, INPUT_FIELDS | {"canonical_question_sha256"})


def extract_external_originals(research):
    """Use the released DARS input field while syntax-skipping all score fields."""
    hashes, counts = set(), {}
    root = research / "e21_p3_external_mechanism/private/dars"
    expected = {"gpqa": {"train": 200, "test": 248}, "math-500": {"train": 200, "test": 300}}
    for task, splits in expected.items():
        counts[task] = {}
        for split, target in splits.items():
            path = root / task / f"{split}_scored_generations.jsonl"
            originals = {}
            with path.open("rb") as stream:
                for raw in stream:
                    row = project(raw, {"query_id", "original_question"})
                    identity = str(row["query_id"])
                    digest = canonical_hash(row["original_question"])
                    if originals.setdefault(identity, digest) != digest:
                        raise ValueError("P3 source has incompatible originals for one question")
            if len(originals) != target:
                raise ValueError("P3 projected question count differs from its frozen protocol")
            counts[task][split] = len(originals)
            hashes.update(originals.values())
    return hashes, counts


def audit(asset_root, private_output, report):
    start = time.perf_counter()
    if private_output.exists() or report.exists():
        raise FileExistsError("E24 audit outputs must be new")
    research = asset_root / "research"
    old = research / "e18_mechanism_and_confirmation_preparation"
    registered_path = old / "private/REGISTERED_EXCLUSION_METADATA.json"
    e18_path = research / "e18_confirmatory_execution/private/EXECUTION_SCOPE.json"
    p2_path = research / "e20_p2_execution/private/EXECUTION_SCOPE.json"
    e23_path = research / "e23_deployment_prior_and_references/run_v1/private/EXECUTION_SCOPE.json"
    registered = read_json(registered_path)["records"]
    e18 = read_json(e18_path)["question_keys"]
    p2 = read_json(p2_path)["question_keys"]
    e23 = read_json(e23_path)["question_keys"]
    for expected_size, scope in ((1708, registered), (3000, e18), (2400, p2), (840, e23)):
        if len(corpus_identities(scope)) != expected_size:
            raise ValueError("Historical scope count differs from its frozen archive")

    registered_ids, e18_ids, p2_ids, e23_ids = map(corpus_identities, (registered, e18, p2, e23))
    if (registered_ids.keys() & e18_ids.keys()) != {row["question_id"] for row in registered if row["split"] == "fresh_dev400"}:
        raise ValueError("E18 scope should share only the original fresh400")
    if p2_ids.keys() & (registered_ids.keys() | e18_ids.keys()):
        raise ValueError("P2 overlaps previous scopes")
    if any(p2_ids.get(key) != value for key, value in e23_ids.items()):
        raise ValueError("E23 supplement is not an exact subset of P2 identity scope")

    exclusions = {row["question_id"]: row for row in registered}
    for label, records in (("e18_executed", e18), ("p2_executed", p2)):
        for row in records:
            existing = exclusions.setdefault(row["question_id"], dict(row, split=label))
            if existing["canonical_question_sha256"] != row["canonical_question_sha256"]:
                raise ValueError("Shared historical question has conflicting canonical identity")
    excluded_hashes = set(corpus_identities(list(exclusions.values())).values())
    if len(exclusions) != 6708:
        raise ValueError("Unique registered/E18/P2 exclusion union differs")

    source_path = old / "private/SOURCE_MANIFEST.json"
    source_manifest = read_json(source_path)
    source_paths = {"data/derived/gsm8k_questions.jsonl"}
    source_paths.update(row["source_file"] for row in source_manifest["official_bbh"])
    declared_sources = {row["path"]: row for row in source_manifest["files"]}
    for relative in source_paths:
        source = asset_root / relative
        if sha(source) != declared_sources[relative]["sha256"]:
            raise ValueError("An official input source differs from the E18 audit snapshot")

    signatures_dir = old / "private/protected_input_fingerprints"
    access = read_json(signatures_dir / "AUTHORIZED_ACCESS_RECEIPT.json")
    index_path = signatures_dir / "OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json"
    signatures_path = signatures_dir / "PROTECTED_INPUT_SIGNATURES.json"
    if (access["status"] != "AUTHORIZED_ONCE_ONLY_INPUT_FINGERPRINT_AUDIT_COMPLETE"
            or sha(index_path) != access["bbh_position_index_sha256"]
            or sha(signatures_path) != access["signature_sha256"]):
        raise ValueError("Previously authorized protected input metadata differs")
    official_index = read_json(index_path)["records"]
    signatures = read_json(signatures_path)["records"]
    key = (signatures_dir / "FINGERPRINT_KEY.bin").read_bytes()
    if len(signatures) != 200 or len(key) != 32:
        raise ValueError("Protected one-way signature inventory is incomplete")

    rows, scans, official_duplicates = safe_source_rows(asset_root, source_manifest, list(exclusions.values()), official_index)
    observed = {row["question_id"] for row in rows}
    protected_ids = {row["question_id"] for row in registered if row["split"] == "sealed_test"}
    missing = set(exclusions) - observed - protected_ids

    for relative in ("e18_confirmatory_execution/private/ADMITTED3000_INPUTS_ONLY.jsonl",):
        for row in inputs_only(research / relative):
            if row["question_id"] in missing:
                _append_reference(rows, row, exclusions, missing)
    if missing:
        for relative in ("data/derived/available_records.jsonl", "data/derived/gsm8k_questions.jsonl"):
            with (asset_root / relative).open("rb") as stream:
                for raw in stream:
                    identity = project(raw, {"question_id"}).get("question_id")
                    if identity in missing:
                        _append_reference(rows, project(raw, INPUT_FIELDS), exclusions, missing)
    if missing:
        raise ValueError("Previously registered inputs are unresolved for duplicate screening")

    equivalent, near_removed, near_edges = _screen_near_duplicates(rows)
    format_removed = set()
    for group in equivalent:
        representative = None
        if not any(rows[index]["registered_excluded"] for index in group):
            representative = min(group, key=lambda idx: _rank("format-representative", rows[idx]["canonical_question_sha256"]))
        format_removed.update(index for index in group if index != representative)
    protected_removed, protected_edges, _ = protected_screen(rows, key, signatures)
    dropped = near_removed | format_removed | protected_removed
    official_candidates = [row for index, row in enumerate(rows) if not row["registered_excluded"] and index not in dropped]

    previous_index = research / "e20_independent_confirmation_preparation/private/inputs/E20_REMAINING_BBH_IDENTITY_INDEX.json"
    previous_bbh = corpus_identities(read_json(previous_index)["records"])
    previous_bbh = {qid: value for qid, value in previous_bbh.items() if qid not in p2_ids}
    projected_bbh = corpus_identities([row for row in official_candidates if row["task"] == "bbh"])
    if projected_bbh != previous_bbh:
        raise ValueError("Official BBH projection differs from the archived eligibility index minus P2")

    external_hashes, external_counts = extract_external_originals(research)
    external_excluded = [row for row in official_candidates if row["canonical_question_sha256"] in external_hashes]
    candidates = [row for row in official_candidates if row["canonical_question_sha256"] not in external_hashes]
    candidate_map = corpus_identities(candidates)
    if candidate_map.keys() & exclusions.keys() or set(candidate_map.values()) & excluded_hashes:
        raise ValueError("E24 candidate frame repeats a history identity")
    fields = INPUT_FIELDS | {"canonical_question_sha256", "source_file", "source_frame", "source_index", "source_revision"}
    output = [{name: row[name] for name in sorted(fields)} for row in candidates]
    output.sort(key=lambda row: (row["task"], row["subtask"] or "", row["canonical_question_sha256"]))
    write_jsonl(private_output, output)

    counts = Counter(row["task"] for row in output)
    subtask_counts = Counter(row["subtask"] for row in output if row["task"] == "bbh")
    for item in source_manifest["official_bbh"]:
        subtask_counts.setdefault(item["subtask"], 0)
    common_capacity = min(count for count in subtask_counts.values() if count)
    old_closure = read_json(old / "public/HISTORICAL_INPUT_REGISTRY_CLOSURE.json")
    audit_files = {name: sha(path) for name, path in {
        "registered_identity_registry": registered_path,
        "e18_identity_scope": e18_path,
        "p2_identity_scope": p2_path,
        "e23_identity_scope": e23_path,
        "previous_bbh_eligibility_index": previous_index,
        "official_source_manifest": source_path,
    }.items()}
    result = {
        "schema": "mab_e24_input_only_candidate_capacity_v1",
        "status": "SOURCE_AUDITED_UNSELECTED_UNADMITTED",
        "official_frame": {"gsm8k_train": 7473, "gsm8k_test": 1319,
                           "bbh_test_source_rows": len(official_index), "bbh_subtasks": len(subtask_counts)},
        "registered_cohorts": dict(Counter(row["split"] for row in registered)),
        "e18_scope_questions": len(e18), "e18_fresh400_inclusion": len(e18_ids.keys() & registered_ids.keys()),
        "p2_scope_questions": len(p2), "e23_subset_of_p2_questions": len(e23),
        "historical_exclusion_union": len(exclusions),
        "historic_registry_modules": old_closure["reviewed_research_modules"],
        "historic_registry_identity_containers": old_closure["explicitly_checked_identity_or_index_containers"],
        "historic_registry_outside_ids": old_closure["outside_registered_question_ids"],
        "historic_registry_outside_canonical": old_closure["outside_registered_canonical_hashes"],
        "external_p3_original_questions_by_split": external_counts,
        "external_p3_exact_exclusions": len(external_excluded),
        "sample_selection_uses_scores_or_predictions": False,
        "relative_dedup_rules": NEAR_RULES,
        "within_source_official_exact_duplicate_groups": len(official_duplicates),
        "format_equivalent_groups": len(equivalent), "high_similarity_edges": len(near_edges),
        "protected_signature_exclusion_edges": len(protected_edges),
        "protected_rows_skipped_by_source": {
            "gsm8k": scans[0]["protected_rows_skipped_before_input_decoding"],
            "bbh": sum(row["protected_rows_skipped_before_input_decoding"] for row in scans[1:]),
        },
        "registered_id_or_text_intersections_remaining": 0,
        "candidate_counts": dict(counts), "bbh_subtask_counts": dict(sorted(subtask_counts.items())),
        "candidate_counts_by_source_frame": dict(Counter(row["source_frame"] for row in output)),
        "maximum_task_balanced_questions": 2 * min(counts.values()),
        "maximum_task_balanced_common_subtask_equal_count_questions": 2 * common_capacity * sum(bool(n) for n in subtask_counts.values()),
        "verified_official_curated_files": len(source_paths),
        "previous_bbh_eligible_index_minus_p2_identically_reconstructed": True,
        "evidence_sha256": audit_files,
        "private_unselected_input_rows_sha256": sha(private_output),
        "private_unselected_input_schema": sorted(fields),
        "protected_input_values_decoded_this_audit": 0,
        "gold_or_reward_values_decoded_this_audit": 0,
        "new_llm_calls": 0, "new_encoder_questions": 0, "admitted_evaluation_questions": 0,
        "wall_seconds": time.perf_counter() - start,
    }
    write_json(report, result)
    return counts, subtask_counts


def _append_reference(rows, row, exclusions, missing):
    identity = row["question_id"]
    digest = canonical_hash(row["question"])
    if digest != exclusions[identity]["canonical_question_sha256"]:
        raise ValueError("Registered input reference digest differs")
    rows.append(dict(row, canonical_question_sha256=digest, registered_excluded=True))
    missing.remove(identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    tasks, subtasks = audit(args.asset_root, args.private_output, args.report)
    print(dict(tasks), dict(sorted(subtasks.items())), flush=True)
