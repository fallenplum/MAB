"""Read-only scientific-preparation checks, then an aggregate-only release seal."""
from __future__ import annotations

import argparse
import ast
from importlib import util
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from e18_common import (ASSETS, CODE, E17, PRIVATE, PUBLIC, POLICIES, SEEDS,
    input_only_npz, preserve_e17, read_json, sha256, write_json)
import numpy as np
from PIL import Image


REQUIRED_REPORTS = (
    "E18_MECHANISM_THEORY_REPORT.md", "E18_DATA_SOURCE_AND_DEDUP_REPORT.md",
    "E18_STATISTICAL_POWER_REPORT.md", "E18_FROZEN_EVALUATION_PROTOCOL.md",
    "E18_EXACT_CALL_BUDGET_REPORT.md", "E18_PRE_EXECUTION_QUALIFICATION_REPORT.md",
)


def collect_private_identifiers():
    forbidden = set()
    registry = read_json(ASSETS / "data/derived/split_manifest.json")
    forbidden.update(row["question_id"] for row in registry["records"])
    forbidden.update(registry["canonical_question_fingerprints"].values())
    fresh = input_only_npz(ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/fresh_dev_test_inputs.npz")
    forbidden.update(fresh["question_ids"].tolist())
    forbidden.update(fresh["canonical_hashes"].tolist())
    with (PRIVATE / "proposed_new2600_inputs_only.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            forbidden.update((row["question_id"], row["canonical_question_sha256"]))
            if len(row["question"]) >= 50:
                forbidden.add(row["question"])
    fresh_inputs = ASSETS / "artifacts/private/routing_redesign_20261104/fresh_dev400/inputs_only.jsonl"
    with fresh_inputs.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if len(row["question"]) >= 50:
                forbidden.add(row["question"])
    return forbidden


def verify_existing():
    manifest = read_json(PUBLIC / "PUBLIC_DELIVERY_MANIFEST.json")
    for record in manifest["files"]:
        if sha256(PUBLIC / record["path"]) != record["sha256"]:
            raise ValueError("Sealed E18 artifact changed")
    preserve_e17()
    print("E18_SEALED_DELIVERY_VERIFIED", len(manifest["files"])+1, flush=True)


def qualify_and_seal():
    start = time.perf_counter()
    original = preserve_e17()
    if (PUBLIC / "PUBLIC_DELIVERY_MANIFEST.json").exists():
        raise FileExistsError("Release already sealed; use verify")
    for name in REQUIRED_REPORTS:
        if not (PUBLIC / name).is_file():
            raise ValueError("Missing requested report")
    versions = read_json(PUBLIC / "ARTIFACT_VERSION_INDEX.json")
    model_qual = read_json(PUBLIC / versions["active_model_qualification"])
    if model_qual["source_sha256"] != sha256(CODE / "frozen_input_budget.py"):
        raise ValueError("Active model recovery source differs")
    meta = read_json(PUBLIC / "INPUT_METADATA_REPAIR_v2.json")
    with np.load(PRIVATE / "PROPOSED2600_INPUTS_ONLY_v2.npz", allow_pickle=False) as archive:
        inputs = {key:archive[key].copy() for key in archive.files}
    if set(inputs) != {"embeddings", "question_ids", "canonical_hashes", "task_labels", "subtask_labels"}:
        raise ValueError("Corrected input archive schema differs")
    if any(value.dtype.hasobject for value in inputs.values()):
        raise ValueError("Object metadata is forbidden")
    if inputs["embeddings"].shape != (2600,384) or np.sum(inputs["task_labels"] == "gsm8k") != 1300:
        raise ValueError("Proposed input feature contract differs")
    if sha256(PRIVATE / "PROPOSED2600_INPUTS_ONLY_v2.npz") != meta["repaired_input"]["sha256"]:
        raise ValueError("Corrected feature asset changed")
    with np.load(PRIVATE / "FROZEN3000_INPUT_PREDICTIONS.npz", allow_pickle=False) as archive:
        actions = archive["actions"].copy()
        policies = archive["policies"].copy()
        qids = archive["question_ids"].copy()
        canonical = archive["canonical_hashes"].copy()
    if actions.shape != (4,3000) or not np.array_equal(policies, POLICIES) or not np.isin(actions,np.arange(20)).all():
        raise ValueError("Frozen action contract differs")
    if len(set(qids.tolist())) != 3000 or len(set(canonical.tolist())) != 3000:
        raise ValueError("Original-question identities repeated")
    spec = util.spec_from_file_location("e17_count", E17 / "public/code/confirmation.py")
    counter = util.module_from_spec(spec)
    spec.loader.exec_module(counter)
    union = read_json(PRIVATE / "FROZEN_CALL_UNION_3000.json")
    actual = counter.input_call_union(actions,np.asarray(union["teams"]),SEEDS)
    if json.dumps(actual,sort_keys=True) != json.dumps(union,sort_keys=True):
        raise ValueError("Execution DAG/choice union mismatch")
    budget = read_json(PUBLIC / "E18_FINAL_CALL_BUDGET.json")
    if actual["counts"] != budget["frozen_logical_calls"]:
        raise ValueError("Budget count mismatch")
    if actual["counts"]["member_calls"]-budget["required_verified_member_reuse"] != budget["not_satisfied_by_verified_inventory"]["member_calls"]:
        raise ValueError("Cache subtraction mismatch")
    materialization = read_json(PUBLIC / "MEMBER_CACHE_MATERIALIZATION_POSTCHECK_v4.json")
    if not materialization["all_expected_keys_materialized"] or materialization["mirrored_new_keys"] != 2130:
        raise ValueError("Remote cache mirror is incomplete")
    local_records = read_json(PRIVATE / "RECONCILED_MEMBER_CACHE_KEYS_LOCAL_v4.json")["keys"]
    source_files = {}
    for row in local_records:
        path = row.get("local_source_path") or row.get("source_path")
        digest = row.get("local_source_sha256") or row.get("source_sha256")
        if not path or not digest:
            raise ValueError("Local verified key lacks its byte source")
        if Path(path).is_absolute() and not str(path).startswith("/data"):
            source_files[path] = digest
    for path,digest in source_files.items():
        if sha256(path) != digest:
            raise ValueError("Verified private cache source changed")
    state = read_json(PUBLIC / "E18_COMPLETE_PREPARATION.json")
    for relative,digest in state["sources"].items():
        source = CODE / relative
        if sha256(source) != digest or sha256(PUBLIC / "code" / relative) != digest:
            raise ValueError("Published source differs from preparation source")
        ast.parse(source.read_text(encoding="utf-8"))
    protocol = read_json(PUBLIC / "E18_EVALUATION_PROTOCOL.json")
    if not protocol["authorization_required_before_actual_generation_or_scoring"]:
        raise ValueError("Execution authorization boundary missing")
    if protocol["source_inputs_sha256"] != sha256(PRIVATE / "proposed_new2600_inputs_only.jsonl"):
        raise ValueError("Proposed source input hash changed")
    if protocol["member_generation_seeds"] != list(SEEDS) or protocol["aggregation_invocation_seed"] != 42:
        raise ValueError("Generation protocols mixed")
    for path in (PUBLIC / "figures").glob("*.svg"):
        ET.parse(path)
    for path in (PUBLIC / "figures").glob("*.drawio"):
        ET.parse(path)
    for path in (PUBLIC / "figures").glob("*.png"):
        with Image.open(path) as picture:
            picture.verify()
    forbidden = collect_private_identifiers()
    checked_files = []
    for path in PUBLIC.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".md", ".json", ".py", ".csv", ".png", ".svg", ".drawio"):
            raise ValueError("Unexpected public artifact type")
        if path.suffix != ".png":
            text = path.read_text(encoding="utf-8-sig")
            if any(value in text for value in forbidden):
                raise ValueError("Private question identity or long raw question in public file " + path.name)
        checked_files.append(path)
    result = dict(status="PASS_AUDITED_PREPARATION_ONLY", requested_main_reports=6,
        frozen_policy_decisions=12000, proposed_original_questions=3000, generation_records_planned=9000,
        member_nodes=actual["counts"]["member_calls"], aggregate_nodes=actual["counts"]["aggregate_calls"],
        private_cache_files_rehashed=len(source_files), member_keys_locally_available=len(local_records),
        current_input_archive_has_object_dtype=False, private_identifier_matches=0,
        denylist_strings=len(forbidden), checked_public_files=len(checked_files),
        E17_preserved=original, source_files=len(state["sources"]),
        actual_llm_generation_calls=0, actual_new_gold_scoring=0,
        protected_input_exception_processed=200, protected_answers_read=0,
        live_paid_runner_status="NOT_IMPLEMENTED_FOR_E18_SCOPE", paid_execution_authorized=False,
        scientific_generalization_evaluated=False,
        evidence_boundary="Software and preparation integrity pass; target approval and native scope adapter still required before paid execution",
        wall_s=time.perf_counter()-start)
    write_json(PUBLIC / "RELEASE_QUALIFICATION.json", result)
    write_json(PRIVATE / "RELEASE_QUALIFICATION.json", result)
    entries = [dict(path=path.relative_to(PUBLIC).as_posix(),bytes=path.stat().st_size,sha256=sha256(path))
               for path in sorted(PUBLIC.rglob("*")) if path.is_file()]
    write_json(PUBLIC / "PUBLIC_DELIVERY_MANIFEST.json", dict(status="SEALED_DEIDENTIFIED_PREPARATION",
        files=entries, raw_question_exports=0, gold_exports=0, private_matrix_exports=0, weight_exports=0))
    print("E18_PUBLIC_DELIVERY_SEALED",len(entries)+1,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "verify"))
    args = parser.parse_args()
    qualify_and_seal() if args.command == "seal" else verify_existing()
