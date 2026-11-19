"""Bind E24 reference row positions using only frozen admitted question text."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e18_mechanism_and_confirmation_preparation"))
from audit_sources import canonical_hash, official_example_objects, project, read_json, sha, write_json

REFERENCE_VERSION = "safe_exact_v2_0_terminal_final_heading_fix"
LOCATOR_SCHEMA = "e24_reference_locator_plan_v1"


def checked_path(assets, relative):
    path = (assets / relative).resolve(strict=True)
    if not path.is_relative_to(assets) or any(part.casefold() in {"sealed", "sealed_test", "sealed200"} for part in path.parts):
        raise PermissionError("A reference source must resolve inside the authorized public benchmark source frame")
    return path


def source_rows(path, task):
    if task == "bbh":
        yield from official_example_objects(path)
    elif task == "gsm8k":
        with path.open("rb") as source:
            yield from source
    else:
        raise ValueError("Unexpected source task")


def prepare(assets, inputs, admission, protocol, output):
    assets = assets.resolve(strict=True)
    identity_root = assets / "research/e18_mechanism_and_confirmation_preparation/private"
    manifest_path = identity_root / "SOURCE_MANIFEST.json"
    registry_path = identity_root / "REGISTERED_EXCLUSION_METADATA.json"
    source_audit = HERE / "SOURCE_AUDIT.json"
    source_binding = read_json(source_audit)["evidence_sha256"]
    admitted = read_json(admission)
    spec = read_json(protocol)
    if (sha(inputs) != admitted.get("input_sha256") or sha(protocol) != admitted.get("protocol_sha256")
            or sha(source_audit) != admitted.get("source_audit_sha256")
            or sha(manifest_path) != source_binding["official_source_manifest"]
            or sha(registry_path) != source_binding["registered_identity_registry"]):
        raise ValueError("The input selection, source inventory or sealed exclusion records changed")
    manifest = read_json(manifest_path)
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    bbh_paths = {row["source_file"] for row in manifest["official_bbh"]}
    gsm_path = "data/raw/gsm8k_official/train.jsonl"
    sealed = [{field: row[field] for field in ("question_id", "canonical_question_sha256")}
              for row in read_json(registry_path)["records"] if row["split"] == "sealed_test"]
    forbidden_ids = {row["question_id"] for row in sealed}
    forbidden_hashes = {row["canonical_question_sha256"] for row in sealed}
    if len(sealed) != len(forbidden_ids) or len(sealed) != len(forbidden_hashes) or not sealed:
        raise ValueError("The prior sealed exclusion registry is incomplete")

    fields = {"question", "question_id", "canonical_question_sha256", "task", "subtask", "source_split",
              "source_file", "source_index", "source_revision", "source_frame"}
    with inputs.open("rb") as stream:
        rows = [project(raw, fields) for raw in stream]
    question_ids = {row["question_id"] for row in rows}
    digests = {row["canonical_question_sha256"] for row in rows}
    strata = Counter("gsm8k" if row["task"] == "gsm8k" else "bbh/" + row["subtask"] for row in rows)
    if (len(rows) != len(question_ids) or len(rows) != len(digests) or question_ids & forbidden_ids
            or digests & forbidden_hashes or strata != spec["cohort"]["stratum_counts"]):
        raise PermissionError("The reference metadata must cover the exact unique, unsealed E24 cohort")

    groups = defaultdict(list)
    for row in rows:
        groups[row["source_file"]].append(row)
    sources, records = [], {}
    for relative, group in sorted(groups.items()):
        task, subtask, split = (group[0][field] for field in ("task", "subtask", "source_split"))
        if (task, split) not in {("gsm8k", "train"), ("bbh", "test")} or not all(
            (row["task"], row["subtask"], row["source_split"]) == (task, subtask, split) for row in group
        ):
            raise ValueError("A source appears with mixed task/split identities")
        if (task == "gsm8k" and (relative != gsm_path or subtask)) or (
            task == "bbh" and (relative not in bbh_paths or Path(relative).stem != subtask)
        ):
            raise ValueError("Source path is outside its official task frame")
        path = checked_path(assets, relative)
        digest = sha(path)
        if digest != expected[relative]:
            raise ValueError("Official source bytes changed")
        source_id = "gsm8k_official_train" if task == "gsm8k" else "bbh_" + subtask
        codec = "gsm8k_official_jsonl_v1" if task == "gsm8k" else "bbh_official_examples_v1"
        sources.append(dict(source_id=source_id, format=codec, path=str(path), sha256=digest,
                            task=task, subtask=subtask or None, source_split=split))
        requested = {row["source_index"]: row for row in group}
        if len(requested) != len(group) or any(type(index) is not int or index < 0 for index in requested):
            raise ValueError("Official source positions must be nonnegative unique integers")
        verified, field = set(), "question" if task == "gsm8k" else "input"
        for index, raw in enumerate(source_rows(path, task)):
            if index not in requested:
                continue
            row = requested[index]
            text = project(raw, {field})[field]
            if canonical_hash(text) != row["canonical_question_sha256"] or text != row["question"]:
                raise ValueError("Official source position does not recover the admitted text")
            qid = row["question_id"]
            records[qid] = dict(question_id=qid, canonical_question_sha256=row["canonical_question_sha256"],
                                task=task, subtask=subtask or None, source_split=split,
                                source_id=source_id, row_index=index)
            verified.add(index)
        if verified != set(requested):
            raise ValueError("Official source position coverage differs from admitted E24 input rows")

    plan = dict(schema=LOCATOR_SCHEMA, status="INPUT_LOCATORS_ONLY_ANSWERS_NOT_READ",
                protocol_sha256=sha(protocol), input_sha256=sha(inputs), admission_sha256=sha(admission),
                source_audit_sha256=sha(source_audit), reference_version=REFERENCE_VERSION,
                sources=sources, records=[records[row["question_id"]] for row in rows], sealed_exclusions=sealed,
                input_source_positions_verified=len(rows), answer_values_decoded=0, sealed_answer_values_decoded=0)
    write_json(output, plan)
    return dict(schema="e24_reference_locator_qualification_v1", status=plan["status"],
                locator_plan_sha256=sha(output), input_sha256=plan["input_sha256"],
                protocol_sha256=plan["protocol_sha256"], scorer_version=REFERENCE_VERSION,
                reference_sources=len(sources), positions_verified=len(rows), sealed_exclusions=len(sealed),
                answer_values_decoded=0, sealed_answer_values_decoded=0, new_llm_calls=0,
                source_sha256=sha(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "admission", "protocol", "output", "qualification"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.qualification, prepare(args.assets, args.inputs, args.admission, args.protocol, args.output))


if __name__ == "__main__":
    main()
