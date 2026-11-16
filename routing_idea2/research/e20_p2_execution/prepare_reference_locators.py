"""Freeze source and row locators using only admitted input text and input projection."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from prepare_independent_pool import ASSETS, PRIVATE, PUBLIC, OLD, now
from audit_sources import canonical_hash, official_example_objects, project, read_json, sha, write_json


def main():
    started = time.perf_counter()
    input_path = PRIVATE / "ADMITTED2400_INPUTS_ONLY.jsonl"
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    manifest = read_json(OLD / "private/SOURCE_MANIFEST.json")
    source_hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    source_hashes.update({item["source_file"]: item["source_sha256"] for item in manifest["official_bbh"]})
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_file"]].append(row)
    sources, records = [], []
    for relative, group in sorted(groups.items()):
        path = ASSETS / relative
        expected_sha = source_hashes[relative]
        if sha(path) != expected_sha:
            raise ValueError("Admitted source changed after historical inventory")
        task, subtask = group[0]["task"], group[0]["subtask"]
        source_id = "gsm8k_official_train" if task == "gsm8k" else "bbh_" + subtask
        sources.append(dict(source_id=source_id, format="gsm8k_official_jsonl_v1" if task == "gsm8k" else "bbh_official_examples_v1",
                            path=str(path), sha256=expected_sha, task=task, source_split=group[0]["source_split"], subtask=subtask))
        requested = {int(row["source_index"]): row for row in group}
        if len(requested) != len(group):
            raise ValueError("Two admitted inputs share a source position")
        verified = set()
        if task == "gsm8k":
            with path.open("rb") as stream:
                for index, raw in enumerate(stream):
                    if index not in requested:
                        continue
                    question = project(raw, {"question"})["question"]
                    if canonical_hash(question) != requested[index]["canonical_question_sha256"]:
                        raise ValueError("GSM source pointer does not recover admitted input")
                    verified.add(index)
        else:
            for index, raw in enumerate(official_example_objects(path)):
                if index not in requested:
                    continue
                question = project(raw, {"input"})["input"]
                if canonical_hash(question) != requested[index]["canonical_question_sha256"]:
                    raise ValueError("BBH source pointer does not recover admitted input")
                verified.add(index)
        if verified != set(requested):
            raise ValueError("Reference source pointer coverage incomplete")
        for row in group:
            records.append(dict(question_id=row["question_id"], canonical_question_sha256=row["canonical_question_sha256"],
                                task=row["task"], subtask=row["subtask"], source_split=row["source_split"],
                                source_id=source_id, row_index=int(row["source_index"])))
    by_id = {row["question_id"]: row for row in records}
    records = [by_id[row["question_id"]] for row in rows]
    sealed = [{key: row[key] for key in ("question_id", "canonical_question_sha256")}
              for row in read_json(OLD / "private/REGISTERED_EXCLUSION_METADATA.json")["records"]
              if row["split"] == "sealed_test"]
    forbidden_ids = {row["question_id"] for row in sealed}
    forbidden_canonical = {row["canonical_question_sha256"] for row in sealed}
    if any(row["question_id"] in forbidden_ids or row["canonical_question_sha256"] in forbidden_canonical for row in records):
        raise ValueError("Sealed identity enters reference plan")
    plan = dict(schema="p2_reference_locator_plan_v1", status="INPUT_LOCATORS_ONLY_ANSWERS_NOT_READ", utc=now(),
                input_sha256=sha(input_path), sources=sources, records=records, sealed_exclusions=sealed,
                reference_version="safe_exact_v2_0_terminal_final_heading_fix",
                source_byte_hashing_performed=True, answer_values_decoded=0, sealed_answer_values_decoded=0,
                input_source_positions_verified=2400)
    target = PRIVATE / "P2_REFERENCE_LOCATORS_FROZEN.json"
    write_json(target, plan)
    summary = dict(status=plan["status"], sources=len(sources), question_locators=len(records),
                   task_counts=dict(Counter(row["task"] for row in records)), input_source_positions_verified=2400,
                   locator_plan_sha256=sha(target), input_sha256=sha(input_path), source_sha256=sha(__file__),
                   source_raw_byte_sha_checks=len(sources), answer_values_decoded=0, sealed_answer_values_decoded=0,
                   wall_seconds=time.perf_counter()-started)
    write_json(PUBLIC / "P2_REFERENCE_LOCATOR_QUALIFICATION.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
