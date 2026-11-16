"""Read original member Store rows for the approved E18 chosen-action union.

Historical native records retain their split, job ID, physical attempt, actual
usage and record SHA. Only an outer private transport wrapper is newly written.
No replacement generation record is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_verified_member_cache(adapter, cache_keys_path, expected_cache_sha256, *, expected_required_count=4717):
    """Return {(qid, seed, member): untouched native row}, plus audit metadata."""
    from cache_audit import FORBIDDEN_KEYS, ROW_FIELDS, digest, json_property_names
    started, cpu_started = time.perf_counter(), time.process_time()
    cache_keys_path = Path(cache_keys_path)
    if sha(cache_keys_path) != expected_cache_sha256:
        raise ValueError("Frozen cache-key source SHA differs")
    keys = read_json(cache_keys_path)["keys"]
    identities = {row["question_id"]: row["canonical_question_sha256"] for row in adapter.scope["question_keys"]}
    required = set(map(tuple, adapter.scope["required_member_keys"]))
    available = {}
    for row in keys:
        logical = row["question_id"], row["generation_seed"], row["member_index"]
        if logical not in required:
            continue
        if row["canonical_id"] != identities[row["question_id"]] or logical in available:
            raise ValueError("Cache canonical identity or unique logical key differs")
        available[logical] = row
    if len(available) != expected_required_count:
        raise ValueError("Required verified cache count differs from the frozen E18 subtraction")
    if set(available) & set(map(tuple, adapter.scope["member_execution_keys"])):
        raise ValueError("A cached member is incorrectly scheduled for new paid generation")
    files, records, references = {}, {}, []
    counters = {}
    for logical, key in sorted(available.items()):
        path = Path(key.get("local_source_path", key["source_path"]))
        expected_sha = key.get("local_source_sha256", key["source_sha256"])
        if path not in files:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected_sha:
                raise ValueError("A private cache source changed since its independent postcheck")
            if any(name.casefold() in FORBIDDEN_KEYS for name in json_property_names(raw)):
                raise ValueError("Reference/reward container rejected before value decoding")
            indexed = {}
            for number, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if set(row) != ROW_FIELDS:
                    raise ValueError("A historical member output has another native schema")
                checksum = row.get("record_sha256")
                if checksum != digest({name: value for name, value in row.items() if name != "record_sha256"}):
                    raise ValueError("Historical native member record hash differs")
                if checksum in indexed:
                    raise ValueError("Duplicate historical native record SHA")
                indexed[checksum] = (number, row)
            files[path] = dict(sha256=expected_sha, bytes=len(raw), records=indexed)
        if files[path]["sha256"] != expected_sha:
            raise ValueError("One source path has conflicting declared file hashes")
        if key["record_sha256"] not in files[path]["records"]:
            raise ValueError("Expected native record is absent from its verified source")
        number, original = files[path]["records"][key["record_sha256"]]
        if original["job_id"] != key["job_id"] or original["attempt_id"] != key["physical_attempt_id"]:
            raise ValueError("Original physical execution identity differs")
        result = original["result"]
        if hashlib.sha256(result["text"].encode()).hexdigest() != key["answer_sha256"]:
            raise ValueError("Historical prediction bytes differ")
        before = digest(original)
        accepted = adapter.checked_member_dependency(*logical, original)
        if accepted is not original or digest(original) != before:
            raise ValueError("Native cache import must preserve the exact historical record")
        records[logical] = original
        group = logical[1], logical[2]
        counters[group] = counters.get(group, 0) + 1
        references.append(dict(question_id=logical[0], generation_seed=logical[1], member_index=logical[2],
            canonical_id=key["canonical_id"], source_path=str(path), source_file_sha256=expected_sha,
            source_line=number, native_record_sha256=key["record_sha256"],
            original_job_id=original["job_id"], original_physical_attempt_id=original["attempt_id"],
            original_split=original["split"]))
    if len({row["attempt_id"] for row in records.values()}) != len(records):
        raise ValueError("Selected logical cache contains duplicate physical executions")
    report = dict(schema="e18_production_member_cache_bridge_audit_v1", status="PASS_UNMODIFIED_ORIGINAL_MEMBER_STORE_ROWS",
        source_cache_keys_sha256=expected_cache_sha256, required_cached_member_keys=len(records),
        source_files=[dict(path=str(path), sha256=data["sha256"], bytes=data["bytes"], rows=len(data["records"])) for path, data in files.items()],
        by_generation_member=[dict(generation_seed=seed, member_index=member, keys=count) for (seed, member), count in sorted(counters.items())],
        original_source_input_tokens=sum(row["result"]["prompt_tokens"] for row in records.values()),
        original_source_output_tokens=sum(row["result"]["completion_tokens"] for row in records.values()),
        original_native_split_counts={split:sum(row["split"] == split for row in records.values()) for split in sorted({row["split"] for row in records.values()})},
        new_actual_records_created=0, original_native_records_rewritten=0,
        original_actual_tokens_changed=0, member_generation_calls=0, aggregation_calls=0, gold_reads=0,
        bridge_operation="Pass untouched historical fresh_dev_test native row to checked_member_dependency; rebuild expected source job only for validation",
        source_references=references, cpu_s=time.process_time()-cpu_started, wall_s=time.perf_counter()-started)
    return records, report


def export_native_wrappers(records, report, path):
    """Private transport wrappers preserve every native record field and SHA."""
    references = {(row["question_id"], row["generation_seed"], row["member_index"]): row for row in report["source_references"]}
    with Path(path).open("x", encoding="utf-8") as stream:
        for key, native in sorted(records.items()):
            json.dump(dict(schema="e18_historical_member_native_wrapper_v1", source=references[key], native_record=native),
                stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--adapter-source-dir", type=Path, required=True)
    parser.add_argument("--inputs-only", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--cache-keys", type=Path, required=True)
    parser.add_argument("--cache-keys-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-native-jsonl", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.export_native_jsonl is not None and args.export_native_jsonl.exists()):
        raise FileExistsError("Production cache-bridge artifacts refuse overwrite")
    sys.path.insert(0, str(args.adapter_source_dir))
    from e18_execution_adapter import E18ExecutionAdapter, load_original_apis
    from cache_audit import read_unscored_rows
    records = read_unscored_rows(args.inputs_only, [])
    scope = read_json(args.scope)
    apis = load_original_apis(args.asset_root)
    pool = read_json(args.asset_root / "configs/pool_v1.json")["models"]
    specs = {row["index"]:apis.inference.ModelSpec(**read_json(args.asset_root / row["config"])) for row in pool}
    agg = apis.inference.ModelSpec(**read_json(args.asset_root / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"))
    adapter = E18ExecutionAdapter(apis, scope, records, specs, agg)
    native_rows, report = load_verified_member_cache(adapter, args.cache_keys, args.cache_keys_sha256)
    report.update(inputs_only_sha256=sha(args.inputs_only), production_scope_sha256=sha(args.scope),
        source_code_sha256=sha(__file__))
    if args.export_native_jsonl is not None:
        export_native_wrappers(native_rows, report, args.export_native_jsonl)
        report["native_wrapper_export"] = dict(path=str(args.export_native_jsonl), sha256=sha(args.export_native_jsonl),
            original_actual_records_rewritten=0, wrapper_rows=len(native_rows))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status", "required_cached_member_keys", "original_source_input_tokens", "original_source_output_tokens", "wall_s")}))


if __name__ == "__main__":
    main()
