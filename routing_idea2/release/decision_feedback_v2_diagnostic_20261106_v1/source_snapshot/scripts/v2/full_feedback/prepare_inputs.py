"""Prepare outcome-free full-feedback inputs with the frozen CPU MiniLM encoder."""
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[3]
PRIVATE = ROOT / "artifacts/private/routing_redesign_20261104"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v2"))

import numpy as np
from budget_router.encoding import FrozenQuestionEncoder
from budget_router_v2.full_feedback import read_inputs, read_json, reference, require, sha_file, write_json
from select_fresh_development import HEADER, input_projection

ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FRESH_SHA256 = "d2bac8bbbb06e9d6dff0aa51197eb52d92c55ab76679bcc0253cd4219bdc4a02"
SPLITS = {"selector_train": "online_stream", "selector_validation": "online_calibration"}
FRESH_DIRECTORY = PRIVATE / "fresh_dev400"


def canonical_hash(question):
    canonical = " ".join(unicodedata.normalize("NFKC", question).split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def identity_rows(data):
    return [{"question_id": str(qid), "canonical_question_sha256": str(canonical),
             "task": str(task), "subtask": str(subtask)}
            for qid, canonical, task, subtask in zip(data["question_ids"], data["canonical_hashes"],
                                                    data["task_labels"], data["subtask_labels"])]


def identity_hash(data):
    value = json.dumps(identity_rows(data), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def metadata_arrays(records, fingerprints):
    return {"question_ids": np.asarray([row["question_id"] for row in records]),
            "canonical_hashes": np.asarray([fingerprints[row["question_id"]] for row in records]),
            "task_labels": np.asarray([row["task"] for row in records]),
            "subtask_labels": np.asarray([row.get("subtask") or "" for row in records])}


def verify_original_questions(manifest, selected):
    """Decode input fields for the 800 permitted IDs, skip all other payloads."""
    allowed = {row["question_id"]: row for row in selected}
    found, scans = set(), []
    for source in manifest["source_files"]:
        if source["path"] not in ("data/derived/available_records.jsonl", "data/derived/gsm8k_questions.jsonl"):
            continue
        path = ROOT / source["path"]
        require(sha_file(ROOT / source["receipt_path"]) == source["receipt_sha256"], "Original question receipt changed")
        checksum = hashlib.sha256()
        count = projected = 0
        with path.open("rb") as stream:
            for raw in stream:
                checksum.update(raw)
                if not raw.strip():
                    continue
                count += 1
                match = HEADER.match(raw.removeprefix(b"\xef\xbb\xbf"))
                require(match is not None, "Question-ID header missing before projection")
                qid = match.group(1).decode("ascii")
                if qid not in allowed:
                    continue
                require(qid not in found, "Duplicate original input")
                row = input_projection(raw)
                expected = allowed[qid]
                require(row["question_id"] == qid and all(row[key] == expected[key]
                        for key in ("task", "subtask", "source_split")), "Original input metadata changed")
                require(canonical_hash(row["question"]) == manifest["canonical_question_fingerprints"][qid],
                        "Original question canonical identity changed")
                found.add(qid)
                projected += 1
        require(checksum.hexdigest() == source["sha256"] and count == source["records"], "Original source changed")
        scans.append({"source": reference(path), "physical_rows_streamed": count,
                      "permitted_question_payloads_projected": projected,
                      "reference_values_json_decoded": 0, "sealed_question_payloads_json_decoded": 0})
    require(found == set(allowed), "Original permitted input coverage differs")
    return scans


def cache_inputs(manifest, audit, cache_path):
    require(audit["status"] == "REAL_FROZEN_FEATURES_READY_NOT_MODEL_COMBINATION_RESULTS" and
            audit["encoder_id"] == ENCODER_ID and audit["encoder_revision"] == ENCODER_REVISION and
            audit["pooling"] == "attention_mask_mean_then_L2" and audit["encoder_max_tokens"] == 256,
            "Original embedding cache encoder policy differs")
    require(audit["split_manifest_sha256"] == sha_file(ROOT / "data/derived/split_manifest.json") and
            audit["feature_artifact"]["path"] == cache_path.relative_to(ROOT).as_posix() and
            audit["feature_artifact"]["sha256"] == sha_file(cache_path), "Original embedding cache binding changed")
    selected = [row for row in manifest["records"] if row["split"] in SPLITS.values()]
    require(len(selected) == 800, "Original 800 input cohort required")
    scans = verify_original_questions(manifest, selected)
    with np.load(cache_path, allow_pickle=False) as archive:
        cached = {key: archive[key] for key in ("question_ids", "task_labels", "splits", "embeddings")}
    ids = cached["question_ids"].tolist()
    require(len(ids) == len(set(ids)) == 800 and set(ids) == {row["question_id"] for row in selected},
            "Original embedding cache question coverage differs")
    require(cached["embeddings"].shape == (800, 384) and cached["embeddings"].dtype == np.float32 and
            np.isfinite(cached["embeddings"]).all(), "Original raw embeddings are invalid")
    positions = {qid: index for index, qid in enumerate(ids)}
    for row in selected:
        index = positions[row["question_id"]]
        require(cached["task_labels"][index] == row["task"] and cached["splits"][index] == row["split"],
                "Original cache row identity differs")
    datasets = {}
    for split, original in SPLITS.items():
        records = [row for row in selected if row["split"] == original]
        data = metadata_arrays(records, manifest["canonical_question_fingerprints"])
        data["embeddings"] = np.stack([cached["embeddings"][positions[row["question_id"]]] for row in records])
        datasets[split] = data
    return datasets, scans


def fresh_inputs(manifest):
    selection_path = FRESH_DIRECTORY / "SELECTION_MANIFEST_v1.json"
    require(sha_file(selection_path) == FRESH_SHA256, "Frozen fresh selection changed")
    selection = read_json(selection_path)
    require(selection["status"] == "SELECTED_INPUTS_ONLY_NOT_EVALUATED" and
            selection["base_manifest_sha256"] == sha_file(ROOT / "data/derived/split_manifest.json") and
            selection["reference_values_decoded"] == selection["sealed_question_payloads_decoded"] == 0,
            "Fresh selection permission boundary differs")
    path = FRESH_DIRECTORY / "inputs_only.jsonl"
    require(sha_file(path) == selection["inputs_sha256"], "Frozen fresh input bytes changed")
    with path.open(encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    expected = selection["records"]
    require(len(records) == len(expected) == 400, "Fresh input count differs")
    for row, selected in zip(records, expected):
        require(set(row) == {"question_id", "task", "subtask", "question", "source_split", "canonical_question_sha256"},
                "Fresh input-only schema differs")
        require({key: value for key, value in row.items() if key != "question"} == selected and
                canonical_hash(row["question"]) == row["canonical_question_sha256"], "Fresh canonical identity changed")
    require(not {row["question_id"] for row in records}.intersection(row["question_id"] for row in manifest["records"]) and
            not {row["canonical_question_sha256"] for row in records}.intersection(manifest["canonical_question_fingerprints"].values()),
            "Fresh cohort overlaps an original question")
    fingerprints = {row["question_id"]: row["canonical_question_sha256"] for row in records}
    return metadata_arrays(records, fingerprints), [row["question"] for row in records], selection_path, path


def validate_assets(output, bundle):
    checked, all_ids, all_hashes = {}, set(), set()
    manifest = read_json(ROOT / "data/derived/split_manifest.json")
    for split in ("selector_train", "selector_validation", "fresh_dev_test"):
        data, path = read_inputs(bundle, output, split)
        ids, hashes = set(data["question_ids"].tolist()), set(data["canonical_hashes"].tolist())
        require(not all_ids.intersection(ids) and not all_hashes.intersection(hashes), "Feature splits overlap")
        all_ids.update(ids)
        all_hashes.update(hashes)
        if split in SPLITS:
            records = [row for row in manifest["records"] if row["split"] == SPLITS[split]]
            expected = metadata_arrays(records, manifest["canonical_question_fingerprints"])
            require(all(np.array_equal(data[key], value) for key, value in expected.items()), "Original cohort identity/order differs")
        checked[split] = {"questions": len(ids), "embedding_shape": list(data["embeddings"].shape),
                          "embedding_dtype": str(data["embeddings"].dtype), "exact_keys": sorted(data),
                          "task_counts": dict(Counter(data["task_labels"].tolist())),
                          "maximum_l2_norm_error": float(np.max(np.abs(np.linalg.norm(data["embeddings"], axis=1) - 1))),
                          "identity_sha256": identity_hash(data), "npz": reference(path)}
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PRIVATE / "full_feedback_inputs/v1")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(output.is_relative_to((PRIVATE / "full_feedback_inputs").resolve()), "Private feature output scope required")
    if args.validate_only:
        receipt = read_json(output / "INPUT_ASSETS_RECEIPT.json")
        for source in receipt["sources"]:
            require(sha_file(source["path"]) == source["sha256"], "Feature source identity changed")
        checked = validate_assets(output, receipt)
        require(checked == receipt["validation"], "Saved input validation differs")
        print(json.dumps({"status": "INPUT_CONTRACT_PASS", "receipt": reference(output / "INPUT_ASSETS_RECEIPT.json"),
                          "questions": {name: value["questions"] for name, value in checked.items()}}, ensure_ascii=False))
        return
    require(not output.exists(), "New immutable output directory required; use --validate-only for an existing run")
    wall, cpu = time.perf_counter(), time.process_time()
    manifest_path = ROOT / "data/derived/split_manifest.json"
    manifest = read_json(manifest_path)
    require(read_json(ROOT / "protocol/E0_LOCK.json")["split_manifest_sha256"] == sha_file(manifest_path), "Original split lock changed")
    audit_path = ROOT / "artifacts/audit/feature_audit.json"
    encoder_receipt = ROOT / "artifacts/audit/encoder_model_download.json"
    audit = read_json(audit_path)
    require(audit["encoder_source_receipt_sha256"] == sha_file(encoder_receipt), "Frozen encoder receipt changed")
    cache_path = ROOT / "data/derived/features/features_v1.npz"
    datasets, scans = cache_inputs(manifest, audit, cache_path)
    fresh, questions, selection_path, fresh_path = fresh_inputs(manifest)
    source_paths = [Path(__file__), ROOT / "src/budget_router/encoding.py", ROOT / "src/budget_router_v2/full_feedback.py",
                    ROOT / "scripts/v2/select_fresh_development.py", Path(__file__).with_name("INPUT_CONTRACT_v1.json"),
                    ROOT / "protocol/E0_LOCK.json", manifest_path, audit_path, encoder_receipt, cache_path, selection_path, fresh_path]
    source_paths.extend(ROOT / item["path"] for item in read_json(encoder_receipt)["files"])
    for source in manifest["source_files"]:
        source_paths.append(ROOT / source["path"])
        if source.get("receipt_path"):
            source_paths.append(ROOT / source["receipt_path"])
    sources = [reference(path) for path in dict.fromkeys(source_paths)]
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), "Local CUDA must remain unused")
    started, started_cpu = time.perf_counter(), time.process_time()
    encoder = FrozenQuestionEncoder(ROOT / "models/sentence-transformers--all-MiniLM-L6-v2",
                                    revision=ENCODER_REVISION, receipt=encoder_receipt, device="cpu", max_length=256)
    require(all(parameter.device.type == "cpu" for parameter in encoder.model.parameters()), "CPU encoder required")
    fresh["embeddings"] = encoder.encode(questions, batch_size=8)
    encode_timing = {"wall_s": time.perf_counter() - started, "cpu_s": time.process_time() - started_cpu}
    require(not torch.cuda.is_initialized(), "Feature preparation initialized local CUDA")
    datasets["fresh_dev_test"] = fresh
    output.mkdir(parents=True, exist_ok=False)
    bundle = {"inputs": {}, "fresh_selection": reference(selection_path)}
    for split, data in datasets.items():
        path = output / (split + "_inputs.npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)
        bundle["inputs"][split] = reference(path)
    checked = validate_assets(output, bundle)
    for source in sources:
        require(sha_file(source["path"]) == source["sha256"], "Source changed during input preparation")
    receipt = {"schema": "routing_full_feedback_input_assets_v1", "status": "LABEL_FREE_FEATURE_INPUTS_READY",
               **bundle, "feature_encoder": {"model_id": ENCODER_ID, "revision": ENCODER_REVISION,
               "pooling": "attention_mask_mean_then_L2", "max_tokens": 256, "device": "cpu", "threads": 2, "batch_size": 8},
               "validation": checked, "sources": sources, "source_scans": scans,
               "cached_raw_embeddings_reused": 800, "fresh_questions_actually_encoded": 400,
               "cache_identity_validation": "Every qid/task/split is checked, canonical hashes are recomputed from input-only projections and bound by the original split/cache receipts.",
               "reference_values_json_decoded": 0, "sealed_question_payloads_json_decoded": 0, "gold_scoring_ops": 0,
               "llm_generation_calls": 0, "local_gpu_used": False, "pca_fitted": False, "old_pca_or_phi_accessed": False,
               "physical_source_scan_includes_locked_and_sealed_bytes": True,
               "formal_stage2_learning_complete": False, "encoder_timing": encode_timing,
               "total_timing": {"wall_s": time.perf_counter() - wall, "cpu_s": time.process_time() - cpu},
               "runtime_versions": {name: version(name) for name in ("numpy", "torch", "transformers", "tokenizers", "safetensors")}}
    destination = output / "INPUT_ASSETS_RECEIPT.json"
    write_json(destination, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": reference(destination),
                      "reused_embeddings": 800, "new_cpu_embeddings": 400, "encoder_timing": encode_timing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
