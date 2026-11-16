"""Encode the already projected sealed input-only file with frozen MiniLM.

The mixed protected source and its answers are unavailable to this interface.
"""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import sys
import time

from fit_final_models import ASSETS, DEFAULT_ROOT, INPUT_ROOT, now, sha, write_json, write_npz
import numpy as np


def main():
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    inputs = DEFAULT_ROOT.parent / "inputs"
    source_path = inputs / "SEALED200_INPUTS_ONLY.jsonl"
    projection_path = inputs / "SEALED200_INPUT_PROJECTION_RECEIPT.json"
    destination = inputs / "SEALED200_INPUTS_ONLY.npz"
    if destination.exists():
        raise FileExistsError("Sealed input vector packet already exists and is immutable")
    freeze_path = DEFAULT_ROOT / "FINAL600_MODEL_FREEZE_RECEIPT.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if sha(DEFAULT_ROOT / "FINAL600_MODELS_FROZEN.npz") != freeze["model_sha256"]:
        raise ValueError("Final B/C models must be frozen before sealed input encoding")
    keys = {"canonical_question_sha256", "question", "question_id", "source_split", "subtask", "task"}
    rows = []
    with source_path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != keys:
                raise ValueError("Unexpected field in previously isolated input-only text packet")
            rows.append(row)
    if len(rows) != 200 or any(sum(row["task"] == task for row in rows) != 100 for task in ("gsm8k", "bbh")):
        raise ValueError("Expected balanced protected200 input-only pool")
    encoder_source = INPUT_ROOT / "INPUT_ASSETS_RECEIPT.json"
    source = json.loads(encoder_source.read_text(encoding="utf-8-sig"))
    versions = {package: metadata.version(package) for package in ("numpy", "torch", "transformers", "tokenizers", "safetensors")}
    if any(versions[package] != source["runtime_versions"][package] for package in versions):
        raise ValueError("Frozen MiniLM runtime differs from original input receipt")
    sys.path.insert(0, str(ASSETS / "src"))
    from budget_router.encoding import FrozenQuestionEncoder
    import torch
    torch.set_num_threads(2)
    write_json(inputs / "SEALED200_ENCODER_PREFREEZE.json", dict(
        schema="e20_sealed_input_encoder_prefreeze_v1", utc=now(), input_sha256=sha(source_path),
        projection_receipt_sha256=sha(projection_path), model_freeze_receipt_sha256=sha(freeze_path),
        final_models_sha256=freeze["model_sha256"], source_sha256=sha(__file__),
        encoder_source_sha256=sha(ASSETS / "src/budget_router/encoding.py"),
        encoder_receipt_sha256=sha(encoder_source), revision=source["feature_encoder"]["revision"],
        runtime_versions=versions, device="cpu", batch_size=8, max_length=256, torch_threads=2,
        answers_decoded=0, rewards_decoded=0, parameter_updates=0, llm_calls=0,
    ))
    encoder = FrozenQuestionEncoder(
        ASSETS / "models/sentence-transformers--all-MiniLM-L6-v2",
        revision=source["feature_encoder"]["revision"],
        receipt=ASSETS / "artifacts/audit/encoder_model_download.json", device="cpu", max_length=256)
    vectors = encoder.encode([row["question"] for row in rows], batch_size=8)
    if vectors.shape != (200, 384) or not np.isfinite(vectors).all():
        raise ValueError("Invalid original MiniLM vector shape")
    norm_error = float(np.max(np.abs(np.linalg.norm(vectors, axis=1)-1)))
    if norm_error > 1e-5:
        raise ValueError("Frozen L2-pooled vector contract failed")
    write_npz(destination, embeddings=vectors, question_ids=np.array([row["question_id"] for row in rows]),
              canonical_hashes=np.array([row["canonical_question_sha256"] for row in rows]),
              task_labels=np.array([row["task"] for row in rows]), subtask_labels=np.array([row["subtask"] for row in rows]))
    receipt = dict(
        schema="e20_sealed_input_encoding_receipt_v1", status="FROZEN_INPUT_ONLY_VECTORS", utc=now(),
        input_sha256=sha(source_path), vector_packet_sha256=sha(destination),
        final_models_sha256=freeze["model_sha256"], frozen_encoder=True, max_norm_error=norm_error,
        question_encodings=200, batch_forward_calls=25, batch_size=8,
        new_llm_calls=0, answer_values_read=0, reward_values_read=0, parameter_updates=0,
        cpu_seconds=time.process_time()-started_cpu, wall_seconds=time.perf_counter()-started_wall,
    )
    write_json(inputs / "SEALED200_INPUT_ENCODING_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
