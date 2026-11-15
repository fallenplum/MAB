"""Append-only E18 preparation helpers. No evaluation or generation entry point."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
sys.path.insert(0, str(ASSETS / "src"))
CODE = Path(__file__).resolve().parent
STUDY = ASSETS / "research/e18_mechanism_and_confirmation_preparation"
PUBLIC, PRIVATE = STUDY / "public", STUDY / "private"
E17 = ASSETS / "research/e17_supervision_distortion_audit"
BRANCH = "codex/e18-mechanism-and-confirmation-preparation"
PARENT = "94a8579f9c8dc16d1f6afb943cee6678c5a9792a"
SEEDS = (42, 137, 2024)
POLICIES = ("Ridge", "SPO_coordinate", "TSPO_uniform", "GlobalFixed")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def input_only_npz(path):
    import numpy as np
    allowed = {"canonical_hashes", "embeddings", "question_ids", "subtask_labels", "task_labels"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != allowed:
            raise ValueError("Input archive contains unexpected fields")
        return {key: archive[key].copy() for key in sorted(allowed)}


def preserve_e17():
    manifest = read_json(E17 / "public/PUBLIC_DELIVERY_MANIFEST.json")
    for record in manifest["files"]:
        if sha256(E17 / "public" / record["path"]) != record["sha256"]:
            raise ValueError("E17 public asset changed")
    private = E17 / "private/confirmation_input_only"
    receipt = read_json(private / "CONFIRMATION_PREPARATION_FROZEN.json")
    for record in receipt["private_artifacts"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("E17 frozen confirmation asset changed")
    return dict(public_files=len(manifest["files"])+1,
                public_manifest_sha256=sha256(E17 / "public/PUBLIC_DELIVERY_MANIFEST.json"),
                frozen_model_receipt_sha256=sha256(private / "CONFIRMATION_PREPARATION_FROZEN.json"),
                private_artifacts_verified=len(receipt["private_artifacts"]))
