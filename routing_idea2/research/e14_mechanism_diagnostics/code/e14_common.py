"""Shared paths and append-only records for the three controlled E14 audits."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

CODE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
E13 = ASSETS / "research/e13_partial_combination_feedback"
E13_CODE = Path(os.environ.get("MAB_E13_CODE", str(E13 / "public/code")))
STUDY = Path(os.environ.get("MAB_E14_ROOT", str(ASSETS / "research/e14_mechanism_diagnostics")))
PUBLIC, PRIVATE = STUDY / "public", STUDY / "private"
BUDGETS = (60, 120, 240, 300, 360, 480)
LOGGERS, SEEDS = (17, 29, 43), (42, 137, 2024)
METHODS = ("Raw", "Group", "GroupResidual", "RawCentered", "SaveRouterFixedMask",
           "GlobalFixed", "TaskFixed", "SubtaskFixed")
A_METHODS = ("Group", "GroupResidual", "SaveRouterFixedMask", "TaskFixed", "SubtaskFixed")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(serializable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty table requires an explicit schema")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(serializable(rows))


def world_path(fold, logger, seed, scenario):
    return E13 / f"private/worlds/fold{fold}/logger{logger}/seed{seed}/{scenario}"


def sources():
    return sorted(CODE.glob("*.py")) + sorted((CODE / "tests").glob("test_*.py"))


def measured(start_cpu, start_wall):
    return {"cpu_s": time.process_time() - start_cpu, "wall_s": time.perf_counter() - start_wall}


def check_protocol():
    protocol = read_json(PUBLIC / "E14_PROTOCOL.json")
    for name, expected in protocol["source_hashes"].items():
        if sha256(CODE / name) != expected:
            raise ValueError("E14 source changed after freeze: " + name)
    for item in read_json(PRIVATE / "INPUT_MANIFEST.json")["files"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Frozen E13 source changed: " + item["path"])
    return protocol
