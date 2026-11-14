"""Frozen E15 configuration and append-only artifact helpers."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

CODE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
E13 = ASSETS / "research/e13_partial_combination_feedback"
E14 = ASSETS / "research/e14_mechanism_diagnostics"
E14_CODE = Path(os.environ.get("MAB_E14_CODE", str(E14 / "public/code")))
STUDY = Path(os.environ.get("MAB_E15_ROOT", str(ASSETS / "research/e15_feedback_structure_interaction")))
PUBLIC, PRIVATE = STUDY / "public", STUDY / "private"
CAPS = (125000, 250000, 500000, 750000)
PRIMARY_CAP = 250000
SEEDS, LOGGERS = (42, 137, 2024), (17, 29, 43)
STRUCTURES = ("Main", "NeighborPair", "RandomPair")
METHODS = ("RawCentered", "Group", "SaveRouterFixedMask")
DESIGNS = ("TokenMatched", "CommonCoverage")

sys.path.insert(0, str(E14_CODE))
from e14_common import array_hash, load_npz, read_json, serializable, sha256, write_json
from run_e14 import fit_existing_prefix
from common import INPUTS, OUTCOMES, verify_source_lock
import numpy as np


def world_path(fold, logger, seed, structure):
    old_name = "Pair" if structure == "NeighborPair" else structure
    return E13 / f"private/worlds/fold{fold}/logger{logger}/seed{seed}/{old_name}"


def new_world_path(fold, logger, seed):
    return PRIVATE / f"worlds/fold{fold}/logger{logger}/seed{seed}"


def sources():
    return sorted(CODE.glob("*.py")) + sorted((CODE / "tests").glob("test_*.py"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        # Stream each row to avoid duplicating large private diagnostic tables.
        for row in rows:
            writer.writerow(serializable(row))


def measured(cpu, wall):
    return dict(cpu_s=time.process_time()-cpu, wall_s=time.perf_counter()-wall)


def verify_protocol():
    p = read_json(PUBLIC / "E15_PROTOCOL.json")
    for name, expected in p["source_hashes"].items():
        if sha256(CODE / name) != expected:
            raise ValueError("Frozen E15 source changed: " + name)
    for item in read_json(PRIVATE / "INPUT_MANIFEST.json")["files"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Historical input changed: " + item["path"])
    return p


def support_statistics(inputs, arrays):
    query, action, y = (arrays[k] for k in ("observed_local_queries", "observed_actions", "observed_rewards"))
    count = np.zeros((28, 20), dtype=np.int64)
    total = np.zeros((28, 20), dtype=float)
    group = inputs["group_h"][query]
    np.add.at(count, (group, action), 1)
    np.add.at(total, (group, action), y)
    return count, total
