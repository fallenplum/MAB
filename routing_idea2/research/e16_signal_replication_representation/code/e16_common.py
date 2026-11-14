"""Frozen axes, source identities and append-only E16 helpers."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
CODE=Path(__file__).resolve().parent
ASSETS=Path(os.environ.get("MAB_ASSET_ROOT",r"E:\语言模型组合路由"))
E13=ASSETS/"research/e13_partial_combination_feedback"
E10=ASSETS/"research/e10_input_prefill_matching_audit"
E8=ASSETS/"research/e8_decision_focused_combination_learning"
STUDY=Path(os.environ.get("MAB_E16_ROOT",str(ASSETS/"research/e16_signal_replication_representation")))
PUBLIC,PRIVATE=STUDY/"public",STUDY/"private"
INPUTS=ASSETS/"artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz"
OUTCOMES=ASSETS/"artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
QWEN=E10/"private/features/PREFILL_FEATURES_FROZEN.npz"
QWEN_SHA="4235fe7a68fd95bd128c6c34543d0c142986e53dcc886182bf5687f058bc3d19"
SEEDS=(42,137,2024)
LOGGERS=(17,29,43)
PACKAGE_BUDGETS=(240,480)
REPS=("MiniLM16","MiniLM64","QwenMatched64","FusionMatched64","FusionFull64")
MULTIPLIERS=(0.1,1.0,10.0)
sys.path.insert(0,str(ASSETS/"research/e14_mechanism_diagnostics/public/code"))
from e14_common import array_hash,load_npz,read_json,serializable,sha256,write_json
import numpy as np


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:writer.writerow(serializable(row))


def source_files():
    return sorted(CODE.glob("*.py"))+sorted((CODE/"tests").glob("test_*.py"))


def timing(cpu,wall):return dict(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)


def verify_protocol():
    protocol=read_json(PUBLIC/"E16_PROTOCOL.json")
    for name,digest in protocol["source_hashes"].items():
        if sha256(CODE/name)!=digest:raise ValueError("E16 source changed: "+name)
    for item in read_json(PRIVATE/"INPUT_MANIFEST.json")["files"]:
        if sha256(item["path"])!=item["sha256"]:raise ValueError("Historical source changed")
    return protocol


def load_input_features():
    x=load_npz(INPUTS)
    q=load_npz(QWEN)
    if not np.array_equal(q["global_indices"],np.arange(600)):raise ValueError("Qwen rows not aligned")
    return x,q


def selected_labels(rows,seeds):
    # Dense cache remains in this manager. The fitter receives only selected
    # original-question/seed action vectors from its legitimate history pool.
    with np.load(OUTCOMES,allow_pickle=False) as archive:
        values=archive["aggregate_scores"]
        return values[np.asarray(seeds)[:,None],np.asarray(rows)[None,:]].copy()
