"""Immutable axes and artifact helpers for E17 supervision auditing."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
CODE=Path(__file__).resolve().parent
ASSETS=Path(os.environ.get("MAB_ASSET_ROOT",r"E:\语言模型组合路由"))
E8=ASSETS/"research/e8_decision_focused_combination_learning"
AUDIT=ASSETS/"research/e8_m4_mechanism_audit"
NS=ASSETS/"research/m4_noise_aware_set_learning"
E13=ASSETS/"research/e13_partial_combination_feedback"
STUDY=Path(os.environ.get("MAB_E17_ROOT",str(ASSETS/"research/e17_supervision_distortion_audit")))
PUBLIC,PRIVATE=STUDY/"public",STUDY/"private"
INPUTS=ASSETS/"artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz"
OUTCOMES=ASSETS/"artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
SEEDS=(42,137,2024)
METHODS=("Ridge","SPO_coordinate","TSPO_uniform")
MC_COUNTS=(1,2,4,8,16)
POLICIES=("semantic_exact","numeric_exact","semantic_near","numeric_near")
NUMERIC_TOLERANCE=1e-8
sys.path.insert(0,str(ASSETS/"research/e14_mechanism_diagnostics/public/code"))
from e14_common import array_hash,load_npz,read_json,serializable,sha256,write_json
import numpy as np


def sources():return sorted(CODE.glob("*.py"))+sorted((CODE/"tests").glob("test_*.py"))


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow(serializable(row))


def timing(cpu,wall):return dict(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)


def posterior_path(fold):return NS/f"private/runtime/noise/private/fold{fold}/POSTERIOR_TRAINING.npz"


def input_packet(fold):return load_npz(E13/f"private/inputs/fold{fold}/INPUTS_ONLY.npz")


def labels(rows):
    with np.load(OUTCOMES,allow_pickle=False) as z:return z["aggregate_scores"][:,rows].mean(axis=0)


def verify_protocol():
    p=read_json(PUBLIC/"E17_PROTOCOL.json")
    for name,digest in p["source_hashes"].items():
        if sha256(CODE/name)!=digest:raise ValueError("Frozen E17 source changed: "+name)
    for item in read_json(PRIVATE/"INPUT_MANIFEST.json")["files"]:
        if sha256(item["path"])!=item["sha256"]:raise ValueError("Historical source changed: "+item["path"])
    return p


def ridge(phi,reward,alpha):
    normal=phi.T@phi+float(alpha)*np.eye(phi.shape[1])
    w=np.linalg.solve(normal,phi.T@reward)
    return w,dict(normal_residual=float(np.max(np.abs(normal@w-phi.T@reward))),alpha=float(alpha))


def relative_calibration(phi,w,reward):
    q=phi@w;q=q-q.mean(axis=1,keepdims=True)
    r=reward-reward.mean(axis=1,keepdims=True)
    denominator=float(np.sum(q*q))
    scale=max(0.0,float(np.sum(q*r))/denominator) if denominator>1e-20 else 0.0
    return dict(nonnegative_scale=scale,score_energy=denominator,
                scope="training-only relative-value metric calibration; no deployment action change")
