"""Fit and freeze E20 B/C on train600, then make input-only predictions.

The fit phase opens historical train600 outcomes only. The prediction phase
accepts a five-member input-only NPZ and has no reward/answer argument.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

CODE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
E19_CODE = CODE.parent / "e19_symmetric_tie_constraint"
if not E19_CODE.is_dir():
    E19_CODE = ASSETS / "research/e19_symmetric_tie_constraint/public/code"
sys.path.insert(0, str(E19_CODE))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from weighted_objectives import fit_weighted_tspo

INPUT_ROOT = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
TRAIN_INPUTS = INPUT_ROOT / "selector_train_inputs.npz"
TRAIN_OUTCOMES = ASSETS / "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
LEGACY_LOG = ASSETS / "research/e17_supervision_distortion_audit/private/confirmation_input_only/FULL600_FIT_LOG.json"
LEGACY_CONFIG = ASSETS / "research/e17_supervision_distortion_audit/public/CONFIRMATION_INPUT_ONLY_CONFIG_FROZEN.json"
DEFAULT_ROOT = ASSETS / "research/e20_independent_confirmation_preparation/private/final_models"
SAFE_INPUT_KEYS = {"canonical_hashes", "embeddings", "question_ids", "subtask_labels", "task_labels"}
WEIGHTED_SHA = "019c505c4a5ab2f90e532eda2842546c444c6480a182504f56c935f3527dd4da"
TRAIN_INPUT_SHA = "57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3"
TRAIN_OUTCOME_SHA = "a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e"
TAU = 1e-8


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
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


def write_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_inputs(path):
    # Reject contaminated archives before any member is decoded.
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != SAFE_INPUT_KEYS:
            raise ValueError("Require exactly the five original label-free input members")
        packet = {key: archive[key].copy() for key in sorted(SAFE_INPUT_KEYS)}
    count = len(packet["embeddings"])
    if packet["embeddings"].shape != (count, 384) or not np.isfinite(packet["embeddings"]).all():
        raise ValueError("Finite original MiniLM384 embeddings required")
    for key in SAFE_INPUT_KEYS - {"embeddings"}:
        if packet[key].shape != (count,):
            raise ValueError("Input-only identity dimension mismatch")
    for key in ("question_ids", "canonical_hashes"):
        if len(set(packet[key].tolist())) != count:
            raise ValueError("Duplicated input identity")
    return packet


def policy(scores, tolerance):
    active = scores >= scores.max(axis=1, keepdims=True) - tolerance
    probabilities = active / active.sum(axis=1, keepdims=True)
    ordered = np.sort(scores, axis=1)
    return probabilities, active.sum(axis=1), ordered[:, -1] - ordered[:, -2]


def fit(root):
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    if root.exists():
        raise FileExistsError("Final-model directory already exists and is immutable")
    for path, expected in ((TRAIN_INPUTS, TRAIN_INPUT_SHA), (TRAIN_OUTCOMES, TRAIN_OUTCOME_SHA),
                           (E19_CODE / "weighted_objectives.py", WEIGHTED_SHA)):
        if sha(path) != expected:
            raise ValueError("Frozen historical training/operator source identity differs")
    legacy_log = json.loads(LEGACY_LOG.read_text(encoding="utf-8-sig"))
    legacy_config = json.loads(LEGACY_CONFIG.read_text(encoding="utf-8-sig"))
    inherited = float(legacy_config["parameters"]["spo_regularization"])
    relevant = [item for item in legacy_log["fits"] if item["method"] in ("SPO_coordinate", "TSPO_uniform")]
    if len(relevant) != 2 or any(float(item["diagnostics"]["regularization"]) != inherited for item in relevant):
        raise ValueError("Frozen final600 configuration and executed fits disagree")
    if inherited != 0.003472222222222222:
        raise ValueError("Unexpected inherited final600 L2 coefficient")
    train = load_inputs(TRAIN_INPUTS)
    if len(train["embeddings"]) != 600 or any(np.sum(train["task_labels"] == task) != 300 for task in ("gsm8k", "bbh")):
        raise ValueError("Original balanced train600 required")
    with np.load(TRAIN_OUTCOMES, allow_pickle=False) as archive:
        required = ("question_ids", "canonical_hashes", "generation_seeds", "aggregate_scores")
        outcome = {key: archive[key].copy() for key in required}
    for key in ("question_ids", "canonical_hashes"):
        if not np.array_equal(outcome[key], train[key]):
            raise ValueError("Historical outcomes and input identity mapping differs")
    if not np.array_equal(outcome["generation_seeds"], [42, 137, 2024]):
        raise ValueError("Frozen generation seed order differs")
    reward = outcome["aggregate_scores"]
    if reward.shape != (3, 600, 20) or not np.isin(reward, [0, 1]).all():
        raise ValueError("Original train600 binary scores required")
    target = reward.mean(axis=0)
    config = dict(
        schema="e20_final600_prefit_v1", status="FROZEN_BEFORE_TWO_FINAL_FITS", utc=now(),
        source_commit="07b01361d425b089e49643f1ec5282eae0ac1ec0", conditions={"B": {"beta": 1, "gamma": 0}, "C": {"beta": 0, "gamma": 0}},
        training_questions=600, generation_seeds=[42, 137, 2024], actions=20,
        feature_recipe="MiniLM384, train600-only PCA16(full SVD), StandardScaler, constant column",
        train600_only_preprocessing=True, feature_count=17, parameters_per_head=340,
        regularization=inherited, penalty="lambda * all 340 coefficients squared including constant",
        normalization="1/600, all epigraph rows retained in both conditions",
        regularization_source_rule=legacy_config["parameters"]["rule"],
        n600_rescaling_performed=False, parameter_grid_search=False,
        training_targets="uniform over empirical maxima of three-generation mean reward",
        primary_deployment="uniform over scores >= maximum - 1e-8", tau=TAU,
        sensitivity_deployment="uniform over raw exactly equal maxima",
        solver_settings={"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "max_iter": 1000, "max_threads": 1},
        source_hashes={"fit_final_models.py": sha(__file__), "weighted_objectives.py": WEIGHTED_SHA},
        historical_sources={"train_inputs": TRAIN_INPUT_SHA, "train_outcomes": TRAIN_OUTCOME_SHA,
                            "legacy_full600_fit_log": sha(LEGACY_LOG), "legacy_full600_prefit": sha(LEGACY_CONFIG)},
        fitting_reward_access=dict(train600_questions=600, fresh400_reward_values=0, sealed200_answer_values=0),
        new_llm_calls=0, new_encoder_forwards=0,
    )
    root.mkdir(parents=True)
    write_json(root / "FINAL600_PREFIT_CONFIG_FROZEN.json", config)
    pca = PCA(n_components=16, svd_solver="full").fit(train["embeddings"].astype(np.float64))
    transformed = pca.transform(train["embeddings"].astype(np.float64))
    scaler = StandardScaler().fit(transformed)
    phi = np.column_stack((np.ones(600), scaler.transform(transformed)))
    coefficients, diagnostics = [], {}
    for name, beta in (("B", 1), ("C", 0)):
        coefficient, diagnostic = fit_weighted_tspo(phi, target, inherited, beta, gamma=0, strict_tolerances=True)
        coefficients.append(coefficient)
        diagnostics[name] = diagnostic
        write_json(root / f"{name}_FIT_LOG.json", diagnostic)
    difference_f0 = diagnostics["B"]["F0"] - diagnostics["C"]["F0"]
    difference_jt = diagnostics["C"]["J_T"] - diagnostics["B"]["J_T"]
    if not (-1e-7 <= difference_f0 <= difference_jt + 1e-7):
        raise AssertionError("Full600 endpoint optimality inequality failed")
    model_path = root / "FINAL600_MODELS_FROZEN.npz"
    write_npz(model_path, weights=np.stack(coefficients), conditions=np.array(["B", "C"]),
              history_indices=np.arange(600), pca_mean=pca.mean_, pca_components=pca.components_,
              scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, question_ids=train["question_ids"],
              canonical_hashes=train["canonical_hashes"], phi_h=phi)
    write_npz(root / "TRAIN600_TARGETS_PRIVATE.npz", mean_rewards=target, all_tied=np.all(target == target[:, :1], axis=1))
    scores = np.einsum("qd,mdc->mqc", phi, np.stack(coefficients))
    write_npz(root / "TRAIN600_SCORES_PRIVATE.npz", scores=scores, centered_scores=scores-scores.mean(axis=2, keepdims=True))
    receipt = dict(
        schema="e20_final600_models_frozen_v1", status="FROZEN_BEFORE_NEW_INPUT_PREDICTIONS", utc=now(),
        model_sha256=sha(model_path), prefit_sha256=sha(root / "FINAL600_PREFIT_CONFIG_FROZEN.json"),
        source_hashes=config["source_hashes"], training_input_sha256=TRAIN_INPUT_SHA,
        training_outcomes_sha256=TRAIN_OUTCOME_SHA, regularization=inherited,
        beta_values=[1, 0], gamma=0, model_parameter_shape=[2, 17, 20],
        all_action_tied_train_questions=diagnostics["B"]["all_action_tied_questions"],
        endpoint_inequality=dict(F0_B_minus_F0_C=difference_f0, JT_C_minus_JT_B=difference_jt, passed=True),
        new_prediction_head_fits=2, failed_real_fits=0,
        fit_cpu_seconds=sum(x["cpu_seconds"] for x in diagnostics.values()),
        fit_wall_seconds=sum(x["wall_seconds"] for x in diagnostics.values()),
        total_cpu_seconds=time.process_time()-started_cpu, total_wall_seconds=time.perf_counter()-started_wall,
        fresh400_reward_values_read=0, sealed200_answer_values_read=0,
        fresh400_input_predictions_completed=False, sealed200_input_predictions_completed=False,
        new_llm_calls=0, new_encoder_forwards=0,
    )
    write_json(root / "FINAL600_MODEL_FREEZE_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def predict(root, input_path, label):
    if label not in ("fresh400", "sealed200"):
        raise ValueError("Prediction pool must be the prespecified fresh400 or sealed200")
    model_path = root / "FINAL600_MODELS_FROZEN.npz"
    receipt = json.loads((root / "FINAL600_MODEL_FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    if sha(model_path) != receipt["model_sha256"] or sha(root / "FINAL600_PREFIT_CONFIG_FROZEN.json") != receipt["prefit_sha256"]:
        raise ValueError("Frozen model or configuration changed")
    packet = load_inputs(input_path)
    expected = 400 if label == "fresh400" else 200
    if len(packet["embeddings"]) != expected:
        raise ValueError("Prespecified input pool size differs")
    with np.load(model_path, allow_pickle=False) as archive:
        model = {key: archive[key].copy() for key in archive.files}
    if set(packet["question_ids"].tolist()) & set(model["question_ids"].tolist()):
        raise ValueError("Prediction pool overlaps historical training IDs")
    if set(packet["canonical_hashes"].tolist()) & set(model["canonical_hashes"].tolist()):
        raise ValueError("Prediction pool overlaps historical training canonical input hashes")
    projected = (packet["embeddings"]-model["pca_mean"]) @ model["pca_components"].T
    standardized = (projected-model["scaler_mean"]) / model["scaler_scale"]
    phi = np.column_stack((np.ones(expected), standardized))
    scores = np.einsum("qd,mdc->mqc", phi, model["weights"])
    main = [policy(value, TAU) for value in scores]
    raw = [policy(value, 0) for value in scores]
    probabilities = np.stack([item[0] for item in main])
    raw_probabilities = np.stack([item[0] for item in raw])
    destination = root / f"{label.upper()}_INPUT_ONLY_PREDICTIONS_FROZEN.npz"
    write_npz(destination, conditions=np.array(["B", "C"]), scores=scores,
              centered_scores=scores-scores.mean(axis=2, keepdims=True), probabilities=probabilities,
              raw_exact_probabilities=raw_probabilities, support_counts=np.stack([item[1] for item in main]),
              margins=np.stack([item[2] for item in main]),
              **{key: packet[key] for key in SAFE_INPUT_KEYS - {"embeddings"}})
    summary = dict(
        schema="e20_input_only_prediction_receipt_v1", pool=label, utc=now(), status="INPUT_ONLY_FROZEN",
        model_sha256=receipt["model_sha256"], model_frozen_utc=receipt["utc"],
        source_input_sha256=sha(input_path), prediction_sha256=sha(destination),
        question_count=expected, task_counts={str(k): int(np.sum(packet["task_labels"] == k)) for k in np.unique(packet["task_labels"])},
        support_summary={name: {"minimum": int(main[index][1].min()), "maximum": int(main[index][1].max()),
                                "mean": float(main[index][1].mean()), "numerically_tied_questions": int(np.sum(main[index][1] > 1)),
                                "raw_exact_tied_questions": int(np.sum(raw[index][1] > 1))}
                         for index, name in enumerate(("B", "C"))},
        both_strict_unique_questions=int(np.sum((main[0][2] > TAU) & (main[1][2] > TAU))),
        strict_switched_questions=int(np.sum((main[0][2] > TAU) & (main[1][2] > TAU) & (scores[0].argmax(axis=1) != scores[1].argmax(axis=1)))),
        average_total_variation=float(np.abs(probabilities[0]-probabilities[1]).sum(axis=1).mean()/2),
        fresh400_reward_values_read=0, sealed200_answer_values_read=0, new_llm_calls=0, new_encoder_forwards=0,
    )
    write_json(root / f"{label.upper()}_INPUT_PREDICTION_RECEIPT.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit", "predict"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--label", choices=("fresh400", "sealed200"))
    args = parser.parse_args()
    if args.phase == "fit":
        fit(args.root)
    else:
        if args.input is None or args.label is None:
            parser.error("Input-only prediction requires --input and --label")
        predict(args.root, args.input, args.label)
