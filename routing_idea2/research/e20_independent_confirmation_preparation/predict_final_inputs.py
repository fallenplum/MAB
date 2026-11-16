"""Replay original sklearn/F-order inference on frozen E20 final models.

This preserves the initial mathematical-equivalent input prediction artifact
and records all policy differences from that input-only numeric implementation.
No coefficient, hyperparameter, training target or evaluation reward is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fit_final_models import (DEFAULT_ROOT, TAU, TRAIN_INPUTS, SAFE_INPUT_KEYS,
                              load_inputs, now, policy, sha, write_json, write_npz)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def features(embeddings, state):
    pca = PCA(n_components=16, svd_solver="full")
    pca.mean_ = state["pca_mean"]
    pca.components_ = state["pca_components"]
    pca.n_features_in_ = 384
    pca.n_components_ = 16
    scaler = StandardScaler()
    scaler.mean_ = state["scaler_mean"]
    scaler.scale_ = state["scaler_scale"]
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = 16
    scaler.n_samples_seen_ = np.int64(600)
    projected = pca._transform(np.asarray(embeddings, dtype=np.float64), xp=np)
    return np.column_stack((np.ones(len(embeddings)), scaler.transform(projected)))


def main(root, input_path, label):
    freeze = json.loads((root / "FINAL600_MODEL_FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    model_path = root / "FINAL600_MODELS_FROZEN.npz"
    if sha(model_path) != freeze["model_sha256"]:
        raise ValueError("Frozen coefficients changed")
    with np.load(model_path, allow_pickle=False) as archive:
        state = {key: archive[key].copy() for key in archive.files}
    train = load_inputs(TRAIN_INPUTS)
    train_phi = features(train["embeddings"], state)
    preprocessing_recovery_error = float(np.max(np.abs(train_phi-state["phi_h"])))
    if preprocessing_recovery_error > 1e-13:
        raise ValueError("Original train600 preprocessing does not recover its saved design")
    packet = load_inputs(input_path)
    expected = 400 if label == "fresh400" else 200
    if len(packet["embeddings"]) != expected:
        raise ValueError("Prespecified input count differs")
    for key in ("question_ids", "canonical_hashes"):
        if set(packet[key].tolist()) & set(state[key].tolist()):
            raise ValueError("Input pool overlaps historical training identities")
    phi = features(packet["embeddings"], state)
    scores = np.stack([phi @ np.asfortranarray(coefficient) for coefficient in state["weights"]])
    main = [policy(value, TAU) for value in scores]
    raw = [policy(value, 0) for value in scores]
    probabilities = np.stack([item[0] for item in main])
    raw_probabilities = np.stack([item[0] for item in raw])
    previous_path = root / f"{label.upper()}_INPUT_ONLY_PREDICTIONS_FROZEN.npz"
    previous_audit = dict(exists=previous_path.is_file())
    if previous_path.is_file():
        with np.load(previous_path, allow_pickle=False) as archive:
            previous_audit.update(
                source_sha256=sha(previous_path), maximum_score_difference=float(np.max(np.abs(scores-archive["scores"]))),
                primary_changed_policy_rows=int(np.any(probabilities != archive["probabilities"], axis=2).sum()),
                raw_exact_changed_policy_rows=int(np.any(raw_probabilities != archive["raw_exact_probabilities"], axis=2).sum()),
            )
    destination = root / f"{label.upper()}_INPUT_ONLY_PREDICTIONS_ORIGINAL_OPERATOR_FROZEN.npz"
    write_npz(destination, conditions=np.array(["B", "C"]), scores=scores,
              centered_scores=scores-scores.mean(axis=2, keepdims=True), probabilities=probabilities,
              raw_exact_probabilities=raw_probabilities, support_counts=np.stack([item[1] for item in main]),
              margins=np.stack([item[2] for item in main]),
              **{key: packet[key] for key in SAFE_INPUT_KEYS - {"embeddings"}})
    receipt = dict(
        schema="e20_original_numeric_operator_prediction_v1", pool=label, utc=now(),
        status="INPUT_ONLY_POLICY_FROZEN", model_sha256=freeze["model_sha256"],
        source_sha256=sha(__file__), source_input_sha256=sha(input_path),
        prediction_sha256=sha(destination), train_preprocessing_recovery_max_error=preprocessing_recovery_error,
        numeric_operator="original sklearn PCA._transform, StandardScaler.transform, per-head phi @ F-order coefficient",
        previous_mathematical_equivalent_input_implementation=previous_audit,
        model_parameter_updates=0, reward_values_read=0, answer_values_read=0,
        new_prediction_head_fits=0, new_llm_calls=0, new_encoder_forwards=0,
        support_summary={name: {"minimum": int(main[index][1].min()), "maximum": int(main[index][1].max()),
                                "mean": float(main[index][1].mean()), "numerically_tied_questions": int(np.sum(main[index][1] > 1)),
                                "raw_exact_tied_questions": int(np.sum(raw[index][1] > 1))}
                         for index, name in enumerate(("B", "C"))},
        both_strict_unique_questions=int(np.sum((main[0][2] > TAU) & (main[1][2] > TAU))),
        strict_switched_questions=int(np.sum((main[0][2] > TAU) & (main[1][2] > TAU) & (scores[0].argmax(axis=1) != scores[1].argmax(axis=1)))),
        average_total_variation=float(np.abs(probabilities[0]-probabilities[1]).sum(axis=1).mean()/2),
    )
    write_json(root / f"{label.upper()}_ORIGINAL_OPERATOR_PREDICTION_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label", choices=("fresh400", "sealed200"), required=True)
    args = parser.parse_args()
    main(args.root, args.input, args.label)
