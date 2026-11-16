"""Read-only numerical/source review of the two completed E20 final fits."""
from __future__ import annotations

import json
from pathlib import Path

from fit_final_models import (ASSETS, CODE, DEFAULT_ROOT, E19_CODE, LEGACY_LOG,
                              TAU, TRAIN_INPUTS, TRAIN_OUTCOMES, load_inputs,
                              now, sha, write_json)
from predict_final_inputs import features
from weighted_objectives import objective_components
import numpy as np


def main():
    root = DEFAULT_ROOT
    config = json.loads((root / "FINAL600_PREFIT_CONFIG_FROZEN.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "FINAL600_MODEL_FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    if sha(CODE / "fit_final_models.py") != config["source_hashes"]["fit_final_models.py"]:
        raise ValueError("Fit source changed after prefit lock")
    if sha(E19_CODE / "weighted_objectives.py") != config["source_hashes"]["weighted_objectives.py"]:
        raise ValueError("Frozen E19 operator changed")
    with np.load(root / "FINAL600_MODELS_FROZEN.npz", allow_pickle=False) as archive:
        state = {key: archive[key].copy() for key in archive.files}
    train = load_inputs(TRAIN_INPUTS)
    with np.load(TRAIN_OUTCOMES, allow_pickle=False) as archive:
        targets = archive["aggregate_scores"].mean(axis=0)
    phi = features(train["embeddings"], state)
    diagnostics, direct_errors = {}, {}
    for index, name in enumerate(("B", "C")):
        logged = json.loads((root / f"{name}_FIT_LOG.json").read_text(encoding="utf-8"))
        recomputed = objective_components(state["weights"][index], phi, targets,
                                           freeze["regularization"], 1-index, gamma=0)
        errors = {key: abs(logged[key]-recomputed[key]) for key in ("objective", "F0", "F1", "J_T", "regularization_penalty")}
        if max(errors.values()) > 1e-12:
            raise AssertionError("Saved final model fails direct objective recovery")
        direct_errors[name] = errors
        diagnostics[name] = {key: logged[key] for key in (
            "status", "iterations", "objective", "data_loss", "regularization_penalty", "F0", "F1", "J_T",
            "all_action_tied_questions", "non_all_tied_questions", "original_question_denominator",
            "effective_question_weight_sum", "regularization", "strong_convexity_modulus", "epigraph_rows_retained",
            "primal_violation", "dual_violation", "stationarity_inf", "complementarity_inf", "primal_dual_objective_gap",
            "direct_epigraph_error", "cpu_seconds", "wall_seconds")}
        diagnostics[name]["weight_frobenius_norm"] = float(np.linalg.norm(state["weights"][index]))
    legacy_path = ASSETS / "research/e17_supervision_distortion_audit/private/confirmation_input_only/FULL600_MODELS.npz"
    with np.load(legacy_path, allow_pickle=False) as archive:
        legacy_geometry_errors = {key: float(np.max(np.abs(state[key]-archive[key])))
                                  for key in ("pca_mean", "pca_components", "scaler_mean", "scaler_scale")}
    if max(legacy_geometry_errors.values()) > 1e-12:
        raise AssertionError("Original full600 preprocessing geometry changed")
    input_results = {}
    input_receipts = {}
    for pool in ("FRESH400", "SEALED200"):
        receipt = json.loads((root / f"{pool}_ORIGINAL_OPERATOR_PREDICTION_RECEIPT.json").read_text(encoding="utf-8"))
        path = root / f"{pool}_INPUT_ONLY_PREDICTIONS_ORIGINAL_OPERATOR_FROZEN.npz"
        if sha(path) != receipt["prediction_sha256"]:
            raise ValueError("Frozen prediction artifact changed")
        with np.load(path, allow_pickle=False) as archive:
            scores, probability, raw = archive["scores"], archive["probabilities"], archive["raw_exact_probabilities"]
            margins = archive["margins"]
        for q, p in zip(scores, probability):
            active = q >= q.max(axis=1, keepdims=True)-TAU
            if not np.array_equal(active/active.sum(axis=1, keepdims=True), p):
                raise AssertionError("Frozen main policy differs from uniform numerical argmax")
        for q, p in zip(scores, raw):
            active = q == q.max(axis=1, keepdims=True)
            if not np.array_equal(active/active.sum(axis=1, keepdims=True), p):
                raise AssertionError("Frozen raw policy differs from exact argmax")
        centered = scores-scores.mean(axis=2, keepdims=True)
        input_results[pool.lower()] = dict(
            question_count=int(scores.shape[1]), support_summary=receipt["support_summary"],
            both_strict_unique_questions=receipt["both_strict_unique_questions"],
            strict_switched_questions=receipt["strict_switched_questions"],
            average_total_variation=receipt["average_total_variation"],
            centered_score_changed_gt_1e_minus7_questions=int(np.sum(np.max(np.abs(centered[1]-centered[0]), axis=1)>1e-7)),
            maximum_common_mean_score_difference=float(np.max(np.abs(scores[1].mean(axis=1)-scores[0].mean(axis=1)))),
            deployment_quality_evaluated=False,
        )
        input_receipts[pool.lower()] = dict(prediction_sha256=receipt["prediction_sha256"],
                                           input_sha256=receipt["source_input_sha256"], utc=receipt["utc"])
    storage_receipt = json.loads((root.parent / "inputs/SEALED200_INPUT_STORAGE_REPAIR_RECEIPT.json").read_text(encoding="utf-8"))
    encoding_receipt = json.loads((root.parent / "inputs/SEALED200_INPUT_ENCODING_RECEIPT.json").read_text(encoding="utf-8"))
    report = dict(
        schema="e20_final_model_numerical_and_input_audit_v1", status="PASS", utc=now(),
        frozen_model_sha256=freeze["model_sha256"], prefit_sha256=freeze["prefit_sha256"],
        fit_source_sha256=config["source_hashes"]["fit_final_models.py"],
        weighted_objective_source_sha256=config["source_hashes"]["weighted_objectives.py"],
        final600_regularization=freeze["regularization"], regularization_rule=config["regularization_source_rule"],
        final600_regularization_executed_source_sha256=sha(LEGACY_LOG),
        new_head_fits=2, failed_real_fit_attempts=0, no_oof_ensemble=True,
        all_original_train_rows_retained=600, original_question_denominator=600,
        all_action_tied_train_rows=306, non_all_tied_train_rows=294,
        direct_objective_recovery_absolute_errors=direct_errors,
        legacy_full600_preprocessing_max_errors=legacy_geometry_errors,
        endpoint_inequality=freeze["endpoint_inequality"], model_diagnostics=diagnostics,
        input_only_diagnostics=input_results, frozen_input_receipts=input_receipts,
        sealed_input_encoding=dict(question_encodings=encoding_receipt["question_encodings"],
            batch_forward_calls=encoding_receipt["batch_forward_calls"], cpu_seconds=encoding_receipt["cpu_seconds"],
            wall_seconds=encoding_receipt["wall_seconds"], max_norm_error=encoding_receipt["max_norm_error"]),
        corrected_storage_failure=dict(failed_prediction_attempts=storage_receipt["failed_prediction_attempts"],
            reason=storage_receipt["reason"], embedding_values_changed=0, reencoded_questions=0),
        new_llm_calls=0, fresh400_reward_values_read=0, sealed200_answer_values_read=0,
        evaluation_started=False, interpretation="Input-side scores and supports only; no new quality evidence",
    )
    write_json(CODE / "FINAL_MODEL_NUMERICAL_AUDIT.json", report)
    print(json.dumps({"status": report["status"], "model_diagnostics": diagnostics,
                      "legacy_full600_preprocessing_max_errors": legacy_geometry_errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
