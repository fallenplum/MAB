"""Read-only replay of E20 frozen preprocessing, decoding and source receipts.

This checks existing artifacts and writes a de-identified review receipt only.
It performs no fitting and reads no fresh reward or sealed answer.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import nct, t

import predict_final_inputs as predictor
import fit_final_models as trainer


def main():
    code = Path(__file__).resolve().parent
    assets = trainer.ASSETS
    root = predictor.DEFAULT_ROOT
    review_path = code / "E20_NUMERICAL_REVIEW.json"
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    config = load(root / "FINAL600_PREFIT_CONFIG_FROZEN.json")
    freeze = load(root / "FINAL600_MODEL_FREEZE_RECEIPT.json")
    legacy = load(trainer.LEGACY_CONFIG)
    legacy_log = load(trainer.LEGACY_LOG)
    logs = {name: load(root / f"{name}_FIT_LOG.json") for name in ("B", "C")}
    regularization = float(legacy["parameters"]["spo_regularization"])
    review = {
        "schema": "e20_independent_numerical_review_v1",
        "review_scope": "Read-only source and frozen-training metadata review, objective reconstruction, input rejection software test, analytic power reproduction; no refit or evaluation",
        "checks": {}, "original_final600_regularization": regularization,
        "regularization_source_rule": legacy["parameters"]["rule"],
        "training_questions": 600, "model_sha256": freeze["model_sha256"],
        "reviewed_fit_source_sha256": trainer.sha(code / "fit_final_models.py"),
        "new_fits_executed": 0, "fresh_reward_values_read": 0,
        "sealed_answer_values_read": 0, "new_llm_calls": 0, "bootstrap_repetitions": 0,
        "limits": [
            "Review of available final input-only receipts does not authorize absent input-pool prediction or any reward evaluation.",
            "Power is conditional on E19 development variance and complete-support expected policy utility.",
            "This review checks implementation and numerical qualifications, without independent scientific quality evidence.",
        ],
    }
    with np.load(root / "FINAL600_MODELS_FROZEN.npz", allow_pickle=False) as source:
        state = {key: source[key].copy() for key in (
            "pca_mean", "pca_components", "scaler_mean", "scaler_scale", "phi_h", "weights", "history_indices"
        )}
    with np.load(predictor.TRAIN_INPUTS, allow_pickle=False) as source:
        original_embeddings = source["embeddings"].copy()
    embeddings = np.asarray(original_embeddings, dtype=np.float64)
    checks = review["checks"]
    checks["regularization_matches_frozen_final600_config"] = freeze["regularization"] == regularization
    checks["regularization_is_median_of_five_training_only_parameters"] = regularization == np.median(
        [row["spo_regularization"] for row in legacy["parameters"]["source_folds"]]
    )
    checks["regularization_matches_old_executed_final600_heads"] = all(
        row["diagnostics"]["regularization"] == regularization for row in legacy_log["fits"]
        if row["method"] in ("SPO_coordinate", "TSPO_uniform")
    )
    checks["no_n600_rescaling_or_new_parameter_search"] = (
        config["n600_rescaling_performed"] is False and config["parameter_grid_search"] is False
    )
    checks["original_E19_operator_hash"] = trainer.sha(trainer.E19_CODE / "weighted_objectives.py") == trainer.WEIGHTED_SHA
    checks["source_code_matches_prefit_freeze"] = trainer.sha(code / "fit_final_models.py") == config["source_hashes"]["fit_final_models.py"]
    checks["model_hash_matches_freeze"] = trainer.sha(root / "FINAL600_MODELS_FROZEN.npz") == freeze["model_sha256"]
    checks["prefit_hash_matches_freeze"] = trainer.sha(root / "FINAL600_PREFIT_CONFIG_FROZEN.json") == freeze["prefit_sha256"]
    checks["weights_exactly_two_17_by_20_heads"] = state["weights"].shape == (2, 17, 20)
    checks["all_600_original_training_rows_retained"] = (
        state["phi_h"].shape == (600, 17) and np.array_equal(state["history_indices"], np.arange(600))
        and all(log["original_question_denominator"] == 600 and log["epigraph_rows_retained"] == 600 for log in logs.values())
    )
    checks["standardized_training_columns_have_zero_mean_unit_variance"] = bool(
        np.allclose(state["phi_h"][:, 1:].mean(axis=0), 0, atol=1e-12)
        and np.allclose(state["phi_h"][:, 1:].var(axis=0), 1, atol=1e-12)
    )
    checks["constant_is_present_and_penalized"] = bool(
        np.array_equal(state["phi_h"][:, 0], np.ones(600)) and all(log["penalized_constant_feature"] for log in logs.values())
    )
    with np.load(root / "TRAIN600_TARGETS_PRIVATE.npz", allow_pickle=False) as source:
        rewards, mask = source["mean_rewards"].copy(), source["all_tied"].copy()
    checks["all_tied_training_mask_reconstructs"] = bool(
        np.array_equal(mask, np.all(rewards == rewards[:, :1], axis=1)) and int(mask.sum()) == 306
    )
    review["all_tied_training_questions"] = int(mask.sum())
    errors = []
    for index, name in enumerate(("B", "C")):
        beta = 1 if name == "B" else 0
        q = state["phi_h"] @ state["weights"][index]
        maxima = rewards == rewards.max(axis=1, keepdims=True)
        alpha = maxima / maxima.sum(axis=1, keepdims=True)
        transformed = 2 * q - rewards
        losses = transformed.max(axis=1) - (alpha * transformed).sum(axis=1)
        penalty = regularization * np.square(state["weights"][index]).sum()
        f0 = losses[~mask].sum() / 600 + penalty
        jt = 2 * (q.max(axis=1) - q.mean(axis=1))[mask].sum() / 600
        objective = (np.where(mask, beta, 1) * losses).sum() / 600 + penalty
        differences = [abs(f0 - logs[name]["F0"]), abs(jt - logs[name]["J_T"]), abs(objective - logs[name]["objective"])]
        errors.extend(differences)
        checks[name + "_independent_fullN_objective_reconstruction"] = max(differences) < 1e-12
        checks[name + "_uniform_empirical_maximum_weights"] = bool(np.allclose(alpha.sum(axis=1), 1) and np.all(alpha[~maxima] == 0))
        checks[name + "_solver_and_KKT_qualified"] = logs[name]["status"] == "Solved" and max(
            logs[name][key] for key in ("primal_violation", "dual_violation", "stationarity_inf", "complementarity_inf", "primal_dual_objective_gap", "direct_epigraph_error")
        ) < 1e-7
    checks["endpoint_optimality_inequality"] = -1e-7 <= logs["B"]["F0"] - logs["C"]["F0"] <= logs["C"]["J_T"] - logs["B"]["J_T"] + 1e-7
    review["maximum_objective_reconstruction_error"] = max(errors)
    review["maximum_KKT_stationarity"] = max(log["stationarity_inf"] for log in logs.values())
    review["maximum_primal_dual_objective_gap"] = max(log["primal_dual_objective_gap"] for log in logs.values())
    clean = dict(embeddings=np.zeros((2, 384)), canonical_hashes=np.array(["a", "b"]),
                 question_ids=np.array(["q1", "q2"]), subtask_labels=np.array(["s", "s"]), task_labels=np.array(["bbh", "bbh"]))
    for extra in ("Y", "reward", "answers"):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **clean, **{extra: np.array([object()], dtype=object)})
        buffer.seek(0)
        try:
            trainer.load_inputs(buffer)
            rejected = False
        except ValueError as error:
            rejected = "five original label-free" in str(error)
        checks["input_interface_rejects_" + extra + "_before_member_decoding"] = rejected
    checks["prefit_configuration_precedes_model_freeze"] = datetime.fromisoformat(config["utc"]) < datetime.fromisoformat(freeze["utc"])
    power = load(code / "E20_POWER_RESULTS.json")
    with np.load(assets / "research/e19_symmetric_tie_constraint/private/OOF_EVALUATION.npz", allow_pickle=False) as source:
        utilities, labels = source["utilities"].copy(), source["task_labels"].copy()
    d = (utilities[1] - utilities[0]).mean(axis=0)
    vg, vb = (d[labels == task].var(ddof=1) for task in ("gsm8k", "bbh"))
    effective_variance = (vg + vb) / 2
    summary = power["decoders"]["tau_1e-8_uniform"]["variance_summary"]
    checks["power_uses_600_question_clusters_and_1800_recorded_events"] = d.shape == (600,) and summary["independent_development_questions"] == 600 and summary["recorded_generation_events"] == 1800
    checks["power_task_weighting_and_variance_units"] = bool(
        np.isclose(effective_variance, summary["effective_variance_for_balanced_total_N"], atol=1e-15)
        and np.isclose(np.sqrt(effective_variance), summary["effective_standard_deviation"], atol=1e-15)
        and np.isclose((d[labels == "gsm8k"].mean() + d[labels == "bbh"].mean()) * 50, summary["macro_mean_difference_pp"], atol=1e-12)
    )
    row = next(r for r in power["decoders"]["tau_1e-8_uniform"]["single_primary_comparison"]["power_grid"] if r["total_independent_questions"] == 400 and r["effect_pp"] == 1)
    se = np.sqrt(0.25 * (vg / 200 + vb / 200))
    df = (vg / 200 + vb / 200)**2 / ((vg / 200)**2 / 199 + (vb / 200)**2 / 199)
    critical = t.isf(0.025, df)
    noncentrality = 0.01 / se
    reproduced_power = nct.sf(critical, df, noncentrality) + nct.cdf(-critical, df, noncentrality)
    checks["power_two_sided_tails_and_pp_conversion"] = bool(abs(reproduced_power - row["welch_noncentral_t_approximation_power"]) < 1e-14 and abs(100 * se - row["standard_error_pp"]) < 1e-12)
    scenario = power["decoders"]["tau_1e-8_uniform"]["sealed199_pending_approval_scenario"]
    unequal_row = next(row for row in scenario["single_primary_comparison_power"] if row["effect_pp"] == 1)
    unequal_se = np.sqrt(0.25 * vg / 100 + 0.25 * vb / 99)
    unequal_df = (vg / 100 + vb / 99)**2 / ((vg / 100)**2 / 99 + (vb / 99)**2 / 98)
    unequal_critical = t.isf(0.025, unequal_df)
    unequal_noncentrality = 0.01 / unequal_se
    unequal_power = nct.sf(unequal_critical, unequal_df, unequal_noncentrality) + nct.cdf(-unequal_critical, unequal_df, unequal_noncentrality)
    checks["sealed199_power_preserves_half_task_weights_with_100_and_99_questions"] = bool(
        abs(unequal_power - unequal_row["welch_noncentral_t_approximation_power"]) < 1e-14
        and abs(100 * unequal_se - unequal_row["standard_error_pp"]) < 1e-12
        and scenario["original_sealed200_pool_preserved"]
        and not scenario["automatic_question_removal_performed"]
    )
    checks["PCA_mean_fitted_on_train600"] = bool(
        np.array_equal(state["pca_mean"], embeddings.mean(axis=0))
    )
    review["reviewer_check_correction"] = {
        "initial_check": "Compared float64 PCA mean against numpy default float32 accumulator mean.",
        "correct_check": "Replay the original explicit float64 training cast before mean computation.",
        "float32_default_mean_max_difference": float(
            np.abs(state["pca_mean"] - original_embeddings.mean(axis=0)).max()
        ),
        "float64_training_mean_max_difference": float(
            np.abs(state["pca_mean"] - embeddings.mean(axis=0)).max()
        ),
        "model_or_fitting_changes": 0,
    }
    pca = PCA(n_components=16, svd_solver="full")
    pca.mean_ = state["pca_mean"]
    pca.components_ = state["pca_components"]
    pca.n_components_ = 16
    pca.n_features_in_ = 384
    # Whitening is disabled. This attribute is inspected for dtype dispatch only.
    pca.explained_variance_ = np.ones(16)
    scaler = StandardScaler()
    scaler.mean_ = state["scaler_mean"]
    scaler.scale_ = state["scaler_scale"]
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = 16
    scaler.n_samples_seen_ = np.int64(600)
    public_phi = np.column_stack((np.ones(600), scaler.transform(pca.transform(embeddings))))
    restored_phi = predictor.features(original_embeddings, state)
    checks["restored_prediction_feature_operator_matches_original_public_sklearn_transform"] = bool(
        np.array_equal(public_phi, restored_phi)
    )
    checks["restored_prediction_features_recover_frozen_design"] = bool(
        np.max(np.abs(public_phi - state["phi_h"])) <= 1e-13
    )
    review["original_prediction_operator_feature_recovery_tolerance"] = 1e-13
    review["original_prediction_operator_feature_recovery_max_error"] = float(
        np.abs(public_phi - state["phi_h"]).max()
    )
    source_hash = hashlib.sha256((code / "predict_final_inputs.py").read_bytes()).hexdigest()
    review["reviewed_prediction_source_sha256"] = source_hash
    pools = []
    corrections = {}
    for path in sorted(root.glob("*ORIGINAL_OPERATOR_PREDICTION_RECEIPT.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        pool = receipt["pool"]
        pools.append(pool)
        checks[pool + "_formal_operator_prediction_follows_weight_freeze"] = bool(
            datetime.fromisoformat(receipt["utc"]) > datetime.fromisoformat(freeze["utc"])
            and receipt["model_sha256"] == freeze["model_sha256"]
        )
        checks[pool + "_formal_operator_prediction_has_no_reward_access"] = (
            receipt["reward_values_read"] == 0 and receipt["answer_values_read"] == 0
            and receipt["new_prediction_head_fits"] == 0
        )
        artifact = root / f"{pool.upper()}_INPUT_ONLY_PREDICTIONS_ORIGINAL_OPERATOR_FROZEN.npz"
        checks[pool + "_formal_prediction_hash_matches_receipt"] = (
            hashlib.sha256(artifact.read_bytes()).hexdigest() == receipt["prediction_sha256"]
        )
        checks[pool + "_formal_prediction_source_matches_receipt"] = source_hash == receipt["source_sha256"]
        with np.load(artifact, allow_pickle=False) as source:
            scores = source["scores"].copy()
            probabilities = source["probabilities"].copy()
            raw_probabilities = source["raw_exact_probabilities"].copy()
        support = scores >= scores.max(axis=2, keepdims=True) - 1e-8
        raw_support = scores == scores.max(axis=2, keepdims=True)
        checks[pool + "_formal_primary_and_raw_decoders_reproduce"] = bool(
            np.array_equal(probabilities, support / support.sum(axis=2, keepdims=True))
            and np.array_equal(raw_probabilities, raw_support / raw_support.sum(axis=2, keepdims=True))
        )
        corrections[pool] = {
            "original_prediction_operator": "Original sklearn PCA transform, StandardScaler transform, per-head phi @ Fortran-order W.",
            "previous_implementation": receipt["previous_mathematical_equivalent_input_implementation"],
            "training_model_changes": 0,
            "correction_selected_using_evaluation_rewards": False,
            "formal_prediction_sha256": receipt["prediction_sha256"],
        }
    review["checked_formal_prediction_pools"] = pools
    review["formal_numeric_operator_correction"] = corrections
    review["checks"] = {key: bool(value) for key, value in checks.items()}
    review["status"] = "PASS" if all(checks.values()) else "FAIL"
    limitation = (
        "The initial algebraically equivalent einsum prediction artifacts remain archived; the formal path "
        "restores the frozen original numeric operator because raw-exact argmax can depend on last-bit differences."
    )
    if limitation not in review["limits"]:
        review["limits"].append(limitation)
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(review, ensure_ascii=False, indent=2))
    if review["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
