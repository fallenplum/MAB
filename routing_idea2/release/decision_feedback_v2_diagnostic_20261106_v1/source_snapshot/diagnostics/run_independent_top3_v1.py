"""Post-validation exploratory IndependentTop3; CPU only, never fresh outcomes.

Fit six independent correctness heads from existing train-member labels. Freeze
model and validation/fresh choices before reading the existing validation matrix.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(r"E:\语言模型组合路由")
PRIVATE = ROOT / "artifacts/private/routing_redesign_20261104"
SEEDS = (42, 137, 2024)
CACHE_RECEIPT_SHA = "e961aa23aa041a5e249a976785e6d8b6a09e0b9b33c88faf86171b1f3429e466"
PINS = {
    "src/budget_router_v2/full_feedback.py": "58b20505182149eef2537a42867c04ba98dfc6959f527fe62c1b8c5ff3f39dc2",
    "src/budget_router_v2/scoring.py": "6fc1d2f3cc81b3e954350143abd1071d6556e1cfe4f321d9333f2016baf2150d",
    "scripts/v2/train_member_score_cache.py": "2170b4f283fa3a66e34bc428f4e9addc17b9191222ec880b73fd0f4d7de5215e",
    "scripts/v2/scientific_matrix_common.py": "14c24afc9da200fd5c145e1f7aee3dbd16dcdfcda48b355882f0129b51b9b17f",
}
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/v2")]
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np
from budget_router.learner import OriginalLabel, SharedRidgeRegressor
from budget_router_v2 import full_feedback as ff
from train_member_score_cache import load_member_score_cache


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ref(path):
    return {"path": str(Path(path).resolve(strict=True)), "sha256": sha(path)}


def put(path, record):
    with path.open("x", encoding="utf8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def contexts(inputs, transform):
    values = (inputs["embeddings"] - transform["pca_mean"]) @ transform["pca_components"].T
    values = (values - transform["scale_mean"]) / transform["scale_scale"]
    return np.column_stack((np.ones(len(values)), values))


def choice_arrays(inputs, transform, theta):
    scores = contexts(inputs, transform) @ theta.T
    require(scores.shape == (len(inputs["question_ids"]), 6) and np.isfinite(scores).all(), "Malformed six-head predictions")
    # Stable descending quality, then original member index ascending. Never clip.
    members = np.sort(np.argsort(-scores, axis=1, kind="stable")[:, :3], axis=1)
    teams = list(itertools.combinations(range(6), 3))
    indexes = np.array([teams.index(tuple(row)) for row in members], dtype=np.int64)
    return {name: inputs[name] for name in ("question_ids", "canonical_hashes", "task_labels", "subtask_labels")} | {
        "predicted_member_scores": scores, "selected_members": members, "selected_team_index": indexes}


def freeze_choices(output, split, inputs, transform, theta, model_ref, identity):
    output.mkdir(exist_ok=False)
    arrays = choice_arrays(inputs, transform, theta)
    path = output / "predictions.npz"
    np.savez_compressed(path, **arrays)
    receipt = {"schema": "routing_v2_independent_member_top3_predictions_v1", "method": "independent_top3",
        "status": "FROZEN_BEFORE_BASELINE_VALIDATION_LOOKUP_AND_ANY_FRESH_OUTCOME_ACCESS",
        "split": split, "exploratory_post_validation_design": True, "identity": identity,
        "model_receipt": model_ref, "predictions": ref(path), "questions": len(inputs["question_ids"]),
        "tie_rule": "descending unrounded member ridge score, member index ascending; selected K3 sorted for team index",
        "member_predictions_are_aggregate_reward_model": False,
        "fresh_outputs_or_gold_semantically_opened": 0, "model_calls": 0}
    put(output / "PREDICTIONS_FROZEN.json", receipt)
    return arrays, output / "PREDICTIONS_FROZEN.json"


def quality_summary(selected, outcomes, inputs):
    tasks, subtasks = inputs["task_labels"], inputs["subtask_labels"]
    oracle = outcomes["aggregate_scores"].max(axis=2)
    regret = oracle - selected
    groups = sorted(set(subtasks[tasks == "bbh"].tolist()))
    return {"task_macro_quality": ff._macro(selected, tasks),
        "per_task_quality": {t: float(selected[:, tasks == t].mean()) for t in ff.TASKS},
        "bbh_subtask_macro_quality": float(np.mean([selected[:, subtasks == s].mean() for s in groups])),
        "per_generation": [{"generation_seed": seed, "task_macro_quality": ff._macro(selected[g], tasks),
            "top1_regret": ff._macro(regret[g], tasks),
            "per_task_quality": {t: float(selected[g, tasks == t].mean()) for t in ff.TASKS}} for g, seed in enumerate(SEEDS)],
        "per_bbh_subtask": {s: {"questions": int(np.sum(subtasks == s)),
            "quality": float(selected[:, subtasks == s].mean())} for s in groups},
        "top1_regret_true_existing20team_oracle": ff._macro(regret, tasks),
        "question_support": len(tasks), "question_generation_cells": int(selected.size)}


def run(output):
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    output = output.resolve()
    require(output.is_relative_to(PRIVATE.resolve()) and not output.exists(), "New private independent baseline directory required")
    for relative, expected in PINS.items():
        require(sha(ROOT / relative) == expected, "Protected source changed: " + relative)
    original = PRIVATE / "ff_v1"
    original_files = [original / "model/MODEL_FROZEN.json", original / "model/model.npz",
                      original / "pred/PREDICTIONS_FROZEN.json", original / "pred/predictions.npz"]
    original_hashes = {str(p): sha(p) for p in original_files}
    bundle_path = original / "BUNDLE_TRAINVAL.json"
    bundle, base, identity = ff.load_bundle(bundle_path)
    inputs = {split: ff.read_inputs(bundle, base, split)[0]
              for split in ("selector_train", "selector_validation", "fresh_dev_test")}
    train, validation, fresh = (inputs[k] for k in ("selector_train", "selector_validation", "fresh_dev_test"))
    require(not (set(train["canonical_hashes"]) & set(validation["canonical_hashes"])) and
            not (set(fresh["canonical_hashes"]) & (set(train["canonical_hashes"]) | set(validation["canonical_hashes"]))),
            "Input canonical partitions overlap")
    original_model_receipt = read(original_files[0])
    require(original_model_receipt["identity"] == identity and original_model_receipt["training"]["feature_fit_scope"] == "selector_train_only"
            and original_model_receipt["training"]["pca_components"] == 16,
            "Original train-only transform identity differs")
    require(sha(original_files[1]) == original_model_receipt["model"]["sha256"], "Original model artifact changed")
    for source in original_model_receipt["algorithm_sources"]:
        ff.verified(source, original_files[0].parent)
    with np.load(original_files[1], allow_pickle=False) as data:
        transform = {key: data[key] for key in ("pca_components", "pca_mean", "scale_mean", "scale_scale")}
    require(transform["pca_components"].shape == (16, 384) and np.all(transform["scale_scale"] > 0), "Original transform malformed")
    phi = contexts(train, transform)
    require(phi.shape == (600, 17), "Same17-dimensional input-only context required")
    cache_path = PRIVATE / "stage2_generation/member_score_cache_actual_20261105_v3/MEMBER_SCORE_CACHE_RECEIPT_v1.json"
    require(sha(cache_path) == CACHE_RECEIPT_SHA, "Original actual train member cache receipt changed")
    cache = load_member_score_cache(cache_path, CACHE_RECEIPT_SHA)
    print(json.dumps({"stage": "TRAIN_MEMBER_CACHE_VERIFIED", "existing_labels": len(cache.metadata), "model_calls": 0}), flush=True)
    cache_receipt = read(cache_path)
    with np.load(ff.verified(cache_receipt["score_cache"], cache_path.parent), allow_pickle=False) as data:
        aliases = data["model_aliases"].tolist()
    pool = ff.read_json(ff.verified(identity["member_pool"], base))["models"]
    require(aliases == [m["alias"] for m in pool], "Member column order differs from original pool")
    qmap = {str(q): i for i, q in enumerate(train["question_ids"])}
    canonical = {str(q): str(c) for q, c in zip(train["question_ids"], train["canonical_hashes"])}
    executed = frozenset(cache.metadata)
    arrived = frozenset(canonical.values())
    targets = np.full((3, 600, 6), np.nan)
    heads = [SharedRidgeRegressor(17, ridge=1., noise_variance=1.) for _ in range(6)]
    label_counter = 0
    for execution_id, item in sorted(cache.metadata.items()):
        qid, member = item["question_id"], item["member_index"]
        require(qid in qmap and item["canonical_hash"] == canonical[qid], "Train member label escaped600 inputs")
        generation = SEEDS.index(item["member_generation_seed"])
        row = qmap[qid]
        require(not np.isfinite(targets[generation, row, member]), "Duplicate train label slot")
        value = cache.query(execution_id, executed_execution_ids=executed,
             arrived_gold_canonical_hashes=arrived, reference_version=cache.reference_version)
        targets[generation, row, member] = value
        heads[member].add_label(OriginalLabel(execution_id, canonical[qid], phi[row], value, 0), now=0)
        label_counter += 1
    require(label_counter == 10800 and np.isfinite(targets).all() and all(h.label_count == 1800 and h.question_count == 600 for h in heads),
            "Each independent head requires1800 genuine member labels")
    theta = np.stack([h.theta for h in heads])
    expected = np.linalg.solve(np.eye(17) + 3 * phi.T @ phi, phi.T @ targets.sum(axis=0)).T
    require(np.allclose(theta, expected, atol=1e-9, rtol=1e-9), "Original ridge update differs from exact sufficient statistics")
    fit_cpu, fit_wall = time.process_time() - started_cpu, time.perf_counter() - started_wall
    output.mkdir()
    fit_path = output / "model.npz"
    np.savez_compressed(fit_path, theta=theta, **transform, train_question_ids=train["question_ids"],
        train_canonical_hashes=train["canonical_hashes"], generation_seeds=np.array(SEEDS), member_aliases=np.array(aliases))
    model_receipt = {"schema": "routing_v2_independent_member_top3_model_v1", "status": "FROZEN_NO_BASELINE_VALIDATION_OUTCOME_LOOKUP_YET",
        "exploratory_post_validation_design": True, "identity": identity, "model": ref(fit_path),
        "training_label_source": ref(cache_path), "training_target": "individual member exact correctness, never aggregate correctness",
        "training_questions": 600, "training_member_labels": 10800, "labels_per_head": 1800,
        "independent_heads": 6, "ridge": 1., "noise_variance": 1., "feature_dimension": 17,
        "intercept_regularized": True, "PCA_scaler_reused_from_original_train_only_model": ref(original_files[0]),
        "feature_fit_or_hyperparameter_search_performed": False,
        "source": ref(Path(__file__)), "protected_sources": PINS,
        "original_ridge_equals_closed_form": True, "fit_and_source_verification_cpu_seconds": fit_cpu,
        "fit_and_source_verification_wall_seconds": fit_wall, "new_model_calls": 0, "new_answer_scoring_operations": 0}
    model_receipt_path = output / "MODEL_FROZEN.json"
    put(model_receipt_path, model_receipt)
    val_pred, val_freeze = freeze_choices(output / "validation", "selector_validation", validation, transform, theta, ref(model_receipt_path), identity)
    fresh_pred, fresh_freeze = freeze_choices(output / "fresh", "fresh_dev_test", fresh, transform, theta, ref(model_receipt_path), identity)
    freeze_barrier = {"schema": "routing_v2_independent_top3_outcome_access_barrier_v1",
        "status": "SIX_HEAD_MODEL_AND_VAL_FRESH_CHOICES_FROZEN_BEFORE_FIRST_VALIDATION_MATRIX_LOOKUP",
        "created_unix": time.time(), "model": ref(model_receipt_path),
        "validation_predictions": ref(val_freeze), "fresh_predictions": ref(fresh_freeze),
        "validation_aggregate_outcomes_read_by_this_helper": 0, "fresh_outputs_or_gold_read": 0,
        "posthoc_design_after_original_validation_seen": True}
    put(output / "OUTCOME_ACCESS_BARRIER.json", freeze_barrier)
    print(json.dumps({"stage": "MODEL_VAL_AND_FRESH_CHOICES_FROZEN", "model_receipt": str(model_receipt_path),
        "fresh_prediction_receipt": str(fresh_freeze), "new_model_calls": 0}), flush=True)
    # Only after the immutable barrier: existing real validation aggregate matrix.
    score_cpu, score_wall = time.process_time(), time.perf_counter()
    outcomes, outcome_path = ff.read_outcomes(bundle, base, "selector_validation", validation)
    original_val_receipt_path = original / "model/validation_predictions/PREDICTIONS_FROZEN.json"
    original_val_receipt = read(original_val_receipt_path)
    require(original_val_receipt["status"] == "PREDICTIONS_FROZEN_BEFORE_OUTCOME_ACCESS" and original_val_receipt["split"] == "selector_validation"
            and original_val_receipt["identity"] == identity, "Original frozen validation predictions differ")
    with np.load(ff.verified(original_val_receipt["predictions"], original_val_receipt_path.parent), allow_pickle=False) as data:
        require(all(np.array_equal(data[k], validation[k]) for k in ("question_ids", "canonical_hashes", "task_labels", "subtask_labels")),
                "Original validation prediction cohort differs")
        choices = {name: np.argmax(data[name], axis=1) for name in ff.METHODS}
    choices["independent_top3"] = val_pred["selected_team_index"]
    g = np.arange(3)[:, None]
    q = np.arange(200)[None, :]
    selected = {name: outcomes["aggregate_scores"][g, q, indexes[None, :]] for name, indexes in choices.items()}
    comparisons = {name: ff._cluster_difference(selected["independent_top3"] - selected[name],
        validation["task_labels"], validation["subtask_labels"], np.array(SEEDS), 2000, 20261104) for name in ff.METHODS}
    report = {"schema": "routing_v2_independent_top3_exploratory_validation_v1", "status": "EXPLORATORY_VALIDATION_ONLY",
        "posthoc_design_after_original_validation_results_seen": True,
        "not_an_original_preregistered_gate_result": True, "fresh_outcomes_evaluated": False,
        "identity": identity, "validation_questions": 200, "generation_seeds": list(SEEDS),
        "methods": {name: quality_summary(values, outcomes, validation) for name, values in selected.items()},
        "independent_top3_minus_original_method": comparisons,
        "comparison_scope": "Exploratory descriptive paired intervals, no multiplicity adjustment or confirmatory claim; question clusters retain all3 generations, task/BBHsubtask strata, original2000 bootstrap and seed20261104",
        "model_freeze": ref(model_receipt_path), "validation_predictions": ref(val_freeze), "fresh_predictions": ref(fresh_freeze),
        "outcome_access_barrier": ref(output / "OUTCOME_ACCESS_BARRIER.json"), "actual_validation_matrix": ref(outcome_path),
        "original_validation_prediction_receipt": ref(original_val_receipt_path),
        "model_target_is_member_accuracy_but_reported_combination_quality_uses_executed_aggregator": True,
        "member_prediction_accuracy_diagnostic": "NOT_EVALUATED_NO_VALIDATION_MEMBER_LABELS_BUILT",
        "new_model_calls": 0, "new_answer_exact_scoring_operations": 0,
        "existing_training_member_labels_reused": 10800, "new_gold_questions_purchased": 0,
        "statistical_lookup_cpu_seconds": time.process_time() - score_cpu,
        "statistical_lookup_wall_seconds": time.perf_counter() - score_wall,
        "total_controller_cpu_seconds": time.process_time() - started_cpu,
        "total_controller_wall_seconds": time.perf_counter() - started_wall,
        "hardware_scope": "Reused S2 validation generation42 includes400 historical local Qwen/Coder members; others A6000. No new generation or hardware claim."}
    require(original_hashes == {str(p): sha(p) for p in original_files}, "Original four frozen artifacts changed")
    require(sha(cache_path) == CACHE_RECEIPT_SHA and all(sha(ROOT / name) == value for name, value in PINS.items()), "Protected sources/cache changed")
    report["original_four_frozen_files_pre_post_sha_identical"] = True
    put(output / "VALIDATION_EXPLORATORY_REPORT.json", report)
    np.savez_compressed(output / "validation_selected_existing_outcomes.npz", question_ids=validation["question_ids"],
        generation_seeds=np.array(SEEDS), **selected)
    summary = {"report": ref(output / "VALIDATION_EXPLORATORY_REPORT.json"),
        "fresh_frozen_choices": ref(fresh_freeze), "independent_top3_validation_quality": report["methods"]["independent_top3"]["task_macro_quality"],
        "independent_top3_validation_regret": report["methods"]["independent_top3"]["top1_regret_true_existing20team_oracle"],
        "comparisons": {name: {key: value[key] for key in ("task_macro_difference", "question_cluster_ci", "positive_generation_replicates")}
                         for name, value in comparisons.items()},
        "new_answer_scoring": 0, "new_model_calls": 0, "fresh_outcomes_read": 0}
    put(output / "DELIVERY.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
