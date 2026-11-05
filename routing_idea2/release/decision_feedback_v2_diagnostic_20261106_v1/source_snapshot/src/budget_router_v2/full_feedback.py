"""Independent v2 full-feedback selector analysis with frozen-prediction scoring.

Formal inputs are outcome-free features and separately hash-bound multi-generation
score/cost matrices. All PCA/scaling and fixed baselines use selector_train only.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from budget_router.features import SubsetFeatureMap
from budget_router.learner import OriginalLabel, SharedRidgeRegressor
from budget_router.ranking_evaluation import ranking_metrics
from budget_router_v2.scoring import BBH_TASKS, SCORER_VERSION

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "artifacts/private/routing_redesign_20261104"
TASKS = ("gsm8k", "bbh")
COUNTS = {"selector_train": 600, "selector_validation": 200, "fresh_dev_test": 400}
METHODS = ("selector", "global_fixed", "task_fixed", "subtask_fixed")
INPUT_KEYS = {"question_ids", "canonical_hashes", "task_labels", "subtask_labels", "embeddings"}
COST_KEYS = tuple(stem + "_" + suffix for stem in ("member", "aggregate") for suffix in
                  ("execution_counts", "prompt_tokens", "completion_tokens", "wall_time_s"))
OUTCOME_KEYS = {"question_ids", "canonical_hashes", "generation_seeds", "aggregate_scores", *COST_KEYS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def reference(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha_file(path)}


def verified(record, base):
    path = Path(record["path"])
    path = (Path(base) / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT.resolve()), "Input asset escapes project scope")
    require(path.is_file() and sha_file(path) == record["sha256"], "Immutable input hash mismatch")
    return path


def new_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), "Outputs must stay in the private v2 development scope")
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_bundle(path):
    path = Path(path).resolve(strict=True)
    bundle = read_json(path)
    require(bundle["schema"] == "routing_full_feedback_bundle_v2", "Unexpected input bundle schema")
    require(bundle["N"] == 6 and bundle["K"] == 3 and bundle["generation_seeds"] == [42, 137, 2024],
            "Formal stage2 requires original fixed K3 and three locked generation seeds")
    require(bundle["scorer"] == {"version": SCORER_VERSION, "no_model_calls": True}, "Pinned local scorer required")
    pool_path = verified(bundle["member_pool"], path.parent)
    pool = read_json(pool_path)["models"]
    require(len(pool) == 6 and [row["index"] for row in pool] == list(range(6)) and
            len({row["model_id"] for row in pool}) == 6, "Pinned six-member column order required")
    operator = bundle["aggregator_operator"]
    require(operator["model_id"] == "Qwen/Qwen3-8B", "Stage2 requires the newly frozen v2 aggregator")
    freeze_path = verified(operator["freeze_evidence"], path.parent)
    freeze = read_json(freeze_path)
    require(freeze["status"] == "FROZEN_AFTER_AGGCAL100" and
            freeze["aggregator_fingerprint"] == operator["fingerprint"] and
            freeze["generation_seeds"] == bundle["generation_seeds"], "Aggregator freeze evidence is incomplete")
    encoder = bundle["feature_encoder"]
    require(encoder["model_id"] == "sentence-transformers/all-MiniLM-L6-v2" and
            encoder["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "Frozen encoder identity differs")
    identity = {name: bundle[name] for name in ("N", "K", "generation_seeds", "scorer", "aggregator_operator", "feature_encoder", "member_pool")}
    return bundle, path.parent, identity


def read_inputs(bundle, base, split):
    path = verified(bundle["inputs"][split], base)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == INPUT_KEYS, "Input-only NPZ must contain only the declared feature/identity arrays")
        data = {name: archive[name] for name in INPUT_KEYS}
    n = COUNTS[split]
    for name in ("question_ids", "canonical_hashes", "task_labels", "subtask_labels"):
        require(data[name].shape == (n,) and data[name].dtype.kind in "US", "Input identity arrays are malformed")
    require(len(set(data["question_ids"].tolist())) == n and len(set(data["canonical_hashes"].tolist())) == n,
            "Duplicate question or canonical identity")
    require(all(len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                for value in data["canonical_hashes"].tolist()), "Canonical identities must be SHA256 hashes")
    require({task: int(np.sum(data["task_labels"] == task)) for task in TASKS} ==
            {task: n // 2 for task in TASKS}, "Original balanced task counts required")
    require(data["embeddings"].shape == (n, 384) and np.isfinite(data["embeddings"]).all(), "Frozen MiniLM embeddings required")
    require(np.allclose(np.linalg.norm(data["embeddings"], axis=1), 1.0, atol=1e-5), "L2-normalized frozen embeddings required")
    for task, subtask in zip(data["task_labels"], data["subtask_labels"]):
        require((task == "gsm8k" and subtask == "") or (task == "bbh" and subtask in BBH_TASKS), "Task/subtask identity invalid")
    if split == "fresh_dev_test":
        selection_path = verified(bundle["fresh_selection"], base)
        selection = read_json(selection_path)
        require(selection["status"] == "SELECTED_INPUTS_ONLY_NOT_EVALUATED", "Frozen score-blind fresh cohort required")
        selected = {row["question_id"]: row for row in selection["records"]}
        require(set(selected) == set(data["question_ids"].tolist()), "Fresh feature cohort differs from frozen selection")
        for qid, canonical, task, subtask in zip(data["question_ids"], data["canonical_hashes"], data["task_labels"], data["subtask_labels"]):
            row = selected[qid]
            require((row["canonical_question_sha256"], row["task"], row.get("subtask") or "") ==
                    (canonical, task, subtask), "Fresh selection metadata differs")
        counts = {name: int(np.sum(data["subtask_labels"] == name)) for name in BBH_TASKS}
        require(all(value in (7, 8) for value in counts.values()), "Fresh BBH must cover all 27 subtasks at seven/eight questions")
    return data, path


def read_outcomes(bundle, base, split, inputs):
    path = verified(bundle["outcomes"][split], base)
    receipt_path = verified(bundle["matrix_receipts"][split], base)
    receipt = read_json(receipt_path)
    require(receipt["schema"] == "routing_v2_real_full_matrix_receipt_v1" and
            receipt["status"] == "REAL_FULL_INFORMATION_READY" and receipt["software_test_only"] is False,
            "Real full-information matrix receipt required")
    flags = ("all_combinations_executed", "real_frozen_llm_aggregation",
             "aggregate_scores_from_executed_aggregator", "failure_costs_included")
    require(all(receipt.get(flag) is True for flag in flags), "Unexecuted or surrogate combination rewards rejected")
    require(receipt["split"] == split and receipt["questions"] == len(inputs["question_ids"]) and
            receipt["generation_seeds"] == bundle["generation_seeds"] and
            receipt["aggregator_fingerprint"] == bundle["aggregator_operator"]["fingerprint"],
            "Matrix receipt operator, split or generation lineage differs")
    require(receipt["combinations"] == [list(team) for team in SubsetFeatureMap(6, 3).subsets] and
            receipt["member_pool_sha256"] == bundle["member_pool"]["sha256"], "Matrix member/combination column order differs")
    require(verified(receipt["outcomes"], receipt_path.parent) == path and
            receipt["outcomes"]["sha256"] == bundle["outcomes"][split]["sha256"] and
            receipt["inputs"]["sha256"] == bundle["inputs"][split]["sha256"], "Receipt asset binding differs")
    require(receipt["scorer_version"] == SCORER_VERSION and
            receipt["scorer_source_sha256"] == sha_file(ROOT / "src/budget_router_v2/scoring.py"),
            "Matrix strict scorer version/source differs")
    require(receipt["sources"], "Actual member/aggregate output source receipts required")
    for source in receipt["sources"]:
        verified(source, receipt_path.parent)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == OUTCOME_KEYS, "Outcome NPZ fields differ from the declared contract")
        data = {name: archive[name] for name in OUTCOME_KEYS}
    seeds = np.asarray(bundle["generation_seeds"])
    n = len(inputs["question_ids"])
    require(data["generation_seeds"].dtype.kind in "iu", "Generation seeds must be integers")
    require(np.array_equal(data["question_ids"], inputs["question_ids"]) and
            np.array_equal(data["canonical_hashes"], inputs["canonical_hashes"]) and
            np.array_equal(data["generation_seeds"], seeds), "Outcome identity/order or generation seeds differ")
    scores = data["aggregate_scores"]
    require(scores.shape == (len(seeds), n, 20) and np.isfinite(scores).all() and
            np.isin(scores, (0.0, 1.0)).all(), "Every real exact-score cell is required")
    for stem, width in (("member", 6), ("aggregate", 20)):
        for suffix in ("execution_counts", "prompt_tokens", "completion_tokens", "wall_time_s"):
            values = data[stem + "_" + suffix]
            require(values.shape == (len(seeds), n, width) and np.isfinite(values).all() and np.all(values >= 0),
                    "Actual source costs must be complete, finite and nonnegative")
            if suffix != "wall_time_s":
                require(values.dtype.kind in "iu", "Execution and token counts must be integers")
            if suffix == "execution_counts":
                require(np.all(values >= 1), "Known failed attempts must remain in source execution counts")
    return data, path


def fit_selector(train, outcomes, components=16):
    """Fit only training inputs and add every original generation/combination score once."""
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    pca = PCA(n_components=components, svd_solver="full")
    reduced = pca.fit_transform(train["embeddings"])
    scaler = StandardScaler().fit(reduced)
    phi = np.column_stack((np.ones(len(reduced)), scaler.transform(reduced)))
    mapper = SubsetFeatureMap(6, 3)
    learner = SharedRidgeRegressor(phi.shape[1] * mapper.basis_dim, ridge=1.0, noise_variance=1.0)
    for row, canonical in enumerate(train["canonical_hashes"]):
        features = mapper.transform(phi[row])
        for generation, seed in enumerate(outcomes["generation_seeds"]):
            for combo in range(20):
                label = OriginalLabel(f"{canonical}:gen:{int(seed)}:combo:{combo}", str(canonical), features[combo],
                                      float(outcomes["aggregate_scores"][generation, row, combo]), 0)
                learner.add_label(label, now=0)
    require(learner.label_count == outcomes["aggregate_scores"].size, "Original observation count differs")
    scores = outcomes["aggregate_scores"]
    task_means = np.stack([scores[:, train["task_labels"] == task].mean(axis=(0, 1)) for task in TASKS])
    subtask_names = sorted(set(train["subtask_labels"][train["task_labels"] == "bbh"].tolist()))
    subtask_means = np.stack([scores[:, train["subtask_labels"] == name].mean(axis=(0, 1)) for name in subtask_names])
    model = {"theta": learner.theta, "pca_components": pca.components_, "pca_mean": pca.mean_,
             "scale_mean": scaler.mean_, "scale_scale": scaler.scale_, "global_means": task_means.mean(axis=0),
             "task_means": task_means, "subtask_names": np.asarray(subtask_names), "subtask_means": subtask_means,
             "used_question_ids": train["question_ids"], "used_canonical_hashes": train["canonical_hashes"],
             "generation_seeds": outcomes["generation_seeds"]}
    metadata = {"training_questions": len(train["question_ids"]), "original_aggregate_observations": learner.label_count,
                "gold_questions": len(train["question_ids"]), "ridge": 1.0, "working_noise_variance": 1.0,
                "pca_components": components, "feature_fit_scope": "selector_train_only",
                "feature_order": "frozen embedding, train PCA, train component standardization, leading constant",
                "subtask_train_support": {name: int(np.sum(train["subtask_labels"] == name)) for name in subtask_names},
                "update_cpu_s": time.process_time() - started_cpu, "update_wall_s": time.perf_counter() - started_wall}
    return model, metadata


def predictions(model, inputs):
    reduced = (inputs["embeddings"] - model["pca_mean"]) @ model["pca_components"].T
    reduced = (reduced - model["scale_mean"]) / model["scale_scale"]
    phi = np.column_stack((np.ones(len(reduced)), reduced))
    mapper = SubsetFeatureMap(6, 3)
    selector = np.stack([mapper.transform(row) @ model["theta"] for row in phi])
    global_predictions = np.broadcast_to(model["global_means"], selector.shape).copy()
    task_predictions = np.stack([model["task_means"][TASKS.index(task)] for task in inputs["task_labels"]])
    subtask_predictions = task_predictions.copy()
    names = {str(name): i for i, name in enumerate(model["subtask_names"])}
    for row, (task, subtask) in enumerate(zip(inputs["task_labels"], inputs["subtask_labels"])):
        if task == "bbh" and subtask in names:
            subtask_predictions[row] = model["subtask_means"][names[subtask]]
    return {"selector": selector, "global_fixed": global_predictions,
            "task_fixed": task_predictions, "subtask_fixed": subtask_predictions}


def _macro(values, tasks):
    return float(np.mean([np.mean(values[..., tasks == task], axis=-1) for task in TASKS if np.any(tasks == task)]))


def _cluster_difference(delta, tasks, subtasks, seeds, replicates, bootstrap_seed):
    """Resample questions within task/subtask. Every generation stays in its question block."""
    rng = np.random.default_rng(bootstrap_seed)
    per_task_draws = []
    strata_support = {}
    for task in TASKS:
        mask = tasks == task
        if not np.any(mask):
            continue
        groups = sorted(set(subtasks[mask].tolist())) if task == "bbh" else [""]
        task_draws = np.zeros((len(seeds), replicates))
        for group in groups:
            indices = np.flatnonzero(mask & ((subtasks == group) if task == "bbh" else True))
            strata_support[task + ("/" + group if group else "")] = len(indices)
            counts = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=replicates)
            task_draws += (delta[:, indices] @ counts.T) / int(mask.sum())
        per_task_draws.append(task_draws)
    draws = np.mean(per_task_draws, axis=0)
    per_seed = np.array([_macro(row, tasks) for row in delta])
    pooled_draws = draws.mean(axis=0)
    return {"task_macro_difference": float(per_seed.mean()),
            "strata_support": strata_support, "adequate_support": all(n >= 2 for n in strata_support.values()),
            "question_cluster_ci": [float(v) for v in np.quantile(pooled_draws, [.025, .975])],
            "question_support": len(tasks), "per_task_support": {task: int(np.sum(tasks == task)) for task in TASKS},
            "per_generation": [{"generation_seed": int(seed), "difference": float(value),
                                "question_cluster_ci": [float(v) for v in np.quantile(draws[i], [.025, .975])]} for i, (seed, value) in enumerate(zip(seeds, per_seed))],
            "positive_generation_replicates": int(np.sum(per_seed > 0)), "confidence": .95,
            "bootstrap_replicates": replicates, "bootstrap_unit": "question; all generations grouped; task/subtask stratified",
            "scope": "Conditional on fitted selector, frozen aggregator and observed generation replicates; no training-set retraining bootstrap"}


def evaluate_predictions(inputs, outcomes, predicted, *, replicates=2000, bootstrap_seed=20261104):
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    require(replicates >= 2, "Bootstrap requires at least two replicates")
    scores = outcomes["aggregate_scores"]
    seeds = outcomes["generation_seeds"]
    tasks, subtasks = inputs["task_labels"], inputs["subtask_labels"]
    quality, methods, costs = {}, {}, {}
    member_incidence = np.asarray([np.isin(range(6), team) for team in SubsetFeatureMap(6, 3).subsets])
    row_indices = np.arange(len(tasks))[None, :]
    generation_indices = np.arange(len(seeds))[:, None]
    for name in METHODS:
        pred = predicted[name]
        require(pred.shape == scores.shape[1:] and np.isfinite(pred).all(), "Frozen prediction shape/value invalid")
        chosen = np.argmax(pred, axis=1)
        selected = scores[generation_indices, row_indices, chosen[None, :]]
        quality[name] = selected
        per_task = {task: float(selected[:, tasks == task].mean()) for task in TASKS if np.any(tasks == task)}
        bbh_groups = sorted(set(subtasks[tasks == "bbh"].tolist()))
        methods[name] = {"task_macro_quality": float(np.mean(list(per_task.values()))), "per_task_quality": per_task,
                         "bbh_subtask_macro_quality": float(np.mean([selected[:, subtasks == group].mean() for group in bbh_groups])),
                         "per_bbh_subtask_quality": {group: {"questions": int(np.sum(subtasks == group)),
                            "quality": float(selected[:, subtasks == group].mean()),
                            "per_generation": [{"generation_seed": int(seed), "quality": float(selected[i, subtasks == group].mean())}
                                               for i, seed in enumerate(seeds)]} for group in bbh_groups},
                         "per_generation": [{"generation_seed": int(seed), "task_macro_quality": _macro(selected[i], tasks),
                                             "bbh_subtask_macro_quality": float(np.mean([selected[i, subtasks == group].mean() for group in bbh_groups])),
                                             "per_task_quality": {task: float(selected[i, tasks == task].mean()) for task in TASKS if np.any(tasks == task)}} for i, seed in enumerate(seeds)]}
        total = {}
        per_generation_costs = [{"generation_seed": int(seed)} for seed in seeds]
        for suffix in ("execution_counts", "prompt_tokens", "completion_tokens", "wall_time_s"):
            aggregate = outcomes["aggregate_" + suffix][generation_indices, row_indices, chosen[None, :]]
            members = (outcomes["member_" + suffix] * member_incidence[chosen][None, :, :]).sum(axis=2)
            values = aggregate + members
            total[suffix] = float(values.sum()) if suffix == "wall_time_s" else int(values.sum())
            for i in range(len(seeds)):
                per_generation_costs[i][suffix] = float(values[i].sum()) if suffix == "wall_time_s" else int(values[i].sum())
        total["per_generation"] = per_generation_costs
        costs[name] = total
    comparisons = {}
    for name in METHODS[1:]:
        delta = quality["selector"] - quality[name]
        comparisons[name] = _cluster_difference(delta, tasks, subtasks, seeds, replicates, bootstrap_seed)
    delta = quality["selector"] - quality["subtask_fixed"]
    within_task = {task: _cluster_difference(delta[:, tasks == task], tasks[tasks == task], subtasks[tasks == task], seeds,
                                            replicates, bootstrap_seed) for task in TASKS if np.any(tasks == task)}
    within_subtask = {name: _cluster_difference(delta[:, subtasks == name], tasks[subtasks == name], subtasks[subtasks == name],
                                               seeds, replicates, bootstrap_seed) for name in sorted(set(subtasks[tasks == "bbh"].tolist()))}
    ranking = []
    for generation, seed in enumerate(seeds):
        metrics = [ranking_metrics(predicted["selector"][row], scores[generation, row]) for row in range(len(tasks))]
        available = [item for item in metrics if item["pairwise_accuracy"] is not None]
        ranking.append({"generation_seed": int(seed), "questions": len(tasks), "rankable_questions": len(available),
                        "ordered_pair_support": sum(item["ordered_pair_support"] for item in metrics),
                        "question_mean_pairwise_accuracy": float(np.mean([item["pairwise_accuracy"] for item in available])) if available else None,
                        "question_mean_value_difference_mae": float(np.mean([item["value_difference_mae"] for item in metrics])),
                        "question_mean_top1_regret": float(np.mean([item["top1_regret_lexicographic"] for item in metrics])),
                        "per_task": {task: {"questions": int(np.sum(tasks == task)),
                            "rankable_questions": sum(item["pairwise_accuracy"] is not None for row, item in enumerate(metrics) if tasks[row] == task),
                            "pairwise_accuracy": float(np.mean([item["pairwise_accuracy"] for row, item in enumerate(metrics)
                                if tasks[row] == task and item["pairwise_accuracy"] is not None]))
                                if any(tasks[row] == task and item["pairwise_accuracy"] is not None for row, item in enumerate(metrics)) else None} for task in TASKS if np.any(tasks == task)}})
    physical = {name: float(outcomes[name].sum()) if name.endswith("wall_time_s") else int(outcomes[name].sum()) for name in COST_KEYS}
    return {"methods": methods, "selector_minus_fixed": comparisons, "primary_reference": "subtask_fixed",
            "within_task_increment": within_task, "within_bbh_subtask_increment": within_subtask,
            "ranking_by_generation": ranking, "source_physical_precomputation": physical,
            "comparison_scope": "Prespecified primary selector minus subtask_fixed; other contrasts and within-group intervals are descriptive without multiplicity adjustment",
            "source_precomputation_by_generation": [{"generation_seed": int(seed), **{
                name: float(outcomes[name][i].sum()) if name.endswith("wall_time_s") else int(outcomes[name][i].sum())
                for name in COST_KEYS}} for i, seed in enumerate(seeds)],
            "logical_service_equivalent_by_method": costs,
            "resource_scope": "Physical source generation counted once; each method logical deployment equivalent is separate; all known failed attempts retained. Controller/scoring CPU is separate.",
            "evaluation_cpu_s": time.process_time() - started_cpu, "evaluation_wall_s": time.perf_counter() - started_wall}


def freeze_predictions(output, inputs, predicted, split, identity, model_receipt):
    output = new_output(output)
    path = output / "predictions.npz"
    np.savez_compressed(path, question_ids=inputs["question_ids"], canonical_hashes=inputs["canonical_hashes"],
                        task_labels=inputs["task_labels"], subtask_labels=inputs["subtask_labels"], **predicted)
    receipt = {"schema": "routing_v2_frozen_predictions_v1", "split": split, "identity": identity,
               "model_receipt": model_receipt, "predictions": reference(path),
               "status": "PREDICTIONS_FROZEN_BEFORE_OUTCOME_ACCESS"}
    write_json(output / "PREDICTIONS_FROZEN.json", receipt)
    return output / "PREDICTIONS_FROZEN.json"


def fit_bundle(bundle_path, output):
    require(not Path(output).exists(), "Preserve existing output directory")
    bundle, base, identity = load_bundle(bundle_path)
    train, train_path = read_inputs(bundle, base, "selector_train")
    validation, validation_path = read_inputs(bundle, base, "selector_validation")
    require(not (set(train["canonical_hashes"]) & set(validation["canonical_hashes"])) and
            not (set(train["question_ids"]) & set(validation["question_ids"])), "Train/validation leakage")
    train_outcomes, train_outcomes_path = read_outcomes(bundle, base, "selector_train", train)
    model, metadata = fit_selector(train, train_outcomes)
    model["used_question_ids"] = np.concatenate((train["question_ids"], validation["question_ids"]))
    model["used_canonical_hashes"] = np.concatenate((train["canonical_hashes"], validation["canonical_hashes"]))
    output = new_output(output)
    model_path = output / "model.npz"
    np.savez_compressed(model_path, **model)
    receipt = {"schema": "routing_v2_full_feedback_model_v1", "status": "FROZEN_FOR_VALIDATION_AND_FRESH_DEVELOPMENT",
               "identity": identity, "model": reference(model_path), "training": metadata,
               "training_inputs": reference(train_path), "training_outcomes": reference(train_outcomes_path),
               "validation_inputs": reference(validation_path), "bundle": reference(bundle_path),
               "algorithm_sources": [reference(path) for path in (
                   ROOT / "src/budget_router_v2/full_feedback.py", ROOT / "scripts/v2/full_feedback/run.py",
                   ROOT / "src/budget_router/learner.py", ROOT / "src/budget_router/features.py",
                   ROOT / "src/budget_router/scoring.py", ROOT / "src/budget_router_v2/scoring.py",
                   ROOT / "src/budget_router/ranking_evaluation.py")]}
    model_receipt = output / "MODEL_FROZEN.json"
    write_json(model_receipt, receipt)
    prediction_receipt = freeze_predictions(output / "validation_predictions", validation,
                                            predictions(model, validation), "selector_validation", identity, reference(model_receipt))
    validation_report = score_bundle(bundle_path, prediction_receipt, output / "validation_evaluation")
    return {"model_receipt": str(model_receipt), "validation_report": str(validation_report), "formal_stage2_complete": False}


def predict_bundle(bundle_path, model_receipt_path, output):
    bundle, base, identity = load_bundle(bundle_path)
    receipt = read_json(model_receipt_path)
    require(receipt["identity"] == identity and receipt["status"] == "FROZEN_FOR_VALIDATION_AND_FRESH_DEVELOPMENT",
            "Frozen model and current operator identity differ")
    for source in receipt["algorithm_sources"]:
        verified(source, Path(model_receipt_path).parent)
    model_path = verified(receipt["model"], Path(model_receipt_path).parent)
    with np.load(model_path, allow_pickle=False) as archive:
        model = {name: archive[name] for name in archive.files}
    fresh, _ = read_inputs(bundle, base, "fresh_dev_test")
    require(not (set(fresh["canonical_hashes"]) & set(model["used_canonical_hashes"])) and
            not (set(fresh["question_ids"]) & set(model["used_question_ids"])), "Fresh cohort overlaps model development inputs")
    return freeze_predictions(output, fresh, predictions(model, fresh), "fresh_dev_test", identity, reference(model_receipt_path))


def score_bundle(bundle_path, prediction_receipt_path, output):
    bundle, base, identity = load_bundle(bundle_path)
    receipt = read_json(prediction_receipt_path)
    require(receipt["identity"] == identity and receipt["status"] == "PREDICTIONS_FROZEN_BEFORE_OUTCOME_ACCESS",
            "Prospective frozen predictions required")
    model_receipt_path = verified(receipt["model_receipt"], Path(prediction_receipt_path).parent)
    model_receipt = read_json(model_receipt_path)
    require(model_receipt["identity"] == identity and
            model_receipt["status"] == "FROZEN_FOR_VALIDATION_AND_FRESH_DEVELOPMENT", "Prediction model lineage changed")
    verified(model_receipt["model"], model_receipt_path.parent)
    for source in model_receipt["algorithm_sources"]:
        verified(source, model_receipt_path.parent)
    path = verified(receipt["predictions"], Path(prediction_receipt_path).parent)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    split = receipt["split"]
    inputs, _ = read_inputs(bundle, base, split)
    require(all(np.array_equal(data[name], inputs[name]) for name in ("question_ids", "canonical_hashes", "task_labels", "subtask_labels")),
            "Frozen prediction identity differs from outcome cohort")
    # Outcome arrays are first opened here, after the immutable prediction receipt.
    outcomes, outcome_path = read_outcomes(bundle, base, split, inputs)
    report = evaluate_predictions(inputs, outcomes, {name: data[name] for name in METHODS})
    primary = report["selector_minus_fixed"]["subtask_fixed"]
    passed = (primary["adequate_support"] and primary["question_cluster_ci"][0] > 0 and
              primary["positive_generation_replicates"] >= 2)
    conclusion = ("SUPPORTED" if passed else "NOT_SUPPORTED"
                  if primary["question_cluster_ci"][1] <= 0 else "INCONCLUSIVE")
    report.update(schema="routing_v2_full_feedback_evaluation_v1", split=split,
                  status="STAGE2_EVALUATED" if split == "fresh_dev_test" else "VALIDATION_ONLY",
                  scientific_learnability_conclusion=conclusion if split == "fresh_dev_test" else None,
                  research_complete=False,
                  primary_gate={"ci_low_must_exceed": 0, "minimum_positive_generation_replicates": 2,
                                "passed": bool(passed) if split == "fresh_dev_test" else None},
                  identity=identity, frozen_predictions=reference(prediction_receipt_path), outcomes=reference(outcome_path),
                  original_aggregate_observations=int(outcomes["aggregate_scores"].size),
                  questions=len(inputs["question_ids"]), generation_seeds=outcomes["generation_seeds"].tolist(),
                  formal_stage2_complete=split == "fresh_dev_test")
    output = new_output(output)
    write_json(output / "evaluation.json", report)
    return output / "evaluation.json"
