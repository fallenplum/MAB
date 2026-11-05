"""Existing train/validation outcome diagnostics; no fits or fresh data access."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np

ROOT = Path(r"E:\语言模型组合路由")
FF = ROOT / "artifacts/private/routing_redesign_20261104/ff_v1"
SEEDS = (42, 137, 2024)
TASKS = ("gsm8k", "bbh")
REPLICATES = 2000
BOOTSTRAP_SEED = 20261104
PAIR_INDEX = np.array(list(itertools.combinations(range(20), 2)))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ref(path):
    return {"path": str(Path(path).resolve(strict=True)), "sha256": sha(path)}


def put(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class QuestionBootstrap:
    """Original task/subtask-stratified question resampling, shared across seeds."""
    def __init__(self, tasks, subtasks):
        self.tasks, self.subtasks = np.asarray(tasks), np.asarray(subtasks)
        self.n = len(tasks)
        self.present = [t for t in TASKS if np.any(self.tasks == t)]
        self.weights = np.zeros((self.n, REPLICATES))
        self.support = {}
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for task in self.present:
            mask = self.tasks == task
            groups = sorted(set(self.subtasks[mask].tolist())) if task == "bbh" else [""]
            for group in groups:
                indexes = np.flatnonzero(mask & ((self.subtasks == group) if task == "bbh" else True))
                n = len(indexes)
                self.support[task + ("/" + group if group else "")] = n
                counts = rng.multinomial(n, np.full(n, 1 / n), size=REPLICATES)
                self.weights[indexes] = counts.T / int(mask.sum()) / len(self.present)

    def stat(self, values, labels):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        require(values.shape == (len(labels), self.n) and np.isfinite(values).all(), "Statistic dimensions/values differ")
        if self.n == 0:
            return {"mean": None, "question_cluster_ci": None, "question_support": 0, "per_replica": []}
        mean = np.mean([values[:, self.tasks == t].mean(axis=1) for t in self.present], axis=0)
        draws = values @ self.weights
        interval = np.quantile(draws.mean(axis=0), [.025, .975])
        return {"mean": float(mean.mean()), "question_cluster_ci": interval.tolist(),
            "question_support": self.n, "per_task_support": {t: int(np.sum(self.tasks == t)) for t in self.present},
            "strata_support": self.support, "adequate_stratum_support_at_least2": all(n >= 2 for n in self.support.values()),
            "per_replica": [{"label": str(label), "mean": float(mean[i]),
                "question_cluster_ci": np.quantile(draws[i], [.025, .975]).tolist()} for i, label in enumerate(labels)],
            "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            "scope": "Question cluster; retain all listed replicas together; task and BBH subtask strata; exploratory unadjusted95% interval"}

    def conditional(self, values, mask, labels):
        mask = np.asarray(mask, dtype=bool)
        return QuestionBootstrap(self.tasks[mask], self.subtasks[mask]).stat(np.asarray(values)[..., mask], labels)


def select(y, indexes):
    return y[np.arange(y.shape[0])[:, None], np.arange(y.shape[1])[None, :], np.asarray(indexes)[None, :]]


def pattern_counts(y):
    weights = np.array([4, 2, 1])[:, None, None]
    patterns = np.sum(y.astype(int) * weights, axis=0)
    counts = {format(i, "03b"): int(np.sum(patterns == i)) for i in range(8)}
    total = int(patterns.size)
    return {"generation_bit_order": list(SEEDS), "question_team_cells": total, "patterns": counts,
        "observed_same_score_all3_cells": counts["000"] + counts["111"],
        "observed_generation_sensitive_cells": total - counts["000"] - counts["111"],
        "generation_sensitive_cell_fraction": (total - counts["000"] - counts["111"]) / total,
        "not_a_causal_or_population_noise_fraction": True}


def rank_pair(y, a, b, bootstrap):
    source, target = y[a], y[b]
    da = source[:, PAIR_INDEX[:, 0]] - source[:, PAIR_INDEX[:, 1]]
    db = target[:, PAIR_INDEX[:, 0]] - target[:, PAIR_INDEX[:, 1]]
    joint = (da != 0) & (db != 0)
    concordant = joint & (da == db)
    reversed_order = joint & (da != db)
    src_count = (da != 0).sum(axis=1)
    target_count = (db != 0).sum(axis=1)
    joint_count = joint.sum(axis=1)
    source_centered, target_centered = source - source.mean(axis=1, keepdims=True), target - target.mean(axis=1, keepdims=True)
    denominator = np.sqrt((source_centered**2).sum(axis=1) * (target_centered**2).sum(axis=1))
    valid = denominator > 0
    correlation = np.zeros(len(source))
    correlation[valid] = (source_centered[valid] * target_centered[valid]).sum(axis=1) / denominator[valid]
    flat_a, flat_b = src_count == 0, target_count == 0
    total_joint = int(joint_count.sum())
    a_best, b_best = source == source.max(axis=1, keepdims=True), target == target.max(axis=1, keepdims=True)
    both_oracle_success = (source.max(axis=1) > 0) & (target.max(axis=1) > 0)
    intersection = (a_best & b_best).sum(axis=1)
    union = (a_best | b_best).sum(axis=1)
    return {"seeds": [SEEDS[a], SEEDS[b]], "questions": len(source),
        "both_rankable_questions": int(valid.sum()), "source_flat_questions": int(flat_a.sum()),
        "target_flat_questions": int(flat_b.sum()), "both_flat_questions": int((flat_a & flat_b).sum()),
        "flat_means_all20_results_equal_not_noiseless_ranking": True,
        "binary_tie_spearman_equivalent_pearson_on_both_rankable": bootstrap.conditional(correlation, valid, [f"{SEEDS[a]}_vs_{SEEDS[b]}"]),
        "joint_strict_pair_support": total_joint, "concordant_joint_strict_pairs": int(concordant.sum()),
        "reversed_joint_strict_pairs": int(reversed_order.sum()),
        "joint_strict_concordance": float(concordant.sum() / total_joint) if total_joint else None,
        "joint_strict_reversal": float(reversed_order.sum() / total_joint) if total_joint else None,
        "source_only_strict_target_tie_pairs": int((src_count - joint_count).sum()),
        "target_only_strict_source_tie_pairs": int((target_count - joint_count).sum()),
        "cell_score_flip_count": int(np.sum(source != target)),
        "cell_score_flip_fraction": float(np.mean(source != target)),
        "both_have_some_correct_team_questions": int(both_oracle_success.sum()),
        "observed_positive_team_set_jaccard_when_both_have_success": bootstrap.conditional(
            intersection / union, both_oracle_success, [f"{SEEDS[a]}_vs_{SEEDS[b]}"]),
        "tie_and_ceiling_warning": "Correlation excludes constant20team rows; high overall answer agreement can coexist with weak discriminative ordering."}


def transfer(y, source, target, baseline_quality, bootstrap):
    choice = np.argmax(source, axis=1)
    best = source == source.max(axis=1, keepdims=True)
    chosen = y[target, np.arange(len(source)), choice]
    uniform_tie = (y[target] * best).sum(axis=1) / best.sum(axis=1)
    informative = source.max(axis=1) > source.min(axis=1)
    return chosen, uniform_tie, {"heldout_generation_seed": SEEDS[target],
        "lexicographic_target_quality": bootstrap.stat(chosen, [SEEDS[target]]),
        "uniform_source_max_tie_expected_target_quality": bootstrap.stat(uniform_tie, [SEEDS[target]]),
        "source_discriminatory_question_support": int(informative.sum()),
        "lexicographic_target_quality_source_discriminatory_only": bootstrap.conditional(chosen, informative, [SEEDS[target]]),
        "source_max_zero_question_support": int(np.sum(source.max(axis=1) == 0)),
        "source_max_tie_size_distribution": {str(n): int(np.sum(best.sum(axis=1) == n)) for n in sorted(set(best.sum(axis=1).tolist()))},
        "target_quality_minus_existing_fixed": {name: bootstrap.stat(chosen - values[target], [SEEDS[target]])
                                                for name, values in baseline_quality.items()},
        "uses_other_generation_true_outcomes_of_this_same_question": True,
        "not_a_deployable_unseen_question_router": True}


def analyze_scope(y, inputs, fixed):
    tasks, subtasks = inputs["task_labels"], inputs["subtask_labels"]
    bootstrap = QuestionBootstrap(tasks, subtasks)
    baselines = {name: select(y, indexes) for name, indexes in fixed.items()}
    oracle = y.max(axis=2)
    shared_best_mean = y.mean(axis=0).max(axis=1)
    oracle_mean = oracle.mean(axis=0)
    stable_correct = (y == 1).all(axis=0).any(axis=1)
    decomposition = {}
    for name, values in baselines.items():
        total = oracle_mean - values.mean(axis=0)
        shared = shared_best_mean - values.mean(axis=0)
        specific = oracle_mean - shared_best_mean
        require(np.allclose(total, shared + specific) and min(shared.min(), specific.min()) >= -1e-12,
                "Observed oracle opportunity decomposition failed")
        opportunity = (oracle > 0) & (values == 0)
        stable_cells = opportunity & stable_correct[None, :]
        total_metric = bootstrap.stat(total, ["all3_generation_average"])
        shared_metric = bootstrap.stat(shared, ["all3_generation_average"])
        specific_metric = bootstrap.stat(specific, ["all3_generation_average"])
        denominator = total_metric["mean"]
        decomposition[name] = {"fixed_quality": bootstrap.stat(values, SEEDS),
            "total_same_seed_hindsight_oracle_advantage": total_metric,
            "generation_shared_best_same_team_hindsight_component": shared_metric,
            "generation_specific_hindsight_increment": specific_metric,
            "shared_fraction_of_observed_oracle_advantage": shared_metric["mean"] / denominator if denominator > 0 else None,
            "generation_specific_fraction_of_observed_oracle_advantage": specific_metric["mean"] / denominator if denominator > 0 else None,
            "baseline_error_cells_recoverable_by_same_seed_oracle": int(opportunity.sum()),
            "those_cells_on_questions_with_a_team_correct_all3": int(stable_cells.sum()),
            "remaining_cells_without_any_team_correct_all3": int((opportunity & ~stable_correct[None, :]).sum()),
            "causal_stable_signal_or_noise_fraction_identified": False}
    rank = [rank_pair(y, a, b, bootstrap) for a, b in itertools.combinations(range(3), 2)]
    singles = []
    for a, b in itertools.permutations(range(3), 2):
        _, _, result = transfer(y, y[a], b, baselines, bootstrap)
        result["source_generation_seeds"] = [SEEDS[a]]
        singles.append(result)
    heldout, uniform, folds = [], [], []
    for target in range(3):
        source_seeds = [i for i in range(3) if i != target]
        chosen, expected, result = transfer(y, y[source_seeds].mean(axis=0), target, baselines, bootstrap)
        result["source_generation_seeds"] = [SEEDS[i] for i in source_seeds]
        heldout.append(chosen)
        uniform.append(expected)
        folds.append(result)
    heldout, uniform = np.stack(heldout), np.stack(uniform)
    return {"question_support": len(tasks), "observed_answer_cells": int(y.size),
        "task_question_support": {t: int(np.sum(tasks == t)) for t in TASKS if np.any(tasks == t)},
        "observed_score_patterns": pattern_counts(y),
        "per_generation_mean_over_all20teams": [{"seed": seed, "quality": float(y[i].mean())} for i, seed in enumerate(SEEDS)],
        "questions_with_at_least_one_team_correct_all3": int(stable_correct.sum()),
        "fraction_questions_with_at_least_one_team_correct_all3": float(stable_correct.mean()),
        "questions_with_oracle_success_all3_but_no_single_team_correct_all3": int(np.sum((oracle == 1).all(axis=0) & ~stable_correct)),
        "hindsight_per_generation_oracle_quality": bootstrap.stat(oracle, SEEDS),
        "best_same_team_per_question_3seed_average_quality": bootstrap.stat(shared_best_mean, ["all3_generation_average"]),
        "ranking_pairs": rank, "single_source_generation_transfer": singles,
        "two_source_generation_to_heldout": {"folds": folds,
            "lexicographic_quality_all3_heldout": bootstrap.stat(heldout, SEEDS),
            "uniform_source_max_tie_quality_all3_heldout": bootstrap.stat(uniform, SEEDS),
            "lexicographic_minus_fixed": {name: bootstrap.stat(heldout - values, SEEDS) for name, values in baselines.items()},
            "uniform_source_tie_minus_fixed": {name: bootstrap.stat(uniform - values, SEEDS) for name, values in baselines.items()},
            "same_question_oracle_minus_heldout_selected": bootstrap.stat(oracle - heldout, SEEDS),
            "scope": "Retrospective reproducibility diagnostic using two true labels for the same question; not a learned feature-based router or new-question generalization result"},
        "oracle_opportunity_decomposition": decomposition}


def load_data(bundle, split):
    inp = bundle["inputs"][split]
    out = bundle["outcomes"][split]
    receipt_ref = bundle["matrix_receipts"][split]
    for item in (inp, out, receipt_ref):
        require(sha(item["path"]) == item["sha256"], "Existing artifact SHA changed")
    receipt = read(receipt_ref["path"])
    require(receipt["status"] == "REAL_FULL_INFORMATION_READY" and not receipt["software_test_only"] and
            receipt["generation_seeds"] == list(SEEDS) and receipt["all_combinations_executed"] is True and
            receipt["outcomes"]["sha256"] == out["sha256"] and receipt["inputs"]["sha256"] == inp["sha256"],
            "Complete existing real matrix identity differs")
    with np.load(inp["path"], allow_pickle=False) as z:
        inputs = {k: z[k] for k in ("question_ids", "canonical_hashes", "task_labels", "subtask_labels")}
    with np.load(out["path"], allow_pickle=False) as z:
        require(np.array_equal(z["generation_seeds"], SEEDS) and all(np.array_equal(z[k], inputs[k]) for k in ("question_ids", "canonical_hashes")), "Matrix question identity differs")
        scores = z["aggregate_scores"].astype(float)
    n = 600 if split == "selector_train" else 200
    require(scores.shape == (3, n, 20) and np.isfinite(scores).all() and np.isin(scores, (0, 1)).all(), "Only exact existing3xNx20 binary outcomes allowed")
    return inputs, scores, {"inputs": inp, "outcomes": out, "matrix_receipt": receipt_ref,
        "verification": "Primary input/outcome/receipt bytes and complete-matrix identity verified; no raw answers, gold or nested mixed source payloads re-read"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    output = args.output_dir.resolve()
    require(output.is_relative_to((ROOT / "artifacts/private").resolve()) and not output.exists(), "New private diagnostic directory required")
    frozen = [FF / "model/MODEL_FROZEN.json", FF / "model/model.npz", FF / "pred/PREDICTIONS_FROZEN.json", FF / "pred/predictions.npz"]
    pre = {str(p): sha(p) for p in frozen}
    bundle_path = FF / "BUNDLE_TRAINVAL.json"
    bundle = read(bundle_path)
    train_in, train_y, train_sources = load_data(bundle, "selector_train")
    val_in, val_y, val_sources = load_data(bundle, "selector_validation")
    require(not set(train_in["canonical_hashes"]) & set(val_in["canonical_hashes"]), "Train/validation overlap")
    # Exact reconstruction of the already-used fixed routes, no regression fit.
    task_means = np.stack([train_y[:, train_in["task_labels"] == t].mean(axis=(0, 1)) for t in TASKS])
    global_team = int(np.argmax(task_means.mean(axis=0)))
    task_team = {t: int(np.argmax(task_means[i])) for i, t in enumerate(TASKS)}
    subtask_team = {s: int(np.argmax(train_y[:, train_in["subtask_labels"] == s].mean(axis=(0, 1))))
                    for s in sorted(set(train_in["subtask_labels"][train_in["task_labels"] == "bbh"].tolist()))}
    def routes(inputs):
        task = np.array([task_team[t] for t in inputs["task_labels"]])
        subtask = np.array([subtask_team[s] if t == "bbh" else task_team[t]
                           for t, s in zip(inputs["task_labels"], inputs["subtask_labels"])])
        return {"global_fixed": np.full(len(task), global_team), "task_fixed": task, "subtask_fixed": subtask}
    val_fixed = routes(val_in)
    old_pred_receipt_path = FF / "model/validation_predictions/PREDICTIONS_FROZEN.json"
    old_pred_receipt = read(old_pred_receipt_path)
    require(sha(old_pred_receipt["predictions"]["path"]) == old_pred_receipt["predictions"]["sha256"], "Original fixed prediction artifact changed")
    with np.load(old_pred_receipt["predictions"]["path"], allow_pickle=False) as z:
        require(all(np.array_equal(z[k], val_in[k]) for k in val_in) and
                all(np.array_equal(np.argmax(z[name], axis=1), values) for name, values in val_fixed.items()),
                "Reconstructed fixed route differs from actual original frozen validation routes")
    datasets = {}
    for name, inputs, y, sources in (("train600", train_in, train_y, train_sources), ("validation200", val_in, val_y, val_sources)):
        fixed = routes(inputs)
        scopes = {}
        for scope, mask in (("all", np.ones(len(y[0]), dtype=bool)), ("gsm8k", inputs["task_labels"] == "gsm8k"), ("bbh", inputs["task_labels"] == "bbh")):
            subset = {k: value[mask] for k, value in inputs.items()}
            scopes[scope] = analyze_scope(y[:, mask], subset, {k: value[mask] for k, value in fixed.items()})
        bbh = {}
        for subtask in sorted(set(inputs["subtask_labels"][inputs["task_labels"] == "bbh"].tolist())):
            mask = inputs["subtask_labels"] == subtask
            bbh[subtask] = analyze_scope(y[:, mask], {k: value[mask] for k, value in inputs.items()}, {k: value[mask] for k, value in fixed.items()})
        datasets[name] = {"sources": sources, "scopes": scopes, "bbh_subtasks": bbh,
            "fixed_baseline_scope": "Original all3-training-generation fixed route, in-sample on train; validation routes are train-derived and verified against original freeze"}
        print(json.dumps({"dataset": name, "questions": len(inputs["question_ids"]), "phase": "DIAGNOSTIC_COMPLETE", "new_model_calls": 0}), flush=True)
    require(pre == {str(p): sha(p) for p in frozen}, "Original frozen files changed")
    report = {"schema": "routing_v2_same_question_cross_generation_diagnostics_v1", "status": "POSTHOC_CPU_DIAGNOSTIC_ONLY",
        "generation_seeds": list(SEEDS), "N": 6, "K": 3, "teams": 20,
        "source_bundle": ref(bundle_path), "diagnostic_source": ref(Path(__file__)),
        "original_frozen_validation_prediction_receipt": ref(old_pred_receipt_path),
        "original_fixed_validation_choices_reconstructed_exactly": True,
        "datasets": datasets, "new_regression_or_model_fits": 0, "new_model_calls": 0,
        "new_answer_scoring_operations": 0, "fresh_or_sealed_gold_or_outputs_read": 0,
        "original_four_frozen_sha_pre_post_identical": True,
        "statistical_scope": "2000 question-cluster draws, seed20261104; task/BBHsubtask strata; all listed generation replicas stay together. Descriptive posthoc intervals, no multiplicity-adjusted or deployment claim.",
        "hardware_limitations": ["All S2train member generations and all aggregate executions use matched A6000 runtime.",
            "Validation generation42 contains400 historical local Qwen/Coder member responses; other member cells are A6000. Comparisons involving42 mix seed and member platform.",
            "Validation137-vs2024 is same-A6000 generation comparison. Three generations cannot identify population or causal generation-noise fractions."],
        "interpretation_boundaries": ["Source-generation answer labels are for the exact same question, so transfer methods are unavailable to an unseen-question router at deployment.",
            "Observed stable000/111 cells and mixed patterns describe three draws only; they do not identify true stable signal versus causal noise.",
            "Oracle decomposition is algebraic hindsight: mean_g max_team minus fixed equals max_team mean_g minus fixed plus mean_g max_team minus max_team mean_g.",
            "Even the generation-shared component uses all three observed outcomes and can overfit; it is not automatically learnable from input features.",
            "Single/two-source maxima can have many ties. Lexicographic and uniform-source-max expectations are both reported; no target outcomes choose the reported route.",
            "Ranks of binary scores are largely tied. Constant20team rows are excluded from correlation with support shown, never assigned perfect ranking."],
        "cpu_seconds": time.process_time() - started_cpu, "wall_seconds": time.perf_counter() - started_wall}
    output.mkdir()
    target = output / "CROSS_GENERATION_SEED_STABILITY_v1.json"
    put(target, report)
    compact = {name: {scope: {"questions": d["question_support"],
        "mixed_cell_fraction": d["observed_score_patterns"]["generation_sensitive_cell_fraction"],
        "has_team_correct_all3_fraction": d["fraction_questions_with_at_least_one_team_correct_all3"],
        "same_seed_oracle_quality": d["hindsight_per_generation_oracle_quality"]["mean"],
        "best_same_team_3seed_average": d["best_same_team_per_question_3seed_average_quality"]["mean"],
        "two_seed_to_heldout_quality": d["two_source_generation_to_heldout"]["lexicographic_quality_all3_heldout"],
        "two_seed_to_heldout_minus_subtask_fixed": d["two_source_generation_to_heldout"]["lexicographic_minus_fixed"]["subtask_fixed"],
        "oracle_decomposition_vs_subtask": {k:v for k,v in d["oracle_opportunity_decomposition"]["subtask_fixed"].items()
            if k in ("shared_fraction_of_observed_oracle_advantage", "generation_specific_fraction_of_observed_oracle_advantage", "baseline_error_cells_recoverable_by_same_seed_oracle", "those_cells_on_questions_with_a_team_correct_all3", "remaining_cells_without_any_team_correct_all3")}}
        for scope,d in dataset["scopes"].items()} for name,dataset in datasets.items()}
    put(output / "SUMMARY_v1.json", {"status": report["status"], "report": ref(target), "summary": compact,
        "new_model_calls": 0, "new_fits": 0, "fresh_or_sealed_access": 0,
        "cpu_seconds": report["cpu_seconds"], "wall_seconds": report["wall_seconds"]})
    print(json.dumps({"report": ref(target), "summary": compact, "cpu_seconds": report["cpu_seconds"], "wall_seconds": report["wall_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
