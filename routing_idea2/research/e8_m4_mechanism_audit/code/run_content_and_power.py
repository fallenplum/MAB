"""Frozen E8 M4 content controls and question-cluster planning only.

The sole decoded outcome split is historical train600. No model is refitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy.stats import norm, nct, t


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + "\n",
        encoding="utf-8")


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def question_ci(values, task, fold, mask=None, seed=2026111403):
    """Stratified paired original-question bootstrap, 10,000 resamples."""
    mask = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, bool)
    rows = np.flatnonzero(mask)
    if not len(rows):
        return {"questions": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    totals = np.zeros(10000)
    for label in sorted(set(task[rows].tolist())):
        for f in sorted(set(fold[rows].tolist())):
            selected = rows[(task[rows] == label) & (fold[rows] == f)]
            if len(selected):
                sampled = rng.choice(selected, size=(10000, len(selected)), replace=True)
                totals += np.asarray(values)[sampled].sum(axis=1)
    estimates = totals / len(rows)
    return {"questions": len(rows), "mean": float(np.mean(values[rows])),
            "ci95": np.quantile(estimates, [.025, .975]).tolist()}


def paired_summary(left, right, task, fold, mask=None):
    mask = np.ones(len(left), dtype=bool) if mask is None else np.asarray(mask, bool)
    difference = left - right
    question_difference = difference.mean(axis=1)
    result = question_ci(question_difference, task, fold, mask)
    result.update(accuracy_left=float(left[mask].mean()), accuracy_right=float(right[mask].mean()),
        corrected_events=int(np.sum((left[mask] == 1) & (right[mask] == 0))),
        destroyed_events=int(np.sum((left[mask] == 0) & (right[mask] == 1))),
        seed_gain=difference[mask].mean(axis=0).tolist(),
        positive_mean_questions=int(np.sum(question_difference[mask] > 0)),
        negative_mean_questions=int(np.sum(question_difference[mask] < 0)),
        zero_mean_questions=int(np.sum(question_difference[mask] == 0)))
    return result


def normal_power(delta, sigma, n, sided):
    critical = norm.ppf(1 - .05 / (2 if sided == "two" else 1))
    noncentrality = delta * math.sqrt(n) / sigma
    upper = norm.sf(critical - noncentrality)
    return float(upper + norm.cdf(-critical - noncentrality)) if sided == "two" else float(upper)


def power_table(difference, task):
    means = difference.mean(axis=1)
    variances = [np.var(means[task == label], ddof=1) for label in ("gsm8k", "bbh")]
    sigma = math.sqrt(.5 * sum(variances))
    pool_sigma = float(np.std(means, ddof=1))
    effects = [1 / 150, .01, .02, .03]
    grid = []
    for sided in ("one", "two"):
        critical = norm.ppf(1 - .05 / (2 if sided == "two" else 1))
        for n in (200, 400, 600):
            for delta in effects:
                ncp = delta * math.sqrt(n) / sigma
                tcritical = t.ppf(1 - .05 / (2 if sided == "two" else 1), n - 1)
                tpower = nct.sf(tcritical, n - 1, ncp)
                if sided == "two":
                    tpower += nct.cdf(-tcritical, n - 1, ncp)
                grid.append(dict(sided=sided, questions=n, delta=delta,
                    standard_error=sigma / math.sqrt(n),
                    power_normal=normal_power(delta, sigma, n, sided),
                    power_noncentral_t_normal_data_model=float(tpower)))
    sizes = []
    for sided in ("one", "two"):
        critical = norm.ppf(1 - .05 / (2 if sided == "two" else 1))
        for power in (.8, .9):
            for delta in effects:
                approximate_n = int(math.ceil((sigma * (critical + norm.ppf(power)) / delta) ** 2))
                sizes.append(dict(sided=sided, target_power=power, delta=delta,
                    approximate_required_questions=approximate_n,
                    required_if_variance_50pct_higher=int(math.ceil(1.5 * approximate_n)),
                    required_if_variance_twice=int(math.ceil(2 * approximate_n))))
    detectable = []
    for sided in ("one", "two"):
        critical = norm.ppf(1 - .05 / (2 if sided == "two" else 1))
        for n in (200, 400, 600):
            for power in (.8, .9):
                detectable.append(dict(sided=sided, questions=n, target_power=power,
                    approximate_minimum_detectable_gain=sigma * (critical + norm.ppf(power)) / math.sqrt(n)))
    return dict(mean_observed=float(means.mean()), sigma_question_mean_pooled=pool_sigma,
        sigma_question_mean_task_balanced=sigma, task_variances=variances,
        var_if_1800_question_seed_events_misused_as_independent=float(np.var(difference, ddof=1) / difference.size),
        variance_of_600_cluster_mean_estimate=float(sigma ** 2 / 600),
        question_mean_zero_count=int(np.sum(means == 0)),
        realized_disagreement_events=int(np.sum(difference != 0)),
        grid=grid, approximate_required_n=sizes, minimum_detectable_gains=detectable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path(r"E:\语言模型组合路由"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.asset_root / "research/e8_decision_focused_combination_learning"
    out = args.output
    private, public = out / "private", out / "public"
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    pred_path = base / "private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz"
    calibration_receipt = read(base / "private/evaluation/CALIBRATION_FROZEN_RECEIPT.json")
    assert sha(pred_path) == calibration_receipt["prediction_and_calibration_sha256"]
    pred = archive(pred_path)
    inputs = {str(pred_path): sha(pred_path)}
    masked_scores = np.empty((600, 20))
    permutation_audits = []
    task = np.empty(600, dtype="<U5")
    sub = np.empty(600, dtype="<U39")
    for f in range(5):
        pack = base / "private/qualification/fold_packs" / f"fold{f}"
        meta_path = pack / "evaluation_metadata_only120.npz"
        meta = archive(meta_path)
        rows = meta["global_indices"]
        task[rows], sub[rows] = meta["task"], meta["sub"]
        assert np.all(pred["fold"][rows] == f)
        state_path = base / "private/training" / f"fold{f}" / "M4/model_state.npz"
        state = archive(state_path)
        assert state["coefficient"].shape == (17, 20)
        masked_scores[rows] = state["coefficient"][0]
        inputs[str(state_path)] = sha(state_path)
        history = archive(pack / "history480.npz")
        inputs[str(pack / "history480.npz")] = sha(pack / "history480.npz")
        folder = base / "private/training" / f"fold{f}" / "SHUFFLE_M4"
        outputs = archive(folder / "train_outputs.npz")
        log = read(folder / "FIT_LOG.json")
        mapping = outputs["fitting_permutation"]
        assert np.array_equal(history["task"], history["task"][mapping])
        assert np.array_equal(history["sub"], history["sub"][mapping])
        assert np.array_equal(outputs["fitting_reward_means"], history["Y"][mapping].mean(axis=1))
        assert sorted(mapping.tolist()) == list(range(480))
        for candidate in log["candidates"]:
            for inner in candidate["fits"]:
                fit = np.asarray(inner["fitting_rows"])
                held = np.asarray(inner["held_rows"])
                local = np.asarray(inner["local_permutation"])
                assert not set(fit.tolist()) & set(held.tolist())
                assert np.array_equal(history["task"][fit], history["task"][fit[local]])
                assert np.array_equal(history["sub"][fit], history["sub"][fit[local]])
        permutation_audits.append(dict(fold=f, full_training_rows=480,
            final_mapping_is_within_semantic_group=True, inner_fit_regions_isolated=True,
            whole_reward_blocks_preserved=True, hyperparameters_not_retuned=not log["parameter_selection_performed"],
            moved_training_questions=int(np.sum(mapping != np.arange(480)))))
        inputs[str(folder / "train_outputs.npz")] = sha(folder / "train_outputs.npz")
        inputs[str(folder / "FIT_LOG.json")] = sha(folder / "FIT_LOG.json")
        inputs[str(meta_path)] = sha(meta_path)
    # A feature mask is an intervention on already-frozen coefficients, not a new fit.
    freeze_path = private / "INPUT_MASK_PREDICTIONS_FROZEN.npz"
    np.savez_compressed(freeze_path, masked_scores=masked_scores,
        actions=np.argmax(masked_scores, axis=1), fold=pred["fold"])
    put(private / "INPUT_MASK_FREEZE_RECEIPT.json", dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        prediction_sha256=sha(freeze_path), source_inputs=inputs,
        action_rule="exact numpy argmax of original frozen M4 constant row coefficient[0]",
        whole_train600_source_payload_decoded_in_current_script=False,
        fold_history480_rewards_already_read=True,
        five_fold_history_packs_jointly_cover_train600=True,
        feature_mask_definition_uses_reward_values=False,
        isolation_scope="application-level source read order, not operating-system isolation",
        no_new_fit=True, no_llm_call=True))
    # The privileged evaluator may now decode historical train600 only.
    sys.path.insert(0, str(args.asset_root / "research/decision_acquisition_e3/public"))
    from assets import load_assets
    data, refs = load_assets(args.asset_root)
    assert np.array_equal(task, data["task"]) and np.array_equal(sub, data["sub"])
    y = data["Y"]
    variants = {}
    for name in ("M0_original", "M4", "SHUFFLE_M4"):
        metrics_path = base / "private/evaluation" / f"{name}_QUESTION_METRICS.npz"
        metrics = archive(metrics_path)
        actions = np.argmax(pred["raw__" + name], axis=1)
        assert np.array_equal(actions, metrics["actions"])
        realized = y[np.arange(600), :, actions]
        assert np.array_equal(realized, metrics["per_generation_rewards"])
        variants[name] = realized
        inputs[str(metrics_path)] = sha(metrics_path)
    variants["M4_input_mask"] = y[np.arange(600), :, np.argmax(masked_scores, axis=1)]
    variants["train_global_fixed"] = y[np.arange(600), :, pred["default_actions"]]
    comparisons = {}
    for name in ("SHUFFLE_M4", "M4_input_mask", "M0_original", "train_global_fixed"):
        comparisons[name] = dict(
            overall=paired_summary(variants["M4"], variants[name], task, pred["fold"]),
            task={label: paired_summary(variants["M4"], variants[name], task, pred["fold"], task == label)
                  for label in ("gsm8k", "bbh")},
            fold=[dict(fold=f, **paired_summary(variants["M4"], variants[name], task, pred["fold"], pred["fold"] == f))
                  for f in range(5)],
            subtask=[dict(subtask=s, **paired_summary(variants["M4"], variants[name], task, pred["fold"], (task == "bbh") & (sub == s)))
                     for s in sorted(set(sub[task == "bbh"].tolist()))])
    power = {name: power_table(variants["M4"] - variants[name], task)
             for name in ("M0_original", "train_global_fixed")}
    # Numerical power implementation qualification under an explicit Normal model.
    rng = np.random.default_rng(20261114031)
    mc_cases = []
    for delta in (0., .01):
        n = 400
        sigma = power["M0_original"]["sigma_question_mean_task_balanced"]
        known_sigma_z = rng.normal(delta * math.sqrt(n) / sigma, 1., 100000)
        observed = float(np.mean(known_sigma_z > norm.ppf(.95)))
        expected = normal_power(delta, sigma, n, "one")
        mc_cases.append(dict(delta=delta, n=n, known_sigma_normal_samples=100000,
            rejection_rate=observed, expected=expected,
            monte_carlo_standard_error=math.sqrt(expected * (1 - expected) / 100000)))
        assert abs(observed - expected) < .006
    result = dict(status="COMPLETE_EXPLORATORY_DEVELOPMENT_DIAGNOSTIC",
        source="frozen E8 original600 outer-CV predictions, never new independent test",
        comparisons=comparisons, negative_control_audit=permutation_audits,
        masked_feature_actions_per_fold=[dict(fold=f, action=int(np.argmax(masked_scores[pred['fold'] == f][0])),
            selected_questions=120) for f in range(5)], power=power,
        qualification_normal_model=mc_cases,
        inference=dict(bootstrap_repetitions=10000, seed=2026111403,
            unit="original_question retaining three generation repeats", strata="outer_fold x task",
            selection="no parameter or model selection from current diagnostic",
            power_alpha=.05, power_sides=["one", "two"], independent_question_counts=[200, 400, 600],
            power_variance_scope="plug-in historical OOF question-mean variance, balanced50:50 tasks"),
        boundaries=dict(decoded_outcome_splits=["selector_train600"],
            validation200_outcomes_read=False, fresh400_outcomes_read=False, sealed200_outcomes_read=False,
            physical_llm_calls=0, prior_results_changed=False),
        literature_sources=[
            dict(title="NIST Analysis of paired observations", url="https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm"),
            dict(title="statsmodels TTestPower paired-sample power", url="https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.power.html"),
            dict(title="statsmodels normal power and sample size implementation", url="https://www.statsmodels.org/stable/_modules/statsmodels/stats/power.html#normal_sample_size_one_tail")])
    exposure_records = []
    for relative, fields in [
        ("research/fewshot_protocol_fix_20261107/public/COMPLETION.json",
         ["fresh400_scores_read", "sealed200_read"]),
        ("research/fewshot_protocol_fix_20261107/public/FROZEN_CANDIDATE.json",
         ["fresh_execution", "fresh_scores_read", "sealed_scores_read"]),
        ("research/b1_reliable_paired_evaluation/public/B1_PROTOCOL.json",
         ["fresh400_outcomes_permitted", "sealed200_outcomes_permitted"]),
        ("research/e8_decision_focused_combination_learning/public/E8_PROTOCOL.json", [])]:
        source = args.asset_root / relative
        metadata = read(source)
        values = {field: metadata[field] for field in fields}
        if not fields:
            values = {key: metadata["data"][key] for key in
                      ("new_validation200_outcome_access", "fresh400_outcome_access", "sealed200_outcome_access")}
        exposure_records.append(dict(source=relative, sha256=sha(source), metadata_values=values))
    result["heldout_role_metadata"] = dict(records=exposure_records,
        source_type="historical declared protocols and completion receipts, no heldout payload audit",
        fresh400_original_role="independent development evaluation after freezing, prior fewshot plan not executed",
        sealed200_original_role="formal final confirmation",
        actual_current_heldout_outcomes_inspected=False,
        observation="Metadata flags are declared boundaries, not a proof that every historical access is globally absent.")
    np.savez_compressed(private / "CONTENT_AND_POWER_PAIRED_VALUES.npz", task=task, sub=sub,
        fold=pred["fold"], **variants)
    put(private / "READ_ONLY_INPUT_HASHES.json", inputs)
    put(private / "REWARD_ACCESS_RECEIPT.json", dict(allowed_split="selector_train600",
        decoded_payload_splits=refs["identity"]["decoded_payload_splits"],
        source_refs=refs, masks_frozen_before_historical_reward_decoding=True))
    put(public / "M4_CONTENT_AND_POWER_RESULTS.json", result)
    print(json.dumps({"status": "COMPLETE", "comparisons": {k: v["overall"] for k, v in comparisons.items()},
        "sigma": {k: v["sigma_question_mean_task_balanced"] for k, v in power.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
