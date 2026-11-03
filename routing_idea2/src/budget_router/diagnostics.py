"""E1 real-output diagnostics. This module does not generate or infer answers.

All score matrices must come from actual frozen-LLM aggregation and the pinned
scorer. Oracle columns use held-out labels and are diagnostic upper references,
never deployable router performance. E1 data must not enter online/PCA training.
"""
from __future__ import annotations

from itertools import combinations as enumerate_combinations
from collections.abc import Mapping, Sequence
import numpy as np


def _matrix(value, name, seeds, width=None):
    arr = np.asarray(value, dtype=float)
    expected_ndim = 2 if width is None else 3
    if arr.ndim == expected_ndim - 1:
        arr = np.broadcast_to(arr, (seeds,) + arr.shape).copy()
    if arr.ndim != expected_ndim or arr.shape[0] != seeds:
        raise ValueError(f"{name}: expected seed-first array (or one explicitly shared matrix).")
    if width is not None and arr.shape[-1] != width:
        raise ValueError(f"{name}: wrong candidate/model width.")
    return arr


def _binary(arr, name):
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: missing/nonfinite scores; never impute unexecuted outputs.")
    if not np.isin(arr, [0.0, 1.0]).all():
        raise ValueError(f"{name}: v1 GSM8K/BBH exact scores must be 0 or 1.")


def _tasks(labels, n, name):
    arr = np.asarray(labels, dtype=str)
    if arr.shape != (n,) or set(arr) != {"gsm8k", "bbh"}:
        raise ValueError(f"{name}: both locked task labels gsm8k/bbh are required.")
    return arr


def _macro(values, tasks):
    return float(np.mean([values[:, tasks == task].mean() for task in sorted(set(tasks))]))


def _interval(values, tasks, *, bootstrap_replicates, bootstrap_seed, confidence):
    """Cluster by question; preserve all candidate outcomes for a sampled question.

    With multiple actual execution seeds, resample whole seed replicates too.
    Resampling randomness is explicitly separate from experimental replication.
    """
    rng = np.random.default_rng(bootstrap_seed)
    strata = [np.flatnonzero(tasks == task) for task in sorted(set(tasks))]
    nseed = len(values)
    draws = np.empty(bootstrap_replicates)
    for i in range(bootstrap_replicates):
        seed_idx = rng.integers(0, nseed, size=nseed) if nseed > 1 else np.array([0])
        draws[i] = np.mean([
            values[np.ix_(seed_idx, rng.choice(indices, len(indices), replace=True))].mean()
            for indices in strata
        ])
    alpha = (1-confidence)/2
    return {
        "mean": _macro(values, tasks),
        "confidence": confidence,
        "low": float(np.quantile(draws, alpha)),
        "high": float(np.quantile(draws, 1-alpha)),
        "per_experiment_seed": [
            float(np.mean([values[i, tasks == task].mean() for task in sorted(set(tasks))]))
            for i in range(nseed)
        ],
        "per_task": {task: float(values[:, tasks == task].mean()) for task in sorted(set(tasks))},
    }


def evaluate_e1(*, aggregate_scores, member_scores, combinations,
                calibration_aggregate_scores, calibration_member_scores,
                question_ids: Sequence[str], calibration_question_ids: Sequence[str],
                task_labels: Sequence[str], calibration_task_labels: Sequence[str],
                experiment_seeds: Sequence[int] = (42,),
                aggregator_alone_scores=None, strong_model_resample_scores=None,
                execution_costs=None, baseline_costs: Mapping | None = None,
                diagnostic_total_execution_cost: float | None = None,
                evidence: Mapping | None = None,
                question_fingerprints=None, calibration_question_fingerprints=None,
                bootstrap_replicates: int = 2000, bootstrap_seed: int = 20261030,
                confidence: float = .95, minimum_effect: float = .01,
                minimum_total_questions: int = 300, maximum_total_questions: int = 500,
                expected_n_models: int = 6, expected_k: int = 3) -> dict:
    """Compare real aggregate scores without selecting baselines on evaluation labels.

    Shapes: aggregate=[seed,n_eval,n_combo], member=[seed,n_eval,n_models];
    calibration equivalents use n_cal instead. Two-dimensional inputs denote a
    fixed shared cache. Baselines/costs are [seed,n_eval] (or shared [n_eval]).
    v1 uses 100 internal E1 selection-calibration + 200 E1 evaluation questions.
    All 300 E1 questions remain outside PCA/online/final-test data.

    `evidence` must attest E0 lock, real raw LLM aggregate/baseline outputs,
    cost matching and split manifest. Missing evidence/baselines => PARTIAL.
    A one-cache/one-generation E1 is allowed; final five-stream robustness belongs
    to E2-E5. This gate never claims that an oracle can be deployed.
    """
    seeds = tuple(experiment_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("experiment_seeds must be nonempty and unique.")
    if bootstrap_replicates < 100 or not 0 < confidence < 1 or minimum_effect < 0:
        raise ValueError("Invalid predeclared CI/gate settings.")
    combos = tuple(tuple(int(i) for i in c) for c in combinations)
    expected = tuple(enumerate_combinations(range(expected_n_models), expected_k))
    if len(set(combos)) != len(expected) or set(combos) != set(expected):
        raise ValueError("E1 requires the complete locked N6/K3 combination enumeration.")
    ids, cal_ids = tuple(question_ids), tuple(calibration_question_ids)
    if len(set(ids)) != len(ids) or len(set(cal_ids)) != len(cal_ids) or set(ids) & set(cal_ids):
        raise ValueError("Question leakage/duplicate IDs across E1 selection and evaluation.")
    n, nc, ns, ncombo = len(ids), len(cal_ids), len(seeds), len(combos)
    if not n or not nc:
        raise ValueError("Separate E1 selection-calibration and evaluation questions are required.")
    if question_fingerprints is not None or calibration_question_fingerprints is not None:
        if question_fingerprints is None or calibration_question_fingerprints is None:
            raise ValueError("Provide normalized question fingerprints for both splits.")
        fp, cfp = tuple(question_fingerprints), tuple(calibration_question_fingerprints)
        if len(fp) != n or len(cfp) != nc or len(set(fp)) != n or len(set(cfp)) != nc or set(fp) & set(cfp):
            raise ValueError("Duplicate normalized questions under different IDs.")
    tasks = _tasks(task_labels, n, "task_labels")
    cal_tasks = _tasks(calibration_task_labels, nc, "calibration_task_labels")
    agg = _matrix(aggregate_scores, "aggregate_scores", ns, ncombo)
    members = _matrix(member_scores, "member_scores", ns, expected_n_models)
    ca = _matrix(calibration_aggregate_scores, "calibration_aggregate_scores", ns, ncombo)
    cm = _matrix(calibration_member_scores, "calibration_member_scores", ns, expected_n_models)
    for arr, name, count in [(agg,"aggregate_scores",n), (members,"member_scores",n),
                             (ca,"calibration_aggregate_scores",nc), (cm,"calibration_member_scores",nc)]:
        if arr.shape[1] != count:
            raise ValueError(f"{name}: question count mismatch.")
        _binary(arr, name)
    calibration_combo_means = np.mean([ca[:, cal_tasks == task, :].mean(axis=1) for task in sorted(set(cal_tasks))], axis=0)
    calibration_model_means = np.mean([cm[:, cal_tasks == task, :].mean(axis=1) for task in sorted(set(cal_tasks))], axis=0)
    # All tie breaks use input model/combination order, not evaluation labels.
    best_fixed = np.argmax(calibration_combo_means, axis=1)
    topk = [tuple(sorted(np.argsort(-row, kind="stable")[:expected_k])) for row in calibration_model_means]
    lookup = {c:i for i,c in enumerate(combos)}
    independent = np.asarray([lookup[c] for c in topk])
    best_single = np.argmax(calibration_model_means, axis=1)
    select = lambda arr, idx: arr[np.arange(ns)[:,None], np.arange(n)[None,:], idx[:,None]]
    values = {
        "best_fixed_calibration": select(agg, best_fixed),
        "independent_topk_calibration": select(agg, independent),
        "strongest_single_calibration": select(members, best_single),
        "diagnostic_oracle_upper_reference": agg.max(axis=2),
    }
    task_topk = np.empty((ns,n), dtype=int)
    task_fixed = np.empty((ns,n), dtype=int)
    for task in sorted(set(tasks)):
        model_rank = cm[:, cal_tasks == task, :].mean(axis=1)
        for s in range(ns):
            task_topk[s, tasks == task] = lookup[tuple(sorted(np.argsort(-model_rank[s],kind="stable")[:expected_k]))]
            task_fixed[s, tasks == task] = np.argmax(ca[s, cal_tasks == task, :].mean(axis=0))
    values["independent_topk_by_task_calibration"] = agg[np.arange(ns)[:,None], np.arange(n)[None,:], task_topk]
    values["best_fixed_by_task_calibration"] = agg[np.arange(ns)[:,None], np.arange(n)[None,:], task_fixed]
    missing = []
    for name, data in [("aggregator_alone",aggregator_alone_scores), ("strong_model_resample",strong_model_resample_scores)]:
        if data is None:
            missing.append(name + "_real_outputs_and_scores")
        else:
            value = _matrix(data, name, ns)
            if value.shape[1] != n: raise ValueError(f"{name}: question count mismatch.")
            _binary(value,name)
            values[name] = value
    summaries = {name:_interval(value,tasks,bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,confidence=confidence) for name,value in values.items()}
    differences = {}
    for name,value in values.items():
        if name == "diagnostic_oracle_upper_reference": continue
        differences["oracle_minus_"+name] = _interval(values["diagnostic_oracle_upper_reference"]-value,tasks,
            bootstrap_replicates=bootstrap_replicates,bootstrap_seed=bootstrap_seed,confidence=confidence)
    differences["fixed_minus_independent_topk"] = _interval(values["best_fixed_calibration"]-values["independent_topk_calibration"],tasks,
        bootstrap_replicates=bootstrap_replicates,bootstrap_seed=bootstrap_seed,confidence=confidence)
    coverage = np.stack([members[:,:,c].max(axis=2) for c in combos],axis=2)
    oracle_idx = agg.argmax(axis=2)
    oracle_coverage = coverage[np.arange(ns)[:,None],np.arange(n)[None,:],oracle_idx]
    aggregate_loss = coverage-agg
    coverage_metrics = {
        "all_pool_candidate_coverage": _macro(members.max(axis=2),tasks),
        "best_fixed_candidate_coverage": _macro(select(coverage,best_fixed),tasks),
        "best_fixed_aggregate_quality": summaries["best_fixed_calibration"]["mean"],
        "best_fixed_coverage_minus_aggregate": _macro(select(aggregate_loss,best_fixed),tasks),
        "oracle_selected_candidate_coverage": _macro(oracle_coverage,tasks),
        "fraction_combo_outputs_correct_when_any_member_correct": float(agg[coverage==1].mean()) if np.any(coverage==1) else None,
        "fraction_combo_outputs_correct_when_no_member_correct": float(agg[coverage==0].mean()) if np.any(coverage==0) else None,
        "note": "Coverage is auxiliary; signed coverage-minus-aggregate may be negative when aggregator solves independently.",
    }
    cost_report = {}
    if execution_costs is None:
        missing.append("complete_aggregation_execution_cost_matrix")
    else:
        costs = _matrix(execution_costs,"execution_costs",ns,ncombo)
        if costs.shape != agg.shape or not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("Costs must be finite, nonnegative and match every actual aggregate output.")
        cost_report = {
            "best_fixed_calibration": _macro(select(costs,best_fixed),tasks),
            "independent_topk_calibration": _macro(select(costs,independent),tasks),
            "oracle_selected_diagnostic_only": _macro(costs[np.arange(ns)[:,None],np.arange(n)[None,:],oracle_idx],tasks),
            "sum_standalone_combo_service_costs_not_diagnostic_actual": float(costs.sum()),
            "note": "Matrix entries are complete standalone logical service costs; their sum overcounts shared diagnostic member generation and is not diagnostic actual cost.",
        }
    if diagnostic_total_execution_cost is None:
        missing.append("full_actual_E1_diagnostic_cost_including_selection_and_controls")
    elif not np.isfinite(diagnostic_total_execution_cost) or diagnostic_total_execution_cost < 0:
        raise ValueError("Invalid actual diagnostic cost.")
    else:
        cost_report["full_actual_E1_diagnostic_cost"] = float(diagnostic_total_execution_cost)
    for task in ("gsm8k", "bbh"):
        if np.count_nonzero(tasks == task) != 100 or np.count_nonzero(cal_tasks == task) != 50:
            missing.append("v1_split_requires_50_selection_100_evaluation_per_task")
    baseline_costs = baseline_costs or {}
    for name in ("aggregator_alone","strong_model_resample","strongest_single_calibration"):
        if name not in baseline_costs:
            missing.append(name+"_execution_costs")
        else:
            arr = _matrix(baseline_costs[name],name+"_costs",ns)
            if arr.shape != (ns,n) or not np.isfinite(arr).all() or (arr<0).any():
                raise ValueError(f"Invalid costs for {name}.")
            cost_report[name] = _macro(arr,tasks)
    evidence = dict(evidence or {})
    required = ["e0_lock_sha256","cache_revision","aggregator_revision","raw_aggregate_artifact_sha256",
                "raw_baseline_artifact_sha256","split_manifest_sha256","scorer_version","cost_unit"]
    missing.extend("evidence:"+key for key in required if not evidence.get(key))
    for flag in ("real_frozen_llm_aggregation","all_combinations_executed","budget_matched_controls","e1_isolated_from_online_and_pca"):
        if evidence.get(flag) is not True: missing.append("evidence:"+flag)
    if not minimum_total_questions <= n+nc <= maximum_total_questions:
        missing.append(f"required_{minimum_total_questions}_to_{maximum_total_questions}_total_E1_questions")
    reference_names = [name for name in values if name != "diagnostic_oracle_upper_reference"]
    # Max of available baselines is chosen only for a conservative diagnostic
    # opportunity check. It is not exported as a deployable selection policy.
    strongest_reference = max(reference_names,key=lambda name:summaries[name]["mean"])
    opportunity = differences["oracle_minus_"+strongest_reference]
    gate_pass = opportunity["mean"] >= minimum_effect and opportunity["low"] > 0
    status = "PARTIAL" if missing else "PASS" if gate_pass else "STOP_E1"
    return {
        "stage":"E1", "status":status, "overall_research_status":"PARTIAL_UNTIL_E2_E5",
        "missing_requirements":sorted(set(missing)),
        "evaluation_scope":"motivation diagnosis only; no online/PCA training labels",
        "n_questions":n, "n_selection_calibration":nc, "n_total_e1":n+nc,
        "n_combinations":ncombo, "n_models":expected_n_models, "k":expected_k,
        "experimental_seed_count":ns, "generation_cache_replicates":ns, "experimental_seeds":list(seeds),
        "robustness_not_yet_checked":ns<5,
        "ci_conditioning":"Question-stratified paired bootstrap conditional on the pinned cache and aggregator; resampling seeds are not new LLM generations.",
        "bootstrap_replicates":bootstrap_replicates, "bootstrap_seed":bootstrap_seed,
        "task_aggregation":"equal-weight task macro; paired question resampling within gsm8k/bbh",
        "selected_on_calibration":{"best_fixed_indices":best_fixed.tolist(),"independent_topk_indices":independent.tolist(),"strongest_single_indices":best_single.tolist()},
        "metrics":summaries,"paired_differences":differences,"coverage_and_aggregation":coverage_metrics,
        "costs":cost_report,"evidence":evidence,
        "gate":{"kind":"engineering_choice_not_teacher_number_or_expected_result",
                "minimum_observed_oracle_headroom":minimum_effect,"requires_positive_paired_ci_low":True,
                "reference":strongest_reference,"observed_headroom":opportunity,
                "passed_statistical_opportunity_check":gate_pass,
                "limitation":"Oracle uses current evaluation labels; PASS licenses E2 feasibility work only and is never deployable accuracy or proof of learnability/cost benefit."},
    }
