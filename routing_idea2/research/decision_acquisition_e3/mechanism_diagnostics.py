"""Post-hoc descriptions of existing E3 traces and reuse-only cost arithmetic.

No policy is executed here. Future-feedback snapshots are evaluator diagnostics,
not information delivered to a learner. Their overlapping future sets cannot be
summed into an independent or additive explanation of final service quality.
"""

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
from scipy.stats import rankdata

from assets import RESOURCE_KEYS, SEEDS, TEAMS, priced_units, zero_resources


CONTEXT_FIELDS = ("predictor", "stage", "method", "B", "history_size")
PATH_FIELDS = ("fold", "gen", "order")
KG_METHODS = {"decision", "decision_cost", "no_cost", "main_only",
              "independent_noise", "random_replacement"}


def _finite(value, name):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    count = len(x)
    if count < 3:
        return {"status": "insufficient_question_clusters", "question_clusters": count,
                "pearson": None, "spearman": None}
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"status": "constant_variable", "question_clusters": count,
                "pearson": None, "spearman": None}
    return {"status": "DESCRIPTIVE", "question_clusters": count,
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
            "p_value": None, "confidence_interval": None}


def _task_name(trace):
    return "gsm8k" if trace["task"] == "gsm8k" else "bbh/" + str(trace["sub"])


def _question_average(records, fields):
    groups = defaultdict(list)
    for record in records:
        groups[int(record["q"])].append(record)
    result = []
    for q in sorted(groups):
        values = groups[q]
        if len({(item["task"], item["sub"]) for item in values}) != 1:
            raise ValueError("Original-question task labels changed across repetitions")
        row = {"q": q, "task": values[0]["task"], "sub": values[0]["sub"],
               "repeated_measurements": len(values)}
        row.update({field: float(np.mean([item[field] for item in values])) for field in fields})
        result.append(row)
    return result


def _service_summary(questions):
    def summarize(rows):
        return {"original_questions": len(rows),
                "accuracy": float(np.mean([row["reward"] for row in rows])),
                "no_feedback_accuracy": float(np.mean([row["baseline"] for row in rows])),
                "realized_prequential_gain": float(np.mean([row["gain"] for row in rows])),
                "correction_probability": float(np.mean([row["corrected"] for row in rows])),
                "harm_probability": float(np.mean([row["harmed"] for row in rows]))}

    by_task = {task: summarize([row for row in questions if row["task"] == task])
               for task in sorted({row["task"] for row in questions})}
    overall = summarize(questions)
    overall["by_task"] = by_task
    overall["balanced_gsm_bbh_gain"] = (
        0.5 * (by_task["gsm8k"]["realized_prequential_gain"] + by_task["bbh"]["realized_prequential_gain"])
        if {"gsm8k", "bbh"} <= set(by_task) else None
    )
    overall["scope"] = "Actual frozen service decisions along the realized adaptive path, versus the same historical predictor without feedback"
    overall["final_frozen_router_test"] = "not_available_in_these_traces"
    return overall


def _diagnose_context(runs):
    all_rows, influence_rows, kg_rows = [], [], []
    excluded = Counter()
    task_paths = defaultdict(list)
    raw_snapshot_totals = Counter()
    for run in runs:
        meta = run["meta"]
        method = str(meta["method"])
        path_tasks = defaultdict(list)
        seen = set()
        for trace in run["traces"]:
            q = int(trace["q"])
            if q in seen:
                raise ValueError("An original question appears twice in one service path")
            seen.add(q)
            reward = _finite(trace["reward"], "reward")
            baseline = _finite(trace["no_feedback_reward"], "no_feedback_reward")
            if reward not in (0, 1) or baseline not in (0, 1):
                raise ValueError("Realized service comparisons require raw binary scores")
            row = {"q": q, "task": str(trace["task"]), "sub": str(trace["sub"]),
                   "reward": reward, "baseline": baseline, "gain": reward - baseline,
                   "corrected": float(reward > baseline), "harmed": float(reward < baseline),
                   "purchased": float(bool(trace["purchase"])),
                   "comparison": float(bool(trace.get("comparison", False)))}
            all_rows.append(row)
            path_tasks[_task_name(trace)].append(row)
            if not trace["purchase"]:
                continue
            influence = trace.get("future_feedback_influence")
            if influence is None:
                excluded["no_logged_future_snapshot_or_no_remaining_query"] += 1
                continue
            remaining = int(influence["remaining_questions"])
            if remaining <= 0:
                excluded["no_remaining_query"] += 1
                continue
            changed = _finite(influence["changed"], "future changed")
            corrected = _finite(influence["corrected"], "future corrected")
            harmed = _finite(influence["harmed"], "future harmed")
            net = _finite(influence["net_reward"], "future net_reward")
            if min(changed, corrected, harmed) < 0 or max(changed, corrected, harmed) > remaining:
                raise ValueError("Future snapshot counts exceed remaining questions")
            if corrected + harmed > changed or not np.isclose(net, corrected - harmed):
                raise ValueError("Future snapshot net reward is inconsistent with corrections and harm")
            future_row = {"q": q, "task": row["task"], "sub": row["sub"],
                          "remaining": float(remaining), "changed_rate": changed / remaining,
                          "correction_rate": corrected / remaining, "harm_rate": harmed / remaining,
                          "net_rate": net / remaining}
            influence_rows.append(future_row)
            raw_snapshot_totals.update({"events": 1, "remaining_question_opportunities": remaining,
                                        "changed": changed, "corrected": corrected,
                                        "harmed": harmed, "net_reward": net})
            if method not in KG_METHODS:
                excluded["kg_not_computed_for_this_method"] += 1
            elif int(trace.get("candidate_count", 0)) <= 0:
                excluded["warmup_or_unplanned_kg_zero_placeholder"] += 1
            elif method == "random_replacement" and trace.get("comparison", False):
                excluded["logged_kg_precedes_random_neighbor_replacement"] += 1
            elif trace.get("chosen_kg") is None:
                excluded["missing_chosen_kg"] += 1
            else:
                kg = _finite(trace["chosen_kg"], "chosen_kg")
                if kg < 0:
                    raise ValueError("Logged Gaussian KG must be nonnegative")
                kg_rows.append({**future_row, "kg": kg})
        for task, rows in path_tasks.items():
            task_paths[task].append({"opportunities": len(rows),
                                     "gold_events": sum(item["purchased"] for item in rows),
                                     "comparison_events": sum(item["comparison"] for item in rows)})

    questions = _question_average(all_rows, ("reward", "baseline", "gain", "corrected", "harmed",
                                                "purchased", "comparison"))
    influence_questions = _question_average(influence_rows, ("remaining", "changed_rate", "correction_rate",
                                                            "harm_rate", "net_rate"))
    kg_questions = _question_average(kg_rows, ("kg", "net_rate", "remaining", "changed_rate",
                                              "correction_rate", "harm_rate"))
    correlation = _correlation([row["kg"] for row in kg_questions], [row["net_rate"] for row in kg_questions])
    correlation.update({
        "repeated_feedback_events": len(kg_rows),
        "predictor": "mean logged chosen_kg per original feedback question",
        "outcome": "mean of future net_reward/remaining_questions per original feedback question",
        "cluster_order": "Average generation/order repetitions within original q before either correlation",
        "scope": "Selected-feedback descriptive association only, conditional on realized paths",
        "limitations": ["Remaining-query sets overlap and can share later feedback histories",
                        "Historical KG reference distribution differs from the remaining stream population",
                        "Acquisition selects the observed feedback events, so this is not an all-candidate calibration test",
                        "No causal mechanism or independent-sample significance claim follows from these correlations"],
    })
    if runs[0]["meta"]["stage"] == "B2":
        correlation["target_alignment"] = "MISMATCH_extra_neighbor_KG_vs_joint_main_and_neighbor_update"
        correlation["marginal_neighbor_realized_gain"] = "not_logged"
        correlation["limitations"].append(
            "B2 chosen_kg is joint-minus-main expected value, while the logged future snapshot measures the entire joint update. This association cannot calibrate marginal neighbor value."
        )
    else:
        correlation["target_alignment"] = "whole_new_feedback_bundle_with_different_reference_population"
    collection = {}
    for task, paths in sorted(task_paths.items()):
        matching = [row for row in questions if ("gsm8k" if row["task"] == "gsm8k" else "bbh/" + row["sub"]) == task]
        collection[task] = {
            "original_question_opportunities": len(matching),
            "path_count": len(paths),
            "repeated_question_opportunities": int(sum(row["opportunities"] for row in paths)),
            "gold_events_across_repeated_paths": int(sum(row["gold_events"] for row in paths)),
            "comparison_events_across_repeated_paths": int(sum(row["comparison_events"] for row in paths)),
            "mean_gold_events_per_path": float(np.mean([row["gold_events"] for row in paths])),
            "question_balanced_purchase_frequency": float(np.mean([row["purchased"] for row in matching])),
            "question_balanced_comparison_frequency": float(np.mean([row["comparison"] for row in matching])),
        }
    future_summary = {"original_feedback_questions": len(influence_questions),
                      "repeated_feedback_events": len(influence_rows),
                      "snapshot_event_totals": dict(raw_snapshot_totals),
                      "scope": "One-update frozen before/after predictions over remaining queries, without subsequent updates",
                      "additive_final_path_attribution": False,
                      "warning": "Snapshot opportunities overlap. Their corrections/harm cannot be added into unique corrected questions or final path gain."}
    for field in ("changed_rate", "correction_rate", "harm_rate", "net_rate"):
        future_summary["question_cluster_mean_" + field] = (
            float(np.mean([row[field] for row in influence_questions])) if influence_questions else None
        )
    return {
        "meta": {field: runs[0]["meta"][field] for field in CONTEXT_FIELDS},
        "path_count": len(runs),
        "kg_vs_future_snapshot": correlation,
        "kg_event_exclusions": dict(excluded),
        "question_clustered_kg_pairs": kg_questions,
        "future_feedback_snapshot": future_summary,
        "actual_realized_service_path": _service_summary(questions),
        "task_feedback_collection": collection,
        "high_uncertainty_low_kg_counterexamples": {
            "status": "UNAVAILABLE_NOT_LOGGED", "examples": [],
            "missing_fields": ["chosen/candidate raw value variance", "pair variance or boundary uncertainty"],
            "reason": "Existing trace records predicted_score and chosen_kg, without raw uncertainty. No uncertainty values are reconstructed or fabricated.",
        },
    }


def diagnose(runs):
    """Aggregate existing complete run dictionaries without executing any strategy."""
    groups = defaultdict(list)
    identities = set()
    excluded_delete_paths = 0
    for run in runs:
        meta = run["meta"]
        if meta.get("delete_group") is not None:
            excluded_delete_paths += 1
            continue
        if not run.get("traces"):
            raise ValueError("A diagnostic run must contain existing service traces")
        key = tuple(meta[field] for field in CONTEXT_FIELDS)
        path = key + tuple(meta[field] for field in PATH_FIELDS)
        if path in identities:
            raise ValueError("A duplicate context/fold/seed/order run would double-count repetitions")
        identities.add(path)
        groups[key].append(run)
    return {
        "schema": "e3_existing_trace_mechanism_diagnostics_v1",
        "new_strategy_runs": 0, "new_llm_calls": 0, "new_feedback_reveals": 0,
        "excluded_delete_group_paths": excluded_delete_paths,
        "contexts": [_diagnose_context(groups[key]) for key in sorted(groups)],
        "inference": "Descriptive original-question clustered associations. Generation seeds and stream orders remain repeated measurements.",
    }


def no_reuse_accounting(run, train):
    """Price identical neighbor events with two shared member outputs charged again.

    This is a ledger counterfactual, not an intervention that regenerates outputs.
    Same cached source token/wall receipts price the hypothetical repeated calls.
    Gold, aggregations, scoring, observations, service choices and quality remain
    those of the existing run. Full-information diagnostic bundles are excluded.
    """
    meta = run["meta"]
    generation = int(meta["gen"])
    if generation not in SEEDS:
        raise ValueError("Unknown generation seed")
    gen_index = SEEDS.index(generation)
    summary = run.get("ledger", {})
    accounts = summary.get("ledger")
    if accounts is None:
        if not run.get("traces"):
            raise ValueError("Run has no existing ledger")
        accounts = run["traces"][-1]["ledger"]
    baseline = deepcopy(accounts)
    if set(("service", "audit", "total")) - set(baseline):
        raise ValueError("Expected service, audit and total resource accounts")
    for account in baseline.values():
        for field in RESOURCE_KEYS:
            value = _finite(account[field], "ledger " + field)
            if value < 0:
                raise ValueError("Resource receipts must be nonnegative")

    additional = zero_resources()
    neighbor_events = 0
    overlap_cells = 0
    excluded_diagnostic_events = 0
    seen = set()
    costs = train["cost"]
    for trace in run["traces"]:
        if not trace["purchase"]:
            continue
        actions = list(map(int, trace.get("actions", [])))
        if len(actions) > 2:
            excluded_diagnostic_events += 1
            continue
        if len(actions) != 2:
            continue
        q, main = int(trace["q"]), int(trace["main"])
        if q in seen:
            raise ValueError("A neighbor feedback event was accounted twice")
        seen.add(q)
        if len(set(actions)) != 2 or main not in actions or any(action < 0 or action >= 20 for action in actions):
            raise ValueError("Malformed accepted neighbor feedback bundle")
        neighbor = next(action for action in actions if action != main)
        shared = sorted(set(TEAMS[main]) & set(TEAMS[neighbor]))
        if len(shared) != 2 or len(set(TEAMS[main]) ^ set(TEAMS[neighbor])) != 2:
            raise ValueError("Counterfactual applies only to a legal single-member replacement")
        if not trace.get("comparison", False):
            raise ValueError("Two-action accepted bundle lacks its original comparison receipt")
        if not 0 <= q < costs["member_execution_counts"].shape[1]:
            raise ValueError("Question is outside cached cost receipts")
        for suffix, field in (("execution_counts", "member_calls"),
                              ("prompt_tokens", "input_tokens"),
                              ("completion_tokens", "output_tokens"),
                              ("wall_time_s", "source_model_wall_s")):
            value = _finite(costs["member_" + suffix][gen_index, q, shared].sum(), suffix)
            if value < 0 or (suffix != "wall_time_s" and value != int(value)):
                raise ValueError("Invalid cached member cost")
            additional[field] += value if suffix == "wall_time_s" else int(value)
        neighbor_events += 1
        overlap_cells += len(shared)
    additional["tokens"] = additional["input_tokens"] + additional["output_tokens"]
    counterfactual = deepcopy(baseline)
    for account in ("audit", "total"):
        for field in RESOURCE_KEYS:
            counterfactual[account][field] += additional[field]
    saved = dict(additional)
    saved["priced_units"] = float(priced_units(additional))
    return {
        "schema": "e3_shared_member_reuse_ledger_counterfactual_v1",
        "meta": dict(meta), "neighbor_events": neighbor_events,
        "shared_member_cells_priced_again": overlap_cells,
        "excluded_full_information_events": excluded_diagnostic_events,
        "baseline_accounts": baseline, "no_reuse_accounts": counterfactual,
        "shared_member_savings": saved,
        "baseline_priced_units": float(priced_units(baseline["total"])),
        "no_reuse_priced_units": float(priced_units(counterfactual["total"])),
        "price_rule": "gold + member_calls + aggregator_calls + tokens/1000 + scoring_ops*.001",
        "unchanged": ["service choices", "feedback timing", "feedback values", "observations",
                      "aggregation outputs", "gold reveals", "scoring count", "service quality"],
        "counterfactual_scope": "Accounting only. Existing cached member receipts price the two shared members again for every accepted neighbor event.",
        "new_llm_calls": 0, "new_feedback_reveals": 0,
        "actual_reexecution": False, "new_decoding": False,
        "wall_time_scope": "Sum of cached source member wall receipts as a pricing proxy, not measured repeated execution or GPU kernel time",
        "budget_scope": "The counterfactual can exceed the original limits because it freezes behavior instead of replaying an affordable alternative policy",
    }
