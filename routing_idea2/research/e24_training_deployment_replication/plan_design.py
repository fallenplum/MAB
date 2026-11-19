"""Plan E24 power from E23 and count exact input-only execution requirements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import nct, t


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e23_deployment_prior_and_references"))
from cache_budget import AGGREGATE_TOKEN_CAP, MEMBER_TOKEN_CAP, SEEDS, TEAMS, budget


POLICIES = ("B_U", "B_P", "C_U", "C_P")
E23_NAMES = ("B_uniform", "B_prior", "C_uniform", "C_prior")
TASKS = ("gsm8k", "bbh")
ALPHA = 0.05
SIZES = (200, 400, 636, 834, 884, 1196, 1200, 1600, 2000, 2400, 3000, 3600, 3700, 5000)
GROUPED_SIZES = (1196, 1600, 2000, 2400, 3000, 3600)
PRESPECIFIED_N = 1196
EXCLUDED_SUBTASK = "penguins_in_a_table"
LIMITED_SUBTASK = "causal_judgement"
PRELIMINARY_CAPACITIES = {LIMITED_SUBTASK: 23, "snarks": 55}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_policies(packet, field, source_names):
    names = packet["names"].tolist()
    if len(names) != len(set(names)) or not set(source_names) <= set(names):
        raise ValueError("The four distinct named policies are required.")
    return np.asarray(packet[field][[names.index(name) for name in source_names]], dtype=float)


def stratify(values, masks):
    if not np.isfinite(values).all():
        raise ValueError("Planning utility must cover every original question.")
    summaries = {}
    for name, mask in masks.items():
        selected = values[mask]
        if selected.size < 2:
            raise ValueError(f"Less than two historical observations for {name}.")
        summaries[name] = dict(questions=int(selected.size), mean_pp=float(100 * selected.mean()),
                               variance=float(selected.var(ddof=1)),
                               questions_with_nonzero_interaction=int(np.count_nonzero(np.abs(selected) > 1e-12)))
    return summaries


def welch_power(strata, weights, counts, effect_pp):
    if set(strata) != set(weights) or set(strata) != set(counts) or min(counts.values()) < 2:
        raise ValueError("Every prespecified stratum requires at least two independent questions.")
    if not np.isclose(sum(weights.values()), 1, rtol=0, atol=1e-12):
        raise ValueError("The estimator's stratum weights must sum to one.")
    variance_terms = {key: weights[key] ** 2 * strata[key]["variance"] / counts[key] for key in strata}
    variance = sum(variance_terms.values())
    if variance <= 0:
        raise ValueError("Planning requires a positive estimated interaction variance.")
    df = variance**2 / sum(term**2 / (counts[key] - 1) for key, term in variance_terms.items())
    critical = t.isf(ALPHA / 2, df)
    noncentrality = (effect_pp / 100) / np.sqrt(variance)
    power = nct.sf(critical, df, noncentrality) + nct.cdf(-critical, df, noncentrality)
    return dict(standard_error_pp=float(100 * np.sqrt(variance)), welch_df=float(df), power=float(power))


def balanced_counts(total_questions):
    if total_questions % 2:
        raise ValueError("Balanced allocation needs an even total.")
    return dict.fromkeys(TASKS, total_questions // 2)


def grouped_counts(total_questions, groups, capacities):
    """Even water filling across retained BBH groups, in fixed lexical order."""
    gsm, bbh = balanced_counts(total_questions).values()
    assigned = dict.fromkeys(groups, 0)
    while sum(assigned.values()) < bbh:
        changed = False
        for key in groups:
            if assigned[key] < capacities.get(key, bbh):
                assigned[key] += 1
                changed = True
                if sum(assigned.values()) == bbh:
                    break
        if not changed:
            return None
    return {"gsm8k": gsm, **assigned}


def minimum_n(strata, weights, effect_pp, target):
    low, high = 2, 2
    while welch_power(strata, weights, balanced_counts(2 * high), effect_pp)["power"] < target:
        high *= 2
        if high > 5_000_000:
            raise RuntimeError("Planning sample size exceeds the supported range.")
    while low < high:
        middle = (low + high) // 2
        if welch_power(strata, weights, balanced_counts(2 * middle), effect_pp)["power"] >= target:
            high = middle
        else:
            low = middle + 1
    if low > 2 and welch_power(strata, weights, balanced_counts(2 * (low - 1)), effect_pp)["power"] >= target:
        raise AssertionError("The reported minimum even sample size is excessive.")
    return 2 * low


def design_row(strata, weights, counts, effects_pp):
    return dict(total_questions=sum(counts.values()), counts=counts,
                alternatives=[dict(effect_pp=effect, **welch_power(strata, weights, counts, effect))
                              for effect in effects_pp])


def summarize_support(task_labels, probabilities):
    if (probabilities.shape != (4, len(task_labels), len(TEAMS)) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=-1), 1, rtol=0, atol=1e-12)
            or set(task_labels) != set(TASKS)):
        raise ValueError("Four valid policies and both task labels are required.")
    support = probabilities > 0
    if np.any(support[1] & ~support[0]) or np.any(support[3] & ~support[2]):
        raise ValueError("Historical-prior supports must lie inside their corresponding uniform supports.")
    union = support[0] | support[2]
    team_membership = np.array([[member in team for member in range(6)] for team in TEAMS], dtype=int)
    members = (union.astype(int) @ team_membership > 0).sum(axis=1)
    combinations = union.sum(axis=1)

    def summarize(mask):
        team_sizes, team_counts = np.unique(combinations[mask], return_counts=True)
        member_sizes, member_counts = np.unique(members[mask], return_counts=True)
        num_members = int(members[mask].sum()) * len(SEEDS)
        num_aggregates = int(combinations[mask].sum()) * len(SEEDS)
        questions = int(mask.sum())
        return dict(questions=questions, member_calls=num_members, aggregation_calls=num_aggregates,
                    member_calls_per_question=num_members / questions,
                    aggregation_calls_per_question=num_aggregates / questions,
                    team_support_histogram={str(k): int(v) for k, v in zip(team_sizes, team_counts)},
                    distinct_member_histogram={str(k): int(v) for k, v in zip(member_sizes, member_counts)},
                    one_success_token_cap=num_members * MEMBER_TOKEN_CAP + num_aggregates * AGGREGATE_TOKEN_CAP)

    return dict(all_questions=summarize(np.ones(len(task_labels), dtype=bool)),
                by_task={task: summarize(task_labels == task) for task in TASKS},
                prior_deployment_extra_aggregate_keys=0, prior_deployment_extra_member_keys=0)


def cost_projection(history, n_values, member_rate, aggregate_rate):
    estimates = []
    for n in n_values:
        members = n * history["member_calls_per_question"]
        aggregates = n * history["aggregation_calls_per_question"]
        estimates.append(dict(
            questions=n, projected_member_calls=int(round(members)), projected_aggregate_calls=int(round(aggregates)),
            projected_actual_tokens_using_p2_success_means=int(round(member_rate * members + aggregate_rate * aggregates)),
            max_member_calls=18*n, max_aggregate_calls=60*n,
            one_success_theoretical_token_cap=n * (18 * MEMBER_TOKEN_CAP + 60 * AGGREGATE_TOKEN_CAP)))
    return estimates


def historical_plan(private_root, summary_path, ledger_path, capacity_path=None):
    utilities_path = private_root / "final_scoring_v1/FINAL_POLICY_UTILITIES_PRIVATE.npz"
    policies_path = private_root / "POLICIES_FROZEN.npz"
    summary = read_json(summary_path)
    ledger = read_json(ledger_path)
    combined_cost = summary["combined_execution_cost"]
    if (summary["status"] != "COMPLETE_FULL2400_REFERENCE_EVALUATION"
            or ledger["status"] != "COMPLETE_SUCCESS_AND_NATIVE_COST_CLOSED"
            or ledger["scope_canonical_sha256"] != summary["parent_scope_canonical_sha256"]
            or not combined_cost["native_token_costs_closed"]
            or ledger["failure_closure_pending"] or ledger["duplicate_paid_success_keys"]
            or sha256(policies_path) != summary["policy_sha256"]
            or sha256(ledger_path) != combined_cost["original_p2_ledger_sha256"]):
        raise ValueError("The historical evidence must match the completed P2/E23 releases.")
    with np.load(utilities_path, allow_pickle=False) as packet:
        bu, bp, cu, cp = select_policies(packet, "utilities", E23_NAMES)
        task_labels, subtask_labels = packet["task_labels"], packet["subtask_labels"]
    if (bu.shape != (len(SEEDS), 2400) or task_labels.shape != (2400,) or subtask_labels.shape != (2400,)
            or any(np.sum(task_labels == key) != 1200 for key in TASKS)):
        raise ValueError("The historical cohort must be the original balanced 2400-question E23 cohort.")
    for name, util in zip(E23_NAMES, (bu, bp, cu, cp)):
        evidence = summary["baseline_strategies"][name]
        if (not np.isfinite(util).all() or np.any(util < -1e-12) or np.any(util > 1 + 1e-12)
                or not np.isclose(100 * util.mean(), evidence["accuracy_pct"], rtol=0, atol=1e-10)
                or any(not np.isclose(100 * util[:, task_labels == task].mean(), evidence["by_task"][task],
                                      rtol=0, atol=1e-10) for task in TASKS)):
            raise ValueError("Private E23 utilities disagree with the de-identified final report.")
    per_seed = (cu - bu) - (cp - bp)
    contrasts = per_seed.mean(axis=0)
    tasks = {task: task_labels == task for task in TASKS}
    strata = stratify(contrasts, tasks)
    old_interaction_pp = (strata["gsm8k"]["mean_pp"] + strata["bbh"]["mean_pp"]) / 2
    if not np.isclose(old_interaction_pp, summary["original_gap_pp"] - summary["remaining_gap_pp"],
                      rtol=0, atol=1e-10):
        raise ValueError("The four-policy historical interaction does not reproduce E23.")
    effects = (0.5, 1.0, old_interaction_pp, 1.5, 2.0)
    weights = dict.fromkeys(TASKS, 0.5)
    if not np.isclose(welch_power(strata, weights, balanced_counts(400), 0)["power"], ALPHA,
                      rtol=0, atol=1e-10):
        raise AssertionError("The analytic two-sided power has the wrong null rejection rate.")
    groups = sorted(set(subtask_labels[tasks["bbh"]]) - {EXCLUDED_SUBTASK})
    if len(groups) != 26 or LIMITED_SUBTASK not in groups:
        raise ValueError("The historical source does not contain the specified 26-subtask common support.")
    grouped_weights = {"gsm8k": 0.5, **dict.fromkeys(groups, 0.5 / len(groups))}
    grouped_strata = stratify(contrasts, {"gsm8k": tasks["gsm8k"], **{
        group: tasks["bbh"] & (subtask_labels == group) for group in groups}})
    capacities = PRELIMINARY_CAPACITIES if capacity_path is None else read_json(capacity_path)
    if "bbh_subtask_counts" in capacities:
        source_audit = capacities
        capacities = dict(source_audit["bbh_subtask_counts"])
        if capacities.pop(EXCLUDED_SUBTASK, None) != 0 or source_audit["registered_id_or_text_intersections_remaining"] != 0:
            raise ValueError("The audited source does not match the fixed common-support design.")
    if (not set(capacities) <= set(groups) or any(type(n) is not int or n < 0 for n in capacities.values())
            or (capacity_path is not None and set(capacities) != set(groups))):
        raise ValueError("Capacity input needs one audited nonnegative integer per retained BBH subtask.")
    grouped_rows = []
    for n in GROUPED_SIZES:
        counts = grouped_counts(n, groups, capacities)
        if counts is None or min(counts.values()) < 2:
            grouped_rows.append(dict(total_questions=n, allocation_feasible=False))
        else:
            grouped_rows.append(dict(allocation_feasible=True, **design_row(grouped_strata, grouped_weights, counts, effects)))
    planned = next(row for row in grouped_rows if row["total_questions"] == PRESPECIFIED_N)
    equal_quota_cap = 2 * len(groups) * min(capacities.values())
    equal_subtask_requirement = next((design_row(grouped_strata, grouped_weights, counts, (1.0,))
        for per_group in range(2, min(capacities.values()) + 1)
        if (counts := grouped_counts(2 * len(groups) * per_group, groups, capacities)) is not None
        and welch_power(grouped_strata, grouped_weights, counts, 1.0)["power"] >= 0.8), None)

    with np.load(policies_path, allow_pickle=False) as packet:
        if not np.array_equal(packet["task_labels"], task_labels) or not np.array_equal(packet["subtask_labels"], subtask_labels):
            raise ValueError("The history policy and utility stratum order differs.")
        hist_budget = summarize_support(task_labels, select_policies(packet, "probabilities", E23_NAMES))
    history = hist_budget["all_questions"]
    member_stats, agg_stats = (ledger["stages"][key] for key in ("member_generation", "aggregation"))
    member_tokens, agg_tokens = (item["successful_prompt_tokens"] + item["successful_completion_tokens"]
                                 for item in (member_stats, agg_stats))
    if (history["member_calls"] != member_stats["unique_successful_keys"]
            or history["aggregation_calls"] != agg_stats["unique_successful_keys"]
            or member_tokens + agg_tokens != ledger["all_attempt_total_tokens"]
            or member_tokens + agg_tokens != combined_cost["original_p2_tokens"]):
        raise ValueError("The four-policy budget or actual Token ledger disagrees with P2.")
    member_rate, agg_rate = member_tokens / history["member_calls"], agg_tokens / history["aggregation_calls"]
    projections = cost_projection(history, SIZES, member_rate, agg_rate)
    for row in grouped_rows:
        row["calls_and_tokens_reference"] = next(p for p in projections if p["questions"] == row["total_questions"])
    return dict(
        schema="mab_e24_interaction_design_planning_v1", status="HISTORICAL_PLANNING_COMPLETE", two_sided_alpha=ALPHA,
        historical_questions=2400, within_question_generation_seeds=list(SEEDS),
        interaction_definition="question_mean_over_three_seeds((C_U-B_U)-(C_P-B_P))",
        old_interaction_pp=old_interaction_pp,
        historical_interaction_by_seed_pp={str(seed): float(100 * row.mean()) for seed, row in zip(SEEDS, per_seed)},
        historical_strata=strata, weight_by_task=weights,
        balanced_effective_variance=sum(row["variance"] for row in strata.values()) / 2,
        planning_power_method="Welch-Satterthwaite noncentral t, historical paired variances, a single primary two-sided test",
        power_grid=[design_row(strata, weights, balanced_counts(n), effects) for n in SIZES],
        minimum_balanced_samples=[dict(effect_pp=effect, power80_n=minimum_n(strata, weights, effect, 0.8),
                                       power90_n=minimum_n(strata, weights, effect, 0.9)) for effect in effects],
        common_support_design=dict(excluded_bbh_subtask=EXCLUDED_SUBTASK, weight_by_stratum=grouped_weights,
                                   capacity_assumption=capacities,
                                   capacity_sha256=None if capacity_path is None else sha256(capacity_path),
                                   capacity_status="Conditional examples based on prior admission counts."
                                   if capacity_path is None else "Count bounds supplied by the separate source audit.",
                                   balanced_equal_subtask_upper_bound=equal_quota_cap,
                                   prespecified_total_questions=PRESPECIFIED_N,
                                   prespecified_allocations_feasible=planned["allocation_feasible"],
                                   planning_observed_alternative_pp=old_interaction_pp,
                                   smallest_meaningful_alternative_pp=1.0,
                                   equal_subtask_1pp_80_percent_minimum=equal_subtask_requirement,
                                   historical_strata=grouped_strata,
                                   historical_interaction_pp=sum(grouped_weights[k] * row["mean_pp"]
                                                                 for k, row in grouped_strata.items()), power_grid=grouped_rows),
        historical_four_policy_support=hist_budget,
        historical_p2_actual_tokens=dict(member_generation=member_tokens, aggregation=agg_tokens,
                                         total=member_tokens + agg_tokens, member_per_call=member_rate,
                                         aggregate_per_call=agg_rate),
        historical_e23_extra_reference_tokens=combined_cost["supplementary_tokens"],
        projections=projections,
        source_sha256=dict(utilities=sha256(utilities_path), policies=sha256(policies_path),
                           historical_public_summary=sha256(summary_path), historical_public_ledger=sha256(ledger_path),
                           planning_source=sha256(Path(__file__))),
        native_aggregation_calls_started=0, member_generation_calls_started=0, new_reference_answers_read=0,
        confidence_intervals_computed=False, bootstrap_replicates=0)


def input_budget(args):
    prediction_hash = sha256(args.predictions)
    if prediction_hash != args.prediction_sha256:
        raise ValueError("Prediction file bytes differ from the frozen receipt.")
    with np.load(args.predictions, allow_pickle=False) as packet:
        if set(packet["names"].tolist()) != set(POLICIES):
            raise ValueError("The prediction archive requires exactly the four frozen E24 policies.")
        probabilities = select_policies(packet, "probabilities", POLICIES)
        question_ids = packet["question_ids"].tolist()
        canonical = packet["canonical_question_sha256"].tolist()
        task_labels = packet["task_labels"]
    if (len(set(question_ids)) != len(question_ids) or len(set(canonical)) != len(question_ids)
            or len(canonical) != len(question_ids) or task_labels.shape != (len(question_ids),)
            or any(not isinstance(question, str) or not question for question in question_ids)
            or any(not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef")
                   for digest in canonical)):
        raise ValueError("Each input requires a unique original identity and canonical question SHA256.")
    result = dict(schema="mab_e24_input_only_support_budget_v1", prediction_sha256=prediction_hash,
                  generation_seeds=list(SEEDS), cache_credits_applied=False,
                  **summarize_support(task_labels, probabilities))
    if args.private_output is not None:
        public, missing = budget(question_ids, dict(zip(POLICIES, probabilities)), (), ())
        union = public["union"]
        if ((union["missing_member_calls"], union["missing_aggregate_calls"]) !=
                (result["all_questions"]["member_calls"], result["all_questions"]["aggregation_calls"])):
            raise AssertionError("Exact keyed and aggregate budget counts disagree.")
        write_json(args.private_output, dict(prediction_sha256=prediction_hash, from_zero=True,
                                             **missing["union"]))
    write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("historical")
    plan.add_argument("--e23-private", type=Path, required=True)
    plan.add_argument("--e23-summary", type=Path, default=HERE.parent / "e23_deployment_prior_and_references/RESULTS_FINAL.json")
    plan.add_argument("--p2-ledger", type=Path, default=HERE.parent / "e20_p2_execution/P2_FINAL_EXECUTION_LEDGER.json")
    plan.add_argument("--bbh-capacities", type=Path,
                      help="Source audit or JSON object with available counts for all 26 retained subtasks")
    plan.add_argument("--output", type=Path, required=True)
    calculate = commands.add_parser("input-budget")
    calculate.add_argument("--predictions", type=Path, required=True)
    calculate.add_argument("--prediction-sha256", required=True)
    calculate.add_argument("--output", type=Path, required=True)
    calculate.add_argument("--private-output", type=Path)
    args = parser.parse_args()
    if args.command == "historical":
        write_json(args.output, historical_plan(args.e23_private, args.e23_summary, args.p2_ledger,
                                                args.bbh_capacities))
    else:
        input_budget(args)


if __name__ == "__main__":
    main()
