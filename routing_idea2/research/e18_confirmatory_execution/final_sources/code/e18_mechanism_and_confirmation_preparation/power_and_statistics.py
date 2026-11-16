"""Frozen two-sided, original-question paired inference and power planning."""
from __future__ import annotations

import argparse
import math
import time

from e18_common import E17, PUBLIC, PRIVATE, preserve_e17, read_json, sha256, write_json
import numpy as np
from scipy import stats

N_GRID = (400, 1200, 2400, 3000, 3700, 4800, 6000, 10000)
EFFECTS_PP = (0.5, 1.0, 1.5, 2.0)
ALPHA = 0.05


def paired_question_test(candidate, baseline, task_labels, alpha=ALPHA):
    """A fixed-time task-stratified paired t test for original-question means.

    Each input has shape (generation, original_question). The three generation
    outcomes are averaged before computing the independent-question variances.
    This pure function performs no data access or permission change.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    labels = np.asarray(task_labels)
    if candidate.shape != baseline.shape or candidate.ndim != 2 or candidate.shape[0] != 3:
        raise ValueError("Require matched three-generation by question reward arrays")
    if labels.shape != (candidate.shape[1],) or set(labels) != {"gsm8k", "bbh"}:
        raise ValueError("Require both GSM8K and BBH question labels")
    if not (np.isin(candidate, (0, 1)).all() and np.isin(baseline, (0, 1)).all()):
        raise ValueError("Final aggregate rewards must be binary")
    if not 0 < alpha < 1:
        raise ValueError("Invalid significance level")
    differences = (candidate - baseline).mean(axis=0)
    groups = {}
    components = []
    for task in ("gsm8k", "bbh"):
        values = differences[labels == task]
        if len(values) < 2:
            raise ValueError("Each task needs at least two independent questions")
        variance = float(values.var(ddof=1))
        component = 0.25 * variance / len(values)
        components.append((component, len(values)-1))
        groups[task] = dict(questions=len(values), mean=float(values.mean()), variance=variance)
    estimate = sum(0.5 * group["mean"] for group in groups.values())
    variance = sum(component for component, _ in components)
    if variance == 0:
        # A constant observed difference does not establish zero population
        # variance. Preserve a valid bounded-support uncertainty output.
        return dict(estimate=estimate, estimate_pp=100*estimate,
                    confidence_interval=[-1.0, 1.0], confidence_level=1-alpha,
                    p_value=None, status="DEGENERATE_VARIANCE_REQUIRES_REVIEW",
                    task_groups=groups, original_questions=len(differences))
    standard_error = math.sqrt(variance)
    degrees = variance**2 / sum(component**2 / df for component, df in components)
    critical = float(stats.t.ppf(1-alpha/2, degrees))
    statistic = estimate / standard_error
    interval = [max(-1.0, estimate-critical*standard_error),
                min(1.0, estimate+critical*standard_error)]
    return dict(estimate=estimate, estimate_pp=100*estimate,
                standard_error=standard_error, degrees_of_freedom=degrees,
                confidence_interval=interval, confidence_level=1-alpha,
                p_value=float(2*stats.t.sf(abs(statistic), degrees)),
                statistic=statistic, task_groups=groups,
                original_questions=len(differences), generation_records=candidate.size,
                method="equal-task-weighted paired t with Satterthwaite df",
                guarantee="model-based independent-question inference, not unconditional finite-sample coverage")


def power(n_questions, effect_pp, variances, variance_multiplier=1.0):
    if n_questions < 4 or n_questions % 2:
        raise ValueError("Balanced question count must be even and at least four")
    if effect_pp < 0 or variance_multiplier <= 0:
        raise ValueError("Use an absolute nonnegative effect and positive variance multiplier")
    per_task = n_questions // 2
    components = [0.25 * float(variances[task]) * variance_multiplier / per_task
                  for task in ("gsm8k", "bbh")]
    variance = sum(components)
    if variance <= 0:
        raise ValueError("Positive planning variance required")
    df = variance**2 / sum(value**2 / (per_task-1) for value in components)
    critical = float(stats.t.ppf(1-ALPHA/2, df))
    ncp = (effect_pp / 100) / math.sqrt(variance)
    value = stats.nct.sf(critical, df, ncp) + stats.nct.cdf(-critical, df, ncp)
    return float(value)


def required_n(effect_pp, variances, target_power):
    if effect_pp <= 0 or not 0 < target_power < 1:
        raise ValueError("Positive effect and valid target power required")
    lower, upper = 2, 4
    while power(upper, effect_pp, variances) < target_power:
        upper *= 2
    while upper-lower > 2:
        midpoint = ((lower+upper)//4)*2
        if power(midpoint, effect_pp, variances) >= target_power:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def qualify():
    rng = np.random.default_rng(20261116018)
    labels = np.array(["gsm8k"]*50+["bbh"]*50)
    baseline = rng.binomial(1, .7, size=(3, 100))
    candidate = baseline.copy()
    candidate[:, :5] = 1
    candidate[:, 50:54] = 1
    forward = paired_question_test(candidate, baseline, labels)
    reverse = paired_question_test(baseline, candidate, labels)
    if abs(forward["estimate"]+reverse["estimate"]) > 1e-15:
        raise AssertionError("Paired sign symmetry failed")
    if abs(forward["p_value"]-reverse["p_value"]) > 1e-15:
        raise AssertionError("Two-sided p symmetry failed")
    if not np.allclose(forward["confidence_interval"], -np.array(reverse["confidence_interval"])[::-1]):
        raise AssertionError("Paired interval sign symmetry failed")
    if forward["original_questions"] != 100 or forward["generation_records"] != 300:
        raise AssertionError("Original-question unit changed")
    if paired_question_test(baseline, baseline, labels)["p_value"] is not None:
        raise AssertionError("Degenerate variance needs explicit review")
    example = dict(gsm8k=.03, bbh=.05)
    if power(3000, 1, example) <= power(1200, 1, example):
        raise AssertionError("Power should increase with sample count")
    if power(3000, 1, example, 2) >= power(3000, 1, example):
        raise AssertionError("Variance sensitivity failed")
    n = required_n(1, example, .8)
    if power(n, 1, example) < .8 or power(n-2, 1, example) >= .8:
        raise AssertionError("Balanced integer sample search failed")
    return dict(status="PASS", checks=8, real_evaluation_rewards_read=0,
                synthetic_intervals_only=True, fresh_rewards_read=0, sealed_payloads_read=0,
                note="Pure numerical qualification does not establish real-data coverage")


def main():
    start = time.perf_counter()
    preserve_e17()
    source = E17 / "public/CONFIRMATION_POWER.json"
    previous = read_json(source)["TSPO_vs_SPO"]
    variances = {task: previous["groups"][task]["sample_variance_ddof1"]
                 for task in ("gsm8k", "bbh")}
    rows = []
    for n in N_GRID:
        for effect in EFFECTS_PP:
            rows.append(dict(n_questions=n, questions_per_task=n//2,
                absolute_effect_pp=effect,
                power=power(n, effect, variances),
                variance_x1_5_power=power(n, effect, variances, 1.5),
                variance_x2_power=power(n, effect, variances, 2.0)))
    targets = [dict(effect_pp=effect,
                    n_80=required_n(effect, variances, .8),
                    n_90=required_n(effect, variances, .9)) for effect in EFFECTS_PP]
    result = dict(schema="E18_two_sided_question_paired_power_v1", status="PLANNING_ONLY",
        primary_comparison="TSPO_uniform minus SPO_coordinate",
        null="equal expected final aggregate accuracy", alternative="two-sided nonzero difference",
        alpha=ALPHA, confidence_level=.95, primary_target_n=3000,
        primary_minimum_effect_pp=1.0, sensitivity_effect_pp=.5,
        target_task_weights=dict(gsm8k=.5, bbh=.5), generation_seeds=[42,137,2024],
        source_variances=variances, source_sigma=previous["sigma_question_mean"],
        source_sha256=sha256(source), source_scope="E17 frozen old600 development OOF question-level paired variance",
        grid=rows, sample_size_targets=targets,
        evaluation_method="task-stratified paired t; question means of three generation outcomes; Satterthwaite df",
        interval_type="fixed-time 95% two-sided question-level paired t interval",
        secondary_comparisons=["TSPO-Ridge", "SPO-Ridge", "TSPO-GlobalFixed", "SPO-GlobalFixed", "Ridge-GlobalFixed"],
        secondary_multiplicity="Holm within the five predeclared secondary comparisons; no promotion to primary",
        fixed_horizon="N fixed before any fresh reward access; planning N grid is not a set of interim looks",
        policy_override=dict(effective_date="2026-11-16", authority="explicit E18 user request",
                             scope="E18 independent confirmation two-sided test and paired interval plan; E17 unchanged"),
        assumptions=["Original questions are the independent units; generation repeats remain clustered.",
                     "Development paired variance transports to the specified new question source frame.",
                     "OOF480 heads provide planning variance for distinct frozen full600 evaluation heads.",
                     "Source-frame expansion and task/subtask composition can alter the variance.",
                     "Noncentral-t power with plug-in task variances is a planning model.",
                     "Detecting a nonzero gain differs from proving a lower bound above one percentage point."],
        future_evaluation_executed=False, fresh_reward_reads=0, sealed_payloads_read=0,
        qualification=qualify(), wall_s=time.perf_counter()-start)
    write_json(PUBLIC / "E18_POWER_ANALYSIS.json", result)
    write_json(PRIVATE / "STATISTICAL_SOFTWARE_QUALIFICATION.json", result["qualification"])
    print("E18_POWER_AND_STATISTICS_FROZEN", flush=True)


if __name__ == "__main__":
    main()
