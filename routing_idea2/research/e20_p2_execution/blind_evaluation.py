"""P2 frozen-policy scope checks and paired evaluation of paid sparse outputs.

This module performs no inference, training, reference access or reward loading
at import. Its default CLI qualifies input-only predictions. The separate score
projection entry consumes an already authorized deterministic-score JSONL, and
requires a complete-output freeze receipt before opening that file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


MODEL_SHA256 = "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92"
OPERATOR_SHA256 = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
SEEDS = (42, 137, 2024)
TAU = 1e-8
TEAMS = tuple(combinations(range(6), 3))
FORBIDDEN = ("reward", "answer", "gold", "target", "reference")


def file_sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        if any(term in name.casefold() for name in archive.files for term in FORBIDDEN):
            raise PermissionError("Reward-bearing data cannot enter the prediction interface")
        required = {"question_ids", "canonical_hashes", "task_labels", "subtask_labels",
                    "scores", "probabilities", "raw_exact_probabilities"}
        if not required <= set(archive.files):
            raise ValueError("Frozen input-only prediction fields are missing")
        return {name: archive[name].copy() for name in required}


def decoder(scores, tolerance):
    selected = scores >= scores.max(axis=-1, keepdims=True) - tolerance
    return selected / selected.sum(axis=-1, keepdims=True)


def required_union(ids, probabilities, raw_probabilities):
    support = np.any((probabilities > 0) | (raw_probabilities > 0), axis=0)
    aggregates, members = set(), set()
    for index, question_id in enumerate(ids):
        for action in np.flatnonzero(support[index]):
            for seed in SEEDS:
                aggregates.add((str(question_id), seed, int(action)))
                members.update((str(question_id), seed, member) for member in TEAMS[action])
    return support, aggregates, members


def qualify_scope(scope, packet, *, expected_questions=2400):
    ids = packet["question_ids"].astype(str)
    canonical = packet["canonical_hashes"].astype(str)
    tasks = packet["task_labels"].astype(str)
    subtasks = packet["subtask_labels"].astype(str)
    count = len(ids)
    if expected_questions is not None and count != expected_questions:
        raise ValueError("The frozen independent cohort size differs")
    if len(set(ids)) != count or len(set(canonical)) != count:
        raise ValueError("Repeated independent input identity")
    if any(len(value) != 64 for value in canonical):
        raise ValueError("Canonical input hashes are required")
    scores = np.asarray(packet["scores"], dtype=np.float64)
    probabilities = np.asarray(packet["probabilities"], dtype=np.float64)
    raw = np.asarray(packet["raw_exact_probabilities"], dtype=np.float64)
    if scores.shape != (2, count, 20) or probabilities.shape != scores.shape or raw.shape != scores.shape:
        raise ValueError("Exactly two frozen20-action heads are required")
    if not np.isfinite(scores).all() or not np.isfinite(probabilities).all() or not np.isfinite(raw).all():
        raise ValueError("Frozen predictions must be finite")
    if not np.array_equal(probabilities, decoder(scores, TAU)):
        raise ValueError("Primary probabilities differ from the frozen tau decoder")
    if not np.array_equal(raw, decoder(scores, 0)):
        raise ValueError("Raw-exact sensitivity probabilities differ")
    model_sha = scope.get("frozen_model_sha256", scope.get("final_model_sha256",
        scope.get("model_sha256", scope.get("bindings", {}).get("final_model_sha256"))))
    if model_sha != MODEL_SHA256:
        raise ValueError("P2 must use the previously frozen final600 models")
    if (scope.get("operator_fingerprint") != OPERATOR_SHA256
            or tuple(scope.get("generation_seeds", ())) != SEEDS
            or scope.get("aggregation_seed") != 42):
        raise ValueError("Frozen aggregation or generation identity differs")
    if "frozen_policy_probabilities" in scope and not np.array_equal(
            np.asarray(scope["frozen_policy_probabilities"]), probabilities):
        raise ValueError("Runtime policy probabilities differ from the frozen prediction packet")
    questions = scope["question_keys"]
    if [row["question_id"] for row in questions] != ids.tolist():
        raise ValueError("Frozen scope/prediction order differs")
    if [row["canonical_question_sha256"] for row in questions] != canonical.tolist():
        raise ValueError("Frozen scope/input canonical identity differs")
    if [row["task"] for row in questions] != tasks.tolist():
        raise ValueError("Input-only task metadata differs")
    counts = Counter(tasks)
    if set(counts) != {"gsm8k", "bbh"} or counts["gsm8k"] != counts["bbh"]:
        raise ValueError("Both tasks must retain the approved50percent allocation")
    if expected_questions == 2400:
        subcounts = Counter(subtasks[tasks == "bbh"])
        if len(subcounts) != 27 or Counter(subcounts.values()) != {44: 15, 45: 12}:
            raise ValueError("The predeclared27-subtask44/45 quotas differ")
        if subcounts["penguins_in_a_table"] != 44:
            raise ValueError("The predeclared rare-subtask quota differs")
    support, aggregate_keys, member_keys = required_union(ids, probabilities, raw)
    stored_aggregates = list(map(tuple, scope["required_aggregate_keys"]))
    if set(stored_aggregates) != aggregate_keys or len(stored_aggregates) != len(aggregate_keys):
        raise ValueError("Aggregation execution union differs from frozen support")
    if "required_member_keys" in scope:
        stored_members = list(map(tuple, scope["required_member_keys"]))
        if set(stored_members) != member_keys or len(stored_members) != len(member_keys):
            raise ValueError("Member execution union differs from legal reuse")
    forbidden_ids = {row["question_id"] for row in scope.get("sealed_exclusions", [])}
    forbidden_hashes = {row["canonical_question_sha256"] for row in scope.get("sealed_exclusions", [])}
    if set(ids) & forbidden_ids or set(canonical) & forbidden_hashes:
        raise PermissionError("Sealed identity entered the independent reference scope")
    return dict(status="PASS_INPUT_ONLY", independent_questions=count, task_counts=dict(counts),
                support_union_teams=int(support.sum()), required_aggregate_calls=len(aggregate_keys),
                required_member_calls=len(member_keys), new_reference_values_decoded=0,
                new_llm_calls=0, primary_tau=TAU, generation_seeds=list(SEEDS))


def paired_point_test(utilities, tasks):
    utilities = np.asarray(utilities, dtype=np.float64)
    tasks = np.asarray(tasks)
    if utilities.shape != (2, 3, len(tasks)) or not np.isfinite(utilities).all():
        raise ValueError("Two three-seed question-cluster utility arrays are required")
    if np.any((utilities < -1e-12) | (utilities > 1 + 1e-12)):
        raise ValueError("Expected correctness must be in[0,1]")
    difference = (utilities[1] - utilities[0]).mean(axis=0)
    components, groups = [], {}
    for task in ("gsm8k", "bbh"):
        mask = tasks == task
        values = difference[mask]
        if len(values) < 2:
            raise ValueError("Each task requires two independent questions")
        variance = float(values.var(ddof=1))
        components.append((0.25 * variance / len(values), len(values) - 1))
        groups[task] = dict(questions=int(mask.sum()),
                            B_accuracy_pct=float(100 * utilities[0][:, mask].mean()),
                            C_accuracy_pct=float(100 * utilities[1][:, mask].mean()),
                            difference_pp=float(100 * values.mean()), sample_variance=variance)
    estimate = sum(group["difference_pp"] for group in groups.values()) / 2
    sampling_variance = sum(value for value, _ in components)
    result = dict(difference_pp=estimate, original_questions=len(tasks),
                  recorded_generation_events_per_policy=3 * len(tasks), task_groups=groups,
                  method="Equal-task paired question mean, Satterthwaite two-sided t",
                  alpha=0.05, primary_comparisons=1, confidence_intervals_computed=False,
                  bootstrap_repetitions=0)
    if sampling_variance <= 0:
        return dict(result, status="DEGENERATE_VARIANCE_REQUIRES_REVIEW", p_value=None)
    standard_error = math.sqrt(sampling_variance)
    degrees = sampling_variance**2 / sum(value**2 / df for value, df in components)
    statistic = (estimate / 100) / standard_error
    return dict(result, status="COMPUTED", standard_error_pp=100 * standard_error,
                degrees_of_freedom=degrees, t_statistic=statistic,
                p_value=float(2 * stats.t.sf(abs(statistic), degrees)))


def expected_corrections(probabilities, rewards):
    """E19 maximal common-action coupling, with independent residual draws."""
    common = np.minimum(probabilities[0], probabilities[1])
    residual_b, residual_c = probabilities[0] - common, probabilities[1] - common
    mass = residual_b.sum(axis=1)
    corrected = np.zeros(rewards.shape[:2])
    damaged = np.zeros_like(corrected)
    active = mass > 1e-14
    for seed_index in range(3):
        reward = rewards[seed_index]
        corrected[seed_index, active] = (
            (residual_b[active] * (1 - reward[active])).sum(axis=1)
            * (residual_c[active] * reward[active]).sum(axis=1) / mass[active]
        )
        damaged[seed_index, active] = (
            (residual_b[active] * reward[active]).sum(axis=1)
            * (residual_c[active] * (1 - reward[active])).sum(axis=1) / mass[active]
        )
    return corrected, damaged


def project_scores(packet, score_rows, excluded_ids=()):
    """Project only paid support scores into frozen randomized-policy utilities."""
    ids = packet["question_ids"].astype(str)
    tasks = packet["task_labels"].astype(str)
    subtasks = packet["subtask_labels"].astype(str)
    probabilities = packet["probabilities"]
    raw = packet["raw_exact_probabilities"]
    support, required, _ = required_union(ids, probabilities, raw)
    excluded = set(excluded_ids)
    if not excluded <= set(ids):
        raise ValueError("Unknown question excluded from evaluation")
    positions = {value: index for index, value in enumerate(ids)}
    values = np.zeros((3, len(ids), 20), dtype=np.float64)
    seen = set()
    for row in score_rows:
        key = (str(row["question_id"]), int(row["generation_seed"]), int(row["combination_index"]))
        if key not in required or key in seen or key[0] in excluded:
            raise ValueError("Unexecuted, duplicated or excluded score entered evaluation")
        if row["score"] not in (0, 1):
            raise ValueError("Frozen correctness labels must be binary")
        values[SEEDS.index(key[1]), positions[key[0]], key[2]] = row["score"]
        seen.add(key)
    needed = {key for key in required if key[0] not in excluded}
    if seen != needed:
        raise ValueError("Executed support scores are incomplete")
    valid = np.array([value not in excluded for value in ids])
    main_utility = np.einsum("mqa,sqa->msq", probabilities[:, valid], values[:, valid])
    raw_utility = np.einsum("mqa,sqa->msq", raw[:, valid], values[:, valid])
    corrections, damages = expected_corrections(probabilities[:, valid], values[:, valid])
    if not np.allclose(corrections-damages, main_utility[1]-main_utility[0], atol=1e-12, rtol=0):
        raise AssertionError("Correction accounting fails the frozen utility difference")
    point = paired_point_test(main_utility, tasks[valid])
    raw_point = paired_point_test(raw_utility, tasks[valid])
    ordered = np.sort(packet["scores"], axis=2)
    margins = ordered[:, :, -1] - ordered[:, :, -2]
    strict_both = np.all(margins > TAU, axis=0)
    strict_changed = strict_both & (packet["scores"][0].argmax(axis=1) != packet["scores"][1].argmax(axis=1))
    strict_valid = strict_changed[valid]
    by_subtask = []
    for subtask in sorted(set(subtasks[(tasks == "bbh") & valid])):
        selected = (subtasks[valid] == subtask) & (tasks[valid] == "bbh")
        by_subtask.append(dict(subtask=str(subtask), questions=int(selected.sum()),
            B_accuracy_pct=float(100 * main_utility[0][:, selected].mean()),
            C_accuracy_pct=float(100 * main_utility[1][:, selected].mean()),
            difference_pp=float(100 * (main_utility[1][:, selected]-main_utility[0][:, selected]).mean())))
    missing_bounds = []
    for task in ("gsm8k", "bbh"):
        original = tasks == task
        known = original & valid
        unknown = original & ~valid
        known_sum = float((main_utility[1][:, tasks[valid] == task]-main_utility[0][:, tasks[valid] == task]).mean(axis=0).sum())
        tv = 0.5 * np.abs(probabilities[1, unknown] - probabilities[0, unknown]).sum(axis=1)
        denominator = int(original.sum())
        missing_bounds.append(((known_sum-tv.sum())/denominator, (known_sum+tv.sum())/denominator))
    centered = packet["scores"] - packet["scores"].mean(axis=2, keepdims=True)
    report = dict(schema="e20_p2_frozen_probability_evaluation_v1", primary=point,
        raw_exact_sensitivity=raw_point, primary_task_weights={"gsm8k":0.5,"bbh":0.5},
        requested_questions=len(ids), valid_questions=int(valid.sum()), excluded_questions=int((~valid).sum()),
        expected_corrected_generation_events=float(corrections.sum()),
        expected_damaged_generation_events=float(damages.sum()),
        expected_net_corrections=float((corrections-damages).sum()),
        strict_both_unique_questions=int(strict_both.sum()), strict_changed_questions=int(strict_changed.sum()),
        strict_valid_questions=int(strict_valid.sum()),
        strict_corrected_events=int(np.sum((main_utility[1][:,strict_valid] == 1)&(main_utility[0][:,strict_valid] == 0))),
        strict_damaged_events=int(np.sum((main_utility[1][:,strict_valid] == 0)&(main_utility[0][:,strict_valid] == 1))),
        centered_score_changed_questions_gt_1e_minus7=int(np.sum(np.max(np.abs(centered[1]-centered[0]),axis=1)>1e-7)),
        bbh_subtasks=by_subtask,
        bbh_subtask_macro_difference_pp=float(np.mean([row["difference_pp"] for row in by_subtask])),
        by_generation_seed=[dict(generation_seed=seed, difference_pp=float(sum(
            50 * (main_utility[1,index,tasks[valid]==task]-main_utility[0,index,tasks[valid]==task]).mean()
            for task in ("gsm8k","bbh")))) for index,seed in enumerate(SEEDS)],
        deterministic_missing_outcome_bounds_pp=[50*sum(row[column] for row in missing_bounds) for column in (0,1)],
        missing_bound_scope="Bounded missing outcomes at original task denominators; not a confidence interval",
        new_llm_calls=0, new_model_fits=0, confidence_intervals_computed=False, bootstrap_repetitions=0)
    arrays = dict(utilities=main_utility, raw_exact_utilities=raw_utility, valid_mask=valid,
                  expected_corrections=corrections, expected_damages=damages,
                  strict_both_unique=strict_both, strict_changed=strict_changed)
    return report, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-projection", action="store_true")
    parser.add_argument("--score-rows", type=Path)
    parser.add_argument("--outputs-frozen", type=Path)
    args = parser.parse_args()
    scope = read_json(args.scope)
    packet = load_predictions(args.predictions)
    qualified = qualify_scope(scope, packet)
    if not args.execute_projection:
        write_json(args.output, qualified)
        print(json.dumps(qualified, ensure_ascii=False))
        return
    if args.score_rows is None or args.outputs_frozen is None:
        parser.error("Projection requires authorized deterministic scores and a complete output freeze")
    freeze = read_json(args.outputs_frozen)
    if (freeze.get("status") != "ALL_REQUIRED_OUTPUTS_FROZEN"
            or freeze.get("scope_sha256") != file_sha(args.scope)
            or freeze.get("prediction_sha256") != file_sha(args.predictions)
            or freeze.get("final_model_sha256") != MODEL_SHA256):
        raise PermissionError("A complete matching frozen execution is required before score access")
    expected_count = qualified["required_aggregate_calls"]
    if freeze.get("completed_required_aggregate_calls") != expected_count:
        raise PermissionError("Incomplete outputs cannot enter final evaluation")
    # The score file is first opened only after every input/output freeze check.
    with args.score_rows.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    report, arrays = project_scores(packet, rows, freeze.get("unscorable_question_ids", []))
    report["scope_sha256"] = file_sha(args.scope)
    report["prediction_sha256"] = file_sha(args.predictions)
    report["score_rows_sha256"] = file_sha(args.score_rows)
    write_json(args.output, report)
    with args.output.with_suffix(".private.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps({"status":"POINT_ANALYSIS_COMPLETE", "questions":report["valid_questions"],
                      "primary_difference_pp":report["primary"]["difference_pp"],
                      "new_llm_calls":0,"bootstrap_repetitions":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
