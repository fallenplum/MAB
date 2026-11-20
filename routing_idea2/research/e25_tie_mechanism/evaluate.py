"""Evaluate frozen E25 score arrays against E24's already closed support rewards."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e24_training_deployment_replication"))
import replication as original
import numpy as np
from blind_evaluation import expected_corrections


TEAMS = tuple(combinations(range(6), 3))
PROTOCOL_SHA = "6c30f10b6ac417e04b33170d8088230dc07d3c4893b65f2a51283ab216dca8f1"
PREDICTIONS_SHA = "9595ee1ddf9e30e85a0c95e51af2155e6b3de4cbd36265114341f32f9c324cef"
RECEIPT_SHA = "9459347fd0a291511f12827ea729bf56310ca44f192cdda01a23500f28e7b393"
SEEDS = tuple(original.SEEDS)
CONTRASTS = {
    "C_U_minus_B_U": (-1, 0, 1, 0),
    "C_P_minus_B_P": (0, -1, 0, 1),
    "interaction": (-1, 1, 1, -1),
}
NAMES = tuple(original.NAMES)
RESULT_LIMIT = 1e-14


def fixed_assets(args):
    for path, digest in ((args.e24_protocol, PROTOCOL_SHA), (args.predictions, PREDICTIONS_SHA),
                         (args.scoring_receipt, RECEIPT_SHA)):
        if original.sha(path) != digest:
            raise ValueError(f"Frozen E24 binding differs: {path.name}")
    receipt = original.read(args.scoring_receipt)
    if receipt["score_sha256"] != original.sha(args.scores) or receipt["rows_scored"] != 53661:
        raise ValueError("The score cache differs from the closed E24 support")
    with np.load(args.predictions, allow_pickle=False) as packet:
        fixed = {k: packet[k] for k in ("names", "scores", "probabilities", "combo_success_counts",
                                        "question_ids", "task_labels", "subtask_labels")}
    if tuple(fixed["names"]) != NAMES or not np.array_equal(
        original.four_policies(fixed["scores"], fixed["combo_success_counts"]), fixed["probabilities"]
    ):
        raise ValueError("Archived probabilities differ from the fixed decoding rules")
    weights = original.group_weights(original.labels_for(fixed["task_labels"], fixed["subtask_labels"]),
                                     original.read(args.e24_protocol))
    return fixed, weights


def load_candidates(path, fixed, expected_cells, seeds=None):
    with np.load(path, allow_pickle=False) as packet:
        scores = packet["scores"]
        if scores.shape != (expected_cells, 2, len(fixed["question_ids"]), 20) or not np.isfinite(scores).all():
            raise ValueError(f"Wrong head-score shape in {path.name}")
        if not np.array_equal(packet["question_ids"], fixed["question_ids"]):
            raise ValueError(f"Question order differs in {path.name}")
        if seeds is not None and packet["seeds"].tolist() != seeds:
            raise ValueError("MLP optimizer seeds differ from the frozen protocol")
        return scores


def weighted_mean(values, strata, weights):
    return sum(w * float(np.mean(values[..., strata == group])) for group, w in weights.items())


def group_mean_points(z, task_labels, strata, weights):
    return dict(overall_pp=100 * weighted_mean(z, strata, weights),
                task_pp={name: 100 * float(z[..., task_labels == name].mean()) for name in ("gsm8k", "bbh")},
                seed_pp={str(seed): 100 * weighted_mean(z[i], strata, weights) for i, seed in enumerate(SEEDS)})


def measure_uncertainty(events, task_labels, strata, weights):
    points = group_mean_points(events, task_labels, strata, weights)
    questions = events.mean(axis=0)
    parts = []
    for name, weight in weights.items():
        z = questions[strata == name]
        parts.append(weight**2 * float(z.var(ddof=1)) / len(z))
    points["question_paired_se_pp"] = 100 * float(np.sqrt(sum(parts)))
    seed = list(points["seed_pp"].values())
    points["recorded_seed_range_pp"] = [min(seed), max(seed)]
    leave_one_out = [float(np.mean([point for j, point in enumerate(seed) if i != j])) for i in range(len(seed))]
    points["drop_one_recorded_seed_range_pp"] = [min(leave_one_out), max(leave_one_out)]
    return points


def partial_events(coefficient, values):
    known = np.isfinite(values)
    observed = np.where(known, values * coefficient, 0).sum(axis=-1)
    missing = np.where(known, 0, coefficient)
    return observed + np.minimum(missing, 0).sum(axis=-1), observed + np.maximum(missing, 0).sum(axis=-1)


def bounds(coefficient, values, task_labels, strata, weights):
    low, high = partial_events(coefficient, values)
    result = {"lower": group_mean_points(low, task_labels, strata, weights),
              "upper": group_mean_points(high, task_labels, strata, weights)}
    if np.max(np.abs(low - high)) <= RESULT_LIMIT:
        result["exact"] = measure_uncertainty(low, task_labels, strata, weights)
    return result


def score_diagnostics(scores, probabilities, tasks, protocol):
    unique_b = (probabilities[0] > 0).sum(axis=1) == 1
    unique_c = (probabilities[2] > 0).sum(axis=1) == 1
    strict_switch = unique_b & unique_c & (scores[0].argmax(1) != scores[1].argmax(1))
    ordered = np.sort(scores, axis=-1)
    margins = ordered[:, :, -1] - ordered[:, :, -2]
    spans = ordered[:, :, -1] - ordered[:, :, 0]
    relative = np.divide(margins, spans, out=np.zeros_like(margins), where=spans > 0)
    threshold_counts = []
    for tau in protocol["deployment"]["input_diagnostic_tolerances"]:
        unique = margins > tau
        if tau == protocol["deployment"]["tau"]:
            unique = np.stack((unique_b, unique_c))
        strict = unique.all(axis=0) & (scores[0].argmax(1) != scores[1].argmax(1))
        threshold_counts.append(dict(tau=tau, B_unique=int(unique[0].sum()), C_unique=int(unique[1].sum()),
                                     strict_switch=int(strict.sum())))
    q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    cluster_counts = {}
    for head, distribution in zip(("B", "C"), probabilities[::2]):
        supports = distribution > 0
        tied = Counter(tuple(np.flatnonzero(row).tolist()) for row in supports if row.sum() > 1)
        cluster_counts[head] = dict(
            number_of_distinct_tied_supports=len(tied),
            largest_tied_supports=[dict(actions=list(actions), questions=count) for actions, count in tied.most_common(3)],
        )
    result = dict(B_unique=int(unique_b.sum()), C_unique=int(unique_c.sum()), strict_switch=int(strict_switch.sum()),
                  margin_quantile_probabilities=q,
                  margin_quantiles={head: np.quantile(margins[i], q).tolist() for i, head in enumerate(("B", "C"))},
                  span_median={head: float(np.median(spans[i])) for i, head in enumerate(("B", "C"))},
                  relative_margin_median={head: float(np.median(relative[i])) for i, head in enumerate(("B", "C"))},
                  tied_support_patterns=cluster_counts,
                  threshold_counts=threshold_counts,
                  task={name: dict(B_unique=int(unique_b[tasks == name].sum()),
                                   C_unique=int(unique_c[tasks == name].sum()),
                                   strict_switch=int(strict_switch[tasks == name].sum()),
                                   n=int((tasks == name).sum())) for name in ("gsm8k", "bbh")})
    return result, unique_b, strict_switch


def unknown_union(support, values):
    missing = support & ~np.isfinite(values)
    return missing, dict(aggregate_keys=int(missing.sum()), questions=int(missing.any(axis=(0, 2)).sum()),
                         questions_with_complete_support=int((~missing.any(axis=(0, 2))).sum()))


def missing_cost(missing, known_members):
    member_union = set()
    for seed, question, action in zip(*np.nonzero(missing)):
        member_union.update((int(seed), int(question), member) for member in TEAMS[action])
    additional = member_union - known_members
    return dict(new_member_calls=len(additional), new_aggregate_calls=int(missing.sum()),
                existing_member_keys_reusable=len(member_union) - len(additional),
                conservative_token_cap=6144 * len(additional) + 8192 * int(missing.sum()))


def known_members_from_values(values):
    return {(int(seed), int(question), member) for seed, question, action in zip(*np.nonzero(np.isfinite(values)))
            for member in TEAMS[action]}


def evaluate_cell(scores, fixed, values, known_members, strata, weights, protocol):
    tasks = fixed["task_labels"]
    p = original.four_policies(scores, fixed["combo_success_counts"])
    diagnostics, unique_b, strict = score_diagnostics(scores, p, tasks, protocol)
    missing, coverage = unknown_union(np.any(p > 0, axis=0)[None, ...], values)
    coverage.update(missing_cost(missing, known_members))
    coverage["per_policy"] = {name: unknown_union(p[i][None, ...] > 0, values)[1]
                              for i, name in enumerate(NAMES)}
    quality = {name: bounds(p[i][None, ...], values, tasks, strata, weights) for i, name in enumerate(NAMES)}
    regions = {"B_unique": unique_b, "B_near_tied": ~unique_b, "both_strict_switch": strict}
    contrasts = {}
    for name, coefficients in CONTRASTS.items():
        effect = np.einsum("p,pqa->qa", coefficients, p)[None, ...]
        entry = bounds(effect, values, tasks, strata, weights)
        entry["regions"] = {label: bounds(effect * mask[None, :, None], values, tasks, strata, weights)
                            for label, mask in regions.items()}
        if name != "interaction":
            indices = (0, 2) if name == "C_U_minus_B_U" else (1, 3)
            selected = p[list(indices)]
            covered = (~(np.any(selected > 0, axis=0)[None, ...] & ~np.isfinite(values))).all(axis=(0, 2))
            supported = np.any(selected[:, covered] > 0, axis=0)
            on_support = np.where(supported[None, ...], values[:, covered], 0)
            correct, damage = expected_corrections(selected[:, covered], on_support)
            entry["corrections_on_paired_complete_questions"] = {
                label: dict(questions=int((mask & covered).sum()),
                            corrected=float(correct[:, mask[covered]].sum()),
                            damaged=float(damage[:, mask[covered]].sum()))
                for label, mask in dict(all=np.ones(len(tasks), dtype=bool), **regions).items()
            }
        contrasts[name] = entry
    return dict(diagnostics=diagnostics, coverage=coverage, policy_accuracy=quality, contrasts=contrasts), missing


def run(args):
    protocol = original.read(args.protocol)
    for source, report_name, binding in (
        ("linear", "LINEAR_FIT_REPORT.json", "input_scores_sha256"),
        ("mlp", "MLP_FIT_AUDIT.json", "private_mlp_output_sha256"),
    ):
        training = original.read(HERE / report_name)
        if training[binding] != original.sha(getattr(args, source)):
            raise ValueError(f"Scores differ from the frozen {source} training audit")
    fixed, weights = fixed_assets(args)
    strata = original.labels_for(fixed["task_labels"], fixed["subtask_labels"])
    grid = load_candidates(args.linear, fixed, 3)
    mlp_seeds = protocol["mlp"]["seeds"]
    mlp = load_candidates(args.mlp, fixed, len(mlp_seeds), mlp_seeds)
    if not np.array_equal(grid[1], fixed["scores"]):
        raise ValueError("The central linear cell must reuse the exact archived scores")
    values = original.project_scores(args.scores, fixed["question_ids"], np.any(fixed["probabilities"] > 0, axis=0))
    known_members = known_members_from_values(values)
    if len(known_members) != 20661:
        raise ValueError("Closed member union does not match the archived bill")
    results = {"linear": [], "mlp": []}
    missing = {}
    for source, rows, labels in (("linear", grid, protocol["linear"]["lambda_factors"]), ("mlp", mlp, mlp_seeds)):
        union = np.zeros_like(values, dtype=bool)
        for label, heads in zip(labels, rows):
            cell, remaining = evaluate_cell(heads, fixed, values, known_members, strata, weights, protocol)
            cell["lambda_factor" if source == "linear" else "seed"] = label
            results[source].append(cell)
            union |= remaining
        missing[source] = union
    report = dict(schema="e25_tie_mechanism_replay_v1", protocol_sha256=original.sha(args.protocol),
                  input_bindings={name: original.sha(getattr(args, name))
                                  for name in ("linear", "mlp", "predictions", "scores", "scoring_receipt")},
                  questions=len(strata), repeated_seeds_per_question=list(SEEDS), primary_weights=weights,
                  archived_support_rows=int(np.isfinite(values).sum()), new_llm_calls=0, reference_values_reopened=0,
                  confidence_intervals_computed=False, bootstrap_repetitions=0,
                  results=results, missing_cost_by_group={group: missing_cost(missing[group], known_members)
                                                           for group in missing},
                  missing_cost_total_union=missing_cost(missing["linear"] | missing["mlp"], known_members))
    original.write(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("linear", "mlp", "predictions", "scores", "scoring-receipt", "e24-protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "PROTOCOL.json")
    run(parser.parse_args())
