"""Describe score margins and action-column proximity for the fixed linear grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geometry import FROZEN_FILES, e24, np, quantiles


SCORE_PATH = "research/e25_tie_mechanism/private/LINEAR_HEADS.npz"


def analyze(assets, fit_report):
    receipt = json.loads(fit_report.read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETE" or receipt["evaluation_reward_reads"] != 0:
        raise ValueError("A completed input-only linear fit is required")
    head_path = assets / SCORE_PATH
    head_sha = e24.sha(head_path)
    if head_sha != receipt["input_scores_sha256"]:
        raise ValueError("Linear score artifact and fit receipt disagree")
    paths = {name: assets / path for name, (path, _) in FROZEN_FILES.items()}
    for name, path in paths.items():
        if e24.sha(path) != FROZEN_FILES[name][1]:
            raise ValueError(f"Frozen {name} bytes changed")
    if e24.sha(e24.E20 / "predict_final_inputs.py") != e24.OPERATOR_SHA:
        raise ValueError("Frozen feature and score operator changed")

    packet = e24.load_inputs(paths["inputs"])
    with np.load(paths["model"], allow_pickle=False) as model:
        design = e24.features(packet["embeddings"], model)
    with np.load(head_path, allow_pickle=False) as heads:
        scores, weights = heads["scores"], heads["weights"]
        factors = heads["lambda_factors"].tolist()
        if not np.array_equal(packet["question_ids"], heads["question_ids"]):
            raise ValueError("Input-only cohort order differs")
        if not np.array_equal(heads["betas"], [1, 0]) or factors != receipt["factors"]:
            raise ValueError("Linear grid order differs from its fit receipt")
    max_norm = float(np.linalg.norm(design, axis=1).max())
    left, right = np.triu_indices(scores.shape[-1], k=1)
    grid = []
    for factor, pair, weight_pair in zip(factors, scores, weights):
        policies = pair >= pair.max(axis=-1, keepdims=True) - e24.TAU
        unique = policies.sum(axis=-1) == 1
        switches = unique.all(axis=0) & (pair[0].argmax(axis=1) != pair[1].argmax(axis=1))
        cells = {}
        for name, score, head, support, unique_rows in zip(("B", "C"), pair, weight_pair, policies, unique):
            order = np.argsort(-score, axis=1, kind="stable")
            margin = score[np.arange(len(score)), order[:, 0]] - score[np.arange(len(score)), order[:, 1]]
            distance = np.linalg.norm(head[:, left] - head[:, right], axis=0)
            patterns, counts = np.unique(support[~unique_rows], axis=0, return_counts=True)
            clusters = []
            for pattern, count in zip(patterns, counts):
                actions = np.flatnonzero(pattern).tolist()
                inside_pairs = pattern[left] & pattern[right]
                max_distance = float(distance[inside_pairs].max())
                bound = max_norm * max_distance
                clusters.append(dict(
                    actions=actions, questions=int(count), maximum_coefficient_pair_distance=max_distance,
                    observed_maximum_score_spread_all_inputs=float(np.ptp(score[:, actions], axis=1).max()),
                    score_spread_bound_on_observed_input_norm_ball=bound,
                    all_inputs_in_bound_have_spread_below_tau=bool(bound < e24.TAU),
                ))
            cells[name] = dict(unique_winners=int(unique_rows.sum()), top_two_margin=quantiles(margin),
                               all_action_pair_distance=quantiles(distance), near_tied_clusters=clusters)
        grid.append(dict(lambda_factor=factor, strict_switch_questions=int(switches.sum()), heads=cells))
    return dict(
        schema="e25_linear_geometry_v1", tau=e24.TAU, questions=len(design), max_design_row_norm=max_norm,
        input_scores_sha256=head_sha, original_fit_report_sha256=e24.sha(fit_report),
        own_source_sha256=e24.sha(Path(__file__)), geometry_source_sha256=e24.sha(Path(__file__).with_name("geometry.py")),
        source_inputs_sha256=FROZEN_FILES["inputs"][1], rows=grid, evaluation_rewards_read=0, llm_calls=0,
        bound="For each action pair, |phi(x)^T(w_i-w_j)| <= ||phi(x)||_2 ||w_i-w_j||_2.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e24.write(args.output, analyze(args.assets, args.fit_report))
