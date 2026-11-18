"""Summarize frozen P2 action probabilities and scored source strata."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from blind_evaluation import (
    SEEDS, TAU, TEAMS, file_sha, load_predictions, read_json, write_json,
)


def summarize(scope_path, predictions_path, results_path, utilities_path, input_path):
    scope = read_json(scope_path)
    result = read_json(results_path)
    if (result["scope_sha256"] != file_sha(scope_path)
            or result["prediction_sha256"] != file_sha(predictions_path)):
        raise ValueError("Scored and frozen input identities differ")
    packet = load_predictions(predictions_path)
    ids = packet["question_ids"].astype(str)
    if [row["question_id"] for row in scope["question_keys"]] != ids.tolist():
        raise ValueError("Scored and frozen question order differs")
    if scope["bindings"]["target_inputs_sha256"] != file_sha(input_path):
        raise ValueError("Source-frame metadata must come from the frozen admitted inputs")
    with input_path.open(encoding="utf-8") as stream:
        input_metadata = [(row["question_id"], row["source_frame"])
                          for row in map(json.loads, stream)]
    if [item[0] for item in input_metadata] != ids.tolist():
        raise ValueError("Source-frame metadata order differs")
    sources = np.array([item[1] for item in input_metadata])
    tasks = packet["task_labels"].astype(str)
    probabilities = packet["probabilities"]
    scores = packet["scores"]
    with np.load(utilities_path, allow_pickle=False) as archive:
        valid = archive["valid_mask"]
        utilities = archive["utilities"]
        corrections = archive["expected_corrections"]
        damages = archive["expected_damages"]
    if (valid.shape != ids.shape or valid.dtype != bool
            or valid.sum() != result["valid_questions"]
            or utilities.shape != (2, len(SEEDS), int(valid.sum()))):
        raise ValueError("Unexpected scored sample membership or utility dimensions")

    def group(mask):
        selected = mask[valid]
        chosen = utilities[:, :, selected]
        return dict(
            requested_questions=int(mask.sum()), scored_questions=int(selected.sum()),
            B_accuracy_pct=float(100 * chosen[0].mean()),
            C_accuracy_pct=float(100 * chosen[1].mean()),
            difference_pp=float(100 * (chosen[1] - chosen[0]).mean()),
            expected_corrected_events=float(corrections[:, selected].sum()),
            expected_damaged_events=float(damages[:, selected].sum()),
        )

    score_drift = scores[1] - scores[1].mean(axis=1, keepdims=True)
    score_drift -= scores[0] - scores[0].mean(axis=1, keepdims=True)
    sorted_scores = np.sort(scores, axis=2)
    margins = sorted_scores[:, :, -1] - sorted_scores[:, :, -2]
    summaries = {}
    for index, name in enumerate(("B", "C")):
        summaries[name] = dict(
            support_counts={str(key): value for key, value in
                            sorted(Counter(np.count_nonzero(probabilities[index], axis=1)).items())},
            median_margin=float(np.median(margins[index])),
            strict_unique_questions=int(np.sum(margins[index] > TAU)),
        )
    probability_delta = probabilities[1] - probabilities[0]
    return dict(
        schema="e20_p2_final_diagnostics_v1",
        requested_questions=int(len(ids)),
        scope_sha256=result["scope_sha256"],
        prediction_sha256=result["prediction_sha256"],
        by_task={task: group(tasks == task) for task in ("gsm8k", "bbh")},
        by_source_frame={source: group(sources == source) for source in sorted(set(sources))},
        policy_summaries=summaries,
        action_probability_summary=[
            dict(action_index=index, members=list(team),
                 B_probability=float(probabilities[0, :, index].mean()),
                 C_probability=float(probabilities[1, :, index].mean()),
                 C_minus_B=float(probability_delta[:, index].mean()),
                 mean_absolute_probability_delta=float(np.abs(probability_delta[:, index]).mean()))
            for index, team in enumerate(TEAMS)
        ],
        mean_total_variation=float(np.abs(probability_delta).sum(axis=1).mean() / 2),
        mean_question_centered_score_l2_change=float(np.linalg.norm(score_drift, axis=1).mean()),
        new_llm_calls=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("scope", "predictions", "results", "utilities", "admitted-inputs", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, summarize(
        args.scope, args.predictions, args.results, args.utilities, args.admitted_inputs))
