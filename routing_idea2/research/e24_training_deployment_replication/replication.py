"""Freeze and evaluate the four E24 policies on one audited question cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.stats import t

CODE = Path(__file__).resolve().parent
E20 = CODE.parent / "e20_independent_confirmation_preparation"
E23 = CODE.parent / "e23_deployment_prior_and_references"
for path in (E20, E23, CODE.parent / "e20_p2_execution"):
    sys.path.insert(0, str(path))

from blind_evaluation import expected_corrections
from cache_budget import SEEDS
from fit_final_models import load_inputs, policy
from predict_final_inputs import features

NAMES = ("B_U", "B_P", "C_U", "C_P")
TAU = 1e-8
MODEL_SHA = "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92"
OPERATOR_SHA = "56bbb2cc43206acbed8575fa6996e0040ad0118a32071b2469ad3a68ccea71bf"
AGGREGATION_FINGERPRINT = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
E23_SUMMARY_SHA = "4987a035169d910a492b26463ae6a199d9a87bb2690cfe35f4ae31e121f2e4e1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def four_policies(scores, counts):
    """Copy the E23 train600 tie-break semantics independently for each head."""
    if scores.ndim != 3 or scores.shape[0] != 2 or scores.shape[2] != 20 or not np.isfinite(scores).all():
        raise ValueError("Two finite 20-action score matrices required")
    if counts.shape != (20,) or counts.dtype.kind not in "iu" or not np.all((0 <= counts) & (counts <= 1800)):
        raise ValueError("Require integer successes over the original 1,800 train600 generation events")
    rows = []
    for head in scores:
        uniform = policy(head, TAU)[0]
        supported_counts = np.where(uniform > 0, counts, -1)
        priority = (uniform > 0) & (supported_counts == supported_counts.max(axis=1, keepdims=True))
        rows.extend((uniform, priority / priority.sum(axis=1, keepdims=True)))
    return np.stack(rows)


def labels_for(tasks, subtasks):
    if not np.isin(tasks, ("gsm8k", "bbh")).all() or np.any(subtasks[tasks == "bbh"] == ""):
        raise ValueError("Recognized task and BBH subtask labels required")
    return np.where(tasks == "gsm8k", "gsm8k", np.char.add("bbh/", subtasks))


def group_weights(strata, spec):
    expected = spec["cohort"]["stratum_counts"]
    if ("gsm8k" not in expected or any(name != "gsm8k" and not name.startswith("bbh/") for name in expected)
            or len(expected) < 2 or min(expected.values()) < 2 or Counter(strata.tolist()) != expected):
        raise ValueError("Scoring denominators differ from the prespecified cohort quotas")
    weight = {name: 0.5 / (len(expected) - 1) for name in expected if name != "gsm8k"}
    weight["gsm8k"] = 0.5
    return weight


def freeze(args):
    spec = read(args.protocol)
    if sha(args.model) != MODEL_SHA or sha(E20 / "predict_final_inputs.py") != OPERATOR_SHA:
        raise ValueError("The original B/C checkpoint or prediction operator changed")
    previous = E23 / "RESULTS_FINAL.json"
    if sha(previous) != E23_SUMMARY_SHA:
        raise ValueError("The original train600 success-count summary changed")
    counts = np.asarray(read(previous)["train_action_success_counts"])
    admission = read(args.admission)
    if (admission.get("schema") != "e24_input_cohort_admission_v1"
            or admission.get("protocol_sha256") != sha(args.protocol)
            or admission.get("vector_packet_sha256") != sha(args.inputs)
            or admission.get("answer_values_decoded") != 0
            or admission.get("reward_values_decoded") != 0):
        raise ValueError("A protocol-bound, input-only cohort and encoding receipt is required")
    packet = load_inputs(args.inputs)
    strata = labels_for(packet["task_labels"].astype(str), packet["subtask_labels"].astype(str))
    group_weights(strata, spec)
    with np.load(args.model, allow_pickle=False) as archive:
        state = {key: archive[key] for key in ("weights", "pca_mean", "pca_components", "scaler_mean",
                                               "scaler_scale", "question_ids", "canonical_hashes")}
    if any(set(packet[key].tolist()) & set(state[key].tolist()) for key in ("question_ids", "canonical_hashes")):
        raise ValueError("Evaluation question overlaps train600")
    design = features(packet["embeddings"], state)
    scores = np.stack([design @ np.asfortranarray(head) for head in state["weights"]])
    probabilities = four_policies(scores, counts)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, names=np.array(NAMES), scores=scores, probabilities=probabilities,
                            combo_success_counts=counts, question_ids=packet["question_ids"],
                            canonical_question_sha256=packet["canonical_hashes"], task_labels=packet["task_labels"],
                            subtask_labels=packet["subtask_labels"])
    support_counts = (probabilities[[0, 2]] > 0).sum(axis=2)
    strict = (support_counts == 1).all(axis=0)
    receipt = dict(schema="e24_frozen_policies_v1", status="INPUT_ONLY_POLICIES_FROZEN",
                   protocol_sha256=sha(args.protocol), input_sha256=sha(args.inputs),
                   input_admission_sha256=sha(args.admission), prediction_sha256=sha(args.output),
                   model_sha256=MODEL_SHA, prediction_operator_sha256=OPERATOR_SHA,
                   historical_prior_source_sha256=E23_SUMMARY_SHA,
                   generation_seeds=list(SEEDS), aggregation_fingerprint=AGGREGATION_FINGERPRINT,
                   questions=len(strata), names=list(NAMES),
                   unique_questions={name: int(np.sum(support_counts[i] == 1)) for i, name in enumerate(("B", "C"))},
                   priors_change_probability_questions={name: int(np.sum(np.any(probabilities[2*i] != probabilities[2*i+1], axis=1)))
                                                         for i, name in enumerate(("B", "C"))},
                   strict_switch_questions=int(np.sum(strict & (scores[0].argmax(1) != scores[1].argmax(1)))),
                   answer_values_decoded=0, reward_values_decoded=0,
                   new_head_fits=0, new_llm_calls=0)
    write(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def project_scores(path, ids, support):
    positions = {value: idx for idx, value in enumerate(ids)}
    values = np.full((len(SEEDS), len(ids), 20), np.nan)
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            question, seed, action = row["question_id"], row["generation_seed"], row["combination_index"]
            if (question not in positions or seed not in SEEDS or type(action) is not int or not 0 <= action < 20
                    or type(row["score"]) not in (float, int) or row["score"] not in (0, 1)):
                raise ValueError("Score row differs from frozen IDs, generation seeds, action space or binary rubric")
            i, j = SEEDS.index(seed), positions[question]
            if not support[j, action] or np.isfinite(values[i, j, action]):
                raise ValueError("Repeated or out-of-scope score")
            values[i, j, action] = row["score"]
    if not np.isfinite(values[:, support]).all():
        raise ValueError("Full four-policy support across all three seeds is required")
    return values


def stratified_test(question_values, strata, weights):
    estimate, components = 0.0, []
    for name, weight in weights.items():
        values = question_values[strata == name]
        estimate += weight * float(values.mean())
        components.append((weight**2 * float(values.var(ddof=1)) / len(values), len(values) - 1))
    variance = sum(value for value, _ in components)
    result = dict(difference_pp=100 * estimate, alpha=0.05, method="Two-sided stratified paired mean with Satterthwaite t")
    if variance == 0:
        return dict(result, status="DEGENERATE_VARIANCE", p_value=None)
    statistic = estimate / math.sqrt(variance)
    degrees = variance**2 / sum(value**2 / df for value, df in components)
    return dict(result, status="COMPUTED", p_value=float(2*t.sf(abs(statistic), degrees)), degrees_of_freedom=degrees)


def summarize(scores, probabilities, values, tasks, subtasks, spec):
    strata = labels_for(tasks, subtasks)
    weights = group_weights(strata, spec)
    # Only audited off-support positions are zeroed for multiplication by zero policy mass.
    legal = np.any(probabilities > 0, axis=0)
    if not np.isfinite(values[:, legal]).all() or not np.isnan(values[:, ~legal]).all():
        raise ValueError("Scores must be present on exact execution support only")
    available = np.where(legal[None], values, 0)
    utilities = np.einsum("pqa,sqa->psq", probabilities, available)

    def mean_pp(samples):
        return 100 * sum(weight * float(samples[..., strata == name].mean()) for name, weight in weights.items())

    def points(samples):
        by_task = {task: float(100 * samples[..., tasks == task].mean()) for task in ("gsm8k", "bbh")}
        return dict(primary=mean_pp(samples), question_weighted_within_task=sum(by_task.values()) / 2,
                    by_task=by_task, by_stratum={name: float(100 * samples[..., strata == name].mean()) for name in weights},
                    by_seed={str(seed): mean_pp(samples[i]) for i, seed in enumerate(SEEDS)})

    unique_b = (probabilities[0] > 0).sum(axis=1) == 1
    unique_c = (probabilities[2] > 0).sum(axis=1) == 1
    strict_switch = unique_b & unique_c & (scores[0].argmax(axis=1) != scores[1].argmax(axis=1))
    regions = {"B_unique": unique_b, "B_near_tied": ~unique_b, "both_strict_switch": strict_switch}
    comparisons = {}
    for label, first, second in (("C_U_minus_B_U", 0, 2), ("C_P_minus_B_P", 1, 3),
                                 ("B_P_minus_B_U", 0, 1), ("C_P_minus_C_U", 2, 3)):
        difference = utilities[second] - utilities[first]
        corrected, damaged = expected_corrections(probabilities[[first, second]], available)
        if not np.allclose(corrected - damaged, difference, atol=1e-12, rtol=0):
            raise ValueError("Expected correction/damage accounting differs from the paired utility")
        comparisons[label] = dict(**points(difference), corrected_events=float(corrected.sum()),
                                  damaged_events=float(damaged.sum()), net_corrected_events=float(difference.sum()),
                                  regions={name: dict(questions=int(mask.sum()),
                                                      contribution_pp=mean_pp(difference * mask[None]),
                                                      corrected_events=float(corrected[:, mask].sum()),
                                                      damaged_events=float(damaged[:, mask].sum()))
                                           for name, mask in regions.items()})
    interaction = (utilities[2] - utilities[0]) - (utilities[3] - utilities[1])
    if not np.allclose(interaction, (utilities[1] - utilities[0]) - (utilities[3] - utilities[2]), atol=1e-12, rtol=0):
        raise AssertionError("Training/deployment interaction does not decompose")
    ordered = np.sort(scores, axis=-1)
    margins = ordered[:, :, -1] - ordered[:, :, -2]
    threshold_rows = []
    for tau in spec["diagnostics"]["score_tolerance_grid"]:
        supports = [(policy(head, tau)[0] > 0).sum(axis=1) for head in scores]
        strict = (supports[0] == 1) & (supports[1] == 1)
        threshold_rows.append(dict(tau=tau, both_unique=int(strict.sum()),
                                   strict_changed=int(np.sum(strict & (scores[0].argmax(1) != scores[1].argmax(1))))))
    return dict(questions=len(tasks), recorded_generation_events_per_policy=len(tasks)*len(SEEDS),
                primary_stratum_weights=weights, accuracies={name: points(utilities[i]) for i, name in enumerate(NAMES)},
                interaction_pp=points(interaction), primary_test=stratified_test(interaction.mean(axis=0), strata, weights),
                comparisons_pp=comparisons, interaction_region_contributions_pp={name: mean_pp(interaction * mask[None])
                                                                                for name, mask in regions.items()},
                input_side=dict(strict_changed_questions=int(strict_switch.sum()), B_unique=int(unique_b.sum()),
                                C_unique=int(unique_c.sum()), thresholds=threshold_rows,
                                strict_change_margin_quantiles={head: (np.quantile(margins[i, strict_switch], [0, .1, .5, .9, 1]).tolist()
                                                                       if strict_switch.any() else [])
                                                                for i, head in enumerate(("B", "C"))}),
                confidence_intervals_computed=False, bootstrap_repetitions=0)


def evaluate(args):
    spec = read(args.protocol)
    frozen = read(args.receipt)
    if (frozen.get("schema") != "e24_frozen_policies_v1" or frozen.get("protocol_sha256") != sha(args.protocol)
            or frozen.get("prediction_sha256") != sha(args.predictions)):
        raise ValueError("Prediction bytes and evaluation protocol must be frozen before scoring")
    scoring = read(args.scoring_receipt)
    if (scoring.get("schema") != "e24_closed_native_scoring_v1"
            or scoring.get("prediction_sha256") != frozen["prediction_sha256"]
            or scoring.get("score_sha256") != sha(args.scores)
            or scoring.get("aggregation_fingerprint") != AGGREGATION_FINGERPRINT
            or scoring.get("sealed_answer_values_decoded") != 0):
        raise ValueError("Require closed native scoring evidence bound to these frozen policies")
    with np.load(args.predictions, allow_pickle=False) as packet:
        if tuple(packet["names"].tolist()) != NAMES:
            raise ValueError("Unexpected policy order")
        scores = packet["scores"]
        probabilities = packet["probabilities"]
        if not np.array_equal(probabilities, four_policies(scores, packet["combo_success_counts"])):
            raise ValueError("Saved policies differ from frozen decoding rules")
        support = np.any(probabilities > 0, axis=0)
        values = project_scores(args.scores, packet["question_ids"].tolist(), support)
        result = summarize(scores, probabilities, values, packet["task_labels"], packet["subtask_labels"], spec)
    report = dict(schema="e24_four_policy_replication_results_v1", protocol_sha256=sha(args.protocol),
                  prediction_sha256=frozen["prediction_sha256"], closed_scoring_receipt_sha256=sha(args.scoring_receipt),
                  **result)
    write(args.output, report)
    print(json.dumps({"questions": report["questions"], "interaction_pp": report["interaction_pp"]["primary"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, keys in (("freeze", ("model", "inputs", "admission", "receipt", "output")),
                          ("evaluate", ("predictions", "receipt", "scores", "scoring-receipt", "output"))):
        sub = subparsers.add_parser(command)
        sub.add_argument("--protocol", type=Path, default=CODE / "E24_PROTOCOL.json")
        for key in keys:
            sub.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
