"""Freeze training-only reference policies, then audit paid P2 score coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parent
ASSETS = Path(r"E:\语言模型组合路由")
P2 = CODE.parent / "e20_p2_execution"
E20 = CODE.parent / "e20_independent_confirmation_preparation"
E22 = CODE.parent / "e22_mechanism_specificity"
for folder in (P2, E20, E22):
    sys.path.insert(0, str(folder))
from predict_final_inputs import features
from row_weighted import fit
from run_audit import controls
from blind_evaluation import SEEDS
from cache_budget import budget

P2_PRIVATE = ASSETS / "research/e20_p2_execution/private"
P2_SCORED = P2_PRIVATE / "final_evaluation_v1"
MODEL = ASSETS / "research/e20_independent_confirmation_preparation/private/final_models/FINAL600_MODELS_FROZEN.npz"
HISTORY = ASSETS / "artifacts/private/routing_redesign_20261104"
RIDGE = ASSETS / "research/e17_supervision_distortion_audit/private/confirmation_input_only/FULL600_MODELS.npz"
MODEL_SHA = "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92"
PREDICTION_SHA = "b790643119b6d85b370d34b6547e89e1316c26b26ad0d4a005e047babc521c75"
RIDGE_SHA = "aabca0b20967f8711d6a700dea301e490afb71de56b73c6d9a65f9db5fb0f360"
INPUT_SHA = "57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3"
REWARD_SHA = "a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def archive(path):
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key].copy() for key in packet.files}


def uniform(scores):
    mask = scores >= scores.max(axis=-1, keepdims=True)-1e-8
    return mask/mask.sum(axis=-1, keepdims=True)


def train_prior(scores, counts):
    support = scores >= scores.max(axis=-1, keepdims=True)-1e-8
    candidates = np.where(support, counts, -1)
    selected = support & (candidates == candidates.max(axis=-1, keepdims=True))
    return selected/selected.sum(axis=-1, keepdims=True)


def binding():
    return {key: digest(CODE / key) for key in ("PROTOCOL.json", "analyze_references.py", "cache_budget.py")}


def freeze(args):
    begin_cpu, begin_wall = time.process_time(), time.perf_counter()
    folder = args.root / "private"
    folder.mkdir(parents=True, exist_ok=False)
    inputs = HISTORY / "full_feedback_inputs/v1/selector_train_inputs.npz"
    outcomes = HISTORY / "ff_v1/tr/selector_train_outcomes.npz"
    for path, expected in ((inputs, INPUT_SHA), (outcomes, REWARD_SHA), (MODEL, MODEL_SHA),
                           (RIDGE, RIDGE_SHA), (P2_PRIVATE/"P2_PREDICTIONS_FROZEN.npz", PREDICTION_SHA)):
        if digest(path) != expected:
            raise ValueError("Frozen history or model identity differs")
    history = archive(inputs)
    with np.load(outcomes, allow_pickle=False) as source:
        outcomes = {key: source[key].copy() for key in
                    ("question_ids", "canonical_hashes", "generation_seeds", "aggregate_scores")}
    if any(not np.array_equal(history[key], outcomes[key]) for key in ("question_ids", "canonical_hashes")):
        raise ValueError("History and reward identity differs")
    reward = outcomes["aggregate_scores"]
    if reward.shape != (3, 600, 20) or not np.isin(reward, (0, 1)).all():
        raise ValueError("Expected original train600 binary aggregate scores")
    if not np.array_equal(outcomes["generation_seeds"], SEEDS):
        raise ValueError("Training generation order differs")
    counts = reward.sum(axis=(0, 1)).astype(np.int64)
    model = archive(MODEL)
    packet = archive(P2_PRIVATE / "P2_PREDICTIONS_FROZEN.npz")
    encoded = archive(P2_PRIVATE / "P2_INPUTS_ONLY.npz")
    if not np.array_equal(encoded["question_ids"], packet["question_ids"]):
        raise ValueError("P2 prediction and input identities differ")
    phi = features(encoded["embeddings"], model)
    phi_h = model["phi_h"]
    reproduced = np.array([phi @ np.asfortranarray(head) for head in model["weights"]])
    if not np.array_equal(reproduced, packet["scores"]):
        raise ValueError("Frozen B/C inference has changed")
    policy = dict(B_uniform=packet["probabilities"][0], C_uniform=packet["probabilities"][1],
                  B_prior=train_prior(packet["scores"][0], counts),
                  C_prior=train_prior(packet["scores"][1], counts),
                  B_raw=packet["raw_exact_probabilities"][0], C_raw=packet["raw_exact_probabilities"][1])
    if not np.array_equal(policy["C_prior"], policy["C_uniform"]):
        raise ValueError("The original unique C policy changed under prior deployment")
    b_unique = (policy["B_uniform"] > 0).sum(axis=1) == 1
    if not np.array_equal(policy["B_prior"][b_unique], policy["B_uniform"][b_unique]):
        raise ValueError("Prior deployment changed an original B unique selection")
    best = counts == counts.max()
    policy["GlobalFixed"] = np.broadcast_to(best/best.sum(), (len(phi), 20)).copy()
    old_ridge = archive(RIDGE)
    if old_ridge["weights"].shape != (3, 17, 20):
        raise ValueError("Unexpected original final600 Ridge checkpoint")
    ridge_phi = features(encoded["embeddings"], old_ridge)
    ridge_scores = ridge_phi @ np.ascontiguousarray(old_ridge["weights"][0])
    policy["Ridge"] = uniform(ridge_scores)
    empirical = reward.mean(axis=0)
    masks, match = controls("final600", phi_h, empirical, history["task_labels"],
                            history["subtask_labels"], history["canonical_hashes"])
    fitted, logs = {}, {}
    for name in ("Uniform", "Matched2026111801", "Matched2026111802", "Matched2026111803"):
        coefficient, log, _ = fit("tspo", phi_h, empirical, 0.003472222222222222, masks[name])
        fitted[name] = coefficient
        logs[name] = log
        policy[name] = uniform(phi @ np.asfortranarray(coefficient))
    names = list(policy)
    target = folder / "POLICIES_FROZEN.npz"
    with target.open("xb") as stream:
        np.savez_compressed(stream, names=np.array(names), probabilities=np.array([policy[key] for key in names]),
                            question_ids=packet["question_ids"], task_labels=packet["task_labels"],
                            subtask_labels=packet["subtask_labels"], counts=counts,
                            control_names=np.array(list(fitted)), control_weights=np.array(list(fitted.values())))
    baseline_scores = packet["scores"]
    sorted_scores = np.sort(baseline_scores, axis=-1)
    strict = np.all(sorted_scores[:, :, -1]-sorted_scores[:, :, -2] > 1e-8, axis=0)
    strict_switched = strict & (baseline_scores[0].argmax(axis=1) != baseline_scores[1].argmax(axis=1))
    with (folder / "REGIONS_FROZEN.npz").open("xb") as stream:
        np.savez_compressed(stream, b_unique=b_unique, strict_switched=strict_switched)
    record = dict(status="ALL_REFERENCES_FROZEN_BEFORE_CACHE_SCORE_ANALYSIS", source_binding=binding(),
                  model_sha256=MODEL_SHA, p2_prediction_sha256=PREDICTION_SHA,
                  full600_ridge_sha256=RIDGE_SHA, policy_sha256=digest(target),
                  regions_sha256=digest(folder/"REGIONS_FROZEN.npz"), train_action_success_counts=counts.tolist(),
                  training_global_best_actions=np.flatnonzero(best).tolist(), matching=match, control_fits=logs,
                  new_prediction_head_fits=4, p2_score_values_read=0, new_llm_calls=0,
                  cpu_seconds=time.process_time()-begin_cpu, wall_seconds=time.perf_counter()-begin_wall)
    write(folder / "FREEZE.json", record)
    print(json.dumps(dict(status=record["status"], models=4, global_actions=record["training_global_best_actions"])), flush=True)


def utility(probabilities, values):
    required = probabilities > 0
    if np.any(~np.isfinite(values) & required[None]):
        raise ValueError("An unexecuted aggregate output cannot enter a full-cohort accuracy")
    return np.sum(np.where(required[None], probabilities[None]*values, 0), axis=-1)


def compare(first, second, values, tasks, subtasks, masks):
    baseline, candidate = utility(first, values), utility(second, values)
    common = np.minimum(first, second)
    residual_first, residual_second = first-common, second-common
    mass = residual_first.sum(axis=1)
    denominator = np.where(mass > 1e-14, mass, np.inf)[None]
    corrected = utility(residual_first, 1-values)*utility(residual_second, values)/denominator
    damaged = utility(residual_first, values)*utility(residual_second, 1-values)/denominator
    difference = candidate-baseline
    if not np.allclose(corrected-damaged, difference, atol=1e-12, rtol=0):
        raise ValueError("Supplemental correction accounting differs")
    def macro(array):
        return float(0.5*sum(array[:, tasks == task].mean() for task in ("gsm8k", "bbh")))
    groups = {}
    for key, mask in masks.items():
        groups[key] = dict(questions=int(mask.sum()), full_cohort_contribution_pp=100*macro(difference*mask[None]),
                           corrected_events=float(corrected[:, mask].sum()), damaged_events=float(damaged[:, mask].sum()))
    return dict(difference_pp=100*macro(difference), corrected_events=float(corrected.sum()),
                damaged_events=float(damaged.sum()), net_corrected_events=float(difference.sum()),
                by_task={task: float(100*difference[:, tasks == task].mean()) for task in ("gsm8k", "bbh")},
                by_seed={str(seed): 100*macro(difference[index:index+1]) for index, seed in enumerate(SEEDS)},
                by_bbh_subtask={key: float(100*difference[:, (tasks == "bbh") & (subtasks == key)].mean())
                                for key in sorted(set(subtasks[tasks == "bbh"]))},
                regions=groups)


def analyze(args):
    folder = args.root / "private"
    receipt = read(folder / "FREEZE.json")
    if (receipt["source_binding"] != binding()
            or digest(folder/"POLICIES_FROZEN.npz") != receipt["policy_sha256"]
            or digest(folder/"REGIONS_FROZEN.npz") != receipt["regions_sha256"]):
        raise ValueError("Frozen reference policies or source changed")
    packet, regions = archive(folder/"POLICIES_FROZEN.npz"), archive(folder/"REGIONS_FROZEN.npz")
    ids, tasks, subtasks = packet["question_ids"].astype(str), packet["task_labels"], packet["subtask_labels"]
    names, probabilities = packet["names"].tolist(), packet["probabilities"]
    policies = dict(zip(names, probabilities))
    scope = read(P2_PRIVATE/"runtime_launch_v2/inputs/EXECUTION_SCOPE_RUNTIME.json")
    existing_member = set(map(tuple, scope["required_member_keys"]))
    existing_agg = set(map(tuple, scope["required_aggregate_keys"]))
    coverage, absent = budget(ids.tolist(), policies, existing_member, existing_agg)
    write(folder/"MISSING_KEYS.json", absent)
    values = np.full((3, len(ids), 20), np.nan)
    positions, seen = {key: index for index, key in enumerate(ids)}, set()
    score_path = P2_SCORED / "PAID_SUPPORT_SCORES.jsonl"
    with score_path.open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            key = (row["question_id"], row["generation_seed"], row["combination_index"])
            if key not in existing_agg or key in seen or row["score"] not in (0, 1):
                raise ValueError("Original P2 score identity, duplication or value error")
            seen.add(key)
            values[SEEDS.index(key[1]), positions[key[0]], key[2]] = row["score"]
    if seen != existing_agg:
        raise ValueError("Original P2 complete paid score coverage changed")
    matrices, rows = {}, {}
    for name, probability in policies.items():
        support = probability > 0
        covered = np.all(np.all(np.isfinite(values) | ~support[None], axis=0), axis=1)
        if covered.all():
            matrices[name] = utility(probability, values)
            rows[name] = dict(status="FULL_COHORT_SCORED", questions=len(ids),
                accuracy_pct=float(50*sum(matrices[name][:, tasks == task].mean() for task in ("gsm8k", "bbh"))),
                by_task={task: float(100*matrices[name][:, tasks == task].mean()) for task in ("gsm8k", "bbh")},
                by_seed={str(seed): float(50*sum(matrices[name][index, tasks == task].mean()
                                                for task in ("gsm8k", "bbh"))) for index, seed in enumerate(SEEDS)})
        else:
            rows[name] = dict(status="FULL_COHORT_ACCURACY_PENDING_MISSING_CALLS",
                             questions=len(ids), fully_covered_questions=int(covered.sum()), accuracy_pct=None)
    masks = dict(original_B_unique=regions["b_unique"], original_B_near_tied=~regions["b_unique"],
                 original_strict_switched=regions["strict_switched"])
    pairs = {}
    for baseline, candidate in (("B_uniform", "C_uniform"), ("B_uniform", "B_prior"),
                                ("B_prior", "C_prior"), ("B_raw", "C_raw")):
        if baseline not in matrices or candidate not in matrices:
            raise ValueError("Required prior deployment result is outside original P2 support")
        pairs[f"{candidate}_minus_{baseline}"] = compare(policies[baseline], policies[candidate], values, tasks, subtasks, masks)
    original = pairs["C_uniform_minus_B_uniform"]["difference_pp"]
    residual = pairs["C_prior_minus_B_prior"]["difference_pp"]
    gain = pairs["B_prior_minus_B_uniform"]["difference_pp"]
    if abs(original-residual-gain) > 1e-10:
        raise ValueError("Deployment-improvement decomposition failed")
    original_report = read(P2_SCORED/"P2_POINT_RESULTS.json")
    with np.load(P2_SCORED/"P2_POLICY_UTILITIES_PRIVATE.npz", allow_pickle=False) as source:
        if (not source["valid_mask"].all()
                or not np.allclose(source["utilities"],
                                   np.array([matrices["B_uniform"], matrices["C_uniform"]]), atol=1e-12, rtol=0)):
            raise ValueError("Original native score projection changed")
    if abs(original_report["primary"]["difference_pp"]-original) > 1e-10:
        raise ValueError("Original frozen primary result changed")
    output = dict(schema="mab_p2_prior_and_references_results_v1",
        source_binding=receipt["source_binding"], policy_sha256=receipt["policy_sha256"],
        existing_scores_sha256=digest(score_path), questions=len(ids),
        original_generation_events_per_policy=3*len(ids), task_weights=dict(gsm8k=0.5, bbh=0.5),
        prior_success_counts=receipt["train_action_success_counts"],
        training_global_best_actions=receipt["training_global_best_actions"],
        matching=receipt["matching"], control_fit_logs=receipt["control_fits"],
        baseline_strategies=rows, paired_comparisons=pairs,
        original_gap_pp=original, B_deployment_gain_pp=gain, remaining_gap_pp=residual,
        coverage_and_missing_budget=coverage, new_llm_calls=0, new_llm_tokens=0,
        new_prediction_head_fits=4, confidence_intervals_computed=False, bootstrap_repetitions=0,
        training_cpu_seconds=receipt["cpu_seconds"], training_wall_seconds=receipt["wall_seconds"])
    write(args.output, output)
    print(json.dumps({key: output[key] for key in ("original_gap_pp", "B_deployment_gain_pp", "remaining_gap_pp", "training_global_best_actions",
                                                  "coverage_and_missing_budget")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "analyze"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else analyze)(args)
