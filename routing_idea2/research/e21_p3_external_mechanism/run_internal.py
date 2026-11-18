"""Original-domain ListNet endpoint contrast on the five frozen E19 folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


CODE = Path(__file__).resolve().parent
E19 = CODE.parent / "e19_symmetric_tie_constraint"
E19_PROTOCOL_SHA256 = "d9715da6b15fe5d35d34c4ec876cfd613284373b2a1d792674248c3b9348269c"
FAMILY = "listnet"
FOLDS = range(5)

sys.path.insert(0, str(E19))
import run_e19 as source
import numpy as np

from endpoints import fit_endpoint_pair


def fixed_inputs(protocol_path, protocol_sha256):
    if source.sha(protocol_path) != protocol_sha256:
        raise ValueError("P3 protocol bytes differ from the pre-fit freeze")
    e19_protocol_path = E19 / "E19_PROTOCOL.json"
    if source.sha(e19_protocol_path) != E19_PROTOCOL_SHA256:
        raise ValueError("E19 source protocol differs")
    e19_protocol = source.read_json(e19_protocol_path)
    if (e19_protocol["head_shape"] != [17, 20]
            or e19_protocol["deployment"]["absolute_score_tie_tolerance"] != 1e-8
            or e19_protocol["branches"] != {"B": {"beta": 1}, "C": {"beta": 0}}
            or e19_protocol["gamma"] != 0):
        raise ValueError("Frozen E19 endpoint configuration differs")
    if source.sha(source.__file__) != e19_protocol["source_hashes"]["run_e19.py"]:
        raise ValueError("Frozen E19 input-reader bytes differ")
    if (source.sha(source.DEFAULT_REWARDS) != source.REWARD_SHA
            or source.sha(source.DEFAULT_INPUTS) != source.INPUT_SHA):
        raise ValueError("Original train600 reward/input assets differ")
    header = source.reward_header(source.DEFAULT_REWARDS)
    if header["shape"] != [3, 600, 20] or header["dtype"] != "<f8":
        raise ValueError("Original reward units differ")
    audit = source.read_json(E19 / "FROZEN_TSPO_REUSE_AUDIT.json")
    if audit["status"] != "PASS_READ_ONLY_SOURCE_AND_FEATURE_AUDIT":
        raise ValueError("Qualified E19 fold evidence is required")
    return e19_protocol, audit


def receipt_binding(protocol_sha256):
    return dict(
        protocol_sha256=protocol_sha256, e19_protocol_sha256=E19_PROTOCOL_SHA256,
        runner_sha256=source.sha(__file__), endpoint_source_sha256=source.sha(CODE / "endpoints.py"),
        input_asset_sha256=source.INPUT_SHA, reward_asset_sha256=source.REWARD_SHA,
    )


def checked_rewards(indices):
    rewards, access = source.read_reward_rows(source.DEFAULT_REWARDS, indices)
    if not np.isin(rewards, (0, 1)).all():
        raise ValueError("E19 native correctness labels must be binary")
    return rewards, access


def run_fit(root, p3_sha, e19_protocol, evidence):
    private = root / "private"
    private.mkdir(parents=True, exist_ok=False)
    all_evaluation_indices, frozen_folds = [], []
    start_wall, start_cpu = time.monotonic(), time.process_time()
    tau = e19_protocol["deployment"]["absolute_score_tie_tolerance"]
    for fold in FOLDS:
        packet, read_evidence = source.load_packet(fold, e19_protocol, evidence)
        history, evaluation = packet["H"], packet["E"]
        history_rewards, access = checked_rewards(history)
        empirical = history_rewards.mean(axis=0)
        regularization = e19_protocol["regularization_by_fold"][fold]
        begin_clock, begin_cpu = time.monotonic(), time.process_time()
        weights, qualification = fit_endpoint_pair(
            FAMILY, packet["phi_h"], empirical, regularization=regularization,
            reverse=(fold == 0), audit_features=packet["phi_e"],
        )
        if qualification["status"] != "PASS":
            raise ValueError("ListNet endpoint qualification failed before evaluation")
        weights = np.asarray(weights)
        if weights.shape != (2, 17, 20) or not np.isfinite(weights).all():
            raise ValueError("Qualified endpoint weights must be finite B/C 17×20 heads")
        score = packet["phi_e"] @ weights
        if score.shape != (2, 120, 20) or not np.isfinite(score).all():
            raise ValueError("Invalid frozen evaluation scores")
        fold_file = private / f"fold{fold}_PREDICTIONS.npz"
        source.write_npz(
            fold_file, W=weights, E=evaluation, scores=score,
            probabilities=np.stack([source.uniform_policy(matrix, tau)["probabilities"] for matrix in score]),
            raw_exact_probabilities=np.stack([
                source.uniform_policy(matrix, 0.0)["probabilities"] for matrix in score]),
        )
        audit_file = private / f"fold{fold}_FIT.json"
        frozen_fold = dict(
            fold=fold, regularization=regularization, history_questions=len(history),
            evaluation_questions=len(evaluation), fully_tied_history_questions=int(np.sum(
                np.all(empirical == empirical[:, :1], axis=1))),
            model_fits=4 if fold == 0 else 2,
            cpu_seconds=time.process_time()-begin_cpu, wall_seconds=time.monotonic()-begin_clock,
            qualification=qualification, fold_input_packet_sha256=read_evidence["input_packet_sha256"],
            predictions_sha256=source.sha(fold_file),
        )
        source.write_json(audit_file, dict(frozen_fold, history_reward_access=access))
        frozen_folds.append(frozen_fold)
        all_evaluation_indices.append(evaluation)
        print(json.dumps(dict(fold=fold, status="ENDPOINT_FITS_FROZEN")), flush=True)
    ordered = np.sort(np.concatenate(all_evaluation_indices))
    if not np.array_equal(ordered, np.arange(600)):
        raise ValueError("The withheld folds must partition the original 600 questions exactly")
    receipt = dict(
        schema="p3_original_domain_listnet_frozen_v1", status="ALL_OOF_PREDICTIONS_FROZEN",
        **receipt_binding(p3_sha), family=FAMILY, folds=frozen_folds, main_fits=10,
        reverse_qualification_fits=2, evaluation_phase_rewards_read=0,
        llm_calls=0, encoder_forwards=0, fit_stage_wall_seconds=time.monotonic()-start_wall,
        fit_stage_cpu_seconds=time.process_time()-start_cpu,
    )
    source.write_json(private / "PREDICTIONS_FROZEN.json", receipt)
    return receipt


def summarize(scores, labels, subtasks, rewards, tolerance):
    policies = [source.uniform_policy(matrix, tolerance) for matrix in scores]
    corrected, damaged, utility_b, utility_c, error = source.expected_coupled_changes(
        policies[0]["probabilities"], policies[1]["probabilities"], rewards)
    result, _, _ = source.point_metrics(
        np.stack([utility_b, utility_c]), corrected, damaged, rewards,
        *policies, labels, subtasks, tolerance)
    centered_delta = policies[1]["centered_scores"] - policies[0]["centered_scores"]
    result.update(
        centered_score_changed_questions_gt_1e_minus7=int(np.sum(
            np.max(np.abs(centered_delta), axis=1) > 1e-7)),
        mean_total_variation=float(np.abs(
            policies[1]["probabilities"]-policies[0]["probabilities"]).sum(axis=1).mean()/2),
        B_median_margin=float(np.median(policies[0]["margin"])),
        C_median_margin=float(np.median(policies[1]["margin"])),
        coupling_net_identity_max_error=error,
    )
    return result


def run_evaluation(root, p3_sha, e19_protocol, evidence):
    private = root / "private"
    receipt_path = private / "PREDICTIONS_FROZEN.json"
    frozen = source.read_json(receipt_path)
    if (frozen["status"] != "ALL_OOF_PREDICTIONS_FROZEN"
            or any(frozen[key] != value for key, value in receipt_binding(p3_sha).items())
            or frozen["evaluation_phase_rewards_read"] != 0
            or [row["fold"] for row in frozen["folds"]] != list(FOLDS)):
        raise PermissionError("Evaluation requires the complete matching pre-reward prediction freeze")
    public_file = root / "public" / "P3_INTERNAL_LISTNET_RESULTS.json"
    evaluation_file = private / "OOF_EVALUATION_RECEIPT.json"
    if public_file.exists() or evaluation_file.exists():
        raise FileExistsError("The original-domain evaluation is already recorded")
    tau = e19_protocol["deployment"]["absolute_score_tie_tolerance"]
    all_indices, all_scores = [], []
    for row in frozen["folds"]:
        fold = row["fold"]
        path = private / f"fold{fold}_PREDICTIONS.npz"
        packet, read_evidence = source.load_packet(fold, e19_protocol, evidence)
        if (row["predictions_sha256"] != source.sha(path)
                or read_evidence["input_packet_sha256"] != row["fold_input_packet_sha256"]):
            raise ValueError("Frozen OOF scores or input binding changed")
        with np.load(path, allow_pickle=False) as data:
            if (not np.array_equal(packet["E"], data["E"])
                    or data["scores"].shape != (2,120,20) or not np.isfinite(data["scores"]).all()):
                raise ValueError("OOF prediction question order or dimensions differ")
            for field, tolerance in (("probabilities", tau), ("raw_exact_probabilities", 0.0)):
                expected = np.stack([source.uniform_policy(matrix, tolerance)["probabilities"]
                                     for matrix in data["scores"]])
                if not np.array_equal(data[field], expected):
                    raise ValueError("Frozen uniform action probabilities differ")
            all_indices.append(data["E"])
            all_scores.append(data["scores"])
    indices = np.concatenate(all_indices)
    if not np.array_equal(np.sort(indices), np.arange(600)):
        raise ValueError("The frozen evaluation folds must partition the 600 questions exactly")
    scores = np.concatenate(all_scores, axis=1)
    with np.load(source.DEFAULT_INPUTS, allow_pickle=False) as inputs:
        labels = inputs["task_labels"][indices]
        subtasks = inputs["subtask_labels"][indices]
    if {key: int(np.sum(labels == key)) for key in ("gsm8k", "bbh")} != {"gsm8k": 300, "bbh": 300}:
        raise ValueError("The original 300/300 task split differs")
    analysis_start, analysis_cpu = time.monotonic(), time.process_time()
    rewards, access = checked_rewards(indices)
    per_fold = []
    for fold in FOLDS:
        selection = slice(120*fold, 120*(fold+1))
        per_fold.append(dict(
            fold=fold, primary=summarize(scores[:, selection], labels[selection], subtasks[selection],
                                         rewards[:, selection], tau),
            raw_exact=summarize(scores[:, selection], labels[selection], subtasks[selection],
                                rewards[:, selection], 0.0),
        ))
    result = dict(
        schema="p3_original_domain_listnet_result_v1",
        status="COMPLETE_ORIGINAL_DOMAIN_DEVELOPMENT_CONTRAST", family=FAMILY,
        pre_evaluation_freeze_sha256=source.sha(receipt_path), **receipt_binding(p3_sha),
        analysis_scope="E19 original six members, original train600 OOF, alternate loss only",
        original_questions=600, generation_events_per_policy=1800, task_weights=dict(gsm8k=0.5, bbh=0.5),
        primary=summarize(scores, labels, subtasks, rewards, tau),
        raw_exact=summarize(scores, labels, subtasks, rewards, 0.0), per_fold=per_fold,
        frozen_fit_summary=[{key: row[key] for key in (
            "fold", "regularization", "fully_tied_history_questions", "model_fits", "cpu_seconds",
            "wall_seconds", "fold_input_packet_sha256", "predictions_sha256", "qualification")}
            for row in frozen["folds"]],
        fit_stage_wall_seconds=frozen["fit_stage_wall_seconds"],
        fit_stage_cpu_seconds=frozen["fit_stage_cpu_seconds"],
        evaluation_wall_seconds=time.monotonic()-analysis_start,
        evaluation_cpu_seconds=time.process_time()-analysis_cpu,
        main_fits=frozen["main_fits"], reverse_qualification_fits=frozen["reverse_qualification_fits"],
        new_llm_calls=0, new_encoder_forwards=0, fresh400_reward_reads=0, sealed200_reward_reads=0,
        confidence_intervals_computed=False, bootstrap_repetitions=0,
    )
    source.write_json(evaluation_file, dict(
        pre_evaluation_freeze_sha256=result["pre_evaluation_freeze_sha256"], **access))
    source.write_json(public_file, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    e19_protocol, audit = fixed_inputs(args.protocol, args.protocol_sha256)
    operation = run_fit if args.phase == "fit" else run_evaluation
    result = operation(args.output_root, args.protocol_sha256, e19_protocol, audit)
    print(json.dumps(dict(
        status=result["status"], main_fits=result["main_fits"],
        C_minus_B_pp=result.get("primary", {}).get("C_minus_B_pp"),
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
