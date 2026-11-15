"""Apply frozen E17 numbering heads to fresh inputs without reading outcomes.

The same400 inputs are processed by five historical training-fold feature maps.
The results describe input-only numbering sensitivity. They do not measure new
question accuracy, reward generalization, or five independent evaluation pools.
No training, encoder forward, API call, gold read, or interval is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from e17_common import (
    ASSETS, CODE, E13, METHODS, NUMERIC_TOLERANCE, POLICIES, PRIVATE, PUBLIC,
    input_packet, read_json, sha256, timing, verify_protocol, write_json,
)
from objectives import select_actions

import numpy as np


INPUT_DIRECTORY = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
FRESH_INPUTS = INPUT_DIRECTORY / "fresh_dev_test_inputs.npz"
INPUT_RECEIPT = INPUT_DIRECTORY / "INPUT_ASSETS_RECEIPT.json"
SAFE_INPUT_KEYS = {"question_ids", "canonical_hashes", "embeddings", "task_labels", "subtask_labels"}


def _describe(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=0)),
                minimum=float(values.min()), maximum=float(values.max()))


def _frequency(actions):
    counts = np.bincount(np.asarray(actions, dtype=int).ravel(), minlength=20)
    proportions = counts / counts.sum()
    nonzero = proportions[proportions > 0]
    return dict(counts=counts.tolist(), maximum_action_share=float(proportions.max()),
                entropy_effective_actions=float(np.exp(-np.sum(nonzero*np.log(nonzero)))))


def _choose(scores, permutation, policy):
    return select_actions(scores, permutation,
                          NUMERIC_TOLERANCE if policy.endswith("near") else 0,
                          semantic=policy.startswith("semantic"))


def _read_fresh_inputs():
    receipt = read_json(INPUT_RECEIPT)
    expected = receipt["inputs"]["fresh_dev_test"]["sha256"]
    if sha256(FRESH_INPUTS) != expected:
        raise ValueError("Cached fresh400 input packet no longer matches the feature receipt")
    with np.load(FRESH_INPUTS, allow_pickle=False) as archive:
        if set(archive.files) != SAFE_INPUT_KEYS:
            raise ValueError("Input packet contains forbidden or missing archive members")
        inputs = {name: archive[name].copy() for name in sorted(SAFE_INPUT_KEYS)}
    if inputs["embeddings"].shape != (400, 384) or not np.isfinite(inputs["embeddings"]).all():
        raise ValueError("Expected400 finite MiniLM384 vectors")
    for name in SAFE_INPUT_KEYS - {"embeddings"}:
        if inputs[name].shape != (400,):
            raise ValueError("Fresh input identity/metadata shape differs")
    if any(int(np.sum(inputs["task_labels"] == task)) != 200 for task in ("gsm8k", "bbh")):
        raise ValueError("Fresh400 task balance differs")
    for name in ("question_ids", "canonical_hashes"):
        if len(set(inputs[name].tolist())) != 400:
            raise ValueError("Duplicate fresh input identity")
    return inputs, expected


def _read_weights(path, expected_hash, permutation, packet):
    if sha256(path) != expected_hash:
        raise ValueError("Previously frozen numbering model changed")
    # Explicit NPZ members exclude even the already frozen outer score matrix.
    with np.load(path, allow_pickle=False) as archive:
        weights = archive["weights"].copy()
        recorded_permutation = archive["permutation"]
        history = archive["history"]
        evaluation = archive["evaluation"]
    if weights.shape != (len(METHODS), 17, 20) or not np.isfinite(weights).all():
        raise ValueError("Frozen numbering coefficient shape differs")
    if not np.array_equal(recorded_permutation, permutation):
        raise ValueError("Numbering model and fixed permutation differ")
    if not np.array_equal(history, packet["H"]) or not np.array_equal(evaluation, packet["E"]):
        raise ValueError("Frozen historical input-feature scope differs")
    return weights


def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    verify_protocol()
    main_freeze_path = PRIVATE / "ALL_PREDICTIONS_FROZEN.json"
    main_freeze = read_json(main_freeze_path)
    if main_freeze["status"] != "FROZEN_BEFORE_OUTER_REWARD_EVALUATION":
        raise ValueError("All numbering model predictions must be frozen before this input-only application")
    source_hashes = {str(Path(item["path"]).resolve()): item["sha256"] for item in main_freeze["files"]}
    permutation_path = PRIVATE / "PERMUTATIONS_FROZEN.npz"
    protocol = read_json(PUBLIC / "E17_PROTOCOL.json")
    if sha256(permutation_path) != protocol["permutation_sha256"]:
        raise ValueError("Frozen permutation collection changed")
    with np.load(permutation_path, allow_pickle=False) as archive:
        permutations = archive["permutations"].copy()
    if permutations.shape != (101, 20) or not np.array_equal(permutations[0], np.arange(20)):
        raise ValueError("Expected identity plus exactly100 predeclared numberings")
    if any(not np.array_equal(np.sort(permutation), np.arange(20)) for permutation in permutations):
        raise ValueError("Invalid frozen action permutation")

    private = PRIVATE / "fresh_numbering_input_only"
    public_output = PUBLIC / "FRESH_NUMBERING_INPUT_ONLY_SUMMARY.json"
    config_output = PUBLIC / "FRESH_NUMBERING_INPUT_ONLY_CONFIG_FROZEN.json"
    if private.exists() or public_output.exists() or config_output.exists():
        raise FileExistsError("Fresh numbering audit artifacts are immutable")
    inputs, input_sha = _read_fresh_inputs()
    config = dict(
        schema="e17_fresh_numbering_input_only_config_v1",
        source_sha256=sha256(CODE/"fresh_numbering_inputs.py"),
        main_prediction_freeze_sha256=sha256(main_freeze_path),
        original_protocol_sha256=sha256(PUBLIC/"E17_PROTOCOL.json"),
        fresh_input_sha256=input_sha, permutation_sha256=sha256(permutation_path),
        methods=METHODS, policies=POLICIES, random_numberings=100, identity_controls=1,
        unique_fresh_inputs=400, historical_training_feature_maps=5,
        near_tolerance=NUMERIC_TOLERANCE, source_models="existing mapped physical-coordinate coefficient matrices",
        outcome_access=0, new_fits=0, encoder_forwards=0, model_calls=0,
        interpretation="numbering sensitivity on new input vectors only; no reward/accuracy generalization claim",
    )
    PUBLIC.mkdir(parents=True, exist_ok=True)
    write_json(config_output, config)
    private.mkdir(parents=True)

    actions = np.empty((5, 101, len(METHODS), len(POLICIES), 400), dtype=np.int16)
    inference_only_actions = np.empty_like(actions)
    top_gaps = np.empty((5, 101, len(METHODS), 400), dtype=np.float64)
    score_spans = np.empty_like(top_gaps)
    centered_errors = np.empty((5, 101, len(METHODS)), dtype=np.float64)
    raw_errors = np.empty_like(centered_errors)
    run_rows, accesses = [], []
    for fold in range(5):
        packet_path = E13 / f"private/inputs/fold{fold}/INPUTS_ONLY.npz"
        packet = input_packet(fold)
        for key, shape in (("pca_mean", (384,)), ("pca_components", (16,384)),
                           ("scaler_mean", (16,)), ("scaler_scale", (16,))):
            if packet[key].shape != shape or not np.isfinite(packet[key]).all():
                raise ValueError("Historical fold preprocessing shape differs")
        if np.any(packet["scaler_scale"] <= 0):
            raise ValueError("Invalid historical StandardScaler scale")
        projected = (np.asarray(inputs["embeddings"], dtype=np.float64)-packet["pca_mean"]) @ packet["pca_components"].T
        phi = np.column_stack((np.ones(400), (projected-packet["scaler_mean"])/packet["scaler_scale"]))
        accesses.append(dict(path=str(packet_path), sha256=sha256(packet_path), purpose="historical H480-fitted input transformation only"))
        identity_scores = None
        for index, permutation in enumerate(permutations):
            model_path = PRIVATE / f"permutations/fold{fold}/p{index:03d}/MODEL.npz"
            expected = source_hashes.get(str(model_path.resolve()))
            if expected is None:
                raise ValueError("Numbering model is absent from the all-predictions freeze")
            weights = _read_weights(model_path, expected, permutation, packet)
            scores = np.einsum("qd,mda->mqa", phi, weights)
            if index == 0:
                identity_scores = scores.copy()
            accesses.append(dict(path=str(model_path), sha256=expected, purpose="frozen coefficient application; no refit"))
            for method_index, method in enumerate(METHODS):
                current, identity = scores[method_index], identity_scores[method_index]
                current_centered = current-current.mean(axis=1, keepdims=True)
                identity_centered = identity-identity.mean(axis=1, keepdims=True)
                centered_errors[fold,index,method_index] = np.max(np.abs(current_centered-identity_centered))
                raw_errors[fold,index,method_index] = np.max(np.abs(current-identity))
                sorted_scores = np.sort(current, axis=1)
                gap = sorted_scores[:,-1]-sorted_scores[:,-2]
                span = sorted_scores[:,-1]-sorted_scores[:,0]
                top_gaps[fold,index,method_index] = gap
                score_spans[fold,index,method_index] = span
                for policy_index, policy in enumerate(POLICIES):
                    selected = _choose(current, permutation, policy)
                    canonical = _choose(identity, permutations[0], policy)
                    inference_only = _choose(identity, permutation, policy)
                    actions[fold,index,method_index,policy_index] = selected
                    inference_only_actions[fold,index,method_index,policy_index] = inference_only
                    run_rows.append(dict(
                        fold=fold, permutation=index, method=method, inference=policy,
                        unique_input_questions=400,
                        identity_action_consistency=float(np.mean(selected == canonical)),
                        canonical_head_inference_priority_only_consistency=float(np.mean(inference_only == canonical)),
                        max_centered_score_equivariance_error=float(centered_errors[fold,index,method_index]),
                        max_raw_score_equivariance_error=float(raw_errors[fold,index,method_index]),
                        top1_minus_top2_gap=_describe(gap), max_minus_min_score_span=_describe(span),
                        exact_top_score_ties=int(np.sum(gap == 0)),
                        near_top_score_ties=int(np.sum(gap <= NUMERIC_TOLERANCE)),
                        physical_action_frequency=_frequency(selected),
                    ))

    summaries = []
    groups = {"all_inputs":np.ones(400,dtype=bool), "GSM8K":inputs["task_labels"]=="gsm8k", "BBH":inputs["task_labels"]=="bbh"}
    for fold_name, fold_indices in [(str(fold),[fold]) for fold in range(5)] + [("all_five_feature_maps",list(range(5)))]:
        for method_index, method in enumerate(METHODS):
            for policy_index, policy in enumerate(POLICIES):
                for group_name, mask in groups.items():
                    selected = actions[fold_indices,:,method_index,policy_index][:,:,mask]
                    reference = selected[:,0:1,:]
                    consistency = (selected[:,1:,:] == reference).mean(axis=2)
                    inference_only = inference_only_actions[fold_indices,:,method_index,policy_index][:,:,mask]
                    summaries.append(dict(
                        feature_map_scope=fold_name, method=method, inference=policy, input_group=group_name,
                        unique_input_questions=int(mask.sum()), historical_feature_maps=len(fold_indices),
                        random_numberings=100,
                        identity_action_consistency_across_fold_numberings=_describe(consistency),
                        canonical_head_inference_priority_only_consistency=_describe((inference_only[:,1:,:] == inference_only[:,0:1,:]).mean(axis=2)),
                        identity_physical_action_frequency=_frequency(selected[:,0,:]),
                        renamed_physical_action_frequency=_frequency(selected[:,1:,:]),
                        input_question_count_is_not_multiplied_by_fold_or_numbering=True,
                    ))
    score_summaries = []
    for fold_name, fold_indices in [(str(fold),[fold]) for fold in range(5)] + [("all_five_feature_maps",list(range(5)))]:
        for method_index, method in enumerate(METHODS):
            gaps = top_gaps[fold_indices,1:,method_index]
            spans = score_spans[fold_indices,1:,method_index]
            score_summaries.append(dict(
                feature_map_scope=fold_name, method=method,
                max_centered_score_equivariance_error_across100=_describe(centered_errors[fold_indices,1:,method_index]),
                max_raw_score_equivariance_error_across100=_describe(raw_errors[fold_indices,1:,method_index]),
                top1_minus_top2_gap=_describe(gaps), max_minus_min_score_span=_describe(spans),
                exact_top_score_tie_fraction=float(np.mean(gaps==0)),
                near_top_score_tie_fraction=float(np.mean(gaps<=NUMERIC_TOLERANCE)),
            ))
    actions_path = private / "FRESH_NUMBERING_ACTIONS_FROZEN.npz"
    np.savez_compressed(actions_path, actions=actions, inference_priority_only_actions=inference_only_actions,
                        top1_top2_gaps=top_gaps, max_min_score_spans=score_spans,
                        max_centered_score_equivariance_errors=centered_errors, max_raw_score_equivariance_errors=raw_errors,
                        question_ids=inputs["question_ids"], canonical_hashes=inputs["canonical_hashes"],
                        task_labels=inputs["task_labels"], permutations=permutations,
                        methods=np.array(METHODS), inference_policies=np.array(POLICIES))
    write_json(private/"RUN_METRICS.json", dict(rows=run_rows))
    write_json(private/"INPUT_ONLY_ACCESS_MANIFEST.json", dict(
        fresh_inputs=dict(path=str(FRESH_INPUTS), sha256=input_sha, decoded_keys=sorted(SAFE_INPUT_KEYS)),
        frozen_source_accesses=accesses, reward_payload_reads=0, answer_payload_reads=0,
        actual_output_length_reads=0, new_model_fits=0, new_embedding_forwards=0,
    ))
    write_json(public_output, dict(
        schema="e17_fresh_numbering_input_only_summary_v1", status="COMPLETE_INPUT_ONLY_NUMBERING_AUDIT",
        config_sha256=sha256(config_output), actions_artifact_sha256=sha256(actions_path),
        unique_fresh_inputs=400, historical_training_feature_maps=5,
        methods=METHODS, inference_policies=POLICIES, random_numberings=100,
        action_consistency_and_frequencies=summaries, score_equivariance_and_gaps=score_summaries,
        interpretation="Numbering sensitivity of frozen heads on new input embeddings. No reward or accuracy was observed; this is not an accuracy-generalization result.",
        repetition_unit="The same400 question inputs recur across five historical feature maps and100 numberings; independent question count remains400.",
        score_scope="Coefficients already mapped to immutable physical-team columns; centered errors remove score offsets shared by all20 actions.",
        inference_control_scope="Canonical fitted head held fixed while numeric priority changes; this is a re-expression control, with zero refits.",
        reward_payload_reads=0, answer_payload_reads=0, actual_output_length_reads=0,
        new_model_fits=0, encoder_forwards=0, llm_calls=0,
        confidence_intervals=0, bootstrap_runs=0, **timing(start_cpu,start_wall),
    ))
    write_json(private/"FROZEN_RECEIPT.json", dict(
        status="FROZEN_INPUT_ONLY_NO_REWARD_ACCESS", config_sha256=sha256(config_output),
        public_summary_sha256=sha256(public_output), actions_sha256=sha256(actions_path),
        run_metrics_sha256=sha256(private/"RUN_METRICS.json"),
        access_manifest_sha256=sha256(private/"INPUT_ONLY_ACCESS_MANIFEST.json"),
    ))
    print("E17_FRESH_NUMBERING_INPUT_ONLY_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit-inputs-only",))
    parser.parse_args()
    main()
