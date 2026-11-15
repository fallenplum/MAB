"""Load the four E17 policies, encode proposed inputs and count execution keys.

This program has no language-model generation or reward-scoring entry point.
Proposed questions remain outside the admitted execution dataset.
"""
from __future__ import annotations

import argparse
from importlib import metadata, util
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import sys
import time
import unicodedata
import hashlib
import json

from e18_common import (ASSETS, CODE, E17, PRIVATE, PUBLIC, POLICIES, SEEDS,
    input_only_npz, preserve_e17, read_json, sha256, write_json)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MODEL_FOLDER = E17 / "private/confirmation_input_only"
INPUT_FOLDER = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
PROPOSED = PRIVATE / "proposed_new2600_inputs_only.jsonl"
AGGREGATOR = ASSETS / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"
AGGREGATOR_FINGERPRINT = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
MODEL_QUALIFICATION = "FROZEN_MODEL_QUALIFICATION_v3.json"
TEAMS = np.array(list(combinations(range(6), 3)), dtype=np.int64)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def restore_feature_state(saved):
    pca = PCA(n_components=16, svd_solver="full")
    pca.mean_ = saved["pca_mean"]
    pca.components_ = saved["pca_components"]
    pca.n_features_in_ = 384
    pca.n_components_ = 16
    scaler = StandardScaler()
    scaler.mean_ = saved["scaler_mean"]
    scaler.scale_ = saved["scaler_scale"]
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = 16
    scaler.n_samples_seen_ = np.int64(600)
    return pca, scaler


def features(embeddings, state):
    pca, scaler = restore_feature_state(state)
    # The frozen state stores the components and mean of a non-whitened PCA.
    # In pinned sklearn1.7.2, _transform is exactly the implementation used by
    # transform after validation. It does not require unsaved eigenvalues.
    projected = pca._transform(np.asarray(embeddings, dtype=np.float64), xp=np)
    transformed = scaler.transform(projected)
    return np.column_stack((np.ones(len(embeddings)), transformed))


def predict(embeddings, state, global_action):
    phi = features(embeddings, state)
    scores = []
    for index, coefficient in enumerate(state["weights"]):
        # Original closed-form Ridge returns C-order; the exact SPO QP returns
        # F-order coefficients. Reproduce the frozen inference memory layout.
        matrix = np.ascontiguousarray(coefficient) if index == 0 else np.asfortranarray(coefficient)
        scores.append(phi @ matrix)
    scores = np.stack(scores)
    choices = np.vstack((scores.argmax(axis=2), np.full((1, len(embeddings)), global_action)))
    return choices.astype(np.int64), scores


def operator_sources():
    sys.path.insert(0, str(ASSETS / "src"))
    from budget_router.inference import ModelSpec, configuration_fingerprint
    from budget_router_v2.scoring import SCORER_VERSION
    sys.path.insert(0, str(ASSETS / "scripts/v2"))
    from typed_operator_contract import validate_typed_freeze_header
    config = read_json(AGGREGATOR)
    bundle_path = ASSETS / "artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json"
    bundle = read_json(bundle_path)
    operator = bundle["aggregator_operator"]
    freeze_path = Path(operator["freeze_evidence"]["path"])
    if sha256(freeze_path) != operator["freeze_evidence"]["sha256"]:
        raise ValueError("Active V2 freeze receipt changed")
    freeze = read_json(freeze_path)
    identity = validate_typed_freeze_header(freeze)
    scientific_model = asdict(ModelSpec(**config))
    scientific_model.pop("model_path")
    if scientific_model != identity["model_config_without_loading_path"]:
        raise ValueError("Active V2 logical model differs")
    if configuration_fingerprint(ModelSpec(**config)) != freeze["logical_model_config_fingerprint"]:
        raise ValueError("Active V2 base-model configuration differs")
    if operator["fingerprint"] != AGGREGATOR_FINGERPRINT:
        raise ValueError("Composite V2 operator differs from E17")
    if sha256(ASSETS / "src/budget_router_v2/scoring.py") != freeze["scorer_sha256"]:
        raise ValueError("Frozen V2 scoring code differs")
    adapter_sha = identity["prompt"]["typed_adapter_source_sha256"]
    adapters = [path for path in (ASSETS / "scripts/v2").glob("*.py") if sha256(path) == adapter_sha]
    if len(adapters) != 1:
        raise ValueError("Cannot locate the exact byte-frozen typed prompt adapter")
    pool = read_json(ASSETS / "configs/pool_v1.json")
    caps = {}
    for model in pool["models"]:
        path = ASSETS / model["config"] if "config" in model else None
        if path is None:
            # Every member has one pinned config whose computed identity must
            # match the original roster. Filename similarity grants no reuse.
            matches = []
            for candidate in (ASSETS / "configs").glob("member_*.json"):
                value = read_json(candidate)
                if configuration_fingerprint(ModelSpec(**value)) == model["fingerprint"]:
                    matches.append(candidate)
            if len(matches) != 1:
                raise ValueError("Cannot identify exactly one original member config")
            path = matches[0]
        value = read_json(path)
        if configuration_fingerprint(ModelSpec(**value)) != model["fingerprint"]:
            raise ValueError("Member execution fingerprint differs")
        caps[int(model["index"])] = int(value["max_input_tokens"]+value["max_new_tokens"])
    source_paths = [AGGREGATOR, ASSETS / "configs/pool_v1.json", bundle_path, freeze_path, adapters[0],
        ASSETS / "scripts/v2/typed_operator_contract.py",
        ASSETS / "src/budget_router/encoding.py", ASSETS / "src/budget_router/inference.py",
        ASSETS / "src/budget_router/execution.py", ASSETS / "src/budget_router/member_execution.py",
        ASSETS / "src/budget_router/scoring.py", ASSETS / "src/budget_router_v2/scoring.py"]
    records = [dict(path=str(path), sha256=sha256(path)) for path in source_paths]
    return dict(aggregator_fingerprint=AGGREGATOR_FINGERPRINT, scorer_version=SCORER_VERSION,
                base_model_configuration_fingerprint=freeze["logical_model_config_fingerprint"],
                prompt_version=identity["prompt"]["version"],
                composite_identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
                member_caps=caps, aggregate_cap=int(config["max_input_tokens"]+config["max_new_tokens"]),
                source_files=records)


def qualify_frozen():
    original = preserve_e17()
    state = load_archive(MODEL_FOLDER / "FULL600_MODELS.npz")
    frozen = load_archive(MODEL_FOLDER / "FRESH_POLICY_ACTIONS_FROZEN.npz")
    fresh = input_only_npz(INPUT_FOLDER / "fresh_dev_test_inputs.npz")
    if not np.array_equal(state["heads"], POLICIES[:3]):
        raise ValueError("Frozen policy identity/order differs")
    if not np.array_equal(frozen["question_ids"], fresh["question_ids"]):
        raise ValueError("Frozen fresh input identity differs")
    fixed = read_json(MODEL_FOLDER / "FULL600_FIT_LOG.json")["fixed_policies"]["global_action"]
    actions, scores = predict(fresh["embeddings"], state, fixed)
    score_error = float(np.max(np.abs(scores-frozen["head_scores"])))
    changed = (actions != frozen["actions"][:4]).sum(axis=1)
    if score_error > 1e-12 or np.any(changed):
        raise ValueError("E17 inference recovery failed")
    versions = read_json(E17 / "public/RUNTIME_VERSIONS.json")
    for package, key in (("numpy", "numpy"), ("scipy", "scipy"), ("scikit-learn", "scikit_learn")):
        if metadata.version(package) != versions[key]:
            raise ValueError("Frozen numerical package version changed")
    result = dict(status="PASS", e17_preservation=original,
        models_sha256=sha256(MODEL_FOLDER / "FULL600_MODELS.npz"),
        frozen_fresh_actions_sha256=sha256(MODEL_FOLDER / "FRESH_POLICY_ACTIONS_FROZEN.npz"),
        policies=POLICIES, frozen_weights=True, pca_refits=0, scaler_refits=0, head_fits=0,
        fresh_recovered_choices=1600, changed_choices_per_policy=changed,
        maximum_score_recovery_error=score_error, service_tie_rule="canonical physical semantic_exact",
        source_sha256=sha256(__file__), active_operator=operator_sources(),
        fresh_rewards_read=0, sealed_inputs_read=0, sealed_rewards_read=0)
    write_json(PUBLIC / MODEL_QUALIFICATION, result)
    write_json(PRIVATE / MODEL_QUALIFICATION, result)
    print("E18_FROZEN_POLICIES_QUALIFIED", flush=True)


def proposed_records():
    import json
    required = {"question_id", "task", "subtask", "source_split", "question",
                "canonical_question_sha256", "source_frame"}
    extra = {"source_file", "source_index", "source_revision", "status",
             "independence_qualification", "source_layer_admission"}
    rows = []
    with PROPOSED.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            if not required.issubset(row) or not set(row).issubset(required | extra):
                raise ValueError("Unexpected non-input fields in proposed cohort")
            normalized = " ".join(unicodedata.normalize("NFKC", row["question"]).split())
            if hashlib.sha256(normalized.encode()).hexdigest() != row["canonical_question_sha256"]:
                raise ValueError("Canonical input identity mismatch")
            rows.append(row)
    if len(rows) != 2600 or any(sum(row["task"] == task for row in rows) != 1300 for task in ("gsm8k", "bbh")):
        raise ValueError("Expected proposed 1300+1300 questions")
    if len({row["question_id"] for row in rows}) != 2600 or len({row["canonical_question_sha256"] for row in rows}) != 2600:
        raise ValueError("Duplicate question identity")
    return rows


def encode_proposed(rows):
    from budget_router.encoding import FrozenQuestionEncoder
    import torch
    source = read_json(INPUT_FOLDER / "INPUT_ASSETS_RECEIPT.json")
    expected = source["runtime_versions"]
    for package in ("numpy", "torch", "transformers", "tokenizers", "safetensors"):
        if metadata.version(package) != expected[package]:
            raise ValueError("Original MiniLM runtime version mismatch")
    torch.set_num_threads(2)
    encoder = FrozenQuestionEncoder(
        ASSETS / "models/sentence-transformers--all-MiniLM-L6-v2",
        revision=source["feature_encoder"]["revision"],
        receipt=ASSETS / "artifacts/audit/encoder_model_download.json", device="cpu", max_length=256)
    vectors = encoder.encode([row["question"] for row in rows], batch_size=8)
    if vectors.shape != (2600,384) or not np.isfinite(vectors).all():
        raise ValueError("Invalid original MiniLM vectors")
    if np.max(np.abs(np.linalg.norm(vectors,axis=1)-1)) > 1e-5:
        raise ValueError("Original L2 pooling contract failed")
    return vectors


def count_budget():
    start = time.perf_counter()
    qualification = read_json(PUBLIC / MODEL_QUALIFICATION)
    if qualification["source_sha256"] != sha256(__file__):
        raise ValueError("Frozen prediction program changed")
    preserve_e17()
    state = load_archive(MODEL_FOLDER / "FULL600_MODELS.npz")
    fresh = input_only_npz(INPUT_FOLDER / "fresh_dev_test_inputs.npz")
    frozen = load_archive(MODEL_FOLDER / "FRESH_POLICY_ACTIONS_FROZEN.npz")
    rows = proposed_records()
    candidate_sha = sha256(PROPOSED)
    ids = fresh["question_ids"].tolist() + [row["question_id"] for row in rows]
    hashes = fresh["canonical_hashes"].tolist() + [row["canonical_question_sha256"] for row in rows]
    if len(set(ids)) != 3000 or len(set(hashes)) != 3000:
        raise ValueError("New proposal overlaps existing fresh input")
    write_json(PRIVATE / "INPUT_PREDICTION_PREFREEZE.json", dict(
        status="FROZEN_BEFORE_NEW_INPUT_ENCODING", source_sha256=sha256(__file__),
        candidate_sha256=candidate_sha, model_qualification_sha256=sha256(PUBLIC / MODEL_QUALIFICATION),
        new_questions=2600, policies=POLICIES, admitted_new_questions=0,
        selection_independent_of_predictions=True, fresh_reward_reads=0, generates_LLM_answers=False))
    embeddings = encode_proposed(rows)
    global_action = int(frozen["actions"][3,0])
    new_actions, scores = predict(embeddings, state, global_action)
    actions = np.column_stack((frozen["actions"][:4], new_actions))
    np.savez_compressed(PRIVATE / "PROPOSED2600_INPUTS_ONLY_v2.npz",
        embeddings=embeddings, question_ids=np.array(ids[400:]), canonical_hashes=np.array(hashes[400:]),
        task_labels=np.array([row["task"] for row in rows], dtype=str),
        subtask_labels=np.array([str(row.get("subtask") or "") for row in rows], dtype=str))
    np.savez_compressed(PRIVATE / "FROZEN3000_INPUT_PREDICTIONS.npz", actions=actions,
        new_head_scores=scores, question_ids=np.array(ids), canonical_hashes=np.array(hashes), policies=np.array(POLICIES))
    spec = util.spec_from_file_location("e17_counter", E17 / "public/code/confirmation.py")
    counter = util.module_from_spec(spec)
    spec.loader.exec_module(counter)
    union = counter.input_call_union(actions, TEAMS, SEEDS)
    positions = {canonical:index for index, canonical in enumerate(hashes)}
    ledger = read_json(PRIVATE / "MEMBER_PHYSICAL_LEDGER.json")["rows"]
    available = {}
    for row in ledger:
        if row["canonical_id"] in positions:
            key = (positions[row["canonical_id"]], row["generation_seed"], row["member_index"])
            if key in available:
                raise ValueError("Duplicate physical member reuse key")
            available[key] = row
    required = set(map(tuple,union["member_keys"]))
    reused = required & set(available)
    missing = required - set(available)
    operator = qualification["active_operator"]
    member_cap = {int(key):value for key,value in operator["member_caps"].items()}
    aggregate_cap = operator["aggregate_cap"]
    actual_tokens = sum(available[key]["tokens"] for key in reused)
    actual_prompt = sum(available[key]["prompt_tokens"] for key in reused)
    actual_completion = sum(available[key]["completion_tokens"] for key in reused)
    missing_token_cap = sum(member_cap[member] for _,_,member in missing) + aggregate_cap*len(union["aggregate_keys"])
    from_zero_cap = sum(member_cap[member] for _,_,member in required) + aggregate_cap*len(union["aggregate_keys"])
    write_json(PRIVATE / "FROZEN_CALL_UNION_3000.json", union)
    write_json(PRIVATE / "VERIFIED_REUSE_AND_LOCAL_MISSING.json", dict(
        reused_member_keys=sorted(reused), local_missing_member_keys=sorted(missing),
        local_missing_aggregate_keys=union["aggregate_keys"],
        source_ledger_sha256=sha256(PRIVATE / "MEMBER_PHYSICAL_LEDGER.json")))
    segment_rows = []
    for name, begin, end in (("existing_fresh400",0,400),("proposed_new2600",400,3000)):
        members = {key for key in required if begin <= key[0] < end}
        aggregates = {tuple(key) for key in union["aggregate_keys"] if begin <= key[0] < end}
        segment_rows.append(dict(segment=name, questions=end-begin, member_calls=len(members),
            aggregate_calls=len(aggregates), verified_member_reuse=len(members & reused),
            new_member_calls_if_no_additional_remote_cache=len(members-set(available))))
    policy_rows = []
    for index, policy in enumerate(POLICIES):
        one = counter.input_call_union(actions[index:index+1], TEAMS, SEEDS)
        policy_rows.append(dict(policy=policy, member_calls=one["counts"]["member_calls"],
                                aggregate_calls=one["counts"]["aggregate_calls"]))
    source_frame_counts = {}
    for row in rows:
        source_frame_counts[row["source_frame"]] = source_frame_counts.get(row["source_frame"],0)+1
    result = dict(schema="E18_frozen_four_policy_exact_call_union_v1",
        status="PREPARATION_COMPLETE_EXECUTION_NOT_APPROVED", proposed_questions=3000,
        existing_fresh_questions=400, proposed_new_independent_questions=2600,
        automatically_admitted_questions=0, proposed_task_counts=dict(gsm8k=1500, bbh=1500),
        proposed_new_source_frames=source_frame_counts, generation_seeds=SEEDS, policies=POLICIES,
        union_required_calls=union["counts"], member_reuse_verified=len(reused), aggregate_reuse_verified=0,
        local_missing=dict(member_calls=len(missing), aggregate_calls=len(union["aggregate_keys"])),
        exact_global_new_physical_calls=None, remote_cache_state="UNKNOWN_PENDING_CONNECTIVITY_AND_INVENTORY",
        physical_upper_bound_if_no_additional_remote_cache=dict(member_calls=len(missing), aggregate_calls=len(union["aggregate_keys"])),
        actual_verified_reused_token_ledger=dict(prompt=actual_prompt, completion=actual_completion, total=actual_tokens),
        all_existing_3200_member_tokens=sum(row["tokens"] for row in ledger),
        new_physical_calls_executed=0, future_actual_token_cost=None,
        token_caps=dict(member_caps=member_cap, aggregate_cap=aggregate_cap,
            from_zero=from_zero_cap, remaining_local_missing=missing_token_cap,
            attempt_scope="one successful execution per unique key; failed/retried calls require additional separate accounting"),
        segments=segment_rows, single_policy_logical_costs=policy_rows,
        no_reuse_naive_calls=dict(member_calls=4*3000*3*3, aggregate_calls=4*3000*3),
        new_encoder_forwards=2600, encoder_frozen=True, head_fits=0, feature_preprocessing_refits=0,
        candidate_sha256=candidate_sha, model_sha256=qualification["models_sha256"],
        source_sha256=sha256(__file__), fresh_rewards_read=0, sealed_inputs_read=0, sealed_rewards_read=0,
        source_permission_note="Any protected-input fingerprint processing is separately authorized and recorded by the data-audit module.",
        wall_s=time.perf_counter()-start)
    write_json(PUBLIC / "E18_EXACT_CALL_BUDGET.json", result)
    write_json(PRIVATE / "FROZEN_BUDGET_RECEIPT.json", dict(status=result["status"],
        public_budget_sha256=sha256(PUBLIC / "E18_EXACT_CALL_BUDGET.json"),
        predictions_sha256=sha256(PRIVATE / "FROZEN3000_INPUT_PREDICTIONS.npz"),
        input_vectors_sha256=sha256(PRIVATE / "PROPOSED2600_INPUTS_ONLY_v2.npz")))
    preserve_e17()
    print("E18_PROPOSED3000_EXACT_CALL_UNION_FROZEN", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("qualify-frozen", "propose-budget"))
    command = parser.parse_args().command
    qualify_frozen() if command == "qualify-frozen" else count_budget()
