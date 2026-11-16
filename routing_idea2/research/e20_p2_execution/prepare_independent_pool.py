"""Admit a frozen input-only P2 cohort, encode it and freeze B/C execution support.

Question selection is independent of model predictions, answers and correctness.
Protected answers are syntax-skipped by the inherited E18 allowlist parser.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

CODE = Path(__file__).resolve().parent
WORKSPACE = CODE.parents[1]
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
STUDY = ASSETS / "research/e20_p2_execution"
PRIVATE = STUDY / "private"
PUBLIC = STUDY / "public"
OLD = ASSETS / "research/e18_mechanism_and_confirmation_preparation"
E18 = ASSETS / "research/e18_confirmatory_execution"
FINAL = ASSETS / "research/e20_independent_confirmation_preparation/private/final_models"
E18_CODE = CODE.parent / "e18_mechanism_and_confirmation_preparation"
E20_CODE = CODE.parent / "e20_independent_confirmation_preparation"
sys.path.insert(0, str(E18_CODE))
sys.path.insert(0, str(E20_CODE))

from audit_sources import (INPUT_FIELDS, METADATA_FIELDS, NEAR_RULES, canonical_hash,
                           project, read_json, sha, write_json, write_jsonl,
                           _screen_near_duplicates)
from candidate_proposal import safe_source_rows, protected_screen

SEED = 20261116020
MODEL_SHA = "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92"
OPERATOR = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
GENERATION_SEEDS = (42, 137, 2024)


def now():
    return datetime.now(timezone.utc).isoformat()


def rank(purpose, value):
    return hashlib.sha256(f"{SEED}|{purpose}|{value}".encode("utf-8")).hexdigest()


def _source_reference(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def select():
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    output = PRIVATE / "ADMITTED2400_INPUTS_ONLY.jsonl"
    if output.exists():
        raise FileExistsError("P2 input cohort is already frozen")
    prior_counts = read_json(E20_CODE / "E20_DATA_PERMISSIONS.json")["additional_pool_plan"]["bbh_subtask_counts"]
    names = sorted(prior_counts)
    if len(names) != 27 or any(prior_counts[name] < 44 for name in names):
        raise ValueError("Predeclared 27-subtask quota lacks input support")
    extra = [name for name in names if name != "penguins_in_a_table" and prior_counts[name] >= 45][:12]
    quotas = {name: 45 if name in extra else 44 for name in names}
    config = dict(schema="e20_p2_input_selection_prefreeze_v1", utc=now(), status="FROZEN_BEFORE_SELECTION",
                  authorization="User instructed completion of P2 preparation and start of P2 after the submitted2400-question proposal",
                  source_commit="031d32ee34911f58c535803c078584c424c8f6e3", source_sha256=sha(__file__),
                  total_questions=2400, task_counts={"gsm8k": 1200, "bbh": 1200}, selection_seed=SEED,
                  source_frame="GSM8K official train; full official BBH test",
                  bbh_subtask_quotas=quotas,
                  quota_rule="Sorted semantic subtask names, first12 with capacity>=45 excluding penguins get45; others44",
                  ordering="GSM first, then semantic BBH subtask order; each group ranks canonical hash using SHA256(seed|purpose|canonical)",
                  duplicate_rules=NEAR_RULES, model_predictions_used_for_selection=False,
                  rewards_or_answers_used_for_selection=False, frozen_model_sha256=MODEL_SHA,
                  protected_pool="sealed200 remains answer-sealed and is excluded by identifiers and one-way signatures")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    write_json(PUBLIC / "P2_INPUT_SELECTION_FROZEN.json", config)

    source_manifest = read_json(OLD / "private/SOURCE_MANIFEST.json")
    registered = read_json(OLD / "private/REGISTERED_EXCLUSION_METADATA.json")["records"]
    prior_scope = read_json(E18 / "private/EXECUTION_SCOPE.json")
    prior_inputs = {}
    for line in (E18 / "private/ADMITTED3000_INPUTS_ONLY.jsonl").read_bytes().splitlines():
        row = project(line, INPUT_FIELDS | {"canonical_question_sha256"})
        if canonical_hash(row["question"]) != row["canonical_question_sha256"]:
            raise ValueError("E18 historical input identity mismatch")
        prior_inputs[row["question_id"]] = row
    exclusions_by_id = {row["question_id"]: row for row in registered}
    for record in prior_scope["question_keys"]:
        question_id = record["question_id"]
        if question_id not in exclusions_by_id:
            exclusions_by_id[question_id] = dict(record, split="e18_executed")
    exclusions = list(exclusions_by_id.values())
    protected_root = OLD / "private/protected_input_fingerprints"
    protected_receipt = read_json(protected_root / "AUTHORIZED_ACCESS_RECEIPT.json")
    index_path = protected_root / "OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json"
    signature_path = protected_root / "PROTECTED_INPUT_SIGNATURES.json"
    if sha(index_path) != protected_receipt["bbh_position_index_sha256"] or sha(signature_path) != protected_receipt["signature_sha256"]:
        raise ValueError("Existing protected input fingerprints changed")
    index = read_json(index_path)["records"]
    signatures = read_json(signature_path)["records"]
    key = (protected_root / "FINGERPRINT_KEY.bin").read_bytes()
    rows, scans, official_duplicates = safe_source_rows(ASSETS, source_manifest, exclusions, index)
    row_ids = {row["question_id"] for row in rows}
    protected_ids = {row["question_id"] for row in registered if row["split"] == "sealed_test"}
    missing_historical = set(exclusions_by_id) - protected_ids - row_ids - set(prior_inputs)
    for path in (ASSETS / "data/derived/available_records.jsonl", ASSETS / "data/derived/gsm8k_questions.jsonl"):
        with path.open("rb") as stream:
            for raw in stream:
                question_id = project(raw, {"question_id"}).get("question_id")
                if question_id not in missing_historical:
                    continue
                row = project(raw, INPUT_FIELDS)
                row["canonical_question_sha256"] = canonical_hash(row["question"])
                if row["canonical_question_sha256"] != exclusions_by_id[question_id]["canonical_question_sha256"]:
                    raise ValueError("Historical input canonical mismatch")
                prior_inputs[question_id] = row
                missing_historical.remove(question_id)
    if missing_historical:
        raise ValueError("Unresolved historical input identities")
    for question_id, row in prior_inputs.items():
        if question_id not in row_ids:
            rows.append(dict(row, registered_excluded=True))
            row_ids.add(question_id)
    format_groups, near_removed, near_edges = _screen_near_duplicates(rows)
    format_removed = set()
    for group in format_groups:
        historical = any(rows[index]["registered_excluded"] for index in group)
        representative = None if historical else min(group, key=lambda index: rank("format-representative", rows[index]["canonical_question_sha256"]))
        format_removed.update(index for index in group if index != representative)
    protected_removed, protected_edges, protected_summary = protected_screen(rows, key, signatures)
    eligible = [row for index, row in enumerate(rows) if not row["registered_excluded"]
                and index not in near_removed and index not in format_removed and index not in protected_removed]
    groups = defaultdict(list)
    for row in eligible:
        groups[(row["task"], row["subtask"] if row["task"] == "bbh" else "")].append(row)
    gsm = sorted(groups[("gsm8k", "")], key=lambda row: rank("gsm1200", row["canonical_question_sha256"]))
    if len(gsm) < 1200:
        raise ValueError("GSM quota unavailable after frozen screens")
    selected = gsm[:1200]
    for name in names:
        group = sorted(groups[("bbh", name)], key=lambda row: rank("bbh1200", row["canonical_question_sha256"]))
        if len(group) < quotas[name]:
            raise ValueError("BBH quota unavailable after frozen screens " + name)
        selected.extend(group[:quotas[name]])
    selected = [{key: value for key, value in row.items() if key != "registered_excluded"} for row in selected]
    selected_ids = {row["question_id"] for row in selected}
    selected_hashes = {row["canonical_question_sha256"] for row in selected}
    old_hashes = {row["canonical_question_sha256"] for row in exclusions}
    if len(selected_ids) != 2400 or len(selected_hashes) != 2400 or selected_ids & set(exclusions_by_id) or selected_hashes & old_hashes:
        raise ValueError("Admitted cohort has historical overlap or repeated identities")
    if Counter(row["task"] for row in selected) != {"gsm8k": 1200, "bbh": 1200}:
        raise ValueError("Task weights differ")
    if any({"reference", "answer", "gold", "target", "reward", "score"} & set(row) for row in selected):
        raise ValueError("Forbidden field in input-only cohort")
    write_jsonl(output, selected)
    write_json(PRIVATE / "P2_INPUT_DUPLICATE_DETAILS.json", dict(
        official_exact_duplicate_groups=official_duplicates, unprotected_high_similarity_edges=near_edges,
        protected_signature_edges=protected_edges,
        format_equivalence_groups=[[rows[index]["question_id"] for index in group] for group in format_groups]))
    summary = dict(schema="e20_p2_admitted_input_cohort_v1", status="ADMITTED_INPUTS_ONLY_REWARDS_HIDDEN", utc=now(),
                   selected_questions=2400, task_counts=dict(Counter(row["task"] for row in selected)),
                   bbh_subtask_counts=dict(sorted(Counter(row["subtask"] for row in selected if row["task"] == "bbh").items())),
                   source_frame_counts=dict(Counter(row["source_frame"] for row in selected)),
                   source_layer_warning="Full official BBH source expands the original common subpool; report source layers separately",
                   candidate_input_rows_screened=len(rows), eligible_task_counts=dict(Counter(row["task"] for row in eligible)),
                   excluded_historical_identities=len(exclusions), format_groups=len(format_groups), high_similarity_edges=len(near_edges),
                   protected_signature_screen=protected_summary, exact_or_near_overlap_after_screen=0,
                   selected_input_sha256=sha(output), preselection_sha256=sha(PUBLIC / "P2_INPUT_SELECTION_FROZEN.json"),
                   source_sha256=sha(__file__), parser_source_sha256=sha(E18_CODE / "audit_sources.py"),
                   source_projection_scans=scans, duplicate_rules=NEAR_RULES,
                   reward_values_decoded=0, answer_values_decoded=0, sealed_inputs_decoded=0,
                   new_llm_calls=0, new_encoder_questions=0,
                   cpu_seconds=time.process_time()-started_cpu, wall_seconds=time.perf_counter()-started_wall)
    write_json(PUBLIC / "P2_INPUT_ADMISSION_REPORT.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "selected_questions", "task_counts", "eligible_task_counts", "selected_input_sha256", "wall_seconds")}, ensure_ascii=False))


def encode():
    import numpy as np
    import torch
    sys.path.insert(0, str(ASSETS / "src"))
    from budget_router.encoding import FrozenQuestionEncoder

    started_cpu, started_wall = time.process_time(), time.perf_counter()
    text_path = PRIVATE / "ADMITTED2400_INPUTS_ONLY.jsonl"
    target = PRIVATE / "P2_INPUTS_ONLY.npz"
    if target.exists():
        raise FileExistsError("P2 encoded inputs already frozen")
    admission = read_json(PUBLIC / "P2_INPUT_ADMISSION_REPORT.json")
    if sha(text_path) != admission["selected_input_sha256"] or sha(FINAL / "FINAL600_MODELS_FROZEN.npz") != MODEL_SHA:
        raise ValueError("Frozen input cohort or models changed")
    rows = [json.loads(line) for line in text_path.read_text(encoding="utf-8").splitlines()]
    if any({"reference", "answer", "gold", "target", "reward", "score"} & set(row) for row in rows):
        raise ValueError("Input encoding schema contaminated")
    input_root = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
    original_receipt = read_json(input_root / "INPUT_ASSETS_RECEIPT.json")
    versions = {package: metadata.version(package) for package in ("numpy", "torch", "transformers", "tokenizers", "safetensors")}
    if any(versions[name] != original_receipt["runtime_versions"][name] for name in versions):
        raise ValueError("Frozen encoder runtime differs")
    torch.set_num_threads(2)
    encoder = FrozenQuestionEncoder(ASSETS / "models/sentence-transformers--all-MiniLM-L6-v2",
                                    revision=original_receipt["feature_encoder"]["revision"],
                                    receipt=ASSETS / "artifacts/audit/encoder_model_download.json", device="cpu", max_length=256)
    vectors = encoder.encode([row["question"] for row in rows], batch_size=8)
    if vectors.shape != (2400, 384) or not np.isfinite(vectors).all():
        raise ValueError("Frozen input vectors invalid")
    with target.open("xb") as stream:
        np.savez_compressed(stream, embeddings=vectors,
                            question_ids=np.array([row["question_id"] for row in rows], dtype=str),
                            canonical_hashes=np.array([row["canonical_question_sha256"] for row in rows], dtype=str),
                            task_labels=np.array([row["task"] for row in rows], dtype=str),
                            subtask_labels=np.array([row["subtask"] or "" for row in rows], dtype=str))
    receipt = dict(schema="e20_p2_frozen_encoder_v1", status="FROZEN_INPUT_VECTORS", utc=now(), questions=2400,
                   batch_forwards=300, device="cpu", batch_size=8, input_sha256=sha(text_path), vector_packet_sha256=sha(target),
                   encoder_source_sha256=sha(ASSETS / "src/budget_router/encoding.py"), revision=original_receipt["feature_encoder"]["revision"],
                   runtime_versions=versions, reward_values_decoded=0, answer_values_decoded=0, new_llm_calls=0,
                   model_parameter_updates=0, cpu_seconds=time.process_time()-started_cpu, wall_seconds=time.perf_counter()-started_wall)
    write_json(PUBLIC / "P2_ENCODER_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def predict():
    import numpy as np
    from predict_final_inputs import features
    from fit_final_models import load_inputs, policy

    started_cpu, started_wall = time.process_time(), time.perf_counter()
    model_path = FINAL / "FINAL600_MODELS_FROZEN.npz"
    if sha(model_path) != MODEL_SHA or sha(E20_CODE / "predict_final_inputs.py") != "56bbb2cc43206acbed8575fa6996e0040ad0118a32071b2469ad3a68ccea71bf":
        raise ValueError("Final models or inference operator changed")
    packet = load_inputs(PRIVATE / "P2_INPUTS_ONLY.npz")
    with np.load(model_path, allow_pickle=False) as archive:
        state = {key: archive[key].copy() for key in archive.files}
    for key in ("question_ids", "canonical_hashes"):
        if set(packet[key].tolist()) & set(state[key].tolist()):
            raise ValueError("P2 evaluation pool overlaps model training")
    phi = features(packet["embeddings"], state)
    scores = np.stack([phi @ np.asfortranarray(coefficient) for coefficient in state["weights"]])
    main = [policy(value, 1e-8) for value in scores]
    raw = [policy(value, 0.0) for value in scores]
    probabilities = np.stack([item[0] for item in main])
    raw_probabilities = np.stack([item[0] for item in raw])
    target = PRIVATE / "P2_PREDICTIONS_FROZEN.npz"
    with target.open("xb") as stream:
        np.savez_compressed(stream, conditions=np.array(["B", "C"]), scores=scores, centered_scores=scores-scores.mean(axis=2, keepdims=True),
                            probabilities=probabilities, raw_exact_probabilities=raw_probabilities,
                            support_counts=np.stack([item[1] for item in main]), margins=np.stack([item[2] for item in main]),
                            **{name: packet[name] for name in ("question_ids", "canonical_hashes", "task_labels", "subtask_labels")})
    rows = [json.loads(line) for line in (PRIVATE / "ADMITTED2400_INPUTS_ONLY.jsonl").read_text(encoding="utf-8").splitlines()]
    teams = tuple(combinations(range(6), 3))
    members, aggregates = set(), set()
    support = np.any((probabilities > 0) | (raw_probabilities > 0), axis=0)
    for index, row in enumerate(rows):
        for action in np.flatnonzero(support[index]):
            for seed in GENERATION_SEEDS:
                aggregates.add((row["question_id"], seed, int(action)))
                members.update((row["question_id"], seed, member) for member in teams[action])
    scope = dict(schema="e20_p2_frozen_execution_scope_v1", status="FROZEN_BEFORE_EXECUTION_AND_GOLD", utc=now(),
                 conditions=["B", "C"], generation_seeds=list(GENERATION_SEEDS), aggregation_seed=42, operator_fingerprint=OPERATOR,
                 model_sha256=MODEL_SHA, prediction_sha256=sha(target), input_sha256=sha(PRIVATE / "ADMITTED2400_INPUTS_ONLY.jsonl"),
                 actions=[list(team) for team in teams], questions=2400,
                 question_keys=[{key: row.get(key) for key in ("question_id", "canonical_question_sha256", "task", "subtask", "source_split", "source_file", "source_index", "source_frame")} for row in rows],
                 required_member_keys=[list(key) for key in sorted(members)], required_aggregate_keys=[list(key) for key in sorted(aggregates)],
                 reward_values_decoded=0, answer_values_decoded=0, sealed_answer_values_decoded=0)
    write_json(PRIVATE / "EXECUTION_SCOPE.json", scope)
    summary = dict(schema="e20_p2_frozen_execution_union_summary_v1", status="PREDICTIONS_AND_EXECUTION_UNION_FROZEN", utc=now(),
                   questions=2400, task_counts=dict(Counter(packet["task_labels"].tolist())), model_sha256=MODEL_SHA,
                   prediction_sha256=sha(target), input_sha256=scope["input_sha256"], scope_sha256=sha(PRIVATE / "EXECUTION_SCOPE.json"),
                   required_member_calls=len(members), required_aggregation_calls=len(aggregates),
                   primary_support_action_pairs=int(np.any(probabilities > 0, axis=0).sum()),
                   raw_extra_action_pairs=int((np.any(raw_probabilities > 0, axis=0) & ~np.any(probabilities > 0, axis=0)).sum()),
                   per_policy_support={name: dict(mean=float(main[index][1].mean()), maximum=int(main[index][1].max()),
                                                   tied_questions=int(np.sum(main[index][1] > 1))) for index, name in enumerate(("B", "C"))},
                   both_strict_unique=int(np.sum((main[0][2] > 1e-8) & (main[1][2] > 1e-8))),
                   strict_switched=int(np.sum((main[0][2] > 1e-8) & (main[1][2] > 1e-8) & (scores[0].argmax(axis=1) != scores[1].argmax(axis=1)))),
                   conditional_member_plus_aggregate_token_cap=len(members)*6144 + len(aggregates)*8192,
                   reward_values_decoded=0, answer_values_decoded=0, sealed_answer_values_decoded=0, new_llm_calls=0,
                   new_model_fits=0, cpu_seconds=time.process_time()-started_cpu, wall_seconds=time.perf_counter()-started_wall)
    write_json(PUBLIC / "P2_FROZEN_EXECUTION_UNION.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "encode", "predict"), required=True)
    phase = parser.parse_args().phase
    {"select": select, "encode": encode, "predict": predict}[phase]()
