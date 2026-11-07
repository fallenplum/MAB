"""Read-only independent verification of frozen E4 artifacts and public delivery.

The receipt reports artifact/source-level access evidence, not an OS file-access
audit. Intentional undefined ranking metrics and Gaussian sandwich placeholders
are distinguished from nonfinite predictions, losses and covariances.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

EXPECTED_E3 = "7995dfdc3afcff8eb8473f6da2af79d55818283d30044200a92c293bf583a499"
NULLABLE = re.compile(r"rank|correlation|spearman|kendall|rho", re.I)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    path = Path(path)
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


class Verification:
    def __init__(self, root):
        self.root = root
        self.work = root / "research/e4_statistical_model_audit"
        self.public, self.private = self.work / "public", self.work / "private"
        self.errors, self.warnings, self.counts = [], [], Counter()
        self.nullable_arrays, self.files = [], []
        self.json_cases = {}

    def check(self, condition, rule, file=None, detail=None):
        self.counts["assertions"] += 1
        if not bool(condition):
            self.errors.append({"rule": rule, "file": str(file) if file else None, "detail": detail})

    def json(self, path):
        value = read(path)
        self.counts["json_opened"] += 1
        self.files.append({"path": str(path.relative_to(self.work)), "sha256": sha(path), "bytes": path.stat().st_size})
        return value

    def arrays(self, path):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        self.counts["npz_opened"] += 1
        self.files.append({"path": str(path.relative_to(self.work)), "sha256": sha(path), "bytes": path.stat().st_size})
        for key, array in arrays.items():
            if array.dtype.kind not in "biufc":
                self.check(array.dtype.kind in "US", "npz_no_object_arrays", path, key)
                continue
            self.check(not np.isinf(array).any(), "npz_no_infinity", path, key)
            missing = int(np.isnan(array).sum()) if array.dtype.kind in "fc" else 0
            if missing:
                if key == "cluster_sandwich_latent_variance" and array.shape[0] == 4:
                    self.check(np.isnan(array[:2]).all() and np.isfinite(array[2:]).all(), "gaussian_sandwich_registered_missing", path)
                else:
                    self.check(bool(NULLABLE.search(key)), "nonfinite_only_registered_undefined_metrics", path, key)
                self.nullable_arrays.append({"file": str(path.relative_to(self.work)), "array": key, "nan_cells": missing})
        return arrays

    def shape(self, arrays, key, expected, path):
        self.check(key in arrays and arrays[key].shape == expected, "array_shape_" + key, path, list(expected))


def import_dependencies(root):
    public = root / "research/e4_statistical_model_audit/public"
    sys.path.insert(0, str(public))
    os.environ["MAB_ASSET_ROOT"] = str(root)
    import e4_common
    path = root / "research/decision_acquisition_e3/public/run_e3.py"
    specification = importlib.util.spec_from_file_location("e3_delivery_frozen_entry", path)
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    train, refs = e4_common.load_train()
    return e4_common, runner, train, refs


def verify_learning(v, train, folds):
    paths = sorted((v.private / "learning").glob("*.json"))
    v.check(len(paths) == 50, "A_50_fit_groups")
    contexts, cases = set(), 0
    for path in paths:
        row, arrays = v.json(path), v.arrays(path.with_suffix(".npz"))
        fold, size, predictor = row["fold"], row["n"], row["predictor"]
        contexts.add((fold, size, predictor))
        v.check(len(row["cases"]) == 13, "A_13_registered_cases", path)
        v.shape(arrays, "predictions", (13, 120, 20), path)
        v.check(np.array_equal(arrays["history"], folds[fold]["nested"][size])
                and np.array_equal(arrays["query"], folds[fold]["stream"]), "A_exact_history_query_ids", path)
        for case in row["cases"]:
            cases += 1
            v.check(case["L"] == len(case["subset"]) and set(case["subset"]) <= {0, 1, 2}, "A_seed_subset_permission", path)
            expected_alpha = 100 if case["regularization"] == "fixed" else 100 * case["L"] / 3
            v.check(np.isclose(case["alpha"], expected_alpha), "A_alpha_rule", path)
            dimension = 255 if predictor == "P1" else 340
            v.check(case["observable_parameter_dimension"] == dimension and 0 <= case["effective_df"] <= dimension + 1e-8,
                    "A_observable_dimension_and_df", path)
            expected_null = 119 / case["alpha"] if predictor == "P1" else 0
            v.check(np.isclose(case["prior_only_null_trace"], expected_null), "A_P1_null_trace", path)
    expected = {(fold, size, predictor) for fold in range(5) for size in (60, 120, 240, 360, 480) for predictor in ("P1", "P2")}
    v.check(contexts == expected and cases == 650, "A_complete_650_fit_grid")
    return {"fit_groups": len(paths), "model_fits": cases}


def verify_cost(v, receipt, permission, expected_gold, served, path):
    ledger = receipt["ledger"]
    v.check(receipt["gold_revealed"] == expected_gold and receipt["served_questions"] == served,
            "gold_and_service_completion", path)
    pair = permission == "pair"
    v.check(ledger["total"]["gold"] == expected_gold and ledger["total"]["scoring_ops"] == expected_gold * (2 if pair else 1),
            "gold_and_original_score_units", path)
    v.check(ledger["service"]["member_calls"] == 3 * served and ledger["service"]["aggregator_calls"] == served,
            "normal_service_calls", path)
    v.check(ledger["audit"]["member_calls"] == (expected_gold if pair else 0)
            and ledger["audit"]["aggregator_calls"] == (expected_gold if pair else 0), "incremental_feedback_calls", path)
    for key in ledger["total"]:
        v.check(np.isclose(ledger["total"][key], ledger["service"][key] + ledger["audit"][key], rtol=1e-11, atol=1e-8),
                "ledger_role_sum", path, key)
    for account in ("service", "audit"):
        v.check(all(ledger[account][key] <= cap for key, cap in receipt[account + "_caps"].items()), "hard_caps", path, account)
    v.check(receipt["new_physical_model_calls"] == 0, "declared_no_new_physical_model_calls", path)


def verify_posterior(v, train, folds, teams):
    paths = sorted((v.private / "posterior").glob("*.json"))
    v.check(len(paths) == 900, "B_900_paths")
    shared, identities = {}, set()
    for path in paths:
        row, arrays = v.json(path), v.arrays(path.with_suffix(".npz"))
        fold, generation, order, permission = (row[key] for key in ("fold", "generation_index", "orderseed", "permission"))
        identities.add((fold, row["predictor"], row["variant"], generation, order, permission))
        expected_order = folds[fold]["audit_orders"][order]
        v.check(np.array_equal(arrays["query"], folds[fold]["audit_query"]), "B_disjoint_fixed_query", path)
        v.shape(arrays, "initial_pred", (100, 20), path)
        v.shape(arrays, "final_pred", (100, 20), path)
        for key, shape in (("chosen", (21, 100)), ("trace", (21,)), ("maxeig", (21,)),
                           ("mean_change", (20,)), ("pair_variance", (21, 100))):
            v.shape(arrays, key, shape, path)
        v.check(np.all(np.diff(arrays["trace"]) <= 1e-9) and np.all(np.diff(arrays["pair_variance"], axis=0) <= 1e-9),
                "B_covariance_fixed_contrast_contraction", path)
        v.check(len(row["feedback"]) == 20, "B_twenty_feedback_bundles", path)
        v.check(row["generation_seed"] == (42, 137, 2024)[generation], "B_generation_identity", path)
        feedback_key = (fold, generation, order, permission)
        observations = []
        for index, event in enumerate(row["feedback"]):
            actions = event["actions"]
            v.check(event["q"] == expected_order[index] and len(actions) == (2 if permission == "pair" else 1), "B_shared_support_action_permission", path)
            v.check(len(event["raw_scores"]) == len(actions) and len(set(actions)) == len(actions)
                    and all(0 <= action < 20 for action in actions), "B_complete_valid_feedback_bundle", path)
            if len(actions) == 2:
                v.check(len(set(teams[actions[0]]) ^ set(teams[actions[1]])) == 2, "B_single_replacement_neighbor", path)
            v.check(all(value == train["Y"][event["q"], generation, action]
                        for action, value in zip(actions, event["raw_scores"])), "B_authorized_raw_score_identity", path)
            observations.append((event["q"], tuple(actions), tuple(event["raw_scores"])))
        if feedback_key in shared:
            v.check(observations == shared[feedback_key], "B_all_predictors_variants_share_feedback", path)
        else:
            shared[feedback_key] = observations
        for index, key in ((0, "initial_pred"), (-1, "final_pred")):
            choice = arrays[key].argmax(1)
            v.check(np.array_equal(choice, arrays["chosen"][index]), "B_saved_argmax", path)
            expected = train["Y"][arrays["query"], :, choice].mean(axis=1)
            v.check(np.allclose(expected, arrays["accuracy"][index], rtol=1e-11, atol=1e-11), "B_query_reward_replay", path)
        verify_cost(v, row["cost"], permission, 20, 20, path)
        v.json_cases[path.name] = row
    expected = {(fold, predictor, variant, generation, order, permission)
                for fold in range(5) for predictor in ("P1", "P2")
                for variant in ("N1", "N2", "N3", "N4", "N5") for generation in range(3)
                for order in (17, 29, 43) for permission in ("main", "pair")}
    v.check(identities == expected, "B_unique_complete_path_identities")
    return {"paths": len(paths), "shared_logging_contexts": len(shared), "different_query_questions": 500}


def verify_binary(v, train, folds):
    paths = sorted((v.private / "binary").glob("*.json"))
    v.check(len(paths) == 30, "C_30_fit_groups")
    counts, total_fits, identities = Counter(), 0, set()
    for path in paths:
        row, arrays = v.json(path), v.arrays(path.with_suffix(".npz"))
        fold, predictor, mode = row["fold"], row["predictor"], row["mode"]
        identities.add((fold, predictor, mode))
        history = folds[fold]["history"]
        counts["primary" if mode in ("text-only", "task-known") else "metadata_sensitivity"] += 1
        v.check(mode in ("text-only", "task-known") or predictor == "P2", "C_metadata_sensitivity_scope", path)
        v.check(row["identity"]["history_rewards"] == hash_array(train["Y"][history]), "C_history_reward_hash", path)
        for name, checksum in row["identity"].items():
            if name != "history_rewards":
                v.check(sha(v.public / name) == checksum, "C_frozen_source_hash", path, name)
        v.shape(arrays, "predictions", (4, 120, 20), path)
        v.check(np.array_equal(arrays["query"], folds[fold]["stream"]), "C_outer_query_ids", path)
        v.check(len(row["models"]) == 4, "C_four_registered_families", path)
        v.check([model["family"] for model in row["models"]] ==
                ["gaussian", "gaussian-correlated", "logistic", "probit"], "C_registered_family_order", path)
        for model in row["models"]:
            search = model["search"]
            v.check(len(search) == 3 and {entry["alpha"] for entry in search} == {1., 10., 100.}, "C_alpha_grid", path)
            key = lambda item: (float(item["brier"]["balanced_v2"]), -float(item["alpha"]))
            winner = min(search, key=key)
            v.check(model["alpha"] == winner["alpha"], "C_history_only_alpha_winner", path)
            metadata = model["metadata"]
            v.check(metadata["history_indices_sha256"] == hash_array(history)
                    and metadata["history_embedding_sha256"] == hash_array(train["E"][history]), "C_feature_history_hash", path)
            v.check(metadata["outcome_arrays_read"] is False and metadata["pca_and_standardization_fit"] == "historical questions only",
                    "C_PCA_metadata_permission_declaration", path)
            for candidate in search:
                held = []
                v.check(len(candidate["inner_fits"]) == 3, "C_three_inner_question_folds", path)
                for fit in candidate["inner_fits"]:
                    a, b = set(fit["fit_questions"]), set(fit["query_questions"])
                    v.check(len(a) == 320 and len(b) == 160 and not a & b and a | b == set(history), "C_inner_training_validation_isolation", path)
                    held.extend(fit["query_questions"])
                v.check(len(held) == len(set(held)) == 480, "C_inner_OOF_every_history_question_once", path)
            total_fits += 10
        v.check(np.all((arrays["predictions"][2:] >= 0) & (arrays["predictions"][2:] <= 1)), "C_binary_probability_range", path)
        v.json_cases[path.name] = row
    v.check(counts == Counter(primary=20, metadata_sensitivity=10) and total_fits == 1200, "C_20plus10_groups_1200_fits")
    expected = {(fold, predictor, mode) for fold in range(5) for predictor in ("P1", "P2")
                for mode in ("text-only", "task-known")}
    expected |= {(fold, "P2", mode) for fold in range(5) for mode in ("subtask-oracle", "subtask-predicted")}
    v.check(identities == expected, "C_unique_complete_fit_identities")
    return {"groups": len(paths), **dict(counts), "inner_and_full_fits": total_fits}


def verify_online(v, train, folds, refs):
    frozen = v.json(v.private / "BINARY_ONLINE_SELECTION_FROZEN.json")
    v.check(frozen["query_used_for_selection"] is False, "C_online_query_not_used_for_selection")
    v.check(hash_json({key: value for key, value in frozen.items() if key != "signature"}) == frozen["signature"], "C_online_selection_signature")
    for name, checksum in frozen["source_hashes"].items():
        v.check(sha(v.public / name) == checksum, "C_online_frozen_source", detail=name)
    choices = {hash_json(choice): choice for choice in frozen["choices"]}
    v.check(len(choices) == 40, "C_online_forty_frozen_models")
    paths = sorted((v.private / "binary_online").glob("*.json"))
    v.check(len(paths) == 720, "C_online_720_paths")
    refit_checkpoints, identities = 0, set()
    for path in paths:
        row, arrays = v.json(path), v.arrays(path.with_suffix(".npz"))
        meta, selection = row["meta"], row["model_selection"]
        identities.add((meta["fold"], meta["predictor"], meta["mode"], meta["family"],
                        meta["generation_index"], meta["orderseed"], meta["permission"]))
        v.check(meta["generation_seed"] == (42, 137, 2024)[meta["generation_index"]], "C_online_generation_identity", path)
        v.check(hash_json(selection) in choices, "C_online_selection_in_frozen_registry", path)
        selected_path = Path(selection["selection_path"])
        v.check(sha(selected_path) == selection["selection_sha256"], "C_online_original_C_selection_hash", path)
        logger_path = Path(row["logger_source"]["path"])
        v.check(sha(logger_path) == row["logger_source"]["sha256"], "C_online_B_logging_hash", path)
        expected = hash_json({"selection": frozen["signature"], "source_hashes": frozen["source_hashes"],
                              "logger_sha256": sha(logger_path), "meta": meta,
                              "inputs": {key: value.get("sha256") for key, value in refs.items() if isinstance(value, dict)}})
        v.check(expected == row["signature"] == str(arrays["signature"]), "C_online_path_identity", path)
        v.check(row["query_feedback_used"] is False and row["legal_support_bundles_consumed"] == 20, "C_online_legal_support_only", path)
        v.check(row["roles"] == {"historical_questions": 480, "support_questions": 20, "query_questions": 100}, "C_online_role_counts", path)
        v.check(np.array_equal(arrays["query"], folds[meta["fold"]]["audit_query"]), "C_online_disjoint_query", path)
        for key in ("initial_pred", "final_pred"):
            v.shape(arrays, key, (100, 20), path)
            v.check(np.all((arrays[key] >= 0) & (arrays[key] <= 1)), "C_online_probability_bounds", path, key)
        v.shape(arrays, "chosen", (21, 100), path)
        v.shape(arrays, "accuracy", (21, 100), path)
        verify_cost(v, row["cost"], meta["permission"], 20, 20, path)
        refit_checkpoints += len(row["full_refit_comparisons"])
    expected = {(fold, predictor, mode, family, generation, order, permission)
                for fold in range(5) for predictor in ("P1", "P2") for mode in ("text-only", "task-known")
                for family in ("logistic", "probit") for generation in range(3)
                for order in (17, 29, 43) for permission in ("main", "pair")}
    v.check(identities == expected, "C_online_unique_complete_path_identities")
    return {"paths": len(paths), "frozen_models": len(choices), "evaluator_full_refit_checkpoints": refit_checkpoints}


def verify_decision(v, train, folds, teams):
    gate = v.json(v.private / "D_GATE_AND_FREEZE.json")
    v.check(gate["live_expected_paths"] == 165 and gate["snapshot_candidates"] == 1200, "D_frozen_workload")
    v.check(sha(v.public / "BINARY_COMPARISON_RESULTS.json") == gate["C_results_sha256"], "D_C_gate_result_hash")
    for name, checksum in gate["source_hashes"].items():
        v.check(sha(v.public / name) == checksum, "D_frozen_source_hash", detail=name)
    source = (v.public / "run_decision_audit.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    live = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "live")
    calls = defaultdict(list)
    for node in ast.walk(live):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls[node.func.attr].append(node.lineno)
    v.check(min(calls["service"]) < min(calls["buy"]) < min(calls["update"]), "D_source_current_service_before_feedback_update")
    paths = sorted((v.private / "decision_live").glob("*.json"))
    v.check(len(paths) == 165, "D_165_live_paths")
    live_identities, snapshot_identities = set(), set()
    for path in paths:
        row = v.json(path)
        live_identities.add((row["fold"], row["generation_index"], row["which"], row["method"]))
        v.check(row["generation_seed"] == (42, 137, 2024)[row["generation_index"]], "D_generation_identity", path)
        order, generation = folds[row["fold"]]["orders"][17], row["generation_index"]
        traces, feedback = row["traces"], row["feedback"]
        v.check(len(traces) == 120 and len(feedback) == 20, "D_complete120_service20_feedback", path)
        events = {event["tick"]: event for event in feedback}
        resources = dict(member_calls=0, aggregator_calls=0, gold=0, scoring_ops=0, input_tokens=0, output_tokens=0, tokens=0, source_model_wall_s=0.)
        for tick, trace in enumerate(traces):
            q, main = trace["q"], trace["main"]
            v.check(q == order[tick] and trace["tick"] == tick, "D_frozen_order", path)
            v.check(trace["reward"] == train["Y"][q, generation, main], "D_frozen_current_selected_quality", path)
            for stem, actions, call in (("member", teams[main], "member_calls"), ("aggregate", [main], "aggregator_calls")):
                resources[call] += len(actions)
                for suffix, key in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"), ("wall_time_s", "source_model_wall_s")):
                    resources[key] += float(train["cost"][stem + "_" + suffix][generation, q, list(actions)].sum())
            if tick in events:
                event = events[tick]
                v.check(trace["purchase"] and event["q"] == q and event["actions"] == [main]
                        and event["raw_scores"] == [float(train["Y"][q, generation, main])], "D_only_frozen_current_raw_score_revealed", path)
                resources["gold"] += 1
                resources["scoring_ops"] += 1
            else:
                v.check(not trace["purchase"], "D_purchase_has_authorized_observation", path)
            resources["tokens"] = resources["input_tokens"] + resources["output_tokens"]
            v.check(all(np.isclose(value, trace["cumulative"][key], rtol=1e-11, atol=1e-8)
                        for key, value in resources.items()), "D_rebuilt_prefix_ledger", path)
        verify_cost(v, row["cost"], "main", 20, 120, path)
    snapshots = sorted((v.private / "decision_snapshots").glob("*.json"))
    v.check(len(snapshots) == 30, "D_30_snapshot_groups")
    candidates = 0
    for path in snapshots:
        row = v.json(path)
        snapshot_identities.add((row["fold"], row["generation_index"], row["which"]))
        v.check(len(row["candidates"]) == 40, "D_40_candidates_per_snapshot_group", path)
        by_step = defaultdict(list)
        for candidate in row["candidates"]:
            by_step[candidate["step"]].append(candidate)
            v.check(candidate["actual_evaluation_role"].startswith("auditor-only"), "D_counterfactual_query_never_actor_feedback", path)
            permission = "pair" if len(candidate["actions"]) == 2 else "main"
            if permission == "pair":
                v.check(len(set(teams[candidate["actions"][0]]) ^ set(teams[candidate["actions"][1]])) == 2, "D_snapshot_neighbor_permission", path)
            verify_cost(v, candidate["counterfactual_cost"], permission, 1, 1, path)
            candidates += 1
        v.check(set(by_step) == {0, 5, 10, 15} and all(len(rows) == 10 for rows in by_step.values()), "D_fixed_four_states_ten_actions", path)
    v.check(candidates == 1200, "D_1200_counterfactual_candidates")
    methods = {"E3": ("Random", "Boundary", "Parameter-IG", "Decision-Aware", "Calibrated-DA", "Probability-DA"),
               "E4": ("Random", "Boundary", "Parameter-IG", "Decision-Aware", "Calibrated-DA")}
    expected_live = {(fold, generation, which, method) for fold in range(5) for generation in range(3)
                     for which, named_methods in methods.items() for method in named_methods}
    expected_snapshots = {(fold, generation, which) for fold in range(5) for generation in range(3) for which in methods}
    v.check(live_identities == expected_live, "D_unique_complete_live_identities")
    v.check(snapshot_identities == expected_snapshots, "D_unique_complete_snapshot_identities")
    return {"live_paths": len(paths), "snapshot_groups": len(snapshots), "snapshot_candidates": candidates,
            "timing_evidence": "source control-flow ordering plus saved trace/feedback/cumulative-ledger consistency; not an OS trace"}


def public_scan(v, train):
    qids = [str(value) for value in train["ids"]]
    reference_path = v.root / "artifacts/private/routing_redesign_20261104/derived/v2/selector_train_references_v1.jsonl"
    questions = [read_row["question"] for read_row in (json.loads(line) for line in reference_path.read_text(encoding="utf8").splitlines())]
    tokens = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})")
    assignments = re.compile(rb"(?i)(?:password|api_key|secret_token|access_token)\s*[:=]\s*[\"'][^\"'\r\n]{6,}[\"']")
    text_files = 0
    structural_findings = []

    def fields(value, path, location="$"):
        if isinstance(value, dict):
            for key, item in value.items():
                where = location + "." + key
                if key == "raw_scores" and isinstance(item, list) and item:
                    structural_findings.append({"file": path.name, "field": where, "kind": "private_raw_observations_in_public"})
                if key in {"gold", "reference", "answer", "reference_answer"} and isinstance(item, str) and item.strip():
                    structural_findings.append({"file": path.name, "field": where, "kind": "review_string_answer_field", "value_sha256": hashlib.sha256(item.encode()).hexdigest()})
                fields(item, path, where)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                fields(item, path, location + f"[{index}]")

    for path in sorted(v.public.rglob("*")):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        if path.suffix == ".gz":
            blob = gzip.decompress(blob)
        secret_hits = len(tokens.findall(blob)) + len(assignments.findall(blob))
        v.check(secret_hits == 0, "public_secret_pattern_scan", path, {"hits": secret_hits})
        if path.suffix not in {".json", ".gz", ".py", ".md", ".txt", ".csv"}:
            continue
        text_files += 1
        text = blob.decode("utf-8-sig")
        id_hits = sum(qid in text for qid in qids)
        question_hits = sum(len(question) >= 24 and (question in text or json.dumps(question, ensure_ascii=True)[1:-1] in text)
                            for question in questions)
        v.check(id_hits == 0 and question_hits == 0, "public_no_original_question_ids_or_text", path,
                {"id_hits": id_hits, "raw_question_hits": question_hits})
        if path.suffix in {".json", ".gz"}:
            fields(json.loads(text), path)
    v.check(not structural_findings, "public_no_raw_gold_or_feedback_fields", detail=structural_findings[:20])
    return {"text_files_scanned": text_files, "structural_findings": structural_findings,
            "methods": ["exact train question ID/text matching", "JSON answer/feedback fields", "credential-format patterns"],
            "limitations": "Pattern scans do not prove absence of all possible secrets; image pixels are outside text-content matching"}


def manifest_check(v):
    path = v.public / "PUBLIC_DELIVERY_MANIFEST.json"
    if not path.exists():
        return {"status": "PENDING_ROOT_MANIFEST"}
    manifest = read(path)
    records = manifest.get("files", manifest.get("artifacts", []))
    if isinstance(records, dict):
        records = [{"path": key, **value} for key, value in records.items()]
    v.check(bool(records), "public_manifest_nonempty", path)
    recorded = set()
    for record in records:
        relative = record.get("path", record.get("relative_path", record.get("file")))
        target = Path(relative)
        if not target.is_absolute():
            target = v.public / target
        v.check(target.resolve().is_relative_to(v.public.resolve()), "public_manifest_path_scope", path, relative)
        recorded.add(target.resolve().relative_to(v.public.resolve()).as_posix())
        v.check(target.exists() and sha(target) == record["sha256"], "public_manifest_hash", path, relative)
        expected_size = record.get("bytes", record.get("size_bytes", record.get("size")))
        if expected_size is not None:
            v.check(target.stat().st_size == expected_size, "public_manifest_bytes", path, relative)
        if "git_blob" in record:
            payload = target.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            v.check(blob == record["git_blob"], "public_manifest_git_blob", path, relative)
    actual = {file.relative_to(v.public).as_posix() for file in v.public.rglob("*") if file.is_file()}
    exceptions = {"PUBLIC_DELIVERY_MANIFEST.json", "DELIVERY_INDEPENDENT_AUDIT.json"}
    v.check(recorded == actual - exceptions, "public_manifest_exact_coverage", path,
            {"missing": sorted(actual - exceptions - recorded), "unexpected": sorted(recorded - (actual - exceptions))})
    v.check(exceptions <= actual and not recorded & exceptions, "public_manifest_two_explicit_self_reference_exceptions", path)
    v.check(manifest.get("separately_versioned_verification_report") == "DELIVERY_INDEPENDENT_AUDIT.json",
            "public_receipt_versioning_rule", path)
    v.check(sha(v.public / "verify_e4_delivery.py") == sha(__file__), "executed_verifier_matches_public_source", path)
    return {"status": "CHECKED", "records": len(records), "public_files": len(actual), "sha256": sha(path),
            "self_reference_exceptions": sorted(exceptions),
            "separate_versioning_scope": "Manifest and receipt excluded from their own content-hash cycle; Git commit binding is parent integration responsibility"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"E:\语言模型组合路由"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("DELIVERY_INDEPENDENT_AUDIT.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    v = Verification(args.root)
    try:
        common, e3, train, refs = import_dependencies(args.root)
        identity, e3_sources = e3.signature()
        v.check(identity == EXPECTED_E3, "original_E3_signature_unchanged", detail=identity)
        splits = common.e4_splits(train)
        folds = splits["folds"]
        v.check(len(train["ids"]) == len(set(train["ids"])) == len(set(train["canonical"])) == 600, "600_original_independent_identities")
        for fold in folds:
            v.check(not set(fold["history"]) & set(fold["stream"])
                    and not set(fold["audit_support"]) & set(fold["audit_query"]), "original_question_roles_disjoint")
        sections = {"A": verify_learning(v, train, folds)}
        print("A_checked", flush=True)
        sections["B"] = verify_posterior(v, train, folds, common.TEAMS)
        print("B_checked", flush=True)
        sections["C"] = verify_binary(v, train, folds)
        print("C_checked", flush=True)
        sections["C_online"] = verify_online(v, train, folds, refs)
        print("C_online_checked", flush=True)
        sections["D"] = verify_decision(v, train, folds, common.TEAMS)
        print("D_checked", flush=True)
        # Also open derived private NPZ archives outside per-case directories.
        for path in sorted(v.private.glob("*.npz")):
            v.arrays(path)
        privacy = public_scan(v, train)
        manifest = manifest_check(v)
        config = read(v.public / "E4_EXPERIMENT_CONFIG.json")
        v.check(config["llm_calls"] == 0 and config["question_folds"] == 5
                and config["data"] == "V2 selector_train600 only", "frozen_train_only_contract")
        v.check(config["primary_weight"] == "GSM8K 1/2, BBH 1/2 with original question weights within BBH"
                and config["secondary_weight"] == "GSM8K 1/2, each of 27 BBH subtasks 1/54", "frozen_primary_secondary_estimands")
        common_source = (v.public / "e4_common.py").read_text(encoding="utf8")
        asset_source = (args.root / "research/decision_acquisition_e3/public/assets.py").read_text(encoding="utf8")
        v.check('bundle[group]["selector_train"]' in asset_source and "load_assets(ROOT)" in common_source,
                "train_only_loader_source_allowlist")
        report = {"schema": "e4_independent_delivery_verification_v1",
                  "status": "ISSUES_FOUND" if v.errors else "PASS_PENDING_PUBLIC_MANIFEST" if manifest["status"] == "PENDING_ROOT_MANIFEST" else "PASS",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "audit_script_sha256": sha(__file__),
                  "sections": sections, "counts": dict(v.counts), "errors": v.errors, "warnings": v.warnings,
                  "registered_undefined_arrays": v.nullable_arrays, "checked_case_files": v.files,
                  "E3_identity": identity, "E3_source_signature": e3_sources, "public_privacy_scan": privacy,
                  "public_manifest": manifest,
                  "independence": {"original_questions": 600, "controlled_B_query_questions": 500,
                    "generation_seeds": [42, 137, 2024], "generation_and_order_repeats_are_new_independent_questions": False},
                  "access_evidence": {"basis": "source allowlists, frozen configuration, asset identities, metadata qualification and saved legal-feedback artifacts",
                    "source_splits_loaded_by_auditor": ["selector_train"], "fresh_or_sealed_score_payloads_loaded_by_auditor": False,
                    "OS_file_access_trace_collected": False, "global_OS_nonaccess_claim": False},
                  "new_llm_calls_by_auditor": 0, "elapsed_s": time.perf_counter() - start}
    except Exception as error:
        report = {"schema": "e4_independent_delivery_verification_v1", "status": "AUDIT_INCOMPLETE",
                  "error_type": type(error).__name__, "error": str(error), "errors_so_far": v.errors,
                  "counts": dict(v.counts), "new_llm_calls_by_auditor": 0}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        raise
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"status": report["status"], "errors": len(v.errors), "counts": dict(v.counts),
                      "receipt_sha256": sha(args.output), "elapsed_s": report["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
