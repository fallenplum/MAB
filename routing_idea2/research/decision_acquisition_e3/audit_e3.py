"""Independently replay current E3 precheck ledgers and test hidden-label isolation.

Reads immutable experiment assets and cached runs. The only output is the
explicit audit receipt. In-memory episode replays never commit experiment files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def import_runner(root):
    public = root / "research/decision_acquisition_e3/public"
    os.environ["MAB_ASSET_ROOT"] = str(root)
    sys.path.insert(0, str(public))
    specification = importlib.util.spec_from_file_location("audited_e3_entry", public / "run_e3.py")
    require(specification is not None and specification.loader is not None, "Runner import unavailable")
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    return runner


def rebuild_run(run, train, teams, seeds):
    """Reconstruct execution unions without trusting cached ledger totals."""
    meta = run["meta"]
    generation = seeds.index(meta["gen"])
    traces = run["traces"]
    require(len(traces) == 120 and len({row["q"] for row in traces}) == 120, "Wrong query population")
    gold, scores, members, aggregates = set(), set(), set(), set()
    for tick, row in enumerate(traces):
        q, main = row["q"], row["main"]
        require(row["tick"] == tick and row["service_frozen_before_feedback"], "Service sequence differs")
        require(row["reward"] == train["Y"][q, generation, main], "Selected service reward differs")
        members.update((q, member) for member in teams[main])
        aggregates.add((q, main))
    for feedback in run["feedback"]:
        q = feedback["q"]
        row = traces[feedback["tick"]]
        require(row["q"] == q and row["purchase"] and row["main"] in feedback["actions"],
                "Feedback lacks a frozen current service")
        require(feedback["source_gen"] == meta["gen"] and feedback["received_before_update"],
                "Feedback source or update sequence differs")
        require(len(feedback["actions"]) == len(feedback["raw_scores"]), "Feedback shape differs")
        gold.add(str(train["canonical"][q]))
        for action, score in zip(feedback["actions"], feedback["raw_scores"]):
            require(score == train["Y"][q, generation, action], "Raw feedback differs from source cache")
            require((q, action) not in scores, "Duplicate raw-score update")
            scores.add((q, action))
            members.update((q, member) for member in teams[action])
            aggregates.add((q, action))
    resources = {"member_calls": len(members), "aggregator_calls": len(aggregates),
                 "gold": len(gold), "scoring_ops": len(scores), "input_tokens": 0,
                 "output_tokens": 0, "source_model_wall_s": 0.0}
    for stem, executed in (("member", members), ("aggregate", aggregates)):
        for q, action in executed:
            resources["input_tokens"] += int(train["cost"][stem + "_prompt_tokens"][generation, q, action])
            resources["output_tokens"] += int(train["cost"][stem + "_completion_tokens"][generation, q, action])
            resources["source_model_wall_s"] += float(train["cost"][stem + "_wall_time_s"][generation, q, action])
    resources["tokens"] = resources["input_tokens"] + resources["output_tokens"]
    stored = run["ledger"]["ledger"]
    require(all(np.isclose(value, stored["total"][name], rtol=1e-11, atol=1e-8)
                for name, value in resources.items()), "Independently rebuilt resource total differs")
    require(len(gold) <= meta["B"] and run["ledger"]["served_questions"] == 120, "Gold or service budget violated")
    require(run["ledger"]["original_score_observations"] == len(scores), "Scoring observation count differs")
    for account in ("service", "audit"):
        caps = run["ledger"][account + "_caps"]
        require(all(stored[account][name] <= maximum for name, maximum in caps.items()), "Hard resource cap violated")
    for name in stored["total"]:
        require(np.isclose(stored["service"][name] + stored["audit"][name], stored["total"][name],
                           rtol=1e-11, atol=1e-8), "Service plus audit does not equal total")
    return resources


def policy_trace(run):
    return [(row["q"], row["main"], row["purchase"], tuple(row["actions"])) for row in run["traces"]]


def update_trace(run):
    return [(row["tick"], row["q"], row["source_gen"], tuple(row["actions"]), tuple(row["raw_scores"]))
            for row in run["feedback"]]


def check_shared_trajectories(runs):
    grouped = {(row["meta"]["predictor"], row["meta"]["stage"], row["meta"]["method"], row["meta"]["B"]): row
               for row in runs}
    checks, full_information = {}, {}
    for predictor in ("P1", "P2", "P3"):
        ig = grouped[(predictor, "B1", "ig", 20)]
        ig_cost = grouped[(predictor, "B1", "ig_cost", 20)]
        require(policy_trace(ig) == policy_trace(ig_cost), "B1 IG/IGCost trajectories differ")
        b2 = [row for row in runs if row["meta"]["predictor"] == predictor and row["meta"]["stage"] == "B2"]
        schedules = [tuple(trace["tick"] for trace in row["traces"] if trace["purchase"]) for row in b2]
        require(len(b2) == 6 and len(set(schedules)) == 1 and len(schedules[0]) == 20,
                "B2 feedback schedules differ")
        require(all(row["ledger"]["ledger"]["audit"]["aggregator_calls"] == 18 for row in b2),
                "B2 neighbor execution count differs")
        decision = grouped[(predictor, "J", "decision_cost", 20)]
        matched = grouped[(predictor, "J", "matched_random", 20)]
        require([(row["purchase"], row["comparison"]) for row in decision["traces"]]
                == [(row["purchase"], row["comparison"]) for row in matched["traces"]],
                "Matched comparison schedule differs")
        full = grouped[(predictor, "A", "full_information", 120)]
        total = full["ledger"]["ledger"]["total"]
        require(total["member_calls"] == 720 and total["aggregator_calls"] == 2400
                and total["gold"] == 120 and total["scoring_ops"] == 2400, "Full-information cost differs")
        full_information[predictor] = full["ledger"]["ledger"]
        checks[predictor] = {"B1_ig_ig_cost_identical": True, "B2_common_gold_purchase_count": 20,
                             "B2_extra_neighbor_calls": 18, "J_matched_purchase_and_comparison_pattern": True}
    return checks, full_information


def hidden_label_flip(runner, train, splits, runs):
    """Change every unpurchased stream reward, retaining all legal observations."""
    source = next(row for row in runs if row["meta"]["predictor"] == "P2"
                  and row["meta"]["stage"] == "B1" and row["meta"]["method"] == "decision_cost"
                  and row["meta"]["B"] == 20)
    meta = dict(source["meta"])
    fold = splits["folds"][meta["fold"]]
    order = fold["orders"][meta["order"]]
    runner._DATA, runner._SPLITS, runner._TEMPLATES = train, splits, {}
    template, weights = runner.template_for(meta)
    reference = runner.episode(train, template, order, meta, weights, audit_future=True)
    require(policy_trace(reference) == policy_trace(source) and update_trace(reference) == update_trace(source),
            "Current-entry replay differs from the bound cached source episode")
    changed = dict(train)
    changed["Y"] = train["Y"].copy()
    changed["Y"][order] = 1.0 - changed["Y"][order]
    purchased = set()
    for feedback in source["feedback"]:
        q, generation = feedback["q"], runner.SEEDS.index(feedback["source_gen"])
        for action in feedback["actions"]:
            purchased.add((q, generation, action))
            changed["Y"][q, generation, action] = train["Y"][q, generation, action]
    require(np.array_equal(changed["Y"][template.history], train["Y"][template.history]),
            "Historical rewards changed during hidden-label intervention")
    changed_cells = int(np.count_nonzero(changed["Y"] != train["Y"]))
    require(changed_cells == len(order) * 3 * 20 - len(purchased), "Hidden-label intervention scope differs")
    counterfactual = runner.episode(changed, template, order, meta, weights, audit_future=True)
    require(policy_trace(reference) == policy_trace(counterfactual), "Hidden labels changed a service/purchase decision")
    require(update_trace(reference) == update_trace(counterfactual), "Hidden labels changed authorized updates")
    require(reference["ledger"]["ledger"] == counterfactual["ledger"]["ledger"], "Hidden labels changed resources")
    require(np.allclose([row["predicted_score"] for row in reference["traces"]],
                        [row["predicted_score"] for row in counterfactual["traces"]], rtol=1e-12, atol=1e-12),
            "Hidden labels changed acquisition scores")
    reward_changes = sum(a["reward"] != b["reward"]
                         for a, b in zip(reference["traces"], counterfactual["traces"]))
    require(reward_changes > 0, "Intervention did not alter evaluator-only service rewards")
    return {"status": "PASS", "meta": meta, "original_cached_replay_identical": True,
            "full_trace_questions": len(order), "preserved_purchased_cells": len(purchased),
            "flipped_hidden_reward_cells": changed_cells, "history_labels_unchanged": True,
            "main_actions_purchases_identical": True, "raw_score_updates_identical": True,
            "acquisition_scores_identical": True, "resource_ledger_identical": True,
            "evaluator_service_reward_changes": reward_changes,
            "auditor_future_reward_snapshots_enabled_in_both_replays": True,
            "scope": "One P2/B1/decision_cost gold20 fold0 generation42 order17 intervention"}


def audit(root):
    runner = import_runner(root)
    identity, source_signature = runner.signature()
    train, refs = runner.load_assets(root)
    splits = runner.make_splits(train)
    index_path = runner.PRIVATE / "PRECHECK_INDEX.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    require(len(index) == 88, "Expected complete 88-run current precheck")
    runs, receipts = [], []
    for entry in index:
        path = Path(entry["path"])
        require(path.resolve().is_relative_to(runner.PRIVATE.resolve()), "Cache path escaped private scope")
        run = read_run(path)
        require(run["identity"] == identity and run["meta"] == entry["meta"], "Current signature/run identity mismatch")
        resources = rebuild_run(run, train, runner.TEAMS, runner.SEEDS)
        receipts.append({"cache_file": path.name, "sha256": sha256(path), "meta": run["meta"],
                         "independently_rebuilt_total": resources, "status": "PASS"})
        runs.append(run)
    trajectories, full_information = check_shared_trajectories(runs)
    intervention = hidden_label_flip(runner, train, splits, runs)
    require(runner.signature() == (identity, source_signature), "Reviewed source changed during audit")
    require(index_path.read_bytes() == index_bytes, "Precheck index changed during audit")
    groups = Counter((row["meta"]["fold"], row["meta"]["gen"], row["meta"]["order"]) for row in runs)
    return {"schema": "independent_precheck_ledger_replay_audit_v2", "status": "PASS_FOR_BOUND_CURRENT_PRECHECK",
            "created_utc": datetime.now(timezone.utc).isoformat(), "reviewed_run_identity": identity,
            "signature_source": "Actual deployed run_e3.signature()", "reviewed_source_signature": source_signature,
            "audit_script_sha256": sha256(__file__),
            "precheck_index_sha256": hashlib.sha256(index_bytes).hexdigest(), "asset_refs": refs,
            "scope": {"runs": len(runs), "fold_generation_order_counts": [
                {"fold": fold, "generation_seed": generation, "order_seed": order, "runs": count}
                for (fold, generation, order), count in groups.items()], "history_size": 240,
                "stream_questions_per_run": 120, "classification": "development_precheck",
                "new_model_calls": 0, "fresh_or_sealed_payloads_opened": False,
                "modified_experiment_assets_or_core_code": False},
            "trajectory_checks": trajectories, "full_information_ledger_by_predictor": full_information,
            "hidden_label_flip_test": intervention, "independent_resource_mismatch_count": 0,
            "member_feedback_permission": {"raw_aggregate_scores_only": True,
                "member_score_requests": 0, "member_score_observations": 0},
            "source_review": {
                "planning_cost_estimates": "Legal historical per-member/per-combination means only",
                "service_timing": "Frozen service before score release; quality remains evaluator-only until legal buy",
                "future_reward_access": "Auditor-only snapshots; hidden-label intervention leaves policy and updates identical",
                "runtime_scope": "Episode process time plus separately recorded template fit or worker cache lookup; planning/update times separate",
                "GPU_scope": "Source call wall sums are available; GPU kernel seconds are unavailable"},
            "limitations": ["Receipt binds only this signature and its 88 current precheck artifacts",
                "One fold/generation/order does not establish scientific gains or all-seed stability",
                "Hidden-label intervention covers one full P2/B1/decision_cost path",
                "Scoring_ops are logical score-release operations over cached true scores",
                "Historical physical preparation remains separate from replay-equivalent costs",
                "Full-information diagnostic is outside the primary constrained-feedback budget"],
            "scientific_success_claimed": False, "runs": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path(r"E:\语言模型组合路由"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("PRECHECK_LEDGER_AUDIT.json"))
    args = parser.parse_args()
    try:
        result = audit(args.asset_root)
    except Exception as error:
        result = {"schema": "independent_precheck_ledger_replay_audit_v2", "status": "FAIL",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "error_type": type(error).__name__, "error": str(error), "scientific_success_claimed": False}
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        raise
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"], "reviewed_identity": result["reviewed_run_identity"],
                      "runs": result["scope"]["runs"], "receipt_sha256": sha256(args.output),
                      "hidden_label_flip": result["hidden_label_flip_test"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
