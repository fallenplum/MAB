"""Post-hoc MC integration sensitivity on frozen E3 posterior snapshots.

Eight and thirty-two antithetic pairs are compared without changing any service
choice, purchase, feedback or primary statistical analysis. Thirty-two pairs are
a numerical reference, not ground truth or a parameter selected for deployment.
"""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
WORK = ROOT / "research/decision_acquisition_e3"
PUBLIC = WORK / "public"
PRIVATE = WORK / "private"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PUBLIC))
from assets import Env, SEEDS, TEAMS, algorithm_seed, load_assets, make_splits
from acquisition import fantasy_table, projected_scores
from predictors import ModelTemplate
from run_e3 import CONFIG, NEIGHBORS, signature


SNAPSHOT_TICKS = (10, 30, 50, 70, 90, 110)
ZERO_TOLERANCE = 1e-12


def _put(path, value, compressed=False):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":") if compressed else None).encode("utf-8")
    if compressed:
        payload = gzip.compress(payload, compresslevel=5, mtime=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_bytes(payload)
    os.replace(pending, path)


def _entries(stage):
    path = PRIVATE / (stage + "_INDEX.json")
    if not path.exists():
        raise ValueError("Completed stage index is absent: " + str(path))
    selected = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        meta = item["meta"]
        if (meta["predictor"] in ("P1", "P2") and meta["stage"] == stage
                and meta["method"] == "decision_cost" and meta["B"] == 20
                and meta["history_size"] == 240 and meta["gen"] == 42
                and meta["order"] == 17 and meta.get("delete_group") is None):
            key = (meta["predictor"], meta["fold"])
            if key in selected and item["path"] != selected[key]["path"]:
                raise ValueError("One frozen path has conflicting cache references")
            selected[key] = item
    expected = {(predictor, fold) for predictor in ("P1", "P2") for fold in range(5)}
    if set(selected) != expected:
        raise ValueError(stage + " lacks the ten prespecified completed paths")
    return [selected[key] for key in sorted(selected)]


def _snapshot(posterior, q, main, meta, weights, tables, cost_oracle, trace):
    stage = meta["stage"]
    by_precision = {}
    for pairs in (8, 32):
        values, _ = projected_scores(posterior, q, main, NEIGHBORS[main], weights,
                                      tables[pairs], include_neighbors=stage != "B1", need_kg=True)
        by_precision[pairs] = values[1:] if stage == "B2" else values
    candidates = by_precision[8]
    if [item["actions"] for item in candidates] != [item["actions"] for item in by_precision[32]]:
        raise ValueError("Integration precisions received different legal candidates")
    # The cached main service has already generated exactly its three members.
    executed = {"members": set(TEAMS[main]), "aggregators": {main}}
    costs = []
    for candidate in candidates:
        estimate = cost_oracle.predict_incremental(candidate["actions"], executed, include_gold=stage != "B2")
        costs.append(estimate["priced_units"] - (0.001 if stage == "B2" else 0.0))
    costs = np.asarray(costs)
    if np.any(costs <= 0):
        raise ValueError("Expected incremental complete costs must be positive")
    kg_field, ig_field = ("extra_kg", "extra_ig") if stage == "B2" else ("kg", "ig")
    kg8 = np.asarray([item[kg_field] for item in candidates])
    kg32 = np.asarray([item[kg_field] for item in by_precision[32]])
    ig8 = np.asarray([item[ig_field] for item in candidates])
    ig32 = np.asarray([item[ig_field] for item in by_precision[32]])
    score8, score32 = kg8 / costs, kg32 / costs
    chosen8, chosen32 = int(np.argmax(score8)), int(np.argmax(score32))
    planned = int(trace.get("candidate_count", 0)) > 0
    reproduction = {"status": "not_claimed_original_log_is_unplanned_or_zero_placeholder"}
    if planned:
        kg_error = abs(float(kg8[chosen8]) - float(trace["chosen_kg"]))
        score_error = abs(float(score8[chosen8]) - float(trace["predicted_score"]))
        if max(kg_error, score_error) > 1e-10:
            raise ValueError("Eight-pair recomputation differs from an actually planned original snapshot")
        action_match = (set(candidates[chosen8]["actions"]) == set(trace["actions"])) if trace["purchase"] else None
        if action_match is False:
            raise ValueError("Recomputed eight-pair bundle differs from the purchased original bundle")
        reproduction = {"status": "planned_original_score_reproduced", "kg_abs_error": kg_error,
                        "score_abs_error": score_error, "purchased_bundle_matches": action_match}
    return {"meta": dict(meta), "q": int(q), "tick": int(trace["tick"]),
            "candidate_actions": [item["actions"] for item in candidates],
            "kg8": kg8.tolist(), "kg32": kg32.tolist(),
            "ig8": ig8.tolist(), "ig32": ig32.tolist(),
            "estimated_incremental_cost": costs.tolist(),
            "score8": score8.tolist(), "score32": score32.tolist(),
            "mc_argmax8": chosen8, "mc_argmax32": chosen32,
            "mc_argmax_agreement": chosen8 == chosen32,
            "mc32_score_opportunity_gap": max(0., float(score32[chosen32] - score32[chosen8])),
            "original8_reproduction": reproduction,
            "actor_service_and_feedback_unchanged": True}


def audit_run(run, train, template, fold, identity):
    if run.get("identity") != identity:
        raise ValueError("Source trace belongs to another execution core")
    meta = run["meta"]
    traces = run["traces"]
    order = np.asarray([int(trace["q"]) for trace in traces])
    if not np.array_equal(order, fold["orders"][17]):
        raise ValueError("Source trace order differs from the prespecified frozen order")
    events = {int(event["tick"]): event for event in run["feedback"]}
    if len(events) != len(run["feedback"]):
        raise ValueError("Duplicate original feedback ticks")
    posterior = template.clone()
    seed = algorithm_seed(meta["fold"], meta["order"], 0)
    tables = {pairs: fantasy_table(seed, pairs) for pairs in (8, 32)}
    oracle = Env(train, template.history, order, SEEDS.index(meta["gen"]), meta["B"])
    rows = []
    for tick, trace in enumerate(traces):
        q, main = int(trace["q"]), int(trace["main"])
        if int(np.argmax(posterior.predict(q))) != main or int(trace["tick"]) != tick:
            raise ValueError("Legal-feedback replay changed the original service choice")
        if tick in SNAPSHOT_TICKS:
            rows.append(_snapshot(posterior, q, main, meta, fold["reference_weights"], tables, oracle, trace))
        event = events.get(tick)
        if event is not None:
            if not trace["purchase"] or int(event["q"]) != q or int(event["source_gen"]) != 42:
                raise ValueError("Feedback has no matching original authorized purchase")
            if set(event["actions"]) != set(trace["actions"]):
                raise ValueError("Feedback bundle differs from source trace")
            posterior.update(q, event["actions"], event["raw_scores"])
        elif trace["purchase"]:
            raise ValueError("Original purchase is missing its raw observation receipt")
    if len(rows) != len(SNAPSHOT_TICKS):
        raise ValueError("Prespecified snapshots are incomplete")
    return rows


def _summarize(rows):
    g8 = np.concatenate([row["kg8"] for row in rows])
    g32 = np.concatenate([row["kg32"] for row in rows])
    absolute = np.abs(g8 - g32)
    meaningful_agreement = [row["mc_argmax_agreement"] for row in rows if row["meta"]["stage"] != "B1"]
    planned = [row["original8_reproduction"] for row in rows
               if row["original8_reproduction"]["status"] == "planned_original_score_reproduced"]
    gaps = np.asarray([row["mc32_score_opportunity_gap"] for row in rows if row["meta"]["stage"] != "B1"])
    return {"snapshots": len(rows), "candidate_values": len(g8),
            "mean_kg8": float(g8.mean()), "mean_kg32": float(g32.mean()),
            "mean_absolute_kg_difference": float(absolute.mean()),
            "p95_absolute_kg_difference": float(np.quantile(absolute, .95)),
            "max_absolute_kg_difference": float(absolute.max()),
            "exact_g8_zero_g32_positive_candidates": int(np.sum((g8 == 0) & (g32 > 0))),
            "tolerance_g8_zero_g32_positive_candidates": int(np.sum((g8 <= ZERO_TOLERANCE) & (g32 > ZERO_TOLERANCE))),
            "tolerance_g32_zero_g8_positive_candidates": int(np.sum((g32 <= ZERO_TOLERANCE) & (g8 > ZERO_TOLERANCE))),
            "snapshots_with_tolerance_zero_to_positive": sum(any(a <= ZERO_TOLERANCE and b > ZERO_TOLERANCE
                                                                 for a, b in zip(row["kg8"], row["kg32"])) for row in rows),
            "cost_argmax_comparison_snapshots": len(meaningful_agreement),
            "cost_argmax_agreement_count": sum(meaningful_agreement),
            "cost_argmax_agreement_fraction": float(np.mean(meaningful_agreement)) if meaningful_agreement else None,
            "mc32_score_opportunity_gap_mean": float(gaps.mean()) if len(gaps) else None,
            "mc32_score_opportunity_gap_max": float(gaps.max()) if len(gaps) else None,
            "max_ig_difference": max(abs(a - b) for row in rows for a, b in zip(row["ig8"], row["ig32"])),
            "planned_original_snapshots_checked": len(planned),
            "unplanned_original_placeholder_snapshots": len(rows) - len(planned),
            "planned_original_max_kg_error": max((item["kg_abs_error"] for item in planned), default=0.0),
            "planned_original_max_score_error": max((item["score_abs_error"] for item in planned), default=0.0)}


def main():
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    train, refs = load_assets(ROOT)
    splits = make_splits(train)
    identity, sources = signature()
    templates, rows = {}, []
    for stage in ("B1", "B2", "J"):
        for entry in _entries(stage):
            run = json.loads(gzip.decompress(Path(entry["path"]).read_bytes()))
            if run["meta"] != entry["meta"]:
                raise ValueError("Index metadata differs from cached path")
            meta = run["meta"]
            key = (meta["predictor"], meta["fold"])
            fold = splits["folds"][meta["fold"]]
            if key not in templates:
                templates[key] = ModelTemplate(train, fold["nested"][240], meta["predictor"], fold["references"][240], CONFIG)
            rows.extend(audit_run(run, train, templates[key], fold, identity))
        print(stage, "ten frozen paths audited", flush=True)
    if len(rows) != 180:
        raise ValueError("The fixed audit must contain exactly 180 snapshots")
    contexts = defaultdict(list)
    for row in rows:
        contexts[(row["meta"]["predictor"], row["meta"]["stage"])].append(row)
    report = {"schema": "e3_fixed_snapshot_kg_mc_precision_audit_v1",
              "classification": "POSTHOC_NUMERICAL_SENSITIVITY_NOT_PARAMETER_SELECTION",
              "fixed_design": {"folds": 5, "predictors": ["P1", "P2"], "stages": ["B1", "B2", "J"],
                               "generation_seed": 42, "order_seed": 17, "gold_budget": 20,
                               "history_size": 240, "snapshot_ticks_zero_based": list(SNAPSHOT_TICKS),
                               "fantasy_pairs": [8, 32], "numerical_zero_tolerance": ZERO_TOLERANCE},
              "overall": _summarize(rows),
              "by_predictor_stage": [{"predictor": key[0], "stage": key[1], **_summarize(contexts[key])}
                                     for key in sorted(contexts)],
              "core_identity": identity, "frozen_core_source_hashes": sources,
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_sha256": {key: value["sha256"] for key, value in refs.items() if isinstance(value, dict) and "sha256" in value},
              "service_decisions_replayed": 30 * 120, "new_strategy_runs": 0,
              "new_llm_calls": 0, "new_feedback_reveals": 0, "raw_question_ids_exported": False,
              "audit_wall_s": time.perf_counter() - started_wall,
              "audit_process_cpu_s": time.process_time() - started_cpu,
              "interpretation": ["The deployed experiment remains frozen at eight pairs",
                                 "Only numerical scores and hypothetical cost-normalized argmax values at identical original snapshots are compared",
                                 "Thirty-two pairs are a finite-MC reference, not true expected value or actual regret",
                                 "Original actor score reproduction is claimed only when candidate_count>0; zero placeholders carry no such claim",
                                 "No new service, purchase or feedback is selected, and no primary statistical result is replaced"]}
    _put(PRIVATE / "KG_PRECISION_AUDIT_STATES.json.gz", rows, compressed=True)
    _put(PUBLIC / "KG_PRECISION_AUDIT.json", report)
    print(json.dumps({"status": "PASS", "overall": report["overall"],
                      "audit_wall_s": report["audit_wall_s"], "new_llm_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
