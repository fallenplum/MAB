"""Replay already-authorized E3 feedback for evaluator-only mechanism diagnostics.

This module makes no acquisition or service decisions and invokes no LLM. The
original service choice is checked at every tick. Unpurchased rewards are never
used for posterior updates. Remaining-query rewards are read only by the auditor.
"""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import rankdata


ROOT = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
WORK = ROOT / "research/decision_acquisition_e3"
CORE = WORK / "public"
PRIVATE = WORK / "private"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CORE))
from assets import SEEDS, algorithm_seed, load_assets, make_splits, sha256
from acquisition import fantasy_table, projected_scores
from predictors import ModelTemplate
from run_e3 import CONFIG, NEIGHBORS, signature

_TRAIN = None
_SPLITS = None
_CORE_IDENTITY = None
_TEMPLATES = {}


def _put(path, value, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":") if compressed else None).encode("utf-8")
    if compressed:
        payload = gzip.compress(payload, compresslevel=5, mtime=0)
    pending = path.with_name(path.name + ".pending")
    pending.write_bytes(payload)
    os.replace(pending, path)


def _predict(posterior, rows):
    template = posterior.template
    coefficient = posterior.mean.reshape(template.question_dimension, template.action_dimension)
    return template.X[rows] @ coefficient @ template.B.T


def _copy_posterior(posterior):
    # Geometry and historical templates are immutable during every replay. All
    # mutable posterior state is deep-copied without duplicating that geometry.
    return deepcopy(posterior, {id(posterior.template): posterior.template,
                                id(posterior.refF): posterior.refF})


def _snapshot(train, remaining, generation, before, after):
    """Evaluator-only comparison. Its output never enters a posterior or policy."""
    count = len(remaining)
    if not count:
        return None
    old = train["Y"][remaining, generation, before]
    new = train["Y"][remaining, generation, after]
    corrected, harmed = int(np.sum(new > old)), int(np.sum(new < old))
    return {"remaining_questions": count, "changed": int(np.sum(before != after)),
            "corrected": corrected, "harmed": harmed, "net_reward": corrected - harmed,
            "net_rate": float(np.mean(new - old)),
            "changed_rate": float(np.mean(before != after)),
            "correction_rate": corrected / count, "harm_rate": harmed / count}


def replay_run(run, train, template, weights, *, core_identity=None):
    """Reconstruct one existing path and return private selected-feedback rows."""
    meta = run["meta"]
    if core_identity is not None and run.get("identity") != core_identity:
        raise ValueError("Cached run belongs to a different frozen execution core")
    if meta["method"] != "decision_cost" or meta["stage"] not in ("B1", "B2", "J"):
        raise ValueError("Replay is restricted to the declared decision_cost mechanisms")
    posterior = template.clone()
    traces = run["traces"]
    order = np.asarray([int(trace["q"]) for trace in traces])
    if len(set(order)) != len(order) or set(order) & set(template.history):
        raise ValueError("Replay stream is duplicated or overlaps history")
    feedback = {}
    for event in run["feedback"]:
        tick = int(event["tick"])
        if tick in feedback:
            raise ValueError("Multiple feedback bundles at one tick")
        if int(event["source_gen"]) != int(meta["gen"]):
            raise ValueError("Feedback generation seed differs from service path")
        feedback[tick] = event
    generation = SEEDS.index(int(meta["gen"]))
    seed = algorithm_seed(meta["fold"], meta["order"], 0)
    fantasies = fantasy_table(seed, CONFIG["fantasy_pairs"])
    rows, kg_errors, snapshot_checks = [], [], 0
    for tick, trace in enumerate(traces):
        q, main = int(trace["q"]), int(trace["main"])
        if int(trace["tick"]) != tick or int(np.argmax(posterior.predict(q))) != main:
            raise ValueError("Replayed pre-feedback service decision differs from cached trace")
        event = feedback.get(tick)
        if event is None:
            if trace["purchase"]:
                raise ValueError("Accepted feedback is absent from original observation receipts")
            continue
        if not trace["purchase"] or int(event["q"]) != q:
            raise ValueError("Feedback receipt has no matching original purchase")
        actions = list(map(int, event["actions"]))
        values = np.asarray(event["raw_scores"], dtype=float)
        if set(actions) != set(trace["actions"]) or main not in actions:
            raise ValueError("Original feedback actions differ from service trace")
        candidates, _ = projected_scores(posterior, q, main, NEIGHBORS[main], weights,
                                         fantasies, include_neighbors=True, need_kg=True)
        selected = [candidate for candidate in candidates if set(candidate["actions"]) == set(actions)]
        if len(selected) != 1:
            raise ValueError("Purchased bundle has no matching legal one-step candidate")
        selected = selected[0]
        planned = int(trace.get("candidate_count", 0)) > 0
        kg_field = "extra_kg" if meta["stage"] == "B2" else "kg"
        if planned:
            error = abs(float(selected[kg_field]) - float(trace["chosen_kg"]))
            kg_errors.append(error)
            if error > 1e-10:
                raise ValueError("Replayed Gaussian KG differs from the original acquisition snapshot")
        remaining = order[tick + 1:]
        before = np.argmax(_predict(posterior, remaining), axis=1) if len(remaining) else np.empty(0, int)
        main_only = None
        if meta["stage"] == "B2" and len(actions) == 2:
            main_only = _copy_posterior(posterior)
            main_only.update(q, [main], [float(values[actions.index(main)])])
        posterior.update(q, actions, values)
        after = np.argmax(_predict(posterior, remaining), axis=1) if len(remaining) else np.empty(0, int)
        joint_snapshot = _snapshot(train, remaining, generation, before, after)
        logged = trace.get("future_feedback_influence")
        if joint_snapshot is not None:
            if logged is None or any(joint_snapshot[field] != logged[field] for field in
                                     ("remaining_questions", "changed", "corrected", "harmed", "net_reward")):
                raise ValueError("Replayed whole-bundle snapshot differs from existing evaluator trace")
            snapshot_checks += 1
        if main_only is not None:
            main_after = np.argmax(_predict(main_only, remaining), axis=1) if len(remaining) else np.empty(0, int)
            aligned_snapshot = _snapshot(train, remaining, generation, main_after, after)
            alignment = "joint_minus_main_only_with_identical_pre_feedback_prior"
        else:
            aligned_snapshot = joint_snapshot
            alignment = "joint_minus_pre_feedback"
        eligible = joint_snapshot is not None and planned and (meta["stage"] != "B2" or len(actions) == 2)
        allowed_candidates = ([candidates[0]] if meta["stage"] == "B1" else
                              candidates[1:] if meta["stage"] == "B2" else candidates)
        candidate_rows = [{"actions": list(candidate["actions"]),
                           "kg": float(candidate[kg_field]),
                           "ig": float(candidate["extra_ig"] if meta["stage"] == "B2" else candidate["ig"]),
                           "variance": float(candidate["variance"]),
                           "pair_variance": float(candidate["pair_variance"]),
                           "selected": set(candidate["actions"]) == set(actions)}
                          for candidate in allowed_candidates]
        rows.append({"meta": dict(meta), "q": q, "tick": tick, "main": main,
                     "task": str(trace["task"]), "sub": str(trace["sub"]),
                     "actions": actions, "planned_acquisition": planned,
                     "primary_calibration_eligible": eligible,
                     "prediction_target": kg_field, "target_alignment": alignment,
                     "kg": float(selected[kg_field]),
                     "ig": float(selected["extra_ig"] if meta["stage"] == "B2" else selected["ig"]),
                     "variance": float(selected["variance"]),
                     "pair_variance": float(selected["pair_variance"]),
                     "aligned_future_snapshot": aligned_snapshot,
                     "joint_future_snapshot": joint_snapshot,
                     "candidates_without_unpurchased_scores": candidate_rows})
    if len(rows) != len(feedback):
        raise ValueError("Some original feedback receipts did not occur in the replay")
    return {"meta": dict(meta), "rows": rows,
            "qualification": {"service_decisions_checked": len(traces),
                              "legal_feedback_bundles_replayed": len(rows),
                              "whole_bundle_snapshot_matches": snapshot_checks,
                              "original_kg_max_error": max(kg_errors, default=0.0),
                              "all_service_choices_identical": True,
                              "new_strategy_decisions": 0, "new_feedback_reveals": 0,
                              "new_llm_calls": 0}}


def _correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    status = "DESCRIPTIVE"
    if len(x) < 3:
        status = "insufficient_original_questions"
    elif np.ptp(x) == 0 or np.ptp(y) == 0:
        status = "constant_variable"
    if status != "DESCRIPTIVE":
        return {"status": status, "original_questions": len(x), "pearson": None, "spearman": None}
    return {"status": status, "original_questions": len(x),
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
            "p_value": None, "confidence_interval": None}


def _cluster(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["primary_calibration_eligible"]:
            groups[int(row["q"])].append(row)
    result = []
    for q in sorted(groups):
        values = groups[q]
        result.append({"task": values[0]["task"], "sub": values[0]["sub"],
                       "repeated_events": len(values),
                       "kg": float(np.mean([row["kg"] for row in values])),
                       "ig": float(np.mean([row["ig"] for row in values])),
                       "variance": float(np.mean([row["variance"] for row in values])),
                       "pair_variance": float(np.mean([row["pair_variance"] for row in values])),
                       "net_rate": float(np.mean([row["aligned_future_snapshot"]["net_rate"] for row in values]))})
    return result


def _correlation_summary(questions):
    return {field + "_vs_aligned_future_net_rate": _correlation(
        [row[field] for row in questions], [row["net_rate"] for row in questions])
        for field in ("kg", "ig", "variance", "pair_variance")}


def _counterexamples(rows):
    packages = defaultdict(list)
    for row in rows:
        if not row["planned_acquisition"]:
            continue
        for candidate in row["candidates_without_unpurchased_scores"]:
            packages[(int(row["q"]), int(row["main"]), tuple(sorted(candidate["actions"])))].append((row, candidate))
    points = []
    for key in sorted(packages):
        values = packages[key]
        points.append({"task": values[0][0]["task"], "sub": values[0][0]["sub"],
                       "kg": float(np.mean([candidate["kg"] for _, candidate in values])),
                       "ig": float(np.mean([candidate["ig"] for _, candidate in values])),
                       "variance": float(np.mean([candidate["variance"] for _, candidate in values])),
                       "pair_variance": float(np.mean([candidate["pair_variance"] for _, candidate in values])),
                       "repeated_candidate_snapshots": len(values)})
    result = {}
    for field in ("variance", "pair_variance"):
        if not points:
            result[field] = {"status": "no_planned_candidate_snapshots", "examples": []}
            continue
        high = float(np.quantile([row[field] for row in points], .75))
        low = float(np.quantile([row["kg"] for row in points], .25))
        examples = [row for row in points if row[field] >= high and row["kg"] <= low]
        examples.sort(key=lambda row: (-row[field], row["kg"]))
        result[field] = {"status": "DESCRIPTIVE", "candidate_question_packages": len(points),
                         "rule": "uncertainty >= within-context 75th percentile and KG <= 25th percentile",
                         "uncertainty_threshold": high, "kg_threshold": low,
                         "matching_packages": len(examples), "examples": examples[:10],
                         "raw_question_ids_exported": False}
    return result


def aggregate(replays):
    contexts = defaultdict(list)
    for replay in replays:
        meta = replay["meta"]
        contexts[(meta["predictor"], meta["stage"], meta["history_size"], meta["B"])].extend(replay["rows"])
    result = []
    for key in sorted(contexts):
        rows = contexts[key]
        questions = _cluster(rows)
        subtasks = sorted({row["sub"] for row in questions if row["task"] == "bbh"})
        result.append({"predictor": key[0], "stage": key[1], "history_size": key[2], "B": key[3],
                       "purchased_events": len(rows), "calibration_events": sum(row["primary_calibration_eligible"] for row in rows),
                       "all_questions": _correlation_summary(questions),
                       "by_task": {task: _correlation_summary([row for row in questions if row["task"] == task])
                                   for task in ("gsm8k", "bbh")},
                       "by_bbh_subtask": {sub: _correlation_summary([row for row in questions if row["task"] == "bbh" and row["sub"] == sub])
                                          for sub in subtasks},
                       "high_uncertainty_low_kg": _counterexamples(rows)})
    return {"schema": "e3_legal_feedback_mechanism_replay_v1", "contexts": result,
            "qualification": {"paths": len(replays),
                              "service_decisions_checked": sum(item["qualification"]["service_decisions_checked"] for item in replays),
                              "legal_feedback_bundles_replayed": sum(item["qualification"]["legal_feedback_bundles_replayed"] for item in replays),
                              "original_kg_max_error": max((item["qualification"]["original_kg_max_error"] for item in replays), default=0.0)},
            "new_strategy_runs": 0, "new_llm_calls": 0, "new_feedback_reveals": 0,
            "raw_question_ids_exported": False,
            "inference": "Average repeated generation/order observations within original q before descriptive Pearson/Spearman; no independent-sample p-values",
            "target_alignment": {"B1/J": "whole feedback update versus its pre-feedback snapshot",
                                 "B2": "joint main+neighbor versus main-only update from the identical pre-feedback posterior"},
            "limitations": ["Feedback was selected by the realized acquisition policy",
                            "Remaining-query snapshots overlap and differ from the historical KG reference distribution",
                            "Frozen branch gains are not additive final-path effects",
                            "Unselected candidate statistics use only features and posterior, never unpurchased output scores",
                            "Post-hoc replay CPU cost is reported separately from the unchanged original policy cost"]}


def _entries(stage, precheck):
    index = PRIVATE / ("PRECHECK_INDEX.json" if precheck else stage + "_INDEX.json")
    if not index.exists():
        raise ValueError("Required completed index is absent: " + str(index))
    selected = {}
    for entry in json.loads(index.read_text(encoding="utf-8")):
        meta = entry["meta"]
        if (meta["stage"] == stage and meta["predictor"] in ("P1", "P2")
                and meta["method"] == "decision_cost" and meta["B"] == 20
                and meta["history_size"] == 240 and meta.get("delete_group") is None):
            key = tuple(meta[field] for field in ("predictor", "stage", "history_size", "B", "fold", "gen", "order"))
            if key in selected and str(selected[key]["path"]) != str(entry["path"]):
                raise ValueError("One replay path has conflicting source cache files")
            selected[key] = entry
    expected = 1 if precheck else 45
    for predictor in ("P1", "P2"):
        count = sum(entry["meta"]["predictor"] == predictor for entry in selected.values())
        if count != expected:
            raise ValueError(f"{stage}/{predictor} requires {expected} completed original paths, got {count}")
    return [selected[key] for key in sorted(selected)]


def _initialize_worker(expected_identity):
    global _TRAIN, _SPLITS, _CORE_IDENTITY, _TEMPLATES
    identity, _ = signature()
    if identity != expected_identity:
        raise ValueError("Frozen core changed before audit worker initialization")
    _TRAIN, _ = load_assets(ROOT)
    _SPLITS = make_splits(_TRAIN)
    _CORE_IDENTITY = identity
    _TEMPLATES = {}


def _replay_entry(entry):
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    path = Path(entry["path"])
    run = json.loads(gzip.decompress(path.read_bytes()))
    if run["meta"] != entry["meta"]:
        raise ValueError("Run receipt metadata differs from its index")
    meta = run["meta"]
    key = (meta["predictor"], meta["fold"])
    fold = _SPLITS["folds"][meta["fold"]]
    if key not in _TEMPLATES:
        _TEMPLATES[key] = ModelTemplate(_TRAIN, fold["nested"][240], meta["predictor"], fold["references"][240], CONFIG)
    if [int(trace["q"]) for trace in run["traces"]] != fold["orders"][meta["order"]].tolist():
        raise ValueError("Run service order differs from frozen original order")
    replay = replay_run(run, _TRAIN, _TEMPLATES[key], fold["reference_weights"], core_identity=_CORE_IDENTITY)
    replay["source_path"] = str(path)
    replay["source_sha256"] = sha256(path)
    replay["audit_runtime"] = {"process_cpu_s": time.process_time() - started_cpu,
                                "wall_s": time.perf_counter() - started_wall}
    return replay


def _collect_stage(entries, executor, stage):
    if executor is None:
        results = [_replay_entry(entry) for entry in entries]
    else:
        futures = [executor.submit(_replay_entry, entry) for entry in entries]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0:
                print(stage, len(results), "replayed", flush=True)
    unique = {}
    for result in results:
        meta = result["meta"]
        key = tuple(meta[field] for field in ("predictor", "stage", "history_size", "B", "fold", "gen", "order"))
        if key in unique:
            raise ValueError("A replay result would duplicate an original path")
        unique[key] = result
    return [unique[key] for key in sorted(unique)]


def main():
    global _TRAIN, _SPLITS, _CORE_IDENTITY
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("B1", "B2", "J", "all"), default="all")
    parser.add_argument("--stageB1", action="store_const", dest="stage", const="B1")
    parser.add_argument("--stageB2", action="store_const", dest="stage", const="B2")
    parser.add_argument("--precheck", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    train, references = load_assets(ROOT)
    splits = make_splits(train)
    identity, core_sources = signature()
    _TRAIN, _SPLITS, _CORE_IDENTITY = train, splits, identity
    all_replays = []
    stages = ("B1", "B2", "J") if args.stage == "all" else (args.stage,)
    suffix = "_PRECHECK" if args.precheck else ""
    pool = (ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize_worker, initargs=(identity,))
            if args.workers > 1 else nullcontext(None))
    with pool as executor:
        for stage in stages:
            replays = _collect_stage(_entries(stage, args.precheck), executor, stage)
            report = aggregate(replays)
            report["classification"] = "PRECHECK_QUALIFICATION" if args.precheck else "POSTHOC_LEGAL_FEEDBACK_AUDIT"
            report["core_identity"] = identity
            _put(PRIVATE / ("MECHANISM_AUDIT_ROWS_" + stage + suffix + ".json.gz"), replays, compressed=True)
            _put(CORE / ("MECHANISM_AUDIT_" + stage + suffix + ".json"), report)
            all_replays.extend(replays)
            print(stage, "PASS", len(replays), "paths", flush=True)
    combined = aggregate(all_replays)
    combined.update({"core_identity": identity, "core_source_hashes": core_sources,
                     "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "input_sha256": {name: ref.get("sha256") for name, ref in references.items() if isinstance(ref, dict) and "sha256" in ref},
                     "classification": "PRECHECK_QUALIFICATION" if args.precheck else "POSTHOC_LEGAL_FEEDBACK_AUDIT",
                     "stages_included": list(stages),
                     "audit_wall_s": time.perf_counter() - started_wall,
                     "audit_controller_process_cpu_s": time.process_time() - started_cpu,
                     "audit_replay_process_cpu_s": sum(item["audit_runtime"]["process_cpu_s"] for item in all_replays),
                     "audit_workers": args.workers,
                     "audit_cpu_scope": "Controller CPU only; spawned-worker CPU is excluded from this field"})
    if args.stage == "all":
        _put(PRIVATE / ("MECHANISM_AUDIT_ROWS" + suffix + ".json.gz"), all_replays, compressed=True)
        _put(CORE / ("MECHANISM_AUDIT" + suffix + ".json"), combined)
    else:
        _put(CORE / ("MECHANISM_AUDIT_" + args.stage + suffix + ".json"), combined)
    print(json.dumps({"status": "PASS", "qualification": combined["qualification"],
                      "audit_wall_s": combined["audit_wall_s"], "new_llm_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
