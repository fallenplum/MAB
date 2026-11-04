"""Replay one immutable calibration question into the independent v2 ledger.

Only existing source receipts and one row of original matrix scores are used.
Every invocation writes a new private audit. Reusing its SQLite account verifies
execution/score/reference idempotence while recording new local audit overhead.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from budget_router_v2 import BudgetFeedbackLedger, ExecutionReceipt, ScoreObservation

PRIVATE = ROOT / "artifacts/private/routing_redesign_20261104"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_path(reference, allowed_root):
    path = Path(reference["path"])
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(allowed_root.resolve()), "Source escapes calibration scope")
    require(path.is_file() and sha_file(path) == reference["sha256"], "Immutable source hash mismatch")
    return path


def selected_records(path, question_id):
    result = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "online_calibration" or row.get("question_id") != question_id:
                continue
            checksum = row.pop("record_sha256", None)
            digest = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                               allow_nan=False).encode()).hexdigest()
            require(checksum == digest, "Selected source receipt checksum mismatch")
            row["record_sha256"] = checksum
            result.append(row)
    return result


def execution_receipt(row, canonical_question, role):
    require(row["cost_status"] == "recorded_actual_usage", "Unknown source usage must be reconciled")
    success = row["status"] == "success"
    usage = row["result"] if success else row.get("actual_usage")
    require(isinstance(usage, dict), "Source usage is missing")
    wall = usage.get("wall_time_s") if success else row.get("failure_wall_time_s")
    require(wall is not None and math.isfinite(wall) and wall >= 0, "Source wall time is unknown")
    for name in ("prompt_tokens", "completion_tokens"):
        require(type(usage.get(name)) is int and usage[name] >= 0, "Source token usage is unknown")
    output_hash = hashlib.sha256(usage["text"].encode()).hexdigest() if success else None
    return ExecutionReceipt(row["job_id"], row["attempt_id"], canonical_question, role,
                            row["model_fingerprint"], row["seed"], output_hash, row["status"],
                            usage["prompt_tokens"], usage["completion_tokens"], float(wall))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PRIVATE / "gold_reuse_audit")
    parser.add_argument("--question-index", type=int, default=0)
    args = parser.parse_args(argv)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    output = args.output_dir.resolve()
    require(output.is_relative_to(PRIVATE.resolve()), "Audit outputs must remain in the authorized private scope")
    manifest_path = ROOT / "data/derived/online_matrices_v1/online_calibration.manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["split"] == "online_calibration" and manifest["all_combinations_executed"] is True,
            "Complete immutable online calibration required")
    require(manifest["scorer"].get("no_model_calls") is True,
            "This replay adapter requires a local verifier; paid judges need their own execution receipts")
    matrix_path = manifest_path.with_name("online_calibration.npz")
    require(sha_file(matrix_path) == manifest["matrix_sha256"], "Calibration matrix hash mismatch")
    arrays = {}
    with np.load(matrix_path, allow_pickle=False) as archive:
        require(0 <= args.question_index < len(archive["question_ids"]), "Question index outside calibration")
        question_id = str(archive["question_ids"][args.question_index])
        for name in ("member_scores", "aggregate_scores", "member_execution_counts", "aggregate_execution_counts",
                     "member_prompt_tokens", "member_completion_tokens", "aggregate_prompt_tokens",
                     "aggregate_completion_tokens", "member_wall_time_s", "aggregate_wall_time_s"):
            arrays[name] = np.array(archive[name][args.question_index], copy=True)
    canonical = manifest["question_fingerprints"][args.question_index]
    # Proxy for the frozen source-reference lineage, independent of scorer and generation.
    reference_version = "v1-source-reference-lineage:" + manifest["split_manifest"]["sha256"]
    combos = list(itertools.combinations(range(6), 3))
    require(manifest["combinations"] == [list(team) for team in combos], "Combination column order changed")
    sources = [r for r in manifest["sources"] if r["kind"] in ("member_outputs", "aggregate_outputs")]
    require(sum(r["kind"] == "member_outputs" for r in sources) == 6 and
            sum(r["kind"] == "aggregate_outputs" for r in sources) == 1, "Six member and one aggregate sources required")
    groups = {}
    for source in sources:
        path = source_path(source, ROOT / "results/raw/online_calibration")
        role = "member" if source["kind"] == "member_outputs" else "aggregator"
        for row in selected_records(path, question_id):
            groups.setdefault(row["job_id"], []).append((role, row))
    successful = [(job, [(role, row) for role, row in attempts if row["status"] == "success"])
                  for job, attempts in groups.items()]
    require(len(groups) == 26 and all(len(rows) == 1 for _, rows in successful), "Exactly 26 successful source executions required")
    identities = {(m["model_id"], m["revision"]): i for i, m in enumerate(manifest["model_pool"])}
    alias_columns = {}
    for _, rows in successful:
        role, row = rows[0]
        if role == "member":
            result = row["result"]
            alias_columns[row["model_alias"]] = identities[(result["model_id"], result["model_revision"])]
    require(len(alias_columns) == len(set(alias_columns.values())) == 6, "Member column identities are incomplete")
    prepared = []
    for job, rows in successful:
        role, success = rows[0]
        column = (alias_columns[success["model_alias"]] if role == "member" else
                  combos.index(tuple(sorted(alias_columns[a] for a in success["members"]))))
        stem = "member" if role == "member" else "aggregate"
        receipts = [execution_receipt(row, canonical, role) for recorded_role, row in groups[job]
                    if recorded_role == role]
        require(len(receipts) == int(arrays[stem + "_execution_counts"][column]), "Source attempts differ from accepted matrix cost")
        require(sum(r.input_tokens for r in receipts) == int(arrays[stem + "_prompt_tokens"][column]) and
                sum(r.output_tokens for r in receipts) == int(arrays[stem + "_completion_tokens"][column]) and
                math.isclose(sum(r.wall_time_s for r in receipts), float(arrays[stem + "_wall_time_s"][column]),
                             rel_tol=1e-12, abs_tol=1e-9),
                "Actual source cost differs from accepted matrix")
        prepared.append((receipts, success, stem, column))
    scope = {"schema": "v2_one_question_replay_scope_v1", "canonical_question": canonical,
             "reference_version": reference_version, "manifest_sha256": sha_file(manifest_path),
             "matrix_sha256": manifest["matrix_sha256"], "sources": sources}
    output.mkdir(parents=True, exist_ok=True)
    scope_path = output / "scope.json"
    if scope_path.exists():
        require(read_json(scope_path) == scope, "Existing database scope differs, choose a new private directory")
    else:
        with scope_path.open("x", encoding="utf-8") as stream:
            json.dump(scope, stream, ensure_ascii=False, allow_nan=False, indent=2)
    invocation = uuid.uuid4().hex
    new_executions = new_scores = cache_hits = 0
    lookup_cpu = lookup_wall = 0.0
    fingerprints = []
    with BudgetFeedbackLedger(output / "ledger.sqlite") as ledger:
        before = ledger.summary()
        new_gold = ledger.request_gold(canonical, reference_version, now=0, available_at=0)
        ledger.arrive_gold(canonical, reference_version, now=0)
        for receipts, success, stem, column in prepared:
            cache_hits += int(ledger.has_success(success["job_id"]))
            for receipt in receipts:
                new_executions += int(ledger.record_execution(receipt))
            cpu, wall = time.process_time(), time.perf_counter()
            value = float(arrays[stem + "_scores"][column])
            cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
            lookup_cpu += cpu
            lookup_wall += wall
            new_scores += int(ledger.record_score(ScoreObservation(success["job_id"],
                manifest["scorer"]["version"], reference_version, value, 0,
                score_cpu_s=cpu, score_wall_s=wall), now=0))
            fingerprints.append({"execution_id": success["job_id"], "role": receipts[-1].role,
                "model_fingerprint": success["model_fingerprint"], "generation_seed": success["seed"],
                "output_hash": hashlib.sha256(success["result"]["text"].encode()).hexdigest(),
                "source_receipt_hashes": [row["record_sha256"] for _, row in groups[success["job_id"]]]})
        ledger.record_overhead(invocation, "planning", cpu_s=max(0.0, time.process_time() - started_cpu - lookup_cpu),
                               wall_s=max(0.0, time.perf_counter() - started_wall - lookup_wall))
        after = ledger.summary()
    require(after["gold_reveals"] == after["gold_arrived"] == 1 and after["scoring_ops"] == 26,
            "Reference reuse or score idempotence failed")
    require(after["roles"]["member"]["successful_executions"] == 6 and
            after["roles"]["aggregator"]["successful_executions"] == 20 and
            after["roles"]["judge"]["calls"] == 0, "Execution replay totals differ")
    audit = {"schema": "v2_gold_reuse_audit_v1", "status": "PASSED_RECORD_REPLAY_AUDIT",
             "scientific_run": False, "new_physical_model_calls": 0,
             "new_gold_reveals": int(new_gold), "new_execution_receipts": new_executions,
             "new_score_observations": new_scores, "cache_hits_before_replay": cache_hits,
             "before": before, "after": after, "execution_sources": fingerprints,
             "timing_scope": "Score clocks measure this invocation's one-row matrix lookup only. Planning overhead measures local source validation, ledger bookkeeping and replay excluding lookup clocks. Original source wall time comes from historical actual receipts. Each invocation adds its own local audit overhead. No learner update occurs.",
             "source_manifest_sha256": scope["manifest_sha256"], "matrix_sha256": scope["matrix_sha256"]}
    audit_path = output / ("audit_" + invocation + ".json")
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"status": audit["status"], "gold_reveals": after["gold_reveals"],
                      "scoring_ops": after["scoring_ops"], "new_execution_receipts": new_executions,
                      "audit_relative_path": audit_path.relative_to(ROOT).as_posix()}))
    return audit


if __name__ == "__main__":
    main()
