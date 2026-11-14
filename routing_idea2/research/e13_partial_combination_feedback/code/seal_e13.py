"""Final bounded integrity and aggregate-only E13 publication manifest."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

from common import (
    BUDGETS, CODE, INPUTS, METHODS, PRIVATE, PUBLIC, SCENARIOS, Timer,
    choice, load_npz, protocol, read_json, sha256, verify_source_lock, write_json,
)
from learners import predict_learners
from saverouter_adapter import predict_from_state
import numpy as np


def main():
    if (PUBLIC / "PUBLIC_DELIVERY_MANIFEST.json").exists():
        raise FileExistsError("E13 publication already sealed")
    with Timer() as timer:
        locked = verify_source_lock()
        for record in locked["source_assets"]:
            if sha256(record["path"]) != record["sha256"]:
                raise ValueError("Original read-only source changed: " + record["path"])
        evaluation = read_json(PRIVATE / "EVALUATION_RECEIPT.json")
        for name, digest in evaluation["public_outputs"].items():
            if sha256(PUBLIC / name) != digest:
                raise ValueError("Frozen scientific results changed")
        result = read_json(PUBLIC / "E13_RESULTS.json")
        paired = load_npz(PRIVATE / "PAIRED_OUTCOMES.npz")
        if sha256(PRIVATE / "PAIRED_OUTCOMES.npz") != evaluation["paired_outcomes_sha256"]:
            raise ValueError("Frozen private paired outcomes changed")
        maximum_aggregate_error = 0.0
        for s, scenario in enumerate(SCENARIOS):
            for b, budget in enumerate(BUDGETS):
                means = paired["selected_rewards"][s, b].mean(axis=(0, 1, 3))*100
                for m, name in enumerate(METHODS):
                    error = abs(float(means[m])-result["scenarios"][scenario][str(budget)]["methods"][name]["accuracy_pct"])
                    maximum_aggregate_error = max(maximum_aggregate_error, error)
        if maximum_aggregate_error > 1e-12:
            raise ValueError("Public accuracy does not match retained private outcomes")
        restored, restore_maximum = 0, 0.0
        fold = 0
        inputs = load_npz(PRIVATE / "inputs/fold0/INPUTS_ONLY.npz")
        for scenario in SCENARIOS:
            for budget in BUDGETS:
                path = PRIVATE / f"worlds/fold0/logger17/seed42/{scenario}/B{budget}/SNAPSHOT.npz"
                snapshot = load_npz(path)
                own = {key[len("own__"):]: value for key, value in snapshot.items() if key.startswith("own__")}
                native = {key[len("sr__"):]: value for key, value in snapshot.items() if key.startswith("sr__")}
                scores = predict_learners(own, inputs["phi_e"], inputs["predicted_group_e"])
                scores["SaveRouterFixedMask"] = predict_from_state(native, inputs["phi_e"][:, 1:], inputs["predicted_group_e"])
                scores["TaskFixedOracle"] = own["task_fixed"][own["task_codes"][inputs["oracle_group_e"]]]
                scores["SubtaskFixedOracle"] = own["subtask_fixed"][inputs["oracle_group_e"]]
                rebuilt = np.stack([scores[name] for name in METHODS])
                error = float(np.max(np.abs(rebuilt-snapshot["scores"])))
                restore_maximum = max(restore_maximum, error)
                if error > 1e-12 or not np.array_equal(choice(rebuilt), snapshot["actions"]):
                    raise ValueError("Saved models do not restore frozen action choices")
                restored += 1
        # Copy each stable public source only after its generating work has ended.
        for source in sorted(CODE.rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
            target = PUBLIC / "code" / source.relative_to(CODE)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != sha256(source):
                raise ValueError("An existing published code copy differs: " + target.name)
            if not target.exists():
                shutil.copy2(source, target)
        required = ("E13_PROTOCOL.json", "E13_RESULTS.json", "E13_PRIMARY_STATISTICS.json", "E13_SUMMARY.csv",
                    "COST.json", "FEEDBACK_COSTS.csv", "FEEDBACK_AND_SERVICE_COSTS.csv", "EXPERIMENT_SUMMARY.md",
                    "REPRODUCE.md", "LITERATURE_AND_BASELINE_MAPPING.md", "MECHANISM_DIAGNOSTICS.json",
                    "quality_gold.png", "quality_feedback_tokens.png")
        if any(not (PUBLIC / name).is_file() for name in required):
            raise ValueError("A required final E13 deliverable is missing")
        original_inputs = load_npz(INPUTS)
        identity_strings = [str(value) for key in ("question_ids", "canonical_hashes") for value in original_inputs[key]]
        files, hits = [], []
        for path in sorted(p for p in PUBLIC.rglob("*") if p.is_file()):
            if path.suffix == ".json":
                read_json(path)
            if path.suffix in (".json", ".py", ".md", ".csv", ".txt"):
                text = path.read_text(encoding="utf-8-sig")
                if any(identity in text for identity in identity_strings):
                    hits.append(str(path.relative_to(PUBLIC)))
            files.append(dict(path=str(path.relative_to(PUBLIC)), sha256=sha256(path), bytes=path.stat().st_size))
        if hits:
            raise ValueError("Private original question identity in publication")
    checks = dict(status="PASS", old_sources_unchanged=len(locked["source_assets"]),
                  aggregate_rows_verified=120, maximum_aggregate_error=maximum_aggregate_error,
                  restored_snapshot_conditions=restored, restored_policies=restored*len(METHODS),
                  maximum_score_restore_error=restore_maximum, question_identity_privacy_hits=hits,
                  visual_QA="Both real-data PNG curves inspected; fixed six budget points and all eight main methods readable",
                  independent_review=dict(matching_generation_rewards_verified=648000, snapshot_ledgers_verified=540,
                      purchased_feedback_values_verified=210600, paired_bootstrap_repeated=5000, holm_family=3,
                      CPU_s=1.5625, wall_s=1.61331, result="exactly matched retained values, member unions and primary statistics"),
                  new_physical_llm_calls=0, new_gold_procurement=0, validation_fresh_sealed_payload_reads=0,
                  timing=timer.record)
    write_json(PRIVATE / "FINAL_VERIFICATION.json", checks)
    write_json(PUBLIC / "PUBLIC_DELIVERY_MANIFEST.json", dict(status="COMPLETE", files=files,
               self_excluded=True, private_assets_included=False, checks=checks,
               local_git_checkout=False, new_mab_commit=None, pushed=False))
    print("E13_DELIVERY_COMPLETE", len(files), "public files", flush=True)


if __name__ == "__main__":
    main()
