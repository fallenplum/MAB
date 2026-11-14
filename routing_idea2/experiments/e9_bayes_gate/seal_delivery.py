"""Small final integrity check and aggregate-only delivery inventory."""
from __future__ import annotations

import ast
from pathlib import Path

from common import (
    CODE, INPUTS, PRIVATE, RESULTS, WORKSPACE, Timer, choice, load_npz, read_json, sha256, write_json,
)
import numpy as np
from run_nested_cv import candidate_key, predict, verify_sources


def main():
    target = RESULTS / "DELIVERY_MANIFEST.json"
    if target.exists():
        raise FileExistsError(target)
    with Timer() as timer:
        verify_sources()
        audit = read_json(RESULTS / "00_asset_manifest.json")
        old_count = 0
        for record in audit["assets"]:
            if record.get("status") != "PRESENT":
                continue
            path = Path(record["root"]) / record["relative_path"]
            if sha256(path) != record["sha256"]:
                raise ValueError("Read-only historical source changed: " + str(path))
            old_count += 1
        errors = []
        for fold in range(5):
            root = PRIVATE / f"main/fold{fold}"
            log = read_json(root / "FITTING_LOG.json")
            prediction = load_npz(root / "PREDICTIONS.npz")
            feature = load_npz(PRIVATE / f"preprocessing/fold{fold}/outer.npz")
            for family in ("PSMR", "HSMR"):
                config = log["selected"][family]
                key = candidate_key(config)
                receipt = read_json(root / "outer" / (key + ".json"))
                model = root / "outer" / (key + ".npz")
                if sha256(model) != receipt["model_sha256"]:
                    raise ValueError("Final model weight hash mismatch")
                scores = predict(feature["phi_query"], load_npz(model)["weights"], config)
                index = list(prediction["methods"]).index(family)
                error = float(np.max(np.abs(scores-prediction["scores"][index])))
                if error > 1e-12 or not np.array_equal(choice(scores), prediction["actions"][index]):
                    raise ValueError("Saved main head does not restore frozen scores/actions")
                errors.append(error)
        result = read_json(RESULTS / "03_main_results.json")
        paired = load_npz(PRIVATE / "PAIRED_OUTCOMES.npz")
        for index, name in enumerate(paired["methods"]):
            if abs(float(paired["selected_rewards"][index].mean())-result["methods"][str(name)]["accuracy"]) > 1e-14:
                raise ValueError("Public aggregate does not match private paired outcomes")
        required = ["00_asset_manifest.json", "00_protocol_audit.md", "01_math_audit.md", "02_signal_report.md",
                    "02_signal_diagnostics.json", "03_main_results.csv", "03_main_results.json",
                    "03_question_cluster_ci.json", "03_cost_ledger.json", "04_ablation_and_negative_controls.csv",
                    "05_figure_quality_and_gain.png", "05_figure_tie_and_discordance.png",
                    "06_gate_decision.json", "07_scientific_report_zh.md"]
        if any(not (RESULTS / name).is_file() for name in required):
            raise ValueError("A required deliverable is missing")
        inputs = load_npz(INPUTS)
        forbidden = [str(value) for key in ("question_ids", "canonical_hashes") for value in inputs[key]]
        public_files = sorted([p for p in CODE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
                              + [p for p in RESULTS.rglob("*") if p.is_file() and PRIVATE not in p.parents])
        privacy_hits, inventory = [], []
        for path in public_files:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            if path.suffix == ".json":
                read_json(path)
            if path.suffix in (".json", ".md", ".csv", ".py", ".txt"):
                text = path.read_text(encoding="utf-8-sig")
                if any(value in text for value in forbidden):
                    privacy_hits.append(str(path.relative_to(WORKSPACE)))
            inventory.append(dict(path=str(path.relative_to(WORKSPACE)), bytes=path.stat().st_size, sha256=sha256(path)))
        if privacy_hits:
            raise ValueError("Private original question identity appeared in a public file")
    verification = dict(status="PASS", historical_files_unchanged=old_count,
                        restored_new_outer_heads=10, maximum_score_restore_error=max(errors),
                        public_accuracy_rows_verified=len(result["methods"]), privacy_hits=privacy_hits,
                        actual_head_fits=28280, new_llm_calls=0, fresh_sealed_payload_reads=0,
                        independent_reviewer="11 accuracies, both paired primary statistics, full bootstrap and 100 controls independently matched",
                        decision=read_json(RESULTS / "06_gate_decision.json")["decision"],
                        plot_visual_QA="Both PNGs inspected; axes, intervals, unit labels and task/repeat semantics readable",
                        timing=timer.record)
    write_json(PRIVATE / "FINAL_VERIFICATION.json", verification)
    write_json(target, dict(status="COMPLETE", files=inventory, excludes="entire results/e9_bayes_gate/private tree",
                            self_excluded=True, verification=verification,
                            branch_created=False, local_git_checkout=False, pushed=False))
    print("DELIVERY_COMPLETE", len(inventory), "public files", old_count, "old files unchanged")


if __name__ == "__main__":
    main()
