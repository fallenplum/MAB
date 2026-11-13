"""Audit frozen E8 artifacts and aggregate costs without reading evaluation rewards."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(root):
    root = Path(root)
    index_path = root / "private/training/E8_FROZEN_TRAINING_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not index.get("complete") or len(index["folds"]) != 5:
        raise RuntimeError("All five E8 folds must be frozen before the final audit")
    summaries = {}
    receipts_count, fits_count = 0, 0
    for fold in index["folds"]:
        if sha(fold["predictions_path"]) != fold["predictions_sha256"]:
            raise RuntimeError("Frozen predictions changed")
        for variant, receipt in fold["variants"].items():
            directory = root / "private/training" / ("fold"+str(fold["fold"])) / variant
            for name, expected in receipt["artifact_hashes"].items():
                if sha(directory / name) != expected:
                    raise RuntimeError("A frozen E8 model artifact changed")
            logs = json.loads((directory / "FIT_LOG.json").read_text(encoding="utf-8"))
            fit_logs = [record["diagnostics"] for candidate in logs["candidates"] for record in candidate["fits"]]
            fit_logs.append(logs["final_diagnostics"])
            if not all(item["success"] for item in fit_logs):
                raise RuntimeError("A failed fit was placed in the frozen results")
            item = summaries.setdefault(variant, dict(
                folds=0, historical_inner_and_final_fits=0,
                sum_training_cpu_seconds=0., sum_training_wall_seconds=0.,
                sum_optimizer_cpu_seconds=0., sum_inference_cpu_seconds=0.,
                sum_inference_wall_seconds=0., max_process_peak_working_set_bytes=0,
                qp_fits=0, max_qp_iterations=0, max_qp_primal_residual=0.,
                max_qp_dual_residual=0., max_qp_objective_discrepancy=0.,
                qp_solver_counts={}, max_qp_iterations_by_solver={},
                max_qp_constraint_violation=0., max_clarabel_primal_dual_gap=0.,
                all_declared_stopping_rules_succeeded=True, selected_parameters_by_fold={}))
            item["folds"] += 1
            item["historical_inner_and_final_fits"] += len(fit_logs)
            item["sum_training_cpu_seconds"] += logs["total_cpu_seconds"]
            item["sum_training_wall_seconds"] += logs["total_wall_seconds"]
            item["sum_optimizer_cpu_seconds"] += sum(fit["cpu_seconds"] for fit in fit_logs)
            item["sum_inference_cpu_seconds"] += logs["inference"]["cpu_seconds"]
            item["sum_inference_wall_seconds"] += logs["inference"]["wall_seconds"]
            item["max_process_peak_working_set_bytes"] = max(
                item["max_process_peak_working_set_bytes"],
                max((fit.get("process_peak_working_set_bytes") or 0) for fit in fit_logs))
            item["selected_parameters_by_fold"][str(fold["fold"])] = logs["selected_parameters"]
            for fit in fit_logs:
                if fit.get("solver") in ("OSQP", "Clarabel"):
                    item["qp_fits"] += 1
                    solver = fit["solver"]
                    item["qp_solver_counts"][solver] = item["qp_solver_counts"].get(solver, 0)+1
                    item["max_qp_iterations_by_solver"][solver] = max(item["max_qp_iterations_by_solver"].get(solver, 0), fit["iterations"])
                    item["max_qp_iterations"] = max(item["max_qp_iterations"], fit["iterations"])
                    item["max_qp_primal_residual"] = max(item["max_qp_primal_residual"], fit["primal_residual"])
                    item["max_qp_dual_residual"] = max(item["max_qp_dual_residual"], fit["dual_residual"])
                    item["max_qp_objective_discrepancy"] = max(item["max_qp_objective_discrepancy"], fit["objective_discrepancy"])
                    item["max_qp_constraint_violation"] = max(item["max_qp_constraint_violation"], fit["constraint_violation"])
                    item["max_clarabel_primal_dual_gap"] = max(item["max_clarabel_primal_dual_gap"], fit.get("primal_dual_objective_gap", 0.))
            receipts_count += 1
            fits_count += len(fit_logs)
    if receipts_count != 70 or set(summaries) != set(index["variants"]):
        raise RuntimeError("The E8 frozen model set is incomplete")
    result = dict(schema="E8_training_cost_audit_v1", status="COMPLETE", models=receipts_count,
                  historical_inner_and_final_objective_fits=fits_count,
                  preprocessing_and_OOF_Ridge_fits_included_in_total_cpu=True,
                  additional_OOF_Ridge_fit_counts_not_in_objective_fit_counter=True,
                  physical_llm_calls=0, evaluation_rewards_read=False,
                  global_index_sha256=sha(index_path), protocol_sha256=index["protocol_sha256"],
                  source_lock_sha256=index["source_lock_sha256"],
                  run_wall_seconds=index["total_wall_seconds"], variants=summaries,
                  total_training_cpu_seconds=sum(v["sum_training_cpu_seconds"] for v in summaries.values()),
                  total_training_cpu_seconds_scope="Successfully completed models only; failed v1 fits and qualification are separately retained",
                  total_inference_wall_seconds=sum(v["sum_inference_wall_seconds"] for v in summaries.values()),
                  peak_memory_interpretation="Windows process peak working set since worker creation, not a fit-exclusive memory increment",
                  inference_timing_interpretation="One frozen 120-question batch per model; CPU timer resolution can yield zero for very short batches")
    for field, filename in (("original_run_cpu_lower_bound", "PROCESS_COST_LOWER_BOUND_V1.json"),
                            ("recovery_run_cpu_lower_bound", "RECOVERY_PROCESS_COST_LOWER_BOUND.json")):
        path = root / "private/training" / filename
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            result[field] = value.get("run_cpu_seconds_lower_bound", value.get("cumulative_cpu_seconds_snapshot_lower_bound"))
    result["run_wall_seconds_scope"] = index.get("total_wall_seconds_scope", "Initial complete run")
    target = root / "public/training/TRAINING_COST_SUMMARY.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    result = summarize(parser.parse_args().root)
    print(json.dumps({key: value for key, value in result.items() if key != "variants"}, ensure_ascii=False))
