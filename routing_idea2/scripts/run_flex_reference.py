"""Run the complete paid-supervision, smoothed Flex coverage-reference grid."""
from pathlib import Path
import argparse
import json
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from budget_router.simulation import MatrixEnvironment, save_simulation
from budget_router.flex_simulation import authorize_flex_pair, run_flex_simulation
from budget_router.flex_validation import make_identity, flex_settings, FlexEpisodeStore
from budget_router.ranking_evaluation import write_cost_curve
from budget_router.stage_validation import (read_json, input_identity, episode_specs,
    episode_name, atomic_json, immutable_json, artifact_reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("online-matrix", "online-manifest", "calibration-matrix", "calibration-manifest",
                 "e3-gate", "mandatory-e4-gate", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--config", default=str(ROOT/"configs/experiment_v1.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    inputs = input_identity(ROOT, online_matrix=args.online_matrix, online_manifest=args.online_manifest,
        calibration_matrix=args.calibration_matrix, calibration_manifest=args.calibration_manifest, config_path=args.config)
    online = MatrixEnvironment.from_npz(args.online_matrix, args.online_manifest, expected_split="online_stream")
    calibration = MatrixEnvironment.from_npz(args.calibration_matrix, args.calibration_manifest, expected_split="online_calibration")
    if online.software_test_only or calibration.software_test_only or len(online) != 600 or len(calibration) != 200:
        raise ValueError("Production Flex reference requires original real 600/200 matrices; no toy mode")
    authorization = authorize_flex_pair(online, calibration, e3_gate_path=args.e3_gate,
        e4_gate_path=args.mandatory_e4_gate, current_inputs=inputs)
    identity = make_identity(ROOT, inputs, e3_gate_path=args.e3_gate, mandatory_e4_gate_path=args.mandatory_e4_gate)
    settings = flex_settings(config)
    output = Path(args.output_dir)
    store = FlexEpisodeStore(output, identity, resume=args.resume, parent=False)
    specs = episode_specs(identity["grid"]); receipts, measurements = [], []
    for spec in specs:
        result = store.load(spec)
        if result is None:
            attempt = output/"attempts"/(episode_name(spec)+"."+uuid.uuid4().hex+".json")
            record = {"schema": "flex_cpu_training_attempt_v1", "spec": spec, "status": "STARTED",
                "started_at_unix": time.time(), "run_identity_sha256": identity["identity_sha256"],
                "scope": "actual CPU FP64200-step paid-only fit and matrix replay; no new GPU/model generation"}
            atomic_json(attempt, record); cpu, wall = time.process_time(), time.perf_counter()
            try:
                result = run_flex_simulation(online, calibration, seed=spec["seed"], label_rate=spec["label_rate"],
                    delay=spec["feedback_delay"], settings=settings, flex_authorization=authorization)
                def writer(value, folder):
                    save_simulation(value, folder); write_cost_curve(value, folder/"cost_curve.csv")
                store.commit(result, spec, writer); record["status"] = "COMMITTED"
            except BaseException as error:
                record.update(status="INTERRUPTED_OR_FAILED", error_type=type(error).__name__)
                raise
            finally:
                record.update(process_time_s=time.process_time()-cpu, wall_time_s=time.perf_counter()-wall)
                atomic_json(attempt, record)
        receipts.append(artifact_reference(output/episode_name(spec)/"episode_receipt.json"))
        summary = result["summary"]
        measurements.append({"spec": spec, "quality_task_macro": summary["quality_task_macro"],
            "candidate_coverage": summary["candidate_coverage"], "aggregation_loss": summary["aggregation_loss"],
            "actual_resources": summary["budget"]["spent"], "limits": summary["budget"]["limits"],
            "controller_process_time_s": summary["controller_process_time_s"],
            "controller_wall_time_s": summary["controller_wall_time_s"],
            "training_process_time_s": summary["runtime_measurements"]["training_process_time_s"],
            "source_service_execution_wall_time_s": summary["source_service_execution_wall_time_s"]})
        atomic_json(output/"progress.json", {"status": "RUNNING_FLEX_REFERENCE_GRID", "validated_complete_episodes": len(receipts),
            "required_episodes": len(specs), "last_episode": episode_name(spec), "research_complete": False})
        print(json.dumps({"validated_episode": episode_name(spec), "complete": len(receipts), "required": len(specs),
            "native_system": "NOT_RUN", "research_complete": False}), flush=True)
    report = {"schema": "flex_grid_summary_v1", "status": "COMPLETE_FLEX_GRID_AWAITING_INDEPENDENT_ACCEPTANCE",
        "run_identity_sha256": identity["identity_sha256"], "grid": identity["grid"], "episode_receipts": receipts,
        "resource_measurements_by_episode": measurements, "variant": identity["variant"], "native_system": "NOT_RUN",
        "physical_generation_replicates": 1, "algorithm_seeds_are_not_independent_generations": True,
        "research_complete": False}
    immutable_json(output/"grid_summary.json", report)
    return report


if __name__ == "__main__": main()
