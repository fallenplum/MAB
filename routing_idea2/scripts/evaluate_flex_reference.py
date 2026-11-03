"""Independently replay the full Flex reference and issue technical E4 gates."""
from pathlib import Path
import argparse
import json
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from budget_router.flex_validation import accept_flex_reference, make_e4_full_gate
from budget_router.stage_validation import read_json, atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", required=True)
    parser.add_argument("--flex-gate", required=True)
    parser.add_argument("--e4-full-gate", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.grid_dir)
    identity = read_json(directory/"grid_plan.json")
    attempt = directory/"acceptance_attempts"/(uuid.uuid4().hex+".json")
    record = {"schema": "flex_independent_acceptance_attempt_v1", "status": "STARTED",
        "started_at_unix": time.time(), "scope": "CPU independent selected-source audit and20 paid-only training replays; no GPU inference"}
    atomic_json(attempt, record); cpu, wall = time.process_time(), time.perf_counter()
    try:
        reference = accept_flex_reference(directory, args.flex_gate)
        full = make_e4_full_gate(args.e4_full_gate, mandatory_e4_gate=identity["mandatory_e4_gate"]["path"],
            flex_reference_gate=args.flex_gate, current_inputs=identity["inputs"])
        record["status"] = "ACCEPTED_TECHNICAL_COMPLETENESS"
    except BaseException as error:
        record.update(status="FAILED_OR_INCOMPLETE", error_type=type(error).__name__)
        raise
    finally:
        record.update(process_time_s=time.process_time()-cpu, wall_time_s=time.perf_counter()-wall)
        atomic_json(attempt, record)
    print(json.dumps({"flex_status": reference["status"], "e4_status": full["status"],
        "native_system": "NOT_RUN", "scientific_positivity_required": False, "research_complete": False}), flush=True)
    return full


if __name__ == "__main__": main()
