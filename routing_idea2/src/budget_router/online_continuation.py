"""Run the already gated real online assets; never claims online-policy validation.

This module only orchestrates local generation after an exact E1 PASS.  It
reuses frozen features through the matrix preparer and does not refit them.
"""
from pathlib import Path
import hashlib
import json


STAGES = {"online_calibration": 200, "online_stream": 600}


def _json(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Online asset path escapes the authorized project")
    return path


def verify_matrix_receipt(root, output_dir):
    """Require all four exact artifacts, not merely a successful child exit."""
    root = Path(root).resolve()
    output_dir = _inside(root, output_dir)
    receipt_path = output_dir / "matrix_preparation_receipt.json"
    receipt = _json(receipt_path)
    if (receipt.get("status") != "REAL_MATRICES_READY"
            or receipt.get("e2_executed") is not False
            or receipt.get("research_complete") is not False):
        raise ValueError("Real online matrices are incomplete or misclassified as E2 validation")
    stages = receipt.get("stages", {})
    if set(stages) != set(STAGES):
        raise ValueError("Both and only the authorized online stages are required")
    for stage, count in STAGES.items():
        item = stages[stage]
        if (item.get("status") != "READY" or item.get("questions") != count
                or item.get("loader_verified") is not True):
            raise ValueError("Online matrix stage failed its exact count/loader gate")
        for kind, suffix in (("matrix", ".npz"), ("manifest", ".manifest.json")):
            expected = output_dir / (stage + suffix)
            entry = item.get(kind, {})
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("Online matrix receipt is missing a concrete artifact path")
            supplied = Path(raw_path)
            candidates = ([supplied.resolve()] if supplied.is_absolute()
                          else [(root / supplied).resolve(), (output_dir / supplied).resolve()])
            if expected.resolve() not in candidates:
                raise ValueError("Online matrix receipt points at a different artifact")
            actual = _inside(root, expected)
            if not actual.is_file() or entry.get("sha256") != _sha(actual):
                raise ValueError("Online matrix artifact is missing or its SHA256 differs")
    return receipt_path, receipt


def continue_online_assets(root, *, run_script, gpu_admission, update_state):
    root = Path(root).resolve()
    if _json(root / "results/e1/e1_diagnostics.json").get("status") != "PASS":
        raise ValueError("Real online generation requires the recorded exact E1 PASS")
    if _json(root / "protocol/E0_LOCK.json").get("status") != "PASS":
        raise ValueError("Real online generation requires a real E0 lock")
    pool = _json(root / "configs/pool_v1.json").get("models", [])
    aliases = [entry.get("alias", "").lower() for entry in pool]
    if len(pool) != 6 or len(set(aliases)) != 6 or any(not alias for alias in aliases):
        raise ValueError("The frozen online pool must contain six distinct aliases")
    for entry in pool:
        config = _inside(root, root / entry["config"])
        if not config.is_file():
            raise ValueError("A frozen member configuration is missing")
    for stage, count in STAGES.items():
        update_state("PREPARING_REAL_ONLINE_MEMBERS", E2="REAL_ASSETS_IN_PROGRESS",
                     online_stage=stage, online_questions=count)
        run_script("prepare_jobs.py", "--kind", "members", "--stage", stage)
        for entry, alias in zip(pool, aliases):
            gpu_admission()
            update_state("GENERATING_REAL_ONLINE_MEMBERS", E2="REAL_ASSETS_IN_PROGRESS",
                         online_stage=stage, active_model=entry["alias"])
            run_script("run_members.py", "--jobs", root / "data/jobs" / stage / f"members_{alias}.jsonl",
                       "--model-config", root / entry["config"],
                       "--output", root / "results/raw" / stage / f"members_{alias}.jsonl",
                       "--qualification-receipt", root / "artifacts/qualification" / f"member_{alias}.json")
        run_script("prepare_jobs.py", "--kind", "aggregations", "--stage", stage)
        gpu_admission()
        update_state("GENERATING_REAL_ONLINE_TEAM_AGGREGATIONS", E2="REAL_ASSETS_IN_PROGRESS",
                     online_stage=stage, online_teams=20)
        run_script("run_aggregation.py", "--jobs", root / "data/jobs" / stage / "aggregations.jsonl",
                   "--model-config", root / "configs/aggregator_v1.json",
                   "--output", root / "results/raw" / stage / "aggregations.jsonl",
                   "--qualification-receipt", root / "artifacts/qualification/aggregator.json")
    output_dir = root / "data/derived/online_matrices_v1"
    update_state("VERIFYING_REAL_ONLINE_MATRICES", E2="REAL_ASSETS_IN_PROGRESS")
    run_script("prepare_online_matrices.py", "--root", root, "--stage", "both",
               "--output-dir", output_dir, allowed_returncodes=(0, 2))
    receipt_path, _ = verify_matrix_receipt(root, output_dir)
    update_state("REAL_ONLINE_MATRICES_READY", E2="REAL_MATRICES_READY_NOT_E2_VALIDATION",
                 overall="PARTIAL", scientific_completion=False, e2_executed=False,
                 matrix_preparation_receipt=receipt_path.relative_to(root).as_posix(),
                 matrix_preparation_receipt_sha256=_sha(receipt_path))
