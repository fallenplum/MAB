"""Run the synthetic common-shift versus decision-boundary acquisition audit."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from budget_router_v2.acquisition import antithetic_fantasies, decision_value

PRIVATE = ROOT / "artifacts/private/routing_redesign_20261104"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PRIVATE / "query_operator_audit")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if not output.is_relative_to(PRIVATE.resolve()):
        raise ValueError("Audit output must remain under the authorized private directory")
    fantasies = antithetic_fantasies(1, pairs=128, seed=20261104)
    common = dict(mean=np.zeros(2), covariance=np.eye(2),
                  reference_features=np.array([[[1.0, 1.0], [1.0, -1.0]]]),
                  noise_covariance=np.eye(1), fantasies=fantasies)
    decoy = decision_value(observations=np.array([[10.0, 0.0]]), **common)
    boundary = decision_value(observations=np.array([[0.0, 1.0]]), **common)
    checks = {"decoy_has_higher_parameter_ig": decoy["ig_theta"] > boundary["ig_theta"],
              "common_shift_kg_is_zero": abs(decoy["decision_kg"]) <= 1e-12,
              "decision_boundary_kg_is_positive": boundary["decision_kg"] > 0}
    if not all(checks.values()):
        raise AssertionError("Synthetic objective-separation audit failed")
    bundle = decision_value(np.zeros(2), np.eye(2),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([[[1.0, value] for value in np.linspace(-1, 1, 20)]]),
        np.eye(3), antithetic_fantasies(3, pairs=32, seed=20261104))
    source_paths = [ROOT / "src/budget_router_v2/acquisition.py", Path(__file__).resolve(),
                    ROOT / "tests/v2/test_decision_value.py"]
    audit = {"schema": "v2_query_operator_software_audit_v1", "status": "SOFTWARE_AUDIT_PASSED",
             "method_family": "KG-informed existing decision-theoretic approach",
             "scientific_efficacy": "UNMEASURED", "real_model_calls": 0,
             "reference_scope": "Synthetic feature arrays only", "checks": checks,
             "common_shift_query": decoy, "decision_boundary_query": boundary,
             "general_bundle_twenty_action_shape": bundle,
             "cost_scope": "Operator returns measured local planning CPU/wall. Resource prices and budget admission belong to the independent feedback ledger.",
             "sources": [{"relative_path": path.relative_to(ROOT).as_posix(),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_paths]}
    output.mkdir(parents=True, exist_ok=True)
    path = output / ("audit_" + uuid.uuid4().hex + ".json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"status": audit["status"], "checks": checks,
                      "audit_relative_path": path.relative_to(ROOT).as_posix()}))
    return audit


if __name__ == "__main__":
    main()
