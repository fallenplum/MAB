"""Regression for native CPS variance with 184 active of 200 total units."""
import json
import os
import sys
from pathlib import Path

HERE = Path(os.environ.get("B1_METHODS_WORK", str(Path(__file__).resolve().parent)))
os.environ["NUMBA_NUM_THREADS"] = "1"
sys.path[:0] = [str(HERE / "dependencies"), str(HERE / "upstream/cps-sampling/src")]
import numpy as np
from finite_pool_supplements import fixed_size_asi_supplement

N, M = 200, 184
active = np.arange(N) < M
d = np.zeros(N)
d[:M] = np.random.default_rng(203).choice([-1/3, 0., 1/3], M)
m = np.zeros(N)
cost = np.full(N, 1000.)
p = np.zeros(N)
p[:M] = .25
result = fixed_size_asi_supplement(d, m, cost, active, p, 46., 46000., repeats=20,
                                  source_root=HERE / "upstream/cps-sampling", dependency_root=HERE / "dependencies")
pi, Pi = result["actual_pi"], result["joint_pi"]
direct = float((d[:M]/pi) @ (Pi-np.outer(pi,pi)) @ (d[:M]/pi) / N**2)
reported = result["diagnostics"]["exact_design_variance_evaluator_only"]
assert abs(reported-direct) < 1e-12
assert result["samples"].shape == (20, 46)
record = {"status": "PASS", "N_total": N, "N_active": M, "sample_count": 46,
          "exact_design_variance": direct, "metadata_variance": reported,
          "absolute_error": abs(reported-direct), "reward_source": "synthetic software regression only"}
(HERE / "native_results/ACTIVE_DENOMINATOR_REGRESSION.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
