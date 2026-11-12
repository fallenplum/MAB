"""Auditable software controls for pinned public baseline adapters."""
import itertools
import json
import os
import sys
from pathlib import Path

for name in ["NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
HERE = Path(os.environ.get("B1_METHODS_WORK", str(Path(__file__).resolve().parent)))
sys.path[:0] = [str(HERE / "dependencies"), str(HERE / "upstream/cps-sampling/src")]
import numpy as np
import pytest
from public_methods import active_testing_lure


def qualify_lure():
    N, n = 5, 3
    losses = np.array([0., .2, .5, .8, 1.])
    shape = np.array([.15, .4, 1., .5, .2])
    records = []
    for order in itertools.permutations(range(N), n):
        probability = 1.
        remaining = list(range(N))
        qs = []
        for index in order:
            q = shape[remaining]/shape[remaining].sum()
            q = np.maximum(.5/len(remaining), q)
            q /= q.sum()
            position = remaining.index(index)
            qs.append(q[position])
            probability *= q[position]
            remaining.remove(index)
        value = active_testing_lure(losses[list(order)], qs, N)
        t = np.arange(1, n+1)
        weight = N*(N-n)/(n*(N-t)*(N-t+1))
        prior = np.concatenate([[0.], np.cumsum(losses[list(order)])[:-1]])
        pure_steps = (prior+losses[list(order)]/qs)/N
        if not np.allclose(value, weight @ pure_steps, atol=1e-12):
            raise AssertionError("LURE weighted martingale decomposition failed")
        records.append((probability, value))
    total = sum(p for p, v in records)
    expectation = sum(p*v for p,v in records)
    assert abs(total-1) < 1e-12
    assert abs(expectation-losses.mean()) < 1e-12
    return {"status": "PASS", "N": N, "n": n, "ordered_samples_enumerated": len(records),
            "probability_sum": total, "expected_estimate": expectation,
            "truth": float(losses.mean()), "absolute_bias": abs(expectation-losses.mean()),
            "independent_theory_review": "sibling b1_inference_synthetic verified weighted-PURE algebra and martingale bound"}


if __name__ == "__main__":
    result = {"LURE": qualify_lure()}
    code = pytest.main([str(HERE / "upstream/cps-sampling/tests"), "-q", "-p", "no:cacheprovider"])
    result["official_CPS_pytest"] = {"exit_code": int(code), "status": "PASS" if code == 0 else "FAIL"}
    (HERE / "native_results/PUBLIC_METHODS_QUALIFICATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
