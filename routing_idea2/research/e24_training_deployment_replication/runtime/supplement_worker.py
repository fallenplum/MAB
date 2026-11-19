"""Reuse the E23 persistent worker with paced E24 dependency polling."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import time


CODE = Path(__file__).resolve().parent
E23_SHA = "6ddbb8c4e795554630c7c5ce8760296344bbd38343954c529adc1e36b3f118e2"


def load_reference_worker():
    local = CODE / "e23_supplement_worker_original.py"
    original = local if local.is_file() else CODE.parents[1] / "e23_deployment_prior_and_references/supplement_worker.py"
    if hashlib.sha256(original.read_bytes()).hexdigest() != E23_SHA:
        raise ValueError("The original E23 persistent worker differs")
    name = "_e24_isolated_e23_worker"
    spec = importlib.util.spec_from_file_location(name, original)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # The E23 module locates the same four verified P2 sources in the deployed directory.
    return module


reference = load_reference_worker()


class PacedDependencyQueue(reference.DependencyQueue):
    """Amortize missing-receipt checks without changing dependency or readiness semantics."""

    next_refresh = 0.0

    def refresh(self):
        now = time.monotonic()
        if self.waiting_members and now >= self.next_refresh:
            super().refresh()
            self.next_refresh = now + 10


reference.DependencyQueue = PacedDependencyQueue


if __name__ == "__main__":
    reference.main()
