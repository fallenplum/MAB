import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_psmr import fit_psmr, predict_psmr
from train_hsmr import fit_hsmr


class RepeatabilityTests(unittest.TestCase):
    def test_fixed_arrays_repeat_exact_predictions(self):
        rng = np.random.default_rng(20261115)
        phi = np.column_stack((np.ones(30), rng.normal(size=(30, 4))))
        rewards = rng.integers(0, 2, size=(30, 3, 20)).astype(float)
        for loss in ("brier", "bce"):
            a = fit_psmr(phi, rewards, .01, loss)
            b = fit_psmr(phi, rewards, .01, loss)
            np.testing.assert_array_equal(predict_psmr(phi, a["weights"]), predict_psmr(phi, b["weights"]))
        task, sub = np.repeat("gsm8k", 30), np.repeat("", 30)
        a = fit_hsmr(phi, rewards, task, sub, 10, 3)
        b = fit_hsmr(phi, rewards, task, sub, 10, 3)
        np.testing.assert_array_equal(a["weights"], b["weights"])


if __name__ == "__main__":
    unittest.main()
