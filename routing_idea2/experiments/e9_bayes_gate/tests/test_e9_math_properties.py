"""Run from the repository root with Python unittest discovery."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "math_consistency.py"
SPEC = importlib.util.spec_from_file_location("e9_math_consistency", MODULE_PATH)
MATH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATH)


class E9MathematicalProperties(unittest.TestCase):
    def test_complete_synthetic_qualifications(self):
        self.assertEqual(MATH.run_qualifications()["status"], "PASS")

    def test_set_loss_recovers_unique_winner_spo(self):
        q = np.asarray([0.4, 0.8, 0.0])
        r = np.asarray([1.0, 0.5, 0.0])
        w = MATH.near_optimal_weights(r, 0.0)
        expected = np.max(2.0 * q - r) - (2.0 * q[0] - r[0])
        self.assertAlmostEqual(float(MATH.set_spo_loss(q, r, w)), float(expected))

    def test_invalid_set_weights_rejected(self):
        with self.assertRaises(ValueError):
            MATH.set_spo_loss(np.zeros(2), np.zeros(2), np.asarray([-0.1, 1.1]))

    def test_counterexample_selects_wrong_action_for_surrogate(self):
        result = MATH.two_state_counterexample()
        self.assertEqual(result["bayes_action"], "B")
        self.assertEqual(result["surrogate_selected_action"], "A")
        self.assertTrue(np.allclose(result["expected_true_spo_loss_per_action"], [0.1, 0.09]))


if __name__ == "__main__":
    unittest.main()
