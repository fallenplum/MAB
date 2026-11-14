"""Small synthetic checks only. No project rewards are read by these tests."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_hsmr import (
    fit_hsmr,
    hierarchical_loo_prior,
    predict_historical_prior,
    shrink_targets,
)
from train_psmr import fit_psmr, objective_gradient, predict_psmr, reward_mean


class TestMeanModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026111401)
        self.design = np.column_stack([np.ones(12), rng.normal(size=(12, 3))])
        self.rewards = rng.integers(0, 2, size=(12, 3, 5)).astype(float)
        self.task = np.array(["gsm8k"] * 4 + ["bbh"] * 8)
        self.subtask = np.array([""] * 4 + ["logic"] * 4 + ["math"] * 4)

    def test_psmr_gradients(self):
        target, _ = reward_mean(self.rewards)
        point = np.linspace(-0.2, 0.2, self.design.shape[1] * target.shape[1])
        for loss in ("brier", "bce"):
            objective = lambda value: objective_gradient(value, self.design, target, 0.01, loss)[0]
            gradient = lambda value: objective_gradient(value, self.design, target, 0.01, loss)[1]
            self.assertLess(check_grad(objective, gradient, point), 1e-6)

    def test_extreme_logits_are_finite(self):
        target, _ = reward_mean(self.rewards)
        for sign in (-1, 1):
            point = np.full(self.design.shape[1] * target.shape[1], sign * 1000.0)
            for loss in ("brier", "bce"):
                objective, gradient = objective_gradient(point, self.design, target, 0.01, loss)
                self.assertTrue(np.isfinite(objective))
                self.assertTrue(np.isfinite(gradient).all())

    def test_repeat_mean_brier_identity(self):
        target, constant = reward_mean(self.rewards)
        probability = np.full_like(target, 0.37)
        repeated = np.mean((probability[:, None, :] - self.rewards) ** 2)
        self.assertAlmostEqual(repeated, np.mean((probability - target) ** 2) + constant, places=14)

    def test_equal_action_labels_have_no_index_preference(self):
        for value in (0.0, 0.4, 1.0):
            rewards = np.full((len(self.design), 3, 5), value)
            for loss in ("brier", "bce"):
                fitted = fit_psmr(self.design, rewards, 0.01, loss)
                self.assertTrue(fitted["diagnostics"]["optimizer_success"])
                predictions = predict_psmr(self.design, fitted["weights"])
                np.testing.assert_allclose(
                    predictions, np.repeat(predictions[:, :1], predictions.shape[1], axis=1), atol=1e-12
                )

    def test_loo_excludes_entire_current_question(self):
        before = hierarchical_loo_prior(self.rewards, self.task, self.subtask)
        for index in (0, 4, 9):
            changed = self.rewards.copy()
            changed[index] = 1.0 - changed[index]
            after = hierarchical_loo_prior(changed, self.task, self.subtask)
            np.testing.assert_allclose(before[index], after[index], atol=1e-14)

    def test_single_question_uses_weak_prior(self):
        prior = hierarchical_loo_prior(self.rewards[:1], self.task[:1], self.subtask[:1])
        np.testing.assert_allclose(prior, 0.5)

    def test_zero_shrinkage_recovers_mean_ridge(self):
        target, _ = reward_mean(self.rewards)
        np.testing.assert_array_equal(shrink_targets(self.rewards, self.task, self.subtask, 0), target)
        fitted = fit_hsmr(self.design, self.rewards, self.task, self.subtask, 2.0, 0.0)
        expected = np.linalg.solve(self.design.T @ self.design + 2 * np.eye(4), self.design.T @ target)
        np.testing.assert_array_equal(fitted["weights"], expected)
        self.assertLess(fitted["diagnostics"]["gradient_max_abs"], 1e-12)

    def test_historical_prior_shape_and_fallback(self):
        prediction = predict_historical_prior(
            self.rewards, self.task, self.subtask, ["gsm8k", "bbh", "unknown"], ["", "missing", ""]
        )
        self.assertEqual(prediction.shape, (3, 5))
        np.testing.assert_allclose(prediction[2], self.rewards.mean(axis=(0, 1)))
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())

    def test_action_relabeling_equivariance(self):
        permutation = np.array([3, 0, 4, 1, 2])
        inverse = np.argsort(permutation)
        for loss in ("brier", "bce"):
            original = fit_psmr(self.design, self.rewards, 0.01, loss)
            permuted = fit_psmr(self.design, self.rewards[:, :, permutation], 0.01, loss)
            np.testing.assert_allclose(original["weights"], permuted["weights"][:, inverse], atol=1e-9)
        original = fit_hsmr(self.design, self.rewards, self.task, self.subtask, 2.0, 3.0)
        permuted = fit_hsmr(self.design, self.rewards[:, :, permutation], self.task, self.subtask, 2.0, 3.0)
        np.testing.assert_allclose(original["weights"], permuted["weights"][:, inverse], atol=1e-12)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            fit_psmr(self.design, self.rewards[:5], 0.01)
        with self.assertRaises(ValueError):
            fit_psmr(self.design, self.rewards[:, :2], 0.01)
        with self.assertRaises(ValueError):
            fit_hsmr(self.design, self.rewards, self.task, self.subtask, 2.0, -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
