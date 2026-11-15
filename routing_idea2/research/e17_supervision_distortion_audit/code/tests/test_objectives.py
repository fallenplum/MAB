"""Synthetic mathematical and numerical qualifications of the E17 operators."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objectives import (
    fit_expected_spo, fit_spo, objective_value, objective_value_and_gradient, select_actions,
)


class ObjectiveTests(unittest.TestCase):
    def test_direct_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(2026111701)
        phi = rng.normal(size=(4, 3))
        samples = rng.uniform(size=(4, 3, 2))
        coefficient = rng.normal(size=(3, 2))
        _, gradient = objective_value_and_gradient(coefficient, phi, samples, .2)
        numeric = np.zeros_like(coefficient)
        step = 1e-6
        for index in np.ndindex(coefficient.shape):
            plus, minus = coefficient.copy(), coefficient.copy()
            plus[index] += step
            minus[index] -= step
            numeric[index] = (objective_value(plus, phi, samples, .2)
                              - objective_value(minus, phi, samples, .2)) / (2 * step)
        np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=0)

    def test_nonnegative_loss_for_every_tie_rule(self):
        rng = np.random.default_rng(2026111702)
        phi = rng.normal(size=(5, 2))
        samples = rng.uniform(size=(5, 2, 20))
        coefficient = rng.normal(size=(2, 20))
        for rule in ("lexicographic", "uniform", "semantic"):
            self.assertGreaterEqual(objective_value(coefficient, phi, samples, 0, rule), 0)

    def test_one_draw_reduces_to_ordinary_spo(self):
        phi = np.array([[1., -1.], [1., 0.], [1., 1.]])
        reward = np.array([[.9, .1], [.8, .2], [.1, .9]])
        ordinary, _ = fit_spo(phi, reward, .1)
        expected, _ = fit_expected_spo(phi, reward[:, None, :], .1)
        np.testing.assert_allclose(ordinary, expected, atol=1e-10, rtol=0)

    def test_all_tied_rewards_have_zero_regularized_solution(self):
        phi = np.array([[1., -1.], [1., 0.], [1., 1.]])
        reward = np.full((3, 2), .6)
        for rule in ("lexicographic", "uniform", "semantic"):
            coefficient, diagnostic = fit_spo(phi, reward, .5, rule, [(1,), (0,)])
            self.assertEqual(objective_value(np.zeros_like(coefficient), phi, reward, .5,
                                             rule, [(1,), (0,)]), 0)
            # J*=0 and nonnegative data loss give lambda*||W||^2 <= J-J*.
            self.assertLessEqual(.5 * np.square(coefficient).sum(), diagnostic["objective"] + 1e-12)
            self.assertLess(np.linalg.norm(coefficient), 1e-4)
            self.assertLess(diagnostic["objective"], 1e-7)

    def test_uniform_and_semantic_losses_are_permutation_equivariant(self):
        rng = np.random.default_rng(2026111703)
        phi = rng.normal(size=(4, 2))
        samples = np.array([[[1., 1., 0.]], [[0., 1., .5]],
                            [[.3, .3, .3]], [[.2, .7, .7]]])
        coefficient = rng.normal(size=(2, 3))
        keys = [(0, 1, 2), (0, 1, 3), (0, 1, 4)]
        permutation = np.array([2, 0, 1])
        for rule in ("uniform", "semantic"):
            value, gradient = objective_value_and_gradient(coefficient, phi, samples, .1, rule, keys)
            new_value, new_gradient = objective_value_and_gradient(
                coefficient[:, permutation], phi, samples[:, :, permutation], .1,
                rule, [keys[i] for i in permutation],
            )
            self.assertAlmostEqual(value, new_value, places=12)
            np.testing.assert_allclose(new_gradient, gradient[:, permutation], atol=1e-12, rtol=0)

    def test_semantic_true_tie_uses_smallest_physical_key(self):
        phi = np.ones((1, 1))
        coefficient = np.array([[-1., 1., 0.]])
        reward = np.array([[1., 1., 0.]])
        keys = [(1, 2, 3), (0, 1, 2), (2, 3, 4)]
        semantic = objective_value(coefficient, phi, reward, 0, "semantic", keys)
        numeric = objective_value(coefficient, phi, reward, 0, "lexicographic", keys)
        self.assertEqual(semantic, 0)
        self.assertEqual(numeric, 4)

    def test_loss_expectation_and_mean_reward_have_different_optima(self):
        phi = np.ones((1, 1))
        samples = np.array([[[.6, .5]] * 9 + [[0., 1.]]])
        mean = samples.mean(axis=1)
        np.testing.assert_allclose(mean, [[.54, .55]], atol=1e-15)
        expected, _ = fit_expected_spo(phi, samples, .2)
        mean_fit, _ = fit_spo(phi, mean, .2)
        self.assertEqual(int(np.argmax(expected[0])), 0)
        self.assertEqual(int(np.argmax(mean_fit[0])), 1)
        self.assertLess(mean[0, np.argmax(expected[0])], mean.max())

    def test_invalid_rule_fails_closed(self):
        phi, reward = np.ones((2, 1)), np.ones((2, 2))
        for rule in ("reverselex", "typo", None):
            with self.assertRaises(ValueError):
                fit_spo(phi, reward, .1, rule)

    def test_twenty_action_reward_maximization_sign(self):
        phi = np.ones((4, 1))
        reward = np.zeros((4, 20))
        reward[:, 17] = 1
        coefficient, diagnostic = fit_spo(phi, reward, .01)
        self.assertTrue(np.all(np.argmax(phi @ coefficient, axis=1) == 17))
        self.assertEqual(diagnostic["coefficient_dimension"], 20)

    def test_epigraph_residuals_and_dimensions(self):
        rng = np.random.default_rng(2026111704)
        phi = rng.normal(size=(6, 2))
        samples = rng.uniform(size=(6, 2, 4))
        coefficient, diagnostic = fit_expected_spo(phi, samples, .1, "uniform")
        self.assertEqual(diagnostic["status"], "Solved")
        self.assertEqual(diagnostic["variables"], 2 * 4 + 6 * 2)
        self.assertEqual(diagnostic["constraints"], 6 * 2 * 4)
        self.assertAlmostEqual(diagnostic["objective"], objective_value(coefficient, phi, samples, .1, "uniform"))
        for field in ("objective_error", "primal_violation", "primal_dual_gap", "stationarity"):
            self.assertLess(diagnostic[field], 1e-7)

    def test_duplicate_draws_preserve_loss_and_optimum(self):
        rng = np.random.default_rng(2026111705)
        phi = rng.normal(size=(4, 2))
        samples = rng.uniform(size=(4, 2, 3))
        duplicated = np.repeat(samples, 2, axis=1)
        coefficient, _ = fit_expected_spo(phi, samples, .3)
        repeated_fit, _ = fit_expected_spo(phi, duplicated, .3)
        np.testing.assert_allclose(coefficient, repeated_fit, atol=1e-6, rtol=0)
        self.assertAlmostEqual(objective_value(coefficient, phi, samples, .3),
                               objective_value(coefficient, phi, duplicated, .3), places=12)

    def test_penalty_is_applied_once_after_draw_average(self):
        phi = np.array([[1., -1.], [1., 2.]])
        samples = np.array([[[.8, .2], [.4, .6]], [[.1, .9], [.7, .3]]])
        coefficient = np.array([[.2, -.1], [.1, .3]])
        per_draw = [objective_value(coefficient, phi, samples[:, b], 0) for b in range(2)]
        expected = np.mean(per_draw) + .4 * np.square(coefficient).sum()
        self.assertAlmostEqual(expected, objective_value(coefficient, phi, samples, .4), places=14)

    def test_exact_and_near_tie_service_rules(self):
        scores = np.array([[1., 1., 0.], [1., 1. + 5e-9, 0.]])
        np.testing.assert_array_equal(select_actions(scores), [0, 1])
        np.testing.assert_array_equal(select_actions(scores, [1, 0, 2], semantic=False), [1, 1])
        np.testing.assert_array_equal(select_actions(scores, [1, 0, 2], tolerance=1e-8), [0, 0])
        np.testing.assert_array_equal(select_actions(scores, [1, 0, 2], tolerance=1e-8, semantic=False), [1, 1])

    def test_service_order_and_physical_keys_are_validated(self):
        for order in ([0, 0], [0., 1.], [1]):
            with self.assertRaises(ValueError):
                select_actions([[0., 1.]], order)
        with self.assertRaises(ValueError):
            objective_value(np.zeros((1, 2)), [[1]], [[0., 1.]], .1, "semantic", [0, 0])


if __name__ == "__main__":
    unittest.main()
