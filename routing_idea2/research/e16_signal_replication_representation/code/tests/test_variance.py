"""Known-answer numerical controls for repeated-generation variance moments."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from variance_decomposition import decompose_rewards


class VarianceDecompositionTests(unittest.TestCase):
    def test_pure_conditional_noise_has_near_zero_signal_in_fixed_large_control(self):
        rng = np.random.default_rng(20261115061)
        y = rng.normal(size=(3, 12000, 2))
        result = decompose_rewards(y)
        signal = result["representations"]["raw_actions"]["global"]["estimated_signal_variance"]
        self.assertLess(float(np.max(np.abs(signal))), .03)
        self.assertFalse(result["estimated_signal_is_truncated"])

    def test_fixed_input_dependent_means_have_exact_nonzero_signal_without_noise(self):
        x = np.linspace(-1, 1, 60)
        means = np.column_stack([x, -x])
        y = np.repeat(means[None], 3, axis=0)
        result = decompose_rewards(y)
        raw = result["representations"]["raw_actions"]["global"]
        pair = result["representations"]["pairs"]["global"]
        np.testing.assert_allclose(raw["within_noise_variance_w"], [0, 0], atol=1e-28)
        np.testing.assert_allclose(raw["estimated_signal_variance"], means.var(axis=0, ddof=1))
        np.testing.assert_allclose(pair["estimated_signal_variance"], [4*x.var(ddof=1)])

    def test_common_question_and_repeat_offsets_are_removed_by_row_centering(self):
        question = np.linspace(0, 2, 30)[None, :, None]
        repeat = np.array([-1., 0., 1.])[:, None, None]
        action = np.array([-.2, .3, .8])[None, None, :]
        y = question + repeat + action
        result = decompose_rewards(y)
        raw = result["representations"]["raw_actions"]["global"]
        centered = result["representations"]["centered_actions"]["global"]
        self.assertGreater(raw["observed_between_question_variance"][0], .1)
        self.assertGreater(raw["within_noise_variance_w"][0], .1)
        np.testing.assert_allclose(centered["observed_between_question_variance"], 0, atol=1e-28)
        np.testing.assert_allclose(centered["within_noise_variance_w"], 0, atol=1e-28)

    def test_correlated_action_noise_uses_paired_covariance(self):
        common = np.array([-1., 0., 1.])[:, None]
        y = np.stack([np.broadcast_to(common, (3, 12)), np.broadcast_to(2*common, (3, 12))], axis=2)
        result = decompose_rewards(y)
        raw = result["representations"]["raw_actions"]["global"]
        pair = result["representations"]["pairs"]["global"]
        self.assertAlmostEqual(raw["within_noise_variance_w"][0], 1.)
        self.assertAlmostEqual(raw["within_noise_variance_w"][1], 4.)
        self.assertAlmostEqual(pair["within_noise_variance_w"][0], 1.)
        self.assertNotAlmostEqual(pair["within_noise_variance_w"][0], sum(raw["within_noise_variance_w"]))

    def test_negative_signal_estimates_are_preserved(self):
        y = np.broadcast_to(np.array([-1., 0., 1.])[:, None, None], (3, 10, 2)).copy()
        result = decompose_rewards(y)
        raw = result["representations"]["raw_actions"]["global"]
        np.testing.assert_allclose(raw["estimated_signal_variance"], [-1/3, -1/3])
        self.assertEqual(raw["negative_signal_estimate_count"], 2)
        np.testing.assert_allclose(result["representations"]["pairs"]["global"]["within_noise_variance_w"], 0)

    def test_pooled_group_variation_uses_degrees_of_freedom_weights(self):
        means = np.array([0., 2., 4., 6., 8.])
        w = np.array([2., 2., 8., 8., 8.])
        repeat = np.array([-1., 0., 1.])[:, None] * np.sqrt(w)[None]
        one_action = means[None] + repeat
        y = np.stack([one_action, np.zeros_like(one_action)], axis=2)
        result = decompose_rewards(y, task=["bbh"]*5, subtask=["a", "a", "b", "b", "b"])
        pooled = result["representations"]["raw_actions"]["groupings"]["bbh_subtask"]["pooled_within_groups"]
        self.assertEqual(pooled["pooled_degrees_of_freedom"], 3)
        self.assertAlmostEqual(pooled["observed_between_question_variance"][0], 10/3)
        self.assertAlmostEqual(pooled["df_weighted_within_noise_variance_w"][0], 6)
        self.assertAlmostEqual(pooled["noise_correction_w_over_L"][0], 2)
        self.assertAlmostEqual(pooled["estimated_signal_variance"][0], 4/3)
        self.assertNotAlmostEqual(pooled["df_weighted_within_noise_variance_w"][0], float(w.mean()))

    def test_singleton_groups_have_zero_pooled_df_and_undefined_variances(self):
        y = np.arange(18, dtype=float).reshape(3, 3, 2)
        result = decompose_rewards(y, subtask=["a", "b", "c"])
        grouped = result["representations"]["raw_actions"]["groupings"]["subtask"]
        pooled = grouped["pooled_within_groups"]
        self.assertEqual(pooled["pooled_degrees_of_freedom"], 0)
        self.assertIsNone(pooled["observed_between_question_variance"])
        self.assertIsNone(pooled["estimated_signal_variance"])
        self.assertTrue(all(row["observed_between_question_variance"] is None for row in grouped["groups"]))
        json.dumps(result, allow_nan=False)

    def test_mixed_singleton_group_contributes_no_false_df_or_noise_weight(self):
        means = np.array([0., 2., 100.])
        w = np.array([3., 3., 1000.])
        one = means[None] + np.array([-1., 0., 1.])[:, None]*np.sqrt(w)
        y = np.stack([one, np.zeros_like(one)], axis=2)
        result = decompose_rewards(y, task=["a", "a", "b"])
        pooled = result["representations"]["raw_actions"]["groupings"]["task"]["pooled_within_groups"]
        self.assertEqual(pooled["pooled_degrees_of_freedom"], 1)
        self.assertEqual(pooled["questions_in_positive_df_groups"], 2)
        self.assertAlmostEqual(pooled["df_weighted_within_noise_variance_w"][0], 3)
        self.assertAlmostEqual(pooled["estimated_signal_variance"][0], 1)

    def test_all190_pairs_are_in_semantic_column_order_and_center_invariant(self):
        rng = np.random.default_rng(29)
        y = rng.normal(size=(3, 15, 20))
        result = decompose_rewards(y)
        self.assertEqual(result["unordered_pair_count"], 190)
        pairs = result["representations"]["pairs"]
        self.assertEqual(pairs["indices"][0], [0, 1])
        self.assertEqual(pairs["indices"][-1], [18, 19])
        centered = y-y.mean(axis=2, keepdims=True)
        i, j = np.triu_indices(20, 1)
        centered_pair_w = (centered[:, :, i]-centered[:, :, j]).var(axis=0, ddof=1).mean(axis=0)
        np.testing.assert_allclose(pairs["global"]["within_noise_variance_w"], centered_pair_w, atol=1e-14)

    def test_invalid_shapes_missing_values_or_metadata_are_rejected(self):
        for bad in (np.zeros((3, 5)), np.zeros((1, 5, 2)), np.full((3, 5, 2), np.nan)):
            with self.assertRaises(ValueError):
                decompose_rewards(bad)
        with self.assertRaisesRegex(ValueError, "task"):
            decompose_rewards(np.zeros((3, 5, 2)), task=["a"])


if __name__ == "__main__":
    unittest.main()
