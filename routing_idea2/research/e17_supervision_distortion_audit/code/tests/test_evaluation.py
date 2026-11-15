"""Fourteen reward-free synthetic checks for the E17 evaluation operators.

Run with ``python -B -m unittest discover -s tests -p test_evaluation.py -v``
from the E17 code directory. All reward arrays are generated here. Importing the
evaluation modules does not invoke their main functions or open outcome files.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import evaluate_e17 as evaluation
import finalize_e17 as finalization
from confirmation import paired_power_plan


class EvaluationQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(170315)
        cls.rewards = rng.integers(0, 2, size=(3, 600, 20)).astype(float)
        cls.actions = rng.integers(0, 20, size=600)
        cls.selected = evaluation.reward_for(cls.rewards, cls.actions)
        cls.fold_rows = rng.choice(600, 120, replace=False)
        cls.fold_scores = rng.normal(size=(3, 120, 20))
        cls.evaluated = rng.integers(0, 2, size=(101, 3, 4, 3, 600)).astype(float)
        cls.choices = rng.integers(0, 20, size=(101, 3, 4, 600))
        cls.mask = np.arange(600) % 7 == 0
        cls.scores = rng.normal(size=(600, 20))
        cls.calibrated = (cls.scores - cls.scores.mean(axis=1, keepdims=True)) * 1.3
        cls.target = cls.rewards.mean(axis=0)
        cls.metrics = evaluation.value_metrics(cls.scores, cls.rewards, cls.calibrated)

    def test_reward_gather_seed_question_axes(self):
        expected = np.array([
            [self.rewards[seed, question, self.actions[question]] for question in range(600)]
            for seed in range(3)
        ])
        np.testing.assert_array_equal(self.selected, expected)

    def test_fold_score_scatter_axes(self):
        destination = np.full((3, 600, 20), np.nan)
        destination[:, self.fold_rows] = self.fold_scores
        np.testing.assert_array_equal(destination[:, self.fold_rows], self.fold_scores)
        unfilled = np.ones(600, dtype=bool)
        unfilled[self.fold_rows] = False
        self.assertTrue(np.isnan(destination[:, unfilled]).all())

    def test_group_accuracy_axes(self):
        actual = 100 * self.evaluated[1:, 1, 2, :, :][:, :, self.mask].mean(axis=(1, 2))
        expected = np.array([
            100 * self.evaluated[permutation, 1, 2][:, self.mask].mean()
            for permutation in range(1, 101)
        ])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_group_consistency_axes(self):
        actual = (
            self.choices[1:, 1, 2][:, self.mask] == self.choices[0, 1, 2, self.mask]
        ).mean(axis=1)
        expected = np.array([
            np.mean(self.choices[permutation, 1, 2, self.mask] == self.choices[0, 1, 2, self.mask])
            for permutation in range(1, 101)
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_group_seed_axes(self):
        actual = 100 * self.evaluated[1:, 1, 2][:, :, self.mask].mean(axis=(0, 2))
        expected = np.array([
            100 * np.mean([
                self.evaluated[permutation, 1, 2, seed, self.mask].mean()
                for permutation in range(1, 101)
            ])
            for seed in range(3)
        ])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_task_balanced_300_300(self):
        expected = 0.5 * (self.selected[:, :300].mean() + self.selected[:, 300:].mean())
        self.assertLess(abs(self.selected.mean() - expected), 1e-15)

    def test_calibrated_centered_error(self):
        centered = self.target - self.target.mean(axis=1, keepdims=True)
        expected = np.mean((self.calibrated - centered) ** 2)
        self.assertLess(abs(self.metrics["centered_calibrated_MSE"] - expected), 1e-15)

    def test_calibrated_pair_gap_error(self):
        errors = []
        for first in range(20):
            for second in range(first + 1, 20):
                errors.extend(np.abs(
                    (self.calibrated[:, first] - self.calibrated[:, second])
                    - (self.target[:, first] - self.target[:, second])
                ))
        expected = np.mean(errors)
        self.assertLess(abs(self.metrics["relative_gap_calibrated_MAE"] - expected), 1e-15)

    def test_strict_pair_rank(self):
        correct = []
        for first in range(20):
            for second in range(first + 1, 20):
                nonzero = self.target[:, first] != self.target[:, second]
                correct.extend(
                    np.sign(self.scores[nonzero, first] - self.scores[nonzero, second])
                    == np.sign(self.target[nonzero, first] - self.target[nonzero, second])
                )
        self.assertEqual(self.metrics["strict_pair_count"], len(correct))
        self.assertLess(abs(self.metrics["strict_pair_rank_accuracy"] - np.mean(correct)), 1e-15)

    def test_all_tied_target_and_unavailable_calibration(self):
        metrics = evaluation.value_metrics(self.scores, np.ones((3, 600, 20)), None)
        self.assertEqual(metrics["strict_pair_count"], 0)
        self.assertIsNone(metrics["strict_pair_rank_accuracy"])
        self.assertIsNone(metrics["centered_calibrated_MSE"])
        self.assertIsNone(metrics["relative_gap_calibrated_MAE"])
        json.dumps(metrics, allow_nan=False)

    def test_exact_near_ties_and_gap(self):
        metrics = evaluation.score_ties(np.zeros((600, 20)))
        self.assertEqual(metrics["exact_score_ties"], 600)
        self.assertEqual(metrics["near_score_ties"], 600)
        self.assertEqual(metrics["top_score_gap_summary"]["maximum"], 0)

    def test_variance_ddof0(self):
        summary = evaluation.describe([1, 2, 3, 4])
        self.assertLess(abs(summary["variance"] - summary["sd"] ** 2), 1e-15)
        self.assertEqual(summary["variance"], 1.25)

    def test_paired_events(self):
        paired = evaluation.paired(self.selected, np.zeros_like(self.selected))
        self.assertEqual(paired["net_correct_events"], int(self.selected.sum()))
        self.assertEqual(paired["corrections_events"], int(self.selected.sum()))
        self.assertEqual(paired["corruptions_events"], 0)

    def test_zero_variance_power_and_optional_format(self):
        tasks = np.array(["gsm8k"] * 300 + ["bbh"] * 300)
        power = paired_power_plan(np.zeros(600), tasks)
        self.assertIsNone(power["effects"][0]["power_by_n"][0]["normal_approximation_power"])
        self.assertEqual(finalization.optional_number(None), "未定义")


if __name__ == "__main__":
    unittest.main(verbosity=2)
