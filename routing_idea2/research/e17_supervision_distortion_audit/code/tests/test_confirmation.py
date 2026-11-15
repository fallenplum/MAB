"""Synthetic qualification of question-level planning and input-only budgets."""
from itertools import combinations
from math import ceil, sqrt
from pathlib import Path
from statistics import NormalDist
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation import input_call_union, paired_power_plan, subtract_verified_cache_keys


TEAMS = np.array(list(combinations(range(6), 3)), dtype=int)


class ConfirmationTests(unittest.TestCase):
    def test_balanced_within_task_variance_does_not_include_task_mean_gap(self):
        differences = np.array([-.2, 0, .2, .4, .6, .8, 1.])
        labels = ["GSM8K"] * 3 + ["BBH"] * 4
        result = paired_power_plan(differences, labels)
        expected = .5 * np.var(differences[:3], ddof=1) + .5 * np.var(differences[3:], ddof=1)
        self.assertAlmostEqual(result["pooled_within_task_variance"], expected)
        self.assertAlmostEqual(result["sigma_question_mean"], sqrt(expected))
        self.assertAlmostEqual(result["observed_task_balanced_difference"], .35)
        self.assertEqual(result["independent_questions"], 7)
        self.assertFalse(result["confidence_intervals_computed"])
        self.assertEqual(result["bootstrap_repetitions"], 0)

    def test_power_and_required_questions_use_percentage_points(self):
        result = paired_power_plan([-.2, .2, -.2, .2], ["gsm8k", "gsm8k", "bbh", "bbh"], (1,), (400,))
        row = result["effects"][0]
        sigma = sqrt(.08)
        normal = NormalDist()
        self.assertAlmostEqual(row["effect_fraction"], .01)
        self.assertAlmostEqual(row["power_by_n"][0]["normal_approximation_power"], normal.cdf(20*.01/sigma-normal.inv_cdf(.95)))
        self.assertEqual(row["required_n"], ceil(((normal.inv_cdf(.95)+normal.inv_cdf(.8))*sigma/.01)**2))

    def test_zero_variance_has_no_zero_question_confirmation_claim(self):
        result = paired_power_plan([.1, .1, -.1, -.1], ["gsm8k", "gsm8k", "bbh", "bbh"])
        self.assertTrue(result["zero_development_variance"])
        for row in result["effects"]:
            self.assertIsNone(row["required_n"])
            self.assertIsNone(row["power_by_n"][0]["normal_approximation_power"])
        self.assertEqual(result["positive_effect_zero_variance_limit"], 1.)

    def test_rejects_generation_expansion_and_insufficient_task_questions(self):
        with self.assertRaises(ValueError):
            paired_power_plan(np.zeros((3, 4)), ["gsm8k", "gsm8k", "bbh", "bbh"])
        with self.assertRaises(ValueError):
            paired_power_plan([0, 0, 0], ["gsm8k", "bbh", "bbh"])
        with self.assertRaises(ValueError):
            paired_power_plan([0, 0, 0, np.nan], ["gsm8k", "gsm8k", "bbh", "bbh"])

    def test_identical_policy_calls_are_shared_and_gold_is_question_level(self):
        union = input_call_union([[0, 1], [0, 1]], TEAMS, [42, 137, 2024])
        self.assertEqual(union["counts"], {"member_calls": 18, "aggregate_calls": 6, "scoring_ops": 6, "gold_questions": 2})
        self.assertEqual(union["gold_question_indices"], [0, 1])
        self.assertFalse(union["authorizes_evaluation"])

    def test_member_union_uses_shared_constituents(self):
        union = input_call_union([[0], [1]], TEAMS, [42, 137, 2024])
        self.assertEqual(union["counts"]["member_calls"], 12)
        self.assertEqual(union["counts"]["aggregate_calls"], 6)
        self.assertEqual(union["counts"]["gold_questions"], 1)

    def test_action_renaming_preserves_semantic_execution_union(self):
        perm = np.arange(20)[::-1]
        inverse = np.argsort(perm)
        original = input_call_union([[0, 1], [19, 1]], TEAMS, [42])
        renamed = input_call_union(inverse[np.array([[0, 1], [19, 1]])], TEAMS[perm], [42])
        self.assertEqual(original["member_keys"], renamed["member_keys"])
        original_semantic = {(q, g, tuple(sorted(TEAMS[a]))) for q, g, a in original["aggregate_keys"]}
        renamed_semantic = {(q, g, tuple(sorted(TEAMS[perm][a]))) for q, g, a in renamed["aggregate_keys"]}
        self.assertEqual(original_semantic, renamed_semantic)

    def test_invalid_actions_duplicate_teams_or_duplicate_seeds_fail_closed(self):
        with self.assertRaises(ValueError):
            input_call_union([[20]], TEAMS, [42])
        with self.assertRaises(ValueError):
            input_call_union([[0]], np.zeros((20, 3), dtype=int), [42])
        with self.assertRaises(ValueError):
            input_call_union([[0]], TEAMS, [42, 42])
        with self.assertRaises(ValueError):
            input_call_union([[0.5]], TEAMS, [42])

    def test_existing_aggregate_avoids_regeneration_without_claiming_member_cache(self):
        union = input_call_union([[0], [19]], TEAMS, [42])
        result = subtract_verified_cache_keys(union, [], [(0, 42, 0)])
        self.assertEqual(len(result["unverified_original_union_member_keys"]), 6)
        self.assertEqual(result["new_member_keys"], [(0, 42, 3), (0, 42, 4), (0, 42, 5)])
        self.assertEqual(result["new_aggregate_keys"], [(0, 42, 19)])
        self.assertEqual(result["counts"]["gold_questions_to_authorize_separately"], 1)
        self.assertEqual(result["counts"]["scoring_ops_to_budget_separately"], 2)

    def test_cache_subtraction_only_uses_verified_intersections(self):
        union = input_call_union([[0], [1]], TEAMS, [42])
        result = subtract_verified_cache_keys(union, [(0, 42, 0), (50, 42, 0)], [(50, 42, 0)])
        self.assertEqual(result["counts"]["new_member_calls"], 3)
        self.assertEqual(result["counts"]["new_aggregate_calls"], 2)
        self.assertEqual(result["ignored_foreign_member_cache_keys"], 1)
        self.assertEqual(result["ignored_foreign_aggregate_cache_keys"], 1)
        self.assertFalse(result["authorizes_evaluation"])

    def test_complete_output_cache_means_zero_new_calls_but_gold_remains_separate(self):
        union = input_call_union([[0]], TEAMS, [42])
        result = subtract_verified_cache_keys(union, [], union["aggregate_keys"])
        self.assertTrue(result["zero_new_model_executions"])
        self.assertEqual(result["counts"]["new_member_calls"], 0)
        self.assertEqual(result["counts"]["new_aggregate_calls"], 0)
        self.assertEqual(result["counts"]["gold_questions_to_authorize_separately"], 1)


if __name__ == "__main__":
    unittest.main()
