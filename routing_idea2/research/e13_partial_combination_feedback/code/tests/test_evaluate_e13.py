"""Synthetic qualification of E13 evaluation axes, clustering and costs."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_e13 import (
    bootstrap_indices, bootstrap_means, holm_adjust, paired_comparison,
    select_matching_generation_rewards, service_cost_summary,
)


class E13EvaluationTests(unittest.TestCase):
    def metadata(self):
        return np.tile(np.array(["gsm8k"]*60+["bbh"]*60), 5), np.repeat(np.arange(5), 120)

    def test_generation_rewards_are_never_crossed(self):
        rewards = np.zeros((3, 7, 20), dtype=int)
        rewards[1, :, 2] = 1
        rewards[2, :, 3] = 1
        actions = np.zeros((2, 6, 3, 3, 10, 7), dtype=np.int16)
        actions[:, :, :, 1] = 2
        actions[:, :, :, 2] = 3
        result = select_matching_generation_rewards(actions, rewards)
        self.assertTrue(np.all(result[:, :, :, 0] == 0))
        self.assertTrue(np.all(result[:, :, :, 1] == 1))
        self.assertTrue(np.all(result[:, :, :, 2] == 1))

    def test_original_question_bootstrap_and_identical_pair(self):
        task, fold = self.metadata()
        indices = bootstrap_indices(task, fold, 5000, 202611150013)
        self.assertEqual(indices.shape, (5000, 600))
        values = np.repeat(np.linspace(0, 1, 600)[None], 2, axis=0)
        boot = bootstrap_means(values, indices)
        np.testing.assert_array_equal(boot[:, 0], boot[:, 1])
        reward = np.ones((3, 3, 600))
        actions = np.zeros((3, 3, 600), dtype=int)
        pair = paired_comparison(reward, reward, actions, actions, boot[:, 0]-boot[:, 1], task, fold)
        self.assertEqual(pair["independent_questions"], 600)
        self.assertEqual(pair["recorded_generation_logger_events"], 5400)
        self.assertEqual(pair["p_unadjusted"], 1)
        self.assertEqual(pair["net_events"], 0)
        np.testing.assert_array_equal(pair["ci95_pp"], [0, 0])

    def test_holm_three_predeclared_comparisons(self):
        np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])

    def test_service_cost_has_three_members_one_aggregate_per_event(self):
        costs = {}
        for stem, count in (("member", 6), ("aggregate", 20)):
            shape = (3, 600, count)
            costs[stem+"_execution_counts"] = np.ones(shape, dtype=int)
            costs[stem+"_prompt_tokens"] = np.full(shape, 10 if stem == "member" else 20, dtype=int)
            costs[stem+"_completion_tokens"] = np.full(shape, 5 if stem == "member" else 7, dtype=int)
            costs[stem+"_wall_time_s"] = np.full(shape, .1 if stem == "member" else .2)
        actions = np.zeros((3, 3, 600), dtype=int)
        total = service_cost_summary(costs, actions)["total_across_nine_replays"]
        self.assertEqual(total["tokens"], 5400*72)
        self.assertEqual(total["member_calls"], 5400*3)
        self.assertEqual(total["aggregator_calls"], 5400)
        self.assertEqual(total["gold"], 0)
        self.assertEqual(total["scoring_ops"], 0)


if __name__ == "__main__":
    unittest.main()
