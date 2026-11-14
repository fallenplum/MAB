"""Synthetic checks of random-action blindness and actual union accounting."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory import (TEAMS, build_random_world, choose_random_second,
                        execution_packages, select_prefix, summarize_prefix)


def synthetic_train(n_questions=3):
    costs = {}
    for stem, width in (("member", 6), ("aggregate", 20)):
        shape = (3, n_questions, width)
        costs[stem + "_execution_counts"] = np.ones(shape, dtype=int)
        costs[stem + "_prompt_tokens"] = np.broadcast_to(np.arange(width) + 10, shape).copy()
        costs[stem + "_completion_tokens"] = np.ones(shape, dtype=int)
        costs[stem + "_wall_time_s"] = np.full(shape, .25)
    rewards = np.zeros((n_questions, 3, 20))
    rewards[:, :, ::2] = 1
    return {"Y": rewards, "cost": costs, "canonical": np.array([f"canonical{i}" for i in range(n_questions)])}


def example_world(seconds=(1, 7, 19), generation=0):
    train = synthetic_train(len(seconds))
    inputs = {"H": np.arange(len(seconds)), "E": np.array([len(seconds)])}
    main = [{"question_index": i, "main_action": 0} for i in range(len(seconds))]
    with patch("trajectory.choose_random_second", side_effect=list(seconds)):
        return build_random_world(train, inputs, main, 0, 17, generation,
                                  [f"question{i}" for i in range(len(seconds))])


class TrajectoryTests(unittest.TestCase):
    def test_random_second_is_deterministic_and_excludes_only_main(self):
        draws = [choose_random_second(f"q{i}", 0, 17, 0) for i in range(1000)]
        self.assertEqual(draws, [choose_random_second(f"q{i}", 0, 17, 0) for i in range(1000)])
        self.assertEqual(set(draws), set(range(1, 20)))
        self.assertTrue(any(len(set(TEAMS[0]) | set(TEAMS[a])) == 4 for a in draws))
        self.assertTrue(any(len(set(TEAMS[0]) | set(TEAMS[a])) == 6 for a in draws))

    def test_second_action_plan_is_shared_across_generation_repeats(self):
        train = synthetic_train()
        inputs = {"H": np.arange(3), "E": np.array([3])}
        mains = [{"question_index": i, "main_action": i} for i in range(3)]
        first = build_random_world(train, inputs, mains, 1, 29, 42, ["a", "b", "c"])
        second = build_random_world(train, inputs, mains, 1, 29, 137, ["a", "b", "c"])
        self.assertEqual(first["plan"], second["plan"])

    def test_all_random_actions_are_planned_before_cache_arrays_are_obtained(self):
        state = {"planned": 0}
        raw = synthetic_train()
        class GuardedTrain(dict):
            def __getitem__(self, key):
                if key in ("Y", "cost") and state["planned"] != 3:
                    raise AssertionError("Cache was accessed before the complete blind plan")
                return super().__getitem__(key)
        def choose(qid, fold, logger, main):
            state["planned"] += 1
            return 1
        with patch("trajectory.choose_random_second", side_effect=choose):
            result = build_random_world(GuardedTrain(raw), {"H": np.arange(3), "E": np.array([3])},
                [{"question_index": i, "main_action": 0} for i in range(3)], 0, 17, 0, ["a", "b", "c"])
        self.assertTrue(result["log"]["plan_frozen_before_reward_cost_reads"])

    def test_arbitrary_pairs_pay_for_exact_four_five_six_member_union(self):
        result = example_world()
        for package, expected in zip(result["packages"], (4, 5, 6)):
            self.assertEqual(package["resources"]["member_calls"], expected)
            self.assertEqual(package["resources"]["aggregator_calls"], 2)
            self.assertEqual(package["resources"]["gold"], 1)
            self.assertEqual(package["resources"]["scoring_ops"], 2)
            self.assertEqual(len(package["observations"]), 2)
        actual = result["log"]["final_summary"]["ledger"]["total"]["tokens"]
        standalone = sum(o["standalone_tokens"] for p in result["packages"] for o in p["observations"])
        self.assertLess(actual, standalone)

    def test_random_pairs_include_only_the_selected_reward_columns(self):
        result = example_world()
        reads = [entry for entry in result["log"]["access_log"] if entry["kind"] == "reward"]
        self.assertEqual(len(reads), 3)
        for entry, second in zip(reads, (1, 7, 19)):
            self.assertEqual(entry["actions"], [0, second])

    def test_neighbor_parser_rejects_the_larger_random_unions(self):
        with self.assertRaisesRegex(ValueError, "replace exactly one member"):
            execution_packages(example_world()["log"], scenario="NeighborPair")

    def test_prefix_cap_never_splits_a_pair_or_skips_a_question(self):
        packages = example_world()["packages"]
        cap = packages[0]["resources"]["tokens"] + packages[1]["resources"]["tokens"] - 1
        chosen = select_prefix(packages, cap)
        self.assertEqual(chosen["n_questions"], 1)
        self.assertEqual(len(chosen["observations"]), 2)
        self.assertGreater(chosen["next_package_excess_over_remaining"], 0)
        self.assertEqual(chosen["full_plan_questions"], 3)
        self.assertGreater(chosen["full_plan_ledger"]["total"]["tokens"], chosen["actual_tokens"])
        self.assertFalse(chosen["online_hard_budget_guarantee"])

    def test_zero_or_insufficient_cap_has_an_empty_legal_prefix(self):
        packages = example_world()["packages"]
        for cap in (0, packages[0]["resources"]["tokens"] - 1):
            chosen = select_prefix(packages, cap)
            self.assertEqual(chosen["n_questions"], 0)
            self.assertEqual(chosen["observations"], [])
            self.assertEqual(chosen["ledger"]["total"]["gold"], 0)
            self.assertEqual(chosen["remaining_tokens"], cap)

    def test_fixed_question_prefix_exposes_both_prefix_and_pool_costs(self):
        packages = example_world()["packages"]
        result = summarize_prefix(packages, 2)
        self.assertEqual(result["ledger"]["total"]["gold"], 2)
        self.assertEqual(result["full_plan_ledger"]["total"]["gold"], 3)
        self.assertEqual(result["selection"], "fixed_original_question_prefix")

    def test_cumulative_ledger_tampering_is_rejected(self):
        log = deepcopy(example_world()["log"])
        log["execution_log"][1]["cumulative"]["gold"] += 1
        with self.assertRaisesRegex(ValueError, "cumulative"):
            execution_packages(log, scenario="RandomPair")

    def test_final_ledger_tampering_is_rejected(self):
        log = deepcopy(example_world()["log"])
        log["final_summary"]["ledger"]["total"]["gold"] += 1
        with self.assertRaisesRegex(ValueError, "Final total"):
            execution_packages(log, scenario="RandomPair")

    def test_reused_member_is_not_accepted_as_an_additional_execution(self):
        log = deepcopy(example_world()["log"])
        log["execution_log"][1]["member_keys"] = [[0, 0, 0]]
        with self.assertRaisesRegex(ValueError, "newly executed union"):
            execution_packages(log, scenario="RandomPair")

    def test_unfinished_packages_are_rejected(self):
        log = deepcopy(example_world()["log"])
        log["execution_log"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            execution_packages(log)


if __name__ == "__main__":
    unittest.main()
