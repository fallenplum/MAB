from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feedback_environment import PartialFeedbackEnv, make_logger_plan


def synthetic_train():
    arrays = {}
    for stem, width in (("member", 6), ("aggregate", 20)):
        for suffix, scale in (("execution_counts", 1), ("prompt_tokens", 11),
                              ("completion_tokens", 7), ("wall_time_s", .1)):
            arrays[f"{stem}_{suffix}"] = np.full((3, 4, width), scale, dtype=float)
    return {"Y": np.tile(np.arange(20) % 2, (4, 3, 1)).astype(float),
            "canonical": np.array(["q0", "q1", "q2", "q3"]), "cost": arrays}


class PartialFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.train = synthetic_train()

    def environment(self, train=None, **kwargs):
        return PartialFeedbackEnv(self.train if train is None else train, [0, 1, 2], 0,
                                  forbidden_questions=[3], **kwargs)

    def test_main_and_neighbor_incremental_counts(self):
        env = self.environment()
        service = env.serve(0, 0)  # (0,1,2)
        first = env.buy(0, [0])
        second = env.buy(0, [0, 1])  # (0,1,3)
        self.assertEqual(service["member_calls"], 3)
        self.assertEqual(service["aggregator_calls"], 1)
        self.assertEqual(service["gold"], 0)
        self.assertEqual(service["scoring_ops"], 0)
        self.assertEqual(first["resources"]["gold"], 1)
        self.assertEqual(first["resources"]["member_calls"], 0)
        self.assertEqual(first["resources"]["aggregator_calls"], 0)
        self.assertEqual(first["resources"]["scoring_ops"], 1)
        self.assertEqual(second["resources"]["member_calls"], 1)
        self.assertEqual(second["resources"]["aggregator_calls"], 1)
        self.assertEqual(second["resources"]["gold"], 0)
        self.assertEqual(second["resources"]["scoring_ops"], 1)
        self.assertEqual(len(second["observations"]), 1)
        self.assertEqual(env.summary()["ledger"]["total"]["tokens"], 6 * 18)
        self.assertEqual(first["observations"][0]["standalone_tokens"], 4 * 18)
        self.assertEqual(second["observations"][0]["standalone_tokens"], 4 * 18)
        self.assertEqual(env.training_packet()["standalone_tokens"].tolist(), [72.0, 72.0])
        self.assertEqual(env.training_packet()["query_indices"].tolist(), [0, 0])
        self.assertEqual(env.training_packet()["action_indices"].tolist(), [0, 1])
        self.assertEqual(len(env.execution_log[-1]["member_keys"]), 1)

    def test_joint_bundle_and_repurchase_are_equivalent(self):
        env = self.environment()
        env.serve(0, 0)
        result = env.buy(0, [0, 1])
        self.assertEqual(result["resources"]["member_calls"], 1)
        self.assertEqual(result["resources"]["aggregator_calls"], 1)
        self.assertEqual(result["resources"]["gold"], 1)
        self.assertEqual(result["resources"]["scoring_ops"], 2)
        repeated = env.buy(0, [0, 1])
        self.assertEqual(repeated["observations"], [])
        self.assertTrue(all(value == 0 for value in repeated["resources"].values()))
        self.assertEqual(env.summary()["score_observations"], 2)

    def test_unobserved_cells_can_be_nan(self):
        poisoned = deepcopy(self.train)
        poisoned["Y"][:] = np.nan
        poisoned["Y"][0, 0, [0, 1]] = self.train["Y"][0, 0, [0, 1]]
        for stem, selected in (("member", [0, 1, 2, 3]), ("aggregate", [0, 1])):
            for suffix in ("execution_counts", "prompt_tokens", "completion_tokens", "wall_time_s"):
                poisoned["cost"][f"{stem}_{suffix}"][:] = np.nan
                poisoned["cost"][f"{stem}_{suffix}"][0, 0, selected] = self.train["cost"][f"{stem}_{suffix}"][0, 0, selected]
        ordinary, guarded = self.environment(), self.environment(poisoned)
        for env in (ordinary, guarded):
            env.serve(0, 0)
            env.buy(0, [0, 1])
        self.assertEqual(ordinary.observations(), guarded.observations())
        self.assertEqual(ordinary.summary(), guarded.summary())
        reward_reads = [row for row in guarded.access_log if row["kind"] == "reward"]
        self.assertEqual(reward_reads[0]["actions"], [0, 1])

    def test_evaluation_and_non_neighbor_are_rejected(self):
        env = self.environment()
        with self.assertRaises(ValueError):
            env.serve(3, 0)
        env.serve(0, 0)
        with self.assertRaises(ValueError):
            env.buy(0, [0, 19])
        self.assertEqual(env.summary()["score_observations"], 0)
        with self.assertRaises(ValueError):
            env.serve(0, 1)

    def test_failed_purchase_does_not_commit_partial_state(self):
        broken = deepcopy(self.train)
        broken["Y"][0, 0, 1] = np.nan
        env = self.environment(broken)
        env.serve(0, 0)
        before = env.summary()
        with self.assertRaises(ValueError):
            env.buy(0, [0, 1])
        self.assertEqual(env.summary(), before)
        self.assertEqual(env.observations(), [])

    def test_public_worst_case_limit_rejects_before_hidden_read(self):
        poisoned = deepcopy(self.train)
        poisoned["cost"]["member_prompt_tokens"][:] = np.nan
        env = self.environment(poisoned, limits={"tokens": 0})
        with self.assertRaisesRegex(ValueError, "reservation"):
            env.serve(0, 0)
        self.assertEqual(env.access_log, [])
        self.assertEqual(env.summary()["ledger"]["total"]["tokens"], 0)

    def test_reference_dedup_uses_canonical_and_version(self):
        train = deepcopy(self.train)
        train["canonical"][1] = train["canonical"][0]
        env = self.environment(train)
        for q in (0, 1):
            env.serve(q, 0)
            env.buy(q, [0])
        self.assertEqual(env.summary()["gold_revealed"], 1)
        self.assertEqual(env.summary()["score_observations"], 2)

    def test_logger_is_independent_of_order_and_rewards(self):
        ids = np.array(["a", "b", "c", "d"])
        plan = make_logger_plan([0, 1, 2], ids, 17, fold=0)
        reverse = make_logger_plan([2, 0, 1], ids, 17, fold=0)
        mapping = {int(q): (int(a), int(b)) for q, a, b in zip(plan["question_indices"], plan["main_actions"], plan["neighbor_actions"])}
        for q, a, b in zip(reverse["question_indices"], reverse["main_actions"], reverse["neighbor_actions"]):
            self.assertEqual(mapping[int(q)], (int(a), int(b)))
        self.assertNotIn("reward", plan)


if __name__ == "__main__":
    unittest.main()
