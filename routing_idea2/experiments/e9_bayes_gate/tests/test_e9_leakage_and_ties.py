import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import choice, permute_blocks, validate_scope


class PermissionAndTieTests(unittest.TestCase):
    def packets(self):
        history = dict(Y=np.zeros((480, 3, 20)), ids=np.arange(480),
                       canonical=np.arange(480), global_indices=np.arange(480))
        evaluation = dict(E=np.zeros((120, 384)), ids=np.arange(480, 600),
                          canonical=np.arange(480, 600), global_indices=np.arange(480, 600))
        return history, evaluation

    def test_clean_evaluation_packet(self):
        validate_scope(*self.packets())

    def test_evaluation_reward_rejected(self):
        for name in ("Y", "reward", "aggregate_scores", "answer"):
            history, evaluation = self.packets()
            evaluation[name] = np.zeros((120, 3, 20))
            with self.assertRaises(ValueError):
                validate_scope(history, evaluation)

    def test_overlap_rejected(self):
        for name in ("ids", "canonical", "global_indices"):
            history, evaluation = self.packets()
            evaluation[name][0] = 0
            with self.assertRaises(ValueError):
                validate_scope(history, evaluation)

    def test_semantic_renaming_preserves_exact_ties(self):
        rng = np.random.default_rng(12)
        scores = rng.integers(0, 3, size=(100, 20)).astype(float)
        permutation = rng.permutation(20)
        renamed = choice(scores[:, permutation], action_order=permutation)
        np.testing.assert_array_equal(permutation[renamed], choice(scores))

    def test_block_permutation_preserves_group_and_dependencies(self):
        values = np.arange(12 * 3 * 20).reshape(12, 3, 20)
        tasks = np.array(["gsm8k"] * 6 + ["bbh"] * 6)
        sub = np.array([""] * 6 + ["a"] * 3 + ["b"] * 3)
        shuffled, permutation = permute_blocks(values, tasks, sub, 20261115, 0, 0)
        np.testing.assert_array_equal(shuffled, values[permutation])
        np.testing.assert_array_equal(tasks, tasks[permutation])
        np.testing.assert_array_equal(sub, sub[permutation])
        self.assertEqual(set(permutation), set(range(12)))


if __name__ == "__main__":
    unittest.main()
