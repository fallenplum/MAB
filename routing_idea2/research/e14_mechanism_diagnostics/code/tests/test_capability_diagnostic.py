from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capability_diagnostic import analyse_capabilities, support_bin


class CapabilityDiagnosticTests(unittest.TestCase):
    def fixture(self):
        n_actions = 20
        capability = np.full((3, n_actions), .2)
        capability[0, 0], capability[1, 1], capability[2, 3] = .75, .8, .9
        phi = np.zeros((3, 17))
        phi[:, 0] = 1
        phi[:, 1] = (-1, 1, 0)
        residual = np.zeros((17, n_actions))
        residual[1, 0] = -.4
        residual[1, 2] = .8
        counts = np.zeros((3, n_actions), dtype=int)
        sums = np.zeros_like(counts, dtype=float)
        counts[0, 0], counts[0, 2], counts[1, 1] = 1, 1, 2
        sums[0, 0], sums[0, 2], sums[1, 1] = 1, 0, 1
        inputs = dict(
            oracle_group_e=np.array([0, 0, 1]), group_h=np.array([0, 0, 1, 1, 2]),
            phi_e=phi, group_keys=np.array([["gsm8k", ""], ["bbh", "a"], ["bbh", "empty"]]),
            E=np.array([10, 11, 12]),
        )
        snapshot = dict(
            own__C=capability, own__residual_weights=residual,
            own_counts__group_count=counts, own_counts__group_sum=sums,
            observed_local_queries=np.array([0, 1, 2, 3]), observed_actions=np.array([0, 2, 1, 1]),
            observed_rewards=np.array([1., 0., 1., 0.]),
        )
        reward = np.zeros((3, n_actions))
        reward[1, 0], reward[0, 2], reward[2, 1] = 1, 1, 1
        mean = reward.copy()
        mean[0, 0] = 1/3
        return inputs, snapshot, reward, mean

    def test_support_bins_frozen(self):
        self.assertEqual([support_bin(x) for x in (0, 1, 2, 3, 4, 5, 9, 10, 99)],
                         ["0", "1", "2", "3-4", "3-4", "5-9", "5-9", "10+", "10+"])

    def test_original_table_unchanged_and_true_group_used(self):
        inputs, snapshot, rewards, means = self.fixture()
        before = snapshot["own__C"].copy()
        output = analyse_capabilities(inputs, snapshot, rewards, means)
        np.testing.assert_array_equal(snapshot["own__C"], before)
        np.testing.assert_array_equal(output["array_diagnostics"]["scores"][0], before[inputs["oracle_group_e"]])
        self.assertEqual(output["array_diagnostics"]["scores"].shape, (2, 3, 20))
        self.assertEqual(len(output["cell_rows"]), 60)
        self.assertEqual(len(output["query_rows"]), 6)

    def test_binary_mse_differs_from_group_mean_error(self):
        output = analyse_capabilities(*self.fixture())
        cell = next(row for row in output["cell_rows"] if row["group_index"] == 0 and row["action"] == 0)
        self.assertEqual(cell["support"], 1)
        self.assertEqual(cell["evaluation_original_questions"], 2)
        self.assertAlmostEqual(cell["heldout_seed_mean"], .5)
        self.assertAlmostEqual(cell["group_seed_signed_mean_error"], .25)
        self.assertAlmostEqual(cell["group_seed_squared_mean_error"], .0625)
        self.assertAlmostEqual(cell["group_individual_binary_reward_mse"], .3125)

    def test_empty_evaluation_group_has_null_error_fields(self):
        output = analyse_capabilities(*self.fixture())
        rows = [row for row in output["cell_rows"] if row["group_index"] == 2]
        self.assertEqual(len(rows), 20)
        for row in rows:
            self.assertEqual(row["evaluation_original_questions"], 0)
            self.assertIsNone(row["heldout_seed_mean"])
            self.assertIsNone(row["group_seed_abs_mean_error"])
            self.assertIsNone(row["group_residual_individual_binary_reward_mse"])

    def test_residual_routes_keep_table_but_change_choices(self):
        output = analyse_capabilities(*self.fixture())
        arrays = output["array_diagnostics"]
        np.testing.assert_array_equal(arrays["actions"][0], [0, 0, 1])
        np.testing.assert_array_equal(arrays["actions"][1], [0, 2, 1])
        self.assertEqual(int(arrays["destructions_vs_group"][1].sum()), 1)
        self.assertEqual(int(arrays["corrections_vs_group"][1].sum()), 0)
        self.assertEqual(float(arrays["net_seed_reward_vs_group"][1].sum()), -1)
        query = next(row for row in output["query_rows"]
                     if row["evaluation_local_index"] == 1 and row["method"] == "GroupResidualOracle")
        self.assertEqual(query["selected_support"], 1)
        self.assertEqual(query["selected_support_bin"], "1")
        self.assertAlmostEqual(query["seed_empirical_selection_loss"], 1)

    def test_changed_or_dense_reward_unit_rejected(self):
        inputs, snapshot, rewards, means = self.fixture()
        rewards[0, 0] = np.nan
        with self.assertRaises(ValueError):
            analyse_capabilities(inputs, snapshot, rewards, means)

    def test_support_must_match_only_purchased_tuples(self):
        inputs, snapshot, rewards, means = self.fixture()
        snapshot["own_counts__group_count"][0, 3] = 1
        with self.assertRaises(ValueError):
            analyse_capabilities(inputs, snapshot, rewards, means)


if __name__ == "__main__":
    unittest.main()
