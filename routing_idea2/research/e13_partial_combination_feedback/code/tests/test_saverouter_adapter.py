"""Small synthetic checks of the unchanged author estimator and COO adapter."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saverouter_adapter import SaveRouterFixedMask, load_upstream, predict_from_state


class SaveRouterAdapterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261114013)
        self.features = rng.normal(size=(240, 16))
        self.groups = np.arange(240) % 8
        selected = np.arange(0, 240, 2)
        self.q = np.repeat(selected, 2)
        first = np.arange(len(selected)) % 20
        self.a = np.column_stack((first, (first+1) % 20)).ravel()
        self.y = rng.integers(0, 2, len(self.q)).astype(float)
        self.cost = rng.integers(100, 1000, len(self.q)).astype(float)
        order = rng.permutation(len(self.q))
        self.q, self.a, self.y, self.cost = [v[order] for v in (self.q, self.a, self.y, self.cost)]
        self.query = rng.normal(size=(25, 16))
        self.query_groups = np.arange(25) % 7

    def test_matches_original_author_class_and_state_restore(self):
        adapter = SaveRouterFixedMask().fit(self.features, self.groups, self.q, self.a, self.y, self.cost)
        Sparse, Native, parameters = load_upstream()
        selected = np.unique(self.q)
        supervision = Sparse(n_queries=len(selected), n_models=20,
                             query_indices=np.searchsorted(selected, self.q), model_indices=self.a,
                             rewards=self.y, group_ids=self.groups[selected], k=2, costs=self.cost)
        original = Native(**parameters).fit(self.features[selected], supervision)
        predicted = adapter.predict_scores(self.query, self.query_groups)
        expected = original.predict_scores(self.query, group_ids=self.query_groups)
        np.testing.assert_array_equal(predicted, expected)
        np.testing.assert_allclose(predict_from_state(adapter.state_dict(), self.query, self.query_groups),
                                   expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(adapter.predict_costs(self.query, self.query_groups),
                                      original.predict_costs(self.query, group_ids=self.query_groups))
        self.assertEqual(len(adapter.native_.residual_models_), 20)

    def test_unobserved_history_features_are_not_fitting_rows(self):
        first = SaveRouterFixedMask().fit(self.features, self.groups, self.q, self.a, self.y)
        changed = self.features.copy()
        changed[1::2] = 1e6
        second = SaveRouterFixedMask().fit(changed, self.groups, self.q, self.a, self.y)
        np.testing.assert_array_equal(first.predict_scores(self.query, self.query_groups),
                                      second.predict_scores(self.query, self.query_groups))

    def test_rejects_dense_unrevealed_rewards(self):
        with self.assertRaises(ValueError):
            SaveRouterFixedMask().fit(self.features, self.groups, self.q, self.a, np.zeros((240, 20)))

    def test_rejects_variable_actions_and_duplicate_generation_pairs(self):
        with self.assertRaises(ValueError):
            SaveRouterFixedMask().fit(self.features, self.groups, self.q[:-1], self.a[:-1], self.y[:-1])
        duplicate = self.a.copy()
        positions = np.flatnonzero(self.q == self.q[0])
        duplicate[positions[1]] = duplicate[positions[0]]
        with self.assertRaises(ValueError):
            SaveRouterFixedMask().fit(self.features, self.groups, self.q, duplicate, self.y)


if __name__ == "__main__":
    unittest.main()
