from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learners import METHODS, fit_learners, predict_learners


class SparseLearnerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026111501)
        self.phi = np.column_stack((np.ones(12), rng.normal(size=(12, 16))))
        self.group = np.repeat(np.arange(3), 4)
        self.tasks = np.array(["gsm8k", "bbh", "bbh"])
        self.query = np.repeat(np.arange(12), 2)
        self.action = np.tile([0, 1], 12)
        self.reward = rng.integers(0, 2, size=len(self.query)).astype(float)

    def fit(self, **updates):
        arguments = dict(phi_h=self.phi, group_h=self.group, group_to_task=self.tasks,
                         query_indices=self.query, action_indices=self.action, rewards=self.reward)
        arguments.update(updates)
        return fit_learners(**arguments)

    def test_shapes_and_no_missing_training_rows(self):
        state = self.fit()
        scores = predict_learners(state, self.phi, self.group)
        self.assertEqual(set(scores), set(METHODS))
        for values in scores.values():
            self.assertEqual(values.shape, (12, 20))
        self.assertEqual(state["raw_weights"].shape, (17, 20))
        self.assertEqual(state["diagnostics"]["n_observations"], 24)
        self.assertEqual(state["diagnostics"]["missing_reward_rows_created"], 0)
        self.assertEqual(state["diagnostics"]["actual_normal_matrix_solves"], 2)
        np.testing.assert_array_equal(state["raw_weights"][:, 2:], 0)

    def test_hierarchical_capabilities_match_manual_formula(self):
        state = self.fit()
        grand = self.reward.mean()
        for action in (0, 1, 2):
            rows_a = self.action == action
            global_a = (self.reward[rows_a].sum() + 10*grand) / (rows_a.sum()+10)
            self.assertAlmostEqual(state["global_a"][action], global_a)
            for group in range(3):
                task = self.tasks[group]
                rows_t = self.tasks[self.group[self.query]] == task
                ta = rows_a & rows_t
                task_a = (self.reward[ta].sum()+10*global_a) / (ta.sum()+10)
                ga = rows_a & (self.group[self.query] == group)
                expected = task_a if task == "gsm8k" else (
                    self.reward[ga].sum()+3*task_a) / (ga.sum()+3)
                self.assertAlmostEqual(state["C"][group, action], expected)

    def test_raw_and_centered_normal_equations(self):
        state = self.fit()
        for action in (0, 1):
            rows = self.action == action
            design = self.phi[self.query[rows]]
            normal = design.T @ design + 1000*np.eye(17)
            expected = np.linalg.solve(normal, design.T @ self.reward[rows])
            np.testing.assert_allclose(state["raw_weights"][:, action], expected, rtol=0, atol=1e-15)
            expected = np.linalg.solve(normal, design.T @ (self.reward[rows]-state["global_a"][action]))
            np.testing.assert_allclose(state["centered_weights"][:, action], expected, rtol=0, atol=1e-15)

    def test_constant_column_receives_penalty(self):
        phi = np.zeros((12, 17))
        phi[:, 0] = 1
        state = self.fit(phi_h=phi)
        expected = self.reward[self.action == 0].sum() / (12+1000)
        self.assertAlmostEqual(state["raw_weights"][0, 0], expected)
        np.testing.assert_array_equal(state["raw_weights"][1:], 0)

    def test_residual_excludes_all_actions_of_current_question(self):
        state = self.fit()
        for question in range(12):
            keep = self.query != question
            reduced = self.fit(query_indices=self.query[keep], action_indices=self.action[keep], rewards=self.reward[keep])
            rows = self.query == question
            expected = reduced["C"][self.group[question], self.action[rows]]
            np.testing.assert_allclose(state["loo_baseline"][rows], expected, rtol=0, atol=1e-14)

    def test_own_rewards_do_not_change_own_loo_baseline(self):
        before = self.fit()
        changed = self.reward.copy()
        changed[self.query == 3] = 1-changed[self.query == 3]
        after = self.fit(rewards=changed)
        np.testing.assert_allclose(before["loo_baseline"][self.query == 3],
                                   after["loo_baseline"][self.query == 3], rtol=0, atol=1e-14)

    def test_single_question_loo_falls_back_to_half(self):
        state = self.fit(query_indices=np.array([2, 2]), action_indices=np.array([1, 3]), rewards=np.array([0., 1.]))
        np.testing.assert_allclose(state["loo_baseline"], .5, rtol=0, atol=1e-14)
        self.assertEqual(state["fallback"]["loo_observations_with_no_other_question_feedback"], 2)

    def test_empty_feedback_records_priors_without_labels(self):
        state = self.fit(query_indices=np.array([], dtype=int), action_indices=np.array([], dtype=int), rewards=np.array([]))
        scores = predict_learners(state, self.phi, self.group)
        np.testing.assert_array_equal(scores["Raw"], 0)
        for name in METHODS[1:]:
            np.testing.assert_array_equal(scores[name], .5)
        self.assertEqual(state["diagnostics"]["n_observations"], 0)
        self.assertTrue(state["fallback"]["grand_weak_prior_used"])

    def test_full_three_repeat_expansion_recovers_e12_mean_scale(self):
        rng = np.random.default_rng(5)
        outcomes = rng.integers(0, 2, size=(12, 3, 20)).astype(float)
        query, seed, action = np.indices(outcomes.shape)
        state = self.fit(query_indices=query.ravel(), action_indices=action.ravel(), rewards=outcomes.ravel())
        expected = np.linalg.solve(self.phi.T@self.phi+(1000/3)*np.eye(17), self.phi.T@outcomes.mean(axis=1))
        np.testing.assert_allclose(state["raw_weights"], expected, rtol=0, atol=2e-16)

    def test_action_relabeling_preserves_scores(self):
        before = self.fit()
        permutation = np.random.default_rng(43).permutation(20)
        inverse = np.argsort(permutation)
        after = self.fit(action_indices=inverse[self.action])
        original = predict_learners(before, self.phi, self.group)
        renamed = predict_learners(after, self.phi, self.group)
        for name in METHODS:
            np.testing.assert_allclose(renamed[name][:, inverse], original[name], rtol=0, atol=1e-13)

    def test_dense_or_nonfinite_reward_input_rejected(self):
        with self.assertRaises(ValueError):
            self.fit(rewards=np.zeros((24, 20)))
        bad = self.reward.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            self.fit(rewards=bad)


if __name__ == "__main__":
    unittest.main()
