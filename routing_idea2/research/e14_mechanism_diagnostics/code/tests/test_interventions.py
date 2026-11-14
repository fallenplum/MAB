"""Regression checks for E14's fixed-table, true-group intervention."""
import unittest
import numpy as np

from run_e14 import interventions


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(predicted_group_e=np.array([0, 0]), oracle_group_e=np.array([0, 1]),
                           phi_e=np.column_stack((np.ones(2), np.zeros((2, 16)))))
        c = np.tile(np.linspace(.1, .9, 20), (2, 1))
        c[1] = c[1, ::-1]
        names = ["Raw", "Group", "GroupResidual", "RawCentered", "SaveRouterFixedMask",
                 "GlobalFixed", "TaskFixed", "SubtaskFixed"]
        scores = np.repeat(c[[0, 0]][None], len(names), axis=0)
        self.snapshot = dict(methods=np.array(names), scores=scores,
            actions=np.argmax(scores, axis=-1), own__C=c, own__residual_weights=np.zeros((17, 20)),
            own__task_fixed=c.copy(), own__task_codes=np.array([0, 1]), own__subtask_fixed=c.copy(),
            sr__group_scores=c.copy(), sr__residual_fitted=np.zeros(20, dtype=bool))

    def test_only_group_lookup_changes(self):
        c = self.snapshot["own__C"].copy()
        actual = interventions(self.inputs, self.snapshot)
        np.testing.assert_array_equal(actual, np.tile([19, 0], (5, 1)))
        np.testing.assert_array_equal(self.snapshot["own__C"], c)

    def test_bad_original_group_scores_rejected(self):
        self.snapshot["scores"][1, 0, 0] += .1
        with self.assertRaises(ValueError):
            interventions(self.inputs, self.snapshot)


if __name__ == "__main__":
    unittest.main()
