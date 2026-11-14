from pathlib import Path
import ast
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from learning import fit_ridge,balanced_order,seed_subset


class LearningTests(unittest.TestCase):
    def test_individual_and_mean_loss_regularization(self):
        rng=np.random.default_rng(161)
        x=np.c_[np.ones(20),rng.normal(size=(20,4))]
        labels=rng.integers(0,2,size=(3,20,5))
        a,_=fit_ridge(np.tile(x,(3,1)),labels.reshape(60,5),100)
        b,_=fit_ridge(x,labels.mean(axis=0),100/3)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

    def test_budget_and_question_balance(self):
        tasks=np.array(["gsm8k"]*240+["bbh"]*240)
        order=balanced_order(np.arange(480),tasks,17,0)
        for budget in (240,480):
            for length in (1,2,3):
                n=budget//length
                self.assertEqual(np.sum(tasks[order[:n]]=="bbh"),n//2)
                self.assertEqual(n*length,budget)

    def test_seed_combinations(self):
        self.assertEqual({tuple(seed_subset(2,r)) for r in range(3)},{(0,1),(1,2),(0,2)})
        for r in range(3):np.testing.assert_array_equal(seed_subset(3,r),[0,1,2])

    def test_all_source_syntax(self):
        root=Path(__file__).resolve().parents[1]
        for p in root.rglob("*.py"):ast.parse(p.read_text(encoding="utf-8"))


if __name__=="__main__":unittest.main()
