"""Exact controlled-comparison arithmetic and frozen grid checks."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from e15_common import CAPS, PRIMARY_CAP, METHODS


class DesignTests(unittest.TestCase):
    def test_exact_coverage_decomposition(self):
        rng=np.random.default_rng(20261115015)
        main_full,alt_full,main_common,alt_common=rng.integers(0,2,size=(4,1000))
        np.testing.assert_array_equal(alt_full-main_full,
            (alt_common-main_common)+(alt_full-alt_common)-(main_full-main_common))

    def test_unique_primary_and_locked_capacity(self):
        self.assertEqual(CAPS,(125000,250000,500000,750000))
        self.assertEqual(PRIMARY_CAP,250000)
        self.assertEqual(METHODS,("RawCentered","Group","SaveRouterFixedMask"))

    def test_zero_integer_net_is_not_positive(self):
        difference=np.array([1,-1,0,0],dtype=int)
        self.assertFalse(int(difference.sum())>0)


if __name__=="__main__":
    unittest.main()
