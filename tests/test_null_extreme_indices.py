from __future__ import annotations

import unittest

import numpy as np

from methods.trainers.isodepth import _null_extreme_indices


class NullExtremeIndicesTests(unittest.TestCase):
    def test_empty_null_metrics_returns_zero_indices(self) -> None:
        low, high = _null_extreme_indices(np.array([1.0]), "mse")
        self.assertEqual((low, high), (0, 0))

    def test_single_null_metric(self) -> None:
        low, high = _null_extreme_indices(np.array([1.0, 2.5]), "mse")
        self.assertEqual((low, high), (0, 0))


if __name__ == "__main__":
    unittest.main()
