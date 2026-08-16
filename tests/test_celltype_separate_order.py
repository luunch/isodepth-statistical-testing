from __future__ import annotations

import unittest

import numpy as np

from methods.permutation import _celltype_indices_by_descending_cell_count


class CelltypeSeparateOrderTests(unittest.TestCase):
    def test_orders_by_descending_cell_count(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        order = _celltype_indices_by_descending_cell_count(labels, n_cell_types=3)
        self.assertEqual(order, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
