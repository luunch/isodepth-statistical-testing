from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.studies.pathway_panel_sweep.pathway_expression_power import (
    compute_pathway_expression_metrics,
)


class TestPathwayExpressionPower(unittest.TestCase):
    def test_low_detection_pathway_has_fewer_passing_genes(self) -> None:
        counts = np.array(
            [
                [10, 0, 5],
                [12, 0, 4],
                [8, 0, 6],
                [0, 0, 0],
            ],
            dtype=np.float64,
        )
        metrics_sparse = compute_pathway_expression_metrics(
            counts,
            ["G1", "G2", "G3"],
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float64),
            ["G1", "G2"],
            min_cells_per_gene=3,
        )
        metrics_dense = compute_pathway_expression_metrics(
            counts,
            ["G1", "G2", "G3"],
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float64),
            ["G1", "G3"],
            min_cells_per_gene=3,
        )
        self.assertLess(metrics_sparse["pathway_genes_min3cells"], metrics_dense["pathway_genes_min3cells"])
        self.assertLess(metrics_sparse["mean_detection_rate"], metrics_dense["mean_detection_rate"])


if __name__ == "__main__":
    unittest.main()
