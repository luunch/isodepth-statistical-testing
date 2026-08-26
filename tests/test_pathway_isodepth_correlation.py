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


class TestPathwayIsodepthCorrelation(unittest.TestCase):
    def test_compute_pathway_mean_score_matches_manual_mean(self) -> None:
        from experiments.studies.pathway_panel_sweep.pathway_isodepth_correlation import (
            compute_pathway_mean_score,
        )

        expression = np.array(
            [
                [1.0, 2.0, 9.0],
                [3.0, 4.0, 1.0],
                [5.0, 6.0, 3.0],
            ],
            dtype=np.float64,
        )
        gene_names = ["G1", "G2", "G3"]
        scores, n_genes = compute_pathway_mean_score(expression, gene_names, ["G1", "G3"])
        self.assertEqual(n_genes, 2)
        expected = np.mean(expression[:, [0, 2]], axis=1)
        np.testing.assert_allclose(scores, expected)


if __name__ == "__main__":
    unittest.main()
