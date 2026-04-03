from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.transforms import apply_expression_transforms, filter_genes_by_min_cells


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestBasicTransforms(unittest.TestCase):
    def test_filter_genes_by_min_cells_drops_underexpressed_genes(self) -> None:
        a = np.asarray(
            [
                [1.0, 0.0, 3.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        filtered = filter_genes_by_min_cells(a, min_cells_per_gene=2)
        self.assertEqual(filtered.shape, (3, 2))
        np.testing.assert_allclose(filtered, a[:, [0, 2]])


@unittest.skipUnless(HAS_TORCH, "torch is required for Poisson low-rank transform tests")
class TestPoissonLowRankTransform(unittest.TestCase):
    def test_q_returns_top_2q_latent_dimensions(self) -> None:
        counts = np.asarray(
            [
                [5.0, 2.0, 0.0, 1.0],
                [4.0, 1.0, 1.0, 0.0],
                [0.0, 3.0, 6.0, 2.0],
                [1.0, 2.0, 5.0, 4.0],
                [2.0, 0.0, 1.0, 3.0],
            ],
            dtype=np.float32,
        )

        latent, metadata = apply_expression_transforms(
            counts,
            min_cells_per_gene=1,
            standardize=True,
            q=2,
            seed=7,
            return_metadata=True,
        )

        self.assertEqual(latent.shape, (5, 4))
        self.assertEqual(metadata["representation"], "poisson_low_rank_latent")
        self.assertEqual(metadata["q"], 2)
        self.assertEqual(metadata["latent_dim"], 4)
        self.assertEqual(len(metadata["feature_names"]), 4)
        np.testing.assert_allclose(latent.mean(axis=0), np.zeros(4, dtype=np.float32), atol=1e-5)

    def test_poisson_transform_is_reproducible_for_fixed_seed(self) -> None:
        counts = np.asarray(
            [
                [3.0, 1.0, 0.0],
                [4.0, 0.0, 1.0],
                [0.0, 2.0, 5.0],
                [1.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        )

        latent_a = apply_expression_transforms(
            counts,
            standardize=False,
            q=1,
            seed=11,
        )
        latent_b = apply_expression_transforms(
            counts,
            standardize=False,
            q=1,
            seed=11,
        )

        np.testing.assert_allclose(latent_a, latent_b, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
