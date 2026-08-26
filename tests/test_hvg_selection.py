from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.h5ad_loader import (  # noqa: E402
    _seurat_hvg_mask_from_counts,
    _select_hvg_mask_from_counts,
    preprocess_celltype_subset,
)


class TestSeuratHvgFallback(unittest.TestCase):
    def test_selects_high_variance_genes_without_scanpy(self) -> None:
        rng = np.random.default_rng(0)
        n_cells, n_low, n_high = 120, 40, 10
        low = rng.poisson(5.0, size=(n_cells, n_low)).astype(np.float32)
        high = np.where(rng.random((n_cells, n_high)) < 0.5, 0.0, 10.0).astype(np.float32)
        counts = np.concatenate([low, high], axis=1)
        mask = _seurat_hvg_mask_from_counts(counts, n_top=n_high)
        self.assertEqual(int(mask.sum()), n_high)
        gene_var = counts.var(axis=0)
        self.assertGreater(float(gene_var[mask].mean()), float(gene_var[~mask].mean()))
        again = _seurat_hvg_mask_from_counts(counts, n_top=n_high)
        np.testing.assert_array_equal(mask, again)

    def test_from_counts_works_when_scanpy_missing(self) -> None:
        rng = np.random.default_rng(1)
        counts = rng.poisson(2.0, size=(50, 30)).astype(np.float32)
        with mock.patch.dict("sys.modules", {"scanpy": None}):
            mask = _select_hvg_mask_from_counts(counts, n_top=8)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(int(mask.sum()), 8)

    def test_preprocess_celltype_subset_without_scanpy(self) -> None:
        rng = np.random.default_rng(2)
        counts = rng.poisson(3.0, size=(40, 25)).astype(np.float32)
        names = [f"g{i}" for i in range(counts.shape[1])]
        with mock.patch.dict("sys.modules", {"scanpy": None}):
            A, kept, space = preprocess_celltype_subset(
                counts,
                names,
                min_cells_per_gene=0,
                top_var_genes=6,
                normalize_total=True,
                log1p=True,
                standardize_expression=True,
            )
        self.assertEqual(A.shape, (40, 6))
        self.assertEqual(len(kept), 6)
        self.assertEqual(space, "gene_expression")


if __name__ == "__main__":
    unittest.main()
