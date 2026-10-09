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

    def test_scanpy_hvg_selection_preserves_dense_and_sparse_counts(self) -> None:
        import anndata as ad
        from scipy import sparse
        from data.h5ad_loader import _select_hvg_mask
        try:
            import scanpy  # noqa: F401
        except ImportError:
            self.skipTest("Scanpy unavailable")
        counts = np.random.default_rng(19).poisson(2.0, (60, 40)).astype(np.float32)
        for matrix in [counts.copy(), sparse.csr_matrix(counts)]:
            with self.subTest(sparse=sparse.issparse(matrix)):
                adata = ad.AnnData(matrix)
                _select_hvg_mask(adata, 10)
                actual = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                np.testing.assert_array_equal(actual, counts)
        before = counts.copy()
        preprocess_celltype_subset(
            counts, [f"g{i}" for i in range(40)], top_var_genes=10,
            normalize_total=True, log1p=True, standardize_expression=True,
        )
        np.testing.assert_array_equal(counts, before)

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

    def test_preprocess_celltype_subset_expression_pcs(self) -> None:
        rng = np.random.default_rng(3)
        n_cells, n_genes = 80, 40
        z = rng.normal(size=n_cells)
        signal = z[:, None] * rng.normal(size=n_genes)
        noise = rng.normal(size=(n_cells, n_genes)) * 0.1
        counts = np.clip(np.exp(signal + noise), 0.0, None).astype(np.float32)
        names = [f"g{i}" for i in range(n_genes)]
        A, kept, space = preprocess_celltype_subset(
            counts,
            names,
            min_cells_per_gene=0,
            top_var_genes=0,
            normalize_total=True,
            log1p=True,
            standardize_expression=True,
            n_expression_pcs=5,
        )
        self.assertEqual(A.shape, (n_cells, 5))
        self.assertEqual(kept, [f"PC{i}" for i in range(1, 6)])
        self.assertEqual(space, "expression_pcs")
        np.testing.assert_allclose(A.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(A.std(axis=0), 1.0, atol=1e-4)


class TestLog1pNormalizedInput(unittest.TestCase):
    def test_restore_inverts_log1p_cpm10k(self) -> None:
        import anndata as ad

        from data.h5ad_loader import _restore_linear_normalized_expression

        counts = np.array([[8000.0, 2000.0], [4000.0, 6000.0]], dtype=np.float32)
        adata = ad.AnnData(X=np.log1p(counts))
        restored, meta = _restore_linear_normalized_expression(
            adata,
            representation="log1p_normalized",
            normalization_target=10000.0,
        )
        np.testing.assert_allclose(np.asarray(restored.X), counts, rtol=1e-5, atol=1e-4)
        self.assertTrue(meta["restored_linear_expression"])
        self.assertLess(float(meta["restored_row_total_max_relative_error"]), 1e-3)

    def test_duplicate_symbols_are_uniquified_in_memory(self) -> None:
        import tempfile

        import anndata as ad

        from data.h5ad_loader import load_h5ad_dataset

        counts = np.array([[8000.0, 2000.0], [4000.0, 6000.0]], dtype=np.float32)
        adata = ad.AnnData(X=np.log1p(counts))
        adata.var_names = ["Gene", "Gene"]
        adata.obs_names = ["c0", "c1"]
        adata.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup.h5ad"
            adata.write_h5ad(path)
            dataset = load_h5ad_dataset(
                h5ad_path=str(path),
                input_expression_representation="log1p_normalized",
                input_normalization_target=10000.0,
                make_var_names_unique=True,
                min_cells_per_gene=0,
                normalize_total=True,
                log1p=True,
                standardize_expression=True,
            )
        self.assertEqual(dataset.n_genes, 2)
        self.assertEqual(len(set(dataset.meta["var_names"])), 2)
        self.assertTrue(dataset.meta["make_var_names_unique"])
        self.assertTrue(dataset.meta["restored_linear_expression"])


if __name__ == "__main__":
    unittest.main()
