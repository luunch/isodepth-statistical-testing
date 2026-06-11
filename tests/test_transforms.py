from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.transforms import (
    _CPM_TARGET,
    apply_expression_transforms,
    apply_expression_transforms_by_celltype,
    celltype_expression_residuals,
    filter_genes_by_min_cells,
    normalize_total_expression,
    zscore_covariate,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestBasicTransforms(unittest.TestCase):
    def test_zscore_covariate_unit_std(self) -> None:
        values = np.asarray([0.84, 0.86, 0.88, 0.90, 0.92], dtype=np.float32)
        normalized = zscore_covariate(values)
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(normalized.std()), 1.0, places=5)

    def test_zscore_covariate_constant_returns_zeros(self) -> None:
        values = np.full(10, 0.5, dtype=np.float32)
        np.testing.assert_allclose(zscore_covariate(values), np.zeros(10, dtype=np.float32))

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

    def test_log1p_runs_before_standardization(self) -> None:
        counts = np.asarray(
            [
                [0.0, 3.0],
                [1.0, 7.0],
                [3.0, 15.0],
            ],
            dtype=np.float32,
        )
        transformed, metadata = apply_expression_transforms(
            counts,
            log1p=True,
            standardize_expression=True,
            return_metadata=True,
        )
        expected_logged = np.log1p(counts)
        expected = (expected_logged - expected_logged.mean(axis=0, keepdims=True)) / (
            expected_logged.std(axis=0, keepdims=True) + 1e-8
        )
        np.testing.assert_allclose(transformed, expected.astype(np.float32), atol=1e-6)
        self.assertTrue(metadata["log1p"])
        self.assertTrue(metadata["standardize_expression"])

    def test_log1p_without_standardization_preserves_logged_scale(self) -> None:
        counts = np.asarray(
            [
                [0.0, 1.0],
                [3.0, 7.0],
            ],
            dtype=np.float32,
        )
        transformed = apply_expression_transforms(
            counts,
            log1p=True,
            standardize_expression=False,
        )
        np.testing.assert_allclose(transformed, np.log1p(counts).astype(np.float32), atol=1e-6)

    def test_normalize_total_scales_to_cpm(self) -> None:
        counts = np.asarray(
            [
                [2.0, 3.0],
                [10.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        normalized = normalize_total_expression(counts)
        np.testing.assert_allclose(normalized.sum(axis=1), [_CPM_TARGET, _CPM_TARGET, 0.0], atol=1e-3)
        np.testing.assert_allclose(
            normalized[0],
            counts[0] / counts[0].sum() * _CPM_TARGET,
            atol=1e-3,
        )

    def test_normalize_total_then_log1p_is_log_cpm(self) -> None:
        counts = np.asarray([[2.0, 3.0], [10.0, 0.0]], dtype=np.float32)
        transformed = apply_expression_transforms(
            counts,
            normalize_total=True,
            log1p=True,
            standardize_expression=False,
        )
        cpm = counts / counts.sum(axis=1, keepdims=True) * _CPM_TARGET
        np.testing.assert_allclose(transformed, np.log1p(cpm).astype(np.float32), atol=1e-6)

    def test_log1p_rejects_negative_values(self) -> None:
        expression = np.asarray(
            [
                [1.0, -0.5],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )
        with self.assertRaises(ValueError):
            apply_expression_transforms(expression, log1p=True, standardize_expression=False)

    def test_log1p_cannot_be_combined_with_q(self) -> None:
        counts = np.asarray(
            [
                [1.0, 0.0],
                [2.0, 3.0],
            ],
            dtype=np.float32,
        )
        with self.assertRaises(ValueError):
            apply_expression_transforms(counts, log1p=True, q=1)


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
            standardize_expression=True,
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
            standardize_expression=False,
            q=1,
            seed=11,
        )
        latent_b = apply_expression_transforms(
            counts,
            standardize_expression=False,
            q=1,
            seed=11,
        )

        np.testing.assert_allclose(latent_a, latent_b, atol=1e-6)

    def test_q_by_celltype_runs_independently_per_type(self) -> None:
        counts = np.asarray(
            [
                [10.0, 0.0, 1.0, 0.0],
                [12.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 15.0],
                [0.0, 0.0, 2.0, 14.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

        per_type, meta = apply_expression_transforms_by_celltype(
            counts,
            labels,
            standardize_expression=True,
            q=1,
            seed=3,
            return_metadata=True,
        )
        global_latent, _ = apply_expression_transforms(
            counts,
            standardize_expression=True,
            q=1,
            seed=3,
            return_metadata=True,
        )

        self.assertEqual(per_type.shape, (4, 2))
        self.assertTrue(meta["q_by_celltype"])
        self.assertFalse(np.allclose(per_type, global_latent, atol=1e-4))
        np.testing.assert_allclose(per_type[labels == 0].mean(axis=0), np.zeros(2), atol=1e-5)
        np.testing.assert_allclose(per_type[labels == 1].mean(axis=0), np.zeros(2), atol=1e-5)

    def test_celltype_expression_residuals_zero_means_per_type(self) -> None:
        expression = np.asarray(
            [
                [1.0, 10.0],
                [3.0, 14.0],
                [20.0, 2.0],
                [24.0, 6.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        residuals = celltype_expression_residuals(expression, labels, n_cell_types=2)
        np.testing.assert_allclose(residuals[labels == 0].mean(axis=0), np.zeros(2), atol=1e-6)
        np.testing.assert_allclose(residuals[labels == 1].mean(axis=0), np.zeros(2), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
