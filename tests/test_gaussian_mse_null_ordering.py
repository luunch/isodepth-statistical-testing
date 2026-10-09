"""nll_gaussian_mse nulls are ranked by MSE; Gaussian NLL is a monotone log of MSE."""
from __future__ import annotations

import unittest

import numpy as np

from data.schemas import TestConfig
from methods.metrics import (
    compute_metric,
    gaussian_mse_from_nll,
    gaussian_nll_from_mse,
    metric_axis_label,
    permutation_p_value,
)
from methods.trainers.isodepth import (
    _compute_masked_metric_np,
    train_parallel_isodepth_model,
)


def _assert_same_order(mse: np.ndarray, nll: np.ndarray) -> None:
    mse = np.asarray(mse, dtype=np.float64)
    nll = np.asarray(nll, dtype=np.float64)
    np.testing.assert_array_equal(
        np.argsort(mse, kind="mergesort"),
        np.argsort(nll, kind="mergesort"),
    )
    mse_ranks = np.argsort(np.argsort(mse, kind="mergesort"), kind="mergesort")
    nll_ranks = np.argsort(np.argsort(nll, kind="mergesort"), kind="mergesort")
    rho = np.corrcoef(mse_ranks, nll_ranks)[0, 1]
    if not np.isfinite(rho):
        np.testing.assert_allclose(mse, mse[0])
        return
    np.testing.assert_allclose(rho, 1.0, atol=1e-12)


class GaussianMseNullOrderingTests(unittest.TestCase):
    def test_mse_nll_roundtrip(self) -> None:
        mse = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float64)
        n_total = 48 * 8
        nll = gaussian_nll_from_mse(mse, n_total=n_total)
        np.testing.assert_allclose(gaussian_mse_from_nll(nll, n_total), mse, rtol=1e-10, atol=1e-12)

    def test_nll_is_strictly_increasing_in_mse(self) -> None:
        mse = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float64)
        nll = gaussian_nll_from_mse(mse, n_total=48 * 8)
        self.assertTrue(np.all(np.diff(nll) > 0.0))
        _assert_same_order(mse, nll)

    def test_random_mse_vectors_preserve_perm_order_and_p_value(self) -> None:
        rng = np.random.default_rng(0)
        n_total = 40 * 12
        for _ in range(8):
            stats_mse = rng.uniform(0.2, 1.8, size=51)
            stats_nll = gaussian_nll_from_mse(stats_mse, n_total)
            _assert_same_order(stats_mse, stats_nll)
            p_mse = permutation_p_value("nll_gaussian_mse", float(stats_mse[0]), stats_mse[1:])
            p_nll = permutation_p_value("nll_gaussian_mse", float(stats_nll[0]), stats_nll[1:])
            self.assertEqual(p_mse, p_nll)

    def test_compute_metric_reports_mse_not_nll(self) -> None:
        rng = np.random.default_rng(1)
        y_true = rng.normal(size=(20, 5))
        y_pred = y_true + rng.normal(scale=0.3, size=y_true.shape)
        mse = float(np.mean((y_pred - y_true) ** 2))
        nll = float(gaussian_nll_from_mse(mse, y_true.size))
        reported = compute_metric("nll_gaussian_mse", y_true, y_pred)
        self.assertAlmostEqual(reported, mse, places=10)
        self.assertGreater(abs(reported - nll), 1.0)
        self.assertAlmostEqual(compute_metric("mse", y_true, y_pred), mse, places=10)

    def test_masked_metric_also_reports_mse(self) -> None:
        rng = np.random.default_rng(2)
        y_true = rng.normal(size=(16, 4))
        y_pred = y_true + rng.normal(scale=0.4, size=y_true.shape)
        mask = rng.integers(0, 2, size=y_true.shape).astype(np.float64)
        mask[0, :] = 1.0
        mse = float((((y_pred - y_true) ** 2) * mask).sum() / mask.sum())
        reported = _compute_masked_metric_np("nll_gaussian_mse", y_true, y_pred, mask)
        self.assertAlmostEqual(reported, mse, places=10)

    def test_axis_label_is_mse(self) -> None:
        self.assertEqual(metric_axis_label("nll_gaussian_mse"), "MSE")
        self.assertEqual(metric_axis_label("nll"), "MSE")

    def test_trained_permutation_null_matches_mse_order(self) -> None:
        rng = np.random.default_rng(3)
        n_cells, n_genes = 36, 6
        S = rng.uniform(size=(n_cells, 2)).astype(np.float32)
        axis = (S[:, 0] - S[:, 0].mean()) / (S[:, 0].std() + 1e-8)
        A = (
            1.6 * axis[:, None] + rng.normal(scale=0.2, size=(n_cells, n_genes))
        ).astype(np.float32)
        A = ((A - A.mean(axis=0, keepdims=True)) / (A.std(axis=0, keepdims=True) + 1e-8)).astype(
            np.float32
        )
        config = TestConfig(
            metric="nll_gaussian_mse",
            decoder="linear",
            n_perms=5,
            n_reruns=2,
            epochs=25,
            lr=0.01,
            seed=4,
            device="cpu",
            verbose=False,
        )
        _model, outputs, _ = train_parallel_isodepth_model(
            S, A, config, model_label="mse null ordering"
        )
        mse_true = float(np.mean((A - outputs.pred_true) ** 2))
        self.assertAlmostEqual(float(outputs.stat_true), mse_true, places=5)
        stats_mse = np.concatenate(
            [[float(outputs.stat_true)], np.asarray(outputs.stat_perm, dtype=np.float64)]
        )
        stats_nll = gaussian_nll_from_mse(stats_mse, n_cells * n_genes)
        _assert_same_order(stats_mse, stats_nll)
        p_mse = permutation_p_value("nll_gaussian_mse", float(stats_mse[0]), stats_mse[1:])
        p_nll = permutation_p_value("nll_gaussian_mse", float(stats_nll[0]), stats_nll[1:])
        self.assertEqual(p_mse, p_nll)
        nll_true = float(gaussian_nll_from_mse(mse_true, n_cells * n_genes))
        # Homoscedastic NLL can be negative when MSE is small, so compare
        # identity rather than assuming NLL > MSE.
        self.assertGreater(abs(float(outputs.stat_true) - nll_true), 1.0)


if __name__ == "__main__":
    unittest.main()
