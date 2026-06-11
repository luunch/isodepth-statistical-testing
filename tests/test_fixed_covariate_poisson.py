"""Tests for Poisson-aware fixed covariate decoder training."""
from __future__ import annotations

import unittest

import numpy as np

from data.schemas import TestConfig
from methods.metrics import compute_metric
from methods.trainers.isodepth import train_fixed_covariate_model


class TestFixedCovariatePoissonTraining(unittest.TestCase):
    def test_poisson_covariate_trains_with_poisson_loss_not_mse(self) -> None:
        rng = np.random.default_rng(0)
        n_cells, n_genes = 48, 12
        covariate = rng.normal(size=n_cells)
        counts = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
        config = TestConfig(
            metric="nll_poisson_mse",
            epochs=40,
            n_reruns=2,
            lr=0.01,
            seed=0,
            device="cpu",
            decoder="linear",
            verbose=False,
        )
        _, pred = train_fixed_covariate_model(
            covariate,
            counts,
            config,
            model_label="test poisson covariate",
        )
        poisson_nll = compute_metric("nll_poisson_mse", counts, pred)
        self.assertTrue(np.isfinite(poisson_nll))
        null_pred = np.broadcast_to(
            np.log(np.maximum(counts.mean(axis=0, keepdims=True), 1e-3)),
            counts.shape,
        )
        null_nll = compute_metric("nll_poisson_mse", counts, null_pred)
        self.assertLess(poisson_nll, null_nll)


if __name__ == "__main__":
    unittest.main()
