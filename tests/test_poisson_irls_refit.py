"""Tests for post-hoc Poisson IRLS decoder refit on isodepth slots."""
from __future__ import annotations

import unittest

import numpy as np

from data.schemas import TestConfig
from methods.metrics import compute_metric
from methods.trainers.isodepth import (
    fit_poisson_glm_irls,
    poisson_parametric_decoder_uses_irls,
    train_parallel_isodepth_model,
)


class TestPoissonIrlsRefit(unittest.TestCase):
    def test_poisson_parametric_decoder_uses_irls_predicate(self) -> None:
        quad_poisson = TestConfig(
            metric="nll_poisson_mse",
            decoder="quadratic",
            n_perms=1,
            n_reruns=1,
            epochs=1,
        )
        nn_poisson = TestConfig(
            metric="nll_poisson_mse",
            decoder="nn",
            n_perms=1,
            n_reruns=1,
            epochs=1,
        )
        gauss_quad = TestConfig(
            metric="nll_gaussian_mse",
            decoder="quadratic",
            n_perms=1,
            n_reruns=1,
            epochs=1,
        )
        self.assertTrue(poisson_parametric_decoder_uses_irls(quad_poisson))
        self.assertFalse(poisson_parametric_decoder_uses_irls(nn_poisson))
        self.assertFalse(poisson_parametric_decoder_uses_irls(gauss_quad))

    def test_predict_latent_applies_fit_coefficients_to_query_cells(self) -> None:
        rng = np.random.default_rng(0)
        n_cells, n_genes = 40, 8
        z = rng.normal(size=n_cells).astype(np.float32)
        counts = rng.poisson(4.0, size=(n_cells, n_genes)).astype(np.float32)
        train_z = z[:20]
        train_counts = counts[:20]
        direct = fit_poisson_glm_irls(train_z, train_counts, "quadratic")
        via_predict = fit_poisson_glm_irls(
            train_z, train_counts, "quadratic", predict_latent=train_z
        )
        np.testing.assert_allclose(direct, via_predict, rtol=1e-5, atol=1e-5)
        all_preds = fit_poisson_glm_irls(
            train_z, train_counts, "quadratic", predict_latent=z
        )
        self.assertEqual(all_preds.shape, counts.shape)
        np.testing.assert_allclose(direct, all_preds[:20], rtol=1e-5, atol=1e-5)

    def test_parallel_poisson_quadratic_uses_irls_metrics(self) -> None:
        rng = np.random.default_rng(1)
        n_cells, n_genes = 48, 10
        S = rng.uniform(size=(n_cells, 2)).astype(np.float32)
        counts = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
        config = TestConfig(
            metric="nll_poisson_mse",
            decoder="quadratic",
            n_perms=3,
            n_reruns=2,
            epochs=25,
            lr=0.01,
            seed=0,
            device="cpu",
            verbose=False,
        )
        _, outputs, _ = train_parallel_isodepth_model(
            S, counts, config, model_label="test poisson irls refit"
        )
        stat_true = float(outputs.stat_true)
        self.assertTrue(np.isfinite(stat_true))
        null_floor = compute_metric(
            "nll_poisson_mse",
            counts,
            np.full_like(counts, np.log(max(float(counts.mean()), 1e-3))),
        )
        self.assertLess(stat_true, null_floor * 1.5)


if __name__ == "__main__":
    unittest.main()
