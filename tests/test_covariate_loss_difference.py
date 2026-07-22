"""Tests for loss-difference covariate whitening."""

from __future__ import annotations

import unittest

import numpy as np

from data.schemas import CovariateWhiteningConfig, DatasetBundle, TestConfig, run_config_from_mapping
from methods.covariate_loss_difference import (
    dataset_uses_loss_difference_whitening,
    run_loss_difference_parallel_training,
)
from methods.trainers import BatchedTrainingOutputs, train_parallel_isodepth_model
from methods.trainers.gpu_selection import resolve_device


class LossDifferenceWhiteningTests(unittest.TestCase):
    def test_covariate_whitening_config_parsing(self) -> None:
        cfg = run_config_from_mapping(
            {
                "data": {
                    "source": "h5ad",
                    "h5ad": "data/h5ad/dummy.h5ad",
                    "covariate_whitening": "loss-difference",
                    "covariate_whitening_obs_key": "calicost_tumor_proportion",
                },
            }
        )
        self.assertIsNotNone(cfg.data.covariate_whitening)
        self.assertEqual(cfg.data.covariate_whitening.method, "loss-difference")
        self.assertTrue(cfg.data.covariate_whitening.is_loss_difference)

    def test_covariate_whitening_alias(self) -> None:
        whitening = CovariateWhiteningConfig(
            method="loss_difference",
            obs_key="tumor_prop",
        ).validate()
        self.assertEqual(whitening.method, "loss-difference")

    def test_parallel_training_with_fixed_covariate(self) -> None:
        rng = np.random.default_rng(0)
        n_cells = 60
        n_genes = 8
        covariate_values = rng.uniform(0.0, 1.0, size=n_cells).astype(np.float32)
        S = rng.normal(size=(n_cells, 2)).astype(np.float32)
        spatial_signal = (S[:, 0] - S[:, 0].mean())[:, None]
        A = (
            spatial_signal
            + 0.5 * covariate_values[:, None]
            + rng.normal(scale=0.05, size=(n_cells, n_genes))
        ).astype(np.float32)
        A = (A - A.mean(axis=0, keepdims=True)) / (A.std(axis=0, keepdims=True) + 1e-8)

        dataset = DatasetBundle(
            S=S,
            A=A,
            meta={
                "covariate_whitening_values": covariate_values,
                "covariate_whitening": {
                    "method": "loss-difference",
                    "obs_key": "tumor_prop",
                },
                "coordinate_standardization": "none",
            },
        ).validate()
        self.assertTrue(dataset_uses_loss_difference_whitening(dataset))

        config = TestConfig(
            metric="nll_gaussian_mse",
            decoder="linear",
            n_perms=5,
            epochs=20,
            n_reruns=2,
            seed=0,
            device="cpu",
            verbose=False,
        ).validate()

        device = resolve_device("cpu")
        model, outputs, s_batched_np, artifacts = run_loss_difference_parallel_training(
            train_parallel_isodepth_model,
            covariate_values=covariate_values,
            model_label="h(d, n)",
            train_kwargs={
                "S": S,
                "A": A,
                "config": config,
                "device": device,
            },
        )

        # outputs are the direct h(d,n) results — one metric per slot
        self.assertEqual(outputs.model_metrics.shape[0], config.n_perms + 1)
        self.assertTrue(np.all(np.isfinite(outputs.model_metrics)))
        self.assertIn("loss_difference_joint_metrics", artifacts)
        np.testing.assert_array_equal(
            artifacts["loss_difference_joint_metrics"],
            outputs.model_metrics,
        )

    def test_loss_difference_detects_spatial_signal(self) -> None:
        """When expression depends on spatial position beyond n, stat_true < permuted losses."""
        rng = np.random.default_rng(42)
        n_cells = 80
        n_genes = 10
        covariate_values = rng.uniform(0.0, 1.0, size=n_cells).astype(np.float32)
        S = rng.uniform(size=(n_cells, 2)).astype(np.float32)
        # Strong spatial signal, weak covariate effect
        A = (
            3.0 * S[:, 0:1]
            + 0.1 * covariate_values[:, None]
            + rng.normal(scale=0.05, size=(n_cells, n_genes))
        ).astype(np.float32)
        A = (A - A.mean(axis=0, keepdims=True)) / (A.std(axis=0, keepdims=True) + 1e-8)

        config = TestConfig(
            metric="nll_gaussian_mse",
            decoder="linear",
            n_perms=19,
            epochs=50,
            n_reruns=1,
            seed=0,
            device="cpu",
            verbose=False,
        ).validate()

        device = resolve_device("cpu")
        _, outputs, _, _ = run_loss_difference_parallel_training(
            train_parallel_isodepth_model,
            covariate_values=covariate_values,
            model_label="h(d, n)",
            train_kwargs={
                "S": S,
                "A": A,
                "config": config,
                "device": device,
            },
        )

        # True slot should have a lower (better) loss than most/all permuted slots
        self.assertLess(float(outputs.stat_true), float(np.median(outputs.stat_perm)))


if __name__ == "__main__":
    unittest.main()
