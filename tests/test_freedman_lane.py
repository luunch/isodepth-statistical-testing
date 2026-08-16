"""Tests for Freedman–Lane covariate whitening."""

from __future__ import annotations

import unittest

import numpy as np

from data.schemas import (
    CovariateWhiteningConfig,
    DatasetBundle,
    DataConfig,
    TestConfig,
    run_config_from_mapping,
)
from methods.freedman_lane import apply_freedman_lane_whitening


class FreedmanLaneWhiteningTests(unittest.TestCase):
    def test_whitening_residualizes_and_restandardizes(self) -> None:
        rng = np.random.default_rng(0)
        n_cells = 80
        n_genes = 12
        covariate_values = rng.uniform(0.0, 1.0, size=n_cells).astype(np.float32)
        S = rng.normal(size=(n_cells, 2)).astype(np.float32)
        A = (
            0.8 * covariate_values[:, None]
            + rng.normal(scale=0.05, size=(n_cells, n_genes))
        ).astype(np.float32)
        A = (A - A.mean(axis=0, keepdims=True)) / (A.std(axis=0, keepdims=True) + 1e-8)

        dataset = DatasetBundle(
            S=S,
            A=A,
            meta={
                "covariate_whitening_values": covariate_values,
                "covariate_whitening": {
                    "method": "freedman-lane",
                    "obs_key": "calicost_tumor_proportion",
                },
                "coordinate_standardization": "none",
            },
        ).validate()

        config = TestConfig(
            metric="nll_gaussian_mse",
            decoder="linear",
            epochs=10,
            n_reruns=2,
            seed=0,
            device="cpu",
            verbose=False,
        ).validate()

        whitened_dataset, artifacts = apply_freedman_lane_whitening(dataset, config)

        pred = np.asarray(artifacts["freedman_lane_pred"], dtype=np.float32)
        self.assertEqual(pred.shape, A.shape)
        whitened = whitened_dataset.A
        np.testing.assert_allclose(whitened.mean(axis=0), np.zeros(n_genes), atol=1e-5)
        np.testing.assert_allclose(whitened.std(axis=0), np.ones(n_genes), atol=0.15)

        rho_before = np.corrcoef(covariate_values, A.mean(axis=1))[0, 1]
        rho_after = np.corrcoef(covariate_values, whitened.mean(axis=1))[0, 1]
        self.assertLess(abs(rho_after), abs(rho_before))

    def test_covariate_whitening_config_parsing(self) -> None:
        cfg = run_config_from_mapping(
            {
                "data": {
                    "source": "h5ad",
                    "h5ad": "data/h5ad/dummy.h5ad",
                    "covariate_whitening": "freedman-lane",
                    "covariate_whitening_obs_key": "calicost_tumor_proportion",
                },
            }
        )
        self.assertIsNotNone(cfg.data.covariate_whitening)
        self.assertEqual(cfg.data.covariate_whitening.method, "freedman-lane")
        self.assertEqual(cfg.data.covariate_whitening.obs_key, "calicost_tumor_proportion")

    def test_covariate_whitening_mapping_form(self) -> None:
        whitening = CovariateWhiteningConfig(
            method="freeman-lane",
            obs_key="tumor_prop",
        ).validate()
        self.assertEqual(whitening.method, "freedman-lane")


if __name__ == "__main__":
    unittest.main()
