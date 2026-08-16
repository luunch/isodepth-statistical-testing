from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import TestConfig, run_config_from_mapping
from data.transforms import gaussian_log_cpm_targets_from_counts


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from methods.trainers import get_training_metadata, train_parallel_isodepth_model


class TestGaussianPoissonPretrainSchema(unittest.TestCase):
    def test_valid_gaussian_pretrain_config(self) -> None:
        config = TestConfig(
            metric="nll_poisson_mse",
            n_perms=3,
            n_reruns=2,
            epochs=10,
            sgd_batch_size=8,
            gaussian_pretrain_epochs=3,
            gaussian_pretrain_freeze_encoder=False,
        )
        self.assertIs(config.validate(), config)

    def test_gaussian_pretrain_requires_poisson_metric(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                metric="nll_gaussian_mse",
                gaussian_pretrain_epochs=5,
                sgd_batch_size=8,
                epochs=10,
            ).validate()

    def test_gaussian_pretrain_requires_minibatch_sgd(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                metric="nll_poisson_mse",
                gaussian_pretrain_epochs=5,
                epochs=10,
            ).validate()

    def test_gaussian_pretrain_epochs_must_be_less_than_total_epochs(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                metric="nll_poisson_mse",
                gaussian_pretrain_epochs=10,
                epochs=10,
                sgd_batch_size=8,
            ).validate()

    def test_run_config_from_mapping_accepts_gaussian_pretrain(self) -> None:
        run_config = run_config_from_mapping(
            {
                "data": {"source": "synthetic", "n_cells": 16, "n_genes": 4},
                "test": {
                    "metric": "nll_poisson_mse",
                    "n_perms": 5,
                    "epochs": 20,
                    "sgd_batch_size": 16,
                    "gaussian_pretrain_epochs": 5,
                    "gaussian_pretrain_freeze_encoder": True,
                },
            }
        )
        self.assertEqual(run_config.test.gaussian_pretrain_epochs, 5)
        self.assertTrue(run_config.test.gaussian_pretrain_freeze_encoder)


class TestGaussianLogCpmTargets(unittest.TestCase):
    def test_log_cpm_targets_are_z_scored(self) -> None:
        counts = np.asarray(
            [
                [10.0, 0.0, 5.0],
                [20.0, 2.0, 8.0],
                [0.0, 1.0, 4.0],
            ],
            dtype=np.float32,
        )
        targets = gaussian_log_cpm_targets_from_counts(counts)
        self.assertEqual(targets.shape, counts.shape)
        np.testing.assert_allclose(targets.mean(axis=0), np.zeros(3), atol=1e-5)
        self.assertTrue(np.all(np.std(targets, axis=0) > 0.5))


@unittest.skipUnless(HAS_TORCH, "torch is required for gaussian pretrain trainer tests")
class TestGaussianPoissonPretrainTrainer(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.n_cells, self.n_genes = 40, 8
        self.s = rng.uniform(size=(self.n_cells, 2)).astype(np.float32)
        self.counts = rng.poisson(4.0, size=(self.n_cells, self.n_genes)).astype(np.float32)

    def _base_config(self, **overrides: object) -> TestConfig:
        params = {
            "metric": "nll_poisson_mse",
            "decoder": "nn",
            "n_perms": 2,
            "n_reruns": 2,
            "epochs": 6,
            "sgd_batch_size": 8,
            "gaussian_pretrain_epochs": 2,
            "lr": 1e-2,
            "seed": 0,
            "device": "cpu",
            "verbose": False,
        }
        params.update(overrides)
        return TestConfig(**params).validate()

    def test_warm_then_poisson_training_completes(self) -> None:
        config = self._base_config(gaussian_pretrain_freeze_encoder=False)
        model, outputs, _ = train_parallel_isodepth_model(
            self.s, self.counts, config, model_label="test gaussian pretrain free"
        )
        metadata = get_training_metadata(model)
        self.assertEqual(metadata["gaussian_pretrain_epochs"], 2)
        self.assertFalse(metadata["gaussian_pretrain_freeze_encoder"])
        self.assertTrue(np.isfinite(float(outputs.stat_true)))

    def test_warm_then_frozen_encoder_training_completes(self) -> None:
        config = self._base_config(gaussian_pretrain_freeze_encoder=True)
        model, outputs, _ = train_parallel_isodepth_model(
            self.s, self.counts, config, model_label="test gaussian pretrain frozen"
        )
        metadata = get_training_metadata(model)
        self.assertTrue(metadata["gaussian_pretrain_freeze_encoder"])
        self.assertTrue(np.isfinite(float(outputs.stat_true)))


if __name__ == "__main__":
    unittest.main()
