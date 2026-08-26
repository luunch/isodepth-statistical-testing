from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import CovariateConfig, DatasetBundle, TestConfig
from data.transforms import zscore_covariate
from methods.architectures import MidlineLatent, ParallelIsoDepthNet
from methods.permutation import run_parallel_permutation_method
from methods.trainers import resolve_device, train_batched_isodepth_model


class TestMidlineLatent(unittest.TestCase):
    def test_median_depth_per_row(self) -> None:
        s = torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [10.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        m = MidlineLatent()
        z = m(s)
        x0 = s[0, :, 0]
        med0 = x0.median()
        depth0 = (x0 - med0).abs()
        expected0 = ((depth0 - depth0.mean()) / depth0.std(unbiased=False)).unsqueeze(-1)
        self.assertTrue(torch.allclose(z[0], expected0))
        x1 = s[1, :, 0]
        med1 = x1.median()
        depth1 = (x1 - med1).abs()
        expected1 = ((depth1 - depth1.mean()) / depth1.std(unbiased=False)).unsqueeze(-1)
        self.assertTrue(torch.allclose(z[1], expected1))


class TestMidlineTraining(unittest.TestCase):
    def test_batched_hybrid_true_midline_perm_full_encoder(self) -> None:
        rng = np.random.default_rng(0)
        s_batched = rng.normal(size=(2, 8, 2)).astype(np.float32) * 0.2
        a = rng.normal(size=(8, 4)).astype(np.float32) * 0.1
        cfg = TestConfig(
            method="parallel_permutation",
            metric="mse",
            n_perms=1,
            n_reruns=1,
            epochs=2,
            patience=5,
            verbose=False,
            device="cpu",
            decoder="linear",
            covariate=CovariateConfig(type="midline"),
        )
        model, outputs = train_batched_isodepth_model(s_batched, a, cfg, latent_dim=1)
        pred = outputs.pred_true
        self.assertEqual(pred.shape, (8, 4))
        self.assertIsNotNone(model.encoder.encoder_perm)
        dev = resolve_device("cpu")
        s_t = torch.tensor(s_batched, dtype=torch.float32, device=dev)
        with torch.no_grad():
            d_b = model.encoder(s_t).detach().cpu().numpy()
        self.assertEqual(d_b.shape, (2, 8, 1))
        x = s_t[0, :, 0]
        med = x.median()
        depth = (x - med).abs()
        expected0 = ((depth - depth.mean()) / depth.std(unbiased=False)).unsqueeze(-1).cpu().numpy()
        np.testing.assert_allclose(d_b[0], expected0, rtol=0, atol=1e-5)

    def test_parallel_permutation_covariate_trains_full_batch_then_covariate_decoder(self) -> None:
        rng = np.random.default_rng(1)
        s = rng.normal(size=(10, 2)).astype(np.float32)
        a = rng.normal(size=(10, 3)).astype(np.float32)
        dataset = DatasetBundle(S=s, A=a).validate()
        cfg = TestConfig(
            method="parallel_permutation",
            metric="mse",
            n_perms=1,
            n_reruns=1,
            epochs=2,
            patience=5,
            verbose=False,
            device="cpu",
            decoder="linear",
            covariate=CovariateConfig(type="midline"),
        ).validate()

        result = run_parallel_permutation_method(dataset, cfg, device=resolve_device("cpu"))

        self.assertEqual(result.method_name, "parallel_permutation")
        self.assertIsInstance(result.artifacts["model"], ParallelIsoDepthNet)
        self.assertEqual(result.stat_perm.shape, (1,))
        self.assertIn("stat_covariate", result.artifacts)
        self.assertIn("p_value_covariate", result.artifacts)
        self.assertIn("pred_true_covariate", result.artifacts)
        self.assertIn("true_isodepth_covariate", result.artifacts)
        self.assertNotIn("pred_true_full_iso", result.artifacts)
        self.assertNotIn("true_isodepth_full_iso", result.artifacts)

        x_t = torch.tensor(s[:, 0], dtype=torch.float32)
        depth = (x_t - x_t.median()).abs().numpy()
        expected_midline = zscore_covariate(depth)
        np.testing.assert_allclose(result.artifacts["true_isodepth_covariate"], expected_midline, rtol=0, atol=1e-5)
