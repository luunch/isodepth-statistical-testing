from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DatasetBundle, TestConfig


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from methods.perturbation import (
        normalize_depth,
        perturb_coordinates,
        run_perturbation_robustness_method,
        score_depth_similarity,
    )
    from methods.trainers import train_parallel_isodepth_model


@unittest.skipUnless(HAS_TORCH, "torch is required for perturbation method tests")
class TestPerturbationHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.S = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

    def test_coordinate_perturbation_is_reproducible_and_clipped(self) -> None:
        perturbed_a = perturb_coordinates(self.S, delta=0.5, seed=7)
        perturbed_b = perturb_coordinates(self.S, delta=0.5, seed=7)
        np.testing.assert_allclose(perturbed_a, perturbed_b)
        self.assertTrue(np.all(perturbed_a >= self.S.min(axis=0, keepdims=True)))
        self.assertTrue(np.all(perturbed_a <= self.S.max(axis=0, keepdims=True)))

    def test_normalize_depth_maps_range_to_unit_interval(self) -> None:
        normalized = normalize_depth(np.asarray([2.0, 4.0, 6.0], dtype=np.float32))
        np.testing.assert_allclose(normalized, np.asarray([0.0, 0.5, 1.0], dtype=np.float32))

    def test_correlation_scores_are_absolute(self) -> None:
        d_true = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        d_flip = np.asarray([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
        score = score_depth_similarity("spearman_corr_mean", d_true, d_flip)
        self.assertAlmostEqual(score, 1.0)

    def test_loss_scores_use_normalized_depths(self) -> None:
        d_true = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        d_scaled = np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
        score = score_depth_similarity("mse", d_true, d_scaled)
        self.assertAlmostEqual(score, 0.0)

    def test_batched_trainer_accepts_arbitrary_s_batched(self) -> None:
        a = np.stack([self.S[:, 0], self.S[:, 1]], axis=1).astype(np.float32)
        s_batched = np.stack(
            [
                self.S,
                perturb_coordinates(self.S, delta=0.1, seed=1),
                perturb_coordinates(self.S, delta=0.1, seed=2),
            ],
            axis=0,
        )
        config = TestConfig(
            method="perturbation_robustness",
            metric="mse",
            n_perms=2,
            n_nulls=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=1,
            device="cpu",
            delta=0.1,
            verbose=False,
        ).validate()
        _, predictions = train_parallel_isodepth_model(
            self.S,
            a,
            config,
            device=None,
            s_batched=s_batched,
            model_label="test batched trainer",
        )
        self.assertEqual(predictions.shape, (3, self.S.shape[0], a.shape[1]))

    def test_batched_trainer_accepts_arbitrary_a_batched(self) -> None:
        a = np.stack([self.S[:, 0], self.S[:, 1]], axis=1).astype(np.float32)
        s_batched = np.stack(
            [
                self.S,
                perturb_coordinates(self.S, delta=0.1, seed=1),
            ],
            axis=0,
        )
        a_batched = np.stack(
            [
                a,
                a[::-1],
            ],
            axis=0,
        ).astype(np.float32)
        config = TestConfig(
            method="perturbation_robustness",
            metric="mse",
            n_perms=1,
            n_nulls=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=1,
            device="cpu",
            delta=0.1,
            verbose=False,
        ).validate()
        _, predictions = train_parallel_isodepth_model(
            self.S,
            a,
            config,
            device=None,
            s_batched=s_batched,
            a_batched=a_batched,
            model_label="test batched trainer with a_batched",
        )
        self.assertEqual(predictions.shape, (2, self.S.shape[0], a.shape[1]))


@unittest.skipUnless(HAS_TORCH, "torch is required for perturbation method tests")
class TestPerturbationMethodIntegration(unittest.TestCase):
    def test_runs_and_returns_expected_artifacts(self) -> None:
        s = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.25],
                [0.25, 0.5],
            ],
            dtype=np.float32,
        )
        depth = s[:, 0] + s[:, 1]
        a = np.stack([depth, depth**2], axis=1).astype(np.float32)
        dataset = DatasetBundle(S=s, A=a).validate()
        config = TestConfig(
            method="perturbation_robustness",
            metric="mse",
            n_perms=5,
            n_nulls=3,
            batch_size=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=3,
            device="cpu",
            delta=0.05,
            verbose=False,
        ).validate()

        result = run_perturbation_robustness_method(dataset, config)

        self.assertEqual(result.method_name, "perturbation_robustness")
        self.assertEqual(result.stat_perm.shape, (3,))
        self.assertGreaterEqual(float(result.stat_true), 0.0)
        self.assertIn("observed_scores", result.artifacts)
        self.assertEqual(len(result.artifacts["observed_scores"]), 5)
        self.assertIn("perturbed_S", result.artifacts)
        self.assertIn("perturbed_isodepth", result.artifacts)
        self.assertIn("lowest_S", result.artifacts)
        self.assertIn("highest_S", result.artifacts)
        self.assertEqual(result.artifacts["perturbed_S"].shape, s.shape)
        self.assertEqual(result.artifacts["perturbed_isodepth"].shape[0], s.shape[0])


if __name__ == "__main__":
    unittest.main()
