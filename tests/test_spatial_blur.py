"""Tests for Gaussian expression blur and metric-distribution overlays."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.plots import save_metric_distribution_plot
from data.schemas import TestResult
from methods.spatial_blur import (
    blur_expression_slots,
    blur_expression_slots_within_groups,
    gaussian_blur_expression,
)


class GaussianBlurExpressionTests(unittest.TestCase):
    def test_tiny_bandwidth_is_near_identity(self) -> None:
        S = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]], dtype=np.float64)
        A = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=np.float64)
        blurred = gaussian_blur_expression(S, A, sigma_um=0.01, cutoff_mult=4.0)
        np.testing.assert_allclose(blurred, A, atol=1e-6)

    def test_close_neighbors_mix_expression(self) -> None:
        S = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
        A = np.array([[1.0], [0.0]], dtype=np.float64)
        blurred = gaussian_blur_expression(S, A, sigma_um=5.0, cutoff_mult=4.0)
        self.assertGreater(float(blurred[0, 0]), 0.5)
        self.assertLess(float(blurred[0, 0]), 1.0)
        self.assertGreater(float(blurred[1, 0]), 0.0)
        self.assertLess(float(blurred[1, 0]), 0.5)

    def test_blur_expression_slots_matches_per_slot_call(self) -> None:
        rng = np.random.default_rng(0)
        S0 = rng.normal(size=(8, 2))
        S1 = rng.normal(size=(8, 2))
        A = rng.normal(size=(8, 3))
        batched = blur_expression_slots(np.stack([S0, S1], axis=0), A, sigma_um=2.0)
        expected = np.stack(
            [
                gaussian_blur_expression(S0, A, 2.0),
                gaussian_blur_expression(S1, A, 2.0),
            ],
            axis=0,
        )
        np.testing.assert_allclose(batched, expected, atol=1e-6)

    def test_within_group_blur_does_not_mix_types(self) -> None:
        S = np.array(
            [[[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [101.0, 0.0]]],
            dtype=np.float64,
        )
        A = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        mixed = blur_expression_slots(S, A, sigma_um=50.0, cutoff_mult=4.0)
        isolated = blur_expression_slots_within_groups(
            S, A, sigma_um=50.0, cutoff_mult=4.0, group_labels=labels
        )
        self.assertGreater(float(isolated[0, 0, 0]), float(mixed[0, 0, 0]))
        np.testing.assert_allclose(
            isolated[0, :2, 0],
            blur_expression_slots(S[:, :2, :], A[:2], 50.0)[0, :, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            isolated[0, 2:, 0],
            blur_expression_slots(S[:, 2:, :], A[2:], 50.0)[0, :, 0],
            atol=1e-6,
        )


class MetricDistributionOverlayTests(unittest.TestCase):
    def test_overlay_writes_plot(self) -> None:
        result = TestResult(
            method_name="parallel_permutation",
            metric="nll_gaussian_mse",
            p_value=0.02,
            stat_true=10.0,
            stat_perm=np.linspace(11.0, 13.0, 40),
            runtime_sec=0.0,
            n_cells=10,
            n_genes=4,
            config={},
        ).validate()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "metric_distribution.png"
            saved = save_metric_distribution_plot(
                result,
                out_path,
                overlays=[
                    (np.array([9.5, 9.7, 9.6]), "20 µm blurred perms", "darkorange"),
                    (np.array([10.1, 10.2]), "50 µm blurred perms", "mediumorchid"),
                ],
            )
            self.assertTrue(saved.is_file())
            self.assertGreater(saved.stat().st_size, 0)

    def test_combined_plot_draws_per_type_blur_overlays(self) -> None:
        from analysis.plots import save_combined_celltype_metric_distribution

        per_type = {
            "CA1": {
                "stat_perm": np.linspace(1.00, 1.02, 40),
                "stat_true": 0.99,
                "p_value": 0.02,
                "n_cells": 12,
                "stat_covariate": 0.995,
                "p_value_covariate": 0.02,
            },
            "CA3": {
                "stat_perm": np.linspace(1.01, 1.03, 40),
                "stat_true": 1.00,
                "p_value": 0.04,
                "n_cells": 9,
            },
        }
        overlays = {
            "CA1": [
                (np.array([0.991, 0.992]), "10 µm blurred perms", "seagreen"),
                (np.array([0.996, 0.997]), "20 µm blurred perms", "darkorange"),
                (np.array([1.000, 1.001]), "50 µm blurred perms", "mediumorchid"),
            ],
            "CA3": [
                (np.array([1.005, 1.006]), "10 µm blurred perms", "seagreen"),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "combined_metric_distribution.png"
            saved = save_combined_celltype_metric_distribution(
                per_type,
                ["CA1", "CA3"],
                out_path,
                covariate_label="Coordinate",
                overlays_by_type=overlays,
            )
            self.assertTrue(saved.is_file())
            self.assertGreater(saved.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
