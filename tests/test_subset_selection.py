from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DataConfig, DatasetBundle, OutputConfig, RunConfig, TestConfig
from experiments.configuration import save_standardized_outputs


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from methods.subsampling import (
        build_subset_masks,
        compute_masked_losses,
        run_comparison_subsampling_test,
        run_subsampling_test,
    )
    from methods.trainers import train_batched_isodepth_model


@unittest.skipUnless(HAS_TORCH, "torch is required for subset selection tests")
class TestSubsetSelectionHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.s = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        depth = self.s[:, 0] + self.s[:, 1]
        self.a = np.stack([depth, depth**2], axis=1).astype(np.float32)

    def test_build_subset_masks_is_reproducible(self) -> None:
        mask_a, fractions_a, sizes_a = build_subset_masks(8, [0.5, 0.75], 2, seed=7)
        mask_b, fractions_b, sizes_b = build_subset_masks(8, [0.5, 0.75], 2, seed=7)
        np.testing.assert_allclose(mask_a, mask_b)
        np.testing.assert_allclose(fractions_a, fractions_b)
        np.testing.assert_array_equal(sizes_a, sizes_b)

    def test_compute_masked_losses_accepts_broadcast_and_expanded_masks(self) -> None:
        predictions = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        targets = np.asarray([[[0.0, 1.0], [1.0, 2.0]]], dtype=np.float32)
        mask_compact = np.asarray([[[1.0], [0.0]]], dtype=np.float32)
        mask_expanded = np.asarray([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)

        compact_loss = compute_masked_losses(predictions, targets, mask_compact)
        expanded_loss = compute_masked_losses(predictions, targets, mask_expanded)

        np.testing.assert_allclose(compact_loss, expanded_loss)
        self.assertAlmostEqual(float(compact_loss[0]), 1.0)

    def test_compute_masked_losses_rejects_empty_masks(self) -> None:
        predictions = np.zeros((1, 2, 1), dtype=np.float32)
        targets = np.zeros((1, 2, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_masked_losses(predictions, targets, np.zeros((1, 2, 1), dtype=np.float32))

    def test_batched_trainer_accepts_loss_masks(self) -> None:
        s_batched = np.repeat(self.s[None, :, :], 2, axis=0)
        a_batched = np.stack([self.a, self.a[::-1]], axis=0).astype(np.float32)
        loss_mask_batched = np.asarray(
            [
                [[1.0], [1.0], [0.0], [0.0]],
                [[0.0], [0.0], [1.0], [1.0]],
            ],
            dtype=np.float32,
        )
        config = TestConfig(
            method="comparison_subsampling_test",
            metric="mse",
            n_perms=0,
            n_nulls=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=1,
            device="cpu",
            verbose=False,
        ).validate()

        _, predictions = train_batched_isodepth_model(
            s_batched,
            self.a,
            config,
            a_batched=a_batched,
            loss_mask_batched=loss_mask_batched,
            model_label="test subset masked trainer",
        )

        self.assertEqual(predictions.shape, (2, self.s.shape[0], self.a.shape[1]))


@unittest.skipUnless(HAS_TORCH, "torch is required for subset selection tests")
class TestSubsetSelectionIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.s = np.asarray(
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
        depth = self.s[:, 0] + self.s[:, 1]
        self.a = np.stack([depth, depth**2], axis=1).astype(np.float32)
        self.dataset = DatasetBundle(S=self.s, A=self.a).validate()

    def _config(self) -> TestConfig:
        return TestConfig(
            method="comparison_subsampling_test",
            metric="mse",
            n_perms=0,
            n_nulls=3,
            batch_size=2,
            subset_fractions=[0.5, 0.75],
            n_subsets=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=3,
            device="cpu",
            verbose=False,
        ).validate()

    def _direct_config(self) -> TestConfig:
        return TestConfig(
            method="subsampling_test",
            metric="mse",
            n_perms=3,
            n_nulls=3,
            batch_size=2,
            subset_fractions=[0.5, 0.75],
            n_subsets=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=3,
            device="cpu",
            verbose=False,
        ).validate()

    def test_direct_runs_and_returns_expected_artifacts(self) -> None:
        result = run_subsampling_test(self.dataset, self._direct_config())

        self.assertEqual(result.method_name, "subsampling_test")
        self.assertEqual(result.stat_perm.shape, (2,))
        self.assertIn("observed_scores", result.artifacts)
        self.assertIn("observed_correlations", result.artifacts)
        self.assertIn("fraction_summaries", result.artifacts)
        self.assertIn("primary_fraction", result.artifacts)
        self.assertIn("fraction_plot_rows", result.artifacts)
        self.assertEqual(len(result.artifacts["observed_scores"]), 4)
        self.assertEqual(len(result.artifacts["observed_correlations"]), 4)
        self.assertEqual(len(result.artifacts["fraction_plot_rows"]), 2)
        self.assertEqual(result.artifacts["summary_statistic"], "scaled_subset_reconstruction_loss")

        fractions = np.asarray(result.artifacts["subset_fraction_per_subset"], dtype=np.float64)
        primary_key = f"{float(result.artifacts['primary_fraction']):.3f}"
        primary_summary = result.artifacts["fraction_summaries"][primary_key]
        primary_mask = np.isclose(fractions, float(result.artifacts["primary_fraction"]))
        expected_scaled = np.asarray(result.artifacts["observed_scores"], dtype=np.float64)[primary_mask]
        np.testing.assert_allclose(result.stat_perm, np.asarray(primary_summary["null_distribution"], dtype=np.float64))
        np.testing.assert_allclose(result.stat_perm, expected_scaled)
        self.assertAlmostEqual(float(result.stat_true), float(primary_summary["loss_mean"]))
        self.assertAlmostEqual(float(result.p_value), float(primary_summary["p_value"]))

    def test_direct_repeated_runs_with_same_seed_match(self) -> None:
        result_a = run_subsampling_test(self.dataset, self._direct_config())
        result_b = run_subsampling_test(self.dataset, self._direct_config())

        np.testing.assert_allclose(result_a.stat_perm, result_b.stat_perm)
        np.testing.assert_allclose(result_a.artifacts["observed_scores"], result_b.artifacts["observed_scores"])
        np.testing.assert_allclose(
            result_a.artifacts["lowest_subset_mask"],
            result_b.artifacts["lowest_subset_mask"],
        )

    def test_runs_and_returns_expected_artifacts(self) -> None:
        result = run_comparison_subsampling_test(self.dataset, self._config())

        self.assertEqual(result.method_name, "comparison_subsampling_test")
        self.assertEqual(result.stat_perm.shape, (3,))
        self.assertIn("observed_scores", result.artifacts)
        self.assertIn("observed_correlations", result.artifacts)
        self.assertIn("fraction_summaries", result.artifacts)
        self.assertIn("primary_fraction", result.artifacts)
        self.assertIn("fraction_plot_rows", result.artifacts)
        self.assertIn("lowest_subset_mask", result.artifacts)
        self.assertIn("highest_subset_mask", result.artifacts)
        self.assertEqual(len(result.artifacts["observed_scores"]), 4)
        self.assertEqual(len(result.artifacts["observed_correlations"]), 4)
        self.assertEqual(len(result.artifacts["fraction_plot_rows"]), 2)
        primary_key = f"{float(result.artifacts['primary_fraction']):.3f}"
        primary_summary = result.artifacts["fraction_summaries"][primary_key]
        np.testing.assert_allclose(result.stat_perm, np.asarray(primary_summary["null_distribution"], dtype=np.float64))
        self.assertAlmostEqual(float(result.stat_true), float(primary_summary["loss_mean"]))
        self.assertAlmostEqual(float(result.p_value), float(primary_summary["p_value"]))
        for summary in result.artifacts["fraction_summaries"].values():
            self.assertIn("p_value", summary)
            self.assertIn("null_distribution", summary)

    def test_repeated_runs_with_same_seed_match(self) -> None:
        result_a = run_comparison_subsampling_test(self.dataset, self._config())
        result_b = run_comparison_subsampling_test(self.dataset, self._config())

        np.testing.assert_allclose(result_a.stat_perm, result_b.stat_perm)
        np.testing.assert_allclose(result_a.artifacts["observed_scores"], result_b.artifacts["observed_scores"])
        np.testing.assert_allclose(
            result_a.artifacts["lowest_subset_mask"],
            result_b.artifacts["lowest_subset_mask"],
        )

    def test_standardized_outputs_are_saved(self) -> None:
        result = run_comparison_subsampling_test(self.dataset, self._config())
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", n_cells=6, n_genes=2),
                test=self._config(),
                output=OutputConfig(out_dir=tmpdir, run_name="comparison_subsampling_test"),
            ).validate()

            payload, result_path = save_standardized_outputs(self.dataset, result, run_config)

            self.assertTrue(result_path.exists())
            self.assertTrue((result_path.parent / "comparison_subsampling_test_dataset.png").exists())
            self.assertTrue((result_path.parent / "observed_scores.npy").exists())
            self.assertTrue((result_path.parent / "observed_correlations.npy").exists())
            self.assertTrue((result_path.parent / "comparison_subsampling_test_isodepth.png").exists())
            self.assertTrue((result_path.parent / "comparison_subsampling_test_subset_fraction_pvalues.png").exists())
            self.assertTrue((result_path.parent / "subset_fraction_0.500_perm_stats.npy").exists())
            self.assertTrue((result_path.parent / "subset_fraction_0.750_perm_stats.npy").exists())
            self.assertEqual(payload["method_name"], "comparison_subsampling_test")

            with open(result_path, "r", encoding="utf-8") as handle:
                saved_payload = json.load(handle)
            self.assertIn("fraction_summaries", saved_payload["artifacts"])

    def test_direct_standardized_outputs_are_saved(self) -> None:
        result = run_subsampling_test(self.dataset, self._direct_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", n_cells=6, n_genes=2),
                test=self._direct_config(),
                output=OutputConfig(out_dir=tmpdir, run_name="subsampling_test"),
            ).validate()

            payload, result_path = save_standardized_outputs(self.dataset, result, run_config)

            self.assertTrue(result_path.exists())
            self.assertTrue((result_path.parent / "subsampling_test_dataset.png").exists())
            self.assertTrue((result_path.parent / "observed_scores.npy").exists())
            self.assertTrue((result_path.parent / "observed_correlations.npy").exists())
            self.assertTrue((result_path.parent / "subsampling_test_isodepth.png").exists())
            self.assertTrue((result_path.parent / "subsampling_test_subset_fraction_pvalues.png").exists())
            self.assertTrue((result_path.parent / "subset_fraction_0.500_perm_stats.npy").exists())
            self.assertTrue((result_path.parent / "subset_fraction_0.750_perm_stats.npy").exists())
            self.assertEqual(payload["method_name"], "subsampling_test")


if __name__ == "__main__":
    unittest.main()
