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

from data.schemas import DatasetBundle, TestConfig
from experiments.configuration import save_standardized_outputs
from data.schemas import DataConfig, OutputConfig, RunConfig


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from methods.perturbation import (
        normalize_depth,
        perturb_coordinates,
        run_comparison_perturbation_test,
        run_perturbation_test,
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
            method="comparison_perturbation_test",
            metric="mse",
            n_perms=2,
            n_reruns=1,
            n_nulls=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=1,
            device="cpu",
            delta=[0.1],
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
            method="comparison_perturbation_test",
            metric="mse",
            n_perms=1,
            n_reruns=1,
            n_nulls=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=1,
            device="cpu",
            delta=[0.1],
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
    def test_perturbation_runs_and_returns_expected_artifacts(self) -> None:
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
            method="perturbation_test",
            metric="mse",
            n_perms=5,
            n_reruns=1,
            n_nulls=3,
            batch_size=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=3,
            device="cpu",
            delta=[0.05, 0.1],
            verbose=False,
        ).validate()

        result = run_perturbation_test(dataset, config)

        self.assertEqual(result.method_name, "perturbation_test")
        self.assertEqual(result.stat_perm.shape, (5,))
        self.assertIn("delta_summaries", result.artifacts)
        self.assertIn("primary_delta", result.artifacts)
        self.assertIn("observed_scores", result.artifacts)
        self.assertIn("rerun_summary", result.artifacts)
        self.assertEqual(result.artifacts["true_rerun_index"], 0)
        self.assertEqual(len(result.artifacts["observed_scores"]), 10)
        self.assertEqual(len(result.artifacts["delta_plot_rows"]), 2)
        for summary in result.artifacts["delta_summaries"].values():
            self.assertEqual(len(summary["null_distribution"]), 5)
            self.assertEqual(summary["observed_distribution"], [float(result.stat_true)])

        primary_key = f"{float(result.artifacts['primary_delta']):.6g}"
        primary_summary = result.artifacts["delta_summaries"][primary_key]
        np.testing.assert_allclose(result.stat_perm, np.asarray(primary_summary["null_distribution"], dtype=np.float64))
        self.assertAlmostEqual(float(result.stat_true), float(primary_summary["score_mean"]))
        self.assertAlmostEqual(float(result.p_value), float(primary_summary["p_value"]))

    def test_perturbation_repeated_runs_with_same_seed_match(self) -> None:
        s = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        depth = s[:, 0] + s[:, 1]
        a = np.stack([depth, depth**2], axis=1).astype(np.float32)
        dataset = DatasetBundle(S=s, A=a).validate()
        config = TestConfig(
            method="perturbation_test",
            metric="mse",
            n_perms=3,
            n_reruns=1,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=5,
            device="cpu",
            delta=[0.05],
            verbose=False,
        ).validate()

        result_a = run_perturbation_test(dataset, config)
        result_b = run_perturbation_test(dataset, config)
        np.testing.assert_allclose(result_a.stat_perm, result_b.stat_perm)
        np.testing.assert_allclose(result_a.artifacts["observed_scores"], result_b.artifacts["observed_scores"])

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
            method="comparison_perturbation_test",
            metric="mse",
            n_perms=5,
            n_reruns=1,
            n_nulls=3,
            batch_size=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=3,
            device="cpu",
            delta=[0.05, 0.1],
            verbose=False,
        ).validate()

        result = run_comparison_perturbation_test(dataset, config)

        self.assertEqual(result.method_name, "comparison_perturbation_test")
        self.assertEqual(result.stat_perm.shape, (3,))
        self.assertGreaterEqual(float(result.stat_true), 0.0)
        self.assertIn("observed_scores", result.artifacts)
        self.assertEqual(len(result.artifacts["observed_scores"]), 10)
        self.assertIn("perturbed_S", result.artifacts)
        self.assertIn("perturbed_isodepth", result.artifacts)
        self.assertIn("rerun_summary", result.artifacts)
        self.assertEqual(result.artifacts["true_rerun_index"], 0)
        self.assertIn("lowest_S", result.artifacts)
        self.assertIn("highest_S", result.artifacts)
        self.assertIn("delta_summaries", result.artifacts)
        self.assertIn("primary_delta", result.artifacts)
        self.assertIn("delta_plot_rows", result.artifacts)
        self.assertEqual(len(result.artifacts["delta_plot_rows"]), 2)
        self.assertEqual(result.artifacts["perturbed_S"].shape, s.shape)
        self.assertEqual(result.artifacts["perturbed_isodepth"].shape[0], s.shape[0])
        primary_key = f"{float(result.artifacts['primary_delta']):.6g}"
        primary_summary = result.artifacts["delta_summaries"][primary_key]
        np.testing.assert_allclose(result.stat_perm, np.asarray(primary_summary["null_distribution"], dtype=np.float64))
        self.assertAlmostEqual(float(result.stat_true), float(primary_summary["score_mean"]))
        self.assertAlmostEqual(float(result.p_value), float(primary_summary["p_value"]))
        for summary in result.artifacts["delta_summaries"].values():
            self.assertIn("p_value", summary)
            self.assertIn("null_distribution", summary)

    def test_standardized_outputs_include_delta_plot(self) -> None:
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
            method="comparison_perturbation_test",
            metric="mse",
            n_perms=3,
            n_reruns=1,
            n_nulls=2,
            batch_size=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=3,
            device="cpu",
            delta=[0.05, 0.1],
            verbose=False,
        ).validate()
        result = run_comparison_perturbation_test(dataset, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", n_cells=6, n_genes=2),
                test=config,
                output=OutputConfig(out_dir=tmpdir, run_name="comparison_perturbation_test"),
            ).validate()

            payload, result_path = save_standardized_outputs(dataset, result, run_config)

            self.assertTrue(result_path.exists())
            self.assertTrue((result_path.parent / "comparison_perturbation_test_dataset.png").exists())
            self.assertTrue((result_path.parent / "comparison_perturbation_test_isodepth.png").exists())
            self.assertTrue((result_path.parent / "comparison_perturbation_test_delta_pvalues.png").exists())
            self.assertFalse((result_path.parent / "delta_0.05_perm_stats.npy").exists())
            self.assertFalse((result_path.parent / "delta_0.1_perm_stats.npy").exists())
            self.assertEqual(payload["method_name"], "comparison_perturbation_test")

            with open(result_path, "r", encoding="utf-8") as handle:
                saved_payload = json.load(handle)
            self.assertIn("delta_summaries", saved_payload["artifacts"])
            self.assertEqual(saved_payload["config"]["test"]["n_reruns"], 1)
            self.assertIn("rerun_summary", saved_payload["artifacts"])

    def test_perturbation_standardized_outputs_include_delta_plot(self) -> None:
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
            method="perturbation_test",
            metric="mse",
            n_perms=3,
            n_reruns=1,
            n_nulls=2,
            batch_size=2,
            epochs=2,
            patience=2,
            lr=1e-2,
            seed=3,
            device="cpu",
            delta=[0.05, 0.1],
            verbose=False,
        ).validate()
        result = run_perturbation_test(dataset, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", n_cells=6, n_genes=2),
                test=config,
                output=OutputConfig(out_dir=tmpdir, run_name="perturbation_test"),
            ).validate()

            payload, result_path = save_standardized_outputs(dataset, result, run_config)

            self.assertTrue(result_path.exists())
            self.assertTrue((result_path.parent / "perturbation_test_dataset.png").exists())
            self.assertTrue((result_path.parent / "perturbation_test_isodepth.png").exists())
            self.assertTrue((result_path.parent / "perturbation_test_delta_pvalues.png").exists())
            self.assertFalse((result_path.parent / "delta_0.05_perm_stats.npy").exists())
            self.assertFalse((result_path.parent / "delta_0.1_perm_stats.npy").exists())
            self.assertEqual(payload["method_name"], "perturbation_test")
            self.assertEqual(payload["config"]["test"]["n_reruns"], 1)
            self.assertIn("rerun_summary", payload["artifacts"])


if __name__ == "__main__":
    unittest.main()
