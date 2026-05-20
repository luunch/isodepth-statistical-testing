from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DataConfig, OutputConfig, RunConfig, TestConfig, TestResult
from data.synthetic import SpatialDataSimulator, generate_synthetic_dataset
from experiments.configuration import save_standardized_outputs


class TestSyntheticGeneration(unittest.TestCase):
    def test_fourier_bounds_set_the_frequency_band(self) -> None:
        simulator = SpatialDataSimulator(N=16, G=2, sigma=0.0)
        x = simulator.S[:, 0]
        y = simulator.S[:, 1]

        with patch("data.synthetic.np.random.randn", return_value=np.ones(2, dtype=np.float64)):
            latent = simulator._generate_fourier_latent(2, 4, dependent_xy=True)

        expected_raw = np.zeros(simulator.N, dtype=np.float64)
        for k1 in (2, 3, 4):
            for k2 in (2, 3, 4):
                angle = 2.0 * np.pi * (k1 * x + k2 * y)
                expected_raw += np.cos(angle)
                expected_raw += np.sin(angle)

        expected = (expected_raw - expected_raw.min()) / (expected_raw.max() - expected_raw.min())
        np.testing.assert_allclose(latent, expected, atol=1e-7)

    def test_independent_fourier_basis_matches_separable_xy_terms(self) -> None:
        simulator = SpatialDataSimulator(N=16, G=2, sigma=0.0)
        x = simulator.S[:, 0]
        y = simulator.S[:, 1]

        with patch("data.synthetic.np.random.randn", return_value=np.ones(4, dtype=np.float64)):
            latent = simulator._generate_fourier_latent(2, 4, dependent_xy=False)

        expected_raw = np.zeros(simulator.N, dtype=np.float64)
        for frequency in (2, 3, 4):
            angle = 2.0 * np.pi * frequency
            expected_raw += np.sin(angle * x)
            expected_raw += np.cos(angle * x)
            expected_raw += np.sin(angle * y)
            expected_raw += np.cos(angle * y)

        expected = (expected_raw - expected_raw.min()) / (expected_raw.max() - expected_raw.min())
        np.testing.assert_allclose(latent, expected, atol=1e-7)

    def test_fourier_dataset_has_expected_shape_and_metadata(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="fourier",
                n_cells=16,
                n_genes=3,
                sigma=0.1,
                k_min=1,
                k_max=2,
                dependent_xy=True,
                seed=7,
                poly_degree=2,
            )
        )

        self.assertEqual(dataset.S.shape, (16, 2))
        self.assertEqual(dataset.A.shape, (16, 3))
        self.assertEqual(dataset.meta["mode"], "fourier")
        self.assertEqual(dataset.meta["k_min"], 1)
        self.assertEqual(dataset.meta["k_max"], 2)
        self.assertEqual(dataset.meta["poly_degree"], 2)
        self.assertTrue(dataset.meta["dependent_xy"])
        self.assertEqual(dataset.meta["fourier_basis"], "interaction_xy")
        self.assertEqual(np.asarray(dataset.meta["synthetic_true_curve"]).shape, (16,))
        self.assertEqual(dataset.meta["grid_height"], 4)
        self.assertEqual(dataset.meta["grid_width"], 4)

    def test_fourier_dataset_can_use_independent_xy_basis(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="fourier",
                n_cells=16,
                n_genes=3,
                sigma=0.1,
                k_min=1,
                k_max=2,
                dependent_xy=False,
                seed=7,
            )
        )

        self.assertFalse(dataset.meta["dependent_xy"])
        self.assertEqual(dataset.meta["fourier_basis"], "independent_xy")

    def test_noise_dataset_records_flat_true_curve(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="noise", n_cells=16, n_genes=3, sigma=0.1, seed=7)
        )

        np.testing.assert_allclose(
            np.asarray(dataset.meta["synthetic_true_curve"], dtype=np.float32),
            np.zeros(16, dtype=np.float32),
        )

    def test_noise_rectangular_grid_matches_side_length(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=24,
                n_genes=3,
                sigma=0.1,
                seed=1,
                side_length=6,
            )
        )
        self.assertEqual(dataset.S.shape, (24, 2))
        self.assertEqual(dataset.meta["grid_height"], 4)
        self.assertEqual(dataset.meta["grid_width"], 6)
        self.assertEqual(dataset.meta["side_length"], 6)
        self.assertEqual(dataset.meta["other_side_length"], 4)
        x_unique = len(np.unique(dataset.S[:, 0]))
        y_unique = len(np.unique(dataset.S[:, 1]))
        self.assertEqual(x_unique, 6)
        self.assertEqual(y_unique, 4)

    def test_noise_semicircle_lattice_cell_count_and_region(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=900,
                n_genes=3,
                sigma=0.1,
                seed=42,
                shape="semicircle",
            )
        )
        self.assertEqual(dataset.n_cells, 900)
        self.assertEqual(dataset.meta["shape"], "semicircle")
        self.assertIn("lattice_resolution", dataset.meta)
        S = np.asarray(dataset.S, dtype=np.float64)
        self.assertTrue(np.all(S[:, 1] >= -1e-5))
        self.assertTrue(np.all((S[:, 0] - 0.5) ** 2 + S[:, 1] ** 2 <= 0.25 + 1e-5))

    def test_noise_square_cutout_lattice_cell_count_and_region(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=900,
                n_genes=3,
                sigma=0.1,
                seed=42,
                shape="square_cutout",
            )
        )
        self.assertEqual(dataset.n_cells, 900)
        self.assertEqual(dataset.meta["shape"], "square_cutout")
        self.assertIn("lattice_resolution", dataset.meta)
        S = np.asarray(dataset.S, dtype=np.float64)
        self.assertTrue(np.all(S >= -1e-5))
        self.assertTrue(np.all(S <= 1.0 + 1e-5))
        in_removed = (S[:, 1] >= -1e-5) & ((S[:, 0] - 0.5) ** 2 + S[:, 1] ** 2 <= 0.25 + 1e-5)
        self.assertFalse(np.any(in_removed))

    def test_noise_uniform_sampling_bias_cell_count_and_region(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=900,
                n_genes=3,
                sigma=0.1,
                seed=42,
                shape="semicircle",
                sampling_bias={"type": "uniform"},
            )
        )
        self.assertEqual(dataset.n_cells, 900)
        self.assertEqual(dataset.meta["sampling_bias"], {"type": "uniform"})
        self.assertNotIn("lattice_resolution", dataset.meta)
        self.assertNotIn("side_length", dataset.meta)
        S = np.asarray(dataset.S, dtype=np.float64)
        self.assertTrue(np.all(S[:, 1] >= -1e-5))
        self.assertTrue(np.all((S[:, 0] - 0.5) ** 2 + S[:, 1] ** 2 <= 0.25 + 1e-5))

    def test_uniform_sampling_bias_ignores_side_length_in_meta(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=100,
                n_genes=2,
                shape="square",
                side_length=10,
                sampling_bias={"type": "uniform"},
                seed=0,
            )
        )
        self.assertNotIn("side_length", dataset.meta)
        self.assertEqual(dataset.n_cells, 100)

    def test_normal_sampling_bias_is_centered_and_in_shape(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=2000,
                n_genes=3,
                sigma=0.1,
                seed=7,
                shape="square",
                sampling_bias={"type": "normal", "variance": 0.05},
            )
        )
        self.assertEqual(dataset.n_cells, 2000)
        self.assertEqual(
            dataset.meta["sampling_bias"], {"type": "normal", "variance": 0.05}
        )
        S = np.asarray(dataset.S, dtype=np.float64)
        self.assertTrue(np.all(S >= -1e-5))
        self.assertTrue(np.all(S <= 1.0 + 1e-5))
        self.assertAlmostEqual(float(np.mean(S[:, 0])), 0.5, delta=0.05)
        self.assertAlmostEqual(float(np.mean(S[:, 1])), 0.5, delta=0.05)

    def test_normal_sampling_bias_respects_spatial_mask(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=600,
                n_genes=2,
                sigma=0.1,
                seed=3,
                shape="square_cutout",
                sampling_bias={"type": "normal", "variance": 0.2},
            )
        )
        S = np.asarray(dataset.S, dtype=np.float64)
        in_removed = (S[:, 1] >= -1e-5) & ((S[:, 0] - 0.5) ** 2 + S[:, 1] ** 2 <= 0.25 + 1e-5)
        self.assertFalse(np.any(in_removed))

    def test_expression_manifold_respects_configured_polynomial_degree(self) -> None:
        simulator = SpatialDataSimulator(N=16, G=2, sigma=0.0, poly_degree=1)
        latent = np.linspace(0.0, 1.0, simulator.N)

        with patch("data.synthetic.np.random.randn", return_value=np.array([2.0, -1.0], dtype=np.float64)):
            manifold = simulator._apply_expression_manifold(latent)

        expected = 2.0 * latent - 1.0
        np.testing.assert_allclose(manifold[:, 0], expected, atol=1e-7)
        np.testing.assert_allclose(manifold[:, 1], expected, atol=1e-7)

    def test_fourier_generation_is_reproducible_for_fixed_seed(self) -> None:
        config = DataConfig(
            source="synthetic", mode="fourier", n_cells=25, n_genes=4, sigma=0.1, k_min=1, k_max=3, seed=11
        )
        dataset_a = generate_synthetic_dataset(config)
        dataset_b = generate_synthetic_dataset(config)

        np.testing.assert_allclose(dataset_a.S, dataset_b.S, atol=1e-7)
        np.testing.assert_allclose(dataset_a.A, dataset_b.A, atol=1e-7)

    def test_fourier_generation_changes_with_seed(self) -> None:
        dataset_a = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="fourier", n_cells=25, n_genes=4, sigma=0.1, k_min=1, k_max=3, seed=11)
        )
        dataset_b = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="fourier", n_cells=25, n_genes=4, sigma=0.1, k_min=1, k_max=3, seed=12)
        )

        self.assertFalse(np.allclose(dataset_a.A, dataset_b.A))

    def test_unknown_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            generate_synthetic_dataset(
                DataConfig(source="synthetic", mode="invalid", n_cells=16, n_genes=3, seed=5)
            )

    def test_parallel_permutation_output_omits_irrelevant_default_fields(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="fourier", n_cells=16, n_genes=3, sigma=0.1, k_min=1, k_max=2, seed=7)
        )
        result = TestResult(
            method_name="parallel_permutation",
            metric="mse",
            p_value=0.25,
            stat_true=0.1,
            stat_perm=np.asarray([0.2, 0.3], dtype=np.float64),
            runtime_sec=0.01,
            n_cells=dataset.n_cells,
            n_genes=dataset.n_genes,
            config={},
            artifacts={
                "true_isodepth": np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32),
                "lowest_isodepth": np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32),
                "lowest_S": np.asarray(dataset.S, dtype=np.float32),
                "lowest_stat": 0.2,
                "highest_isodepth": np.linspace(1.0, 0.0, dataset.n_cells, dtype=np.float32),
                "highest_S": np.asarray(dataset.S, dtype=np.float32),
                "highest_stat": 0.3,
                "null_summary": {"mean": 0.25},
            },
        ).validate()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(
                    source="synthetic",
                    mode="fourier",
                    n_cells=16,
                    n_genes=3,
                    sigma=0.1,
                    k_min=1,
                    k_max=2,
                    seed=7,
                ),
                output=OutputConfig(out_dir=tmpdir, run_name="parallel_permutation_test"),
            ).validate()

            payload, _ = save_standardized_outputs(dataset, result, run_config)

        self.assertNotIn("perturb_target", payload["config"]["test"])
        self.assertNotIn("subset_fractions", payload["config"]["test"])
        self.assertNotIn("delta", payload["config"]["test"])
        self.assertNotIn("n_nulls", payload["config"]["test"])
        self.assertNotIn("perturb_target", payload["artifacts"])
        self.assertNotIn("subset_fractions", payload["artifacts"])
        self.assertNotIn("delta", payload["artifacts"])
        self.assertIn("true_isodepth", payload["artifacts"])
        self.assertEqual(len(payload["artifacts"]["true_isodepth"]), dataset.n_cells)
        self.assertIn("synthetic_true_curve_plot", payload["artifacts"])
        self.assertNotIn("synthetic_true_curve", payload["artifacts"]["dataset_meta"])
        self.assertTrue(payload["artifacts"]["dataset_meta"]["has_synthetic_true_curve"])

    def test_standardized_outputs_writes_selected_genes_with_predictions(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="fourier",
                n_cells=16,
                n_genes=4,
                sigma=0.1,
                k_min=1,
                k_max=2,
                seed=7,
            )
        )
        rng = np.random.RandomState(0)
        pred = np.asarray(dataset.A, dtype=np.float32) + 0.01 * rng.randn(*dataset.A.shape).astype(np.float32)
        depth = np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32)
        result = TestResult(
            method_name="parallel_permutation",
            metric="mse",
            p_value=0.25,
            stat_true=0.1,
            stat_perm=np.asarray([0.2, 0.3], dtype=np.float64),
            runtime_sec=0.01,
            n_cells=dataset.n_cells,
            n_genes=dataset.n_genes,
            config={},
            artifacts={
                "pred_true": pred,
                "true_isodepth": depth,
                "lowest_isodepth": np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32),
                "lowest_S": np.asarray(dataset.S, dtype=np.float32),
                "lowest_stat": 0.2,
                "highest_isodepth": np.linspace(1.0, 0.0, dataset.n_cells, dtype=np.float32),
                "highest_S": np.asarray(dataset.S, dtype=np.float32),
                "highest_stat": 0.3,
                "null_summary": {"mean": 0.25},
            },
        ).validate()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(
                    source="synthetic",
                    mode="fourier",
                    n_cells=16,
                    n_genes=4,
                    sigma=0.1,
                    k_min=1,
                    k_max=2,
                    seed=7,
                ),
                output=OutputConfig(out_dir=tmpdir, run_name="selected_genes_test"),
            ).validate()

            payload, _ = save_standardized_outputs(dataset, result, run_config)

            sg = Path(payload["artifacts"]["selected_genes_dir"])
            self.assertTrue(sg.is_dir())
            self.assertTrue((sg / "manifest.json").exists())
            manifest = json.loads((sg / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["top_k"], 5)
            self.assertEqual(len(manifest["scenarios"]), 1)
            self.assertEqual(manifest["scenarios"][0]["subdir"], "primary")
            pngs = list((sg / "primary").glob("*.png"))
            self.assertEqual(len(pngs), min(5, dataset.n_genes))

    def test_standardized_outputs_save_synthetic_true_curve_plot(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="radial", n_cells=16, n_genes=3, sigma=0.1, seed=7)
        )
        result = TestResult(
            method_name="parallel_permutation",
            metric="mse",
            p_value=0.25,
            stat_true=0.1,
            stat_perm=np.asarray([0.2, 0.3], dtype=np.float64),
            runtime_sec=0.01,
            n_cells=dataset.n_cells,
            n_genes=dataset.n_genes,
            config={},
            artifacts={
                "true_isodepth": np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32),
                "lowest_isodepth": np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32),
                "lowest_S": np.asarray(dataset.S, dtype=np.float32),
                "lowest_stat": 0.2,
                "highest_isodepth": np.linspace(1.0, 0.0, dataset.n_cells, dtype=np.float32),
                "highest_S": np.asarray(dataset.S, dtype=np.float32),
                "highest_stat": 0.3,
                "null_summary": {"mean": 0.25},
            },
        ).validate()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", mode="radial", n_cells=16, n_genes=3, sigma=0.1, seed=7),
                output=OutputConfig(out_dir=tmpdir, run_name="synthetic_true_curve_test"),
            ).validate()

            payload, result_path = save_standardized_outputs(dataset, result, run_config)

            self.assertTrue(result_path.exists())
            self.assertTrue((result_path.parent / "synthetic_true_curve_test_true_curve.png").exists())
            self.assertEqual(
                payload["artifacts"]["synthetic_true_curve_plot"],
                str(result_path.parent / "synthetic_true_curve_test_true_curve.png"),
            )

    def test_standardized_outputs_save_true_rerun_isodepth_grid_plot(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="radial", n_cells=16, n_genes=3, sigma=0.1, seed=7)
        )

        class _MockModel:
            def __init__(self) -> None:
                depth_a = np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32)
                depth_b = np.linspace(1.0, 0.0, dataset.n_cells, dtype=np.float32)
                depth_c = np.sin(np.linspace(0.0, np.pi, dataset.n_cells, dtype=np.float32))
                self.training_metadata = {
                    "n_reruns": 3,
                    "selection_loss": "training_reconstruction_loss",
                    "best_train_loss_per_model": np.asarray([0.2], dtype=np.float64),
                    "best_rerun_index_per_model": np.asarray([1], dtype=np.int64),
                    "train_loss_per_rerun": np.asarray([[0.4, 0.2, 0.3]], dtype=np.float64),
                    "true_rerun_isodepths": np.stack([depth_a, depth_b, depth_c], axis=0),
                }

        result = TestResult(
            method_name="parallel_permutation",
            metric="mse",
            p_value=0.25,
            stat_true=0.1,
            stat_perm=np.asarray([0.2, 0.3], dtype=np.float64),
            runtime_sec=0.01,
            n_cells=dataset.n_cells,
            n_genes=dataset.n_genes,
            config={},
            artifacts={
                "model": _MockModel(),
                "true_isodepth": np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32),
                "lowest_isodepth": np.linspace(0.0, 1.0, dataset.n_cells, dtype=np.float32),
                "lowest_S": np.asarray(dataset.S, dtype=np.float32),
                "lowest_stat": 0.2,
                "highest_isodepth": np.linspace(1.0, 0.0, dataset.n_cells, dtype=np.float32),
                "highest_S": np.asarray(dataset.S, dtype=np.float32),
                "highest_stat": 0.3,
                "null_summary": {"mean": 0.25},
            },
        ).validate()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", mode="radial", n_cells=16, n_genes=3, sigma=0.1, seed=7),
                test=TestConfig(method="parallel_permutation", metric="mse", n_perms=2, n_reruns=3),
                output=OutputConfig(out_dir=tmpdir, run_name="rerun_isodepth_test"),
            ).validate()

            payload, result_path = save_standardized_outputs(dataset, result, run_config)

            expected_path = result_path.parent / "rerun_isodepth_test_true_rerun_isodepths.png"
            self.assertTrue(expected_path.exists())
            self.assertEqual(payload["artifacts"]["true_rerun_isodepth_grid_plot"], str(expected_path))


if __name__ == "__main__":
    unittest.main()
