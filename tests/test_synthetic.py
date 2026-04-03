from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DataConfig, OutputConfig, RunConfig, TestResult
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


if __name__ == "__main__":
    unittest.main()
