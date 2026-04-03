from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DataConfig, TestConfig
from methods.metrics import compute_metric, permutation_p_value


class TestDataSchema(unittest.TestCase):
    def test_q_is_optional(self) -> None:
        config = DataConfig(source="synthetic", n_cells=8, n_genes=3, q=None)
        self.assertIs(config.validate(), config)

    def test_positive_q_is_valid_for_h5ad(self) -> None:
        config = DataConfig(source="h5ad", h5ad="data/example.h5ad", q=2)
        self.assertIs(config.validate(), config)

    def test_non_positive_q_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", n_cells=8, n_genes=3, q=0).validate()

    def test_q_is_rejected_for_synthetic(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", n_cells=8, n_genes=3, q=2).validate()

    def test_fourier_mode_accepts_frequency_bounds(self) -> None:
        config = DataConfig(source="synthetic", mode="fourier", n_cells=16, n_genes=3, k_min=2, k_max=4)
        self.assertIs(config.validate(), config)

    def test_fourier_mode_accepts_dependent_xy_toggle(self) -> None:
        config = DataConfig(
            source="synthetic",
            mode="fourier",
            n_cells=16,
            n_genes=3,
            k_min=2,
            k_max=4,
            dependent_xy=False,
        )
        self.assertIs(config.validate(), config)

    def test_fourier_mode_requires_frequency_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", mode="fourier", n_cells=16, n_genes=3).validate()

    def test_non_positive_k_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", mode="fourier", n_cells=16, n_genes=3, k_min=0, k_max=2).validate()
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", mode="fourier", n_cells=16, n_genes=3, k_min=1, k_max=0).validate()

    def test_fourier_mode_rejects_inverted_frequency_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", mode="fourier", n_cells=16, n_genes=3, k_min=4, k_max=2).validate()

    def test_negative_poly_degree_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", n_cells=16, n_genes=3, poly_degree=-1).validate()

    def test_poly_degree_is_rejected_for_h5ad_configs(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="h5ad", h5ad="data/example.h5ad", poly_degree=2).validate()

    def test_frequency_bounds_are_rejected_for_non_fourier_synthetic_modes(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", mode="noise", n_cells=16, n_genes=3, k_min=1, k_max=2).validate()

    def test_dependent_xy_is_rejected_for_non_fourier_modes(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=16,
                n_genes=3,
                dependent_xy=False,
            ).validate()

    def test_frequency_bounds_are_rejected_for_h5ad_configs(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="h5ad", h5ad="data/example.h5ad", k_min=1, k_max=2).validate()

    def test_legacy_k_maps_to_k_max_with_default_k_min(self) -> None:
        config = DataConfig(source="synthetic", mode="fourier", n_cells=16, n_genes=3, k=2).validate()
        self.assertEqual(config.k_min, 1)
        self.assertEqual(config.k_max, 2)

    def test_unsupported_synthetic_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", mode="unknown", n_cells=16, n_genes=3).validate()


class TestPerturbationSchema(unittest.TestCase):
    def test_comparison_perturbation_config_is_valid(self) -> None:
        config = TestConfig(
            method="comparison_perturbation_test",
            metric="spearman_corr_mean",
            delta=[0.1],
            perturb_target="coordinates",
        )
        self.assertIs(config.validate(), config)

    def test_invalid_perturb_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_perturbation_test",
                delta=[0.1],
                perturb_target="expression",
            ).validate()

    def test_invalid_delta_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="comparison_perturbation_test", delta=[0.0]).validate()

    def test_delta_list_is_validated(self) -> None:
        config = TestConfig(
            method="comparison_perturbation_test",
            delta=[0.01, 0.05, 0.1],
        )
        self.assertIs(config.validate(), config)

    def test_invalid_delta_list_entries_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_perturbation_test",
                delta=[0.01, 0.0],
            ).validate()

    def test_perturbation_test_config_is_valid(self) -> None:
        config = TestConfig(
            method="perturbation_test",
            metric="mse",
            n_perms=3,
            delta=[0.1],
            perturb_target="coordinates",
        )
        self.assertIs(config.validate(), config)

    def test_perturbation_test_rejects_correlation_metric(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="perturbation_test",
                metric="spearman_corr_mean",
                n_perms=3,
                delta=[0.1],
            ).validate()


class TestSubsetSelectionSchema(unittest.TestCase):
    def test_comparison_subsampling_config_is_valid(self) -> None:
        config = TestConfig(
            method="comparison_subsampling_test",
            metric="mse",
            n_perms=0,
            n_nulls=5,
            subset_fractions=[0.4, 0.8],
            n_subsets=3,
        )
        self.assertIs(config.validate(), config)

    def test_invalid_subset_fraction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_subsampling_test",
                metric="mse",
                subset_fractions=[0.5, 1.0],
            ).validate()

    def test_invalid_n_subsets_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_subsampling_test",
                metric="mse",
                n_subsets=0,
            ).validate()

    def test_comparison_subsampling_rejects_correlation_metric(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_subsampling_test",
                metric="spearman_corr_mean",
            ).validate()

    def test_subsampling_config_is_valid(self) -> None:
        config = TestConfig(
            method="subsampling_test",
            metric="mse",
            n_perms=3,
            subset_fractions=[0.4, 0.8],
            n_subsets=3,
        )
        self.assertIs(config.validate(), config)

    def test_subsampling_rejects_correlation_metric(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="subsampling_test",
                metric="spearman_corr_mean",
                n_perms=3,
            ).validate()


class TestMetricUtilities(unittest.TestCase):
    def test_spearman_metric_handles_reversed_order(self) -> None:
        y_true = [[0.0], [1.0], [2.0], [3.0]]
        y_pred = [[3.0], [2.0], [1.0], [0.0]]
        self.assertAlmostEqual(compute_metric("spearman_corr_mean", y_true, y_pred), -1.0)

    def test_permutation_p_value_respects_metric_tail(self) -> None:
        lower_tail = permutation_p_value("mse", 0.05, [0.1, 0.2, 0.3])
        upper_tail = permutation_p_value("spearman_corr_mean", 0.9, [0.2, 0.3, 0.4])
        self.assertLess(lower_tail, 0.5)
        self.assertLess(upper_tail, 0.5)


if __name__ == "__main__":
    unittest.main()
