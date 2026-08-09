from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import CovariateConfig, DataConfig, TestConfig, run_config_from_mapping
from methods.metrics import compute_metric, permutation_p_value
from run_permutation import _build_arg_parser, _build_cli_overrides


class TestDataSchema(unittest.TestCase):
    def test_q_is_optional(self) -> None:
        config = DataConfig(source="synthetic", n_cells=8, n_genes=3, q=None)
        self.assertIs(config.validate(), config)

    def test_positive_q_is_valid_for_h5ad(self) -> None:
        config = DataConfig(source="h5ad", h5ad="data/example.h5ad", q=2)
        self.assertIs(config.validate(), config)

    def test_obs_drop_na_must_be_non_empty_list(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="h5ad", h5ad="data/example.h5ad", obs_drop_na=[]).validate()
        with self.assertRaises(ValueError):
            DataConfig(source="h5ad", h5ad="data/example.h5ad", obs_drop_na=[""]).validate()

    def test_obs_drop_na_accepts_column_names(self) -> None:
        config = DataConfig(
            source="h5ad",
            h5ad="data/example.h5ad",
            obs_drop_na=["manual_layer_label"],
        )
        self.assertIs(config.validate(), config)

    def test_exclude_gene_patterns_accepts_regex_list_for_h5ad(self) -> None:
        config = DataConfig(
            source="h5ad",
            h5ad="data/example.h5ad",
            exclude_gene_patterns=["^MT-", "^RPL", "^MALAT1$"],
        )
        self.assertIs(config.validate(), config)

    def test_exclude_gene_patterns_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="h5ad",
                h5ad="data/example.h5ad",
                exclude_gene_patterns=[],
            ).validate()
        with self.assertRaises(ValueError):
            DataConfig(
                source="h5ad",
                h5ad="data/example.h5ad",
                exclude_gene_patterns=["["],
            ).validate()
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                exclude_gene_patterns=["^MT-"],
            ).validate()

    def test_gene_list_accepts_gene_symbol_list_for_h5ad(self) -> None:
        config = DataConfig(
            source="h5ad",
            h5ad="data/example.h5ad",
            gene_list=["ACTB", "VEGFA", "HK1"],
            top_var_genes=0,
        )
        self.assertIs(config.validate(), config)

    def test_gene_list_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="h5ad", h5ad="data/example.h5ad", gene_list=[]).validate()
        with self.assertRaises(ValueError):
            DataConfig(source="h5ad", h5ad="data/example.h5ad", gene_list=[""]).validate()
        with self.assertRaises(ValueError):
            DataConfig(
                source="h5ad", h5ad="data/example.h5ad", gene_list=["ACTB", "ACTB"]
            ).validate()
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", gene_list=["ACTB"]).validate()

    def test_gene_list_cannot_be_combined_with_top_var_genes(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="h5ad",
                h5ad="data/example.h5ad",
                gene_list=["ACTB", "VEGFA"],
                top_var_genes=3000,
            ).validate()

    def test_log1p_cannot_be_combined_with_q(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="h5ad", h5ad="data/example.h5ad", log1p=True, q=2).validate()

    def test_non_positive_q_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", n_cells=8, n_genes=3, q=0).validate()

    def test_q_is_rejected_for_synthetic(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", n_cells=8, n_genes=3, q=2).validate()

    def test_expression_distribution_is_rejected_for_h5ad(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="h5ad",
                h5ad="data/example.h5ad",
                expression_distribution="poisson",
            ).validate()

    def test_poisson_expression_distribution_is_valid_for_synthetic(self) -> None:
        config = DataConfig(
            source="synthetic",
            mode="radial",
            n_cells=16,
            n_genes=3,
            expression_distribution="poisson",
            mean_count=10.0,
        )
        self.assertIs(config.validate(), config)

    def test_non_positive_mean_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                n_cells=16,
                n_genes=3,
                expression_distribution="poisson",
                mean_count=0.0,
            ).validate()

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

    def test_side_length_is_valid_for_noise_mode(self) -> None:
        config = DataConfig(source="synthetic", mode="noise", n_cells=24, n_genes=3, side_length=6)
        self.assertIs(config.validate(), config)

    def test_side_length_is_rejected_when_shape_is_not_square(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=24,
                n_genes=3,
                shape="semicircle",
                side_length=6,
            ).validate()

    def test_non_square_shape_is_valid_without_side_length(self) -> None:
        config = DataConfig(
            source="synthetic",
            mode="noise",
            n_cells=24,
            n_genes=3,
            shape="semicircle",
        )
        self.assertIs(config.validate(), config)

    def test_invalid_shape_string_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=16,
                n_genes=3,
                shape="hexagon",
            ).validate()

    def test_invalid_sampling_bias_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=16,
                n_genes=3,
                sampling_bias={"type": "gaussian"},
            ).validate()

    def test_sampling_bias_is_rejected_for_h5ad(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="h5ad",
                h5ad="data/example.h5ad",
                sampling_bias={"type": "uniform"},
            ).validate()

    def test_uniform_sampling_bias_allows_side_length_ignored_for_lattice(self) -> None:
        config = DataConfig(
            source="synthetic",
            mode="noise",
            n_cells=25,
            n_genes=3,
            shape="square",
            side_length=6,
            sampling_bias={"type": "uniform"},
        )
        self.assertIs(config.validate(), config)

    def test_uniform_sampling_bias_is_valid_without_side_length(self) -> None:
        config = DataConfig(
            source="synthetic",
            mode="noise",
            n_cells=120,
            n_genes=3,
            sampling_bias={"type": "uniform"},
        )
        self.assertIs(config.validate(), config)

    def test_legacy_string_sampling_bias_is_accepted(self) -> None:
        config = DataConfig(
            source="synthetic",
            mode="noise",
            n_cells=120,
            n_genes=3,
            sampling_bias="uniform",
        ).validate()
        self.assertEqual(config.sampling_bias.type, "uniform")
        self.assertIsNone(config.sampling_bias.variance)

    def test_normal_sampling_bias_requires_variance(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=16,
                n_genes=3,
                sampling_bias={"type": "normal"},
            ).validate()

    def test_normal_sampling_bias_rejects_nonpositive_variance(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=16,
                n_genes=3,
                sampling_bias={"type": "normal", "variance": 0.0},
            ).validate()

    def test_uniform_sampling_bias_rejects_variance(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=16,
                n_genes=3,
                sampling_bias={"type": "uniform", "variance": 0.05},
            ).validate()

    def test_unknown_sampling_bias_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                mode="noise",
                n_cells=16,
                n_genes=3,
                sampling_bias={"type": "uniform", "stddev": 0.1},
            ).validate()

    def test_side_length_requires_divisible_n_cells_for_noise(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", mode="noise", n_cells=25, n_genes=3, side_length=6).validate()

    def test_side_length_is_rejected_for_non_noise_modes(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="synthetic", mode="radial", n_cells=16, n_genes=3, side_length=4).validate()

    def test_side_length_is_rejected_for_h5ad_configs(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(source="h5ad", h5ad="data/example.h5ad", side_length=4).validate()

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
    def test_n_reruns_defaults_to_thirty(self) -> None:
        self.assertEqual(TestConfig().n_reruns, 30)

    def test_decoder_defaults_to_nn(self) -> None:
        self.assertEqual(TestConfig().decoder, "nn")

    def test_invalid_decoder_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(decoder="bad").validate()

    def test_cli_decoder_override_is_parsed(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--decoder", "nn"])
        overrides = _build_cli_overrides(args)
        self.assertEqual(overrides["test"]["decoder"], "nn")

    def test_cli_log1p_override_is_parsed(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--log1p"])
        overrides = _build_cli_overrides(args)
        self.assertTrue(overrides["data"]["log1p"])

    def test_cli_no_log1p_override_is_parsed(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--no-log1p"])
        overrides = _build_cli_overrides(args)
        self.assertFalse(overrides["data"]["log1p"])

    def test_non_positive_n_reruns_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(n_reruns=0).validate()

    def test_cli_n_reruns_override_is_parsed(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--n-reruns", "7"])
        overrides = _build_cli_overrides(args)
        self.assertEqual(overrides["test"]["n_reruns"], 7)

    def test_zero_sgd_batch_size_keeps_full_batch_behavior(self) -> None:
        config = TestConfig(
            method="comparison_perturbation_test",
            metric="mse",
            delta=[0.1],
            sgd_batch_size=0,
        )
        self.assertIs(config.validate(), config)

    def test_negative_sgd_batch_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_perturbation_test",
                metric="mse",
                delta=[0.1],
                sgd_batch_size=-1,
            ).validate()

    def test_cli_sgd_batch_size_override_is_parsed(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--sgd-batch-size", "11"])
        overrides = _build_cli_overrides(args)
        self.assertEqual(overrides["test"]["sgd_batch_size"], 11)

    def test_cosine_lr_decay_requires_positive_sgd_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_perturbation_test",
                metric="mse",
                delta=[0.1],
                sgd_cosine_lr_decay=True,
                sgd_batch_size=0,
            ).validate()
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_perturbation_test",
                metric="mse",
                delta=[0.1],
                sgd_cosine_lr_decay=True,
                sgd_batch_size=None,
            ).validate()

    def test_cosine_eta_min_must_be_in_range(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_perturbation_test",
                metric="mse",
                delta=[0.1],
                sgd_batch_size=4,
                sgd_cosine_lr_decay=True,
                sgd_cosine_eta_min=-1e-6,
            ).validate()
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_perturbation_test",
                metric="mse",
                delta=[0.1],
                sgd_batch_size=4,
                lr=1e-3,
                sgd_cosine_lr_decay=True,
                sgd_cosine_eta_min=1e-2,
            ).validate()

    def test_cosine_t_max_steps_must_be_positive_when_set(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_perturbation_test",
                metric="mse",
                delta=[0.1],
                sgd_batch_size=4,
                sgd_cosine_lr_decay=True,
                sgd_cosine_t_max_steps=0,
            ).validate()

    def test_cli_sgd_cosine_overrides_are_parsed(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--sgd-cosine-lr-decay",
                "--sgd-cosine-eta-min",
                "1e-5",
                "--sgd-cosine-t-max-steps",
                "400",
            ]
        )
        overrides = _build_cli_overrides(args)
        self.assertTrue(overrides["test"]["sgd_cosine_lr_decay"])
        self.assertEqual(overrides["test"]["sgd_cosine_eta_min"], 1e-5)
        self.assertEqual(overrides["test"]["sgd_cosine_t_max_steps"], 400)

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
            n_perms=3,
            n_nulls=5,
            subset_fractions=[0.4, 0.8],
        )
        self.assertIs(config.validate(), config)

    def test_invalid_subset_fraction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_subsampling_test",
                metric="mse",
                subset_fractions=[0.5, 1.0],
            ).validate()

    def test_invalid_comparison_subsampling_n_perms_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="comparison_subsampling_test",
                metric="mse",
                n_perms=0,
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
        )
        self.assertIs(config.validate(), config)

    def test_subsampling_rejects_correlation_metric(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="subsampling_test",
                metric="spearman_corr_mean",
                n_perms=3,
            ).validate()


class TestCovariateSchema(unittest.TestCase):
    def test_unknown_covariate_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CovariateConfig(type="unknown").validate()

    def test_run_config_parses_nested_covariate(self) -> None:
        cfg = run_config_from_mapping(
            {
                "data": {"source": "synthetic", "n_cells": 16, "n_genes": 3, "mode": "noise"},
                "test": {
                    "method": "parallel_permutation",
                    "metric": "mse",
                    "n_perms": 2,
                    "covariate": {"type": "midline"},
                },
                "output": {"out_dir": "results", "run_name": "t"},
            }
        )
        self.assertIsNotNone(cfg.test.covariate)
        self.assertEqual(cfg.test.covariate.type, "midline")

    def test_run_config_accepts_string_covariate_shorthand(self) -> None:
        cfg = run_config_from_mapping(
            {
                "data": {"source": "synthetic", "n_cells": 16, "n_genes": 3, "mode": "noise"},
                "test": {
                    "method": "parallel_permutation",
                    "metric": "mse",
                    "n_perms": 2,
                    "covariate": "midline",
                },
                "output": {"out_dir": "results", "run_name": "t"},
            }
        )
        self.assertEqual(cfg.test.covariate.type, "midline")


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
