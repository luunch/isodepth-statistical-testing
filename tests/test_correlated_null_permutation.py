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

from analysis.plots import save_variogram_matching_plots
from data.schemas import DatasetBundle, TestConfig
from data.synthetic import generate_synthetic_dataset, DataConfig
from methods.correlated_null_permutation import (
    _adaptive_blur_expression,
    _build_variogram_geometry,
    _classical_variogram,
    _covariance_variance_contribution,
    _fit_variance_constrained_coefficients,
    _fit_variogram_sse,
    _median_nearest_neighbor_distance,
    _precompute_adaptive_blur_plan,
    _score_restored_candidate,
    _standardize_noise_columns,
    _strictly_better,
    build_correlated_null_adaptive_surrogates,
    build_correlated_null_surrogates,
)
from methods.spatial_blur import apply_gaussian_weights, gaussian_smoothing_weights
from methods.permutation import (
    run_correlated_null_adaptive_permutation_method,
    run_correlated_null_permutation_method,
)


def _brute_classical_variogram(expression: np.ndarray, geom, pairs, bin_index) -> np.ndarray:
    n_bins = len(geom.bin_matrices)
    n_genes = expression.shape[1] if expression.ndim == 2 else 1
    y = np.asarray(expression, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    out = np.full((n_bins, n_genes), np.nan, dtype=np.float64)
    for j in range(n_genes):
        difference = y[pairs[:, 0], j] - y[pairs[:, 1], j]
        squared_sum = np.bincount(bin_index, weights=difference**2, minlength=n_bins)
        out[geom.valid_bins, j] = (
            squared_sum[geom.valid_bins] / (2.0 * geom.pair_counts[geom.valid_bins])
        )
    return out


def _grid_coords(n_side: int = 12, spacing: float = 1.0) -> np.ndarray:
    xs = np.arange(n_side, dtype=np.float64) * spacing
    xx, yy = np.meshgrid(xs, xs)
    return np.column_stack([xx.ravel(), yy.ravel()])


class TestCorrelatedNullPermutation(unittest.TestCase):
    def test_sparse_variogram_matches_brute_force(self) -> None:
        coords = _grid_coords(10)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((coords.shape[0], 4))
        geom = _build_variogram_geometry(coords, n_bins=13)
        fast = _classical_variogram(x, geom)
        slow = _brute_classical_variogram(x, geom, geom.pairs, geom.bin_index)
        np.testing.assert_allclose(fast, slow, equal_nan=True, rtol=1e-10)

    def test_surrogates_are_z_scored_with_expected_shape(self) -> None:
        coords = _grid_coords(8)
        rng = np.random.default_rng(1)
        x = rng.standard_normal((coords.shape[0], 6))
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
        surr, _ = build_correlated_null_surrogates(
            x,
            coords,
            n_perms=3,
            seed=2,
            sigma_multiples=[1.0, 2.0],
            n_bins=8,
            n_plot_genes=4,
            verbose=False,
        )
        self.assertEqual(surr.shape, (3, coords.shape[0], 6))
        for p in range(surr.shape[0]):
            np.testing.assert_allclose(surr[p].mean(axis=0), 0.0, atol=1e-5)
            np.testing.assert_allclose(surr[p].std(axis=0, ddof=0), 1.0, atol=1e-5)

    def test_preserves_input_moments_and_is_affine_equivariant(self) -> None:
        coords = _grid_coords(8)
        x = np.random.default_rng(12).standard_normal((len(coords), 5))
        x[:, -1] = 0.0
        scale = np.array([0.3, 2.0, 7.0, 4.0, 1.0])
        shift = np.array([10.0, -20.0, 30.0, 5.0, 9.0])
        for builder, candidates in [
            (build_correlated_null_surrogates, {"sigma_multiples": [1.0, 2.0]}),
            (build_correlated_null_adaptive_surrogates, {"nn_candidates": [0.0, 0.2, 0.6]}),
        ]:
            kwargs = dict(n_perms=3, seed=13, n_bins=8, n_plot_genes=3, verbose=False, **candidates)
            base, _ = builder(x, coords, **kwargs)
            transformed = x * scale + shift
            actual, diag = builder(transformed, coords, **kwargs)
            np.testing.assert_allclose(actual, base * scale + shift, atol=1e-5)
            np.testing.assert_allclose(actual.mean(axis=1), np.broadcast_to(transformed.mean(0), (3, 5)), atol=1e-5)
            np.testing.assert_allclose(actual.var(axis=1), np.broadcast_to(transformed.var(0), (3, 5)), atol=1e-5, rtol=1e-5)
            indices = diag["plot_gene_indices"]
            np.testing.assert_allclose(diag["X_true"], transformed[:, indices], atol=1e-5)
            geom = _build_variogram_geometry(coords, 8)
            np.testing.assert_allclose(diag["variogram_real"], _classical_variogram(transformed, geom)[:, indices], rtol=1e-5, atol=1e-5)

    def test_plot_gene_count_does_not_change_nulls(self) -> None:
        coords = _grid_coords(8)
        x = np.random.default_rng(10).standard_normal((len(coords), 5))
        kwargs = dict(n_perms=2, seed=3, nn_candidates=[0.0, 0.3], n_bins=8, verbose=False)
        a, _ = build_correlated_null_adaptive_surrogates(x, coords, n_plot_genes=1, **kwargs)
        b, _ = build_correlated_null_adaptive_surrogates(x, coords, n_plot_genes=5, **kwargs)
        np.testing.assert_array_equal(a, b)

    def test_nonnegative_variogram_fit_matches_nnls(self) -> None:
        from scipy.optimize import nnls
        rng = np.random.default_rng(11)
        candidates = rng.uniform(0, 2, (3, 8, 6))
        candidates[0, :, 0] = 0  # rank-deficient design
        candidates[1, :, 1] = 1
        target = rng.uniform(0, 2, (8, 6))
        target[:, 2] = 3 - candidates[0, :, 2]  # negative unconstrained slope
        target[:, 3] = np.maximum(0, 2 * candidates[0, :, 3] - 1)
        valid = np.ones(8, dtype=bool)
        sse, intercept, slope = _fit_variogram_sse(target, candidates, valid)
        self.assertTrue(np.all(intercept >= 0))
        self.assertTrue(np.all(slope >= 0))
        for k in range(3):
            for g in range(6):
                _, residual = nnls(np.column_stack([candidates[k, :, g], np.ones(8)]), target[:, g])
                self.assertAlmostEqual(sse[k, g], residual**2, places=10)

    def test_identity_is_joint_permutation_on_original_scale(self) -> None:
        coords = _grid_coords(8)
        x = np.random.default_rng(7).normal(5, 3, (len(coords), 4))
        surr, diag = build_correlated_null_adaptive_surrogates(
            x, coords, n_perms=2, seed=6, nn_candidates=[0.0],
            n_bins=8, n_plot_genes=4, verbose=False,
        )
        for sample in surr:
            order = np.argsort(sample[:, 0])
            np.testing.assert_allclose(sample[order], x[np.argsort(x[:, 0])], atol=1e-6)
        np.testing.assert_allclose(diag["fitted_variograms_candidates"][0], diag["variogram_permuted"], equal_nan=True)

    def test_variogram_includes_all_pairs_below_cutoff(self) -> None:
        coords = _grid_coords(8)
        geom = _build_variogram_geometry(coords, 8)
        from scipy.spatial.distance import pdist
        self.assertEqual(len(geom.pairs), int(np.sum(pdist(coords) < geom.max_dist)))

    def test_correlated_null_rejects_poisson_metric(self) -> None:
        for method in ["correlated_null_permutation", "correlated_null_adaptive_permutation"]:
            with self.assertRaisesRegex(ValueError, "Poisson"):
                TestConfig(method=method, metric="nll_poisson_mse").validate()

    def test_gene_correlation_is_partially_preserved(self) -> None:
        coords = _grid_coords(10)
        rng = np.random.default_rng(3)
        latent = rng.standard_normal(coords.shape[0])
        x = np.column_stack([latent + 0.1 * rng.standard_normal(coords.shape[0]) for _ in range(5)])
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
        corr_true = np.corrcoef(x.T)
        surr, _ = build_correlated_null_surrogates(
            x,
            coords,
            n_perms=1,
            seed=4,
            sigma_multiples=[1.0, 3.0],
            n_bins=8,
            n_plot_genes=3,
            verbose=False,
        )
        corr_surr = np.corrcoef(surr[0].T)
        off_diag = ~np.eye(corr_true.shape[0], dtype=bool)
        mean_abs_true = float(np.mean(np.abs(corr_true[off_diag])))
        mean_abs_surr = float(np.mean(np.abs(corr_surr[off_diag])))
        self.assertGreater(mean_abs_surr, 0.05)
        self.assertGreater(mean_abs_surr, 0.3 * mean_abs_true)

    def test_end_to_end_writes_variogram_matching_plots(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="radial",
                n_cells=48,
                n_genes=8,
                sigma=0.15,
                seed=7,
            )
        )
        config = TestConfig(
            method="correlated_null_permutation",
            metric="mse",
            n_perms=2,
            epochs=30,
            n_reruns=1,
            seed=11,
            device="cpu",
            decoder="linear",
            verbose=False,
            variogram_sigma_multiples=[1.0, 2.0],
            variogram_n_plot_genes=4,
        ).validate()
        result = run_correlated_null_permutation_method(dataset, config)
        self.assertEqual(result.method_name, "correlated_null_permutation")
        self.assertIsNotNone(result.p_value)
        diag = result.artifacts.get("variogram_matching_diagnostics")
        self.assertIsNotNone(diag)
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_variogram_matching_plots(diag, tmp)
            self.assertTrue((Path(tmp) / "variogram-matching" / "variogram_candidates.png").exists())
            self.assertTrue((Path(tmp) / "variogram-matching" / "variogram_best_match.png").exists())
            self.assertIn("variogram_candidates", paths)
            with np.load(paths["diagnostics"], allow_pickle=False) as saved:
                self.assertEqual(int(saved["algorithm_version"]), 3)
                self.assertEqual(
                    str(saved["selection_objective"]),
                    "final_surrogate_empirical_variogram",
                )
                np.testing.assert_allclose(saved["surrogate_gene_variance"], saved["input_gene_variance"], atol=1e-6)

    def test_config_validation_rejects_empty_sigma_multiples(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="correlated_null_permutation",
                variogram_sigma_multiples=[],
            ).validate()

    def test_adaptive_requires_single_zero_nn_candidate(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="correlated_null_adaptive_permutation",
                variogram_nn_candidates=[0.1, 0.2],
            ).validate()

    def test_adaptive_surrogates_shape_and_identity_candidate(self) -> None:
        coords = _grid_coords(8)
        rng = np.random.default_rng(5)
        x = rng.standard_normal((coords.shape[0], 4))
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
        plan = _precompute_adaptive_blur_plan(coords, [0.0, 0.5])
        self.assertEqual(int(plan.identity_index), 0)
        surr, diag = build_correlated_null_adaptive_surrogates(
            x,
            coords,
            n_perms=2,
            seed=3,
            nn_candidates=[0.0, 0.5],
            n_bins=8,
            n_plot_genes=2,
            verbose=False,
        )
        self.assertEqual(surr.shape, (2, coords.shape[0], 4))
        self.assertEqual(diag["variogram_blur_mode"], "adaptive")

    def test_adaptive_end_to_end(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(
                source="synthetic",
                mode="radial",
                n_cells=40,
                n_genes=6,
                sigma=0.15,
                seed=9,
            )
        )
        config = TestConfig(
            method="correlated_null_adaptive_permutation",
            metric="mse",
            n_perms=2,
            epochs=20,
            n_reruns=1,
            seed=11,
            device="cpu",
            decoder="linear",
            verbose=False,
            variogram_nn_candidates=[0.0, 0.3, 0.6],
            variogram_n_plot_genes=3,
        ).validate()
        result = run_correlated_null_adaptive_permutation_method(dataset, config)
        self.assertEqual(result.method_name, "correlated_null_adaptive_permutation")
        self.assertIsNotNone(result.artifacts.get("variogram_matching_diagnostics"))

    def test_backend_name_validation(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="correlated_null_adaptive_permutation",
                variogram_backend="tensor",
            ).validate()
        config = TestConfig(
            method="correlated_null_adaptive_permutation",
            variogram_backend="gpu",
            variogram_perm_batch_size=4,
        ).validate()
        self.assertEqual(config.variogram_backend, "cuda")
        self.assertEqual(config.variogram_perm_batch_size, 4)

    def test_cpu_backend_defaults_to_one_permutation_per_batch(self) -> None:
        coords = _grid_coords(6)
        expression = np.random.default_rng(0).standard_normal((len(coords), 3))
        _, diag = build_correlated_null_adaptive_surrogates(
            expression,
            coords,
            n_perms=2,
            seed=1,
            nn_candidates=[0.0, 0.3],
            n_bins=6,
            n_plot_genes=1,
            verbose=False,
            backend="cpu",
        )
        self.assertEqual(int(diag["variogram_perm_batch_size"]), 1)

    def test_batched_cpu_matches_sequential_draws_and_fits(self) -> None:
        coords = _grid_coords(8)
        expression = np.random.default_rng(4).normal(2.0, 0.5, (len(coords), 5))
        common = dict(
            n_perms=3,
            seed=8,
            n_bins=8,
            n_plot_genes=3,
            verbose=False,
            collect_trace=True,
            collect_timings=True,
        )
        cases = [
            (
                build_correlated_null_adaptive_surrogates,
                {"nn_candidates": [0.2, 0.0, 0.5]},
            ),
            (
                build_correlated_null_surrogates,
                {"sigma_multiples": [1.0, 2.0]},
            ),
        ]
        for builder, candidates in cases:
            reference, ref_diag = builder(expression, coords, backend="sequential", **common, **candidates)
            cached, cached_diag = builder(
                expression, coords, backend="cpu", perm_batch_size=1, **common, **candidates
            )
            blocked, blocked_diag = builder(
                expression,
                coords,
                backend="cpu",
                perm_batch_size=3,
                weight_cache_bytes=0,
                **common,
                **candidates,
            )
            wide, wide_diag = builder(
                expression, coords, backend="cpu", perm_batch_size=3, **common, **candidates
            )
            for actual, diag in (
                (cached, cached_diag),
                (blocked, blocked_diag),
                (wide, wide_diag),
            ):
                _assert_backend_matches_reference(reference, ref_diag, actual, diag)

    def test_batched_cuda_matches_sequential_within_tolerance(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        coords = _grid_coords(8)
        expression = np.random.default_rng(4).normal(2.0, 0.5, (len(coords), 5))
        common = dict(
            n_perms=3,
            seed=8,
            n_bins=8,
            n_plot_genes=3,
            verbose=False,
            collect_trace=True,
        )
        for fit_mode in ("legacy", "variance_constrained"):
            common["fit_mode"] = fit_mode
            for builder, candidates in (
                (build_correlated_null_adaptive_surrogates, {"nn_candidates": [0.2, 0.0, 0.5]}),
                (build_correlated_null_surrogates, {"sigma_multiples": [1.0, 2.0]}),
            ):
                reference, ref_diag = builder(expression, coords, backend="sequential", **common, **candidates)
                actual, diag = builder(
                    expression,
                    coords,
                    backend="cuda",
                    perm_batch_size=2,
                    **common,
                    **candidates,
                )
                np.testing.assert_array_equal(diag["trace_permutations"], ref_diag["trace_permutations"])
                np.testing.assert_array_equal(diag["trace_noise"], ref_diag["trace_noise"])
                np.testing.assert_allclose(
                    diag["trace_gamma_y"], ref_diag["trace_gamma_y"], rtol=1e-6, atol=1e-6, equal_nan=True
                )
                np.testing.assert_allclose(
                    diag["trace_sse"], ref_diag["trace_sse"], rtol=1e-6, atol=1e-6, equal_nan=True
                )
                np.testing.assert_allclose(
                    diag["trace_coefficient_sse"],
                    ref_diag["trace_coefficient_sse"],
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
                np.testing.assert_allclose(diag["trace_slope"], ref_diag["trace_slope"], rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(
                    diag["trace_intercept"], ref_diag["trace_intercept"], rtol=1e-6, atol=1e-6
                )
                _assert_bandwidths_agree(ref_diag, diag, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
                self.assertEqual(diag["variogram_backend"], "cuda")
                self.assertEqual(diag["variogram_fit_mode"], fit_mode)

    def test_draw_order_is_permutation_then_noise(self) -> None:
        coords = _grid_coords(6)
        expression = np.random.default_rng(1).standard_normal((len(coords), 3))
        _, diag = build_correlated_null_adaptive_surrogates(
            expression,
            coords,
            n_perms=4,
            seed=19,
            nn_candidates=[0.0, 0.4],
            n_bins=6,
            n_plot_genes=2,
            verbose=False,
            backend="cpu",
            perm_batch_size=2,
            collect_trace=True,
        )
        rng = np.random.default_rng(19)
        for perm, noise in zip(diag["trace_permutations"], diag["trace_noise"]):
            np.testing.assert_array_equal(perm, rng.permutation(len(coords)))
            np.testing.assert_array_equal(noise, rng.standard_normal((len(coords), len(coords))))

    def test_realized_ranking_can_differ_from_coefficient_fit(self) -> None:
        coords = _grid_coords(8)
        geom = _build_variogram_geometry(coords, 8)
        gradient = coords[:, 0]
        gradient = (gradient - gradient.mean()) / gradient.std(ddof=0)
        white = np.random.default_rng(0).standard_normal(len(coords))
        white = (white - white.mean()) / white.std(ddof=0)
        checker = (
            ((coords[:, 0].astype(int) + coords[:, 1].astype(int)) % 2) * 2 - 1
        ).astype(np.float64)
        checker = (checker - checker.mean()) / checker.std(ddof=0)

        def _standardize(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            return (values - values.mean()) / values.std(ddof=0)

        # The smooth field fits the target with a positive nugget, so the shared
        # checkerboard noise enters only that candidate and makes its finished
        # variogram worse than the second field.
        target = _standardize(0.7 * gradient + 0.7 * white)[:, None]
        y_smooth = gradient[:, None]
        y_other = _standardize(0.10 * gradient + 0.90 * white)[:, None]
        gamma_x = _classical_variogram(target, geom)
        gamma_y = np.stack([
            _classical_variogram(y_smooth, geom),
            _classical_variogram(y_other, geom),
        ])
        coefficient_sse, intercept, slope = _fit_variogram_sse(gamma_x, gamma_y, geom.valid_bins)
        self.assertGreater(float(intercept[0, 0]), 0.0)
        noise = checker[:, None]
        realized = []
        finalized = []
        for k, y in enumerate((y_smooth, y_other)):
            restored, _gamma, sse = _score_restored_candidate(
                y,
                slope[k],
                intercept[k],
                noise,
                identity=False,
                input_mean=np.zeros(1),
                input_std=np.ones(1),
                gamma_target=gamma_x,
                geom=geom,
            )
            brute = _brute_classical_variogram(restored, geom, geom.pairs, geom.bin_index)
            brute_sse = np.sum((brute[geom.valid_bins] - gamma_x[geom.valid_bins]) ** 2, axis=0)
            np.testing.assert_allclose(sse, brute_sse, rtol=1e-10, atol=1e-10)
            realized.append(float(sse[0]))
            finalized.append(restored[:, 0])
        coefficient_choice = int(np.argmin(coefficient_sse[:, 0]))
        realized_choice = int(np.argmin(realized))
        self.assertEqual(coefficient_choice, 0)
        self.assertEqual(realized_choice, 1)
        self.assertEqual(realized_choice, int(np.argmin(realized)))
        winner = finalized[realized_choice]
        winner_gamma = _brute_classical_variogram(winner, geom, geom.pairs, geom.bin_index)
        winner_sse = float(np.sum((winner_gamma[geom.valid_bins] - gamma_x[geom.valid_bins]) ** 2))
        self.assertAlmostEqual(winner_sse, realized[realized_choice], places=8)

    def test_returned_winner_is_the_scored_finalized_candidate(self) -> None:
        coords = _grid_coords(6)
        expression = np.random.default_rng(21).normal(size=(len(coords), 3))
        expression[:, 0] = coords[:, 0]
        nn_candidates = [0.0, 0.3, 0.7]
        surr, diag = build_correlated_null_adaptive_surrogates(
            expression,
            coords,
            n_perms=2,
            seed=4,
            nn_candidates=nn_candidates,
            n_bins=6,
            n_plot_genes=3,
            verbose=False,
            collect_trace=True,
        )
        _assert_finalized_selection(
            expression,
            coords,
            surr,
            diag,
            candidates=nn_candidates,
            n_bins=6,
            adaptive=True,
        )
        np.testing.assert_allclose(
            surr.mean(axis=1),
            np.broadcast_to(expression.mean(axis=0), surr.mean(axis=1).shape),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            surr.var(axis=1),
            np.broadcast_to(expression.var(axis=0), surr.var(axis=1).shape),
            rtol=1e-5,
            atol=1e-5,
        )
        # One noise matrix per permutation, drawn after that permutation.
        self.assertEqual(diag["trace_noise"].shape[0], 2)

    def test_variance_constrained_coefficients_match_the_clipped_projection(self) -> None:
        variance = np.array([1.5, 0.5, 0.0, 2.0])
        gamma_y = np.array(
            [
                [1.0, 0.2, 0.0, 2.0],
                [2.0, 0.8, 0.0, 2.0],
                [1.5, 0.5, 0.0, 2.0],
            ]
        )
        # Column 0 solves at b=0.5. Column 1 is above 1/v. Column 2 is a
        # constant smooth. Column 3 has q=0, so the denominator vanishes.
        gamma_x = np.array(
            [
                [0.25 + 0.5 * 1.0, 0.0, 0.0, 1.0],
                [0.25 + 0.5 * 2.0, 0.0, 0.0, 1.0],
                [0.25 + 0.5 * 1.5, 0.0, 0.0, 1.0],
            ]
        )
        gamma_x[:, 1] = 1.0 + 10.0 * (gamma_y[:, 1] - variance[1])
        valid = np.ones(3, dtype=bool)
        noise_variance, smooth_multiplier, sse = _fit_variance_constrained_coefficients(
            gamma_x, gamma_y, variance, valid
        )
        np.testing.assert_allclose(smooth_multiplier, [0.5, 2.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(noise_variance, [0.25, 0.0, 1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(noise_variance + smooth_multiplier * variance, 1.0, atol=1e-12)
        fitted = noise_variance + smooth_multiplier * gamma_y
        expected_sse = np.sum((fitted - gamma_x) ** 2, axis=0)
        np.testing.assert_allclose(sse, expected_sse, atol=1e-12)

    def test_constrained_fit_mode_is_explicit_and_legacy_stays_the_default(self) -> None:
        legacy = TestConfig(method="correlated_null_adaptive_permutation").validate()
        self.assertEqual(legacy.variogram_fit_mode, "legacy")
        constrained = TestConfig(
            method="correlated_null_adaptive_permutation",
            variogram_fit_mode="variance_constrained",
        ).validate()
        self.assertEqual(constrained.variogram_fit_mode, "variance_constrained")
        with self.assertRaisesRegex(ValueError, "variogram_fit_mode"):
            TestConfig(
                method="correlated_null_permutation",
                variogram_fit_mode="notebook",
            ).validate()

    def test_constrained_ties_keep_the_earlier_bandwidth(self) -> None:
        best = np.array([np.inf])
        chosen = -1
        for index, score in enumerate((0.4, 0.4, 0.2, 0.2)):
            improved = _strictly_better(np.array([score]), best)
            if index == 0:
                improved = np.array([True])
            if improved[0]:
                best[0] = score
                chosen = index
        self.assertEqual(chosen, 2)

    def test_variance_constrained_winner_restores_input_moments(self) -> None:
        coords = _grid_coords(8)
        expression = np.random.default_rng(4).normal(2.0, 0.5, (len(coords), 4))
        expression[:, -1] = 7.0
        surr, diag = build_correlated_null_adaptive_surrogates(
            expression,
            coords,
            n_perms=2,
            seed=8,
            nn_candidates=[0.0, 0.3, 0.7],
            n_bins=8,
            n_plot_genes=4,
            verbose=False,
            collect_trace=True,
            fit_mode="variance_constrained",
        )
        self.assertEqual(diag["variogram_fit_mode"], "variance_constrained")
        self.assertEqual(int(diag["algorithm_version"]), 6)
        self.assertEqual(diag["selection_objective"], "final_surrogate_empirical_variogram")
        self.assertEqual(diag["variance_matching"], "input_gene_moments")
        np.testing.assert_array_equal(
            diag["trace_best_k"],
            np.argmin(diag["trace_sse"], axis=1),
        )
        _assert_constrained_selection(expression, coords, surr, diag, nn_candidates=[0.0, 0.3, 0.7], n_bins=8)
        nonconstant = expression.std(axis=0, ddof=0) > 0
        np.testing.assert_allclose(diag["component_variance"][nonconstant], 1.0, atol=1e-8)
        np.testing.assert_allclose(
            diag["pre_normalization_variance_ratio"][nonconstant],
            diag["component_variance"][nonconstant] + diag["covariance_variance_contribution"][nonconstant],
            atol=1e-6,
        )
        np.testing.assert_allclose(diag["realized_variance_ratio"][nonconstant], 1.0, atol=1e-6)
        np.testing.assert_allclose(surr.var(axis=1), np.broadcast_to(expression.var(axis=0), (2, 4)), atol=1e-6)
        np.testing.assert_allclose(surr.mean(axis=1), np.broadcast_to(expression.mean(axis=0), (2, 4)), atol=1e-6)
        geom = _build_variogram_geometry(coords, 8)
        # Independent pair-difference calculation validates the analytic final curve.
        gamma = _brute_classical_variogram(surr[0].astype(np.float64), geom, geom.pairs, geom.bin_index)
        indices = diag["plot_gene_indices"]
        np.testing.assert_allclose(diag["variogram_final"], gamma[:, indices], rtol=1e-5, atol=1e-6, equal_nan=True)
        candidate_gamma = diag["realized_variograms_candidates"]
        selected = np.column_stack([candidate_gamma[diag["trace_best_k"][0, g], :, j] for j, g in enumerate(indices)])
        np.testing.assert_allclose(diag["variogram_final"], selected, rtol=1e-6, equal_nan=True)
        actual_sse = np.sum((gamma[geom.valid_bins][:, indices] - diag["variogram_real"][geom.valid_bins])**2, axis=0)
        np.testing.assert_allclose(diag["realized_sse_best_plot"], actual_sse, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(diag["sse_best_plot"], actual_sse, rtol=1e-4, atol=1e-6)
        before_gamma = gamma[:, indices] / diag["final_variogram_scale"][indices]
        before_sse = np.sum((before_gamma[geom.valid_bins] - diag["variogram_real"][geom.valid_bins])**2, axis=0)
        np.testing.assert_allclose(diag["pre_normalization_sse_best_plot"], before_sse, rtol=1e-4, atol=1e-6)
        self.assertEqual(float(diag["component_variance"][-1]), 0.0)
        np.testing.assert_allclose(surr[:, :, -1], 7.0, atol=1e-5)
        for key in (
            "upper_boundary_frequency",
            "pure_noise_frequency",
            "identity_selection_frequency",
            "upper_boundary_label",
        ):
            self.assertIn(key, diag)
        self.assertGreaterEqual(float(diag["identity_selection_frequency"]), 0.0)
        self.assertLessEqual(float(diag["upper_boundary_frequency"]), 1.0)

    def test_constrained_identity_is_the_exact_permutation(self) -> None:
        coords = _grid_coords(8)
        expression = np.random.default_rng(7).normal(5.0, 3.0, (len(coords), 4))
        surr, diag = build_correlated_null_adaptive_surrogates(
            expression,
            coords,
            n_perms=2,
            seed=6,
            nn_candidates=[0.0],
            n_bins=8,
            n_plot_genes=4,
            verbose=False,
            fit_mode="variance_constrained",
        )
        for sample in surr:
            order = np.argsort(sample[:, 0])
            np.testing.assert_allclose(sample[order], expression[np.argsort(expression[:, 0])], atol=1e-6)
        self.assertEqual(float(diag["identity_selection_frequency"]), 1.0)
        self.assertEqual(float(diag["pure_noise_frequency"]), 0.0)

    def test_constrained_field_is_affine_equivariant_after_restoring_moments(self) -> None:
        coords = _grid_coords(8)
        expression = np.random.default_rng(12).standard_normal((len(coords), 4))
        scale = np.array([0.3, 2.0, 7.0, 4.0])
        shift = np.array([10.0, -20.0, 30.0, 5.0])
        kwargs = dict(
            n_perms=2,
            seed=13,
            nn_candidates=[0.0, 0.25, 0.6],
            n_bins=8,
            n_plot_genes=2,
            verbose=False,
            fit_mode="variance_constrained",
        )
        base, diag = build_correlated_null_adaptive_surrogates(expression, coords, **kwargs)
        transformed = expression * scale + shift
        actual, _ = build_correlated_null_adaptive_surrogates(transformed, coords, **kwargs)
        np.testing.assert_allclose(actual, base * scale + shift, atol=1e-5)
        ratio = diag["realized_variance_ratio"]
        self.assertTrue(np.all(np.isfinite(ratio)))

    def test_constrained_tensor_scoring_and_finalization_match_numpy(self) -> None:
        import torch
        from methods.correlated_null_permutation import (
            _score_variance_constrained_candidate, _finalize_constrained_field,
        )
        from methods.variogram_matching_accel import (
            _score_variance_constrained_candidate_torch, _finalize_constrained_field_torch,
            _TorchVariogram,
        )
        coords = _grid_coords(6)
        geom = _build_variogram_geometry(coords, 6)
        x = np.random.default_rng(27).normal(size=(len(coords), 4))
        x[:, -1] = 0.0
        x[:, :-1] = (x[:, :-1] - x[:, :-1].mean(axis=0)) / x[:, :-1].std(axis=0)
        noise = _standardize_noise_columns(np.random.default_rng(28).normal(size=x.shape))
        noise[:, -1] = 0.0
        mean = np.array([2., 3., -5., 7.])
        std = np.array([.5, 3., 2., 0.])
        gamma_x = _classical_variogram(x, geom)
        # Exercise the actual sparse tensor variogram implementation on CPU.
        operator = _TorchVariogram.__new__(_TorchVariogram)
        operator._torch = torch
        operator.pair_counts = geom.pair_counts
        operator._bins, operator._degrees = [], []
        for k, matrix in enumerate(geom.bin_matrices):
            if not geom.valid_bins[k]:
                operator._bins.append(None); operator._degrees.append(None)
                continue
            coo = matrix.tocoo()
            operator._bins.append(torch.sparse_coo_tensor(
                np.vstack([coo.row, coo.col]), coo.data, coo.shape, dtype=torch.float64
            ).coalesce())
            operator._degrees.append(torch.tensor(geom.degrees[k], dtype=torch.float64))
        for identity in (False, True):
            for columns in (None, np.array([0, 3])):
                kwargs = dict(identity=identity, gamma_target=gamma_x * std**2, realized_columns=columns)
                expected = _score_variance_constrained_candidate(
                    x, noise, gamma_x, geom.valid_bins, geom, input_mean=mean, input_std=std, **kwargs
                )
                actual = _score_variance_constrained_candidate_torch(
                    torch.tensor(x), torch.tensor(noise), gamma_x, geom.valid_bins, operator,
                    input_mean=torch.tensor(mean), input_std=torch.tensor(std), **kwargs
                )
                for key in expected:
                    if key.startswith("timing"):
                        continue
                    value = actual[key].numpy() if isinstance(actual[key], torch.Tensor) else actual[key]
                    # At a pure-smooth optimum, O(eps) nugget rounding produces
                    # O(sqrt(eps)) differences in the generated field.
                    tolerance = 5e-7 if key in {"field", "standardized"} else 1e-10
                    np.testing.assert_allclose(value, expected[key], atol=tolerance, rtol=1e-9, equal_nan=True)
                mask = np.full(x.shape[1], identity, dtype=bool)
                expected_field, expected_scale = _finalize_constrained_field(expected["standardized"], mean, std, mask)
                actual_field, actual_scale = _finalize_constrained_field_torch(
                    actual["standardized"], torch.tensor(mean), torch.tensor(std), mask
                )
                np.testing.assert_allclose(actual_field.numpy(), expected_field, atol=5e-7)
                np.testing.assert_allclose(actual_scale, expected_scale, atol=5e-7)

    def test_constrained_backends_match_on_cpu(self) -> None:
        coords = _grid_coords(8)
        expression = np.random.default_rng(4).normal(2.0, 0.5, (len(coords), 5))
        common = dict(
            n_perms=3,
            seed=8,
            n_bins=8,
            n_plot_genes=3,
            verbose=False,
            collect_trace=True,
            fit_mode="variance_constrained",
        )
        cases = [
            (build_correlated_null_adaptive_surrogates, {"nn_candidates": [0.2, 0.0, 0.5]}),
            (build_correlated_null_surrogates, {"sigma_multiples": [1.0, 2.0]}),
        ]
        for builder, candidates in cases:
            reference, ref_diag = builder(expression, coords, backend="sequential", **common, **candidates)
            wide, wide_diag = builder(
                expression, coords, backend="cpu", perm_batch_size=3, **common, **candidates
            )
            blocked, blocked_diag = builder(
                expression,
                coords,
                backend="cpu",
                perm_batch_size=3,
                weight_cache_bytes=0,
                **common,
                **candidates,
            )
            for actual, diag in ((wide, wide_diag), (blocked, blocked_diag)):
                _assert_backend_matches_reference(reference, ref_diag, actual, diag)
                np.testing.assert_allclose(
                    diag["trace_realized_sse"],
                    ref_diag["trace_realized_sse"],
                    rtol=1e-8,
                    atol=1e-8,
                    equal_nan=True,
                )
                self.assertEqual(diag["variogram_fit_mode"], "variance_constrained")
                self.assertEqual(int(diag["algorithm_version"]), 6)

    def test_constrained_and_legacy_diagnostics_both_plot(self) -> None:
        coords = _grid_coords(8)
        expression = np.random.default_rng(1).normal(size=(len(coords), 4))
        _, constrained = build_correlated_null_adaptive_surrogates(
            expression,
            coords,
            n_perms=1,
            seed=2,
            nn_candidates=[0.0, 0.4],
            n_bins=6,
            n_plot_genes=4,
            verbose=False,
            fit_mode="variance_constrained",
        )
        _, legacy = build_correlated_null_adaptive_surrogates(
            expression,
            coords,
            n_perms=1,
            seed=2,
            nn_candidates=[0.0, 0.4],
            n_bins=6,
            n_plot_genes=4,
            verbose=False,
            fit_mode="legacy",
        )
        older = dict(legacy)
        older.pop("variogram_fit_mode", None)
        older["algorithm_version"] = 2
        older.pop("realized_variograms_candidates", None)
        with tempfile.TemporaryDirectory() as tmp:
            constrained_paths = save_variogram_matching_plots(constrained, Path(tmp) / "constrained")
            legacy_paths = save_variogram_matching_plots(legacy, Path(tmp) / "legacy")
            older_paths = save_variogram_matching_plots(older, Path(tmp) / "older")
            self.assertIn("variogram_candidates", constrained_paths)
            self.assertIn("variogram_best_match", legacy_paths)
            self.assertIn("variogram_candidates", older_paths)
            summary = (Path(tmp) / "constrained" / "variogram-matching" / "selected_sigma_summary.csv").read_text()
            self.assertIn("noise_variance", summary)
            self.assertIn("smooth_multiplier", summary)
            self.assertIn("component_variance", summary)
            self.assertIn("final_surrogate_empirical_variogram", summary)
            self.assertIn("upper_boundary_frequency", summary)

    def test_constrained_match_scores_every_returned_candidate(self) -> None:
        coords = _grid_coords(6)
        expression = np.random.default_rng(3).normal(size=(len(coords), 4))
        n_candidates = 2
        n_perms = 4
        batch = 2
        n_plot = 1
        shapes: list[tuple[int, ...]] = []
        real = _classical_variogram

        def _record(expression_columns: np.ndarray, geom):
            values = np.asarray(expression_columns)
            shapes.append(tuple(values.shape))
            return real(values, geom)

        with patch(
            "methods.correlated_null_permutation._classical_variogram", side_effect=_record,
        ), patch(
            "methods.variogram_matching_accel._classical_variogram", side_effect=_record,
        ):
            _surrogates, diag = build_correlated_null_adaptive_surrogates(
                expression,
                coords,
                n_perms=n_perms,
                seed=5,
                nn_candidates=[0.0, 0.4],
                n_bins=6,
                n_plot_genes=n_plot,
                verbose=False,
                backend="cpu",
                perm_batch_size=batch,
                fit_mode="variance_constrained",
            )
        wide = sum(1 for shape in shapes if shape[1] == expression.shape[1] * batch)
        plot = sum(1 for shape in shapes if shape[1] == n_plot)
        # Every bandwidth scores both the smooth field and the actual
        # normalized noisy candidate for all genes, even without traces.
        self.assertEqual(wide, 2 * n_candidates * (n_perms // batch))
        self.assertEqual(plot, 0)
        # Only the input target and slot-0 permutation use an all-gene narrow pass.
        self.assertEqual(sum(shape[1] == expression.shape[1] for shape in shapes), 2)
        valid = np.asarray(diag["valid_bins"], dtype=bool)
        curves = np.asarray(diag["realized_variograms_candidates"])
        self.assertTrue(np.all(np.isfinite(curves[:, valid, :])))
        self.assertTrue(np.all(np.isfinite(diag["realized_sse_candidates"])))
        self.assertEqual(diag["selection_objective"], "final_surrogate_empirical_variogram")


def _assert_constrained_selection(
    expression: np.ndarray,
    coords: np.ndarray,
    surrogates: np.ndarray,
    diagnostics: dict,
    *,
    nn_candidates: list[float],
    n_bins: int,
) -> None:
    expression = np.asarray(expression, dtype=np.float64)
    input_mean = expression.mean(axis=0)
    input_std = expression.std(axis=0, ddof=0)
    standardized = (expression - input_mean) / np.where(input_std > 0, input_std, 1.0)
    geom = _build_variogram_geometry(coords, n_bins)
    gamma_x = _classical_variogram(standardized, geom)
    plan = _precompute_adaptive_blur_plan(coords, nn_candidates)
    n_genes = expression.shape[1]
    n_cells = expression.shape[0]
    for perm_index in range(surrogates.shape[0]):
        perm = diagnostics["trace_permutations"][perm_index]
        noise = diagnostics["trace_noise"][perm_index]
        raw_noise = (noise @ standardized) / np.sqrt(float(n_cells))
        r_noise = _standardize_noise_columns(raw_noise)
        permuted = standardized[perm]
        blurred = _adaptive_blur_expression(permuted, coords, plan)
        theoretical = np.empty((len(nn_candidates), n_genes), dtype=np.float64)
        empirical = np.empty_like(theoretical)
        fields = []
        for k in range(len(nn_candidates)):
            identity = k == int(plan.identity_index)
            if identity:
                smooth = permuted
                variance = smooth.var(axis=0, ddof=0)
                gamma_y = _classical_variogram(smooth, geom)
                noise_variance = np.zeros(n_genes)
                smooth_multiplier = np.ones(n_genes)
                z = np.array(permuted, copy=True)
            else:
                smooth = blurred[k] - blurred[k].mean(axis=0)
                variance = smooth.var(axis=0, ddof=0)
                gamma_y = _classical_variogram(smooth, geom)
                noise_variance, smooth_multiplier, _sse = _fit_variance_constrained_coefficients(
                    gamma_x, gamma_y, variance, geom.valid_bins
                )
                z = np.sqrt(smooth_multiplier) * smooth + np.sqrt(noise_variance) * r_noise
            constant = input_std <= 0
            if np.any(constant):
                z[:, constant] = 0.0
                if not identity:
                    noise_variance = np.array(noise_variance, copy=True)
                    smooth_multiplier = np.array(smooth_multiplier, copy=True)
                    noise_variance[constant] = 0.0
                    smooth_multiplier[constant] = 0.0
            fitted = noise_variance + smooth_multiplier * gamma_y
            theoretical[k] = np.sum((fitted[geom.valid_bins] - gamma_x[geom.valid_bins]) ** 2, axis=0)
            theoretical[k, constant] = 0.0
            np.testing.assert_allclose(
                smooth_multiplier, diagnostics["trace_slope"][perm_index, k], atol=1e-8
            )
            np.testing.assert_allclose(
                noise_variance, diagnostics["trace_intercept"][perm_index, k], atol=1e-8
            )
            centered_z = z - z.mean(axis=0)
            z_std = centered_z.std(axis=0)
            restored = centered_z / np.where(z_std > 0, z_std, 1.0) * input_std + input_mean
            if identity:
                restored = permuted * input_std + input_mean
            fields.append(restored)
            gamma_final = _brute_classical_variogram(restored, geom, geom.pairs, geom.bin_index)
            empirical[k] = np.sum(
                (gamma_final[geom.valid_bins] - (gamma_x * input_std**2)[geom.valid_bins])**2,
                axis=0,
            )
            if not identity:
                contribution = _covariance_variance_contribution(
                    noise_variance, smooth_multiplier, np.mean(smooth * r_noise, axis=0)
                )
                realized = z.var(axis=0, ddof=0)
                component = noise_variance + smooth_multiplier * variance
                active = ~constant
                np.testing.assert_allclose(realized[active], (component + contribution)[active], atol=1e-8)
        np.testing.assert_allclose(theoretical, diagnostics["trace_coefficient_sse"][perm_index], atol=1e-8)
        np.testing.assert_allclose(empirical, diagnostics["trace_sse"][perm_index], atol=1e-8)
        chosen = np.argmin(empirical, axis=0)
        np.testing.assert_array_equal(chosen, diagnostics["trace_best_k"][perm_index])
        winner = np.column_stack([fields[int(chosen[g])][:, g] for g in range(n_genes)])
        np.testing.assert_allclose(surrogates[perm_index], winner, rtol=1e-5, atol=1e-5)


def _assert_finalized_selection(
    expression: np.ndarray,
    coords: np.ndarray,
    surrogates: np.ndarray,
    diagnostics: dict,
    *,
    candidates: list[float],
    n_bins: int,
    adaptive: bool,
) -> None:
    expression = np.asarray(expression, dtype=np.float64)
    input_mean = expression.mean(axis=0)
    input_std = expression.std(axis=0, ddof=0)
    standardized = (expression - input_mean) / np.where(input_std > 0, input_std, 1.0)
    geom = _build_variogram_geometry(coords, n_bins)
    gamma_target = _classical_variogram(standardized, geom) * input_std**2
    plan = _precompute_adaptive_blur_plan(coords, candidates) if adaptive else None
    weights = None
    if not adaptive:
        base = _median_nearest_neighbor_distance(coords)
        weights = [gaussian_smoothing_weights(coords, sigma=float(m) * base) for m in candidates]
    n_genes = expression.shape[1]
    for perm_index in range(surrogates.shape[0]):
        perm = diagnostics["trace_permutations"][perm_index]
        noise = diagnostics["trace_noise"][perm_index]
        r_noise = (noise @ standardized) / np.sqrt(float(len(coords)))
        permuted = standardized[perm]
        if adaptive:
            blurred = _adaptive_blur_expression(permuted, coords, plan)
            identity_index = int(plan.identity_index)
        else:
            blurred = np.stack(
                [np.asarray(apply_gaussian_weights(w, permuted), dtype=np.float64) for w in weights],
                axis=0,
            )
            identity_index = None
        slope = diagnostics["trace_slope"][perm_index]
        intercept = diagnostics["trace_intercept"][perm_index]
        realized = np.empty((len(candidates), n_genes), dtype=np.float64)
        fields = []
        for k in range(len(candidates)):
            restored, _gamma, sse = _score_restored_candidate(
                blurred[k],
                slope[k],
                intercept[k],
                r_noise,
                identity=identity_index is not None and k == identity_index,
                input_mean=input_mean,
                input_std=input_std,
                gamma_target=gamma_target,
                geom=geom,
            )
            brute = _brute_classical_variogram(restored, geom, geom.pairs, geom.bin_index)
            brute_sse = np.sum(
                (brute[geom.valid_bins] - gamma_target[geom.valid_bins]) ** 2, axis=0
            )
            np.testing.assert_allclose(sse, brute_sse, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(
                sse, diagnostics["trace_sse"][perm_index, k], rtol=1e-8, atol=1e-8
            )
            realized[k] = sse
            fields.append(restored)
        chosen = np.argmin(realized, axis=0)
        np.testing.assert_array_equal(chosen, diagnostics["trace_best_k"][perm_index])
        winner = np.column_stack([fields[int(chosen[g])][:, g] for g in range(n_genes)])
        np.testing.assert_allclose(surrogates[perm_index], winner, rtol=1e-5, atol=1e-5)
        winner_gamma = _brute_classical_variogram(surrogates[perm_index], geom, geom.pairs, geom.bin_index)
        winner_sse = np.sum(
            (winner_gamma[geom.valid_bins] - gamma_target[geom.valid_bins]) ** 2, axis=0
        )
        recorded = np.array(
            [
                diagnostics["trace_sse"][perm_index, int(chosen[g]), g]
                for g in range(n_genes)
            ]
        )
        np.testing.assert_allclose(winner_sse, recorded, rtol=1e-5, atol=1e-5)


def _assert_backend_matches_reference(reference, ref_diag, actual, diag) -> None:
    np.testing.assert_array_equal(diag["trace_permutations"], ref_diag["trace_permutations"])
    np.testing.assert_array_equal(diag["trace_noise"], ref_diag["trace_noise"])
    np.testing.assert_allclose(
        diag["trace_gamma_y"], ref_diag["trace_gamma_y"], rtol=1e-8, atol=1e-8, equal_nan=True
    )
    np.testing.assert_allclose(diag["trace_sse"], ref_diag["trace_sse"], rtol=1e-8, atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(
        diag["trace_coefficient_sse"],
        ref_diag["trace_coefficient_sse"],
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    )
    np.testing.assert_allclose(diag["trace_slope"], ref_diag["trace_slope"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(diag["trace_intercept"], ref_diag["trace_intercept"], rtol=1e-8, atol=1e-8)
    _assert_bandwidths_agree(ref_diag, diag, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual.mean(axis=1), reference.mean(axis=1), atol=1e-5)
    np.testing.assert_allclose(actual.var(axis=1), reference.var(axis=1), rtol=1e-5, atol=1e-5)
    for key in (
        "variogram_real",
        "variogram_permuted",
        "variogram_final",
        "fitted_variograms_candidates",
        "realized_variograms_candidates",
        "realized_sse_candidates",
        "slope_best_plot",
        "intercept_best_plot",
        "sse_best_plot",
        "coefficient_sse_best_plot",
        "X_final",
        "Y_blurred_best",
    ):
        np.testing.assert_allclose(diag[key], ref_diag[key], rtol=1e-5, atol=1e-5, equal_nan=True)
    if np.array_equal(diag["trace_best_k"], ref_diag["trace_best_k"]):
        np.testing.assert_array_equal(diag["selection_counts"], ref_diag["selection_counts"])
        np.testing.assert_array_equal(
            diag["best_candidate_per_plot_gene"], ref_diag["best_candidate_per_plot_gene"]
        )


def _assert_bandwidths_agree(ref_diag, diag, *, rtol: float, atol: float) -> None:
    ref_k = np.asarray(ref_diag["trace_best_k"])
    new_k = np.asarray(diag["trace_best_k"])
    if np.array_equal(ref_k, new_k):
        return
    ref_sse = np.asarray(ref_diag["trace_sse"])
    new_sse = np.asarray(diag["trace_sse"])
    for perm, gene in np.argwhere(ref_k != new_k):
        chosen_ref = int(ref_k[perm, gene])
        chosen_new = int(new_k[perm, gene])
        ref_gap = abs(float(ref_sse[perm, chosen_ref, gene] - ref_sse[perm, chosen_new, gene]))
        new_gap = abs(float(new_sse[perm, chosen_ref, gene] - new_sse[perm, chosen_new, gene]))
        scale = max(abs(float(ref_sse[perm, chosen_ref, gene])), 1.0)
        limit = atol + rtol * scale
        if ref_gap > limit or new_gap > limit:
            raise AssertionError(
                f"Bandwidth changed at perm {perm} gene {gene}: "
                f"reference candidate {chosen_ref}, batched candidate {chosen_new}, "
                f"reference SSE gap {ref_gap}, batched SSE gap {new_gap}"
            )


if __name__ == "__main__":
    unittest.main()
