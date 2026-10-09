import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.plots import (
    _bin_mean_series,
    _decoder_fit_stats,
    _expression_label,
    _expression_y_axis_label,
    _gene_ols_slopes,
    _mse_per_gene,
    _quantile_bin_assignments,
    _r2_per_gene,
    _top_genes_by_abs_slope,
    _top_genes_by_decoder_loss,
    compute_isodepth_sig_genes,
    save_combined_celltype_residual_ratio_outputs,
    save_gene_expression_vs_coordinates_comparison,
    save_gene_expression_vs_isodepth_plot,
    save_svg_spatial_expression_plots,
)
from data.schemas import DatasetBundle
from experiments.configuration import _decoder_df_from_config


class GeneExpressionPlotHelpersTests(unittest.TestCase):
    def test_expression_y_axis_label_poisson_raw_counts(self):
        label = _expression_y_axis_label(
            {"log1p": False, "standardize_expression": False}
        )
        self.assertEqual(label, "Expression")

    def test_expression_y_axis_label_gaussian_log_zscore(self):
        label = _expression_y_axis_label(
            {"log1p": True, "standardize_expression": True}
        )
        self.assertEqual(label, "Expression (log₁p, z-scored)")

    def test_expression_y_axis_label_synthetic_gaussian(self):
        meta = {"source": "synthetic", "expression_distribution": "gaussian"}
        self.assertEqual(_expression_y_axis_label(meta), "Expression (z-scored)")

    def test_expression_label_synthetic_gaussian(self):
        meta = {"source": "synthetic", "expression_distribution": "gaussian"}
        self.assertEqual(_expression_label(meta), "Standardized expression")

    def test_expression_label_poisson_raw_counts(self):
        label = _expression_label({"log1p": False, "standardize_expression": False})
        self.assertEqual(label, "Expression (raw counts)")

    def test_decoder_df_from_config_skips_nn(self):
        self.assertIsNone(_decoder_df_from_config("nn"))
        self.assertEqual(_decoder_df_from_config("linear"), 1)
        self.assertEqual(_decoder_df_from_config("quadratic"), 2)

    def test_binned_decoder_curve_is_smoother_than_per_cell_predictions(self):
        rng = np.random.default_rng(0)
        coord = np.linspace(-3.4, -3.15, 120)
        trend = 2.0 + 0.5 * (coord - coord.min()) / (coord.max() - coord.min())
        decoder_preds = trend + rng.normal(0.0, 5.0, size=coord.shape)

        bin_idx, actual_n = _quantile_bin_assignments(coord, n_bins=12)
        centers, means = _bin_mean_series(
            coord, decoder_preds, bin_idx, actual_n, min_bin_cells=3
        )
        self.assertGreaterEqual(len(centers), 4)

        fit_x = np.linspace(float(coord.min()), float(coord.max()), 300)
        deg = min(3, len(centers) - 1)
        poly = np.poly1d(np.polyfit(centers, means, deg))
        fit_y = poly(fit_x)
        per_cell_range = float(np.max(decoder_preds) - np.min(decoder_preds))
        smooth_range = float(np.max(fit_y) - np.min(fit_y))
        self.assertGreater(per_cell_range, 10.0)
        self.assertLess(smooth_range, 0.5 * per_cell_range)

    def test_top_genes_by_decoder_loss_are_lowest_mse(self):
        losses = np.array([0.40, 0.05, 0.20, 0.01, 0.30])
        top = _top_genes_by_decoder_loss(losses, 3)
        np.testing.assert_array_equal(top, np.array([3, 1, 2]))

    def test_decoder_fit_stats_rank_by_pred_true_not_spearman(self):
        rng = np.random.default_rng(1)
        n, g = 80, 6
        iso = np.linspace(-1.0, 1.0, n)
        A = rng.normal(size=(n, g))
        A[:, 0] = 3.0 * iso + rng.normal(scale=0.05, size=n)
        A[:, 1] = rng.normal(size=n)
        A[:, 1] += np.where(np.arange(n) < 4, 8.0, 0.0)
        pred = np.zeros_like(A)
        pred[:, 1] = A[:, 1]
        pred[:, 2] = 0.9 * A[:, 2]
        mse, r2 = _decoder_fit_stats(A, iso, pred, decoder_df=1)
        self.assertIsNotNone(mse)
        top = _top_genes_by_decoder_loss(mse, 2)
        self.assertEqual(int(top[0]), 1)
        self.assertGreater(float(r2[1]), float(r2[0]))
        self.assertLess(float(mse[1]), float(_mse_per_gene(A, pred)[0]))

    def test_gene_ols_slopes_match_linregress(self):
        rng = np.random.default_rng(2)
        n = 50
        iso = np.linspace(-2.0, 2.0, n)
        A = np.column_stack([
            1.5 * iso + 0.4,
            -2.0 * iso,
            rng.normal(scale=0.01, size=n),
        ])
        slopes = _gene_ols_slopes(A, iso)
        self.assertAlmostEqual(float(slopes[0]), 1.5, places=6)
        self.assertAlmostEqual(float(slopes[1]), -2.0, places=6)
        self.assertLess(abs(float(slopes[2])), 0.05)

    def test_top_genes_by_abs_slope_prefer_steep_over_low_mse(self):
        iso = np.linspace(-1.0, 1.0, 80)
        A = np.zeros((80, 4))
        A[:, 0] = 3.0 * iso
        A[:, 1] = -2.5 * iso
        A[:, 2] = 0.01 * iso
        A[:, 3] = 0.5
        slopes = _gene_ols_slopes(A, iso)
        top = _top_genes_by_abs_slope(slopes, 2)
        np.testing.assert_array_equal(top, np.array([0, 1]))
        pred = np.zeros_like(A)
        pred[:, 3] = A[:, 3]
        mse, _ = _decoder_fit_stats(A, iso, pred, decoder_df=1)
        mse_top = _top_genes_by_decoder_loss(mse, 1)
        self.assertEqual(int(mse_top[0]), 3)
        self.assertNotEqual(int(top[0]), int(mse_top[0]))

    def test_isodepth_and_spatial_plots_are_always_written(self):
        rng = np.random.default_rng(3)
        n, g = 40, 6
        iso = np.linspace(-1.0, 1.0, n)
        S = np.column_stack([np.linspace(0.0, 1.0, n), np.linspace(0.0, 0.5, n)])
        A = rng.normal(size=(n, g))
        A[:, 0] = 4.0 * iso
        A[:, 1] = -3.0 * iso
        dataset = DatasetBundle(
            S=S,
            A=A,
            meta={"var_names": [f"g{i}" for i in range(g)]},
        ).validate()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run_gene_expression_vs_isodepth.png"
            save_gene_expression_vs_isodepth_plot(
                dataset,
                iso,
                out,
                n_top_genes=3,
                decoder_preds=A,
                decoder_df=1,
            )
            spatial = out.parent / (out.stem + "_svg_spatial_expression.png")
            self.assertTrue(out.exists(), "1D gene vs isodepth plot was not written")
            self.assertTrue(spatial.exists(), "2D spatial gene scatter was not written")
            self.assertGreater(out.stat().st_size, 0)
            self.assertGreater(spatial.stat().st_size, 0)

    def test_comparison_plot_writes_spatial_companion(self):
        rng = np.random.default_rng(4)
        n, g = 40, 5
        iso = np.linspace(-1.0, 1.0, n)
        cov = np.linspace(1.0, -1.0, n)
        S = np.column_stack([np.linspace(0.0, 1.0, n), rng.normal(size=n)])
        A = rng.normal(size=(n, g))
        A[:, 0] = 2.0 * iso
        A[:, 1] = 2.0 * cov
        dataset = DatasetBundle(
            S=S,
            A=A,
            meta={"var_names": [f"g{i}" for i in range(g)]},
        ).validate()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run_gene_expression_vs_coordinates.png"
            save_gene_expression_vs_coordinates_comparison(
                dataset,
                iso,
                cov,
                out,
                n_top_genes=2,
                pred_isodepth=A,
                pred_covariate=A,
                decoder_df=1,
            )
            spatial = out.parent / (out.stem + "_svg_spatial_expression.png")
            self.assertTrue(out.exists())
            self.assertTrue(spatial.exists())

    def test_top_genes_by_abs_slope_among_restricts_to_subset(self):
        slopes = np.array([3.0, 0.1, -2.5, 0.4])
        among = np.array([1, 2, 3])
        top = _top_genes_by_abs_slope(slopes, 2, among=among)
        np.testing.assert_array_equal(top, np.array([2, 3]))

    def _steep_nonsig_vs_moderate_sig_bundle(self):
        rng = np.random.default_rng(5)
        n, g = 80, 5
        iso = np.linspace(-1.0, 1.0, n)
        S = np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])
        A = rng.normal(scale=0.05, size=(n, g))
        A[:, 0] = 5.0 * iso
        A[:, 1] = 1.2 * iso
        A[:, 2] = -1.0 * iso
        pred = np.zeros_like(A)
        pred[:, 1] = A[:, 1]
        pred[:, 2] = A[:, 2]
        names = [f"g{i}" for i in range(g)]
        dataset = DatasetBundle(S=S, A=A, meta={"var_names": names}).validate()
        return dataset, iso, pred, names

    def test_significant_only_ranking_drops_steep_nonsignificant_gene(self):
        dataset, iso, pred, names = self._steep_nonsig_vs_moderate_sig_bundle()
        slopes = _gene_ols_slopes(dataset.A, iso)
        unrestricted = _top_genes_by_abs_slope(slopes, 3)
        self.assertEqual(int(unrestricted[0]), 0)

        svg = compute_isodepth_sig_genes(
            dataset.A, names, pred, 1, coord=iso, alpha=0.05,
        )
        self.assertNotIn(0, set(int(i) for i in svg["sig_indices"]))
        self.assertIn(1, set(int(i) for i in svg["sig_indices"]))
        self.assertIn(2, set(int(i) for i in svg["sig_indices"]))
        top = _top_genes_by_abs_slope(slopes, 5, among=svg["sig_indices"])
        self.assertNotIn(0, set(int(i) for i in top))
        self.assertEqual(int(top[0]), 1)
        self.assertEqual(int(top[1]), 2)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run_gene_expression_vs_isodepth.png"
            save_gene_expression_vs_isodepth_plot(
                dataset, iso, out, n_top_genes=5,
                decoder_preds=pred, decoder_df=1,
            )
            csv_path = out.parent / (out.stem + "_isodepth_sig_genes.csv")
            self.assertTrue(out.exists())
            self.assertTrue(csv_path.exists())
            sig_names = [row.split(",")[0] for row in csv_path.read_text().splitlines()[1:]]
            self.assertIn("g1", sig_names)
            self.assertIn("g2", sig_names)
            self.assertNotIn("g0", sig_names)

    def test_significant_only_false_keeps_steep_nonsignificant_gene(self):
        dataset, iso, pred, names = self._steep_nonsig_vs_moderate_sig_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run_gene_expression_vs_isodepth.png"
            save_gene_expression_vs_isodepth_plot(
                dataset, iso, out, n_top_genes=3,
                decoder_preds=pred, decoder_df=1,
                significant_only=False,
            )
            self.assertTrue(out.exists())
            slopes = _gene_ols_slopes(dataset.A, iso)
            top = _top_genes_by_abs_slope(slopes, 3)
            self.assertEqual(int(top[0]), 0)

    def test_no_significant_genes_writes_placeholder_plots(self):
        rng = np.random.default_rng(6)
        n, g = 40, 4
        iso = np.linspace(-1.0, 1.0, n)
        S = np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])
        A = rng.normal(scale=0.05, size=(n, g))
        pred = np.zeros_like(A)
        dataset = DatasetBundle(
            S=S, A=A, meta={"var_names": [f"g{i}" for i in range(g)]},
        ).validate()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run_gene_expression_vs_isodepth.png"
            save_gene_expression_vs_isodepth_plot(
                dataset, iso, out, n_top_genes=5,
                decoder_preds=pred, decoder_df=1,
            )
            spatial = out.parent / (out.stem + "_svg_spatial_expression.png")
            csv_path = out.parent / (out.stem + "_isodepth_sig_genes.csv")
            self.assertTrue(out.exists())
            self.assertTrue(spatial.exists())
            self.assertGreater(out.stat().st_size, 0)
            self.assertGreater(spatial.stat().st_size, 0)
            rows = csv_path.read_text().splitlines()
            self.assertEqual(rows[0], "gene,p_value,q_value")
            self.assertEqual(len(rows), 1)

    def test_spatial_expression_plot_accepts_clipped_color_limits(self):
        rng = np.random.default_rng(7)
        n, g = 30, 3
        S = rng.normal(size=(n, 2))
        A = rng.normal(size=(n, g))
        A[0, 0] = -12.0
        names = [f"g{i}" for i in range(g)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "clip_spatial.png"
            save_svg_spatial_expression_plots(
                S, A, names, np.array([0, 1, 2]), out, vmin=-2.0, vmax=2.0,
            )
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_combined_residual_ratio_aligns_per_type_gene_sets(self):
        per_type = {
            "type0": {
                "A": np.array([[2.0, 4.0]], dtype=np.float64),
                "pred_true": np.array([[0.0, 0.0]], dtype=np.float64),
                "pred_true_covariate": np.array([[1.0, 2.0]], dtype=np.float64),
                "var_names": ["geneA", "geneB"],
            },
            "type1": {
                "A": np.array([[3.0, 5.0]], dtype=np.float64),
                "pred_true": np.array([[0.0, 0.0]], dtype=np.float64),
                "pred_true_covariate": np.array([[0.0, 0.0]], dtype=np.float64),
                "var_names": ["geneB", "geneC"],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, plot_path = save_combined_celltype_residual_ratio_outputs(
                per_type,
                ["type0", "type1"],
                ["unused"],
                Path(tmp) / "ratio.csv",
                Path(tmp) / "ratio.png",
            )
            self.assertIsNotNone(csv_path)
            self.assertTrue(csv_path.exists())
            self.assertIsNotNone(plot_path)
            rows = {
                row[1]: row
                for row in csv.reader(csv_path.read_text().splitlines()[1:])
            }
            self.assertEqual(set(rows), {"geneA", "geneB", "geneC"})
            self.assertAlmostEqual(float(rows["geneA"][2]), 1.0)
            self.assertAlmostEqual(float(rows["geneA"][3]), 4.0)
            self.assertAlmostEqual(float(rows["geneB"][2]), 13.0)
            self.assertAlmostEqual(float(rows["geneB"][3]), 25.0)
            self.assertAlmostEqual(float(rows["geneC"][2]), 25.0)
            self.assertAlmostEqual(float(rows["geneC"][3]), 25.0)
            self.assertEqual(int(rows["geneA"][5]), 2)
            self.assertEqual(int(rows["geneA"][6]), 2)


if __name__ == "__main__":
    unittest.main()
