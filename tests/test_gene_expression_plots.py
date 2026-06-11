import unittest

import numpy as np

from analysis.plots import (
    _bin_mean_series,
    _expression_y_axis_label,
    _quantile_bin_assignments,
)
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


if __name__ == "__main__":
    unittest.main()
