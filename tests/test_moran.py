"""Tests for Moran's I permutation diagnostics."""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from data.schemas import TestConfig
from methods.moran import (
    build_inverse_distance_weights,
    compute_moran_permutation_diagnostics,
    maybe_compute_moran_artifacts,
    morans_i_per_gene,
    summarize_moran_slots,
)


class MoranDiagnosticsTests(unittest.TestCase):
    def test_morans_i_computed_on_neighbor_graph(self) -> None:
        S_um = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float64)
        W, s0 = build_inverse_distance_weights(S_um, radius_um=30.0)
        A = np.array([[1.0], [1.0], [0.0]], dtype=np.float64)
        I = morans_i_per_gene(W, s0, A)
        self.assertEqual(I.shape, (1,))
        self.assertTrue(np.isfinite(I[0]))

    def test_summarize_moran_slots_empirical_p_value(self) -> None:
        I_by_slot = np.array(
            [
                [0.8, 0.7],
                [0.1, 0.2],
                [0.15, 0.25],
            ],
            dtype=np.float64,
        )
        summary = summarize_moran_slots(I_by_slot)
        self.assertAlmostEqual(summary["moran_true_mean"], 0.75)
        self.assertEqual(summary["moran_n_perms"], 2)
        self.assertEqual(summary["moran_p_value"], 1.0)

    def test_maybe_compute_moran_disabled_returns_empty(self) -> None:
        config = TestConfig(moran=False, n_perms=2)
        out = maybe_compute_moran_artifacts(
            config,
            {"coordinate_um_per_unit": 1.0},
            np.zeros((3, 4, 2), dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
        )
        self.assertEqual(out, {})

    def test_compute_moran_fixed_coords_expression_slots(self) -> None:
        S_native = np.tile(
            np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)[np.newaxis, :, :],
            (3, 1, 1),
        )
        a_batched = np.array(
            [
                [[0.0], [0.0], [1.0]],
                [[1.0], [0.0], [0.0]],
                [[0.5], [0.5], [0.5]],
            ],
            dtype=np.float32,
        )
        summary = compute_moran_permutation_diagnostics(
            S_native,
            a_batched,
            um_per_unit=1.0,
            neighbor_radius_um=30.0,
        )
        self.assertEqual(summary["moran_i_per_gene_per_slot"].shape, (3, 1))
        self.assertEqual(summary["moran_n_perms"], 2)
        self.assertEqual(summary["moran_n_slots"], 3)

    def test_maybe_compute_skips_when_cells_too_sparse(self) -> None:
        config = TestConfig(moran=True, n_perms=2, verbose=False)
        s = np.tile(
            np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)[np.newaxis, :, :],
            (3, 1, 1),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = maybe_compute_moran_artifacts(
                config,
                {"coordinate_um_per_unit": 1.0},
                s,
                np.zeros((2, 1), dtype=np.float32),
            )
        self.assertTrue(out.get("moran_skipped"))
        self.assertIn("No cell pairs", out.get("moran_skip_reason", ""))
        self.assertEqual(len(caught), 1)

    def test_build_weights_raises_when_not_allow_empty(self) -> None:
        S_um = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            build_inverse_distance_weights(S_um, radius_um=30.0, allow_empty=False)

    def test_schema_moran_radius_validation(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(moran=True, moran_neighbor_radius_um=0.0).validate()


if __name__ == "__main__":
    unittest.main()
