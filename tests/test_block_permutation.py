from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import TestConfig
from methods.block_permutation import (
    assign_block_ids,
    block_centroid_permute,
    block_stats,
    block_ids_to_axial_qr,
    block_ids_to_square_ij,
    build_block_permuted_coordinate_batch,
    hex_bin_ids,
    hex_center_coord,
    hex_polygons_for_block_ids,
    square_bin_ids,
    square_polygons_for_block_ids,
    square_block_grid_line_segments,
)


class TestBlockPermutation(unittest.TestCase):
    def test_block_radius_required_for_method(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="block_permutation", n_perms=5).validate()

    def test_block_radius_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="block_permutation", n_perms=5, block_radius=0.0).validate()

    def test_valid_block_permutation_config(self) -> None:
        config = TestConfig(
            method="block_permutation",
            n_perms=5,
            block_radius=50.0,
            block_shape="hexagon",
            coordinate_um_per_unit=1.0,
        )
        self.assertIs(config.validate(), config)

    def test_invalid_block_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="block_permutation",
                n_perms=5,
                block_radius=50.0,
                block_shape="triangle",
            ).validate()

    def test_block_shape_defaults_to_hexagon(self) -> None:
        config = TestConfig(
            method="block_permutation",
            n_perms=5,
            block_radius=50.0,
        )
        self.assertEqual(config.block_shape, "hexagon")
        self.assertIs(config.validate(), config)

    def test_hex_bin_ids_are_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 500, size=(200, 2)).astype(np.float64)
        ids_a = hex_bin_ids(coords, 50.0, (0.0, 0.0))
        ids_b = hex_bin_ids(coords, 50.0, (0.0, 0.0))
        np.testing.assert_array_equal(ids_a, ids_b)

    def test_jitter_changes_block_assignments(self) -> None:
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 500, size=(200, 2)).astype(np.float64)
        ids_no_jitter = hex_bin_ids(coords, 50.0, (0.0, 0.0))
        ids_jitter = hex_bin_ids(coords, 50.0, (12.3, -7.8))
        self.assertFalse(np.array_equal(ids_no_jitter, ids_jitter))

    def test_square_bin_ids_are_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 500, size=(200, 2)).astype(np.float64)
        ids_a = square_bin_ids(coords, 50.0, (0.0, 0.0))
        ids_b = square_bin_ids(coords, 50.0, (0.0, 0.0))
        np.testing.assert_array_equal(ids_a, ids_b)

    def test_square_and_hex_assignments_differ(self) -> None:
        rng = np.random.default_rng(9)
        coords = rng.uniform(0, 500, size=(200, 2)).astype(np.float64)
        hex_ids = assign_block_ids(coords, 50.0, block_shape="hexagon")
        square_ids = assign_block_ids(coords, 50.0, block_shape="square")
        self.assertFalse(np.array_equal(hex_ids, square_ids))

    def test_square_mesh_polygons_match_bin_centers(self) -> None:
        radius = 50.0
        side = 2.0 * radius
        ix_vals = np.array([0, 1, 2], dtype=np.int64)
        iy_vals = np.array([0, 1, -1], dtype=np.int64)
        ix, iy = np.meshgrid(ix_vals, iy_vals, indexing="ij")
        ix = ix.ravel()
        iy = iy.ravel()
        cx = (ix.astype(np.float64) + 0.5) * side
        cy = (iy.astype(np.float64) + 0.5) * side
        centers = np.column_stack([cx, cy])
        block_ids = square_bin_ids(centers, radius, (0.0, 0.0))
        polys = square_polygons_for_block_ids(block_ids, radius)
        self.assertEqual(len(polys), len(np.unique(block_ids)))
        ix_dec, iy_dec = block_ids_to_square_ij(block_ids)
        np.testing.assert_array_equal(ix_dec, ix)
        np.testing.assert_array_equal(iy_dec, iy)

    def test_square_block_grid_lines_tile_unit_square(self) -> None:
        radius = 30.0 / 960.0
        side = 2.0 * radius
        lines = square_block_grid_line_segments(
            radius,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        )
        self.assertEqual(len(lines), 34)
        xs = sorted({float(seg[0, 0]) for seg in lines if seg[0, 0] == seg[1, 0]})
        expected_xs = [k * side for k in range(17)]
        np.testing.assert_allclose(xs, expected_xs, rtol=0.0, atol=1e-12)

    def test_centroid_permute_preserves_intra_block_offsets(self) -> None:
        rng = np.random.default_rng(2)
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [100.0, 100.0],
                [101.0, 100.0],
                [100.0, 101.0],
            ],
            dtype=np.float64,
        )
        block_ids = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
        permuted = block_centroid_permute(coords, block_ids, rng)
        for block in (1, 2):
            mask = block_ids == block
            offsets_before = coords[mask] - coords[mask].mean(axis=0)
            offsets_after = permuted[mask] - permuted[mask].mean(axis=0)
            np.testing.assert_allclose(offsets_before, offsets_after, atol=1e-5)

    def test_build_batch_slot_zero_is_true_coordinates(self) -> None:
        rng = np.random.default_rng(3)
        S = rng.uniform(0, 200, size=(80, 2)).astype(np.float32)
        batch = build_block_permuted_coordinate_batch(
            S,
            radius_um=50.0,
            coordinate_um_per_unit=1.0,
            n_perms=3,
            seed=42,
            block_jitter=True,
        )
        self.assertEqual(batch.shape, (4, S.shape[0], 2))
        np.testing.assert_array_equal(batch[0], S)

    def test_build_batch_is_reproducible(self) -> None:
        rng = np.random.default_rng(4)
        S = rng.uniform(0, 200, size=(80, 2)).astype(np.float32)
        kwargs = dict(
            radius_um=50.0,
            coordinate_um_per_unit=1.0,
            n_perms=2,
            seed=99,
            block_jitter=True,
        )
        batch_a = build_block_permuted_coordinate_batch(S, **kwargs)
        batch_b = build_block_permuted_coordinate_batch(S, **kwargs)
        np.testing.assert_array_equal(batch_a, batch_b)

    def test_per_cell_type_meshes_are_independent(self) -> None:
        rng = np.random.default_rng(5)
        S = rng.uniform(0, 200, size=(60, 2)).astype(np.float32)
        labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)
        batch = build_block_permuted_coordinate_batch(
            S,
            radius_um=50.0,
            coordinate_um_per_unit=1.0,
            n_perms=1,
            seed=7,
            cell_type_labels=labels,
            n_cell_types=2,
            block_jitter=True,
        )
        self.assertEqual(batch.shape[0], 2)
        self.assertFalse(np.allclose(batch[1], S))

    def test_standardized_coords_round_trip(self) -> None:
        rng = np.random.default_rng(8)
        S_raw = rng.uniform(9000, 18000, size=(100, 2)).astype(np.float32)
        mean = S_raw.mean(axis=0)
        std = S_raw.std(axis=0)
        safe_std = np.where(std > 1e-8, std, 1.0)
        S_std = ((S_raw - mean) / safe_std).astype(np.float32)
        meta = {
            "coordinate_standardization": "zscore",
            "coord_mean": mean.astype(np.float32),
            "coord_std": std.astype(np.float32),
        }
        from data import raw_coordinates_from_standardized, standardize_coordinate_batch

        recovered = raw_coordinates_from_standardized(S_std, meta)
        np.testing.assert_allclose(recovered, S_raw, rtol=1e-5, atol=1e-4)
        batch_raw = np.stack([S_raw, S_raw + 10.0], axis=0)
        batch_std = standardize_coordinate_batch(batch_raw, meta)
        np.testing.assert_allclose(batch_std[0], S_std, rtol=1e-5, atol=1e-4)

    def test_hex_mesh_polygons_match_bin_centers(self) -> None:
        radius = 50.0
        q_vals = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
        r_vals = np.array([0, 1, -1, 2, -2], dtype=np.int64)
        cx, cy = hex_center_coord(q_vals, r_vals, radius)
        centers = np.column_stack([cx, cy])
        block_ids = hex_bin_ids(centers, radius, (0.0, 0.0))
        polys = hex_polygons_for_block_ids(block_ids, radius)
        self.assertEqual(len(polys), len(np.unique(block_ids)))
        q_dec, r_dec = block_ids_to_axial_qr(block_ids)
        cx_dec, cy_dec = hex_center_coord(q_dec, r_dec, radius)
        np.testing.assert_allclose(cx_dec, cx, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(cy_dec, cy, rtol=1e-5, atol=1e-4)

    def test_block_stats_reports_counts(self) -> None:
        rng = np.random.default_rng(6)
        S = rng.uniform(0, 200, size=(120, 2)).astype(np.float32)
        stats = block_stats(S, radius_um=50.0, coordinate_um_per_unit=1.0)
        self.assertGreater(stats["n_blocks"], 0)
        self.assertIn("mean_cells", stats)

    def test_separate_celltype_mode_runs(self) -> None:
        """block_permutation cell_type='separate' should not raise NotImplementedError."""
        import torch
        from data.schemas import DatasetBundle, TestConfig
        from methods.permutation import run_block_permutation_method

        rng = np.random.default_rng(99)
        n_cells, n_genes, n_types = 120, 8, 3
        S_raw = rng.uniform(0, 300, size=(n_cells, 2)).astype(np.float32)
        mean_s = S_raw.mean(axis=0)
        std_s = np.maximum(S_raw.std(axis=0), 1e-8)
        S = ((S_raw - mean_s) / std_s).astype(np.float32)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        labels = (np.arange(n_cells) % n_types).astype(np.int64)
        dataset = DatasetBundle(
            S=S,
            A=A,
            meta={
                "cell_type_mode": "separate",
                "cell_type_labels": labels,
                "cell_type_names": [f"type{i}" for i in range(n_types)],
                "n_cell_types": n_types,
                "coordinate_standardization": "zscore",
                "coord_mean": mean_s.astype(np.float32),
                "coord_std": std_s.astype(np.float32),
                "var_names": [f"gene{i}" for i in range(n_genes)],
            },
        ).validate()
        config = TestConfig(
            method="block_permutation",
            n_perms=3,
            epochs=5,
            n_reruns=1,
            block_radius=80.0,
            coordinate_um_per_unit=1.0,
            block_jitter=False,
            seed=7,
            device="cpu",
            verbose=False,
        ).validate()
        result = run_block_permutation_method(dataset, config)
        self.assertEqual(result.method_name, "block_permutation")
        self.assertIn("per_type_results", result.artifacts)
        self.assertEqual(result.artifacts["cell_type_mode"], "separate")
        per_type = result.artifacts["per_type_results"]
        self.assertEqual(set(per_type.keys()), {f"type{i}" for i in range(n_types)})
        for type_name, tr in per_type.items():
            self.assertIn("p_value", tr)
            self.assertIn("stat_perm", tr)
            self.assertEqual(len(tr["stat_perm"]), 3)
            self.assertIn("S_raw", tr)
            self.assertIn("block_ids_true", tr)
            self.assertIn("s_permuted_slot1_raw", tr)
            self.assertIn("block_radius_units", tr)

    def test_separate_block_permutation_saves_overlay_plots(self) -> None:
        import tempfile

        from data.schemas import DataConfig, DatasetBundle, OutputConfig, RunConfig
        from experiments.configuration import save_standardized_outputs
        from methods.permutation import run_block_permutation_method

        rng = np.random.default_rng(11)
        n_cells, n_genes, n_types = 60, 6, 2
        S_raw = rng.uniform(0, 300, size=(n_cells, 2)).astype(np.float32)
        mean_s = S_raw.mean(axis=0)
        std_s = np.maximum(S_raw.std(axis=0), 1e-8)
        S = ((S_raw - mean_s) / std_s).astype(np.float32)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        labels = (np.arange(n_cells) % n_types).astype(np.int64)
        dataset = DatasetBundle(
            S=S,
            A=A,
            meta={
                "cell_type_mode": "separate",
                "cell_type_labels": labels,
                "cell_type_names": [f"type{i}" for i in range(n_types)],
                "n_cell_types": n_types,
                "coordinate_standardization": "zscore",
                "coord_mean": mean_s.astype(np.float32),
                "coord_std": std_s.astype(np.float32),
                "var_names": [f"gene{i}" for i in range(n_genes)],
            },
        ).validate()
        config = TestConfig(
            method="block_permutation",
            n_perms=2,
            epochs=2,
            n_reruns=1,
            sgd_batch_size=16,
            block_radius=80.0,
            coordinate_um_per_unit=1.0,
            block_jitter=False,
            save_permutation_null_comparison=True,
            seed=11,
            device="cpu",
            verbose=False,
        ).validate()
        result = run_block_permutation_method(dataset, config)
        with tempfile.TemporaryDirectory() as tmp:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", n_cells=n_cells, n_genes=n_genes),
                test=config,
                output=OutputConfig(out_dir=tmp, run_name="blk_sep"),
            )
            save_standardized_outputs(dataset, result, run_config)
            run_dir = Path(tmp) / "blk_sep"
            self.assertTrue((run_dir / "blk_sep_block_permutation_overlay.png").exists())
            self.assertTrue((run_dir / "blk_sep_permutation_null_comparison.png").exists())
            for type_name in result.artifacts["per_type_results"]:
                safe = type_name.replace(" ", "_").replace("/", "_")
                self.assertTrue(
                    (run_dir / safe / f"{safe}_block_permutation_overlay.png").exists()
                )
                self.assertTrue(
                    (run_dir / safe / f"{safe}_permutation_null_comparison.png").exists()
                )

    def test_save_permutation_null_comparison_disabled_by_default(self) -> None:
        import tempfile

        from data.schemas import DataConfig, DatasetBundle, OutputConfig, RunConfig
        from experiments.configuration import save_standardized_outputs
        from methods.permutation import run_block_permutation_method

        rng = np.random.default_rng(12)
        n_cells, n_genes = 40, 5
        S = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        dataset = DatasetBundle(S=S, A=A, meta={"var_names": [f"g{i}" for i in range(n_genes)]}).validate()
        config = TestConfig(
            method="block_permutation",
            n_perms=2,
            epochs=2,
            n_reruns=1,
            sgd_batch_size=16,
            block_radius=50.0,
            coordinate_um_per_unit=1.0,
            block_jitter=False,
            seed=12,
            device="cpu",
            verbose=False,
        ).validate()
        self.assertFalse(config.save_permutation_null_comparison)
        result = run_block_permutation_method(dataset, config)
        with tempfile.TemporaryDirectory() as tmp:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", n_cells=n_cells, n_genes=n_genes),
                test=config,
                output=OutputConfig(out_dir=tmp, run_name="blk_off"),
            )
            save_standardized_outputs(dataset, result, run_config)
            run_dir = Path(tmp) / "blk_off"
            self.assertFalse((run_dir / "blk_off_permutation_null_comparison.png").exists())

    def test_global_coordinate_permute_slot_matches_batch(self) -> None:
        import torch

        from methods.permutation import _build_permuted_coordinate_batch, global_coordinate_permute_slot

        rng = np.random.default_rng(7)
        S = rng.uniform(0, 100, size=(40, 2)).astype(np.float32)
        seed = 123
        batch, _ = _build_permuted_coordinate_batch(
            S, n_perms=3, seed=seed, device=torch.device("cpu")
        )
        slot1 = global_coordinate_permute_slot(S, seed=seed, slot=1)
        np.testing.assert_allclose(slot1, np.asarray(batch[1], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
