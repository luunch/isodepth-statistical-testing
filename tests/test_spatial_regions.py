from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.plots import save_spatial_region_split_plot
from data.schemas import DataConfig, DatasetBundle, run_config_from_mapping
from data.spatial_regions import (
    dbscan_middle_region_mask,
    drop_label_border_spots,
    label_border_spot_mask,
    split_spatial_regions,
)


class SpatialRegionSplitTests(unittest.TestCase):
    def _two_blob_dataset(self) -> DatasetBundle:
        rng = np.random.default_rng(0)
        left = rng.normal(loc=(-2.0, 0.0), scale=0.15, size=(40, 2))
        right = rng.normal(loc=(2.0, 0.0), scale=0.15, size=(40, 2))
        noise = rng.uniform(-4.0, 4.0, size=(8, 2))
        S = np.vstack([left, right, noise]).astype(np.float64)
        A = rng.normal(size=(S.shape[0], 5)).astype(np.float64)
        return DatasetBundle(
            S=S,
            A=A,
            meta={
                "cell_type_labels": np.zeros(len(S), dtype=np.int64),
                "cell_type_names": ["region"],
                "n_cell_types": 1,
            },
        )

    def test_dbscan_diag_and_plot(self) -> None:
        dataset = self._two_blob_dataset()
        config = DataConfig(
            source="h5ad",
            h5ad="dummy.h5ad",
            spatial_region_split=True,
            spatial_region_split_eps=0.5,
            spatial_region_split_min_samples=3,
            spatial_region_split_min_cells=10,
            cell_type="separate",
        )
        n_before = len(dataset.S)
        out = split_spatial_regions(dataset, config)

        diag = out.meta.get("spatial_region_split_diag")
        self.assertIsNotNone(diag)
        assert diag is not None
        self.assertEqual(diag["S"].shape[0], n_before)
        self.assertTrue(np.any(diag["removed"]))
        self.assertGreater(len(diag["region_color_names"]), 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_spatial_region_split_plot(
                out,
                Path(tmp) / "spatial_split.png",
            )
            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.exists())

    def test_dbscan_middle_region_selects_central_band(self) -> None:
        rng = np.random.default_rng(0)
        left = rng.normal(loc=(-2.0, 0.0), scale=0.12, size=(30, 2))
        middle = rng.normal(loc=(0.0, 0.0), scale=0.12, size=(60, 2))
        right = rng.normal(loc=(2.0, 0.0), scale=0.12, size=(30, 2))
        xy = np.vstack([left, middle, right]).astype(np.float64)

        mask, diag = dbscan_middle_region_mask(
            xy,
            eps=0.45,
            min_samples=3,
            min_cells=10,
            axis="x",
        )
        self.assertEqual(int(diag["n_clusters_valid"]), 3)
        self.assertEqual(int(mask.sum()), 60)
        self.assertTrue(np.allclose(xy[mask].mean(axis=0), (0.0, 0.0), atol=0.15))


def _layer_grid(columns_per_label: tuple[int, ...]) -> DatasetBundle:
    """Unit-spacing grid. Column blocks receive successive integer labels."""
    coords = []
    labels = []
    x = 0
    for label, width in enumerate(columns_per_label):
        for _ in range(width):
            for y in range(3):
                coords.append((float(x), float(y)))
                labels.append(label)
            x += 1
    s = np.asarray(coords, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    return DatasetBundle(
        S=s,
        A=np.ones((len(s), 2), dtype=np.float32),
        meta={
            "cell_type_labels": labels_arr,
            "cell_type_names": [f"L{i}" for i in range(len(columns_per_label))],
            "n_cell_types": len(columns_per_label),
            "covariate_values": np.arange(len(s), dtype=np.float32),
        },
    )


class LabelBorderSpotTests(unittest.TestCase):
    def test_interface_spots_are_marked_and_interiors_kept(self) -> None:
        dataset = _layer_grid((2, 2))
        border, pitch, radius = label_border_spot_mask(
            dataset.S, dataset.meta["cell_type_labels"], radius_mult=1.2
        )
        self.assertAlmostEqual(pitch, 1.0)
        self.assertAlmostEqual(radius, 1.2)
        # Facing columns only. Diagonal distance sqrt(2) is outside the radius.
        self.assertEqual(int(border.sum()), 6)
        kept_x = dataset.S[~border, 0]
        self.assertTrue(np.array_equal(np.unique(kept_x), np.array([0.0, 3.0])))

    def test_uniform_labels_drop_nothing(self) -> None:
        dataset = _layer_grid((4,))
        config = DataConfig(
            source="h5ad",
            h5ad="dummy.h5ad",
            cell_type="separate",
            drop_label_border_spots=True,
        )
        out = drop_label_border_spots(dataset, config)
        self.assertEqual(out.S.shape[0], 12)
        self.assertEqual(out.meta["label_border"]["n_dropped"], 0)
        self.assertEqual(out.meta["cell_type_names"], ["L0"])

    def test_border_drop_removes_interface_and_realigns_covariates(self) -> None:
        dataset = _layer_grid((2, 2))
        config = DataConfig(
            source="h5ad",
            h5ad="dummy.h5ad",
            cell_type="separate",
            drop_label_border_spots=True,
            min_cells_per_celltype=1,
        )
        out = drop_label_border_spots(dataset, config)
        self.assertEqual(out.S.shape[0], 6)
        self.assertEqual(out.A.shape[0], 6)
        self.assertTrue(np.array_equal(out.meta["covariate_values"], np.array([0, 1, 2, 9, 10, 11])))
        self.assertEqual(out.meta["label_border"]["kept_by_label"], {"L0": 3, "L1": 3})
        self.assertEqual(out.meta["n_cell_types"], 2)

    def test_layer_left_below_min_cells_is_removed(self) -> None:
        dataset = _layer_grid((2, 1))
        config = DataConfig(
            source="h5ad",
            h5ad="dummy.h5ad",
            cell_type="separate",
            drop_label_border_spots=True,
            min_cells_per_celltype=1,
        )
        out = drop_label_border_spots(dataset, config)
        self.assertEqual(out.meta["cell_type_names"], ["L0"])
        self.assertTrue(np.all(out.meta["cell_type_labels"] == 0))
        self.assertEqual(out.meta["label_border"]["removed_labels"], ["L1"])
        self.assertEqual(out.S.shape[0], 3)

    def test_config_requires_labeled_h5ad(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(
                source="synthetic",
                drop_label_border_spots=True,
                cell_type="separate",
            ).validate()
        with self.assertRaises(ValueError):
            DataConfig(
                source="h5ad",
                h5ad="data/example.h5ad",
                drop_label_border_spots=True,
                cell_type=False,
            ).validate()
        with self.assertRaises(ValueError):
            DataConfig(
                source="h5ad",
                h5ad="data/example.h5ad",
                label_border_radius_mult=1.0,
            ).validate()

    def test_dlpfc_no_layer_border_config_matches_separate_run(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base_path = (
            root
            / "configs/maynard_dlpfc/maynard_dlpfc_151673_correlated_null_adaptive_separate_linear_p99.json"
        )
        path = (
            root
            / "configs/maynard_dlpfc/maynard_dlpfc_151673_correlated_null_adaptive_separate_linear_p99_no_layer_border.json"
        )
        base = json.loads(base_path.read_text())
        raw = json.loads(path.read_text())
        cfg = run_config_from_mapping(raw)
        self.assertTrue(cfg.data.drop_label_border_spots)
        self.assertEqual(cfg.data.label_border_radius_mult, 1.2)
        self.assertEqual(cfg.data.cell_type, "separate")
        self.assertEqual(cfg.data.cell_type_key, "Region")
        self.assertEqual(cfg.test.variogram_fit_mode, "variance_constrained")
        self.assertEqual(cfg.test.n_perms, base["test"]["n_perms"])
        self.assertEqual(cfg.test.metric, base["test"]["metric"])
        self.assertEqual(cfg.data.h5ad, base["data"]["h5ad"])
        self.assertEqual(cfg.data.top_var_genes, base["data"]["top_var_genes"])
        self.assertEqual(cfg.data.min_cells_per_celltype, 100)
        self.assertIn("no_layer_border", cfg.output.run_name)
        self.assertNotEqual(cfg.output.run_name, base["output"]["run_name"])

    def test_dlpfc_all_spots_min_cells_100_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base_path = (
            root
            / "configs/maynard_dlpfc/maynard_dlpfc_151673_correlated_null_adaptive_separate_linear_p99.json"
        )
        path = (
            root
            / "configs/maynard_dlpfc/maynard_dlpfc_151673_correlated_null_adaptive_separate_linear_p99_min_cells_100.json"
        )
        base = json.loads(base_path.read_text())
        raw = json.loads(path.read_text())
        cfg = run_config_from_mapping(raw)
        self.assertFalse(cfg.data.drop_label_border_spots)
        self.assertEqual(cfg.data.min_cells_per_celltype, 100)
        self.assertEqual(cfg.data.cell_type, "separate")
        self.assertEqual(cfg.data.min_cells_per_gene, base["data"]["min_cells_per_gene"])
        self.assertEqual(cfg.test.variogram_fit_mode, base["test"]["variogram_fit_mode"])
        self.assertEqual(cfg.test.decoder, base["test"]["decoder"])
        self.assertTrue(cfg.output.run_name.endswith("min_cells_100"))


if __name__ == "__main__":
    unittest.main()
