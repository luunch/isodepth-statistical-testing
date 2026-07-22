from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.plots import save_spatial_region_split_plot
from data.schemas import DataConfig, DatasetBundle
from data.spatial_regions import dbscan_middle_region_mask, split_spatial_regions


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


if __name__ == "__main__":
    unittest.main()
