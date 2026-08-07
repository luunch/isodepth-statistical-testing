"""Tests for MOSTA Stereo-seq coordinate scale detection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from data.h5ad_loader import COSMX_UM_PER_UNIT, MOSTA_BIN50_UM_PER_UNIT, _detect_coordinate_um_per_unit


class TestMostaCoordinateScale(unittest.TestCase):
    def test_detects_from_mosta_filename(self) -> None:
        adata = ad.AnnData(
            X=np.zeros((3, 2), dtype=np.float32),
            obsm={"spatial": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)},
        )
        scale = _detect_coordinate_um_per_unit(
            adata,
            h5ad_path="data/h5ad/mouse-organogenesis/E10.5_E1S1.MOSTA.h5ad",
        )
        self.assertEqual(scale, MOSTA_BIN50_UM_PER_UNIT)

    def test_detects_from_cosmx_filename(self) -> None:
        adata = ad.AnnData(
            X=np.zeros((3, 2), dtype=np.float32),
            obsm={"spatial": np.zeros((3, 2), dtype=np.float64)},
        )
        scale = _detect_coordinate_um_per_unit(
            adata,
            h5ad_path="data/h5ad/cosmx_human_nsclc_annotated.h5ad",
        )
        self.assertEqual(scale, COSMX_UM_PER_UNIT)

    def test_detects_from_stereo_seq_uns(self) -> None:
        adata = ad.AnnData(
            X=np.zeros((2, 2), dtype=np.float32),
            obsm={"spatial": np.zeros((2, 2), dtype=np.float64)},
        )
        adata.uns["stereo_seq"] = {"coordinate_um_per_unit": 25.0}
        scale = _detect_coordinate_um_per_unit(adata, h5ad_path="other.h5ad")
        self.assertEqual(scale, 25.0)

    def test_unknown_file_returns_none_without_metadata(self) -> None:
        adata = ad.AnnData(
            X=np.zeros((2, 2), dtype=np.float32),
            obsm={"spatial": np.zeros((2, 2), dtype=np.float64)},
        )
        scale = _detect_coordinate_um_per_unit(adata, h5ad_path="random.h5ad")
        self.assertIsNone(scale)

    def test_loader_excludes_genes_by_regex_before_preprocessing(self) -> None:
        adata = ad.AnnData(
            X=np.ones((4, 5), dtype=np.float32),
            obsm={"spatial": np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                dtype=np.float32,
            )},
        )
        adata.var_names = ["MT-CO1", "RPL41", "MALAT1", "HSPA1A", "ACTB"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.h5ad"
            adata.write_h5ad(path)
            from data.h5ad_loader import load_h5ad_dataset

            dataset = load_h5ad_dataset(
                h5ad_path=str(path),
                spatial_key="spatial",
                exclude_gene_patterns=["^MT-", "^RPL", "^MALAT1$", "^HSP"],
                min_cells_per_gene=0,
                top_var_genes=0,
                normalize_total=False,
                log1p=False,
                standardize_expression=False,
            )

        self.assertEqual(dataset.meta["var_names"], ["ACTB"])
        self.assertEqual(dataset.meta["excluded_gene_count"], 4)

    def test_loader_meta_on_real_mosta_file_if_present(self) -> None:
        path = Path("data/h5ad/mouse-organogenesis/E10.5_E1S1.MOSTA.h5ad")
        if not path.is_file():
            self.skipTest(f"{path} not available")

        from data.h5ad_loader import load_h5ad_dataset

        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / path.name
            link.symlink_to(path.resolve())
            dataset = load_h5ad_dataset(
                h5ad_path=str(link),
                spatial_key="spatial",
                layer="count",
                min_cells_per_gene=0,
                top_var_genes=0,
                max_cells=100,
                seed=0,
            )
        self.assertEqual(dataset.meta.get("coordinate_um_per_unit"), MOSTA_BIN50_UM_PER_UNIT)


if __name__ == "__main__":
    unittest.main()
