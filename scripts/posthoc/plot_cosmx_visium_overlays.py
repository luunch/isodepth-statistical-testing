"""Visium 55 µm / 100 µm overlays on previously annotated CosMx cell-type regions.

Each figure uses only the ``obs_indices`` cells from one existence-test region
(no surrounding tissue). A cells-only scatter is written first, then the same
cells with the Visium hex/circle overlay.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.plots import save_visium_pseudobulk_overlay, _point_size, _spatial_axis_limits
from data.schemas import DatasetBundle, run_config_from_mapping
from methods.visium_pseudobulk import build_visium_pseudobulk_dataset

H5AD = ROOT / "data" / "h5ad" / "cosmx_human_nsclc_annotated.h5ad"
OUT = ROOT / "results" / "cosmx_visium_pseudobulk_overlays"
UM_PER_PX = 0.12028

# Regions we have tested the most (extra gaussian / block-perm variants).
REGIONS = [
    {
        "label": "LUAD-9 R2 neutrophil c88",
        "indices": "data/h5ad/cosmx_celltype_regions/luad_9_r2/cosmx_luad_9_r2_neutrophil_c88.indices.npy",
    },
    {
        "label": "LUAD-9 R1 neutrophil c35",
        "indices": "data/h5ad/cosmx_celltype_regions/luad_9_r1/cosmx_luad_9_r1_neutrophil_c35.indices.npy",
    },
    {
        "label": "LUAD-5 R3 B-cell c22",
        "indices": "data/h5ad/cosmx_celltype_regions/luad_5_r3/cosmx_luad_5_r3_b_cell_c22.indices.npy",
    },
]


def _read_categorical(grp: h5py.Group | h5py.Dataset) -> np.ndarray:
    if isinstance(grp, h5py.Group):
        codes = grp["codes"][:]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in grp["categories"][:]]
        return pd.Categorical.from_codes(codes, cats).astype(str)
    arr = grp[:]
    if arr.dtype.kind == "S":
        arr = np.array([x.decode() for x in arr])
    return np.asarray(arr).astype(str)


def _test_config(*, tissue_radius_um: float | None = None):
    mapping = {
        "data": {"source": "synthetic"},
        "test": {
            "method": "visium_pseudobulk",
            "n_perms": 1,
            "seed": 42,
            "visium_spot_distance_um": 100.0,
            "visium_spot_diameter_um": 55.0,
            "coordinate_um_per_unit": UM_PER_PX,
            "block_jitter": False,
            "visium_tissue_radius_um": tissue_radius_um,
        },
    }
    return run_config_from_mapping(mapping).validate().test


def _region_tissue_radius_um(xy_um: np.ndarray, *, eps_mult: float = 3.0) -> float:
    """Same connectivity radius used to define this DBSCAN region in the first place.

    ``scripts/data_prep/segment_cosmx_celltype_regions.py`` uses
    ``percentile(nn_dist, 90) * eps_mult`` as the DBSCAN ``eps`` when carving out
    single-cell-type regions. Reusing it here means the Visium tissue mask treats
    every gap *within* the region's own connectivity radius as tissue, instead of
    the much tighter generic ``1.5x median NN`` heuristic (tuned for dense, full
    tissue sections) which incorrectly punches holes through sparse cell types like
    scattered neutrophils or B-cells.
    """
    nn = cKDTree(xy_um).query(xy_um, k=2)[0][:, 1]
    return float(np.percentile(nn, 90) * eps_mult)


def _bundle(xy_px: np.ndarray) -> DatasetBundle:
    n = int(xy_px.shape[0])
    return DatasetBundle(
        S=np.asarray(xy_px, dtype=np.float32),
        A=np.ones((n, 1), dtype=np.float32),
        meta={
            "source": "h5ad",
            "coordinate_um_per_unit": UM_PER_PX,
            "normalize_total": False,
            "log1p": False,
            "standardize_expression": False,
        },
    ).validate()


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _save_cells_only(xy_um: np.ndarray, out_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.scatter(
        xy_um[:, 0],
        xy_um[:, 1],
        c="#4c78a8",
        s=max(_point_size(xy_um), 4.0),
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"{title}\n{xy_um.shape[0]:,} annotated-region cells")
    xlim, ylim = _spatial_axis_limits(xy_um)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5AD, "r") as f:
        xy = np.asarray(f["obsm"]["spatial"][:], dtype=np.float64)

    summaries: list[dict] = []
    for spec in REGIONS:
        idx = np.load(ROOT / spec["indices"])
        xy_px = xy[idx]
        xy_um = xy_px * UM_PER_PX
        name = f"cosmx_{_slug(spec['label'])}_region"
        cells_path = OUT / f"{name}_cells.png"
        overlay_path = OUT / f"{name}_overlay.png"
        tissue_radius_um = _region_tissue_radius_um(xy_um)
        test = _test_config(tissue_radius_um=tissue_radius_um)
        print(
            f"[region] {spec['label']}: {xy_px.shape[0]:,} annotated cells, "
            f"tissue_radius={tissue_radius_um:.1f} um"
        )
        _save_cells_only(xy_um, cells_path, spec["label"])

        _, artifacts = build_visium_pseudobulk_dataset(_bundle(xy_px), test)
        summary = dict(artifacts["visium_pseudobulk_summary"])
        save_visium_pseudobulk_overlay(
            artifacts["visium_overlay_S_um"],
            artifacts["visium_overlay_assigned"],
            artifacts["visium_overlay_centers_in_tissue_um"],
            artifacts["visium_overlay_occupied_ids"],
            hex_radius_um=float(artifacts["visium_overlay_hex_radius_um"]),
            spot_radius_um=float(artifacts["visium_overlay_spot_radius_um"]),
            out_path=overlay_path,
            run_name=name,
            spot_distance_um=summary.get("spot_distance_um"),
            spot_diameter_um=summary.get("spot_diameter_um"),
        )
        summary.update(
            {
                "kind": "annotated_region",
                "region": spec["label"],
                "n_cells": int(xy_px.shape[0]),
                "tissue_radius_um": tissue_radius_um,
                "cells_plot": str(cells_path),
                "plot": str(overlay_path),
            }
        )
        summaries.append(summary)
        print(
            f"  occupied={summary['n_pseudospots']}  "
            f"dropped_frac={summary['fraction_cells_dropped_in_gaps']:.3f}  "
            f"median_cells/spot={summary['median_cells_per_pseudospot']:.1f}  "
            f"-> {overlay_path}"
        )

    out_json = OUT / "summary.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
