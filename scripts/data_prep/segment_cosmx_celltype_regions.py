"""Segment CosMx samples into contiguous spatial regions per annotated cell type.

For each sample and each ``obs['cell_type']``, runs DBSCAN on that type's
coordinates and exports every connected component with ``>= min_cells`` cells.
Regions are 100% single cell type by construction (unlike cluster20-based
``segment_cosmx_sample_regions.py``).

Writes global ``obs_indices`` .npy files (no per-region h5ad), a manifest CSV,
per-sample overview PNGs, and run_permutation configs.

Example (all 8 samples, default min_cells=500 → ~77 regions):
  mamba run -n isodepth_env python scripts/segment_cosmx_celltype_regions.py

Example (one sample):
  mamba run -n isodepth_env python scripts/segment_cosmx_celltype_regions.py \\
      --sample "LUAD-9 R1"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "h5ad" / "cosmx_human_nsclc_annotated.h5ad"
OUT_BASE = ROOT / "data" / "h5ad" / "cosmx_celltype_regions"
CONFIG_DIR = ROOT / "configs" / "cosmx_celltype_regions"
MANIFEST = OUT_BASE / "manifest.csv"
FULL_H5AD_REL = "data/h5ad/cosmx_human_nsclc_annotated.h5ad"
KEEP_OBS = ["sample", "patient", "fov", "niche", "cell_type", "n_counts"]


def _read_categorical(grp: h5py.Group | h5py.Dataset) -> np.ndarray:
    if isinstance(grp, h5py.Group):
        codes = grp["codes"][:]
        cats = [c.decode() if isinstance(c, bytes) else c for c in grp["categories"][:]]
        return pd.Categorical.from_codes(codes, cats)
    arr = grp[:]
    if arr.dtype.kind == "S":
        arr = np.array([x.decode() for x in arr])
    return arr


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _config_for(region_name: str, indices_rel: str, run_name: str) -> dict:
    return {
        "data": {
            "source": "h5ad",
            "h5ad": FULL_H5AD_REL,
            "obs_indices": indices_rel,
            "spatial_key": "spatial",
            "obs_x_col": None,
            "obs_y_col": None,
            "layer": "counts",
            "use_raw": False,
            "cell_type_key": "cell_type",
            "cell_type": False,
            "min_cells_per_gene": 1,
            "normalize_total": False,
            "standardize_expression": False,
            "standardize_coordinates": True,
            "log1p": False,
            "max_cells": None,
            "seed": 42,
        },
        "test": {
            "method": "parallel_permutation",
            "metric": "nll_poisson_mse",
            "n_perms": 99,
            "epochs": 500,
            "n_reruns": 10,
            "sgd_batch_size": 128,
            "lr": 0.001,
            "seed": 42,
            "device": "cuda",
            "decoder": "nn",
            "batch_size": None,
            "verbose": False,
        },
        "output": {
            "out_dir": "results/cosmx_celltype_regions",
            "run_name": run_name,
            "save_preds": False,
            "save_perm_stats": True,
        },
    }


def discover_components(
    global_idx: np.ndarray,
    xy: np.ndarray,
    cell_type: str,
    eps: float,
    min_cells: int,
) -> list[dict]:
    rows: list[dict] = []
    labels = DBSCAN(eps=eps, min_samples=10).fit_predict(xy)
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        sub = labels == lab
        n = int(sub.sum())
        if n < min_cells:
            continue
        cx, cy = float(xy[sub, 0].mean()), float(xy[sub, 1].mean())
        rows.append({
            "cell_type": cell_type,
            "component_id": int(lab),
            "n_cells": n,
            "centroid_x": cx,
            "centroid_y": cy,
            "global_indices": global_idx[sub],
        })
    return rows


def segment_sample(
    sample: str,
    *,
    idx_all: np.ndarray,
    obs: pd.DataFrame,
    xy: np.ndarray,
    min_cells: int,
    eps_mult: float,
    write_configs: bool,
    dry_run: bool,
) -> list[dict]:
    sample_slug = _sanitize(sample)
    out_dir = OUT_BASE / sample_slug
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    eps = float(np.percentile(nn, 90) * eps_mult)

    regions: list[dict] = []
    cell_types = sorted(set(obs["cell_type"].astype(str)))
    for ct in cell_types:
        local_mask = obs["cell_type"].astype(str).to_numpy() == ct
        if int(local_mask.sum()) < min_cells // 2:
            continue
        local_idx = np.flatnonzero(local_mask)
        comps = discover_components(
            idx_all[local_idx], xy[local_idx], ct, eps, min_cells
        )
        regions.extend(comps)

    if dry_run:
        print(f"[dry-run] {sample}: {len(regions)} regions (eps={eps:.1f})")
        for r in sorted(regions, key=lambda x: -x["n_cells"]):
            print(f"  {r['cell_type']:20s} c{r['component_id']:3d}  n={r['n_cells']}")
        return []

    if not regions:
        print(f"[skip] {sample}: no regions >= {min_cells} cells")
        return []

    if write_configs:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    fig, axes = plt.subplots(1, min(len(regions), 6) + 1,
                             figsize=(4.2 * (min(len(regions), 6) + 1), 4.5))
    if len(regions) == 0:
        plt.close(fig)
        return []

    axes[0].scatter(xy[:, 0], xy[:, 1], s=0.3, c="0.85", linewidths=0, rasterized=True)
    axes[0].set_title(f"{sample}\nall cells (n={len(xy):,})", fontsize=9)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    top_regions = sorted(regions, key=lambda r: -r["n_cells"])[:6]
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_regions)))

    for ax, reg, color in zip(axes[1:], top_regions, colors):
        gidx = reg["global_indices"]
        local = np.isin(idx_all, gidx)
        ct_slug = _sanitize(reg["cell_type"])
        region_name = f"cosmx_{sample_slug}_{ct_slug}_c{reg['component_id']}"
        run_name = f"{region_name}_poisson"

        indices_rel = f"data/h5ad/cosmx_celltype_regions/{sample_slug}/{region_name}.indices.npy"
        np.save(ROOT / indices_rel, gidx.astype(np.int64))

        if write_configs:
            cfg_path = CONFIG_DIR / f"{region_name}.json"
            cfg_path.write_text(json.dumps(
                _config_for(region_name, indices_rel, run_name), indent=2
            ) + "\n")

        manifest_rows.append({
            "region_name": region_name,
            "run_name": run_name,
            "sample": sample,
            "cell_type": reg["cell_type"],
            "component_id": reg["component_id"],
            "n_cells": reg["n_cells"],
            "centroid_x": round(reg["centroid_x"], 1),
            "centroid_y": round(reg["centroid_y"], 1),
            "obs_indices": indices_rel,
            "config": str((CONFIG_DIR / f"{region_name}.json").relative_to(ROOT))
            if write_configs else "",
        })

        axes[0].scatter(xy[local, 0], xy[local, 1], s=1.5, c=[color],
                        linewidths=0, rasterized=True)
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="0.9", linewidths=0, rasterized=True)
        ax.scatter(xy[local, 0], xy[local, 1], s=2, c=[color], linewidths=0, rasterized=True)
        ax.set_title(f"{reg['cell_type']}\nc{reg['component_id']}\nn={reg['n_cells']:,}",
                     fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Export remaining regions (beyond top 6 for plot) without re-plotting
    plotted = {id(r) for r in top_regions}
    for reg in regions:
        if id(reg) in plotted:
            continue
        gidx = reg["global_indices"]
        ct_slug = _sanitize(reg["cell_type"])
        region_name = f"cosmx_{sample_slug}_{ct_slug}_c{reg['component_id']}"
        run_name = f"{region_name}_poisson"
        indices_rel = f"data/h5ad/cosmx_celltype_regions/{sample_slug}/{region_name}.indices.npy"
        np.save(ROOT / indices_rel, gidx.astype(np.int64))
        if write_configs:
            cfg_path = CONFIG_DIR / f"{region_name}.json"
            cfg_path.write_text(json.dumps(
                _config_for(region_name, indices_rel, run_name), indent=2
            ) + "\n")
        manifest_rows.append({
            "region_name": region_name,
            "run_name": run_name,
            "sample": sample,
            "cell_type": reg["cell_type"],
            "component_id": reg["component_id"],
            "n_cells": reg["n_cells"],
            "centroid_x": round(reg["centroid_x"], 1),
            "centroid_y": round(reg["centroid_y"], 1),
            "obs_indices": indices_rel,
            "config": str((CONFIG_DIR / f"{region_name}.json").relative_to(ROOT))
            if write_configs else "",
        })

    fig.suptitle(f"CosMx cell-type regions — {sample}", fontsize=11, y=1.02)
    plt.tight_layout()
    overview = out_dir / f"{sample_slug}_celltype_regions_overview.png"
    plt.savefig(overview, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"[done] {sample}: {len(manifest_rows)} regions -> {out_dir}")
    for row in sorted(manifest_rows, key=lambda r: -r["n_cells"])[:8]:
        print(f"  {row['region_name']:50s} n={row['n_cells']}")
    if len(manifest_rows) > 8:
        print(f"  ... +{len(manifest_rows) - 8} more")
    return manifest_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=None, help="One sample (default: all 8).")
    ap.add_argument("--min-cells", type=int, default=500)
    ap.add_argument("--eps-mult", type=float, default=3.0)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-configs", dest="write_configs", action="store_false", default=True)
    args = ap.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"[load] obs + spatial from {args.src.name}", flush=True)
    with h5py.File(args.src, "r") as f:
        sample_col = np.asarray(_read_categorical(f["obs"]["sample"]))
        samples = sorted(set(sample_col))
        if args.sample:
            if args.sample not in samples:
                raise SystemExit(f"sample not found: {args.sample}")
            samples = [args.sample]

        all_rows: list[dict] = []
        for sample in samples:
            mask = sample_col == sample
            idx_all = np.flatnonzero(mask)
            obs = pd.DataFrame({
                k: np.asarray(_read_categorical(f["obs"][k]))[mask]
                for k in KEEP_OBS
            })
            xy = f["obsm"]["spatial"][idx_all]
            rows = segment_sample(
                sample,
                idx_all=idx_all,
                obs=obs,
                xy=xy,
                min_cells=args.min_cells,
                eps_mult=args.eps_mult,
                write_configs=args.write_configs,
                dry_run=args.dry_run,
            )
            all_rows.extend(rows)

    if args.dry_run:
        return

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).sort_values(
        ["sample", "cell_type", "n_cells"], ascending=[True, True, False]
    ).to_csv(MANIFEST, index=False)
    print(f"\n[manifest] {len(all_rows)} regions -> {MANIFEST}")
    if args.write_configs:
        print(f"[configs]  {CONFIG_DIR} ({len(list(CONFIG_DIR.glob('*.json')))} files)")


if __name__ == "__main__":
    main()
