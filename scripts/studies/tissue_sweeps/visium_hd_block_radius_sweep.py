"""Visium HD block-radius diagnostic sweep (config-driven).

For each candidate hex ``block_radius`` (µm):
  1. Summarize tiling on raw physical coordinates (block counts / cells per block).
  2. Save a two-panel overlay (true tiling + one sample centroid permutation).

Does not train isodepth — cheap calibration for choosing ``test.block_radius``.

Usage:
  python scripts/visium_hd_block_radius_sweep.py
  python scripts/visium_hd_block_radius_sweep.py --spec configs/experiments/visium_hd_block_radius_sweep.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.core.paths import repo_root
REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from analysis.plots import save_block_permutation_overlay
from methods.block_permutation import (
    block_occupancy_counts,
    block_stats,
    build_block_permuted_coordinate_batch,
    hex_bin_ids,
    resolve_um_per_unit,
)

DEFAULT_SPEC = REPO / "configs/experiments/visium_hd_block_radius_sweep.json"


def load_spec(spec_path: str | Path) -> dict:
    with open(spec_path, encoding="utf-8") as fh:
        return json.load(fh)


def _detect_bin_size_um(adata: ad.AnnData) -> float | None:
    sp = adata.uns.get("spatial")
    if not isinstance(sp, dict):
        return None
    for lib in sp.values():
        if isinstance(lib, dict) and isinstance(lib.get("scalefactors"), dict):
            val = lib["scalefactors"].get("bin_size_um")
            if val is not None:
                return float(val)
    return None


def _singleton_fraction(block_ids: np.ndarray) -> float:
    _unique, counts = np.unique(np.asarray(block_ids, dtype=np.int64), return_counts=True)
    if counts.size == 0:
        return 0.0
    return float((counts == 1).sum()) / float(counts.size)


def _save_cells_per_block_histogram(
    counts: np.ndarray,
    out_path: Path,
    *,
    radius_um: float,
    bin_size_um: float,
) -> None:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        return
    max_count = int(counts.max())
    bins = np.arange(0.5, max_count + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(counts, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.6)
    ax.axvline(float(np.median(counts)), color="#d62728", ls="--", lw=1.2, label=f"median={np.median(counts):.0f}")
    ax.axvline(float(np.mean(counts)), color="#ff7f0e", ls=":", lw=1.2, label=f"mean={np.mean(counts):.1f}")
    ax.set_xlabel("cells per occupied hex block")
    ax.set_ylabel("number of blocks")
    title = f"Block occupancy histogram — r={radius_um:g} µm"
    if bin_size_um == bin_size_um:
        title += f" ({radius_um / bin_size_um:.1f}× bin)"
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(spec_path: str | Path = DEFAULT_SPEC) -> Path:
    spec = load_spec(spec_path)
    data_cfg = spec["data"]
    sweep_cfg = spec["sweep"]
    out_dir = REPO / spec["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = REPO / data_cfg["h5ad"]
    adata = ad.read_h5ad(h5ad_path, backed="r")
    spatial_key = str(data_cfg.get("spatial_key", "spatial"))
    S_raw = np.asarray(adata.obsm[spatial_key][:, :2], dtype=np.float32)

    # Apply optional obs_filter: dict of {obs_column: value} to subset cells.
    obs_filter = data_cfg.get("obs_filter", {})
    if obs_filter:
        mask = np.ones(S_raw.shape[0], dtype=bool)
        for col, val in obs_filter.items():
            mask &= (adata.obs[col] == val).values
        S_raw = S_raw[mask]
        n_filtered = int(mask.sum())
        print(f"obs_filter applied: {obs_filter} → {n_filtered}/{int(mask.size)} cells retained")

    meta_um = adata.uns.get("spatial")
    detected_um = None
    if isinstance(meta_um, dict):
        for lib in meta_um.values():
            if isinstance(lib, dict) and isinstance(lib.get("scalefactors"), dict):
                mpp = lib["scalefactors"].get("microns_per_pixel")
                if mpp is not None:
                    detected_um = float(mpp)
                    break

    um_per_unit = resolve_um_per_unit(
        data_cfg.get("coordinate_um_per_unit"),
        detected_um,
    )
    bin_size_um = float(data_cfg.get("bin_size_um") or _detect_bin_size_um(adata) or np.nan)
    adata.file.close()

    radii_um = [float(r) for r in sweep_cfg["block_radii_um"]]
    block_jitter = bool(sweep_cfg.get("block_jitter", True))
    seed = int(sweep_cfg.get("seed", 0))

    rows: list[dict[str, object]] = []
    for radius_um in radii_um:
        stats = block_stats(S_raw, radius_um, um_per_unit)
        block_counts = block_occupancy_counts(S_raw, radius_um, um_per_unit)
        S_um = np.asarray(S_raw, dtype=np.float64) * um_per_unit
        block_ids = hex_bin_ids(S_um, radius_um, (0.0, 0.0))
        singleton_frac = _singleton_fraction(block_ids)

        s_batched_raw = build_block_permuted_coordinate_batch(
            S_raw,
            radius_um=radius_um,
            coordinate_um_per_unit=um_per_unit,
            n_perms=1,
            seed=seed,
            block_jitter=block_jitter,
        )
        overlay_path = out_dir / f"block_radius_{int(radius_um)}um_overlay.png"
        save_block_permutation_overlay(
            S_raw,
            np.asarray(s_batched_raw[1], dtype=np.float32),
            block_ids,
            overlay_path,
            run_name=f"r={int(radius_um)} µm",
            radius_units=float(radius_um / um_per_unit),
        )
        hist_path = out_dir / f"block_radius_{int(radius_um)}um_cells_per_block_hist.png"
        _save_cells_per_block_histogram(
            block_counts,
            hist_path,
            radius_um=radius_um,
            bin_size_um=bin_size_um,
        )

        row = {
            "radius_um": radius_um,
            "radius_bin_multiples": (radius_um / bin_size_um) if bin_size_um == bin_size_um else "",
            "radius_coord_units": radius_um / um_per_unit,
            "n_blocks": stats["n_blocks"],
            "mean_cells_per_block": stats["mean_cells"],
            "median_cells_per_block": stats["median_cells"],
            "min_cells_per_block": stats["min_cells"],
            "max_cells_per_block": stats["max_cells"],
            "fraction_singleton_blocks": singleton_frac,
            "overlay_plot": str(overlay_path.relative_to(REPO)),
            "cells_per_block_hist": str(hist_path.relative_to(REPO)),
        }
        rows.append(row)
        bin_label = (
            f"{row['radius_bin_multiples']:.1f}× bin"
            if isinstance(row["radius_bin_multiples"], float)
            else "no bin_size"
        )
        print(
            f"r={radius_um:4.0f} µm ({bin_label}): "
            f"{stats['n_blocks']} blocks, median={stats['median_cells']:.0f} cells/block, "
            f"singletons={singleton_frac:.1%}"
        )

    summary_csv = out_dir / "block_radius_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [float(r["radius_um"]) for r in rows]
    axes[0].plot(x, [int(r["n_blocks"]) for r in rows], marker="o")
    axes[0].set_xlabel("block_radius (µm)")
    axes[0].set_ylabel("occupied hex blocks")
    axes[0].set_title("Block count vs radius")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, [float(r["median_cells_per_block"]) for r in rows], marker="o", label="median")
    axes[1].plot(x, [float(r["mean_cells_per_block"]) for r in rows], marker="s", label="mean", alpha=0.8)
    axes[1].set_xlabel("block_radius (µm)")
    axes[1].set_ylabel("cells per block")
    axes[1].set_title("Block occupancy vs radius")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    if bin_size_um == bin_size_um:
        for ax in axes:
            sec = ax.secondary_xaxis(
                "top",
                functions=(lambda r, b=bin_size_um: r / b, lambda m, b=bin_size_um: m * b),
            )
            sec.set_xlabel("× Visium HD bin_size_um")
    exp_name = spec.get("experiment_name") or h5ad_path.name
    bin_str = f"bin={bin_size_um:g} µm, " if bin_size_um == bin_size_um else ""
    obs_filter_str = (
        " [" + ", ".join(f"{k}={v}" for k, v in obs_filter.items()) + "]"
        if obs_filter
        else ""
    )
    fig.suptitle(
        f"Block-radius sweep — {exp_name}{obs_filter_str} "
        f"({bin_str}scale={um_per_unit:.4f} µm/coord unit)",
        fontsize=11,
    )
    fig.tight_layout()
    summary_plot = out_dir / "block_radius_summary.png"
    fig.savefig(summary_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "experiment_name": spec.get("experiment_name"),
        "spec_path": str(Path(spec_path).resolve()),
        "h5ad": str(h5ad_path),
        "obs_filter": obs_filter,
        "n_cells": int(S_raw.shape[0]),
        "coordinate_um_per_unit": um_per_unit,
        "bin_size_um": None if bin_size_um != bin_size_um else bin_size_um,
        "block_radii_um": radii_um,
        "summary_csv": str(summary_csv.relative_to(REPO)),
        "summary_plot": str(summary_plot.relative_to(REPO)),
        "overlays": [str(r["overlay_plot"]) for r in rows],
        "cells_per_block_histograms": [str(r["cells_per_block_hist"]) for r in rows],
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nSaved summary -> {summary_csv}")
    print(f"Saved plot     -> {summary_plot}")
    print(f"Saved manifest -> {manifest_path}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visium HD block-radius diagnostic sweep.")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    main(parser.parse_args().spec)
