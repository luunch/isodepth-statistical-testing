"""Visualize the CosMx tumor-clone regions: each clone's spatial footprint
highlighted within its home tissue section.

For each primary (sample, tumor-clone) region (clone present with >= --min-cells
in that section), draws the section's full cell cloud in grey with the clone's
cells highlighted on top — so you can see the actual spatial extent that the
existence test would run on.

Output: results/cosmx_clone_regions/clone_regions_overview.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "h5ad" / "cosmx_human_nsclc_annotated.h5ad"
OUT = ROOT / "results" / "cosmx_clone_regions"

CLONE_COLORS = {
    "tumor 5": "#1f77b4", "tumor 6": "#d62728", "tumor 9": "#2ca02c",
    "tumor 12": "#9467bd", "tumor 13": "#ff7f0e",
}


def _col(f, name):
    g = f["obs"][name]
    if isinstance(g, h5py.Group):
        return pd.Categorical.from_codes(
            g["codes"][:], [c.decode() if isinstance(c, bytes) else c
                            for c in g["categories"][:]])
    return g[:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-cells", type=int, default=1000)
    args = ap.parse_args()

    with h5py.File(SRC, "r") as f:
        sample = np.asarray(_col(f, "sample"))
        cell_type = np.asarray(_col(f, "cell_type"))
        xy = f["obsm"]["spatial"][:]

    df = pd.DataFrame({"sample": sample, "cell_type": cell_type,
                       "x": xy[:, 0], "y": xy[:, 1]})
    df["is_tumor"] = pd.Series(cell_type).astype(str).str.startswith("tumor").values

    tum = df[df["is_tumor"]]
    sizes = tum.groupby(["cell_type", "sample"], observed=True).size()
    regions = (sizes[sizes >= args.min_cells]
               .sort_values(ascending=False).index.tolist())
    print(f"{len(regions)} clone regions with >= {args.min_cells} cells")

    n = len(regions)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (clone, samp) in zip(axes, regions):
        samp_cells = df[df["sample"] == samp]
        clone_cells = samp_cells[samp_cells["cell_type"] == clone]
        ax.scatter(samp_cells["x"], samp_cells["y"], s=1, c="0.82",
                   linewidths=0, rasterized=True)
        ax.scatter(clone_cells["x"], clone_cells["y"], s=1.5,
                   c=CLONE_COLORS.get(clone, "#000000"),
                   linewidths=0, rasterized=True)
        ax.set_title(f"{clone} in {samp}\nn={len(clone_cells):,} "
                     f"({100*len(clone_cells)/len(samp_cells):.0f}% of section)",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("CosMx NSCLC tumor-clone regions (clone highlighted within its section)",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "clone_regions_overview.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
