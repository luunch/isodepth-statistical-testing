"""Build the thin Visium HD CA ribbon used by the pyramidal.h5ad runs.

``hippocampal_pyramidal_CA_morphogam_midline.h5ad`` (the p499 input) includes
off-ribbon cells that thicken CA1/CA3. ``hippocampal_pyramidal.h5ad`` is the
denser cell-body ribbon from the earlier visium-HD runs, but it has no CA
labels or MorphoGAM ``coordinate``. This script copies that thin geometry and
transfers ``cell_type`` / ``coordinate`` from the morphogam file (barcode
match, else spatial KNN).

Usage (repo root, isodepth_env):
    python scripts/data_prep/build_hippocampus_thin_ribbon_h5ad.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_PYR = "data/h5ad/hippocampal_pyramidal.h5ad"
DEFAULT_MORPH = "data/h5ad/hippocampal_pyramidal_CA_morphogam_midline.h5ad"
DEFAULT_OUT = "data/h5ad/hippocampal_pyramidal_thin_ca_morphogam.h5ad"
DEFAULT_PREVIEW = "configs/hippocampus/thin_ribbon_preview.png"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def transfer_morphogam_labels(
    pyr: ad.AnnData,
    morph: ad.AnnData,
    *,
    k_neighbors: int = 15,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    pyr_ids = np.asarray(pyr.obs_names, dtype=str)
    morph_ids = np.asarray(morph.obs_names, dtype=str)
    morph_type = np.asarray(morph.obs["cell_type"].astype(str))
    morph_coord = np.asarray(morph.obs["coordinate"], dtype=np.float32)
    morph_map = {i: j for j, i in enumerate(morph_ids)}

    cell_type = np.empty(len(pyr_ids), dtype=object)
    coordinate = np.empty(len(pyr_ids), dtype=np.float32)
    matched = np.zeros(len(pyr_ids), dtype=bool)
    for i, barcode in enumerate(pyr_ids):
        j = morph_map.get(barcode)
        if j is None:
            continue
        matched[i] = True
        cell_type[i] = morph_type[j]
        coordinate[i] = morph_coord[j]

    n_matched = int(matched.sum())
    n_knn = int((~matched).sum())
    if n_knn:
        s_morph = np.asarray(morph.obsm["spatial"], dtype=np.float64)
        s_pyr = np.asarray(pyr.obsm["spatial"], dtype=np.float64)
        k = min(k_neighbors, len(morph))
        nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(s_morph)
        _, idx = nn.kneighbors(s_pyr[~matched])
        for row, neighbors in zip(np.flatnonzero(~matched), idx):
            votes = morph_type[neighbors]
            cell_type[row] = Counter(votes).most_common(1)[0][0]
            coordinate[row] = float(np.mean(morph_coord[neighbors]))

    stats = {
        "n_pyr": int(len(pyr_ids)),
        "n_morph": int(len(morph_ids)),
        "n_barcode_match": n_matched,
        "n_knn": n_knn,
    }
    return cell_type, coordinate, stats


def save_preview(
    path: Path,
    morph: ad.AnnData,
    thin: ad.AnnData,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    panels = [
        (axes[0], morph, "p499 morphogam (thick)"),
        (axes[1], thin, "thin pyramidal ribbon"),
    ]
    colors = {"CA1": "#1f77b4", "CA2": "#d62728", "CA3": "#17becf"}
    for ax, adata, title in panels:
        s = np.asarray(adata.obsm["spatial"])
        labels = np.asarray(adata.obs["cell_type"].astype(str))
        for lab, color in colors.items():
            m = labels == lab
            ax.scatter(
                s[m, 0],
                s[m, 1],
                s=3,
                c=color,
                linewidths=0,
                label=f"{lab} n={int(m.sum())}",
            )
        ax.set_title(f"{title}\nn={adata.n_obs}")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=4, fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyramidal-h5ad", default=DEFAULT_PYR)
    parser.add_argument("--morphogam-h5ad", default=DEFAULT_MORPH)
    parser.add_argument("--out-h5ad", default=DEFAULT_OUT)
    parser.add_argument("--preview", default=DEFAULT_PREVIEW)
    parser.add_argument("--k-neighbors", type=int, default=15)
    args = parser.parse_args()

    pyr = ad.read_h5ad(_resolve(args.pyramidal_h5ad))
    morph = ad.read_h5ad(_resolve(args.morphogam_h5ad))
    cell_type, coordinate, stats = transfer_morphogam_labels(
        pyr, morph, k_neighbors=args.k_neighbors
    )
    out = pyr.copy()
    out.obs["cell_type"] = cell_type
    out.obs["coordinate"] = coordinate
    out.obs["cell_type"] = out.obs["cell_type"].astype("category")
    out.uns["thin_ribbon"] = {
        **stats,
        "source_pyramidal": str(_resolve(args.pyramidal_h5ad)),
        "source_morphogam": str(_resolve(args.morphogam_h5ad)),
        "k_neighbors": int(args.k_neighbors),
    }

    out_path = _resolve(args.out_h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(out_path)

    preview_path = _resolve(args.preview)
    save_preview(preview_path, morph, out)

    counts = out.obs["cell_type"].astype(str).value_counts().to_dict()
    print(f"wrote {out_path} shape={out.shape}")
    print(f"barcode match={stats['n_barcode_match']} knn={stats['n_knn']}")
    print(f"cell_type {counts}")
    print(f"preview {preview_path}")


if __name__ == "__main__":
    main()
