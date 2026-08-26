"""QC-covariate-vs-isodepth diagnostic plot for the ``loss_diff_clone2_linear`` run.

Mirrors the "isodepth vs. technical covariate" diagnostic style used elsewhere in this
repo (scatter with Spearman rho/p + spatial maps colored by isodepth and by the
covariate), but targets a single specific run:

    results/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2/loss_diff_clone2_linear

This run whitens on ``calicost_tumor_proportion`` (tumor clone proportion) only --
it does **not** whiten on total counts / library size (see its
``loss_diff_clone2_linear_result.json`` -> ``config.data.covariate_whitening``).
The two covariates plotted here (``pct_mt``, ``total_counts``) are therefore both
*unmodeled* technical covariates from the point of view of this run's test, which is
exactly what this diagnostic is checking for residual confounding.

Why coordinate matching is needed: this run predates ``obs_numeric_filters`` /
``spatial_crop`` tracking in the saved config (its ``loss_diff_clone2_linear_result.json``
config has no such keys), so naively reloading via ``load_dataset`` on that saved config
yields 197 clone-2.0 cells (only the 2 NaN-tumor-proportion spots dropped) rather than the
195 cells the run actually trained on. Since ``pct_mt``/``total_counts`` must line up
1:1 with the saved ``true_isodepth`` array (order-sensitive), we instead recover the
exact 195-cell subset by un-standardizing the saved (z-scored) spatial coordinates
using the run's saved ``coord_mean``/``coord_std`` and nearest-neighbor-matching them
back to raw spot coordinates in the source h5ad (matches are near-exact, max residual
~1e-3 in raw pixel units; see agent.md).

Usage (from repo root, isodepth_env):
    python -m scripts.posthoc.loss_diff_clone2_linear_qc_covariate_diagnostics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from experiments.core.paths import repo_root

REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "results/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2/loss_diff_clone2_linear"
RESULT_JSON = RUN_DIR / "loss_diff_clone2_linear_result.json"
CELL_TYPE = "2.0"
NPZ_PATH = RUN_DIR / CELL_TYPE / f"{CELL_TYPE}_isodepths.npz"
OUT_PATH = RUN_DIR / "loss_diff_clone2_linear_pct_mt_total_counts_diagnostics.png"

COVARIATES = [
    ("pct_mt", "pct_mt (%)"),
    ("total_counts", "total_counts"),
]


def _match_indices(raw_xy_target: np.ndarray, candidate_xy: np.ndarray, *, atol: float = 0.1) -> np.ndarray:
    """Nearest-neighbor match each row of ``raw_xy_target`` to ``candidate_xy``.

    Raises if any match is not (near-)exact or if matches are not unique, since a bad
    match would silently misalign ``true_isodepth`` against the QC covariates.
    """
    tree = cKDTree(candidate_xy)
    dist, idx = tree.query(raw_xy_target, k=1)
    if dist.max() > atol:
        raise ValueError(f"Worst spatial match residual {dist.max():.4f} exceeds tolerance {atol}")
    if len(set(idx.tolist())) != len(idx):
        raise ValueError("Non-unique nearest-neighbor matches; cannot safely align covariates to isodepth")
    return idx


def main() -> None:
    with RESULT_JSON.open("r", encoding="utf-8") as f:
        result = json.load(f)
    dataset_meta = result["artifacts"]["dataset_meta"]
    coord_mean = np.asarray(dataset_meta["coord_mean"], dtype=np.float64)
    coord_std = np.asarray(dataset_meta["coord_std"], dtype=np.float64)
    h5ad_path = Path(dataset_meta["h5ad"])
    p_value = float(result["p_value"])

    npz = np.load(NPZ_PATH)
    S = np.asarray(npz["S"], dtype=np.float64)  # standardized coords, training order
    true_isodepth = np.asarray(npz["true_isodepth"], dtype=np.float64)
    n_cells = true_isodepth.shape[0]
    raw_xy_target = S * coord_std + coord_mean

    print(f"[load] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs
    clone_mask = (obs["calicost_clone_label"].astype(str) == CELL_TYPE) & (
        ~obs["calicost_tumor_proportion"].isna()
    )
    candidate_xy = np.asarray(adata.obsm["spatial"])[clone_mask.values]
    candidate_obs = obs.loc[clone_mask]

    match_idx = _match_indices(raw_xy_target, candidate_xy)
    matched_obs = candidate_obs.iloc[match_idx]
    print(f"[match] {n_cells} training spots matched to source h5ad (max residual ok)")

    covariate_values = {name: np.asarray(matched_obs[name], dtype=np.float64) for name, _ in COVARIATES}

    n_cov = len(COVARIATES)
    n_spatial = 1 + n_cov  # isodepth + each covariate, each shown once
    ncols = 2 * n_spatial  # common denominator so both rows can be centered/evenly spaced
    fig = plt.figure(figsize=(3.0 * ncols, 8.5))
    gs = fig.add_gridspec(2, ncols)

    scatter_span = ncols // n_cov
    for col, (name, xlabel) in enumerate(COVARIATES):
        x = covariate_values[name]
        ax = fig.add_subplot(gs[0, col * scatter_span : (col + 1) * scatter_span])
        ax.scatter(x, true_isodepth, s=14, alpha=0.6)
        rho, p = spearmanr(x, true_isodepth)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("true_isodepth")
        ax.set_title(f"{name}: rho={rho:.2f}, p={p:.2g}")

    spatial_span = ncols // n_spatial
    spatial_panels = [("isodepth", true_isodepth)] + [(name, covariate_values[name]) for name, _ in COVARIATES]
    for col, (name, values) in enumerate(spatial_panels):
        ax = fig.add_subplot(gs[1, col * spatial_span : (col + 1) * spatial_span])
        sc = ax.scatter(S[:, 0], S[:, 1], c=values, s=14, cmap="viridis")
        ax.set_title(name)
        fig.colorbar(sc, ax=ax)

    fig.suptitle(
        f"loss_diff_clone2_linear | clone {CELL_TYPE} (n={n_cells}, p={p_value:.3g}) | "
        "covariate whitening: calicost_tumor_proportion only (not total_counts)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)
    print(f"[write] {OUT_PATH}")


if __name__ == "__main__":
    main()
