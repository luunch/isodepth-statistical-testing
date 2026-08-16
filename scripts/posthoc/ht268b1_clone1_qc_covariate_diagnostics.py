"""QC-covariate-vs-isodepth diagnostic plot for HT268B1 clone 1 (linear, tumor-prop whitening).

Same layout as ``loss_diff_clone2_linear_qc_covariate_diagnostics`` / the reference
``pct_mt`` / ``total_counts`` figure: Spearman scatter vs ``true_isodepth`` on top,
spatial maps of isodepth / pct_mt / total_counts on the bottom.

Target run (tumor-proportion-only whitening — total_counts is an *unmodeled*
technical covariate, matching the clone-2 reference figure):

    results/calicost/HT268B1-Th1K3Fc2U1Z1Bs1/loss_diff_clone1_gt0p7_linear

Reload note: the sibling config JSON has ``spatial_denoise_radius_um=300``, but the
saved ``result.json`` ``dataset_meta`` omits denoise. Reloading with denoise=300
recovers the exact 426 training spots; per-clone re-standardized ``S`` matches the
NPZ. QC columns (``pct_mt``, ``total_counts``) are read from the source h5ad via
nearest-neighbor match on raw spatial coordinates.

Usage (from repo root, isodepth_env):
    python -m scripts.posthoc.ht268b1_clone1_qc_covariate_diagnostics
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

from data import load_dataset  # noqa: E402
from data.schemas import run_config_from_mapping  # noqa: E402
from experiments.configuration import load_json_config  # noqa: E402

RUN_DIR = REPO / "results/calicost/HT268B1-Th1K3Fc2U1Z1Bs1/loss_diff_clone1_gt0p7_linear"
RESULT_JSON = RUN_DIR / "loss_diff_clone1_gt0p7_linear_result.json"
CONFIG_PATH = REPO / "configs/calicost/HT268B1_slice1_U1_loss_difference.json"
CELL_TYPE = "1.0"
NPZ_PATH = RUN_DIR / CELL_TYPE / f"{CELL_TYPE}_isodepths.npz"
OUT_PATH = RUN_DIR / CELL_TYPE / f"{CELL_TYPE}_pct_mt_total_counts_diagnostics.png"

COVARIATES = [
    ("pct_mt", "pct_mt (%)"),
    ("total_counts", "total_counts"),
]


def _match_indices(raw_xy_target: np.ndarray, candidate_xy: np.ndarray, *, atol: float = 0.1) -> np.ndarray:
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

    cfg = load_json_config(str(CONFIG_PATH))
    run_cfg = run_config_from_mapping(cfg)
    # Saved meta omits denoise; config (and the actual training run) used 300 µm.
    run_cfg.data.spatial_denoise_radius_um = 300.0
    run_cfg.data.spatial_crop = None
    run_cfg.data.obs_numeric_filters = dataset_meta.get("obs_numeric_filters")
    run_cfg.data.coordinate_um_per_unit = dataset_meta.get("coordinate_um_per_unit") or getattr(
        run_cfg.test, "coordinate_um_per_unit", None
    )

    print(f"[load] dataset via {CONFIG_PATH.name} (denoise=300)")
    dataset = load_dataset(run_cfg.data)
    if int(dataset.S.shape[0]) != int(result["n_cells"]):
        raise RuntimeError(
            f"reloaded n_cells={dataset.S.shape[0]} != result n_cells={result['n_cells']}"
        )

    npz = np.load(NPZ_PATH)
    S_npz = np.asarray(npz["S"], dtype=np.float64)
    true_isodepth = np.asarray(npz["true_isodepth"], dtype=np.float64)
    n_cells = true_isodepth.shape[0]

    labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    names = [str(n) for n in dataset.meta["cell_type_names"]]
    type_index = names.index(CELL_TYPE)
    mask = labels == type_index
    if int(mask.sum()) != n_cells:
        raise RuntimeError(f"clone {CELL_TYPE}: reload n={int(mask.sum())} != npz n={n_cells}")

    S_c = np.asarray(dataset.S[mask], dtype=np.float64)
    S_rez = (S_c - S_c.mean(axis=0)) / np.maximum(S_c.std(axis=0), 1e-8)
    if not np.allclose(S_npz, S_rez, atol=1e-3):
        raise RuntimeError(
            f"per-clone S alignment failed (max |diff|={np.max(np.abs(S_npz - S_rez)):.4g})"
        )

    raw_xy_target = S_c * coord_std + coord_mean
    print(f"[load] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs
    for col in ("pct_mt", "total_counts", "calicost_clone_label", "calicost_tumor_proportion"):
        if col not in obs.columns:
            raise KeyError(f"Required obs column missing from h5ad: {col}")

    clone_mask = (obs["calicost_clone_label"].astype(str) == CELL_TYPE) & (
        ~obs["calicost_tumor_proportion"].isna()
    )
    candidate_xy = np.asarray(adata.obsm["spatial"])[clone_mask.values]
    candidate_obs = obs.loc[clone_mask]
    match_idx = _match_indices(raw_xy_target, candidate_xy)
    matched_obs = candidate_obs.iloc[match_idx]
    print(f"[match] {n_cells} training spots matched to source h5ad")

    covariate_values = {name: np.asarray(matched_obs[name], dtype=np.float64) for name, _ in COVARIATES}

    n_cov = len(COVARIATES)
    n_spatial = 1 + n_cov
    ncols = 2 * n_spatial
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
        sc = ax.scatter(S_npz[:, 0], S_npz[:, 1], c=values, s=14, cmap="viridis")
        ax.set_title(name)
        fig.colorbar(sc, ax=ax)

    fig.suptitle(
        f"loss_diff_clone1_gt0p7_linear | clone {CELL_TYPE} (n={n_cells}, p={p_value:.3g}) | "
        "covariate whitening: calicost_tumor_proportion only (not total_counts)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)

    rho_mt, p_mt = spearmanr(covariate_values["pct_mt"], true_isodepth)
    rho_tc, p_tc = spearmanr(covariate_values["total_counts"], true_isodepth)
    print(f"[write] {OUT_PATH}")
    print(f"  rho(pct_mt)={rho_mt:.2f} (p={p_mt:.2g})")
    print(f"  rho(total_counts)={rho_tc:.2f} (p={p_tc:.2g})")


if __name__ == "__main__":
    main()
