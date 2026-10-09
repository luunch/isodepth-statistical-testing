"""Read-only CalicoST cohort survey: QC cache, summary tables, and contact-sheet plots.

Visualizes all CalicoST h5ad slices together without applying filters, writing configs,
or modifying h5ad files. Candidate thresholds appear as reference lines/overlays only.

Usage (from repo root, isodepth_env):
    python scripts/posthoc/calicost_cohort_survey.py --stage all
    python scripts/posthoc/calicost_cohort_survey.py --stage qc
    python scripts/posthoc/calicost_cohort_survey.py --stage tables
    python scripts/posthoc/calicost_cohort_survey.py --stage plots
    python scripts/posthoc/calicost_cohort_survey.py --stage spotsweeper --out-subdir gt1clone --min-clones 2
    python scripts/posthoc/calicost_cohort_survey.py --stage inclusion --out-subdir gt1clone/inclusion_gt0p7_qc60 --min-clones 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.core.paths import repo_root  # noqa: E402

REPO = repo_root(__file__)

from analysis.plots import (  # noqa: E402
    _calicost_clone_base_color,
    _clone_tumor_proportion_colors,
    _normalize_depth,
    _point_size,
    _square_spatial_axis_limits,
)
from data.h5ad_loader import _detect_coordinate_um_per_unit  # noqa: E402
from data.spatial_regions import _auto_eps, _split_dbscan  # noqa: E402

H5AD_DIR = REPO / "data/h5ad/calicost"
OUT_DIR = REPO / "results/calicost/cohort_survey"
QC_CACHE_DIR = OUT_DIR / "qc_cache"

TUMOR_THRESHOLDS = (0.3, 0.5, 0.7, 0.8)
DENOISE_RADIUS_UM = 300.0
CHUNK_SIZE = 512

SPOTSWEEPER_K = 36  # 3 concentric Visium hex rings (OSTA / SpotSweeper default)
SPOTSWEEPER_CUTOFF = 3.0
SPOTSWEEPER_METRICS: tuple[tuple[str, str, str], ...] = (
    ("total_counts", "lower", "total_counts_outliers"),
    ("detected", "lower", "detected_outliers"),
    ("pct_counts_mt", "higher", "pct_counts_mt_outliers"),
)

INCLUSION_TUMOR_THRESHOLD = 0.7
INCLUSION_MIN_SPOTS = 60
INCLUSION_MIN_CLUSTER_SPOTS = 50
INCLUSION_DEFAULT_SUBDIR = "gt1clone/inclusion_gt0p7_qc60"
SELECTED_EXISTENCE_CLUSTERS_CSV = (
    REPO / "configs/calicost/inclusion_gt0p7_qc60/selected_clusters.csv"
)
ISODEPTH_CMAP = "viridis"
ISODEPTH_COLORBAR_LABEL = "Normalized isodepth"
DBSCAN_EPS_MULT = 3.0
DBSCAN_MIN_SAMPLES = 10

SPECIAL_FLAGS: dict[str, str] = {
    "HT270P1-S1H1Fs5U1Bp1": "NO CLONE LABELS",
    "HT112C1-U1_ST_Bn1_paste_fig4": "DUPLICATE of HT112C1-U1",
}


def _list_h5ad_paths(h5ad_dir: Path) -> list[Path]:
    return sorted(h5ad_dir.glob("*.h5ad"))


def _gene_masks(var_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    upper = np.asarray(var_names, dtype=str)
    upper_u = np.char.upper(upper)
    mt = np.char.startswith(upper_u, "MT-")
    ribo = np.char.startswith(upper_u, "RPL") | (
        np.char.startswith(upper_u, "RPS") & ~np.char.startswith(upper_u, "RPS6K")
    )
    return mt, ribo


def _top20_gene_mask(counts_layer, n_genes: int) -> np.ndarray:
    gene_totals = np.zeros(n_genes, dtype=np.float64)
    n_obs = counts_layer.shape[0]
    for start in range(0, n_obs, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_obs)
        block = counts_layer[start:end]
        if sp.issparse(block):
            block = block.toarray()
        gene_totals += np.asarray(block, dtype=np.float64).sum(axis=0)
    top_idx = np.argsort(gene_totals)[::-1][:20]
    mask = np.zeros(n_genes, dtype=bool)
    mask[top_idx] = True
    return mask


def _compute_qc_for_slice(h5ad_path: Path, *, force: bool = False) -> Path:
    slice_id = h5ad_path.stem
    cache_path = QC_CACHE_DIR / f"{slice_id}.csv"
    meta_path = QC_CACHE_DIR / f"{slice_id}_meta.json"
    if cache_path.exists() and meta_path.exists() and not force:
        return cache_path

    QC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(h5ad_path, backed="r")

    if "counts" not in adata.layers:
        raise KeyError(f"{h5ad_path} missing layers['counts']")
    counts_layer = adata.layers["counts"]
    n_obs, n_genes = adata.n_obs, adata.n_vars
    var_names = np.asarray(adata.var_names)
    mt_mask, ribo_mask = _gene_masks(var_names)
    top20_mask = _top20_gene_mask(counts_layer, n_genes)

    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    um_per_unit = _detect_coordinate_um_per_unit(adata, h5ad_path=str(h5ad_path))

    clone_label = (
        adata.obs["calicost_clone_label"].astype(str).to_numpy()
        if "calicost_clone_label" in adata.obs.columns
        else np.full(n_obs, "", dtype=object)
    )
    tumor_prop = (
        adata.obs["calicost_tumor_proportion"].to_numpy(dtype=np.float64)
        if "calicost_tumor_proportion" in adata.obs.columns
        else np.full(n_obs, np.nan, dtype=np.float64)
    )

    total_counts = np.zeros(n_obs, dtype=np.float64)
    n_genes_detected = np.zeros(n_obs, dtype=np.int32)
    mt_counts = np.zeros(n_obs, dtype=np.float64)
    ribo_counts = np.zeros(n_obs, dtype=np.float64)
    top20_counts = np.zeros(n_obs, dtype=np.float64)

    for start in range(0, n_obs, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_obs)
        block = counts_layer[start:end]
        if sp.issparse(block):
            block = block.toarray()
        block = np.asarray(block, dtype=np.float64)
        row_sums = block.sum(axis=1)
        total_counts[start:end] = row_sums
        n_genes_detected[start:end] = (block > 0).sum(axis=1)
        mt_counts[start:end] = block[:, mt_mask].sum(axis=1)
        ribo_counts[start:end] = block[:, ribo_mask].sum(axis=1)
        top20_counts[start:end] = block[:, top20_mask].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_mt = np.where(total_counts > 0, 100.0 * mt_counts / total_counts, 0.0)
        pct_ribo = np.where(total_counts > 0, 100.0 * ribo_counts / total_counts, 0.0)
        pct_top20 = np.where(total_counts > 0, 100.0 * top20_counts / total_counts, 0.0)

    log1p_total_counts = np.log1p(np.maximum(total_counts, 0.0))
    log1p_n_genes = np.log1p(np.maximum(n_genes_detected.astype(np.float64), 0.0))

    tree = cKDTree(spatial)
    nn_dist = tree.query(spatial, k=2)[0][:, 1]
    median_nn_coord = float(np.median(nn_dist))
    median_nn_um = median_nn_coord * float(um_per_unit) if um_per_unit else np.nan

    df = pd.DataFrame(
        {
            "barcode": adata.obs_names.to_numpy(),
            "x": spatial[:, 0],
            "y": spatial[:, 1],
            "x_um": spatial[:, 0] * um_per_unit if um_per_unit else np.nan,
            "y_um": spatial[:, 1] * um_per_unit if um_per_unit else np.nan,
            "calicost_clone_label": clone_label,
            "calicost_tumor_proportion": tumor_prop,
            "total_counts": total_counts,
            "log1p_total_counts": log1p_total_counts,
            "n_genes": n_genes_detected,
            "log1p_n_genes": log1p_n_genes,
            "pct_mt": pct_mt,
            "pct_ribo": pct_ribo,
            "pct_top20": pct_top20,
            "nn_dist_coord": nn_dist,
            "nn_dist_um": nn_dist * um_per_unit if um_per_unit else np.nan,
        }
    )
    df.to_csv(cache_path, index=False)

    meta = {
        "slice_id": slice_id,
        "h5ad_path": str(h5ad_path),
        "n_spots": n_obs,
        "n_genes": n_genes,
        "um_per_unit": um_per_unit,
        "median_nn_coord": median_nn_coord,
        "median_nn_um": median_nn_um,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[qc] cached {slice_id}: {n_obs} spots -> {cache_path.name}")
    return cache_path


def run_qc_stage(h5ad_dir: Path, *, force: bool = False) -> list[Path]:
    paths = _list_h5ad_paths(h5ad_dir)
    if not paths:
        raise FileNotFoundError(f"No h5ad files in {h5ad_dir}")
    return [_compute_qc_for_slice(p, force=force) for p in paths]


def _load_slice_dataframe(h5ad_path: Path) -> tuple[pd.DataFrame, dict]:
    slice_id = h5ad_path.stem
    cache_path = QC_CACHE_DIR / f"{slice_id}.csv"
    meta_path = QC_CACHE_DIR / f"{slice_id}_meta.json"
    if not cache_path.exists():
        _compute_qc_for_slice(h5ad_path)
    df = pd.read_csv(cache_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return df, meta


def _count_components(xy: np.ndarray, radius_um: float, um_per_unit: float) -> int:
    n = len(xy)
    if n <= 1:
        return n
    radius_coord = radius_um / um_per_unit
    tree = cKDTree(xy)
    pairs = tree.query_pairs(radius_coord, output_type="ndarray")
    if len(pairs) == 0:
        return n
    row, col = pairs[:, 0], pairs[:, 1]
    data_vals = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix(
        (
            np.concatenate([data_vals, data_vals]),
            (np.concatenate([row, col]), np.concatenate([col, row])),
        ),
        shape=(n, n),
    )
    n_comp, _ = connected_components(adj, directed=False)
    return int(n_comp)


def _build_tables(h5ad_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    slice_rows: list[dict] = []
    clone_rows: list[dict] = []

    for h5ad_path in h5ad_paths:
        df, meta = _load_slice_dataframe(h5ad_path)
        slice_id = h5ad_path.stem
        um = meta.get("um_per_unit")
        n_spots = len(df)
        labeled = df["calicost_clone_label"].astype(str).str.len() > 0
        labeled &= df["calicost_clone_label"].astype(str) != "nan"
        labeled_frac = float(labeled.sum()) / n_spots if n_spots else 0.0

        clones = sorted(
            c for c in df["calicost_clone_label"].astype(str).unique()
            if c and c != "nan"
        )
        n_clones = len(clones)

        x_um = df["x_um"].to_numpy(dtype=np.float64)
        y_um = df["y_um"].to_numpy(dtype=np.float64)
        valid_um = np.isfinite(x_um) & np.isfinite(y_um)
        if valid_um.any():
            bbox_x = float(x_um[valid_um].max() - x_um[valid_um].min())
            bbox_y = float(y_um[valid_um].max() - y_um[valid_um].min())
        else:
            bbox_x = bbox_y = np.nan

        tp = df["calicost_tumor_proportion"].to_numpy(dtype=np.float64)
        retained = {}
        for thr in TUMOR_THRESHOLDS:
            key = f"retained_frac_{str(thr).replace('.', 'p')}"
            retained[key] = float(np.nanmean(tp > thr)) if np.isfinite(tp).any() else np.nan

        slice_rows.append(
            {
                "slice_id": slice_id,
                "n_spots": n_spots,
                "n_genes": meta.get("n_genes"),
                "n_clones": n_clones,
                "labeled_fraction": labeled_frac,
                "median_total_counts": float(np.median(df["total_counts"])),
                "iqr_total_counts": float(
                    np.subtract(*np.percentile(df["total_counts"], [75, 25]))
                ),
                "median_n_genes": float(np.median(df["n_genes"])),
                "median_pct_mt": float(np.median(df["pct_mt"])),
                "p90_pct_mt": float(np.percentile(df["pct_mt"], 90)),
                "coordinate_um_per_unit": um,
                "median_nn_spacing_um": meta.get("median_nn_um"),
                "bbox_extent_um_x": bbox_x,
                "bbox_extent_um_y": bbox_y,
                "special_flag": SPECIAL_FLAGS.get(slice_id, ""),
                **retained,
            }
        )

        for clone in clones:
            mask = df["calicost_clone_label"].astype(str) == clone
            sub = df.loc[mask]
            xy = sub[["x", "y"]].to_numpy(dtype=np.float64)
            n_clone = len(sub)
            tp_clone = sub["calicost_tumor_proportion"].to_numpy(dtype=np.float64)

            row: dict = {
                "slice_id": slice_id,
                "clone_label": clone,
                "n_spots_all": n_clone,
            }
            for thr in TUMOR_THRESHOLDS:
                key = f"n_spots_gt_{str(thr).replace('.', 'p')}"
                row[key] = int(np.nansum(tp_clone > thr)) if n_clone else 0

            if um and n_clone > 0:
                row["n_components_300um"] = _count_components(xy, DENOISE_RADIUS_UM, float(um))
            else:
                row["n_components_300um"] = np.nan

            clone_rows.append(row)

    return pd.DataFrame(slice_rows), pd.DataFrame(clone_rows)


def run_tables_stage(h5ad_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _list_h5ad_paths(h5ad_dir)
    slice_df, clone_df = _build_tables(paths)
    slice_path = OUT_DIR / "survey_slice_table.csv"
    clone_path = OUT_DIR / "survey_clone_table.csv"
    slice_df.to_csv(slice_path, index=False)
    clone_df.to_csv(clone_path, index=False)
    print(f"[tables] wrote {slice_path.name} ({len(slice_df)} rows)")
    print(f"[tables] wrote {clone_path.name} ({len(clone_df)} rows)")
    return slice_df, clone_df


def _grid_shape(n: int) -> tuple[int, int]:
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _contact_sheet_figsize(
    n: int,
    *,
    panel_w: float = 3.2,
    panel_h: float = 3.5,
) -> tuple[int, int, tuple[float, float]]:
    nrows, ncols = _grid_shape(n)
    return nrows, ncols, (panel_w * ncols, panel_h * nrows)


def _style_spatial_ax(
    ax: plt.Axes,
    title: str,
    *,
    title_fontsize: float = 7,
) -> None:
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=title_fontsize, pad=4)
    ax.set_xlabel("x (um)", fontsize=6)
    ax.set_ylabel("y (um)", fontsize=6)
    ax.tick_params(labelsize=5)


def _save_contact_sheet(
    fig: plt.Figure,
    out_path: Path,
    *,
    suptitle: str | None = None,
    title_lines: int = 2,
    has_colorbar: bool = False,
    cbar_mappable=None,
    cbar_label: str | None = None,
) -> None:
    """Save a grid contact sheet with spacing that avoids title overlap."""
    hspace = 0.40 + 0.16 * max(title_lines - 1, 0)
    top = 0.90 if suptitle else 0.96
    right = 0.90 if has_colorbar else 0.98
    fig.subplots_adjust(
        left=0.08,
        right=right,
        top=top,
        bottom=0.08,
        hspace=hspace,
        wspace=0.30,
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.98)
    if has_colorbar and cbar_mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.62])
        fig.colorbar(cbar_mappable, cax=cbar_ax, label=cbar_label)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _short_title(slice_id: str, n_spots: int, n_clones: int) -> str:
    flag = SPECIAL_FLAGS.get(slice_id, "")
    base = f"{slice_id}\n(n={n_spots}, clones={n_clones})"
    if flag:
        base += f"\n[{flag}]"
    return base


def _clone_color_map(clones: list[str]) -> dict[str, np.ndarray]:
    return {
        clone: _calicost_clone_base_color(clone, fallback_index=i, n_types=len(clones))
        for i, clone in enumerate(clones)
    }


def _filter_h5ad_paths(
    h5ad_paths: list[Path],
    slice_df: pd.DataFrame,
    *,
    min_clones: int = 1,
) -> list[Path]:
    if min_clones <= 1:
        return h5ad_paths
    lookup = slice_df.set_index("slice_id")["n_clones"]
    return [p for p in h5ad_paths if int(lookup.get(p.stem, 0)) > min_clones - 1]


def _plot_spatial_clones(
    h5ad_paths: list[Path],
    slice_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    n = len(h5ad_paths)
    nrows, ncols, figsize = _contact_sheet_figsize(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    lookup = slice_df.set_index("slice_id")

    title_lines = 3 if any(SPECIAL_FLAGS.get(p.stem) for p in h5ad_paths) else 2

    for ax, h5ad_path in zip(axes, h5ad_paths):
        df, _ = _load_slice_dataframe(h5ad_path)
        slice_id = h5ad_path.stem
        info = lookup.loc[slice_id]
        clones = sorted(
            c for c in df["calicost_clone_label"].astype(str).unique()
            if c and c != "nan"
        )
        color_map = _clone_color_map(clones)
        tp = df["calicost_tumor_proportion"].to_numpy(dtype=np.float32)
        tp = np.nan_to_num(tp, nan=0.0)

        for clone in clones:
            mask = df["calicost_clone_label"].astype(str).values == clone
            if not mask.any():
                continue
            base = color_map[clone]
            colors = _clone_tumor_proportion_colors(base, tp[mask])
            ax.scatter(df.loc[mask, "x_um"], df.loc[mask, "y_um"], c=colors, s=4, linewidths=0)

        unlabeled = ~df["calicost_clone_label"].astype(str).isin(clones)
        if unlabeled.any():
            ax.scatter(
                df.loc[unlabeled, "x_um"],
                df.loc[unlabeled, "y_um"],
                c="#cccccc",
                s=3,
                alpha=0.5,
                linewidths=0,
            )

        _style_spatial_ax(
            ax,
            _short_title(slice_id, int(info["n_spots"]), int(info["n_clones"])),
        )

    for ax in axes[n:]:
        ax.axis("off")

    out_path = out_dir / "survey_spatial_clones.png"
    _save_contact_sheet(
        fig,
        out_path,
        suptitle="CalicoST Cohort - Clone Labels (fade = tumor proportion)",
        title_lines=title_lines,
    )
    print(f"[plots] wrote {out_path}")


def _plot_spatial_tumor_prop(
    h5ad_paths: list[Path],
    slice_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    n = len(h5ad_paths)
    nrows, ncols, figsize = _contact_sheet_figsize(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    lookup = slice_df.set_index("slice_id")
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    title_lines = 3 if any(SPECIAL_FLAGS.get(p.stem) for p in h5ad_paths) else 2

    for ax, h5ad_path in zip(axes, h5ad_paths):
        df, _ = _load_slice_dataframe(h5ad_path)
        slice_id = h5ad_path.stem
        info = lookup.loc[slice_id]
        tp = df["calicost_tumor_proportion"].to_numpy(dtype=np.float64)
        ax.scatter(
            df["x_um"],
            df["y_um"],
            c=np.clip(tp, 0, 1),
            cmap=cmap,
            norm=norm,
            s=5,
            linewidths=0,
        )
        _style_spatial_ax(
            ax,
            _short_title(slice_id, int(info["n_spots"]), int(info["n_clones"])),
        )

    for ax in axes[n:]:
        ax.axis("off")

    out_path = out_dir / "survey_spatial_tumor_prop.png"
    _save_contact_sheet(
        fig,
        out_path,
        suptitle="CalicoST Cohort - Tumor Proportion",
        title_lines=title_lines,
        has_colorbar=True,
        cbar_mappable=sm,
        cbar_label="calicost_tumor_proportion",
    )
    print(f"[plots] wrote {out_path}")


def _plot_spatial_tumor_gating(
    h5ad_paths: list[Path],
    slice_df: pd.DataFrame,
    out_dir: Path,
    thresholds: tuple[float, ...] = TUMOR_THRESHOLDS,
) -> pd.DataFrame:
    """One contact sheet per tumor-proportion gate; clone color = pass, gray = excluded."""
    lookup = slice_df.set_index("slice_id")
    summary_rows: list[dict] = []

    for thr in thresholds:
        n = len(h5ad_paths)
        nrows, ncols, figsize = _contact_sheet_figsize(n, panel_h=3.8)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()
        title_lines = 4 if any(SPECIAL_FLAGS.get(p.stem) for p in h5ad_paths) else 3

        for ax, h5ad_path in zip(axes, h5ad_paths):
            df, _ = _load_slice_dataframe(h5ad_path)
            slice_id = h5ad_path.stem
            info = lookup.loc[slice_id]
            tp = df["calicost_tumor_proportion"].to_numpy(dtype=np.float64)
            clone_labels = df["calicost_clone_label"].astype(str).values
            clones = sorted(c for c in np.unique(clone_labels) if c and c != "nan")
            color_map = _clone_color_map(clones)

            labeled = np.isin(clone_labels, clones)
            pass_mask = labeled & np.isfinite(tp) & (tp > thr)
            fail_mask = ~pass_mask

            if fail_mask.any():
                ax.scatter(
                    df.loc[fail_mask, "x_um"],
                    df.loc[fail_mask, "y_um"],
                    c="#cccccc",
                    s=3,
                    alpha=0.45,
                    linewidths=0,
                )

            for clone in clones:
                mask = pass_mask & (clone_labels == clone)
                if not mask.any():
                    continue
                ax.scatter(
                    df.loc[mask, "x_um"],
                    df.loc[mask, "y_um"],
                    c=[color_map[clone]],
                    s=4,
                    linewidths=0,
                )

            n_pass = int(pass_mask.sum())
            frac = n_pass / len(df) if len(df) else 0.0
            clone_parts = [
                f"{clone}:{int((pass_mask & (clone_labels == clone)).sum())}"
                for clone in clones
            ]
            subtitle = _short_title(slice_id, int(info["n_spots"]), int(info["n_clones"]))
            _style_spatial_ax(
                ax,
                f"{subtitle}\npass={frac:.1%} ({', '.join(clone_parts)})",
                title_fontsize=6,
            )

            row: dict = {
                "slice_id": slice_id,
                "threshold": thr,
                "n_spots": len(df),
                "n_pass": n_pass,
                "frac_pass": frac,
            }
            for clone in clones:
                key = f"clone_{clone.replace('.', 'p')}"
                row[key] = int((pass_mask & (clone_labels == clone)).sum())
            summary_rows.append(row)

        for ax in axes[n:]:
            ax.axis("off")

        thr_tag = str(thr).replace(".", "p")
        out_path = out_dir / f"survey_spatial_gating_gt{thr_tag}.png"
        _save_contact_sheet(
            fig,
            out_path,
            suptitle=f"Tumor proportion gate > {thr} — clone color = pass, gray = excluded",
            title_lines=title_lines,
        )
        print(f"[plots] wrote {out_path}")

    return pd.DataFrame(summary_rows)


def _ensure_violin_data(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size >= 2:
        return finite
    if finite.size == 1:
        return np.array([finite[0], finite[0]], dtype=np.float64)
    return np.array([0.0, 0.0], dtype=np.float64)


def _plot_tumor_prop_dist(h5ad_paths: list[Path]) -> None:
    data = []
    labels = []
    retained_text = []
    for h5ad_path in h5ad_paths:
        df, _ = _load_slice_dataframe(h5ad_path)
        slice_id = h5ad_path.stem
        tp_raw = df["calicost_tumor_proportion"].to_numpy(dtype=np.float64)
        tp_finite = tp_raw[np.isfinite(tp_raw)]
        data.append(_ensure_violin_data(tp_raw))
        labels.append(slice_id)
        parts = []
        for thr in TUMOR_THRESHOLDS:
            frac = float(np.mean(tp_finite > thr)) if tp_finite.size else np.nan
            parts.append(f">{thr:.1f}:{frac:.0%}")
        retained_text.append(" ".join(parts))

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.35), 6))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.6)
    for thr in TUMOR_THRESHOLDS:
        ax.axhline(thr, color="crimson", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("calicost_tumor_proportion")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Tumor Proportion Distribution per Slice (dashed = candidate thresholds)")

    for i, txt in enumerate(retained_text, start=1):
        ax.text(i, 1.04, txt, ha="center", va="bottom", fontsize=4, rotation=90)

    fig.subplots_adjust(top=0.82, bottom=0.22)
    fig.savefig(OUT_DIR / "survey_tumor_prop_dist.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("[plots] wrote survey_tumor_prop_dist.png")


def _plot_qc_distributions(h5ad_paths: list[Path]) -> None:
    metrics = [
        ("log1p_total_counts", "log1p(total counts)"),
        ("n_genes", "n_genes detected"),
        ("pct_mt", "pct_mt"),
        ("pct_top20", "pct counts in top-20 genes"),
    ]
    slice_ids = [p.stem for p in h5ad_paths]
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(slice_ids) * 0.25), 8))
    axes = axes.ravel()

    for ax, (col, ylabel) in zip(axes, metrics):
        data = []
        for h5ad_path in h5ad_paths:
            df, _ = _load_slice_dataframe(h5ad_path)
            vals = df[col].to_numpy(dtype=np.float64)
            data.append(_ensure_violin_data(vals))
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.6)
        ax.set_xticks(np.arange(1, len(slice_ids) + 1))
        ax.set_xticklabels(slice_ids, rotation=90, fontsize=5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    fig.suptitle("Per-Slice QC Metric Distributions", fontsize=11, y=0.98)
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
    fig.savefig(OUT_DIR / "survey_qc_distributions.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("[plots] wrote survey_qc_distributions.png")


def _plot_spatial_qc_metric(
    h5ad_paths: list[Path],
    slice_df: pd.DataFrame,
    metric_col: str,
    out_name: str,
    title: str,
    out_dir: Path,
    cmap_name: str = "viridis",
) -> None:
    n = len(h5ad_paths)
    nrows, ncols, figsize = _contact_sheet_figsize(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    lookup = slice_df.set_index("slice_id")
    title_lines = 3 if any(SPECIAL_FLAGS.get(p.stem) for p in h5ad_paths) else 2

    all_vals = []
    for h5ad_path in h5ad_paths:
        df, _ = _load_slice_dataframe(h5ad_path)
        all_vals.append(df[metric_col].to_numpy(dtype=np.float64))
    vmin = float(np.nanpercentile(np.concatenate(all_vals), 2))
    vmax = float(np.nanpercentile(np.concatenate(all_vals), 98))
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for ax, h5ad_path in zip(axes, h5ad_paths):
        df, _ = _load_slice_dataframe(h5ad_path)
        slice_id = h5ad_path.stem
        info = lookup.loc[slice_id]
        vals = df[metric_col].to_numpy(dtype=np.float64)
        ax.scatter(
            df["x_um"],
            df["y_um"],
            c=vals,
            cmap=cmap,
            norm=norm,
            s=5,
            linewidths=0,
        )
        _style_spatial_ax(
            ax,
            _short_title(slice_id, int(info["n_spots"]), int(info["n_clones"])),
        )

    for ax in axes[n:]:
        ax.axis("off")

    _save_contact_sheet(
        fig,
        out_dir / out_name,
        suptitle=title,
        title_lines=title_lines,
        has_colorbar=True,
        cbar_mappable=sm,
        cbar_label=metric_col,
    )
    print(f"[plots] wrote {out_dir / out_name}")


def _plot_clone_composition(slice_df: pd.DataFrame, clone_df: pd.DataFrame) -> None:
    slice_ids = slice_df["slice_id"].tolist()
    thresholds = [None, *TUMOR_THRESHOLDS]
    ncols = len(thresholds)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.5 * ncols, max(6, len(slice_ids) * 0.22)),
        sharey=True,
    )
    if ncols == 1:
        axes = [axes]

    clone_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    clone_labels = sorted(clone_df["clone_label"].astype(str).unique())

    for ax, thr in zip(axes, thresholds):
        bottom = np.zeros(len(slice_ids))
        for ci, clone in enumerate(clone_labels):
            heights = []
            for sid in slice_ids:
                row = clone_df[
                    (clone_df["slice_id"] == sid) & (clone_df["clone_label"].astype(str) == clone)
                ]
                if row.empty:
                    heights.append(0)
                    continue
                if thr is None:
                    heights.append(int(row.iloc[0]["n_spots_all"]))
                else:
                    col = f"n_spots_gt_{str(thr).replace('.', 'p')}"
                    heights.append(int(row.iloc[0][col]))
            heights_arr = np.asarray(heights, dtype=np.float64)
            ax.barh(slice_ids, heights_arr, left=bottom, color=clone_palette[ci % 20], height=0.8)
            bottom += heights_arr

        title = "all spots" if thr is None else f"tumor_prop > {thr}"
        ax.set_title(title)
        ax.set_xlabel("spot count")
        ax.tick_params(axis="y", labelsize=6)

    axes[0].set_ylabel("slice")
    fig.suptitle("Clone Composition per Slice at Candidate Purity Thresholds", fontsize=11, y=0.98)
    fig.subplots_adjust(top=0.88, wspace=0.25)
    fig.savefig(OUT_DIR / "survey_clone_composition.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("[plots] wrote survey_clone_composition.png")


def _spotsweeper_discard_mask(df: pd.DataFrame, slice_id: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run SpotSweeper local_outliers on standard Visium QC metrics."""
    import spotsweeper.local_outliers as lo

    adata = ad.AnnData(X=np.zeros((len(df), 1), dtype=np.float32))
    adata.obsm["spatial"] = df[["x", "y"]].to_numpy(dtype=np.float64)
    adata.obs["sample_id"] = slice_id
    adata.obs["total_counts"] = df["total_counts"].to_numpy(dtype=np.float64)
    adata.obs["detected"] = df["n_genes"].to_numpy(dtype=np.float64)
    adata.obs["pct_counts_mt"] = df["pct_mt"].to_numpy(dtype=np.float64)

    per_metric: dict[str, np.ndarray] = {}
    for metric, direction, out_col in SPOTSWEEPER_METRICS:
        lo.local_outliers(
            adata,
            metric=metric,
            direction=direction,
            n_neighbors=SPOTSWEEPER_K,
            sample_key="sample_id",
            log=True,
            cutoff=SPOTSWEEPER_CUTOFF,
            coord_key="spatial",
        )
        per_metric[out_col] = adata.obs[out_col].to_numpy(dtype=bool)

    combined = np.zeros(len(df), dtype=bool)
    for flags in per_metric.values():
        combined |= flags
    return combined, per_metric


def _plot_spotsweeper_discard(
    h5ad_paths: list[Path],
    slice_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    n = len(h5ad_paths)
    nrows, ncols, figsize = _contact_sheet_figsize(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    lookup = slice_df.set_index("slice_id")
    summary_rows: list[dict] = []
    title_lines = 3 if any(SPECIAL_FLAGS.get(p.stem) for p in h5ad_paths) else 2

    for ax, h5ad_path in zip(axes, h5ad_paths):
        df, _ = _load_slice_dataframe(h5ad_path)
        slice_id = h5ad_path.stem
        info = lookup.loc[slice_id]
        discard, per_metric = _spotsweeper_discard_mask(df, slice_id)
        keep = ~discard

        if keep.any():
            ax.scatter(
                df.loc[keep, "x_um"],
                df.loc[keep, "y_um"],
                c="#4c78a8",
                s=4,
                alpha=0.5,
                linewidths=0,
            )
        if discard.any():
            ax.scatter(
                df.loc[discard, "x_um"],
                df.loc[discard, "y_um"],
                c="#e45756",
                s=8,
                alpha=0.95,
                linewidths=0,
            )

        frac = float(discard.mean()) if len(df) else 0.0
        subtitle = _short_title(slice_id, int(info["n_spots"]), int(info["n_clones"]))
        _style_spatial_ax(ax, f"{subtitle}\nSpotSweeper discard={frac:.1%}")

        summary_rows.append(
            {
                "slice_id": slice_id,
                "n_spots": len(df),
                "n_discard": int(discard.sum()),
                "frac_discard": frac,
                "n_total_counts": int(per_metric["total_counts_outliers"].sum()),
                "n_detected": int(per_metric["detected_outliers"].sum()),
                "n_pct_mt": int(per_metric["pct_counts_mt_outliers"].sum()),
            }
        )

    for ax in axes[n:]:
        ax.axis("off")

    out_path = out_dir / "survey_spotsweeper_discard.png"
    _save_contact_sheet(
        fig,
        out_path,
        suptitle=(
            "SpotSweeper local QC (k=36, 3 MAD) — red = would remove "
            "(low counts/genes or high pct_mt vs neighbors)"
        ),
        title_lines=title_lines,
    )
    print(f"[spotsweeper] wrote {out_path}")
    return pd.DataFrame(summary_rows)


def _is_survey_duplicate(slice_id: str) -> bool:
    flag = SPECIAL_FLAGS.get(slice_id, "")
    return flag.upper().startswith("DUPLICATE")


def _collect_inclusion_clones(
    h5ad_paths: list[Path],
    *,
    tumor_threshold: float = INCLUSION_TUMOR_THRESHOLD,
    min_spots: int = INCLUSION_MIN_SPOTS,
) -> tuple[list[dict], pd.DataFrame]:
    """Return passing (slice, clone) pairs and a full audit table of all clones."""
    passing: list[dict] = []
    audit_rows: list[dict] = []

    for h5ad_path in h5ad_paths:
        slice_id = h5ad_path.stem
        if _is_survey_duplicate(slice_id):
            continue

        df, _ = _load_slice_dataframe(h5ad_path)
        discard, _ = _spotsweeper_discard_mask(df, slice_id)
        tp = df["calicost_tumor_proportion"].to_numpy(dtype=np.float64)
        clone_labels = df["calicost_clone_label"].astype(str).values
        clones = sorted(c for c in np.unique(clone_labels) if c and c != "nan")
        color_map = _clone_color_map(clones)

        for clone in clones:
            clone_mask = clone_labels == clone
            n_all = int(clone_mask.sum())
            n_discard = int((clone_mask & discard).sum())
            included = clone_mask & (~discard) & np.isfinite(tp) & (tp > tumor_threshold)
            n_included = int(included.sum())
            row = {
                "slice_id": slice_id,
                "clone_label": clone,
                "n_spots_all": n_all,
                "n_spotsweeper_discard": n_discard,
                "n_included": n_included,
                "passes_min_spots": n_included >= min_spots,
            }
            audit_rows.append(row)
            if n_included >= min_spots:
                passing.append(
                    {
                        "slice_id": slice_id,
                        "clone_label": clone,
                        "n_included": n_included,
                        "included_mask": included,
                        "df": df,
                        "discard": discard,
                        "color": color_map[clone],
                    }
                )

    passing.sort(key=lambda r: (r["slice_id"], r["clone_label"]))
    return passing, pd.DataFrame(audit_rows)


def _plot_inclusion_survey(
    passing: list[dict],
    out_dir: Path,
    *,
    tumor_threshold: float = INCLUSION_TUMOR_THRESHOLD,
    min_spots: int = INCLUSION_MIN_SPOTS,
) -> None:
    if not passing:
        print("[inclusion] no clones pass filters; skipping plot")
        return

    n = len(passing)
    nrows, ncols, figsize = _contact_sheet_figsize(n, panel_w=3.0, panel_h=3.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax, entry in zip(axes, passing):
        df = entry["df"]
        included = entry["included_mask"]
        discard = entry["discard"]
        background = ~included

        if background.any():
            bg = df.loc[background]
            bg_discard = discard[background]
            if (~bg_discard).any():
                ax.scatter(
                    bg.loc[~bg_discard, "x_um"],
                    bg.loc[~bg_discard, "y_um"],
                    c="#dddddd",
                    s=2,
                    alpha=0.35,
                    linewidths=0,
                )
            if bg_discard.any():
                ax.scatter(
                    bg.loc[bg_discard, "x_um"],
                    bg.loc[bg_discard, "y_um"],
                    c="#f5f5f5",
                    s=2,
                    alpha=0.25,
                    linewidths=0,
                )

        ax.scatter(
            df.loc[included, "x_um"],
            df.loc[included, "y_um"],
            c=[entry["color"]],
            s=5,
            linewidths=0,
        )

        clone = entry["clone_label"]
        sid = entry["slice_id"]
        n_inc = entry["n_included"]
        _style_spatial_ax(ax, f"{sid}\nclone {clone}  n={n_inc}", title_fontsize=6)

    for ax in axes[n:]:
        ax.axis("off")

    out_path = out_dir / "survey_inclusion_clones.png"
    _save_contact_sheet(
        fig,
        out_path,
        suptitle=(
            f"Inclusion survey: SpotSweeper pass + tumor_prop > {tumor_threshold} + "
            f">= {min_spots} spots per clone (one panel per passing clone)"
        ),
        title_lines=3,
    )
    print(f"[inclusion] wrote {out_path}")


def _cluster_inclusion_clones(
    passing: list[dict],
    *,
    min_cluster_spots: int = INCLUSION_MIN_CLUSTER_SPOTS,
    eps_mult: float = DBSCAN_EPS_MULT,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> tuple[list[dict], pd.DataFrame]:
    """DBSCAN each passing clone's included spots; keep clusters >= min_cluster_spots."""
    cluster_entries: list[dict] = []
    audit_rows: list[dict] = []

    for entry in passing:
        df = entry["df"]
        included = entry["included_mask"]
        xy = df.loc[included, ["x_um", "y_um"]].to_numpy(dtype=np.float64)
        n_included = int(included.sum())

        if n_included < min_samples:
            audit_rows.append(
                {
                    "slice_id": entry["slice_id"],
                    "clone_label": entry["clone_label"],
                    "cluster_id": np.nan,
                    "n_spots": n_included,
                    "passes_min_cluster_spots": False,
                    "note": "too_few_for_dbscan",
                }
            )
            continue

        eps = _auto_eps(xy, eps_mult)
        labels = _split_dbscan(xy, eps=eps, min_samples=min_samples)
        included_indices = np.flatnonzero(included)

        for cluster_id in sorted({int(c) for c in labels}):
            local_mask = labels == cluster_id
            n_spots = int(local_mask.sum())
            is_noise = cluster_id == -1
            passes = (not is_noise) and n_spots >= min_cluster_spots
            audit_rows.append(
                {
                    "slice_id": entry["slice_id"],
                    "clone_label": entry["clone_label"],
                    "cluster_id": cluster_id,
                    "n_spots": n_spots,
                    "passes_min_cluster_spots": passes,
                    "note": "noise" if is_noise else "",
                    "dbscan_eps_um": eps,
                }
            )
            if not passes:
                continue

            cluster_mask = np.zeros(len(df), dtype=bool)
            cluster_mask[included_indices[local_mask]] = True
            cluster_entries.append(
                {
                    "slice_id": entry["slice_id"],
                    "clone_label": entry["clone_label"],
                    "cluster_id": cluster_id,
                    "n_spots": n_spots,
                    "cluster_mask": cluster_mask,
                    "df": df,
                    "discard": entry["discard"],
                    "included_mask": included,
                    "color": entry["color"],
                    "dbscan_eps_um": eps,
                }
            )

    cluster_entries.sort(
        key=lambda r: (r["slice_id"], r["clone_label"], r["cluster_id"]),
    )
    return cluster_entries, pd.DataFrame(audit_rows)


def _draw_inclusion_cluster_background(
    ax: plt.Axes,
    df: pd.DataFrame,
    cluster_mask: np.ndarray,
    included: np.ndarray,
    discard: np.ndarray,
) -> None:
    background = ~cluster_mask
    if not background.any():
        return
    bg = df.loc[background]
    bg_discard = discard[background]
    bg_other_included = included[background]
    bg_plain = (~bg_discard) & (~bg_other_included)
    if bg_plain.any():
        ax.scatter(
            bg.loc[bg_plain, "x_um"],
            bg.loc[bg_plain, "y_um"],
            c="#dddddd",
            s=2,
            alpha=0.35,
            linewidths=0,
        )
    if (bg_other_included & ~bg_discard).any():
        ax.scatter(
            bg.loc[bg_other_included & ~bg_discard, "x_um"],
            bg.loc[bg_other_included & ~bg_discard, "y_um"],
            c="#bbbbbb",
            s=2,
            alpha=0.45,
            linewidths=0,
        )
    if bg_discard.any():
        ax.scatter(
            bg.loc[bg_discard, "x_um"],
            bg.loc[bg_discard, "y_um"],
            c="#f5f5f5",
            s=2,
            alpha=0.25,
            linewidths=0,
        )


def _plot_inclusion_dbscan_clusters(
    clusters: list[dict],
    out_dir: Path,
    *,
    min_cluster_spots: int = INCLUSION_MIN_CLUSTER_SPOTS,
) -> None:
    if not clusters:
        print("[inclusion] no DBSCAN clusters pass min size; skipping cluster plot")
        return

    n = len(clusters)
    nrows, ncols, figsize = _contact_sheet_figsize(n, panel_w=3.0, panel_h=3.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax, entry in zip(axes, clusters):
        df = entry["df"]
        cluster_mask = entry["cluster_mask"]
        included = entry["included_mask"]
        discard = entry["discard"]
        _draw_inclusion_cluster_background(ax, df, cluster_mask, included, discard)

        ax.scatter(
            df.loc[cluster_mask, "x_um"],
            df.loc[cluster_mask, "y_um"],
            c=[entry["color"]],
            s=5,
            linewidths=0,
        )

        _style_spatial_ax(
            ax,
            f"{entry['slice_id']}\nclone {entry['clone_label']}  "
            f"cluster {entry['cluster_id']}  n={entry['n_spots']}",
            title_fontsize=6,
        )

    for ax in axes[n:]:
        ax.axis("off")

    out_path = out_dir / "survey_inclusion_dbscan_clusters.png"
    _save_contact_sheet(
        fig,
        out_path,
        suptitle=(
            f"DBSCAN spatial clusters on included spots (eps=auto x{DBSCAN_EPS_MULT}, "
            f"min_samples={DBSCAN_MIN_SAMPLES}) — one panel per cluster with "
            f">= {min_cluster_spots} spots"
        ),
        title_lines=3,
    )
    print(f"[inclusion] wrote {out_path}")


def _norm_clone_label(value: object) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _cluster_identity(
    slice_id: object, clone_label: object, cluster_id: object
) -> tuple[str, str, int]:
    return str(slice_id), _norm_clone_label(clone_label), int(cluster_id)


def _select_existence_test_clusters(
    clusters: list[dict],
    csv_path: Path,
) -> list[dict]:
    """Keep the DBSCAN clusters used for existence / permutation tests, in panel order."""
    selection = pd.read_csv(csv_path)
    by_key = {
        _cluster_identity(entry["slice_id"], entry["clone_label"], entry["cluster_id"]): entry
        for entry in clusters
    }
    selected: list[dict] = []
    missing: list[str] = []
    for rec in selection.to_dict("records"):
        key = _cluster_identity(rec["slice_id"], rec["clone_label"], rec["cluster_id"])
        entry = by_key.get(key)
        if entry is None:
            missing.append(
                f"panel {int(rec['panel'])} {key[0]} clone {key[1]} cluster {key[2]}"
            )
            continue
        selected.append({**entry, "panel": int(rec["panel"])})
    if missing:
        raise ValueError(
            "selected existence clusters missing from DBSCAN results:\n  "
            + "\n  ".join(missing)
        )
    return selected


def _plot_inclusion_cluster_metric_heatmaps(
    clusters: list[dict],
    out_dir: Path,
    *,
    selected_clusters_csv: Path | None = SELECTED_EXISTENCE_CLUSTERS_CSV,
) -> None:
    if selected_clusters_csv is not None and selected_clusters_csv.exists():
        clusters = _select_existence_test_clusters(clusters, selected_clusters_csv)
        print(
            f"[inclusion] metric heatmaps restricted to {len(clusters)} "
            f"existence-test clusters from {selected_clusters_csv}"
        )
    if not clusters:
        print("[inclusion] no clusters for metric heatmaps; skipping")
        return

    metric_specs: tuple[tuple[str, str, str], ...] = (
        (
            "calicost_tumor_proportion",
            "survey_inclusion_spatial_tumor_prop.png",
            "Tumor Proportion — existence-test clusters",
        ),
        (
            "pct_mt",
            "survey_inclusion_spatial_qc_pctmt.png",
            "Mitochondrial Fraction (pct_mt) — existence-test clusters",
        ),
        (
            "total_counts",
            "survey_inclusion_spatial_qc_totalcounts.png",
            "Library Depth (total counts) — existence-test clusters",
        ),
    )
    cmap = plt.get_cmap(ISODEPTH_CMAP)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for metric_col, out_name, title in metric_specs:
        n = len(clusters)
        nrows, ncols, figsize = _contact_sheet_figsize(n, panel_w=3.0, panel_h=3.6)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for ax, entry in zip(axes, clusters):
            df = entry["df"]
            cluster_mask = entry["cluster_mask"]
            vals = df.loc[cluster_mask, metric_col].to_numpy(dtype=np.float64)
            finite = np.isfinite(vals)
            plot_vals = np.full(vals.shape, np.nan, dtype=np.float32)
            if finite.any():
                plot_vals[finite] = _normalize_depth(vals[finite])
            x = df.loc[cluster_mask, "x_um"].to_numpy(dtype=np.float64)
            y = df.loc[cluster_mask, "y_um"].to_numpy(dtype=np.float64)
            ok = np.isfinite(plot_vals)
            coords = np.column_stack([x[ok], y[ok]])
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=plot_vals[ok],
                cmap=cmap,
                norm=norm,
                s=_point_size(coords),
                linewidths=0,
            )
            panel = entry.get("panel")
            panel_prefix = f"panel {int(panel):02d}  " if panel is not None else ""
            _style_spatial_ax(
                ax,
                f"{panel_prefix}{entry['slice_id']}\nclone {entry['clone_label']}  "
                f"cluster {entry['cluster_id']}  n={entry['n_spots']}",
                title_fontsize=6,
            )
            if coords.shape[0] > 0:
                xlim, ylim = _square_spatial_axis_limits(coords)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

        for ax in axes[n:]:
            ax.axis("off")

        out_path = out_dir / out_name
        _save_contact_sheet(
            fig,
            out_path,
            suptitle=title,
            title_lines=3,
            has_colorbar=True,
            cbar_mappable=sm,
            cbar_label=ISODEPTH_COLORBAR_LABEL,
        )
        print(f"[inclusion] wrote {out_path}")


def run_inclusion_survey_stage(
    h5ad_dir: Path,
    slice_df: pd.DataFrame,
    *,
    out_dir: Path,
    min_clones: int = 1,
    tumor_threshold: float = INCLUSION_TUMOR_THRESHOLD,
    min_spots: int = INCLUSION_MIN_SPOTS,
    min_cluster_spots: int = INCLUSION_MIN_CLUSTER_SPOTS,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _filter_h5ad_paths(_list_h5ad_paths(h5ad_dir), slice_df, min_clones=min_clones)
    paths = [p for p in paths if not _is_survey_duplicate(p.stem)]
    if not paths:
        print(f"[inclusion] no slices with n_clones >= {min_clones}; skipping")
        return

    passing, audit_df = _collect_inclusion_clones(
        paths,
        tumor_threshold=tumor_threshold,
        min_spots=min_spots,
    )
    audit_path = out_dir / "survey_inclusion_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"[inclusion] wrote {audit_path} ({int(audit_df['passes_min_spots'].sum())} passing clones)")

    if passing:
        summary_rows = [
            {
                "slice_id": e["slice_id"],
                "clone_label": e["clone_label"],
                "n_included": e["n_included"],
            }
            for e in passing
        ]
        summary_path = out_dir / "survey_inclusion_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"[inclusion] wrote {summary_path}")

    _plot_inclusion_survey(
        passing,
        out_dir,
        tumor_threshold=tumor_threshold,
        min_spots=min_spots,
    )

    clusters, cluster_audit_df = _cluster_inclusion_clones(
        passing,
        min_cluster_spots=min_cluster_spots,
    )
    cluster_audit_path = out_dir / "survey_inclusion_dbscan_audit.csv"
    cluster_audit_df.to_csv(cluster_audit_path, index=False)
    n_pass_clusters = int(cluster_audit_df["passes_min_cluster_spots"].sum())
    print(f"[inclusion] wrote {cluster_audit_path} ({n_pass_clusters} passing clusters)")

    if clusters:
        cluster_summary = pd.DataFrame(
            [
                {
                    "slice_id": e["slice_id"],
                    "clone_label": e["clone_label"],
                    "cluster_id": e["cluster_id"],
                    "n_spots": e["n_spots"],
                    "dbscan_eps_um": e["dbscan_eps_um"],
                }
                for e in clusters
            ]
        )
        cluster_summary_path = out_dir / "survey_inclusion_dbscan_summary.csv"
        cluster_summary.to_csv(cluster_summary_path, index=False)
        print(f"[inclusion] wrote {cluster_summary_path}")

    _plot_inclusion_dbscan_clusters(
        clusters,
        out_dir,
        min_cluster_spots=min_cluster_spots,
    )
    _plot_inclusion_cluster_metric_heatmaps(clusters, out_dir)


def run_spotsweeper_stage(
    h5ad_dir: Path,
    slice_df: pd.DataFrame,
    *,
    out_dir: Path,
    min_clones: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _filter_h5ad_paths(_list_h5ad_paths(h5ad_dir), slice_df, min_clones=min_clones)
    if not paths:
        print(f"[spotsweeper] no slices with n_clones >= {min_clones}; skipping")
        return
    summary = _plot_spotsweeper_discard(paths, slice_df, out_dir)
    summary_path = out_dir / "survey_spotsweeper_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[spotsweeper] wrote {summary_path}")


def run_plots_stage(
    h5ad_dir: Path,
    slice_df: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    min_clones: int = 1,
    spatial_only: bool = True,
) -> None:
    plot_dir = out_dir or OUT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = _filter_h5ad_paths(_list_h5ad_paths(h5ad_dir), slice_df, min_clones=min_clones)
    if not paths:
        print(f"[plots] no slices with n_clones >= {min_clones}; skipping")
        return

    _plot_spatial_clones(paths, slice_df, plot_dir)
    _plot_spatial_tumor_prop(paths, slice_df, plot_dir)
    gating_summary = _plot_spatial_tumor_gating(paths, slice_df, plot_dir)
    gating_summary_path = plot_dir / "survey_tumor_gating_summary.csv"
    gating_summary.to_csv(gating_summary_path, index=False)
    print(f"[plots] wrote {gating_summary_path}")
    if not spatial_only:
        _plot_tumor_prop_dist(paths)
        _plot_qc_distributions(paths)
        clone_df = pd.read_csv(OUT_DIR / "survey_clone_table.csv")
        _plot_clone_composition(slice_df, clone_df)
    _plot_spatial_qc_metric(
        paths,
        slice_df,
        "total_counts",
        "survey_spatial_qc_totalcounts.png",
        "Spatial Library Depth (total counts)",
        plot_dir,
    )
    _plot_spatial_qc_metric(
        paths,
        slice_df,
        "pct_mt",
        "survey_spatial_qc_pctmt.png",
        "Spatial Mitochondrial Fraction (pct_mt)",
        plot_dir,
        cmap_name="magma",
    )
    _plot_spatial_qc_metric(
        paths,
        slice_df,
        "n_genes",
        "survey_spatial_qc_ngenes.png",
        "Spatial Genes Detected per Spot (n_genes)",
        plot_dir,
        cmap_name="plasma",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("qc", "tables", "plots", "spotsweeper", "inclusion", "all"), default="all")
    parser.add_argument(
        "--h5ad-dir",
        type=Path,
        default=H5AD_DIR,
        help="Directory containing CalicoST *.h5ad files",
    )
    parser.add_argument("--force-qc", action="store_true", help="Recompute QC cache even if present")
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Write plot PNGs under OUT_DIR/<subdir> instead of OUT_DIR",
    )
    parser.add_argument(
        "--min-clones",
        type=int,
        default=1,
        help="Include only slices with at least this many clones (use 2 for multi-clone)",
    )
    parser.add_argument(
        "--include-distribution-plots",
        action="store_true",
        help="Also write violin/bar distribution plots (off by default)",
    )
    parser.add_argument(
        "--inclusion-tumor-threshold",
        type=float,
        default=INCLUSION_TUMOR_THRESHOLD,
        help="Tumor proportion gate for inclusion survey (default 0.7)",
    )
    parser.add_argument(
        "--inclusion-min-spots",
        type=int,
        default=INCLUSION_MIN_SPOTS,
        help="Minimum SpotSweeper-passing gt-gated spots per clone (default 60)",
    )
    parser.add_argument(
        "--inclusion-min-cluster-spots",
        type=int,
        default=INCLUSION_MIN_CLUSTER_SPOTS,
        help="Minimum DBSCAN cluster size to include (default 50)",
    )
    args = parser.parse_args()

    h5ad_dir = args.h5ad_dir if args.h5ad_dir.is_absolute() else REPO / args.h5ad_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "inclusion" and args.out_subdir is None:
        args.out_subdir = INCLUSION_DEFAULT_SUBDIR
    plot_dir = OUT_DIR / args.out_subdir if args.out_subdir else OUT_DIR

    slice_df: pd.DataFrame | None = None

    if args.stage in ("qc", "all"):
        run_qc_stage(h5ad_dir, force=args.force_qc)

    if args.stage in ("tables", "all"):
        slice_df, _ = run_tables_stage(h5ad_dir)

    if args.stage in ("plots", "all"):
        if slice_df is None:
            slice_path = OUT_DIR / "survey_slice_table.csv"
            if not slice_path.exists():
                slice_df, _ = run_tables_stage(h5ad_dir)
            else:
                slice_df = pd.read_csv(slice_path)
        run_plots_stage(
            h5ad_dir,
            slice_df,
            out_dir=plot_dir,
            min_clones=args.min_clones,
            spatial_only=not args.include_distribution_plots,
        )

    if args.stage == "spotsweeper":
        if slice_df is None:
            slice_path = OUT_DIR / "survey_slice_table.csv"
            if not slice_path.exists():
                slice_df, _ = run_tables_stage(h5ad_dir)
            else:
                slice_df = pd.read_csv(slice_path)
        run_spotsweeper_stage(
            h5ad_dir,
            slice_df,
            out_dir=plot_dir,
            min_clones=args.min_clones,
        )

    if args.stage == "inclusion":
        if slice_df is None:
            slice_path = OUT_DIR / "survey_slice_table.csv"
            if not slice_path.exists():
                slice_df, _ = run_tables_stage(h5ad_dir)
            else:
                slice_df = pd.read_csv(slice_path)
        run_inclusion_survey_stage(
            h5ad_dir,
            slice_df,
            out_dir=plot_dir,
            min_clones=args.min_clones,
            tumor_threshold=args.inclusion_tumor_threshold,
            min_spots=args.inclusion_min_spots,
            min_cluster_spots=args.inclusion_min_cluster_spots,
        )

    done_dir = plot_dir if args.stage in ("plots", "spotsweeper", "inclusion", "all") else OUT_DIR
    print(f"[done] outputs in {done_dir}")


if __name__ == "__main__":
    main()
