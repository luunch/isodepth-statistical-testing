"""Generate publication-quality composite figures from existing run results.

Usage
-----
    python -m experiments.publication_figures \\
        --result-json results/<run>/<run>_result.json \\
        [--config configs/<dataset>.json] \\
        [--out figures/<run>/main_figure.pdf] \\
        [--format pdf]

The script reads the saved *_result.json and (optionally) the original config
to reload the dataset.  It never re-trains models — it composes spatial maps,
null distributions, and covariate comparisons into single multi-panel PDF/SVG
figures suitable for journal submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analysis.publication_plots import (
    add_panel_label,
    friendly_metric,
    mm_to_inches,
    plot_celltype_map,
    plot_null_ecdf,
    plot_null_histogram,
    plot_spatial_field,
    publication_style,
    savefig,
    shared_spatial_limits,
    DOUBLE_COL_MM,
    _strip_spatial_axes,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_result_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_dataset_from_result(
    result: dict[str, Any],
    config_path: str | None = None,
):
    """Reload the ``DatasetBundle`` used in the original run."""
    from data import load_dataset
    from data.schemas import DataConfig

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        data_raw = file_cfg.get("data", {})
    else:
        data_raw = dict(result.get("config", {}).get("data", {}))

    data_raw.pop("sampling_bias", None)
    for k in list(data_raw):
        if k not in DataConfig.__dataclass_fields__:
            data_raw.pop(k, None)

    data_cfg = DataConfig(**data_raw)
    return load_dataset(data_cfg)


def _null_plot(ax, stat_perm, stat_true, p_value, metric, *,
               n_perms=None, q_value=None, **kw):
    """Pick ECDF or histogram based on permutation count."""
    perm = np.asarray(stat_perm, dtype=np.float64)
    fn = plot_null_ecdf if perm.size > 50 else plot_null_histogram
    fn(ax, perm, stat_true, p_value, metric_label=metric,
       n_perms=n_perms, q_value=q_value, **kw)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values (FDR) for an array of p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks
    # enforce monotonicity: q[i] = min(q[i], q[i+1]) walking from largest rank down
    q_sorted = q[order]
    for i in range(m - 2, -1, -1):
        q_sorted[i + 1] = min(q_sorted[i + 1], 1.0)
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_sorted[0] = min(q_sorted[0], 1.0)
    q[order] = q_sorted
    return q


# ---------------------------------------------------------------------------
# Standard (non-cell-type-separate) figure
# ---------------------------------------------------------------------------

def _build_standard_figure(
    result: dict[str, Any],
    dataset,
    out_path: Path,
    fmt: str,
) -> Path:
    arts = result["artifacts"]
    meta = dataset.meta

    has_celltype = meta.get("cell_type_labels") is not None
    true_isodepth = arts.get("true_isodepth")
    has_isodepth = true_isodepth is not None
    has_covariate = "stat_covariate" in arts
    cov_isodepth = arts.get("true_isodepth_covariate")

    n_top = 1 + int(has_covariate) + int(has_isodepth)
    n_top = max(n_top, 2)

    fig_w = mm_to_inches(DOUBLE_COL_MM)
    panel_h = fig_w / n_top * 0.85
    null_h = panel_h * 0.50 if has_isodepth else 0
    fig_h = panel_h + null_h + mm_to_inches(12)

    fig = plt.figure(figsize=(fig_w, fig_h))
    if has_isodepth:
        gs = fig.add_gridspec(2, n_top, height_ratios=[1.0, 0.45],
                              hspace=0.45, wspace=0.30,
                              left=0.02, right=0.98, top=0.92, bottom=0.08)
    else:
        gs = fig.add_gridspec(1, n_top, wspace=0.30,
                              left=0.02, right=0.98, top=0.92, bottom=0.08)

    limits = shared_spatial_limits(dataset.S)
    col = 0
    panel_i = 0
    labels = "abcdefghij"

    # (a) cell-type map or mean expression
    ax_a = fig.add_subplot(gs[0, col])
    if has_celltype:
        ct_labels = np.asarray(meta["cell_type_labels"], dtype=np.int64)
        ct_names = list(meta["cell_type_names"])
        plot_celltype_map(ax_a, dataset.S, ct_labels, ct_names, limits=limits,
                          title="Cell types")
        ax_a.legend(
            fontsize=4, markerscale=1.8,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, handletextpad=0.3,
            borderpad=0.2, labelspacing=0.25,
        )
    else:
        signal = np.mean(np.abs(np.asarray(dataset.A, dtype=np.float32)), axis=1)
        plot_spatial_field(ax_a, dataset.S, signal, cmap="Reds", limits=limits,
                          title="Mean expression", colorbar_label="Expression",
                          contours=False)
    add_panel_label(ax_a, labels[panel_i])
    col += 1; panel_i += 1

    # (b) covariate (optional)
    if has_covariate and cov_isodepth is not None:
        ax_b = fig.add_subplot(gs[0, col])
        cov_arr = np.asarray(cov_isodepth, dtype=np.float32).ravel()
        cov_raw = result.get("config", {}).get("test", {}).get("covariate", {})
        cov_name = cov_raw.get("type", "covariate") if isinstance(cov_raw, dict) else str(cov_raw)
        cov_title = f"{cov_name.capitalize()} covariate"
        plot_spatial_field(ax_b, dataset.S, cov_arr, cmap="Reds",
                          limits=limits, title=cov_title,
                          colorbar_label="Isodepth", contours=True)
        add_panel_label(ax_b, labels[panel_i])
        col += 1; panel_i += 1

    # (c) learned isodepth + null ECDF
    if has_isodepth:
        ax_c = fig.add_subplot(gs[0, col])
        iso = np.asarray(true_isodepth, dtype=np.float32).ravel()
        plot_spatial_field(ax_c, dataset.S, iso, cmap="Reds", limits=limits,
                          title="Learned isodepth", colorbar_label="Isodepth",
                          contours=True)
        add_panel_label(ax_c, labels[panel_i]); panel_i += 1

        null_span = min(2, n_top)
        null_start = (n_top - null_span) // 2
        ax_null = fig.add_subplot(gs[1, null_start:null_start + null_span])
        n_perms = len(result["stat_perm"])
        _null_plot(
            ax_null,
            result["stat_perm"], float(result["stat_true"]),
            float(result["p_value"]), result.get("metric", ""),
            n_perms=n_perms,
            stat_covariate=(float(arts["stat_covariate"])
                            if "stat_covariate" in arts else None),
            p_value_covariate=(float(arts["p_value_covariate"])
                               if "p_value_covariate" in arts else None),
            title="Permutation null",
        )
        add_panel_label(ax_null, labels[panel_i])

    savefig(fig, str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Cell-type separate: overview figure (mean expression + cell-type map)
# ---------------------------------------------------------------------------

def _build_celltype_overview_figure(
    dataset,
    ct_names: list[str],
    out_path: Path,
) -> Path:
    """(a) Mean expression, (b) cell-type map with legend."""
    meta = dataset.meta
    ct_labels = np.asarray(meta.get("cell_type_labels", []), dtype=np.int64)
    ct_name_list = list(meta.get("cell_type_names", ct_names))
    limits = shared_spatial_limits(dataset.S)
    _title_fs = 5.5

    fig_w = mm_to_inches(DOUBLE_COL_MM)
    fig_h = fig_w * 0.38

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 0.22],
        left=0.04, right=0.75,
        top=0.90, bottom=0.05,
        wspace=0.06,
    )

    ax_expr = fig.add_subplot(gs[0, 0])
    signal = np.mean(np.abs(np.asarray(dataset.A, dtype=np.float32)), axis=1)
    plot_spatial_field(ax_expr, dataset.S, signal, cmap="Reds", limits=limits,
                       title="", colorbar_label="Expression",
                       contours=False, point_scale=0.20)
    ax_expr.set_title("Mean expression", fontsize=_title_fs, fontstyle="italic", pad=3)
    add_panel_label(ax_expr, "a")

    ax_ct = fig.add_subplot(gs[0, 1])
    if ct_labels.size > 0:
        plot_celltype_map(ax_ct, dataset.S, ct_labels, ct_name_list,
                          limits=limits, title="", point_scale=0.20)
        ax_ct.set_title("Cell types", fontsize=_title_fs, fontstyle="italic", pad=3)
        ax_leg = fig.add_subplot(gs[0, 2])
        ax_leg.set_axis_off()
        handles, leg_labels = ax_ct.get_legend_handles_labels()
        ax_leg.legend(
            handles, leg_labels,
            fontsize=4, markerscale=1.8,
            loc="center left", bbox_to_anchor=(0.0, 0.5),
            frameon=False, handletextpad=0.3,
            borderpad=0.2, labelspacing=0.25,
        )
    add_panel_label(ax_ct, "b")

    savefig(fig, str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Cell-type separate: per-type isodepth + null distribution grid
# ---------------------------------------------------------------------------

def _build_celltype_pertype_figure(
    result: dict[str, Any],
    dataset,
    out_path: Path,
) -> Path:
    """Grid of (spatial isodepth | null distribution) pairs, one per cell type."""
    arts = result["artifacts"]
    meta = dataset.meta
    ct_names: list[str] = arts["cell_type_names"]
    per_type: dict[str, dict] = arts["per_type_summaries"]

    ct_labels = np.asarray(meta.get("cell_type_labels", []), dtype=np.int64)
    ct_name_list = list(meta.get("cell_type_names", ct_names))
    limits = shared_spatial_limits(dataset.S)
    metric = result.get("metric", "")
    n_types = len(ct_names)
    n_perms = len(per_type[ct_names[0]].get("stat_perm", []))

    raw_pvals = np.array([float(per_type[ct]["p_value"]) for ct in ct_names])
    q_values = _benjamini_hochberg(raw_pvals) if n_types > 1 else raw_pvals

    _title_fs = 5.5
    pairs_per_row = 2 if n_types > 3 else 1
    n_rows = int(np.ceil(n_types / pairs_per_row))
    n_ax_cols = pairs_per_row * 2

    fig_w = mm_to_inches(DOUBLE_COL_MM)
    row_h = fig_w / n_ax_cols * 0.82
    fig_h = n_rows * row_h + mm_to_inches(6)

    fig, axes_flat = plt.subplots(
        n_rows, n_ax_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.55, "hspace": 0.60},
        squeeze=False,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05)

    from scipy.spatial import cKDTree

    labels = "abcdefghijklmnopqrstuvwxyz"

    for type_idx, type_name in enumerate(ct_names):
        td = per_type[type_name]
        row = type_idx // pairs_per_row
        pair_col = type_idx % pairs_per_row
        sp_col = pair_col * 2
        ec_col = sp_col + 1

        ax_sp = axes_flat[row, sp_col]
        ax_ec = axes_flat[row, ec_col]

        n_cells = int(td["n_cells"])
        p_val = float(td["p_value"])

        type_mask = None
        if type_name in ct_name_list:
            type_mask = ct_labels == ct_name_list.index(type_name)

        # Shared cell-type label centered above the pair
        mid_x = (ax_sp.get_position().x0 + ax_ec.get_position().x1) / 2
        top_y = ax_sp.get_position().y1
        fig.text(mid_x, top_y + 0.012, f"{type_name} (n={n_cells})",
                 ha="center", va="bottom", fontsize=_title_fs,
                 fontstyle="italic")

        if type_mask is not None and np.any(type_mask):
            S_type = np.asarray(dataset.S, dtype=np.float32)[type_mask]
            if S_type.shape[0] > 2:
                tree = cKDTree(S_type)
                dists, _ = tree.query(S_type, k=min(6, S_type.shape[0]))
                density = 1.0 / (np.mean(dists[:, 1:], axis=1) + 1e-8)
                plot_spatial_field(
                    ax_sp, S_type, density, cmap="Reds", limits=limits,
                    title="",
                    colorbar=True, colorbar_label="Isodepth",
                    contours=True, point_scale=0.30,
                )
            else:
                ax_sp.scatter(S_type[:, 0], S_type[:, 1], s=1.5,
                              c="#777777", edgecolors="none", rasterized=True)
                _strip_spatial_axes(ax_sp, limits)
        else:
            ax_sp.text(0.5, 0.5, type_name, ha="center", va="center",
                       transform=ax_sp.transAxes, fontsize=5)
            _strip_spatial_axes(ax_sp, limits)

        add_panel_label(ax_sp, labels[type_idx * 2])

        is_bottom = (row == n_rows - 1) or (type_idx + pairs_per_row >= n_types)
        is_left_ecdf = (pair_col == 0)

        q_val = float(q_values[type_idx]) if n_types > 1 else None
        _null_plot(
            ax_ec,
            td["stat_perm"], float(td["stat_true"]), p_val, metric,
            n_perms=n_perms,
            q_value=q_val,
            title="",
        )
        add_panel_label(ax_ec, labels[type_idx * 2 + 1])

        if not is_bottom:
            ax_ec.set_xlabel("")
        if not is_left_ecdf:
            ax_ec.set_ylabel("")

    # Hide unused cells
    for idx in range(n_types, n_rows * pairs_per_row):
        row = idx // pairs_per_row
        pair_col = idx % pairs_per_row
        axes_flat[row, pair_col * 2].set_visible(False)
        axes_flat[row, pair_col * 2 + 1].set_visible(False)

    savefig(fig, str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Supplement: per-type ECDF grid
# ---------------------------------------------------------------------------

def _build_supplement_ecdf_grid(
    result: dict[str, Any],
    out_path: Path,
) -> Path:
    arts = result["artifacts"]
    ct_names: list[str] = arts["cell_type_names"]
    per_type: dict[str, dict] = arts["per_type_summaries"]
    metric = result.get("metric", "")

    n = len(ct_names)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    cell_w = mm_to_inches(DOUBLE_COL_MM) / ncols
    cell_h = cell_w * 0.62
    fig_w = cell_w * ncols
    fig_h = cell_h * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             constrained_layout=True, squeeze=False)

    n_perms = len(per_type[ct_names[0]].get("stat_perm", []))
    for idx, type_name in enumerate(ct_names):
        r, c = divmod(idx, ncols)
        td = per_type[type_name]
        plot_null_ecdf(
            axes[r][c],
            np.asarray(td["stat_perm"], dtype=np.float64),
            float(td["stat_true"]), float(td["p_value"]),
            metric_label=metric,
            n_perms=n_perms,
            title=f"{type_name} (n={td['n_cells']})",
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    savefig(fig, str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def make_publication_figure(
    result_json_path: str,
    config_path: str | None = None,
    out_path: str | None = None,
    fmt: str = "pdf",
) -> list[Path]:
    result = _load_result_json(result_json_path)
    result_dir = Path(result_json_path).resolve().parent
    run_name = Path(result_json_path).stem.replace("_result", "")

    if out_path is None:
        figures_dir = result_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        out_main = figures_dir / f"{run_name}_main.{fmt}"
    else:
        out_main = Path(out_path)
        out_main.parent.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset_from_result(result, config_path)
    arts = result.get("artifacts", {})
    ct_mode = (arts.get("cell_type_mode")
               or arts.get("dataset_meta", {}).get("cell_type_mode"))

    outputs: list[Path] = []
    with publication_style():
        if ct_mode == "separate":
            ct_names = arts.get("cell_type_names", [])
            overview_path = out_main.with_name(f"{run_name}_overview.{fmt}")
            pertype_path = out_main.with_name(f"{run_name}_pertype.{fmt}")
            outputs.append(
                _build_celltype_overview_figure(dataset, ct_names, overview_path))
            outputs.append(
                _build_celltype_pertype_figure(result, dataset, pertype_path))
        else:
            outputs.append(
                _build_standard_figure(result, dataset, out_main, fmt))

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from run results.",
    )
    parser.add_argument(
        "--result-json", required=True,
        help="Path to the *_result.json from a completed run.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Original config JSON (to reload dataset).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path for the main figure.",
    )
    parser.add_argument(
        "--format", default="pdf", choices=["pdf", "svg", "png"],
        help="Output format (default: pdf).",
    )
    args = parser.parse_args()

    outputs = make_publication_figure(
        result_json_path=args.result_json,
        config_path=args.config,
        out_path=args.out,
        fmt=args.format,
    )
    for p in outputs:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
