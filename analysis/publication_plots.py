"""Publication-quality figure primitives.

Styled to match the GASTON paper aesthetic: clean white backgrounds, minimal
spines, light-weight sans-serif type, small edgeless scatter points, and
contour-line overlays on spatial fields.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

from analysis.plots import (
    _masked_triangulation,
    _point_size,
    _spatial_axis_limits,
)


MM_PER_INCH = 25.4
SINGLE_COL_MM = 89.0
DOUBLE_COL_MM = 183.0

_PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.weight": "light",
    "font.size": 5.5,
    "axes.labelsize": 5.5,
    "axes.titlesize": 6,
    "axes.titleweight": "regular",
    "axes.linewidth": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.labelsize": 4.5,
    "ytick.labelsize": 4.5,
    "xtick.major.width": 0.3,
    "ytick.major.width": 0.3,
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 4.5,
    "legend.frameon": False,
    "legend.borderpad": 0.2,
    "legend.handletextpad": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 0.6,
    "patch.linewidth": 0.25,
}


@contextmanager
def publication_style():
    """Context manager that temporarily applies publication rcParams."""
    old = {k: mpl.rcParams.get(k) for k in _PUB_RC}
    mpl.rcParams.update(_PUB_RC)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is not None:
                mpl.rcParams[k] = v


def mm_to_inches(mm: float) -> float:
    return mm / MM_PER_INCH


def add_panel_label(ax, label: str, x: float = -0.06, y: float = 1.06) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="right",
        fontfamily="sans-serif",
    )


def shared_spatial_limits(
    *coord_arrays: np.ndarray,
    padding_frac: float = 0.03,
) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack([np.asarray(c, dtype=np.float32) for c in coord_arrays])
    return _spatial_axis_limits(stacked, padding_frac=padding_frac)


def _strip_spatial_axes(ax, limits=None) -> None:
    """Remove all spines, ticks, and labels from a spatial scatter axes."""
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])


def plot_spatial_field(
    ax,
    S: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str = "Reds",
    vmin: float | None = None,
    vmax: float | None = None,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    colorbar: bool = True,
    colorbar_label: str = "",
    title: str = "",
    contours: bool = True,
    point_alpha: float = 0.70,
    point_scale: float = 0.35,
) -> None:
    """Scatter + optional tricontour for a continuous spatial field."""
    S = np.asarray(S, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32).ravel()

    resolved_vmin = float(v.min()) if vmin is None else float(vmin)
    resolved_vmax = float(v.max()) if vmax is None else float(vmax)

    norm = mcolors.Normalize(vmin=resolved_vmin, vmax=resolved_vmax)
    sc = ax.scatter(
        S[:, 0], S[:, 1],
        c=v, cmap=cmap, norm=norm,
        s=_point_size(S) * point_scale,
        edgecolors="none",
        alpha=point_alpha, rasterized=True,
    )

    if contours and S.shape[0] >= 3:
        try:
            tri = _masked_triangulation(S)
            levels = np.linspace(resolved_vmin, resolved_vmax, 5)
            if levels[-1] > levels[0] + 1e-12:
                cs = ax.tricontour(
                    tri, v, levels=levels,
                    colors="black", linewidths=0.7, alpha=0.6,
                )
        except (RuntimeError, ValueError):
            pass

    if title:
        ax.set_title(title, pad=3, fontstyle="italic")
    if colorbar:
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                          label=colorbar_label)
        cb.set_label(colorbar_label, fontsize=4.5)
        cb.ax.tick_params(labelsize=4, length=1.2, width=0.25)
        cb.outline.set_linewidth(0.25)
    _strip_spatial_axes(ax, limits)


def plot_celltype_map(
    ax,
    S: np.ndarray,
    labels: np.ndarray,
    type_names: Sequence[str],
    *,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    title: str = "",
    max_legend_types: int = 20,
    point_scale: float = 0.30,
) -> None:
    """Cell-type scatter with compact legend (GASTON style)."""
    S = np.asarray(S, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n_types = len(type_names)
    cmap = plt.cm.get_cmap("tab20" if n_types > 10 else "tab10")

    for c in range(n_types):
        mask = labels == c
        if not np.any(mask):
            continue
        ax.scatter(
            S[mask, 0], S[mask, 1],
            c=[cmap(c / max(n_types - 1, 1))],
            s=_point_size(S) * point_scale,
            edgecolors="none",
            label=type_names[c] if c < max_legend_types else None,
            alpha=0.65, rasterized=True,
        )

    if title:
        ax.set_title(title, pad=3, fontstyle="italic")
    _strip_spatial_axes(ax, limits)


_METRIC_DISPLAY = {
    "nll_gaussian_mse": "Loss",
    "mse": "MSE",
    "pearson_corr_mean": "Pearson r",
    "spearman_corr_mean": "Spearman \u03c1",
}


def friendly_metric(raw: str) -> str:
    """Map internal metric names to human-readable labels."""
    return _METRIC_DISPLAY.get(raw, raw)


def plot_null_ecdf(
    ax,
    stat_perm: np.ndarray,
    stat_observed: float,
    p_value: float,
    *,
    q_value: float | None = None,
    stat_covariate: float | None = None,
    p_value_covariate: float | None = None,
    metric_label: str = "",
    title: str = "",
    n_perms: int | None = None,
) -> None:
    """ECDF of null distribution with observed statistic line(s)."""
    perm = np.sort(np.asarray(stat_perm, dtype=np.float64))
    n = perm.size
    ecdf_y = np.arange(1, n + 1) / n

    m_label = f" (m={n_perms or n})" if (n_perms or n) else ""

    ax.fill_between(perm, 0, ecdf_y, color="#d4d4d4", alpha=0.45,
                    step="post")
    ax.step(perm, ecdf_y, where="post", color="#888888", linewidth=0.5,
            label=f"Null distribution{m_label}")
    ax.axvline(stat_observed, color="#c0392b", linestyle="-", linewidth=0.8,
               label="Observed")

    if stat_covariate is not None:
        ax.axvline(stat_covariate, color="#16a085", linestyle="--",
                   linewidth=0.8, label="Covariate")

    xlabel = friendly_metric(metric_label) if metric_label else "Loss"
    ax.set_xlabel(xlabel, fontsize=4.5)
    ax.set_ylabel("Cumulative prob.", fontsize=4.5)
    if title:
        ax.set_title(title, pad=3, fontstyle="italic", fontsize=5.5)

    if q_value is not None:
        p_text = f"q = {q_value:.3g}"
    else:
        p_text = f"p = {p_value:.3g}"
    if stat_covariate is not None and p_value_covariate is not None:
        p_text += f"\np (cov) = {p_value_covariate:.3g}"
    ax.text(0.04, 0.96, p_text, transform=ax.transAxes, fontsize=4,
            va="top", ha="left", color="#444444")

    ax.legend(fontsize=3.5, loc="lower right", handlelength=1.2,
              borderpad=0.2)
    ax.tick_params(labelsize=4)


def plot_null_histogram(
    ax,
    stat_perm: np.ndarray,
    stat_observed: float,
    p_value: float,
    *,
    q_value: float | None = None,
    stat_covariate: float | None = None,
    p_value_covariate: float | None = None,
    metric_label: str = "",
    title: str = "",
    n_perms: int | None = None,
) -> None:
    """Histogram of null distribution with observed-stat line."""
    perm = np.asarray(stat_perm, dtype=np.float64)
    n = perm.size
    m_label = f" (m={n_perms or n})" if (n_perms or n) else ""

    ax.hist(perm, bins=30, color="#d4d4d4", edgecolor="#aaaaaa",
            linewidth=0.25, label=f"Null distribution{m_label}", density=True)
    ax.axvline(stat_observed, color="#c0392b", linewidth=0.8,
               label="Observed")

    if stat_covariate is not None:
        ax.axvline(stat_covariate, color="#16a085", linestyle="--",
                   linewidth=0.8, label="Covariate")

    xlabel = friendly_metric(metric_label) if metric_label else "Loss"
    ax.set_xlabel(xlabel, fontsize=4.5)
    ax.set_ylabel("Density", fontsize=4.5)
    if title:
        ax.set_title(title, pad=3, fontstyle="italic", fontsize=5.5)

    if q_value is not None:
        p_text = f"q = {q_value:.3g}"
    else:
        p_text = f"p = {p_value:.3g}"
    if stat_covariate is not None and p_value_covariate is not None:
        p_text += f"\np (cov) = {p_value_covariate:.3g}"
    ax.text(0.96, 0.96, p_text, transform=ax.transAxes, fontsize=4,
            va="top", ha="right", color="#444444")

    ax.legend(fontsize=3.5, loc="upper right", handlelength=1.2,
              borderpad=0.2)
    ax.tick_params(labelsize=4)


def savefig(fig: Figure, path: str, *, close: bool = True) -> None:
    """Save with tight bbox and white background."""
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04,
                facecolor="white", edgecolor="none")
    if close:
        plt.close(fig)
