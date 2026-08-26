"""Spatial context plots for CosMx region subset configs."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from data.schemas import DataConfig

COSMX_ANNOTATED_NAME = "cosmx_human_nsclc_annotated.h5ad"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FULL_COSMX_H5AD = _REPO_ROOT / "data" / "h5ad" / COSMX_ANNOTATED_NAME
# CosMx ``obsm['spatial']`` is global pixel coords; 0.12028 µm/px (AtoMx ReadMe).
from data.h5ad_loader import COSMX_UM_PER_UNIT

_COORD_UM_PER_UNIT = COSMX_UM_PER_UNIT
_SCALE_BAR_UM = 500.0

# Stable colors for common NSCLC cell types (CellCharter-style discrete map).
_CELL_TYPE_COLORS: dict[str, str] = {
    "tumor 5": "#E41A1C",
    "tumor 6": "#984EA3",
    "tumor 9": "#FF7F00",
    "tumor 12": "#A65628",
    "tumor 13": "#F781BF",
    "fibroblast": "#4DAF4A",
    "neutrophil": "#377EB8",
    "macrophage": "#A6CEE3",
    "B-cell": "#1B9E77",
    "T CD4 naive": "#66C2A5",
    "T CD8": "#FC8D62",
    "endothelial": "#8DA0CB",
    "NK": "#E78AC3",
    "mdc": "#FFD92F",
    "plasmablast": "#E5C494",
}


def _full_cosmx_h5ad_path(data: DataConfig) -> str:
    if data.h5ad and COSMX_ANNOTATED_NAME in Path(str(data.h5ad)).name:
        return str(data.h5ad)
    if _FULL_COSMX_H5AD.exists():
        return str(_FULL_COSMX_H5AD)
    raise FileNotFoundError(f"full CosMx h5ad not found: {_FULL_COSMX_H5AD}")


def _read_categorical(grp: h5py.Group | h5py.Dataset) -> np.ndarray:
    if isinstance(grp, h5py.Group):
        codes = grp["codes"][:]
        cats = [c.decode() if isinstance(c, bytes) else c for c in grp["categories"][:]]
        return pd.Categorical.from_codes(codes, cats)
    arr = grp[:]
    if arr.dtype.kind == "S":
        arr = np.array([x.decode() for x in arr])
    return arr


@lru_cache(maxsize=1)
def _load_full_cosmx_context(h5ad_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (spatial Nx2, sample labels) for the full stitched CosMx file."""
    with h5py.File(h5ad_path, "r") as f:
        xy = np.asarray(f["obsm"]["spatial"][:], dtype=np.float32)
        sample = np.asarray(_read_categorical(f["obs"]["sample"]), dtype=object)
    return xy, sample


@lru_cache(maxsize=1)
def _load_full_cell_types(h5ad_path: str) -> np.ndarray:
    with h5py.File(h5ad_path, "r") as f:
        return np.asarray(_read_categorical(f["obs"]["cell_type"])).astype(str)


def _color_for_cell_type(cell_type: str, fallback_cmap_idx: int) -> str:
    if cell_type in _CELL_TYPE_COLORS:
        return _CELL_TYPE_COLORS[cell_type]
    cmap = plt.get_cmap("tab20")
    return cmap(fallback_cmap_idx % 20)


def _add_scale_bar(ax, length_um: float = _SCALE_BAR_UM) -> None:
    """Horizontal scale bar in microns."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + 0.04 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.04 * (ylim[1] - ylim[0])
    length_coord = float(length_um) / _COORD_UM_PER_UNIT
    ax.plot([x0, x0 + length_coord], [y0, y0], color="black", lw=2, solid_capstyle="butt")
    label = f"{length_um / 1000:.1f} mm" if length_um >= 1000 else f"{int(length_um)} µm"
    if length_um == 500:
        label = "0.5 mm"
    ax.text(x0 + length_coord / 2, y0 + 0.02 * (ylim[1] - ylim[0]), label,
            ha="center", va="bottom", fontsize=8, color="black")


def _region_bbox(xy: np.ndarray, pad_frac: float = 0.08) -> tuple[float, float, float, float]:
    x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
    y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
    pad_x = max((x1 - x0) * pad_frac, 200.0)
    pad_y = max((y1 - y0) * pad_frac, 200.0)
    return x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y


def _scatter_by_cell_type(
    ax,
    xy: np.ndarray,
    cell_types: np.ndarray,
    *,
    point_size: float = 1.5,
    alpha: float = 0.85,
    highlight_type: str | None = None,
    zorder_base: int = 1,
) -> dict[str, str]:
    """Scatter cells colored by cell type; return color map used."""
    used: dict[str, str] = {}
    types = sorted(set(cell_types.astype(str)))
    for i, ct in enumerate(types):
        if highlight_type and ct != highlight_type:
            continue
        m = cell_types.astype(str) == ct
        color = _color_for_cell_type(ct, i)
        used[ct] = color
        ax.scatter(
            xy[m, 0], xy[m, 1], s=point_size, color=color, linewidths=0,
            rasterized=True, alpha=alpha, zorder=zorder_base, label=ct,
        )
    return used


def _is_cosmx_subset_config(data: DataConfig) -> bool:
    if data.source != "h5ad" or not data.h5ad:
        return False
    if COSMX_ANNOTATED_NAME not in Path(str(data.h5ad)).name:
        return False
    return bool(data.obs_indices or data.obs_filters)


def _slugify_sample(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _match_sample_slug(slug: str, samples: np.ndarray) -> str:
    for s in samples:
        if _slugify_sample(s) == slug:
            return str(s)
    raise ValueError(f"no sample matches slug {slug!r}")


def resolve_legacy_cluster20_component_indices(
    h5ad_path: str,
    *,
    sample: str,
    cluster20: str,
    component_id: int,
    eps_mult: float = 3.0,
) -> np.ndarray:
    with h5py.File(h5ad_path, "r") as f:
        sample_col = np.asarray(_read_categorical(f["obs"]["sample"]))
        mask = sample_col.astype(str) == str(sample)
        idx_all = np.flatnonzero(mask)
        cluster = np.asarray(_read_categorical(f["obs"]["cluster20"]))[mask].astype(str)
        xy = f["obsm"]["spatial"][idx_all]

    cl_mask = cluster == str(cluster20)
    if not cl_mask.any():
        raise ValueError(f"cluster20={cluster20} not in sample {sample}")
    nn = cKDTree(xy[cl_mask]).query(xy[cl_mask], k=2)[0][:, 1]
    eps = float(np.percentile(nn, 90) * eps_mult)
    labels = DBSCAN(eps=eps, min_samples=10).fit_predict(xy[cl_mask])
    comp_mask = labels == int(component_id)
    if not comp_mask.any():
        raise ValueError(
            f"component {component_id} not found for cluster20={cluster20} "
            f"(labels={sorted(set(labels))})"
        )
    base = idx_all[cl_mask]
    return base[comp_mask].astype(np.int64)


_LEGACY_RUN_RE = re.compile(
    r"^cosmx_(?P<slug>.+)_cl(?P<cl>\d+)_c(?P<comp>\d+)_(?P<rest>.+?)(?:_poisson|_gaussian)?$"
)


def resolve_region_indices_from_run_name(h5ad_path: str, run_name: str) -> np.ndarray:
    m = _LEGACY_RUN_RE.match(run_name)
    if not m:
        raise ValueError(f"run_name does not match legacy cluster20 pattern: {run_name}")
    with h5py.File(h5ad_path, "r") as f:
        samples = np.unique(np.asarray(_read_categorical(f["obs"]["sample"])))
    sample = _match_sample_slug(m.group("slug"), samples)
    return resolve_legacy_cluster20_component_indices(
        h5ad_path,
        sample=sample,
        cluster20=m.group("cl"),
        component_id=int(m.group("comp")),
    )


def resolve_region_indices(data: DataConfig, *, run_name: str | None = None) -> np.ndarray:
    data.validate()
    if data.obs_indices:
        path = Path(data.obs_indices)
        if path.exists():
            return np.asarray(np.load(path), dtype=np.int64)
        if run_name and _LEGACY_RUN_RE.match(run_name):
            return resolve_region_indices_from_run_name(_full_cosmx_h5ad_path(data), run_name)
        raise FileNotFoundError(f"obs_indices not found: {data.obs_indices}")
    if run_name and _LEGACY_RUN_RE.match(run_name):
        return resolve_region_indices_from_run_name(_full_cosmx_h5ad_path(data), run_name)
    if not data.obs_filters:
        raise ValueError("CosMx region context requires obs_indices or obs_filters")

    with h5py.File(data.h5ad, "r") as f:
        sample = np.asarray(_read_categorical(f["obs"]["sample"]))
        n = len(sample)
        mask = np.ones(n, dtype=bool)
        for col, val in data.obs_filters.items():
            col_vals = np.asarray(_read_categorical(f["obs"][col]))
            mask &= col_vals.astype(str) == str(val)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"obs_filters matched no cells: {data.obs_filters}")
    return idx.astype(np.int64)


def save_cosmx_region_context_plot(
    data: DataConfig,
    out_path: str | Path,
    *,
    run_name: Optional[str] = None,
    title: Optional[str] = None,
    sample_hint: Optional[str] = None,
) -> Path:
    """Paper-style region context: full section (cell types) + ROI box + zoom inset."""
    if not _is_cosmx_subset_config(data) and not (
        run_name and _LEGACY_RUN_RE.match(run_name)
    ):
        raise ValueError("save_cosmx_region_context_plot requires a CosMx subset DataConfig")

    region_idx = resolve_region_indices(data, run_name=run_name)
    h5ad_context = _full_cosmx_h5ad_path(data)
    xy_all, sample_all = _load_full_cosmx_context(h5ad_context)
    if region_idx.max() >= xy_all.shape[0] or region_idx.min() < 0:
        raise ValueError(
            f"region indices out of range: max={region_idx.max()}, n_obs={xy_all.shape[0]}"
        )

    region_xy = xy_all[region_idx]
    region_samples = sample_all[region_idx]
    if sample_hint is None:
        uniq, counts = np.unique(region_samples, return_counts=True)
        sample_hint = str(uniq[int(np.argmax(counts))])

    sample_mask = sample_all.astype(str) == str(sample_hint)
    sample_xy = xy_all[sample_mask]
    ct_all = _load_full_cell_types(h5ad_context)
    sample_ct = ct_all[sample_mask]

    if data.obs_filters and "cell_type" in data.obs_filters:
        cell_type = str(data.obs_filters["cell_type"])
    else:
        ct_region = ct_all[region_idx]
        uniq, counts = np.unique(ct_region.astype(str), return_counts=True)
        cell_type = str(uniq[int(np.argmax(counts))])

    bx0, bx1, by0, by1 = _region_bbox(region_xy)
    n_sample = int(sample_mask.sum())
    n_region = int(region_idx.size)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 5.2))
    ax_full = fig.add_axes([0.06, 0.12, 0.52, 0.78])
    ax_zoom = fig.add_axes([0.64, 0.12, 0.32, 0.78])

    # Full section: all cell types (background), then test region outlined.
    _scatter_by_cell_type(ax_full, sample_xy, sample_ct, point_size=1.2, alpha=0.75)
    ax_full.scatter(
        region_xy[:, 0], region_xy[:, 1], s=4.0, facecolors="none",
        edgecolors="black", linewidths=0.35, rasterized=True, zorder=10, label="test region",
    )
    rect = Rectangle(
        (bx0, by0), bx1 - bx0, by1 - by0,
        fill=False, edgecolor="black", linewidth=2.0, zorder=11,
    )
    ax_full.add_patch(rect)

    ax_full.set_aspect("equal")
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    ax_full.set_title(f"{sample_hint}  (n={n_sample:,} cells)", fontsize=10)

    # Zoom inset: cells inside bbox, test region filled.
    zoom_mask = (
        (sample_xy[:, 0] >= bx0) & (sample_xy[:, 0] <= bx1)
        & (sample_xy[:, 1] >= by0) & (sample_xy[:, 1] <= by1)
    )
    zoom_xy = sample_xy[zoom_mask]
    zoom_ct = sample_ct[zoom_mask]
    other = zoom_ct.astype(str) != cell_type
    if other.any():
        ax_zoom.scatter(
            zoom_xy[other, 0], zoom_xy[other, 1], s=3.0, color="0.82",
            linewidths=0, rasterized=True, alpha=0.9, zorder=1,
        )
    same = zoom_ct.astype(str) == cell_type
    region_color = _color_for_cell_type(cell_type, 0)
    if same.any():
        ax_zoom.scatter(
            zoom_xy[same, 0], zoom_xy[same, 1], s=5.0, color=region_color,
            linewidths=0, rasterized=True, alpha=0.95, zorder=2,
        )
    ax_zoom.scatter(
        region_xy[:, 0], region_xy[:, 1], s=8.0, facecolors="none",
        edgecolors="black", linewidths=0.5, rasterized=True, zorder=3,
    )
    ax_zoom.set_xlim(bx0, bx1)
    ax_zoom.set_ylim(by0, by1)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_title(f"Test region: {cell_type}\n(n={n_region:,})", fontsize=10)

    # Connector lines (paper-style zoom bracket).
    for xy_a, xy_b in [
        ((bx0, by0), (bx0, by0)),
        ((bx1, by0), (bx1, by0)),
        ((bx0, by1), (bx0, by1)),
        ((bx1, by1), (bx1, by1)),
    ]:
        con = ConnectionPatch(
            xyA=xy_a, xyB=xy_b, coordsA="data", coordsB="data",
            axesA=ax_full, axesB=ax_zoom, color="black", linewidth=0.8, alpha=0.7,
        )
        fig.add_artist(con)

    _add_scale_bar(ax_full)
    _add_scale_bar(ax_zoom, length_um=min(_SCALE_BAR_UM, (bx1 - bx0) * 0.25))

    # Compact legend for cell types present in section (top types only).
    type_counts = pd.Series(sample_ct).value_counts()
    legend_types = list(type_counts.head(8).index)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_color_for_cell_type(str(ct), i),
               markersize=6, label=str(ct))
        for i, ct in enumerate(legend_types)
    ]
    handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=6, markeredgewidth=1.2, label="test region")
    )
    ax_full.legend(handles=handles, loc="lower left", fontsize=6.5, framealpha=0.9,
                   markerscale=0.9, ncol=2, borderpad=0.4, handletextpad=0.3)

    main_title = title or f"CosMx region — {sample_hint} — {cell_type} (n={n_region:,})"
    fig.suptitle(main_title, fontsize=11, y=0.98)

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
