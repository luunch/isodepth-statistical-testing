from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import f as _f_dist
from scipy.stats import gaussian_kde, linregress, spearmanr

from data.schemas import DatasetBundle, TestResult
from methods.metrics import metric_prefers_lower

# Exclude points treated as zero expression when ``hide_zero_expression`` is enabled.
_EXPRESSION_ZERO_EPS = 1e-15

# Low ≈ white / very light, high = dark — cycle across per-gene figures for distinction.
_EXPRESSION_WHITE_TO_DARK_CYCLIC = (
    "Reds",
    "Blues",
    "Purples",
    "Oranges",
    "Greens",
    "BuPu",
    "PuRd",
    "YlOrRd",
)

# Single aggregate over genes (e.g. dataset triptych mean |expression|): white-low → dark-high.
DEFAULT_EXPRESSION_AGGREGATE_COLORMAP = "Reds"


def expression_colormap_for_index(index: int) -> str:
    """Sequential colormap name for per-gene expression plots (near-white low → dark high)."""
    return _EXPRESSION_WHITE_TO_DARK_CYCLIC[
        int(index) % len(_EXPRESSION_WHITE_TO_DARK_CYCLIC)
    ]


def _normalize_depth(
    values: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    resolved_vmin = float(values.min()) if vmin is None else float(vmin)
    resolved_vmax = float(values.max()) if vmax is None else float(vmax)
    return (values - resolved_vmin) / (resolved_vmax - resolved_vmin + 1e-8)


def _point_size(S: np.ndarray) -> float:
    n = max(int(S.shape[0]), 1)
    return float(max(2, min(16, 500 / np.sqrt(n))))


def _clone_overview_point_size(S: np.ndarray) -> float:
    """Larger markers for CalicoST clone / tumor-proportion overview scatter plots."""
    return float(max(28.0, _point_size(S) * 3.0))


_CLONE_OVERVIEW_LOW_COLOR = np.asarray((0.851, 0.851, 0.851, 1.0), dtype=np.float32)  # paper Normal grey
_CLONE_OVERVIEW_LEGEND_MARKERSIZE = 5.0

# CalicoST Nature Fig. 4 clone palette (HT112C1 panel a; HT268B1 panel c uses clones 1–2).
_CALICOST_PAPER_CLONE_HEX: dict[int, str] = {
    1: "#66c2a5",  # teal
    2: "#fc8d62",  # coral
    3: "#8da0cb",  # slate blue
}


def _hex_rgba(hex_color: str, *, alpha: float = 1.0) -> np.ndarray:
    rgb = mcolors.to_rgb(hex_color)
    return np.asarray([rgb[0], rgb[1], rgb[2], alpha], dtype=np.float32)


def _parse_calicost_clone_id(label_name: str) -> int | None:
    text = str(label_name).strip()
    if not text:
        return None
    try:
        value = float(text)
        if value == int(value):
            return int(value)
    except ValueError:
        return None
    return None


def _calicost_clone_base_color(
    label_name: str,
    *,
    fallback_index: int,
    n_types: int,
) -> np.ndarray:
    clone_id = _parse_calicost_clone_id(label_name)
    if clone_id is not None and clone_id in _CALICOST_PAPER_CLONE_HEX:
        return _hex_rgba(_CALICOST_PAPER_CLONE_HEX[clone_id])
    cmap = plt.cm.get_cmap("tab20" if n_types > 10 else "tab10")
    return np.asarray(cmap(fallback_index / max(n_types - 1, 1)), dtype=np.float32)


def _clone_overview_max_color(base_color: np.ndarray) -> np.ndarray:
    """Vivid, fully opaque clone color at tumor proportion = 1."""
    out = np.asarray(base_color, dtype=np.float32).copy()
    out[:3] = np.clip(out[:3], 0.0, 1.0)
    out[3] = 1.0
    return out


def _clone_tumor_proportion_blend_weights(weights: np.ndarray) -> np.ndarray:
    """Map tumor proportion to blend weight; high values reach full color before tp=1."""
    w = np.clip(np.asarray(weights, dtype=np.float32).reshape(-1), 0.0, 1.0)
    return np.power(w, 0.7).reshape(-1, 1)


def _clone_tumor_proportion_colors(
    base_color: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    blend = _clone_tumor_proportion_blend_weights(weights)
    max_color = _clone_overview_max_color(base_color)
    colors = _CLONE_OVERVIEW_LOW_COLOR * (1.0 - blend) + max_color * blend
    colors[:, 3] = alpha
    return colors


def _clone_legend_handles(
    names: list[str],
    base_colors: list[np.ndarray],
    *,
    markersize: float = _CLONE_OVERVIEW_LEGEND_MARKERSIZE,
) -> list[Line2D]:
    """Legend swatches at full clone color (tumor proportion = 1), not point-averaged."""
    handles: list[Line2D] = []
    for name, base_color in zip(names, base_colors):
        face = _clone_overview_max_color(np.asarray(base_color, dtype=np.float32))
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=face,
                markeredgecolor=face,
                markeredgewidth=0.0,
                markersize=markersize,
                label=name,
            )
        )
    return handles


def _spatial_axis_limits(
    S: np.ndarray,
    *,
    padding_frac: float = 0.02,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits ((xmin, xmax), (ymin, ymax)) with light padding."""
    coords = np.asarray(S, dtype=np.float64)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-8)
    pad = spans * float(padding_frac)
    return (
        (float(mins[0] - pad[0]), float(maxs[0] + pad[0])),
        (float(mins[1] - pad[1]), float(maxs[1] + pad[1])),
    )


def _square_spatial_axis_limits(
    S: np.ndarray,
    *,
    padding_frac: float = 0.02,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Square axis limits with the same numeric span for x and y."""
    coords = np.asarray(S, dtype=np.float64)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(max(np.max(maxs - mins), 1e-8))
    half_span = span * (0.5 + float(padding_frac))
    return (
        (float(center[0] - half_span), float(center[0] + half_span)),
        (float(center[1] - half_span), float(center[1] + half_span)),
    )


def _masked_triangulation(S: np.ndarray) -> mtri.Triangulation:
    triangulation = mtri.Triangulation(S[:, 0], S[:, 1])
    if triangulation.triangles.size == 0:
        return triangulation

    triangles = triangulation.triangles
    tri_points = np.asarray(S, dtype=np.float64)[triangles]
    edge_lengths = np.stack(
        [
            np.linalg.norm(tri_points[:, 0] - tri_points[:, 1], axis=1),
            np.linalg.norm(tri_points[:, 1] - tri_points[:, 2], axis=1),
            np.linalg.norm(tri_points[:, 2] - tri_points[:, 0], axis=1),
        ],
        axis=1,
    )
    positive_edges = edge_lengths[edge_lengths > 0]
    if positive_edges.size == 0:
        return triangulation

    # Mask thin boundary triangles and triangles that span unusually large gaps.
    analyzer = mtri.TriAnalyzer(triangulation)
    mask = analyzer.get_flat_tri_mask(min_circle_ratio=0.01)
    long_edge_threshold = 3.0 * float(np.median(positive_edges))
    mask |= edge_lengths.max(axis=1) > long_edge_threshold
    triangulation.set_mask(mask)
    return triangulation


def _plot_spatial_isodepth(
    ax,
    S: np.ndarray,
    depth: np.ndarray,
    title: str,
    *,
    normalize_bounds: tuple[float, float] | None = None,
    spatial_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    colorbar_label: str = "Normalized isodepth",
) -> None:
    bounds = None if normalize_bounds is None else (float(normalize_bounds[0]), float(normalize_bounds[1]))
    depth = _normalize_depth(
        depth,
        vmin=None if bounds is None else bounds[0],
        vmax=None if bounds is None else bounds[1],
    )
    scatter = ax.scatter(
        S[:, 0],
        S[:, 1],
        c=depth,
        cmap="viridis",
        s=_point_size(S),
        linewidths=0,
        alpha=0.9,
    )
    if S.shape[0] >= 3:
        try:
            triangulation = _masked_triangulation(np.asarray(S, dtype=np.float32))
            contour_levels = np.linspace(0.1, 0.9, 7)
            contour_colors = plt.cm.Reds(np.linspace(0.35, 0.95, contour_levels.size))
            ax.tricontour(
                triangulation,
                depth,
                levels=contour_levels,
                colors=contour_colors,
                linewidths=0.9,
                alpha=0.9,
            )
        except (RuntimeError, ValueError):
            pass
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    if spatial_limits is not None:
        xlim, ylim = spatial_limits
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def _flatten_isodepth_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    raise ValueError(f"Expected 1D isodepth or single-column 2D isodepth, got shape {array.shape}")


def save_parallelization_grid(
    spatial_batches: np.ndarray,
    isodepth_batches: np.ndarray,
    out_path: str | Path,
    *,
    panel_titles: list[str] | None = None,
    figure_title: str | None = None,
) -> Path:
    spatial_array = np.asarray(spatial_batches, dtype=np.float32)
    isodepth_array = np.asarray(isodepth_batches, dtype=np.float32)
    if spatial_array.ndim != 3 or spatial_array.shape[-1] != 2:
        raise ValueError(f"spatial_batches must have shape (M, N, 2), got {spatial_array.shape}")
    if isodepth_array.ndim == 3 and isodepth_array.shape[-1] == 1:
        isodepth_array = isodepth_array[:, :, 0]
    if isodepth_array.ndim != 2:
        raise ValueError(f"isodepth_batches must have shape (M, N) or (M, N, 1), got {isodepth_array.shape}")
    if spatial_array.shape[:2] != isodepth_array.shape:
        raise ValueError(
            "spatial_batches and isodepth_batches must agree on (M, N), "
            f"got {spatial_array.shape[:2]} vs {isodepth_array.shape}"
        )

    n_panels = int(spatial_array.shape[0])
    titles = panel_titles or [f"Model {index + 1}" for index in range(n_panels)]
    n_cols = min(4, max(1, int(np.ceil(np.sqrt(n_panels)))))
    n_rows = int(np.ceil(n_panels / n_cols))

    out_path = Path(out_path)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows), squeeze=False)
    for axis, spatial, isodepth, title in zip(axes.flat, spatial_array, isodepth_array, titles):
        _plot_spatial_isodepth(axis, spatial, isodepth, title)
    for axis in axes.flat[n_panels:]:
        axis.axis("off")
    if figure_title:
        fig.suptitle(figure_title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_parallelization_paired_comparison(
    spatial_batches: np.ndarray,
    parallel_isodepths: np.ndarray,
    sequential_isodepths: np.ndarray,
    out_path: str | Path,
    *,
    row_titles: list[str] | None = None,
) -> Path:
    spatial_array = np.asarray(spatial_batches, dtype=np.float32)
    parallel_array = np.asarray(parallel_isodepths, dtype=np.float32)
    sequential_array = np.asarray(sequential_isodepths, dtype=np.float32)
    if parallel_array.ndim == 3 and parallel_array.shape[-1] == 1:
        parallel_array = parallel_array[:, :, 0]
    if sequential_array.ndim == 3 and sequential_array.shape[-1] == 1:
        sequential_array = sequential_array[:, :, 0]
    if spatial_array.ndim != 3 or spatial_array.shape[-1] != 2:
        raise ValueError(f"spatial_batches must have shape (M, N, 2), got {spatial_array.shape}")
    if parallel_array.ndim != 2 or sequential_array.ndim != 2:
        raise ValueError(
            "parallel_isodepths and sequential_isodepths must have shape (M, N) or (M, N, 1), "
            f"got {parallel_array.shape} and {sequential_array.shape}"
        )
    if spatial_array.shape[:2] != parallel_array.shape or parallel_array.shape != sequential_array.shape:
        raise ValueError(
            "spatial and isodepth batches must agree on (M, N), "
            f"got {spatial_array.shape[:2]}, {parallel_array.shape}, and {sequential_array.shape}"
        )

    n_rows = int(spatial_array.shape[0])
    titles = row_titles or [f"Model {index + 1}" for index in range(n_rows)]
    out_path = Path(out_path)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.8 * n_rows), squeeze=False)
    for row_index in range(n_rows):
        spatial = spatial_array[row_index]
        parallel_depth = _flatten_isodepth_vector(parallel_array[row_index])
        sequential_depth = _flatten_isodepth_vector(sequential_array[row_index])
        shared_bounds = (
            float(min(parallel_depth.min(), sequential_depth.min())),
            float(max(parallel_depth.max(), sequential_depth.max())),
        )
        diff_depth = np.abs(parallel_depth - sequential_depth)
        _plot_spatial_isodepth(
            axes[row_index, 0],
            spatial,
            parallel_depth,
            f"{titles[row_index]}\nParallel",
            normalize_bounds=shared_bounds,
        )
        _plot_spatial_isodepth(
            axes[row_index, 1],
            spatial,
            sequential_depth,
            f"{titles[row_index]}\nSequential",
            normalize_bounds=shared_bounds,
        )
        _plot_spatial_isodepth(
            axes[row_index, 2],
            spatial,
            diff_depth,
            f"{titles[row_index]}\nAbs diff",
            colorbar_label="Normalized abs diff",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _bias_detection_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt(float((a_c * a_c).sum()) * float((b_c * b_c).sum())))
    if denom <= 1e-12:
        return float("nan")
    return float((a_c * b_c).sum() / denom)


def compute_isodepth_bias_detection_similarity(
    isodepths_by_device: dict[str, np.ndarray],
) -> dict[str, list[dict[str, float]]]:
    """Per-device, per-slot Pearson against that device's true-data slot (column 0).

    Returns ``{device: [ {model_index, perm_index, pearson}, ... ]}``; the entry for
    ``model_index == 0`` (the true-data slot itself) is included with ``perm_index = None``
    and ``pearson = 1.0`` for completeness.
    """
    similarity: dict[str, list[dict[str, float]]] = {}
    for device_name, batch in isodepths_by_device.items():
        depth_batch = np.asarray(batch, dtype=np.float32)
        if depth_batch.ndim == 3 and depth_batch.shape[-1] == 1:
            depth_batch = depth_batch[:, :, 0]
        if depth_batch.ndim != 2:
            raise ValueError(
                f"isodepths for {device_name!r} must have shape (M, N) or (M, N, 1), got {depth_batch.shape}"
            )
        true_depth = _flatten_isodepth_vector(depth_batch[0]).astype(np.float64)
        device_rows: list[dict[str, float]] = []
        for model_index in range(depth_batch.shape[0]):
            other = _flatten_isodepth_vector(depth_batch[model_index]).astype(np.float64)
            entry: dict[str, float | None | int] = {
                "model_index": int(model_index),
                "perm_index": None if model_index == 0 else int(model_index - 1),
            }
            entry["pearson"] = (
                1.0 if model_index == 0 else _bias_detection_pearson(true_depth, other)
            )
            device_rows.append(entry)
        similarity[str(device_name)] = device_rows
    return similarity


def compute_isodepth_cross_correlation_matrices(
    isodepths_by_device: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-device M\u00d7M matrix of pairwise Pearson correlations between every pair of slots
    (true + permutations). ``M[i, j] = Pearson(isodepth_slot_i, isodepth_slot_j)``; symmetric,
    diagonal is exactly 1.0."""
    matrices: dict[str, np.ndarray] = {}
    for device_name, batch in isodepths_by_device.items():
        depth_batch = np.asarray(batch, dtype=np.float32)
        if depth_batch.ndim == 3 and depth_batch.shape[-1] == 1:
            depth_batch = depth_batch[:, :, 0]
        if depth_batch.ndim != 2:
            raise ValueError(
                f"isodepths for {device_name!r} must have shape (M, N) or (M, N, 1), got {depth_batch.shape}"
            )
        m = int(depth_batch.shape[0])
        flat = np.stack(
            [_flatten_isodepth_vector(depth_batch[i]).astype(np.float64) for i in range(m)],
            axis=0,
        )
        matrix = np.empty((m, m), dtype=np.float64)
        for i in range(m):
            matrix[i, i] = 1.0
            for j in range(i + 1, m):
                value = _bias_detection_pearson(flat[i], flat[j])
                matrix[i, j] = value
                matrix[j, i] = value
        matrices[str(device_name)] = matrix
    return matrices


def save_isodepth_cross_correlation_matrix_figure(
    matrices_by_device: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    panel_titles: list[str] | None = None,
    figure_title: str | None = None,
    annotate_threshold: int = 30,
) -> Path:
    """Heatmap(s) of the per-device M\u00d7M Pearson correlation matrix from
    ``compute_isodepth_cross_correlation_matrices``. One subplot per device, axes labeled by
    ``panel_titles``. Cells are annotated with the correlation value when ``M <= annotate_threshold``.
    """
    if not matrices_by_device:
        raise ValueError("matrices_by_device must be non-empty")
    out_path = Path(out_path)
    devices = list(matrices_by_device.keys())
    n_devices = len(devices)
    first_matrix = np.asarray(next(iter(matrices_by_device.values())), dtype=np.float64)
    m = int(first_matrix.shape[0])
    titles = panel_titles or [f"Slot {index + 1}" for index in range(m)]
    if len(titles) != m:
        raise ValueError(f"panel_titles length {len(titles)} must equal matrix size {m}")

    fig_w = max(5.5, 0.45 * m + 2.5) * n_devices
    fig_h = max(5.5, 0.45 * m + 2.5)
    fig, axes = plt.subplots(1, n_devices, figsize=(fig_w, fig_h), squeeze=False)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    for ax, device_name in zip(axes.flat, devices):
        matrix = np.asarray(matrices_by_device[device_name], dtype=np.float64)
        if matrix.shape != (m, m):
            raise ValueError(
                f"matrix for {device_name!r} must be square ({m}, {m}), got {matrix.shape}"
            )
        im = ax.imshow(matrix, cmap="coolwarm", norm=norm, interpolation="nearest")
        ax.set_title(f"{device_name}\nPearson cross-correlation")
        ax.set_xticks(range(m))
        ax.set_yticks(range(m))
        ax.set_xticklabels(titles, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(titles, fontsize=8)
        if m <= int(annotate_threshold):
            for i in range(m):
                for j in range(m):
                    val = matrix[i, j]
                    text_color = "white" if abs(val) > 0.55 else "0.1"
                    ax.text(
                        j, i,
                        "nan" if not np.isfinite(val) else f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=7, color=text_color,
                    )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")
    if figure_title:
        fig.suptitle(figure_title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_isodepth_bias_detection_figure(
    spatial_batches: np.ndarray,
    isodepths_by_device: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    device_order: list[str],
    panel_titles: list[str] | None = None,
    figure_title: str | None = None,
) -> Path:
    """Grid: one column per permutation slot; one row per device (shared color scale within each column).

    When at least two devices are present, an extra row plots their signed isodepth difference (first − second
    in ``device_order``) on the **raw** latent scale (color axis in model units, not min–max compressed).

    For each permutation column (col_index >= 1), each device's panel is annotated with the per-device
    ``Pearson(true_isodepth_for_device, perm_isodepth_for_device)`` (a single rankable similarity number).
    The full M\u00d7M cross-correlation matrix is computed separately by
    ``compute_isodepth_cross_correlation_matrices`` and rendered by
    ``save_isodepth_cross_correlation_matrix_figure``.
    """
    spatial_array = np.asarray(spatial_batches, dtype=np.float32)
    if spatial_array.ndim != 3 or spatial_array.shape[-1] != 2:
        raise ValueError(f"spatial_batches must have shape (M, N, 2), got {spatial_array.shape}")

    n_models = int(spatial_array.shape[0])
    titles = panel_titles or [f"Slot {index + 1}" for index in range(n_models)]

    ordered_devices = [str(d) for d in device_order]
    if not ordered_devices:
        raise ValueError("device_order must be non-empty")
    for name in ordered_devices:
        if name not in isodepths_by_device:
            raise ValueError(f"Missing isodepth batch for device {name!r}")

    depth_rows: list[np.ndarray] = []
    for device_name in ordered_devices:
        batch = np.asarray(isodepths_by_device[device_name], dtype=np.float32)
        if batch.ndim == 3 and batch.shape[-1] == 1:
            batch = batch[:, :, 0]
        if batch.ndim != 2 or batch.shape[0] != n_models:
            raise ValueError(
                f"isodepths for {device_name!r} must have shape ({n_models}, N), got {batch.shape}"
            )
        if spatial_array.shape[:2] != batch.shape:
            raise ValueError(f"Spatial vs isodepth mismatch for {device_name}: {spatial_array.shape[:2]} vs {batch.shape}")
        depth_rows.append(batch)

    show_diff_row = len(ordered_devices) >= 2
    n_rows = len(depth_rows) + (1 if show_diff_row else 0)

    out_path = Path(out_path)
    fig, axes = plt.subplots(n_rows, n_models, figsize=(4.6 * n_models, 3.9 * n_rows), squeeze=False)

    for col_index in range(n_models):
        spatial = spatial_array[col_index]
        shared_bounds = (
            float(min(row[col_index].min() for row in depth_rows)),
            float(max(row[col_index].max() for row in depth_rows)),
        )
        for row_index, device_name in enumerate(ordered_devices):
            depth = _flatten_isodepth_vector(depth_rows[row_index][col_index])
            row_title = f"{titles[col_index]}\n{device_name}"
            if col_index >= 1:
                true_depth_for_device = _flatten_isodepth_vector(depth_rows[row_index][0])
                pearson = _bias_detection_pearson(true_depth_for_device, depth)
                row_title += f"\nr={pearson:.3f}"
            _plot_spatial_isodepth(
                axes[row_index, col_index],
                spatial,
                depth,
                row_title,
                normalize_bounds=shared_bounds,
            )
        if show_diff_row:
            d0 = _flatten_isodepth_vector(depth_rows[0][col_index]).astype(np.float64)
            d1 = _flatten_isodepth_vector(depth_rows[1][col_index]).astype(np.float64)
            signed_diff = d0 - d1
            diff_row_index = len(ordered_devices)
            _plot_spatial_signed_difference(
                axes[diff_row_index, col_index],
                spatial,
                signed_diff,
                f"{titles[col_index]}\nΔ ({ordered_devices[0]} − {ordered_devices[1]})",
            )

    for axis in axes.flat:
        axis.label_outer()

    if figure_title:
        fig.suptitle(figure_title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_spatial_signed_difference(ax, S: np.ndarray, diff: np.ndarray, title: str) -> None:
    """Scatter signed difference with a diverging colormap; colorbar ticks are raw model units (not rescaled to [0, 1])."""
    diff = np.asarray(diff, dtype=np.float64).reshape(-1)
    bound = float(max(abs(float(diff.min())), abs(float(diff.max())), 1e-15))
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    scatter = ax.scatter(
        S[:, 0],
        S[:, 1],
        c=diff,
        cmap="coolwarm",
        norm=norm,
        s=_point_size(S),
        linewidths=0,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Isodepth difference")


def _overlay_subsampling(ax, S: np.ndarray, subset_mask: np.ndarray | None) -> None:
    if subset_mask is None:
        return
    selected = np.asarray(subset_mask, dtype=np.float32).reshape(-1) > 0
    if not np.any(selected):
        return
    ax.scatter(
        S[selected, 0],
        S[selected, 1],
        facecolors="none",
        edgecolors="black",
        s=max(18.0, _point_size(S) * 3.0),
        linewidths=0.8,
        alpha=0.95,
    )


def _cell_expression_signal(A: np.ndarray) -> np.ndarray:
    expression = np.asarray(A, dtype=np.float32)
    if expression.ndim != 2:
        raise ValueError(f"Expected 2D expression matrix, got shape {expression.shape}")
    return np.mean(np.abs(expression), axis=1)


def _plot_spatial_dataset_heatmap(
    ax,
    S: np.ndarray,
    signal: np.ndarray,
    title: str,
    *,
    subset_mask: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    sig = np.asarray(signal, dtype=np.float32)
    if vmin is not None and vmax is not None:
        norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    else:
        norm = None
    scatter = ax.scatter(
        S[:, 0],
        S[:, 1],
        c=sig,
        cmap=DEFAULT_EXPRESSION_AGGREGATE_COLORMAP,
        norm=norm,
        s=_point_size(S),
        linewidths=0,
        alpha=0.9,
    )
    _overlay_subsampling(ax, S, subset_mask)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Mean absolute expression")


def save_dataset_triptych(
    dataset: DatasetBundle,
    result: TestResult,
    out_path: str | Path,
) -> Path | None:
    lowest_S = result.artifacts.get("lowest_S")
    highest_S = result.artifacts.get("highest_S")
    if lowest_S is None or highest_S is None:
        return None

    signal = _cell_expression_signal(dataset.A)
    vmin = float(signal.min())
    vmax = float(signal.max())
    title_prefix = "True Synthetic Dataset" if dataset.meta.get("source") == "synthetic" else "True Dataset"
    n_cells = int(dataset.S.shape[0])
    title_prefix += f" (n={n_cells})"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_spatial_dataset_heatmap(
        axes[0],
        np.asarray(dataset.S, dtype=np.float32),
        signal,
        title_prefix,
        vmin=vmin,
        vmax=vmax,
    )
    _plot_spatial_dataset_heatmap(
        axes[1],
        np.asarray(lowest_S, dtype=np.float32),
        signal,
        f"Lowest Metric Dataset\n{float(result.artifacts.get('lowest_stat', np.nan)):.4g}",
        subset_mask=result.artifacts.get("lowest_subset_mask"),
        vmin=vmin,
        vmax=vmax,
    )
    _plot_spatial_dataset_heatmap(
        axes[2],
        np.asarray(highest_S, dtype=np.float32),
        signal,
        f"Highest Metric Dataset\n{float(result.artifacts.get('highest_stat', np.nan)):.4g}",
        subset_mask=result.artifacts.get("highest_subset_mask"),
        vmin=vmin,
        vmax=vmax,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_celltype_dataset_plot(
    dataset: DatasetBundle,
    out_path: str | Path,
) -> Path | None:
    """Scatter plot of cells colored by their cell-type assignment."""
    cell_type_labels = dataset.meta.get("cell_type_labels")
    cell_type_names = dataset.meta.get("cell_type_names")
    if cell_type_labels is None or cell_type_names is None:
        return None

    labels = np.asarray(cell_type_labels, dtype=np.int64)
    S = np.asarray(dataset.S, dtype=np.float32)
    n_types = len(cell_type_names)
    tumor_prop = dataset.meta.get("calicost_tumor_proportion")
    tumor_prop_arr = None
    if tumor_prop is not None:
        tumor_prop_arr = np.asarray(tumor_prop, dtype=np.float32).reshape(-1)
        if tumor_prop_arr.shape[0] != S.shape[0]:
            tumor_prop_arr = None

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    legend_names: list[str] = []
    legend_base_colors: list[np.ndarray] = []
    for c in range(n_types):
        mask = labels == c
        if not np.any(mask):
            continue
        base_color = _calicost_clone_base_color(
            cell_type_names[c],
            fallback_index=c,
            n_types=n_types,
        )
        colors = [base_color]
        alpha = 0.7
        if tumor_prop_arr is not None:
            colors = _clone_tumor_proportion_colors(base_color, tumor_prop_arr[mask])
            alpha = None
        ax.scatter(
            S[mask, 0],
            S[mask, 1],
            c=colors,
            s=_clone_overview_point_size(S),
            alpha=alpha,
            linewidths=0,
        )
        legend_names.append(cell_type_names[c])
        legend_base_colors.append(base_color)
    if tumor_prop_arr is not None:
        ax.set_title("CalicoST Clones (Color Intensity = Tumor Proportion)")
    else:
        ax.set_title("Dataset Colored by Cell Type")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(
        handles=_clone_legend_handles(legend_names, legend_base_colors),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize="x-small",
        frameon=False,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _covariate_scatter_panel(
    ax,
    S: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plain continuous-colormap scatter of raw (unnormalized) covariate values (white to dark red)."""
    scatter = ax.scatter(
        S[:, 0],
        S[:, 1],
        c=values,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        s=_clone_overview_point_size(S),
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    return scatter


def save_covariate_whitening_spatial_plot(
    dataset: DatasetBundle,
    out_path: str | Path,
) -> Path | None:
    """Spatial map of the raw ``data.covariate_whitening`` obs column(s).

    Shows the untransformed obs values on a real (non-normalized) colorbar, independent
    of any clone/cell-type color coding, so the covariate can be inspected on its own scale.
    One panel per cell type (shared color scale) when the dataset used
    ``cell_type='separate'``; otherwise a single panel over all cells.  When multiple
    whitening covariates are configured, writes one figure with one column per covariate.
    Returns ``None`` when the run does not use ``data.covariate_whitening``.
    """
    cov_values = dataset.meta.get("covariate_whitening_values")
    obs_key = dataset.meta.get("covariate_whitening_obs_key")
    if cov_values is None or not obs_key:
        return None

    S = np.asarray(dataset.S, dtype=np.float32)
    cov = np.asarray(cov_values, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    if cov.shape[0] != S.shape[0]:
        return None
    if isinstance(obs_key, (list, tuple)):
        labels = [str(k) for k in obs_key]
    else:
        labels = [str(obs_key)]
    if cov.shape[1] != len(labels):
        labels = [f"{labels[0]}[{j}]" for j in range(cov.shape[1])] if len(labels) == 1 else [
            f"cov{j}" for j in range(cov.shape[1])
        ]

    cell_type_labels = dataset.meta.get("cell_type_labels")
    cell_type_names = dataset.meta.get("cell_type_names")
    use_grid = (
        dataset.meta.get("cell_type_mode") == "separate"
        and cell_type_labels is not None
        and cell_type_names
    )

    n_cov = cov.shape[1]
    if use_grid:
        type_labels = np.asarray(cell_type_labels, dtype=np.int64)
        n_types = len(cell_type_names)
        fig, axes = plt.subplots(
            n_cov, n_types, figsize=(4.6 * n_types, 4.2 * n_cov), squeeze=False
        )
        for j in range(n_cov):
            col = cov[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                for c in range(n_types):
                    axes[j, c].axis("off")
                continue
            vmin = float(col[finite].min())
            vmax = float(col[finite].max())
            for c in range(n_types):
                ax = axes[j, c]
                mask = type_labels == c
                if not np.any(mask):
                    ax.axis("off")
                    continue
                title = str(cell_type_names[c]) if n_cov == 1 else f"{cell_type_names[c]} · {labels[j]}"
                _covariate_scatter_panel(
                    ax,
                    S[mask],
                    col[mask],
                    title,
                    colorbar_label=labels[j],
                    vmin=vmin,
                    vmax=vmax,
                )
        fig.suptitle("Covariate whitening: " + ", ".join(labels))
    else:
        fig, axes = plt.subplots(1, n_cov, figsize=(5.2 * n_cov, 5.0), squeeze=False)
        for j in range(n_cov):
            col = cov[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                axes[0, j].axis("off")
                continue
            vmin = float(col[finite].min())
            vmax = float(col[finite].max())
            _covariate_scatter_panel(
                axes[0, j],
                S,
                col,
                f"Covariate: {labels[j]}",
                colorbar_label=labels[j],
                vmin=vmin,
                vmax=vmax,
            )

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_single_type_covariate_plot(
    S: np.ndarray,
    values: np.ndarray,
    out_path: str | Path,
    *,
    covariate_label: str,
    type_name: str,
) -> Path | None:
    """Single-panel raw-covariate scatter for one cell type's spots (own color scale)."""
    S = np.asarray(S, dtype=np.float32)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if S.shape[0] == 0 or values.shape[0] != S.shape[0]:
        return None
    finite = np.isfinite(values)
    if not finite.any():
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.0))
    _covariate_scatter_panel(
        ax,
        S,
        values,
        f"{type_name}: {covariate_label}",
        colorbar_label=covariate_label,
        vmin=float(values[finite].min()),
        vmax=float(values[finite].max()),
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _numeric_filter_label(obs_key: str, filter_spec: dict) -> str:
    op_labels = {
        "gt": ">",
        "ge": ">=",
        "gte": ">=",
        "lt": "<",
        "le": "<=",
        "lte": "<=",
        "eq": "=",
        "ne": "!=",
    }
    parts = []
    for op, threshold in filter_spec.items():
        parts.append(f"{obs_key} {op_labels.get(str(op), str(op))} {threshold:g}")
    return " and ".join(parts)


def save_obs_numeric_filter_histogram(
    values: np.ndarray,
    keep_mask: np.ndarray,
    obs_key: str,
    filter_spec: dict,
    out_path: str | Path,
    *,
    subset_label: str | None = None,
) -> Path | None:
    """Histogram of an obs numeric filter variable for all pre-threshold cells."""
    del keep_mask, filter_spec  # retained for call-site compatibility
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] == 0:
        return None

    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = max(10, min(50, int(np.sqrt(values.size)) + 1))
    ax.hist(values, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6)

    title = obs_key if subset_label is None else f"{obs_key} ({subset_label})"
    ax.set_title(title)
    ax.set_xlabel(obs_key)
    ax.set_ylabel("Number of cells")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_obs_numeric_filter_diagnostic_plot(
    S: np.ndarray,
    *,
    labels: np.ndarray | None,
    label_names: list[str] | None,
    values: np.ndarray,
    keep_mask: np.ndarray,
    obs_key: str,
    filter_spec: dict,
    out_path: str | Path,
    label_title: str = "CNV clone",
) -> Path | None:
    """Two-panel pre/post view for an obs numeric threshold.

    Left panel shows all pre-threshold spots colored by label, with color intensity
    scaled by the numeric obs value. Right panel shows kept spots (solid clone color)
    vs removed spots (faint grey).
    """
    S = np.asarray(S, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if S.ndim != 2 or S.shape[1] != 2 or S.shape[0] == 0:
        return None
    if values.shape[0] != S.shape[0] or keep_mask.shape[0] != S.shape[0]:
        return None

    if labels is not None:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.shape[0] != S.shape[0]:
            labels_arr = None
    else:
        labels_arr = None

    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        weights = np.clip(values, 0.0, 1.0)
    else:
        weights = np.ones(S.shape[0], dtype=np.float32)

    threshold_text = _numeric_filter_label(obs_key, filter_spec)
    point_size = _clone_overview_point_size(S)
    limits = _square_spatial_axis_limits(S)

    # Dynamic figsize: make each panel roughly square based on the data aspect.
    # _square_spatial_axis_limits already equalises x/y span, so each panel IS square
    # in data space. We size the figure so the rendered panels are square in inches.
    span = limits[0][1] - limits[0][0]          # same for x and y (already equal)
    x_span = (limits[0][1] - limits[0][0])
    y_span = (limits[1][1] - limits[1][0])
    # Per-panel height in inches; width follows the data aspect ratio
    panel_h = 5.5
    panel_w = panel_h * (x_span / y_span)
    fig_w = panel_w * 2 + 2.0   # 2 panels + inner gap + outer margins
    fig_h = panel_h + 1.2       # panel + title / xlabel headroom

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharex=True, sharey=True)

    # --- Left panel: all spots, clone colour × tumor-proportion intensity ---
    ax = axes[0]
    legend_names: list[str] = []
    legend_base_colors: list[np.ndarray] = []
    if labels_arr is not None:
        unique_labels = np.unique(labels_arr)
        n_types = len(unique_labels)
        resolved_names = list(label_names or [])
        for i, label in enumerate(unique_labels):
            mask = labels_arr == label
            label_name = (
                str(resolved_names[int(label)])
                if 0 <= int(label) < len(resolved_names)
                else str(label)
            )
            base_color = _calicost_clone_base_color(
                label_name, fallback_index=i, n_types=n_types
            )
            colors = _clone_tumor_proportion_colors(base_color, weights[mask])
            ax.scatter(S[mask, 0], S[mask, 1], c=colors, s=point_size, linewidths=0)
            legend_names.append(label_name)
            legend_base_colors.append(base_color)
        ax.legend(
            handles=_clone_legend_handles(legend_names, legend_base_colors),
            title=label_title,
            loc="lower right",
            fontsize="x-small",
            title_fontsize="x-small",
            frameon=True,
            framealpha=0.75,
        )
    else:
        scatter = ax.scatter(
            S[:, 0], S[:, 1], c=values, cmap="viridis",
            s=point_size, alpha=0.9, linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=obs_key)
    ax.set_title(f"All Spots by {label_title}\nColor intensity = {obs_key}")

    # --- Right panel: kept (solid clone colour) vs removed (faint grey) ---
    ax = axes[1]
    dropped_mask = ~keep_mask
    dropped_size = max(point_size * 0.55, 8.0)

    # Removed spots: uniform faint grey, plotted first (background)
    if np.any(dropped_mask):
        ax.scatter(
            S[dropped_mask, 0], S[dropped_mask, 1],
            c="#c8c8c8", s=dropped_size, linewidths=0,
            alpha=0.45, zorder=1,
        )

    # Kept spots: solid clone colour, foreground
    if labels_arr is not None:
        unique_labels = np.unique(labels_arr)
        n_types = len(unique_labels)
        resolved_names = list(label_names or [])
        for i, label in enumerate(unique_labels):
            label_mask = labels_arr == label
            kept_label_mask = label_mask & keep_mask
            if not np.any(kept_label_mask):
                continue
            label_name = (
                str(resolved_names[int(label)])
                if 0 <= int(label) < len(resolved_names)
                else str(label)
            )
            base_color = _calicost_clone_base_color(
                label_name, fallback_index=i, n_types=n_types
            )
            clone_colors = _clone_tumor_proportion_colors(base_color, weights[kept_label_mask], alpha=1.0)
            ax.scatter(
                S[kept_label_mask, 0], S[kept_label_mask, 1],
                c=clone_colors, s=point_size, linewidths=0,
                zorder=2,
            )
    else:
        if np.any(keep_mask):
            scatter = ax.scatter(
                S[keep_mask, 0], S[keep_mask, 1],
                c=values[keep_mask], cmap="viridis",
                s=point_size, alpha=0.95, linewidths=0, zorder=2,
            )

    # Legend: grey swatch = left out, dark dot = kept
    ax.scatter([], [], c="#c8c8c8", s=14, linewidths=0,
               label=f"left out (n={int(dropped_mask.sum())})")
    ax.scatter([], [], c="#555555", s=14, linewidths=0,
               label=f"kept (n={int(keep_mask.sum())})")
    ax.legend(
        loc="lower right",
        fontsize="x-small",
        frameon=True,
        framealpha=0.75,
    )
    ax.set_title(f"Threshold Selection\n{threshold_text}")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_spatial_region_split_plot(
    dataset: DatasetBundle,
    out_path: str | Path,
) -> Path | None:
    """Overview of spatial region splitting: grey kept spots, red removed, colored regions."""
    diag = dataset.meta.get("spatial_region_split_diag")
    if not diag:
        return None

    S = np.asarray(diag["S"], dtype=np.float32)
    removed = np.asarray(diag["removed"], dtype=bool)
    region_color_ids = np.asarray(diag["region_color_ids"], dtype=np.int64)
    region_color_names = list(diag.get("region_color_names") or [])
    algorithm = str(diag.get("algorithm", "spatial region split"))
    point_size = _point_size(S)

    kept = ~removed
    grey_mask = kept & (region_color_ids < 0)
    removed_mask = removed

    n_colors = len(region_color_names)
    cmap = plt.cm.get_cmap("tab20" if n_colors > 10 else "tab10")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if np.any(grey_mask):
        ax.scatter(
            S[grey_mask, 0],
            S[grey_mask, 1],
            c="#b0b0b0",
            s=point_size,
            label="kept",
            alpha=0.85,
            linewidths=0,
            zorder=1,
        )

    for color_id, region_name in enumerate(region_color_names):
        mask = kept & (region_color_ids == color_id)
        if not np.any(mask):
            continue
        ax.scatter(
            S[mask, 0],
            S[mask, 1],
            c=[cmap(color_id / max(n_colors - 1, 1))],
            s=point_size,
            label=region_name,
            alpha=0.85,
            linewidths=0,
            zorder=2,
        )

    if np.any(removed_mask):
        ax.scatter(
            S[removed_mask, 0],
            S[removed_mask, 1],
            c="#d62728",
            s=point_size,
            label="removed (noise / below min_cells)",
            alpha=0.95,
            linewidths=0,
            zorder=3,
        )

    ax.set_title(f"Spatial Region Split ({algorithm})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize="x-small",
        markerscale=2.0,
        frameon=False,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_celltype_or_spatial_split_plot(
    dataset: DatasetBundle,
    out_path: str | Path,
) -> Path | None:
    """Save spatial split overview when available, otherwise cell-type scatter."""
    if dataset.meta.get("spatial_region_split_diag"):
        return save_spatial_region_split_plot(dataset, out_path)
    return save_celltype_dataset_plot(dataset, out_path)


def save_synthetic_true_curve_plot(
    dataset: DatasetBundle,
    out_path: str | Path,
) -> Path | None:
    if dataset.meta.get("source") != "synthetic":
        return None

    mode = str(dataset.meta.get("mode", ""))
    if mode not in {"radial", "fourier", "noise"}:
        return None

    true_curve = dataset.meta.get("synthetic_true_curve")
    if true_curve is None:
        return None

    title = "True Synthetic Isodepth"
    if mode == "noise":
        title = "True Synthetic Isodepth (Flat Null)"
    elif mode == "fourier":
        title = "True Synthetic Isodepth (Fourier)"
    elif mode == "radial":
        title = "True Synthetic Isodepth (Radial)"

    gh = int(dataset.meta.get("grid_height", 0) or 0)
    gw = int(dataset.meta.get("grid_width", 0) or 0)
    if gh > 0 and gw > 0 and gh != gw:
        aspect = float(gw) / float(gh)
        fig_w = float(np.clip(6.0 * aspect, 3.5, 14.0))
        fig_h = float(np.clip(5.0 / max(aspect, 1e-6), 3.5, 10.0))
        figsize = (fig_w, fig_h)
    else:
        figsize = (6, 5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _plot_spatial_isodepth(
        ax,
        np.asarray(dataset.S, dtype=np.float32),
        np.asarray(true_curve, dtype=np.float32),
        title,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_synthetic_kernel_plot(
    dataset: DatasetBundle,
    out_path: "str | Path",
) -> "dict[str, Path] | None":
    """Kernel diagnostics as two separate, normally-sized figures.

    Writes ``{run}_kernel_correlation.png`` (decay curve) and
    ``{run}_kernel_noise_sample.png`` (square spatial noise draw).

    *out_path* may be the legacy ``*_kernel_diagnostics.png`` path (used only to
    derive the output directory and run-name prefix); the combined file is no
    longer written.

    Returns a dict of saved paths or None if the dataset has no kernel metadata.
    """
    if dataset.meta.get("source") != "synthetic":
        return None
    kernel_meta = dataset.meta.get("kernel")
    if kernel_meta is None:
        return None
    delta = float(dataset.meta.get("delta", 0.0))
    scale_um = float(dataset.meta.get("scale_um", 1.0))
    p = float(kernel_meta["distance"])
    kernel_type = str(kernel_meta.get("type", "exp"))
    r_max_explicit = kernel_meta.get("max_interaction_distance")
    if kernel_type == "trunc":
        r_max = float(r_max_explicit) if r_max_explicit is not None else p
    else:
        r_max = float(r_max_explicit) if r_max_explicit is not None else 4.0 * p
    local_fraction = delta / (1.0 + delta)

    out_path = Path(out_path)
    prefix = out_path.name.replace("_kernel_diagnostics.png", "").replace(".png", "")
    corr_path = out_path.parent / f"{prefix}_kernel_correlation.png"
    noise_path = out_path.parent / f"{prefix}_kernel_noise_sample.png"

    # ── Correlation curve ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    r = np.linspace(0.0, r_max * 1.05, 400)
    if kernel_type == "trunc":
        k_shape = np.where(r <= r_max, 1.0, 0.0)
        corr = delta * k_shape / (1.0 + delta)
        ax.plot(r, corr, color="steelblue", lw=2, label=r"$\delta / (1+\delta)$ for $d \leq r_{\max}$")
        ax.axvline(r_max, color="tomato", lw=1.2, ls="--", label=f"cutoff = {r_max:g} µm")
        if r_max_explicit is not None and abs(r_max - p) > 1e-9:
            ax.axvline(p, color="gray", lw=1.2, ls="--", label=f"$p$ = {p:g} µm")
        title = f"Kernel: trunc, p={p:g} µm, cutoff={r_max:g} µm, δ={delta:g}"
    else:
        corr = delta * np.exp(-r / p) / (1.0 + delta)
        ax.plot(r, corr, color="steelblue", lw=2, label=r"$\delta \cdot e^{-r/p}\,/\,(1+\delta)$")
        ax.axvline(p, color="gray", lw=1.2, ls="--", label=f"$p$ = {p:g} µm")
        if r_max_explicit is not None:
            ax.axvline(r_max, color="tomato", lw=1.2, ls="--", label=f"cutoff = {r_max:g} µm")
        title = f"Kernel: exp, p={p:g} µm, δ={delta:g}"
    ax.axhline(local_fraction, color="steelblue", lw=0.8, ls=":", alpha=0.6)
    ax.annotate(
        f"local fraction\n$\\delta/(1+\\delta)$ = {local_fraction:.2f}",
        xy=(0, local_fraction),
        xytext=(r_max * 0.3, local_fraction + 0.04),
        fontsize=8,
        color="steelblue",
    )
    ax.set_xlabel("Distance (µm)")
    ax.set_ylabel("Spatial correlation")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Spatial noise sample (square axes, scatter at cell coords) ────────
    # Noise is one value per cell (same ordering as ``dataset.S``). Plot with
    # scatter like the dataset/isodepth panels — not imshow reshape, which
    # falsely renders a filled pixel grid even when cells are irregular.
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    noise_sample = dataset.meta.get("kernel_noise_sample")
    S = np.asarray(dataset.S, dtype=np.float32)
    spatial_extent = (0.0, scale_um, 0.0, scale_um)
    sigma = dataset.meta.get("sigma")
    sigma_label = f"{float(sigma):g}" if sigma is not None else "?"
    if noise_sample is not None:
        noise = np.asarray(noise_sample, dtype=np.float32)
        if noise.shape[0] != S.shape[0]:
            ax.text(
                0.5, 0.5,
                f"noise length {noise.shape[0]} != n_cells {S.shape[0]}",
                ha="center", va="center", transform=ax.transAxes, color="gray",
            )
        else:
            vmax = float(np.abs(noise).max()) or 1.0
            sc = ax.scatter(
                S[:, 0] * scale_um,
                S[:, 1] * scale_um,
                c=noise,
                cmap="RdBu_r",
                s=_point_size(S),
                vmin=-vmax,
                vmax=vmax,
                linewidths=0,
                alpha=0.9,
            )
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cx = scale_um * 0.5
            cy = scale_um * 0.5
            circle = plt.Circle((cx, cy), p, fill=False, color="black", lw=1.2, ls="--")
            ax.add_patch(circle)
        ax.set_xlim(spatial_extent[0], spatial_extent[1])
        ax.set_ylim(spatial_extent[2], spatial_extent[3])
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    else:
        ax.text(
            0.5, 0.5, "noise sample not available",
            ha="center", va="center", transform=ax.transAxes, color="gray",
        )
    ax.set_title(f"Correlated noise (1 gene, σ={sigma_label}, δ={delta:g})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(noise_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "kernel_correlation_plot": corr_path,
        "kernel_noise_sample_plot": noise_path,
    }


def _format_stat_suffix(label: str, value: Any) -> str:
    if value is None:
        return label
    try:
        return f"{label}\n{float(value):.4g}"
    except (TypeError, ValueError):
        return label


def _true_isodepth_panels_for_permutation_result(
    dataset: DatasetBundle,
    result: TestResult,
    true_isodepth: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    spatial = np.asarray(dataset.S, dtype=np.float32)
    artifacts = result.artifacts
    full_iso_depth = artifacts.get("true_isodepth_full_iso")
    covariate_depth = artifacts.get("true_isodepth_covariate")
    stat_covariate = artifacts.get("stat_covariate")

    if full_iso_depth is not None:
        return [
            (
                spatial,
                np.asarray(full_iso_depth, dtype=np.float32),
                _format_stat_suffix("Trained True Isodepth", result.stat_true),
            ),
            (
                spatial,
                true_isodepth,
                _format_stat_suffix("Covariate Isodepth", stat_covariate),
            ),
        ]

    n_cells = int(spatial.shape[0])
    true_label = "Trained True Isodepth" if covariate_depth is not None else "True Data Isodepth"
    true_label += f" (n={n_cells})"
    panels = [
        (
            spatial,
            true_isodepth,
            _format_stat_suffix(true_label, result.stat_true),
        )
    ]
    if covariate_depth is not None:
        panels.append(
            (
                spatial,
                np.asarray(covariate_depth, dtype=np.float32),
                _format_stat_suffix("Covariate Isodepth", stat_covariate),
            )
        )
    return panels


def _save_permutation_triptych(
    dataset: DatasetBundle,
    result: TestResult,
    out_path: Path,
    true_isodepth: np.ndarray,
    lowest_isodepth: np.ndarray,
    lowest_S: np.ndarray,
    highest_isodepth: np.ndarray,
    highest_S: np.ndarray,
) -> Path:
    panels = _true_isodepth_panels_for_permutation_result(dataset, result, true_isodepth)
    panels.extend(
        [
            (
                lowest_S,
                lowest_isodepth,
                f"Lowest Null Isodepth\n{float(result.artifacts.get('lowest_stat')):.4g}",
            ),
            (
                highest_S,
                highest_isodepth,
                f"Highest Null Isodepth\n{float(result.artifacts.get('highest_stat')):.4g}",
            ),
        ]
    )
    n_panels = len(panels)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    for axis, (panel_S, panel_depth, title) in zip(axes.flat, panels):
        _plot_spatial_isodepth(axis, panel_S, panel_depth, title)
    for axis in axes.flat[n_panels:]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_perturbation_triptych(
    dataset: DatasetBundle,
    result: TestResult,
    out_path: Path,
    true_isodepth: np.ndarray,
) -> Path:
    rows = result.artifacts.get("delta_plot_rows") or []
    if not rows:
        fallback_delta = result.artifacts.get("delta", [0.0])
        if isinstance(fallback_delta, (list, tuple, np.ndarray)):
            fallback_delta = float(np.asarray(fallback_delta, dtype=np.float64).reshape(-1)[0])
        rows = [
            {
                "delta": float(fallback_delta),
                "lowest_isodepth": np.asarray(result.artifacts["lowest_isodepth"], dtype=np.float32),
                "lowest_S": np.asarray(result.artifacts["lowest_S"], dtype=np.float32),
                "lowest_stat": float(result.artifacts.get("lowest_stat", np.nan)),
                "highest_isodepth": np.asarray(result.artifacts["highest_isodepth"], dtype=np.float32),
                "highest_S": np.asarray(result.artifacts["highest_S"], dtype=np.float32),
                "highest_stat": float(result.artifacts.get("highest_stat", np.nan)),
            }
        ]

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)
    spatial = np.asarray(dataset.S, dtype=np.float32)
    for row_index, row in enumerate(rows):
        delta = float(row["delta"])
        _plot_spatial_isodepth(
            axes[row_index, 0],
            spatial,
            true_isodepth,
            f"Original Isodepth\nDelta = {delta:.4g}",
        )
        _plot_spatial_isodepth(
            axes[row_index, 1],
            np.asarray(row["lowest_S"], dtype=np.float32),
            np.asarray(row["lowest_isodepth"], dtype=np.float32),
            f"Lowest Metric\n{float(row['lowest_stat']):.4g}",
        )
        _plot_spatial_isodepth(
            axes[row_index, 2],
            np.asarray(row["highest_S"], dtype=np.float32),
            np.asarray(row["highest_isodepth"], dtype=np.float32),
            f"Highest Metric\n{float(row['highest_stat']):.4g}",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_subsampling_triptych(
    dataset: DatasetBundle,
    result: TestResult,
    out_path: Path,
    true_isodepth: np.ndarray,
) -> Path:
    rows = result.artifacts.get("fraction_plot_rows") or []
    if not rows:
        rows = [
            {
                "fraction": float(result.artifacts.get("lowest_subset_fraction", 0.0)),
                "lowest_isodepth": np.asarray(result.artifacts["lowest_isodepth"], dtype=np.float32),
                "lowest_mask": result.artifacts.get("lowest_subset_mask"),
                "lowest_stat": float(result.artifacts.get("lowest_stat", np.nan)),
                "highest_isodepth": np.asarray(result.artifacts["highest_isodepth"], dtype=np.float32),
                "highest_mask": result.artifacts.get("highest_subset_mask"),
                "highest_stat": float(result.artifacts.get("highest_stat", np.nan)),
            }
        ]

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)
    spatial = np.asarray(dataset.S, dtype=np.float32)
    for row_index, row in enumerate(rows):
        fraction = float(row["fraction"])
        _plot_spatial_isodepth(
            axes[row_index, 0],
            spatial,
            true_isodepth,
            f"Full-Data Isodepth\nSubset Fraction = {fraction:.2f}",
        )
        _plot_spatial_isodepth(
            axes[row_index, 1],
            spatial,
            np.asarray(row["lowest_isodepth"], dtype=np.float32),
            f"Lowest Loss\n{float(row['lowest_stat']):.4g}",
        )
        _overlay_subsampling(axes[row_index, 1], spatial, row.get("lowest_mask"))
        _plot_spatial_isodepth(
            axes[row_index, 2],
            spatial,
            np.asarray(row["highest_isodepth"], dtype=np.float32),
            f"Highest Loss\n{float(row['highest_stat']):.4g}",
        )
        _overlay_subsampling(axes[row_index, 2], spatial, row.get("highest_mask"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_true_rerun_isodepth_grid(
    dataset: DatasetBundle,
    rerun_isodepths: np.ndarray,
    out_path: str | Path,
    *,
    rerun_losses: np.ndarray | None = None,
    selected_rerun_index: int | None = None,
) -> Path | None:
    rerun_array = np.asarray(rerun_isodepths, dtype=np.float32)
    if rerun_array.ndim == 2:
        rerun_array = rerun_array[:, :, None]
    if rerun_array.ndim != 3:
        raise ValueError(
            "rerun_isodepths must have shape (n_reruns, n_cells) or (n_reruns, n_cells, latent_dim), "
            f"got {rerun_array.shape}"
        )

    n_reruns, n_cells, latent_dim = rerun_array.shape
    if n_reruns <= 1:
        return None
    if n_cells != dataset.n_cells:
        raise ValueError(
            f"rerun_isodepths cell count must match dataset.n_cells, got {n_cells} vs {dataset.n_cells}"
        )

    losses = None if rerun_losses is None else np.asarray(rerun_losses, dtype=np.float64).reshape(-1)
    spatial = np.asarray(dataset.S, dtype=np.float32)
    panel_specs: list[tuple[np.ndarray, str]] = []
    for rerun_index in range(n_reruns):
        rerun_suffix = f"Rerun {rerun_index + 1}"
        if selected_rerun_index is not None and rerun_index == int(selected_rerun_index):
            rerun_suffix += " (selected)"
        loss_suffix = ""
        if losses is not None and rerun_index < losses.size and np.isfinite(losses[rerun_index]):
            loss_suffix = f"\nloss={losses[rerun_index]:.4g}"
        for dim_index in range(latent_dim):
            title = rerun_suffix
            if latent_dim > 1:
                title += f" · d{dim_index + 1}"
            title += loss_suffix
            panel_specs.append((rerun_array[rerun_index, :, dim_index], title))

    n_panels = len(panel_specs)
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows), squeeze=False)

    for axis, (depth_values, title) in zip(axes.flat, panel_specs):
        _plot_spatial_isodepth(axis, spatial, depth_values, title)

    for axis in axes.flat[n_panels:]:
        axis.axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_freedman_lane_covariate_plot(
    dataset: DatasetBundle,
    result: TestResult,
    out_path: str | Path,
) -> Path | None:
    """Spatial maps of the Freedman–Lane covariate and decoder expression predictions."""
    cov_values = result.artifacts.get("freedman_lane_covariate_values")
    pred = result.artifacts.get("freedman_lane_pred")
    if cov_values is None or pred is None:
        return None

    spatial = np.asarray(dataset.S, dtype=np.float32)
    cov = np.asarray(cov_values, dtype=np.float64).reshape(-1)
    pred_mean = np.asarray(pred, dtype=np.float64).mean(axis=1)
    obs_key = str(result.artifacts.get("freedman_lane_obs_key", "covariate"))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), squeeze=False)
    _plot_spatial_isodepth(
        axes[0, 0],
        spatial,
        cov,
        f"Covariate: {obs_key}",
        colorbar_label=obs_key,
    )
    _plot_spatial_isodepth(
        axes[0, 1],
        spatial,
        pred_mean,
        "Decoder predictions (gene mean)",
        colorbar_label="Mean predicted expression",
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_isodepth_triptych(
    dataset: DatasetBundle,
    result: TestResult,
    out_path: str | Path,
) -> Path | None:
    true_isodepth = result.artifacts.get("true_isodepth")
    lowest_isodepth = result.artifacts.get("lowest_isodepth")
    lowest_S = result.artifacts.get("lowest_S")
    highest_isodepth = result.artifacts.get("highest_isodepth")
    highest_S = result.artifacts.get("highest_S")
    if (
        true_isodepth is None
        or lowest_isodepth is None
        or lowest_S is None
        or highest_isodepth is None
        or highest_S is None
    ):
        perturbed_isodepth = result.artifacts.get("perturbed_isodepth")
        perturbed_S = result.artifacts.get("perturbed_S")
        if true_isodepth is None or perturbed_isodepth is None or perturbed_S is None:
            return None
        out_path = Path(out_path)
        return _save_perturbation_triptych(
            dataset,
            result,
            out_path,
            np.asarray(true_isodepth, dtype=np.float32),
            np.asarray(perturbed_isodepth, dtype=np.float32),
            np.asarray(perturbed_S, dtype=np.float32),
            np.asarray(perturbed_isodepth, dtype=np.float32),
            np.asarray(perturbed_S, dtype=np.float32),
        )

    out_path = Path(out_path)
    if result.method_name in {"comparison_perturbation_test", "perturbation_test"}:
        return _save_perturbation_triptych(
            dataset,
            result,
            out_path,
            np.asarray(true_isodepth, dtype=np.float32),
        )
    if result.method_name in {"comparison_subsampling_test", "subsampling_test"}:
        return _save_subsampling_triptych(
            dataset,
            result,
            out_path,
            np.asarray(true_isodepth, dtype=np.float32),
        )
    return _save_permutation_triptych(
        dataset,
        result,
        out_path,
        np.asarray(true_isodepth, dtype=np.float32),
        np.asarray(lowest_isodepth, dtype=np.float32),
        np.asarray(lowest_S, dtype=np.float32),
        np.asarray(highest_isodepth, dtype=np.float32),
        np.asarray(highest_S, dtype=np.float32),
    )


def save_metric_distribution_plot(result: TestResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    if result.method_name in {"comparison_perturbation_test", "perturbation_test"}:
        delta_summaries = result.artifacts.get("delta_summaries")
        if isinstance(delta_summaries, dict) and delta_summaries:
            rows = sorted(
                (summary for summary in delta_summaries.values() if isinstance(summary, dict)),
                key=lambda summary: float(summary["delta"]),
            )
            fig, axes = plt.subplots(len(rows), 1, figsize=(6, 4 * len(rows)), squeeze=False)
            for ax, summary in zip(axes[:, 0], rows):
                stat_perm = np.asarray(summary["null_distribution"], dtype=np.float64)
                stat_true = float(summary["score_mean"])
                p_value = float(summary["p_value"])
                delta = float(summary["delta"])
                ax.hist(stat_perm, bins=30, color="lightsteelblue", edgecolor="black")
                ax.axvline(stat_true, color="crimson", linestyle="--", label=f"Observed Mean: {stat_true:.4g}")
                ax.set_title(f"Delta = {delta:.4g}\np-value = {p_value:.4g}")
                ax.set_xlabel(result.metric)
                ax.set_ylabel("Count")
                ax.legend()
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return out_path
    if result.method_name in {"comparison_subsampling_test", "subsampling_test"}:
        fraction_summaries = result.artifacts.get("fraction_summaries")
        if isinstance(fraction_summaries, dict) and fraction_summaries:
            rows = sorted(
                (summary for summary in fraction_summaries.values() if isinstance(summary, dict)),
                key=lambda summary: float(summary["fraction"]),
            )
            fig, axes = plt.subplots(len(rows), 1, figsize=(6, 4 * len(rows)), squeeze=False)
            for ax, summary in zip(axes[:, 0], rows):
                stat_perm = np.asarray(summary["null_distribution"], dtype=np.float64)
                stat_true = float(summary["loss_mean"])
                p_value = float(summary["p_value"])
                fraction = float(summary["fraction"])
                ax.hist(stat_perm, bins=30, color="lightsteelblue", edgecolor="black")
                ax.axvline(stat_true, color="crimson", linestyle="--", label=f"Observed Mean: {stat_true:.4g}")
                ax.set_title(f"Fraction = {fraction:.3f}\np-value = {p_value:.4g}")
                ax.set_xlabel(result.metric)
                ax.set_ylabel("Count")
                ax.legend()
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return out_path

    stat_perm_arr = np.asarray(result.stat_perm, dtype=np.float64)
    stat_cov = result.artifacts.get("stat_covariate")
    p_cov = result.artifacts.get("p_value_covariate")
    has_cov_dual = stat_cov is not None and p_cov is not None

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.2))
    ax.hist(
        stat_perm_arr,
        bins=30,
        color="lightsteelblue",
        edgecolor="black",
        label="Null (permutations)",
    )
    if has_cov_dual:
        ax.axvline(
            result.stat_true,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Fully trained isodepth: {result.stat_true:.4g}",
        )
        ax.axvline(
            float(stat_cov),
            color="teal",
            linestyle="--",
            linewidth=1.5,
            label=f"Covariate (midline) decoder: {float(stat_cov):.4g}",
        )
        ax.set_title(
            "Null distribution\n"
            f"p (fully trained) = {result.p_value:.4g}  |  p (covariate) = {float(p_cov):.4g}"
        )
    else:
        ax.axvline(
            result.stat_true,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Observed: {result.stat_true:.4g}",
        )
        ax.set_title(f"Null distribution\np-value = {result.p_value:.4g}")
    ax.set_xlabel(result.metric)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_cross_validation_fold_isodepth_grid(
    dataset: DatasetBundle,
    fold_isodepths: list[np.ndarray],
    out_path: str | Path,
    *,
    fold_test_sizes: np.ndarray | None = None,
) -> Path | None:
    """Spatial grid of true-model isodepths, one panel per CV fold."""
    if not fold_isodepths:
        return None

    panels = [np.asarray(depth, dtype=np.float32).reshape(-1) for depth in fold_isodepths]
    n_folds = len(panels)
    if panels[0].shape[0] != dataset.n_cells:
        raise ValueError(
            f"fold isodepth length must match dataset.n_cells={dataset.n_cells}, "
            f"got {panels[0].shape[0]}"
        )

    n_cols = min(n_folds, 3)
    n_rows = int(np.ceil(n_folds / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows), squeeze=False)
    spatial = np.asarray(dataset.S, dtype=np.float32)

    stacked = np.concatenate(panels, axis=0)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))
    normalize_bounds = (vmin, vmax) if vmax > vmin else None

    for fold_index, depth in enumerate(panels):
        title = f"Fold {fold_index + 1}"
        if fold_test_sizes is not None and fold_index < fold_test_sizes.size:
            title += f" (held-out n={int(fold_test_sizes[fold_index])})"
        _plot_spatial_isodepth(
            axes.flat[fold_index],
            spatial,
            depth,
            title,
            normalize_bounds=normalize_bounds,
        )

    for axis in axes.flat[n_folds:]:
        axis.axis("off")

    fig.suptitle("True-model isodepth by cross-validation fold", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_cross_validation_per_fold_metric_distributions(
    result: TestResult,
    out_path: str | Path,
) -> Path | None:
    """Per-fold null histograms using fold-local permutation p-values."""
    if result.method_name != "cross_validation":
        return None

    per_fold_true = result.artifacts.get("per_fold_true_loss")
    per_fold_perm = result.artifacts.get("per_fold_perm_loss")
    per_fold_p = result.artifacts.get("per_fold_p_values")
    if per_fold_true is None or per_fold_perm is None:
        return None

    true_losses = np.asarray(per_fold_true, dtype=np.float64).reshape(-1)
    perm_losses = np.asarray(per_fold_perm, dtype=np.float64)
    if perm_losses.ndim != 2 or perm_losses.shape[0] != true_losses.size:
        raise ValueError(
            "per_fold_perm_loss must have shape (n_folds, n_perms) matching per_fold_true_loss"
        )

    p_values = (
        np.asarray(per_fold_p, dtype=np.float64).reshape(-1)
        if per_fold_p is not None
        else np.full(true_losses.size, np.nan, dtype=np.float64)
    )

    n_folds = true_losses.size
    fig, axes = plt.subplots(n_folds, 1, figsize=(6.4, 3.8 * n_folds), squeeze=False)
    for fold_index, ax in enumerate(axes[:, 0]):
        stat_perm = perm_losses[fold_index]
        stat_true = float(true_losses[fold_index])
        p_value = float(p_values[fold_index]) if np.isfinite(p_values[fold_index]) else float("nan")
        ax.hist(
            stat_perm,
            bins=30,
            color="lightsteelblue",
            edgecolor="black",
            label="Null (permutations)",
        )
        ax.axvline(
            stat_true,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Observed: {stat_true:.4g}",
        )
        ax.set_title(f"Fold {fold_index + 1}  |  p-value = {p_value:.4g}")
        ax.set_xlabel(result.metric)
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", framealpha=0.95, fontsize=8)

    fig.suptitle("Per-fold null distributions (unweighted permutation tests)", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_combined_celltype_metric_distribution(
    per_type_results: dict[str, dict],
    cell_type_names: list[str],
    out_path: str | Path,
    *,
    metric: str = "nll_gaussian_mse",
) -> Path:
    """Grid of null-distribution histograms, one panel per cell type."""
    n = len(cell_type_names)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.0 * nrows),
        squeeze=False,
    )
    for idx, type_name in enumerate(cell_type_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        data = per_type_results[type_name]
        stat_perm = np.asarray(data["stat_perm"], dtype=np.float64)
        stat_true = float(data["stat_true"])
        p_value = float(data["p_value"])
        n_cells = int(data["n_cells"])
        stat_cov = data.get("stat_covariate")
        p_cov = data.get("p_value_covariate")
        has_cov_dual = stat_cov is not None and p_cov is not None

        ax.hist(stat_perm, bins=30, color="lightsteelblue", edgecolor="black",
                label="Null (permutations)")
        if has_cov_dual:
            ax.axvline(stat_true, color="crimson", linestyle="--", linewidth=1.5,
                       label=f"Isodepth: {stat_true:.4g}")
            ax.axvline(float(stat_cov), color="teal", linestyle="--", linewidth=1.5,
                       label=f"Covariate: {float(stat_cov):.4g}")
            ax.set_title(
                f"{type_name} (n={n_cells})\n"
                f"p = {p_value:.4g}  |  p (cov) = {float(p_cov):.4g}",
                fontsize=10,
            )
        else:
            ax.axvline(stat_true, color="crimson", linestyle="--", linewidth=1.5,
                       label=f"Observed: {stat_true:.4g}")
            ax.set_title(f"{type_name} (n={n_cells})\np = {p_value:.4g}", fontsize=10)
        ax.set_xlabel(metric, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax.tick_params(labelsize=8)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Per-Cell-Type Null Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_combined_celltype_isodepth_grid(
    per_type_results: dict[str, dict],
    cell_type_names: list[str],
    out_path: str | Path,
    *,
    full_spatial: np.ndarray | None = None,
) -> Path:
    """Grid of true-data isodepth scatter plots, one panel per cell type."""
    if full_spatial is not None:
        tissue_limits = _spatial_axis_limits(full_spatial)
    else:
        all_coords = [
            np.asarray(
                per_type_results[name].get("S_original", per_type_results[name]["S"]),
                dtype=np.float32,
            )
            for name in cell_type_names
        ]
        tissue_limits = _spatial_axis_limits(np.vstack(all_coords))

    n = len(cell_type_names)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    for idx, type_name in enumerate(cell_type_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        data = per_type_results[type_name]
        S_plot = np.asarray(
            data.get("S_original", data["S"]),
            dtype=np.float32,
        )
        true_isodepth = np.asarray(data["true_isodepth"], dtype=np.float32)
        n_cells = int(data["n_cells"])
        p_value = float(data["p_value"])

        _plot_spatial_isodepth(
            ax,
            S_plot,
            true_isodepth,
            f"{type_name} (n={n_cells})\np = {p_value:.4g}",
            spatial_limits=tissue_limits,
        )

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Per-Cell-Type Learned Isodepths (True Data)", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_region_isodepth_timeline(
    stage_panels: list[dict[str, object]],
    out_path: str | Path,
    *,
    region_name: str,
    model_label: str | None = None,
) -> Path | None:
    """Spatial true-isodepth panels for one region across embryonic stages.

    Each entry in ``stage_panels`` must provide ``S``, ``true_isodepth``,
    ``stage_label``, ``n_cells``, and ``p_value``.
    """
    if not stage_panels:
        return None

    out_path = Path(out_path)
    n_panels = len(stage_panels)
    n_cols = min(n_panels, 4)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )

    for idx, panel in enumerate(stage_panels):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        S = np.asarray(panel["S"], dtype=np.float32)
        true_isodepth = np.asarray(panel["true_isodepth"], dtype=np.float32)
        stage_label = str(panel["stage_label"])
        n_cells = int(panel["n_cells"])
        p_value = float(panel["p_value"])
        title = f"{stage_label}\nn = {n_cells:,}\np = {p_value:.4g}"
        _plot_spatial_isodepth(
            ax,
            S,
            true_isodepth,
            title,
            spatial_limits=_spatial_axis_limits(S),
        )

    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    title_parts = [f"{region_name}: isodepth over time"]
    if model_label:
        title_parts.append(f"({model_label})")
    fig.suptitle(" ".join(title_parts), fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _recursive_celltype_grid_shape(
    per_type_recursive_results: dict[str, dict],
    cell_type_names: list[str],
) -> tuple[int, int]:
    ncols = max(1, len(cell_type_names))
    nrows = 1
    for type_name in cell_type_names:
        tested = per_type_recursive_results.get(type_name, {}).get("tested_gradients", [])
        nrows = max(nrows, len(tested))
    return nrows, ncols


def _recursive_celltype_panel_title(type_name: str, entry: dict) -> str:
    gradient_index = int(entry.get("gradient_index", 0))
    p_value = float(entry.get("p_value", np.nan))
    n_svgs = int(entry.get("n_svgs", 0))
    if bool(entry.get("passed_permutation", False)):
        status = f"{n_svgs} SVGs" if n_svgs > 0 else "no SVGs"
    else:
        status = "not significant"
    return f"{type_name}\nGradient {gradient_index} | p = {p_value:.4g} | {status}"


def save_recursive_celltype_isodepth_grid(
    per_type_recursive_results: dict[str, dict],
    cell_type_names: list[str],
    out_path: str | Path,
    *,
    full_spatial: np.ndarray | None = None,
) -> Path | None:
    """Grid of recursive cell-type isodepths.

    Columns are cell types/regions. Rows are tested recursive gradients,
    including the terminal non-significant gradient when present.
    """
    if not cell_type_names:
        return None

    if full_spatial is not None:
        tissue_limits = _spatial_axis_limits(full_spatial)
    else:
        all_coords: list[np.ndarray] = []
        for type_name in cell_type_names:
            for entry in per_type_recursive_results.get(type_name, {}).get("tested_gradients", []):
                S_plot = entry.get("S_plot")
                if S_plot is not None:
                    all_coords.append(np.asarray(S_plot, dtype=np.float32))
                    break
        tissue_limits = _spatial_axis_limits(np.vstack(all_coords)) if all_coords else None

    nrows, ncols = _recursive_celltype_grid_shape(per_type_recursive_results, cell_type_names)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.4 * nrows),
        squeeze=False,
    )

    for col, type_name in enumerate(cell_type_names):
        tested = per_type_recursive_results.get(type_name, {}).get("tested_gradients", [])
        for row in range(nrows):
            ax = axes[row][col]
            if row >= len(tested):
                ax.set_visible(False)
                continue
            entry = tested[row]
            S_plot = entry.get("S_plot")
            iso = entry.get("true_isodepth")
            if S_plot is None or iso is None:
                ax.set_visible(False)
                continue
            _plot_spatial_isodepth(
                ax,
                np.asarray(S_plot, dtype=np.float32),
                np.asarray(iso, dtype=np.float32),
                _recursive_celltype_panel_title(type_name, entry),
                spatial_limits=tissue_limits,
            )
            if col == 0:
                ax.set_ylabel(f"Test {row + 1}", fontsize=10)

    fig.suptitle("Recursive Cell-Type Isodepths", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_recursive_celltype_metric_distribution_grid(
    per_type_recursive_results: dict[str, dict],
    cell_type_names: list[str],
    out_path: str | Path,
    *,
    metric: str = "nll_gaussian_mse",
) -> Path | None:
    """Grid of recursive cell-type permutation null distributions."""
    if not cell_type_names:
        return None

    nrows, ncols = _recursive_celltype_grid_shape(per_type_recursive_results, cell_type_names)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for col, type_name in enumerate(cell_type_names):
        tested = per_type_recursive_results.get(type_name, {}).get("tested_gradients", [])
        for row in range(nrows):
            ax = axes[row][col]
            if row >= len(tested):
                ax.set_visible(False)
                continue
            entry = tested[row]
            stat_perm = np.asarray(entry.get("stat_perm", []), dtype=np.float64)
            stat_true = float(entry.get("stat_true", np.nan))
            if stat_perm.size == 0:
                ax.set_visible(False)
                continue
            ax.hist(
                stat_perm,
                bins=30,
                color="lightsteelblue",
                edgecolor="black",
                label="Null (permutations)",
            )
            ax.axvline(
                stat_true,
                color="crimson",
                linestyle="--",
                linewidth=1.5,
                label=f"Observed: {stat_true:.4g}",
            )
            ax.set_title(_recursive_celltype_panel_title(type_name, entry), fontsize=9)
            ax.set_xlabel(metric, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(f"Test {row + 1}\nCount", fontsize=9)

    fig.suptitle("Recursive Cell-Type Null Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_subset_fraction_pvalue_plot(result: TestResult, out_path: str | Path) -> Path | None:
    if result.method_name not in {"comparison_subsampling_test", "subsampling_test"}:
        return None

    fraction_summaries = result.artifacts.get("fraction_summaries")
    if not isinstance(fraction_summaries, dict) or not fraction_summaries:
        return None

    rows = sorted(
        (summary for summary in fraction_summaries.values() if isinstance(summary, dict)),
        key=lambda summary: float(summary["fraction"]),
    )
    fractions = np.asarray([float(summary["fraction"]) for summary in rows], dtype=np.float64)
    p_values = np.asarray([float(summary["p_value"]) for summary in rows], dtype=np.float64)

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(fractions, p_values, color="crimson", s=55)
    ax.plot(fractions, p_values, color="lightcoral", linewidth=1.0, alpha=0.8)
    ax.set_title("Subset Fraction vs p-value")
    ax.set_xlabel("Sampling Fraction")
    ax.set_ylabel("p-value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_perturbation_delta_pvalue_plot(result: TestResult, out_path: str | Path) -> Path | None:
    if result.method_name not in {"comparison_perturbation_test", "perturbation_test"}:
        return None

    delta_summaries = result.artifacts.get("delta_summaries")
    if not isinstance(delta_summaries, dict) or not delta_summaries:
        return None

    rows = sorted(
        (summary for summary in delta_summaries.values() if isinstance(summary, dict)),
        key=lambda summary: float(summary["delta"]),
    )
    deltas = np.asarray([float(summary["delta"]) for summary in rows], dtype=np.float64)
    p_values = np.asarray([float(summary["p_value"]) for summary in rows], dtype=np.float64)

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(deltas, p_values, color="crimson", s=55)
    ax.plot(deltas, p_values, color="lightcoral", linewidth=1.0, alpha=0.8)
    ax.set_title("Perturbation Delta vs p-value")
    ax.set_xlabel("Delta")
    ax.set_ylabel("p-value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _plot_spatial_expression_panel(
    ax,
    S: np.ndarray,
    z: np.ndarray,
    title: str,
    *,
    vmin: float,
    vmax: float,
    show_contours: bool = True,
    hide_zero_expression: bool = False,
    cmap: str = "Reds",
):
    """Scatter (and optional tricontour) for one gene on one spatial layout; values use vmin/vmax directly.

    Uses a sequential colormap (default ``Reds``): near-white at low expression, dark at high.
    Returns the scatter ``PathCollection`` when points were drawn, else ``None``.
    """
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    S = np.asarray(S, dtype=np.float32)
    if hide_zero_expression:
        nz = np.abs(z) > _EXPRESSION_ZERO_EPS
        S = S[nz]
        z = z[nz]
    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    scatter = None
    if S.shape[0] > 0:
        scatter = ax.scatter(
            S[:, 0],
            S[:, 1],
            c=z,
            cmap=cmap,
            norm=norm,
            s=_point_size(S),
            linewidths=0,
            alpha=0.9,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No plotted cells\n(all zero or masked)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="0.35",
        )
    if show_contours and S.shape[0] >= 3:
        try:
            triangulation = _masked_triangulation(S)
            levels = np.linspace(float(vmin), float(vmax), num=8)
            if levels.size > 1 and float(vmax) > float(vmin) + 1e-12:
                try:
                    from matplotlib import colormaps as _mpl_colormaps

                    cm = _mpl_colormaps[cmap]
                except (KeyError, TypeError, AttributeError):
                    cm = plt.get_cmap(cmap)
                contour_colors = cm(np.linspace(0.35, 0.98, levels.size))
                ax.tricontour(
                    triangulation,
                    z,
                    levels=levels,
                    colors=contour_colors,
                    linewidths=0.9,
                    alpha=0.9,
                )
        except (RuntimeError, ValueError):
            pass
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return scatter


def save_gene_spatial_contour_grid(
    spatial_views: list[tuple[str, np.ndarray]],
    expression: np.ndarray,
    out_path: str | Path,
    *,
    vmin: float,
    vmax: float,
    colorbar_label: str = "Expression",
    figure_title: str | None = None,
    show_contours: bool = True,
    hide_zero_expression: bool = False,
    cmap: str = "Reds",
) -> Path:
    """Save a grid (e.g. true + spatial nulls) of spatial expression maps; optional contour lines."""
    out_path = Path(out_path)
    z = np.asarray(expression, dtype=np.float32).reshape(-1)
    n_panels = len(spatial_views)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    if n_panels == 1:
        axes_arr = np.array([[axes]])
    elif nrows == 1:
        axes_arr = np.asarray(axes).reshape(1, -1)
    else:
        axes_arr = np.asarray(axes)

    for idx, (title, S_view) in enumerate(spatial_views):
        r, c = divmod(idx, ncols)
        ax = axes_arr[r, c]
        _plot_spatial_expression_panel(
            ax,
            S_view,
            z,
            title,
            vmin=vmin,
            vmax=vmax,
            show_contours=show_contours,
            hide_zero_expression=hide_zero_expression,
            cmap=cmap,
        )

    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes_arr[r, c].set_visible(False)

    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label=colorbar_label)
    if figure_title:
        fig.suptitle(figure_title, fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96] if figure_title else [0, 0, 0.9, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_multi_gene_spatial_expression_grid(
    S: np.ndarray,
    expression_matrix: np.ndarray,
    gene_labels: list[str],
    out_path: str | Path,
    *,
    show_contours: bool = False,
    hide_zero_expression: bool = False,
    figure_title: str | None = None,
) -> Path:
    """Save one figure with one panel per gene on the true spatial layout."""
    out_path = Path(out_path)
    S = np.asarray(S, dtype=np.float32)
    expr = np.asarray(expression_matrix, dtype=np.float32)
    if expr.ndim != 2:
        raise ValueError(f"expression_matrix must be 2D (n_cells, n_genes), got {expr.shape}")
    if S.shape[0] != expr.shape[0]:
        raise ValueError(
            "S and expression_matrix must agree on n_cells, "
            f"got {S.shape[0]} vs {expr.shape[0]}"
        )
    if expr.shape[1] != len(gene_labels):
        raise ValueError(
            "gene_labels length must match expression_matrix second dimension, "
            f"got {len(gene_labels)} vs {expr.shape[1]}"
        )

    # One shared scale for all genes so panels are comparable; color bar starts at 0.
    if hide_zero_expression:
        nz = np.abs(expr) > _EXPRESSION_ZERO_EPS
        if np.any(nz):
            vmax = float(np.max(expr[nz]))
        else:
            vmax = 1.0
    else:
        vmax = float(np.max(expr))
    vmin = 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-8

    n_genes = expr.shape[1]
    ncols = min(3, n_genes)
    nrows = int(np.ceil(n_genes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows), squeeze=False)
    for idx in range(n_genes):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        z = np.asarray(expr[:, idx], dtype=np.float32)
        cmap_i = expression_colormap_for_index(idx)
        sc = _plot_spatial_expression_panel(
            ax,
            S,
            z,
            gene_labels[idx],
            vmin=vmin,
            vmax=vmax,
            show_contours=show_contours,
            hide_zero_expression=hide_zero_expression,
            cmap=cmap_i,
        )
        if sc is not None:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="Expression")

    for idx in range(n_genes, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    if figure_title:
        fig.suptitle(figure_title, fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96] if figure_title else [0, 0, 0.9, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_spatial_pc_gradient_contour_panel(
    ax,
    S: np.ndarray,
    z: np.ndarray,
    title: str,
    *,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    """Filled gradient (``tricontourf``) plus line contours on spatial coordinates."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    S = np.asarray(S, dtype=np.float32)
    vmin, vmax = float(np.min(z)), float(np.max(z))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    if S.shape[0] >= 3:
        try:
            triangulation = _masked_triangulation(S)
            fill_levels = np.linspace(vmin, vmax, 28)
            line_levels = np.linspace(vmin, vmax, 11)
            cf = ax.tricontourf(
                triangulation,
                z,
                levels=fill_levels,
                cmap=cmap,
                norm=norm,
                alpha=0.92,
            )
            ax.tricontour(
                triangulation,
                z,
                levels=line_levels,
                colors="0.12",
                linewidths=0.65,
                alpha=0.9,
            )
            plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
        except (RuntimeError, ValueError):
            sc = ax.scatter(S[:, 0], S[:, 1], c=z, cmap=cmap, norm=norm, s=_point_size(S), linewidths=0, alpha=0.9)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    else:
        sc = ax.scatter(
            S[:, 0],
            S[:, 1],
            c=z,
            cmap=cmap,
            norm=norm,
            s=max(20.0, _point_size(S)),
            linewidths=0,
            alpha=0.9,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def save_spatial_principal_axes_plot(
    S: np.ndarray,
    out_path: str | Path,
    *,
    cmap: str = "viridis",
) -> Path:
    """PCA on spatial coordinates only: PC1/PC2 scores over (x,y) with gradient fill + contours.

    Uses centered coordinates and ``numpy.linalg.svd`` (scores ``S_centered @ V.T``).
    """
    out_path = Path(out_path)
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != 2:
        raise ValueError(f"Expected S with shape (N, 2), got {S.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    ax0, ax1 = axes[0], axes[1]

    if S.shape[0] < 2:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "Need at least 2 cells",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
        fig.suptitle("Principal axes of spatial coordinates", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    S0 = S - np.mean(S, axis=0, keepdims=True)
    _U, sing, Vt = np.linalg.svd(S0, full_matrices=False)
    scores = S0 @ Vt.T
    evr = (sing**2) / (np.sum(sing**2) + 1e-20)
    pc1 = np.asarray(scores[:, 0], dtype=np.float64)
    if scores.shape[1] > 1:
        pc2 = np.asarray(scores[:, 1], dtype=np.float64)
        title1 = f"PC2 ({100.0 * float(evr[1]):.1f}% variance)"
        colorbar2 = "PC2 score"
    else:
        pc2 = np.zeros_like(pc1)
        title1 = "PC2 (undefined in 1D arrangement)"
        colorbar2 = "—"

    title0 = f"PC1 ({100.0 * float(evr[0]):.1f}% variance)"

    S32 = np.asarray(S, dtype=np.float32)
    _plot_spatial_pc_gradient_contour_panel(
        ax0,
        S32,
        pc1,
        title0,
        colorbar_label="PC1 score",
        cmap=cmap,
    )
    _plot_spatial_pc_gradient_contour_panel(
        ax1,
        S32,
        pc2,
        title1,
        colorbar_label=colorbar2,
        cmap=cmap,
    )
    fig.suptitle("Principal coordinate axes (spatial layout only)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_spatial_pointcloud_kde_plot(
    S: np.ndarray,
    out_path: str | Path,
    *,
    cmap: str = "magma",
    grid_size: int = 220,
) -> Path:
    """Plot 2D KDE density of spatial coordinates with contour lines."""
    out_path = Path(out_path)
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != 2:
        raise ValueError(f"Expected S with shape (N, 2), got {S.shape}")
    if S.shape[0] < 3:
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.2))
        ax.scatter(S[:, 0], S[:, 1], c="0.2", s=max(20.0, _point_size(np.asarray(S, dtype=np.float32))))
        ax.set_title("Spatial point cloud density (KDE)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    x = S[:, 0]
    y = S[:, 1]
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xpad = max((xmax - xmin) * 0.03, 1e-6)
    ypad = max((ymax - ymin) * 0.03, 1e-6)
    gx, gy = np.mgrid[(xmin - xpad):(xmax + xpad):complex(grid_size), (ymin - ypad):(ymax + ypad):complex(grid_size)]
    positions = np.vstack([gx.ravel(), gy.ravel()])
    gz = kde(positions).reshape(gx.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.8))
    cf = ax.contourf(gx, gy, gz, levels=26, cmap=cmap, alpha=0.95)
    contour_levels = np.linspace(float(np.min(gz)), float(np.max(gz)), num=10)
    if contour_levels.size > 1 and float(np.max(gz)) > float(np.min(gz)) + 1e-20:
        ax.contour(gx, gy, gz, levels=contour_levels, colors="0.12", linewidths=0.75, alpha=0.9)
    ax.scatter(x, y, c="white", s=2.0, linewidths=0, alpha=0.18)
    ax.set_title("Spatial point cloud density (KDE)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="KDE density")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path



def save_spatial_binned_density_plot(
    S: np.ndarray,
    out_path: str | Path,
    *,
    n_bins: int = 60,
    cmap: str = "viridis",
) -> Path:
    """Plot spatial density by counting cells in 2D bins."""
    out_path = Path(out_path)
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != 2:
        raise ValueError(f"Expected S with shape (N, 2), got {S.shape}")
    if S.shape[0] == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.2))
        ax.text(0.5, 0.5, "No cells available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Spatial binned density")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    x = S[:, 0]
    y = S[:, 1]
    n_bins = max(8, int(n_bins))
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.8))
    mesh = ax.pcolormesh(x_edges, y_edges, counts.T, shading="auto", cmap=cmap)
    contour_levels = np.linspace(float(np.min(counts)), float(np.max(counts)), num=8)
    if contour_levels.size > 1 and float(np.max(counts)) > float(np.min(counts)) + 1e-12:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        gx, gy = np.meshgrid(x_centers, y_centers, indexing="xy")
        ax.contour(gx, gy, counts.T, levels=contour_levels, colors="0.12", linewidths=0.7, alpha=0.9)
    ax.set_title(f"Spatial binned density ({n_bins}x{n_bins})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="Cells per bin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _flatten_isodepth_for_axis(depth: np.ndarray) -> np.ndarray:
    """Use the first latent dimension as the 1D isodepth axis."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 1:
        return d
    if d.ndim == 2:
        return np.asarray(d[:, 0], dtype=np.float64)
    raise ValueError(f"Expected isodepth with ndim 1 or 2, got shape {d.shape}")


def _safe_filename_fragment(text: str, *, max_len: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))
    cleaned = cleaned.strip("_") or "gene"
    return cleaned[:max_len]


def _gene_display_name(meta: dict[str, Any], gene_idx: int) -> str:
    names = meta.get("var_names")
    if isinstance(names, list) and 0 <= gene_idx < len(names):
        return str(names[gene_idx])
    return f"gene_{gene_idx}"


def _expression_preprocessing_kind(meta: dict[str, Any] | None) -> str:
    """Semantic preprocessing tag for expression axis / colorbar labels."""
    if meta is None:
        return "unknown"
    if meta.get("q") is not None or meta.get("feature_space") == "poisson_low_rank_latent":
        return "poisson_latent"
    if meta.get("source") == "synthetic" and meta.get("expression_distribution") == "gaussian":
        return "z_scored"
    log1p = bool(meta.get("log1p", False))
    standardized = bool(meta.get("standardize_expression", False))
    if log1p and standardized:
        return "log_z_scored"
    if log1p:
        return "log"
    if standardized:
        return "z_scored"
    return "raw_counts"


def _expression_y_axis_label(meta: dict[str, Any]) -> str:
    """Describe ``dataset.A`` columns (model input / observed expression space)."""
    kind = _expression_preprocessing_kind(meta)
    if kind == "poisson_latent":
        return "Feature value (Poisson low-rank latent)"
    if kind == "log_z_scored":
        return "Expression (log₁p, z-scored)"
    if kind == "log":
        return "Expression (log₁p)"
    if kind == "z_scored":
        return "Expression (z-scored)"
    return "Expression"


# ---------------------------------------------------------------------------
# Gene expression vs isodepth / covariate — summary comparison plots
# ---------------------------------------------------------------------------


def _ftest_decoder_pvalues(
    A: np.ndarray,
    preds: np.ndarray,
    df_model: int,
) -> np.ndarray:
    """F-test p-value per gene: does the decoder explain significant variance?

    Parameters
    ----------
    A : (n_cells, G) observed expression (z-scored).
    preds : (n_cells, G) decoder-predicted expression.
    df_model : effective number of model parameters minus intercept (1 for a
        linear decoder, 2 for quadratic, 3 for cubic, etc.).

    Returns
    -------
    pvalues : (G,) array of raw F-test p-values.
    """
    A = np.asarray(A, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    n, G = A.shape
    df_error = n - df_model - 1
    if df_error <= 0:
        return np.ones(G, dtype=np.float64)

    y_mean = A.mean(axis=0)                         # (G,)
    sst = np.sum((A - y_mean) ** 2, axis=0)         # total SS
    sse = np.sum((A - preds) ** 2, axis=0)          # residual SS
    ssr = np.maximum(sst - sse, 0.0)                # explained SS (clamp ≥ 0)

    # Guard against degenerate genes (zero variance)
    safe_sse = np.where(sse > 0, sse, np.finfo(float).tiny)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        f_stat = (ssr / df_model) / (safe_sse / df_error)
    pvalues = 1.0 - _f_dist.cdf(f_stat, df_model, df_error)
    # Genes with zero total variance → p = 1
    pvalues = np.where(sst > 0, pvalues, 1.0)
    return pvalues.astype(np.float64)


def _bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values from an array of p-values."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    G = len(pvalues)
    if G == 0:
        return pvalues.copy()
    order = np.argsort(pvalues)
    ranks = np.arange(1, G + 1, dtype=np.float64)
    q_sorted = pvalues[order] * G / ranks
    # Enforce monotonicity from right to left
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(G, dtype=np.float64)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def _save_sig_genes_csv(
    out_path: Path,
    gene_names: list[str],
    pvalues: np.ndarray,
    qvalues: np.ndarray,
    q_threshold: float = 0.05,
) -> Path:
    """Write significant genes (q < q_threshold) to a CSV file.

    Columns: gene, p_value, q_value — sorted by p_value ascending.
    Always writes the file; if no genes pass the threshold the file contains
    only the header row.  Returns the path written.
    """
    sig_mask = qvalues < q_threshold
    rows = sorted(
        [
            (gene_names[g], float(pvalues[g]), float(qvalues[g]))
            for g in range(len(gene_names))
            if sig_mask[g]
        ],
        key=lambda r: r[1],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene", "p_value", "q_value"])
        writer.writerows(rows)
    return out_path


def compute_isodepth_sig_genes(
    A: np.ndarray,
    gene_names: list[str],
    pred_isodepth: np.ndarray | None,
    decoder_df: int,
    coord: np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict:
    """Compute significant SVGs for an isodepth gradient via F-test + BH correction.

    Parameters
    ----------
    A : (n_cells, G) observed expression matrix.
    gene_names : length-G list of gene name strings.
    pred_isodepth : (n_cells, G) decoder predictions, or None. When None,
        ``coord`` must be provided and a polynomial of degree ``decoder_df``
        is fit per gene to approximate the decoder.
    decoder_df : degrees of freedom for the decoder (1=linear, 2=quadratic).
    coord : (n_cells,) isodepth values; required only when pred_isodepth is None.
    alpha : BH q-value threshold; genes with q < alpha are significant.

    Returns
    -------
    dict with keys:
        ``sig_indices``   : np.ndarray of significant gene indices into A columns
        ``sig_names``     : list[str] of significant gene names
        ``pvalues``       : (G,) raw F-test p-values for all genes
        ``qvalues``       : (G,) BH q-values for all genes
    """
    A = np.asarray(A, dtype=np.float64)
    G = A.shape[1]

    if pred_isodepth is not None:
        fitted = np.asarray(pred_isodepth, dtype=np.float64)
    else:
        if coord is None:
            raise ValueError("coord must be provided when pred_isodepth is None")
        c = np.asarray(coord, dtype=np.float64).reshape(-1)
        fitted = np.stack(
            [np.poly1d(np.polyfit(c, A[:, g], decoder_df))(c) for g in range(G)],
            axis=1,
        )

    pvals = _ftest_decoder_pvalues(A, fitted, df_model=decoder_df)
    qvals = _bh_qvalues(pvals)
    sig_mask = qvals < alpha
    sig_indices = np.flatnonzero(sig_mask)
    sig_names = [gene_names[i] for i in sig_indices]
    return {
        "sig_indices": sig_indices,
        "sig_names": sig_names,
        "pvalues": pvals,
        "qvalues": qvals,
    }


def _gene_spearman_rhos(A: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """|Spearman ρ| between every gene column of A and coord. Shape (G,)."""
    A = np.asarray(A, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    G = A.shape[1]
    rhos = np.zeros(G, dtype=np.float64)
    for g in range(G):
        r, _ = spearmanr(A[:, g], coord)
        rhos[g] = abs(float(r)) if np.isfinite(r) else 0.0
    return rhos


def _top_genes_by_abs_rho(rhos: np.ndarray, n_top: int) -> np.ndarray:
    """Indices of top n_top genes by |ρ|, descending."""
    return np.argsort(-np.asarray(rhos, dtype=np.float64))[: min(int(n_top), int(rhos.size))]


def _decoder_fitted_values(
    A: np.ndarray,
    coord: np.ndarray,
    decoder_preds: np.ndarray | None,
    decoder_df: int | None,
) -> np.ndarray | None:
    """Return fitted expression values for residual summaries, if available."""
    if decoder_preds is not None:
        fitted = np.asarray(decoder_preds, dtype=np.float64)
        if fitted.shape == np.asarray(A).shape:
            return fitted
        return None
    if decoder_df is None:
        return None
    A = np.asarray(A, dtype=np.float64)
    c = np.asarray(coord, dtype=np.float64).reshape(-1)
    return np.stack(
        [np.poly1d(np.polyfit(c, A[:, g], int(decoder_df)))(c) for g in range(A.shape[1])],
        axis=1,
    )


def _rss_per_gene(A: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    return np.sum((A - fitted) ** 2, axis=0)


def _residual_ratio_hist_bins(values: np.ndarray, n_bins: int = 40) -> np.ndarray:
    """Equal-width histogram bins symmetric about 1.0."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n_bins = max(2, int(n_bins))
    if vals.size == 0:
        return np.linspace(0.5, 1.5, n_bins + 1)

    vmin = float(vals.min())
    vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin - 1.0), 0.05)
        return np.linspace(1.0 - delta, 1.0 + delta, n_bins + 1)

    half_extent = max(1.0 - vmin, vmax - 1.0, 1e-6)
    return np.linspace(1.0 - half_extent, 1.0 + half_extent, n_bins + 1)


def _save_correlation_distribution_plot(
    out_path: Path,
    series: list[tuple[str, np.ndarray, Any]],
) -> Path | None:
    """Save overlaid histograms of per-gene |Spearman ρ| values."""
    valid_series: list[tuple[str, np.ndarray, Any]] = []
    for label, values, color in series:
        vals = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
        vals = vals[np.isfinite(vals)]
        if vals.size:
            valid_series.append((label, vals, color))
    if not valid_series:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))
    bins = np.linspace(0.0, 1.0, 41)
    for label, vals, color in valid_series:
        ax.hist(
            vals,
            bins=bins,
            alpha=0.45,
            label=f"{label} (n={vals.size})",
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(float(np.median(vals)), color=color, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Per-gene |Spearman correlation|")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of gene-coordinate correlations")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_residual_ratio_outputs(
    *,
    csv_path: Path,
    plot_path: Path,
    gene_names: list[str],
    A: np.ndarray,
    fitted_coord: np.ndarray | None,
    fitted_covariate: np.ndarray | None,
    rhos_coord: np.ndarray,
    rhos_covariate: np.ndarray,
    coord_label: str,
    covariate_label: str,
) -> tuple[Path | None, Path | None]:
    """Save RSS_covariate/RSS_fitted-coordinate rankings and a ratio histogram."""
    if fitted_coord is None or fitted_covariate is None:
        return None, None

    rss_coord = _rss_per_gene(A, fitted_coord)
    rss_cov = _rss_per_gene(A, fitted_covariate)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rss_cov / rss_coord
    ratio = np.asarray(ratio, dtype=np.float64)
    order = np.argsort(-np.nan_to_num(ratio, nan=-np.inf, posinf=np.inf, neginf=-np.inf))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank",
            "gene",
            "residual_covariate",
            "residual_fitted_coordinate",
            "residual_cov_over_fitted",
            "spearman_covariate",
            "spearman_fitted_coordinate",
            "abs_spearman_covariate",
            "abs_spearman_fitted_coordinate",
            "covariate_label",
            "fitted_coordinate_label",
        ])
        for rank, g in enumerate(order, start=1):
            writer.writerow([
                rank,
                gene_names[int(g)],
                float(rss_cov[int(g)]),
                float(rss_coord[int(g)]),
                float(ratio[int(g)]),
                float(rhos_covariate[int(g)]),
                float(rhos_coord[int(g)]),
                float(abs(rhos_covariate[int(g)])),
                float(abs(rhos_coord[int(g)])),
                covariate_label,
                coord_label,
            ])

    finite_ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    plot_out: Path | None = None
    if finite_ratio.size:
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))
        ax.hist(
            finite_ratio,
            bins=_residual_ratio_hist_bins(finite_ratio),
            color="mediumpurple",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(1.0, color="0.15", linestyle="--", linewidth=1.2)
        ax.set_xlabel(f"RSS {covariate_label} / RSS {coord_label}")
        ax.set_ylabel("Genes")
        ax.set_title("Residual ratio distribution")
        ax.grid(alpha=0.18, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_out = plot_path

    return csv_path, plot_out


def save_combined_celltype_residual_ratio_outputs(
    per_type_results: dict[str, dict],
    cell_type_names: list[str],
    gene_names: list[str],
    csv_path: str | Path,
    plot_path: str | Path,
    *,
    coord_label: str = "Isodepth",
    covariate_label: str = "Covariate",
) -> tuple[Path | None, Path | None]:
    """Combine per-cell-type residuals into one gene-level piecewise RSS ratio.

    Separate cell-type mode fits one coordinate/decoder per cell type.  For a
    global gene-level residual comparison, treat those fits as one piecewise
    function: for each gene, sum RSS across all cell-type subsets before taking
    ``RSS_covariate / RSS_fitted_coordinate``.
    """
    rss_coord: np.ndarray | None = None
    rss_cov: np.ndarray | None = None
    used_cell_types: list[str] = []
    n_cells_total = 0

    for type_name in cell_type_names:
        type_data = per_type_results.get(type_name, {})
        A = type_data.get("A")
        fitted_coord = type_data.get("pred_true")
        fitted_cov = type_data.get("pred_true_covariate")
        if A is None or fitted_coord is None or fitted_cov is None:
            continue

        A_arr = np.asarray(A, dtype=np.float64)
        coord_arr = np.asarray(fitted_coord, dtype=np.float64)
        cov_arr = np.asarray(fitted_cov, dtype=np.float64)
        if A_arr.shape != coord_arr.shape or A_arr.shape != cov_arr.shape:
            continue

        type_rss_coord = _rss_per_gene(A_arr, coord_arr)
        type_rss_cov = _rss_per_gene(A_arr, cov_arr)
        rss_coord = type_rss_coord if rss_coord is None else rss_coord + type_rss_coord
        rss_cov = type_rss_cov if rss_cov is None else rss_cov + type_rss_cov
        used_cell_types.append(str(type_name))
        n_cells_total += int(A_arr.shape[0])

    if rss_coord is None or rss_cov is None or not used_cell_types:
        return None, None

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(rss_cov / rss_coord, dtype=np.float64)
    order = np.argsort(-np.nan_to_num(ratio, nan=-np.inf, posinf=np.inf, neginf=-np.inf))

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank",
            "gene",
            "residual_covariate_sum_across_cell_types",
            "residual_fitted_coordinate_sum_across_cell_types",
            "residual_cov_over_fitted",
            "n_cells_total",
            "n_cell_types",
            "cell_types",
            "covariate_label",
            "fitted_coordinate_label",
        ])
        cell_types_joined = ";".join(used_cell_types)
        for rank, g in enumerate(order, start=1):
            writer.writerow([
                rank,
                gene_names[int(g)] if int(g) < len(gene_names) else f"gene_{int(g)}",
                float(rss_cov[int(g)]),
                float(rss_coord[int(g)]),
                float(ratio[int(g)]),
                n_cells_total,
                len(used_cell_types),
                cell_types_joined,
                covariate_label,
                coord_label,
            ])

    finite_ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    plot_out: Path | None = None
    if finite_ratio.size:
        plot_path = Path(plot_path)
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))
        ax.hist(
            finite_ratio,
            bins=_residual_ratio_hist_bins(finite_ratio),
            color="mediumpurple",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(1.0, color="0.15", linestyle="--", linewidth=1.2)
        ax.set_xlabel(f"RSS {covariate_label} / RSS piecewise {coord_label}")
        ax.set_ylabel("Genes")
        ax.set_title("Piecewise cell-type residual ratio distribution")
        ax.grid(alpha=0.18, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_out = plot_path

    return csv_path, plot_out


def _quantile_bin_assignments(
    coord: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, int]:
    """Assign each coordinate to a quantile bin; return bin index and bin count."""
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    bin_edges = np.unique(np.quantile(coord, np.linspace(0.0, 1.0, n_bins + 1)))
    actual_n = max(len(bin_edges) - 1, 1)
    if len(bin_edges) < 2:
        bin_edges = np.array([float(coord.min()), float(coord.max())])
    bin_idx = np.clip(np.digitize(coord, bin_edges) - 1, 0, actual_n - 1)
    return bin_idx, actual_n


def _bin_mean_series(
    coord: np.ndarray,
    values: np.ndarray,
    bin_idx: np.ndarray,
    actual_n: int,
    *,
    min_bin_cells: int,
) -> tuple[list[float], list[float]]:
    """Per-bin mean coordinate and mean value for bins with enough cells."""
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centers: list[float] = []
    means: list[float] = []
    for b in range(actual_n):
        mask = bin_idx == b
        if int(mask.sum()) < min_bin_cells:
            continue
        centers.append(float(np.mean(coord[mask])))
        means.append(float(np.mean(values[mask])))
    return centers, means


def _plot_trend_curve_from_bin_means(
    ax,
    centers: list[float],
    means: list[float],
    coord: np.ndarray,
    *,
    fallback_x: np.ndarray | None = None,
    fallback_y: np.ndarray | None = None,
) -> None:
    """Draw a smooth trend line through quantile bin means on ``coord``'s range."""
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    if len(centers) >= 4:
        deg = min(3, len(centers) - 1)
        poly = np.poly1d(np.polyfit(centers, means, deg))
        fit_x = np.linspace(float(coord.min()), float(coord.max()), 300)
        ax.plot(fit_x, poly(fit_x), color="0.20", linewidth=1.8, alpha=0.9, zorder=4)
    elif len(centers) >= 2:
        order = np.argsort(centers)
        ax.plot(
            np.asarray(centers, dtype=np.float64)[order],
            np.asarray(means, dtype=np.float64)[order],
            color="0.20", linewidth=1.8, alpha=0.9, zorder=4,
        )
    elif fallback_x is not None and fallback_y is not None:
        slope, intercept, *_ = linregress(fallback_x, fallback_y)
        fit_x = np.array([float(coord.min()), float(coord.max())])
        ax.plot(fit_x, slope * fit_x + intercept, color="0.20", linewidth=1.8, alpha=0.9, zorder=4)


def _plot_gene_binned_vs_coord(
    ax,
    expr_col: np.ndarray,
    coord: np.ndarray,
    gene_name: str,
    xlabel: str,
    *,
    n_bins: int = 20,
    min_bin_cells: int = 3,
    rho: float | None = None,
    color: Any = "steelblue",
    max_cells_scatter: int = 2000,
    decoder_preds: np.ndarray | None = None,
    show_background_cells: bool = False,
    expression_y_label: str = "Expression",
) -> None:
    """Per-bin mean dots + trend curve, with optional per-cell background scatter.

    When ``decoder_preds`` is provided (model decoder predictions for this gene,
    shape ``(n_cells,)``), the fit curve is the actual decoder output: cells are
    sorted by coordinate and the predictions are smoothed with a uniform running
    mean (~10% window).  This shows NN non-linearities faithfully rather than
    collapsing them to a polynomial.
    When ``decoder_preds`` is None, a degree-3 polynomial is fit through the
    per-bin expression means as a non-linear trend approximation.

    Set ``show_background_cells=True`` to overlay a subsampled per-cell scatter
    (useful for visualising per-cell noise alongside the bin means).
    """
    expr_col = np.asarray(expr_col, dtype=np.float64).reshape(-1)
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    n_cells = len(coord)

    # --- optional per-cell scatter (subsampled so large datasets don't slow down) ---
    if show_background_cells:
        if n_cells <= max_cells_scatter:
            sc_idx = np.arange(n_cells)
        else:
            rng = np.random.default_rng(0)
            sc_idx = rng.choice(n_cells, size=max_cells_scatter, replace=False)
        ax.scatter(
            coord[sc_idx], expr_col[sc_idx],
            s=2, alpha=0.12, c="0.65", linewidths=0, zorder=1,
            rasterized=True,
        )

    bin_idx, actual_n = _quantile_bin_assignments(coord, n_bins)
    centers, means = _bin_mean_series(
        coord, expr_col, bin_idx, actual_n, min_bin_cells=min_bin_cells
    )

    if centers:
        ax.scatter(
            centers, means,
            s=40, color=color, alpha=0.95, linewidths=0.4,
            edgecolors="white", zorder=3,
        )

    # --- fit curve: actual NN predictions (smoothed) or poly through expression bin means ---
    if decoder_preds is not None:
        # Show the actual decoder curve: sort cells by coordinate, then apply a
        # uniform running mean whose window is ~10% of cells (min 5, max 200).
        # This preserves NN non-linearities that a degree-3 polynomial would destroy.
        dp = np.asarray(decoder_preds, dtype=np.float64).reshape(-1)
        sort_order = np.argsort(coord)
        dp_sorted = dp[sort_order]
        coord_sorted = coord[sort_order]
        window = max(5, min(200, int(round(len(dp_sorted) * 0.10))))
        dp_smooth = uniform_filter1d(dp_sorted, size=window, mode="nearest")
        ax.plot(
            coord_sorted, dp_smooth,
            color="0.20", linewidth=1.8, alpha=0.9, zorder=4,
        )
    else:
        _plot_trend_curve_from_bin_means(
            ax, centers, means, coord, fallback_x=coord, fallback_y=expr_col
        )

    title = gene_name
    if rho is not None:
        title += f"\n|Spearman r| = {abs(float(rho)):.3f}"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(expression_y_label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.4)


def save_gene_expression_vs_isodepth_plot(
    dataset: "DatasetBundle",
    isodepth: np.ndarray,
    out_path: str | Path,
    *,
    n_top_genes: int = 5,
    n_bins: int = 20,
    min_bin_cells: int = 3,
    coord_label: str = "Isodepth",
    decoder_preds: np.ndarray | None = None,
    decoder_df: int | None = None,
    q_threshold: float = 0.05,
    gene_indices: np.ndarray | list[int] | None = None,
    figure_title: str | None = None,
    pvalues: np.ndarray | None = None,
    qvalues: np.ndarray | None = None,
    spatial_S: np.ndarray | None = None,
) -> Path:
    """Top n_top_genes genes (by |Spearman ρ| with isodepth) as binned mean
    expression vs isodepth.

    ``decoder_preds`` (n_cells, G): when provided the fit curve uses the binned
    mean decoder output smoothed over isodepth.  ``decoder_df`` triggers an
    F-test across all G genes (parametric decoders only); significant genes
    (BH q < q_threshold) are saved to
    ``<stem>_isodepth_sig_genes.csv`` beside the PNG, and a companion spatial
    expression scatter is saved to ``<stem>_svg_spatial_expression.png``.

    ``spatial_S``: optional (N, 2) spatial coordinates used for the companion
    scatter.  Defaults to ``dataset.S`` when not provided.
    """
    out_path = Path(out_path)
    A = np.asarray(dataset.A, dtype=np.float64)
    coord = np.asarray(isodepth, dtype=np.float64).reshape(-1)
    G = A.shape[1]

    var_names = dataset.meta.get("var_names")
    gene_names: list[str] = (
        [str(var_names[i]) for i in range(G)] if var_names else [f"gene_{i}" for i in range(G)]
    )

    rhos = _gene_spearman_rhos(A, coord)
    if gene_indices is None:
        top_idx = _top_genes_by_abs_rho(rhos, n_top_genes)
        title = f"Top {{n_top}} genes by |Spearman r| with {coord_label}"
    else:
        top_idx = np.asarray(gene_indices, dtype=np.intp)[: int(n_top_genes)]
        title = figure_title or f"Selected genes vs {coord_label}"
    n_top = len(top_idx)

    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(n_top, 1)))
    fig, axes = plt.subplots(1, n_top, figsize=(4.0 * n_top, 3.6), squeeze=False)
    pv_arr = np.asarray(pvalues, dtype=np.float64) if pvalues is not None else None
    qv_arr = np.asarray(qvalues, dtype=np.float64) if qvalues is not None else None
    expression_y_label = _expression_y_axis_label(dataset.meta)

    for col, gene_idx in enumerate(top_idx):
        dp = (
            np.asarray(decoder_preds, dtype=np.float64)[:, gene_idx]
            if decoder_preds is not None else None
        )
        show_rho = None if (pv_arr is not None and qv_arr is not None) else float(rhos[gene_idx])
        _plot_gene_binned_vs_coord(
            axes[0, col], A[:, gene_idx], coord,
            gene_names[gene_idx], coord_label,
            n_bins=n_bins, min_bin_cells=min_bin_cells,
            rho=show_rho, color=colors[col],
            decoder_preds=dp,
            expression_y_label=expression_y_label,
        )
        if pv_arr is not None and qv_arr is not None:
            axes[0, col].set_title(
                f"{gene_names[gene_idx]}\np={pv_arr[gene_idx]:.2e}  q={qv_arr[gene_idx]:.2e}",
                fontsize=9,
            )

    fig.suptitle(title.format(n_top=n_top), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    stem = out_path.parent / out_path.stem
    _save_correlation_distribution_plot(
        Path(f"{stem}_correlation_distribution.png"),
        [(coord_label, rhos, "steelblue")],
    )

    S_plot = (
        np.asarray(spatial_S, dtype=np.float32)
        if spatial_S is not None
        else np.asarray(dataset.S, dtype=np.float32)
    )
    spatial_out = out_path.parent / (out_path.stem + "_svg_spatial_expression.png")

    # --- F-test significance CSV + companion spatial expression scatter ---
    if decoder_df is not None:
        svg_result = compute_isodepth_sig_genes(
            A, gene_names, decoder_preds, decoder_df,
            coord=coord, alpha=q_threshold,
        )
        csv_path = out_path.parent / (out_path.stem + "_isodepth_sig_genes.csv")
        _save_sig_genes_csv(
            csv_path, gene_names,
            svg_result["pvalues"], svg_result["qvalues"],
            q_threshold=q_threshold,
        )
        if svg_result["sig_indices"].size > 0:
            try:
                save_svg_spatial_expression_plots(
                    S_plot, A, gene_names,
                    svg_result["sig_indices"],
                    spatial_out,
                    pvalues=svg_result["pvalues"],
                    qvalues=svg_result["qvalues"],
                    expression_meta=dataset.meta,
                    suptitle=f"Top SVG Spatial Expression — {coord_label}",
                )
            except Exception:
                pass
    else:
        stale_csv = out_path.parent / (out_path.stem + "_isodepth_sig_genes.csv")
        if stale_csv.exists():
            stale_csv.unlink()
        # nn decoder: no F-test, use the same top-|rho| genes shown in the binned plot
        if top_idx.size > 0:
            try:
                save_svg_spatial_expression_plots(
                    S_plot, A, gene_names,
                    top_idx,
                    spatial_out,
                    rhos=rhos,
                    expression_meta=dataset.meta,
                    suptitle=f"Top Gene Spatial Expression — {coord_label}",
                )
            except Exception:
                pass

    return out_path


def save_gene_expression_vs_coordinates_comparison(
    dataset: "DatasetBundle",
    isodepth: np.ndarray,
    covariate_isodepth: np.ndarray,
    out_path: str | Path,
    *,
    n_top_genes: int = 5,
    n_bins: int = 20,
    min_bin_cells: int = 3,
    isodepth_label: str = "Isodepth",
    covariate_label: str = "Covariate",
    pred_isodepth: np.ndarray | None = None,
    pred_covariate: np.ndarray | None = None,
    decoder_df: int | None = None,
    q_threshold: float = 0.05,
    spatial_S: np.ndarray | None = None,
) -> Path:
    """4-row × n_top_genes comparison grid.

    Row 0: top isodepth genes vs isodepth
    Row 1: top isodepth genes vs covariate  (ρ shown vs covariate)
    Row 2: top covariate genes vs covariate
    Row 3: top covariate genes vs isodepth  (ρ shown vs isodepth)

    When ``pred_isodepth`` / ``pred_covariate`` are provided (model decoder
    predictions, shape ``(n_cells, G)``), the fit curve shows the actual decoder
    output: cells sorted by coordinate, predictions smoothed with a uniform
    running mean (~10% window) to remove per-cell noise while preserving NN
    non-linearities.  Otherwise a degree-3 polynomial fit through the per-bin
    expression means is used as a non-linear trend approximation.

    When ``decoder_df`` is set (e.g. 1 for a linear decoder, 2 for quadratic)
    an F-test is run across **all** genes for each decoder model and significant
    genes (Benjamini–Hochberg q < 0.05) are written to CSVs beside the PNG:
    ``<stem>_isodepth_sig_genes.csv`` and/or ``<stem>_covariate_sig_genes.csv``.
    When ``pred_isodepth``/``pred_covariate`` is None but ``decoder_df`` is set,
    a fresh polynomial of degree ``decoder_df`` is fitted to the raw per-cell
    data for every gene so that the F-test covers all genes, not just the
    displayed top-k.

    Scatter dots are per-bin means with quantile-based (equal-count) bins.
    """
    out_path = Path(out_path)
    A = np.asarray(dataset.A, dtype=np.float64)
    iso = np.asarray(isodepth, dtype=np.float64).reshape(-1)
    cov = np.asarray(covariate_isodepth, dtype=np.float64).reshape(-1)
    G = A.shape[1]

    var_names = dataset.meta.get("var_names")
    gene_names: list[str] = (
        [str(var_names[i]) for i in range(G)] if var_names else [f"gene_{i}" for i in range(G)]
    )

    rhos_iso = _gene_spearman_rhos(A, iso)
    rhos_cov = _gene_spearman_rhos(A, cov)
    top_iso = _top_genes_by_abs_rho(rhos_iso, n_top_genes)
    top_cov = _top_genes_by_abs_rho(rhos_cov, n_top_genes)
    n_top = max(len(top_iso), len(top_cov))

    iso_colors = plt.cm.Blues(np.linspace(0.45, 0.85, max(len(top_iso), 1)))
    cov_colors = plt.cm.Greens(np.linspace(0.45, 0.85, max(len(top_cov), 1)))

    # (gene_indices, rhos_to_display, coordinate, x_label, colors, pred_array)
    # pred_array shape: (n_cells, G) or None
    row_specs = [
        (top_iso, rhos_iso, iso, isodepth_label, iso_colors, pred_isodepth),
        (top_iso, rhos_cov, cov, covariate_label, iso_colors, pred_covariate),
        (top_cov, rhos_cov, cov, covariate_label, cov_colors, pred_covariate),
        (top_cov, rhos_iso, iso, isodepth_label, cov_colors, pred_isodepth),
    ]
    row_titles = [
        f"Top {len(top_iso)} isodepth genes — vs {isodepth_label}",
        f"Top {len(top_iso)} isodepth genes — vs {covariate_label}",
        f"Top {len(top_cov)} covariate genes — vs {covariate_label}",
        f"Top {len(top_cov)} covariate genes — vs {isodepth_label}",
    ]

    fig, axes = plt.subplots(4, n_top, figsize=(4.0 * n_top, 3.6 * 4), squeeze=False)
    expression_y_label = _expression_y_axis_label(dataset.meta)

    for row_idx, (gene_indices, rhos_for_row, coord, xlabel, colors, preds) in enumerate(row_specs):
        for col, gene_idx in enumerate(gene_indices):
            dp = (
                np.asarray(preds, dtype=np.float64)[:, gene_idx]
                if preds is not None else None
            )
            _plot_gene_binned_vs_coord(
                axes[row_idx, col], A[:, gene_idx], coord,
                gene_names[gene_idx], xlabel,
                n_bins=n_bins, min_bin_cells=min_bin_cells,
                rho=float(rhos_for_row[gene_idx]), color=colors[col],
                decoder_preds=dp,
                expression_y_label=expression_y_label,
            )
        for col in range(len(gene_indices), n_top):
            axes[row_idx, col].set_visible(False)
        axes[row_idx, 0].set_ylabel(
            f"Mean expression\n({row_titles[row_idx]})", fontsize=8
        )

    fig.suptitle(
        f"Gene expression vs learned coordinates\n({isodepth_label} | {covariate_label})",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    stem = out_path.parent / out_path.stem
    _save_correlation_distribution_plot(
        Path(f"{stem}_correlation_distribution.png"),
        [
            (isodepth_label, rhos_iso, "steelblue"),
            (covariate_label, rhos_cov, "seagreen"),
        ],
    )
    fitted_iso = _decoder_fitted_values(A, iso, pred_isodepth, decoder_df)
    fitted_cov = _decoder_fitted_values(A, cov, pred_covariate, decoder_df)
    _save_residual_ratio_outputs(
        csv_path=Path(f"{stem}_residual_ratio_rankings.csv"),
        plot_path=Path(f"{stem}_residual_ratio_distribution.png"),
        gene_names=gene_names,
        A=A,
        fitted_coord=fitted_iso,
        fitted_covariate=fitted_cov,
        rhos_coord=rhos_iso,
        rhos_covariate=rhos_cov,
        coord_label=isodepth_label,
        covariate_label=covariate_label,
    )

    S_plot = (
        np.asarray(spatial_S, dtype=np.float32)
        if spatial_S is not None
        else np.asarray(dataset.S, dtype=np.float32)
    )

    # --- F-test significance CSVs (parametric decoders only) ---
    if decoder_df is not None:
        iso_svg = compute_isodepth_sig_genes(
            A, gene_names, pred_isodepth, decoder_df, coord=iso, alpha=q_threshold,
        )
        cov_svg = compute_isodepth_sig_genes(
            A, gene_names, pred_covariate, decoder_df, coord=cov, alpha=q_threshold,
        )
        _save_sig_genes_csv(
            Path(f"{stem}_isodepth_sig_genes.csv"), gene_names,
            iso_svg["pvalues"], iso_svg["qvalues"], q_threshold=q_threshold,
        )
        _save_sig_genes_csv(
            Path(f"{stem}_covariate_sig_genes.csv"), gene_names,
            cov_svg["pvalues"], cov_svg["qvalues"], q_threshold=q_threshold,
        )
        # rows for the combined spatial plot: (gene_indices, pvalues, qvalues, rhos, label)
        spatial_row_specs = [
            (iso_svg["sig_indices"], iso_svg["pvalues"], iso_svg["qvalues"], None,
             f"Top {isodepth_label} SVGs  (q\u2009<\u2009{q_threshold})"),
            (cov_svg["sig_indices"], cov_svg["pvalues"], cov_svg["qvalues"], None,
             f"Top {covariate_label} SVGs  (q\u2009<\u2009{q_threshold})"),
        ]
    else:
        # nn decoder: top-|rho| genes already computed for the binned plot
        spatial_row_specs = [
            (top_iso, None, None, rhos_iso, f"Top {isodepth_label} genes  (by |Spearman r|)"),
            (top_cov, None, None, rhos_cov, f"Top {covariate_label} genes  (by |Spearman r|)"),
        ]

    # --- Combined 2 × n_top spatial expression grid ---
    # Mirrors the layout of the 4-row comparison binned-plot: rows = gene sets,
    # columns = individual genes.  Each panel shows cells at (x, y) coloured by
    # gene expression, so the spatial pattern is immediately comparable across
    # the two coordinate systems.
    try:
        n_cols = int(n_top_genes)
        n_rows = len(spatial_row_specs)
        ps = _point_size(S_plot)
        expr_label = _expression_label(dataset.meta)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 4.5 * n_rows),
            squeeze=False,
        )

        for row_idx, (g_indices, pvals, qvals, rho_arr, row_label) in enumerate(spatial_row_specs):
            g_indices = np.asarray(g_indices, dtype=np.intp)
            pv_arr = np.asarray(pvals, dtype=np.float64) if pvals is not None else None
            qv_arr = np.asarray(qvals, dtype=np.float64) if qvals is not None else None
            rh_arr = np.asarray(rho_arr, dtype=np.float64) if rho_arr is not None else None

            if pv_arr is not None and g_indices.size > 0:
                order = np.argsort(pv_arr[g_indices])
                top = g_indices[order[:n_cols]]
            else:
                top = g_indices[:n_cols]

            for col, gene_idx in enumerate(top):
                ax = axes[row_idx, col]
                expr = A[:, int(gene_idx)]
                scatter = ax.scatter(
                    S_plot[:, 0], S_plot[:, 1],
                    c=expr, cmap="Reds", s=ps,
                    linewidths=0, alpha=0.9, rasterized=True,
                )
                ax.set_aspect("equal")
                ax.set_xlabel("x", fontsize=7)
                ax.set_ylabel("y", fontsize=7)
                gname = gene_names[int(gene_idx)] if int(gene_idx) < len(gene_names) else f"gene_{gene_idx}"
                if pv_arr is not None and qv_arr is not None:
                    ann = f"p={pv_arr[int(gene_idx)]:.2e}  q={qv_arr[int(gene_idx)]:.2e}"
                elif rh_arr is not None:
                    ann = f"|Spearman r| = {abs(float(rh_arr[int(gene_idx)])):.3f}"
                else:
                    ann = ""
                ax.set_title(f"{gname}\n{ann}" if ann else gname, fontsize=7)
                plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=expr_label)

            for col in range(len(top), n_cols):
                axes[row_idx, col].set_visible(False)

            axes[row_idx, 0].set_ylabel(
                f"{row_label}\n\ny", fontsize=7,
            )

        fig.suptitle(
            f"Spatial expression — {isodepth_label} | {covariate_label}",
            fontsize=10, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(
            Path(f"{stem}_svg_spatial_expression.png"),
            dpi=200, bbox_inches="tight",
        )
        plt.close(fig)
    except Exception:
        pass

    return out_path


def _expression_label(meta: dict | None) -> str:
    """Human-readable colorbar label describing the expression normalisation."""
    kind = _expression_preprocessing_kind(meta)
    if kind == "unknown":
        return "Expression"
    if kind == "poisson_latent":
        return "Feature value (Poisson low-rank latent)"
    if kind == "log_z_scored":
        return "Log-norm. standardized expression"
    if kind == "log":
        return "Log-norm. expression"
    if kind == "z_scored":
        return "Standardized expression"
    return "Expression (raw counts)"


def save_svg_spatial_expression_plots(
    S: np.ndarray,
    A: np.ndarray,
    gene_names: list[str],
    gene_indices: np.ndarray,
    out_path: str | Path,
    *,
    pvalues: np.ndarray | None = None,
    qvalues: np.ndarray | None = None,
    rhos: np.ndarray | None = None,
    expression_meta: dict | None = None,
    n_top: int = 5,
    suptitle: str = "",
) -> Path:
    """Spatial scatter plots for the top ``n_top`` genes colored by expression.

    Each panel shows cells at their (x, y) spatial coordinates colored by the
    expression level of one gene — the same style as the dataset-triptych total
    expression heatmap but resolved to individual gene expression values.

    Gene selection and panel annotation depend on which statistics are supplied:

    * **Parametric decoders** (``pvalues`` and ``qvalues`` provided): genes are
      ranked by raw F-test p-value (ascending); each panel shows ``p=...  q=...``.
      ``gene_indices`` should be the significant-gene subset (q < alpha).
    * **nn decoder** (``pvalues``/``qvalues`` absent, ``rhos`` provided): genes
      are displayed in the order supplied in ``gene_indices`` (caller ranks by
      |rho|); each panel shows |Spearman rho|.

    Parameters
    ----------
    S            : (N, 2) spatial coordinates.
    A            : (N, G) expression matrix.
    gene_names   : length-G list of gene name strings.
    gene_indices : gene indices (into G columns of A) to consider.
    out_path     : destination PNG path.
    pvalues      : (G,) raw F-test p-values — used for ranking and annotation.
    qvalues      : (G,) BH q-values — used for annotation only.
    rhos         : (G,) |Spearman rho| values — used when pvalues are absent.
    n_top        : maximum panels to display (default 5).
    suptitle     : optional figure super-title.
    """
    out_path = Path(out_path)
    S = np.asarray(S, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    gene_indices = np.asarray(gene_indices, dtype=np.intp)

    if gene_indices.size == 0:
        return out_path


    pv_arr = np.asarray(pvalues, dtype=np.float64) if pvalues is not None else None
    qv_arr = np.asarray(qvalues, dtype=np.float64) if qvalues is not None else None
    rho_arr = np.asarray(rhos, dtype=np.float64) if rhos is not None else None

    # Ranking: p-value order for parametric, caller-supplied order for nn
    if pv_arr is not None:
        order = np.argsort(pv_arr[gene_indices])
        top_indices = gene_indices[order[: int(n_top)]]
    else:
        top_indices = gene_indices[: int(n_top)]

    n_panels = int(len(top_indices))
    ps = _point_size(S)
    expr_label = _expression_label(expression_meta)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5), squeeze=False)
    axes_flat = axes[0]

    for col, gene_idx in enumerate(top_indices):
        ax = axes_flat[col]
        expr = A[:, int(gene_idx)]
        scatter = ax.scatter(
            S[:, 0], S[:, 1],
            c=expr,
            cmap="Reds",
            s=ps,
            linewidths=0,
            alpha=0.9,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        gname = gene_names[int(gene_idx)] if int(gene_idx) < len(gene_names) else f"gene_{gene_idx}"
        if pv_arr is not None and qv_arr is not None:
            annotation = f"p={pv_arr[int(gene_idx)]:.2e}  q={qv_arr[int(gene_idx)]:.2e}"
        elif rho_arr is not None:
            annotation = f"|Spearman r| = {abs(float(rho_arr[int(gene_idx)])):.3f}"
        else:
            annotation = ""
        ax.set_title(f"{gname}\n{annotation}" if annotation else gname, fontsize=8)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=expr_label)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_recursive_svg_count_plot(
    svg_counts_by_label: dict[str, list[dict]],
    out_path: str | Path,
    *,
    title: str = "Recursive SVG counts by gradient",
) -> Path | None:
    """Line plot of the number of SVGs detected at each recursive gradient."""
    out_path = Path(out_path)

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, entries in svg_counts_by_label.items():
        points: list[tuple[int, int]] = []
        for entry in entries:
            if "gradient_index" not in entry or "n_svgs" not in entry:
                continue
            points.append((int(entry["gradient_index"]), int(entry["n_svgs"])))
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        xs = np.asarray([p[0] for p in points], dtype=np.int64)
        ys = np.asarray([p[1] for p in points], dtype=np.int64)
        series.append((str(label), xs, ys))

    if not series:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(len(series), 1)))
    max_y = 0
    all_x: list[int] = []
    for color, (label, xs, ys) in zip(colors, series):
        max_y = max(max_y, int(ys.max()))
        all_x.extend(int(x) for x in xs)
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=5, label=label, color=color)
        for x, y in zip(xs, ys):
            ax.annotate(
                str(int(y)),
                (int(x), int(y)),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=color,
            )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Gradient")
    ax.set_ylabel("SVGs detected")
    ax.set_xticks(sorted(set(all_x)))
    ax.set_ylim(bottom=0, top=max(1, max_y) * 1.15)
    ax.grid(alpha=0.25, linewidth=0.6)
    if len(series) > 1:
        ax.legend(title="Cell type", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_block_permutation_overlay(
    S_true: np.ndarray,
    S_permuted: np.ndarray | None,
    block_ids: np.ndarray | None,
    out_path: str | Path,
    *,
    run_name: str = "",
    radius_units: float | None = None,
    block_shape: str = "hexagon",
) -> Path | None:
    """Two-panel diagnostic: true layout with block mesh, then a sample centroid permutation.

    Left panel shows cells on the tissue with the block mesh drawn on top.
    Right panel (when provided) colours cells by their *original* block ID at permuted
    coordinates so it is clear how blocks were scrambled.
    """
    from matplotlib.collections import LineCollection

    from methods.block_permutation import (
        block_polygons_for_block_ids,
        square_block_grid_line_segments,
    )

    out_path = Path(out_path)
    S_true = np.asarray(S_true, dtype=np.float32)
    if S_true.ndim != 2 or S_true.shape[1] != 2 or S_true.shape[0] == 0:
        return None

    n_panels = 2 if S_permuted is not None else 1
    spatial_limits = _spatial_axis_limits(S_true)
    ps = _point_size(S_true)
    shape_label = "square" if block_shape == "square" else "hex"

    block_ids_arr: np.ndarray | None = None
    block_colors: np.ndarray | None = None
    n_blocks = 0
    if block_ids is not None:
        block_ids_arr = np.asarray(block_ids, dtype=np.int64)
        unique_ids = np.unique(block_ids_arr)
        id_to_idx = {int(bid): i for i, bid in enumerate(unique_ids)}
        block_colors = np.array([id_to_idx[int(b)] for b in block_ids_arr], dtype=float)
        n_blocks = len(unique_ids)

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    xlim, ylim = spatial_limits

    ax_true = axes[0]
    ax_true.scatter(
        S_true[:, 0],
        S_true[:, 1],
        c="#bdbdbd",
        s=ps,
        linewidths=0,
        alpha=0.55,
        rasterized=True,
        zorder=1,
    )
    if radius_units is not None and float(radius_units) > 0:
        radius = float(radius_units)
        if block_shape == "square":
            grid_lines = square_block_grid_line_segments(
                radius,
                x_min=float(xlim[0]),
                x_max=float(xlim[1]),
                y_min=float(ylim[0]),
                y_max=float(ylim[1]),
            )
            if grid_lines:
                mesh = LineCollection(
                    grid_lines,
                    colors="#404040",
                    linewidths=0.45,
                    alpha=0.85,
                    zorder=2,
                )
                ax_true.add_collection(mesh)
        elif block_ids_arr is not None:
            block_polys = block_polygons_for_block_ids(
                block_ids_arr, radius, block_shape=block_shape,
            )
            if block_polys:
                mesh = PolyCollection(
                    block_polys,
                    facecolors="none",
                    edgecolors="#404040",
                    linewidths=0.45,
                    alpha=0.85,
                    zorder=2,
                )
                ax_true.add_collection(mesh)
    ax_true.set_aspect("equal")
    ax_true.set_xlabel("x")
    ax_true.set_ylabel("y")
    ax_true.set_title(f"True layout — {n_blocks} {shape_label} blocks")
    ax_true.set_xlim(float(xlim[0]), float(xlim[1]))
    ax_true.set_ylim(float(ylim[0]), float(ylim[1]))

    if S_permuted is not None:
        S_perm = np.asarray(S_permuted, dtype=np.float32)
        ax_perm = axes[1]
        cmap = "tab20" if n_blocks <= 20 else "nipy_spectral"
        ax_perm.scatter(
            S_perm[:, 0],
            S_perm[:, 1],
            c=block_colors if block_colors is not None else "#bdbdbd",
            cmap=cmap if block_colors is not None else None,
            s=ps,
            linewidths=0,
            alpha=0.85,
            rasterized=True,
        )
        ax_perm.set_aspect("equal")
        ax_perm.set_xlabel("x")
        ax_perm.set_ylabel("y")
        ax_perm.set_title("Centroid permutation — colour = original block")
        ax_perm.set_xlim(float(xlim[0]), float(xlim[1]))
        ax_perm.set_ylim(float(ylim[0]), float(ylim[1]))

        if block_ids_arr is not None and n_blocks <= 50:
            for bid in np.unique(block_ids_arr):
                mask = block_ids_arr == bid
                if int(mask.sum()) == 0:
                    continue
                old_c = S_true[mask].mean(axis=0)
                new_c = S_perm[mask].mean(axis=0)
                ax_perm.annotate(
                    "",
                    xy=(new_c[0], new_c[1]),
                    xytext=(old_c[0], old_c[1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.6, alpha=0.45),
                )

    title_parts = ["Block permutation overlay"]
    if run_name:
        title_parts = [f"Block permutation overlay — {run_name}"]
    if radius_units is not None:
        title_parts.append(f"radius = {radius_units:.1f} coord units")
    title_parts.append(f"shape = {block_shape}")
    fig.suptitle("  ".join(title_parts), fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_permutation_null_comparison(
    S_true: np.ndarray,
    S_block_perm: np.ndarray,
    A: np.ndarray,
    out_path: str | Path,
    *,
    seed: int,
    run_name: str = "",
) -> Path | None:
    """Three-panel mean |expression| map: true layout, block null, global coordinate null."""
    from methods.permutation import global_coordinate_permute_slot

    out_path = Path(out_path)
    S_true = np.asarray(S_true, dtype=np.float32)
    S_block_perm = np.asarray(S_block_perm, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    n_cells = int(S_true.shape[0])
    if S_true.ndim != 2 or S_true.shape[1] != 2 or n_cells == 0:
        return None
    if S_block_perm.shape != S_true.shape:
        raise ValueError(
            f"S_block_perm shape {S_block_perm.shape} must match S_true {S_true.shape}"
        )
    if A.shape[0] != n_cells:
        raise ValueError(f"A must have {n_cells} rows, got {A.shape[0]}")

    S_global_perm = global_coordinate_permute_slot(S_true, seed=int(seed), slot=1)
    signal = _cell_expression_signal(A)
    vmin = float(signal.min())
    vmax = float(signal.max())
    spatial_limits = _spatial_axis_limits(S_true)
    xlim, ylim = spatial_limits

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = [
        (S_true, "True coordinates"),
        (S_block_perm, "Block null (slot 1)"),
        (S_global_perm, "Global coordinate null (slot 1)"),
    ]
    for ax, (S_panel, title) in zip(axes, panels):
        _plot_spatial_dataset_heatmap(
            ax,
            S_panel,
            signal,
            title,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    title_parts = ["Permutation null comparison (mean |expression|)"]
    if run_name:
        title_parts[0] = f"Permutation null comparison — {run_name}"
    fig.suptitle(title_parts[0], fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_msr_surrogate_example_plot(
    dataset: "DatasetBundle",
    result: "TestResult",
    out_path: "str | Path",
    *,
    n_genes: int = 5,
) -> "Path | None":
    """Two-row spatial scatter: real expression (row 1) vs MSR surrogate (row 2).

    Genes are selected by descending absolute Spearman correlation with the
    true isodepth, giving the most spatially structured genes.

    Parameters
    ----------
    dataset : DatasetBundle with the real expression matrix (``dataset.A``).
    result : TestResult with ``artifacts["msr_surrogate_example"]`` and
             ``artifacts["true_isodepth"]``.
    out_path : destination path for the PNG.
    n_genes : number of top genes to display (default 5).
    """
    A_surr = result.artifacts.get("msr_surrogate_example")
    true_isodepth = result.artifacts.get("true_isodepth")
    if A_surr is None or true_isodepth is None:
        return None

    out_path = Path(out_path)
    A_real = np.asarray(dataset.A, dtype=np.float32)
    A_surr = np.asarray(A_surr, dtype=np.float32)
    isodepth = np.asarray(true_isodepth, dtype=np.float32).reshape(-1)
    S = np.asarray(dataset.S, dtype=np.float32)
    G = A_real.shape[1]
    n_genes = min(int(n_genes), G)

    # rank genes by |Spearman r| with isodepth
    rhos = np.array(
        [float(abs(spearmanr(isodepth, A_real[:, g]).statistic)) for g in range(G)],
        dtype=np.float32,
    )
    top_indices = np.argsort(rhos)[::-1][:n_genes]

    gene_names = dataset.meta.get("var_names") or dataset.meta.get("gene_names", [])
    ps = _point_size(S)
    color_limits = []
    for g_idx in top_indices:
        combined = np.concatenate(
            [A_real[:, int(g_idx)], A_surr[:, int(g_idx)]]
        )
        vmin, vmax = float(combined.min()), float(combined.max())
        if vmax <= vmin:
            vmax = vmin + 1e-8
        color_limits.append((vmin, vmax))

    fig, axes = plt.subplots(2, n_genes, figsize=(4.5 * n_genes, 8.0))
    if n_genes == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["Real expression", "MSR surrogate"]
    row_data = [A_real, A_surr]

    for row, (label, A_row) in enumerate(zip(row_labels, row_data)):
        for col, g_idx in enumerate(top_indices):
            ax = axes[row, col]
            expr = A_row[:, int(g_idx)]
            vmin, vmax = color_limits[col]
            sc = ax.scatter(
                S[:, 0], S[:, 1],
                c=expr,
                cmap="Reds",
                vmin=vmin, vmax=vmax,
                s=ps,
                linewidths=0,
                alpha=0.9,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_xlabel("x", fontsize=7)
            ax.tick_params(labelsize=6)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                gname = (
                    gene_names[int(g_idx)]
                    if int(g_idx) < len(gene_names)
                    else f"gene_{g_idx}"
                )
                ax.set_title(f"{gname}\n|r|={rhos[int(g_idx)]:.3f}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{label}\n\ny", fontsize=7)
            else:
                ax.set_ylabel("y", fontsize=7)

    fig.suptitle(
        "MSR surrogate example — real vs surrogate expression (top genes by Spearman |r|)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_fourier_surrogate_example_plot(
    dataset: "DatasetBundle",
    result: "TestResult",
    out_path: "str | Path",
    *,
    n_genes: int = 5,
) -> "Path | None":
    """Two-row spatial scatter: real expression vs Fourier phase-randomized surrogate."""
    A_surr = result.artifacts.get("fourier_surrogate_example")
    true_isodepth = result.artifacts.get("true_isodepth")
    if A_surr is None or true_isodepth is None:
        return None

    out_path = Path(out_path)
    A_real = np.asarray(dataset.A, dtype=np.float32)
    A_surr = np.asarray(A_surr, dtype=np.float32)
    isodepth = np.asarray(true_isodepth, dtype=np.float32).reshape(-1)
    S = np.asarray(dataset.S, dtype=np.float32)
    G = A_real.shape[1]
    n_genes = min(int(n_genes), G)

    rhos = np.array(
        [float(abs(spearmanr(isodepth, A_real[:, g]).statistic)) for g in range(G)],
        dtype=np.float32,
    )
    top_indices = np.argsort(rhos)[::-1][:n_genes]
    gene_names = dataset.meta.get("var_names") or dataset.meta.get("gene_names", [])
    ps = _point_size(S)
    color_limits = []
    for g_idx in top_indices:
        combined = np.concatenate([A_real[:, int(g_idx)], A_surr[:, int(g_idx)]])
        vmin, vmax = float(combined.min()), float(combined.max())
        if vmax <= vmin:
            vmax = vmin + 1e-8
        color_limits.append((vmin, vmax))

    fig, axes = plt.subplots(2, n_genes, figsize=(4.5 * n_genes, 8.0))
    if n_genes == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["Real expression", "Fourier surrogate"]
    row_data = [A_real, A_surr]
    for row, (label, A_row) in enumerate(zip(row_labels, row_data)):
        for col, g_idx in enumerate(top_indices):
            ax = axes[row, col]
            expr = A_row[:, int(g_idx)]
            vmin, vmax = color_limits[col]
            sc = ax.scatter(
                S[:, 0],
                S[:, 1],
                c=expr,
                cmap="Reds",
                vmin=vmin,
                vmax=vmax,
                s=ps,
                linewidths=0,
                alpha=0.9,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_xlabel("x", fontsize=7)
            ax.tick_params(labelsize=6)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                gname = (
                    gene_names[int(g_idx)]
                    if int(g_idx) < len(gene_names)
                    else f"gene_{g_idx}"
                )
                ax.set_title(f"{gname}\n|r|={rhos[int(g_idx)]:.3f}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{label}\n\ny", fontsize=7)
            else:
                ax.set_ylabel("y", fontsize=7)

    fig.suptitle(
        "Fourier spectral randomization — real vs phase-randomized expression",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_moran_distribution_plot(result: "TestResult", out_path: "str | Path") -> "Path | None":
    """Histogram of null mean Moran's I across genes with true value marked."""
    true_mean = result.artifacts.get("moran_true_mean")
    null_means = result.artifacts.get("moran_null_mean_per_perm")
    if true_mean is None or null_means is None:
        return None

    out_path = Path(out_path)
    true_mean_f = float(true_mean)
    null_means_arr = np.asarray(null_means, dtype=np.float64)
    n_perms = int(null_means_arr.shape[0])
    radius_um = result.artifacts.get("moran_neighbor_radius_um")
    p_value = result.artifacts.get("moran_p_value")
    rank = result.artifacts.get("moran_rank")

    null_mean_of_means = float(null_means_arr.mean()) if n_perms else float("nan")
    null_std_of_means = float(null_means_arr.std()) if n_perms else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    radius_label = f"{float(radius_um):.0f}" if radius_um is not None else "?"
    fig.suptitle(
        f"Global Moran's I — null vs true (neighbor r={radius_label} µm)",
        fontsize=10,
        fontweight="bold",
    )

    if n_perms > 0:
        bins = max(15, min(40, n_perms // 3))
        ax.hist(
            null_means_arr,
            bins=bins,
            color="#27ae60",
            alpha=0.72,
            edgecolor="k",
            linewidth=0.4,
            label=f"null mean I (n={n_perms})",
        )
    ax.axvline(
        true_mean_f,
        color="crimson",
        lw=2.2,
        ls="--",
        label=f"true mean = {true_mean_f:.4f}",
    )
    if n_perms > 0:
        ax.axvline(
            null_mean_of_means,
            color="k",
            lw=1.2,
            ls=":",
            alpha=0.8,
            label=f"null avg = {null_mean_of_means:.4f} ± {null_std_of_means:.4f}",
        )
    ax.set_xlabel("Mean Moran's I across genes")
    ax.set_ylabel("Count")
    if p_value is not None and rank is not None and n_perms > 0:
        pct = 100.0 * float(rank) / (n_perms + 1)
        ax.set_title(
            f"true rank {int(rank)}/{n_perms + 1} ({pct:.1f} percentile), "
            f"p = {float(p_value):.4g}",
            fontsize=9,
        )
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _training_metadata_from_result(result: TestResult) -> dict[str, Any] | None:
    metadata = result.artifacts.get("training_metadata")
    if isinstance(metadata, dict):
        return metadata
    model = result.artifacts.get("model")
    if model is not None:
        metadata = getattr(model, "training_metadata", None)
        if isinstance(metadata, dict):
            return metadata
    return None



def _extract_loss_history(result: TestResult) -> np.ndarray | None:
    history = result.artifacts.get("loss_history")
    if history is None:
        metadata = _training_metadata_from_result(result)
        if metadata is not None:
            history = metadata.get("loss_history")
    if history is None:
        return None
    arr = np.asarray(history, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _extract_loss_history_per_slot(result: TestResult) -> tuple[np.ndarray | None, int]:
    """Return ``(epochs, slots)`` train-loss history and ``n_reruns`` when available."""
    metadata = _training_metadata_from_result(result)
    n_reruns = 1
    history = result.artifacts.get("loss_history_per_slot")
    if metadata is not None:
        n_reruns = max(1, int(metadata.get("n_reruns", 1)))
        if history is None:
            history = metadata.get("loss_history_per_slot")
    if history is None:
        return None, n_reruns
    arr = np.asarray(history, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None, n_reruns
    return arr, n_reruns


def _select_fixed_rerun_curves_from_slots(
    per_slot: np.ndarray,
    n_reruns: int,
) -> np.ndarray | None:
    """Collapse expanded slots to one curve per logical model, using a *fixed* rerun.

    Slot layout is ``(true + n_perms) * n_reruns`` with reruns contiguous per model.
    For each model, the rerun with the lowest loss **at the final recorded epoch**
    is chosen once (mirroring ``best_rerun_index_per_model = argmin(final loss)``
    in ``methods/trainers/isodepth.py``, which is how the real training pipeline
    selects a single rerun per model). That same rerun's full trajectory is then
    used across *all* epochs.

    This intentionally does NOT re-select the best rerun independently at every
    epoch: doing so would silently stitch together different reruns at different
    points in training into an artificial "lower envelope" that no single actual
    training run ever achieved, which is misleading and inconsistent with the
    fixed-rerun model that the real test statistic is computed from.

    Returns an array of shape ``(n_epochs, n_models)``, or ``None`` if slots
    can't be grouped into reruns (e.g. no rerun metadata available).
    """
    n_epochs, n_slots = per_slot.shape
    n_reruns = max(1, int(n_reruns))
    if n_slots < n_reruns or n_slots % n_reruns != 0:
        return None
    n_models = n_slots // n_reruns
    reshaped = per_slot.reshape(n_epochs, n_models, n_reruns)
    final_losses = reshaped[-1, :, :]
    # np.nanargmin raises on all-NaN rows; treat NaNs as +inf so such a model
    # falls back to rerun 0 instead of crashing the plot.
    safe_final_losses = np.where(np.isnan(final_losses), np.inf, final_losses)
    best_rerun_index = np.argmin(safe_final_losses, axis=1)
    return reshaped[:, np.arange(n_models), best_rerun_index]


def _true_and_perm_loss_curves_from_slots(
    per_slot: np.ndarray,
    n_reruns: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split fixed-rerun model curves into a true curve and permutation curves.

    Both curves use the single rerun per model that wins at the final epoch (see
    :func:`_select_fixed_rerun_curves_from_slots`), matching the epoch-wise p-value
    computation in :func:`_pvalue_trajectory_from_slots`, so the plotted "True" line
    is directly comparable to the permutation curves and the p-value trajectory.

    Returns ``(true_curve, perm_curves)`` where ``true_curve`` has shape
    ``(n_epochs,)`` and ``perm_curves`` has shape ``(n_epochs, n_perms)`` or is
    ``None`` when there are no permutation models.
    """
    n_epochs, n_slots = per_slot.shape
    selected = _select_fixed_rerun_curves_from_slots(per_slot, n_reruns)
    if selected is None:
        # Fall back: treat every non-leading slot as its own faded curve, and the
        # leading slot (no rerun grouping available) as the true curve as-is.
        true_curve = np.asarray(per_slot[:, 0], dtype=np.float64)
        perm_curves = None if n_slots <= 1 else np.asarray(per_slot[:, 1:], dtype=np.float64)
        return true_curve, perm_curves
    true_curve = selected[:, 0]
    if selected.shape[1] <= 1:
        return true_curve, None
    return true_curve, selected[:, 1:]


def _pvalue_trajectory_from_slots(
    per_slot: np.ndarray,
    n_reruns: int,
    metric: str,
) -> np.ndarray | None:
    """Epoch-wise permutation p-value from per-slot train losses (fixed rerun).

    Each model (true and every permutation) is represented by the single rerun
    that wins at the final epoch (see :func:`_select_fixed_rerun_curves_from_slots`),
    matching final model selection. The Monte Carlo p-value is then computed from
    that fixed trajectory's loss at each epoch, so this traces how the *actual*
    kept models ranked against each other over the course of training, rather than
    re-selecting the best rerun independently at every epoch. For
    ``nll_gaussian_mse`` the recorded history is MSE, which is rank-equivalent to
    the Gaussian-NLL test statistic.
    """
    per_slot = np.asarray(per_slot, dtype=np.float64)
    if per_slot.ndim != 2 or per_slot.shape[0] == 0 or per_slot.shape[1] == 0:
        return None
    selected = _select_fixed_rerun_curves_from_slots(per_slot, n_reruns)
    if selected is None or selected.shape[1] <= 1:
        return None
    true = selected[:, 0]
    perms = selected[:, 1:]
    n_perms = int(perms.shape[1])
    if metric_prefers_lower(metric):
        counts = np.sum(perms <= true[:, None], axis=1)
    else:
        counts = np.sum(perms >= true[:, None], axis=1)
    return (1.0 + counts.astype(np.float64)) / float(n_perms + 1)


def save_loss_curve_plot(
    result: TestResult,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Save training loss vs epoch; faded permutation curves behind the true curve.

    When per-slot loss history is available, the plotted "True" curve and the
    permutation curves both use the min-over-reruns selection at each epoch
    (matching final model selection), so they are directly comparable and the
    epoch-wise permutation p-value overlaid on a right-hand axis reflects exactly
    what's drawn. Without per-slot history, falls back to the single recorded
    ``loss_history`` (rerun 0) and no permutation curves or p-value are shown.
    """
    per_slot, n_reruns = _extract_loss_history_per_slot(result)
    perm_curves = None
    if per_slot is not None:
        losses, perm_curves = _true_and_perm_loss_curves_from_slots(per_slot, n_reruns)
    else:
        losses = _extract_loss_history(result)
    if losses is None:
        return None

    out_path = Path(out_path)
    epochs = np.arange(1, losses.size + 1, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    if perm_curves is not None and perm_curves.shape[0] == losses.size:
        n_perms = int(perm_curves.shape[1])
        alpha = float(min(0.35, max(0.04, 4.0 / max(n_perms, 1))))
        for perm_index in range(n_perms):
            ax.plot(
                epochs,
                perm_curves[:, perm_index],
                color="0.55",
                alpha=alpha,
                linewidth=0.8,
                zorder=1,
            )
        ax.plot([], [], color="0.55", alpha=min(0.8, alpha * 4.0), linewidth=1.2, label=f"Permutations (n={n_perms})")

    true_label = "True (best rerun)" if perm_curves is not None or per_slot is not None else "True"
    ax.plot(epochs, losses, color="steelblue", linewidth=1.8, zorder=3, label=true_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(title or "Training loss by epoch")
    ax.grid(alpha=0.25, linewidth=0.6)

    p_traj = (
        None
        if per_slot is None or per_slot.shape[0] != losses.size
        else _pvalue_trajectory_from_slots(per_slot, n_reruns, result.metric)
    )
    if p_traj is not None:
        ax_p = ax.twinx()
        ax_p.plot(
            epochs,
            p_traj,
            color="crimson",
            linewidth=1.6,
            zorder=4,
            label="p-value",
        )
        ax_p.set_ylabel("p-value", color="crimson")
        ax_p.tick_params(axis="y", labelcolor="crimson")
        ax_p.set_ylim(0.0, 1.0)
        handles_l, labels_l = ax.get_legend_handles_labels()
        handles_p, labels_p = ax_p.get_legend_handles_labels()
        ax.legend(handles_l + handles_p, labels_l + labels_p, loc="best", frameon=False)
    else:
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


