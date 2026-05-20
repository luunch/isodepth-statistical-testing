from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import colors as mcolors
import numpy as np
from scipy.stats import gaussian_kde

from data.schemas import DatasetBundle, TestResult

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


def _as_dimension_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected isodepth array with 1 or 2 dimensions, got shape {array.shape}")


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

    cmap = plt.cm.get_cmap("tab20" if n_types > 10 else "tab10")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for c in range(n_types):
        mask = labels == c
        if not np.any(mask):
            continue
        ax.scatter(
            S[mask, 0],
            S[mask, 1],
            c=[cmap(c / max(n_types - 1, 1))],
            s=_point_size(S),
            label=cell_type_names[c],
            alpha=0.7,
            linewidths=0,
        )
    ax.set_title("Dataset Colored by Cell Type")
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


def save_celltype_expression_plot(
    dataset: DatasetBundle,
    result: TestResult,
    out_path: str | Path,
) -> Path | None:
    """Scatter plot of cells colored by mean predicted expression, labeled by cell type."""
    cell_type_labels = dataset.meta.get("cell_type_labels")
    cell_type_names = dataset.meta.get("cell_type_names")
    pred_true = result.artifacts.get("pred_true")
    if cell_type_labels is None or cell_type_names is None or pred_true is None:
        return None

    labels = np.asarray(cell_type_labels, dtype=np.int64)
    S = np.asarray(dataset.S, dtype=np.float32)
    preds = np.asarray(pred_true, dtype=np.float32)
    signal = np.mean(np.abs(preds), axis=1)

    n_types = len(cell_type_names)
    cmap_cat = plt.cm.get_cmap("tab20" if n_types > 10 else "tab10")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter = axes[0].scatter(
        S[:, 0], S[:, 1],
        c=signal,
        cmap="Reds",
        s=_point_size(S),
        linewidths=0,
        alpha=0.8,
    )
    axes[0].set_title("Predicted Expression (mean |pred|)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)

    for c in range(n_types):
        mask = labels == c
        if not np.any(mask):
            continue
        axes[1].scatter(
            S[mask, 0],
            S[mask, 1],
            c=[cmap_cat(c / max(n_types - 1, 1))],
            s=_point_size(S),
            label=cell_type_names[c],
            alpha=0.7,
            linewidths=0,
        )
    axes[1].set_title("Cell Types")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    axes[1].legend(
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


def _save_exact_existence_triptych(
    dataset: DatasetBundle,
    result: TestResult,
    out_path: Path,
) -> Path | None:
    rows = result.artifacts.get("dimension_plot_rows")
    if not isinstance(rows, list) or not rows:
        return None

    spatial = np.asarray(dataset.S, dtype=np.float32)
    panels: list[tuple[np.ndarray, np.ndarray, str]] = []

    for row in rows:
        dim = int(row["tested_dim"])
        true_depths = _as_dimension_matrix(row["true_isodepth"])
        low_depths = _as_dimension_matrix(row["lowest_isodepth"])
        high_depths = _as_dimension_matrix(row["highest_isodepth"])
        low_S = np.asarray(row["lowest_S"], dtype=np.float32)
        high_S = np.asarray(row["highest_S"], dtype=np.float32)
        labels = list(row.get("dimension_labels") or [f"d{i + 1}" for i in range(dim)])
        for dim_index in range(dim):
            label = labels[dim_index] if dim_index < len(labels) else f"d{dim_index + 1}"
            title_suffix = (
                f"dim {dim}\np={float(row['p_value']):.4g}"
                if dim_index == 0
                else f"dim {dim}"
            )
            panels.append((spatial, true_depths[:, dim_index], f"True {label}\n{title_suffix}"))
            panels.append((low_S, low_depths[:, dim_index], f"Lowest {label}\n{float(row['lowest_stat']):.4g}"))
            panels.append((high_S, high_depths[:, dim_index], f"Highest {label}\n{float(row['highest_stat']):.4g}"))

    n_cols = min(3, max(len(panels), 1))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for axis, (panel_S, panel_depth, panel_title) in zip(axes.flat, panels):
        _plot_spatial_isodepth(axis, panel_S, panel_depth, panel_title)

    for axis in axes.flat[len(panels):]:
        axis.axis("off")

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
    if result.method_name == "exact_existence":
        out_path = Path(out_path)
        return _save_exact_existence_triptych(dataset, result, out_path)
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
    if result.method_name == "exact_existence":
        step_summaries = result.artifacts.get("step_summaries")
        if isinstance(step_summaries, dict) and step_summaries:
            ordered_keys = sorted(step_summaries.keys(), key=lambda value: int(value))
            fig, axes = plt.subplots(len(ordered_keys), 1, figsize=(6, 4 * len(ordered_keys)), squeeze=False)
            for ax, key in zip(axes[:, 0], ordered_keys):
                summary = step_summaries[key]
                stat_perm = np.asarray(summary["null_distribution"], dtype=np.float64)
                if "observed_delta" in summary:
                    stat_true = float(summary["observed_delta"])
                    title = f"k={int(summary['tested_dim']) - 1} -> {int(summary['tested_dim'])}"
                    xlabel = "Loss Reduction Scale"
                    label = f"Observed Reduction-Scale Stat: {stat_true:.4g}"
                else:
                    stat_true = float(summary["observed_stat"])
                    title = "Existence Test"
                    xlabel = result.metric
                    label = f"Observed: {stat_true:.4g}"
                p_value = float(summary["p_value"])
                significance = "significant" if bool(summary["significant"]) else "not significant"
                ax.hist(stat_perm, bins=30, color="lightsteelblue", edgecolor="black")
                ax.axvline(stat_true, color="crimson", linestyle="--", label=label)
                ax.set_title(f"{title}\np-value = {p_value:.4g} ({significance})")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.legend()
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return out_path
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

        ax.hist(stat_perm, bins=30, color="lightsteelblue", edgecolor="black",
                label="Null (permutations)")
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


def _expression_y_axis_label(meta: dict[str, Any]) -> str:
    """Describe ``dataset.A`` columns (model input / observed expression space)."""
    if meta.get("q") is not None:
        return "Feature value (Poisson low-rank latent)"
    if meta.get("feature_space") == "poisson_low_rank_latent":
        return "Feature value (Poisson low-rank latent)"
    parts: list[str] = []
    if meta.get("log1p"):
        parts.append("log₁p")
    if meta.get("standardize_expression"):
        parts.append("z-scored")
    if parts:
        return "Expression (" + ", ".join(parts) + ")"
    return "Expression"


def _top_genes_by_prediction_correlation(
    A: np.ndarray,
    pred: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gene_indices, pearson_r) for the top ``top_k`` genes by Pearson(A_g, pred_g)."""
    A = np.asarray(A, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if A.shape != pred.shape:
        raise ValueError(f"A shape {A.shape} does not match pred shape {pred.shape}")
    G = A.shape[1]
    corrs = np.full(G, -np.inf, dtype=np.float64)
    for g in range(G):
        a_col = A[:, g]
        p_col = pred[:, g]
        if np.std(a_col) < 1e-12 or np.std(p_col) < 1e-12:
            continue
        corrs[g] = float(np.corrcoef(a_col, p_col)[0, 1])
    order = np.argsort(-corrs)
    k = min(int(top_k), G)
    top_idx = order[:k]
    return top_idx, corrs


def _prediction_scenarios_for_selected_genes(result: TestResult) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """List (subdir_slug, panel_title, pred, isodepth) for top-gene plots."""
    art = result.artifacts
    pred_full_iso = art.get("pred_true_full_iso")
    depth_full_iso = art.get("true_isodepth_full_iso")
    pred_cov = art.get("pred_true_covariate")
    depth_cov = art.get("true_isodepth_covariate")
    pred_primary = art.get("pred_true")
    depth_primary = art.get("true_isodepth")

    scenarios: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    # Parallel midline: slot 0 is midline; separate full model artifacts are present.
    if pred_full_iso is not None and depth_full_iso is not None:
        if pred_primary is None or depth_primary is None:
            return scenarios
        scenarios.append(
            (
                "covariate_midline",
                "Covariate (midline) — true slot",
                np.asarray(pred_primary, dtype=np.float32),
                np.asarray(depth_primary, dtype=np.float32),
            )
        )
        scenarios.append(
            (
                "full_isodepth",
                "Full isodepth (true layout, no covariate)",
                np.asarray(pred_full_iso, dtype=np.float32),
                np.asarray(depth_full_iso, dtype=np.float32),
            )
        )
        return scenarios

    # Covariate comparisons: primary = full learned model; covariate artifacts are separate.
    if pred_cov is not None and depth_cov is not None:
        if pred_primary is None or depth_primary is None:
            return scenarios
        scenarios.append(
            (
                "full_isodepth",
                "Full isodepth (encoder + decoder)",
                np.asarray(pred_primary, dtype=np.float32),
                np.asarray(depth_primary, dtype=np.float32),
            )
        )
        scenarios.append(
            (
                "covariate_midline",
                "Covariate (midline)",
                np.asarray(pred_cov, dtype=np.float32),
                np.asarray(depth_cov, dtype=np.float32),
            )
        )
        return scenarios

    if pred_primary is not None and depth_primary is not None:
        scenarios.append(
            (
                "primary",
                "True-layout model",
                np.asarray(pred_primary, dtype=np.float32),
                np.asarray(depth_primary, dtype=np.float32),
            )
        )
    return scenarios


def _plot_gene_piecewise_linear(
    ax,
    x: np.ndarray,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_bins: int = 10,
    pt_size: float = 50.0,
    scatter_color: str = "mediumseagreen",
    fit_color: str = "0.45",
    fit_lw: float = 3.0,
    bin_color: str = "0.45",
    bin_lw: float = 2.5,
    scatter_alpha: float = 0.65,
) -> None:
    """Scatter + model-prediction fit line + binned-mean horizontal segments.

    Mirrors the GASTON ``plot_gene_pwlinear`` visual style for a single
    piecewise-linear segment: colored scatter, overlaid regression curve
    (from model predictions sorted by isodepth), and stepped horizontal
    bars showing the mean observed expression within each isodepth bin.
    """
    sort_idx = np.argsort(x)
    ax.scatter(x, y_obs, s=pt_size, alpha=scatter_alpha, c=scatter_color, zorder=1)

    ax.plot(
        x[sort_idx],
        y_pred[sort_idx],
        color=fit_color,
        lw=fit_lw,
        solid_capstyle="round",
        zorder=3,
    )

    bin_edges = np.linspace(float(x.min()), float(x.max()), int(num_bins) + 1)
    for i in range(int(num_bins)):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if i == int(num_bins) - 1:
            mask |= x == bin_edges[i + 1]
        if mask.sum() == 0:
            continue
        bin_mean = float(np.mean(y_obs[mask]))
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [bin_mean, bin_mean],
            color=bin_color,
            lw=bin_lw,
            solid_capstyle="butt",
            zorder=2,
        )


def save_selected_genes_expression_vs_isodepth(
    dataset: DatasetBundle,
    result: TestResult,
    out_dir: str | Path,
    *,
    top_k: int = 5,
    num_bins: int = 10,
) -> Path | None:
    """Plot top predicted genes (by per-gene Pearson) vs 1D isodepth; write under ``selected_genes/``.

    Each per-gene figure uses a GASTON-inspired piecewise-linear style:
    colored scatter points, the model prediction as a fit line (sorted by
    isodepth), and horizontal segments showing binned mean expression.

    Y-axis: observed expression in the same space as ``dataset.A``.
    X-axis: first dimension of the model isodepth for that prediction path.

    Returns the ``selected_genes`` directory path, or ``None`` if no usable predictions exist.
    """
    scenarios = _prediction_scenarios_for_selected_genes(result)
    if not scenarios:
        return None

    out_dir = Path(out_dir)
    selected_root = out_dir / "selected_genes"
    selected_root.mkdir(parents=True, exist_ok=True)

    y_label = _expression_y_axis_label(dataset.meta)
    manifest: dict[str, Any] = {
        "top_k": int(top_k),
        "num_bins": int(num_bins),
        "ranking": "per-gene Pearson correlation between observed expression and model prediction",
        "y_axis": y_label,
        "scenarios": [],
    }

    A = np.asarray(dataset.A, dtype=np.float64)
    pt_size = max(8.0, _point_size(np.asarray(dataset.S, dtype=np.float32)))

    for subdir, panel_title, pred, depth in scenarios:
        pred = np.asarray(pred, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)
        if pred.shape[0] != A.shape[0] or pred.shape[1] != A.shape[1]:
            continue
        try:
            x = _flatten_isodepth_for_axis(depth)
        except ValueError:
            continue
        if x.shape[0] != A.shape[0]:
            continue

        top_idx, corrs = _top_genes_by_prediction_correlation(A, pred, top_k=top_k)
        scenario_dir = selected_root / subdir
        scenario_dir.mkdir(parents=True, exist_ok=True)

        gene_entries: list[dict[str, Any]] = []
        for rank, g in enumerate(top_idx):
            r = float(corrs[int(g)])
            y_obs = A[:, int(g)]
            y_pred = pred[:, int(g)]
            gene_name = _gene_display_name(dataset.meta, int(g))
            stem = f"{int(rank):02d}_{int(g):04d}_{_safe_filename_fragment(gene_name)}_r{r:.4f}".replace(".", "p")
            out_png = scenario_dir / f"{stem}.png"

            fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
            _plot_gene_piecewise_linear(
                ax, x, y_obs, y_pred,
                num_bins=num_bins,
                pt_size=pt_size,
            )
            ax.set_xlabel("Isodepth", fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(f"{panel_title}\n{gene_name}  (r={r:.3f})", fontsize=11)
            ax.tick_params(labelsize=10)
            fig.tight_layout()
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)

            gene_entries.append(
                {
                    "rank": int(rank),
                    "gene_index": int(g),
                    "gene_name": gene_name,
                    "pearson_r": r,
                    "filename": str(out_png.relative_to(selected_root)),
                }
            )

        manifest["scenarios"].append(
            {
                "subdir": subdir,
                "title": panel_title,
                "genes": gene_entries,
            }
        )

    manifest_path = selected_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return selected_root