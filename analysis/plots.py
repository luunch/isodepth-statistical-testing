from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import colors as mcolors
import numpy as np
from scipy.stats import gaussian_kde

from data.schemas import DatasetBundle, TestResult

# Exclude points treated as zero expression when ``hide_zero_expression`` is enabled.
_EXPRESSION_ZERO_EPS = 1e-15


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
    scatter = ax.scatter(
        S[:, 0],
        S[:, 1],
        c=np.asarray(signal, dtype=np.float32),
        cmap="magma",
        s=_point_size(S),
        linewidths=0,
        alpha=0.9,
        vmin=vmin,
        vmax=vmax,
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_spatial_isodepth(
        axes[0],
        np.asarray(dataset.S, dtype=np.float32),
        true_isodepth,
        "True Data Isodepth",
    )
    _plot_spatial_isodepth(
        axes[1],
        lowest_S,
        lowest_isodepth,
        f"Lowest Metric Isodepth\n{float(result.artifacts.get('lowest_stat')):.4g}",
    )
    _plot_spatial_isodepth(
        axes[2],
        highest_S,
        highest_isodepth,
        f"Highest Metric Isodepth\n{float(result.artifacts.get('highest_stat')):.4g}",
    )
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

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(np.asarray(result.stat_perm, dtype=np.float64), bins=30, color="lightsteelblue", edgecolor="black")
    ax.axvline(result.stat_true, color="crimson", linestyle="--", label=f"Observed: {result.stat_true:.4g}")
    if "lowest_stat" in result.artifacts:
        ax.axvline(float(result.artifacts["lowest_stat"]), color="darkgreen", linestyle=":", label=f"Lowest: {float(result.artifacts['lowest_stat']):.4g}")
    if "highest_stat" in result.artifacts:
        ax.axvline(float(result.artifacts["highest_stat"]), color="darkorange", linestyle=":", label=f"Highest: {float(result.artifacts['highest_stat']):.4g}")
    ax.set_title(f"Null Distribution\np-value = {result.p_value:.4g}")
    ax.set_xlabel(result.metric)
    ax.set_ylabel("Count")
    fig.tight_layout()
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
) -> None:
    """Scatter (and optional tricontour) for one gene on one spatial layout; values use vmin/vmax directly."""
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
            cmap="viridis",
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
                contour_colors = plt.cm.Reds(np.linspace(0.35, 0.95, levels.size))
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
        )

    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes_arr[r, c].set_visible(False)

    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
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
        _plot_spatial_expression_panel(
            ax,
            S,
            z,
            gene_labels[idx],
            vmin=vmin,
            vmax=vmax,
            show_contours=show_contours,
            hide_zero_expression=hide_zero_expression,
        )

    for idx in range(n_genes, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="Expression")

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