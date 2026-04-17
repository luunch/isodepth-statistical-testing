from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from data.schemas import DatasetBundle, TestResult


def _normalize_depth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    vmin = float(values.min())
    vmax = float(values.max())
    return (values - vmin) / (vmax - vmin + 1e-8)


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


def _plot_spatial_isodepth(ax, S: np.ndarray, depth: np.ndarray, title: str) -> None:
    depth = _normalize_depth(depth)
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
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Normalized isodepth")


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

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
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
