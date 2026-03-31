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
        "Original Isodepth",
    )
    _plot_spatial_isodepth(
        axes[1],
        lowest_S,
        lowest_isodepth,
        f"Most Stable Perturbed Fit\n{float(result.artifacts.get('lowest_stat')):.4g}",
    )
    _plot_spatial_isodepth(
        axes[2],
        highest_S,
        highest_isodepth,
        f"Least Stable Perturbed Fit\n{float(result.artifacts.get('highest_stat')):.4g}",
    )
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
    if result.method_name == "perturbation_robustness":
        return _save_perturbation_triptych(
            dataset,
            result,
            out_path,
            np.asarray(true_isodepth, dtype=np.float32),
            np.asarray(lowest_isodepth, dtype=np.float32),
            np.asarray(lowest_S, dtype=np.float32),
            np.asarray(highest_isodepth, dtype=np.float32),
            np.asarray(highest_S, dtype=np.float32),
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
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(np.asarray(result.stat_perm, dtype=np.float64), bins=30, color="lightsteelblue", edgecolor="black")
    observed_label = "Observed"
    title_prefix = "Null Distribution"
    if result.method_name == "perturbation_robustness":
        observed_label = "Observed Mean"
        title_prefix = "Null Distribution of Mean Perturbation Score"
    ax.axvline(result.stat_true, color="crimson", linestyle="--", label=f"{observed_label}: {result.stat_true:.4g}")
    if result.method_name != "perturbation_robustness" and "lowest_stat" in result.artifacts:
        ax.axvline(float(result.artifacts["lowest_stat"]), color="darkgreen", linestyle=":", label=f"Lowest: {float(result.artifacts['lowest_stat']):.4g}")
    if result.method_name != "perturbation_robustness" and "highest_stat" in result.artifacts:
        ax.axvline(float(result.artifacts["highest_stat"]), color="darkorange", linestyle=":", label=f"Highest: {float(result.artifacts['highest_stat']):.4g}")
    ax.set_title(f"{title_prefix}\np-value = {result.p_value:.4g}")
    ax.set_xlabel(result.metric)
    ax.set_ylabel("Count")
    ax.text(
        0.02,
        0.98,
        f"p = {result.p_value:.4g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
