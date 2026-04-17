from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


def _sort_sigma(values: Iterable[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _sort_k(values: Iterable[int | str]) -> list[int]:
    parsed = []
    for value in values:
        if value in {"", None}:
            continue
        parsed.append(int(value))
    return sorted(set(parsed))


def save_rate_vs_sigma_plot(
    summary_rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
    *,
    truth_label: str,
    title: str,
    y_label: str,
) -> Path | None:
    rows = [row for row in summary_rows if str(row["truth_label"]) == truth_label]
    if not rows:
        return None

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["family_label"]), []).append(row)

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for family_label, family_rows in sorted(grouped.items()):
        family_rows = sorted(family_rows, key=lambda row: float(row["sigma"]))
        sigma = np.asarray([float(row["sigma"]) for row in family_rows], dtype=np.float64)
        rate = np.asarray([float(row["rate"]) for row in family_rows], dtype=np.float64)
        lower = np.asarray([float(row["ci_lower"]) for row in family_rows], dtype=np.float64)
        upper = np.asarray([float(row["ci_upper"]) for row in family_rows], dtype=np.float64)
        ax.plot(sigma, rate, marker="o", linewidth=1.5, label=family_label)
        ax.fill_between(sigma, lower, upper, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("Sigma")
    ax.set_ylabel(y_label)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_fourier_heatmap(
    summary_rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
    *,
    truth_label: str,
    title: str,
) -> Path | None:
    rows = [
        row
        for row in summary_rows
        if str(row["truth_label"]) == truth_label and str(row["mode"]) == "fourier"
    ]
    if not rows:
        return None

    sigma_values = _sort_sigma(float(row["sigma"]) for row in rows)
    k_values = _sort_k(row["k"] for row in rows)
    if not sigma_values or not k_values:
        return None

    matrix = np.full((len(k_values), len(sigma_values)), np.nan, dtype=np.float64)
    sigma_index = {value: index for index, value in enumerate(sigma_values)}
    k_index = {value: index for index, value in enumerate(k_values)}
    for row in rows:
        matrix[k_index[int(row["k"])]][sigma_index[float(row["sigma"])]] = float(row["rate"])

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Fourier k")
    ax.set_xticks(np.arange(len(sigma_values)))
    ax.set_xticklabels([f"{value:.3g}" for value in sigma_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(k_values)))
    ax.set_yticklabels([str(value) for value in k_values])
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_null_pvalue_histograms(
    per_run_rows: Iterable[Mapping[str, object]],
    out_dir: str | Path,
) -> list[Path]:
    rows = [row for row in per_run_rows if str(row["truth_label"]) == "null"]
    if not rows:
        return []

    sigma_values = _sort_sigma(float(row["sigma"]) for row in rows)
    if not sigma_values:
        return []
    representative = [sigma_values[0], sigma_values[len(sigma_values) // 2], sigma_values[-1]]
    chosen = []
    for value in representative:
        if value not in chosen:
            chosen.append(value)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for sigma in chosen:
        sigma_rows = [row for row in rows if float(row["sigma"]) == sigma]
        families = sorted({str(row["family_label"]) for row in sigma_rows})
        if not families:
            continue
        fig, axes = plt.subplots(len(families), 1, figsize=(6, 3.5 * len(families)), squeeze=False)
        for ax, family_label in zip(axes[:, 0], families):
            family_rows = [row for row in sigma_rows if str(row["family_label"]) == family_label]
            p_values = np.asarray([float(row["p_value"]) for row in family_rows], dtype=np.float64)
            ax.hist(p_values, bins=10, range=(0.0, 1.0), color="lightsteelblue", edgecolor="black")
            ax.set_title(f"Null p-values: {family_label}, sigma={sigma:.3g}")
            ax.set_xlabel("p-value")
            ax.set_ylabel("Count")
            ax.set_xlim(-0.02, 1.02)
        fig.tight_layout()
        out_path = out_dir / f"null_pvalues_sigma_{str(sigma).replace('.', 'p')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


def save_pvalue_vs_kmax_plot(
    per_run_rows: Iterable[Mapping[str, object]],
    summary_rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
    *,
    title: str,
) -> Path | None:
    rows = list(per_run_rows)
    summaries = list(summary_rows)
    if not rows or not summaries:
        return None

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x = np.asarray([int(row["k_max"]) for row in rows], dtype=np.int64)
    y = np.asarray([float(row["p_value"]) for row in rows], dtype=np.float64)
    ax.scatter(x, y, alpha=0.7, color="steelblue", edgecolors="none", label="Per-run p-values")

    summary_x = np.asarray([int(row["k_max"]) for row in summaries], dtype=np.int64)
    summary_y = np.asarray([float(row["p_value_mean"]) for row in summaries], dtype=np.float64)
    summary_std = np.asarray([float(row["p_value_std"]) for row in summaries], dtype=np.float64)
    ax.plot(summary_x, summary_y, color="black", linewidth=1.5, marker="o", label="Mean p-value")
    if np.any(np.isfinite(summary_std)):
        lower = np.clip(summary_y - summary_std, 0.0, 1.0)
        upper = np.clip(summary_y + summary_std, 0.0, 1.0)
        ax.fill_between(summary_x, lower, upper, alpha=0.15, color="black", label="Mean +/- 1 std")

    ax.set_title(title)
    ax.set_xlabel("k_max")
    ax.set_ylabel("p-value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_grouped_pvalue_vs_kmax_plot(
    per_run_rows: Iterable[Mapping[str, object]],
    summary_rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
    *,
    group_field: str,
    title: str,
    group_label_prefix: str = "",
) -> Path | None:
    rows = list(per_run_rows)
    summaries = list(summary_rows)
    if not rows or not summaries:
        return None

    grouped_rows: dict[str, list[Mapping[str, object]]] = {}
    grouped_summaries: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row[group_field]), []).append(row)
    for row in summaries:
        grouped_summaries.setdefault(str(row[group_field]), []).append(row)

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(grouped_summaries), 1)))

    for color, group_key in zip(colors, sorted(grouped_summaries, key=lambda value: float(value))):
        row_group = grouped_rows.get(group_key, [])
        summary_group = sorted(grouped_summaries[group_key], key=lambda row: int(row["k_max"]))
        if row_group:
            x = np.asarray([int(row["k_max"]) for row in row_group], dtype=np.int64)
            y = np.asarray([float(row["p_value"]) for row in row_group], dtype=np.float64)
            ax.scatter(x, y, alpha=0.25, color=color, edgecolors="none")

        summary_x = np.asarray([int(row["k_max"]) for row in summary_group], dtype=np.int64)
        summary_y = np.asarray([float(row["p_value_mean"]) for row in summary_group], dtype=np.float64)
        summary_std = np.asarray([float(row["p_value_std"]) for row in summary_group], dtype=np.float64)
        label = f"{group_label_prefix}{float(group_key):.6g}" if group_key not in {"", None} else group_label_prefix.rstrip("=")
        ax.plot(summary_x, summary_y, color=color, linewidth=1.5, marker="o", label=label)
        if np.any(np.isfinite(summary_std)):
            lower = np.clip(summary_y - summary_std, 0.0, 1.0)
            upper = np.clip(summary_y + summary_std, 0.0, 1.0)
            ax.fill_between(summary_x, lower, upper, alpha=0.08, color=color)

    ax.set_title(title)
    ax.set_xlabel("k_max")
    ax.set_ylabel("p-value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_boxplots_by_kmax(
    rows: Iterable[Mapping[str, object]],
    out_dir: str | Path,
) -> list[Path]:
    boxplot_rows = list(rows)
    if not boxplot_rows:
        return []

    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in boxplot_rows:
        grouped.setdefault(int(row["k_max"]), []).append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for k_max in sorted(grouped):
        k_rows = grouped[k_max]
        categories = sorted(
            {str(row["test_label"]) for row in k_rows},
            key=lambda label: (-1 if label == "existence" else 0, 0.0 if label == "existence" else float(label.split("=", 1)[1])),
        )
        values = [
            np.asarray([float(row["p_value"]) for row in k_rows if str(row["test_label"]) == category], dtype=np.float64)
            for category in categories
        ]
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(categories) * 1.2), 5))
        ax.boxplot(values, tick_labels=categories)
        ax.set_title(f"p-value Distribution by Test for k_max={k_max}")
        ax.set_xlabel("Test")
        ax.set_ylabel("p-value")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        out_path = out_dir / f"boxplots_kmax_{k_max}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


def save_repeat_pvalue_plot(
    rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
    *,
    title: str = "Existence p-value by Repeat",
) -> Path | None:
    ordered_rows = sorted(rows, key=lambda row: int(row["repeat_index"]))
    if not ordered_rows:
        return None

    repeat_index = np.asarray([int(row["repeat_index"]) for row in ordered_rows], dtype=np.int64)
    p_values = np.asarray([float(row["p_value"]) for row in ordered_rows], dtype=np.float64)

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(repeat_index, p_values, color="steelblue", linewidth=1.3, alpha=0.9)
    ax.scatter(repeat_index, p_values, color="crimson", s=45, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Repeat Index")
    ax.set_ylabel("p-value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_repeat_true_loss_plot(
    rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
    *,
    title: str = "True Loss by Repeat",
) -> Path | None:
    ordered_rows = sorted(rows, key=lambda row: int(row["repeat_index"]))
    if not ordered_rows:
        return None

    repeat_index = np.asarray([int(row["repeat_index"]) for row in ordered_rows], dtype=np.int64)
    losses = np.asarray([float(row["stat_true"]) for row in ordered_rows], dtype=np.float64)

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(repeat_index, losses, color="darkorange", linewidth=1.3, alpha=0.9)
    ax.scatter(repeat_index, losses, color="black", s=40, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Repeat Index")
    ax.set_ylabel("True Loss")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_repeat_null_boxplot(
    null_rows: Iterable[Mapping[str, object]],
    per_run_rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
    *,
    title: str = "Null Loss Distributions by Repeat",
) -> Path | None:
    ordered_run_rows = sorted(per_run_rows, key=lambda row: int(row["repeat_index"]))
    if not ordered_run_rows:
        return None

    grouped_nulls: dict[int, list[float]] = {}
    for row in null_rows:
        grouped_nulls.setdefault(int(row["repeat_index"]), []).append(float(row["null_loss"]))

    positions = [int(row["repeat_index"]) for row in ordered_run_rows]
    values = [np.asarray(grouped_nulls.get(index, []), dtype=np.float64) for index in positions]
    if not any(value.size for value in values):
        return None

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(positions) * 0.55), 5))
    ax.boxplot(values, positions=positions, widths=0.6, patch_artist=True)
    for patch in ax.artists:
        patch.set_facecolor("lightsteelblue")
        patch.set_alpha(0.7)

    true_losses = np.asarray([float(row["stat_true"]) for row in ordered_run_rows], dtype=np.float64)
    ax.scatter(positions, true_losses, color="crimson", s=40, zorder=3, label="True Loss")
    ax.plot(positions, true_losses, color="crimson", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Repeat Index")
    ax.set_ylabel("Loss")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_repeat_null_density_overlay(
    null_rows: Iterable[Mapping[str, object]],
    per_run_rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
    *,
    title: str = "Null Loss Density Overlay",
) -> Path | None:
    ordered_run_rows = sorted(per_run_rows, key=lambda row: int(row["repeat_index"]))
    if not ordered_run_rows:
        return None

    grouped_nulls: dict[int, list[float]] = {}
    for row in null_rows:
        grouped_nulls.setdefault(int(row["repeat_index"]), []).append(float(row["null_loss"]))
    all_nulls = [value for values in grouped_nulls.values() for value in values]
    if not all_nulls:
        return None

    values = np.asarray(all_nulls, dtype=np.float64)
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max <= value_min:
        value_max = value_min + 1e-6
    bins = np.linspace(value_min, value_max, 40)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(ordered_run_rows), 1)))

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for color, row in zip(colors, ordered_run_rows):
        repeat_index = int(row["repeat_index"])
        null_values = np.asarray(grouped_nulls.get(repeat_index, []), dtype=np.float64)
        if null_values.size == 0:
            continue
        density, edges = np.histogram(null_values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, density, color=color, linewidth=1.0, alpha=0.45)
        ax.axvline(float(row["stat_true"]), color=color, linestyle="--", linewidth=0.8, alpha=0.45)

    ax.set_title(title)
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_spearman_matrix_heatmap(
    matrix: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "True Isodepth Spearman Matrix",
) -> Path | None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        return None

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Repeat Index")
    ax.set_ylabel("Repeat Index")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_spearman_histogram(
    values: Iterable[float],
    out_path: str | Path,
    *,
    title: str = "Off-Diagonal Spearman Distribution",
) -> Path | None:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        return None

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(values_array, bins=20, color="lightsteelblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Spearman")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
