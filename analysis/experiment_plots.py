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
