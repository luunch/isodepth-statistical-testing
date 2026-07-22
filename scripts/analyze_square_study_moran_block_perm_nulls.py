"""Aggregate block-permutation Moran-null artifacts for kernel-noise square studies.

Produces:
  - analysis/moran_block_perm_per_run.csv
  - analysis/moran_block_perm_summary_by_condition.csv
  - analysis/kernel_noise_block_moran_percentile_comparison.png
  - analysis/moran_block_perm_analysis_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PER_RUN_FIELDS = [
    "result_json_path",
    "run_name",
    "dataset_key",
    "kernel_distance_um",
    "delta",
    "data_seed",
    "test_method",
    "block_radius_um",
    "neighbor_radius_um",
    "n_perms",
    "true_mean_morans_i",
    "null_mean_of_means",
    "null_std_of_means",
    "z_score_like",
    "true_rank",
    "true_percentile",
    "top_rank",
]

SUMMARY_FIELDS = [
    "kernel_distance_um",
    "delta",
    "test_method",
    "block_radius_um",
    "neighbor_radius_um",
    "n_runs",
    "n_top_rank",
    "top_rank_rate",
    "median_true_percentile",
    "mean_true_percentile",
    "median_true_mean_morans_i",
    "mean_true_mean_morans_i",
    "mean_null_mean_of_means",
    "mean_null_std_of_means",
    "mean_z_score_like",
    "mean_gap_true_minus_null",
]


def _run_name_from_moran_stem(stem: str) -> str:
    pattern = r"_block_perm_moran_i_br\d+_nr\d+_n\d+$"
    return re.sub(pattern, "", stem)


def _seed_from_dataset_key(dataset_key: str) -> int:
    match = re.search(r"_seed(\d+)$", dataset_key)
    return int(match.group(1)) if match else -1


def _load_rows(study_root: Path) -> list[dict[str, Any]]:
    moran_json_paths = sorted((study_root / "runs").glob("*_block_r*/*_block_perm_moran_i_*.json"))
    rows: list[dict[str, Any]] = []
    for path in moran_json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_name = _run_name_from_moran_stem(path.stem)
        dataset_key = str(payload.get("dataset_key", ""))
        n_perms = int(payload.get("n_perms", 0))
        true_mean = float(payload.get("true_mean_morans_i", np.nan))
        null_means = np.asarray(payload.get("null_mean_morans_i_per_perm", []), dtype=np.float64)
        null_mean = float(np.mean(null_means)) if null_means.size else float("nan")
        null_std = float(np.std(null_means)) if null_means.size else float("nan")
        z_score_like = float((true_mean - null_mean) / null_std) if null_std > 0 else float("nan")
        true_rank = int(payload.get("true_rank", 0))
        if "true_percentile" in payload:
            true_percentile = float(payload["true_percentile"])
        elif n_perms > 0 and true_rank > 0:
            true_percentile = float(100.0 * true_rank / (n_perms + 1))
        else:
            true_percentile = float("nan")
        top_rank = int(true_rank == (n_perms + 1)) if n_perms > 0 else 0

        rows.append(
            {
                "result_json_path": str(path.resolve()),
                "run_name": run_name,
                "dataset_key": dataset_key,
                "kernel_distance_um": float(payload.get("kernel_rho_um", np.nan)),
                "delta": float(payload.get("delta", np.nan)),
                "data_seed": _seed_from_dataset_key(dataset_key),
                "test_method": "block_permutation_moran_null",
                "block_radius_um": float(payload.get("block_radius_um", np.nan)),
                "neighbor_radius_um": float(payload.get("neighbor_radius_um", np.nan)),
                "n_perms": n_perms,
                "true_mean_morans_i": true_mean,
                "null_mean_of_means": null_mean,
                "null_std_of_means": null_std,
                "z_score_like": z_score_like,
                "true_rank": true_rank,
                "true_percentile": true_percentile,
                "top_rank": top_rank,
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["kernel_distance_um"]),
            float(row["delta"]),
            str(row["test_method"]),
            float(row["block_radius_um"]),
            float(row["neighbor_radius_um"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (distance, delta, method, block_radius, neighbor_radius), group in sorted(grouped.items()):
        true_percentiles = [float(r["true_percentile"]) for r in group]
        true_means = [float(r["true_mean_morans_i"]) for r in group]
        null_means = [float(r["null_mean_of_means"]) for r in group]
        null_stds = [float(r["null_std_of_means"]) for r in group]
        z_scores = [float(r["z_score_like"]) for r in group]
        top_rank_count = int(sum(int(r["top_rank"]) for r in group))
        summary_rows.append(
            {
                "kernel_distance_um": distance,
                "delta": delta,
                "test_method": method,
                "block_radius_um": block_radius,
                "neighbor_radius_um": neighbor_radius,
                "n_runs": len(group),
                "n_top_rank": top_rank_count,
                "top_rank_rate": float(top_rank_count / len(group)) if group else 0.0,
                "median_true_percentile": float(np.median(np.asarray(true_percentiles, dtype=np.float64))),
                "mean_true_percentile": float(np.mean(np.asarray(true_percentiles, dtype=np.float64))),
                "median_true_mean_morans_i": float(np.median(np.asarray(true_means, dtype=np.float64))),
                "mean_true_mean_morans_i": float(np.mean(np.asarray(true_means, dtype=np.float64))),
                "mean_null_mean_of_means": float(np.mean(np.asarray(null_means, dtype=np.float64))),
                "mean_null_std_of_means": float(np.mean(np.asarray(null_stds, dtype=np.float64))),
                "mean_z_score_like": _nanmean(z_scores),
                "mean_gap_true_minus_null": float(
                    np.mean(np.asarray(true_means, dtype=np.float64) - np.asarray(null_means, dtype=np.float64))
                ),
            }
        )
    return summary_rows


def _plot_percentiles(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to plot")

    deltas = sorted({float(r["delta"]) for r in rows})
    distances = sorted({float(r["kernel_distance_um"]) for r in rows})
    block_radii = sorted({float(r["block_radius_um"]) for r in rows})
    fig, axes = plt.subplots(
        len(deltas),
        len(distances),
        figsize=(4.2 * len(distances), 3.4 * len(deltas)),
        squeeze=False,
        sharey=True,
    )
    cmap = plt.get_cmap("tab10")
    radius_to_color = {radius: cmap(i % 10) for i, radius in enumerate(block_radii)}
    x_positions = np.arange(len(block_radii), dtype=np.float64)
    x_labels = [f"B{int(r) if math.isclose(r, round(r)) else r:g}" for r in block_radii]

    for i, delta in enumerate(deltas):
        for j, distance in enumerate(distances):
            ax = axes[i, j]
            panel = [r for r in rows if float(r["delta"]) == delta and float(r["kernel_distance_um"]) == distance]
            for ridx, radius in enumerate(block_radii):
                subset = [r for r in panel if float(r["block_radius_um"]) == radius]
                if not subset:
                    continue
                y = np.asarray([float(r["true_percentile"]) for r in subset], dtype=np.float64)
                ax.scatter(
                    np.full(y.size, x_positions[ridx]),
                    y,
                    s=24,
                    color=radius_to_color[radius],
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.9,
                    label=f"Block r={radius:g} µm" if i == 0 and j == 0 else None,
                )
            ax.axhline(100.0, color="0.35", lw=1.0, ls="--")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlim(-0.5, len(block_radii) - 0.5)
            ax.set_ylim(-2.0, 102.0)
            if j == 0:
                ax.set_ylabel(f"δ={delta:g}\nTrue percentile", fontsize=9)
            if i == 0:
                ax.set_title(f"ρ={distance:g} µm", fontsize=10)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Block-permutation Moran null: true percentile vs null means", fontsize=12, y=0.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=min(4, len(labels)), frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("results/experiments/kernel_noise_square_study"),
        help="Study root containing runs/, datasets/, and analysis/ directories.",
    )
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    analysis_dir = study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(study_root)
    if not rows:
        raise SystemExit(f"No block-permutation Moran JSON files found under {study_root / 'runs'}")

    per_run_csv = analysis_dir / "moran_block_perm_per_run.csv"
    summary_csv = analysis_dir / "moran_block_perm_summary_by_condition.csv"
    comparison_plot = analysis_dir / "kernel_noise_block_moran_percentile_comparison.png"

    _write_csv(per_run_csv, PER_RUN_FIELDS, rows)
    summary_rows = _summarize(rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _plot_percentiles(rows, comparison_plot)

    neighbor_values = sorted({float(r["neighbor_radius_um"]) for r in rows})
    summary_payload = {
        "n_runs": len(rows),
        "neighbor_radius_um_fixed": neighbor_values[0] if len(neighbor_values) == 1 else None,
        "summary_by_condition": summary_rows,
        "artifacts": {
            "per_run_csv": str(per_run_csv.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "comparison_plot": str(comparison_plot.resolve()),
        },
    }
    summary_json = analysis_dir / "moran_block_perm_analysis_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {per_run_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {comparison_plot}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
