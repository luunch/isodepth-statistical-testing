"""Aggregate MSR-surrogate Moran-null artifacts for kernel-noise MSR studies.

Produces:
  - analysis/moran_msr_per_run.csv
  - analysis/moran_msr_summary_by_condition.csv
  - analysis/kernel_noise_msr_moran_percentile_comparison.png
  - analysis/moran_msr_analysis_summary.json
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
    "truncate_um",
    "msr_radius_um",
    "neighbor_radius_um",
    "variant_label",
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
    "truncate_um",
    "msr_radius_um",
    "neighbor_radius_um",
    "variant_label",
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
    pattern = r"_msr_moran_i_t\d+_nr\d+_n\d+$"
    return re.sub(pattern, "", stem)


def _seed_from_dataset_key(dataset_key: str) -> int:
    match = re.search(r"_seed(\d+)$", dataset_key)
    return int(match.group(1)) if match else -1


def _parse_dataset_key_from_run_name(run_name: str) -> tuple[float, float, int]:
    m = re.match(
        r"d(\d+(?:p\d+)?)"
        r"_delta(\d+(?:p\d+)?)"
        r"_seed(\d+)",
        run_name,
    )
    if not m:
        return float("nan"), float("nan"), -1

    def _unslug(s: str) -> float:
        return float(s.replace("p", ".").replace("m", "-"))

    return _unslug(m.group(1)), _unslug(m.group(2)), int(m.group(3))


def _format_variant_label(truncate_um: float, msr_radius_um: float) -> str:
    return f"T{truncate_um:g}|N{msr_radius_um:g}"


def _load_rows(study_root: Path) -> list[dict[str, Any]]:
    moran_json_paths = sorted((study_root / "runs").rglob("*_msr_moran_i_*.json"))
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

        kernel_rho = float(payload.get("kernel_rho_um", np.nan))
        delta_val = float(payload.get("delta", np.nan))
        data_seed = _seed_from_dataset_key(dataset_key)
        if np.isnan(kernel_rho) or np.isnan(delta_val) or data_seed == -1:
            fb_dist, fb_delta, fb_seed = _parse_dataset_key_from_run_name(run_name)
            if np.isnan(kernel_rho):
                kernel_rho = fb_dist
            if np.isnan(delta_val):
                delta_val = fb_delta
            if data_seed == -1:
                data_seed = fb_seed

        truncate_um = float(payload.get("truncate_um", np.nan))
        msr_radius = float(payload.get("msr_radius_um", np.nan))
        neighbor_radius = float(payload.get("neighbor_radius_um", np.nan))
        variant_label = _format_variant_label(truncate_um, msr_radius)

        rows.append(
            {
                "result_json_path": str(path.resolve()),
                "run_name": run_name,
                "dataset_key": dataset_key,
                "kernel_distance_um": kernel_rho,
                "delta": delta_val,
                "data_seed": data_seed,
                "test_method": "msr_moran_null",
                "truncate_um": truncate_um,
                "msr_radius_um": msr_radius,
                "neighbor_radius_um": neighbor_radius,
                "variant_label": variant_label,
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


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["kernel_distance_um"]),
            float(row["delta"]),
            str(row["test_method"]),
            float(row["truncate_um"]),
            float(row["msr_radius_um"]),
            float(row["neighbor_radius_um"]),
            str(row["variant_label"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (distance, delta, method, trunc, msr_r, nr, vlabel), group in sorted(grouped.items()):
        true_percentiles = np.asarray([float(r["true_percentile"]) for r in group], dtype=np.float64)
        true_means = np.asarray([float(r["true_mean_morans_i"]) for r in group], dtype=np.float64)
        null_means = np.asarray([float(r["null_mean_of_means"]) for r in group], dtype=np.float64)
        null_stds = np.asarray([float(r["null_std_of_means"]) for r in group], dtype=np.float64)
        z_scores = np.asarray([float(r["z_score_like"]) for r in group], dtype=np.float64)
        top_rank_count = int(sum(int(r["top_rank"]) for r in group))
        summary_rows.append(
            {
                "kernel_distance_um": distance,
                "delta": delta,
                "test_method": method,
                "truncate_um": trunc,
                "msr_radius_um": msr_r,
                "neighbor_radius_um": nr,
                "variant_label": vlabel,
                "n_runs": len(group),
                "n_top_rank": top_rank_count,
                "top_rank_rate": float(top_rank_count / len(group)) if group else 0.0,
                "median_true_percentile": float(np.nanmedian(true_percentiles)),
                "mean_true_percentile": float(np.nanmean(true_percentiles)),
                "median_true_mean_morans_i": float(np.nanmedian(true_means)),
                "mean_true_mean_morans_i": float(np.nanmean(true_means)),
                "mean_null_mean_of_means": float(np.nanmean(null_means)),
                "mean_null_std_of_means": float(np.nanmean(null_stds)),
                "mean_z_score_like": float(np.nanmean(z_scores)),
                "mean_gap_true_minus_null": float(np.nanmean(true_means - null_means)),
            }
        )
    return summary_rows


def _plot_percentiles(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to plot")

    deltas = sorted({float(r["delta"]) for r in rows})
    distances = sorted({float(r["kernel_distance_um"]) for r in rows})
    variant_records = sorted(
        {
            (
                str(r["variant_label"]),
                float(r["truncate_um"]),
                float(r["msr_radius_um"]),
            )
            for r in rows
        },
        key=lambda x: (x[1], x[2], x[0]),
    )
    x_labels = [rec[0] for rec in variant_records]
    x_positions = np.arange(len(x_labels), dtype=np.float64)
    color_palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]
    color_by_variant = {
        label: color_palette[idx % len(color_palette)] for idx, label in enumerate(x_labels)
    }

    fig, axes = plt.subplots(
        len(deltas),
        len(distances),
        figsize=(4.2 * len(distances), 3.4 * len(deltas)),
        squeeze=False,
        sharey=True,
    )

    for i, delta in enumerate(deltas):
        for j, distance in enumerate(distances):
            ax = axes[i, j]
            panel = [
                r for r in rows
                if float(r["delta"]) == delta
                and float(r["kernel_distance_um"]) == distance
            ]
            for x_idx, label in enumerate(x_labels):
                subset = [r for r in panel if str(r["variant_label"]) == label]
                if not subset:
                    continue
                y = np.asarray([float(r["true_percentile"]) for r in subset], dtype=np.float64)
                ax.scatter(
                    np.full(y.size, x_positions[x_idx]),
                    y,
                    s=24,
                    color=color_by_variant[label],
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.9,
                    label=f"MSR {label}" if i == 0 and j == 0 else None,
                )
            ax.axhline(100.0, color="0.35", lw=1.0, ls="--")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlim(-0.5, len(x_labels) - 0.5)
            ax.set_ylim(-2.0, 102.0)
            if j == 0:
                ax.set_ylabel(f"δ={delta:g}\nTrue percentile", fontsize=9)
            if i == 0:
                ax.set_title(f"ρ={distance:g} µm", fontsize=10)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle("MSR Moran null: true percentile vs null means", fontsize=12, y=0.98)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    if handles:
        fig.legend(
            handles, labels_, loc="upper center",
            bbox_to_anchor=(0.5, 0.03), ncol=min(4, len(labels_)), frameon=False,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="Study root containing runs/ and analysis/ directories.",
    )
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    analysis_dir = study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(study_root)
    if not rows:
        raise SystemExit(f"No MSR Moran JSON files found under {study_root / 'runs'}")

    per_run_csv = analysis_dir / "moran_msr_per_run.csv"
    summary_csv = analysis_dir / "moran_msr_summary_by_condition.csv"
    comparison_plot = analysis_dir / "kernel_noise_msr_moran_percentile_comparison.png"

    _write_csv(per_run_csv, PER_RUN_FIELDS, rows)
    summary_rows = _summarize(rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _plot_percentiles(rows, comparison_plot)

    neighbor_values = sorted({float(r["neighbor_radius_um"]) for r in rows})
    summary_payload: dict[str, Any] = {
        "n_runs": len(rows),
        "neighbor_radius_um_fixed": neighbor_values[0] if len(neighbor_values) == 1 else None,
        "summary_by_condition": summary_rows,
        "artifacts": {
            "per_run_csv": str(per_run_csv.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "comparison_plot": str(comparison_plot.resolve()),
        },
    }
    summary_json = analysis_dir / "moran_msr_analysis_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {per_run_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {comparison_plot}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
