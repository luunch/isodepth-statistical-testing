"""Aggregate kernel-noise study runs and plot coordinate vs block p-values."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.kernel_noise_study import (
    KernelNoiseStudySpec,
    analysis_dir_for_spec,
    load_kernel_noise_study_spec,
    manifest_path_for_spec,
)

PER_RUN_FIELDS = [
    "result_json_path",
    "run_name",
    "dataset_key",
    "kernel_distance_um",
    "delta",
    "data_seed",
    "test_method",
    "block_radius_um",
    "p_value",
    "reject",
    "stat_true",
    "runtime_sec",
]

SUMMARY_FIELDS = [
    "kernel_distance_um",
    "delta",
    "test_method",
    "block_radius_um",
    "alpha",
    "n_runs",
    "n_reject",
    "median_p_value",
    "mean_p_value",
    "reject_rate",
]


def _read_result_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _extract_rows_from_manifest(manifest_path: Path, *, alpha: float) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in payload.get("runs", []):
        result_path = Path(str(entry["result_json_path"]))
        if not result_path.exists():
            continue
        result = _read_result_json(result_path)
        test_method = str(entry.get("test_method") or result.get("method_name"))
        block_radius = entry.get("block_radius_um")
        if block_radius is None and test_method == "block_permutation":
            block_radius = result.get("config", {}).get("test", {}).get("block_radius")
        p_value = float(result["p_value"])
        rows.append(
            {
                "result_json_path": str(result_path.resolve()),
                "run_name": str(entry.get("run_name") or result_path.parent.name),
                "dataset_key": str(entry.get("dataset_key") or ""),
                "kernel_distance_um": float(entry.get("kernel_distance_um", np.nan)),
                "delta": float(entry.get("delta", np.nan)),
                "data_seed": int(entry.get("data_seed", -1)),
                "test_method": test_method,
                "block_radius_um": "" if block_radius in (None, "") else float(block_radius),
                "p_value": p_value,
                "reject": int(p_value <= alpha),
                "stat_true": float(result.get("stat_true", np.nan)),
                "runtime_sec": float(result.get("runtime_sec", np.nan)),
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_rows(rows: list[dict[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["kernel_distance_um"]),
            float(row["delta"]),
            str(row["test_method"]),
            row["block_radius_um"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (distance, delta, test_method, block_radius), group in sorted(grouped.items()):
        p_values = np.asarray([float(row["p_value"]) for row in group], dtype=np.float64)
        n_reject = int(np.sum(p_values <= alpha))
        summary_rows.append(
            {
                "kernel_distance_um": distance,
                "delta": delta,
                "test_method": test_method,
                "block_radius_um": block_radius,
                "alpha": alpha,
                "n_runs": len(group),
                "n_reject": n_reject,
                "median_p_value": float(np.median(p_values)),
                "mean_p_value": float(np.mean(p_values)),
                "reject_rate": float(n_reject / len(group)) if group else 0.0,
            }
        )
    return summary_rows


def save_kernel_noise_pvalue_comparison_plot(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    spec: KernelNoiseStudySpec,
    alpha: float,
) -> Path:
    """One summary figure: 3×3 panels (delta × kernel distance).

    Each panel compares p-values across seeds for coordinate permutation vs three
    block radii (strip/swarm with α line).
    """
    out_path = Path(out_path)
    deltas = list(spec.deltas)
    distances = list(spec.kernel_distances_um)
    block_radii = list(spec.block_radii_um)

    fig, axes = plt.subplots(
        len(deltas),
        len(distances),
        figsize=(4.2 * len(distances), 3.4 * len(deltas)),
        squeeze=False,
        sharey=True,
    )

    method_styles = {
        "parallel_permutation": {"label": "Coord perm", "color": "#c0392b", "marker": "o"},
    }
    block_colors = ["#2980b9", "#27ae60", "#8e44ad"]
    for idx, radius in enumerate(block_radii):
        method_styles[f"block_{radius:g}"] = {
            "label": f"Block r={radius:g} µm",
            "color": block_colors[idx % len(block_colors)],
            "marker": "s",
        }

    x_labels: list[str] = ["Coord"] + [f"B{int(r) if r == int(r) else r:g}" for r in block_radii]
    x_positions = np.arange(len(x_labels), dtype=np.float64)

    for i, delta in enumerate(deltas):
        for j, distance in enumerate(distances):
            ax = axes[i, j]
            panel_rows = [
                row
                for row in rows
                if float(row["delta"]) == float(delta)
                and float(row["kernel_distance_um"]) == float(distance)
            ]
            for x_idx, label in enumerate(x_labels):
                if label == "Coord":
                    subset = [row for row in panel_rows if row["test_method"] == "parallel_permutation"]
                    style = method_styles["parallel_permutation"]
                else:
                    radius = block_radii[x_idx - 1]
                    subset = [
                        row
                        for row in panel_rows
                        if row["test_method"] == "block_permutation"
                        and float(row["block_radius_um"]) == float(radius)
                    ]
                    style = method_styles[f"block_{radius:g}"]
                if not subset:
                    continue
                p_values = np.asarray([float(row["p_value"]) for row in subset], dtype=np.float64)
                jitter = (np.random.default_rng(0).random(p_values.size) - 0.5) * 0.12
                ax.scatter(
                    np.full(p_values.size, x_positions[x_idx]) + jitter,
                    p_values,
                    s=22,
                    alpha=0.85,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolors="white",
                    linewidths=0.4,
                    label=style["label"] if i == 0 and j == 0 else None,
                )

            ax.axhline(alpha, color="0.35", ls="--", lw=1.0)
            ax.set_xticks(x_positions, x_labels, fontsize=8)
            ax.set_xlim(-0.5, len(x_labels) - 0.5)
            ax.set_ylim(-0.02, 1.02)
            if j == 0:
                ax.set_ylabel(f"δ={delta:g}\np-value", fontsize=9)
            if i == 0:
                ax.set_title(f"ρ={distance:g} µm", fontsize=10)
            ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=9, frameon=False)
    fig.suptitle(
        "Coordinate vs block permutation p-values on cached kernel-noise datasets "
        f"(α={alpha:g}, {len(spec.seeds)} seeds per cell)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze_kernel_noise_study(spec_path: str | Path, *, plots_only: bool = False) -> dict[str, Any]:
    spec = load_kernel_noise_study_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = _extract_rows_from_manifest(manifest_path_for_spec(spec), alpha=float(spec.alpha))
    if not rows:
        raise FileNotFoundError(
            f"No completed runs found. Expected manifest at {manifest_path_for_spec(spec)} "
            "with result JSON paths."
        )

    summary_rows = _summarize_rows(rows, alpha=float(spec.alpha))
    per_run_csv = analysis_dir / "per_run.csv"
    summary_csv = analysis_dir / "summary_by_condition.csv"
    _write_csv(per_run_csv, PER_RUN_FIELDS, rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)

    plot_path = analysis_dir / "kernel_noise_study_pvalue_comparison.png"
    save_kernel_noise_pvalue_comparison_plot(rows, plot_path, spec=spec, alpha=float(spec.alpha))

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "per_run_csv": str(per_run_csv),
        "summary_csv": str(summary_csv),
        "comparison_plot": str(plot_path),
        "n_rows": len(rows),
        "plots_only": plots_only,
    }
    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {per_run_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_path}")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze kernel-noise study outputs.")
    parser.add_argument("--spec", required=True, help="Path to configs/experiments/kernel_noise_study.json")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate CSVs/plots from existing manifest (same as default; kept for symmetry).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_kernel_noise_study(args.spec, plots_only=args.plots_only)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
