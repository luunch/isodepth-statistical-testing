"""Aggregate kernel-noise single-method study runs and plot p-values."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.kernel_noise_msr_study import (
    KernelNoiseSingleMethodStudySpec,
    analysis_dir_for_spec,
    load_kernel_noise_single_method_study_spec,
    spot_distances_for_delta,
)

PER_RUN_FIELDS = [
    "result_json_path",
    "run_name",
    "dataset_key",
    "kernel_distance_um",
    "delta",
    "data_seed",
    "test_method",
    "msr_truncate_um",
    "msr_neighbor_radius_um",
    "spot_distance_um",
    "variant_label",
    "p_value",
    "reject",
    "stat_true",
    "runtime_sec",
]

SUMMARY_FIELDS = [
    "kernel_distance_um",
    "delta",
    "test_method",
    "msr_truncate_um",
    "msr_neighbor_radius_um",
    "spot_distance_um",
    "variant_label",
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


def _format_variant_label(
    *,
    truncate_um: float | None,
    neighbor_um: float | None,
    spot_distance_um: float | None = None,
) -> str:
    if spot_distance_um is not None:
        return f"spot{spot_distance_um:g}"
    if truncate_um is None:
        return "MSR"
    if neighbor_um is None:
        return f"T{truncate_um:g}"
    return f"T{truncate_um:g}|N{neighbor_um:g}"


def _parse_dataset_key_from_run_name(run_name: str) -> tuple[float, float, int]:
    """Parse (kernel_distance, delta, seed) from slug ``d{N}_delta{N}_seed{N}_...``."""
    import re
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


def _extract_rows_from_runs(
    spec: KernelNoiseSingleMethodStudySpec, *, alpha: float
) -> list[dict[str, Any]]:
    runs_dir = spec.output_root / "runs"
    if not runs_dir.exists():
        return []
    result_paths = sorted(runs_dir.rglob("*_result.json"))
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        result = _read_result_json(result_path)
        config = dict(result.get("config", {}) or {})
        data_cfg = dict(config.get("data", {}) or {})
        test_cfg = dict(config.get("test", {}) or {})
        output_cfg = dict(config.get("output", {}) or {})

        kernel_cfg = dict(data_cfg.get("kernel", {}) or {})
        kernel_distance_um = float(kernel_cfg.get("distance", np.nan))
        delta = float(data_cfg.get("delta", np.nan))
        data_seed = int(test_cfg.get("seed", data_cfg.get("seed", -1)))

        if np.isnan(kernel_distance_um) or np.isnan(delta):
            run_name_fallback = str(output_cfg.get("run_name", result_path.parent.name))
            fb_dist, fb_delta, fb_seed = _parse_dataset_key_from_run_name(run_name_fallback)
            if np.isnan(kernel_distance_um):
                kernel_distance_um = fb_dist
            if np.isnan(delta):
                delta = fb_delta
            if data_seed == -1:
                data_seed = fb_seed
        test_method = str(result.get("method_name") or test_cfg.get("method", ""))
        truncate_raw = test_cfg.get("msr_truncate_um", None)
        neighbor_raw = test_cfg.get("msr_neighbor_radius_um", None)
        spot_raw = test_cfg.get("bin_spot_distance_um", None)
        truncate_um = None if truncate_raw in (None, "") else float(truncate_raw)
        neighbor_um = None if neighbor_raw in (None, "") else float(neighbor_raw)
        spot_distance_um = None if spot_raw in (None, "") else float(spot_raw)
        variant_label = _format_variant_label(
            truncate_um=truncate_um,
            neighbor_um=neighbor_um,
            spot_distance_um=spot_distance_um,
        )
        p_value = float(result["p_value"])
        rows.append(
            {
                "result_json_path": str(result_path.resolve()),
                "run_name": str(output_cfg.get("run_name", result_path.parent.name)),
                "dataset_key": "",
                "kernel_distance_um": kernel_distance_um,
                "delta": delta,
                "data_seed": data_seed,
                "test_method": test_method,
                "msr_truncate_um": "" if truncate_um is None else truncate_um,
                "msr_neighbor_radius_um": "" if neighbor_um is None else neighbor_um,
                "spot_distance_um": "" if spot_distance_um is None else spot_distance_um,
                "variant_label": variant_label,
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
            row["msr_truncate_um"],
            row["msr_neighbor_radius_um"],
            row["spot_distance_um"],
            str(row["variant_label"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (distance, delta, test_method, truncate_um, neighbor_um, spot_distance_um, variant_label), group in sorted(
        grouped.items()
    ):
        p_values = np.asarray([float(row["p_value"]) for row in group], dtype=np.float64)
        n_reject = int(np.sum(p_values <= alpha))
        summary_rows.append(
            {
                "kernel_distance_um": distance,
                "delta": delta,
                "test_method": test_method,
                "msr_truncate_um": truncate_um,
                "msr_neighbor_radius_um": neighbor_um,
                "spot_distance_um": spot_distance_um,
                "variant_label": variant_label,
                "alpha": alpha,
                "n_runs": len(group),
                "n_reject": n_reject,
                "median_p_value": float(np.median(p_values)),
                "mean_p_value": float(np.mean(p_values)),
                "reject_rate": float(n_reject / len(group)) if group else 0.0,
            }
        )
    return summary_rows


def save_kernel_noise_msr_pvalue_plot(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    spec: KernelNoiseSingleMethodStudySpec,
    alpha: float,
) -> Path:
    """One summary figure in kernel_noise_study strip/swarm style.

    Panel grid shape is (len(deltas), len(kernel_distances_um)).
    """
    out_path = Path(out_path)
    deltas = list(spec.deltas)
    distances = list(spec.kernel_distances_um)
    method_name = str(rows[0]["test_method"]) if rows else "method"
    if spec.spot_distances_um:
        all_spot_labels = sorted(
            {
                f"spot{float(v):g}"
                for delta in spec.deltas
                for v in (spot_distances_for_delta(spec, float(delta)) or [])
            },
            key=lambda label: float(label.removeprefix("spot")),
        )
        x_labels = all_spot_labels
    else:
        variant_records = sorted(
            {
                (
                    str(row["variant_label"]),
                    float(row["msr_truncate_um"]) if row["msr_truncate_um"] != "" else float("inf"),
                    float(row["msr_neighbor_radius_um"]) if row["msr_neighbor_radius_um"] != "" else float("inf"),
                )
                for row in rows
            },
            key=lambda x: (x[1], x[2], x[0]),
        )
        x_labels = [record[0] for record in variant_records]
    x_positions = np.arange(len(x_labels), dtype=np.float64)
    color_palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]
    color_by_variant = {
        label: color_palette[idx % len(color_palette)] for idx, label in enumerate(x_labels)
    }

    fig, axes = plt.subplots(
        len(deltas),
        len(distances),
        figsize=(4.0 * len(distances), 3.2 * len(deltas)),
        squeeze=False,
        sharey=True,
    )

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
                subset = [row for row in panel_rows if str(row["variant_label"]) == label]
                if not subset:
                    continue
                p_values = np.asarray([float(row["p_value"]) for row in subset], dtype=np.float64)
                jitter_y = (np.random.default_rng(0).random(p_values.size) - 0.5) * 0.012
                ax.scatter(
                    np.full(p_values.size, x_positions[x_idx]),
                    p_values + jitter_y,
                    s=22,
                    alpha=0.85,
                    color=color_by_variant[label],
                    marker="o",
                    edgecolors="white",
                    linewidths=0.4,
                )
            ax.axhline(alpha, color="0.35", ls="--", lw=1.0)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlim(-0.5, len(x_labels) - 0.5)
            ax.set_ylim(-0.02, 1.02)
            if j == 0:
                ax.set_ylabel(f"δ={delta:g}\np-value", fontsize=9)
            if i == 0:
                ax.set_title(f"ρ={distance:g} µm", fontsize=10)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Kernel-noise {method_name} p-values on cached datasets "
        f"(α={alpha:g}, {len(spec.seeds)} seeds per cell)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze_kernel_noise_msr_study(spec_path: str | Path, *, plots_only: bool = False) -> dict[str, Any]:
    spec = load_kernel_noise_single_method_study_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = _extract_rows_from_runs(spec, alpha=float(spec.alpha))
    if not rows:
        raise FileNotFoundError(
            f"No completed runs found under: {spec.output_root / 'runs'}"
        )

    summary_rows = _summarize_rows(rows, alpha=float(spec.alpha))
    per_run_csv = analysis_dir / "per_run.csv"
    summary_csv = analysis_dir / "summary_by_condition.csv"
    _write_csv(per_run_csv, PER_RUN_FIELDS, rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)

    plot_path = analysis_dir / (
        "kernel_noise_binning_study_pvalues.png"
        if spec.spot_distances_um
        else "kernel_noise_msr_study_pvalues.png"
    )
    save_kernel_noise_msr_pvalue_plot(rows, plot_path, spec=spec, alpha=float(spec.alpha))

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "per_run_csv": str(per_run_csv),
        "summary_csv": str(summary_csv),
        "pvalue_plot": str(plot_path),
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
    parser = argparse.ArgumentParser(description="Analyze kernel-noise MSR-study outputs.")
    parser.add_argument("--spec", required=True, help="Path to configs/experiments/kernel_noise_msr_study.json")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate CSVs/plots from existing manifest (same as default; kept for symmetry).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_kernel_noise_msr_study(args.spec, plots_only=args.plots_only)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
