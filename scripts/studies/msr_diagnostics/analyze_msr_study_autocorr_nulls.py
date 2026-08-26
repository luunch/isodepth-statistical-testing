"""Aggregate MSR-surrogate autocorrelation-null artifacts for kernel-noise MSR studies."""
from __future__ import annotations

import argparse
import csv
import json
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
    "truncate_um",
    "msr_radius_um",
    "calibration_um",
    "variant_label",
    "n_perms",
    "true_half_max_um",
    "null_mean_half_max_um",
    "null_std_half_max_um",
    "true_rank",
    "true_percentile",
]


def _run_name_from_stem(stem: str) -> str:
    return re.sub(r"_msr_autocorr_t\d+_n\d+$", "", stem)


def _parse_dataset_key_from_run_name(run_name: str) -> tuple[float, float, int]:
    m = re.match(r"d(\d+(?:p\d+)?)_delta(\d+(?:p\d+)?)_seed(\d+)", run_name)
    if not m:
        return float("nan"), float("nan"), -1

    def _unslug(s: str) -> float:
        return float(s.replace("p", ".").replace("m", "-"))

    return _unslug(m.group(1)), _unslug(m.group(2)), int(m.group(3))


def _format_variant_label(truncate_um: float, msr_radius_um: float = 30.0) -> str:
    return f"T{truncate_um:g}|N{msr_radius_um:g}"


def _load_rows(study_root: Path) -> list[dict[str, Any]]:
    paths = sorted((study_root / "runs").rglob("*_msr_autocorr_*.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_name = _run_name_from_stem(path.stem)
        dataset_key = str(payload.get("dataset_key", ""))
        truncate_um = float(payload.get("truncate_um", np.nan))
        msr_radius = float(payload.get("msr_radius_um", 30.0))
        cal_um = float(payload.get("calibration_um", np.nan))
        n_perms = int(payload.get("n_perms", 0))
        true_half = float(payload.get("true_half_max_um", np.nan))
        null_halves = np.asarray(payload.get("null_half_max_um_per_perm", []), dtype=np.float64)
        null_mean = float(null_halves.mean()) if null_halves.size else float("nan")
        null_std = float(null_halves.std()) if null_halves.size else float("nan")
        true_rank = int(payload.get("true_rank", 0))
        true_pct = float(payload.get("true_percentile", np.nan))

        kernel_rho = float(payload.get("kernel_rho_um", np.nan))
        delta_val = float(payload.get("delta", np.nan))
        fb_dist, fb_delta, fb_seed = _parse_dataset_key_from_run_name(run_name)
        if np.isnan(kernel_rho):
            kernel_rho = fb_dist
        if np.isnan(delta_val):
            delta_val = fb_delta
        data_seed = fb_seed

        rows.append(
            {
                "result_json_path": str(path.resolve()),
                "run_name": run_name,
                "dataset_key": dataset_key,
                "kernel_distance_um": kernel_rho,
                "delta": delta_val,
                "data_seed": data_seed,
                "truncate_um": truncate_um,
                "msr_radius_um": msr_radius,
                "calibration_um": cal_um,
                "variant_label": _format_variant_label(truncate_um, msr_radius),
                "n_perms": n_perms,
                "true_half_max_um": true_half,
                "null_mean_half_max_um": null_mean,
                "null_std_half_max_um": null_std,
                "true_rank": true_rank,
                "true_percentile": true_pct,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_percentiles(rows: list[dict[str, Any]], out_path: Path) -> None:
    deltas = sorted({float(r["delta"]) for r in rows})
    distances = sorted({float(r["kernel_distance_um"]) for r in rows})
    variant_records = sorted(
        {(str(r["variant_label"]), float(r["truncate_um"])) for r in rows},
        key=lambda x: x[1],
    )
    x_labels = [rec[0] for rec in variant_records]
    x_positions = np.arange(len(x_labels), dtype=np.float64)
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, axes = plt.subplots(len(deltas), len(distances), figsize=(4.2 * len(distances), 3.4 * len(deltas)), squeeze=False)
    cal_values = sorted({float(r["calibration_um"]) for r in rows if not np.isnan(float(r["calibration_um"]))})
    cal_note = f"cal={cal_values[0]:g} µm" if len(cal_values) == 1 else "mixed cal"

    for i, delta in enumerate(deltas):
        for j, distance in enumerate(distances):
            ax = axes[i, j]
            panel = [
                r for r in rows
                if float(r["delta"]) == delta and float(r["kernel_distance_um"]) == distance
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
                    color=colors[x_idx % len(colors)],
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.9,
                )
            ax.axhline(50.0, color="0.35", lw=1.0, ls="--")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlim(-0.5, len(x_labels) - 0.5)
            ax.set_ylim(-2.0, 102.0)
            if j == 0:
                ax.set_ylabel(f"δ={delta:g}\nTrue percentile", fontsize=9)
            if i == 0:
                ax.set_title(f"ρ={distance:g} µm", fontsize=10)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"MSR autocorr null: true half-max length percentile ({cal_note})",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_example_curves(study_root: Path, rows: list[dict[str, Any]], out_path: Path) -> None:
    """Overlay true vs null-mean c(d) for one run per variant (first delta/seed)."""
    if not rows:
        return
    delta = sorted({float(r["delta"]) for r in rows})[0]
    seed = sorted({int(r["data_seed"]) for r in rows if int(r["data_seed"]) >= 0})[0]
    panel = [
        r for r in rows if float(r["delta"]) == delta and int(r["data_seed"]) == seed
    ]
    if not panel:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for idx, row in enumerate(sorted(panel, key=lambda r: float(r["truncate_um"]))):
        payload = json.loads(Path(row["result_json_path"]).read_text(encoding="utf-8"))
        centers = np.asarray(payload["true_centers_um"], dtype=np.float64)
        true_c = np.asarray(payload["true_c_hat"], dtype=np.float64)
        color = ["#1f77b4", "#2ca02c", "#9467bd"][idx % 3]
        label = str(row["variant_label"])
        ax.plot(centers, true_c, color=color, lw=2.0, ls="--", label=f"true ({label})")
        # Recompute null mean curve from stored per-null diags if present — use half-max only in JSON.
        # For overlay, read one representative null from the PNG companion by re-loading not stored;
        # store null mean c in future. For now plot true only per variant + note.
    ax.set_xlabel("Distance (µm)")
    ax.set_ylabel("Pooled c(d)")
    ax.set_title(f"True c(d) by MSR variant (δ={delta:g}, seed={seed}) — see per-run PNGs for null bands")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    analysis_dir = study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(study_root)
    if not rows:
        raise SystemExit(f"No MSR autocorr JSON files found under {study_root / 'runs'}")

    per_run_csv = analysis_dir / "msr_autocorr_per_run.csv"
    comparison_plot = analysis_dir / "kernel_noise_msr_autocorr_percentile_comparison.png"
    _write_csv(per_run_csv, rows)
    _plot_percentiles(rows, comparison_plot)

    # Flag identical T30/T60 pairs (same dataset seed)
    dup_pairs = 0
    by_key: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["kernel_distance_um"]), float(row["delta"]), int(row["data_seed"]))
        by_key.setdefault(key, []).append(row)
    for group in by_key.values():
        if len(group) < 2:
            continue
        null_means = [float(r["null_mean_half_max_um"]) for r in group]
        if len(set(f"{v:.6g}" for v in null_means)) == 1:
            dup_pairs += 1

    summary = {
        "n_runs": len(rows),
        "n_dataset_keys_with_identical_null_means_across_variants": dup_pairs,
        "artifacts": {
            "per_run_csv": str(per_run_csv.resolve()),
            "comparison_plot": str(comparison_plot.resolve()),
        },
    }
    summary_path = analysis_dir / "msr_autocorr_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {per_run_csv}")
    print(f"Wrote {comparison_plot}")
    print(f"Wrote {summary_path}")
    if dup_pairs:
        print(
            f"WARNING: {dup_pairs} dataset(s) have identical null autocorr summaries across "
            "variants — check msr_calibration_um (use fixed cal, not cal=truncate)."
        )


if __name__ == "__main__":
    main()
