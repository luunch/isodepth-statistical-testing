"""Analyze kernel_blurred_sweep runs: p-value swarm + Moran true-vs-perm scatter."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.studies.kernel_blurred_sweep.sweep import (
    DatasetKey,
    KernelBlurredSweepSpec,
    analysis_dir_for_spec,
    dataset_cache_path,
    load_kernel_blurred_sweep_spec,
)

VISIUM_MORAN_RADIUS_UM = 120.0
VISIUM_AUTOCORR_N_PERMS = 20

PER_RUN_FIELDS = [
    "result_json_path",
    "run_name",
    "dataset_key",
    "bandwidth_um",
    "data_seed",
    "test_method",
    "p_value",
    "reject",
    "stat_true",
    "runtime_sec",
]

SUMMARY_FIELDS = [
    "bandwidth_um",
    "test_method",
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


def _extract_rows_from_runs(
    spec: KernelBlurredSweepSpec, *, alpha: float
) -> list[dict[str, Any]]:
    runs_dir = spec.output_root / "runs"
    if not runs_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for result_path in sorted(runs_dir.rglob("*_result.json")):
        result = _read_result_json(result_path)
        config = dict(result.get("config", {}) or {})
        data_cfg = dict(config.get("data", {}) or {})
        test_cfg = dict(config.get("test", {}) or {})
        output_cfg = dict(config.get("output", {}) or {})
        kernel_cfg = dict(data_cfg.get("kernel", {}) or {})
        bandwidth_um = float(kernel_cfg.get("distance", np.nan))
        data_seed = int(test_cfg.get("seed", data_cfg.get("seed", -1)))
        test_method = str(result.get("method_name") or test_cfg.get("method", ""))
        p_value = float(result["p_value"])
        rows.append(
            {
                "result_json_path": str(result_path.resolve()),
                "run_name": str(output_cfg.get("run_name", result_path.parent.name)),
                "dataset_key": "",
                "bandwidth_um": bandwidth_um,
                "data_seed": data_seed,
                "test_method": test_method,
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
        key = (float(row["bandwidth_um"]), str(row["test_method"]))
        grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (bandwidth_um, test_method), group in sorted(grouped.items()):
        p_values = np.asarray([float(row["p_value"]) for row in group], dtype=np.float64)
        n_reject = int(np.sum(p_values <= alpha))
        summary_rows.append(
            {
                "bandwidth_um": bandwidth_um,
                "test_method": test_method,
                "alpha": alpha,
                "n_runs": len(group),
                "n_reject": n_reject,
                "median_p_value": float(np.median(p_values)),
                "mean_p_value": float(np.mean(p_values)),
                "reject_rate": float(n_reject / len(group)) if group else 0.0,
            }
        )
    return summary_rows


def save_kernel_blurred_pvalue_plot(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    spec: KernelBlurredSweepSpec,
    alpha: float,
) -> Path:
    """Same swarm style as the kernel-noise single-method plots; panels are bandwidths."""
    out_path = Path(out_path)
    bandwidths = list(spec.bandwidths_um)
    method_name = str(rows[0]["test_method"]) if rows else "method"
    fig, axes = plt.subplots(
        1,
        len(bandwidths),
        figsize=(4.0 * len(bandwidths), 3.2),
        squeeze=False,
        sharey=True,
    )
    for j, bandwidth in enumerate(bandwidths):
        ax = axes[0, j]
        p_values = np.asarray(
            [
                float(row["p_value"])
                for row in rows
                if float(row["bandwidth_um"]) == float(bandwidth)
            ],
            dtype=np.float64,
        )
        if p_values.size:
            jitter_y = (np.random.default_rng(0).random(p_values.size) - 0.5) * 0.012
            ax.scatter(
                np.zeros(p_values.size),
                p_values + jitter_y,
                s=22,
                alpha=0.85,
                color="#1f77b4",
                marker="o",
                edgecolors="white",
                linewidths=0.4,
            )
        ax.axhline(alpha, color="0.35", ls="--", lw=1.0)
        ax.set_xticks([0.0])
        ax.set_xticklabels([method_name], fontsize=8)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.02, 1.02)
        if j == 0:
            ax.set_ylabel("p-value", fontsize=9)
        ax.set_title(f"Gaussian blur σ={bandwidth:g} µm", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Gaussian-blurred {method_name} p-values "
        f"(α={alpha:g}, {len(spec.seeds)} seeds per blur σ)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _zscore_genes(A: np.ndarray) -> np.ndarray:
    a = np.asarray(A, dtype=np.float64)
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    return (a - mu) / np.maximum(sd, 1e-8)


def _spots_to_um(spots) -> np.ndarray:
    summary = spots.meta.get("visium_pseudobulk") or {}
    um_per_unit = float(summary.get("coordinate_um_per_unit", 1.0))
    if spots.meta.get("coordinate_standardization") == "zscore":
        mean = np.asarray(spots.meta["coord_mean"], dtype=np.float64)
        std = np.asarray(spots.meta["coord_std"], dtype=np.float64)
        S_native = np.asarray(spots.S, dtype=np.float64) * std + mean
    else:
        S_native = np.asarray(spots.S, dtype=np.float64)
    return S_native * um_per_unit


def _mean_moran_i(S_um: np.ndarray, A: np.ndarray, *, radius_um: float) -> float:
    from methods.moran import build_inverse_distance_weights, morans_i_per_gene

    weights = build_inverse_distance_weights(S_um, float(radius_um), allow_empty=True)
    if weights is None:
        return float("nan")
    W, s0 = weights
    return float(np.nanmean(morans_i_per_gene(W, s0, A)))


def _visium_true_vs_perm_moran(
    cache_path: Path,
    test_cfg: dict[str, Any],
    *,
    n_perms: int,
    radius_um: float,
) -> tuple[float, float]:
    import torch

    from data.schemas import run_config_from_mapping
    from experiments.core.dataset_cache import load_dataset_cache
    from methods.permutation import _build_permuted_coordinate_batch
    from methods.visium_pseudobulk import build_visium_pseudobulk_dataset

    mapping = {
        "data": {"source": "synthetic"},
        "test": {
            "method": "visium_pseudobulk",
            "n_perms": int(n_perms),
            "seed": int(test_cfg.get("seed", 0)),
            "visium_spot_distance_um": float(test_cfg.get("visium_spot_distance_um", 100.0)),
            "visium_spot_diameter_um": float(test_cfg.get("visium_spot_diameter_um", 55.0)),
            "coordinate_um_per_unit": float(test_cfg.get("coordinate_um_per_unit", 960.0)),
            "block_jitter": bool(test_cfg.get("block_jitter", False)),
        },
    }
    test = run_config_from_mapping(mapping).validate().test
    spots, _ = build_visium_pseudobulk_dataset(load_dataset_cache(cache_path), test)
    A = _zscore_genes(spots.A)
    S_um = _spots_to_um(spots)
    I_true = _mean_moran_i(S_um, A, radius_um=radius_um)
    s_batched, _ = _build_permuted_coordinate_batch(
        S_um.astype(np.float32),
        n_perms=int(n_perms),
        seed=int(test.seed),
        device=torch.device("cpu"),
    )
    I_null = [
        _mean_moran_i(s_batched[slot].detach().cpu().numpy(), A, radius_um=radius_um)
        for slot in range(1, s_batched.shape[0])
    ]
    return I_true, float(np.nanmean(I_null))


def _attach_visium_autocorr(
    spec: KernelBlurredSweepSpec,
    rows: list[dict[str, Any]],
    *,
    n_perms: int = VISIUM_AUTOCORR_N_PERMS,
    radius_um: float = VISIUM_MORAN_RADIUS_UM,
) -> list[dict[str, Any]]:
    for row in rows:
        key = DatasetKey(
            bandwidth_um=float(row["bandwidth_um"]),
            data_seed=int(row["data_seed"]),
        )
        cache_path = dataset_cache_path(spec, key)
        result = _read_result_json(Path(row["result_json_path"]))
        test_cfg = dict((result.get("config") or {}).get("test") or {})
        I_true, I_null = _visium_true_vs_perm_moran(
            cache_path, test_cfg, n_perms=n_perms, radius_um=radius_um
        )
        row["moran_radius_um"] = float(radius_um)
        row["true_mean_morans_i"] = I_true
        row["null_mean_morans_i"] = I_null
        row["dataset_key"] = key.slug
    return rows


def save_kernel_blurred_correlation_scatter(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    spec: KernelBlurredSweepSpec,
    radius_um: float,
) -> Path:
    """Same panel grid as the p-value plot; x-ticks are true vs whole-spot permutation."""
    out_path = Path(out_path)
    bandwidths = list(spec.bandwidths_um)
    x_labels = ["true", "perm"]
    x_positions = np.arange(len(x_labels), dtype=np.float64)
    colors = {"true": "#c0392b", "perm": "#2980b9"}
    fig, axes = plt.subplots(
        1,
        len(bandwidths),
        figsize=(4.0 * len(bandwidths), 3.2),
        squeeze=False,
        sharey=True,
    )
    all_vals = [
        float(row[key])
        for row in rows
        for key in ("true_mean_morans_i", "null_mean_morans_i")
        if np.isfinite(float(row.get(key, np.nan)))
    ]
    y_pad = 0.02
    if all_vals:
        y_lo = min(all_vals) - y_pad
        y_hi = max(all_vals) + y_pad
    else:
        y_lo, y_hi = -0.1, 0.1

    for j, bandwidth in enumerate(bandwidths):
        ax = axes[0, j]
        panel_rows = [
            row for row in rows if float(row["bandwidth_um"]) == float(bandwidth)
        ]
        for x_idx, label in enumerate(x_labels):
            key = "true_mean_morans_i" if label == "true" else "null_mean_morans_i"
            values = np.asarray(
                [float(row[key]) for row in panel_rows if np.isfinite(float(row.get(key, np.nan)))],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            jitter_y = (np.random.default_rng(0).random(values.size) - 0.5) * 0.004
            ax.scatter(
                np.full(values.size, x_positions[x_idx]),
                values + jitter_y,
                s=22,
                alpha=0.85,
                color=colors[label],
                marker="o",
                edgecolors="white",
                linewidths=0.4,
            )
        ax.axhline(0.0, color="0.35", ls="--", lw=1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlim(-0.5, len(x_labels) - 0.5)
        ax.set_ylim(y_lo, y_hi)
        if j == 0:
            ax.set_ylabel("mean Moran's I", fontsize=9)
        ax.set_title(f"Gaussian blur σ={bandwidth:g} µm", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Gaussian-blurred visium_pseudobulk spatial correlation "
        f"(Moran's I, r={radius_um:g} µm, {len(spec.seeds)} seeds; perm = whole-spot shuffle)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze_kernel_blurred_sweep(spec_path: str | Path, *, plots_only: bool = False) -> dict[str, Any]:
    spec = load_kernel_blurred_sweep_spec(spec_path)
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

    method_name = str(rows[0]["test_method"]) if rows else "method"
    plot_stem = (
        "kernel_blurred_visium_pseudobulk"
        if method_name == "visium_pseudobulk"
        else "kernel_blurred_pseudobulk"
    )
    plot_path = analysis_dir / f"{plot_stem}_pvalues.png"
    save_kernel_blurred_pvalue_plot(rows, plot_path, spec=spec, alpha=float(spec.alpha))

    payload: dict[str, Any] = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "per_run_csv": str(per_run_csv),
        "summary_csv": str(summary_csv),
        "pvalue_plot": str(plot_path),
        "n_rows": len(rows),
        "plots_only": plots_only,
    }
    method_name = str(rows[0]["test_method"]) if rows else ""
    if method_name == "visium_pseudobulk":
        rows = _attach_visium_autocorr(spec, rows)
        autocorr_fields = PER_RUN_FIELDS + [
            "dataset_key",
            "moran_radius_um",
            "true_mean_morans_i",
            "null_mean_morans_i",
        ]
        autocorr_csv = analysis_dir / "per_run_spatial_correlation.csv"
        _write_csv(autocorr_csv, autocorr_fields, rows)
        corr_plot = analysis_dir / f"{plot_stem}_correlations.png"
        save_kernel_blurred_correlation_scatter(
            rows, corr_plot, spec=spec, radius_um=VISIUM_MORAN_RADIUS_UM
        )
        payload["spatial_correlation_csv"] = str(autocorr_csv)
        payload["spatial_correlation_plot"] = str(corr_plot)
        print(f"Wrote {autocorr_csv}")
        print(f"Wrote {corr_plot}")
    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {per_run_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_path}")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze kernel_blurred_sweep outputs.")
    parser.add_argument(
        "--spec",
        required=True,
        help=(
            "Sweep spec. Use "
            "configs/experiments/kernel_blurred_visium_pseudobulk_sweep.json "
            "or configs/experiments/kernel_blurred_pseudobulk_sweep.json."
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate CSVs/plots from existing runs (same as default; kept for symmetry).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_kernel_blurred_sweep(args.spec, plots_only=args.plots_only)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
