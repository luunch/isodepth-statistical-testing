"""
MSR-surrogate null vs true pooled spatial autocorrelation c(d) (kernel noise).

For a cached kernel-noise dataset, estimates the empirical pooled covariance-vs-
distance curve on the true expression matrix and on each MSR surrogate (coordinates
fixed).  Outputs:

  - c(d) overlay: true vs null mean ± 1 SD band (each null is its own c_hat curve)
  - histogram of null half-max autocorrelation lengths vs true

Usage:
  python scripts/kernel_noise_msr_autocorr_null.py \\
      --cache results/experiments/.../datasets/d15_delta0p1_seed0.npz \\
      --n-perms 99 --truncate-um 30 --msr-radius 30 \\
      --calibration-um 100 --coordinate-um-per-unit 960 \\
      --out runs/.../d15_delta0p1_seed0_joint_truncated_msr_msr_autocorr_t30_n99.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.core.paths import repo_root
REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from experiments.kernel_noise_msr_study import load_dataset_cache
from scripts.studies.msr_diagnostics.msr_moran_null import (
    DEFAULT_SEED,
    _effective_calibration_um,
    _resolve_um_per_unit,
    _true_coords_native,
)
from scripts.studies.msr_diagnostics.msr_null_smoke_test import estimate_pooled_autocorr_length_um

AUTOCORR_EST_SEED = 0


def _pooled_autocorr_on_coords_um(
    S_um: np.ndarray,
    A: np.ndarray,
    *,
    seed: int,
) -> tuple[float, dict]:
    """Estimate c(d) with coordinates already in microns (um_per_unit=1)."""
    return estimate_pooled_autocorr_length_um(
        S_um,
        A,
        um_per_unit=1.0,
        seed=int(seed),
    )


def plot_autocorr_results(
    *,
    dataset_key: str,
    kernel_rho_um: float,
    delta: float,
    truncate_um: float,
    msr_radius_um: float,
    calibration_um: float,
    true_diag: dict,
    null_diags: list[dict],
    out_path: Path,
    cal_warning: str | None,
) -> None:
    true_half = float(true_diag["half_max_um"])
    null_halves = np.asarray([float(d["half_max_um"]) for d in null_diags], dtype=np.float64)
    n_perms = int(null_halves.size)
    rank = int(np.sum(null_halves < true_half)) + 1
    pct = 100.0 * rank / (n_perms + 1)

    centers = np.asarray(true_diag["centers_um"], dtype=np.float64)
    true_c = np.asarray(true_diag["c_hat"], dtype=np.float64)
    null_c_stack = np.stack(
        [np.asarray(d["c_hat"], dtype=np.float64) for d in null_diags],
        axis=0,
    )
    null_mean = null_c_stack.mean(axis=0)
    null_std = null_c_stack.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    title = (
        f"Pooled spatial autocorrelation — MSR null vs true\n"
        f"{dataset_key}  ρ={kernel_rho_um:.0f} µm  δ={delta:g}  "
        f"T={truncate_um:g} µm  msr_r={msr_radius_um:g} µm  cal={calibration_um:g} µm  "
        f"n_null={n_perms}"
    )
    if cal_warning:
        title += "\n(warn: cal=truncate — T variants may be identical)"
    fig.suptitle(title, fontsize=10, fontweight="bold")

    ax = axes[0]
    ax.fill_between(
        centers,
        null_mean - null_std,
        null_mean + null_std,
        color="#2980b9",
        alpha=0.25,
        label="null mean ± 1 SD",
    )
    ax.plot(centers, null_mean, color="#2980b9", lw=1.8, label="null mean c(d)")
    ax.plot(centers, true_c, color="crimson", lw=2.2, ls="--", label="true c(d)")
    ax.axhline(0.5 * float(true_diag["c_max"]), color="gray", ls=":", lw=1, alpha=0.7)
    ax.axvline(true_half, color="crimson", ls=":", lw=1.2, label=f"true half={true_half:.1f} µm")
    if kernel_rho_um > 0:
        ax.axvline(kernel_rho_um, color="green", ls=":", lw=1.2, alpha=0.8, label=f"kernel ρ={kernel_rho_um:g} µm")
    ax.set_xlabel("Distance (µm)", fontsize=9)
    ax.set_ylabel("Pooled c(d)", fontsize=9)
    ax.set_title("Empirical c(d) on expression (true vs MSR null band)", fontsize=9)
    ax.legend(fontsize=7.5, loc="best")
    ax.tick_params(labelsize=8)

    ax2 = axes[1]
    bins = max(12, min(35, n_perms // 3))
    ax2.hist(
        null_halves,
        bins=bins,
        color="#2980b9",
        alpha=0.72,
        edgecolor="k",
        linewidth=0.4,
        label=f"null half-max (n={n_perms})",
    )
    ax2.axvline(true_half, color="crimson", lw=2.2, ls="--", label=f"true = {true_half:.2f} µm")
    ax2.axvline(float(null_halves.mean()), color="k", lw=1.2, ls=":", alpha=0.8,
                label=f"null avg = {null_halves.mean():.2f} ± {null_halves.std():.2f} µm")
    ax2.set_xlabel("Half-max autocorrelation length (µm)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title(f"Half-max length vs null\nrank {rank}/{n_perms + 1} ({pct:.1f} percentile)", fontsize=9)
    ax2.legend(fontsize=7.5, loc="best")
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(
    cache_path: Path,
    *,
    n_perms: int,
    truncate_um: float,
    msr_radius_um: float,
    seed: int,
    coordinate_um_per_unit: float | None,
    calibration_um: float | None,
    out_path: Path,
) -> dict:
    from methods.msr import build_joint_truncated_msr_surrogates

    cal_um, cal_warning = _effective_calibration_um(truncate_um, calibration_um)
    if cal_warning:
        warnings.warn(cal_warning, stacklevel=2)
        print(f"WARNING: {cal_warning}", flush=True)

    ds = load_dataset_cache(cache_path)
    S, A, meta = ds.S, ds.A, ds.meta
    um_per_unit = _resolve_um_per_unit(meta, coordinate_um_per_unit)
    S_native = _true_coords_native(S, meta)
    S_um = np.asarray(S_native, dtype=np.float64) * um_per_unit

    kernel_rho = float((meta.get("kernel") or {}).get("distance", np.nan))
    delta = float(meta.get("delta", np.nan))
    dataset_key = cache_path.stem

    print(
        f"Dataset {dataset_key}: {S.shape[0]} cells × {A.shape[1]} genes  "
        f"ρ={kernel_rho:.0f} µm  δ={delta:g}",
        flush=True,
    )
    print(
        f"MSR autocorr: trunc={truncate_um:.0f} µm  msr_r={msr_radius_um:.0f} µm  "
        f"cal={cal_um:.0f} µm  n_perms={n_perms}  seed={seed}",
        flush=True,
    )

    t0 = time.time()
    true_half, true_diag = _pooled_autocorr_on_coords_um(
        S_um, A, seed=AUTOCORR_EST_SEED,
    )
    print(f"  True half-max AC length: {true_half:.2f} µm", flush=True)

    surrogates = build_joint_truncated_msr_surrogates(
        S_um,
        A,
        n_perms,
        seed,
        radius=msr_radius_um,
        truncate_scale_um=truncate_um,
        calibration_um=cal_um,
    )
    print(f"  Built {n_perms} MSR surrogates in {time.time() - t0:.1f}s", flush=True)

    null_diags: list[dict] = []
    null_halves: list[float] = []
    t1 = time.time()
    for i in range(n_perms):
        half_um, diag = _pooled_autocorr_on_coords_um(
            S_um,
            surrogates[i],
            seed=AUTOCORR_EST_SEED + 1 + i,
        )
        null_halves.append(float(half_um))
        null_diags.append(diag)
    print(f"  Computed null c(d) curves in {time.time() - t1:.1f}s", flush=True)

    null_halves_arr = np.asarray(null_halves, dtype=np.float64)
    rank = int(np.sum(null_halves_arr < true_half)) + 1

    plot_autocorr_results(
        dataset_key=dataset_key,
        kernel_rho_um=kernel_rho,
        delta=delta,
        truncate_um=truncate_um,
        msr_radius_um=msr_radius_um,
        calibration_um=cal_um,
        true_diag=true_diag,
        null_diags=null_diags,
        out_path=out_path,
        cal_warning=cal_warning,
    )

    summary = {
        "dataset_key": dataset_key,
        "cache_path": str(cache_path.resolve()),
        "kernel_rho_um": kernel_rho,
        "delta": delta,
        "truncate_um": truncate_um,
        "msr_radius_um": msr_radius_um,
        "calibration_um": cal_um,
        "calibration_warning": cal_warning,
        "coordinate_um_per_unit": um_per_unit,
        "n_perms": n_perms,
        "seed": seed,
        "true_half_max_um": true_half,
        "true_centers_um": np.asarray(true_diag["centers_um"], dtype=np.float64).tolist(),
        "true_c_hat": np.asarray(true_diag["c_hat"], dtype=np.float64).tolist(),
        "null_half_max_um_per_perm": null_halves_arr.tolist(),
        "null_mean_half_max_um": float(null_halves_arr.mean()),
        "null_std_half_max_um": float(null_halves_arr.std()),
        "true_rank": rank,
        "true_percentile": 100.0 * rank / (n_perms + 1),
        "plot_path": str(out_path.resolve()),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 65, flush=True)
    print(f"  true half-max AC (µm)        {true_half:10.3f}", flush=True)
    print(f"  null mean ± std (half-max)   {null_halves_arr.mean():10.3f} ± {null_halves_arr.std():.3f}", flush=True)
    print(f"  true rank vs nulls           {rank}/{n_perms + 1}", flush=True)
    print(f"  plot → {out_path}", flush=True)
    print(f"  json → {json_path}", flush=True)
    print("=" * 65, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pooled spatial autocorrelation on true vs MSR-surrogate expression.",
    )
    parser.add_argument("--cache", required=True, help="Path to cached dataset .npz")
    parser.add_argument("--n-perms", type=int, default=99)
    parser.add_argument("--truncate-um", type=float, required=True, help="MSR truncation scale in µm")
    parser.add_argument("--msr-radius", type=float, required=True, help="MSR neighbor graph radius in µm")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--coordinate-um-per-unit", type=float, default=None)
    parser.add_argument(
        "--calibration-um",
        type=float,
        default=None,
        help="Fixed MEM scale calibration in µm (recommended when comparing truncate values)",
    )
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached dataset not found: {cache_path}")

    run(
        cache_path,
        n_perms=int(args.n_perms),
        truncate_um=float(args.truncate_um),
        msr_radius_um=float(args.msr_radius),
        seed=int(args.seed),
        coordinate_um_per_unit=args.coordinate_um_per_unit,
        calibration_um=args.calibration_um,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
