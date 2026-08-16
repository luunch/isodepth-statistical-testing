"""
MSR-surrogate null vs true global Moran's I (kernel noise).

For a cached kernel-noise dataset, computes Moran's I per gene on the true
expression matrix and on each MSR surrogate expression matrix (coordinates
fixed).  Spatial weights: inverse-distance within ``--neighbor-radius`` µm.

Outputs a histogram of null mean Moran's I (mean across genes) with the true
value marked, plus a JSON summary.

Usage:
  python scripts/kernel_noise_msr_moran_null.py \\
      --cache results/experiments/.../datasets/d15_delta0p1_seed0.npz \\
      --n-perms 99 --truncate-um 30 --neighbor-radius 30 \\
      --msr-radius 30 --coordinate-um-per-unit 960
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.core.paths import repo_root
REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from data import raw_coordinates_from_standardized
from experiments.kernel_noise_msr_study import load_dataset_cache
from methods.moran import DEFAULT_MORAN_NEIGHBOR_RADIUS_UM
from scripts.studies.msr_diagnostics.block_perm_autocorr_length import (
    build_inverse_distance_weights,
    morans_i_per_gene,
)

DEFAULT_SEED = 42
DEFAULT_MSR_CALIBRATION_UM = 100.0


def _effective_calibration_um(truncate_um: float, calibration_um: float | None) -> tuple[float, str | None]:
    if calibration_um is not None and float(calibration_um) > 0:
        return float(calibration_um), None
    cal_um = float(truncate_um)
    msg = (
        f"calibration_um defaults to truncate_um ({cal_um:g} µm). "
        "Truncation uses |λ| < 1, so T30 and T60 with cal=truncate yield identical surrogates."
    )
    return cal_um, msg


def _resolve_um_per_unit(meta: dict, override: float | None) -> float:
    if override is not None and float(override) > 0:
        return float(override)
    if meta.get("coordinate_um_per_unit") is not None:
        return float(meta["coordinate_um_per_unit"])
    if meta.get("scale_um") is not None:
        return float(meta["scale_um"])
    return 1000.0


def _true_coords_native(S: np.ndarray, meta: dict) -> np.ndarray:
    if meta.get("coordinate_standardization") == "zscore":
        return raw_coordinates_from_standardized(S, meta)
    return np.asarray(S, dtype=np.float32)


def estimate_morans_i_msr(
    S_um: np.ndarray,
    A_true: np.ndarray,
    A_surrogates: np.ndarray,
    *,
    neighbor_radius_um: float,
) -> np.ndarray:
    """Moran's I per gene for true expression (slot 0) and each surrogate.

    Returns shape (1 + n_surrogates, n_genes).
    """
    W, s0 = build_inverse_distance_weights(S_um, neighbor_radius_um)
    n_surrogates = A_surrogates.shape[0]
    n_genes = A_true.shape[1]
    out = np.empty((1 + n_surrogates, n_genes), dtype=np.float64)
    out[0] = morans_i_per_gene(W, s0, A_true)
    for i in range(n_surrogates):
        out[1 + i] = morans_i_per_gene(W, s0, A_surrogates[i])
    return out


def plot_results(
    *,
    dataset_key: str,
    kernel_rho_um: float,
    delta: float,
    truncate_um: float,
    msr_radius_um: float,
    neighbor_radius_um: float,
    I_by_slot: np.ndarray,
    out_path: Path,
) -> None:
    I_by_slot = np.asarray(I_by_slot, dtype=np.float64)
    true_I = I_by_slot[0]
    null_I = I_by_slot[1:]
    n_perms = null_I.shape[0]

    true_mean = float(np.nanmean(true_I))
    null_means = np.nanmean(null_I, axis=1)
    null_mean_of_means = float(null_means.mean())
    null_std_of_means = float(null_means.std())
    rank = int(np.sum(null_means < true_mean)) + 1
    pct = 100.0 * rank / (n_perms + 1)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.suptitle(
        f"Global Moran's I — MSR null vs true\n"
        f"{dataset_key}  ρ={kernel_rho_um:.0f} µm  δ={delta:g}  "
        f"trunc={truncate_um:.0f} µm  msr_r={msr_radius_um:.0f} µm  "
        f"neighbor r={neighbor_radius_um:.0f} µm  n_null={n_perms}",
        fontsize=10,
        fontweight="bold",
    )

    bins = max(15, min(40, n_perms // 3))
    ax.hist(
        null_means,
        bins=bins,
        color="#2980b9",
        alpha=0.72,
        edgecolor="k",
        linewidth=0.4,
        label=f"null mean I (n={n_perms})",
    )
    ax.axvline(true_mean, color="crimson", lw=2.2, ls="--", label=f"true mean = {true_mean:.4f}")
    ax.axvline(
        null_mean_of_means,
        color="k",
        lw=1.2,
        ls=":",
        alpha=0.8,
        label=f"null avg = {null_mean_of_means:.4f} ± {null_std_of_means:.4f}",
    )
    ax.set_xlabel("Mean Moran's I across genes", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(
        f"Null histogram vs true\n"
        f"true rank {rank}/{n_perms + 1} ({pct:.1f} percentile)",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, loc="best")
    ax.tick_params(labelsize=8)

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
    neighbor_radius_um: float,
    seed: int,
    coordinate_um_per_unit: float | None,
    calibration_um: float | None,
    out_path: Path,
) -> dict:
    from methods.msr import build_joint_truncated_msr_surrogates

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
    cal_um, cal_warning = _effective_calibration_um(truncate_um, calibration_um)
    if cal_warning:
        import warnings

        warnings.warn(cal_warning, stacklevel=2)
        print(f"WARNING: {cal_warning}", flush=True)

    print(
        f"MSR: trunc={truncate_um:.0f} µm  msr_r={msr_radius_um:.0f} µm  "
        f"cal={cal_um:.0f} µm  neighbor r={neighbor_radius_um:.0f} µm  "
        f"n_perms={n_perms}  seed={seed}",
        flush=True,
    )

    t0 = time.time()
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

    t1 = time.time()
    I_by_slot = estimate_morans_i_msr(
        S_um,
        A,
        surrogates,
        neighbor_radius_um=neighbor_radius_um,
    )
    print(f"  Computed Moran's I in {time.time() - t1:.1f}s", flush=True)

    true_I = I_by_slot[0]
    null_I = I_by_slot[1:]
    true_mean = float(np.nanmean(true_I))
    null_means = np.nanmean(null_I, axis=1)
    null_mean_of_means = float(null_means.mean())
    null_std_of_means = float(null_means.std())
    rank = int(np.sum(null_means < true_mean)) + 1

    plot_results(
        dataset_key=dataset_key,
        kernel_rho_um=kernel_rho,
        delta=delta,
        truncate_um=truncate_um,
        msr_radius_um=msr_radius_um,
        neighbor_radius_um=neighbor_radius_um,
        I_by_slot=I_by_slot,
        out_path=out_path,
    )

    summary = {
        "dataset_key": dataset_key,
        "cache_path": str(cache_path.resolve()),
        "kernel_rho_um": kernel_rho,
        "delta": delta,
        "truncate_um": truncate_um,
        "msr_radius_um": msr_radius_um,
        "neighbor_radius_um": neighbor_radius_um,
        "calibration_um": cal_um,
        "calibration_warning": cal_warning,
        "coordinate_um_per_unit": um_per_unit,
        "n_perms": n_perms,
        "seed": seed,
        "true_morans_i_per_gene": true_I.tolist(),
        "true_mean_morans_i": true_mean,
        "null_mean_morans_i_per_perm": null_means.tolist(),
        "null_mean_of_means": null_mean_of_means,
        "null_std_of_means": null_std_of_means,
        "true_rank": rank,
        "true_percentile": 100.0 * rank / (n_perms + 1),
        "plot_path": str(out_path.resolve()),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 65, flush=True)
    print(f"  true mean Moran's I          {true_mean:10.5f}", flush=True)
    print(f"  null mean ± std (of means)   {null_mean_of_means:10.5f} ± {null_std_of_means:.5f}", flush=True)
    print(f"  true rank vs nulls           {rank}/{n_perms + 1}", flush=True)
    print(f"  plot → {out_path}", flush=True)
    print(f"  json → {json_path}", flush=True)
    print("=" * 65, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare global Moran's I on true vs MSR-surrogate null expression.",
    )
    parser.add_argument("--cache", required=True, help="Path to cached dataset .npz")
    parser.add_argument("--n-perms", type=int, default=99)
    parser.add_argument("--truncate-um", type=float, required=True, help="MSR truncation scale in µm")
    parser.add_argument("--msr-radius", type=float, required=True, help="MSR neighbor graph radius in µm")
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=DEFAULT_MORAN_NEIGHBOR_RADIUS_UM,
        help=f"Moran W radius in µm (default: {DEFAULT_MORAN_NEIGHBOR_RADIUS_UM:g})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--coordinate-um-per-unit", type=float, default=None)
    parser.add_argument("--calibration-um", type=float, default=None)
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
        neighbor_radius_um=float(args.neighbor_radius),
        seed=int(args.seed),
        coordinate_um_per_unit=args.coordinate_um_per_unit,
        calibration_um=args.calibration_um,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
