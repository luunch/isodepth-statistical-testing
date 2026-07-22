"""
Block-permutation null vs true global Moran's I (kernel noise).

For a cached kernel-noise dataset (default: generative ρ=30 µm), computes
Moran's I per gene on true coordinates and on each of ~100 block-permutation
coordinate layouts (expression fixed). Spatial weights: inverse-distance within
``--neighbor-radius`` µm (same convention as MSR).

Outputs a histogram of null mean Moran's I (mean across genes) with the true value marked.

Usage:
  python scripts/kernel_noise_block_perm_autocorr_length.py
  python scripts/kernel_noise_block_perm_autocorr_length.py \\
      --dataset d30_delta0p01_seed1 --n-perms 100 --block-radius 60
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data import raw_coordinates_from_standardized
from experiments.kernel_noise_study import load_dataset_cache
from methods.block_permutation import block_stats, build_block_permuted_coordinate_batch
from methods.moran import DEFAULT_MORAN_NEIGHBOR_RADIUS_UM
from scripts.msr_null_smoke_test import SEED

DEFAULT_DATASET = "d30_delta0p01_seed1"
OUTDIR = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)


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


def build_inverse_distance_weights(S_um: np.ndarray, radius_um: float) -> tuple[np.ndarray, float]:
    """Row-stochastic-free symmetric W: w_ij = 1/d_ij for pairs with d < radius."""
    from scipy.spatial import KDTree

    S_um = np.asarray(S_um, dtype=np.float64)
    radius = float(radius_um)
    tree = KDTree(S_um)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if pairs.shape[0] == 0:
        raise ValueError(
            f"No cell pairs within neighbor radius {radius_um:.1f} µm. "
            "Increase --neighbor-radius."
        )

    d_vals = np.linalg.norm(S_um[pairs[:, 0]] - S_um[pairs[:, 1]], axis=1)
    d_min = max(float(np.median(d_vals)) * 1e-4, 1e-9)
    w_vals = 1.0 / np.maximum(d_vals, d_min)

    n_cells = int(S_um.shape[0])
    W = np.zeros((n_cells, n_cells), dtype=np.float64)
    W[pairs[:, 0], pairs[:, 1]] = w_vals
    W[pairs[:, 1], pairs[:, 0]] = w_vals
    s0 = float(W.sum())
    return W, s0


def morans_i_per_gene(W: np.ndarray, s0: float, A: np.ndarray) -> np.ndarray:
    """Global Moran's I for each gene column of A (already z-scored per gene)."""
    A64 = np.asarray(A, dtype=np.float64)
    n_cells, n_genes = A64.shape
    n = float(n_cells)
    I = np.empty(n_genes, dtype=np.float64)

    for g in range(n_genes):
        x = A64[:, g]
        xc = x - x.mean()
        denom = float(xc @ xc)
        if denom <= 1e-12:
            I[g] = np.nan
            continue
        numer = float(xc @ W @ xc)
        I[g] = (n / s0) * (numer / denom)
    return I


def estimate_morans_i_slots(
    s_batched_native: np.ndarray,
    A: np.ndarray,
    *,
    um_per_unit: float,
    neighbor_radius_um: float,
) -> np.ndarray:
    """Moran's I per gene for each coordinate slot: shape (n_slots, n_genes)."""
    n_slots = int(s_batched_native.shape[0])
    n_genes = int(A.shape[1])
    out = np.empty((n_slots, n_genes), dtype=np.float64)

    for slot in range(n_slots):
        S_um = np.asarray(s_batched_native[slot], dtype=np.float64) * float(um_per_unit)
        W, s0 = build_inverse_distance_weights(S_um, neighbor_radius_um)
        out[slot] = morans_i_per_gene(W, s0, A)
    return out


def plot_results(
    *,
    dataset_key: str,
    kernel_rho_um: float,
    delta: float,
    block_radius_um: float,
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
        f"Global Moran's I — block perm null vs true\n"
        f"{dataset_key}  generative ρ={kernel_rho_um:.0f} µm  δ={delta:g}  "
        f"block r={block_radius_um:.0f} µm  neighbor r={neighbor_radius_um:.0f} µm  "
        f"n_null={n_perms}",
        fontsize=10,
        fontweight="bold",
    )

    bins = max(15, min(40, n_perms // 3))
    ax.hist(
        null_means,
        bins=bins,
        color="#27ae60",
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
    block_radius_um: float,
    neighbor_radius_um: float,
    seed: int,
    coordinate_um_per_unit: float | None,
    block_jitter: bool,
    block_shape: str,
    out_path: Path | None,
) -> dict:
    ds = load_dataset_cache(cache_path)
    S, A, meta = ds.S, ds.A, ds.meta
    um_per_unit = _resolve_um_per_unit(meta, coordinate_um_per_unit)
    S_native = _true_coords_native(S, meta)

    kernel_rho = float((meta.get("kernel") or {}).get("distance", np.nan))
    delta = float(meta.get("delta", np.nan))
    dataset_key = cache_path.stem

    stats = block_stats(
        S_native,
        block_radius_um,
        um_per_unit,
        block_shape=block_shape,
    )
    print(
        f"Dataset {dataset_key}: {S.shape[0]} cells × {A.shape[1]} genes  "
        f"ρ={kernel_rho:.0f} µm  δ={delta:g}",
        flush=True,
    )
    print(
        f"Block tiling: {stats['n_blocks']} blocks  "
        f"median cells/block={stats['median_cells']:.0f}  "
        f"block r={block_radius_um:.0f} µm  neighbor r={neighbor_radius_um:.0f} µm  "
        f"n_perms={n_perms}",
        flush=True,
    )

    t0 = time.time()
    s_batched_native = build_block_permuted_coordinate_batch(
        S_native,
        radius_um=block_radius_um,
        coordinate_um_per_unit=um_per_unit,
        n_perms=n_perms,
        seed=seed,
        block_jitter=block_jitter,
        block_shape=block_shape,
    )
    print(f"  Built {n_perms + 1} coordinate slots in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    I_by_slot = estimate_morans_i_slots(
        s_batched_native,
        A,
        um_per_unit=um_per_unit,
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

    if out_path is None:
        slug_br = int(round(block_radius_um))
        slug_nr = int(round(neighbor_radius_um))
        out_path = OUTDIR / (
            f"kernel_noise_block_perm_moran_i_br{slug_br}_nr{slug_nr}_"
            f"n{n_perms}_{dataset_key}.png"
        )

    plot_results(
        dataset_key=dataset_key,
        kernel_rho_um=kernel_rho,
        delta=delta,
        block_radius_um=block_radius_um,
        neighbor_radius_um=neighbor_radius_um,
        I_by_slot=I_by_slot,
        out_path=out_path,
    )

    summary = {
        "dataset_key": dataset_key,
        "cache_path": str(cache_path.resolve()),
        "kernel_rho_um": kernel_rho,
        "delta": delta,
        "block_radius_um": block_radius_um,
        "neighbor_radius_um": neighbor_radius_um,
        "n_perms": n_perms,
        "seed": seed,
        "true_morans_i_per_gene": true_I.tolist(),
        "true_mean_morans_i": true_mean,
        "null_mean_morans_i_per_perm": null_means.tolist(),
        "null_morans_i_per_gene_per_perm": null_I.tolist(),
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
        description="Compare global Moran's I on true vs block-perm null coordinates.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset slug under kernel_noise_study/datasets/ (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Optional explicit path to cached .npz (overrides --dataset)",
    )
    parser.add_argument("--n-perms", type=int, default=999)
    parser.add_argument("--block-radius", type=float, default=60.0)
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=DEFAULT_MORAN_NEIGHBOR_RADIUS_UM,
        help=(
            "Moran W radius in µm "
            f"(default: {DEFAULT_MORAN_NEIGHBOR_RADIUS_UM:g}, "
            "matching test.moran_neighbor_radius_um)"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--coordinate-um-per-unit", type=float, default=None)
    parser.add_argument("--block-jitter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-shape", choices=("hexagon", "square"), default="hexagon")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    cache_path = Path(args.cache) if args.cache else (
        REPO / f"results/experiments/kernel_noise_study/datasets/{args.dataset}.npz"
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached dataset not found: {cache_path}")

    neighbor_radius = float(args.neighbor_radius)
    out_path = Path(args.out) if args.out else None
    run(
        cache_path,
        n_perms=int(args.n_perms),
        block_radius_um=float(args.block_radius),
        neighbor_radius_um=neighbor_radius,
        seed=int(args.seed),
        coordinate_um_per_unit=args.coordinate_um_per_unit,
        block_jitter=bool(args.block_jitter),
        block_shape=str(args.block_shape),
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
