"""Empirical expression variogram: true layout vs coordinate permutations.

For z-scored expression, the gene-pooled semivariogram and covariance are dual:
    γ(d) = 1 - c(d),   c(d) = mean_g [A_i,g * A_j,g] over pairs at distance d.

This mirrors the existence-test null (shuffle cell coordinates, keep expression)
and asks whether real spatial structure in A exceeds what random labelings produce.

Usage:
  python -m scripts.posthoc.plot_expression_variogram_true_vs_perm \\
      configs/jfan_merfish.json \\
      results/jfan_merfish/260810_jfan_merfish_linear_decoder/260810_jfan_merfish_linear_decoder_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data import load_dataset
from data.schemas import run_config_from_mapping
from experiments.configuration import load_json_config
from methods.permutation import _build_permuted_coordinate_batch


def _cell_gram_pooled(A: np.ndarray) -> np.ndarray:
    """N×N gene-pooled cross-cell products: Gram[i,j] = mean_g A[i,g]*A[j,g]."""
    A64 = np.asarray(A, dtype=np.float64)
    G = float(A64.shape[1])
    return (A64 @ A64.T) / G


def _estimate_pooled_cov_and_variogram(
    S: np.ndarray,
    gram: np.ndarray,
    *,
    bins: np.ndarray,
    n_est_pairs: int | None = None,
    seed: int = 0,
    pair_ii: np.ndarray | None = None,
    pair_jj: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin gene-pooled c(d) and γ(d)=1-c(d) on fixed distance bins (same units as S).

    ``gram`` is the precomputed N×N matrix of mean_g A_i A_j products so we never
    materialize (n_pairs × G) temporary arrays. Pass shared ``pair_ii``/``pair_jj``
    to compare true vs perms on identical cell pairs (only distances change).
    """
    coords = np.asarray(S, dtype=np.float64)
    N = coords.shape[0]
    n_bins = int(bins.size - 1)

    if pair_ii is None or pair_jj is None:
        if n_est_pairs is None or int(n_est_pairs) >= N * (N - 1) // 2:
            # Exact: all unordered pairs
            ii, jj = np.triu_indices(N, k=1)
        else:
            rng = np.random.default_rng(int(seed))
            n_est = int(n_est_pairs)
            ii = rng.integers(0, N, size=n_est)
            jj = rng.integers(0, N, size=n_est)
            keep = ii != jj
            ii, jj = ii[keep], jj[keep]
    else:
        ii = np.asarray(pair_ii)
        jj = np.asarray(pair_jj)

    d_est = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    prod_est = gram[ii, jj]

    # Keep only pairs inside the open-closed interval covered by bins
    in_range = (d_est >= bins[0]) & (d_est <= bins[-1])
    d_est = d_est[in_range]
    prod_est = prod_est[in_range]

    idx = np.clip(np.digitize(d_est, bins) - 1, 0, n_bins - 1)
    # digitize puts values == bins[-1] into n_bins (past last edge); clip handles that
    csum = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.float64)
    np.add.at(csum, idx, prod_est)
    np.add.at(cnt, idx, 1.0)
    c_hat = np.where(cnt > 0, csum / np.maximum(cnt, 1.0), np.nan)
    gamma_hat = 1.0 - c_hat
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, c_hat, gamma_hat


def _coords_to_um(S: np.ndarray, coord_mean, coord_std) -> np.ndarray:
    if coord_mean is None or coord_std is None:
        return np.asarray(S, dtype=np.float64)
    cm = np.asarray(coord_mean, dtype=np.float64).reshape(1, -1)
    cs = np.asarray(coord_std, dtype=np.float64).reshape(1, -1)
    return np.asarray(S, dtype=np.float64) * cs + cm


def _make_distance_bins(
    d_pairs: np.ndarray,
    *,
    n_bins: int,
    spacing: str,
    d_max_um: float | None,
    d_min_um: float,
) -> tuple[np.ndarray, float]:
    """Build distance bin edges.

    ``linear``: equal-width bins on [0, d_hi].
    ``log``: log-spaced bins on [d_min_um, d_hi] (finer resolution at short range).
    ``near_linear``: equal-width bins forced onto [0, d_max_um] (use with a small d_max).
    """
    d_hi_data = float(np.percentile(d_pairs, 99.5))
    d_hi = float(d_max_um) if d_max_um is not None else d_hi_data
    d_hi = min(d_hi, d_hi_data)
    spacing = str(spacing).lower()
    if spacing == "linear":
        bins = np.linspace(0.0, d_hi, int(n_bins) + 1)
    elif spacing == "near_linear":
        bins = np.linspace(0.0, d_hi, int(n_bins) + 1)
    elif spacing == "log":
        d_lo = max(float(d_min_um), 1e-3)
        # pairs below d_lo go into the first log bin via a leading edge at 0
        log_edges = np.geomspace(d_lo, d_hi, int(n_bins))
        bins = np.concatenate([[0.0], log_edges])
    else:
        raise ValueError(f"Unknown --bin-spacing {spacing!r} (use linear|log|near_linear)")
    return bins.astype(np.float64), d_hi


def _pair_counts_per_bin(
    S: np.ndarray,
    bins: np.ndarray,
    pair_ii: np.ndarray,
    pair_jj: np.ndarray,
) -> np.ndarray:
    d = np.linalg.norm(S[pair_ii] - S[pair_jj], axis=1)
    n_bins = int(bins.size - 1)
    idx = np.clip(np.digitize(d, bins) - 1, 0, n_bins - 1)
    # only count pairs that fall inside [bins[0], bins[-1]]
    in_range = d <= bins[-1]
    cnt = np.zeros(n_bins, dtype=np.int64)
    np.add.at(cnt, idx[in_range], 1)
    return cnt


def _run_variogram_panel(
    S_um: np.ndarray,
    gram: np.ndarray,
    perms: list[np.ndarray],
    *,
    bins: np.ndarray,
    pair_ii: np.ndarray,
    pair_jj: np.ndarray,
) -> dict[str, np.ndarray]:
    centers, c_true, g_true = _estimate_pooled_cov_and_variogram(
        S_um, gram, bins=bins, pair_ii=pair_ii, pair_jj=pair_jj
    )
    n_perms = len(perms)
    g_null = np.empty((n_perms, centers.size), dtype=np.float64)
    c_null = np.empty((n_perms, centers.size), dtype=np.float64)
    for p_i, perm in enumerate(perms):
        S_p = S_um[np.asarray(perm)]
        _, c_hat, g_hat = _estimate_pooled_cov_and_variogram(
            S_p, gram, bins=bins, pair_ii=pair_ii, pair_jj=pair_jj
        )
        c_null[p_i] = c_hat
        g_null[p_i] = g_hat
    g_null_mean = np.nanmean(g_null, axis=0)
    g_null_std = np.nanstd(g_null, axis=0)
    c_null_mean = np.nanmean(c_null, axis=0)
    c_null_std = np.nanstd(c_null, axis=0)
    counts = _pair_counts_per_bin(S_um, bins, pair_ii, pair_jj)
    return {
        "centers": centers,
        "bins": bins,
        "counts": counts.astype(np.float64),
        "g_true": g_true,
        "c_true": c_true,
        "g_null_mean": g_null_mean,
        "g_null_std": g_null_std,
        "c_null_mean": c_null_mean,
        "c_null_std": c_null_std,
        "g_diff": g_true - g_null_mean,
        "c_diff": c_true - c_null_mean,
    }


def _write_csv(path: Path, panel: dict[str, np.ndarray]) -> None:
    with path.open("w") as f:
        f.write(
            "distance_um,n_pairs,bin_lo_um,bin_hi_um,"
            "gamma_true,gamma_perm_mean,gamma_perm_std,gamma_true_minus_perm_mean,"
            "c_true,c_perm_mean,c_perm_std,c_true_minus_perm_mean\n"
        )
        bins = panel["bins"]
        for i in range(panel["centers"].size):
            f.write(
                f"{panel['centers'][i]:.6g},{int(panel['counts'][i])},"
                f"{bins[i]:.6g},{bins[i+1]:.6g},"
                f"{panel['g_true'][i]:.6g},{panel['g_null_mean'][i]:.6g},"
                f"{panel['g_null_std'][i]:.6g},{panel['g_diff'][i]:.6g},"
                f"{panel['c_true'][i]:.6g},{panel['c_null_mean'][i]:.6g},"
                f"{panel['c_null_std'][i]:.6g},{panel['c_diff'][i]:.6g}\n"
            )


def _plot_row(axes, panel: dict[str, np.ndarray], *, row_title: str, log_x: bool) -> None:
    centers = panel["centers"]
    ax = axes[0]
    ax.fill_between(
        centers,
        panel["g_null_mean"] - panel["g_null_std"],
        panel["g_null_mean"] + panel["g_null_std"],
        color="#7f8c8d",
        alpha=0.3,
        label="perm mean ± 1 SD",
    )
    ax.plot(centers, panel["g_null_mean"], color="#7f8c8d", lw=1.8, label="perm mean γ(d)")
    ax.plot(centers, panel["g_true"], color="crimson", lw=2.2, label="true γ(d)")
    ax.set_ylabel("Semivariogram γ(d)")
    ax.set_title(f"{row_title}: γ(d)")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    ax.fill_between(
        centers,
        panel["c_null_mean"] - panel["c_null_std"],
        panel["c_null_mean"] + panel["c_null_std"],
        color="#2980b9",
        alpha=0.3,
        label="perm mean ± 1 SD",
    )
    ax.plot(centers, panel["c_null_mean"], color="#2980b9", lw=1.8, label="perm mean c(d)")
    ax.plot(centers, panel["c_true"], color="crimson", lw=2.2, ls="--", label="true c(d)")
    ax.axhline(0.0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_ylabel("Pooled covariance c(d)")
    ax.set_title(f"{row_title}: c(d)")
    ax.legend(fontsize=7, loc="best")

    ax = axes[2]
    ax.axhline(0.0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.plot(centers, panel["g_diff"], color="crimson", lw=2.0, label="γ_true − γ_perm")
    ax.plot(centers, panel["c_diff"], color="#2980b9", lw=1.8, ls="--", label="c_true − c_perm")
    ax.set_ylabel("Difference")
    ax.set_title(f"{row_title}: true − perm")
    ax.legend(fontsize=7, loc="best")

    for ax in axes:
        ax.set_xlabel("Distance (µm)")
        if log_x:
            ax.set_xscale("log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config_path", type=Path)
    ap.add_argument("result_json", type=Path)
    ap.add_argument("--n-bins", type=int, default=40, help="Bins for the full-range row")
    ap.add_argument(
        "--near-n-bins",
        type=int,
        default=40,
        help="Bins for the near-range fine row (default 40 over --near-d-max-um)",
    )
    ap.add_argument(
        "--near-d-max-um",
        type=float,
        default=800.0,
        help="Max distance (µm) for the fine near-range row",
    )
    ap.add_argument(
        "--bin-spacing",
        choices=("linear", "log"),
        default="linear",
        help="Spacing for the full-range row (near row is always fine linear)",
    )
    ap.add_argument(
        "--d-min-um",
        type=float,
        default=20.0,
        help="Left edge for log-spaced full-range bins",
    )
    ap.add_argument("--n-est-pairs", type=int, default=2_000_000)
    ap.add_argument(
        "--n-perms",
        type=int,
        default=None,
        help="Override n_perms (default: from result.json test config)",
    )
    ap.add_argument("--seed", type=int, default=None, help="Override pair-sampling seed")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional filename suffix before extension (e.g. _log)",
    )
    args = ap.parse_args()

    with args.result_json.open() as f:
        result = json.load(f)
    test_cfg = result.get("config", {}).get("test", {})
    n_perms = int(args.n_perms if args.n_perms is not None else test_cfg.get("n_perms", 99))
    perm_seed = int(test_cfg.get("seed", 42))

    cfg = load_json_config(str(args.config_path))
    meta = result.get("artifacts", {}).get("dataset_meta", {})
    if meta.get("h5ad"):
        cfg["data"]["h5ad"] = meta["h5ad"]
    ds = load_dataset(run_config_from_mapping(cfg).data)
    A = np.asarray(ds.A, dtype=np.float64)
    S_std = np.asarray(ds.S, dtype=np.float64)
    S_um = _coords_to_um(S_std, meta.get("coord_mean"), meta.get("coord_std"))
    print(f"Building N×N gene-pooled Gram (N={A.shape[0]}, G={A.shape[1]})...", flush=True)
    gram = _cell_gram_pooled(A)

    pair_ii, pair_jj = np.triu_indices(S_um.shape[0], k=1)
    d_true_pairs = np.linalg.norm(S_um[pair_ii] - S_um[pair_jj], axis=1)

    bins_full, d_hi_full = _make_distance_bins(
        d_true_pairs,
        n_bins=int(args.n_bins),
        spacing=args.bin_spacing,
        d_max_um=None,
        d_min_um=float(args.d_min_um),
    )
    bins_near, d_hi_near = _make_distance_bins(
        d_true_pairs,
        n_bins=int(args.near_n_bins),
        spacing="near_linear",
        d_max_um=float(args.near_d_max_um),
        d_min_um=float(args.d_min_um),
    )
    near_width = float(bins_near[1] - bins_near[0])
    print(
        f"Pairs={pair_ii.size:,}  full: {args.bin_spacing} n_bins={args.n_bins} "
        f"d_hi={d_hi_full:.1f} µm",
        flush=True,
    )
    print(
        f"Near-fine: linear n_bins={args.near_n_bins} over 0–{d_hi_near:.1f} µm "
        f"(bin width ≈ {near_width:.1f} µm)",
        flush=True,
    )

    _, perms = _build_permuted_coordinate_batch(
        S_um.astype(np.float32),
        n_perms=n_perms,
        seed=perm_seed,
        device=torch.device("cpu"),
    )

    print("Computing near-fine panel...", flush=True)
    near = _run_variogram_panel(
        S_um, gram, perms, bins=bins_near, pair_ii=pair_ii, pair_jj=pair_jj
    )
    print("Computing full-range panel...", flush=True)
    full = _run_variogram_panel(
        S_um, gram, perms, bins=bins_full, pair_ii=pair_ii, pair_jj=pair_jj
    )

    out_dir = args.out_dir or args.result_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.result_json.stem.replace("_result", "")
    suffix = str(args.suffix)
    plot_path = out_dir / f"{stem}_expression_variogram_true_vs_perm{suffix}.png"
    csv_near = out_dir / f"{stem}_expression_variogram_true_vs_perm_near{suffix}.csv"
    csv_full = out_dir / f"{stem}_expression_variogram_true_vs_perm{suffix}.csv"

    _write_csv(csv_near, near)
    _write_csv(csv_full, full)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    fig.suptitle(
        f"Gene-pooled expression variogram — true vs coordinate permutations\n"
        f"{stem}  N={A.shape[0]}  G={A.shape[1]}  n_perms={n_perms}  "
        f"(γ=1−c; top=fine near-range ~{near_width:.0f} µm bins, "
        f"bottom=full {args.bin_spacing})",
        fontsize=10,
        fontweight="bold",
    )
    _plot_row(
        axes[0],
        near,
        row_title=f"Near 0–{d_hi_near:.0f} µm ({near_width:.0f} µm bins)",
        log_x=False,
    )
    _plot_row(
        axes[1],
        full,
        row_title=f"Full 0–{d_hi_full:.0f} µm ({args.bin_spacing})",
        log_x=(args.bin_spacing == "log"),
    )
    # Annotate sparse near bins (<50 pairs)
    sparse = near["counts"] < 50
    if np.any(sparse):
        axes[0, 1].scatter(
            near["centers"][sparse],
            near["c_true"][sparse],
            s=18,
            facecolors="none",
            edgecolors="orange",
            linewidths=1.0,
            zorder=5,
            label="n_pairs<50",
        )
        axes[0, 1].legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_near}")
    print(f"Wrote {csv_full}")
    print(
        "Near-bin pair counts: "
        f"min={int(near['counts'].min())} median={int(np.median(near['counts']))} "
        f"max={int(near['counts'].max())}"
    )
    peak_i = int(np.nanargmax(np.abs(near["c_diff"])))
    print(
        f"Near peak |Δc|={abs(near['c_diff'][peak_i]):.4f} at "
        f"{near['centers'][peak_i]:.1f} µm "
        f"(bin {near['bins'][peak_i]:.1f}–{near['bins'][peak_i+1]:.1f}, "
        f"n_pairs={int(near['counts'][peak_i])})"
    )


if __name__ == "__main__":
    main()
