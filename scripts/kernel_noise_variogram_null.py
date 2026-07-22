"""
Variogram-matched null on kernel noise — non-parametric SA-preserving surrogate.

Estimates the empirical covariance-vs-distance c_hat(d) directly from the data
(no assumed kernel form, no fixed correlation length ρ), builds a stationary
isotropic covariance, and Cholesky-draws fresh GP realizations matching it.

This is the real-data-ready cousin of Cholesky-GP: same cell-space resampling
principle, but the spatial structure is learned from the data rather than assumed.

Compares against:
  - Cholesky-GP (parametric, uses true ρ=30) — calibrated (p=0.64)
  - SpGP (MEM basis fresh draw)              — false positive (p=0.01)
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.kernel_noise_study import load_dataset_cache
from methods.metrics import permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model
from scripts.msr_null_smoke_test import (
    N_RERUNS,
    SEED,
    base_cfg,
    build_variogram_matched_surrogates,
)

N_PERMS = 99
EPOCHS = 450
DATASET = "d30_delta0p1_seed1"
CACHE_PATH = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
OUTDIR = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Prior parallel n=99 runs for reference
PRIOR = {
    "spgp":        dict(stat_true=141592.99, null_mean=141691.57, null_std=23.83, p=0.010, rank=1),
    "cholesky_gp": dict(stat_true=141592.99, null_mean=141580.81, null_std=30.37, p=0.640, rank=64),
}


def run_variogram_parallel(S: np.ndarray, A: np.ndarray) -> dict:
    print(f"\n  [Variogram-matched parallel] epochs={EPOCHS}  n_perms={N_PERMS}", flush=True)
    t0 = time.time()

    surrogates = build_variogram_matched_surrogates(S, A, N_PERMS, seed=SEED + 900)
    n_slots = N_PERMS + 1
    s_batched = np.repeat(S[np.newaxis, :, :], n_slots, axis=0).astype(np.float32)
    a_batched = np.empty((n_slots, S.shape[0], A.shape[1]), dtype=np.float32)
    a_batched[0] = A
    for i in range(N_PERMS):
        a_batched[i + 1] = surrogates[i]

    cfg = replace(base_cfg(N_PERMS, SEED), epochs=EPOCHS)
    device = resolve_device("cuda")

    print(f"    Training {n_slots} parallel slots …", flush=True)
    _, out, _ = train_parallel_isodepth_model(
        S, A, cfg, device=device,
        s_batched=s_batched,
        a_batched=a_batched,
        model_label=f"Variogram_parallel_e{EPOCHS}",
    )

    elapsed = time.time() - t0
    stat_true = out.stat_true
    stat_perm = out.stat_perm
    null_mean = float(stat_perm.mean())
    null_std = float(stat_perm.std())
    z = (null_mean - stat_true) / (null_std + 1e-12)
    p = permutation_p_value(cfg.metric, stat_true, stat_perm)
    rank = int(np.sum(stat_perm < stat_true)) + 1

    print(
        f"    stat_true={stat_true:.2f}  null_mean={null_mean:.2f}  "
        f"null_std={null_std:.2f}  z={z:.2f}  p={p:.3f}  rank={rank}/{N_PERMS+1}  [{elapsed:.0f}s]",
        flush=True,
    )
    for i, s in enumerate(stat_perm):
        if (i + 1) % 10 == 0 or i == 0 or i == N_PERMS - 1:
            print(f"    perm {i+1:2d}/{N_PERMS}: {s:.2f}", flush=True)

    return dict(
        method="variogram", stat_true=stat_true, stat_perm=stat_perm,
        null_mean=null_mean, null_std=null_std, z=z, p=p, rank=rank, elapsed=elapsed,
    )


def plot_result(r: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Variogram-matched null — kernel noise {DATASET}\n"
        f"non-parametric (empirical c(d), no assumed ρ)  |  parallel e={EPOCHS}  n_perms={N_PERMS}",
        fontsize=10, fontweight="bold",
    )

    def sig(p: float) -> str:
        return "p<0.05 ✓" if p < 0.05 else ("p<0.10 ~" if p < 0.10 else "p≥0.10 ns")

    ax = axes[0]
    ax.hist(r["stat_perm"], bins=25, color="#1f77b4", alpha=0.72, edgecolor="k", linewidth=0.5)
    ax.axvline(r["stat_true"], color="crimson", lw=2.0, ls="--", label=f"true={r['stat_true']:.0f}")
    ax.axvline(r["null_mean"], color="k", lw=1.0, ls=":", alpha=0.65, label=f"null={r['null_mean']:.0f}")
    xl = ax.get_xlim()
    ax.axvspan(xl[0], r["stat_true"], alpha=0.07, color="crimson")
    ax.set_xlabel("NLL Gaussian MSE", fontsize=9)
    ax.set_title(
        f"Variogram-matched (non-parametric)\n"
        f"z={r['z']:.2f}  p={r['p']:.3f}  rank={r['rank']}/{N_PERMS+1}  {sig(r['p'])}",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    ax2 = axes[1]
    labels = ["SpGP\n(MEM)", "Cholesky-GP\n(param ρ)", "Variogram\n(non-param)"]
    keys = ["spgp", "cholesky_gp", None]
    nulls = [PRIOR["spgp"]["null_mean"], PRIOR["cholesky_gp"]["null_mean"], r["null_mean"]]
    errs = [PRIOR["spgp"]["null_std"], PRIOR["cholesky_gp"]["null_std"], r["null_std"]]
    trues = [PRIOR["spgp"]["stat_true"], PRIOR["cholesky_gp"]["stat_true"], r["stat_true"]]
    ps = [PRIOR["spgp"]["p"], PRIOR["cholesky_gp"]["p"], r["p"]]
    colors = ["#55a868", "#2ca02c", "#1f77b4"]
    xs = np.arange(3)
    ax2.bar(xs, nulls, yerr=errs, capsize=6, color=colors, alpha=0.75, edgecolor="k")
    for xi, t, nm, ns, p_val in zip(xs, trues, nulls, errs, ps):
        ax2.plot([xi - 0.28, xi + 0.28], [t, t], color="crimson", lw=2.5, zorder=5)
        ax2.text(xi, nm + ns * 1.4, f"p={p_val:.3f}", ha="center", fontsize=9, color="crimson")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("reconstruction loss", fontsize=9)
    ax2.set_title("Null mean ± std vs stat_true (—)", fontsize=9)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = OUTDIR / f"kernel_noise_variogram_e{EPOCHS}_n{N_PERMS}_{DATASET}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out}", flush=True)


def main() -> None:
    t0 = time.time()
    print("=" * 65)
    print(f"VARIOGRAM-MATCHED NULL — {DATASET}")
    print(f"n_perms={N_PERMS}  epochs={EPOCHS}  n_reruns={N_RERUNS}  SEED={SEED}")
    print("=" * 65)

    dataset = load_dataset_cache(CACHE_PATH)
    S, A = dataset.S, dataset.A
    print(f"\nData: {S.shape[0]} cells × {A.shape[1]} genes")

    r = run_variogram_parallel(S, A)

    print("\n" + "=" * 70)
    print(f"{'Null':<22} {'stat_true':>10} {'null_mean':>10} {'p':>6} {'rank':>8}")
    print("-" * 70)
    print(f"{'SpGP (MEM)':<22} {PRIOR['spgp']['stat_true']:>10.2f} "
          f"{PRIOR['spgp']['null_mean']:>10.2f} {PRIOR['spgp']['p']:>6.3f} {PRIOR['spgp']['rank']:>5}/100")
    print(f"{'Cholesky-GP (param ρ)':<22} {PRIOR['cholesky_gp']['stat_true']:>10.2f} "
          f"{PRIOR['cholesky_gp']['null_mean']:>10.2f} {PRIOR['cholesky_gp']['p']:>6.3f} {PRIOR['cholesky_gp']['rank']:>5}/100")
    print(f"{'Variogram (non-param)':<22} {r['stat_true']:>10.2f} "
          f"{r['null_mean']:>10.2f} {r['p']:>6.3f} {r['rank']:>5}/{N_PERMS+1}")
    print("=" * 70)

    plot_result(r)
    print(f"\nTotal: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
