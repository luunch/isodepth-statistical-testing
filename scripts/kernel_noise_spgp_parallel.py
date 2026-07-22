"""
SpGP null on kernel noise — parallel batched training, 3× epochs.

Compares SpGP at epochs=150 (sequential, prior run) vs epochs=450 (parallel batched).
Uses a_batched so true + all null surrogates train in one parallel call.
"""
from __future__ import annotations
import sys, time
from dataclasses import replace
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.kernel_noise_study import load_dataset_cache
from methods.metrics import permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model
from scripts.msr_null_smoke_test import (
    N_RERUNS, SEED, base_cfg, build_spectral_gp_surrogates,
)

N_PERMS    = 99
EPOCHS     = 450          # 3× default 150
DATASET    = "d30_delta0p1_seed1"
CACHE_PATH = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
OUTDIR     = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Prior parallel run at 150 epochs / 10 perms (for comparison)
PRIOR = dict(stat_true=141577.78, null_mean=141677.99, null_std=29.5, z=3.40, p=0.091, n_perms=10)


def run_spgp_parallel(S: np.ndarray, A: np.ndarray) -> dict:
    print(f"\n  [SpGP parallel] epochs={EPOCHS}  n_perms={N_PERMS}", flush=True)
    t0 = time.time()

    surrogates = build_spectral_gp_surrogates(S, A, N_PERMS, seed=SEED + 500)
    n_slots    = N_PERMS + 1  # slot 0 = true; slots 1..N_PERMS = surrogates
    s_batched  = np.repeat(S[np.newaxis, :, :], n_slots, axis=0).astype(np.float32)
    a_batched  = np.empty((n_slots, S.shape[0], A.shape[1]), dtype=np.float32)
    a_batched[0] = A
    for i in range(N_PERMS):
        a_batched[i + 1] = surrogates[i]

    cfg    = replace(base_cfg(N_PERMS, SEED), epochs=EPOCHS)
    device = resolve_device("cuda")

    print(f"    Training {n_slots} parallel slots (true + {N_PERMS} SpGP surrogates) …",
          flush=True)
    _, out, _ = train_parallel_isodepth_model(
        S, A, cfg, device=device,
        s_batched=s_batched,
        a_batched=a_batched,
        model_label=f"SpGP_parallel_e{EPOCHS}",
    )

    elapsed   = time.time() - t0
    stat_true = out.stat_true
    stat_perm = out.stat_perm
    null_mean = float(stat_perm.mean())
    null_std  = float(stat_perm.std())
    z         = (null_mean - stat_true) / (null_std + 1e-12)
    p         = permutation_p_value(cfg.metric, stat_true, stat_perm)

    print(f"    stat_true={stat_true:.2f}  null_mean={null_mean:.2f}  "
          f"null_std={null_std:.2f}  z={z:.2f}  p={p:.3f}  rank={int(np.sum(stat_perm < stat_true))+1}/{N_PERMS+1}  [{elapsed:.0f}s]", flush=True)
    for i, s in enumerate(stat_perm):
        if (i + 1) % 10 == 0 or i == 0 or i == N_PERMS - 1:
            print(f"    perm {i+1:2d}/{N_PERMS}: {s:.2f}", flush=True)

    return dict(method="spectral_gp_parallel", stat_true=stat_true, stat_perm=stat_perm,
                null_mean=null_mean, null_std=null_std, z=z, p=p, elapsed=elapsed)


def plot_result(r: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"SpGP null — kernel noise {DATASET} (SA only, no gradient)\n"
        f"Parallel batched  |  epochs={EPOCHS}  |  n_perms={N_PERMS}  |  SEED={SEED}",
        fontsize=11, fontweight="bold",
    )

    def sig(p):
        return "p<0.05 ✓" if p < 0.05 else ("p<0.10 ~" if p < 0.10 else "p≥0.10 ns")

    ax = axes[0]
    ax.hist(r["stat_perm"], bins=min(25, max(12, N_PERMS // 4)), color="#55a868", alpha=0.72, edgecolor="k", linewidth=0.5)
    ax.axvline(r["stat_true"], color="crimson", lw=2.0, ls="--",
               label=f"true={r['stat_true']:.0f}")
    ax.axvline(r["null_mean"], color="k", lw=1.0, ls=":", alpha=0.65,
               label=f"null={r['null_mean']:.0f}")
    xl = ax.get_xlim()
    ax.axvspan(xl[0], r["stat_true"], alpha=0.07, color="crimson")
    ax.set_xlabel("NLL Gaussian MSE", fontsize=9)
    ax.set_title(f"SpGP parallel e={EPOCHS}\nz={r['z']:.2f}  p={r['p']:.3f}  {sig(r['p'])}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    ax2 = axes[1]
    labels = [f"SpGP par\ne=450 n=10", f"SpGP par\ne={EPOCHS} n={N_PERMS}"]
    nulls  = [PRIOR["null_mean"], r["null_mean"]]
    errs   = [PRIOR["null_std"], r["null_std"]]
    trues  = [PRIOR["stat_true"], r["stat_true"]]
    xs     = np.arange(2)
    ax2.bar(xs, nulls, yerr=errs, capsize=6, color=["#DD8452", "#55a868"],
            alpha=0.75, edgecolor="k", linewidth=0.8)
    for xi, t, nm, ns, p_val in zip(xs, trues, nulls, errs,
                                    [PRIOR["p"], r["p"]]):
        ax2.plot([xi - 0.28, xi + 0.28], [t, t], color="crimson", lw=2.5, zorder=5)
        ax2.text(xi, nm + ns * 1.4, f"p={p_val:.3f}", ha="center", fontsize=9, color="crimson")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("reconstruction loss", fontsize=9)
    ax2.set_title("vs prior parallel run (n=10)", fontsize=10)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = OUTDIR / f"kernel_noise_spgp_parallel_e{EPOCHS}_n{N_PERMS}_{DATASET}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out}", flush=True)


def main():
    t0 = time.time()
    print("=" * 65)
    print(f"SpGP PARALLEL — {DATASET}")
    print(f"n_perms={N_PERMS}  epochs={EPOCHS}  n_reruns={N_RERUNS}  SEED={SEED}")
    print("=" * 65)

    dataset = load_dataset_cache(CACHE_PATH)
    S, A = dataset.S, dataset.A
    print(f"\nData: {S.shape[0]} cells × {A.shape[1]} genes")

    r = run_spgp_parallel(S, A)

    print("\n" + "=" * 65)
    print(f"SpGP parallel e=450 n=10: stat_true={PRIOR['stat_true']:.2f}  "
          f"null_mean={PRIOR['null_mean']:.2f}  p={PRIOR['p']:.3f}")
    print(f"SpGP parallel e={EPOCHS} n={N_PERMS}: stat_true={r['stat_true']:.2f}  "
          f"null_mean={r['null_mean']:.2f}  p={r['p']:.3f}  rank={int(np.sum(r['stat_perm'] < r['stat_true']))+1}/{N_PERMS+1}")
    print("=" * 65)

    plot_result(r)
    print(f"\nTotal: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
