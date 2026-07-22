"""SpGP null on kernel noise d30 — 450 epochs (3x default) convergence test."""
from __future__ import annotations
import sys, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.kernel_noise_study import load_dataset_cache
from methods.metrics import permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model
from scripts.msr_null_smoke_test import SEED, N_RERUNS, build_spectral_gp_surrogates
from data.schemas import TestConfig

N_PERMS = 10
EPOCHS  = 450
DATASET = "d30_delta0p1_seed1"
CACHE   = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
OUTDIR  = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)

def cfg():
    return TestConfig(
        method="parallel_permutation", metric="nll_gaussian_mse",
        n_perms=0, epochs=EPOCHS, n_reruns=N_RERUNS,
        sgd_batch_size=128, lr=1e-3, seed=SEED,
        device="cuda", decoder="nn", verbose=False,
    )

def main():
    t0 = time.time()
    print(f"SpGP — {DATASET}  epochs={EPOCHS}  n_perms={N_PERMS}  SEED={SEED}")
    ds = load_dataset_cache(CACHE)
    S, A = ds.S, ds.A
    print(f"Data: {S.shape[0]} x {A.shape[1]}")
    device = resolve_device("cuda")

    print("\n[true model]", flush=True)
    _, out_true, _ = train_parallel_isodepth_model(S, A, cfg(), device=device)
    stat_true = out_true.stat_true
    print(f"  stat_true = {stat_true:.2f}", flush=True)

    print("\n[SpGP surrogates]", flush=True)
    surr = build_spectral_gp_surrogates(S, A, N_PERMS, seed=SEED + 500)

    stat_perm = np.empty(N_PERMS)
    for i in range(N_PERMS):
        _, oi, _ = train_parallel_isodepth_model(S, surr[i], cfg(), device=device)
        stat_perm[i] = oi.stat_true
        print(f"  perm {i+1:2d}/{N_PERMS}: {stat_perm[i]:.2f}", flush=True)

    nm  = float(stat_perm.mean()); ns = float(stat_perm.std())
    z   = (nm - stat_true) / (ns + 1e-12)
    p   = permutation_p_value("nll_gaussian_mse", stat_true, stat_perm)
    gap = nm - stat_true

    print(f"\nstat_true={stat_true:.2f}  null_mean={nm:.2f}  null_std={ns:.2f}")
    print(f"gap={gap:.2f}  z={z:.2f}  p={p:.3f}")
    print(f"Result: {'FIRE' if p<0.05 else ('~' if p<0.10 else 'ns')}")
    print(f"Reference 150ep: gap=71.6  z=4.04  p=0.091")
    print("gap shrank -> optimization issue; gap same -> distributional difference")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        f"SpGP null — Kernel noise {DATASET}\n"
        f"epochs={EPOCHS} (3x)  n_perms={N_PERMS}  SEED={SEED}",
        fontsize=11, fontweight="bold",
    )
    ax = axes[0]
    ax.hist(stat_perm, bins=6, color="#55a868", alpha=0.72, edgecolor="k", lw=0.5)
    ax.axvline(stat_true, color="crimson", lw=2.0, ls="--", label=f"true={stat_true:.0f}")
    ax.axvline(nm, color="k", lw=1.0, ls=":", alpha=0.65, label=f"null={nm:.0f}")
    xl = ax.get_xlim(); ax.axvspan(xl[0], stat_true, alpha=0.07, color="crimson")
    ax.set_xlabel("NLL Gaussian MSE", fontsize=10); ax.set_ylabel("count", fontsize=10)
    ax.set_title(f"z={z:.2f}  p={p:.3f}  gap={gap:.1f}", fontsize=10)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.bar([0], [71.6], color="#55a868", alpha=0.45, edgecolor="k")
    ax2.bar([1], [gap],  color="#55a868", alpha=0.90, edgecolor="k")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["150 ep\n(prev)", f"{EPOCHS} ep\n(now)"], fontsize=10)
    ax2.set_ylabel("null_mean - stat_true\n(smaller = better calibrated)", fontsize=9)
    ax2.set_title("Gap comparison", fontsize=10)
    for xi, g in zip([0, 1], [71.6, gap]):
        ax2.text(xi, g + 1, f"{g:.1f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = OUTDIR / f"kernel_noise_spgp_{EPOCHS}ep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Plot -> {out}")
    print(f"Total: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
