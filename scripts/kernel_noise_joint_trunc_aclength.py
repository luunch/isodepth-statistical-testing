"""
Joint truncated MSR on kernel noise — truncate scale = empirical autocorrelation length.

Estimates pooled half-max autocorrelation length from data, then runs joint trunc-MSR
with truncate_um = calibration_um = that length (same recipe as r=60µm calibration).
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
    SEED, N_RERUNS, base_cfg, build_joint_truncated_msr_surrogates,
    estimate_pooled_autocorr_length_um,
)

N_PERMS = 99
EPOCHS  = 450
DATASET = "d30_delta0p1_seed1"
CACHE   = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
OUTDIR  = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)

PRIOR = {
    "joint_trunc_60um": dict(p=0.740, rank=74, stat_true=141592.99, null_mean=141578.93),
    "cholesky_gp":        dict(p=0.640, rank=64, stat_true=141592.99, null_mean=141580.81),
}


def run(S, A, meta, ac_um: float):
    print(f"\n  [Joint trunc-MSR] truncate=ac_len={ac_um:.1f}µm  "
          f"epochs={EPOCHS}  n_perms={N_PERMS}", flush=True)
    t0 = time.time()
    surrogates = build_joint_truncated_msr_surrogates(
        S, A, N_PERMS, seed=SEED + 750,
        truncate_scale_um=ac_um,
        calibration_um=ac_um,
    )
    n_slots = N_PERMS + 1
    s_batched = np.repeat(S[np.newaxis], n_slots, axis=0).astype(np.float32)
    a_batched = np.empty((n_slots, S.shape[0], A.shape[1]), dtype=np.float32)
    a_batched[0] = A
    for i in range(N_PERMS):
        a_batched[i + 1] = surrogates[i]

    cfg = replace(base_cfg(N_PERMS, SEED), epochs=EPOCHS)
    device = resolve_device("cuda")
    _, out, _ = train_parallel_isodepth_model(
        S, A, cfg, device=device,
        s_batched=s_batched, a_batched=a_batched,
        model_label=f"JointTrunc_ac{int(round(ac_um))}_e{EPOCHS}",
    )
    elapsed = time.time() - t0
    stat_true = out.stat_true
    stat_perm = out.stat_perm
    nm = float(stat_perm.mean()); ns = float(stat_perm.std())
    z = (nm - stat_true) / (ns + 1e-12)
    p = permutation_p_value(cfg.metric, stat_true, stat_perm)
    rank = int(np.sum(stat_perm < stat_true)) + 1
    print(f"    stat_true={stat_true:.2f}  null_mean={nm:.2f}  z={z:.2f}  "
          f"p={p:.3f}  rank={rank}/{N_PERMS+1}  [{elapsed:.0f}s]", flush=True)
    return dict(stat_true=stat_true, stat_perm=stat_perm, null_mean=nm, null_std=ns,
                z=z, p=p, rank=rank, ac_um=ac_um)


def plot_ac_curve(diag, meta, r):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    kernel_rho = float((meta.get("kernel") or {}).get("distance", 0))
    fig.suptitle(
        f"Joint trunc-MSR @ empirical AC length — {DATASET}\n"
        f"truncate={r['ac_um']:.1f}µm (half-max)  generative ρ={kernel_rho}µm  "
        f"e={EPOCHS}  n={N_PERMS}",
        fontsize=10, fontweight="bold",
    )
    ax = axes[0]
    ax.plot(diag["centers_um"], diag["c_hat"], "o-", color="#1f77b4", lw=1.5, ms=4)
    ax.axhline(0.5 * diag["c_max"], color="gray", ls="--", lw=1, label="half-max")
    ax.axvline(r["ac_um"], color="crimson", ls="--", lw=2, label=f"AC half={r['ac_um']:.1f}µm")
    if kernel_rho > 0:
        ax.axvline(kernel_rho, color="green", ls=":", lw=2, label=f"kernel ρ={kernel_rho}µm")
    ax.set_xlabel("distance (µm)", fontsize=9)
    ax.set_ylabel("pooled c(d)", fontsize=9)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    ax.set_title("Empirical pooled covariance vs distance", fontsize=9)

    ax2 = axes[1]
    ax2.hist(r["stat_perm"], bins=25, color="#e377c2", alpha=0.75, edgecolor="k", lw=0.5)
    ax2.axvline(r["stat_true"], color="crimson", lw=2, ls="--")
    ax2.axvline(r["null_mean"], color="k", lw=1, ls=":", alpha=0.65)
    sig = "p<0.05" if r["p"] < 0.05 else "ns"
    ax2.set_title(f"Joint trunc @ AC len  z={r['z']:.2f}  p={r['p']:.3f}  "
                  f"rank={r['rank']}/{N_PERMS+1}  {sig}", fontsize=9)
    ax2.set_xlabel("NLL Gaussian MSE", fontsize=9)
    ax2.tick_params(labelsize=8)

    slug = f"ac{int(round(r['ac_um']))}"
    out = OUTDIR / f"kernel_noise_joint_trunc_{slug}_e{EPOCHS}_n{N_PERMS}_{DATASET}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Plot → {out}", flush=True)


def main():
    t0 = time.time()
    print("=" * 65)
    print(f"JOINT TRUNC MSR @ AUTOCORR LENGTH — {DATASET}")
    print("=" * 65)
    ds = load_dataset_cache(CACHE)
    S, A, meta = ds.S, ds.A, ds.meta
    um_per_unit = float(meta.get("scale_um", 1000.0))
    kernel_rho = float((meta.get("kernel") or {}).get("distance", 0))

    ac_um, diag = estimate_pooled_autocorr_length_um(
        S, A, um_per_unit=um_per_unit, seed=SEED,
    )
    print(f"Empirical half-max AC length: {ac_um:.2f} µm  (generative kernel ρ={kernel_rho} µm)")
    print(f"  c(0 bin)={diag['c_hat_first_bin']:.4f}  c_max={diag['c_max']:.4f}")

    r = run(S, A, meta, ac_um)
    plot_ac_curve(diag, meta, r)

    print("\n" + "=" * 65)
    print(f"  Joint trunc r=60µm (prior)     : p={PRIOR['joint_trunc_60um']['p']:.3f}  "
          f"rank={PRIOR['joint_trunc_60um']['rank']}/100")
    print(f"  Joint trunc @ AC={ac_um:.1f}µm : p={r['p']:.3f}  rank={r['rank']}/{N_PERMS+1}")
    print(f"  Cholesky-GP (prior)            : p={PRIOR['cholesky_gp']['p']:.3f}  "
          f"rank={PRIOR['cholesky_gp']['rank']}/100")
    print("=" * 65)
    print(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
