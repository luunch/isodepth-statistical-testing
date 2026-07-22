"""
4-null comparison on kernel noise d30_delta0p1_seed1 (SA only, no gradient).

Nulls compared
--------------
  1. Coord shuffle   — permutes S; destroys all spatial structure
  2. MSR plain       — sign-flip spectral coefficients; preserves per-gene SA but
                       scrambles spatial orientation → rougher optimization landscape
  3. MSR-Recolored   — MSR + restores gene-gene covariance; no effect here because
                       kernel-noise covariance is near-zero
  4. SpGP (new)      — draws FRESH realizations from the estimated SA spectral
                       density; same distribution AND same landscape difficulty
                       as real data → should be properly calibrated

Expected
--------
  Nulls 1-3: fire (stat_true < all null stats)
  SpGP: NOT fire (stat_true comparable to null stats)
"""
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
from scripts.msr_null_smoke_test import (
    EPOCHS, N_RERUNS, SEED, base_cfg,
    build_msr_surrogates, build_msr_recolored_surrogates,
    build_spectral_gp_surrogates,
)

N_PERMS    = 10
DATASET    = "d30_delta0p1_seed1"
CACHE_PATH = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
OUTDIR     = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)

COLORS  = {
    "coord_shuffle": "#4C72B0",
    "msr":           "#DD8452",
    "msr_recolored": "#8172B3",
    "spectral_gp":   "#55a868",
}
MLABELS = {
    "coord_shuffle": "Coord shuffle",
    "msr":           "MSR plain",
    "msr_recolored": "MSR-Recolored",
    "spectral_gp":   "SpGP (fresh draw)",
}
METHODS = ["coord_shuffle", "msr", "msr_recolored", "spectral_gp"]


def _pack(method, stat_true, stat_perm, elapsed):
    nm = float(stat_perm.mean()); ns = float(stat_perm.std())
    z  = (nm - stat_true) / (ns + 1e-12)
    p  = permutation_p_value("nll_gaussian_mse", stat_true, stat_perm)
    return dict(method=method, stat_true=stat_true, stat_perm=stat_perm.copy(),
                null_mean=nm, null_std=ns, z=z, p=p, elapsed=elapsed)


def run_coord_shuffle(S, A):
    print(f"\n  [Coord shuffle]", flush=True)
    t0     = time.time()
    cfg    = base_cfg(N_PERMS, SEED)
    device = resolve_device("cuda")
    _, out, _ = train_parallel_isodepth_model(S, A, cfg, device=device)
    r = _pack("coord_shuffle", out.stat_true, out.stat_perm, time.time()-t0)
    print(f"    stat_true={r['stat_true']:.2f}  null_mean={r['null_mean']:.2f}  z={r['z']:.2f}  p={r['p']:.3f}  [{r['elapsed']:.0f}s]", flush=True)
    return r


def _run_expr_null(S, A, surrogates, method):
    device = resolve_device("cuda")
    _, out_true, _ = train_parallel_isodepth_model(S, A, base_cfg(0, SEED), device=device)
    stat_true = out_true.stat_true
    print(f"    stat_true={stat_true:.2f}", flush=True)
    stat_perm = np.empty(N_PERMS, dtype=np.float64)
    for i in range(N_PERMS):
        _, oi, _ = train_parallel_isodepth_model(S, surrogates[i], base_cfg(0, SEED), device=device)
        stat_perm[i] = oi.stat_true
        print(f"    perm {i+1:2d}/{N_PERMS}: {stat_perm[i]:.2f}", flush=True)
    return stat_true, stat_perm


def run_msr(S, A):
    print(f"\n  [MSR plain]", flush=True)
    t0   = time.time()
    surr = build_msr_surrogates(S, A, N_PERMS, seed=SEED+500)
    st, sp = _run_expr_null(S, A, surr, "msr")
    r = _pack("msr", st, sp, time.time()-t0)
    print(f"    null_mean={r['null_mean']:.2f}  z={r['z']:.2f}  p={r['p']:.3f}  [{r['elapsed']:.0f}s]", flush=True)
    return r


def run_msr_recolored(S, A):
    print(f"\n  [MSR-Recolored]", flush=True)
    t0   = time.time()
    surr = build_msr_recolored_surrogates(S, A, N_PERMS, seed=SEED+500)
    st, sp = _run_expr_null(S, A, surr, "msr_recolored")
    r = _pack("msr_recolored", st, sp, time.time()-t0)
    print(f"    null_mean={r['null_mean']:.2f}  z={r['z']:.2f}  p={r['p']:.3f}  [{r['elapsed']:.0f}s]", flush=True)
    return r


def run_spectral_gp(S, A):
    print(f"\n  [SpGP — fresh draw]", flush=True)
    t0   = time.time()
    surr = build_spectral_gp_surrogates(S, A, N_PERMS, seed=SEED+500)
    st, sp = _run_expr_null(S, A, surr, "spectral_gp")
    r = _pack("spectral_gp", st, sp, time.time()-t0)
    print(f"    null_mean={r['null_mean']:.2f}  z={r['z']:.2f}  p={r['p']:.3f}  [{r['elapsed']:.0f}s]", flush=True)
    return r


def plot_results(results):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(
        f"4-null comparison — Kernel noise {DATASET} (SA only, no gradient)\n"
        f"n_perms={N_PERMS}  epochs={EPOCHS}  SEED={SEED} fixed for all runs",
        fontsize=11, fontweight="bold",
    )
    rdict = {r["method"]: r for r in results}

    def sig(p):
        return "p<0.05 ✓" if p < 0.05 else ("p<0.10 ~" if p < 0.10 else "p≥0.10 ns")

    for col, method in enumerate(METHODS):
        ax = axes[col]
        r  = rdict[method]
        ax.hist(r["stat_perm"], bins=6, color=COLORS[method], alpha=0.72,
                edgecolor="k", linewidth=0.5)
        ax.axvline(r["stat_true"], color="crimson", lw=2.0, ls="--",
                   label=f"true={r['stat_true']:.0f}")
        ax.axvline(r["null_mean"], color="k", lw=1.0, ls=":", alpha=0.65,
                   label=f"null={r['null_mean']:.0f}")
        xl = ax.get_xlim()
        ax.axvspan(xl[0], r["stat_true"], alpha=0.07, color="crimson")
        ax.set_xlabel("NLL Gaussian MSE", fontsize=9)
        ax.set_title(f"{MLABELS[method]}\nz={r['z']:.2f}  p={r['p']:.3f}  {sig(r['p'])}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    # Summary bar
    ax_s  = axes[4]
    xs    = np.arange(len(METHODS))
    ys    = [rdict[m]["null_mean"] for m in METHODS]
    errs  = [rdict[m]["null_std"]  for m in METHODS]
    clrs  = [COLORS[m]             for m in METHODS]
    ax_s.bar(xs, ys, yerr=errs, capsize=5, color=clrs, alpha=0.72,
             edgecolor="k", linewidth=0.8)
    for xi, m in zip(xs, METHODS):
        r = rdict[m]
        ax_s.plot([xi-0.28, xi+0.28], [r["stat_true"]]*2, color="crimson", lw=2.5, zorder=5)
        sym = "**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else "ns")
        ax_s.text(xi, r["null_mean"] + r["null_std"]*1.5,
                  f"p={r['p']:.3f}\n{sym}", ha="center", fontsize=7.5, color="crimson")
    ax_s.set_xticks(xs)
    ax_s.set_xticklabels([MLABELS[m].replace(" ", "\n") for m in METHODS], fontsize=8)
    ax_s.set_ylabel("reconstruction loss", fontsize=9)
    ax_s.set_title("Null mean ± std\nvs stat_true (—)", fontsize=9)
    ax_s.tick_params(labelsize=8)

    plt.tight_layout()
    out = OUTDIR / f"kernel_noise_four_null_{DATASET}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out}", flush=True)


def main():
    t_total = time.time()
    print("="*65)
    print(f"4-NULL COMPARISON — {DATASET}")
    print(f"n_perms={N_PERMS}  epochs={EPOCHS}  SEED={SEED}")
    print("="*65)

    dataset = load_dataset_cache(CACHE_PATH)
    S, A = dataset.S, dataset.A
    print(f"\nData: {S.shape[0]} cells × {A.shape[1]} genes  "
          f"(SA only, ρ=30µm, δ=0.1, no gradient)")

    results = [
        run_coord_shuffle(S, A),
        run_msr(S, A),
        run_msr_recolored(S, A),
        run_spectral_gp(S, A),
    ]

    print("\n" + "="*72)
    print(f"{'Method':<18} {'stat_true':>10} {'null_mean':>10} {'z':>7} {'p':>6}  result")
    print("-"*72)
    for r in results:
        res = "FIRE" if r["p"] < 0.05 else ("~" if r["p"] < 0.10 else "ns")
        print(f"{r['method']:<18} {r['stat_true']:>10.2f} {r['null_mean']:>10.2f} "
              f"{r['z']:>7.2f} {r['p']:>6.3f}  {res}")
    print("="*72)

    plot_results(results)
    print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
