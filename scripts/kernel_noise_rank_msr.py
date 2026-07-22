"""
Rank-matched MSR null on kernel noise — 99 perms, 450 epochs, parallel.

Rank-matched MSR: generate MSR sign-flip reference field per gene, then
rank-assign the original expression values into that spatial ordering.
Preserves exact marginal distribution per gene + approximate SA.
No new values generated — only spatial rearrangement.
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
from scripts.msr_null_smoke_test import SEED, N_RERUNS, base_cfg, build_rank_matched_msr_surrogates

N_PERMS = 99
EPOCHS  = 450
DATASET = "d30_delta0p1_seed1"
CACHE   = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
OUTDIR  = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)

PRIOR = {
    "msr":         dict(null_mean=141758.47, null_std=17.50, stat_true=141699.76, p=0.010, rank=1),
    "spgp":        dict(null_mean=141691.57, null_std=23.83, stat_true=141592.99, p=0.010, rank=1),
    "cholesky_gp": dict(null_mean=141580.81, null_std=30.37, stat_true=141592.99, p=0.640, rank=64),
}


def run(S, A):
    print(f"\n  [Rank-matched MSR] epochs={EPOCHS}  n_perms={N_PERMS}", flush=True)
    t0 = time.time()

    surrogates = build_rank_matched_msr_surrogates(S, A, N_PERMS, seed=SEED + 600)
    n_slots = N_PERMS + 1
    s_batched = np.repeat(S[np.newaxis], n_slots, axis=0).astype(np.float32)
    a_batched = np.empty((n_slots, S.shape[0], A.shape[1]), dtype=np.float32)
    a_batched[0] = A
    for i in range(N_PERMS):
        a_batched[i + 1] = surrogates[i]

    cfg    = replace(base_cfg(N_PERMS, SEED), epochs=EPOCHS)
    device = resolve_device("cuda")
    print(f"    Training {n_slots} parallel slots …", flush=True)
    _, out, _ = train_parallel_isodepth_model(
        S, A, cfg, device=device,
        s_batched=s_batched, a_batched=a_batched,
        model_label=f"RankMSR_e{EPOCHS}",
    )

    elapsed   = time.time() - t0
    stat_true = out.stat_true
    stat_perm = out.stat_perm
    nm = float(stat_perm.mean()); ns = float(stat_perm.std())
    z  = (nm - stat_true) / (ns + 1e-12)
    p  = permutation_p_value(cfg.metric, stat_true, stat_perm)
    rank = int(np.sum(stat_perm < stat_true)) + 1
    print(f"    stat_true={stat_true:.2f}  null_mean={nm:.2f}  z={z:.2f}  "
          f"p={p:.3f}  rank={rank}/{N_PERMS+1}  [{elapsed:.0f}s]", flush=True)
    for i, s in enumerate(stat_perm):
        if (i + 1) % 10 == 0 or i == 0 or i == N_PERMS - 1:
            print(f"    perm {i+1:2d}/{N_PERMS}: {s:.2f}", flush=True)
    return dict(stat_true=stat_true, stat_perm=stat_perm,
                null_mean=nm, null_std=ns, z=z, p=p, rank=rank, elapsed=elapsed)


def plot(r):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Rank-matched MSR null — kernel noise {DATASET} (SA only, no gradient)\n"
        f"Parallel  e={EPOCHS}  n_perms={N_PERMS}  SEED={SEED}",
        fontsize=10, fontweight="bold",
    )
    def sig(p): return "p<0.05 ✓" if p < 0.05 else ("p<0.10 ~" if p < 0.10 else "p≥0.10 ns")

    ax = axes[0]
    ax.hist(r["stat_perm"], bins=25, color="#9467bd", alpha=0.72, edgecolor="k", lw=0.5)
    ax.axvline(r["stat_true"], color="crimson", lw=2.0, ls="--", label=f"true={r['stat_true']:.0f}")
    ax.axvline(r["null_mean"], color="k", lw=1.0, ls=":", alpha=0.65, label=f"null={r['null_mean']:.0f}")
    xl = ax.get_xlim()
    ax.axvspan(xl[0], r["stat_true"], alpha=0.07, color="crimson")
    ax.set_xlabel("NLL Gaussian MSE", fontsize=9)
    ax.set_title(f"Rank-matched MSR\nz={r['z']:.2f}  p={r['p']:.3f}  "
                 f"rank={r['rank']}/{N_PERMS+1}  {sig(r['p'])}", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.tick_params(labelsize=8)

    ax2 = axes[1]
    methods   = ["MSR\n(sign-flip)", "SpGP\n(MEM draw)", "Cholesky-GP\n(param ρ)", "Rank-MSR\n(this run)"]
    nm_list   = [PRIOR["msr"]["null_mean"], PRIOR["spgp"]["null_mean"],
                 PRIOR["cholesky_gp"]["null_mean"], r["null_mean"]]
    ns_list   = [PRIOR["msr"]["null_std"], PRIOR["spgp"]["null_std"],
                 PRIOR["cholesky_gp"]["null_std"], r["null_std"]]
    tr_list   = [PRIOR["msr"]["stat_true"], PRIOR["spgp"]["stat_true"],
                 PRIOR["cholesky_gp"]["stat_true"], r["stat_true"]]
    p_list    = [PRIOR["msr"]["p"], PRIOR["spgp"]["p"],
                 PRIOR["cholesky_gp"]["p"], r["p"]]
    colors    = ["#DD8452", "#55a868", "#2ca02c", "#9467bd"]
    xs = np.arange(4)
    ax2.bar(xs, nm_list, yerr=ns_list, capsize=5, color=colors, alpha=0.75, edgecolor="k")
    for xi, t, nm, ns, pv in zip(xs, tr_list, nm_list, ns_list, p_list):
        ax2.plot([xi - 0.3, xi + 0.3], [t, t], color="crimson", lw=2.5, zorder=5)
        ax2.text(xi, nm + ns * 1.5, f"p={pv:.3f}", ha="center", fontsize=8, color="crimson")
    ax2.set_xticks(xs); ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylabel("reconstruction loss", fontsize=9)
    ax2.set_title("Null mean ± std vs stat_true (—)", fontsize=9)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = OUTDIR / f"kernel_noise_rank_msr_e{EPOCHS}_n{N_PERMS}_{DATASET}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out}", flush=True)


def main():
    t0 = time.time()
    print("=" * 65)
    print(f"RANK-MATCHED MSR NULL — {DATASET}")
    print(f"n_perms={N_PERMS}  epochs={EPOCHS}  n_reruns={N_RERUNS}  SEED={SEED}")
    print("=" * 65)
    ds = load_dataset_cache(CACHE)
    S, A = ds.S, ds.A
    print(f"Data: {S.shape[0]} cells × {A.shape[1]} genes")
    r = run(S, A)
    print("\n" + "=" * 65)
    for label, d in [("MSR sign-flip (seq,e150)", PRIOR["msr"]),
                     ("SpGP par e=450", PRIOR["spgp"]),
                     ("Cholesky-GP par e=450", PRIOR["cholesky_gp"])]:
        print(f"  {label:<28}: p={d['p']:.3f}  rank={d['rank']}/100")
    print(f"  {'Rank-MSR par e=450':<28}: p={r['p']:.3f}  rank={r['rank']}/{N_PERMS+1}")
    print("=" * 65)
    plot(r)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
