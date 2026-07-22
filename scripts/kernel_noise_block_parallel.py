"""
Block-permutation null on kernel noise — 99 perms, 450 epochs, parallel.

Head-to-head with rank-matched MSR (same training settings).
Block perm permutes coordinates in raw physical space; expression fixed.
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

from data import raw_coordinates_from_standardized, standardize_coordinate_batch
from experiments.kernel_noise_study import load_dataset_cache
from methods.block_permutation import block_stats, build_block_permuted_coordinate_batch
from methods.metrics import permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model
from scripts.msr_null_smoke_test import SEED, N_RERUNS, base_cfg

N_PERMS = 99
EPOCHS  = 450
DATASET = "d30_delta0p1_seed1"
CACHE   = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
OUTDIR  = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)

BLOCK_RADIUS_UM = 60.0
UM_PER_UNIT     = 1000.0

PRIOR = {
    "rank_msr":    dict(null_mean=141692.16, null_std=23.0, stat_true=141592.99, p=0.010, rank=1),
    "cholesky_gp": dict(null_mean=141580.81, null_std=30.37, stat_true=141592.99, p=0.640, rank=64),
    "three_way":   dict(null_mean=141784.0, stat_true=141663.0, z=5.16, p=0.05, n_perms=19, epochs=150),
}


def run(ds):
    S, A, meta = ds.S, ds.A, ds.meta
    S_raw = raw_coordinates_from_standardized(S, meta)
    um_per_unit = float(meta.get("coordinate_um_per_unit") or UM_PER_UNIT)

    stats = block_stats(S_raw, BLOCK_RADIUS_UM, um_per_unit, block_shape="hexagon")
    print(f"\n  [Block perm r={BLOCK_RADIUS_UM}µm] epochs={EPOCHS}  n_perms={N_PERMS}", flush=True)
    print(f"    {stats['n_blocks']} blocks  cells/block median={stats['median_cells']:.0f}", flush=True)

    t0 = time.time()
    s_batched_raw = build_block_permuted_coordinate_batch(
        S_raw,
        radius_um=BLOCK_RADIUS_UM,
        coordinate_um_per_unit=um_per_unit,
        n_perms=N_PERMS,
        seed=SEED,
        block_jitter=True,
        block_shape="hexagon",
    )
    s_batched = standardize_coordinate_batch(s_batched_raw, meta)

    cfg    = replace(base_cfg(N_PERMS, SEED), epochs=EPOCHS)
    device = resolve_device("cuda")
    print(f"    Training {N_PERMS + 1} parallel slots …", flush=True)
    _, out, _ = train_parallel_isodepth_model(
        S, A, cfg, device=device,
        s_batched=s_batched,
        model_label=f"BlockPerm_r{int(BLOCK_RADIUS_UM)}_e{EPOCHS}",
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
        f"Block perm null — kernel noise {DATASET} (SA only, no gradient)\n"
        f"r={BLOCK_RADIUS_UM}µm  parallel  e={EPOCHS}  n_perms={N_PERMS}  SEED={SEED}",
        fontsize=10, fontweight="bold",
    )
    def sig(p): return "p<0.05 ✓" if p < 0.05 else ("p<0.10 ~" if p < 0.10 else "p≥0.10 ns")

    ax = axes[0]
    ax.hist(r["stat_perm"], bins=25, color="#27ae60", alpha=0.72, edgecolor="k", lw=0.5)
    ax.axvline(r["stat_true"], color="crimson", lw=2.0, ls="--", label=f"true={r['stat_true']:.0f}")
    ax.axvline(r["null_mean"], color="k", lw=1.0, ls=":", alpha=0.65, label=f"null={r['null_mean']:.0f}")
    xl = ax.get_xlim()
    ax.axvspan(xl[0], r["stat_true"], alpha=0.07, color="crimson")
    ax.set_xlabel("NLL Gaussian MSE", fontsize=9)
    ax.set_title(f"Block perm r={BLOCK_RADIUS_UM}µm\nz={r['z']:.2f}  p={r['p']:.3f}  "
                 f"rank={r['rank']}/{N_PERMS+1}  {sig(r['p'])}", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.tick_params(labelsize=8)

    ax2 = axes[1]
    methods = ["Rank-MSR\ne=450 n=99", "Block perm\ne=450 n=99", "Cholesky-GP\n(calibrated)"]
    nm_list = [PRIOR["rank_msr"]["null_mean"], r["null_mean"], PRIOR["cholesky_gp"]["null_mean"]]
    ns_list = [PRIOR["rank_msr"]["null_std"], r["null_std"], PRIOR["cholesky_gp"]["null_std"]]
    tr_list = [PRIOR["rank_msr"]["stat_true"], r["stat_true"], PRIOR["cholesky_gp"]["stat_true"]]
    p_list  = [PRIOR["rank_msr"]["p"], r["p"], PRIOR["cholesky_gp"]["p"]]
    colors  = ["#9467bd", "#27ae60", "#2ca02c"]
    xs = np.arange(3)
    ax2.bar(xs, nm_list, yerr=ns_list, capsize=5, color=colors, alpha=0.75, edgecolor="k")
    for xi, t, nm, ns, pv in zip(xs, tr_list, nm_list, ns_list, p_list):
        ax2.plot([xi - 0.3, xi + 0.3], [t, t], color="crimson", lw=2.5, zorder=5)
        ax2.text(xi, nm + ns * 1.5, f"p={pv:.3f}", ha="center", fontsize=8, color="crimson")
    ax2.set_xticks(xs); ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylabel("reconstruction loss", fontsize=9)
    ax2.set_title("Null mean ± std vs stat_true (—)", fontsize=9)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = OUTDIR / f"kernel_noise_block_r{int(BLOCK_RADIUS_UM)}_e{EPOCHS}_n{N_PERMS}_{DATASET}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out}", flush=True)


def main():
    t0 = time.time()
    print("=" * 65)
    print(f"BLOCK PERM NULL — {DATASET}")
    print(f"r={BLOCK_RADIUS_UM}µm  n_perms={N_PERMS}  epochs={EPOCHS}  "
          f"n_reruns={N_RERUNS}  SEED={SEED}")
    print("=" * 65)
    ds = load_dataset_cache(CACHE)
    print(f"Data: {ds.S.shape[0]} cells × {ds.A.shape[1]} genes")
    r = run(ds)
    print("\n" + "=" * 65)
    print(f"  {'Rank-MSR par e=450':<28}: p={PRIOR['rank_msr']['p']:.3f}  rank={PRIOR['rank_msr']['rank']}/100")
    print(f"  {'Block perm par e=450':<28}: p={r['p']:.3f}  rank={r['rank']}/{N_PERMS+1}")
    print(f"  {'Cholesky-GP par e=450':<28}: p={PRIOR['cholesky_gp']['p']:.3f}  rank={PRIOR['cholesky_gp']['rank']}/100")
    print(f"  {'3-way block (e=150 n=19)':<28}: z={PRIOR['three_way']['z']:.2f}  p={PRIOR['three_way']['p']:.3f}")
    print("=" * 65)
    plot(r)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
