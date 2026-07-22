"""
Joint truncated MSR on positive controls: STARmap cortex + MOSTA E13.5 liver.

Compares joint trunc-MSR vs coord shuffle (same parallel training settings).
Truncate scale matches block-permutation radius used in prior smoke tests.
"""
from __future__ import annotations
import sys, time
from dataclasses import replace
from pathlib import Path

import anndata as ad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.h5ad_loader import _select_hvg_mask_from_counts
from methods.metrics import permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model
from scripts.msr_null_smoke_test import (
    SEED, N_RERUNS, EPOCHS, N_PERMS, base_cfg,
    build_joint_truncated_msr_surrogates, load_starmap,
)

OUTDIR = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRUNCATE_UM = 50.0
CALIBRATION_UM = 100.0  # fixed reference; truncate alone sets cutoff scale

# Prior smoke-test reference (19 perms, 150 epochs)
PRIOR = {
    "starmap_cortex": {
        "coord_shuffle": dict(z=45.5, p=0.05),
        "msr_per_gene": dict(z=16.5, p=0.05),
        "joint_trunc_100um": dict(z=0.51, p=0.40),
    },
    "liver_e135": {
        "coord_shuffle": dict(z=28.6, p=0.05),
        "msr_per_gene": dict(z=20.4, p=0.05),
        "block_perm": dict(z=4.46, p=0.05),
        "joint_trunc_150um": dict(z=1.06, p=0.20),
    },
}


def load_liver_e135(top_genes: int = 1000) -> tuple[np.ndarray, np.ndarray, str]:
    h5ad = REPO / "data/h5ad/mouse-organogenesis/E13.5_E1S1.MOSTA.h5ad"
    adata = ad.read_h5ad(h5ad)
    adata = adata[adata.obs["annotation"] == "Liver"].copy()
    raw = adata.layers["count"]
    counts = np.array(raw.toarray() if sp.issparse(raw) else raw, dtype=np.float32)
    hvg = _select_hvg_mask_from_counts(counts, top_genes)
    counts = counts[:, hvg]
    row_sums = counts.sum(axis=1, keepdims=True).clip(1.0)
    A = np.log1p(counts / row_sums * 1e4)
    mu = A.mean(0, keepdims=True)
    sd = A.std(0, keepdims=True).clip(1e-8)
    A = ((A - mu) / sd).astype(np.float32)
    S_raw = np.array(adata.obsm["spatial"], dtype=np.float32)
    S = ((S_raw - S_raw.mean(0, keepdims=True)) / S_raw.std(0, keepdims=True).clip(1e-8)).astype(np.float32)
    return S, A, f"liver E13.5 (N={len(adata)}, G={A.shape[1]})"


def run_joint_trunc(S, A, truncate_um: float, label: str, seed_offset: int):
    print(f"\n  [Joint trunc-MSR] {label}  truncate>{truncate_um}µm", flush=True)
    t0 = time.time()
    surrogates = build_joint_truncated_msr_surrogates(
        S, A, N_PERMS, seed=SEED + seed_offset,
        truncate_scale_um=truncate_um,
        calibration_um=CALIBRATION_UM,
    )
    n_slots = N_PERMS + 1
    s_batched = np.repeat(S[np.newaxis], n_slots, axis=0).astype(np.float32)
    a_batched = np.empty((n_slots, S.shape[0], A.shape[1]), dtype=np.float32)
    a_batched[0] = A
    for i in range(N_PERMS):
        a_batched[i + 1] = surrogates[i]

    cfg = base_cfg(N_PERMS, SEED)
    device = resolve_device("cuda")
    _, out, _ = train_parallel_isodepth_model(
        S, A, cfg, device=device,
        s_batched=s_batched, a_batched=a_batched,
        model_label=f"JointTrunc_{label}",
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
    return dict(
        method="joint_trunc_msr", label=label, truncate_um=truncate_um,
        stat_true=stat_true, stat_perm=stat_perm,
        null_mean=nm, null_std=ns, z=z, p=p, rank=rank, elapsed=elapsed,
    )


def run_coord_shuffle(S, A, label: str):
    print(f"\n  [Coord shuffle] {label}", flush=True)
    t0 = time.time()
    cfg = base_cfg(N_PERMS, SEED)
    device = resolve_device("cuda")
    _, out, _ = train_parallel_isodepth_model(S, A, cfg, device=device, model_label=f"CoordShuffle_{label}")
    elapsed = time.time() - t0
    stat_true = out.stat_true
    stat_perm = out.stat_perm
    nm = float(stat_perm.mean()); ns = float(stat_perm.std())
    z = (nm - stat_true) / (ns + 1e-12)
    p = permutation_p_value(cfg.metric, stat_true, stat_perm)
    rank = int(np.sum(stat_perm < stat_true)) + 1
    print(f"    stat_true={stat_true:.2f}  null_mean={nm:.2f}  z={z:.2f}  "
          f"p={p:.3f}  rank={rank}/{N_PERMS+1}  [{elapsed:.0f}s]", flush=True)
    return dict(
        method="coord_shuffle", label=label,
        stat_true=stat_true, stat_perm=stat_perm,
        null_mean=nm, null_std=ns, z=z, p=p, rank=rank, elapsed=elapsed,
    )


def plot_dataset(key: str, results: list[dict], truncate_um: float):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(
        f"Joint trunc-MSR — {results[0]['label']}\n"
        f"truncate>{truncate_um}µm  parallel  e={EPOCHS}  n_perms={N_PERMS}",
        fontsize=10, fontweight="bold",
    )
    joint = next(r for r in results if r["method"] == "joint_trunc_msr")
    coord = next(r for r in results if r["method"] == "coord_shuffle")

    for ax, r, color, title in [
        (axes[0], coord, "#4C72B0", "Coord shuffle"),
        (axes[1], joint, "#e377c2", f"Joint trunc-MSR >{truncate_um}µm"),
    ]:
        ax.hist(r["stat_perm"], bins=12, color=color, alpha=0.75, edgecolor="k", lw=0.5)
        ax.axvline(r["stat_true"], color="crimson", lw=2, ls="--")
        ax.axvline(r["null_mean"], color="k", lw=1, ls=":", alpha=0.65)
        sig = "p<0.05 ✓" if r["p"] < 0.05 else ("p<0.10" if r["p"] < 0.10 else "ns")
        ax.set_title(f"{title}\nz={r['z']:.1f}  p={r['p']:.3f}  rank={r['rank']}/{N_PERMS+1}  {sig}", fontsize=9)
        ax.set_xlabel("NLL Gaussian MSE", fontsize=8)
        ax.tick_params(labelsize=8)

    out = OUTDIR / f"joint_trunc_msr_{key}_r{int(truncate_um)}_e{EPOCHS}_n{N_PERMS}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Plot → {out}", flush=True)


def main():
    t0 = time.time()
    print("=" * 65)
    print("JOINT TRUNCATED MSR — POSITIVE CONTROLS")
    print(f"n_perms={N_PERMS}  epochs={EPOCHS}  truncate>{TRUNCATE_UM}µm  cal={CALIBRATION_UM}µm  SEED={SEED}")
    print("=" * 65)

    all_results: dict[str, list[dict]] = {}

    # STARmap full cortex
    S, A, lbl = load_starmap()
    trunc = TRUNCATE_UM
    print(f"\n### STARmap {lbl}  truncate>{trunc}µm")
    rs = [
        run_coord_shuffle(S, A, "starmap_cortex"),
        run_joint_trunc(S, A, trunc, "starmap_cortex", seed_offset=800),
    ]
    all_results["starmap_cortex"] = rs
    plot_dataset("starmap_cortex", rs, trunc)

    # MOSTA liver
    S, A, lbl = load_liver_e135()
    trunc = TRUNCATE_UM
    print(f"\n### {lbl}  truncate>{trunc}µm")
    rs = [
        run_coord_shuffle(S, A, "liver_e135"),
        run_joint_trunc(S, A, trunc, "liver_e135", seed_offset=900),
    ]
    all_results["liver_e135"] = rs
    plot_dataset("liver_e135", rs, trunc)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for key, prior in PRIOR.items():
        trunc = TRUNCATE_UM
        print(f"\n{key} (truncate>{trunc}µm):")
        if "coord_shuffle" in prior:
            print(f"  prior coord shuffle     z={prior['coord_shuffle']['z']:.1f}  p={prior['coord_shuffle']['p']:.3f}")
        if "msr_per_gene" in prior:
            print(f"  prior MSR per-gene      z={prior['msr_per_gene']['z']:.1f}  p={prior['msr_per_gene']['p']:.3f}")
        if "block_perm" in prior:
            print(f"  prior block perm        z={prior['block_perm']['z']:.1f}  p={prior['block_perm']['p']:.3f}")
        for r in all_results[key]:
            print(f"  this {r['method']:<18} z={r['z']:.2f}  p={r['p']:.3f}  rank={r['rank']}/{N_PERMS+1}")
    print("=" * 65)
    print(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
