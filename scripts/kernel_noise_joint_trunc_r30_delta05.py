"""Joint trunc MSR on d30_delta0p5_seed1 — truncate at ρ=30µm."""
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
    SEED, base_cfg, build_joint_truncated_msr_surrogates, estimate_pooled_autocorr_length_um,
)

N_PERMS = 99
EPOCHS = 450
DATASET = "d30_delta0p5_seed1"
CACHE = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
OUTDIR = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)
TRUNCATE_UM = 30.0  # generative kernel ρ


def main():
    ds = load_dataset_cache(CACHE)
    S, A, meta = ds.S, ds.A, ds.meta
    um = float(meta.get("scale_um", 1000))
    rho = float((meta.get("kernel") or {}).get("distance", 30))
    delta = float(meta.get("delta", 0.5))
    ac_um, diag = estimate_pooled_autocorr_length_um(S, A, um_per_unit=um, seed=SEED)
    print(f"Dataset: ρ={rho}µm  δ={delta}  empirical AC half-max={ac_um:.1f}µm")
    print(f"Running joint trunc-MSR truncate=cal={TRUNCATE_UM}µm  n={N_PERMS}  e={EPOCHS}")

    t0 = time.time()
    surr = build_joint_truncated_msr_surrogates(
        S, A, N_PERMS, seed=SEED + 760,
        truncate_scale_um=TRUNCATE_UM, calibration_um=TRUNCATE_UM,
    )
    n_slots = N_PERMS + 1
    s_batched = np.repeat(S[np.newaxis], n_slots, axis=0).astype(np.float32)
    a_batched = np.empty((n_slots, S.shape[0], A.shape[1]), dtype=np.float32)
    a_batched[0] = A
    for i in range(N_PERMS):
        a_batched[i + 1] = surr[i]
    cfg = replace(base_cfg(N_PERMS, SEED), epochs=EPOCHS)
    _, out, _ = train_parallel_isodepth_model(
        S, A, cfg, device=resolve_device("cuda"),
        s_batched=s_batched, a_batched=a_batched,
        model_label=f"JointTrunc_r30_d{delta}",
    )
    st, sp = out.stat_true, out.stat_perm
    nm, ns = float(sp.mean()), float(sp.std())
    z = (nm - st) / (ns + 1e-12)
    p = permutation_p_value(cfg.metric, st, sp)
    rank = int(np.sum(sp < st)) + 1
    print(f"stat_true={st:.2f}  null_mean={nm:.2f}  z={z:.2f}  p={p:.3f}  rank={rank}/{N_PERMS+1}")
    print(f"Compare d30_δ0.1 @ AC≈51µm: p=0.670  |  d30_δ0.1 @ r=60µm: p=0.740")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Joint trunc @ ρ=30µm — {DATASET} (δ={delta})", fontsize=10, fontweight="bold")
    axes[0].plot(diag["centers_um"], diag["c_hat"], "o-")
    axes[0].axvline(TRUNCATE_UM, color="crimson", ls="--", label=f"truncate={TRUNCATE_UM}µm")
    axes[0].axvline(ac_um, color="green", ls=":", label=f"emp AC={ac_um:.0f}µm")
    axes[0].legend(fontsize=8); axes[0].set_xlabel("distance (µm)"); axes[0].set_title("pooled c(d)")
    axes[1].hist(sp, bins=25, color="#e377c2", alpha=0.75, edgecolor="k")
    axes[1].axvline(st, color="crimson", lw=2, ls="--")
    axes[1].axvline(nm, color="k", ls=":",)
    axes[1].set_title(f"z={z:.2f}  p={p:.3f}  rank={rank}/{N_PERMS+1}")
    out = OUTDIR / f"kernel_noise_joint_trunc_r30_e{EPOCHS}_n{N_PERMS}_{DATASET}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Plot → {out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
