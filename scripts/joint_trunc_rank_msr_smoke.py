"""Smoke test: joint truncated rank-MSR on kernel noise + liver."""
from __future__ import annotations
import sys, time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.kernel_noise_study import load_dataset_cache
from methods.metrics import permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model
from scripts.joint_trunc_msr_positive_controls import load_liver_e135, TRUNCATE_UM, CALIBRATION_UM
from scripts.msr_null_smoke_test import (
    SEED, N_PERMS, EPOCHS, base_cfg, build_joint_truncated_rank_msr_surrogates,
)

DATASETS = [
    ("kernel_noise", REPO / "results/experiments/kernel_noise_study/datasets/d30_delta0p1_seed1.npz", None),
    ("liver_e135", None, load_liver_e135),
]


def run(label: str, S, A, shared_rank: bool):
    tag = "shared" if shared_rank else "pergene"
    print(f"\n=== {label} joint trunc rank-MSR ({tag}) ===", flush=True)
    t0 = time.time()
    surr = build_joint_truncated_rank_msr_surrogates(
        S, A, N_PERMS, seed=SEED + 950 + int(shared_rank),
        truncate_scale_um=TRUNCATE_UM, calibration_um=CALIBRATION_UM,
        shared_rank=shared_rank,
    )
    n_slots = N_PERMS + 1
    s_batched = __import__("numpy").repeat(S[None], n_slots, axis=0).astype("float32")
    a_batched = __import__("numpy").empty((n_slots, S.shape[0], A.shape[1]), dtype="float32")
    a_batched[0] = A
    for i in range(N_PERMS):
        a_batched[i + 1] = surr[i]
    cfg = replace(base_cfg(N_PERMS, SEED), epochs=EPOCHS)
    _, out, _ = train_parallel_isodepth_model(
        S, A, cfg, device=resolve_device("cuda"),
        s_batched=s_batched, a_batched=a_batched,
        model_label=f"JtRank_{tag}_{label}",
    )
    st, sp = out.stat_true, out.stat_perm
    nm, ns = float(sp.mean()), float(sp.std())
    z = (nm - st) / (ns + 1e-12)
    p = permutation_p_value(cfg.metric, st, sp)
    rank = int(__import__("numpy").sum(sp < st)) + 1
    print(f"  stat_true={st:.2f}  null_mean={nm:.2f}  z={z:.2f}  p={p:.3f}  "
          f"rank={rank}/{N_PERMS+1}  [{time.time()-t0:.0f}s]", flush=True)
    return dict(label=label, shared_rank=shared_rank, z=z, p=p, rank=rank, stat_true=st, null_mean=nm)


def main():
    print(f"truncate>{TRUNCATE_UM}µm  cal={CALIBRATION_UM}µm  n_perms={N_PERMS}  epochs={EPOCHS}")
    for key, cache, loader in DATASETS:
        if cache is not None:
            ds = load_dataset_cache(cache)
            S, A = ds.S, ds.A
            lbl = key
        else:
            S, A, lbl = loader()
        for shared in (False, True):
            run(lbl, S, A, shared)


if __name__ == "__main__":
    main()
