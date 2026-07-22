"""Multiscale spatial/gene-covariance null smoke test on kernel-noise data.

This is a standalone prototype for the proposed common-axis null:

  1. Build a Moran spatial basis from the fixed cell coordinates.
  2. Project expression into spatial modes.
  3. Bin modes by spatial scale and estimate a gene-gene covariance per bin.
  4. Draw fresh Gaussian coefficients per bin, preserving gene covariance by scale.
  5. Rank-map each gene back to its observed marginal values.
  6. Train GASTON/isodepth on observed + null matrices in one parallel batch.

The goal is calibration on a negative-control synthetic kernel-noise dataset
with spatial autocorrelation but no true gradient.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.kernel_noise_study import load_dataset_cache
from methods.metrics import permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model
from scripts.msr_null_smoke_test import SEED, N_RERUNS, base_cfg, _msr_basis


N_PERMS = 9
EPOCHS = 75
DATASET = "d30_delta0p1_seed1"
CACHE_PATH = REPO / f"results/experiments/kernel_noise_study/datasets/{DATASET}.npz"
# Store smoke artifacts under the writable home root; the repo's results path is
# a scratch symlink that can be read-only under Codex's sandbox.
OUTDIR = Path("/home/ajain71/isodepth_smoke_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

N_SCALE_BINS = 5
COV_SHRINKAGE = 0.10


def _rank_map_to_observed_marginals(reference: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Assign observed per-gene sorted values into the reference per-gene rank order."""
    ref = np.asarray(reference, dtype=np.float32)
    obs = np.asarray(observed, dtype=np.float32)
    n_cells, n_genes = obs.shape
    sorted_vals = np.sort(obs, axis=0)
    ranks = np.argsort(ref, axis=0)
    mapped = np.empty((n_cells, n_genes), dtype=np.float32)
    for g in range(n_genes):
        mapped[ranks[:, g], g] = sorted_vals[:, g]
    return mapped


def _scale_bin_indices(eigvals: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Bin Moran modes from long-range to short-range using |eigenvalue| quantiles."""
    abs_eig = np.abs(np.asarray(eigvals, dtype=np.float64))
    order = np.argsort(abs_eig)
    chunks = np.array_split(order, int(n_bins))
    return [chunk.astype(np.int64) for chunk in chunks if chunk.size > 0]


def _shrink_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - float(shrinkage)) * cov + float(shrinkage) * diag
    eps = max(1e-8, 1e-6 * float(np.trace(shrunk)) / max(shrunk.shape[0], 1))
    return shrunk + eps * np.eye(shrunk.shape[0], dtype=np.float64)


def build_multiscale_gene_cov_surrogates(
    S: np.ndarray,
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    *,
    n_scale_bins: int = N_SCALE_BINS,
    cov_shrinkage: float = COV_SHRINKAGE,
) -> tuple[np.ndarray, dict]:
    """Fresh multiscale spatial draws with gene covariance preserved per scale bin."""
    n_cells, n_genes = A.shape
    print(
        f"    Multiscale gene-cov null: N={n_cells}, G={n_genes}, "
        f"bins={n_scale_bins}, shrink={cov_shrinkage}",
        flush=True,
    )
    t0 = time.time()
    V, C, eigvals = _msr_basis(S, A, return_eigvals=True)
    bins = _scale_bin_indices(eigvals, n_scale_bins)
    rng = np.random.default_rng(seed)

    bin_payload: list[dict] = []
    bin_factors: list[tuple[np.ndarray, np.ndarray]] = []
    total_energy = float(np.sum(np.asarray(C, dtype=np.float64) ** 2))

    for bin_index, mode_idx in enumerate(bins):
        C_bin = np.asarray(C[mode_idx, :], dtype=np.float64)
        cov = (C_bin.T @ C_bin) / max(int(C_bin.shape[0]), 1)
        cov = _shrink_covariance(cov, cov_shrinkage)
        L = np.linalg.cholesky(cov)
        V_bin = np.asarray(V[:, mode_idx], dtype=np.float64)
        bin_factors.append((V_bin, L))

        energy = float(np.sum(C_bin**2))
        abs_eig = np.abs(np.asarray(eigvals[mode_idx], dtype=np.float64))
        bin_payload.append(
            {
                "bin_index": int(bin_index),
                "n_modes": int(mode_idx.size),
                "abs_eig_min": float(abs_eig.min()),
                "abs_eig_max": float(abs_eig.max()),
                "energy_fraction": float(energy / max(total_energy, 1e-12)),
                "mean_gene_cov_diag": float(np.mean(np.diag(cov))),
                "mean_abs_gene_corr": _mean_abs_offdiag_corr(cov),
            }
        )
        print(
            f"      bin {bin_index}: modes={mode_idx.size:4d} "
            f"|eig|=[{abs_eig.min():.3g},{abs_eig.max():.3g}] "
            f"energy={energy / max(total_energy, 1e-12):.3f} "
            f"mean|corr|={bin_payload[-1]['mean_abs_gene_corr']:.3f}",
            flush=True,
        )

    surrogates = np.empty((n_surrogates, n_cells, n_genes), dtype=np.float32)
    ts = time.time()
    for b in range(n_surrogates):
        draw = np.zeros((n_cells, n_genes), dtype=np.float64)
        for V_bin, L in bin_factors:
            Z = rng.standard_normal((V_bin.shape[1], n_genes))
            C_new = Z @ L.T
            draw += V_bin @ C_new

        # Stabilize before empirical-marginal rank mapping.
        draw = draw - draw.mean(axis=0, keepdims=True)
        draw = draw / np.clip(draw.std(axis=0, keepdims=True), 1e-8, None)
        surrogates[b] = _rank_map_to_observed_marginals(draw.astype(np.float32), A)

    print(
        f"    Surrogates: {time.time() - ts:.1f}s | total null build: {time.time() - t0:.1f}s",
        flush=True,
    )

    corr_o = np.corrcoef(np.asarray(A, dtype=np.float64).T)
    corr_s = np.corrcoef(np.asarray(surrogates[0], dtype=np.float64).T)
    tri = np.triu_indices(n_genes, k=1)
    diagnostics = {
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_scale_bins": int(len(bins)),
        "cov_shrinkage": float(cov_shrinkage),
        "bins": bin_payload,
        "orig_mean_abs_gene_corr": float(np.mean(np.abs(corr_o[tri]))),
        "surr0_mean_abs_gene_corr": float(np.mean(np.abs(corr_s[tri]))),
        "orig_var": float(np.mean(np.asarray(A, dtype=np.float32) ** 2)),
        "surr0_var": float(np.mean(surrogates[0] ** 2)),
        "surr0_sorted_max_diff": float(
            np.max(np.abs(np.sort(np.asarray(A, dtype=np.float32), axis=0) - np.sort(surrogates[0], axis=0)))
        ),
    }
    print(
        f"    Marginal check: sorted max diff={diagnostics['surr0_sorted_max_diff']:.2e}; "
        f"mean|gene corr| orig={diagnostics['orig_mean_abs_gene_corr']:.3f} "
        f"surr0={diagnostics['surr0_mean_abs_gene_corr']:.3f}",
        flush=True,
    )
    return surrogates, diagnostics


def _mean_abs_offdiag_corr(cov: np.ndarray) -> float:
    diag = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(diag, diag)
    tri = np.triu_indices(cov.shape[0], k=1)
    return float(np.mean(np.abs(corr[tri]))) if tri[0].size else 0.0


def run_smoke(S: np.ndarray, A: np.ndarray) -> dict:
    print(f"\n  [Multiscale gene-cov null] epochs={EPOCHS} n_perms={N_PERMS}", flush=True)
    t0 = time.time()
    surrogates, null_diag = build_multiscale_gene_cov_surrogates(
        S,
        A,
        N_PERMS,
        seed=SEED + 820,
    )

    n_slots = N_PERMS + 1
    s_batched = np.repeat(S[np.newaxis, :, :], n_slots, axis=0).astype(np.float32)
    a_batched = np.empty((n_slots, S.shape[0], A.shape[1]), dtype=np.float32)
    a_batched[0] = A
    a_batched[1:] = surrogates

    requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = replace(base_cfg(N_PERMS, SEED), epochs=EPOCHS, device=requested_device)
    device = resolve_device(requested_device)
    print(f"    Device: {device}", flush=True)

    print(f"    Training {n_slots} parallel slots ...", flush=True)
    _, out, _ = train_parallel_isodepth_model(
        S,
        A,
        cfg,
        device=device,
        s_batched=s_batched,
        a_batched=a_batched,
        model_label=f"MultiscaleGeneCov_e{EPOCHS}_n{N_PERMS}",
    )

    stat_true = float(out.stat_true)
    stat_perm = np.asarray(out.stat_perm, dtype=np.float64)
    null_mean = float(stat_perm.mean())
    null_std = float(stat_perm.std())
    z = float((null_mean - stat_true) / (null_std + 1e-12))
    p = float(permutation_p_value(cfg.metric, stat_true, stat_perm))
    rank = int(np.sum(stat_perm < stat_true)) + 1
    elapsed = time.time() - t0
    print(
        f"    stat_true={stat_true:.2f} null_mean={null_mean:.2f} null_std={null_std:.2f} "
        f"z={z:.2f} p={p:.3f} rank={rank}/{N_PERMS + 1} [{elapsed:.0f}s]",
        flush=True,
    )
    return {
        "method": "multiscale_gene_cov_rank",
        "dataset": DATASET,
        "n_perms": int(N_PERMS),
        "epochs": int(EPOCHS),
        "n_reruns": int(N_RERUNS),
        "seed": int(SEED),
        "stat_true": stat_true,
        "stat_perm": stat_perm.tolist(),
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z,
        "p": p,
        "rank": rank,
        "runtime_sec": float(elapsed),
        "null_diagnostics": null_diag,
    }


def plot_result(result: dict) -> Path:
    stat_perm = np.asarray(result["stat_perm"], dtype=np.float64)
    stat_true = float(result["stat_true"])
    null_mean = float(result["null_mean"])
    p = float(result["p"])
    rank = int(result["rank"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(stat_perm, bins=min(12, max(6, N_PERMS // 2)), color="#4c78a8", alpha=0.75, edgecolor="black")
    ax.axvline(stat_true, color="crimson", linestyle="--", linewidth=2.0, label=f"observed={stat_true:.0f}")
    ax.axvline(null_mean, color="black", linestyle=":", linewidth=1.5, label=f"null mean={null_mean:.0f}")
    ax.set_title(
        f"Multiscale gene-cov rank null — {DATASET}\n"
        f"e={EPOCHS}, n={N_PERMS}, p={p:.3f}, rank={rank}/{N_PERMS + 1}",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("NLL Gaussian MSE")
    ax.set_ylabel("Null count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUTDIR / f"kernel_noise_multiscale_gene_cov_e{EPOCHS}_n{N_PERMS}_{DATASET}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print("=" * 72)
    print(f"MULTISCALE GENE-COV NULL SMOKE — {DATASET}")
    print(f"n_perms={N_PERMS} epochs={EPOCHS} n_reruns={N_RERUNS} seed={SEED}")
    print("=" * 72)
    dataset = load_dataset_cache(CACHE_PATH)
    S = np.asarray(dataset.S, dtype=np.float32)
    A = np.asarray(dataset.A, dtype=np.float32)
    print(f"Data: {S.shape[0]} cells x {A.shape[1]} genes")
    result = run_smoke(S, A)
    plot_path = plot_result(result)
    json_path = OUTDIR / f"kernel_noise_multiscale_gene_cov_e{EPOCHS}_n{N_PERMS}_{DATASET}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"JSON -> {json_path}")
    print(f"Plot -> {plot_path}")


if __name__ == "__main__":
    main()
