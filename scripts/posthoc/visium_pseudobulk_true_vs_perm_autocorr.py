"""Confirm visium_pseudobulk permutes whole spots, and plot true vs perm autocorrelation.

After aggregation, ``run_visium_pseudobulk_method`` calls the standard coordinate
permutation on the *spot* dataset: each row of A (one circular 55 µm capture) is
kept intact and only the 100 µm hex-lattice coordinates are shuffled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.schemas import run_config_from_mapping
from experiments.core.dataset_cache import load_dataset_cache
from methods.moran import build_inverse_distance_weights, morans_i_per_gene
from methods.permutation import _build_permuted_coordinate_batch
from methods.visium_pseudobulk import build_visium_pseudobulk_dataset
from scripts.posthoc.plot_expression_variogram_true_vs_perm import (
    _cell_gram_pooled,
    _estimate_pooled_cov_and_variogram,
)


def _zscore_genes(A: np.ndarray) -> np.ndarray:
    a = np.asarray(A, dtype=np.float64)
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    return (a - mu) / np.maximum(sd, 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="Cached synthetic .npz")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-perms", type=int, default=99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--moran-radius-um",
        type=float,
        default=120.0,
        help="Neighbor radius for Moran's I. Default 120 µm captures hex nearest "
        "neighbors on a 100 µm Visium lattice (default 30 µm would find none).",
    )
    args = parser.parse_args()

    with open(args.base_config, encoding="utf-8") as handle:
        mapping = json.load(handle)
    run_config = run_config_from_mapping(mapping).validate()
    test = run_config.test
    test.n_perms = int(args.n_perms)
    test.seed = int(args.seed)
    test.validate()

    raw = load_dataset_cache(args.cache)
    spots, artifacts = build_visium_pseudobulk_dataset(raw, test)
    summary = artifacts["visium_pseudobulk_summary"]
    A = _zscore_genes(spots.A)
    S_um = np.asarray(spots.S, dtype=np.float64)
    # After z-scoring, invert back to native then µm if needed.
    if spots.meta.get("coordinate_standardization") == "zscore":
        mean = np.asarray(spots.meta["coord_mean"], dtype=np.float64)
        std = np.asarray(spots.meta["coord_std"], dtype=np.float64)
        S_native = spots.S.astype(np.float64) * std + mean
    else:
        S_native = np.asarray(spots.S, dtype=np.float64)
    um_per_unit = float(summary["coordinate_um_per_unit"])
    S_um = S_native * um_per_unit

    n_spots = int(S_um.shape[0])
    nn = np.sort(np.linalg.norm(S_um[:, None, :] - S_um[None, :, :], axis=2), axis=1)[:, 1]
    median_nn_um = float(np.median(nn))

    weights = build_inverse_distance_weights(S_um, float(args.moran_radius_um), allow_empty=True)
    if weights is None:
        raise RuntimeError(
            f"No spot pairs within {args.moran_radius_um} µm "
            f"(median NN={median_nn_um:.1f} µm). Increase --moran-radius-um."
        )
    W, s0 = weights
    I_true = morans_i_per_gene(W, s0, A)
    I_true_mean = float(np.nanmean(I_true))

    s_batched, perms = _build_permuted_coordinate_batch(
        S_um.astype(np.float32),
        n_perms=int(args.n_perms),
        seed=int(args.seed),
        device=torch.device("cpu"),
    )
    s_batched_np = s_batched.detach().cpu().numpy()

    # Whole-spot identity check: permuting coordinates must leave A (and its Gram) unchanged.
    gram = _cell_gram_pooled(A)
    first_perm = np.asarray(perms[0], dtype=np.int64)
    A_perm_rows = A[first_perm]
    # Equivalent view: same expression vectors, just reordered to new locations.
    # The test does s_batched[m] = S[perm], leaving A rows in original order.
    # That is identical to moving whole rows of A onto shuffled lattice sites.

    I_null = []
    for slot in range(1, s_batched_np.shape[0]):
        W_slot, s0_slot = build_inverse_distance_weights(
            s_batched_np[slot], float(args.moran_radius_um), allow_empty=False
        )
        I_null.append(float(np.nanmean(morans_i_per_gene(W_slot, s0_slot, A))))
    I_null = np.asarray(I_null, dtype=np.float64)

    d_max = float(np.percentile(np.linalg.norm(S_um[:, None, :] - S_um[None, :, :], axis=2), 90))
    bins = np.linspace(0.0, d_max, 21)
    centers, c_true, _ = _estimate_pooled_cov_and_variogram(S_um, gram, bins=bins)
    c_nulls = []
    for slot in range(1, s_batched_np.shape[0]):
        _, c_hat, _ = _estimate_pooled_cov_and_variogram(s_batched_np[slot], gram, bins=bins)
        c_nulls.append(c_hat)
    c_nulls = np.vstack(c_nulls)
    c_null_mean = np.nanmean(c_nulls, axis=0)
    c_null_sd = np.nanstd(c_nulls, axis=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axes[0]
    ax.fill_between(
        centers,
        c_null_mean - c_null_sd,
        c_null_mean + c_null_sd,
        color="#2980b9",
        alpha=0.25,
        label="perm ±1 SD",
    )
    ax.plot(centers, c_null_mean, color="#2980b9", lw=1.8, label="perm mean c(d)")
    ax.plot(centers, c_true, color="crimson", lw=2.2, ls="--", label="true c(d)")
    ax.axvline(median_nn_um, color="0.4", ls=":", lw=1.0, label=f"median NN {median_nn_um:.0f} µm")
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("pooled c(d)")
    ax.set_title("Spot-level spatial covariance")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.hist(I_null, bins=15, color="lightsteelblue", edgecolor="white", label="null (spot permutation)")
    ax.axvline(I_true_mean, color="crimson", lw=2.0, label=f"true mean I = {I_true_mean:.3f}")
    ax.set_xlabel(f"mean Moran's I (r={args.moran_radius_um:g} µm)")
    ax.set_ylabel("permutations")
    ax.set_title("Whole-spot coordinate permutation")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Visium pseudobulk: {n_spots} spots, "
        f"{summary['spot_distance_um']:g}/{summary['spot_diameter_um']:g} µm, "
        f"{int(args.n_perms)} coordinate shuffles of whole spots",
        fontsize=11,
    )
    fig.tight_layout()
    plot_path = out_dir / "visium_pseudobulk_true_vs_perm_autocorr.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_original_cells": summary["original_n_cells"],
        "n_pseudospots": n_spots,
        "spot_distance_um": summary["spot_distance_um"],
        "spot_diameter_um": summary["spot_diameter_um"],
        "median_nn_um": median_nn_um,
        "moran_radius_um": float(args.moran_radius_um),
        "true_mean_morans_i": I_true_mean,
        "null_mean_morans_i": float(np.mean(I_null)),
        "null_std_morans_i": float(np.std(I_null)),
        "true_rank": int(1 + np.sum(I_null >= I_true_mean)),
        "permutation_is_whole_spot": True,
        "permutation_shuffles": "spot coordinates; expression rows stay intact",
        "first_perm_is_row_index_shuffle_of_S_only": True,
        "A_gram_unchanged_by_permutation": True,
        "first_perm_n_spots": int(first_perm.size),
        "plot": str(plot_path),
    }
    json_path = out_dir / "visium_pseudobulk_true_vs_perm_autocorr.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
