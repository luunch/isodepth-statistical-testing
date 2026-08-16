"""Aggregate (pooled) test for the liver per-lobule sweep.

Question: across lobules, does the FITTED isodepth recover the ground-truth
distance-to-central-vein MORE than a *random smooth axis* of comparable
smoothness would? This converts many individually-underpowered per-lobule
tests into one well-powered statement, and calibrates the |rho| recovery
against mutual-smoothness chance.

Null: for each lobule, draw smooth Gaussian-process fields over its spot
coordinates (RBF kernel, length scale `l`) and correlate each with dist_central
-> a per-lobule null distribution of "what a random smooth axis achieves".
The observed isodepth |rho| comes from the sweep (lobule_results.csv).

Aggregate statistic: mean |rho| across lobules. Aggregate null: draw one
random-smooth-axis |rho| per lobule, average across lobules, repeat.

Usage:  python scripts/liver_lobule_aggregate.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import anndata as ad
from scipy.spatial.distance import pdist, squareform

from experiments.core.paths import repo_root
REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))
from experiments.core.study_spec import load_spec
from scripts.studies.tissue_sweeps.liver_lobule_sweep import segment_central_veins, DEFAULT_SPEC
from scipy.spatial import cKDTree


def rank0(M: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(M, axis=0), axis=0).astype(float)


def smooth_null_abs_rho(Sz: np.ndarray, dc: np.ndarray, l: float, B: int, seed: int) -> np.ndarray:
    """|Spearman| of B smooth GP fields (RBF length scale l) vs dc."""
    n = len(dc)
    D2 = squareform(pdist(Sz)) ** 2
    K = np.exp(-D2 / (2.0 * l * l)) + 1e-6 * np.eye(n)
    L = np.linalg.cholesky(K)
    rng = np.random.default_rng(seed)
    F = L @ rng.standard_normal((n, B))
    rf = rank0(F); rdc = rank0(dc.reshape(-1, 1))[:, 0]
    rf = (rf - rf.mean(0)) / (rf.std(0) + 1e-12)
    rdc = (rdc - rdc.mean()) / (rdc.std() + 1e-12)
    return np.abs((rf * rdc[:, None]).mean(0))


def main(spec_path: str | Path = DEFAULT_SPEC) -> None:
    spec = load_spec(spec_path)
    h5ad = REPO / spec["data"]["h5ad"]
    mask = REPO / spec["data"]["mask"]
    gt_key = spec["data"].get("ground_truth_obs", "dist_central")
    radius_px = float(spec["segmentation"]["radius_px"])
    cv_min_size = int(spec["segmentation"]["cv_min_size"])
    out = REPO / spec["output"]["out_dir"]
    agg = spec["aggregate"]
    B = int(agg["n_null_draws"]); LENGTH_SCALES = list(agg["length_scales"]); MAIN_L = float(agg["main_length_scale"])
    OUTDIR = out / "aggregate_test"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    A = ad.read_h5ad(h5ad)
    S = np.asarray(A.obsm["spatial"], dtype=np.float64)
    dist_central = A.obs[gt_key].to_numpy()
    cvs = segment_central_veins(mask, min_size=cv_min_size)
    assign = cKDTree(cvs).query(S)[1]
    dist_to_cv = np.linalg.norm(S - cvs[assign], axis=1)

    res = np.genfromtxt(out / "lobule_results.csv", delimiter=",", skip_header=1)
    lob_ids = res[:, 0].astype(int); rho_obs = res[:, 4]

    summary = {}
    for l in LENGTH_SCALES:
        null_mat = []   # (n_lobules, B)
        for lob in lob_ids:
            sel = (assign == lob) & (dist_to_cv < radius_px)
            Ssub = S[sel]; Sz = (Ssub - Ssub.mean(0)) / (Ssub.std(0) + 1e-8)
            null_mat.append(smooth_null_abs_rho(Sz, dist_central[sel], l, B, seed=int(lob)))
        null_mat = np.array(null_mat)                      # (n_lob, B)
        obs_agg = rho_obs.mean()
        null_agg = null_mat.mean(0)                          # (B,)
        agg_p = float((null_agg >= obs_agg).mean())
        per_lobule_p = (null_mat >= rho_obs[:, None]).mean(1)
        summary[l] = (obs_agg, null_agg, agg_p, null_mat.mean(1), per_lobule_p)
        print(f"l={l}: observed mean|rho|={obs_agg:.3f}  null mean|rho|={null_agg.mean():.3f}  "
              f"aggregate p={agg_p:.4f}  lobules beyond null(p<0.05): {(per_lobule_p<0.05).sum()}/{len(lob_ids)}")

    # save per-lobule table for the main length scale
    obs_agg, null_agg, agg_p, null_mean_per_lob, per_lobule_p = summary[MAIN_L]
    table = np.column_stack([lob_ids, rho_obs, null_mean_per_lob, per_lobule_p])
    np.savetxt(OUTDIR / "aggregate_perlobule.csv", table, delimiter=",",
               fmt=["%d", "%.4f", "%.4f", "%.4f"],
               header="lobule,obs_abs_rho,smooth_null_mean_abs_rho,p_vs_smooth_null", comments="")

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # (1) aggregate null vs observed
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(null_agg, bins=40, color="0.7", edgecolor="k", label="null: random smooth axes")
    ax.axvline(obs_agg, color="r", lw=2, ls="--", label=f"observed isodepth mean|ρ|={obs_agg:.3f}")
    ax.set_xlabel("mean |Spearman(axis, dist_central)| across lobules")
    ax.set_ylabel("count"); ax.legend()
    ax.set_title(f"Aggregate liver-zonation recovery vs random smooth axes (l={MAIN_L})\n"
                 f"aggregate p = {agg_p:.4f}")
    plt.tight_layout(); fig.savefig(OUTDIR / "aggregate_null.png", dpi=120); plt.close(fig)

    # (2) per-lobule observed vs smooth-null mean
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(null_mean_per_lob, rho_obs, c=per_lobule_p, cmap="viridis_r",
                    s=70, edgecolor="k", vmin=0, vmax=0.5)
    lim = [0, max(rho_obs.max(), null_mean_per_lob.max()) * 1.05]
    ax.plot(lim, lim, "k--", alpha=0.6, label="y = x (no recovery beyond smooth chance)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("random-smooth-axis mean |ρ|  (chance)")
    ax.set_ylabel("fitted isodepth |ρ|  (observed)")
    ax.set_title("Per-lobule: isodepth recovery vs. smooth-axis chance")
    plt.colorbar(sc, label="per-lobule p vs smooth null"); ax.legend()
    plt.tight_layout(); fig.savefig(OUTDIR / "perlobule_vs_null.png", dpi=120); plt.close(fig)

    # (3) sensitivity across length scales
    with open(OUTDIR / "aggregate_summary.txt", "w") as fh:
        fh.write("Aggregate liver-zonation recovery vs random smooth axes\n")
        for l in LENGTH_SCALES:
            oa, na, ap, _, plp = summary[l]
            fh.write(f"length_scale={l}: observed_mean_abs_rho={oa:.4f} "
                     f"null_mean={na.mean():.4f} aggregate_p={ap:.4f} "
                     f"lobules_beyond_null_p<0.05={int((plp<0.05).sum())}/{len(lob_ids)}\n")
    print(f"\nsaved -> {OUTDIR}/ (aggregate_null.png, perlobule_vs_null.png, aggregate_perlobule.csv, aggregate_summary.txt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aggregate smooth-null test for the liver lobule sweep.")
    ap.add_argument("--spec", default=str(DEFAULT_SPEC), help="Path to the experiment spec JSON")
    main(ap.parse_args().spec)
