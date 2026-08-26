"""Per-lobule existence-test sweep on ST-mLiver (ground-truth liver zonation).

Segments central veins from the histological mask, assigns spots to lobules
(nearest central vein), and runs the parallel-permutation existence test on each
lobule. Reports, per lobule: p-value, effect size, and |Spearman| between the
recovered isodepth and the ground-truth distance-to-central-vein.

Usage:
    python scripts/liver_lobule_sweep.py [--spec configs/experiments/liver_lobule_study.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import anndata as ad
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from experiments.core.paths import repo_root
REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from data.transforms import apply_expression_transforms
from data.schemas import DatasetBundle, TestConfig, DataConfig, OutputConfig, RunConfig
from methods.permutation import run_permutation_method
from experiments.configuration import save_standardized_outputs

DEFAULT_SPEC = REPO / "configs/experiments/liver_lobule_study.json"


from experiments.core.study_spec import load_spec


def segment_central_veins(mask_path: Path, min_size: int = 20) -> np.ndarray:
    m = np.array(Image.open(mask_path))[:, :, :3]
    red = (m[:, :, 0] > 180) & (m[:, :, 1] < 80) & (m[:, :, 2] < 80)
    lab, n = ndimage.label(red)
    cents = ndimage.center_of_mass(red, lab, range(1, n + 1))
    sizes = ndimage.sum(red, lab, range(1, n + 1))
    cents = np.array([(c[1], c[0]) for c in cents])  # (x, y)
    return cents[sizes >= min_size]


def main(spec_path: str | Path = DEFAULT_SPEC) -> None:
    spec = load_spec(spec_path)
    h5ad = REPO / spec["data"]["h5ad"]
    mask = REPO / spec["data"]["mask"]
    gt_key = spec["data"].get("ground_truth_obs", "dist_central")
    seg = spec["segmentation"]; pp = spec["preprocessing"]; tst = spec["test"]
    out = REPO / spec["output"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    A = ad.read_h5ad(h5ad)
    counts = A.layers["counts"]
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    counts = counts.astype(np.float32)
    gene_names = np.asarray([str(g) for g in A.var_names], dtype=object)
    S = np.asarray(A.obsm["spatial"], dtype=np.float64)
    ground_truth = A.obs[gt_key].to_numpy()

    cvs = segment_central_veins(mask, min_size=int(seg["cv_min_size"]))
    assign = cKDTree(cvs).query(S)[1]
    dist_to_cv = np.linalg.norm(S - cvs[assign], axis=1)

    rows = []
    for lob in range(len(cvs)):
        sel = (assign == lob) & (dist_to_cv < float(seg["radius_px"]))
        if sel.sum() < int(seg["min_spots"]):
            continue
        Aexpr, tmeta = apply_expression_transforms(
            counts[sel], min_cells_per_gene=int(pp["min_cells_per_gene"]),
            normalize_total=bool(pp["normalize_total"]), log1p=bool(pp["log1p"]),
            standardize_expression=bool(pp["standardize_expression"]), return_metadata=True,
        )
        keep = np.asarray(tmeta["gene_keep_mask"], dtype=bool)
        names = [str(g) for g in gene_names[keep]]
        Ssub = S[sel]
        Sz = (Ssub - Ssub.mean(0)) / (Ssub.std(0) + 1e-8)
        bundle = DatasetBundle(S=Sz.astype(np.float32), A=Aexpr.astype(np.float32),
                               meta={"var_names": names}).validate()
        cfg = TestConfig(method=tst["method"], metric=tst["metric"],
                         n_perms=int(tst["n_perms"]), epochs=int(tst["epochs"]),
                         n_reruns=int(tst["n_reruns"]), lr=float(tst["lr"]),
                         patience=int(tst["patience"]), seed=int(tst["seed"]),
                         device=tst["device"], decoder=tst["decoder"], verbose=False)
        res = run_permutation_method(bundle, cfg)
        # full standard output folder per lobule (dataset/isodepth/metric/gene plots/result.json)
        run_name = f"lobule_{lob:02d}"
        run_config = RunConfig(
            data=DataConfig(source="h5ad", h5ad=str(h5ad)),
            test=cfg,
            output=OutputConfig(out_dir=str(out), run_name=run_name),
        )
        try:
            save_standardized_outputs(bundle, res, run_config)
        except Exception as e:  # keep the sweep going even if one plot fails
            print(f"  [warn] save_standardized_outputs failed for {run_name}: {e}")
        sp = np.asarray(res.stat_perm); z = (sp.mean() - res.stat_true) / (sp.std() + 1e-12)
        iso = np.asarray(res.artifacts["true_isodepth"]).ravel()
        rho = abs(spearmanr(iso, ground_truth[sel]).statistic)
        rows.append((lob, int(sel.sum()), float(res.p_value), float(z), float(rho)))
        print(f"lobule {lob:2d}: n={sel.sum():3d}  p={res.p_value:.3f}  z={z:6.1f}  "
              f"|rho(isodepth, {gt_key})|={rho:.3f}")

    arr = np.array([r for r in rows], dtype=float)
    np.savetxt(out / "lobule_results.csv", arr, delimiter=",",
               fmt=["%d", "%d", "%.4f", "%.4f", "%.4f"],
               header="lobule,n_spots,p_value,effect_z,abs_spearman_isodepth_vs_distCV", comments="")
    OUT = out  # for the print at the end
    make_summary_plots(arr, out)
    sig = arr[:, 2] < 0.05
    print("\n=== SUMMARY ===")
    print(f"lobules tested: {len(arr)} | significant (p<0.05): {int(sig.sum())} ({sig.mean():.0%})")
    print(f"median |rho(isodepth, dist_central)|: all={np.median(arr[:,4]):.3f} | "
          f"significant-only={np.median(arr[sig,4]) if sig.any() else float('nan'):.3f}")
    print(f"saved -> {OUT / 'lobule_results.csv'}")


def make_summary_plots(arr: np.ndarray, out: Path) -> None:
    """Cross-lobule summary figures. arr columns: lobule,n_spots,p,z,|rho|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lob, n, p, z, rho = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    chance = 1.0 / np.sqrt(np.median(n) - 1)  # ~1σ chance |Spearman| at median spot count

    # (1) the 2D scatter: ground-truth recovery vs p-value, sized/coloured by #spots
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(rho, p, s=n * 2.5, c=n, cmap="viridis", edgecolor="k", alpha=0.85)
    ax.axhline(0.05, ls="--", c="r", label="p = 0.05")
    ax.axvline(2 * chance, ls=":", c="grey", label="~2σ chance |ρ|")
    ax.set_xlabel("|Spearman(recovered isodepth, distance-to-central-vein)|")
    ax.set_ylabel("existence-test p-value")
    ax.set_title("Per-lobule: ground-truth recovery vs. test significance")
    plt.colorbar(sc, label="# spots in lobule"); ax.legend(loc="upper right")
    plt.tight_layout(); fig.savefig(out / "lobule_summary.png", dpi=120); plt.close(fig)

    # (2) everything-together 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    a = axes[0, 0]
    a.hist(p, bins=np.linspace(0, 1, 11), color="0.6", edgecolor="k")
    a.axvline(0.05, ls="--", c="r"); a.set_xlabel("p-value"); a.set_ylabel("# lobules")
    a.set_title(f"p-value distribution ({int((p<0.05).sum())}/{len(p)} significant)")

    a = axes[0, 1]
    a.hist(rho, bins=np.linspace(0, 1, 11), color="#4C72B0", edgecolor="k")
    a.axvline(2 * chance, ls=":", c="grey", label="~2σ chance")
    a.set_xlabel("|Spearman(isodepth, dist_central)|"); a.set_ylabel("# lobules")
    a.set_title("Ground-truth recovery (median %.2f)" % np.median(rho)); a.legend()

    a = axes[1, 0]
    a.scatter(n, rho, c=z, cmap="coolwarm", edgecolor="k", s=60)
    a.axhline(2 * chance, ls=":", c="grey")
    a.set_xlabel("# spots in lobule"); a.set_ylabel("|ρ(isodepth, dist_central)|")
    a.set_title("Recovery vs lobule size")

    a = axes[1, 1]
    a.scatter(z, rho, c=n, cmap="viridis", edgecolor="k", s=60)
    a.axhline(2 * chance, ls=":", c="grey"); a.axvline(0, ls="-", c="0.7")
    a.set_xlabel("effect size  z = (null_mean - obs)/null_std"); a.set_ylabel("|ρ(isodepth, dist_central)|")
    a.set_title("Effect size vs ground-truth recovery")
    plt.suptitle("ST-mLiver per-lobule sweep — everything together", fontsize=14)
    plt.tight_layout(); fig.savefig(out / "lobule_overview.png", dpi=120); plt.close(fig)
    print(f"saved -> {out / 'lobule_summary.png'}, {out / 'lobule_overview.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-lobule existence-test sweep on ST-mLiver.")
    ap.add_argument("--spec", default=str(DEFAULT_SPEC), help="Path to the experiment spec JSON")
    main(ap.parse_args().spec)
