"""STARmap mouse visual cortex within-stratum study (config-driven).

The cortical-depth axis is (geometrically) almost a function of x, so feeding the
layer ordinal as a covariate is degenerate (a 6-step staircase along x). The
cleaner decomposition: handle the trivial BETWEEN-layer variation by segmenting
on the layer label, and ask the non-trivial WITHIN-layer question directly.

Positive: full cortex -> existence test + recovery |Spearman(isodepth, cortical_depth)|.
Negative: each single layer crop (depth axis removed by construction) -> existence
test; reports per-layer p / effect size. A clean layer should be ~null; firing
indicates residual tangential/sublaminar structure (or over-sensitivity).

This mirrors scripts/dlpfc_study.py (reuses its `_run_unit`) but on a single
small section. All parameters come from configs/experiments/starmap_within_layer.json.

Usage:  python scripts/starmap_within_layer.py [--spec ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import anndata as ad

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.liver_lobule_sweep import load_spec
from scripts.dlpfc_study import _run_unit, _select_hvg

DEFAULT_SPEC = REPO / "configs/experiments/starmap_within_layer.json"


def main(spec_path=DEFAULT_SPEC):
    spec = load_spec(spec_path)
    d = spec["data"]; neg = spec["negative"]; pp = spec["preprocessing"]; tst = spec["test"]
    out = REPO / spec["output"]["out_dir"]; out.mkdir(parents=True, exist_ok=True)

    A = ad.read_h5ad(REPO / d["h5ad"])
    C = (A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)).astype(np.float32)
    names = np.asarray([str(g) for g in A.var_names], dtype=object)
    S = np.asarray(A.obsm["spatial"], dtype=np.float64)
    layer = A.obs[d["layer_obs"]].astype(str).to_numpy()
    depth = np.asarray(A.obs[d["depth_obs"]].values, dtype=float)

    n_hvg = int(d.get("n_hvg", 0))
    if 0 < n_hvg < C.shape[1]:
        hvg = _select_hvg(C, n_hvg)
        C = C[:, hvg]; names = names[hvg]

    rows = []  # (label, n, p, z, rho)

    # POSITIVE: full cortex, recover the laminar depth axis
    p, z, rho, n, _ = _run_unit(C, S, names, pp, tst, out, "cortex_full", gt=depth)
    rows.append(("cortex_full", n, p, z, rho))
    print(f"[POS] cortex_full: n={n} p={p:.3f} z={z:.1f} |rho(iso,depth)|={rho:.3f}")

    # NEGATIVE: within-layer crops (depth axis removed by construction)
    for L in neg["layers"]:
        sel = layer == L
        if sel.sum() < int(neg["within_layer_min_cells"]):
            print(f"[skip] {L}: n={int(sel.sum())} < {neg['within_layer_min_cells']}")
            continue
        safe = L.replace("/", "")
        p, z, _, n, _ = _run_unit(C[sel], S[sel], names, pp, tst, out, f"layer_{safe}", gt=None)
        rows.append((f"layer_{safe}", n, p, z, np.nan))
        print(f"[NEG] {L}: n={n} p={p:.3f} z={z:.1f}")

    # ---- save summary csv + plot ----
    import csv
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "kind", "n", "p_value", "effect_z", "abs_spearman_iso_vs_depth"])
        for (lab, n, p, z, rho) in rows:
            kind = "positive" if lab == "cortex_full" else "negative_within_layer"
            w.writerow([lab, kind, n, f"{p:.4f}", f"{z:.4f}", ("" if rho != rho else f"{rho:.4f}")])

    neg_rows = [r for r in rows if r[0] != "cortex_full"]
    nsig = sum(1 for r in neg_rows if r[2] < 0.05)
    print(f"\n=== STARmap within-layer negatives: {nsig}/{len(neg_rows)} significant (p<0.05) ===")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    labels = [r[0].replace("layer_", "") for r in neg_rows]
    zs = [r[3] for r in neg_rows]
    colors = ["#d62728" if r[2] < 0.05 else "#4C72B0" for r in neg_rows]
    ax[0].bar(labels, zs, color=colors)
    ax[0].set_title("Within-layer effect size (red = p<0.05)\nlower power than DLPFC (small n)")
    ax[0].set_ylabel("effect z = (null_mean - stat_true)/null_std"); ax[0].axhline(0, color="0.6")
    ax[1].bar([r[0] for r in rows], [r[2] for r in rows],
              color=["#2ca02c"] + colors)
    ax[1].axhline(0.05, color="r", ls="--", label="p=0.05")
    ax[1].set_title("p-values: cortex_full (green=positive) vs within-layer")
    ax[1].set_ylabel("existence-test p-value"); ax[1].tick_params(axis="x", rotation=45); ax[1].legend()
    plt.suptitle("STARmap V1: full cortex (positive) vs within-layer (negatives)", fontsize=13)
    plt.tight_layout(); fig.savefig(out / "starmap_within_layer_summary.png", dpi=120); plt.close(fig)
    print(f"saved -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="STARmap within-layer study.")
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    main(ap.parse_args().spec)
