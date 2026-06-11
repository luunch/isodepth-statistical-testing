"""DLPFC within-layer study (config-driven).

Default: run existence tests on within-layer crops only (depth axis removed by
construction). Set ``run_full_sections: true`` to also run full-section positives.

All parameters come from a spec JSON (default configs/experiments/dlpfc_study.json).
Set ``output.save_per_unit_outputs: true`` to write per-unit plot folders; default is
false (summary CSV/PNG only) for faster batch runs.

Usage:  python scripts/dlpfc_study.py [--spec configs/experiments/dlpfc_study.json]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import anndata as ad
import scanpy as sc
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.liver_lobule_sweep import load_spec
from data.transforms import apply_expression_transforms
from data.schemas import DatasetBundle, TestConfig, DataConfig, OutputConfig, RunConfig
from methods.permutation import run_permutation_method
from experiments.configuration import save_standardized_outputs

DEFAULT_SPEC = REPO / "configs/experiments/dlpfc_study.json"
LAYER_ORDER = {"Layer1": 1, "Layer2": 2, "Layer3": 3, "Layer4": 4,
               "Layer5": 5, "Layer6": 6, "WM": 7}


def _select_hvg_mask(adata: ad.AnnData, n_hvg: int) -> np.ndarray:
    """HVG mask on sparse counts — avoids densifying all 33k genes."""
    tmp = ad.AnnData(adata.X)
    tmp.var_names = adata.var_names
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(
        tmp, flavor="seurat", n_top_genes=min(n_hvg, adata.n_vars - 1),
    )
    return tmp.var["highly_variable"].to_numpy()


def _load_section(repo: Path, h5ad_dir: str, section: str, n_hvg: int, layer_obs: str):
    """Load one section once; return HVG-only dense counts + metadata."""
    A = ad.read_h5ad(repo / h5ad_dir / f"{section}.h5ad")
    hvg = _select_hvg_mask(A, int(n_hvg))
    X_hvg = A[:, hvg].X
    C = (X_hvg.toarray() if hasattr(X_hvg, "toarray") else np.asarray(X_hvg)).astype(np.float32)
    names = np.asarray([str(g) for g in A.var_names[hvg]], dtype=object)
    S = np.asarray(A.obsm["spatial"], dtype=np.float64)
    region = A.obs[layer_obs].astype(str).to_numpy()
    depth = np.array([LAYER_ORDER.get(r, np.nan) for r in region])
    return C, names, S, region, depth


def _test_config_from_spec(tst: dict) -> TestConfig:
    """Build TestConfig from study spec (optional keys fall back to TestConfig defaults)."""
    cfg_kwargs = dict(
        method=tst["method"], metric=tst["metric"], n_perms=int(tst["n_perms"]),
        epochs=int(tst["epochs"]), n_reruns=int(tst["n_reruns"]), lr=float(tst["lr"]),
        patience=int(tst.get("patience", 0)), seed=int(tst["seed"]), device=tst["device"],
        decoder=tst["decoder"], verbose=False,
    )
    if tst.get("sgd_batch_size") is not None:
        cfg_kwargs["sgd_batch_size"] = int(tst["sgd_batch_size"])
    return TestConfig(**cfg_kwargs)


def _release_gpu_memory() -> None:
    """Drop cached CUDA allocations between sequential study units."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_unit(counts, S, gene_names, pp, tst, out, run_name, gt=None, *, save_outputs=False):
    """Run the existence test on a spot subset; return (p, z, |rho_gt|, n)."""
    patience = int(tst.get("patience", 0))
    stop_note = "no early stopping" if patience <= 0 else f"patience={patience}"
    print(f"  [run] {run_name}: n={len(S)} — training true + {tst['n_perms']} perms "
          f"({tst['epochs']} epochs, {stop_note}) …", flush=True)
    t0 = time.perf_counter()
    Aexpr, tmeta = apply_expression_transforms(
        counts, min_cells_per_gene=int(pp["min_cells_per_gene"]),
        normalize_total=bool(pp["normalize_total"]), log1p=bool(pp["log1p"]),
        standardize_expression=bool(pp["standardize_expression"]), return_metadata=True)
    keep = np.asarray(tmeta["gene_keep_mask"], dtype=bool)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-8)
    bundle = DatasetBundle(S=Sz.astype(np.float32), A=Aexpr.astype(np.float32),
                           meta={"var_names": [str(g) for g in np.asarray(gene_names)[keep]]}).validate()
    cfg = _test_config_from_spec(tst)
    res = run_permutation_method(bundle, cfg)
    if save_outputs:
        try:
            save_standardized_outputs(bundle, res, RunConfig(
                data=DataConfig(source="h5ad", h5ad="dlpfc"), test=cfg,
                output=OutputConfig(out_dir=str(out), run_name=run_name)))
        except Exception as e:
            print(f"  [warn] plot save failed for {run_name}: {e}")
    sp = np.asarray(res.stat_perm); z = (sp.mean() - res.stat_true) / (sp.std() + 1e-12)
    iso = np.asarray(res.artifacts["true_isodepth"]).ravel()
    rho = abs(spearmanr(iso, gt).statistic) if gt is not None else np.nan
    p_value = float(res.p_value)
    del res, bundle, cfg
    _release_gpu_memory()
    print(f"  [done] {run_name}: {time.perf_counter() - t0:.1f}s", flush=True)
    return p_value, float(z), float(rho), int(len(iso)), iso


def main(spec_path=DEFAULT_SPEC):
    spec = load_spec(spec_path)
    d = spec["data"]; neg = spec["negative"]; pp = spec["preprocessing"]; tst = spec["test"]
    out_cfg = spec.get("output", {})
    save_unit_outputs = bool(out_cfg.get("save_per_unit_outputs", False))
    out = REPO / out_cfg["out_dir"]; out.mkdir(parents=True, exist_ok=True)
    layer_obs = d["layer_obs"]

    run_full_sections = bool(spec.get("run_full_sections", False))
    pos_rows = []
    neg_rows = []
    for s in d["sections"]:
        print(f"[section] loading {s} …", flush=True)
        t_sec = time.perf_counter()
        C, names, S, region, depth = _load_section(REPO, d["h5ad_dir"], s, int(d["n_hvg"]), layer_obs)
        print(f"[section] {s} ready in {time.perf_counter() - t_sec:.1f}s "
              f"({C.shape[0]} spots × {C.shape[1]} HVGs)", flush=True)

        if run_full_sections:
            ann = ~np.isnan(depth)
            p, z, rho, n, _ = _run_unit(
                C[ann], S[ann], names, pp, tst, out, f"section_{s}_full", gt=depth[ann],
                save_outputs=save_unit_outputs,
            )
            pos_rows.append((s, n, p, z, rho))
            print(f"[POS] {s} full: n={n} p={p:.3f} z={z:.1f} |rho(iso,depth)|={rho:.3f}")

        for layer in neg["layers"]:
            sel = region == layer
            if sel.sum() < int(neg["within_layer_min_spots"]):
                continue
            p, z, _, n, _ = _run_unit(
                C[sel], S[sel], names, pp, tst, out, f"section_{s}_{layer}", gt=None,
                save_outputs=save_unit_outputs,
            )
            neg_rows.append((s, layer, n, p, z))
            print(f"[LAYER] {s} {layer}: n={n} p={p:.3f} z={z:.1f}")

    if pos_rows:
        np.savetxt(out / "positive_sections.csv", np.array([(float(s), n, p, z, rho) for (s, n, p, z, rho) in pos_rows]),
                   delimiter=",", fmt=["%d", "%d", "%.4f", "%.4f", "%.4f"],
                   header="section,n_spots,p_value,effect_z,abs_spearman_iso_vs_depth", comments="")
    neg_arr = np.array([(float(s), LAYER_ORDER.get(L, 0), n, p, z)
                        for (s, L, n, p, z) in neg_rows])
    np.savetxt(out / "within_layer.csv", neg_arr, delimiter=",",
               fmt=["%d", "%d", "%d", "%.4f", "%.4f"],
               header="section,layer_code,n_spots,p_value,effect_z", comments="")
    _save_summary_csv(out, pos_rows, neg_rows)

    signeg = neg_arr[:, 3] < 0.05 if len(neg_arr) else np.array([])
    if pos_rows:
        print("\n=== POSITIVE (full sections) ===")
        for s, n, p, z, rho in pos_rows:
            print(f"  {s}: n={n} p={p:.4f} z={z:.1f} |rho(iso,depth)|={rho:.3f}")
    print("\n=== WITHIN-LAYER ===")
    print(f"  crops tested: {len(neg_arr)} | significant (p<0.05): {int(signeg.sum())} "
          f"({(signeg.mean() if len(neg_arr) else 0):.0%})  [false-positive rate; want ~5%]")
    _make_pvalue_summary_plots(out, pos_rows, neg_rows)
    print(f"saved -> {out}/")


def _save_summary_csv(out: Path, pos_rows, neg_rows) -> None:
    import csv

    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "kind", "section", "layer", "n_spots", "p_value", "effect_z", "abs_spearman_iso_vs_depth"])
        for s, n, p, z, rho in pos_rows:
            w.writerow([f"section_{s}_full", "positive_full_section", s, "", n, f"{p:.4f}", f"{z:.4f}", f"{rho:.4f}"])
        for s, layer, n, p, z in neg_rows:
            w.writerow([f"section_{s}_{layer}", "negative_within_layer", s, layer, n, f"{p:.4f}", f"{z:.4f}", ""])


def _annotate_bars(ax, bars, values, fmt="{:.3f}") -> None:
    ymax = max(float(v) for v in values) if len(values) else 1.0
    pad = 0.03 * ymax if ymax > 0 else 0.03
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _make_pvalue_summary_plots(out, pos_rows, neg_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neg_labels = [f"{s}\n{layer}" for s, layer, *_ in neg_rows]
    neg_p = [p for _, _, _, p, _ in neg_rows]
    neg_sig = [p < 0.05 for p in neg_p]
    n_neg = len(neg_p)
    n_false_pos = int(sum(neg_sig))
    fpr = n_false_pos / n_neg if n_neg else 0.0

    ncols = 3 if pos_rows else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]
    ax_idx = 0

    if pos_rows:
        pos_labels = [str(s) for s, *_ in pos_rows]
        pos_p = [p for _, _, p, _, _ in pos_rows]
        bars = axes[ax_idx].bar(pos_labels, pos_p, color="#2ca02c", edgecolor="k", linewidth=0.6)
        axes[ax_idx].axhline(0.05, color="r", ls="--", lw=1.2, label="p = 0.05")
        axes[ax_idx].set_ylim(0, max(max(pos_p, default=0.05) * 1.25, 0.12))
        axes[ax_idx].set_title("Full-section existence tests")
        axes[ax_idx].set_xlabel("Visium section")
        axes[ax_idx].set_ylabel("existence-test p-value")
        axes[ax_idx].legend(loc="upper right")
        _annotate_bars(axes[ax_idx], bars, pos_p)
        ax_idx += 1

    neg_colors = ["#d62728" if sig else "#4C72B0" for sig in neg_sig]
    bars = axes[ax_idx].bar(neg_labels, neg_p, color=neg_colors, edgecolor="k", linewidth=0.6)
    axes[ax_idx].axhline(0.05, color="r", ls="--", lw=1.2, label="p = 0.05")
    axes[ax_idx].set_ylim(0, max(max(neg_p, default=0.05) * 1.25, 0.12))
    axes[ax_idx].set_title(
        f"Within-layer existence tests\n"
        f"{n_false_pos}/{n_neg} significant (FPR = {fpr:.0%}; target ~5%)"
    )
    axes[ax_idx].set_xlabel("section / layer")
    axes[ax_idx].set_ylabel("existence-test p-value")
    axes[ax_idx].tick_params(axis="x", labelsize=8)
    axes[ax_idx].legend(loc="upper right")
    _annotate_bars(axes[ax_idx], bars, neg_p)
    ax_idx += 1

    if n_neg:
        axes[ax_idx].hist(neg_p, bins=np.linspace(0, 1, 11), color="#4C72B0", edgecolor="k")
        axes[ax_idx].axvline(0.05, color="r", ls="--", lw=1.2, label="p = 0.05")
        axes[ax_idx].set_title("Within-layer p-value calibration")
        axes[ax_idx].set_xlabel("existence-test p-value")
        axes[ax_idx].set_ylabel("# crops")
        axes[ax_idx].legend(loc="upper right")
    else:
        axes[ax_idx].text(0.5, 0.5, "No within-layer crops", ha="center", va="center", transform=axes[ax_idx].transAxes)
        axes[ax_idx].set_axis_off()

    title = "DLPFC within-layer study"
    if pos_rows:
        title = "DLPFC study: full sections + within-layer"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(out / "dlpfc_pvalue_summary.png", dpi=140)
    plt.close(fig)


def _load_rows_from_csv(out: Path):
    layer_names = {v: k for k, v in LAYER_ORDER.items()}
    pos_path = out / "positive_sections.csv"
    pos_rows = []
    if pos_path.exists():
        pos = np.genfromtxt(pos_path, delimiter=",", names=True, dtype=None, encoding=None)
        pos_rows = [(int(r["section"]), int(r["n_spots"]), float(r["p_value"]), float(r["effect_z"]),
                     float(r["abs_spearman_iso_vs_depth"])) for r in np.atleast_1d(pos)]
    neg_path = out / "within_layer.csv"
    if not neg_path.exists():
        neg_path = out / "negative_within_layer.csv"
    neg = np.genfromtxt(neg_path, delimiter=",", names=True, dtype=None, encoding=None)
    neg_rows = [(int(r["section"]), layer_names.get(int(r["layer_code"]), str(r["layer_code"])),
                 int(r["n_spots"]), float(r["p_value"]), float(r["effect_z"]))
                for r in np.atleast_1d(neg)]
    return pos_rows, neg_rows


def plots_only(spec_path=DEFAULT_SPEC) -> None:
    spec = load_spec(spec_path)
    out = REPO / spec["output"]["out_dir"]
    pos_rows, neg_rows = _load_rows_from_csv(out)
    _save_summary_csv(out, pos_rows, neg_rows)
    _make_pvalue_summary_plots(out, pos_rows, neg_rows)
    print(f"regenerated summary plots -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DLPFC matched positive/negative study.")
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--plots-only", action="store_true", help="Regenerate summary.csv and p-value plots from existing CSVs.")
    args = ap.parse_args()
    if args.plots_only:
        plots_only(args.spec)
    else:
        main(args.spec)
