"""Scan completed CalicoST isodepth runs for a library-size (depth) confound.

For each completed CalicoST permutation-test run listed in ``RUNS`` below, this
script:

1. Reloads the exact training-time dataset via ``load_dataset`` (deterministic:
   same obs filters, spatial denoise, crop, z-scoring as the original run), which
   gives per-clone **raw** counts (``cell_type="separate"`` defers HVG/normalize/
   log1p/z-score, so ``dataset.A`` is still raw UMI counts at this point).
2. Computes per-spot technical covariates directly from those raw counts:
   ``total_counts`` (library size), ``n_genes_detected``, ``pct_mt``.
3. Loads the saved ``true_isodepth`` and preprocessed expression matrix ``A``
   from ``{type}/{type}_isodepths.npz`` (same row order as the raw counts subset,
   since both come from boolean-masking the same ordered array).
4. Correlates isodepth against: library size, gene count, pct_mt, distance-from-
   centroid (geometry proxy), and the loss-difference whitening covariate
   (typically ``calicost_tumor_proportion``) when present.
5. Correlates isodepth against every gene's expression (Spearman), and
   separately correlates *library size* against every gene's expression, then
   compares the two per-gene correlation vectors. If they are strongly aligned,
   the genes whose expression tracks isodepth are (mostly) the same genes whose
   expression tracks sequencing depth -- i.e. the "isodepth vs. gene expression"
   signal is largely a depth/library-size signature, not independent biology.

Outputs (see --out-dir):
    <sample>__<run>__<celltype>_summary.json  (per-run/per-celltype scalars)
    <sample>__<run>__<celltype>_diagnostics.png (4-panel scatter figure)
    summary.csv / summary.json                (one row per run x celltype)
    summary_bar.png                            (cross-run comparison bar chart)
    gene_axis_alignment.png                    (per-gene rho_iso vs rho_depth, all runs)

Usage (from repo root):
    python -m scripts.posthoc.qc_library_size_confound_scan
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from experiments.core.paths import repo_root

REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from data import load_dataset  # noqa: E402
from data.h5ad_loader import preprocess_celltype_subset  # noqa: E402
from data.schemas import run_config_from_mapping  # noqa: E402
from experiments.configuration import load_json_config  # noqa: E402

# Completed CalicoST permutation-test runs with an exact, current config file
# (i.e. the config's ``output.out_dir``/``output.run_name`` maps to an existing
# ``*_result.json``), spanning 4 distinct tissue/clone samples. See agent.md
# ("Calicost isodepth-confounder collinearity") for provenance.
RUNS: list[dict] = [
    dict(
        config="configs/calicost/HT112C1_U1_fig4_loss_diff_tumor_prop.json",
        sample="HT112C1_U1",
        variant="tumor_gt0p7",
    ),
    dict(
        config="configs/calicost/HT112C1_U1_fig4_loss_diff_tumor_prop_cropx.json",
        sample="HT112C1_U1",
        variant="tumor_gt0p7_cropx_3000hvg",
    ),
    dict(
        config="configs/calicost/HT112C1_U1_fig4_loss_diff_tumor_prop_gt0p8.json",
        sample="HT112C1_U1",
        variant="tumor_gt0p8",
    ),
    dict(
        config="configs/calicost/HT112C1_U2_loss_difference.json",
        sample="HT112C1_U2",
        variant="tumor_gt0p7",
    ),
    dict(
        config="configs/calicost/HT112C1_U2_loss_difference_cropx.json",
        sample="HT112C1_U2",
        variant="tumor_gt0p7_cropx",
    ),
    dict(
        config="configs/calicost/HT268B1_slice1_U1_loss_difference.json",
        sample="HT268B1_slice1_clone1",
        variant="tumor_gt0p7_linear",
    ),
    dict(
        config="configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2.json",
        sample="HT306P1_clone2",
        variant="tumor_gt0p5_cropy",
    ),
    dict(
        config="configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_no_mtribo_stress.json",
        sample="HT306P1_clone2",
        variant="tumor_gt0p5_cropy_no_mtribo_stress",
    ),
]


@dataclass
class RunCellTypeSummary:
    sample: str
    variant: str
    run_name: str
    cell_type: str
    n_cells: int
    n_genes_raw: int
    n_genes_hvg: int
    # Spearman rho of isodepth vs...
    rho_iso_log_total_counts: float
    rho_iso_n_genes: float
    rho_iso_pct_mt: float
    rho_iso_centroid_dist: float
    rho_iso_whitening_covariate: Optional[float]
    whitening_covariate_name: Optional[str]
    # R^2 (OLS with intercept) of isodepth on each covariate set
    r2_library_size: float  # [log_total_counts, n_genes]
    r2_pct_mt: float
    r2_geometry: float  # [centroid_dist_x, centroid_dist_y] (2D)
    r2_whitening_covariate: Optional[float]
    r2_all_technical: float  # library_size + pct_mt + geometry (+ whitening if present)
    # Gene-level: does the isodepth gene signature look like a depth signature?
    gene_axis_alignment_pearson: float  # corr(rho_iso_gene, rho_depth_gene) across genes
    top50_jaccard_overlap: float  # overlap of top-50 |rho| genes (iso-ranked vs depth-ranked)
    top_iso_genes: list  # [(gene, rho_iso, rho_depth), ...] top 15 by |rho_iso|


def _safe_name(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rho, _ = spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else float("nan")


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of an OLS fit of ``y`` on covariates ``X`` (with intercept)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    design = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _gene_level_scores(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-column Spearman rho of ``A[:, g]`` vs 1-D ``x``."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    A = np.asarray(A, dtype=np.float64)
    out = np.zeros(A.shape[1], dtype=np.float64)
    for g in range(A.shape[1]):
        rho, _ = spearmanr(x, A[:, g])
        out[g] = rho if np.isfinite(rho) else 0.0
    return out


def _jaccard_top_k(rho_a: np.ndarray, rho_b: np.ndarray, k: int) -> float:
    k = min(k, rho_a.size)
    top_a = set(np.argsort(-np.abs(rho_a))[:k].tolist())
    top_b = set(np.argsort(-np.abs(rho_b))[:k].tolist())
    union = top_a | top_b
    if not union:
        return float("nan")
    return len(top_a & top_b) / len(union)


def _plot_diagnostics(
    out_path: Path,
    *,
    title: str,
    iso: np.ndarray,
    log_total_counts: np.ndarray,
    n_genes_detected: np.ndarray,
    pct_mt: np.ndarray,
    rho_iso_gene: np.ndarray,
    rho_depth_gene: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.scatter(log_total_counts, iso, s=8, alpha=0.5, color="#1f77b4")
    rho = _spearman(iso, log_total_counts)
    ax.set_xlabel("log1p(total_counts)")
    ax.set_ylabel("true_isodepth")
    ax.set_title(f"isodepth vs. library size (rho={rho:.2f})")

    ax = axes[0, 1]
    ax.scatter(n_genes_detected, iso, s=8, alpha=0.5, color="#ff7f0e")
    rho = _spearman(iso, n_genes_detected)
    ax.set_xlabel("n_genes_detected")
    ax.set_ylabel("true_isodepth")
    ax.set_title(f"isodepth vs. gene count (rho={rho:.2f})")

    ax = axes[1, 0]
    ax.scatter(pct_mt, iso, s=8, alpha=0.5, color="#2ca02c")
    rho = _spearman(iso, pct_mt)
    ax.set_xlabel("pct_mt (%)")
    ax.set_ylabel("true_isodepth")
    ax.set_title(f"isodepth vs. pct_mt (rho={rho:.2f})")

    ax = axes[1, 1]
    ax.scatter(rho_depth_gene, rho_iso_gene, s=8, alpha=0.5, color="#9467bd")
    lim = float(np.nanmax(np.abs(np.concatenate([rho_iso_gene, rho_depth_gene])))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.8, linestyle="--")
    r, _ = pearsonr(rho_iso_gene, rho_depth_gene)
    ax.set_xlabel("per-gene rho(gene, log_total_counts)")
    ax.set_ylabel("per-gene rho(gene, isodepth)")
    ax.set_title(f"gene axis alignment (pearson={r:.2f})")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _process_run(run_spec: dict, out_dir: Path) -> list[RunCellTypeSummary]:
    config_path = REPO / run_spec["config"]
    cfg = load_json_config(str(config_path))
    run_cfg = run_config_from_mapping(cfg)
    run_name = str(cfg["output"]["run_name"])
    result_dir = REPO / cfg["output"]["out_dir"] / run_name
    result_json_path = result_dir / f"{run_name}_result.json"
    if not result_json_path.exists():
        print(f"[skip] no result JSON at {result_json_path}", flush=True)
        return []

    print(f"[load] {run_spec['sample']} / {run_spec['variant']} <- {config_path.name}", flush=True)
    dataset = load_dataset(run_cfg.data)

    cell_type_names = list(dataset.meta.get("cell_type_names") or [])
    cell_type_labels = dataset.meta.get("cell_type_labels")
    if not cell_type_names or cell_type_labels is None:
        print(f"[skip] {run_name}: not cell_type='separate' output, unsupported here", flush=True)
        return []
    cell_type_labels = np.asarray(cell_type_labels, dtype=np.int64)

    var_names_full = dataset.meta.get("var_names")
    if var_names_full is None:
        var_names_full = [f"gene_{i}" for i in range(dataset.A.shape[1])]
    var_names_full = np.asarray([str(v) for v in var_names_full], dtype=object)
    mt_mask = np.array([v.upper().startswith("MT-") for v in var_names_full], dtype=bool)

    whitening_vals_all = dataset.meta.get("covariate_whitening_values")
    whitening_key = dataset.meta.get("covariate_whitening_obs_key")
    pp = dict(dataset.meta.get("separate_preprocessing", {}))
    pp_params = {k: v for k, v in pp.items() if k != "seed"}
    pp_seed = int(pp.get("seed", 0))

    summaries: list[RunCellTypeSummary] = []
    for type_index, type_name in enumerate(cell_type_names):
        mask = cell_type_labels == type_index
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        safe_name = _safe_name(type_name)
        npz_path = result_dir / safe_name / f"{safe_name}_isodepths.npz"
        if not npz_path.exists():
            print(f"  [skip] missing NPZ for cell type {type_name}: {npz_path}", flush=True)
            continue
        npz = np.load(npz_path, allow_pickle=False)
        if "true_isodepth" not in npz:
            continue
        iso = np.asarray(npz["true_isodepth"], dtype=np.float64).reshape(-1)
        if iso.shape[0] != n_c:
            print(
                f"  [skip] {type_name}: isodepth length {iso.shape[0]} != cell count {n_c}",
                flush=True,
            )
            continue

        raw_counts = np.asarray(dataset.A[mask], dtype=np.float64)
        total_counts = raw_counts.sum(axis=1)
        log_total_counts = np.log1p(np.maximum(total_counts, 0.0))
        n_genes_detected = (raw_counts > 0).sum(axis=1).astype(np.float64)
        mt_counts = raw_counts[:, mt_mask].sum(axis=1)
        pct_mt = 100.0 * mt_counts / np.maximum(total_counts, 1.0)

        S_c = np.asarray(dataset.S[mask], dtype=np.float64)
        centroid = S_c.mean(axis=0)
        centroid_dist = np.linalg.norm(S_c - centroid, axis=1)

        whitening_vals = None
        if whitening_vals_all is not None:
            arr = np.asarray(whitening_vals_all)
            if arr.ndim >= 1 and arr.shape[0] == cell_type_labels.shape[0]:
                whitening_vals = np.asarray(arr[mask], dtype=np.float64).reshape(-1)

        # Preprocessed (HVG + normalize/log1p/zscore) expression, matching npz "A".
        A_pre, var_names_c, _ = preprocess_celltype_subset(
            np.asarray(raw_counts, dtype=np.float32),
            list(var_names_full),
            seed=pp_seed + int(type_index),
            **pp_params,
        )
        A_expr = A_pre
        if "A" in npz:
            A_npz = np.asarray(npz["A"], dtype=np.float32)
            if A_npz.shape == (n_c, len(var_names_c)):
                A_expr = A_npz

        rho_iso_gene = _gene_level_scores(A_expr, iso)
        rho_depth_gene = _gene_level_scores(A_expr, log_total_counts)
        align_r, _ = pearsonr(rho_iso_gene, rho_depth_gene)
        jacc = _jaccard_top_k(rho_iso_gene, rho_depth_gene, k=50)

        order = np.argsort(-np.abs(rho_iso_gene))[:15]
        top_iso_genes = [
            (str(var_names_c[i]), round(float(rho_iso_gene[i]), 4), round(float(rho_depth_gene[i]), 4))
            for i in order.tolist()
        ]

        r2_lib = _ols_r2(iso, np.column_stack([log_total_counts, n_genes_detected]))
        r2_mt = _ols_r2(iso, pct_mt)
        r2_geom = _ols_r2(iso, S_c)
        r2_white = _ols_r2(iso, whitening_vals) if whitening_vals is not None else None
        tech_cols = [log_total_counts, n_genes_detected, pct_mt, S_c[:, 0], S_c[:, 1]]
        if whitening_vals is not None:
            tech_cols.append(whitening_vals)
        r2_all = _ols_r2(iso, np.column_stack(tech_cols))

        summary = RunCellTypeSummary(
            sample=run_spec["sample"],
            variant=run_spec["variant"],
            run_name=run_name,
            cell_type=str(type_name),
            n_cells=n_c,
            n_genes_raw=int(raw_counts.shape[1]),
            n_genes_hvg=int(A_expr.shape[1]),
            rho_iso_log_total_counts=_spearman(iso, log_total_counts),
            rho_iso_n_genes=_spearman(iso, n_genes_detected),
            rho_iso_pct_mt=_spearman(iso, pct_mt),
            rho_iso_centroid_dist=_spearman(iso, centroid_dist),
            rho_iso_whitening_covariate=(_spearman(iso, whitening_vals) if whitening_vals is not None else None),
            whitening_covariate_name=(str(whitening_key) if whitening_vals is not None else None),
            r2_library_size=r2_lib,
            r2_pct_mt=r2_mt,
            r2_geometry=r2_geom,
            r2_whitening_covariate=r2_white,
            r2_all_technical=r2_all,
            gene_axis_alignment_pearson=float(align_r) if np.isfinite(align_r) else float("nan"),
            top50_jaccard_overlap=jacc,
            top_iso_genes=top_iso_genes,
        )
        summaries.append(summary)

        stem = f"{run_spec['sample']}__{run_name}__{safe_name}"
        with (out_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        _plot_diagnostics(
            out_dir / f"{stem}_diagnostics.png",
            title=f"{run_spec['sample']} / {run_name} / clone {type_name} (n={n_c})",
            iso=iso,
            log_total_counts=log_total_counts,
            n_genes_detected=n_genes_detected,
            pct_mt=pct_mt,
            rho_iso_gene=rho_iso_gene,
            rho_depth_gene=rho_depth_gene,
        )
        print(
            f"  [{type_name}] n={n_c} rho(iso,log_counts)={summary.rho_iso_log_total_counts:.2f} "
            f"rho(iso,n_genes)={summary.rho_iso_n_genes:.2f} R2_libsize={r2_lib:.2f} "
            f"R2_geom={r2_geom:.2f} gene_axis_align={summary.gene_axis_alignment_pearson:.2f} "
            f"top50_jaccard={jacc:.2f}",
            flush=True,
        )

    return summaries


def _write_summary_table(rows: list[RunCellTypeSummary], out_dir: Path) -> None:
    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

    csv_path = out_dir / "summary.csv"
    field_names = [f for f in RunCellTypeSummary.__dataclass_fields__ if f != "top_iso_genes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(field_names)
        for r in rows:
            d = asdict(r)
            writer.writerow([d[k] for k in field_names])
    print(f"[write] {csv_path}", flush=True)
    print(f"[write] {json_path}", flush=True)


def _plot_summary_bar(rows: list[RunCellTypeSummary], out_dir: Path) -> None:
    if not rows:
        return
    labels = [f"{r.sample}\n{r.variant}\nclone {r.cell_type}" for r in rows]
    metrics = {
        "|rho| log_counts": [abs(r.rho_iso_log_total_counts) for r in rows],
        "|rho| n_genes": [abs(r.rho_iso_n_genes) for r in rows],
        "|rho| pct_mt": [abs(r.rho_iso_pct_mt) for r in rows],
        "|rho| centroid_dist": [abs(r.rho_iso_centroid_dist) for r in rows],
    }
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(rows)), 5.5))
    for i, (name, vals) in enumerate(metrics.items()):
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|Spearman rho| with isodepth")
    ax.set_title("Isodepth vs. technical/geometric covariates, across CalicoST runs")
    ax.legend(fontsize=8)
    ax.axhline(0.3, color="grey", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bar_rho.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(rows)), 5.5))
    r2_metrics = {
        "R2 library-size": [r.r2_library_size for r in rows],
        "R2 pct_mt": [r.r2_pct_mt for r in rows],
        "R2 geometry": [r.r2_geometry for r in rows],
        "R2 all-technical": [r.r2_all_technical for r in rows],
    }
    for i, (name, vals) in enumerate(r2_metrics.items()):
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("R^2 with isodepth (OLS)")
    ax.set_title("Variance of isodepth explained by covariate groups, across CalicoST runs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bar_r2.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(rows)), 5.0))
    ax.bar(x, [r.gene_axis_alignment_pearson for r in rows], color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("pearson(rho_iso_gene, rho_depth_gene)")
    ax.set_title(
        "Gene-level axis alignment: is the isodepth gene signature\n"
        "the same as the library-size gene signature?"
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "gene_axis_alignment.png", dpi=160)
    plt.close(fig)


def main() -> None:
    out_dir = REPO / "results" / "calicost" / "isodepth_library_size_confound"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[RunCellTypeSummary] = []
    for run_spec in RUNS:
        all_rows.extend(_process_run(run_spec, out_dir))

    if not all_rows:
        print("No runs produced summaries.", flush=True)
        return

    _write_summary_table(all_rows, out_dir)
    _plot_summary_bar(all_rows, out_dir)

    n = len(all_rows)
    med_lib_rho = float(np.median([abs(r.rho_iso_log_total_counts) for r in all_rows]))
    med_ngenes_rho = float(np.median([abs(r.rho_iso_n_genes) for r in all_rows]))
    med_r2_lib = float(np.median([r.r2_library_size for r in all_rows]))
    med_r2_geom = float(np.median([r.r2_geometry for r in all_rows]))
    med_r2_white = float(
        np.median([r.r2_whitening_covariate for r in all_rows if r.r2_whitening_covariate is not None])
    )
    med_align = float(np.median([r.gene_axis_alignment_pearson for r in all_rows]))
    med_jaccard = float(np.median([r.top50_jaccard_overlap for r in all_rows]))

    print("\n=== Cross-run summary (median across {} run x cell-type rows) ===".format(n))
    print(f"median |rho(isodepth, log_total_counts)|      = {med_lib_rho:.3f}")
    print(f"median |rho(isodepth, n_genes_detected)|       = {med_ngenes_rho:.3f}")
    print(f"median R^2(isodepth ~ library-size axis)        = {med_r2_lib:.3f}")
    print(f"median R^2(isodepth ~ geometry [S_x,S_y])        = {med_r2_geom:.3f}")
    print(f"median R^2(isodepth ~ whitening covariate)       = {med_r2_white:.3f}")
    print(f"median pearson(rho_iso_gene, rho_depth_gene)     = {med_align:.3f}")
    print(f"median top-50 |rho|-gene overlap (iso vs depth)  = {med_jaccard:.3f}")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
