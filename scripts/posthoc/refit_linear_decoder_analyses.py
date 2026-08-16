"""OLS-refit a linear/quadratic decoder on frozen true_isodepth, then redo GSEA + sig genes.

Given a finished gaussian isodepth run, the jointly trained decoder is often far from the
closed-form MSE optimum for the final latent (see agent.md). This script:

1. Loads ``true_isodepth`` + expression from the result / dataset
2. Refits the decoder with ``fit_closed_form_decoder`` (exact OLS / poly OLS)
3. Writes ``*_pred_true_ols_refit.npz``
4. Recomputes F-test SVG list (``*_isodepth_sig_genes.csv``) and optionally the
   gene-expression-vs-isodepth plot using the OLS predictions
5. Re-runs pre-ranked GSEA with ``--score-method decoder`` on the OLS predictions

Usage (from repo root, isodepth_env):
    python -m scripts.posthoc.refit_linear_decoder_analyses \
        configs/jfan_merfish.json \
        results/jfan_merfish/260810_jfan_merfish_linear_decoder/260810_jfan_merfish_linear_decoder_result.json \
        --gmt data/gmt/h.all.v2026.1.Hs.symbols.gmt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from experiments.core.paths import repo_root

REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from analysis.plots import (  # noqa: E402
    _save_sig_genes_csv,
    compute_isodepth_sig_genes,
    save_gene_expression_vs_isodepth_plot,
)
from data import load_dataset  # noqa: E402
from data.schemas import run_config_from_mapping  # noqa: E402
from experiments.configuration import (  # noqa: E402
    _dataset_for_gene_expression_plots,
    _decoder_df_from_config,
    load_json_config,
)
from methods.trainers.isodepth import fit_closed_form_decoder  # noqa: E402
from scripts.posthoc.postprocess_gsea_isodepth import (  # noqa: E402
    _collapse_duplicate_genes,
    _gsea_preranked,
    _load_gmt,
    _plot_top_pathways,
    _safe_name,
    _score_genes,
    _write_gsea_csv,
    _write_prerank_csv,
)


def _archive_if_exists(path: Path, suffix: str) -> None:
    if not path.exists():
        return
    dest = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    if dest.exists():
        return
    shutil.copy2(path, dest)
    print(f"[archive] {path.name} -> {dest.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("result_json_path", type=Path)
    parser.add_argument("--gmt", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to the result JSON's parent directory",
    )
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--n-permutations", type=int, default=250)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--max-size", type=int, default=500)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-plot-pathways", type=int, default=10)
    parser.add_argument("--leading-edge-limit", type=int, default=30)
    parser.add_argument(
        "--skip-expression-plot",
        action="store_true",
        help="Only write sig-gene CSV; do not regenerate the expression-vs-isodepth PNG",
    )
    args = parser.parse_args()

    config_path = args.config_path.resolve()
    result_json_path = args.result_json_path.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else result_json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_json_config(str(config_path))
    run_cfg = run_config_from_mapping(cfg)
    decoder_type = str(getattr(run_cfg.test, "decoder", "nn"))
    decoder_df = _decoder_df_from_config(decoder_type)
    if decoder_df is None:
        raise ValueError(
            f"OLS refit only supports linear/quadratic decoders; got decoder={decoder_type!r}"
        )

    with result_json_path.open("r", encoding="utf-8") as f:
        result_json = json.load(f)
    artifacts = result_json.get("artifacts", {})
    iso_raw = artifacts.get("true_isodepth")
    if iso_raw is None:
        raise ValueError("Result JSON artifacts do not contain 'true_isodepth'.")
    iso = np.asarray(iso_raw, dtype=np.float64).reshape(-1)

    dataset = load_dataset(run_cfg.data)
    plot_dataset = _dataset_for_gene_expression_plots(dataset)
    A = np.asarray(plot_dataset.A, dtype=np.float64)
    if A.shape[0] != iso.shape[0]:
        raise ValueError(
            f"true_isodepth length ({iso.shape[0]}) != dataset cells ({A.shape[0]})"
        )
    gene_names = [str(v) for v in (plot_dataset.meta.get("var_names") or [
        f"gene_{i}" for i in range(A.shape[1])
    ])]
    run_name = result_json_path.stem.replace("_result", "")
    stem = _safe_name(run_name)

    print(
        f"[refit] decoder={decoder_type} df={decoder_df} N={A.shape[0]} G={A.shape[1]}",
        flush=True,
    )
    pred_ols = fit_closed_form_decoder(iso, A, decoder_type)
    pred_ols_f64 = np.asarray(pred_ols, dtype=np.float64)

    mse_ols = float(np.mean((A - pred_ols_f64) ** 2))
    mse_mean = float(np.mean((A - A.mean(axis=0, keepdims=True)) ** 2))
    pred_gd = artifacts.get("pred_true")
    if pred_gd is not None:
        pred_gd_arr = np.asarray(pred_gd, dtype=np.float64)
        mse_gd = float(np.mean((A - pred_gd_arr) ** 2))
        print(
            f"[refit] MSE mean-only={mse_mean:.6f}  gd={mse_gd:.6f}  ols={mse_ols:.6f}",
            flush=True,
        )
    else:
        print(f"[refit] MSE mean-only={mse_mean:.6f}  ols={mse_ols:.6f}", flush=True)

    npz_path = out_dir / f"{stem}_pred_true_ols_refit.npz"
    np.savez_compressed(
        npz_path,
        true_isodepth=iso.astype(np.float64),
        pred_true_ols=pred_ols.astype(np.float32),
        decoder_type=np.asarray([decoder_type]),
        mse_ols=np.asarray([mse_ols]),
    )
    print(f"[refit] wrote {npz_path.name}", flush=True)

    # --- Sig genes (archive GD-based CSV if present) ---
    expr_plot_stem = out_dir / f"{stem}_gene_expression_vs_isodepth"
    sig_csv = Path(f"{expr_plot_stem}_isodepth_sig_genes.csv")
    _archive_if_exists(sig_csv, "gd_decoder")

    svg = compute_isodepth_sig_genes(
        A,
        gene_names,
        pred_ols_f64,
        int(decoder_df),
        coord=iso,
        alpha=float(args.q_threshold),
    )
    _save_sig_genes_csv(
        sig_csv,
        gene_names,
        svg["pvalues"],
        svg["qvalues"],
        q_threshold=float(args.q_threshold),
    )
    n_sig = int(len(svg["sig_names"]))
    print(
        f"[sig] wrote {sig_csv.name}  (q<{args.q_threshold}: {n_sig} genes)",
        flush=True,
    )

    if not args.skip_expression_plot:
        plot_png = Path(f"{expr_plot_stem}.png")
        _archive_if_exists(plot_png, "gd_decoder")
        corr_png = Path(f"{expr_plot_stem}_correlation_distribution.png")
        _archive_if_exists(corr_png, "gd_decoder")
        save_gene_expression_vs_isodepth_plot(
            plot_dataset,
            iso,
            plot_png,
            coord_label="Isodepth",
            decoder_preds=pred_ols_f64,
            decoder_df=int(decoder_df),
            q_threshold=float(args.q_threshold),
            pvalues=svg["pvalues"],
            qvalues=svg["qvalues"],
        )
        print(f"[plot] wrote {plot_png.name}", flush=True)

    # --- GSEA on OLS-refit decoder scores ---
    gsea_dir = out_dir / "gsea_isodepth"
    gsea_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        f"{stem}_prerank_scores.csv",
        f"{stem}_gsea_results.csv",
        f"{stem}_top_pathways.png",
    ):
        _archive_if_exists(gsea_dir / name, "gd_decoder")

    gene_sets = _load_gmt(args.gmt.resolve())
    scores, pvals = _score_genes(
        A.astype(np.float32),
        iso,
        gene_names,
        method="decoder",
        pred=pred_ols_f64,
    )
    uniq_genes, uniq_scores, uniq_pvals = _collapse_duplicate_genes(gene_names, scores, pvals)
    order = np.argsort(-uniq_scores)
    results = _gsea_preranked(
        ranked_genes=uniq_genes[order],
        ranked_scores=uniq_scores[order],
        gene_sets=gene_sets,
        min_size=int(args.min_size),
        max_size=int(args.max_size),
        n_permutations=int(args.n_permutations),
        weight=float(args.weight),
        seed=int(args.seed),
    )
    prerank_csv = gsea_dir / f"{stem}_prerank_scores.csv"
    gsea_csv = gsea_dir / f"{stem}_gsea_results.csv"
    gsea_plot = gsea_dir / f"{stem}_top_pathways.png"
    _write_prerank_csv(prerank_csv, uniq_genes, uniq_scores, uniq_pvals)
    _write_gsea_csv(gsea_csv, results, args.leading_edge_limit)
    _plot_top_pathways(gsea_plot, results, top_n=int(args.top_plot_pathways))

    n_q = sum(1 for r in results if r.q_value < float(args.q_threshold))
    print(
        f"[gsea] wrote {prerank_csv.name}, {gsea_csv.name}, {gsea_plot.name} "
        f"(pathways={len(results)}, q<{args.q_threshold}={n_q})",
        flush=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
