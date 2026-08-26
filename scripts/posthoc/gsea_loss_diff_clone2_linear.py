"""Pre-ranked GSEA for the ``loss_diff_clone2_linear`` run (HT306P1 clone 2, tumor-proportion-
only covariate).

Target run:
    results/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2/loss_diff_clone2_linear

This run whitens on ``calicost_tumor_proportion`` only (see its
``loss_diff_clone2_linear_result.json`` -> ``config.data.covariate_whitening`` and ``agent.md``) --
i.e. it is the "tumor clone percentage as the only covariate" run for clone 2 of HT306P1, as
opposed to sibling configs that also whiten out total-counts/library-size.

Why this can't just use ``scripts/posthoc/postprocess_gsea_isodepth.py`` directly: that script's
"separate" cell-type branch reloads the dataset via ``load_dataset(data_cfg)`` from the run's
saved config, then subsets by ``cell_type_labels``. But this run's saved config predates
``obs_numeric_filters``/``spatial_crop`` serialization, so a naive reload yields 197 clone-2.0
cells instead of the 195 the model actually trained on (see agent.md) -- the row-count check in
``_extract_groups`` would then silently skip this group entirely (shape mismatch between the
reloaded 197-cell reconstruction and the saved 195-cell NPZ).

Fix used here: recover the exact 195-cell subset (in training order) by un-standardizing the
saved (z-scored) spatial coordinates using the run's saved ``coord_mean``/``coord_std`` and
nearest-neighbor-matching them back to raw spot coordinates in the source h5ad (same technique as
``scripts/posthoc/loss_diff_clone2_linear_qc_covariate_diagnostics.py``). We then re-run the exact
per-celltype preprocessing (``preprocess_celltype_subset`` with the run's saved
``separate_preprocessing`` params) on the correctly-matched raw counts to recover gene names
aligned 1:1 with the saved NPZ's ``A`` matrix columns, and sanity-check the reconstruction against
the saved ``A`` before using it for GSEA (the saved ``A`` itself is used for scoring, not the
reconstruction, since it's the ground truth training-space matrix).

Usage (from repo root, isodepth_env):
    python -m scripts.posthoc.gsea_loss_diff_clone2_linear --gmt data/gmt/h.all.v2026.1.Hs.symbols.gmt

Outputs (matching the convention of sibling clone-2 GSEA runs, e.g.
``results/calicost/HT306P1_S1H1Fc2U1Z1Bs1/loss_diff_clone2_linear_gt0p5_cropy/gsea_isodepth/``):
    <run_dir>/gsea_isodepth/2.0_prerank_scores.csv
    <run_dir>/gsea_isodepth/2.0_gsea_results.csv
    <run_dir>/gsea_isodepth/2.0_top_pathways.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

from experiments.core.paths import repo_root

REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from data.h5ad_loader import preprocess_celltype_subset  # noqa: E402
from scripts.posthoc.postprocess_gsea_isodepth import (  # noqa: E402
    _collapse_duplicate_genes,
    _gsea_preranked,
    _load_gmt,
    _plot_top_pathways,
    _score_genes,
    _write_gsea_csv,
    _write_prerank_csv,
)

RUN_DIR = REPO / "results/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2/loss_diff_clone2_linear"
RESULT_JSON = RUN_DIR / "loss_diff_clone2_linear_result.json"
CELL_TYPE = "2.0"
NPZ_PATH = RUN_DIR / CELL_TYPE / f"{CELL_TYPE}_isodepths.npz"


def _match_indices(raw_xy_target: np.ndarray, candidate_xy: np.ndarray, *, atol: float = 0.1) -> np.ndarray:
    """Nearest-neighbor match each row of ``raw_xy_target`` to ``candidate_xy``.

    Raises if any match is not (near-)exact or if matches are not unique, since a bad match
    would silently misalign ``true_isodepth``/``A`` against reconstructed gene names.
    """
    tree = cKDTree(candidate_xy)
    dist, idx = tree.query(raw_xy_target, k=1)
    if dist.max() > atol:
        raise ValueError(f"Worst spatial match residual {dist.max():.4f} exceeds tolerance {atol}")
    if len(set(idx.tolist())) != len(idx):
        raise ValueError("Non-unique nearest-neighbor matches; cannot safely align covariates to isodepth")
    return idx


def _recover_gene_names(result: dict, npz: np.lib.npyio.NpzFile) -> list[str]:
    dataset_meta = result["artifacts"]["dataset_meta"]
    coord_mean = np.asarray(dataset_meta["coord_mean"], dtype=np.float64)
    coord_std = np.asarray(dataset_meta["coord_std"], dtype=np.float64)
    h5ad_path = Path(dataset_meta["h5ad"])

    S = np.asarray(npz["S"], dtype=np.float64)
    A_saved = np.asarray(npz["A"], dtype=np.float32)
    n_cells = S.shape[0]
    raw_xy_target = S * coord_std + coord_mean

    print(f"[load] {h5ad_path}", flush=True)
    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs
    clone_mask = (obs["calicost_clone_label"].astype(str) == CELL_TYPE) & (
        ~obs["calicost_tumor_proportion"].isna()
    )
    candidate_xy = np.asarray(adata.obsm["spatial"])[clone_mask.values]
    candidate_full_idx = np.flatnonzero(clone_mask.values)
    print(f"[match] {n_cells} training spots vs {candidate_xy.shape[0]} h5ad candidates", flush=True)

    match_idx = _match_indices(raw_xy_target, candidate_xy)
    matched_full_idx = candidate_full_idx[match_idx]

    counts = adata.layers["counts"]
    import scipy.sparse as sp

    if sp.issparse(counts):
        counts = counts.toarray()
    counts_matched = np.asarray(counts[matched_full_idx], dtype=np.float32)
    var_names_full = [str(v) for v in adata.var_names]

    pp = dataset_meta["separate_preprocessing"]
    pp_params = {k: v for k, v in pp.items() if k != "seed"}
    pp_seed = int(pp.get("seed", 0))
    type_index = list(dataset_meta["cell_type_names"]).index(CELL_TYPE)

    A_recon, var_names_c, _feature_space = preprocess_celltype_subset(
        counts_matched,
        var_names_full,
        seed=pp_seed + type_index,
        **pp_params,
    )

    if A_recon.shape != A_saved.shape:
        raise ValueError(
            f"Reconstructed expression shape {A_recon.shape} != saved NPZ A shape {A_saved.shape}; "
            "cell matching or preprocessing params are wrong."
        )
    max_abs_diff = float(np.max(np.abs(A_recon - A_saved)))
    print(f"[verify] reconstructed vs saved A: max_abs_diff={max_abs_diff:.6g}", flush=True)
    if max_abs_diff > 1e-3:
        raise ValueError(
            f"Reconstructed expression does not match saved NPZ A closely enough "
            f"(max_abs_diff={max_abs_diff:.6g}); refusing to trust recovered gene names."
        )
    return var_names_c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmt", type=Path, required=True, help="Gene-set GMT file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--score-method",
        choices=["decoder", "spearman", "pearson"],
        default="decoder",
        help=(
            "Per-gene ranking statistic. 'decoder' (default) uses "
            "slope(pred, isodepth) * max(Pearson(obs, pred), 0)."
        ),
    )
    parser.add_argument(
        "--decoder-refit",
        choices=["closed-form", "none"],
        default="closed-form",
        help="For score-method=decoder: 'closed-form' (default) recomputes predictions via "
        "exact least-squares on the final isodepth instead of the noisy joint-trained decoder.",
    )
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--max-size", type=int, default=500)
    parser.add_argument("--n-permutations", type=int, default=250)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-plot-pathways", type=int, default=10)
    parser.add_argument("--leading-edge-limit", type=int, default=30)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve() if args.out_dir is not None else RUN_DIR / "gsea_isodepth"
    out_dir.mkdir(parents=True, exist_ok=True)

    with RESULT_JSON.open("r", encoding="utf-8") as f:
        result = json.load(f)

    npz = np.load(NPZ_PATH, allow_pickle=False)
    true_isodepth = np.asarray(npz["true_isodepth"], dtype=np.float64).reshape(-1)
    A = np.asarray(npz["A"], dtype=np.float32)
    pred = None
    if "pred_true" in npz:
        pred_arr = np.asarray(npz["pred_true"], dtype=np.float64)
        if pred_arr.shape == A.shape:
            pred = pred_arr
        else:
            raise ValueError(
                f"NPZ pred_true shape {pred_arr.shape} != A shape {A.shape}; "
                "cannot run decoder scoring."
            )

    gene_names = _recover_gene_names(result, npz)
    gene_sets = _load_gmt(args.gmt.resolve())
    print(f"Loaded {len(gene_sets)} pathways from {args.gmt}", flush=True)
    print(f"[group={CELL_TYPE}] N={A.shape[0]}, G={A.shape[1]}", flush=True)

    scores, pvals = _score_genes(
        A,
        true_isodepth,
        gene_names,
        method=args.score_method,
        pred=pred,
        decoder_type="linear",  # this run's run_name (this file's docstring) is the linear-decoder clone-2 GSEA
        decoder_refit=args.decoder_refit,
    )
    uniq_genes, uniq_scores, uniq_pvals = _collapse_duplicate_genes(gene_names, scores, pvals)
    order = np.argsort(-uniq_scores)
    ranked_genes = uniq_genes[order]
    ranked_scores = uniq_scores[order]

    results = _gsea_preranked(
        ranked_genes=ranked_genes,
        ranked_scores=ranked_scores,
        gene_sets=gene_sets,
        min_size=int(args.min_size),
        max_size=int(args.max_size),
        n_permutations=int(args.n_permutations),
        weight=float(args.weight),
        seed=int(args.seed),
    )

    prerank_csv = out_dir / f"{CELL_TYPE}_prerank_scores.csv"
    gsea_csv = out_dir / f"{CELL_TYPE}_gsea_results.csv"
    gsea_plot = out_dir / f"{CELL_TYPE}_top_pathways.png"
    _write_prerank_csv(prerank_csv, uniq_genes, uniq_scores, uniq_pvals)
    _write_gsea_csv(gsea_csv, results, args.leading_edge_limit)
    _plot_top_pathways(gsea_plot, results, top_n=int(args.top_plot_pathways))

    print(
        f"wrote: {prerank_csv.name}, {gsea_csv.name}, {gsea_plot.name} "
        f"(pathways tested: {len(results)})",
        flush=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
