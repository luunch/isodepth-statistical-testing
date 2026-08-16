"""Post-process isodepth runs with pre-ranked GSEA.

This script is a downstream analysis step: it reads a finished result JSON,
builds a per-gene ranking from the fitted decoder's association with isodepth
(default), and runs a pre-ranked GSEA against a user-provided GMT gene-set file.

Default gene score (``--score-method decoder``), for each gene j:
    slope_j = Cov(pred_j, isodepth) / Var(isodepth)   # decoder effect along isodepth
    fit_j   = Pearson(obs_j, pred_j)                  # how well decoder tracks data
    score_j = slope_j * max(fit_j, 0)

For a 1-D linear decoder, ``pred_j`` is affine in isodepth, so this reduces to a
signed, fit-weighted decoder slope (closely related to Pearson(obs, isodepth)).
Legacy ``spearman`` / ``pearson`` methods rank by direct obs-vs-isodepth correlation
and do not require ``pred_true``.

Usage (single run):
    python -m scripts.posthoc.postprocess_gsea_isodepth \
        configs/calicost/HT268B1_Th1H3Fc2U2Z1Bs1_cnv_profile_existence_gaussian_loss_difference_tumor_prop.json \
        results/calicost/HT268B1_Th1H3Fc2U2Z1Bs1_cnv_profile_existence_gaussian_loss_difference_tumor_prop/HT268B1_Th1H3Fc2U2Z1Bs1_cnv_profile_existence_gaussian_loss_difference_tumor_prop_result.json \
        --gmt /path/to/msigdb/h.all.v2026.1.Hs.symbols.gmt

Outputs:
    <result_dir>/gsea_isodepth/<group_name>_prerank_scores.csv
    <result_dir>/gsea_isodepth/<group_name>_gsea_results.csv
    <result_dir>/gsea_isodepth/<group_name>_top_pathways.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# This script lives in scripts/; add project root for package imports.
from experiments.core.paths import repo_root
REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from data import load_dataset  # noqa: E402
from data.h5ad_loader import preprocess_celltype_subset  # noqa: E402
from data.schemas import DataConfig, RunConfig, run_config_from_mapping  # noqa: E402
from experiments.configuration import (  # noqa: E402
    _dataset_for_gene_expression_plots,
    load_json_config,
)
from methods.trainers import fit_closed_form_decoder  # noqa: E402

CLOSED_FORM_DECODER_TYPES = ("linear", "quadratic")


@dataclass
class GroupData:
    name: str
    gene_names: list[str]
    expression: np.ndarray  # (N, G)
    isodepth: np.ndarray  # (N,)
    pred: np.ndarray | None = None  # (N, G) decoder predictions; required for method=decoder
    decoder_type: str = "nn"  # from run config's test.decoder; drives closed-form refit eligibility


@dataclass
class PathwayResult:
    pathway: str
    size: int
    overlap_size: int
    es: float
    nes: float
    p_value: float
    q_value: float
    leading_edge_size: int
    leading_edge_genes: list[str]


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=np.float64)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        running = min(running, val)
        q[i] = running
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _load_gmt(gmt_path: Path) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    with gmt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) < 3:
                continue
            pathway = parts[0].strip()
            genes = {g.strip() for g in parts[2:] if g.strip()}
            if pathway and genes:
                gene_sets[pathway] = genes
    if not gene_sets:
        raise ValueError(f"No gene sets found in GMT: {gmt_path}")
    return gene_sets


def _pearson_pvalues_from_r(r: np.ndarray, n: int) -> np.ndarray:
    """Two-sided p-values for Pearson correlations given r and sample size n."""
    from scipy.stats import t as student_t

    r = np.asarray(r, dtype=np.float64)
    pvals = np.ones(r.shape[0], dtype=np.float64)
    if n <= 2:
        return pvals
    valid = np.isfinite(r) & (np.abs(r) < 1.0)
    dof = float(n - 2)
    rr = np.clip(r[valid], -1.0 + 1e-15, 1.0 - 1e-15)
    t_stat = rr * np.sqrt(dof / np.maximum(1.0 - rr * rr, 1e-15))
    pvals[valid] = 2.0 * student_t.sf(np.abs(t_stat), dof)
    pvals[np.isfinite(r) & (np.abs(r) >= 1.0)] = 0.0
    return pvals


def _refit_closed_form_pred(
    isodepth: np.ndarray,
    expression: np.ndarray,
    decoder_type: str,
) -> np.ndarray:
    """Exact MSE-optimal decoder predictions given the model's *final* learned isodepth.

    Joint encoder+decoder training (Adam/SGD) leaves the decoder chasing a moving
    target (the encoder's latent keeps shifting), so the saved ``pred_true`` is a
    noisy, shrunk, sometimes sign-flipped estimate relative to the decoder you'd get
    by exactly solving OLS (linear) / polynomial least squares (quadratic) against the
    *frozen* final isodepth. This closed-form refit removes that optimization-noise
    artifact without touching training or the existence-test statistic at all -- it
    only changes the GSEA-and-friends "does the model track the data" view.
    """
    return fit_closed_form_decoder(isodepth, expression, decoder_type)


def _score_genes_decoder(
    expression: np.ndarray,
    isodepth: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank genes by decoder isodepth slope, shrunk by how well pred matches obs.

    score_j = slope_j * max(Pearson(obs_j, pred_j), 0)
    where slope_j = Cov(pred_j, isodepth) / Var(isodepth).
    """
    x = np.asarray(isodepth, dtype=np.float64).reshape(-1)
    A = np.asarray(expression, dtype=np.float64)
    P = np.asarray(pred, dtype=np.float64)
    if A.shape != P.shape:
        raise ValueError(f"expression shape {A.shape} != pred shape {P.shape}")
    if A.shape[0] != x.shape[0]:
        raise ValueError(
            f"Row mismatch between expression ({A.shape[0]}) and isodepth ({x.shape[0]})"
        )

    n, g = A.shape
    scores = np.zeros(g, dtype=np.float64)
    if n < 3:
        return scores, np.ones(g, dtype=np.float64)

    x_c = x - x.mean()
    var_x = float(np.dot(x_c, x_c) / n)
    if var_x <= 0.0:
        return scores, np.ones(g, dtype=np.float64)

    P_c = P - P.mean(axis=0, keepdims=True)
    A_c = A - A.mean(axis=0, keepdims=True)
    # Population-style covariances / correlations over cells.
    cov_xp = (x_c @ P_c) / n
    slopes = cov_xp / var_x

    ss_a = np.sum(A_c * A_c, axis=0)
    ss_p = np.sum(P_c * P_c, axis=0)
    denom = np.sqrt(ss_a * ss_p)
    fits = np.zeros(g, dtype=np.float64)
    ok = denom > 0.0
    fits[ok] = np.sum(A_c[:, ok] * P_c[:, ok], axis=0) / denom[ok]
    fits = np.where(np.isfinite(fits), fits, 0.0)

    scores = slopes * np.maximum(fits, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    pvals = _pearson_pvalues_from_r(fits, n)
    return scores, pvals


def _score_genes(
    expression: np.ndarray,
    isodepth: np.ndarray,
    gene_names: list[str],
    *,
    method: str,
    pred: np.ndarray | None = None,
    decoder_type: str = "nn",
    decoder_refit: str = "closed-form",
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(isodepth, dtype=np.float64).reshape(-1)
    A = np.asarray(expression, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"Expected expression to be 2D, got shape {A.shape}")
    if A.shape[0] != x.shape[0]:
        raise ValueError(
            f"Row mismatch between expression ({A.shape[0]}) and isodepth ({x.shape[0]})"
        )
    if len(gene_names) != A.shape[1]:
        raise ValueError(
            f"gene_names length {len(gene_names)} does not match expression columns {A.shape[1]}"
        )

    scores = np.zeros(A.shape[1], dtype=np.float64)
    pvals = np.ones(A.shape[1], dtype=np.float64)

    if method == "decoder":
        if decoder_refit == "closed-form" and decoder_type in CLOSED_FORM_DECODER_TYPES:
            pred = _refit_closed_form_pred(x, A, decoder_type)
        elif pred is None:
            raise ValueError(
                "score-method=decoder requires decoder predictions (pred_true) in the result "
                "artifacts / isodepths NPZ (or --decoder-refit closed-form with a linear/"
                "quadratic decoder)."
            )
        return _score_genes_decoder(A, x, pred)

    if method == "pearson":
        x_center = x - x.mean()
        x_denom = float(np.sqrt(np.sum(x_center**2)))
        if x_denom <= 0.0:
            return scores, pvals
        for j in range(A.shape[1]):
            y = A[:, j]
            y_center = y - y.mean()
            y_denom = float(np.sqrt(np.sum(y_center**2)))
            if y_denom <= 0.0:
                continue
            r = float(np.dot(x_center, y_center) / (x_denom * y_denom))
            scores[j] = r
        pvals = _pearson_pvalues_from_r(scores, A.shape[0])
        return scores, pvals

    # Spearman.
    for j in range(A.shape[1]):
        rho, p = spearmanr(x, A[:, j])
        if not np.isfinite(rho):
            rho = 0.0
        if not np.isfinite(p):
            p = 1.0
        scores[j] = float(rho)
        pvals[j] = float(p)
    return scores, pvals


def _collapse_duplicate_genes(
    gene_names: Iterable[str],
    scores: np.ndarray,
    pvals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = [str(g) for g in gene_names]
    best_idx_by_gene: dict[str, int] = {}
    for i, g in enumerate(names):
        if g not in best_idx_by_gene:
            best_idx_by_gene[g] = i
            continue
        prev = best_idx_by_gene[g]
        if abs(scores[i]) > abs(scores[prev]):
            best_idx_by_gene[g] = i

    selected = np.array(sorted(best_idx_by_gene.values()), dtype=np.int64)
    uniq_genes = np.array([names[i] for i in selected], dtype=object)
    return uniq_genes, scores[selected], pvals[selected]


def _enrichment_score(
    ranked_scores: np.ndarray,
    hit_mask: np.ndarray,
    *,
    weight: float,
) -> tuple[float, int]:
    N = ranked_scores.size
    k = int(hit_mask.sum())
    if k <= 0 or k >= N:
        return 0.0, 0

    abs_w = np.abs(ranked_scores) ** weight
    hit_w_sum = float(abs_w[hit_mask].sum())
    if hit_w_sum <= 0.0:
        return 0.0, 0

    miss_step = 1.0 / float(N - k)
    running = np.where(hit_mask, abs_w / hit_w_sum, -miss_step)
    running = np.cumsum(running)

    i_max = int(np.argmax(running))
    i_min = int(np.argmin(running))
    max_v = float(running[i_max])
    min_v = float(running[i_min])
    if abs(max_v) >= abs(min_v):
        return max_v, i_max
    return min_v, i_min


def _gsea_preranked(
    ranked_genes: np.ndarray,
    ranked_scores: np.ndarray,
    gene_sets: dict[str, set[str]],
    *,
    min_size: int,
    max_size: int,
    n_permutations: int,
    weight: float,
    seed: int,
) -> list[PathwayResult]:
    rng = np.random.default_rng(seed)
    N = ranked_genes.size
    gene_index = {g: i for i, g in enumerate(ranked_genes.tolist())}

    results: list[PathwayResult] = []
    for pathway, genes in gene_sets.items():
        overlap = [g for g in genes if g in gene_index]
        k = len(overlap)
        if k < int(min_size) or k > int(max_size):
            continue
        idx = np.array(sorted(gene_index[g] for g in overlap), dtype=np.int64)
        hit_mask = np.zeros(N, dtype=bool)
        hit_mask[idx] = True

        es_obs, lead_idx = _enrichment_score(ranked_scores, hit_mask, weight=weight)
        if abs(es_obs) <= 0.0:
            continue

        if es_obs >= 0:
            lead_hits = idx[idx <= lead_idx]
        else:
            lead_hits = idx[idx >= lead_idx]
        leading_edge = [str(ranked_genes[i]) for i in lead_hits]

        null_es = np.zeros(int(n_permutations), dtype=np.float64)
        for b in range(int(n_permutations)):
            rand_idx = rng.choice(N, size=k, replace=False)
            rand_mask = np.zeros(N, dtype=bool)
            rand_mask[rand_idx] = True
            null_es[b], _ = _enrichment_score(ranked_scores, rand_mask, weight=weight)

        if es_obs >= 0:
            same_sign = null_es[null_es > 0]
            denom = float(np.mean(same_sign)) if same_sign.size else np.nan
            p_num = float(np.sum(null_es >= es_obs) + 1.0)
        else:
            same_sign = null_es[null_es < 0]
            denom = float(np.mean(np.abs(same_sign))) if same_sign.size else np.nan
            p_num = float(np.sum(null_es <= es_obs) + 1.0)
        p_val = p_num / float(n_permutations + 1)
        if not np.isfinite(denom) or denom <= 1e-12:
            nes = float("nan")
        else:
            nes = float(es_obs / denom)

        results.append(
            PathwayResult(
                pathway=pathway,
                size=len(genes),
                overlap_size=k,
                es=float(es_obs),
                nes=nes,
                p_value=float(np.clip(p_val, 0.0, 1.0)),
                q_value=1.0,  # filled after all pathways are processed
                leading_edge_size=len(leading_edge),
                leading_edge_genes=leading_edge,
            )
        )

    if not results:
        return []

    pvals = np.asarray([r.p_value for r in results], dtype=np.float64)
    qvals = _bh_qvalues(pvals)
    for r, q in zip(results, qvals.tolist()):
        r.q_value = float(q)

    # Stable ordering: best q, then strongest |NES|, then pathway name.
    results.sort(
        key=lambda r: (
            r.q_value,
            -abs(r.nes) if np.isfinite(r.nes) else -math.inf,
            r.pathway,
        )
    )
    return results


def _write_prerank_csv(
    out_path: Path,
    genes: np.ndarray,
    scores: np.ndarray,
    pvals: np.ndarray,
) -> None:
    order = np.argsort(-scores)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "score", "p_value"])
        for i in order.tolist():
            writer.writerow([str(genes[i]), float(scores[i]), float(pvals[i])])


def _write_gsea_csv(out_path: Path, results: list[PathwayResult], leading_edge_limit: int) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pathway",
                "size",
                "overlap_size",
                "es",
                "nes",
                "p_value",
                "q_value",
                "leading_edge_size",
                "leading_edge_genes",
            ]
        )
        for r in results:
            leading = ";".join(r.leading_edge_genes[: int(leading_edge_limit)])
            writer.writerow(
                [
                    r.pathway,
                    int(r.size),
                    int(r.overlap_size),
                    float(r.es),
                    float(r.nes) if np.isfinite(r.nes) else "",
                    float(r.p_value),
                    float(r.q_value),
                    int(r.leading_edge_size),
                    leading,
                ]
            )


def _plot_top_pathways(out_path: Path, results: list[PathwayResult], top_n: int) -> None:
    finite = [r for r in results if np.isfinite(r.nes)]
    if not finite:
        return
    pos = [r for r in finite if r.nes > 0]
    neg = [r for r in finite if r.nes < 0]
    pos = sorted(pos, key=lambda r: r.nes, reverse=True)[:top_n]
    neg = sorted(neg, key=lambda r: r.nes)[:top_n]
    selected = pos + neg
    if not selected:
        return

    labels = [r.pathway for r in selected]
    vals = [r.nes for r in selected]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]

    fig_h = max(4.0, 0.35 * len(selected) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(len(selected))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title("Top Enriched Pathways")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _extract_groups(
    run_cfg: RunConfig,
    result_json_path: Path,
    result_json: dict,
) -> list[GroupData]:
    data_cfg = run_cfg.data
    dataset = load_dataset(data_cfg)
    decoder_type = str(getattr(run_cfg.test, "decoder", "nn"))

    artifacts = result_json.get("artifacts", {})
    run_name = result_json_path.stem.replace("_result", "")
    groups: list[GroupData] = []

    if artifacts.get("cell_type_mode") == "separate" or artifacts.get("cell_type_names"):
        cell_type_names = list(dataset.meta.get("cell_type_names", []))
        if not cell_type_names:
            raise ValueError("Result appears to be separate mode, but dataset has no cell_type_names.")
        cell_type_labels = np.asarray(dataset.meta.get("cell_type_labels"), dtype=np.int64)
        if cell_type_labels.ndim != 1:
            raise ValueError("dataset.meta['cell_type_labels'] missing or malformed.")

        var_names = dataset.meta.get("var_names")
        if var_names is None:
            var_names = [f"gene_{i}" for i in range(dataset.A.shape[1])]
        pp = dataset.meta.get("separate_preprocessing", {})
        pp_params = {k: v for k, v in pp.items() if k != "seed"}
        pp_seed = int(pp.get("seed", 0))

        for type_index, type_name in enumerate(cell_type_names):
            safe_name = _safe_name(type_name)
            npz_path = result_json_path.parent / safe_name / f"{safe_name}_isodepths.npz"
            if not npz_path.exists():
                print(f"[warn] missing NPZ for {type_name}: {npz_path}", flush=True)
                continue
            npz = np.load(npz_path, allow_pickle=False)
            if "true_isodepth" not in npz:
                print(f"[warn] missing true_isodepth in {npz_path}", flush=True)
                continue
            iso = np.asarray(npz["true_isodepth"], dtype=np.float64).reshape(-1)

            mask = cell_type_labels == int(type_index)
            A_raw = np.asarray(dataset.A[mask], dtype=np.float32)
            A_pre, var_names_c, _ = preprocess_celltype_subset(
                A_raw,
                [str(v) for v in var_names],
                seed=pp_seed + int(type_index),
                **pp_params,
            )
            A_npz = np.asarray(npz["A"], dtype=np.float32) if "A" in npz else None
            A_expr = A_pre
            if A_npz is not None:
                # Use the saved per-type matrix when present so downstream
                # analyses preserve the exact training-space representation
                # (e.g. Freedman-Lane residualized expression).
                if A_npz.shape[0] == iso.shape[0] and A_npz.shape[1] == len(var_names_c):
                    A_expr = A_npz
                else:
                    print(
                        f"[warn] NPZ A shape mismatch for {type_name}: "
                        f"A_npz={A_npz.shape}, expected=({iso.shape[0]}, {len(var_names_c)}); "
                        "falling back to reconstructed preprocessed expression.",
                        flush=True,
                    )

            pred = None
            if "pred_true" in npz:
                pred_arr = np.asarray(npz["pred_true"], dtype=np.float64)
                if pred_arr.shape == A_expr.shape:
                    pred = pred_arr
                else:
                    print(
                        f"[warn] NPZ pred_true shape mismatch for {type_name}: "
                        f"pred_true={pred_arr.shape}, expected={A_expr.shape}; "
                        "decoder scoring will be unavailable for this group.",
                        flush=True,
                    )

            if A_expr.shape[0] != iso.shape[0]:
                print(
                    f"[warn] skipping {type_name}: cell mismatch A={A_expr.shape[0]} vs isodepth={iso.shape[0]}",
                    flush=True,
                )
                continue
            groups.append(
                GroupData(
                    name=str(type_name),
                    gene_names=[str(v) for v in var_names_c],
                    expression=np.asarray(A_expr, dtype=np.float32),
                    isodepth=iso,
                    pred=pred,
                    decoder_type=decoder_type,
                )
            )
        return groups

    plot_dataset = _dataset_for_gene_expression_plots(dataset)
    iso_raw = artifacts.get("true_isodepth")
    if iso_raw is None:
        raise ValueError("Result JSON artifacts do not contain 'true_isodepth'.")
    iso = np.asarray(iso_raw, dtype=np.float64).reshape(-1)
    if iso.shape[0] != plot_dataset.A.shape[0]:
        raise ValueError(
            f"true_isodepth length ({iso.shape[0]}) does not match dataset cells ({plot_dataset.A.shape[0]})."
        )
    var_names = plot_dataset.meta.get("var_names")
    if var_names is None:
        var_names = [f"gene_{i}" for i in range(plot_dataset.A.shape[1])]
    pred = None
    pred_raw = artifacts.get("pred_true")
    if pred_raw is not None:
        pred_arr = np.asarray(pred_raw, dtype=np.float64)
        if pred_arr.shape == tuple(plot_dataset.A.shape):
            pred = pred_arr
        else:
            print(
                f"[warn] pred_true shape mismatch: pred_true={pred_arr.shape}, "
                f"expected={plot_dataset.A.shape}; decoder scoring will be unavailable.",
                flush=True,
            )
    groups.append(
        GroupData(
            name=run_name,
            gene_names=[str(v) for v in var_names],
            expression=np.asarray(plot_dataset.A, dtype=np.float32),
            isodepth=iso,
            pred=pred,
            decoder_type=decoder_type,
        )
    )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path, help="Run config JSON path")
    parser.add_argument("result_json_path", type=Path, help="Saved *_result.json path")
    parser.add_argument("--gmt", type=Path, required=True, help="Gene-set GMT file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <result_dir>/gsea_isodepth)",
    )
    parser.add_argument(
        "--score-method",
        choices=["decoder", "spearman", "pearson"],
        default="decoder",
        help=(
            "Per-gene ranking statistic. 'decoder' (default) uses "
            "slope(pred, isodepth) * max(Pearson(obs, pred), 0); "
            "'spearman'/'pearson' use direct obs-vs-isodepth correlation."
        ),
    )
    parser.add_argument(
        "--decoder-refit",
        choices=["closed-form", "none"],
        default="closed-form",
        help=(
            "For score-method=decoder with a linear/quadratic decoder: 'closed-form' (default) "
            "recomputes predictions by exactly solving least-squares against the model's final "
            "isodepth, instead of trusting the noisy jointly-trained decoder weights (see module "
            "docstring). 'none' uses the raw saved pred_true."
        ),
    )
    parser.add_argument("--min-size", type=int, default=15, help="Min overlapping genes per pathway")
    parser.add_argument("--max-size", type=int, default=500, help="Max overlapping genes per pathway")
    parser.add_argument("--n-permutations", type=int, default=250, help="Permutation count for null ES")
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="GSEA running-sum weighting exponent p (classic=0, weighted=1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--top-plot-pathways",
        type=int,
        default=10,
        help="Top positive and negative pathways to show in plot",
    )
    parser.add_argument(
        "--leading-edge-limit",
        type=int,
        default=30,
        help="How many leading-edge genes to write per pathway row",
    )
    args = parser.parse_args()

    if args.min_size <= 0:
        raise ValueError("--min-size must be > 0")
    if args.max_size < args.min_size:
        raise ValueError("--max-size must be >= --min-size")
    if args.n_permutations <= 0:
        raise ValueError("--n-permutations must be > 0")
    if args.weight < 0.0:
        raise ValueError("--weight must be >= 0")

    config_path = args.config_path.resolve()
    result_json_path = args.result_json_path.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else result_json_path.parent / "gsea_isodepth"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_json_config(str(config_path))
    run_cfg = run_config_from_mapping(cfg)
    with result_json_path.open("r", encoding="utf-8") as f:
        result_json = json.load(f)

    gene_sets = _load_gmt(args.gmt.resolve())
    groups = _extract_groups(run_cfg, result_json_path, result_json)
    if not groups:
        raise ValueError("No analyzable groups found (missing isodepth/NPZs?).")

    print(f"Loaded {len(gene_sets)} pathways from {args.gmt}", flush=True)
    print(f"Analyzing {len(groups)} group(s)...", flush=True)

    for group in groups:
        print(
            f"[group={group.name}] N={group.expression.shape[0]}, G={group.expression.shape[1]}",
            flush=True,
        )
        scores, pvals = _score_genes(
            group.expression,
            group.isodepth,
            group.gene_names,
            method=args.score_method,
            pred=group.pred,
            decoder_type=group.decoder_type,
            decoder_refit=args.decoder_refit,
        )
        uniq_genes, uniq_scores, uniq_pvals = _collapse_duplicate_genes(group.gene_names, scores, pvals)
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

        stem = _safe_name(group.name)
        prerank_csv = out_dir / f"{stem}_prerank_scores.csv"
        gsea_csv = out_dir / f"{stem}_gsea_results.csv"
        gsea_plot = out_dir / f"{stem}_top_pathways.png"
        _write_prerank_csv(prerank_csv, uniq_genes, uniq_scores, uniq_pvals)
        _write_gsea_csv(gsea_csv, results, args.leading_edge_limit)
        _plot_top_pathways(gsea_plot, results, top_n=int(args.top_plot_pathways))

        print(
            f"  wrote: {prerank_csv.name}, {gsea_csv.name}, {gsea_plot.name} "
            f"(pathways tested: {len(results)})",
            flush=True,
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
