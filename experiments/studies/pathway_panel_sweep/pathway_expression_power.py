from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from data import load_dataset
from data.schemas import run_config_from_mapping
from experiments.configuration import build_run_config
from experiments.core.study_io import write_csv
from experiments.studies.pathway_panel_sweep.lib import (
    analysis_dir_for_spec,
    load_gmt_gene_sets,
    load_pathway_panel_sweep_spec,
)
from experiments.studies.pathway_panel_sweep.pathway_isodepth_correlation import (
    _load_isodepth_from_result,
    _reference_result_path,
)

POWER_FIELDS = [
    "pathway_name",
    "pathway_genes_in_gmt",
    "pathway_genes_in_h5ad",
    "pathway_genes_min3cells",
    "gene_retention_vs_gmt",
    "mean_detection_rate",
    "median_detection_rate",
    "mean_log1p_cpm",
    "median_log1p_cpm",
    "mean_pathway_raw_counts_per_cell",
    "median_pathway_raw_counts_per_cell",
    "spearman_pathway_score_vs_spatial_y",
    "spearman_pathway_score_vs_reference_isodepth",
    "existence_p_value",
    "existence_significant",
    "n_genes_surviving_from_run",
]


def _load_raw_counts_and_coords(base_config_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    run_config = build_run_config(str(base_config_path), {})
    mapping = copy.deepcopy(run_config.to_dict())
    mapping.setdefault("data", {})
    mapping["data"]["gene_list"] = None
    mapping["data"]["top_var_genes"] = 0
    mapping["data"]["normalize_total"] = False
    mapping["data"]["log1p"] = False
    mapping["data"]["standardize_expression"] = False
    raw_cfg = run_config_from_mapping(mapping).data
    dataset = load_dataset(raw_cfg)

    cell_type_labels = np.asarray(dataset.meta.get("cell_type_labels"), dtype=np.int64)
    if cell_type_labels.ndim != 1:
        raise ValueError("Expected cell_type_labels for separate-mode dataset")
    mask = cell_type_labels == 0
    counts = np.asarray(dataset.A[mask], dtype=np.float64)
    coords = np.asarray(dataset.S[mask], dtype=np.float64)
    var_names = [str(v) for v in dataset.meta.get("var_names", [])]
    if counts.shape[1] != len(var_names):
        raise ValueError("Raw counts and var_names length mismatch")
    return counts, var_names, coords


def _log1p_cpm(counts: np.ndarray) -> np.ndarray:
    library_size = counts.sum(axis=1, keepdims=True)
    library_size = np.maximum(library_size, 1.0)
    cpm = counts / library_size * 1e4
    return np.log1p(cpm)


def compute_pathway_expression_metrics(
    counts: np.ndarray,
    var_names: list[str],
    coords: np.ndarray,
    pathway_genes: list[str],
    *,
    min_cells_per_gene: int,
    reference_isodepth: np.ndarray | None = None,
) -> dict[str, float | int]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
    gmt_genes = list(pathway_genes)
    in_h5ad = [gene for gene in gmt_genes if gene in gene_to_idx]
    min_cells = int(min_cells_per_gene)
    passing: list[str] = []
    detection_rates: list[float] = []
    for gene in in_h5ad:
        idx = gene_to_idx[gene]
        column = counts[:, idx]
        n_detected = int(np.sum(column > 0))
        if n_detected >= min_cells:
            passing.append(gene)
        detection_rates.append(float(n_detected / counts.shape[0]))

    if not passing:
        return {
            "pathway_genes_in_gmt": int(len(gmt_genes)),
            "pathway_genes_in_h5ad": int(len(in_h5ad)),
            "pathway_genes_min3cells": 0,
            "gene_retention_vs_gmt": 0.0,
            "mean_detection_rate": float(np.mean(detection_rates)) if detection_rates else 0.0,
            "median_detection_rate": float(np.median(detection_rates)) if detection_rates else 0.0,
            "mean_log1p_cpm": float("nan"),
            "median_log1p_cpm": float("nan"),
            "mean_pathway_raw_counts_per_cell": float("nan"),
            "median_pathway_raw_counts_per_cell": float("nan"),
            "spearman_pathway_score_vs_spatial_y": float("nan"),
            "spearman_pathway_score_vs_reference_isodepth": float("nan"),
        }

    indices = [gene_to_idx[gene] for gene in passing]
    sub_counts = counts[:, indices]
    log1p_cpm = _log1p_cpm(sub_counts)
    per_cell_pathway_sum = sub_counts.sum(axis=1)
    per_gene_mean_log = log1p_cpm.mean(axis=0)
    pathway_score = log1p_cpm.mean(axis=1)

    spatial_y = coords[:, 1]
    rho_y, _ = spearmanr(pathway_score, spatial_y)
    rho_iso = float("nan")
    if reference_isodepth is not None and reference_isodepth.shape[0] == pathway_score.shape[0]:
        rho_iso, _ = spearmanr(pathway_score, reference_isodepth)

    return {
        "pathway_genes_in_gmt": int(len(gmt_genes)),
        "pathway_genes_in_h5ad": int(len(in_h5ad)),
        "pathway_genes_min3cells": int(len(passing)),
        "gene_retention_vs_gmt": float(len(passing) / len(gmt_genes)) if gmt_genes else 0.0,
        "mean_detection_rate": float(np.mean(detection_rates)) if detection_rates else 0.0,
        "median_detection_rate": float(np.median(detection_rates)) if detection_rates else 0.0,
        "mean_log1p_cpm": float(np.mean(per_gene_mean_log)),
        "median_log1p_cpm": float(np.median(per_gene_mean_log)),
        "mean_pathway_raw_counts_per_cell": float(np.mean(per_cell_pathway_sum)),
        "median_pathway_raw_counts_per_cell": float(np.median(per_cell_pathway_sum)),
        "spearman_pathway_score_vs_spatial_y": float(rho_y) if np.isfinite(rho_y) else float("nan"),
        "spearman_pathway_score_vs_reference_isodepth": float(rho_iso),
    }


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_float(value: object) -> float | None:
    if value == "" or value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def save_pathway_expression_power_scatter(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Scatter existence p-value vs gene count / pathway counts, colored by significance."""
    points: list[tuple[float, float, float, bool, str]] = []
    for row in rows:
        p_value = _finite_float(row.get("existence_p_value"))
        n_genes = _finite_float(row.get("n_genes_surviving_from_run"))
        if n_genes is None:
            n_genes = _finite_float(row.get("pathway_genes_min3cells"))
        counts = _finite_float(row.get("mean_pathway_raw_counts_per_cell"))
        if p_value is None or n_genes is None or counts is None:
            continue
        points.append(
            (
                n_genes,
                counts,
                p_value,
                _as_bool(row.get("existence_significant")),
                str(row.get("pathway_name", "")).replace("HALLMARK_", ""),
            )
        )
    if not points:
        raise ValueError("No finite rows available for pathway expression-power scatter")

    n_genes_arr = np.asarray([p[0] for p in points], dtype=np.float64)
    counts_arr = np.asarray([p[1] for p in points], dtype=np.float64)
    p_arr = np.asarray([p[2] for p in points], dtype=np.float64)
    sig_mask = np.asarray([p[3] for p in points], dtype=bool)
    labels = [p[4] for p in points]

    rho_genes, rho_genes_p = spearmanr(n_genes_arr, p_arr)
    rho_counts, rho_counts_p = spearmanr(counts_arr, p_arr)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    panels = [
        (
            axes[0],
            n_genes_arr,
            "# genes surviving",
            float(rho_genes),
            float(rho_genes_p),
        ),
        (
            axes[1],
            counts_arr,
            "Mean pathway raw counts / cell",
            float(rho_counts),
            float(rho_counts_p),
        ),
    ]
    for ax, x, xlabel, rho, rho_p in panels:
        if np.any(~sig_mask):
            ax.scatter(
                x[~sig_mask],
                p_arr[~sig_mask],
                c="#4C78A8",
                s=42,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                label=f"not significant (n={int((~sig_mask).sum())})",
                zorder=2,
            )
        if np.any(sig_mask):
            ax.scatter(
                x[sig_mask],
                p_arr[sig_mask],
                c="#F58518",
                s=42,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                label=f"significant (n={int(sig_mask.sum())})",
                zorder=3,
            )
        ax.axhline(0.05, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.02, min(1.05, max(p_arr.max() * 1.05, 0.1)))
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(f"Spearman ρ={rho:.2f} (p={rho_p:.1e})")
        # Label a few informative outliers: small significant / large nonsignificant.
        ranked = sorted(
            range(len(points)),
            key=lambda i: (0 if sig_mask[i] else 1, x[i] if sig_mask[i] else -x[i]),
        )
        for idx in ranked[:2]:
            ax.annotate(
                labels[idx],
                (x[idx], p_arr[idx]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.9,
            )

    axes[0].set_ylabel("Existence p-value")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze_pathway_expression_power(
    spec_path: str,
    *,
    reference_pathway_name: str = "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
) -> dict[str, object]:
    spec = load_pathway_panel_sweep_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    base_run_config = build_run_config(str(spec.base_config), {})
    min_cells_per_gene = int(base_run_config.data.min_cells_per_gene)

    reference_result_path = _reference_result_path(
        spec_path,
        reference_pathway_name=reference_pathway_name,
    )
    reference_isodepth = _load_isodepth_from_result(reference_result_path)
    counts, var_names, coords = _load_raw_counts_and_coords(spec.base_config)
    if counts.shape[0] != reference_isodepth.shape[0]:
        raise ValueError(
            f"Cell count mismatch: raw counts {counts.shape[0]} vs reference isodepth "
            f"{reference_isodepth.shape[0]}"
        )

    gene_sets = load_gmt_gene_sets(spec.gmt_path)
    existence_rows: dict[str, dict[str, str]] = {}
    per_pathway_csv = analysis_dir / "per_pathway_results.csv"
    if per_pathway_csv.exists():
        with open(per_pathway_csv, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existence_rows[str(row["pathway_name"])] = dict(row)

    rows: list[dict[str, object]] = []
    for pathway_name in sorted(gene_sets.keys()):
        metrics = compute_pathway_expression_metrics(
            counts,
            var_names,
            coords,
            gene_sets[pathway_name],
            min_cells_per_gene=min_cells_per_gene,
            reference_isodepth=reference_isodepth,
        )
        existence = existence_rows.get(pathway_name, {})
        row = {
            "pathway_name": pathway_name,
            **metrics,
            "existence_p_value": existence.get("p_value", ""),
            "existence_significant": existence.get("significant", ""),
            "n_genes_surviving_from_run": existence.get("n_genes_surviving", ""),
        }
        rows.append(row)

    out_csv = analysis_dir / "pathway_expression_power.csv"
    write_csv(out_csv, rows, fieldnames=POWER_FIELDS)

    scatter_path = save_pathway_expression_power_scatter(
        rows,
        analysis_dir / "pathway_expression_power_scatter.png",
        title=f"{spec.experiment_name}: existence p vs pathway size / counts",
    )

    sig_rows = [row for row in rows if _as_bool(row.get("existence_significant"))]
    nonsig_rows = [row for row in rows if not _as_bool(row.get("existence_significant"))]

    def _metric_values(selected: list[dict[str, object]], key: str) -> np.ndarray:
        values = []
        for row in selected:
            value = row.get(key)
            if value == "" or value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                values.append(numeric)
        return np.asarray(values, dtype=np.float64)

    comparisons: dict[str, dict[str, float | int | None]] = {}
    for metric in [
        "pathway_genes_min3cells",
        "gene_retention_vs_gmt",
        "mean_detection_rate",
        "mean_log1p_cpm",
        "mean_pathway_raw_counts_per_cell",
        "spearman_pathway_score_vs_reference_isodepth",
    ]:
        sig_vals = _metric_values(sig_rows, metric)
        nonsig_vals = _metric_values(nonsig_rows, metric)
        entry: dict[str, float | int | None] = {
            "sig_mean": float(np.mean(sig_vals)) if sig_vals.size else None,
            "nonsig_mean": float(np.mean(nonsig_vals)) if nonsig_vals.size else None,
            "sig_median": float(np.median(sig_vals)) if sig_vals.size else None,
            "nonsig_median": float(np.median(nonsig_vals)) if nonsig_vals.size else None,
            "mannwhitney_p": None,
        }
        if sig_vals.size and nonsig_vals.size:
            try:
                _, p_value = mannwhitneyu(sig_vals, nonsig_vals, alternative="two-sided")
                entry["mannwhitney_p"] = float(p_value)
            except ValueError:
                entry["mannwhitney_p"] = None
        comparisons[metric] = entry

    high_gene_nonsig = sorted(
        [
            row
            for row in nonsig_rows
            if int(row.get("pathway_genes_min3cells", 0)) >= 100
        ],
        key=lambda row: float(row.get("existence_p_value") or 1.0),
        reverse=True,
    )
    low_expr_nonsig = sorted(
        nonsig_rows,
        key=lambda row: float(row.get("mean_log1p_cpm") or float("inf")),
    )[:8]

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "reference_pathway_name": reference_pathway_name,
        "n_pathways": len(rows),
        "n_existence_significant": len(sig_rows),
        "n_existence_nonsignificant": len(nonsig_rows),
        "sig_vs_nonsig_comparisons": comparisons,
        "high_gene_count_nonsignificant_pathways": [
            {
                "pathway_name": row["pathway_name"],
                "pathway_genes_min3cells": row["pathway_genes_min3cells"],
                "mean_log1p_cpm": row["mean_log1p_cpm"],
                "mean_detection_rate": row["mean_detection_rate"],
                "spearman_vs_isodepth": row["spearman_pathway_score_vs_reference_isodepth"],
                "existence_p_value": row["existence_p_value"],
            }
            for row in high_gene_nonsig
        ],
        "lowest_expression_nonsignificant_pathways": [
            {
                "pathway_name": row["pathway_name"],
                "pathway_genes_min3cells": row["pathway_genes_min3cells"],
                "mean_log1p_cpm": row["mean_log1p_cpm"],
                "mean_detection_rate": row["mean_detection_rate"],
                "existence_p_value": row["existence_p_value"],
            }
            for row in low_expr_nonsig
        ],
        "pathway_expression_power_csv": str(out_csv),
        "pathway_expression_power_scatter_png": str(scatter_path),
    }
    summary_path = analysis_dir / "pathway_expression_power_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["pathway_expression_power_summary_json"] = str(summary_path)
    return payload
