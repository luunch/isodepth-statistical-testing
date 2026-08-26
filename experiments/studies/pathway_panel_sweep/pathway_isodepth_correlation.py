from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr

from data import load_dataset
from data.h5ad_loader import preprocess_celltype_subset
from data.schemas import run_config_from_mapping
from experiments.configuration import build_run_config
from experiments.core.study_io import load_result_payload, write_csv
from experiments.studies.pathway_panel_sweep.lib import (
    analysis_dir_for_spec,
    load_gmt_gene_sets,
    load_manifest_entries,
    load_pathway_panel_sweep_spec,
    manifest_path_for_spec,
)
from scripts.posthoc.postprocess_gsea_isodepth import _bh_qvalues

CORRELATION_FIELDS = [
    "pathway_name",
    "pathway_genes_in_gmt",
    "pathway_genes_matched",
    "pathway_score_n_genes",
    "spearman_rho_reference",
    "spearman_p_reference",
    "spearman_q_reference",
    "spearman_rho_own_isodepth",
    "spearman_p_own_isodepth",
    "reference_pathway_name",
    "reference_isodepth_source",
    "existence_p_value",
    "existence_significant",
    "existence_stat_true",
]


def _reference_result_path(
    spec_path: str | Path,
    *,
    reference_pathway_name: str,
) -> Path:
    spec = load_pathway_panel_sweep_spec(spec_path)
    manifest = load_manifest_entries(manifest_path_for_spec(spec))
    for entry in manifest.values():
        if str(entry.get("pathway_name")) == reference_pathway_name:
            return Path(str(entry["result_json_path"])).resolve()
    safe_suffix = reference_pathway_name.replace("HALLMARK_", "").lower()
    candidate = (
        spec.output_root
        / "runs"
        / f"{spec.experiment_name}__{safe_suffix}"
        / f"{spec.experiment_name}__{safe_suffix}_result.json"
    )
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(
        f"Could not find reference pathway result for {reference_pathway_name!r} under {spec.output_root}"
    )


def _load_broad_expression_matrix(
    base_config_path: Path,
    *,
    top_var_genes: int,
    expected_n_cells: int,
) -> tuple[np.ndarray, list[str]]:
    run_config = build_run_config(str(base_config_path), {})
    mapping = copy.deepcopy(run_config.to_dict())
    mapping.setdefault("data", {})
    mapping["data"]["gene_list"] = None
    mapping["data"]["top_var_genes"] = int(top_var_genes)
    broad_cfg = run_config_from_mapping(mapping).data
    dataset = load_dataset(broad_cfg)

    cell_type_names = list(dataset.meta.get("cell_type_names", []))
    if len(cell_type_names) != 1:
        raise ValueError(
            f"Expected exactly one cell type for pathway correlation analysis, got {cell_type_names}"
        )
    type_index = 0
    cell_type_labels = np.asarray(dataset.meta.get("cell_type_labels"), dtype=np.int64)
    var_names = [str(v) for v in dataset.meta.get("var_names", [])]
    pp = dataset.meta.get("separate_preprocessing", {})
    pp_params = {k: v for k, v in pp.items() if k != "seed"}
    pp_seed = int(pp.get("seed", 0))

    mask = cell_type_labels == int(type_index)
    counts = np.asarray(dataset.A[mask], dtype=np.float32)
    expression, var_names_out, _ = preprocess_celltype_subset(
        counts,
        var_names,
        seed=pp_seed + int(type_index),
        **pp_params,
    )
    if int(expression.shape[0]) != int(expected_n_cells):
        raise ValueError(
            f"Broad expression cell count {expression.shape[0]} != reference isodepth "
            f"length {expected_n_cells}"
        )
    return np.asarray(expression, dtype=np.float64), [str(v) for v in var_names_out]


def compute_pathway_mean_score(
    expression: np.ndarray,
    gene_names: list[str],
    pathway_genes: list[str] | set[str],
) -> tuple[np.ndarray, int]:
    wanted = {str(g) for g in pathway_genes}
    indices = [idx for idx, gene in enumerate(gene_names) if gene in wanted]
    if not indices:
        raise ValueError("No pathway genes matched the expression matrix")
    scores = np.mean(expression[:, indices], axis=1)
    return scores, len(indices)


def _isodepth_npz_path_from_result(payload: Mapping[str, Any]) -> Path:
    summaries = payload.get("artifacts", {}).get("per_type_summaries", {})
    if not isinstance(summaries, Mapping) or not summaries:
        raise ValueError("result JSON missing artifacts.per_type_summaries")
    for type_summary in summaries.values():
        if not isinstance(type_summary, Mapping):
            continue
        artifact_paths = type_summary.get("artifact_paths", {})
        if not isinstance(artifact_paths, Mapping):
            continue
        npz_value = artifact_paths.get("isodepths_npz")
        if npz_value:
            return Path(str(npz_value)).resolve()
    raise ValueError("Could not locate isodepths_npz in result JSON")


def _load_isodepth_from_result(result_json_path: Path) -> np.ndarray:
    payload = load_result_payload(result_json_path)
    npz_path = _isodepth_npz_path_from_result(payload)
    npz = np.load(npz_path, allow_pickle=False)
    if "true_isodepth" not in npz:
        raise ValueError(f"Missing true_isodepth in {npz_path}")
    return np.asarray(npz["true_isodepth"], dtype=np.float64).reshape(-1)


def _load_own_isodepth_and_score(result_json_path: Path) -> tuple[float, float, int]:
    payload = load_result_payload(result_json_path)
    npz_path = _isodepth_npz_path_from_result(payload)
    npz = np.load(npz_path, allow_pickle=False)
    iso = np.asarray(npz["true_isodepth"], dtype=np.float64).reshape(-1)
    if "A" not in npz:
        raise ValueError(f"Missing expression matrix A in {npz_path}")
    expression = np.asarray(npz["A"], dtype=np.float64)
    if expression.ndim != 2 or expression.shape[0] != iso.shape[0]:
        raise ValueError(f"NPZ A/isodepth shape mismatch in {npz_path}")
    scores = np.mean(expression, axis=1)
    rho, p_value = spearmanr(iso, scores)
    if not np.isfinite(rho):
        rho = 0.0
    if not np.isfinite(p_value):
        p_value = 1.0
    return float(rho), float(p_value), int(expression.shape[1])


def analyze_pathway_isodepth_correlations(
    spec_path: str,
    *,
    reference_pathway_name: str = "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    expression_top_var_genes: int = 3000,
    pathway_score_min_genes: int = 5,
) -> dict[str, object]:
    spec = load_pathway_panel_sweep_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    gene_sets = load_gmt_gene_sets(spec.gmt_path)
    reference_result_path = _reference_result_path(
        spec_path,
        reference_pathway_name=reference_pathway_name,
    )
    reference_isodepth = _load_isodepth_from_result(reference_result_path)

    broad_expression, broad_gene_names = _load_broad_expression_matrix(
        spec.base_config,
        top_var_genes=int(expression_top_var_genes),
        expected_n_cells=int(reference_isodepth.shape[0]),
    )

    existence_rows: dict[str, dict[str, object]] = {}
    per_pathway_csv = analysis_dir / "per_pathway_results.csv"
    if per_pathway_csv.exists():
        import csv

        with open(per_pathway_csv, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existence_rows[str(row["pathway_name"])] = dict(row)

    manifest = load_manifest_entries(manifest_path_for_spec(spec))
    result_path_by_pathway: dict[str, Path] = {}
    for entry in manifest.values():
        pathway = str(entry.get("pathway_name", ""))
        if pathway:
            result_path_by_pathway[pathway] = Path(str(entry["result_json_path"])).resolve()

    rows: list[dict[str, object]] = []
    reference_rhos: list[float] = []
    reference_ps: list[float] = []

    for pathway_name in sorted(gene_sets.keys()):
        pathway_genes = gene_sets[pathway_name]
        try:
            scores, n_matched = compute_pathway_mean_score(
                broad_expression,
                broad_gene_names,
                pathway_genes,
            )
        except ValueError:
            continue
        if n_matched < int(pathway_score_min_genes):
            continue

        rho_ref, p_ref = spearmanr(reference_isodepth, scores)
        if not np.isfinite(rho_ref):
            rho_ref = 0.0
        if not np.isfinite(p_ref):
            p_ref = 1.0

        result_json_path = result_path_by_pathway.get(pathway_name)
        rho_own = float("nan")
        p_own = float("nan")
        if result_json_path is not None and result_json_path.exists():
            try:
                rho_own, p_own, _ = _load_own_isodepth_and_score(result_json_path)
            except Exception:
                rho_own = float("nan")
                p_own = float("nan")

        existence = existence_rows.get(pathway_name, {})
        row = {
            "pathway_name": pathway_name,
            "pathway_genes_in_gmt": int(len(pathway_genes)),
            "pathway_genes_matched": int(n_matched),
            "pathway_score_n_genes": int(n_matched),
            "spearman_rho_reference": float(rho_ref),
            "spearman_p_reference": float(p_ref),
            "spearman_q_reference": float("nan"),
            "spearman_rho_own_isodepth": float(rho_own),
            "spearman_p_own_isodepth": float(p_own),
            "reference_pathway_name": reference_pathway_name,
            "reference_isodepth_source": str(reference_result_path),
            "existence_p_value": existence.get("p_value", ""),
            "existence_significant": existence.get("significant", ""),
            "existence_stat_true": existence.get("stat_true", ""),
        }
        rows.append(row)
        reference_rhos.append(float(rho_ref))
        reference_ps.append(float(p_ref))

    q_values = _bh_qvalues(np.asarray(reference_ps, dtype=np.float64))
    for row, q_value in zip(rows, q_values.tolist()):
        row["spearman_q_reference"] = float(q_value)

    rows.sort(key=lambda row: abs(float(row["spearman_rho_reference"])), reverse=True)

    out_csv = analysis_dir / "pathway_isodepth_spearman_correlations.csv"
    write_csv(out_csv, rows, fieldnames=CORRELATION_FIELDS)

    abs_rho = np.abs(np.asarray(reference_rhos, dtype=np.float64))
    sig_q05 = [row for row in rows if float(row["spearman_q_reference"]) < 0.05]
    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "reference_pathway_name": reference_pathway_name,
        "reference_isodepth_source": str(reference_result_path),
        "reference_n_cells": int(reference_isodepth.shape[0]),
        "expression_top_var_genes": int(expression_top_var_genes),
        "pathway_score_min_genes": int(pathway_score_min_genes),
        "n_pathways_analyzed": len(rows),
        "n_pathways_spearman_q_lt_0p05": len(sig_q05),
        "spearman_rho_reference_abs_mean": float(np.mean(abs_rho)) if abs_rho.size else None,
        "pathway_isodepth_spearman_correlations_csv": str(out_csv),
        "top_positive_spearman_pathways": [
            row["pathway_name"]
            for row in sorted(rows, key=lambda r: float(r["spearman_rho_reference"]), reverse=True)[:8]
        ],
        "top_negative_spearman_pathways": [
            row["pathway_name"]
            for row in sorted(rows, key=lambda r: float(r["spearman_rho_reference"]))[:8]
        ],
    }
    summary_path = analysis_dir / "pathway_isodepth_spearman_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["pathway_isodepth_spearman_summary_json"] = str(summary_path)
    return payload
