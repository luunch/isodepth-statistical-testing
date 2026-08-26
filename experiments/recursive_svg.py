"""Recursive spatial gradient discovery.

Each iteration fits a parallel-permutation isodepth on the current gene set,
identifies significant SVGs (F-test + BH q < alpha), removes them, then
repeats on the remaining genes.  Stops when:
  - the permutation p-value >= alpha  (dataset not significant -> no SVGs),
  - no genes pass the F-test threshold,
  - no genes remain, or
  - max_gradients iterations have been completed.

Supports ``parallel_permutation`` with a ``linear`` or ``quadratic`` decoder.
Cell-type mode ``together`` is supported.
Cell-type mode ``separate`` is supported by splitting the dataset by cell
type and running an independent recursive test for each region.
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from analysis.plots import (
    compute_isodepth_sig_genes,
    save_celltype_dataset_plot,
    save_gene_expression_vs_isodepth_plot,
    save_isodepth_triptych,
    save_metric_distribution_plot,
    save_recursive_celltype_isodepth_grid,
    save_recursive_celltype_metric_distribution_grid,
    save_recursive_svg_count_plot,
    save_svg_spatial_expression_plots,
    _save_sig_genes_csv,
)
from data.schemas import DatasetBundle, RunConfig, TestConfig, TestResult
from experiments.configuration import _decoder_df_from_config
from methods.metrics import summarize_metric_distribution
from methods.permutation import run_permutation_method


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subset_dataset(dataset: DatasetBundle, gene_indices: np.ndarray) -> DatasetBundle:
    """Return a new DatasetBundle with only the columns in gene_indices."""
    gene_indices = np.asarray(gene_indices, dtype=np.intp)
    new_A = np.asarray(dataset.A, dtype=np.float32)[:, gene_indices]
    new_meta = dict(dataset.meta)
    var_names = dataset.meta.get("var_names")
    if var_names is not None:
        new_meta["var_names"] = [var_names[i] for i in gene_indices]
    return DatasetBundle(S=dataset.S, A=new_A, meta=new_meta).validate()


def _gene_names_from_meta(meta: dict, n_genes: int) -> list[str]:
    var_names = meta.get("var_names")
    if var_names is not None:
        return [str(v) for v in var_names]
    return [f"gene_{i}" for i in range(n_genes)]


def _safe_celltype_name(type_name: str) -> str:
    return str(type_name).replace(" ", "_").replace("/", "_")


def _standardize_spatial_within_subset(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float32)
    mean = S.mean(axis=0)
    std = S.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return np.asarray((S - mean) / safe_std, dtype=np.float32)


def _empty_svg_info(n_genes: int) -> dict[str, Any]:
    return {
        "sig_indices": np.array([], dtype=np.intp),
        "sig_names": [],
        "pvalues": np.ones(int(n_genes), dtype=np.float64),
        "qvalues": np.ones(int(n_genes), dtype=np.float64),
    }


def _polynomial_predictions(A: np.ndarray, coord: np.ndarray, degree: int) -> np.ndarray:
    """OLS polynomial predictions for every gene against a learned isodepth."""
    A = np.asarray(A, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    degree = int(degree)
    return np.stack(
        [np.poly1d(np.polyfit(coord, A[:, g], degree))(coord) for g in range(A.shape[1])],
        axis=1,
    )


def _json_loop_summary(loop_summary: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in loop_summary.items() if not k.startswith("_")}


def _save_gradient_outputs(
    gradient_idx: int,
    dataset: DatasetBundle,
    result: TestResult,
    svg_info: dict,
    gradient_dir: Path,
    decoder_df: int,
    alpha: float,
    *,
    S_for_plot: np.ndarray | None = None,
) -> dict[str, str]:
    """Save all per-gradient artefacts and return a dict of saved paths."""
    k = gradient_idx
    name = f"gradient_{k}"
    gradient_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    p = save_isodepth_triptych(
        dataset, result,
        gradient_dir / f"{name}_isodepth.png",
    )
    if p is not None:
        paths["isodepth_triptych_plot"] = str(p)

    dist_path = save_metric_distribution_plot(
        result,
        gradient_dir / f"{name}_metric_distribution.png",
    )
    paths["metric_distribution_plot"] = str(dist_path)

    iso_raw = result.artifacts.get("true_isodepth")
    if iso_raw is not None:
        iso_path = gradient_dir / f"{name}_isodepth.npy"
        np.save(iso_path, np.asarray(iso_raw, dtype=np.float32))
        paths["isodepth_npy"] = str(iso_path)

    null_path = gradient_dir / f"{name}_null_distribution.npy"
    np.save(null_path, np.asarray(result.stat_perm, dtype=np.float64))
    paths["null_distribution_npy"] = str(null_path)

    iso_coord = np.asarray(iso_raw, dtype=np.float64).reshape(-1) if iso_raw is not None else None
    if iso_coord is not None and len(svg_info["sig_indices"]) > 0:
        pvalues = np.asarray(svg_info["pvalues"], dtype=np.float64)
        sig_indices = np.asarray(svg_info["sig_indices"], dtype=np.intp)
        sig_order = sig_indices[np.argsort(pvalues[sig_indices])]
        svg_genes_path = gradient_dir / f"{name}_svg_genes.png"
        try:
            fitted = _polynomial_predictions(dataset.A, iso_coord, decoder_df)
            save_gene_expression_vs_isodepth_plot(
                dataset,
                iso_coord,
                svg_genes_path,
                coord_label=f"Gradient {k} Isodepth",
                decoder_preds=fitted,
                decoder_df=None,
                q_threshold=alpha,
                gene_indices=sig_order,
                figure_title=f"Gradient {k} significant SVGs by F-test/BH",
                pvalues=svg_info["pvalues"],
                qvalues=svg_info["qvalues"],
            )
            paths["svg_genes_plot"] = str(svg_genes_path)
        except Exception:
            pass

    if len(svg_info["sig_indices"]) > 0:
        local_gene_names = _gene_names_from_meta(dataset.meta, dataset.n_genes)

        sig_csv_path = gradient_dir / f"{name}_sig_genes.csv"
        _save_sig_genes_csv(
            sig_csv_path,
            local_gene_names,
            svg_info["pvalues"],
            svg_info["qvalues"],
            q_threshold=alpha,
        )
        paths["sig_genes_csv"] = str(sig_csv_path)

        # Spatial expression map: top-5 significant SVGs, cells coloured by
        # individual gene expression at their (x, y) spatial positions.
        # S_for_plot is the original (unstandardised) spatial coordinates when
        # the recursive loop was run on a per-type standardised bundle; falls
        # back to dataset.S when not provided.
        S_spatial = (
            np.asarray(S_for_plot, dtype=np.float32)
            if S_for_plot is not None
            else np.asarray(dataset.S, dtype=np.float32)
        )
        spatial_expr_path = gradient_dir / f"{name}_svg_spatial_expression.png"
        try:
            save_svg_spatial_expression_plots(
                S_spatial,
                dataset.A,
                local_gene_names,
                svg_info["sig_indices"],
                spatial_expr_path,
                pvalues=svg_info["pvalues"],
                qvalues=svg_info["qvalues"],
                expression_meta=dataset.meta,
                suptitle=f"Gradient {k} — Top SVG Spatial Expression",
            )
            paths["svg_spatial_expression_plot"] = str(spatial_expr_path)
        except Exception:
            pass

    result_payload = result.to_json_dict(
        config={"gradient_index": k},
        artifacts={
            **{k2: paths[k2] for k2 in paths},
            "perm_summary": summarize_metric_distribution(result.stat_perm),
            "n_sig_genes": int(len(svg_info["sig_indices"])),
            "sig_gene_names": list(svg_info["sig_names"]),
        },
    )
    result_json_path = gradient_dir / f"{name}_result.json"
    with open(result_json_path, "w", encoding="utf-8") as fh:
        json.dump(result_payload, fh, indent=2)
    paths["result_json"] = str(result_json_path)

    return paths


def _write_combined_sig_genes_csv(
    out_path: Path,
    gradient_entries: list[dict],
) -> None:
    """Write combined_sig_genes.csv with columns: gene, p_value, q_value, corresponding_gradient."""
    rows: list[tuple[str, float, float, int]] = []
    for entry in gradient_entries:
        k = entry["gradient_idx"]
        gnames = entry["gene_names"]
        pvals = entry["pvalues"]
        qvals = entry["qvalues"]
        sig_idx = entry["sig_indices"]
        for i in sorted(sig_idx, key=lambda idx: pvals[idx]):
            rows.append((gnames[i], float(pvals[i]), float(qvals[i]), k))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene", "p_value", "q_value", "corresponding_gradient"])
        writer.writerows(rows)


def _write_combined_celltype_sig_genes_csv(
    out_path: Path,
    region_sig_gene_csvs: list[tuple[str, Path]],
) -> Path | None:
    """Write a run-level SVG table that preserves per-cell-type discoveries."""
    rows: list[tuple[str, str, float, float, int]] = []
    for cell_type, csv_path in region_sig_gene_csvs:
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append((
                    cell_type,
                    str(row["gene"]),
                    float(row["p_value"]),
                    float(row["q_value"]),
                    int(row["corresponding_gradient"]),
                ))

    if not rows:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_type", "gene", "p_value", "q_value", "corresponding_gradient"])
        writer.writerows(rows)
    return out_path


def _run_recursive_loop(
    dataset: DatasetBundle,
    config: TestConfig,
    recursive_dir: Path,
    decoder_df: int,
    alpha: float,
    max_gradients: int,
    label: str = "",
    *,
    include_terminal_result: bool = False,
    collect_plot_entries: bool = False,
    plot_S: np.ndarray | None = None,
) -> dict[str, Any]:
    """Core gradient-peeling loop.

    Runs ``max_gradients`` iterations of: fit isodepth -> identify SVGs -> peel.
    Significant gradients are recorded in ``gradients``. When requested,
    ``tested_gradients`` also includes the terminal non-significant/no-SVG test
    so combined recursive plots show exactly where the loop stopped.
    """
    prefix = f"[recursive_svg{(' ' + label) if label else ''}]"
    total_genes = int(dataset.n_genes)
    all_gene_names = _gene_names_from_meta(dataset.meta, total_genes)
    remaining_global = np.arange(total_genes, dtype=np.intp)

    current_dataset = dataset
    gradient_entries: list[dict] = []
    per_gradient_summaries: list[dict[str, Any]] = []
    tested_gradient_summaries: list[dict[str, Any]] = []
    plot_entries: list[dict[str, Any]] = []
    plot_coords = np.asarray(plot_S, dtype=np.float32) if plot_S is not None else np.asarray(dataset.S, dtype=np.float32)

    for k in range(1, max_gradients + 1):
        print(
            f"\n{prefix} Gradient {k} — {int(current_dataset.n_genes)} genes remaining",
            flush=True,
        )

        result: TestResult = run_permutation_method(current_dataset, config)
        p_value = float(result.p_value)
        print(f"{prefix} Gradient {k} p_value={p_value:.4g}", flush=True)

        iso_raw = result.artifacts.get("true_isodepth")
        iso_coord = np.asarray(iso_raw, dtype=np.float64).reshape(-1) if iso_raw is not None else None
        local_gene_names = _gene_names_from_meta(current_dataset.meta, current_dataset.n_genes)

        passed_permutation = p_value < alpha
        stop_reason: str | None = None
        if not passed_permutation:
            svg_info = _empty_svg_info(current_dataset.n_genes)
            n_sig = 0
            stop_reason = f"p_value {p_value:.4g} >= alpha {alpha}"
            print(f"{prefix} {stop_reason} — stopping.", flush=True)
        else:
            if iso_coord is None:
                raise ValueError("recursive SVG detection requires result.artifacts['true_isodepth']")
            svg_info = compute_isodepth_sig_genes(
                np.asarray(current_dataset.A, dtype=np.float64),
                local_gene_names,
                None,
                decoder_df,
                coord=iso_coord,
                alpha=alpha,
            )
            n_sig = int(len(svg_info["sig_indices"]))
            print(f"{prefix} Gradient {k}: {n_sig} significant SVGs (q < {alpha})", flush=True)
            if n_sig == 0:
                stop_reason = "no significant SVGs"
                print(f"{prefix} No SVGs found — stopping.", flush=True)

        should_save = passed_permutation and n_sig > 0
        should_save_terminal = include_terminal_result and (not passed_permutation or n_sig == 0)
        if not should_save and not should_save_terminal:
            break

        gradient_dir = recursive_dir / f"gradient_{k}"
        artifact_paths = _save_gradient_outputs(
            k, current_dataset, result, svg_info,
            gradient_dir, decoder_df, alpha,
            S_for_plot=plot_coords,
        )

        local_sig = np.asarray(svg_info["sig_indices"], dtype=np.intp)
        global_sig = remaining_global[local_sig] if local_sig.size else np.array([], dtype=np.intp)
        tested_summary = {
            "gradient_index": k,
            "n_genes_in": int(current_dataset.n_genes),
            "p_value": p_value,
            "stat_true": float(result.stat_true),
            "perm_summary": summarize_metric_distribution(result.stat_perm),
            "n_svgs": n_sig,
            "passed_permutation": bool(passed_permutation),
            "stop_reason": stop_reason,
            "svg_genes": [all_gene_names[g] for g in global_sig],
            "artifact_paths": artifact_paths,
        }
        tested_gradient_summaries.append(tested_summary)

        if collect_plot_entries:
            plot_entries.append({
                "gradient_index": k,
                "p_value": p_value,
                "stat_true": float(result.stat_true),
                "stat_perm": np.asarray(result.stat_perm, dtype=np.float64),
                "n_svgs": n_sig,
                "passed_permutation": bool(passed_permutation),
                "stop_reason": stop_reason,
                "true_isodepth": None if iso_raw is None else np.asarray(iso_raw, dtype=np.float32),
                "S_plot": plot_coords,
            })

        if not should_save:
            break

        gradient_entries.append({
            "gradient_idx": k,
            "gene_names": local_gene_names,
            "pvalues": svg_info["pvalues"],
            "qvalues": svg_info["qvalues"],
            "sig_indices": local_sig,
            "global_gene_indices": global_sig,
        })

        per_gradient_summaries.append(tested_summary)

        local_mask = np.ones(int(current_dataset.n_genes), dtype=bool)
        local_mask[local_sig] = False
        remaining_local = np.flatnonzero(local_mask)

        if remaining_local.size == 0:
            print(f"{prefix} All genes assigned — stopping.", flush=True)
            remaining_global = np.array([], dtype=np.intp)
            break

        remaining_global = remaining_global[remaining_local]
        current_dataset = _subset_dataset(current_dataset, remaining_local)

    combined_csv_path = recursive_dir / "combined_sig_genes.csv"
    if gradient_entries:
        _write_combined_sig_genes_csv(combined_csv_path, gradient_entries)

    return {
        "n_gradients_found": len(per_gradient_summaries),
        "n_tested_gradients": len(tested_gradient_summaries),
        "alpha": alpha,
        "max_gradients": max_gradients,
        "decoder_df": decoder_df,
        "total_genes": total_genes,
        "total_svgs": sum(e["n_svgs"] for e in per_gradient_summaries),
        "combined_sig_genes_csv": str(combined_csv_path) if gradient_entries else None,
        "gradients": per_gradient_summaries,
        "tested_gradients": tested_gradient_summaries,
        "_plot_entries": plot_entries,
    }

def _run_recursive_svg_separate(
    dataset: DatasetBundle,
    run_config: RunConfig,
    decoder_df: int,
) -> tuple[dict[str, Any], Path]:
    """Run independent recursive tests for each cell type/region."""
    config = run_config.test
    alpha = float(config.alpha)
    max_gradients = int(config.max_gradients)

    out_root = Path(run_config.output.out_dir)
    out_dir = out_root / run_config.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_type_labels = dataset.meta.get("cell_type_labels")
    cell_type_names = dataset.meta.get("cell_type_names")
    if cell_type_labels is None or cell_type_names is None:
        raise ValueError(
            "data.cell_type='separate' recursive mode requires dataset.meta "
            "to include cell_type_labels and cell_type_names."
        )
    labels = np.asarray(cell_type_labels, dtype=np.int64)
    cell_type_names = [str(name) for name in cell_type_names]
    all_gene_names = _gene_names_from_meta(dataset.meta, int(dataset.n_genes))

    celltype_overview_path = save_celltype_dataset_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_celltype.png",
    )

    # Each region is treated as its own plain recursive run. Any covariate was
    # only meaningful for the old aggregate separate-mode test, so recursive
    # peeling uses the learned coordinate alone.
    plain_config = replace(config, covariate=None)
    per_type_recursive: dict[str, Any] = {}
    per_type_plot_data: dict[str, dict] = {}
    per_type_svg_count_data: dict[str, list[dict[str, Any]]] = {}
    region_sig_gene_csvs: list[tuple[str, Path]] = []

    for type_idx, type_name in enumerate(cell_type_names):
        mask = labels == type_idx
        n_cells = int(mask.sum())
        safe_name = _safe_celltype_name(type_name)
        type_dir = out_dir / safe_name
        type_dir.mkdir(parents=True, exist_ok=True)

        if n_cells == 0:
            per_type_recursive[type_name] = {
                "skipped": True,
                "reason": "no cells for cell type",
            }
            per_type_plot_data[type_name] = {"tested_gradients": []}
            per_type_svg_count_data[type_name] = []
            continue

        print(
            f"\n[recursive_svg] === Cell type: {type_name} ({n_cells} cells) ===",
            flush=True,
        )

        S_plot = np.asarray(dataset.S[mask], dtype=np.float32)
        type_bundle = DatasetBundle(
            S=_standardize_spatial_within_subset(S_plot),
            A=np.asarray(dataset.A[mask], dtype=np.float32),
            meta={"var_names": list(all_gene_names)},
        ).validate()

        loop_summary = _run_recursive_loop(
            type_bundle,
            plain_config,
            type_dir,
            decoder_df,
            alpha,
            max_gradients,
            label=type_name,
            include_terminal_result=True,
            collect_plot_entries=True,
            plot_S=S_plot,
        )
        plot_entries = loop_summary.pop("_plot_entries", [])
        loop_summary["cell_type"] = type_name
        loop_summary["n_cells"] = n_cells

        per_type_svg_count_data[type_name] = list(loop_summary["tested_gradients"])
        svg_count_path = save_recursive_svg_count_plot(
            {type_name: per_type_svg_count_data[type_name]},
            type_dir / "svg_counts_by_gradient.png",
            title=f"{type_name} recursive SVG counts by gradient",
        )
        if svg_count_path is not None:
            loop_summary["svg_count_plot"] = str(svg_count_path)

        recursive_summary_path = type_dir / "recursive_summary.json"
        with open(recursive_summary_path, "w", encoding="utf-8") as fh:
            json.dump(_json_loop_summary(loop_summary), fh, indent=2)

        per_type_recursive[type_name] = {
            "skipped": False,
            "recursive_summary": recursive_summary_path.relative_to(out_dir).as_posix(),
            "n_gradients_found": loop_summary["n_gradients_found"],
            "n_tested_gradients": loop_summary["n_tested_gradients"],
            "total_svgs": loop_summary["total_svgs"],
        }
        if svg_count_path is not None:
            per_type_recursive[type_name]["svg_count_plot"] = svg_count_path.relative_to(out_dir).as_posix()
        combined_sig_genes_csv = loop_summary.get("combined_sig_genes_csv")
        if combined_sig_genes_csv is not None:
            combined_sig_genes_path = Path(combined_sig_genes_csv)
            region_sig_gene_csvs.append((type_name, combined_sig_genes_path))
            per_type_recursive[type_name]["combined_sig_genes_csv"] = (
                combined_sig_genes_path.relative_to(out_dir).as_posix()
            )
        per_type_plot_data[type_name] = {"tested_gradients": plot_entries}

        print(
            f"[recursive_svg] {type_name}: {loop_summary['n_gradients_found']} significant "
            f"gradient(s), {loop_summary['n_tested_gradients']} tested, "
            f"{loop_summary['total_svgs']} SVGs.",
            flush=True,
        )

    combined_isodepth_path = save_recursive_celltype_isodepth_grid(
        per_type_plot_data,
        cell_type_names,
        out_dir / f"{run_config.output.run_name}_combined_isodepths.png",
        full_spatial=dataset.S,
    )
    combined_dist_path = save_recursive_celltype_metric_distribution_grid(
        per_type_plot_data,
        cell_type_names,
        out_dir / f"{run_config.output.run_name}_combined_metric_distribution.png",
        metric=config.metric,
    )
    svg_count_path = save_recursive_svg_count_plot(
        {type_name: per_type_svg_count_data.get(type_name, []) for type_name in cell_type_names},
        out_dir / f"{run_config.output.run_name}_svg_counts_by_gradient.png",
        title="Recursive SVG counts by gradient",
    )
    combined_sig_genes_path = _write_combined_celltype_sig_genes_csv(
        out_dir / f"{run_config.output.run_name}_combined_sig_genes.csv",
        region_sig_gene_csvs,
    )

    top_summary: dict[str, Any] = {
        "run_name": run_config.output.run_name,
        "alpha": alpha,
        "max_gradients": max_gradients,
        "decoder": config.decoder,
        "decoder_df": decoder_df,
        "cell_type_names": cell_type_names,
        "per_type_recursive": per_type_recursive,
    }
    if celltype_overview_path is not None:
        top_summary["celltype_dataset_plot"] = str(celltype_overview_path)
    if combined_isodepth_path is not None:
        top_summary["combined_isodepth_plot"] = str(combined_isodepth_path)
    if combined_dist_path is not None:
        top_summary["combined_metric_distribution_plot"] = str(combined_dist_path)
    if svg_count_path is not None:
        top_summary["svg_count_plot"] = str(svg_count_path)
    if combined_sig_genes_path is not None:
        top_summary["combined_sig_genes_csv"] = str(combined_sig_genes_path)

    summary_path = out_dir / f"{run_config.output.run_name}_celltype_recursive_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(top_summary, fh, indent=2)

    print(
        f"\n[recursive_svg] Done. Saved celltype recursive summary to: {summary_path}",
        flush=True,
    )
    return top_summary, summary_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_recursive_svg(
    dataset: DatasetBundle,
    run_config: RunConfig,
) -> tuple[dict[str, Any], Path]:
    """Run recursive gradient peeling and save all outputs.

    Handles both plain and cell_type='separate' datasets.

    Returns
    -------
    (summary_payload, summary_json_path)
    """
    config = run_config.test
    alpha = float(config.alpha)
    max_gradients = int(config.max_gradients)

    decoder_df = _decoder_df_from_config(getattr(config, "decoder", None))
    if decoder_df is None:
        raise ValueError(
            f"test.recursive requires a parametric decoder ('linear' or 'quadratic'); "
            f"got '{config.decoder}'. The nn decoder does not support recursive SVG detection."
        )

    # --- separate cell-type mode ---
    if run_config.data.cell_type_mode == "separate":
        return _run_recursive_svg_separate(dataset, run_config, decoder_df)

    # --- plain / together mode ---
    out_root = Path(run_config.output.out_dir)
    recursive_dir = out_root / run_config.output.run_name / "recursive"
    recursive_dir.mkdir(parents=True, exist_ok=True)

    loop_summary = _run_recursive_loop(
        dataset, config, recursive_dir, decoder_df, alpha, max_gradients,
    )
    loop_summary.pop("_plot_entries", None)
    svg_count_path = save_recursive_svg_count_plot(
        {"All cells": loop_summary["tested_gradients"]},
        recursive_dir / "svg_counts_by_gradient.png",
    )
    if svg_count_path is not None:
        loop_summary["svg_count_plot"] = str(svg_count_path)

    summary_payload: dict[str, Any] = {
        "run_name": run_config.output.run_name,
        "decoder": config.decoder,
        **loop_summary,
    }
    summary_path = recursive_dir / "recursive_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    print(
        f"\n[recursive_svg] Done. Found {loop_summary['n_gradients_found']} gradient(s). "
        f"Outputs saved to: {recursive_dir}",
        flush=True,
    )
    return summary_payload, summary_path
