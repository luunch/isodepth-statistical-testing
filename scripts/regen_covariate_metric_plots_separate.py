"""Recompute Poisson covariate metrics and regenerate metric-distribution plots.

The covariate fit for ``decoder`` in {``linear``, ``quadratic``} with
``metric=nll_poisson_mse`` was switched from gradient descent to IRLS
(``fit_poisson_glm_irls``).  Runs completed before that fix can have
``stat_covariate`` values far above the null (``p_value_covariate`` ≈ 1.0)
even though the isodepth permutation stats are fine.

This script reloads per-cell-type expression/coordinates from the h5ad,
re-fits the covariate with the current code, updates ``per_type_summaries``
in the saved result JSON, and rewrites the per-type and combined metric
distribution PNGs.  It does **not** rerun the isodepth permutation test.

Usage::

    python scripts/regen_covariate_metric_plots_separate.py \\
        configs/hypothalamus/hypothalamus_existence_separatecell_poisson.json \\
        results/hypothalamus_existence_quadratic_separatecell_poisson/hypothalamus_existence_quadratic_separatecell_poisson_result.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from analysis.plots import save_combined_celltype_metric_distribution, save_metric_distribution_plot
from data import load_dataset
from data.schemas import TestResult, run_config_from_mapping
from experiments.configuration import load_json_config, summarize_metric_distribution
from methods.permutation import _train_covariate_artifacts


def _standardize_coords(S: np.ndarray) -> np.ndarray:
    mean = S.mean(axis=0)
    std = S.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return np.asarray((S - mean) / safe_std, dtype=np.float32)


def main(config_path: str, result_json_path: str) -> None:
    run_cfg = run_config_from_mapping(load_json_config(config_path))
    if run_cfg.test.covariate is None:
        sys.exit("ERROR: config has no covariate — nothing to recompute.")

    result_path = Path(result_json_path)
    out_root = result_path.parent
    with open(result_path, encoding="utf-8") as f:
        saved = json.load(f)

    arts = saved.get("artifacts", {})
    cell_type_names: list[str] = list(arts.get("cell_type_names", []))
    summaries: dict[str, dict] = dict(arts.get("per_type_summaries", {}))
    if not cell_type_names or not summaries:
        sys.exit("ERROR: result JSON is missing separate cell-type summaries.")

    metric = saved.get("metric", run_cfg.test.metric)
    device = torch.device("cpu")

    print(f"Loading dataset ({run_cfg.data.h5ad}) …")
    dataset = load_dataset(run_cfg.data, covariate=run_cfg.test.covariate)
    labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    names = list(dataset.meta["cell_type_names"])

    per_type_results: dict[str, dict] = {}

    for type_name in cell_type_names:
        if type_name not in summaries:
            print(f"  SKIP {type_name}: not in per_type_summaries")
            continue
        type_summary = summaries[type_name]
        stat_perm = np.asarray(type_summary["stat_perm"], dtype=np.float64)

        idx = names.index(type_name)
        mask = labels == idx
        S_c = _standardize_coords(np.asarray(dataset.S[mask], dtype=np.float32))
        A_c = np.asarray(dataset.A[mask], dtype=np.float32)

        cov_art = _train_covariate_artifacts(
            S_c,
            A_c,
            run_cfg.test,
            device,
            metric,
            stat_perm,
            model_label=f"covariate regen ({type_name})",
        )
        if "stat_covariate" not in cov_art:
            print(f"  WARN {type_name}: covariate fit returned no stat")
            continue

        old_cov = type_summary.get("stat_covariate")
        new_cov = float(cov_art["stat_covariate"])
        new_p_cov = float(cov_art["p_value_covariate"])
        print(
            f"  {type_name}: stat_covariate {old_cov:.4g} -> {new_cov:.4g}, "
            f"p (cov) -> {new_p_cov:.4g}"
        )

        type_summary["stat_covariate"] = new_cov
        type_summary["p_value_covariate"] = new_p_cov
        summaries[type_name] = type_summary

        per_type_results[type_name] = {
            "stat_perm": stat_perm,
            "stat_true": float(type_summary["stat_true"]),
            "p_value": float(type_summary["p_value"]),
            "n_cells": int(type_summary["n_cells"]),
            "stat_covariate": new_cov,
            "p_value_covariate": new_p_cov,
        }

        safe_name = type_name.replace(" ", "_").replace("/", "_")
        type_dir = out_root / safe_name
        type_dir.mkdir(parents=True, exist_ok=True)
        subset_result = TestResult(
            method_name="parallel_permutation",
            metric=metric,
            p_value=float(type_summary["p_value"]),
            stat_true=float(type_summary["stat_true"]),
            stat_perm=stat_perm,
            runtime_sec=0.0,
            n_cells=int(type_summary["n_cells"]),
            n_genes=int(A_c.shape[1]),
            config={},
            artifacts={
                "stat_covariate": new_cov,
                "p_value_covariate": new_p_cov,
            },
        ).validate()
        plot_path = save_metric_distribution_plot(
            subset_result,
            type_dir / f"{safe_name}_metric_distribution.png",
        )
        apaths = dict(type_summary.get("artifact_paths", {}))
        apaths["metric_distribution_plot"] = str(plot_path)
        type_summary["artifact_paths"] = apaths

    run_name = out_root.name
    combined_path = save_combined_celltype_metric_distribution(
        per_type_results,
        cell_type_names,
        out_root / f"{run_name}_combined_metric_distribution.png",
        metric=metric,
    )
    print(f"Wrote combined plot: {combined_path}")

    arts["per_type_summaries"] = summaries
    arts["combined_metric_distribution_plot"] = str(combined_path)
    saved["artifacts"] = arts

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    print(f"Updated result JSON: {result_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
