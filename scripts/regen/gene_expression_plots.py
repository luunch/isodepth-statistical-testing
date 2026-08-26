"""Regenerate gene expression vs coordinates summary plots from saved results.

Usage:
    python scripts/regen_gene_expression_plots.py configs/hypothalamus_existence.json \
        results/hypothalamus_existence_one_perm/hypothalamus_existence_one_perm_result.json

The script loads the dataset using the config, reads the saved isodepth arrays
from the result JSON, and re-generates the comparison plot (or isodepth-only
plot when no covariate was used) without re-running any training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# This script lives in scripts/; add the project root so package imports work
# when invoked as `python scripts/<name>.py`.
from experiments.core.paths import repo_root
sys.path.insert(0, str(repo_root(__file__)))

import numpy as np

from data.schemas import DataConfig
from data import load_dataset
from experiments.configuration import (
    load_json_config,
    _decoder_df_from_config,
    _dataset_for_gene_expression_plots,
)
from analysis.plots import (
    save_gene_expression_vs_coordinates_comparison,
    save_gene_expression_vs_isodepth_plot,
)


def main(config_path: str, result_json_path: str) -> None:
    cfg = load_json_config(config_path)
    data_cfg = DataConfig(**cfg["data"])
    out_cfg = cfg.get("output", {})
    run_name = out_cfg.get("run_name", Path(result_json_path).stem.replace("_result", ""))
    out_dir = Path(result_json_path).parent

    print(f"Loading dataset from {data_cfg.h5ad} …", flush=True)
    # covariate=None: midline is computed from coords (not loaded from h5ad),
    # so no obs-column loading is required to reconstruct expression arrays.
    dataset = load_dataset(data_cfg)
    plot_dataset = _dataset_for_gene_expression_plots(dataset)
    print(f"  {dataset.A.shape[0]} cells × {dataset.A.shape[1]} genes", flush=True)
    if plot_dataset is not dataset:
        print("  using cell-type expression residuals for gene-expression plots", flush=True)

    with open(result_json_path) as f:
        saved = json.load(f)

    artifacts = saved.get("artifacts", {})
    iso_raw = artifacts.get("true_isodepth")
    if iso_raw is None:
        sys.exit("ERROR: 'true_isodepth' not found in result JSON artifacts.")

    iso = np.asarray(iso_raw, dtype=np.float64).reshape(-1)
    cov_raw = artifacts.get("true_isodepth_covariate")

    # Load decoder predictions if saved.  pred_true: isodepth-model decoder;
    # pred_true_covariate: covariate-model decoder.  Both shape (n_cells, G).
    pred_iso_raw = artifacts.get("pred_true")
    pred_cov_raw = artifacts.get("pred_true_covariate")
    pred_iso = np.asarray(pred_iso_raw, dtype=np.float64) if pred_iso_raw is not None else None
    pred_cov = np.asarray(pred_cov_raw, dtype=np.float64) if pred_cov_raw is not None else None

    decoder_type = cfg.get("test", {}).get("decoder", None)
    decoder_df = _decoder_df_from_config(decoder_type)

    if pred_iso is not None:
        print(f"  pred_true (isodepth decoder) loaded, shape {pred_iso.shape}", flush=True)
    else:
        print("  pred_true not in JSON — using polynomial fit through bin means", flush=True)
    if pred_cov is not None:
        print(f"  pred_true_covariate (covariate decoder) loaded, shape {pred_cov.shape}", flush=True)
    if decoder_df is not None:
        print(f"  decoder='{decoder_type}' → F-test df_model={decoder_df}, writing sig-gene CSVs", flush=True)

    if cov_raw is not None:
        cov = np.asarray(cov_raw, dtype=np.float64).reshape(-1)
        out_path = out_dir / f"{run_name}_gene_expression_vs_coordinates.png"
        print(f"Generating 4-row comparison plot → {out_path}", flush=True)
        covariate_str = cfg.get("test", {}).get("covariate", "covariate")
        covariate_label = "Midline" if covariate_str == "midline" else str(covariate_str)
        save_gene_expression_vs_coordinates_comparison(
            plot_dataset, iso, cov, out_path,
            isodepth_label="Isodepth",
            covariate_label=covariate_label,
            pred_isodepth=pred_iso,
            pred_covariate=pred_cov,
            decoder_df=decoder_df,
        )
    else:
        out_path = out_dir / f"{run_name}_gene_expression_vs_isodepth.png"
        print(f"Generating isodepth-only plot → {out_path}", flush=True)
        save_gene_expression_vs_isodepth_plot(
            plot_dataset, iso, out_path,
            decoder_preds=pred_iso,
            decoder_df=decoder_df,
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
