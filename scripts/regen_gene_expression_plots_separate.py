"""Regenerate per-cell-type gene expression vs coordinate plots.

Reads the saved ``{type_name}_isodepths.npz`` files (written automatically by
the pipeline for every future run).  Falls back to reloading the h5ad when
the NPZ files are absent (legacy runs).

Usage:
    python scripts/regen_gene_expression_plots_separate.py \\
        configs/dlpfc_new/dlpfc_poisson.json \\
        results/dlpfc_new/poisson_1000_genes/poisson_1000_genes_result.json

The script iterates over every cell type found in the result JSON and
generates (or re-generates) ``{type_name}_gene_expression_vs_coordinates.png``
in each per-type subdirectory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# This script lives in scripts/; add the project root so package imports work
# when invoked as `python scripts/<name>.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data.schemas import DataConfig, DatasetBundle
from data import load_dataset
from experiments.configuration import load_json_config, _decoder_df_from_config
from analysis.plots import (
    save_gene_expression_vs_coordinates_comparison,
    save_gene_expression_vs_isodepth_plot,
)


def _covariate_label(cfg: dict) -> str:
    cov = cfg.get("test", {}).get("covariate")
    if not cov:
        return "Covariate"
    if isinstance(cov, str):
        return "Midline" if cov == "midline" else cov.capitalize()
    return str(cov).capitalize()


def _load_per_type_expression(
    h5ad_path: str,
    data_cfg: DataConfig,
    cell_type_names: list[str],
    covariate: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Load the full dataset and subset A/S per cell type."""
    print(f"  Loading dataset from {h5ad_path} for expression sub-setting …")
    from data.schemas import CovariateConfig

    cov_obj = None
    if covariate and covariate != "midline":
        try:
            cov_obj = CovariateConfig(type=covariate)
        except Exception:
            pass
    dataset = load_dataset(data_cfg, covariate=cov_obj)
    meta = dict(dataset.meta)
    cell_type_labels = np.asarray(meta.get("cell_type_labels", []), dtype=np.int64)
    names: list[str] = list(meta.get("cell_type_names", []))
    expr_by_type: dict[str, np.ndarray] = {}
    spatial_by_type: dict[str, np.ndarray] = {}
    for ct in cell_type_names:
        if ct not in names:
            continue
        idx = names.index(ct)
        mask = cell_type_labels == idx
        expr_by_type[ct] = np.asarray(dataset.A[mask], dtype=np.float32)
        spatial_by_type[ct] = np.asarray(dataset.S[mask], dtype=np.float32)
    return expr_by_type, spatial_by_type, meta


def _subset_meta(dataset_meta: dict) -> dict:
    subset_meta = dict(dataset_meta)
    for key in ("cell_type_labels", "cell_type_names", "n_cell_types", "cell_type_mode"):
        subset_meta.pop(key, None)
    return subset_meta


def main(config_path: str, result_json_path: str) -> None:
    cfg = load_json_config(config_path)
    data_cfg = DataConfig(**cfg["data"])
    covariate_str = cfg.get("test", {}).get("covariate", None)
    cov_lbl = _covariate_label(cfg)
    decoder_df = _decoder_df_from_config(cfg.get("test", {}).get("decoder", None))
    out_root = Path(result_json_path).parent

    with open(result_json_path) as f:
        saved = json.load(f)

    arts = saved.get("artifacts", {})
    cell_type_names: list[str] = arts.get("cell_type_names", [])
    if not cell_type_names:
        sys.exit("ERROR: 'cell_type_names' not found in result JSON — is this a separate cell-type run?")

    npz_cache: dict[str, np.ndarray] = {}
    iso_cache: dict[str, np.ndarray] = {}
    cov_cache: dict[str, np.ndarray] = {}
    pred_cache: dict[str, np.ndarray] = {}
    pred_cov_cache: dict[str, np.ndarray] = {}
    spatial_cache: dict[str, np.ndarray] = {}

    needs_h5ad = False
    for ct in cell_type_names:
        safe_name = ct.replace(" ", "_").replace("/", "_")
        type_dir = out_root / safe_name
        npz_path = type_dir / f"{safe_name}_isodepths.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=False)
            iso_cache[ct] = data["true_isodepth"].astype(np.float64).reshape(-1)
            if "true_isodepth_covariate" in data:
                cov_cache[ct] = data["true_isodepth_covariate"].astype(np.float64).reshape(-1)
            if "A" in data:
                npz_cache[ct] = data["A"].astype(np.float32)
            if "S" in data:
                spatial_cache[ct] = data["S"].astype(np.float32)
            if "pred_true" in data:
                pred_cache[ct] = data["pred_true"].astype(np.float32)
            if "pred_true_covariate" in data:
                pred_cov_cache[ct] = data["pred_true_covariate"].astype(np.float32)
        else:
            print(f"  No NPZ for {ct} — will reload dataset from h5ad (legacy run).")
            needs_h5ad = True

    dataset_meta: dict = dict(saved.get("artifacts", {}).get("dataset_meta", {}))
    expr_by_type: dict[str, np.ndarray] = {}
    if needs_h5ad:
        print("Reloading full dataset from h5ad to extract per-type expression …")
        expr_by_type, spatial_by_type, dataset_meta = _load_per_type_expression(
            data_cfg.h5ad, data_cfg, cell_type_names, covariate_str,
        )
        spatial_cache.update(spatial_by_type)
    elif not dataset_meta:
        _, _, dataset_meta = _load_per_type_expression(
            data_cfg.h5ad, data_cfg, cell_type_names, covariate_str,
        )

    subset_meta = _subset_meta(dataset_meta)

    for ct in cell_type_names:
        safe_name = ct.replace(" ", "_").replace("/", "_")
        type_dir = out_root / safe_name

        iso = iso_cache.get(ct)
        if iso is None:
            print(f"  Skipping {ct}: no isodepth data available.")
            continue

        A = npz_cache.get(ct, expr_by_type.get(ct))
        if A is None:
            print(f"  Skipping {ct}: no expression data available.")
            continue

        S = spatial_cache.get(ct)
        if S is None:
            S = np.zeros((A.shape[0], 2), dtype=np.float32)

        bundle = DatasetBundle(S=S, A=A, meta=subset_meta).validate()
        pred_iso = pred_cache.get(ct)
        pred_cov = pred_cov_cache.get(ct)
        cov = cov_cache.get(ct)

        if cov is not None:
            out_path = type_dir / f"{safe_name}_gene_expression_vs_coordinates.png"
            print(f"Generating 4-row plot for {ct} → {out_path}")
            save_gene_expression_vs_coordinates_comparison(
                bundle, iso, cov, out_path,
                isodepth_label="Isodepth",
                covariate_label=cov_lbl,
                pred_isodepth=pred_iso,
                pred_covariate=pred_cov,
                decoder_df=decoder_df,
                spatial_S=S,
            )
        else:
            out_path = type_dir / f"{safe_name}_gene_expression_vs_isodepth.png"
            print(f"Generating isodepth-only plot for {ct} → {out_path}")
            save_gene_expression_vs_isodepth_plot(
                bundle, iso, out_path,
                decoder_preds=pred_iso,
                decoder_df=decoder_df,
                spatial_S=S,
            )

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
