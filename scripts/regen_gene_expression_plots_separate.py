"""Regenerate per-cell-type gene expression vs coordinate plots.

Reads the saved ``{type_name}_isodepths.npz`` files (written automatically by
the pipeline for every future run).  Falls back to reloading the h5ad when
the NPZ files are absent (legacy runs).

Usage:
    python regen_gene_expression_plots_separate.py \\
        configs/mouse_hippocampus_separate.json \\
        results/hippocampus_separate/hippocampus_separate_result.json

The script iterates over every cell type found in the result JSON and
generates (or re-generates) ``{type_name}_gene_expression_vs_coordinates.png``
in each per-type subdirectory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from data.schemas import DataConfig
from data import load_dataset
from experiments.configuration import load_json_config
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
) -> dict[str, np.ndarray]:
    """Load the full dataset and subset A per cell type."""
    print(f"  Loading dataset from {h5ad_path} for expression sub-setting …")
    from data.schemas import CovariateConfig
    cov_obj = None
    if covariate and covariate != "midline":
        try:
            cov_obj = CovariateConfig(type=covariate)
        except Exception:
            pass
    dataset = load_dataset(data_cfg, covariate=cov_obj)
    meta = dataset.meta
    cell_type_labels = np.asarray(meta.get("cell_type_labels", []), dtype=np.int64)
    names: list[str] = list(meta.get("cell_type_names", []))
    out: dict[str, np.ndarray] = {}
    for ct in cell_type_names:
        if ct not in names:
            continue
        idx = names.index(ct)
        mask = cell_type_labels == idx
        out[ct] = np.asarray(dataset.A[mask], dtype=np.float32)
    return out


def main(config_path: str, result_json_path: str) -> None:
    cfg = load_json_config(config_path)
    data_cfg = DataConfig(**cfg["data"])
    covariate_str = cfg.get("test", {}).get("covariate", None)
    cov_lbl = _covariate_label(cfg)
    out_root = Path(result_json_path).parent

    with open(result_json_path) as f:
        saved = json.load(f)

    arts = saved.get("artifacts", {})
    cell_type_names: list[str] = arts.get("cell_type_names", [])
    if not cell_type_names:
        sys.exit("ERROR: 'cell_type_names' not found in result JSON — is this a separate cell-type run?")

    per_type_summaries = arts.get("per_type_summaries", {})

    # Try to load expression per type from NPZ (written by new pipeline runs).
    # If absent, fall back to h5ad reload.
    npz_cache: dict[str, np.ndarray] = {}
    iso_cache: dict[str, np.ndarray] = {}
    cov_cache: dict[str, np.ndarray] = {}

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
        else:
            print(f"  No NPZ for {ct} — will reload dataset from h5ad (legacy run).")
            needs_h5ad = True

    if needs_h5ad:
        print("Reloading full dataset from h5ad to extract per-type expression …")
        expr_by_type = _load_per_type_expression(
            data_cfg.h5ad, data_cfg, cell_type_names, covariate_str,
        )
    else:
        expr_by_type = {}

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

        # Build a minimal DatasetBundle for the plot functions.
        from data.schemas import DatasetBundle
        dummy_S = np.zeros((A.shape[0], 2), dtype=np.float32)
        bundle = DatasetBundle(S=dummy_S, A=A, meta={}).validate()

        cov = cov_cache.get(ct)
        if cov is not None:
            out_path = type_dir / f"{safe_name}_gene_expression_vs_coordinates.png"
            print(f"Generating 4-row plot for {ct} → {out_path}")
            save_gene_expression_vs_coordinates_comparison(
                bundle, iso, cov, out_path,
                isodepth_label="Isodepth",
                covariate_label=cov_lbl,
            )
        else:
            out_path = type_dir / f"{safe_name}_gene_expression_vs_isodepth.png"
            print(f"Generating isodepth-only plot for {ct} → {out_path}")
            save_gene_expression_vs_isodepth_plot(bundle, iso, out_path)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
