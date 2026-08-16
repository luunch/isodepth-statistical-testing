"""Extract a normal human cortex Visium section from spatialDLPFC for external controls.

Greenwald et al. (*Cell* 2024) use Ravi et al. UKF256_C / UKF265_C normal cortex as a
non-tumor reference, but those samples are not on the Greenwald Zenodo record (only GBM/IDH-mut
tumor sections). Dryad hosting for Ravi 2022 is script-inaccessible (WAF). As a substitute
negative control with the same tissue context (human brain cortex, non-malignant), we export
``Br6522_ant`` from the Lieber Institute spatialDLPFC resource already in this repo.

Output matches ``data/h5ad/external_controls/ZH1007-*.h5ad`` conventions:
raw counts in ``X``, ``obsm['spatial']`` in pixel coordinates, QC obs columns, metadata.

Usage (from repo root, isodepth_env):
    python -m scripts.data_prep.build_dlpfc_normal_cortex_h5ad
    python -m scripts.data_prep.build_dlpfc_normal_cortex_h5ad --sample-id Br6522_mid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from experiments.core.paths import repo_root

REPO = repo_root(__file__)
DEFAULT_SOURCE = REPO / "data/h5ad/dlpfc_new/spatialDLPFC_Visium.h5ad"
DEFAULT_OUT = REPO / "data/h5ad/external_controls/DLPFC-Br6522_ant-normal-cortex.h5ad"


def _compute_qc(adata: ad.AnnData) -> None:
    counts = adata.X
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    counts = counts.astype(np.float64)
    total_counts = counts.sum(axis=1)
    n_genes = (counts > 0).sum(axis=1)
    adata.obs["total_counts"] = total_counts.astype(np.float32)
    adata.obs["log1p_total_counts"] = np.log1p(np.maximum(total_counts, 0.0)).astype(np.float32)
    adata.obs["n_genes"] = n_genes.astype(np.float32)


def build_normal_cortex_h5ad(
    *,
    source_h5ad: Path,
    sample_id: str,
    out_path: Path,
) -> ad.AnnData:
    adata = ad.read_h5ad(source_h5ad)
    if "sample_id" not in adata.obs.columns:
        raise KeyError(f"{source_h5ad} missing obs column 'sample_id'")
    mask = adata.obs["sample_id"].astype(str) == sample_id
    if mask.sum() == 0:
        available = sorted(adata.obs["sample_id"].astype(str).unique().tolist())
        raise ValueError(f"sample_id {sample_id!r} not found; available: {available[:5]}...")

    sub = adata[mask].copy()
    if "gene_name" in sub.var.columns:
        symbols = sub.var["gene_name"].astype(str)
        symbols = symbols.where(symbols.ne("") & symbols.ne("nan"), sub.var_names.astype(str))
        sub.var["ensembl_id"] = sub.var_names.astype(str)
        sub.var_names = symbols.to_numpy()
        sub.var_names_make_unique()
    sub.obs["in_tissue"] = True
    sub.obs["region_label"] = "normal_cortex"
    sub.obs["patient"] = sample_id.split("_")[0]
    sub.obs["cancer_type"] = "normal"
    sub.obs["slice_id"] = sample_id
    if "array_row" not in sub.obs.columns and "row" in sub.obs.columns:
        sub.obs["array_row"] = sub.obs["row"]
    if "array_col" not in sub.obs.columns and "col" in sub.obs.columns:
        sub.obs["array_col"] = sub.obs["col"]
    if "pxl_row" not in sub.obs.columns and "pxl_row_in_fullres" in sub.obs.columns:
        sub.obs["pxl_row"] = sub.obs["pxl_row_in_fullres"]
        sub.obs["pxl_col"] = sub.obs["pxl_col_in_fullres"]

    _compute_qc(sub)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(out_path)
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sample-id", type=str, default="Br6522_ant")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = args.source_h5ad if args.source_h5ad.is_absolute() else REPO / args.source_h5ad
    out = args.out if args.out.is_absolute() else REPO / args.out

    sub = build_normal_cortex_h5ad(source_h5ad=source, sample_id=args.sample_id, out_path=out)
    print(f"Wrote {out} ({sub.n_obs} spots, {sub.n_vars} genes)")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    main()
