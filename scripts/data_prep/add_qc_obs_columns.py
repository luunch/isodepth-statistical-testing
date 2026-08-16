"""Add per-spot QC obs columns (total_counts, log1p_total_counts, n_genes, pct_mt)
to a CalicoST h5ad file, in place.

These columns are required by ``data.covariate_whitening_obs_key`` /
``data.obs_drop_na`` whenever a config whitens on library depth (e.g.
``log1p_total_counts``) in addition to ``calicost_tumor_proportion`` -- unlike
``test.covariate.compute_total_counts_covariate``, the ``loss-difference``
covariate-whitening path reads straight from ``adata.obs`` and does not compute
depth on the fly (see ``data/h5ad_loader.py::load_h5ad_dataset``).

Computed from the raw ``counts`` layer (all genes, before any HVG selection /
normalization), matching the columns already present on
``HT268B1-Th1K3Fc2U1Z1Bs1.h5ad``:
    total_counts        -- sum of raw counts per spot
    log1p_total_counts  -- log1p(total_counts), natural log
    n_genes             -- number of genes with nonzero raw count per spot
    pct_mt              -- percent of raw counts from "MT-" prefixed genes

Usage (from repo root, isodepth_env):
    python -m scripts.data_prep.add_qc_obs_columns data/h5ad/calicost/HT112C1-U1_ST_Bn1.h5ad
    python -m scripts.data_prep.add_qc_obs_columns data/h5ad/calicost/HT112C1-U2_ST_Bn1.h5ad
    python -m scripts.data_prep.add_qc_obs_columns data/h5ad/calicost/HT306P1-S1H1Fc2U1Z1Bs1.h5ad

Pass --dry-run to compute and print summary stats without writing the file.
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
sys.path.insert(0, str(REPO))

QC_COLUMNS = ["total_counts", "log1p_total_counts", "n_genes", "pct_mt"]


def compute_qc_columns(adata: ad.AnnData, layer: str = "counts") -> dict[str, np.ndarray]:
    counts = adata.layers[layer]
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    counts = counts.astype(np.float64)

    total_counts = counts.sum(axis=1)
    n_genes = (counts > 0).sum(axis=1)

    mt_mask = np.asarray(adata.var_names.str.upper().str.startswith("MT-"))
    mt_counts = counts[:, mt_mask].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_mt = np.where(total_counts > 0, 100.0 * mt_counts / total_counts, 0.0)

    return {
        "total_counts": total_counts.astype(np.float32),
        "log1p_total_counts": np.log1p(np.maximum(total_counts, 0.0)).astype(np.float32),
        "n_genes": n_genes.astype(np.float32),
        "pct_mt": pct_mt.astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad_path", type=str)
    parser.add_argument("--layer", type=str, default="counts")
    parser.add_argument("--force", action="store_true", help="overwrite existing QC columns")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.h5ad_path)
    if not path.is_absolute():
        path = REPO / path
    adata = ad.read_h5ad(path)

    existing = [c for c in QC_COLUMNS if c in adata.obs.columns]
    if existing and not args.force:
        raise SystemExit(
            f"{path} already has QC columns {existing}; pass --force to overwrite."
        )

    qc = compute_qc_columns(adata, layer=args.layer)
    for col, values in qc.items():
        adata.obs[col] = values

    print(f"{path.name}: n_obs={adata.n_obs}")
    for col, values in qc.items():
        print(
            f"  {col}: mean={values.mean():.4f} std={values.std():.4f} "
            f"min={values.min():.4f} max={values.max():.4f}"
        )

    if args.dry_run:
        print("(dry run, not writing)")
        return

    adata.write_h5ad(path)
    print(f"Wrote QC columns to {path}")


if __name__ == "__main__":
    main()
