"""Per-batch SpotSweeper local-outlier QC for the JFan MERFISH U2OS dataset.

Motivation (see agent.md, 2026-09-04 "JFan per-batch 499-perm rerun" /
significance-vs-batch-density investigation): each of B1/B2/B3 has one
standout low-total-count cell whose normalize_total-rescaled expression
profile is 3-5x more extreme (in z-scored mean-abs-expression) than the next
most extreme cell in that batch. This is a classic sparse-cell normalization
artifact, not necessarily biology. This script runs SpotSweeper's local,
neighbor-relative MAD outlier detector (the same `spotsweeper.local_outliers`
routine already used for Visium QC in
`scripts/posthoc/calicost_cohort_survey.py`) on each batch's `n_counts` /
`n_genes`, using each cell's own spatial neighbors as the local reference
distribution (so a "low count" cell is only flagged if it's low *relative to
its neighbors*, not relative to the whole batch).

Outputs, per batch, a `.npy` of GLOBAL positional indices (into the full,
unfiltered h5ad) of the cells that PASS QC. Point `data.obs_indices` at that
file directly (do not also set `data.obs_filters.batch` -- the two are
mutually exclusive in DataConfig.validate(), and the saved indices already
restrict to the batch).

Usage (repo root, isodepth_env):
    python -m scripts.data_prep.jfan_merfish_spotsweeper_qc
    python -m scripts.data_prep.jfan_merfish_spotsweeper_qc --n-neighbors 20 --cutoff 3.0 --plot

Writes:
    data/qc/jfan_merfish/jfan_merfish_spotsweeper_keep_B1.npy (etc. for B2, B3)
    data/qc/jfan_merfish/jfan_merfish_spotsweeper_summary.csv
    data/qc/jfan_merfish/jfan_merfish_spotsweeper_qc.png   (if --plot)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from experiments.core.paths import repo_root

REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

DEFAULT_H5AD = "data/h5ad/merfish_xia2019_u2os.h5ad"
DEFAULT_BATCHES = ["B1", "B2", "B3"]
DEFAULT_OUT_DIR = "data/qc/jfan_merfish"

# (obs column to feed local_outliers, direction, output column name)
QC_METRICS = [
    ("n_counts", "lower", "n_counts_outliers"),
    ("n_genes", "lower", "n_genes_outliers"),
]


def _run_local_outliers(
    coords: np.ndarray,
    metrics: dict[str, np.ndarray],
    *,
    batch: str,
    n_neighbors: int,
    cutoff: float,
) -> dict[str, np.ndarray]:
    """Wrap `spotsweeper.local_outliers` for one batch's cells.

    Builds a throwaway AnnData (SpotSweeper's API operates on AnnData.obs /
    obsm) purely to reuse the tested local_outliers implementation, matching
    the pattern in scripts/posthoc/calicost_cohort_survey.py.
    """
    import spotsweeper.local_outliers as lo

    n = coords.shape[0]
    k = min(n_neighbors, n - 1)
    if k < n_neighbors:
        print(f"  [warn] {batch}: only {n} cells; reducing n_neighbors {n_neighbors} -> {k}")

    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["spatial"] = np.asarray(coords, dtype=np.float64)
    adata.obs["sample_id"] = batch
    for col, values in metrics.items():
        adata.obs[col] = np.asarray(values, dtype=np.float64)

    per_metric: dict[str, np.ndarray] = {}
    for metric, direction, out_col in QC_METRICS:
        lo.local_outliers(
            adata,
            metric=metric,
            direction=direction,
            n_neighbors=k,
            sample_key="sample_id",
            log=True,
            cutoff=cutoff,
            coord_key="spatial",
        )
        per_metric[out_col] = adata.obs[out_col].to_numpy(dtype=bool)
    return per_metric


def _plot_qc(
    adata: ad.AnnData,
    batches: list[str],
    discard_by_batch: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4.5))
    axes = np.atleast_1d(axes)
    for ax, batch in zip(axes, batches):
        mask = (adata.obs["batch"] == batch).to_numpy()
        xy = adata.obsm["spatial"][mask]
        discard = discard_by_batch[batch]
        keep = ~discard
        ax.scatter(xy[keep, 0], xy[keep, 1], s=10, c="#4c78a8", alpha=0.6, linewidths=0, label="keep")
        if discard.any():
            ax.scatter(
                xy[discard, 0], xy[discard, 1], s=40, c="crimson", marker="x", linewidths=1.5,
                label=f"SpotSweeper discard (n={int(discard.sum())})",
            )
        ax.set_title(f"{batch} (n={mask.sum()})")
        ax.set_xlabel("x (\u00b5m)")
        ax.set_ylabel("y (\u00b5m)")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("JFan MERFISH SpotSweeper local-outlier QC (n_counts + n_genes, lower-tail)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", type=str, default=DEFAULT_H5AD)
    parser.add_argument("--batches", type=str, default=",".join(DEFAULT_BATCHES))
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad)
    if not h5ad_path.is_absolute():
        h5ad_path = REPO / h5ad_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = [b.strip() for b in args.batches.split(",") if b.strip()]

    adata = ad.read_h5ad(h5ad_path)
    if "batch" not in adata.obs.columns:
        raise SystemExit(f"{h5ad_path} has no obs['batch'] column")

    rows = []
    discard_by_batch: dict[str, np.ndarray] = {}
    for batch in batches:
        mask = (adata.obs["batch"] == batch).to_numpy()
        global_idx = np.where(mask)[0]
        if global_idx.size == 0:
            raise SystemExit(f"No cells found for batch={batch!r}")

        sub = adata[mask]
        coords = sub.obsm["spatial"]
        metrics = {
            "n_counts": sub.obs["n_counts"].to_numpy(dtype=np.float64),
            "n_genes": sub.obs["n_genes"].to_numpy(dtype=np.float64),
        }
        per_metric = _run_local_outliers(
            coords, metrics, batch=batch, n_neighbors=args.n_neighbors, cutoff=args.cutoff
        )
        combined_discard = np.zeros(global_idx.size, dtype=bool)
        for flags in per_metric.values():
            combined_discard |= flags
        discard_by_batch[batch] = combined_discard

        keep_global_idx = np.sort(global_idx[~combined_discard])
        out_path = out_dir / f"jfan_merfish_spotsweeper_keep_{batch}.npy"
        np.save(out_path, keep_global_idx.astype(np.int64))

        discarded_cell_ids = sub.obs["cell_id"].to_numpy()[combined_discard].tolist() if "cell_id" in sub.obs else []
        print(
            f"[{batch}] n={global_idx.size} discarded={int(combined_discard.sum())} "
            f"(n_counts_outliers={int(per_metric['n_counts_outliers'].sum())}, "
            f"n_genes_outliers={int(per_metric['n_genes_outliers'].sum())}) "
            f"-> wrote {out_path} ({keep_global_idx.size} kept)"
        )
        if discarded_cell_ids:
            print(f"  discarded cell_id(s): {discarded_cell_ids}")

        rows.append(
            {
                "batch": batch,
                "n_cells": int(global_idx.size),
                "n_discarded": int(combined_discard.sum()),
                "n_kept": int(keep_global_idx.size),
                "n_counts_outliers": int(per_metric["n_counts_outliers"].sum()),
                "n_genes_outliers": int(per_metric["n_genes_outliers"].sum()),
                "n_neighbors": min(args.n_neighbors, global_idx.size - 1),
                "cutoff": args.cutoff,
                "keep_indices_path": str(out_path.relative_to(REPO)),
            }
        )

    summary_path = out_dir / "jfan_merfish_spotsweeper_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[summary] wrote {summary_path}")

    if args.plot:
        _plot_qc(adata, batches, discard_by_batch, out_dir / "jfan_merfish_spotsweeper_qc.png")


if __name__ == "__main__":
    main()
