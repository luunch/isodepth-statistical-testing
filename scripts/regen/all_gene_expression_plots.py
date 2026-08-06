"""Regenerate gene-expression summary plots for every saved result JSON.

Walks ``results/**/*_result.json``, reloads the dataset and saved isodepths,
and rewrites gene-expression PNGs (including spatial SVG companion plots).

Usage:
    python scripts/regen_all_gene_expression_plots.py
    python scripts/regen_all_gene_expression_plots.py --results-root results/synthetic
    python scripts/regen_all_gene_expression_plots.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.core.paths import repo_root
REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

import numpy as np

from analysis.plots import (
    save_gene_expression_vs_coordinates_comparison,
    save_gene_expression_vs_isodepth_plot,
)
from data import load_dataset
from data.schemas import CovariateConfig, DataConfig, DatasetBundle
from experiments.configuration import (
    _dataset_for_gene_expression_plots,
    _decoder_df_from_config,
)


def _covariate_label(cfg: dict) -> str:
    cov = cfg.get("test", {}).get("covariate")
    if not cov:
        return "Covariate"
    if isinstance(cov, str):
        return "Midline" if cov == "midline" else cov.capitalize()
    return str(cov).capitalize()


def _subset_meta(dataset_meta: dict) -> dict:
    subset_meta = dict(dataset_meta)
    for key in ("cell_type_labels", "cell_type_names", "n_cell_types", "cell_type_mode"):
        subset_meta.pop(key, None)
    return subset_meta


def _load_covariate(cfg: dict) -> CovariateConfig | None:
    covariate_str = cfg.get("test", {}).get("covariate")
    if not covariate_str or covariate_str == "midline":
        return None
    try:
        return CovariateConfig(type=covariate_str)
    except Exception:
        return None


def _load_per_type_expression(
    data_cfg: DataConfig,
    cell_type_names: list[str],
    covariate: CovariateConfig | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    dataset = load_dataset(data_cfg, covariate=covariate)
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


def regen_separate_celltype_plots(result_path: Path, *, dry_run: bool = False) -> str:
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = saved.get("config") or {}
    if not cfg.get("data"):
        return "skip: no data config"

    arts = saved.get("artifacts", {})
    cell_type_names: list[str] = arts.get("cell_type_names", [])
    if not cell_type_names:
        return "skip: no cell_type_names"

    data_cfg = DataConfig(**cfg["data"])
    covariate = _load_covariate(cfg)
    cov_lbl = _covariate_label(cfg)
    decoder_df = _decoder_df_from_config(cfg.get("test", {}).get("decoder"))
    out_root = result_path.parent

    npz_cache: dict[str, np.ndarray] = {}
    iso_cache: dict[str, np.ndarray] = {}
    cov_cache: dict[str, np.ndarray] = {}
    pred_cache: dict[str, np.ndarray] = {}
    pred_cov_cache: dict[str, np.ndarray] = {}
    spatial_cache: dict[str, np.ndarray] = {}

    needs_h5ad = False
    for ct in cell_type_names:
        safe_name = ct.replace(" ", "_").replace("/", "_")
        npz_path = out_root / safe_name / f"{safe_name}_isodepths.npz"
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
            needs_h5ad = True

    dataset_meta: dict = dict(arts.get("dataset_meta", {}))
    expr_by_type: dict[str, np.ndarray] = {}
    if needs_h5ad or not dataset_meta:
        expr_by_type, spatial_by_type, dataset_meta = _load_per_type_expression(
            data_cfg, cell_type_names, covariate,
        )
        spatial_cache.update(spatial_by_type)

    subset_meta = _subset_meta(dataset_meta)
    regenerated = 0

    for ct in cell_type_names:
        safe_name = ct.replace(" ", "_").replace("/", "_")
        type_dir = out_root / safe_name
        iso = iso_cache.get(ct)
        if iso is None:
            continue
        A = npz_cache.get(ct, expr_by_type.get(ct))
        if A is None:
            continue
        S = spatial_cache.get(ct)
        if S is None:
            S = np.zeros((A.shape[0], 2), dtype=np.float32)

        bundle = DatasetBundle(S=S, A=A, meta=subset_meta).validate()
        pred_iso = pred_cache.get(ct)
        pred_cov = pred_cov_cache.get(ct)
        cov = cov_cache.get(ct)

        if dry_run:
            regenerated += 1
            continue

        if cov is not None:
            out_path = type_dir / f"{safe_name}_gene_expression_vs_coordinates.png"
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
            save_gene_expression_vs_isodepth_plot(
                bundle, iso, out_path,
                decoder_preds=pred_iso,
                decoder_df=decoder_df,
                spatial_S=S,
            )
        regenerated += 1

    return f"ok: {regenerated} cell types"


def regen_standard_plots(result_path: Path, *, dry_run: bool = False) -> str:
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = saved.get("config") or {}
    if not cfg.get("data"):
        return "skip: no data config"

    arts = saved.get("artifacts", {})
    iso_raw = arts.get("true_isodepth")
    if iso_raw is None:
        return "skip: no true_isodepth"

    data_cfg = DataConfig(**cfg["data"])
    covariate = _load_covariate(cfg)
    run_name = result_path.stem.replace("_result", "")
    out_dir = result_path.parent

    if dry_run:
        return "ok: dry-run"

    dataset = load_dataset(data_cfg, covariate=covariate)
    plot_dataset = _dataset_for_gene_expression_plots(dataset)
    iso = np.asarray(iso_raw, dtype=np.float64).reshape(-1)

    pred_iso_raw = arts.get("pred_true")
    pred_cov_raw = arts.get("pred_true_covariate")
    pred_iso = np.asarray(pred_iso_raw, dtype=np.float64) if pred_iso_raw is not None else None
    pred_cov = np.asarray(pred_cov_raw, dtype=np.float64) if pred_cov_raw is not None else None
    decoder_df = _decoder_df_from_config(cfg.get("test", {}).get("decoder"))

    cov_raw = arts.get("true_isodepth_covariate")
    if cov_raw is not None:
        cov = np.asarray(cov_raw, dtype=np.float64).reshape(-1)
        covariate_label = _covariate_label(cfg)
        save_gene_expression_vs_coordinates_comparison(
            plot_dataset, iso, cov,
            out_dir / f"{run_name}_gene_expression_vs_coordinates.png",
            isodepth_label="Isodepth",
            covariate_label=covariate_label,
            pred_isodepth=pred_iso,
            pred_covariate=pred_cov,
            decoder_df=decoder_df,
        )
    else:
        save_gene_expression_vs_isodepth_plot(
            plot_dataset, iso,
            out_dir / f"{run_name}_gene_expression_vs_isodepth.png",
            decoder_preds=pred_iso,
            decoder_df=decoder_df,
        )
    return "ok"


def regen_from_result_json(result_path: Path, *, dry_run: bool = False) -> str:
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    arts = saved.get("artifacts", {})
    if arts.get("cell_type_names"):
        return regen_separate_celltype_plots(result_path, dry_run=dry_run)
    return regen_standard_plots(result_path, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO / "results",
        help="Root directory to search for *_result.json files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count eligible runs only")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index into sorted result JSON list (for resuming)",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    result_paths = sorted(results_root.rglob("*_result.json"))
    if not result_paths:
        print(f"No result JSONs under {results_root}")
        return

    counts: dict[str, int] = {}
    for index, result_path in enumerate(result_paths, start=1):
        if index < int(args.start_index):
            continue
        try:
            status = regen_from_result_json(result_path, dry_run=args.dry_run)
        except Exception as exc:
            status = f"error: {exc}"
        counts[status] = counts.get(status, 0) + 1
        if status.startswith("ok"):
            print(f"[{index}/{len(result_paths)}] {result_path.relative_to(REPO)} -> {status}", flush=True)
        elif not status.startswith("skip"):
            print(f"[{index}/{len(result_paths)}] {result_path.relative_to(REPO)} -> {status}", flush=True)

    print("\nSummary:", flush=True)
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}", flush=True)


if __name__ == "__main__":
    main()
