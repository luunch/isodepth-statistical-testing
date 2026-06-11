"""DLPFC Layer 3 Phase-1 experiment: Poisson with frozen Gaussian isodepth covariate.

Prepares a Layer-3-only h5ad with ``obs['gaussian_isodepth']`` from an existing
Gaussian separate-cell-type run, then runs the parallel-permutation Poisson test
with that axis as a fixed covariate.  Summarizes how the free Poisson isodepth
compares to the frozen Gaussian axis and to prior Layer-3 baselines.

Usage:
    # Prep + run + summarize (single command):
    python scripts/dlpfc_layer3_gaussian_axis_poisson.py

    # Or split across the generic config runner:
    python scripts/dlpfc_layer3_gaussian_axis_poisson.py --prep-only
    sbatch --export=ALL,CONFIG=configs/dlpfc_new/dlpfc_layer3_poisson_gaussian_covariate.json run_config.sh
    python scripts/dlpfc_layer3_gaussian_axis_poisson.py --summarize-only

    python scripts/dlpfc_layer3_gaussian_axis_poisson.py --spec configs/experiments/dlpfc_layer3_gaussian_axis_poisson.json
    python scripts/dlpfc_layer3_gaussian_axis_poisson.py --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data import load_dataset
from data.h5ad_loader import _apply_obs_subset, _select_hvg_mask
from data.schemas import CovariateConfig, DataConfig, OutputConfig, RunConfig, TestConfig
from experiments.configuration import save_standardized_outputs
from methods.permutation import run_permutation_method
from scripts.liver_lobule_sweep import load_spec

DEFAULT_SPEC = REPO / "configs/experiments/dlpfc_layer3_gaussian_axis_poisson.json"
PY = sys.executable


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO / path


def _layer3_mask_from_parent(
    parent_h5ad: Path,
    *,
    obs_filters: dict,
    obs_drop_na: list[str],
    layer_obs_key: str,
    layer_label: str,
    parent_layer: str | None = None,
) -> tuple[ad.AnnData, np.ndarray]:
    adata = ad.read_h5ad(parent_h5ad, backed="r")
    adata = _apply_obs_subset(
        adata,
        obs_filters=obs_filters,
        obs_drop_na=obs_drop_na,
    )
    if getattr(adata, "isbacked", False) or parent_layer is not None:
        adata = adata.to_memory()
    if parent_layer is not None:
        if parent_layer not in adata.layers:
            raise ValueError(
                f"Parent layer '{parent_layer}' not found in adata.layers; "
                f"available: {list(adata.layers.keys())}"
            )
        adata.X = adata.layers[parent_layer]

    if layer_obs_key not in adata.obs.columns:
        raise ValueError(
            f"Layer obs key '{layer_obs_key}' missing from adata.obs; "
            f"available: {list(adata.obs.columns)}"
        )
    layer_mask = adata.obs[layer_obs_key].astype(str).to_numpy() == str(layer_label)
    if not layer_mask.any():
        raise ValueError(
            f"No cells matched layer '{layer_label}' in column '{layer_obs_key}' "
            f"after parent filters {obs_filters}."
        )
    return adata, layer_mask


def _align_gaussian_isodepth(
    gaussian_npz: Path,
    expected_n: int,
) -> np.ndarray:
    payload = np.load(gaussian_npz)
    if "true_isodepth" not in payload:
        raise ValueError(f"{gaussian_npz} missing 'true_isodepth'")
    values = np.asarray(payload["true_isodepth"], dtype=np.float32).reshape(-1)
    if values.size != expected_n:
        raise ValueError(
            f"Gaussian isodepth length {values.size} != subset cell count {expected_n}. "
            "Regenerate the Gaussian separate-cell-type run or verify region/sample filters."
        )
    return values


def prepare_layer3_h5ad(spec: dict, *, force: bool = False) -> Path:
    data = spec["data"]
    prepared = _resolve(data["prepared_h5ad"])
    if prepared.exists() and not force:
        print(f"[prep] reusing existing h5ad: {prepared}", flush=True)
        return prepared

    parent = _resolve(data["parent_h5ad"])
    gaussian_npz = _resolve(spec["artifacts"]["gaussian_isodepth_npz"])
    top_var_genes = int(data["top_var_genes"])
    parent_layer = data.get("parent_layer")

    adata, layer_mask = _layer3_mask_from_parent(
        parent,
        obs_filters=dict(data.get("obs_filters", {})),
        obs_drop_na=list(data["obs_drop_na"]),
        layer_obs_key=str(data["layer_obs_key"]),
        layer_label=str(data["layer_label"]),
        parent_layer=parent_layer,
    )
    gaussian_iso = _align_gaussian_isodepth(gaussian_npz, int(layer_mask.sum()))

    if top_var_genes > 0 and top_var_genes < adata.n_vars:
        hvg_mask = _select_hvg_mask(adata, top_var_genes)
        adata = adata[:, hvg_mask].copy()
    else:
        adata = adata.copy()

    adata = adata[layer_mask].copy()
    cov_key = str(data["covariate_obs_key"])
    adata.obs[cov_key] = gaussian_iso

    prepared.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(prepared)
    print(
        f"[prep] wrote {prepared}  n={adata.n_obs} genes={adata.n_vars} "
        f"covariate='{cov_key}'",
        flush=True,
    )
    return prepared


def _test_config_from_spec(tst: dict) -> TestConfig:
    cov_raw = tst.get("covariate")
    covariate = None
    if cov_raw is not None:
        covariate = (
            CovariateConfig(type=cov_raw)
            if isinstance(cov_raw, str)
            else CovariateConfig(**dict(cov_raw))
        )
    return TestConfig(
        method=tst["method"],
        metric=tst["metric"],
        covariate=covariate,
        n_perms=int(tst["n_perms"]),
        epochs=int(tst["epochs"]),
        n_reruns=int(tst["n_reruns"]),
        lr=float(tst["lr"]),
        patience=int(tst.get("patience", 0)),
        seed=int(tst["seed"]),
        device=tst["device"],
        decoder=tst["decoder"],
        sgd_batch_size=tst.get("sgd_batch_size"),
        verbose=bool(tst.get("verbose", True)),
    )


def _data_config_from_spec(data: dict, pp: dict, prepared_h5ad: Path) -> DataConfig:
    return DataConfig(
        source="h5ad",
        h5ad=str(prepared_h5ad.relative_to(REPO)),
        spatial_key=str(data.get("spatial_key", "spatial")),
        layer=data.get("layer"),
        min_cells_per_gene=int(pp["min_cells_per_gene"]),
        top_var_genes=0,
        normalize_total=bool(pp["normalize_total"]),
        log1p=bool(pp["log1p"]),
        standardize_expression=bool(pp["standardize_expression"]),
        standardize_coordinates=bool(pp["standardize_coordinates"]),
        seed=int(data.get("seed", 42)),
    )


def run_experiment(spec: dict, *, skip_run: bool = False) -> Path:
    out_cfg = spec["output"]
    run_name = str(out_cfg["run_name"])
    run_dir = _resolve(out_cfg["out_dir"]) / run_name
    result_json = run_dir / f"{run_name}_result.json"

    if result_json.exists() and not skip_run:
        print(f"[run] reusing existing result: {result_json}", flush=True)
        return result_json

    if skip_run:
        raise FileNotFoundError(f"No existing result at {result_json}; run without --summarize-only.")

    prepared = prepare_layer3_h5ad(spec, force=False)
    data_cfg = _data_config_from_spec(spec["data"], spec["preprocessing"], prepared)
    test_cfg = _test_config_from_spec(spec["test"])
    run_config = RunConfig(
        data=data_cfg,
        test=test_cfg,
        output=OutputConfig(
            out_dir=str(out_cfg["out_dir"]),
            run_name=run_name,
            save_preds=False,
            save_perm_stats=True,
        ),
    )
    run_config.validate()

    dataset = load_dataset(run_config.data, covariate=run_config.test.covariate)
    print(
        f"[run] {run_name}: n={dataset.S.shape[0]} genes={dataset.A.shape[1]} "
        f"metric={run_config.test.metric} covariate={run_config.test.covariate.type}",
        flush=True,
    )
    result = run_permutation_method(dataset, run_config.test)
    if bool(out_cfg.get("save_per_unit_outputs", True)):
        _, result_json = save_standardized_outputs(dataset, result, run_config)
    else:
        result_json = run_dir / f"{run_name}_result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        with open(result_json, "w", encoding="utf-8") as fh:
            json.dump(
                result.to_json_dict(config=run_config.to_dict()),
                fh,
                indent=2,
            )
    print(f"[run] wrote {result_json}", flush=True)
    return result_json


def _baseline_layer3_stats(path: Path, layer_key: str) -> dict:
    payload = json.load(open(path))
    by_type = payload.get("artifacts", {}).get("per_type_summaries", {})
    if layer_key not in by_type:
        raise KeyError(f"Layer '{layer_key}' not found in {path}")
    row = by_type[layer_key]
    perm = np.asarray(row["stat_perm"], dtype=np.float64)
    return {
        "metric": payload.get("metric"),
        "p_value": float(row["p_value"]),
        "stat_true": float(row["stat_true"]),
        "null_mean": float(perm.mean()),
        "null_std": float(perm.std()),
        "z": float((perm.mean() - row["stat_true"]) / (perm.std() + 1e-12)),
        "n_cells": int(row.get("n_cells", 0)),
    }


def _axis_correlation(spec: dict, learned_iso: np.ndarray) -> dict:
    out: dict[str, float | None] = {}
    gaussian_npz = _resolve(spec["artifacts"]["gaussian_isodepth_npz"])
    gaussian_iso = np.asarray(np.load(gaussian_npz)["true_isodepth"], dtype=np.float64).reshape(-1)
    if gaussian_iso.size == learned_iso.size:
        out["abs_spearman_vs_gaussian_axis"] = float(abs(spearmanr(learned_iso, gaussian_iso).statistic))
        out["pearson_vs_gaussian_axis"] = float(pearsonr(learned_iso, gaussian_iso)[0])
    else:
        out["abs_spearman_vs_gaussian_axis"] = None
        out["pearson_vs_gaussian_axis"] = None

    poisson_npz = _resolve(spec["artifacts"]["poisson_isodepth_npz"])
    if poisson_npz.exists():
        poisson_iso = np.asarray(np.load(poisson_npz)["true_isodepth"], dtype=np.float64).reshape(-1)
        if poisson_iso.size == learned_iso.size:
            out["abs_spearman_vs_baseline_poisson_axis"] = float(
                abs(spearmanr(learned_iso, poisson_iso).statistic)
            )
        else:
            out["abs_spearman_vs_baseline_poisson_axis"] = None
    else:
        out["abs_spearman_vs_baseline_poisson_axis"] = None
    return out


def summarize(spec: dict, result_json: Path | None = None) -> dict:
    out_cfg = spec["output"]
    run_name = str(out_cfg["run_name"])
    if result_json is None:
        result_json = _resolve(out_cfg["out_dir"]) / run_name / f"{run_name}_result.json"
    if not result_json.exists():
        raise FileNotFoundError(f"Missing result JSON: {result_json}")

    result = json.load(open(result_json))
    art = result.get("artifacts", {})
    perm = np.asarray(result["stat_perm"], dtype=np.float64)
    learned_iso = np.asarray(art.get("true_isodepth", []), dtype=np.float64).reshape(-1)
    cov_iso = np.asarray(art.get("true_isodepth_covariate", []), dtype=np.float64).reshape(-1)

    baselines = spec["baselines"]
    layer_key = str(baselines["layer_key"])
    gaussian_base = _baseline_layer3_stats(_resolve(baselines["gaussian_layer3_result"]), layer_key)
    poisson_base = _baseline_layer3_stats(_resolve(baselines["poisson_layer3_result"]), layer_key)

    stat_true = float(result["stat_true"])
    stat_cov = art.get("stat_covariate")
    null_mean = float(perm.mean())
    null_std = float(perm.std())
    summary = {
        "experiment": spec["experiment_name"],
        "run_name": run_name,
        "n_cells": int(learned_iso.size),
        "metric": result.get("metric"),
        "p_value": float(result["p_value"]),
        "z_effect": float((null_mean - stat_true) / (null_std + 1e-12)),
        "stat_true": stat_true,
        "stat_covariate": stat_cov,
        "p_value_covariate": art.get("p_value_covariate"),
        "poisson_beats_gaussian_axis": (
            stat_cov is not None and stat_true < float(stat_cov)
        ),
        "delta_stat_cov_minus_true": (
            float(stat_cov) - stat_true if stat_cov is not None else None
        ),
        "baseline_gaussian_p": gaussian_base["p_value"],
        "baseline_gaussian_z": gaussian_base["z"],
        "baseline_poisson_p": poisson_base["p_value"],
        "baseline_poisson_z": poisson_base["z"],
        "baseline_poisson_stat_true": poisson_base["stat_true"],
        **(_axis_correlation(spec, learned_iso) if learned_iso.size else {}),
        "abs_spearman_learned_vs_covariate": (
            float(abs(spearmanr(learned_iso, cov_iso).statistic))
            if learned_iso.size and cov_iso.size
            else None
        ),
    }

    out_dir = _resolve(out_cfg["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "experiment_summary.json"
    summary_csv = out_dir / "experiment_summary.csv"
    with open(summary_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n=== Layer 3 Gaussian-axis Poisson experiment ===", flush=True)
    print(f"run: {run_name}  n={summary['n_cells']}", flush=True)
    print(
        f"Poisson free axis: p={summary['p_value']:.4g}  z={summary['z_effect']:.3f}  "
        f"stat={summary['stat_true']:.6g}",
        flush=True,
    )
    if stat_cov is not None:
        print(
            f"Poisson on frozen Gaussian axis: stat={float(stat_cov):.6g}  "
            f"p={summary['p_value_covariate']:.4g}  "
            f"free beats frozen={summary['poisson_beats_gaussian_axis']}",
            flush=True,
    )
    print(
        f"Baselines Layer 3: gaussian p={summary['baseline_gaussian_p']:.4g} "
        f"(z={summary['baseline_gaussian_z']:.2f}) | "
        f"poisson p={summary['baseline_poisson_p']:.4g} "
        f"(z={summary['baseline_poisson_z']:.2f})",
        flush=True,
    )
    if summary.get("abs_spearman_vs_gaussian_axis") is not None:
        print(
            f"Axis recovery: |Spearman|(learned, gaussian)="
            f"{summary['abs_spearman_vs_gaussian_axis']:.4f}",
            flush=True,
        )
    print(f"wrote {summary_json}", flush=True)
    print(f"wrote {summary_csv}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC), help="Experiment spec JSON")
    parser.add_argument("--prep-only", action="store_true", help="Only build the prepared h5ad")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize an existing run")
    parser.add_argument("--force-prep", action="store_true", help="Rebuild the prepared h5ad")
    parser.add_argument(
        "--via-run-permutation",
        action="store_true",
        help="Call run_permutation.py with the paired run config instead of the in-process API",
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    if args.force_prep or args.prep_only:
        prepare_layer3_h5ad(spec, force=args.force_prep)
        if args.prep_only:
            return

    result_json: Path | None = None
    if not args.summarize_only:
        if args.via_run_permutation:
            prepare_layer3_h5ad(spec, force=args.force_prep)
            cfg = _resolve(out_cfg.get("run_config", "configs/dlpfc_new/dlpfc_layer3_poisson_gaussian_covariate.json"))
            proc = subprocess.run(
                [PY, str(REPO / "run_permutation.py"), "--config", str(cfg), "--quiet"],
                cwd=str(REPO),
            )
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
            run_name = spec["output"]["run_name"]
            result_json = _resolve(spec["output"]["out_dir"]) / run_name / f"{run_name}_result.json"
        else:
            result_json = run_experiment(spec)

    summarize(spec, result_json=result_json)


if __name__ == "__main__":
    main()
