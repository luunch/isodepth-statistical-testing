from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.plots import (
    save_celltype_dataset_plot,
    save_celltype_expression_plot,
    save_combined_celltype_isodepth_grid,
    save_combined_celltype_metric_distribution,
    save_dataset_triptych,
    save_isodepth_triptych,
    save_metric_distribution_plot,
    save_perturbation_delta_pvalue_plot,
    save_selected_genes_expression_vs_isodepth,
    save_synthetic_true_curve_plot,
    save_subset_fraction_pvalue_plot,
    save_true_rerun_isodepth_grid,
)
from data.schemas import DatasetBundle, RunConfig, TestResult, run_config_from_mapping
from methods.metrics import summarize_metric_distribution


def load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_config_relative_paths(config: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return config

    base_dir = Path(config_path).resolve().parent.parent
    data_cfg = config.get("data")
    if isinstance(data_cfg, Mapping):
        h5ad_path = data_cfg.get("h5ad")
        if isinstance(h5ad_path, str) and h5ad_path and not Path(h5ad_path).is_absolute():
            data_cfg = dict(data_cfg)
            data_cfg["h5ad"] = str((base_dir / h5ad_path).resolve())
            config = dict(config)
            config["data"] = data_cfg

    output_cfg = config.get("output")
    if isinstance(output_cfg, Mapping):
        out_dir = output_cfg.get("out_dir")
        if isinstance(out_dir, str) and out_dir and not Path(out_dir).is_absolute():
            output_cfg = dict(output_cfg)
            output_cfg["out_dir"] = str((base_dir / out_dir).resolve())
            config = dict(config)
            config["output"] = output_cfg

    return config


def deep_merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_run_config(config_path: str | None, cli_overrides: Mapping[str, Any]) -> RunConfig:
    file_config = _resolve_config_relative_paths(load_json_config(config_path), config_path)
    merged = deep_merge_dicts(file_config, cli_overrides)
    if "data" in merged and "seed" in merged["data"]:
        merged.setdefault("test", {})
        merged["test"].setdefault("seed", merged["data"]["seed"])
    return run_config_from_mapping(merged)


def build_manifest_config_snapshot(
    spec_path: str | Path,
    base_config_paths: dict[str, str | Path],
) -> dict[str, Any]:
    """Return a dict with the full raw JSON contents of the experiment spec and
    all base config files, suitable for embedding in a manifest.json so that
    every specified parameter is preserved."""
    snapshot: dict[str, Any] = {}
    spec_path = Path(spec_path).resolve()
    snapshot["spec_path"] = str(spec_path)
    snapshot["spec"] = load_json_config(str(spec_path))
    snapshot["base_configs"] = {}
    for key, cfg_path in base_config_paths.items():
        resolved = Path(cfg_path).resolve()
        snapshot["base_configs"][key] = {
            "path": str(resolved),
            "contents": load_json_config(str(resolved)),
        }
    return snapshot


def _compact_dataset_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(meta)
    var_names = compact.pop("var_names", None)
    if var_names is not None:
        compact["n_var_names"] = int(len(var_names))
        compact["var_names_preview"] = [str(x) for x in list(var_names)[:10]]
    synthetic_true_curve = compact.pop("synthetic_true_curve", None)
    if synthetic_true_curve is not None:
        compact["has_synthetic_true_curve"] = True
    return compact


def _filter_mapping(mapping: Mapping[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    return {key: mapping[key] for key in allowed_keys if key in mapping}


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _compact_run_config(run_config: RunConfig) -> dict[str, Any]:
    config_dict = run_config.to_dict()
    data = dict(config_dict.get("data", {}))
    test = dict(config_dict.get("test", {}))
    output = dict(config_dict.get("output", {}))
    method = str(test.get("method", run_config.test.method))

    data_keys = {
        "source",
        "seed",
        "mode",
        "n_cells",
        "n_genes",
        "sigma",
        "poly_degree",
        "standardize_coordinates",
    }
    if data.get("source") == "h5ad":
        data_keys |= {
            "h5ad",
            "spatial_key",
            "obs_x_col",
            "obs_y_col",
            "layer",
            "use_raw",
            "min_cells_per_gene",
            "log1p",
            "standardize_expression",
            "q",
            "max_cells",
            "cell_type",
            "cell_type_key",
            "min_cells_per_celltype",
        }
    if data.get("mode") == "fourier":
        data_keys |= {"k_min", "k_max"}
    if data.get("source") == "synthetic":
        data_keys.add("dependent_xy")
        data_keys.add("shape")
        data_keys.add("sampling_bias")
    if data.get("mode") == "noise":
        data_keys.add("side_length")

    test_keys = {
        "method",
        "metric",
        "epochs",
        "lr",
        "patience",
        "seed",
        "device",
        "decoder",
        "verbose",
        "n_reruns",
        "covariate",
        "sgd_batch_size",
        "sgd_cosine_lr_decay",
        "sgd_cosine_eta_min",
        "sgd_cosine_t_max_steps",
    }
    if method in {
        "parallel_permutation",
        "cross_validation",
        "exact_existence",
        "full_retraining",
        "comparison_perturbation_test",
        "perturbation_test",
        "comparison_subsampling_test",
        "subsampling_test",
    }:
        test_keys.add("n_perms")
    if method == "cross_validation":
        test_keys.add("train_fraction")
    if method == "exact_existence":
        test_keys |= {"max_spatial_dims", "alpha"}
    if method in {"comparison_perturbation_test", "perturbation_test"}:
        test_keys |= {"n_nulls", "batch_size", "delta", "perturb_target"}
    elif method in {"comparison_subsampling_test", "subsampling_test"}:
        test_keys |= {"n_nulls", "subset_fractions"}

    output_keys = {"out_dir", "run_name", "save_preds", "save_perm_stats"}

    compact = {
        "data": _filter_mapping(data, data_keys),
        "test": _filter_mapping(test, test_keys),
        "output": _filter_mapping(output, output_keys),
    }
    return compact


def _method_artifact_keys(method_name: str) -> set[str]:
    shared = {
        "perm_summary",
        "dataset_meta",
        "lowest_stat",
        "highest_stat",
        "rerun_summary",
        "true_rerun_index",
        "true_train_loss",
        "lowest_rerun_index",
        "lowest_train_loss",
        "highest_rerun_index",
        "highest_train_loss",
    }
    if method_name in {
        "parallel_permutation",
        "cross_validation",
        "exact_existence",
        "full_retraining",
        "subsampling_test",
    }:
        extra = {
            "null_summary",
            "true_isodepth",
            "stat_covariate",
            "p_value_covariate",
            "pred_true_covariate",
            "true_isodepth_covariate",
            "pred_true_full_iso",
            "true_isodepth_full_iso",
        }
        if method_name == "cross_validation":
            extra |= {
                "train_mask",
                "test_mask",
                "train_fraction",
                "test_fraction",
                "train_size",
                "test_size",
                "observed_test_loss",
            }
        if method_name == "exact_existence":
            extra |= {"selected_spatial_dims", "tested_spatial_dims", "step_summaries", "alpha", "max_spatial_dims"}
        return shared | extra
    if method_name == "perturbation_test":
        return shared | {
            "delta",
            "delta_summaries",
            "primary_delta",
            "perturb_target",
            "observed_summary",
            "null_summary",
            "summary_statistic",
            "n_nulls",
        }
    if method_name == "comparison_perturbation_test":
        return shared | {
            "delta",
            "delta_summaries",
            "primary_delta",
            "perturb_target",
            "observed_summary",
            "null_summary",
            "summary_statistic",
            "n_nulls",
        }
    if method_name == "comparison_subsampling_test":
        return shared | {
            "observed_summary",
            "observed_correlation_summary",
            "null_summary",
            "fraction_summaries",
            "primary_fraction",
            "summary_statistic",
            "n_nulls",
            "n_perms",
            "subset_fractions",
            "lowest_subset_fraction",
            "highest_subset_fraction",
        }
    return shared


def _save_single_type_outputs(
    type_name: str,
    type_data: dict,
    dataset_meta: dict,
    type_dir: Path,
    *,
    metric: str = "nll_gaussian_mse",
) -> dict[str, str]:
    """Generate the standard plot set for one cell type in separate mode."""
    S_c = np.asarray(type_data["S"], dtype=np.float32)
    A_c = np.asarray(type_data["A"], dtype=np.float32)
    subset_meta = dict(dataset_meta)
    subset_meta.pop("cell_type_labels", None)
    subset_meta.pop("cell_type_names", None)
    subset_meta.pop("n_cell_types", None)
    subset_meta.pop("cell_type_mode", None)
    subset_dataset = DatasetBundle(S=S_c, A=A_c, meta=subset_meta).validate()

    subset_result = TestResult(
        method_name="parallel_permutation",
        metric=metric,
        p_value=float(type_data["p_value"]),
        stat_true=float(type_data["stat_true"]),
        stat_perm=np.asarray(type_data["stat_perm"], dtype=np.float64),
        runtime_sec=0.0,
        n_cells=int(S_c.shape[0]),
        n_genes=int(A_c.shape[1]),
        config={},
        artifacts={
            "model": type_data.get("model"),
            "pred_true": type_data.get("pred_true"),
            "true_isodepth": type_data.get("true_isodepth"),
            "rerun_summary": type_data.get("rerun_summary", {}),
            "true_rerun_index": type_data.get("true_rerun_index", 0),
            "true_train_loss": type_data.get("true_train_loss", 0.0),
            "lowest_isodepth": type_data.get("lowest_isodepth"),
            "lowest_S": type_data.get("lowest_S"),
            "lowest_stat": type_data.get("lowest_stat", 0.0),
            "lowest_perm_index": type_data.get("lowest_perm_index", 0),
            "lowest_rerun_index": type_data.get("lowest_rerun_index", 0),
            "lowest_train_loss": type_data.get("lowest_train_loss", 0.0),
            "highest_isodepth": type_data.get("highest_isodepth"),
            "highest_S": type_data.get("highest_S"),
            "highest_stat": type_data.get("highest_stat", 0.0),
            "highest_perm_index": type_data.get("highest_perm_index", 0),
            "highest_rerun_index": type_data.get("highest_rerun_index", 0),
            "highest_train_loss": type_data.get("highest_train_loss", 0.0),
        },
    ).validate()

    type_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    p = save_dataset_triptych(subset_dataset, subset_result, type_dir / f"{type_name}_dataset.png")
    if p is not None:
        paths["dataset_triptych_plot"] = str(p)

    p = save_isodepth_triptych(subset_dataset, subset_result, type_dir / f"{type_name}_isodepth.png")
    if p is not None:
        paths["isodepth_triptych_plot"] = str(p)

    p_dist = save_metric_distribution_plot(subset_result, type_dir / f"{type_name}_metric_distribution.png")
    paths["metric_distribution_plot"] = str(p_dist)

    p_genes = save_selected_genes_expression_vs_isodepth(
        subset_dataset, subset_result, type_dir, top_k=5,
    )
    if p_genes is not None:
        paths["selected_genes_dir"] = str(p_genes)

    model_c = type_data.get("model")
    training_metadata = getattr(model_c, "training_metadata", None)
    if isinstance(training_metadata, Mapping):
        true_rerun_isodepths = training_metadata.get("true_rerun_isodepths")
        if true_rerun_isodepths is not None:
            rerun_losses = np.asarray(
                training_metadata.get("train_loss_per_rerun", [[0.0]]),
                dtype=np.float64,
            )
            selected_arr = np.asarray(
                training_metadata.get("best_rerun_index_per_model", [0]),
                dtype=np.int64,
            )
            sel_idx = int(selected_arr[0]) if selected_arr.size else 0
            p_rerun = save_true_rerun_isodepth_grid(
                subset_dataset,
                np.asarray(true_rerun_isodepths, dtype=np.float32),
                type_dir / f"{type_name}_true_rerun_isodepths.png",
                rerun_losses=rerun_losses[0] if rerun_losses.ndim >= 2 and rerun_losses.shape[0] > 0 else None,
                selected_rerun_index=sel_idx,
            )
            if p_rerun is not None:
                paths["true_rerun_isodepth_grid_plot"] = str(p_rerun)

    return paths


def save_standardized_outputs(
    dataset: DatasetBundle,
    result: TestResult,
    run_config: RunConfig,
) -> tuple[dict[str, Any], Path]:
    out_root = Path(run_config.output.out_dir)
    out_dir = out_root / run_config.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.artifacts.get("cell_type_mode") == "separate":
        return _save_separate_celltype_outputs(dataset, result, run_config, out_dir)

    artifact_paths: dict[str, str] = {}

    dataset_triptych_path = save_dataset_triptych(
        dataset,
        result,
        out_dir / f"{run_config.output.run_name}_dataset.png",
    )
    if dataset_triptych_path is not None:
        artifact_paths["dataset_triptych_plot"] = str(dataset_triptych_path)

    synthetic_true_curve_path = save_synthetic_true_curve_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_true_curve.png",
    )
    if synthetic_true_curve_path is not None:
        artifact_paths["synthetic_true_curve_plot"] = str(synthetic_true_curve_path)

    isodepth_plot_path = save_isodepth_triptych(
        dataset,
        result,
        out_dir / f"{run_config.output.run_name}_isodepth.png",
    )
    if isodepth_plot_path is not None:
        artifact_paths["isodepth_triptych_plot"] = str(isodepth_plot_path)

    distribution_plot_path = save_metric_distribution_plot(
        result,
        out_dir / f"{run_config.output.run_name}_metric_distribution.png",
    )
    artifact_paths["metric_distribution_plot"] = str(distribution_plot_path)

    selected_genes_dir = save_selected_genes_expression_vs_isodepth(dataset, result, out_dir, top_k=5)
    if selected_genes_dir is not None:
        artifact_paths["selected_genes_dir"] = str(selected_genes_dir)

    subset_fraction_plot_path = save_subset_fraction_pvalue_plot(
        result,
        out_dir / f"{run_config.output.run_name}_subset_fraction_pvalues.png",
    )
    if subset_fraction_plot_path is not None:
        artifact_paths["subset_fraction_pvalue_plot"] = str(subset_fraction_plot_path)

    perturbation_delta_plot_path = save_perturbation_delta_pvalue_plot(
        result,
        out_dir / f"{run_config.output.run_name}_delta_pvalues.png",
    )
    if perturbation_delta_plot_path is not None:
        artifact_paths["delta_pvalue_plot"] = str(perturbation_delta_plot_path)

    celltype_plot_path = save_celltype_dataset_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_celltype.png",
    )
    if celltype_plot_path is not None:
        artifact_paths["celltype_dataset_plot"] = str(celltype_plot_path)

    celltype_expr_plot_path = save_celltype_expression_plot(
        dataset,
        result,
        out_dir / f"{run_config.output.run_name}_celltype_expression.png",
    )
    if celltype_expr_plot_path is not None:
        artifact_paths["celltype_expression_plot"] = str(celltype_expr_plot_path)

    model = result.artifacts.get("model")
    training_metadata = getattr(model, "training_metadata", None)
    if isinstance(training_metadata, Mapping):
        true_rerun_isodepths = training_metadata.get("true_rerun_isodepths")
        if true_rerun_isodepths is not None:
            rerun_losses = np.asarray(
                training_metadata.get("train_loss_per_rerun", [[0.0]]),
                dtype=np.float64,
            )
            selected_rerun_index_array = np.asarray(
                training_metadata.get("best_rerun_index_per_model", [0]),
                dtype=np.int64,
            )
            selected_rerun_index = int(selected_rerun_index_array[0]) if selected_rerun_index_array.size else 0
            true_rerun_plot_path = save_true_rerun_isodepth_grid(
                dataset,
                np.asarray(true_rerun_isodepths, dtype=np.float32),
                out_dir / f"{run_config.output.run_name}_true_rerun_isodepths.png",
                rerun_losses=rerun_losses[0] if rerun_losses.ndim >= 2 and rerun_losses.shape[0] > 0 else None,
                selected_rerun_index=selected_rerun_index,
            )
            if true_rerun_plot_path is not None:
                artifact_paths["true_rerun_isodepth_grid_plot"] = str(true_rerun_plot_path)

    payload = result.to_json_dict(
        config=_compact_run_config(run_config),
        artifacts=_json_compatible(
            {
                **artifact_paths,
                **{
                    key: result.artifacts[key]
                    for key in _method_artifact_keys(result.method_name)
                    if key in result.artifacts
                },
                "perm_summary": summarize_metric_distribution(result.stat_perm),
                "dataset_meta": _compact_dataset_meta(dataset.meta),
            }
        ),
    )

    result_path = out_dir / f"{run_config.output.run_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload, result_path


def _save_separate_celltype_outputs(
    dataset: DatasetBundle,
    result: TestResult,
    run_config: RunConfig,
    out_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Generate per-cell-type output directories with full plot sets."""
    per_type_results: dict[str, dict] = result.artifacts["per_type_results"]
    cell_type_names: list[str] = result.artifacts["cell_type_names"]

    celltype_overview_path = save_celltype_dataset_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_celltype.png",
    )

    per_type_summaries: dict[str, dict[str, Any]] = {}
    per_type_artifact_paths: dict[str, dict[str, str]] = {}

    for type_name in cell_type_names:
        type_data = per_type_results[type_name]
        safe_name = type_name.replace(" ", "_").replace("/", "_")
        type_dir = out_dir / safe_name

        type_artifact_paths = _save_single_type_outputs(
            safe_name, type_data, dataset.meta, type_dir,
            metric=result.metric,
        )
        per_type_artifact_paths[type_name] = type_artifact_paths
        per_type_summaries[type_name] = {
            "p_value": float(type_data["p_value"]),
            "stat_true": float(type_data["stat_true"]),
            "stat_perm": [float(x) for x in np.asarray(type_data["stat_perm"]).tolist()],
            "n_cells": int(type_data["n_cells"]),
            "perm_summary": summarize_metric_distribution(type_data["stat_perm"]),
            "artifact_paths": type_artifact_paths,
        }

    combined_dist_path = save_combined_celltype_metric_distribution(
        per_type_results,
        cell_type_names,
        out_dir / f"{run_config.output.run_name}_combined_metric_distribution.png",
        metric=result.metric,
    )

    combined_isodepth_path = save_combined_celltype_isodepth_grid(
        per_type_results,
        cell_type_names,
        out_dir / f"{run_config.output.run_name}_combined_isodepths.png",
        full_spatial=dataset.S,
    )

    top_level_artifacts: dict[str, Any] = {
        "cell_type_mode": "separate",
        "cell_type_names": cell_type_names,
        "per_type_summaries": per_type_summaries,
        "perm_summary": summarize_metric_distribution(result.stat_perm),
        "dataset_meta": _compact_dataset_meta(dataset.meta),
        "combined_metric_distribution_plot": str(combined_dist_path),
        "combined_isodepth_plot": str(combined_isodepth_path),
    }
    if celltype_overview_path is not None:
        top_level_artifacts["celltype_dataset_plot"] = str(celltype_overview_path)

    payload = result.to_json_dict(
        config=_compact_run_config(run_config),
        artifacts=_json_compatible(top_level_artifacts),
    )

    result_path = out_dir / f"{run_config.output.run_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload, result_path
