from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.plots import (
    save_dataset_triptych,
    save_isodepth_triptych,
    save_metric_distribution_plot,
    save_perturbation_delta_pvalue_plot,
    save_synthetic_true_curve_plot,
    save_subset_fraction_pvalue_plot,
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
            "standardize",
            "q",
            "max_cells",
        }
    if data.get("mode") == "fourier":
        data_keys |= {"k_min", "k_max"}

    test_keys = {
        "method",
        "metric",
        "epochs",
        "lr",
        "patience",
        "seed",
        "device",
        "verbose",
        "n_reruns",
        "sgd_batch_size",
    }
    if method in {
        "parallel_permutation",
        "exact_existence",
        "full_retraining",
        "comparison_perturbation_test",
        "perturbation_test",
        "comparison_subsampling_test",
        "subsampling_test",
    }:
        test_keys.add("n_perms")
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
        "exact_existence",
        "full_retraining",
        "subsampling_test",
    }:
        extra = {"null_summary", "true_isodepth"}
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


def save_standardized_outputs(
    dataset: DatasetBundle,
    result: TestResult,
    run_config: RunConfig,
) -> tuple[dict[str, Any], Path]:
    out_root = Path(run_config.output.out_dir)
    out_dir = out_root / run_config.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

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
