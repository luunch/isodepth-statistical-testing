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
    return compact


def _filter_mapping(mapping: Mapping[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    return {key: mapping[key] for key in allowed_keys if key in mapping}


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

    test_keys = {"method", "metric", "epochs", "lr", "patience", "seed", "device", "verbose"}
    if method in {
        "parallel_permutation",
        "full_retraining",
        "frozen_encoder",
        "gaston_mix_closed_form",
        "comparison_perturbation_test",
        "perturbation_test",
        "subsampling_test",
    }:
        test_keys.add("n_perms")
    if method in {"comparison_perturbation_test", "perturbation_test"}:
        test_keys |= {"n_nulls", "batch_size", "n_experts", "delta", "perturb_target"}
    elif method == "gaston_mix_closed_form":
        test_keys.add("n_experts")
    elif method in {"comparison_subsampling_test", "subsampling_test"}:
        test_keys |= {"n_nulls", "subset_fractions", "n_subsets"}

    output_keys = {"out_dir", "run_name", "save_preds", "save_perm_stats"}

    compact = {
        "data": _filter_mapping(data, data_keys),
        "test": _filter_mapping(test, test_keys),
        "output": _filter_mapping(output, output_keys),
    }
    return compact


def _method_artifact_keys(method_name: str) -> set[str]:
    shared = {"perm_summary", "dataset_meta", "lowest_stat", "highest_stat"}
    if method_name in {
        "parallel_permutation",
        "full_retraining",
        "frozen_encoder",
        "gaston_mix_closed_form",
        "perturbation_test",
        "subsampling_test",
    }:
        return shared | {"null_summary"}
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
            "n_subsets",
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

    if run_config.output.save_perm_stats:
        perm_path = out_dir / "perm_stats.npy"
        np.save(perm_path, result.stat_perm)
        artifact_paths["perm_stats"] = str(perm_path)

        delta_summaries = result.artifacts.get("delta_summaries")
        if isinstance(delta_summaries, Mapping):
            for key, summary in delta_summaries.items():
                if not isinstance(summary, Mapping) or "null_distribution" not in summary:
                    continue
                delta_perm_path = out_dir / f"delta_{key}_perm_stats.npy"
                np.save(delta_perm_path, np.asarray(summary["null_distribution"], dtype=np.float64))
                artifact_paths[f"delta_{key}_perm_stats"] = str(delta_perm_path)

        fraction_summaries = result.artifacts.get("fraction_summaries")
        if isinstance(fraction_summaries, Mapping):
            for key, summary in fraction_summaries.items():
                if not isinstance(summary, Mapping) or "null_distribution" not in summary:
                    continue
                fraction_perm_path = out_dir / f"subset_fraction_{key}_perm_stats.npy"
                np.save(fraction_perm_path, np.asarray(summary["null_distribution"], dtype=np.float64))
                artifact_paths[f"subset_fraction_{key}_perm_stats"] = str(fraction_perm_path)

    if "observed_scores" in result.artifacts:
        observed_scores_path = out_dir / "observed_scores.npy"
        np.save(observed_scores_path, np.asarray(result.artifacts["observed_scores"], dtype=np.float64))
        artifact_paths["observed_scores"] = str(observed_scores_path)

    if "observed_correlations" in result.artifacts:
        observed_correlations_path = out_dir / "observed_correlations.npy"
        np.save(
            observed_correlations_path,
            np.asarray(result.artifacts["observed_correlations"], dtype=np.float64),
        )
        artifact_paths["observed_correlations"] = str(observed_correlations_path)

    if "subset_fraction_per_subset" in result.artifacts:
        subset_fraction_path = out_dir / "subset_fraction_per_subset.npy"
        np.save(
            subset_fraction_path,
            np.asarray(result.artifacts["subset_fraction_per_subset"], dtype=np.float32),
        )
        artifact_paths["subset_fraction_per_subset"] = str(subset_fraction_path)

    if "subset_size_per_subset" in result.artifacts:
        subset_size_path = out_dir / "subset_size_per_subset.npy"
        np.save(
            subset_size_path,
            np.asarray(result.artifacts["subset_size_per_subset"], dtype=np.int64),
        )
        artifact_paths["subset_size_per_subset"] = str(subset_size_path)

    if run_config.output.save_preds and "pred_true" in result.artifacts:
        pred_true_path = out_dir / "pred_true.npy"
        np.save(pred_true_path, np.asarray(result.artifacts["pred_true"], dtype=np.float32))
        artifact_paths["pred_true"] = str(pred_true_path)

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
        artifacts={
            **artifact_paths,
            **{
                key: result.artifacts[key]
                for key in _method_artifact_keys(result.method_name)
                if key in result.artifacts
            },
            "perm_summary": summarize_metric_distribution(result.stat_perm),
            "dataset_meta": _compact_dataset_meta(dataset.meta),
        },
    )

    result_path = out_dir / f"{run_config.output.run_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload, result_path
