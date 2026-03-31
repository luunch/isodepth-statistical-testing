from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.plots import save_isodepth_triptych, save_metric_distribution_plot
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


def save_standardized_outputs(
    dataset: DatasetBundle,
    result: TestResult,
    run_config: RunConfig,
) -> tuple[dict[str, Any], Path]:
    out_root = Path(run_config.output.out_dir)
    out_dir = out_root / run_config.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, str] = {}

    s_path = out_dir / "S.npy"
    a_path = out_dir / "A.npy"
    np.save(s_path, dataset.S)
    np.save(a_path, dataset.A)
    artifact_paths["S"] = str(s_path)
    artifact_paths["A"] = str(a_path)

    if run_config.output.save_perm_stats:
        perm_path = out_dir / "perm_stats.npy"
        np.save(perm_path, result.stat_perm)
        artifact_paths["perm_stats"] = str(perm_path)

    if "observed_scores" in result.artifacts:
        observed_scores_path = out_dir / "observed_scores.npy"
        np.save(observed_scores_path, np.asarray(result.artifacts["observed_scores"], dtype=np.float64))
        artifact_paths["observed_scores"] = str(observed_scores_path)

    if run_config.output.save_preds and "pred_true" in result.artifacts:
        pred_true_path = out_dir / "pred_true.npy"
        np.save(pred_true_path, np.asarray(result.artifacts["pred_true"], dtype=np.float32))
        artifact_paths["pred_true"] = str(pred_true_path)

    isodepth_plot_path = save_isodepth_triptych(
        dataset,
        result,
        out_dir / f"{run_config.output.run_name}_isodepth_triptych.png",
    )
    if isodepth_plot_path is not None:
        artifact_paths["isodepth_triptych_plot"] = str(isodepth_plot_path)

    distribution_plot_path = save_metric_distribution_plot(
        result,
        out_dir / f"{run_config.output.run_name}_metric_distribution.png",
    )
    artifact_paths["metric_distribution_plot"] = str(distribution_plot_path)

    payload = result.to_json_dict(
        config=run_config.to_dict(),
        artifacts={
            **artifact_paths,
            "perm_summary": summarize_metric_distribution(result.stat_perm),
            **{
                key: result.artifacts[key]
                for key in [
                    "delta",
                    "perturb_target",
                    "observed_summary",
                    "null_summary",
                    "summary_statistic",
                    "n_nulls",
                    "lowest_stat",
                    "highest_stat",
                ]
                if key in result.artifacts
            },
            "dataset_meta": _compact_dataset_meta(dataset.meta),
        },
    )

    result_path = out_dir / f"{run_config.output.run_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload, result_path
