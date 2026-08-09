from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.cosmx_region_context import (
    save_cosmx_region_context_plot,
    _is_cosmx_subset_config,
    _LEGACY_RUN_RE,
)
from analysis.plots import (
    save_block_permutation_overlay,
    save_celltype_dataset_plot,
    save_celltype_or_spatial_split_plot,
    save_combined_celltype_residual_ratio_outputs,
    save_combined_celltype_isodepth_grid,
    save_combined_celltype_metric_distribution,
    save_cross_validation_fold_isodepth_grid,
    save_cross_validation_per_fold_metric_distributions,
    save_dataset_triptych,
    save_gene_expression_vs_coordinates_comparison,
    save_gene_expression_vs_isodepth_plot,
    save_isodepth_triptych,
    save_loss_curve_plot,
    save_metric_distribution_plot,
    save_moran_distribution_plot,
    save_msr_surrogate_example_plot,
    save_obs_numeric_filter_diagnostic_plot,
    save_obs_numeric_filter_histogram,
    save_permutation_null_comparison,
    save_perturbation_delta_pvalue_plot,
    save_synthetic_kernel_plot,
    save_synthetic_true_curve_plot,
    save_subset_fraction_pvalue_plot,
    save_true_rerun_isodepth_grid,
    save_freedman_lane_covariate_plot,
    save_covariate_whitening_spatial_plot,
    save_single_type_covariate_plot,
    save_fourier_surrogate_example_plot,
)
from data.schemas import (
    DatasetBundle,
    MSR_METHODS,
    RunConfig,
    TestResult,
    TOTAL_COUNTS_COVARIATE,
    run_config_from_mapping,
)
from data import raw_coordinates_from_standardized
from data.transforms import celltype_expression_residuals
from methods.metrics import summarize_metric_distribution
from methods.trainers.isodepth import get_training_metadata


def load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _repo_root_from_config_path(config_path: str) -> Path:
    """Return the repo root for a config file under ``configs/`` (any nesting depth)."""
    config_dir = Path(config_path).resolve().parent
    for ancestor in [config_dir, *config_dir.parents]:
        if ancestor.name == "configs":
            return ancestor.parent
    return config_dir.parent


def _resolve_relative_path(config_path: str, rel_path: str) -> str:
    """Resolve a config-relative path to an absolute path.

    - ``../`` or ``./`` paths are resolved from the config file's directory
      (supports arbitrary nesting depth under ``configs/``).
    - Bare paths (e.g. ``data/h5ad/...``) are resolved from the repo root.
    """
    if not rel_path or Path(rel_path).is_absolute():
        return rel_path
    config_dir = Path(config_path).resolve().parent
    if rel_path.startswith("../") or rel_path.startswith("./"):
        return str((config_dir / rel_path).resolve())
    repo_root = _repo_root_from_config_path(config_path)
    return str((repo_root / rel_path).resolve())


def _resolve_config_relative_paths(config: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return config

    data_cfg = config.get("data")
    if isinstance(data_cfg, Mapping):
        h5ad_path = data_cfg.get("h5ad")
        if isinstance(h5ad_path, str) and h5ad_path:
            data_cfg = dict(data_cfg)
            data_cfg["h5ad"] = _resolve_relative_path(config_path, h5ad_path)
            config = dict(config)
            config["data"] = data_cfg
        obs_indices = data_cfg.get("obs_indices") if isinstance(data_cfg, Mapping) else None
        if isinstance(obs_indices, str) and obs_indices:
            data_cfg = dict(config.get("data", data_cfg))
            data_cfg["obs_indices"] = _resolve_relative_path(config_path, obs_indices)
            config = dict(config)
            config["data"] = data_cfg

    output_cfg = config.get("output")
    if isinstance(output_cfg, Mapping):
        out_dir = output_cfg.get("out_dir")
        if isinstance(out_dir, str) and out_dir:
            output_cfg = dict(output_cfg)
            output_cfg["out_dir"] = _resolve_relative_path(config_path, out_dir)
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


def _obs_numeric_filter_mask(obs, obs_numeric_filters: Mapping[str, Any]) -> tuple[np.ndarray, str, dict]:
    import pandas as pd

    allowed_ops = {"gt", "ge", "gte", "lt", "le", "lte", "eq", "ne"}
    mask = np.ones(obs.shape[0], dtype=bool)
    first_key: str | None = None
    first_spec: dict | None = None

    for col, spec_raw in obs_numeric_filters.items():
        spec = dict(spec_raw)
        unknown_ops = set(spec.keys()) - allowed_ops
        if unknown_ops:
            raise ValueError(
                f"obs_numeric_filters['{col}'] has unsupported operators "
                f"{sorted(unknown_ops)}; allowed: {sorted(allowed_ops)}"
            )
        if col not in obs.columns:
            raise ValueError(
                f"obs_numeric_filters key '{col}' not in adata.obs; "
                f"available: {list(obs.columns)}"
            )
        if first_key is None:
            first_key = str(col)
            first_spec = spec
        values = pd.to_numeric(obs[col], errors="coerce").to_numpy(dtype=np.float64)
        col_mask = np.isfinite(values)
        for op, threshold_raw in spec.items():
            threshold = float(threshold_raw)
            if op == "gt":
                col_mask &= values > threshold
            elif op in {"ge", "gte"}:
                col_mask &= values >= threshold
            elif op == "lt":
                col_mask &= values < threshold
            elif op in {"le", "lte"}:
                col_mask &= values <= threshold
            elif op == "eq":
                col_mask &= values == threshold
            elif op == "ne":
                col_mask &= values != threshold
        mask &= col_mask

    if first_key is None or first_spec is None:
        raise ValueError("obs_numeric_filters must be non-empty")
    return mask, first_key, first_spec


def _load_obs_numeric_filter_context(run_config: RunConfig) -> dict[str, Any] | None:
    if run_config.data.source != "h5ad" or not run_config.data.obs_numeric_filters:
        return None

    import anndata as ad
    import pandas as pd

    from data.h5ad_loader import _apply_obs_subset, _extract_coordinates

    adata = ad.read_h5ad(run_config.data.h5ad, backed="r")
    pre_threshold = _apply_obs_subset(
        adata,
        obs_filters=run_config.data.obs_filters,
        obs_indices=run_config.data.obs_indices,
        obs_drop_na=run_config.data.obs_drop_na,
        obs_numeric_filters=None,
    )

    keep_mask, obs_key, filter_spec = _obs_numeric_filter_mask(
        pre_threshold.obs,
        run_config.data.obs_numeric_filters,
    )
    values = pd.to_numeric(pre_threshold.obs[obs_key], errors="coerce").to_numpy(
        dtype=np.float32
    )
    S = _extract_coordinates(
        pre_threshold,
        spatial_key=run_config.data.spatial_key,
        obs_x_col=run_config.data.obs_x_col,
        obs_y_col=run_config.data.obs_y_col,
    )

    labels = None
    label_names = None
    if run_config.data.cell_type_key in pre_threshold.obs.columns:
        raw_labels = pre_threshold.obs[run_config.data.cell_type_key].values
        label_names = sorted(set(str(v) for v in raw_labels))
        label_to_idx = {name: i for i, name in enumerate(label_names)}
        labels = np.asarray([label_to_idx[str(v)] for v in raw_labels], dtype=np.int64)

    return {
        "S": S,
        "labels": labels,
        "label_names": label_names,
        "values": values,
        "keep_mask": keep_mask,
        "obs_key": obs_key,
        "filter_spec": filter_spec,
        "label_title": (
            "CNV clone"
            if run_config.data.cell_type_key == "calicost_clone_label"
            else run_config.data.cell_type_key
        ),
    }


def _save_obs_numeric_filter_diagnostics(
    run_config: RunConfig,
    out_dir: Path,
    *,
    cell_type_names: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Save spatial and histogram diagnostics for obs numeric filters."""
    try:
        ctx = _load_obs_numeric_filter_context(run_config)
        if ctx is None:
            return {}, {}

        top_level_paths: dict[str, str] = {}
        per_type_paths: dict[str, str] = {}

        spatial_path = save_obs_numeric_filter_diagnostic_plot(
            ctx["S"],
            labels=ctx["labels"],
            label_names=ctx["label_names"],
            values=ctx["values"],
            keep_mask=ctx["keep_mask"],
            obs_key=ctx["obs_key"],
            filter_spec=ctx["filter_spec"],
            out_path=out_dir / f"{run_config.output.run_name}_obs_threshold.png",
            label_title=ctx["label_title"],
        )
        if spatial_path is not None:
            top_level_paths["obs_numeric_filter_diagnostic_plot"] = str(spatial_path)

        if cell_type_names:
            labels_arr = ctx["labels"]
            label_names = ctx["label_names"] or []
            if labels_arr is None:
                return top_level_paths, per_type_paths
            for type_name in cell_type_names:
                if type_name not in label_names:
                    continue
                label_idx = label_names.index(type_name)
                type_mask = labels_arr == label_idx
                if not np.any(type_mask):
                    continue
                safe_name = type_name.replace(" ", "_").replace("/", "_")
                type_dir = out_dir / safe_name
                type_dir.mkdir(parents=True, exist_ok=True)
                hist_path = save_obs_numeric_filter_histogram(
                    ctx["values"][type_mask],
                    ctx["keep_mask"][type_mask],
                    ctx["obs_key"],
                    ctx["filter_spec"],
                    type_dir / f"{safe_name}_obs_threshold_histogram.png",
                    subset_label=type_name,
                )
                if hist_path is not None:
                    per_type_paths[type_name] = str(hist_path)
        else:
            hist_path = save_obs_numeric_filter_histogram(
                ctx["values"],
                ctx["keep_mask"],
                ctx["obs_key"],
                ctx["filter_spec"],
                out_dir / f"{run_config.output.run_name}_obs_threshold_histogram.png",
            )
            if hist_path is not None:
                top_level_paths["obs_numeric_filter_histogram_plot"] = str(hist_path)

        return top_level_paths, per_type_paths
    except Exception as exc:
        import warnings

        warnings.warn(f"Could not save obs numeric-filter diagnostic plots: {exc}")
        return {}, {}


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
    spatial_split_diag = compact.pop("spatial_region_split_diag", None)
    if spatial_split_diag is not None:
        compact["has_spatial_region_split_diag"] = True
        compact["spatial_region_split_algorithm"] = spatial_split_diag.get("algorithm")
        compact["spatial_region_split_n_removed"] = int(
            np.sum(np.asarray(spatial_split_diag.get("removed", []), dtype=bool))
        )
        compact["spatial_region_split_region_names"] = list(
            spatial_split_diag.get("region_color_names") or []
        )
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
            "exclude_gene_patterns",
            "gene_list",
            "log1p",
            "standardize_expression",
            "q",
            "max_cells",
            "cell_type",
            "cell_type_key",
            "min_cells_per_celltype",
            "obs_filters",
            "obs_indices",
        "obs_drop_na",
        "covariate_whitening",
    }
    if data.get("mode") == "fourier":
        data_keys |= {"k_min", "k_max"}
    if data.get("source") == "synthetic":
        data_keys.add("dependent_xy")
        data_keys.add("shape")
        data_keys.add("sampling_bias")
        data_keys.add("expression_distribution")
        if data.get("expression_distribution") == "poisson":
            data_keys.add("mean_count")
        if data.get("kernel") is not None:
            data_keys |= {"kernel", "delta", "scale"}
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
        "alpha",
        "recursive",
        "max_gradients",
        "gaussian_pretrain_epochs",
        "gaussian_pretrain_freeze_encoder",
        "moran",
        "moran_neighbor_radius_um",
    }
    if method in {
        "parallel_permutation",
        "block_permutation",
        "binning",
        "joint_truncated_msr",
        "joint_truncated_rank_msr",
        "fourier_spectral_randomization",
        "cross_validation",
        "full_retraining",
        "comparison_perturbation_test",
        "perturbation_test",
        "comparison_subsampling_test",
        "subsampling_test",
    }:
        test_keys.add("n_perms")
    if method == "block_permutation":
        test_keys |= {
            "block_radius",
            "coordinate_um_per_unit",
            "block_jitter",
            "save_permutation_null_comparison",
        }
    if method == "binning":
        test_keys |= {
            "bin_shape",
            "bin_spot_distance_um",
            "coordinate_um_per_unit",
            "block_jitter",
        }
    if method in MSR_METHODS:
        test_keys |= {
            "msr_truncate_um",
            "msr_neighbor_radius_um",
            "msr_calibration_um",
            "coordinate_um_per_unit",
        }
        if method == "joint_truncated_rank_msr":
            test_keys.add("msr_shared_rank")
    if method == "cross_validation":
        test_keys.add("n_folds")
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
        "block_permutation",
        "binning",
        "joint_truncated_msr",
        "joint_truncated_rank_msr",
        "cross_validation",
        "full_retraining",
        "subsampling_test",
    }:
        extra = {
            "null_summary",
            "true_isodepth",
            "stat_covariate",
            "p_value_covariate",
            "pred_true",
            "pred_true_covariate",
            "true_isodepth_covariate",
            "pred_true_full_iso",
            "true_isodepth_full_iso",
        }
        if method_name == "cross_validation":
            extra |= {
                "train_mask",
                "test_mask",
                "n_folds",
                "fold_test_sizes",
                "fold_weights",
                "per_fold_true_loss",
                "per_fold_perm_loss",
                "per_fold_p_values",
                "per_fold_true_isodepth",
                "train_size",
                "test_size",
                "observed_test_loss",
            }
        if method_name == "block_permutation":
            extra.add("block_stats")
        if method_name == "binning":
            extra |= {
                "binning_summary",
                "binning_cell_counts",
            }
        if method_name in MSR_METHODS:
            extra.add("msr_surrogate_example")
        if method_name == "fourier_spectral_randomization":
            extra.add("fourier_surrogate_example")
        extra |= {
            "moran_skipped",
            "moran_skip_reason",
            "moran_true_mean",
            "moran_p_value",
            "moran_rank",
            "moran_neighbor_radius_um",
            "moran_mean_per_slot",
            "moran_null_mean_per_perm",
        "moran_n_slots",
        "moran_n_perms",
        "freedman_lane_obs_key",
        "freedman_lane_covariate_values",
        "freedman_lane_latent",
        "freedman_lane_pred",
    }
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


def _loss_history_arrays_from_model(model: Any) -> dict[str, np.ndarray]:
    """Pull serializable loss-history arrays out of a trained model (if present)."""
    if model is None:
        return {}
    metadata = get_training_metadata(model)
    payload: dict[str, np.ndarray] = {
        "n_reruns": np.asarray([int(metadata.get("n_reruns", 1))], dtype=np.int32),
    }
    loss_history = metadata.get("loss_history")
    if loss_history is not None:
        payload["loss_history"] = np.asarray(loss_history, dtype=np.float32).reshape(-1)
    per_slot = metadata.get("loss_history_per_slot")
    if per_slot is not None:
        arr = np.asarray(per_slot, dtype=np.float32)
        if arr.ndim == 2 and arr.size > 0:
            payload["loss_history_per_slot"] = arr
    return payload


def _save_loss_history_npz(
    model: Any,
    out_path: Path,
) -> Path | None:
    """Persist train-loss histories so loss/p-value curves can be regenerated offline."""
    payload = _loss_history_arrays_from_model(model)
    if "loss_history" not in payload and "loss_history_per_slot" not in payload:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


def _save_single_type_outputs(
    type_name: str,
    type_data: dict,
    dataset_meta: dict,
    type_dir: Path,
    *,
    method_name: str = "parallel_permutation",
    metric: str = "nll_gaussian_mse",
    decoder_df: int | None = None,
) -> dict[str, str]:
    """Generate the standard plot set for one cell type in separate mode."""
    S_c = np.asarray(type_data["S"], dtype=np.float32)
    A_c = np.asarray(type_data["A"], dtype=np.float32)
    subset_meta = dict(dataset_meta)
    subset_meta.pop("cell_type_labels", None)
    subset_meta.pop("cell_type_names", None)
    subset_meta.pop("n_cell_types", None)
    subset_meta.pop("cell_type_mode", None)
    if type_data.get("var_names") is not None:
        subset_meta["var_names"] = list(type_data["var_names"])
    if type_data.get("feature_space") is not None:
        subset_meta["feature_space"] = type_data["feature_space"]
    subset_dataset = DatasetBundle(S=S_c, A=A_c, meta=subset_meta).validate()

    subset_result = TestResult(
        method_name=method_name,
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
            "true_isodepth_covariate": type_data.get("true_isodepth_covariate"),
            "stat_covariate": type_data.get("stat_covariate"),
            "p_value_covariate": type_data.get("p_value_covariate"),
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
            "msr_surrogate_example": type_data.get("msr_surrogate_example"),
        },
    ).validate()

    type_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    p = save_isodepth_triptych(subset_dataset, subset_result, type_dir / f"{type_name}_isodepth.png")
    if p is not None:
        paths["isodepth_triptych_plot"] = str(p)

    if type_data.get("freedman_lane_pred") is not None:
        fl_spatial = np.asarray(type_data.get("S_original", S_c), dtype=np.float32)
        fl_subset_dataset = DatasetBundle(S=fl_spatial, A=A_c, meta=subset_meta).validate()
        fl_subset_result = TestResult(
            method_name=method_name,
            metric=metric,
            p_value=float(type_data["p_value"]),
            stat_true=float(type_data["stat_true"]),
            stat_perm=np.asarray(type_data["stat_perm"], dtype=np.float64),
            runtime_sec=0.0,
            n_cells=int(fl_spatial.shape[0]),
            n_genes=int(A_c.shape[1]),
            config={},
            artifacts={
                "freedman_lane_obs_key": type_data.get("freedman_lane_obs_key"),
                "freedman_lane_covariate_values": type_data.get("freedman_lane_covariate_values"),
                "freedman_lane_pred": type_data.get("freedman_lane_pred"),
            },
        ).validate()
        p_fl = save_freedman_lane_covariate_plot(
            fl_subset_dataset,
            fl_subset_result,
            type_dir / f"{type_name}_freedman_lane_covariate.png",
        )
        if p_fl is not None:
            paths["freedman_lane_covariate_plot"] = str(p_fl)

    p_dist = save_metric_distribution_plot(subset_result, type_dir / f"{type_name}_metric_distribution.png")
    paths["metric_distribution_plot"] = str(p_dist)

    p_loss = save_loss_curve_plot(
        subset_result,
        type_dir / f"{type_name}_loss_curve.png",
        title=f"{type_name} — training loss by epoch",
    )
    if p_loss is not None:
        paths["loss_curve_plot"] = str(p_loss)

    if type_data.get("moran_true_mean") is not None and not type_data.get("moran_skipped"):
        moran_result = TestResult(
            method_name=method_name,
            metric=metric,
            p_value=float(type_data["p_value"]),
            stat_true=float(type_data["stat_true"]),
            stat_perm=np.asarray(type_data["stat_perm"], dtype=np.float64),
            runtime_sec=0.0,
            n_cells=int(S_c.shape[0]),
            n_genes=int(A_c.shape[1]),
            config={},
            artifacts={
                "moran_true_mean": type_data.get("moran_true_mean"),
                "moran_null_mean_per_perm": type_data.get("moran_null_mean_per_perm"),
                "moran_p_value": type_data.get("moran_p_value"),
                "moran_rank": type_data.get("moran_rank"),
                "moran_neighbor_radius_um": type_data.get("moran_neighbor_radius_um"),
            },
        ).validate()
        p_moran = save_moran_distribution_plot(
            moran_result,
            type_dir / f"{type_name}_moran_distribution.png",
        )
        if p_moran is not None:
            paths["moran_distribution_plot"] = str(p_moran)

    if method_name in MSR_METHODS:
        p_msr = save_msr_surrogate_example_plot(
            subset_dataset,
            subset_result,
            type_dir / f"{type_name}_msr_surrogate_example.png",
            n_genes=5,
        )
        if p_msr is not None:
            paths["msr_surrogate_example_plot"] = str(p_msr)

    if type_data.get("per_fold_true_isodepth") is not None:
        fold_isodepths = np.asarray(type_data["per_fold_true_isodepth"], dtype=np.float32)
        p_fold_iso = save_cross_validation_fold_isodepth_grid(
            subset_dataset,
            [np.asarray(row, dtype=np.float32) for row in fold_isodepths],
            type_dir / f"{type_name}_cv_fold_isodepths.png",
            fold_test_sizes=np.asarray(type_data.get("fold_test_sizes", []), dtype=np.int64),
        )
        if p_fold_iso is not None:
            paths["cross_validation_fold_isodepth_grid_plot"] = str(p_fold_iso)

        cv_subset_result = TestResult(
            method_name="cross_validation",
            metric=metric,
            p_value=float(type_data["p_value"]),
            stat_true=float(type_data["stat_true"]),
            stat_perm=np.asarray(type_data["stat_perm"], dtype=np.float64),
            runtime_sec=0.0,
            n_cells=int(S_c.shape[0]),
            n_genes=int(A_c.shape[1]),
            config={},
            artifacts={
                "per_fold_true_loss": type_data.get("per_fold_true_loss"),
                "per_fold_perm_loss": type_data.get("per_fold_perm_loss"),
                "per_fold_p_values": type_data.get("per_fold_p_values"),
            },
        ).validate()
        p_fold_dist = save_cross_validation_per_fold_metric_distributions(
            cv_subset_result,
            type_dir / f"{type_name}_cv_per_fold_metric_distributions.png",
        )
        if p_fold_dist is not None:
            paths["cross_validation_per_fold_metric_distribution_plot"] = str(p_fold_dist)

    iso_raw = type_data.get("true_isodepth")
    cov_raw = type_data.get("true_isodepth_covariate")

    # Gene expression vs isodepth/covariate summary plot (quantile bins).
    if iso_raw is not None:
        cov_label = str(dataset_meta.get("covariate_obs_key") or "covariate")
        fake_result: TestResult = TestResult(
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
                "true_isodepth": iso_raw,
                "true_isodepth_covariate": cov_raw,
                "pred_true": type_data.get("pred_true"),
                "pred_true_covariate": type_data.get("pred_true_covariate"),
            },
        ).validate()
        _save_gene_expression_summary_plots(
            subset_dataset, fake_result, type_name, type_dir, paths,
            isodepth_label="Isodepth",
            covariate_label=cov_label,
            decoder_df=decoder_df,
        )

    iso_raw = type_data.get("true_isodepth")
    if iso_raw is not None:
        npz_path = type_dir / f"{type_name}_isodepths.npz"
        npz_payload: dict[str, np.ndarray] = {
            "true_isodepth": np.asarray(iso_raw, dtype=np.float32).reshape(-1),
            "A": A_c,
            "S": S_c,
        }
        pred_true = type_data.get("pred_true")
        if pred_true is not None:
            npz_payload["pred_true"] = np.asarray(pred_true, dtype=np.float32)
        cov_raw = type_data.get("true_isodepth_covariate")
        if cov_raw is not None:
            npz_payload["true_isodepth_covariate"] = np.asarray(
                cov_raw, dtype=np.float32
            ).reshape(-1)
        pred_cov = type_data.get("pred_true_covariate")
        if pred_cov is not None:
            npz_payload["pred_true_covariate"] = np.asarray(pred_cov, dtype=np.float32)
        msr_surrogate = type_data.get("msr_surrogate_example")
        if msr_surrogate is not None:
            npz_payload["msr_surrogate_example"] = np.asarray(
                msr_surrogate, dtype=np.float32
            )
        moran_i_slots = type_data.get("moran_i_per_gene_per_slot")
        if moran_i_slots is not None:
            npz_payload["moran_i_per_gene_per_slot"] = np.asarray(
                moran_i_slots, dtype=np.float64
            )
        model_c = type_data.get("model")
        npz_payload.update(_loss_history_arrays_from_model(model_c))
        np.savez(npz_path, **npz_payload)
        paths["isodepths_npz"] = str(npz_path)

    model_c = type_data.get("model")
    loss_hist_path = _save_loss_history_npz(
        model_c,
        type_dir / f"{type_name}_loss_history.npz",
    )
    if loss_hist_path is not None:
        paths["loss_history_npz"] = str(loss_hist_path)

    training_metadata = getattr(model_c, "training_metadata", None) if model_c is not None else None
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


def _dataset_for_gene_expression_plots(dataset: DatasetBundle) -> DatasetBundle:
    """Use cell-type residuals for together-mode gene-expression summary plots."""
    if dataset.meta.get("cell_type_mode") != "together":
        return dataset
    cell_type_labels = dataset.meta.get("cell_type_labels")
    if cell_type_labels is None:
        return dataset
    residuals = celltype_expression_residuals(
        dataset.A,
        np.asarray(cell_type_labels, dtype=np.int64),
        n_cell_types=int(dataset.meta.get("n_cell_types", 0)),
    )
    return replace(dataset, A=residuals)


def _save_gene_expression_summary_plots(
    dataset: DatasetBundle,
    result: TestResult,
    run_name: str,
    out_dir: Path,
    artifact_paths: dict[str, str],
    *,
    isodepth_label: str = "Isodepth",
    covariate_label: str | None = None,
    decoder_df: int | None = None,
) -> None:
    """Generate gene-expression-vs-coordinate summary plots from saved isodepths.

    Uses ``true_isodepth`` (and optionally ``true_isodepth_covariate``) from
    ``result.artifacts``.  When ``pred_true`` / ``pred_true_covariate`` are
    also present in ``result.artifacts`` (either from training or a rerun), the
    decoder predictions are passed to the plot so the fit curve reflects the
    actual learned (possibly non-linear) decoder rather than a polynomial
    approximation.

    When ``decoder_df`` is set (1 for linear, 2 for quadratic …) an F-test is
    run for every gene and significant genes (BH q < 0.05) are written to CSVs
    beside the PNG.
    """
    iso_raw = result.artifacts.get("true_isodepth")
    if iso_raw is None:
        return
    plot_dataset = _dataset_for_gene_expression_plots(dataset)
    iso = np.asarray(iso_raw, dtype=np.float64).reshape(-1)

    pred_iso_raw = result.artifacts.get("pred_true")
    pred_cov_raw = result.artifacts.get("pred_true_covariate")
    pred_iso = np.asarray(pred_iso_raw, dtype=np.float64) if pred_iso_raw is not None else None
    pred_cov = np.asarray(pred_cov_raw, dtype=np.float64) if pred_cov_raw is not None else None

    cov_raw = result.artifacts.get("true_isodepth_covariate")
    cov_lbl = covariate_label or "Covariate"
    if cov_raw is not None:
        cov = np.asarray(cov_raw, dtype=np.float64).reshape(-1)
        try:
            p = save_gene_expression_vs_coordinates_comparison(
                plot_dataset, iso, cov,
                out_dir / f"{run_name}_gene_expression_vs_coordinates.png",
                isodepth_label=isodepth_label,
                covariate_label=cov_lbl,
                pred_isodepth=pred_iso,
                pred_covariate=pred_cov,
                decoder_df=decoder_df,
            )
            artifact_paths["gene_expression_plot"] = str(p)
            stem = p.parent / p.stem
            companion_artifacts = {
                "gene_expression_correlation_distribution_plot": Path(f"{stem}_correlation_distribution.png"),
                "gene_expression_residual_ratio_distribution_plot": Path(f"{stem}_residual_ratio_distribution.png"),
                "gene_expression_residual_ratio_rankings_csv": Path(f"{stem}_residual_ratio_rankings.csv"),
            }
            for key, path in companion_artifacts.items():
                if path.exists():
                    artifact_paths[key] = str(path)
        except Exception:
            pass
    else:
        try:
            p = save_gene_expression_vs_isodepth_plot(
                plot_dataset, iso,
                out_dir / f"{run_name}_gene_expression_vs_isodepth.png",
                coord_label=isodepth_label,
                decoder_preds=pred_iso,
                decoder_df=decoder_df,
            )
            artifact_paths["gene_expression_plot"] = str(p)
            corr_path = p.parent / f"{p.stem}_correlation_distribution.png"
            if corr_path.exists():
                artifact_paths["gene_expression_correlation_distribution_plot"] = str(corr_path)
        except Exception:
            pass


def _decoder_df_from_config(decoder: str | None) -> int | None:
    """Return the F-test model degrees of freedom for the given decoder type.

    Only well-defined for parametric decoders:
      ``"linear"``    → 1 (y = w·z + b; one slope parameter)
      ``"quadratic"`` → 2 (y = w₁·z + w₂·z² + b; two slope parameters)
    All other decoders (``"nn"``, None, …) return None → F-test skipped.
    """
    if decoder == "linear":
        return 1
    if decoder == "quadratic":
        return 2
    return None


def _resolve_covariate_label(run_config: RunConfig) -> str:
    """Human-readable label for the covariate coordinate axis."""
    cov = run_config.test.covariate
    if cov is None:
        return "Covariate"
    if cov.type == "midline":
        return "Midline"
    if cov.type == TOTAL_COUNTS_COVARIATE:
        return "Log total counts"
    if cov.is_obs_key and cov.type:
        return str(cov.type).capitalize()
    return "Covariate"


def _save_block_permutation_overlay_artifact(
    S_true_raw: np.ndarray,
    s_permuted_slot1_raw: np.ndarray | None,
    block_ids_true: np.ndarray | None,
    out_path: Path,
    *,
    run_name: str,
    radius_units: float | None,
    block_shape: str = "hexagon",
) -> str | None:
    overlay_path = save_block_permutation_overlay(
        S_true_raw,
        s_permuted_slot1_raw,
        block_ids_true,
        out_path,
        run_name=run_name,
        radius_units=radius_units,
        block_shape=block_shape,
    )
    return str(overlay_path) if overlay_path is not None else None


def _save_permutation_null_comparison_artifact(
    S_true_raw: np.ndarray,
    s_permuted_slot1_raw: np.ndarray | None,
    A: np.ndarray,
    out_path: Path,
    *,
    seed: int,
    run_name: str,
) -> str | None:
    if s_permuted_slot1_raw is None:
        return None
    comparison_path = save_permutation_null_comparison(
        S_true_raw,
        np.asarray(s_permuted_slot1_raw, dtype=np.float32),
        A,
        out_path,
        seed=int(seed),
        run_name=run_name,
    )
    return str(comparison_path) if comparison_path is not None else None


def save_standardized_outputs(
    dataset: DatasetBundle,
    result: TestResult,
    run_config: RunConfig,
) -> tuple[dict[str, Any], Path]:
    out_root = Path(run_config.output.out_dir)
    out_dir = out_root / run_config.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.method_name == "binning":
        binned_dataset = result.artifacts.get("binned_dataset")
        if isinstance(binned_dataset, DatasetBundle):
            dataset = binned_dataset

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

    if _is_cosmx_subset_config(run_config.data) or _LEGACY_RUN_RE.match(run_config.output.run_name):
        try:
            region_context_path = save_cosmx_region_context_plot(
                run_config.data,
                out_dir / f"{run_config.output.run_name}_region_context.png",
                run_name=run_config.output.run_name,
            )
            artifact_paths["region_context_plot"] = str(region_context_path)
        except Exception as exc:
            import warnings
            warnings.warn(f"Could not save CosMx region context plot: {exc}")

    synthetic_true_curve_path = save_synthetic_true_curve_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_true_curve.png",
    )
    if synthetic_true_curve_path is not None:
        artifact_paths["synthetic_true_curve_plot"] = str(synthetic_true_curve_path)

    synthetic_kernel_paths = save_synthetic_kernel_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_kernel_diagnostics.png",
    )
    if synthetic_kernel_paths is not None:
        for key, path in synthetic_kernel_paths.items():
            artifact_paths[key] = str(path)

    isodepth_plot_path = save_isodepth_triptych(
        dataset,
        result,
        out_dir / f"{run_config.output.run_name}_isodepth.png",
    )
    if isodepth_plot_path is not None:
        artifact_paths["isodepth_triptych_plot"] = str(isodepth_plot_path)

    freedman_lane_plot_path = save_freedman_lane_covariate_plot(
        dataset,
        result,
        out_dir / f"{run_config.output.run_name}_freedman_lane_covariate.png",
    )
    if freedman_lane_plot_path is not None:
        artifact_paths["freedman_lane_covariate_plot"] = str(freedman_lane_plot_path)

    covariate_whitening_plot_path = save_covariate_whitening_spatial_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_covariate_whitening.png",
    )
    if covariate_whitening_plot_path is not None:
        artifact_paths["covariate_whitening_plot"] = str(covariate_whitening_plot_path)

    distribution_plot_path = save_metric_distribution_plot(
        result,
        out_dir / f"{run_config.output.run_name}_metric_distribution.png",
    )
    artifact_paths["metric_distribution_plot"] = str(distribution_plot_path)

    loss_curve_plot_path = save_loss_curve_plot(
        result,
        out_dir / f"{run_config.output.run_name}_loss_curve.png",
        title=f"{run_config.output.run_name} — training loss by epoch",
    )
    if loss_curve_plot_path is not None:
        artifact_paths["loss_curve_plot"] = str(loss_curve_plot_path)

    moran_plot_path = save_moran_distribution_plot(
        result,
        out_dir / f"{run_config.output.run_name}_moran_distribution.png",
    )
    if moran_plot_path is not None:
        artifact_paths["moran_distribution_plot"] = str(moran_plot_path)

    if result.method_name == "block_permutation":
        S_true_raw = raw_coordinates_from_standardized(dataset.S, dataset.meta)
        block_overlay_path = _save_block_permutation_overlay_artifact(
            S_true_raw,
            result.artifacts.get("s_permuted_slot1_raw"),
            result.artifacts.get("block_ids_true"),
            out_dir / f"{run_config.output.run_name}_block_permutation_overlay.png",
            run_name=run_config.output.run_name,
            radius_units=result.artifacts.get("block_radius_units"),
            block_shape=run_config.test.block_shape,
        )
        if block_overlay_path is not None:
            artifact_paths["block_permutation_overlay_plot"] = block_overlay_path
        if run_config.test.save_permutation_null_comparison:
            null_comparison_path = _save_permutation_null_comparison_artifact(
                S_true_raw,
                result.artifacts.get("s_permuted_slot1_raw"),
                np.asarray(dataset.A, dtype=np.float32),
                out_dir / f"{run_config.output.run_name}_permutation_null_comparison.png",
                seed=int(run_config.test.seed),
                run_name=run_config.output.run_name,
            )
            if null_comparison_path is not None:
                artifact_paths["permutation_null_comparison_plot"] = null_comparison_path

    if result.method_name in MSR_METHODS:
        surr_plot_path = save_msr_surrogate_example_plot(
            dataset,
            result,
            out_dir / f"{run_config.output.run_name}_msr_surrogate_example.png",
        )
        if surr_plot_path is not None:
            artifact_paths["msr_surrogate_example_plot"] = str(surr_plot_path)

    if result.method_name == "fourier_spectral_randomization":
        fourier_surr_plot_path = save_fourier_surrogate_example_plot(
            dataset,
            result,
            out_dir / f"{run_config.output.run_name}_fourier_surrogate_example.png",
        )
        if fourier_surr_plot_path is not None:
            artifact_paths["fourier_surrogate_example_plot"] = str(fourier_surr_plot_path)

    if result.method_name == "cross_validation":
        fold_isodepths = result.artifacts.get("per_fold_true_isodepth")
        if fold_isodepths is not None:
            fold_grid_path = save_cross_validation_fold_isodepth_grid(
                dataset,
                [np.asarray(row, dtype=np.float32) for row in np.asarray(fold_isodepths)],
                out_dir / f"{run_config.output.run_name}_cv_fold_isodepths.png",
                fold_test_sizes=np.asarray(
                    result.artifacts.get("fold_test_sizes", []),
                    dtype=np.int64,
                ),
            )
            if fold_grid_path is not None:
                artifact_paths["cross_validation_fold_isodepth_grid_plot"] = str(fold_grid_path)

        per_fold_dist_path = save_cross_validation_per_fold_metric_distributions(
            result,
            out_dir / f"{run_config.output.run_name}_cv_per_fold_metric_distributions.png",
        )
        if per_fold_dist_path is not None:
            artifact_paths["cross_validation_per_fold_metric_distribution_plot"] = str(
                per_fold_dist_path
            )

    _save_gene_expression_summary_plots(
        dataset, result, run_config.output.run_name, out_dir, artifact_paths,
        covariate_label=_resolve_covariate_label(run_config),
        decoder_df=_decoder_df_from_config(getattr(run_config.test, "decoder", None)),
    )

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

    celltype_plot_path = save_celltype_or_spatial_split_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_celltype.png",
    )
    if celltype_plot_path is not None:
        artifact_paths["celltype_dataset_plot"] = str(celltype_plot_path)

    obs_diag_paths, _ = _save_obs_numeric_filter_diagnostics(run_config, out_dir)
    artifact_paths.update(obs_diag_paths)

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

    loss_hist_path = _save_loss_history_npz(
        model,
        out_dir / f"{run_config.output.run_name}_loss_history.npz",
    )
    if loss_hist_path is not None:
        artifact_paths["loss_history_npz"] = str(loss_hist_path)

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

    celltype_overview_path = save_celltype_or_spatial_split_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_celltype.png",
    )

    per_type_summaries: dict[str, dict[str, Any]] = {}
    per_type_artifact_paths: dict[str, dict[str, str]] = {}

    _ct_decoder_df = _decoder_df_from_config(getattr(run_config.test, "decoder", None))

    obs_diag_paths, obs_hist_by_type = _save_obs_numeric_filter_diagnostics(
        run_config,
        out_dir,
        cell_type_names=cell_type_names,
    )

    cov_values_all = dataset.meta.get("covariate_whitening_values")
    cov_obs_key = dataset.meta.get("covariate_whitening_obs_key")
    full_labels = dataset.meta.get("cell_type_labels")
    full_type_names = dataset.meta.get("cell_type_names")

    for type_name in cell_type_names:
        type_data = per_type_results[type_name]
        safe_name = type_name.replace(" ", "_").replace("/", "_")
        type_dir = out_dir / safe_name

        type_artifact_paths = _save_single_type_outputs(
            safe_name, type_data, dataset.meta, type_dir,
            method_name=result.method_name,
            metric=result.metric,
            decoder_df=_ct_decoder_df,
        )
        if result.method_name == "block_permutation" and type_data.get("S_raw") is not None:
            overlay_path = _save_block_permutation_overlay_artifact(
                np.asarray(type_data["S_raw"], dtype=np.float32),
                type_data.get("s_permuted_slot1_raw"),
                type_data.get("block_ids_true"),
                type_dir / f"{safe_name}_block_permutation_overlay.png",
                run_name=safe_name,
                radius_units=type_data.get("block_radius_units"),
                block_shape=run_config.test.block_shape,
            )
            if overlay_path is not None:
                type_artifact_paths["block_permutation_overlay_plot"] = overlay_path
            if run_config.test.save_permutation_null_comparison:
                null_comparison_path = _save_permutation_null_comparison_artifact(
                    np.asarray(type_data["S_raw"], dtype=np.float32),
                    type_data.get("s_permuted_slot1_raw"),
                    np.asarray(type_data["A"], dtype=np.float32),
                    type_dir / f"{safe_name}_permutation_null_comparison.png",
                    seed=int(run_config.test.seed),
                    run_name=safe_name,
                )
                if null_comparison_path is not None:
                    type_artifact_paths["permutation_null_comparison_plot"] = null_comparison_path
        if type_name in obs_hist_by_type:
            type_artifact_paths["obs_numeric_filter_histogram_plot"] = obs_hist_by_type[
                type_name
            ]
        if cov_values_all is not None and cov_obs_key and full_labels is not None and full_type_names:
            if type_name in full_type_names:
                type_idx = list(full_type_names).index(type_name)
                type_mask = np.asarray(full_labels) == type_idx
                cov_for_type = np.asarray(cov_values_all)[type_mask]
                if cov_for_type.shape[0] == int(type_data["n_cells"]):
                    cov_plot_path = save_single_type_covariate_plot(
                        np.asarray(type_data["S"], dtype=np.float32),
                        cov_for_type,
                        type_dir / f"{safe_name}_covariate_whitening.png",
                        covariate_label=cov_obs_key,
                        type_name=type_name,
                    )
                    if cov_plot_path is not None:
                        type_artifact_paths["covariate_whitening_plot"] = str(cov_plot_path)
        per_type_artifact_paths[type_name] = type_artifact_paths
        type_summary: dict[str, Any] = {
            "p_value": float(type_data["p_value"]),
            "stat_true": float(type_data["stat_true"]),
            "stat_perm": [float(x) for x in np.asarray(type_data["stat_perm"]).tolist()],
            "n_cells": int(type_data["n_cells"]),
            "perm_summary": summarize_metric_distribution(type_data["stat_perm"]),
            "artifact_paths": type_artifact_paths,
        }
        if type_data.get("stat_covariate") is not None:
            type_summary["stat_covariate"] = float(type_data["stat_covariate"])
        if type_data.get("p_value_covariate") is not None:
            type_summary["p_value_covariate"] = float(type_data["p_value_covariate"])
        per_type_summaries[type_name] = type_summary

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

    var_names = dataset.meta.get("var_names")
    gene_names = (
        [str(var_names[i]) for i in range(dataset.n_genes)]
        if var_names is not None else [f"gene_{i}" for i in range(dataset.n_genes)]
    )
    combined_residual_csv_path, combined_residual_plot_path = save_combined_celltype_residual_ratio_outputs(
        per_type_results,
        cell_type_names,
        gene_names,
        out_dir / f"{run_config.output.run_name}_piecewise_residual_ratio_rankings.csv",
        out_dir / f"{run_config.output.run_name}_piecewise_residual_ratio_distribution.png",
        coord_label="Isodepth",
        covariate_label=_resolve_covariate_label(run_config),
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
    top_level_artifacts.update(obs_diag_paths)
    if combined_residual_csv_path is not None:
        top_level_artifacts["piecewise_residual_ratio_rankings_csv"] = str(combined_residual_csv_path)
    if combined_residual_plot_path is not None:
        top_level_artifacts["piecewise_residual_ratio_distribution_plot"] = str(combined_residual_plot_path)

    for key in (
        "freedman_lane_obs_key",
        "freedman_lane_covariate_values",
        "freedman_lane_latent",
        "freedman_lane_pred",
    ):
        if key in result.artifacts:
            top_level_artifacts[key] = result.artifacts[key]

    freedman_lane_plot_path = save_freedman_lane_covariate_plot(
        dataset,
        result,
        out_dir / f"{run_config.output.run_name}_freedman_lane_covariate.png",
    )
    if freedman_lane_plot_path is not None:
        top_level_artifacts["freedman_lane_covariate_plot"] = str(freedman_lane_plot_path)

    covariate_whitening_plot_path = save_covariate_whitening_spatial_plot(
        dataset,
        out_dir / f"{run_config.output.run_name}_covariate_whitening.png",
    )
    if covariate_whitening_plot_path is not None:
        top_level_artifacts["covariate_whitening_plot"] = str(covariate_whitening_plot_path)

    if result.method_name == "block_permutation":
        S_true_raw = raw_coordinates_from_standardized(dataset.S, dataset.meta)
        block_overlay_path = _save_block_permutation_overlay_artifact(
            S_true_raw,
            result.artifacts.get("s_permuted_slot1_raw"),
            result.artifacts.get("block_ids_true"),
            out_dir / f"{run_config.output.run_name}_block_permutation_overlay.png",
            run_name=run_config.output.run_name,
            radius_units=result.artifacts.get("block_radius_units"),
            block_shape=run_config.test.block_shape,
        )
        if block_overlay_path is not None:
            top_level_artifacts["block_permutation_overlay_plot"] = block_overlay_path
        if run_config.test.save_permutation_null_comparison:
            null_comparison_path = _save_permutation_null_comparison_artifact(
                S_true_raw,
                result.artifacts.get("s_permuted_slot1_raw"),
                np.asarray(dataset.A, dtype=np.float32),
                out_dir / f"{run_config.output.run_name}_permutation_null_comparison.png",
                seed=int(run_config.test.seed),
                run_name=run_config.output.run_name,
            )
            if null_comparison_path is not None:
                top_level_artifacts["permutation_null_comparison_plot"] = null_comparison_path

    payload = result.to_json_dict(
        config=_compact_run_config(run_config),
        artifacts=_json_compatible(top_level_artifacts),
    )

    result_path = out_dir / f"{run_config.output.run_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload, result_path
