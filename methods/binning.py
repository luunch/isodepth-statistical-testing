"""Pseudospot binning transform for the binning permutation test."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from data import raw_coordinates_from_standardized
from data.schemas import DatasetBundle, TestConfig
from data.transforms import apply_expression_transforms
from methods.block_permutation import (
    assign_block_ids,
    block_ids_to_axial_qr,
    block_ids_to_square_ij,
    hex_center_coord,
    resolve_um_per_unit,
    square_center_coord,
)
from methods.metrics import canonicalize_metric_name


def _bin_radius_from_center_distance(distance_um: float, shape: str) -> float:
    if shape == "square":
        return float(distance_um) / 2.0
    if shape == "hexagon":
        return float(distance_um) / math.sqrt(3.0)
    raise ValueError(f"Unsupported bin shape {shape!r}")


def _centers_for_block_ids(
    block_ids: np.ndarray,
    radius_um: float,
    shape: str,
    origin_xy_um: tuple[float, float],
) -> np.ndarray:
    bids = np.asarray(block_ids, dtype=np.int64)
    if shape == "square":
        ix, iy = block_ids_to_square_ij(bids)
        cx, cy = square_center_coord(ix, iy, radius_um)
    else:
        q, r = block_ids_to_axial_qr(bids)
        cx, cy = hex_center_coord(q, r, radius_um)
    centers = np.column_stack([cx, cy]).astype(np.float64)
    centers[:, 0] += float(origin_xy_um[0])
    centers[:, 1] += float(origin_xy_um[1])
    return centers.astype(np.float32)


def _bin_origin_um(
    S_um: np.ndarray,
    radius_um: float,
    shape: str,
    *,
    jitter: bool,
    seed: int,
) -> tuple[float, float]:
    lower_left = np.asarray(S_um, dtype=np.float64).min(axis=0)
    if not jitter:
        return float(lower_left[0]), float(lower_left[1])

    if shape == "square":
        period_x = period_y = 2.0 * float(radius_um)
    else:
        period_x = math.sqrt(3.0) * float(radius_um)
        period_y = 1.5 * float(radius_um)
    rng = np.random.default_rng(int(seed))
    return (
        float(lower_left[0] - rng.uniform(0.0, period_x)),
        float(lower_left[1] - rng.uniform(0.0, period_y)),
    )


def _zscore_coordinates_if_needed(S_raw: np.ndarray, source_meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    meta_update: dict[str, Any] = {}
    if source_meta.get("coordinate_standardization") != "zscore":
        return np.asarray(S_raw, dtype=np.float32), meta_update

    s = np.asarray(S_raw, dtype=np.float32)
    mean = s.mean(axis=0)
    std = s.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    meta_update["coordinate_standardization"] = "zscore"
    meta_update["coord_mean"] = np.asarray(mean, dtype=np.float32)
    meta_update["coord_std"] = np.asarray(safe_std, dtype=np.float32)
    return np.asarray((s - mean) / safe_std, dtype=np.float32), meta_update


def _aggregate_vectors(
    values: np.ndarray | None,
    inverse: np.ndarray,
    n_groups: int,
) -> np.ndarray | None:
    if values is None:
        return None
    v = np.asarray(values, dtype=np.float32)
    out = np.zeros((n_groups,) + v.shape[1:], dtype=np.float64)
    np.add.at(out, inverse, v.astype(np.float64, copy=False))
    counts = np.bincount(inverse, minlength=n_groups).astype(np.float64)
    reshape = (n_groups,) + (1,) * (out.ndim - 1)
    return np.asarray(out / counts.reshape(reshape), dtype=np.float32)


def _expression_preprocessing_from_meta(dataset: DatasetBundle, config: TestConfig) -> dict[str, Any]:
    pp = dict(dataset.meta.get("binning_preprocessing") or {})
    metric = canonicalize_metric_name(config.metric)
    if metric == "nll_poisson_mse":
        pp.setdefault("normalize_total", False)
        pp.setdefault("log1p", False)
        pp.setdefault("standardize_expression", False)
    else:
        pp.setdefault("normalize_total", bool(dataset.meta.get("normalize_total", True)))
        pp.setdefault("log1p", bool(dataset.meta.get("log1p", True)))
        pp.setdefault("standardize_expression", bool(dataset.meta.get("standardize_expression", True)))
    pp.setdefault("min_cells_per_gene", int(dataset.meta.get("min_cells_per_gene", 0)))
    pp.setdefault("top_var_genes", int(dataset.meta.get("top_var_genes", 0)))
    pp.setdefault("q", dataset.meta.get("q"))
    pp.setdefault("seed", int(dataset.meta.get("seed", config.seed)))
    return pp


def _transform_binned_expression(
    A_sum: np.ndarray,
    dataset: DatasetBundle,
    config: TestConfig,
) -> tuple[np.ndarray, list[str] | None, str | None, dict[str, Any]]:
    pp = _expression_preprocessing_from_meta(dataset, config)
    if bool(pp.get("normalize_total")) and np.any(np.asarray(A_sum) < 0):
        raise ValueError(
            "test.method='binning' expects non-negative raw counts before CPM normalization. "
            "For h5ad configs, run through run_permutation.py so binning can load counts "
            "before data.normalize_total/log1p, or disable CPM-style binning preprocessing."
        )

    transformed, transform_meta = apply_expression_transforms(
        A_sum,
        min_cells_per_gene=int(pp.get("min_cells_per_gene", 0)),
        normalize_total=bool(pp.get("normalize_total", True)),
        log1p=bool(pp.get("log1p", True)),
        standardize_expression=bool(pp.get("standardize_expression", True)),
        q=pp.get("q"),
        seed=int(pp.get("seed", config.seed)),
        return_metadata=True,
    )
    var_names = dataset.meta.get("var_names")
    if "feature_names" in transform_meta:
        new_var_names = [str(x) for x in transform_meta["feature_names"]]
    elif var_names is not None:
        keep = np.asarray(transform_meta["gene_keep_mask"], dtype=bool)
        new_var_names = [str(x) for x in np.asarray(var_names, dtype=object)[keep]]
    else:
        new_var_names = None
    return (
        np.asarray(transformed, dtype=np.float32),
        new_var_names,
        str(transform_meta.get("representation", "gene_expression")),
        transform_meta,
    )


def bin_dataset_to_pseudospots(
    dataset: DatasetBundle,
    config: TestConfig,
) -> tuple[DatasetBundle, dict[str, Any]]:
    """Aggregate rows into occupied square/hex pseudospots.

    Cell-type modes keep cell types separate by using ``(cell_type, bin_id)`` as
    the grouping key.  Plain mode uses only the spatial bin.
    """
    dataset.validate()
    config.validate()
    shape = "hexagon" if str(config.bin_shape) == "hexagonal" else str(config.bin_shape)
    spot_distance_um = float(config.bin_spot_distance_um)
    um_per_unit = resolve_um_per_unit(
        config.coordinate_um_per_unit,
        dataset.meta.get("coordinate_um_per_unit"),
    )
    radius_um = _bin_radius_from_center_distance(spot_distance_um, shape)

    S_raw = raw_coordinates_from_standardized(dataset.S, dataset.meta)
    S_um = np.asarray(S_raw, dtype=np.float64) * float(um_per_unit)
    origin_xy_um = _bin_origin_um(
        S_um,
        radius_um,
        shape,
        jitter=bool(getattr(config, "block_jitter", False)),
        seed=int(config.seed),
    )
    block_ids = assign_block_ids(S_um, radius_um, origin_xy_um, block_shape=shape)

    cell_type_mode = dataset.meta.get("cell_type_mode", "none")
    cell_type_labels = dataset.meta.get("cell_type_labels")
    if cell_type_mode in {"together", "separate"} and cell_type_labels is not None:
        labels = np.asarray(cell_type_labels, dtype=np.int64)
        keys = np.column_stack([labels, block_ids.astype(np.int64)])
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        binned_labels = unique_keys[:, 0].astype(np.int64)
        binned_block_ids = unique_keys[:, 1].astype(np.int64)
    else:
        binned_block_ids, inverse = np.unique(block_ids.astype(np.int64), return_inverse=True)
        binned_labels = None

    n_bins = int(binned_block_ids.shape[0])
    A_raw = np.asarray(dataset.A, dtype=np.float32)
    A_sum = np.zeros((n_bins, A_raw.shape[1]), dtype=np.float64)
    np.add.at(A_sum, inverse, A_raw.astype(np.float64, copy=False))
    A_sum = np.asarray(A_sum, dtype=np.float32)
    bin_counts = np.bincount(inverse, minlength=n_bins).astype(np.int64)

    centers_um = _centers_for_block_ids(binned_block_ids, radius_um, shape, origin_xy_um)
    centers_raw = np.asarray(centers_um / float(um_per_unit), dtype=np.float32)
    S_binned, coord_meta = _zscore_coordinates_if_needed(centers_raw, dataset.meta)

    meta = dict(dataset.meta)
    for stale_key in (
        "plot_cell_type_labels",
        "plot_cell_type_names",
        "synthetic_true_curve",
        "kernel_noise_sample",
    ):
        meta.pop(stale_key, None)
    meta.update(coord_meta)
    meta["binning"] = {
        "shape": shape,
        "spot_distance_um": spot_distance_um,
        "radius_um": float(radius_um),
        "coordinate_um_per_unit": float(um_per_unit),
        "origin_x_um": float(origin_xy_um[0]),
        "origin_y_um": float(origin_xy_um[1]),
        "jitter": bool(getattr(config, "block_jitter", False)),
        "original_n_cells": int(dataset.n_cells),
        "n_pseudospots": n_bins,
        "mean_cells_per_pseudospot": float(bin_counts.mean()) if n_bins else 0.0,
        "median_cells_per_pseudospot": float(np.median(bin_counts)) if n_bins else 0.0,
        "min_cells_per_pseudospot": int(bin_counts.min()) if n_bins else 0,
        "max_cells_per_pseudospot": int(bin_counts.max()) if n_bins else 0,
    }
    meta["binning_cell_counts"] = bin_counts
    meta["binning_block_ids"] = binned_block_ids

    if binned_labels is not None:
        meta["cell_type_labels"] = binned_labels
        meta["cell_type_mode"] = cell_type_mode
        if "cell_type_names" in dataset.meta:
            meta["cell_type_names"] = list(dataset.meta["cell_type_names"])
        if "n_cell_types" in dataset.meta:
            meta["n_cell_types"] = int(dataset.meta["n_cell_types"])

    for key in ("covariate_values", "covariate_whitening_values", "calicost_tumor_proportion"):
        aggregated = _aggregate_vectors(dataset.meta.get(key), inverse, n_bins)
        if aggregated is not None:
            meta[key] = aggregated

    if cell_type_mode == "separate" and binned_labels is not None:
        # Existing separate-mode training will apply per-type preprocessing from
        # raw counts.  Re-seed it so binning and original runs remain reproducible
        # but distinct when both are launched from the same base config.
        pp = dict(meta.get("separate_preprocessing") or {})
        if not pp:
            pp = _expression_preprocessing_from_meta(dataset, config)
        pp["seed"] = int(pp.get("seed", config.seed))
        meta["separate_preprocessing"] = pp
        A_binned = A_sum
        var_names = dataset.meta.get("var_names")
        feature_space = "raw_counts_binned_deferred"
    else:
        A_binned, var_names, feature_space, transform_meta = _transform_binned_expression(
            A_sum, dataset, config,
        )
        meta["normalize_total"] = bool(transform_meta.get("normalize_total", False))
        meta["log1p"] = bool(transform_meta.get("log1p", False))
        meta["standardize_expression"] = bool(transform_meta.get("standardize_expression", False))

    if var_names is not None:
        meta["var_names"] = list(var_names)
    if feature_space is not None:
        meta["feature_space"] = feature_space

    binned = DatasetBundle(S=S_binned, A=A_binned, meta=meta).validate()
    artifacts = {
        "binned_dataset": binned,
        "binning_summary": dict(meta["binning"]),
        "binning_cell_counts": bin_counts,
    }
    return binned, artifacts


def parallel_config_for_binned_run(config: TestConfig) -> TestConfig:
    """Run the standard permutation machinery after the binning transform."""
    return replace(config, method="parallel_permutation")
