"""Visium-style pseudospot pseudobulk transform.

Unlike ``methods.binning`` (which fully tiles the tissue so every cell belongs to
exactly one pseudospot), this overlays a hexagonal array of fixed-diameter circular
spots spaced ``test.visium_spot_distance_um`` apart (default 100 µm), matching the
10x Visium array geometry. Only points within ``test.visium_spot_diameter_um / 2``
(default 27.5 µm, i.e. a 55 µm-diameter spot) of a spot center are aggregated into
that spot; cells that fall in the gaps between spots are dropped, exactly as they
would be for a real Visium capture area. Spots whose full circular footprint would
extend outside the tissue mask are discarded up front to avoid edge artifacts from
partially-covered boundary spots.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from data import raw_coordinates_from_standardized
from data.schemas import DatasetBundle, TestConfig
from data.transforms import apply_expression_transforms
from methods.block_permutation import hex_center_coord, resolve_um_per_unit
from methods.metrics import canonicalize_metric_name

_N_BOUNDARY_SAMPLES = 32


def _hex_lattice_centers(
    *,
    q_range: tuple[int, int],
    r_range: tuple[int, int],
    radius_units: float,
    origin_xy: tuple[float, float],
) -> np.ndarray:
    """Enumerate hex lattice centers, spaced ``radius_units * sqrt(3)`` apart."""
    q_vals = np.arange(q_range[0], q_range[1] + 1, dtype=np.int64)
    r_vals = np.arange(r_range[0], r_range[1] + 1, dtype=np.int64)
    q_grid, r_grid = np.meshgrid(q_vals, r_vals, indexing="ij")
    cx, cy = hex_center_coord(q_grid.ravel(), r_grid.ravel(), radius_units)
    centers = np.column_stack([cx, cy]).astype(np.float64)
    centers[:, 0] += float(origin_xy[0])
    centers[:, 1] += float(origin_xy[1])
    return centers


def _qr_range_for_bbox(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    radius_units: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Axial (q, r) index bounds guaranteed to cover ``[x_min, x_max] x [y_min, y_max]``."""
    r_min = math.floor(y_min / (1.5 * radius_units)) - 1
    r_max = math.ceil(y_max / (1.5 * radius_units)) + 1
    q_min = math.floor(x_min / (radius_units * math.sqrt(3.0))) - abs(r_max) - 1
    q_max = math.ceil(x_max / (radius_units * math.sqrt(3.0))) + abs(r_max) + 1
    return (int(q_min), int(q_max)), (int(r_min), int(r_max))


def _grid_origin_um(
    S_um: np.ndarray,
    *,
    jitter: bool,
    seed: int,
    spacing_um: float,
) -> tuple[float, float]:
    lower_left = np.asarray(S_um, dtype=np.float64).min(axis=0)
    if not jitter:
        return float(lower_left[0]), float(lower_left[1])
    rng = np.random.default_rng(int(seed))
    return (
        float(lower_left[0] - rng.uniform(0.0, spacing_um)),
        float(lower_left[1] - rng.uniform(0.0, spacing_um)),
    )


def _synthetic_tissue_membership_fn(dataset: DatasetBundle):
    """Point-membership predicate on native (raw, pre-µm-scaling) coordinates.

    Returns ``None`` when the dataset was not generated with a known synthetic shape
    (e.g. h5ad data), signalling callers should fall back to a density-based mask.
    """
    if dataset.meta.get("source") != "synthetic":
        return None
    shape = str(dataset.meta.get("shape", "square"))
    from data.synthetic import spatial_shape_mask

    def _membership(xy_native: np.ndarray) -> np.ndarray:
        return spatial_shape_mask(np.asarray(xy_native, dtype=np.float64), shape)

    return _membership


def _density_tissue_membership_fn(S_um: np.ndarray, *, radius_um: float | None = None):
    """Fallback tissue mask for non-synthetic data: a point is "in tissue" if a real
    cell lies within ``radius_um`` of it (default: 1.5x the median nearest-neighbor
    spacing of the observed cells)."""
    tree = cKDTree(S_um)
    if radius_um is None:
        nn_dist, _ = tree.query(S_um, k=2)
        median_nn = float(np.median(nn_dist[:, 1])) if S_um.shape[0] > 1 else 1.0
        radius_um = 1.5 * median_nn

    def _membership(xy_um: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(np.asarray(xy_um, dtype=np.float64), k=1)
        return np.asarray(dist) <= float(radius_um)

    return _membership


def _circle_sample_points(center_xy: np.ndarray, radius: float, n_samples: int = _N_BOUNDARY_SAMPLES) -> np.ndarray:
    """Center + evenly spaced boundary points on a circle, one column per candidate spot.

    ``center_xy`` has shape ``(n_spots, 2)``; returns ``(n_spots, n_samples + 1, 2)``.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * float(radius)  # (n_samples, 2)
    boundary = center_xy[:, None, :] + offsets[None, :, :]  # (n_spots, n_samples, 2)
    return np.concatenate([center_xy[:, None, :], boundary], axis=1)  # (n_spots, n_samples+1, 2)


def _spots_fully_in_tissue(
    centers_native: np.ndarray,
    *,
    spot_radius_native: float,
    membership_fn,
) -> np.ndarray:
    """Boolean mask: True where every sampled point of the spot's circle is in tissue."""
    sample_points = _circle_sample_points(centers_native, spot_radius_native)
    n_spots, n_points, _ = sample_points.shape
    flat = sample_points.reshape(n_spots * n_points, 2)
    in_tissue_flat = membership_fn(flat)
    in_tissue = np.asarray(in_tissue_flat, dtype=bool).reshape(n_spots, n_points)
    return in_tissue.all(axis=1)


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


def _expression_preprocessing_from_meta(dataset: DatasetBundle, config: TestConfig) -> dict[str, Any]:
    pp = dict(dataset.meta.get("visium_pseudobulk_preprocessing") or {})
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


def _transform_pseudobulk_expression(
    A_sum: np.ndarray,
    dataset: DatasetBundle,
    config: TestConfig,
) -> tuple[np.ndarray, list[str] | None, str | None, dict[str, Any]]:
    pp = _expression_preprocessing_from_meta(dataset, config)
    if bool(pp.get("normalize_total")) and np.any(np.asarray(A_sum) < 0):
        raise ValueError(
            "test.method='visium_pseudobulk' expects non-negative raw counts before CPM "
            "normalization. For h5ad configs, run through run_permutation.py so pseudobulking "
            "can load counts before data.normalize_total/log1p, or disable CPM-style "
            "preprocessing."
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


def build_visium_pseudobulk_dataset(
    dataset: DatasetBundle,
    config: TestConfig,
) -> tuple[DatasetBundle, dict[str, Any]]:
    """Aggregate cells into a Visium-style array of fixed-diameter circular pseudospots.

    Cells that fall outside every spot's circle (the inter-spot gaps) are dropped.
    Spots whose full circular footprint would extend past the tissue mask are dropped
    up front. Cell-type modes keep types as separate pseudospots via a ``(cell_type,
    spot_id)`` grouping key, matching ``methods.binning``.
    """
    dataset.validate()
    config.validate()

    spacing_um = float(config.visium_spot_distance_um)
    spot_radius_um = float(config.visium_spot_diameter_um) / 2.0
    radius_units_um = spacing_um / math.sqrt(3.0)  # hex-lattice spacing parameter, not the spot radius
    um_per_unit = resolve_um_per_unit(
        config.coordinate_um_per_unit,
        dataset.meta.get("coordinate_um_per_unit"),
    )

    S_raw = raw_coordinates_from_standardized(dataset.S, dataset.meta)
    S_native = np.asarray(S_raw, dtype=np.float64)
    S_um = S_native * float(um_per_unit)

    origin_um = _grid_origin_um(
        S_um,
        jitter=bool(getattr(config, "block_jitter", False)),
        seed=int(config.seed),
        spacing_um=spacing_um,
    )
    q_range, r_range = _qr_range_for_bbox(
        x_min=float(S_um[:, 0].min()) - origin_um[0],
        x_max=float(S_um[:, 0].max()) - origin_um[0],
        y_min=float(S_um[:, 1].min()) - origin_um[1],
        y_max=float(S_um[:, 1].max()) - origin_um[1],
        radius_units=radius_units_um,
    )
    candidate_centers_um = _hex_lattice_centers(
        q_range=q_range,
        r_range=r_range,
        radius_units=radius_units_um,
        origin_xy=origin_um,
    )

    membership_fn = _synthetic_tissue_membership_fn(dataset)
    if membership_fn is not None:
        candidate_centers_native = candidate_centers_um / float(um_per_unit)
        spot_radius_native = spot_radius_um / float(um_per_unit)
        in_tissue = _spots_fully_in_tissue(
            candidate_centers_native,
            spot_radius_native=spot_radius_native,
            membership_fn=membership_fn,
        )
    else:
        density_membership_fn = _density_tissue_membership_fn(S_um)
        in_tissue = _spots_fully_in_tissue(
            candidate_centers_um,
            spot_radius_native=spot_radius_um,
            membership_fn=density_membership_fn,
        )
    valid_centers_um = candidate_centers_um[in_tissue]

    if valid_centers_um.shape[0] == 0:
        raise ValueError(
            "test.method='visium_pseudobulk' found no spot whose full circular footprint "
            "fits inside the tissue mask; check visium_spot_distance_um/visium_spot_diameter_um "
            "relative to the tissue extent."
        )

    spot_tree = cKDTree(valid_centers_um)
    nearest_dist, nearest_spot = spot_tree.query(S_um, k=1)
    cell_assigned = nearest_dist <= spot_radius_um
    assigned_spot_idx = np.where(cell_assigned, nearest_spot, -1)

    occupied_spot_ids, occupancy_inverse = np.unique(
        assigned_spot_idx[cell_assigned], return_inverse=True
    )
    n_original_spots = int(valid_centers_um.shape[0])
    n_dropped_cells = int((~cell_assigned).sum())

    cell_type_mode = dataset.meta.get("cell_type_mode", "none")
    cell_type_labels = dataset.meta.get("cell_type_labels")
    kept_cell_mask = cell_assigned
    if cell_type_mode in {"together", "separate"} and cell_type_labels is not None:
        labels_all = np.asarray(cell_type_labels, dtype=np.int64)
        labels_kept = labels_all[kept_cell_mask]
        spot_ids_kept = assigned_spot_idx[kept_cell_mask]
        keys = np.column_stack([labels_kept, spot_ids_kept])
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        binned_labels = unique_keys[:, 0].astype(np.int64)
        binned_spot_ids = unique_keys[:, 1].astype(np.int64)
    else:
        binned_spot_ids, inverse = occupied_spot_ids, occupancy_inverse
        binned_labels = None

    n_bins = int(binned_spot_ids.shape[0])
    A_raw = np.asarray(dataset.A, dtype=np.float32)[kept_cell_mask]
    A_sum = np.zeros((n_bins, A_raw.shape[1]), dtype=np.float64)
    np.add.at(A_sum, inverse, A_raw.astype(np.float64, copy=False))
    A_sum = np.asarray(A_sum, dtype=np.float32)
    bin_counts = np.bincount(inverse, minlength=n_bins).astype(np.int64)

    centers_um = valid_centers_um[binned_spot_ids]
    centers_native = np.asarray(centers_um / float(um_per_unit), dtype=np.float32)
    S_binned, coord_meta = _zscore_coordinates_if_needed(centers_native, dataset.meta)

    meta = dict(dataset.meta)
    for stale_key in (
        "plot_cell_type_labels",
        "plot_cell_type_names",
        "synthetic_true_curve",
        "kernel_noise_sample",
    ):
        meta.pop(stale_key, None)
    meta.update(coord_meta)
    meta["visium_pseudobulk"] = {
        "spot_distance_um": spacing_um,
        "spot_diameter_um": float(config.visium_spot_diameter_um),
        "spot_radius_um": float(spot_radius_um),
        "hex_radius_um": float(radius_units_um),
        "coordinate_um_per_unit": float(um_per_unit),
        "origin_x_um": float(origin_um[0]),
        "origin_y_um": float(origin_um[1]),
        "jitter": bool(getattr(config, "block_jitter", False)),
        "original_n_cells": int(dataset.n_cells),
        "n_candidate_spots": int(candidate_centers_um.shape[0]),
        "n_spots_in_tissue": n_original_spots,
        "n_pseudospots": int(n_bins),
        "n_cells_dropped_in_gaps": n_dropped_cells,
        "fraction_cells_dropped_in_gaps": float(n_dropped_cells) / float(dataset.n_cells),
        "mean_cells_per_pseudospot": float(bin_counts.mean()) if n_bins else 0.0,
        "median_cells_per_pseudospot": float(np.median(bin_counts)) if n_bins else 0.0,
        "min_cells_per_pseudospot": int(bin_counts.min()) if n_bins else 0,
        "max_cells_per_pseudospot": int(bin_counts.max()) if n_bins else 0,
    }
    meta["visium_pseudobulk_cell_counts"] = bin_counts

    if binned_labels is not None:
        meta["cell_type_labels"] = binned_labels
        meta["cell_type_mode"] = cell_type_mode
        if "cell_type_names" in dataset.meta:
            meta["cell_type_names"] = list(dataset.meta["cell_type_names"])
        if "n_cell_types" in dataset.meta:
            meta["n_cell_types"] = int(dataset.meta["n_cell_types"])

    for key in ("covariate_values", "covariate_whitening_values", "calicost_tumor_proportion"):
        source_values = dataset.meta.get(key)
        if source_values is not None:
            source_values = np.asarray(source_values)[kept_cell_mask]
        aggregated = _aggregate_vectors(source_values, inverse, n_bins)
        if aggregated is not None:
            meta[key] = aggregated

    if cell_type_mode == "separate" and binned_labels is not None:
        pp = dict(meta.get("separate_preprocessing") or {})
        if not pp:
            pp = _expression_preprocessing_from_meta(dataset, config)
        pp["seed"] = int(pp.get("seed", config.seed))
        meta["separate_preprocessing"] = pp
        A_binned = A_sum
        var_names = dataset.meta.get("var_names")
        feature_space = "raw_counts_visium_pseudobulk_deferred"
    else:
        A_binned, var_names, feature_space, transform_meta = _transform_pseudobulk_expression(
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
        "visium_pseudobulk_dataset": binned,
        "visium_pseudobulk_summary": dict(meta["visium_pseudobulk"]),
        "visium_pseudobulk_cell_counts": bin_counts,
        "visium_overlay_S_um": np.asarray(S_um, dtype=np.float32),
        "visium_overlay_assigned": np.asarray(cell_assigned, dtype=bool),
        "visium_overlay_centers_in_tissue_um": np.asarray(valid_centers_um, dtype=np.float32),
        "visium_overlay_occupied_ids": np.asarray(occupied_spot_ids, dtype=np.int64),
        "visium_overlay_hex_radius_um": float(radius_units_um),
        "visium_overlay_spot_radius_um": float(spot_radius_um),
    }
    return binned, artifacts


def parallel_config_for_visium_pseudobulk_run(config: TestConfig) -> TestConfig:
    """Run the standard permutation machinery after the pseudobulk transform."""
    return replace(config, method="parallel_permutation")
