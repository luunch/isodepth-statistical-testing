"""Post-load spatial region splitting and denoising for DatasetBundle.

Provides :func:`denoise_spatial_outliers`, which removes per-clone outlier spots
that are not in the largest connected component within a given radius (applied
before coordinate z-scoring and before splitting).

Provides :func:`split_spatial_regions`, which runs DBSCAN (when
``config.spatial_region_split is True``) or K-Means (when
``config.spatial_region_split`` is a positive integer >= 2) on raw spatial
coordinates within each existing cell-type label group.  The result is an
expanded set of sub-region labels that feed directly into the
``cell_type="separate"`` pipeline.

Usage
-----
Called automatically by :func:`data.load_dataset` when
``config.spatial_region_split`` is set.  Requires ``cell_type="separate"``
so that per-region expression preprocessing is correctly deferred.

Naming convention
-----------------
- A label group that splits into k > 1 regions produces sub-labels
  ``"<name>_r0"``, ``"<name>_r1"``, … sorted by **descending size**
  (``r0`` is always the largest component).
- A label group that maps to a single region keeps its original name
  unchanged (no ``_r0`` suffix).
- DBSCAN noise cells (cluster id -1) and cells in sub-regions smaller than
  ``config.spatial_region_split_min_cells`` are dropped from the dataset.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from data.schemas import DataConfig, DatasetBundle


def apply_spatial_crop(dataset: DatasetBundle, config: DataConfig) -> DatasetBundle:
    """Drop cells outside an axis-aligned box on ``dataset.S`` (already-standardized coords).

    ``config.spatial_crop`` maps axis names ``"x"``/``"y"`` to operator dicts using the
    same ``{"gt"/"ge"/"lt"/"le"/"eq"/"ne": threshold}`` syntax as ``obs_numeric_filters``.
    Intended as a simple, geometry-only way to excise disconnected/off-tissue spots (e.g.
    a handful of far-away outlier spots) by coordinate, as an alternative to the
    DBSCAN/K-Means-based :func:`split_spatial_regions`. No-op when ``spatial_crop`` is unset.
    """
    spec = config.spatial_crop
    if not spec:
        return dataset

    axis_to_col = {"x": 0, "y": 1}
    S = np.asarray(dataset.S, dtype=np.float64)
    n_total = len(S)
    mask = np.ones(n_total, dtype=bool)
    for axis, ops in spec.items():
        col = S[:, axis_to_col[axis]]
        for op, threshold_raw in ops.items():
            threshold = float(threshold_raw)
            if op == "gt":
                mask &= col > threshold
            elif op in ("ge", "gte"):
                mask &= col >= threshold
            elif op == "lt":
                mask &= col < threshold
            elif op in ("le", "lte"):
                mask &= col <= threshold
            elif op == "eq":
                mask &= col == threshold
            elif op == "ne":
                mask &= col != threshold

    n_dropped = int((~mask).sum())
    if n_dropped == 0:
        return dataset
    if not mask.any():
        raise ValueError(f"data.spatial_crop={spec} matched no cells")

    print(
        f"[spatial_crop] dropping {n_dropped}/{n_total} cells "
        f"({100.0 * n_dropped / n_total:.1f}%) outside crop bounds {spec}"
    )
    dataset.S = dataset.S[mask]
    dataset.A = dataset.A[mask]
    for key in (
        "covariate_values",
        "covariate_whitening_values",
        "calicost_tumor_proportion",
        "plot_cell_type_labels",
        "cell_type_labels",
    ):
        val = dataset.meta.get(key)
        if val is not None:
            arr = np.asarray(val)
            if arr.ndim >= 1 and len(arr) == n_total:
                dataset.meta[key] = arr[mask]
    dataset.meta["spatial_crop"] = dict(spec)
    dataset.meta["spatial_crop_n_dropped"] = n_dropped
    return dataset


_PER_SPOT_META_KEYS = (
    "covariate_values",
    "covariate_whitening_values",
    "calicost_tumor_proportion",
    "plot_cell_type_labels",
    "cell_type_labels",
)


def label_border_spot_mask(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    radius_mult: float = 1.2,
) -> tuple[np.ndarray, float, float]:
    """Mark spots that touch a different label within one lattice step.

    The radius is ``radius_mult`` times the median nearest-neighbor distance.
    On Visium, 1.2 includes the six immediate hex neighbors (spacing = pitch)
    and excludes the next ring (spacing = pitch * sqrt(3) ≈ 1.73).
    """
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels)
    n = int(coords.shape[0])
    border = np.zeros(n, dtype=bool)
    if n < 2:
        return border, float("nan"), float("nan")

    tree = cKDTree(coords)
    nn = tree.query(coords, k=2)[0][:, 1]
    positive = nn[np.isfinite(nn) & (nn > 0.0)]
    if positive.size == 0:
        return border, float("nan"), float("nan")

    pitch = float(np.median(positive))
    radius = pitch * float(radius_mult)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return border, pitch, radius

    different = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    if not np.any(different):
        return border, pitch, radius
    hits = pairs[different]
    border[hits[:, 0]] = True
    border[hits[:, 1]] = True
    return border, pitch, radius


def _subset_bundle_rows(dataset: DatasetBundle, keep_mask: np.ndarray) -> None:
    keep_mask = np.asarray(keep_mask, dtype=bool)
    n_total = int(keep_mask.shape[0])
    if keep_mask.shape[0] != dataset.S.shape[0]:
        raise ValueError(
            f"Row mask has length {keep_mask.shape[0]} but dataset has {dataset.S.shape[0]} spots"
        )
    if not keep_mask.any():
        raise ValueError("data.drop_label_border_spots removed every spot")
    if keep_mask.all():
        return
    dataset.S = dataset.S[keep_mask]
    dataset.A = dataset.A[keep_mask]
    for key in _PER_SPOT_META_KEYS:
        val = dataset.meta.get(key)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == n_total:
            dataset.meta[key] = arr[keep_mask]


def drop_label_border_spots(dataset: DatasetBundle, config: DataConfig) -> DatasetBundle:
    """Remove spots that sit on the boundary between two layer labels.

    A spot is removed when any spatial neighbor inside one lattice step has a
    different ``cell_type_labels`` value. Remaining labels smaller than
    ``min_cells_per_celltype`` are dropped and the surviving labels are
    renumbered. Tissue-edge spots whose neighbors share their label are kept.
    """
    if not getattr(config, "drop_label_border_spots", False):
        return dataset

    raw_labels = dataset.meta.get("cell_type_labels")
    raw_names = dataset.meta.get("cell_type_names")
    if raw_labels is None or not raw_names:
        raise ValueError(
            "data.drop_label_border_spots requires cell-type labels from "
            f"data.cell_type_key={config.cell_type_key!r}"
        )

    original_names = [str(name) for name in raw_names]
    labels = np.asarray(raw_labels, dtype=np.int64)
    if labels.shape[0] != dataset.S.shape[0]:
        raise ValueError(
            "cell_type_labels length does not match the number of spots "
            f"({labels.shape[0]} vs {dataset.S.shape[0]})"
        )
    if int(labels.max()) >= len(original_names) or int(labels.min()) < 0:
        raise ValueError(
            "cell_type_labels must be contiguous ids into cell_type_names"
        )

    border, pitch, radius = label_border_spot_mask(
        dataset.S,
        labels,
        radius_mult=float(config.label_border_radius_mult),
    )
    n_total = int(border.shape[0])
    n_dropped = int(border.sum())
    dropped_by_label = {
        original_names[i]: int(np.sum(border & (labels == i)))
        for i in range(len(original_names))
    }
    if n_dropped:
        radius_text = "nan" if not np.isfinite(radius) else f"{radius:.4g}"
        pitch_text = "nan" if not np.isfinite(pitch) else f"{pitch:.4g}"
        dropped_text = ", ".join(
            f"{name}={count}" for name, count in dropped_by_label.items() if count
        )
        print(
            f"[label_border] dropping {n_dropped}/{n_total} spots "
            f"on a layer boundary (radius={radius_text}, pitch={pitch_text}): "
            f"{dropped_text}"
        )
        _subset_bundle_rows(dataset, ~border)

    labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    min_cells = int(config.min_cells_per_celltype)
    counts = np.bincount(labels, minlength=len(original_names))
    keep_types = [i for i, count in enumerate(counts) if int(count) >= min_cells]
    removed_types = [
        original_names[i] for i in range(len(original_names)) if i not in set(keep_types)
    ]
    if not keep_types:
        raise ValueError(
            "data.drop_label_border_spots left no layer with "
            f">= {min_cells} spots"
        )
    if removed_types:
        print(
            "[label_border] dropping layers below "
            f"min_cells_per_celltype={min_cells}: {removed_types}"
        )
        _subset_bundle_rows(dataset, np.isin(labels, keep_types))
        labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
        old_to_new = {old: new for new, old in enumerate(keep_types)}
        labels = np.array([old_to_new[int(value)] for value in labels], dtype=np.int64)
        names = [original_names[i] for i in keep_types]
        dataset.meta["cell_type_labels"] = labels
        dataset.meta["cell_type_names"] = names
    else:
        names = original_names
        dataset.meta["cell_type_labels"] = labels

    dataset.meta["n_cell_types"] = len(names)
    kept_labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    dataset.meta["label_border"] = {
        "n_spots_before": n_total,
        "n_dropped": n_dropped,
        "n_kept": int(dataset.S.shape[0]),
        "pitch": None if not np.isfinite(pitch) else float(pitch),
        "radius": None if not np.isfinite(radius) else float(radius),
        "radius_mult": float(config.label_border_radius_mult),
        "dropped_by_label": dropped_by_label,
        "kept_by_label": {
            names[i]: int(np.sum(kept_labels == i)) for i in range(len(names))
        },
        "removed_labels": removed_types,
    }
    return dataset


def denoise_spatial_outliers(dataset: DatasetBundle, config: DataConfig) -> DatasetBundle:
    """Drop per-clone spots not in the largest connected component.

    For each clone, builds a radius adjacency graph (connecting spots within
    ``config.spatial_denoise_radius_um`` µm of each other), finds connected
    components, and retains only spots in the largest component.  Isolated spots
    and small stray groups are discarded.

    Applied before coordinate z-scoring and before :func:`split_spatial_regions`
    so that downstream splitting operates on the cleaned coordinate set.

    Parameters
    ----------
    dataset:
        Loaded bundle with raw (pre-z-score) coordinates in ``S``.
    config:
        Data config.  Must have ``spatial_denoise_radius_um > 0`` and
        ``coordinate_um_per_unit > 0`` set.
    """
    radius_um = getattr(config, "spatial_denoise_radius_um", None)
    if not radius_um:
        return dataset

    um_per_unit = (
        getattr(config, "coordinate_um_per_unit", None)
        or dataset.meta.get("coordinate_um_per_unit")
    )
    if not um_per_unit:
        raise ValueError(
            "data.spatial_denoise_radius_um requires coordinate_um_per_unit to be set "
            "(in test.coordinate_um_per_unit or auto-detected from the h5ad file)."
        )
    radius_coord = float(radius_um) / float(um_per_unit)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    S = np.asarray(dataset.S, dtype=np.float64)
    n_total = len(S)
    keep_mask = np.ones(n_total, dtype=bool)

    raw_labels = dataset.meta.get("cell_type_labels")
    raw_names = dataset.meta.get("cell_type_names")
    if raw_labels is None or raw_names is None or len(raw_names) == 0:
        groups: list[tuple[np.ndarray, str]] = [(np.arange(n_total), "all")]
    else:
        labels = np.asarray(raw_labels, dtype=np.int64)
        groups = [
            (np.flatnonzero(labels == i), name)
            for i, name in enumerate(raw_names)
        ]

    total_dropped = 0
    for indices, name in groups:
        if len(indices) == 0:
            continue
        xy = S[indices]
        n = len(indices)
        tree = cKDTree(xy)
        pairs = tree.query_pairs(radius_coord, output_type="ndarray")

        if len(pairs) == 0:
            # No pairs at all — every spot is its own component; keep all.
            continue

        row, col = pairs[:, 0], pairs[:, 1]
        data_vals = np.ones(len(row), dtype=np.float32)
        adj = csr_matrix(
            (
                np.concatenate([data_vals, data_vals]),
                (np.concatenate([row, col]), np.concatenate([col, row])),
            ),
            shape=(n, n),
        )
        n_comp, comp_labels = connected_components(adj, directed=False)
        comp_sizes = np.bincount(comp_labels, minlength=n_comp)
        largest = int(np.argmax(comp_sizes))
        outlier_local = comp_labels != largest
        keep_mask[indices[outlier_local]] = False
        n_dropped = int(outlier_local.sum())
        total_dropped += n_dropped
        if n_dropped > 0:
            print(
                f"[spatial_denoise] {name}: dropped {n_dropped} outlier spot(s) "
                f"(kept {comp_sizes[largest]}, radius={radius_um:.0f} µm, "
                f"{n_comp} component(s))"
            )

    if not keep_mask.all():
        print(
            f"[spatial_denoise] total: dropped {total_dropped} / {n_total} spots"
        )
        dataset.S = dataset.S[keep_mask]
        dataset.A = dataset.A[keep_mask]
        for key in ("covariate_values", "covariate_whitening_values", "cell_type_labels"):
            if key in dataset.meta and dataset.meta[key] is not None:
                arr = np.asarray(dataset.meta[key])
                if arr.ndim >= 1 and len(arr) == n_total:
                    dataset.meta[key] = arr[keep_mask]

    return dataset


def split_spatial_regions(dataset: DatasetBundle, config: DataConfig) -> DatasetBundle:
    """Sub-divide each cell-type label into spatially contiguous or forced-K regions.

    Parameters
    ----------
    dataset:
        Loaded bundle with raw (pre-z-score) coordinates in ``S``.
        Must have ``meta["cell_type_labels"]`` and ``meta["cell_type_names"]`` populated
        (requires ``cell_type_mode == "separate"``).
    config:
        Data config.  ``config.spatial_region_split`` controls the algorithm:
        - ``True``     → DBSCAN with auto-eps (no K required)
        - int >= 2    → K-Means with exactly that many clusters per label group

    Returns
    -------
    DatasetBundle
        Updated bundle.  ``meta["cell_type_labels"]``, ``meta["cell_type_names"]``, and
        ``meta["n_cell_types"]`` are replaced with the expanded sub-region values.
        When DBSCAN drops noise cells or K-Means/min_cells filtering removes cells,
        ``dataset.S``, ``dataset.A``, and any cell-indexed meta arrays are pruned
        to match.
    """
    rs = config.spatial_region_split
    if rs is False or rs == 0:
        return dataset

    use_kmeans = isinstance(rs, int) and not isinstance(rs, bool)
    k_fixed: int = int(rs) if use_kmeans else 0

    S = np.asarray(dataset.S, dtype=np.float64)
    n_total = len(S)

    # When cell_type=false was configured there are no cell-type labels yet.
    # Create a single synthetic "region" group covering all cells so the
    # splitting logic below runs uniformly.
    raw_labels = dataset.meta.get("cell_type_labels")
    raw_names = dataset.meta.get("cell_type_names")
    if raw_labels is None or raw_names is None or len(raw_names) == 0:
        labels_in = np.zeros(n_total, dtype=np.int64)
        names_in = ["region"]
    else:
        labels_in = np.asarray(raw_labels, dtype=np.int64)
        names_in = list(raw_names)

    new_labels = np.full(n_total, -1, dtype=np.int64)
    new_names: list[str] = []
    keep_mask = np.ones(n_total, dtype=bool)
    # -1 = kept as a single unsplit region (grey in overview plot); >=0 indexes
    # ``region_color_names`` for kept multi-region components.
    region_color_ids = np.full(n_total, -1, dtype=np.int64)
    region_color_names: list[str] = []

    alg_name = f"K-Means (K={k_fixed})" if use_kmeans else "DBSCAN"
    print(f"[spatial_region_split] algorithm={alg_name}")

    for ct_idx, ct_name in enumerate(names_in):
        group_mask = labels_in == ct_idx
        global_indices = np.flatnonzero(group_mask)
        xy = S[global_indices]
        n = len(global_indices)

        if use_kmeans:
            cluster_ids = _split_kmeans(xy, k_fixed, seed=config.seed)
        else:
            eps = config.spatial_region_split_eps
            if eps is None:
                eps = _auto_eps(xy, config.spatial_region_split_eps_mult)
            cluster_ids = _split_dbscan(
                xy, eps=eps, min_samples=config.spatial_region_split_min_samples
            )

        # Mark noise cells (DBSCAN label == -1) for removal
        noise_local = cluster_ids == -1
        if noise_local.any():
            keep_mask[global_indices[noise_local]] = False

        unique_clusters = sorted({int(c) for c in cluster_ids if c >= 0})

        # Drop sub-regions smaller than min_cells
        valid_clusters = [
            c for c in unique_clusters
            if int((cluster_ids == c).sum()) >= config.spatial_region_split_min_cells
        ]
        tiny_clusters = [c for c in unique_clusters if c not in valid_clusters]
        for c in tiny_clusters:
            keep_mask[global_indices[cluster_ids == c]] = False

        if not valid_clusters:
            # Fallback: no component survived min_cells filter.
            # Re-include all non-noise cells under the original name.
            non_noise_local = ~noise_local
            keep_mask[global_indices[non_noise_local]] = True
            new_region_idx = len(new_names)
            new_names.append(ct_name)
            new_labels[global_indices[non_noise_local]] = new_region_idx
            print(
                f"  {ct_name} (n={n}) -> [{ct_name} (n={non_noise_local.sum()}, "
                "no valid clusters after min_cells filter — kept as one region)]"
            )
            continue

        # Sort by descending size so r0 is always the largest component
        valid_clusters_sorted = sorted(
            valid_clusters, key=lambda c: -int((cluster_ids == c).sum())
        )

        if len(valid_clusters_sorted) == 1:
            c = valid_clusters_sorted[0]
            n_kept = int((cluster_ids == c).sum())
            n_dropped = n - n_kept
            new_region_idx = len(new_names)
            new_names.append(ct_name)
            new_labels[global_indices[cluster_ids == c]] = new_region_idx
            drop_str = f", {n_dropped} noise/tiny dropped" if n_dropped > 0 else ""
            print(f"  {ct_name} (n={n}) -> [{ct_name} (n={n_kept}){drop_str}]")
        else:
            base_idx = len(new_names)
            sub_strs: list[str] = []
            for sub_i, c in enumerate(valid_clusters_sorted):
                sub_name = f"{ct_name}_r{sub_i}"
                new_names.append(sub_name)
                region_color_names.append(sub_name)
                n_sub = int((cluster_ids == c).sum())
                sub_strs.append(f"{sub_name} (n={n_sub})")

            for sub_i, c in enumerate(valid_clusters_sorted):
                new_region_idx = base_idx + sub_i
                sub_name = new_names[base_idx + sub_i]
                color_id = region_color_names.index(sub_name)
                local_mask = cluster_ids == c
                new_labels[global_indices[local_mask]] = new_region_idx
                region_color_ids[global_indices[local_mask]] = color_id

            n_kept = sum(int((cluster_ids == c).sum()) for c in valid_clusters_sorted)
            n_dropped = n - n_kept
            drop_str = f" [{n_dropped} noise/tiny dropped]" if n_dropped > 0 else ""
            print(f"  {ct_name} (n={n}) -> [{', '.join(sub_strs)}]{drop_str}")

    dataset.meta["spatial_region_split_diag"] = {
        "algorithm": alg_name,
        "S": S.copy(),
        "removed": ~keep_mask,
        "region_color_ids": region_color_ids.copy(),
        "region_color_names": list(region_color_names),
    }

    # Apply keep_mask to prune noise and tiny-cluster cells
    if not keep_mask.all():
        n_dropped_total = int((~keep_mask).sum())
        print(
            f"[spatial_region_split] dropping {n_dropped_total} cells "
            f"({100.0 * n_dropped_total / n_total:.1f}%) "
            "classified as noise or below min_cells threshold"
        )
        dataset.S = dataset.S[keep_mask]
        dataset.A = dataset.A[keep_mask]
        new_labels = new_labels[keep_mask]
        # Prune any other cell-indexed meta arrays that may be present
        for key in ("covariate_values", "covariate_whitening_values"):
            if key in dataset.meta and dataset.meta[key] is not None:
                arr = np.asarray(dataset.meta[key])
                if arr.ndim >= 1 and len(arr) == n_total:
                    dataset.meta[key] = arr[keep_mask]

    dataset.meta["cell_type_labels"] = new_labels
    dataset.meta["cell_type_names"] = new_names
    dataset.meta["n_cell_types"] = len(new_names)
    # Ensure downstream dispatch (run_permutation_method, save_standardized_outputs)
    # treats this dataset as separate-mode regardless of what cell_type was in the config.
    dataset.meta["cell_type_mode"] = "separate"
    return dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _axis_scores(xy: np.ndarray, axis: str) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"xy must be (N, 2), got {coords.shape}")
    if axis == "x":
        return coords[:, 0]
    if axis == "y":
        return coords[:, 1]
    if axis == "pc1":
        centered = coords - coords.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[0]
    raise ValueError(f"axis must be 'pc1', 'x', or 'y'; got '{axis}'")


def dbscan_middle_region_mask(
    xy: np.ndarray,
    *,
    eps: float | None = None,
    eps_mult: float = 3.0,
    min_samples: int = 10,
    min_cells: int = 50,
    axis: str = "pc1",
) -> tuple[np.ndarray, dict[str, object]]:
    """DBSCAN spatial regions, then keep the middle component along ``axis``.

    Valid clusters (not noise, size >= ``min_cells``) are ordered by centroid
    position on ``axis``; the cluster at the median rank is returned.  For
    three tangential bands this selects the central band while dropping edge
    fragments and DBSCAN noise.
    """
    coords = np.asarray(xy, dtype=np.float64)
    if coords.shape[0] == 0:
        raise ValueError("xy must contain at least one point")

    eps_value = float(_auto_eps(coords, eps_mult) if eps is None else eps)
    cluster_ids = _split_dbscan(
        coords, eps=eps_value, min_samples=int(min_samples)
    )

    valid_clusters = sorted(
        {
            int(c)
            for c in cluster_ids
            if c >= 0 and int((cluster_ids == c).sum()) >= int(min_cells)
        }
    )
    if not valid_clusters:
        raise ValueError(
            "DBSCAN found no clusters meeting min_cells="
            f"{min_cells} (try lowering min_cells or eps_mult)"
        )

    centroid_scores = {
        c: float(_axis_scores(coords[cluster_ids == c], axis=axis).mean())
        for c in valid_clusters
    }
    ordered = sorted(valid_clusters, key=lambda c: centroid_scores[c])
    selected = ordered[len(ordered) // 2]
    mask = cluster_ids == selected

    diag: dict[str, object] = {
        "mode": "dbscan_middle",
        "algorithm": "DBSCAN",
        "axis": axis,
        "eps": eps_value,
        "eps_mult": float(eps_mult),
        "min_samples": int(min_samples),
        "min_cells": int(min_cells),
        "n_cells_before": int(coords.shape[0]),
        "n_cells_after": int(mask.sum()),
        "n_clusters_valid": len(valid_clusters),
        "cluster_sizes": {
            int(c): int((cluster_ids == c).sum()) for c in valid_clusters
        },
        "cluster_centroid_axis": {
            int(c): centroid_scores[c] for c in valid_clusters
        },
        "selected_cluster": int(selected),
        "cluster_ids": cluster_ids.copy(),
        "noise_cells": int((cluster_ids == -1).sum()),
    }
    return mask, diag


def _auto_eps(xy: np.ndarray, eps_mult: float) -> float:
    """Estimate DBSCAN eps from the 90th-percentile nearest-neighbour distance."""
    if len(xy) < 2:
        return 1.0
    nn_dists = cKDTree(xy).query(xy, k=2)[0][:, 1]
    return float(np.percentile(nn_dists, 90) * eps_mult)


def _split_dbscan(xy: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run DBSCAN and return integer cluster labels (noise = -1)."""
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)


def _split_kmeans(xy: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Run K-Means with ``min(k, n_cells)`` clusters and return integer labels."""
    from sklearn.cluster import KMeans
    k_actual = min(k, len(xy))
    return KMeans(n_clusters=k_actual, random_state=seed, n_init="auto").fit_predict(xy)
