"""Post-load spatial region splitting for DatasetBundle.

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
                n_sub = int((cluster_ids == c).sum())
                sub_strs.append(f"{sub_name} (n={n_sub})")

            for sub_i, c in enumerate(valid_clusters_sorted):
                new_region_idx = base_idx + sub_i
                new_labels[global_indices[cluster_ids == c]] = new_region_idx

            n_kept = sum(int((cluster_ids == c).sum()) for c in valid_clusters_sorted)
            n_dropped = n - n_kept
            drop_str = f" [{n_dropped} noise/tiny dropped]" if n_dropped > 0 else ""
            print(f"  {ct_name} (n={n}) -> [{', '.join(sub_strs)}]{drop_str}")

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
        for key in ("covariate_values",):
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
