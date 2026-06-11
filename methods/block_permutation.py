"""Block-permutation null for spatial transcriptomics permutation tests.

Tiles the tissue with a pointy-top hexagonal grid then randomly permutes entire
block clusters via centroid translation.  This breaks large-scale spatial
gradients while preserving within-block expression–coordinate coupling (local
niche / signalling structure).

For each permutation slot the mesh is optionally jittered so block boundaries
do not always align with the same cells.  When cell-type labels are provided
each cell type receives an independent mesh (independent jitter seed), so
within-type local structure is preserved separately.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

_SQRT3_OVER_3 = math.sqrt(3.0) / 3.0
_ONE_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0

# Cantor-like pairing offset.  Supports hex axial coords in [-HASH_OFFSET, HASH_OFFSET].
# 1 000 000 covers tissues up to ~100 mm across at 0.1 µm resolution.
_HASH_OFFSET = np.int64(1_000_000)
_HASH_MODULUS = np.int64(2_000_001)


def hex_bin_ids(
    coords_um: np.ndarray,
    radius_um: float,
    jitter_xy_um: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Assign each cell to a hexagonal bin.

    Uses a pointy-top hex grid with the given radius (centre-to-vertex).

    Parameters
    ----------
    coords_um:
        ``(N, 2)`` float array of coordinates in microns.
    radius_um:
        Hex radius in microns (centre to vertex).
    jitter_xy_um:
        ``(dx, dy)`` offset applied before binning — shifts the mesh.

    Returns
    -------
    block_ids:
        ``(N,)`` int64 array; each unique integer identifies one hex cell.
    """
    x = coords_um[:, 0].astype(np.float64) - float(jitter_xy_um[0])
    y = coords_um[:, 1].astype(np.float64) - float(jitter_xy_um[1])

    # Pointy-top axial coordinates
    q_float = (_SQRT3_OVER_3 * x - _ONE_THIRD * y) / radius_um
    r_float = (_TWO_THIRDS * y) / radius_um
    s_float = -q_float - r_float

    q_round = np.round(q_float).astype(np.int64)
    r_round = np.round(r_float).astype(np.int64)
    s_round = np.round(s_float).astype(np.int64)

    q_diff = np.abs(q_round.astype(np.float64) - q_float)
    r_diff = np.abs(r_round.astype(np.float64) - r_float)
    s_diff = np.abs(s_round.astype(np.float64) - s_float)

    # Cube-coordinate rounding: correct the component with the largest error
    q_wins = (q_diff > r_diff) & (q_diff > s_diff)
    r_wins = (~q_wins) & (r_diff > s_diff)

    q_out = np.where(q_wins, -r_round - s_round, q_round)
    r_out = np.where(r_wins, -q_round - s_round, r_round)

    return ((q_out + _HASH_OFFSET) * _HASH_MODULUS + (r_out + _HASH_OFFSET)).astype(np.int64)


def block_ids_to_axial_qr(block_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode packed hex ``block_ids`` back to axial cube coordinates ``(q, r)``."""
    bids = np.asarray(block_ids, dtype=np.int64)
    q = bids // _HASH_MODULUS - _HASH_OFFSET
    r = bids % _HASH_MODULUS - _HASH_OFFSET
    return q, r


def hex_center_coord(
    q: np.ndarray,
    r: np.ndarray,
    radius_units: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hex centre positions in native coordinate units (pointy-top grid)."""
    q_arr = np.asarray(q, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    radius = float(radius_units)
    cx = radius * math.sqrt(3.0) * (q_arr + r_arr / 2.0)
    cy = radius * 1.5 * r_arr
    return cx, cy


def hex_vertex_offsets_pointy_top(radius_units: float) -> np.ndarray:
    """Closed pointy-top hex polygon offsets from centre, shape ``(7, 2)``."""
    radius = float(radius_units)
    half_sqrt3 = math.sqrt(3.0) / 2.0
    return np.array(
        [
            [0.0, radius],
            [half_sqrt3 * radius, radius / 2.0],
            [half_sqrt3 * radius, -radius / 2.0],
            [0.0, -radius],
            [-half_sqrt3 * radius, -radius / 2.0],
            [-half_sqrt3 * radius, radius / 2.0],
            [0.0, radius],
        ],
        dtype=np.float64,
    )


def hex_polygons_for_block_ids(
    block_ids: np.ndarray,
    radius_units: float,
) -> list[np.ndarray]:
    """One closed hex polygon per occupied block, in native coordinate units."""
    unique_ids = np.unique(np.asarray(block_ids, dtype=np.int64))
    if unique_ids.size == 0:
        return []
    q, r = block_ids_to_axial_qr(unique_ids)
    cx, cy = hex_center_coord(q, r, radius_units)
    offsets = hex_vertex_offsets_pointy_top(radius_units)
    return [
        np.column_stack([cx[i] + offsets[:, 0], cy[i] + offsets[:, 1]])
        for i in range(unique_ids.size)
    ]

def block_centroid_permute(
    coords_um: np.ndarray,
    block_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Translate each block's cells so block centroids are randomly permuted.

    Each cell's position relative to its block centroid is preserved exactly.
    All occupied blocks participate (including singletons).
    """
    unique_ids, inverse, counts = np.unique(block_ids, return_inverse=True, return_counts=True)
    n_unique = len(unique_ids)
    if n_unique < 2:
        return np.asarray(coords_um, dtype=np.float32)

    # Vectorised centroid computation via scatter-add
    sum_xy = np.zeros((n_unique, 2), dtype=np.float64)
    np.add.at(sum_xy, inverse, coords_um.astype(np.float64))
    centroids = sum_xy / counts[:, None].clip(1)  # (n_unique, 2)

    new_centroids = centroids[rng.permutation(n_unique)]
    delta = new_centroids - centroids

    return (coords_um.astype(np.float64) + delta[inverse]).astype(np.float32)


def build_block_permuted_coordinate_batch(
    S: np.ndarray,
    *,
    radius_um: float,
    coordinate_um_per_unit: float,
    n_perms: int,
    seed: int,
    cell_type_labels: np.ndarray | None = None,
    n_cell_types: int | None = None,
    block_jitter: bool = True,
) -> np.ndarray:
    """Build an ``(n_perms+1, N, 2)`` block-permuted coordinate batch.

    Slot 0 contains the true (unmodified) coordinates.
    Slots 1 … n_perms each carry an independent block-centroid permutation with
    an optional random mesh jitter.

    When *cell_type_labels* is given each cell type gets its own hex grid and
    independent jitter seed per slot.

    Parameters
    ----------
    S:
        ``(N, 2)`` true coordinates in native units.
    radius_um:
        Hex radius in **microns**.
    coordinate_um_per_unit:
        Microns per one native coordinate unit.
    n_perms:
        Number of permuted slots to generate.
    seed:
        Master RNG seed (fully determines all permutations and jitter).
    cell_type_labels:
        Optional ``(N,)`` int64 cell-type indices (0-based).
    n_cell_types:
        Number of distinct cell types (required when *cell_type_labels* is set).
    block_jitter:
        When True, shift the hex mesh by a uniform random offset in
        ``(-radius_um, radius_um)^2`` independently per slot (and per type).

    Returns
    -------
    s_batched:
        ``(n_perms+1, N, 2)`` float32 in native coordinate units.
    """
    S = np.asarray(S, dtype=np.float32)
    n_cells = S.shape[0]
    n_models = n_perms + 1
    um_scale = float(coordinate_um_per_unit)

    S_um = S.astype(np.float64) * um_scale

    s_batched = np.empty((n_models, n_cells, 2), dtype=np.float32)
    s_batched[0] = S  # slot 0: true coordinates unchanged

    # Pre-draw per-slot seeds for full determinism
    master_rng = np.random.default_rng(int(seed))
    slot_seeds = master_rng.integers(0, 2**31, size=n_models)

    for slot in range(1, n_models):
        if cell_type_labels is None:
            rng_slot = np.random.default_rng(int(slot_seeds[slot]))
            if block_jitter:
                jx = float(rng_slot.uniform(-radius_um, radius_um))
                jy = float(rng_slot.uniform(-radius_um, radius_um))
            else:
                jx, jy = 0.0, 0.0
            bids = hex_bin_ids(S_um, radius_um, (jx, jy))
            new_um = block_centroid_permute(S_um, bids, rng_slot).astype(np.float64)
        else:
            assert n_cell_types is not None
            slot_parent = np.random.default_rng(int(slot_seeds[slot]))
            type_seeds = slot_parent.integers(0, 2**31, size=int(n_cell_types))
            new_um = S_um.copy()
            for ct in range(int(n_cell_types)):
                ct_mask = cell_type_labels == ct
                if int(ct_mask.sum()) == 0:
                    continue
                rng_ct = np.random.default_rng(int(type_seeds[ct]))
                S_ct = S_um[ct_mask]
                if block_jitter:
                    jx = float(rng_ct.uniform(-radius_um, radius_um))
                    jy = float(rng_ct.uniform(-radius_um, radius_um))
                else:
                    jx, jy = 0.0, 0.0
                bids_ct = hex_bin_ids(S_ct, radius_um, (jx, jy))
                new_um[ct_mask] = block_centroid_permute(
                    S_ct, bids_ct, rng_ct
                ).astype(np.float64)

        s_batched[slot] = (new_um / um_scale).astype(np.float32)

    return s_batched


def block_stats(
    S: np.ndarray,
    radius_um: float,
    coordinate_um_per_unit: float,
) -> dict:
    """Diagnostic statistics for the hex tiling of true coordinates.

    Returns a dict with keys:
    ``n_blocks``, ``mean_cells``, ``median_cells``, ``min_cells``, ``max_cells``.
    """
    counts = block_occupancy_counts(S, radius_um, coordinate_um_per_unit)
    return {
        "n_blocks": int(len(counts)),
        "mean_cells": float(counts.mean()) if len(counts) else 0.0,
        "median_cells": float(np.median(counts)) if len(counts) else 0.0,
        "min_cells": int(counts.min()) if len(counts) else 0,
        "max_cells": int(counts.max()) if len(counts) else 0,
    }


def block_occupancy_counts(
    S: np.ndarray,
    radius_um: float,
    coordinate_um_per_unit: float,
) -> np.ndarray:
    """Return per-occupied-block cell counts for a hex tiling of true coordinates."""
    S_um = np.asarray(S, dtype=np.float64) * float(coordinate_um_per_unit)
    bids = hex_bin_ids(S_um, radius_um, (0.0, 0.0))
    _unique, counts = np.unique(bids, return_counts=True)
    return np.asarray(counts, dtype=np.int64)


def resolve_um_per_unit(
    config_value: float | None,
    meta_value: float | None,
) -> float:
    """Resolve microns-per-coordinate-unit from config or auto-detected metadata.

    Priority: config value > auto-detected from file > 1.0 with warning.
    """
    if config_value is not None:
        return float(config_value)
    if meta_value is not None:
        return float(meta_value)
    warnings.warn(
        "Block permutation: coordinate_um_per_unit not specified and could not be "
        "auto-detected from file metadata.  Treating native coordinates as microns "
        "(scale=1.0).  Set test.coordinate_um_per_unit in config if coordinates are "
        "not already in microns.",
        UserWarning,
        stacklevel=3,
    )
    return 1.0
