"""Global Moran's I diagnostics for permutation / MSR null slots."""
from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np

from methods.block_permutation import resolve_um_per_unit


DEFAULT_MORAN_NEIGHBOR_RADIUS_UM = 30.0


def _moran_skip_artifacts(radius_um: float, reason: str) -> dict[str, Any]:
    return {
        "moran_skipped": True,
        "moran_skip_reason": reason,
        "moran_neighbor_radius_um": float(radius_um),
    }


def build_inverse_distance_weights(
    S_um: np.ndarray,
    radius_um: float,
    *,
    allow_empty: bool = False,
) -> tuple[np.ndarray, float] | None:
    """Symmetric W with w_ij = 1/d_ij for pairs with d < radius_um.

    Returns None when ``allow_empty=True`` and no pairs fall within the radius.
    """
    from scipy.spatial import KDTree

    S_um = np.asarray(S_um, dtype=np.float64)
    radius = float(radius_um)
    tree = KDTree(S_um)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if pairs.shape[0] == 0:
        if allow_empty:
            return None
        raise ValueError(
            f"No cell pairs within neighbor radius {radius_um:.1f} µm. "
            "Increase moran_neighbor_radius_um."
        )

    d_vals = np.linalg.norm(S_um[pairs[:, 0]] - S_um[pairs[:, 1]], axis=1)
    d_min = max(float(np.median(d_vals)) * 1e-4, 1e-9)
    w_vals = 1.0 / np.maximum(d_vals, d_min)

    n_cells = int(S_um.shape[0])
    W = np.zeros((n_cells, n_cells), dtype=np.float64)
    W[pairs[:, 0], pairs[:, 1]] = w_vals
    W[pairs[:, 1], pairs[:, 0]] = w_vals
    s0 = float(W.sum())
    return W, s0


def morans_i_per_gene(W: np.ndarray, s0: float, A: np.ndarray) -> np.ndarray:
    """Global Moran's I for each gene column of A (z-scored per gene)."""
    A64 = np.asarray(A, dtype=np.float64)
    n_cells, n_genes = A64.shape
    n = float(n_cells)
    I = np.empty(n_genes, dtype=np.float64)

    for g in range(n_genes):
        x = A64[:, g]
        xc = x - x.mean()
        denom = float(xc @ xc)
        if denom <= 1e-12:
            I[g] = np.nan
            continue
        numer = float(xc @ W @ xc)
        I[g] = (n / s0) * (numer / denom)
    return I


def native_coords_batched_from_standardized(
    s_batched: np.ndarray,
    meta: Mapping[str, Any],
) -> np.ndarray:
    """Invert coordinate z-scoring for each slot when applicable."""
    s = np.asarray(s_batched, dtype=np.float64)
    if meta.get("coordinate_standardization") != "zscore":
        return s.astype(np.float32)
    mean = np.asarray(meta["coord_mean"], dtype=np.float64)
    std = np.asarray(meta["coord_std"], dtype=np.float64)
    return ((s * std) + mean).astype(np.float32)


def estimate_morans_i_slots(
    s_batched_native: np.ndarray,
    a_batched: np.ndarray,
    *,
    um_per_unit: float,
    neighbor_radius_um: float,
) -> np.ndarray | None:
    """Moran's I per gene for each slot: shape (n_slots, n_genes).

    Returns None when no cell pairs fall within ``neighbor_radius_um`` for any slot.
    """
    s_batched_native = np.asarray(s_batched_native, dtype=np.float32)
    a_batched = np.asarray(a_batched, dtype=np.float32)
    n_slots = int(s_batched_native.shape[0])
    n_genes = int(a_batched.shape[-1])
    out = np.empty((n_slots, n_genes), dtype=np.float64)

    # Fixed coordinates: reuse one weight matrix.
    if n_slots > 1 and np.allclose(s_batched_native[0], s_batched_native[1:]):
        S_um = np.asarray(s_batched_native[0], dtype=np.float64) * float(um_per_unit)
        built = build_inverse_distance_weights(
            S_um, neighbor_radius_um, allow_empty=True,
        )
        if built is None:
            return None
        W, s0 = built
        for slot in range(n_slots):
            out[slot] = morans_i_per_gene(W, s0, a_batched[slot])
        return out

    for slot in range(n_slots):
        S_um = np.asarray(s_batched_native[slot], dtype=np.float64) * float(um_per_unit)
        built = build_inverse_distance_weights(
            S_um, neighbor_radius_um, allow_empty=True,
        )
        if built is None:
            return None
        W, s0 = built
        out[slot] = morans_i_per_gene(W, s0, a_batched[slot])
    return out


def summarize_moran_slots(I_by_slot: np.ndarray) -> dict[str, Any]:
    """Summarize per-slot Moran's I (slot 0 = true, slots 1.. = null)."""
    I_by_slot = np.asarray(I_by_slot, dtype=np.float64)
    if I_by_slot.ndim != 2 or I_by_slot.shape[0] < 1:
        raise ValueError(f"I_by_slot must be 2D with at least one row, got {I_by_slot.shape}")

    mean_per_slot = np.nanmean(I_by_slot, axis=1)
    true_mean = float(mean_per_slot[0])
    null_means = mean_per_slot[1:]
    n_perms = int(null_means.shape[0])
    if n_perms == 0:
        p_value = 1.0
        rank = 1
    else:
        rank = int(np.sum(null_means < true_mean)) + 1
        p_value = float(rank) / float(n_perms + 1)

    return {
        "moran_skipped": False,
        "moran_i_per_gene_per_slot": I_by_slot,
        "moran_mean_per_slot": mean_per_slot,
        "moran_true_mean": true_mean,
        "moran_true_i_per_gene": I_by_slot[0],
        "moran_null_mean_per_perm": null_means,
        "moran_null_i_per_gene_per_perm": I_by_slot[1:],
        "moran_p_value": p_value,
        "moran_rank": int(rank),
        "moran_n_slots": int(I_by_slot.shape[0]),
        "moran_n_perms": n_perms,
    }


def compute_moran_permutation_diagnostics(
    s_batched_native: np.ndarray,
    a_batched: np.ndarray,
    *,
    um_per_unit: float,
    neighbor_radius_um: float = DEFAULT_MORAN_NEIGHBOR_RADIUS_UM,
) -> dict[str, Any] | None:
    """Build local inverse-distance W (radius in µm) and Moran's I for each slot."""
    I_by_slot = estimate_morans_i_slots(
        s_batched_native,
        a_batched,
        um_per_unit=float(um_per_unit),
        neighbor_radius_um=float(neighbor_radius_um),
    )
    if I_by_slot is None:
        return None
    summary = summarize_moran_slots(I_by_slot)
    summary["moran_neighbor_radius_um"] = float(neighbor_radius_um)
    return summary


def maybe_compute_moran_artifacts(
    config: Any,
    dataset_meta: Mapping[str, Any],
    s_batched: np.ndarray,
    A: np.ndarray,
    *,
    a_batched: np.ndarray | None = None,
    s_batched_native: np.ndarray | None = None,
    coord_mean: np.ndarray | None = None,
    coord_std: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return Moran's I artifact dict when ``config.moran`` is enabled."""
    if not getattr(config, "moran", False):
        return {}

    um_per_unit = resolve_um_per_unit(
        getattr(config, "coordinate_um_per_unit", None),
        dataset_meta.get("coordinate_um_per_unit"),
    )
    radius_um = float(
        getattr(config, "moran_neighbor_radius_um", DEFAULT_MORAN_NEIGHBOR_RADIUS_UM)
    )

    meta = dict(dataset_meta)
    if coord_mean is not None and coord_std is not None:
        meta["coord_mean"] = np.asarray(coord_mean, dtype=np.float32)
        meta["coord_std"] = np.asarray(coord_std, dtype=np.float32)
        meta["coordinate_standardization"] = "zscore"

    if s_batched_native is not None:
        s_native = np.asarray(s_batched_native, dtype=np.float32)
    else:
        s_native = native_coords_batched_from_standardized(s_batched, meta)

    A_np = np.asarray(A, dtype=np.float32)
    if a_batched is None:
        n_slots = int(s_native.shape[0])
        a_slots = np.tile(A_np[np.newaxis, :, :], (n_slots, 1, 1))
    else:
        a_slots = np.asarray(a_batched, dtype=np.float32)

    if config.verbose:
        print(
            f"Moran's I: neighbor_radius={radius_um:.1f} µm, "
            f"slots={s_native.shape[0]}, genes={A_np.shape[1]}",
            flush=True,
        )

    summary = compute_moran_permutation_diagnostics(
        s_native,
        a_slots,
        um_per_unit=um_per_unit,
        neighbor_radius_um=radius_um,
    )
    if summary is None:
        reason = (
            f"No cell pairs within neighbor radius {radius_um:.1f} µm "
            f"({int(s_native.shape[1])} cells). "
            "Increase test.moran_neighbor_radius_um or disable test.moran."
        )
        warnings.warn(f"Moran's I skipped: {reason}", stacklevel=2)
        if config.verbose:
            print(f"Moran's I skipped: {reason}", flush=True)
        return _moran_skip_artifacts(radius_um, reason)

    return summary
