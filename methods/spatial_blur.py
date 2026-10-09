"""Nadaraya–Watson Gaussian blur of a cell-by-gene expression field."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree


DEFAULT_BLUR_CUTOFF_MULT = 4.0
# One float64 N×N weight matrix at this size is about 512 MB. Larger fields
# use a sparse matrix instead of materializing every pairwise distance.
_DENSE_WEIGHT_MAX_CELLS = 8000


def gaussian_smoothing_weights(
    S: np.ndarray,
    sigma: float,
    *,
    cutoff_mult: float = DEFAULT_BLUR_CUTOFF_MULT,
) -> np.ndarray | csr_matrix:
    """Row-normalized Gaussian weights on ``S``.

    ``W_ij ∝ exp(-‖x_i-x_j‖² / (2σ²))`` for neighbors within ``cutoff_mult * σ``,
    including the spot itself. The same matrix smooths every permutation,
    because the kernel depends only on coordinates.
    """
    coords = np.asarray(S, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"S must be (n_cells, 2), got {coords.shape}")
    bandwidth = float(sigma)
    if bandwidth <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    cutoff = float(cutoff_mult)
    if cutoff <= 0.0:
        raise ValueError(f"cutoff_mult must be > 0, got {cutoff_mult}")

    n_cells = int(coords.shape[0])
    if n_cells == 0:
        return np.zeros((0, 0), dtype=np.float64)
    radius = cutoff * bandwidth
    inv_two_sigma_sq = 1.0 / (2.0 * bandwidth * bandwidth)
    if _kernel_is_dense(coords, radius):
        return _dense_gaussian_weights(coords, radius, inv_two_sigma_sq)
    return _sparse_gaussian_weights(coords, radius, inv_two_sigma_sq)


def _kernel_is_dense(coords: np.ndarray, radius: float) -> bool:
    """Use a full matrix only when most spots fall inside the kernel.

    A local bandwidth on thousands of cells is almost all zeros. Multiplying
    that pattern as a dense matrix still costs a multiply per spot pair.
    """
    n_cells = int(coords.shape[0])
    if n_cells <= 64:
        return True
    if n_cells > _DENSE_WEIGHT_MAX_CELLS:
        return False
    sample_count = min(32, n_cells)
    sample = coords[np.linspace(0, n_cells - 1, num=sample_count, dtype=int)]
    counts = [len(neighbors) for neighbors in KDTree(coords).query_ball_point(sample, r=radius)]
    return float(np.median(counts)) > 0.25 * n_cells


def apply_gaussian_weights(
    weights: np.ndarray | csr_matrix | None,
    A: np.ndarray,
) -> np.ndarray:
    """Apply a precomputed smoother. ``weights is None`` is the identity."""
    expression = np.asarray(A, dtype=np.float64)
    if weights is None:
        return expression.astype(np.float32, copy=False)
    if expression.ndim == 1:
        expression = expression[:, None]
        squeezed = True
    else:
        squeezed = False
    smoothed = np.asarray(weights @ expression, dtype=np.float64)
    if squeezed:
        smoothed = smoothed[:, 0]
    return smoothed.astype(np.float32, copy=False)


def gaussian_blur_expression(
    S_um: np.ndarray,
    A: np.ndarray,
    sigma_um: float,
    *,
    cutoff_mult: float = DEFAULT_BLUR_CUTOFF_MULT,
) -> np.ndarray:
    """Row-normalized Gaussian smooth of ``A`` using distances in microns.

    ``A'_i = Σ_j w_ij A_j`` with ``w_ij ∝ exp(-‖x_i-x_j‖² / (2σ²))`` for
    neighbors within ``cutoff_mult * σ`` (self is always included).  Matches the
    ``kernel.type="smooth"`` field used in synthetic kernel-noise generation.
    """
    S_um = np.asarray(S_um, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    if S_um.ndim != 2 or S_um.shape[1] != 2:
        raise ValueError(f"S_um must be (n_cells, 2), got {S_um.shape}")
    if A.ndim != 2 or A.shape[0] != S_um.shape[0]:
        raise ValueError(
            f"A must be (n_cells, n_genes) aligned with S_um, got {A.shape} vs {S_um.shape}"
        )
    sigma = float(sigma_um)
    if sigma <= 0.0:
        raise ValueError(f"sigma_um must be > 0, got {sigma_um}")
    if S_um.shape[0] == 0:
        return A.astype(np.float32, copy=False)
    weights = gaussian_smoothing_weights(S_um, sigma, cutoff_mult=cutoff_mult)
    return apply_gaussian_weights(weights, A)


def _dense_gaussian_weights(
    coords: np.ndarray,
    radius: float,
    inv_two_sigma_sq: float,
) -> np.ndarray:
    squared_norm = np.sum(coords * coords, axis=1)
    distance_sq = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (coords @ coords.T)
    np.maximum(distance_sq, 0.0, out=distance_sq)
    np.fill_diagonal(distance_sq, 0.0)
    within = distance_sq <= (radius * radius)
    weights = np.exp(-distance_sq * inv_two_sigma_sq)
    weights[~within] = 0.0
    row_sum = weights.sum(axis=1, keepdims=True)
    weights /= np.maximum(row_sum, 1e-300)
    return weights


def _sparse_gaussian_weights(
    coords: np.ndarray,
    radius: float,
    inv_two_sigma_sq: float,
) -> csr_matrix:
    n_cells = int(coords.shape[0])
    neighbor_lists = KDTree(coords).query_ball_point(coords, r=radius)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for i, neighbors in enumerate(neighbor_lists):
        idx = np.asarray(neighbors, dtype=np.int32)
        if idx.size == 0:
            rows.append(np.array([i], dtype=np.int32))
            cols.append(np.array([i], dtype=np.int32))
            values.append(np.ones(1, dtype=np.float64))
            continue
        delta = coords[idx] - coords[i]
        row_weights = np.exp(-np.sum(delta * delta, axis=1) * inv_two_sigma_sq)
        weight_sum = float(row_weights.sum())
        if weight_sum <= 0.0:
            rows.append(np.array([i], dtype=np.int32))
            cols.append(np.array([i], dtype=np.int32))
            values.append(np.ones(1, dtype=np.float64))
            continue
        row_weights /= weight_sum
        rows.append(np.full(idx.size, i, dtype=np.int32))
        cols.append(idx)
        values.append(row_weights)
    return csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_cells, n_cells),
    )


def blur_expression_slots(
    s_um_batched: np.ndarray,
    A: np.ndarray,
    sigma_um: float,
    *,
    cutoff_mult: float = DEFAULT_BLUR_CUTOFF_MULT,
) -> np.ndarray:
    """Blur ``A`` independently on each coordinate slot in ``s_um_batched``."""
    slots = np.asarray(s_um_batched, dtype=np.float64)
    if slots.ndim != 3 or slots.shape[2] != 2:
        raise ValueError(f"s_um_batched must be (n_slots, n_cells, 2), got {slots.shape}")
    return np.stack(
        [
            gaussian_blur_expression(slot, A, sigma_um, cutoff_mult=cutoff_mult)
            for slot in slots
        ],
        axis=0,
    )


def blur_expression_slots_within_groups(
    s_slots_um: np.ndarray,
    expression: np.ndarray,
    sigma_um: float,
    *,
    cutoff_mult: float = DEFAULT_BLUR_CUTOFF_MULT,
    group_labels: np.ndarray | None,
) -> np.ndarray:
    """Blur independently within each integer group so types cannot mix expression."""
    if group_labels is None:
        return blur_expression_slots(
            s_slots_um,
            expression,
            sigma_um,
            cutoff_mult=cutoff_mult,
        )
    labels = np.asarray(group_labels, dtype=np.int64)
    if labels.shape != (expression.shape[0],):
        raise ValueError(
            f"group_labels shape {labels.shape} != ({expression.shape[0]},)"
        )
    blurred = np.empty(
        (s_slots_um.shape[0], expression.shape[0], expression.shape[1]),
        dtype=np.float32,
    )
    for group in np.unique(labels):
        mask = labels == group
        blurred[:, mask, :] = blur_expression_slots(
            s_slots_um[:, mask, :],
            expression[mask],
            sigma_um,
            cutoff_mult=cutoff_mult,
        )
    return blurred
