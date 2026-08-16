"""Moran Spectral Randomization (MSR) surrogate generation for the main pipeline.

Implements joint truncated MSR with radius-based inverse-distance spatial weights.
The W matrix is built by finding all cell pairs within ``msr_neighbor_radius_um``
and weighting each pair by ``1 / d_{ij}``, then doubly-centering to form the Moran
operator Ω, decomposing into Moran Eigenvector Maps (MEMs), and applying a joint
sign-flip on long-range modes (scale > ``msr_truncate_um``).

This module is used exclusively by the main pipeline (methods/permutation.py).
The smoke-test scripts in scripts/ retain their own independent copies.
"""
from __future__ import annotations

import time
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _msr_basis(
    S: np.ndarray,
    A: np.ndarray,
    radius: float,
    *,
    return_eigvals: bool = False,
) -> tuple:
    """Build radius-based inverse-distance graph → Omega → eigendecomposition.

    Parameters
    ----------
    S : (N, 2) spatial coordinates in physical µm.
    A : (N, G) expression matrix (already log-normalised).
    radius : float — include all cell pairs with d < radius (same units as S).
    return_eigvals : if True, return (V, C, eigvals); else return (V, C).

    Returns
    -------
    V : (N, K) Moran eigenvectors (MEMs), K non-trivial modes.
    C : (K, G) spectral coefficients = V^T @ A.
    eigvals : (K,) eigenvalues, only when return_eigvals=True.
    """
    from scipy.spatial import KDTree

    S = np.asarray(S, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    N, G = A.shape

    print(f"    MSR basis: N={N}, G={G}, neighbor_radius={radius:.1f}µm", flush=True)

    # --- build sparse inverse-distance W via radius query --------------------
    tree = KDTree(S)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")  # (n_pairs, 2) i < j

    if pairs.shape[0] == 0:
        raise ValueError(
            f"No cell pairs found within radius {radius:.1f} µm. "
            "Increase msr_neighbor_radius_um so that at least some neighbors exist."
        )

    d_vals = np.linalg.norm(S[pairs[:, 0]] - S[pairs[:, 1]], axis=1)
    d_min = float(np.median(d_vals)) * 1e-4
    d_min = max(d_min, 1e-9)
    w_vals = 1.0 / np.maximum(d_vals, d_min)

    W = np.zeros((N, N), dtype=np.float64)
    W[pairs[:, 0], pairs[:, 1]] = w_vals
    W[pairs[:, 1], pairs[:, 0]] = w_vals

    # warn about isolated cells (zero neighbors)
    n_neighbors = (W > 0).sum(axis=1)
    n_isolated = int((n_neighbors == 0).sum())
    if n_isolated > 0:
        warnings.warn(
            f"MSR: {n_isolated}/{N} cells have zero neighbors within radius "
            f"{radius:.1f} µm. These produce degenerate eigenvectors. "
            "Consider increasing msr_neighbor_radius_um.",
            stacklevel=3,
        )

    print(
        f"    {pairs.shape[0]} pairs | d range [{d_vals.min():.2f}, {d_vals.max():.2f}] µm | "
        f"isolated cells: {n_isolated}",
        flush=True,
    )

    # --- Moran operator (doubly-centered W) ----------------------------------
    rm = W.mean(1, keepdims=True)
    cm = W.mean(0, keepdims=True)
    gm = W.mean()
    Omega = W - rm - cm + gm

    t = time.time()
    eigvals_all, eigvecs_all = np.linalg.eigh(Omega)
    print(f"    Eigendecomp: {time.time()-t:.1f}s", flush=True)

    keep = np.abs(eigvals_all) > 1e-8
    eig_kept = eigvals_all[keep]
    V = eigvecs_all[:, keep]
    print(
        f"    K={V.shape[1]} non-trivial MEMs "
        f"({eig_kept.min():.3f} … {eig_kept.max():.3f})",
        flush=True,
    )

    C = V.T @ A
    if return_eigvals:
        return V, C, eig_kept
    return V, C


def _mem_characteristic_scales_um(
    eigvals: np.ndarray,
    calibration_um: float,
) -> np.ndarray:
    """Map Moran eigenvalues to approximate spatial scale in µm.

    Low |λ| → smooth / long-range MEM; high |λ| → local / short-range MEM.
    scale_um ≈ calibration_um / sqrt(|λ|).
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    return calibration_um / np.sqrt(np.clip(np.abs(eigvals), 1e-8, None))


def _joint_trunc_mode_mask(
    eigvals: np.ndarray,
    truncate_scale_um: float,
    calibration_um: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (scales_um, rand_mask, calibration_um_used).

    rand_mask[k] is True when mode k should be sign-flipped (long-range).
    When truncate_scale_um == 0 all modes are randomised.
    """
    cal_um = float(calibration_um if calibration_um is not None else truncate_scale_um)
    if cal_um <= 0.0:
        # fallback: can't compute meaningful scales, randomise everything
        scales_um = np.zeros_like(np.asarray(eigvals, dtype=np.float64))
        rand_mask = np.ones(len(eigvals), dtype=bool)
        return scales_um, rand_mask, cal_um
    scales_um = _mem_characteristic_scales_um(eigvals, cal_um)
    if truncate_scale_um <= 0.0:
        # randomise all modes but still report actual scale ranges
        rand_mask = np.ones(len(eigvals), dtype=bool)
    else:
        rand_mask = scales_um > float(truncate_scale_um)
    return scales_um, rand_mask, cal_um


def _log_joint_trunc_split(
    scales_um: np.ndarray,
    rand_mask: np.ndarray,
    truncate_scale_um: float,
    K_modes: int,
) -> None:
    n_rand = int(rand_mask.sum())
    n_keep = K_modes - n_rand
    print(
        f"    Mode split: randomise {n_rand}/{K_modes} long-range modes "
        f"(scale>{truncate_scale_um}µm), keep {n_keep} short-range fixed",
        flush=True,
    )
    if n_rand > 0 and scales_um.size > 0:
        rand_scales = scales_um[rand_mask]
        print(
            f"    Randomised scale range: {rand_scales.min():.1f} … {rand_scales.max():.1f} µm",
            flush=True,
        )
    if n_keep > 0 and scales_um.size > 0:
        kept_scales = scales_um[~rand_mask]
        print(
            f"    Kept scale range: {kept_scales.min():.1f} … {kept_scales.max():.1f} µm",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_joint_truncated_msr_surrogates(
    S: np.ndarray,
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    *,
    radius: float,
    truncate_scale_um: float = 0.0,
    calibration_um: float | None = None,
) -> np.ndarray:
    """Build joint truncated MSR expression surrogates.

    Each surrogate preserves spatial autocorrelation structure at scales shorter
    than ``truncate_scale_um`` while destroying long-range spatial patterns.
    When ``truncate_scale_um == 0``, all Moran modes are randomised (plain joint MSR).

    Parameters
    ----------
    S : (N, 2) physical coordinates in µm (already converted from standardised).
    A : (N, G) log-normalised expression matrix (the pipeline's ``dataset.A``).
    n_surrogates : number of surrogate matrices to generate.
    seed : RNG seed for reproducibility.
    radius : neighbour-graph radius in µm for the Moran weight matrix.
    truncate_scale_um : only modes with characteristic scale > this value are
        sign-flipped; 0 means randomise all modes.
    calibration_um : reference µm for eigenvalue→scale mapping; defaults to
        ``truncate_scale_um`` when None (or to ``radius`` when both are 0).

    Returns
    -------
    surrogates : (n_surrogates, N, G) float32 expression arrays.
    """
    S = np.asarray(S, dtype=np.float64)
    A = np.asarray(A, dtype=np.float32)
    N, G = A.shape

    # use radius as calibration fallback when truncate_scale_um == 0
    cal_um = calibration_um
    if cal_um is None:
        cal_um = float(truncate_scale_um) if truncate_scale_um > 0.0 else float(radius)

    print(
        f"    Joint trunc-MSR: N={N}, G={G}, "
        f"neighbor_radius={radius:.1f}µm, truncate>{truncate_scale_um}µm, "
        f"cal={cal_um:.1f}µm",
        flush=True,
    )
    t0 = time.time()

    V, C, eigvals = _msr_basis(S, A, radius, return_eigvals=True)
    K_modes = V.shape[1]

    scales_um, rand_mask, _ = _joint_trunc_mode_mask(eigvals, truncate_scale_um, cal_um)
    _log_joint_trunc_split(scales_um, rand_mask, truncate_scale_um, K_modes)
    n_rand = int(rand_mask.sum())

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    ts = time.time()
    for i in range(n_surrogates):
        signs = np.ones(K_modes, dtype=np.float64)
        if n_rand > 0:
            signs[rand_mask] = rng.integers(0, 2, size=n_rand) * 2 - 1
        # joint flip: same sign per mode for all genes
        surrogates[i] = (V @ (C * signs[:, np.newaxis])).astype(np.float32)

    print(
        f"    Surrogates: {time.time()-ts:.1f}s | "
        f"total joint trunc-MSR: {time.time()-t0:.1f}s",
        flush=True,
    )

    # sanity checks
    orig_var = float((A ** 2).mean())
    surr_var = float((surrogates[0] ** 2).mean())
    print(
        f"    Variance check: orig={orig_var:.4f}  surrogate[0]={surr_var:.4f}",
        flush=True,
    )
    if G <= 300:
        corr_o = np.corrcoef(A.T)
        corr_s = np.corrcoef(surrogates[0].T)
        m = np.triu_indices(G, k=1)
        print(
            f"    Gene corr: orig off-diag mean={corr_o[m].mean():.4f}  "
            f"surr[0]={corr_s[m].mean():.4f}  "
            "(joint MSR preserves more than per-gene)",
            flush=True,
        )

    return surrogates


def build_joint_truncated_rank_msr_surrogates(
    S: np.ndarray,
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    *,
    radius: float,
    truncate_scale_um: float = 0.0,
    calibration_um: float | None = None,
    shared_rank: bool = False,
) -> np.ndarray:
    """Build rank-matched joint truncated MSR expression surrogates.

    Same truncated joint sign-flip reference as ``build_joint_truncated_msr_surrogates``,
    but instead of using the reference values directly, rank-matches the original
    per-gene expression values into the reference ordering (BrainSMASH-style).

    Preserves exactly:
      - Per-gene marginal distribution (same multiset of values per gene)

    When ``shared_rank=True``, one spatial permutation derived from the mean
    reference field is applied to all genes.
    """
    S = np.asarray(S, dtype=np.float64)
    A = np.asarray(A, dtype=np.float32)
    N, G = A.shape

    cal_um = calibration_um
    if cal_um is None:
        cal_um = float(truncate_scale_um) if truncate_scale_um > 0.0 else float(radius)

    mode = "shared" if shared_rank else "per-gene"
    print(
        f"    Joint trunc rank-MSR ({mode}): N={N}, G={G}, "
        f"neighbor_radius={radius:.1f}µm, truncate>{truncate_scale_um}µm, "
        f"cal={cal_um:.1f}µm",
        flush=True,
    )
    t0 = time.time()

    V, C, eigvals = _msr_basis(S, A, radius, return_eigvals=True)
    K_modes = V.shape[1]

    scales_um, rand_mask, _ = _joint_trunc_mode_mask(eigvals, truncate_scale_um, cal_um)
    _log_joint_trunc_split(scales_um, rand_mask, truncate_scale_um, K_modes)
    n_rand = int(rand_mask.sum())

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    sorted_vals = np.sort(A, axis=0)

    ts = time.time()
    for i in range(n_surrogates):
        signs = np.ones(K_modes, dtype=np.float64)
        if n_rand > 0:
            signs[rand_mask] = rng.integers(0, 2, size=n_rand) * 2 - 1
        ref = (V @ (C * signs[:, np.newaxis])).astype(np.float32)
        surr = np.empty_like(A)
        if shared_rank:
            ranks = np.argsort(ref.mean(axis=1))
            for g in range(G):
                surr[ranks, g] = sorted_vals[:, g]
        else:
            ranks = np.argsort(ref, axis=0)
            for g in range(G):
                surr[ranks[:, g], g] = sorted_vals[:, g]
        surrogates[i] = surr

    print(
        f"    Surrogates: {time.time()-ts:.1f}s | "
        f"total joint trunc rank-MSR: {time.time()-t0:.1f}s",
        flush=True,
    )

    orig_var = float((A ** 2).mean())
    surr_var = float((surrogates[0] ** 2).mean())
    print(
        f"    Variance check: orig={orig_var:.4f}  surrogate[0]={surr_var:.4f}",
        flush=True,
    )
    if G <= 300:
        orig_sorted = np.sort(A, axis=0)
        surr_sorted = np.sort(surrogates[0], axis=0)
        max_diff = float(np.abs(orig_sorted - surr_sorted).max())
        print(
            f"    Value preservation: max diff={max_diff:.2e}  (should be ~0)",
            flush=True,
        )

    return surrogates
