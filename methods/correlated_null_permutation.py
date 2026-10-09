"""Variogram-matched correlated null expression surrogates (fixed coordinates)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

from data import raw_coordinates_from_standardized
from data.schemas import DatasetBundle, TestConfig
from methods.spatial_blur import apply_gaussian_weights, gaussian_smoothing_weights

LOCFIT_GAUSSIAN_FACTOR = 2.5
# Row-block size for adaptive weights. The sequential blur and the batched
# backend both build Gaussian rows in blocks of this height.
ADAPTIVE_BLUR_BLOCK_SIZE = 256
DEFAULT_ADAPTIVE_NN_CANDIDATES: tuple[float, ...] = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


@dataclass
class _VariogramGeometry:
    lag_centers: np.ndarray
    max_dist: float
    valid_bins: np.ndarray
    pair_counts: np.ndarray
    bin_matrices: list[csr_matrix]
    degrees: list[np.ndarray]
    pairs: np.ndarray
    bin_index: np.ndarray


@dataclass
class _AdaptiveBlurPlan:
    nn_candidates: np.ndarray
    identity_index: int
    positive_candidate_indices: np.ndarray
    kth_distance_sq: list[np.ndarray]


def default_adaptive_nn_candidates() -> list[float]:
    return [float(v) for v in DEFAULT_ADAPTIVE_NN_CANDIDATES]


def _build_variogram_geometry(coords: np.ndarray, n_bins: int) -> _VariogramGeometry:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n_cells, 2), got {coords.shape}")
    n_cells = coords.shape[0]
    if n_cells < 3:
        raise ValueError("Need at least 3 cells for variogram matching.")

    all_distances = pdist(coords, metric="euclidean")
    max_dist = float(np.quantile(all_distances, 0.25))
    below_cutoff = all_distances[all_distances < max_dist]
    if below_cutoff.size == 0 or all_distances.min() <= 1e-12:
        raise ValueError("Expected distinct, separated coordinates for variogram bins.")
    bin_max_dist = max_dist
    if max_dist <= 0:
        raise ValueError("The 25th-percentile distance must be positive.")

    bin_edges = np.linspace(0.0, bin_max_dist, n_bins + 1)
    bin_edges[0] = 1e-12
    lag_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    pairs = cKDTree(coords).query_pairs(max_dist, output_type="ndarray")
    if pairs.size == 0:
        raise ValueError("No spatial pairs within variogram max.dist.")
    pair_distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    inside = (pair_distances >= bin_edges[0]) & (pair_distances < bin_edges[-1])
    pairs = pairs[inside]
    pair_distances = pair_distances[inside]
    bin_index = np.searchsorted(bin_edges[1:-1], pair_distances, side="right")
    pair_counts = np.bincount(bin_index, minlength=n_bins)
    valid_bins = pair_counts >= 2

    bin_matrices: list[csr_matrix] = []
    degrees: list[np.ndarray] = []
    for k in range(n_bins):
        mask = bin_index == k
        if not np.any(mask):
            bin_matrices.append(csr_matrix((n_cells, n_cells), dtype=np.float64))
            degrees.append(np.zeros(n_cells, dtype=np.float64))
            continue
        p_k = pairs[mask]
        rows = np.concatenate([p_k[:, 0], p_k[:, 1]])
        cols = np.concatenate([p_k[:, 1], p_k[:, 0]])
        data = np.ones(rows.size, dtype=np.float64)
        b_k = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        bin_matrices.append(b_k)
        degrees.append(np.asarray(b_k.sum(axis=1)).ravel())

    return _VariogramGeometry(
        lag_centers=lag_centers,
        max_dist=max_dist,
        valid_bins=valid_bins,
        pair_counts=pair_counts,
        bin_matrices=bin_matrices,
        degrees=degrees,
        pairs=pairs,
        bin_index=bin_index,
    )


def _classical_variogram(expression: np.ndarray, geom: _VariogramGeometry) -> np.ndarray:
    """Semivariance per bin and gene, shape (n_bins, n_genes)."""
    y = np.asarray(expression, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n_bins = len(geom.bin_matrices)
    n_genes = y.shape[1]
    out = np.full((n_bins, n_genes), np.nan, dtype=np.float64)
    y_sq = y * y
    for k, (b_k, deg) in enumerate(zip(geom.bin_matrices, geom.degrees)):
        if not geom.valid_bins[k]:
            continue
        cross = y * np.asarray(b_k @ y, dtype=np.float64)
        numer = (deg @ y_sq) - cross.sum(axis=0)
        out[k] = numer / (2.0 * float(geom.pair_counts[k]))
    return out


def _median_nearest_neighbor_distance(coords: np.ndarray) -> float:
    dist = cKDTree(coords).query(coords, k=2)[0][:, 1]
    return float(np.median(dist))


def _fit_variogram_sse(
    gamma_x: np.ndarray,
    gamma_y_candidates: np.ndarray,
    valid_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit gamma_x = a * gamma_y + b with nonnegative variance components.

    Evaluate the feasible interior solution and both boundary solutions. These
    coefficients are mixed into the surrogate. The bandwidth is chosen later
    from the restored surrogate's empirical variogram, not from this residual.
    Taking abs() of unconstrained coefficients would mix a different field
    from the one these nonnegative coefficients describe.
    """
    n_candidates, _n_bins, n_genes = gamma_y_candidates.shape
    v = np.flatnonzero(valid_bins)
    if v.size < 2:
        raise ValueError("Need at least two valid variogram bins for fitting.")
    target = gamma_x[v, :]
    sse = np.empty((n_candidates, n_genes), dtype=np.float64)
    intercept = np.empty_like(sse)
    slope = np.empty_like(sse)
    target_mean = target.mean(axis=0)
    for k in range(n_candidates):
        candidate = gamma_y_candidates[k, v, :]
        candidate_mean = candidate.mean(axis=0)
        centered = candidate - candidate_mean
        denom = np.sum(centered**2, axis=0)
        a = np.divide(
            np.sum(centered * (target - target_mean), axis=0), denom,
            out=np.zeros(n_genes), where=denom > 0,
        )
        b = target_mean - a * candidate_mean
        norm_sq = np.sum(candidate**2, axis=0)
        boundary_a = np.divide(
            np.sum(candidate * target, axis=0), norm_sq,
            out=np.zeros(n_genes), where=norm_sq > 0,
        )
        slopes = np.stack([a, np.maximum(boundary_a, 0), np.zeros(n_genes)])
        intercepts = np.stack([b, np.zeros(n_genes), np.maximum(target_mean, 0)])
        errors = np.sum(
            (target[None] - slopes[:, None] * candidate[None] - intercepts[:, None])**2,
            axis=1,
        )
        errors[0, (a < 0) | (b < 0)] = np.inf
        choice = errors.argmin(axis=0)
        genes = np.arange(n_genes)
        sse[k] = errors[choice, genes]
        slope[k] = slopes[choice, genes]
        intercept[k] = intercepts[choice, genes]
    return sse, intercept, slope


def _apply_blur_weights(
    weights: np.ndarray | csr_matrix,
    expression: np.ndarray,
) -> np.ndarray:
    return apply_gaussian_weights(weights, expression)


def _precompute_adaptive_blur_plan(
    coords: np.ndarray,
    nn_candidates: list[float],
) -> _AdaptiveBlurPlan:
    coords = np.asarray(coords, dtype=np.float64)
    nn = np.asarray([float(v) for v in nn_candidates], dtype=np.float64)
    if nn.size == 0:
        raise ValueError("nn_candidates must be non-empty")
    if np.any((nn < 0) | (nn > 1)):
        raise ValueError("variogram_nn_candidates must lie in [0, 1]")
    identity_matches = np.flatnonzero(nn == 0.0)
    if identity_matches.size != 1:
        raise ValueError("variogram_nn_candidates must include exactly one nn=0 entry")
    identity_index = int(identity_matches[0])
    positive = np.flatnonzero(nn > 0)
    n_cells = int(coords.shape[0])
    neighbor_counts = np.ceil(nn[positive] * (n_cells - 1)).astype(int)
    kth_distance_sq: list[np.ndarray] = []
    if positive.size:
        distance_sq = cdist(coords, coords, metric="sqeuclidean")
        for count in neighbor_counts:
            if count <= 0 or count >= n_cells:
                raise ValueError(f"Invalid adaptive neighbor count {count} for n_cells={n_cells}")
            kth = np.partition(distance_sq, count, axis=1)[:, count]
            if np.any(kth <= 0):
                raise ValueError("Adaptive bandwidth is zero; check for duplicate coordinates.")
            kth_distance_sq.append(kth.astype(np.float64, copy=False))
    return _AdaptiveBlurPlan(
        nn_candidates=nn,
        identity_index=identity_index,
        positive_candidate_indices=positive,
        kth_distance_sq=kth_distance_sq,
    )


def _adaptive_gaussian_weight_block(
    coords: np.ndarray,
    kth_sq: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    """Row-normalized locfit Gaussian weights for cells ``start:stop``.

    The weights depend only on coordinates and the candidate bandwidth
    ``kth_sq``, not on expression.
    """
    distance2 = cdist(coords[start:stop], coords, metric="sqeuclidean")
    weights = np.exp(
        -0.5 * LOCFIT_GAUSSIAN_FACTOR**2 * distance2 / kth_sq[start:stop, None]
    )
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    return weights


def _adaptive_blur_expression(
    x_pi: np.ndarray,
    coords: np.ndarray,
    plan: _AdaptiveBlurPlan,
    *,
    block_size: int = ADAPTIVE_BLUR_BLOCK_SIZE,
) -> np.ndarray:
    """STCompare/locfit adaptive Gaussian smooth; the nn=0 candidate is raw.

    This is the reference implementation. It rebuilds coordinate weights on
    every call. The batched backend reuses those weights instead.
    """
    x_pi = np.asarray(x_pi, dtype=np.float64)
    n_cells, n_genes = x_pi.shape
    n_candidates = int(plan.nn_candidates.size)
    out = np.empty((n_candidates, n_cells, n_genes), dtype=np.float64)
    out[plan.identity_index] = x_pi
    if plan.positive_candidate_indices.size == 0:
        return out
    for cand_idx, kth_sq in zip(plan.positive_candidate_indices, plan.kth_distance_sq):
        blurred = np.empty_like(x_pi)
        for start in range(0, n_cells, block_size):
            stop = min(start + block_size, n_cells)
            weights = _adaptive_gaussian_weight_block(coords, kth_sq, start, stop)
            blurred[start:stop] = weights @ x_pi
        out[cand_idx] = blurred
    return out


def _mix_matched_expression(
    y_best: np.ndarray,
    a_best: np.ndarray,
    b_best: np.ndarray,
    r_noise: np.ndarray,
    best_k: np.ndarray,
    identity_index: int | None,
) -> np.ndarray:
    mixed = np.empty_like(y_best)
    n_genes = y_best.shape[1]
    for g in range(n_genes):
        if identity_index is not None and int(best_k[g]) == identity_index:
            mixed[:, g] = y_best[:, g]
            continue
        mixed[:, g] = (
            np.sqrt(a_best[g]) * y_best[:, g]
            + np.sqrt(b_best[g]) * r_noise[:, g]
        )
    return mixed


def _strictly_better(candidate: np.ndarray, best: np.ndarray) -> np.ndarray:
    """Columns where ``candidate`` would replace ``best`` under ``np.argmin``.

    Ties keep the earlier candidate. A NaN replaces a numeric value and does
    not replace an earlier NaN, which is what ``np.argmin`` does on this NumPy.
    """
    return (candidate < best) | (np.isnan(candidate) & ~np.isnan(best))


def _restore_input_moments(
    mixed: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
) -> np.ndarray:
    mixed_std = mixed.std(axis=0, ddof=0)
    if np.any((input_std > 0) & (mixed_std == 0)):
        raise ValueError("Variogram matching generated a constant nonconstant gene")
    safe_std = np.where(mixed_std > 0, mixed_std, 1.0)
    restored = (mixed - mixed.mean(axis=0)) / safe_std
    return restored * input_std + input_mean


def _empirical_variogram_sse(
    gamma: np.ndarray,
    gamma_target: np.ndarray,
    valid_bins: np.ndarray,
) -> np.ndarray:
    difference = gamma[valid_bins] - gamma_target[valid_bins]
    return np.sum(difference**2, axis=0)


def _score_restored_candidate(
    y: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
    r_noise: np.ndarray,
    *,
    identity: bool,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    gamma_target: np.ndarray,
    geom: _VariogramGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix one bandwidth, restore input moments, and score that field.

    ``identity`` keeps the raw permutation: no nugget is added. The returned
    expression is the field whose variogram produced ``sse``.
    """
    y = np.asarray(y, dtype=np.float64)
    if identity:
        mixed = np.array(y, copy=True)
    else:
        mixed = np.sqrt(slope) * y + np.sqrt(intercept) * r_noise
    restored = _restore_input_moments(mixed, input_mean, input_std)
    gamma = _classical_variogram(restored, geom)
    sse = _empirical_variogram_sse(gamma, gamma_target, geom.valid_bins)
    return restored, gamma, sse


def _cuda_variogram_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _normalize_variogram_backend(backend: str) -> str:
    name = str(backend).strip().lower()
    aliases = {
        "sequential": "sequential",
        "reference": "sequential",
        "cpu": "cpu",
        "batched": "cpu",
        "batched_cpu": "cpu",
        "gpu": "cuda",
        "cuda": "cuda",
    }
    if name not in aliases:
        raise ValueError(
            f"Unknown variogram backend {backend!r}. Expected sequential, cpu, or cuda."
        )
    return aliases[name]


def _slot0_variogram_diagnostics(
    *,
    blur_mode: str,
    backend: str,
    input_variance: np.ndarray,
    input_std: np.ndarray,
    input_mean: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    geom: _VariogramGeometry,
    gamma_x: np.ndarray,
    gamma_pi: np.ndarray,
    gamma_y: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
    sse: np.ndarray,
    realized_sse: np.ndarray,
    realized_gamma: np.ndarray,
    best_k: np.ndarray,
    y_best: np.ndarray,
    a_best: np.ndarray,
    b_best: np.ndarray,
    x_pi: np.ndarray,
    x_final: np.ndarray,
    labels: np.ndarray,
    plot_gene_indices: np.ndarray,
    plot_gene_names: list[str],
    extra_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Diagnostics for permutation slot 0, on the original expression scale.

    ``sse`` is the nonnegative coefficient residual on the standardized
    blurred field. ``realized_sse`` and ``realized_gamma`` are the empirical
    variogram of each moment-restored candidate, which is the selection
    objective.
    """
    n_candidates = int(labels.size)
    n_bins = int(geom.lag_centers.size)
    n_plot = int(plot_gene_indices.size)
    fitted_curves = np.empty((n_candidates, n_bins, n_plot), dtype=np.float64)
    for k in range(n_candidates):
        for j, g in enumerate(plot_gene_indices):
            fitted_curves[k, :, j] = (
                (slope[k, g] * gamma_y[k, :, g] + intercept[k, g]) * input_variance[g]
            )
    best_label_slot0 = np.array([labels[int(best_k[g])] for g in plot_gene_indices])
    realized_plot = np.asarray(realized_gamma, dtype=np.float64)[:, :, plot_gene_indices]
    diagnostics = {
        "variogram_blur_mode": blur_mode,
        "variogram_backend": backend,
        "variance_matching": "input_gene_moments",
        "algorithm_version": 3,
        "selection_objective": "final_surrogate_empirical_variogram",
        "variogram_fit": "nonnegative_least_squares",
        "input_gene_variance": input_variance.copy(),
        "surrogate_gene_variance": x_final.var(axis=0),
        "coords": coords.astype(np.float32),
        "lag_centers": geom.lag_centers.astype(np.float32),
        "max_dist": float(geom.max_dist),
        "valid_bins": geom.valid_bins.copy(),
        "plot_gene_indices": plot_gene_indices.copy(),
        "plot_gene_names": plot_gene_names,
        "candidate_labels": labels.astype(np.float32),
        "X_true": (X * input_std + input_mean)[:, plot_gene_indices].astype(np.float32),
        "X_permuted": (x_pi * input_std + input_mean)[:, plot_gene_indices].astype(np.float32),
        "Y_blurred_best": (y_best * input_std + input_mean)[:, plot_gene_indices].astype(np.float32),
        "X_final": x_final[:, plot_gene_indices].astype(np.float32),
        "variogram_real": (gamma_x * input_variance)[:, plot_gene_indices].astype(np.float32),
        "variogram_permuted": (gamma_pi * input_variance)[:, plot_gene_indices].astype(np.float32),
        "variogram_final": _classical_variogram(x_final, geom)[:, plot_gene_indices].astype(np.float32),
        "fitted_variograms_candidates": fitted_curves.astype(np.float32),
        "realized_variograms_candidates": realized_plot.astype(np.float32),
        "best_candidate_per_plot_gene": best_label_slot0.astype(np.float32),
        "intercept_best_plot": (b_best * input_variance)[plot_gene_indices].astype(np.float32),
        "slope_best_plot": a_best[plot_gene_indices].astype(np.float32),
        "coefficient_sse_best_plot": np.array(
            [sse[best_k[g], g] * input_variance[g] ** 2 for g in plot_gene_indices],
            dtype=np.float32,
        ),
        "sse_best_plot": np.array(
            [realized_sse[int(best_k[g]), g] for g in plot_gene_indices],
            dtype=np.float32,
        ),
        "realized_sse_candidates": np.asarray(realized_sse, dtype=np.float64)[:, plot_gene_indices].astype(np.float32),
        **extra_diagnostics,
    }
    if blur_mode == "uniform":
        diagnostics["sigma_multiples"] = labels.astype(np.float32)
        diagnostics["best_multiplier_per_plot_gene"] = best_label_slot0.astype(np.float32)
    else:
        diagnostics["nn_candidates"] = labels.astype(np.float32)
        diagnostics["best_nn_per_plot_gene"] = best_label_slot0.astype(np.float32)
    return diagnostics


def _run_variogram_matching(
    X: np.ndarray,
    coords: np.ndarray,
    *,
    n_perms: int,
    seed: int,
    n_bins: int,
    n_plot_genes: int,
    var_names: list[str] | None,
    candidate_builder: Callable[[np.ndarray], np.ndarray],
    candidate_labels: np.ndarray,
    blur_mode: str,
    identity_index: int | None,
    extra_diagnostics: dict[str, Any],
    verbose: bool = True,
    progress_label: str | None = None,
    collect_trace: bool = False,
    collect_timings: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    t_total = time.perf_counter()
    timing_smooth = 0.0
    timing_variogram = 0.0
    timing_fit = 0.0
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite cell-by-gene matrix")
    n_cells, n_genes = X.shape
    if coords.shape != (n_cells, 2) or not np.all(np.isfinite(coords)):
        raise ValueError("coords must be finite and aligned with X")
    # Match in standardized units so the nugget noise has unit marginal
    # variance. Restore the input moments before training or plotting.
    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0, ddof=0)
    input_variance = input_std**2
    safe_std = np.where(input_std > 0, input_std, 1.0)
    X = (X - input_mean) / safe_std
    if n_perms <= 0:
        raise ValueError("n_perms must be > 0")
    labels = np.asarray(candidate_labels, dtype=np.float64)
    n_candidates = int(labels.size)
    if n_candidates <= 0:
        raise ValueError("At least one blur candidate is required.")

    geom = _build_variogram_geometry(coords, n_bins)
    t_var = time.perf_counter()
    gamma_x = _classical_variogram(X, geom)
    timing_variogram += time.perf_counter() - t_var
    rng = np.random.default_rng(seed)
    n_plot = min(n_plot_genes, n_genes)
    plot_rng = np.random.default_rng(seed)
    plot_gene_indices = plot_rng.choice(n_genes, size=n_plot, replace=False)
    plot_gene_indices.sort()
    if var_names is not None:
        plot_gene_names = [str(var_names[i]) for i in plot_gene_indices]
    else:
        plot_gene_names = [f"gene_{i}" for i in plot_gene_indices]

    A_surr = np.empty((n_perms, n_cells, n_genes), dtype=np.float32)
    selection_counts = np.zeros((n_genes, n_candidates), dtype=np.int64)
    diagnostics: dict[str, Any] | None = None
    trace_perms = trace_noise = None
    trace_gamma = trace_sse = trace_coefficient_sse = None
    trace_slope = trace_intercept = trace_best = None
    if collect_trace:
        trace_perms = np.empty((n_perms, n_cells), dtype=np.int64)
        trace_noise = np.empty((n_perms, n_cells, n_cells), dtype=np.float64)
        trace_gamma = np.empty((n_perms, n_candidates, n_bins, n_genes), dtype=np.float64)
        trace_sse = np.empty((n_perms, n_candidates, n_genes), dtype=np.float64)
        trace_coefficient_sse = np.empty_like(trace_sse)
        trace_slope = np.empty_like(trace_sse)
        trace_intercept = np.empty_like(trace_sse)
        trace_best = np.empty((n_perms, n_genes), dtype=np.int64)
    gamma_target = gamma_x * input_variance
    desc = "variogram matching"
    if progress_label:
        desc = f"{desc}: {progress_label}"

    for p in tqdm(range(n_perms), desc=desc, unit="perm", disable=not verbose):
        # Permutation, then one noise matrix. The same matrix is reused for
        # every bandwidth so the draw sequence matches the previous generator
        # order and bandwidths are not scored on different nuggets.
        perm = rng.permutation(n_cells)
        w_noise = rng.standard_normal((n_cells, n_cells))
        r_noise = (w_noise @ X) / np.sqrt(float(n_cells))
        x_pi = X[perm]
        t_var = time.perf_counter()
        gamma_pi = _classical_variogram(x_pi, geom)
        timing_variogram += time.perf_counter() - t_var

        t_smooth = time.perf_counter()
        y_blurred = candidate_builder(x_pi).astype(np.float64)
        timing_smooth += time.perf_counter() - t_smooth
        t_var = time.perf_counter()
        gamma_y = np.stack(
            [_classical_variogram(y_blurred[k], geom) for k in range(n_candidates)],
            axis=0,
        )
        timing_variogram += time.perf_counter() - t_var
        t_fit = time.perf_counter()
        sse, intercept, slope = _fit_variogram_sse(gamma_x, gamma_y, geom.valid_bins)
        if identity_index is not None:
            # The identity branch is the raw row permutation, not an affine
            # mixture. Coefficients stay a=1, b=0; selection scores the
            # moment-restored permutation itself.
            slope[identity_index] = 1.0
            intercept[identity_index] = 0.0
            sse[identity_index] = np.sum(
                (gamma_x[geom.valid_bins] - gamma_y[identity_index, geom.valid_bins])**2,
                axis=0,
            )
        timing_fit += time.perf_counter() - t_fit

        best_sse = np.full(n_genes, np.inf, dtype=np.float64)
        best_k = np.zeros(n_genes, dtype=np.int64)
        best_final = np.empty((n_cells, n_genes), dtype=np.float64)
        y_best = np.empty((n_cells, n_genes), dtype=np.float64)
        a_best = np.empty(n_genes, dtype=np.float64)
        b_best = np.empty(n_genes, dtype=np.float64)
        realized_sse = np.empty((n_candidates, n_genes), dtype=np.float64)
        realized_gamma = np.empty((n_candidates, geom.lag_centers.size, n_genes), dtype=np.float64)
        for k in range(n_candidates):
            t_var = time.perf_counter()
            restored, gamma_k, sse_k = _score_restored_candidate(
                y_blurred[k],
                slope[k],
                intercept[k],
                r_noise,
                identity=identity_index is not None and k == int(identity_index),
                input_mean=input_mean,
                input_std=input_std,
                gamma_target=gamma_target,
                geom=geom,
            )
            timing_variogram += time.perf_counter() - t_var
            realized_sse[k] = sse_k
            realized_gamma[k] = gamma_k
            improved = _strictly_better(sse_k, best_sse)
            best_sse[improved] = sse_k[improved]
            best_k[improved] = k
            best_final[:, improved] = restored[:, improved]
            y_best[:, improved] = y_blurred[k][:, improved]
            a_best[improved] = slope[k, improved]
            b_best[improved] = intercept[k, improved]
        selection_counts[np.arange(n_genes), best_k] += 1
        A_surr[p] = best_final
        if collect_trace:
            assert trace_perms is not None and trace_noise is not None
            assert trace_gamma is not None and trace_sse is not None
            assert trace_coefficient_sse is not None
            assert trace_slope is not None and trace_intercept is not None and trace_best is not None
            trace_perms[p] = perm
            trace_noise[p] = w_noise
            trace_gamma[p] = gamma_y
            trace_sse[p] = realized_sse
            trace_coefficient_sse[p] = sse
            trace_slope[p] = slope
            trace_intercept[p] = intercept
            trace_best[p] = best_k

        if p == 0:
            diagnostics = _slot0_variogram_diagnostics(
                blur_mode=blur_mode,
                backend="sequential",
                input_variance=input_variance,
                input_std=input_std,
                input_mean=input_mean,
                X=X,
                coords=coords,
                geom=geom,
                gamma_x=gamma_x,
                gamma_pi=gamma_pi,
                gamma_y=gamma_y,
                slope=slope,
                intercept=intercept,
                sse=sse,
                realized_sse=realized_sse,
                realized_gamma=realized_gamma,
                best_k=best_k,
                y_best=y_best,
                a_best=a_best,
                b_best=b_best,
                x_pi=x_pi,
                x_final=best_final,
                labels=labels,
                plot_gene_indices=plot_gene_indices,
                plot_gene_names=plot_gene_names,
                extra_diagnostics=extra_diagnostics,
            )

    assert diagnostics is not None
    diagnostics["selection_counts"] = selection_counts
    diagnostics["selection_counts_columns"] = labels.astype(np.float64)
    if collect_timings:
        diagnostics["timing_smooth_sec"] = float(timing_smooth)
        diagnostics["timing_variogram_sec"] = float(timing_variogram)
        diagnostics["timing_fit_sec"] = float(timing_fit)
        diagnostics["timing_total_sec"] = float(time.perf_counter() - t_total)
    if collect_trace:
        diagnostics["trace_permutations"] = trace_perms
        diagnostics["trace_noise"] = trace_noise
        diagnostics["trace_gamma_y"] = trace_gamma
        diagnostics["trace_sse"] = trace_sse
        diagnostics["trace_coefficient_sse"] = trace_coefficient_sse
        diagnostics["trace_slope"] = trace_slope
        diagnostics["trace_intercept"] = trace_intercept
        diagnostics["trace_best_k"] = trace_best
    return A_surr, diagnostics


def build_correlated_null_surrogates(
    X: np.ndarray,
    coords: np.ndarray,
    *,
    n_perms: int,
    seed: int,
    sigma_multiples: list[float],
    n_bins: int = 13,
    n_plot_genes: int = 10,
    var_names: list[str] | None = None,
    verbose: bool = True,
    progress_label: str | None = None,
    backend: str = "sequential",
    perm_batch_size: int | None = None,
    weight_cache_bytes: int | None = None,
    collect_trace: bool = False,
    collect_timings: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Uniform-bandwidth variogram nulls (σ = m × median NN distance)."""
    multiples = [float(m) for m in sigma_multiples]
    if not multiples or any(m <= 0 for m in multiples):
        raise ValueError("sigma_multiples must be a non-empty list of positive values.")
    resolved_backend = _normalize_variogram_backend(backend)
    if resolved_backend == "cuda" and not _cuda_variogram_available():
        raise RuntimeError(
            "variogram_backend is 'cuda' but CUDA is not available. "
            "Set test.variogram_backend to 'sequential' or 'cpu'."
        )
    base_d = _median_nearest_neighbor_distance(coords)
    blur_weights = [
        gaussian_smoothing_weights(coords, sigma=m * base_d) for m in multiples
    ]
    labels = np.asarray(multiples, dtype=np.float64)
    if resolved_backend != "sequential":
        from methods.variogram_matching_accel import run_batched_variogram_matching

        return run_batched_variogram_matching(
            X,
            coords,
            n_perms=n_perms,
            seed=seed,
            n_bins=n_bins,
            n_plot_genes=n_plot_genes,
            var_names=var_names,
            candidate_labels=labels,
            blur_mode="uniform",
            identity_index=None,
            extra_diagnostics={"median_nn_distance": base_d},
            verbose=verbose,
            progress_label=progress_label,
            backend=resolved_backend,
            perm_batch_size=perm_batch_size,
            weight_cache_bytes=weight_cache_bytes,
            collect_trace=collect_trace,
            blur_weights=blur_weights,
            adaptive_plan=None,
        )

    def _builder(x_pi: np.ndarray) -> np.ndarray:
        return np.stack([_apply_blur_weights(w, x_pi) for w in blur_weights], axis=0)

    return _run_variogram_matching(
        X,
        coords,
        n_perms=n_perms,
        seed=seed,
        n_bins=n_bins,
        n_plot_genes=n_plot_genes,
        var_names=var_names,
        candidate_builder=_builder,
        candidate_labels=labels,
        blur_mode="uniform",
        identity_index=None,
        extra_diagnostics={"median_nn_distance": base_d},
        verbose=verbose,
        progress_label=progress_label,
        collect_trace=collect_trace,
        collect_timings=collect_timings,
    )


def build_correlated_null_adaptive_surrogates(
    X: np.ndarray,
    coords: np.ndarray,
    *,
    n_perms: int,
    seed: int,
    nn_candidates: list[float],
    n_bins: int = 13,
    n_plot_genes: int = 10,
    var_names: list[str] | None = None,
    verbose: bool = True,
    progress_label: str | None = None,
    backend: str = "sequential",
    perm_batch_size: int | None = None,
    weight_cache_bytes: int | None = None,
    collect_trace: bool = False,
    collect_timings: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """STCompare-style adaptive locfit Gaussian blur (nn fraction candidates, nn=0 identity)."""
    resolved_backend = _normalize_variogram_backend(backend)
    if resolved_backend == "cuda" and not _cuda_variogram_available():
        raise RuntimeError(
            "variogram_backend is 'cuda' but CUDA is not available. "
            "Set test.variogram_backend to 'sequential' or 'cpu'."
        )
    plan = _precompute_adaptive_blur_plan(coords, nn_candidates)
    coords64 = np.asarray(coords, dtype=np.float64)
    if resolved_backend != "sequential":
        from methods.variogram_matching_accel import run_batched_variogram_matching

        return run_batched_variogram_matching(
            X,
            coords64,
            n_perms=n_perms,
            seed=seed,
            n_bins=n_bins,
            n_plot_genes=n_plot_genes,
            var_names=var_names,
            candidate_labels=plan.nn_candidates,
            blur_mode="adaptive",
            identity_index=plan.identity_index,
            extra_diagnostics={"locfit_gaussian_factor": LOCFIT_GAUSSIAN_FACTOR},
            verbose=verbose,
            progress_label=progress_label,
            backend=resolved_backend,
            perm_batch_size=perm_batch_size,
            weight_cache_bytes=weight_cache_bytes,
            collect_trace=collect_trace,
            blur_weights=None,
            adaptive_plan=plan,
        )

    def _builder(x_pi: np.ndarray) -> np.ndarray:
        return _adaptive_blur_expression(x_pi, coords64, plan)

    return _run_variogram_matching(
        X,
        coords,
        n_perms=n_perms,
        seed=seed,
        n_bins=n_bins,
        n_plot_genes=n_plot_genes,
        var_names=var_names,
        candidate_builder=_builder,
        candidate_labels=plan.nn_candidates,
        blur_mode="adaptive",
        identity_index=plan.identity_index,
        extra_diagnostics={"locfit_gaussian_factor": LOCFIT_GAUSSIAN_FACTOR},
        verbose=verbose,
        progress_label=progress_label,
        collect_trace=collect_trace,
        collect_timings=collect_timings,
    )


def build_correlated_null_surrogates_from_dataset(
    dataset: DatasetBundle,
    config: TestConfig,
    *,
    A_override: np.ndarray | None = None,
    coords_override: np.ndarray | None = None,
    seed_offset: int = 0,
    progress_label: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    X = np.asarray(A_override if A_override is not None else dataset.A, dtype=np.float64)
    if coords_override is not None:
        coords = np.asarray(coords_override, dtype=np.float64)
    else:
        coords = raw_coordinates_from_standardized(dataset.S, dataset.meta)
    var_names = dataset.meta.get("var_names")
    names_list = list(var_names) if var_names is not None else None
    return build_correlated_null_surrogates(
        X,
        coords,
        n_perms=int(config.n_perms),
        seed=int(config.seed) + int(seed_offset),
        sigma_multiples=list(config.variogram_sigma_multiples),
        n_bins=int(config.variogram_n_bins),
        n_plot_genes=int(config.variogram_n_plot_genes),
        var_names=names_list,
        verbose=bool(config.verbose),
        progress_label=progress_label,
        backend=str(config.variogram_backend),
        perm_batch_size=config.variogram_perm_batch_size,
        collect_timings=True,
    )


def build_correlated_null_adaptive_surrogates_from_dataset(
    dataset: DatasetBundle,
    config: TestConfig,
    *,
    A_override: np.ndarray | None = None,
    coords_override: np.ndarray | None = None,
    seed_offset: int = 0,
    progress_label: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    X = np.asarray(A_override if A_override is not None else dataset.A, dtype=np.float64)
    if coords_override is not None:
        coords = np.asarray(coords_override, dtype=np.float64)
    else:
        coords = raw_coordinates_from_standardized(dataset.S, dataset.meta)
    var_names = dataset.meta.get("var_names")
    names_list = list(var_names) if var_names is not None else None
    return build_correlated_null_adaptive_surrogates(
        X,
        coords,
        n_perms=int(config.n_perms),
        seed=int(config.seed) + int(seed_offset),
        nn_candidates=list(config.variogram_nn_candidates),
        n_bins=int(config.variogram_n_bins),
        n_plot_genes=int(config.variogram_n_plot_genes),
        var_names=names_list,
        verbose=bool(config.verbose),
        progress_label=progress_label,
        backend=str(config.variogram_backend),
        perm_batch_size=config.variogram_perm_batch_size,
        collect_timings=True,
    )
