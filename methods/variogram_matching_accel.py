"""Batched variogram matching. The fit is the sequential one; the products are wider.

Each permutation still draws its row shuffle and its Gaussian noise matrix from
the NumPy generator, in that order. Those draws are taken before the batched
linear algebra. One noise matrix is shared across bandwidths. Legacy selection
walks candidates in order, scores the moment-restored surrogate, and keeps the
first minimum. Variance-constrained selection uses the same draw order and the
shared constrained solver, and keeps the first theoretical-SSE minimum.
The winner is normalized once; its existing diagnostic variogram is scaled
analytically. Candidate mixture variograms are diagnostic only.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm

from methods.correlated_null_permutation import (
    ADAPTIVE_BLUR_BLOCK_SIZE,
    _AdaptiveBlurPlan,
    _VariogramGeometry,
    _adaptive_gaussian_weight_block,
    _build_variogram_geometry,
    _classical_variogram,
    _empirical_variogram_sse,
    _fit_variogram_sse,
    _fit_variance_constrained_coefficients,
    _identity_constrained_coefficients,
    _finalize_constrained_field,
    _normalize_variogram_fit_mode,
    _score_restored_candidate,
    _diagnostic_realized_columns,
    _score_variance_constrained_candidate,
    _selection_frequency_counts,
    _slot0_variogram_diagnostics,
    _standardize_noise_columns,
    _strictly_better,
)

# One cached candidate is an N×N float64 matrix. Above this total, each
# candidate is built in cell blocks, applied to the whole permutation batch,
# and discarded.
DEFAULT_VARIAGRAM_WEIGHT_CACHE_BYTES = 512 * 1024 * 1024
# One cells × (batch × genes) float64 matrix stays near this size when the
# caller does not set a batch size.
_EXPRESSION_BATCH_BYTES = 256 * 1024 * 1024


def _stack_permuted_columns(expression: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    """Cells × (B × genes). Column ``b * n_genes + g`` is permutation ``b``, gene ``g``."""
    gathered = expression[permutations]
    return np.transpose(gathered, (1, 0, 2)).reshape(expression.shape[0], -1)


def _split_columns(values: np.ndarray, n_perms: int, n_genes: int) -> np.ndarray:
    """(n_bins, B × genes) -> (B, n_bins, genes), or (B × genes,) -> (B, genes)."""
    if values.ndim == 1:
        return values.reshape(n_perms, n_genes)
    n_bins = values.shape[0]
    return np.transpose(values.reshape(n_bins, n_perms, n_genes), (1, 0, 2))


def _fit_candidate_columns(
    gamma_x: np.ndarray,
    gamma_columns: np.ndarray,
    valid_bins: np.ndarray,
    *,
    identity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit every permutation–gene column. ``gamma_x`` is shared across permutations."""
    n_genes = int(gamma_x.shape[1])
    n_cols = int(gamma_columns.shape[1])
    if n_genes == 0 or n_cols % n_genes != 0:
        raise ValueError(f"Column count {n_cols} is not a multiple of n_genes={n_genes}")
    gamma_x_rep = np.tile(gamma_x, (1, n_cols // n_genes))
    if identity:
        difference = gamma_x_rep[valid_bins] - gamma_columns[valid_bins]
        sse = np.sum(difference**2, axis=0)
        slope = np.ones(n_cols, dtype=np.float64)
        intercept = np.zeros(n_cols, dtype=np.float64)
        return sse, intercept, slope
    sse, intercept, slope = _fit_variogram_sse(
        gamma_x_rep,
        gamma_columns[None, :, :],
        valid_bins,
    )
    return sse[0], intercept[0], slope[0]


def _resolve_perm_batch_size(
    n_cells: int,
    n_genes: int,
    n_perms: int,
    requested: int | None,
    device: str,
) -> int:
    if requested is not None:
        batch = int(requested)
        if batch <= 0:
            raise ValueError("perm_batch_size must be > 0")
        return min(batch, n_perms)
    if device == "cpu":
        # SciPy's sparse variogram products get slower as the column count
        # grows. On the Fan2019 B1 matrix, a CPU batch of 8 or 32 was slower
        # than one permutation at a time. Weight reuse is the CPU speedup.
        return 1
    bytes_per_perm = int(n_cells) * int(n_genes) * 8
    batch = max(1, _EXPRESSION_BATCH_BYTES // max(bytes_per_perm, 1))
    return min(batch, n_perms)


def _fill_adaptive_weights(
    coords: np.ndarray,
    kth_sq: np.ndarray,
    block_size: int,
) -> np.ndarray:
    n_cells = int(coords.shape[0])
    weights = np.empty((n_cells, n_cells), dtype=np.float64)
    for start in range(0, n_cells, block_size):
        stop = min(start + block_size, n_cells)
        weights[start:stop] = _adaptive_gaussian_weight_block(coords, kth_sq, start, stop)
    return weights


def _apply_dense_row_blocks(
    weights: np.ndarray,
    expression: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Apply a cached dense smoother in the same row blocks as the reference blur."""
    out = np.empty((weights.shape[0], expression.shape[1]), dtype=np.float64)
    for start in range(0, weights.shape[0], block_size):
        stop = min(start + block_size, weights.shape[0])
        out[start:stop] = weights[start:stop] @ expression
    return out


def _apply_adaptive_blocks(
    coords: np.ndarray,
    kth_sq: np.ndarray,
    expression: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Build each weight block, apply it to every column, then drop it."""
    n_cells = int(expression.shape[0])
    out = np.empty_like(expression)
    for start in range(0, n_cells, block_size):
        stop = min(start + block_size, n_cells)
        block = _adaptive_gaussian_weight_block(coords, kth_sq, start, stop)
        out[start:stop] = block @ expression
    return out


class _AdaptiveBatchSmoother:
    def __init__(
        self,
        coords: np.ndarray,
        plan: _AdaptiveBlurPlan,
        *,
        block_size: int,
        cache_bytes: int,
        device: str,
    ) -> None:
        self.coords = np.asarray(coords, dtype=np.float64)
        self.plan = plan
        self.block_size = int(block_size)
        self.device = device
        self.identity_index = int(plan.identity_index)
        n_cells = int(self.coords.shape[0])
        n_positive = int(plan.positive_candidate_indices.size)
        nbytes = n_positive * n_cells * n_cells * 8
        self.weight_cache = "cached" if nbytes <= int(cache_bytes) and n_positive > 0 else "blocked"
        if n_positive == 0:
            self.weight_cache = "identity"
        self._kth_by_candidate: dict[int, np.ndarray] = {
            int(cand): kth
            for cand, kth in zip(plan.positive_candidate_indices, plan.kth_distance_sq)
        }
        self._weights: dict[int, np.ndarray] = {}
        self._gpu_weights: dict[int, Any] = {}
        if self.weight_cache == "cached":
            for cand, kth_sq in self._kth_by_candidate.items():
                self._weights[cand] = _fill_adaptive_weights(self.coords, kth_sq, self.block_size)
            if device == "cuda":
                import torch

                for cand, matrix in self._weights.items():
                    self._gpu_weights[cand] = torch.as_tensor(
                        matrix, dtype=torch.float64, device="cuda"
                    )

    def apply_numpy(self, expression: np.ndarray, candidate: int) -> np.ndarray:
        if candidate == self.identity_index:
            return expression
        if self.weight_cache == "cached":
            return _apply_dense_row_blocks(self._weights[candidate], expression, self.block_size)
        return _apply_adaptive_blocks(
            self.coords, self._kth_by_candidate[candidate], expression, self.block_size
        )

    def apply_torch(self, expression: Any, candidate: int) -> Any:
        import torch

        if candidate == self.identity_index:
            return expression
        if self.weight_cache == "cached":
            return self._gpu_weights[candidate] @ expression
        n_cells = int(expression.shape[0])
        out = torch.empty_like(expression)
        kth_sq = self._kth_by_candidate[candidate]
        for start in range(0, n_cells, self.block_size):
            stop = min(start + self.block_size, n_cells)
            block = _adaptive_gaussian_weight_block(self.coords, kth_sq, start, stop)
            weights = torch.as_tensor(block, dtype=torch.float64, device=expression.device)
            out[start:stop] = weights @ expression
        return out


def _as_torch_operator(weights: np.ndarray | csr_matrix) -> Any:
    import torch

    if issparse(weights):
        matrix = weights.tocsr()
        crow = torch.tensor(np.asarray(matrix.indptr), dtype=torch.int64, device="cuda")
        cols = torch.tensor(np.asarray(matrix.indices), dtype=torch.int64, device="cuda")
        values = torch.tensor(np.asarray(matrix.data), dtype=torch.float64, device="cuda")
        sparse = torch.sparse_csr_tensor(
            crow,
            cols,
            values,
            size=matrix.shape,
            dtype=torch.float64,
            device="cuda",
        )

        def _multiply(expression: Any) -> Any:
            return torch.sparse.mm(sparse, expression)

        return _multiply
    dense = torch.as_tensor(np.asarray(weights, dtype=np.float64), device="cuda")

    def _multiply_dense(expression: Any) -> Any:
        return dense @ expression

    return _multiply_dense


class _UniformBatchSmoother:
    """Precomputed Gaussian weights. Outputs are rounded to float32, as in the reference."""

    def __init__(self, blur_weights: list[np.ndarray | csr_matrix], device: str) -> None:
        self.weights = list(blur_weights)
        self.device = device
        self.weight_cache = "cached"
        self._gpu = [_as_torch_operator(weights) for weights in self.weights] if device == "cuda" else []

    def apply_numpy(self, expression: np.ndarray, candidate: int) -> np.ndarray:
        smoothed = np.asarray(self.weights[candidate] @ expression, dtype=np.float64)
        return smoothed.astype(np.float32).astype(np.float64)

    def apply_torch(self, expression: Any, candidate: int) -> Any:
        # The reference blur casts each smoothed field to float32 before the variogram.
        return self._gpu[candidate](expression).float().double()


class _TorchVariogram:
    def __init__(self, geom: _VariogramGeometry) -> None:
        import warnings

        import torch

        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state",
            category=UserWarning,
        )
        self._torch = torch
        self.pair_counts = geom.pair_counts
        self._bins: list[Any | None] = []
        self._degrees: list[Any | None] = []
        for valid, matrix, degree in zip(geom.valid_bins, geom.bin_matrices, geom.degrees):
            if not valid:
                self._bins.append(None)
                self._degrees.append(None)
                continue
            csr = matrix.tocsr()
            crow = torch.tensor(np.asarray(csr.indptr), dtype=torch.int64, device="cuda")
            cols = torch.tensor(np.asarray(csr.indices), dtype=torch.int64, device="cuda")
            values = torch.tensor(np.asarray(csr.data, dtype=np.float64), dtype=torch.float64, device="cuda")
            self._bins.append(
                torch.sparse_csr_tensor(
                    crow,
                    cols,
                    values,
                    size=csr.shape,
                    dtype=torch.float64,
                    device="cuda",
                )
            )
            self._degrees.append(
                torch.tensor(np.asarray(degree, dtype=np.float64).ravel(), dtype=torch.float64, device="cuda")
            )

    def evaluate(self, expression: Any) -> np.ndarray:
        torch = self._torch
        n_bins = len(self._bins)
        n_cols = int(expression.shape[1])
        out = torch.full((n_bins, n_cols), float("nan"), dtype=expression.dtype, device=expression.device)
        squares = expression * expression
        for k, matrix in enumerate(self._bins):
            if matrix is None or self._degrees[k] is None:
                continue
            cross = expression * torch.sparse.mm(matrix, expression)
            numer = self._degrees[k] @ squares - cross.sum(dim=0)
            out[k] = numer / (2.0 * float(self.pair_counts[k]))
        return out.detach().cpu().numpy()


def _sync_cuda(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def _score_restored_candidate_torch(
    smoothed: Any,
    slope: np.ndarray,
    intercept: np.ndarray,
    noise_columns: Any,
    input_mean: Any,
    input_std: Any,
    gamma_target: np.ndarray,
    valid_bins: np.ndarray,
    torch_variogram: _TorchVariogram,
    *,
    identity: bool,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """GPU mix and moment restoration. The variogram returns to NumPy for the SSE."""
    import torch

    if identity:
        mixed = smoothed
    else:
        slope_t = torch.as_tensor(slope, dtype=torch.float64, device=smoothed.device)
        intercept_t = torch.as_tensor(intercept, dtype=torch.float64, device=smoothed.device)
        mixed = smoothed * torch.sqrt(slope_t) + noise_columns * torch.sqrt(intercept_t)
    center = mixed.mean(dim=0)
    mixed_std = torch.sqrt(((mixed - center) ** 2).mean(dim=0))
    bad = (input_std > 0) & (mixed_std == 0)
    if bool(bad.detach().any()):
        raise ValueError("Variogram matching generated a constant nonconstant gene")
    denom = torch.where(mixed_std > 0, mixed_std, torch.ones_like(mixed_std))
    restored = (mixed - center) / denom * input_std + input_mean
    gamma = torch_variogram.evaluate(restored)
    sse = _empirical_variogram_sse(gamma, gamma_target, valid_bins)
    return restored, gamma, sse


def _score_variance_constrained_candidate_torch(
    smooth_raw: Any,
    noise: Any,
    gamma_x: np.ndarray,
    valid_bins: np.ndarray,
    torch_variogram: _TorchVariogram,
    *,
    identity: bool,
    input_mean: Any,
    input_std: Any,
    gamma_target: np.ndarray,
    realized_columns: np.ndarray | None,
) -> dict[str, Any]:
    """Keep expression-sized constrained operations on the tensor device.

    Only bin-by-gene variograms and gene-wise moments cross to NumPy for the
    common coefficient solver. No cells-by-genes CPU copy per bandwidth.
    """
    import torch

    centered = smooth_raw - smooth_raw.mean(dim=0, keepdim=True)
    variance = (centered * centered).mean(dim=0).detach().cpu().numpy()
    t_var = time.perf_counter()
    gamma = torch_variogram.evaluate(smooth_raw if identity else centered)
    variogram_seconds = time.perf_counter() - t_var
    if identity:
        nugget, multiplier, sse = _identity_constrained_coefficients(gamma_x, gamma, valid_bins)
        z = smooth_raw.clone()
    else:
        nugget, multiplier, sse = _fit_variance_constrained_coefficients(gamma_x, gamma, variance, valid_bins)
        z = (centered * torch.as_tensor(np.sqrt(multiplier), device=smooth_raw.device)
             + noise * torch.as_tensor(np.sqrt(nugget), device=smooth_raw.device))
    constant = (input_std <= 0.0).detach().cpu().numpy()
    constant_t = torch.as_tensor(constant, device=smooth_raw.device)
    z[:, constant_t] = 0.0
    sse[constant] = 0.0
    variance[constant] = 0.0
    if not identity:
        nugget[constant] = 0.0
        multiplier[constant] = 0.0
    # Score the normalized noisy realization that will be returned. The
    # theoretical constrained curve is retained as a coefficient diagnostic,
    # but finite-sample noise and final moment normalization can change the
    # realized variogram.
    identity_columns = np.full(int(z.shape[1]), bool(identity), dtype=bool)
    selection_field, _selection_scale = _finalize_constrained_field_torch(
        z, input_mean, input_std, identity_columns
    )
    covariance = (centered * noise).mean(dim=0).detach().cpu().numpy()
    covariance[constant] = 0.0
    realized_gamma = np.full_like(gamma, np.nan)
    t_var = time.perf_counter()
    selection_gamma = torch_variogram.evaluate(selection_field)
    variogram_seconds += time.perf_counter() - t_var
    selection_sse = _empirical_variogram_sse(selection_gamma, gamma_target, valid_bins)
    if realized_columns is not None and len(realized_columns):
        realized_gamma[:, realized_columns] = selection_gamma[:, realized_columns]
    field = z * input_std + input_mean
    return dict(timing_variogram_sec=variogram_seconds, gamma_y=gamma, noise_variance=nugget, smooth_multiplier=multiplier,
                theoretical_sse=sse, realized_sse=selection_sse, realized_gamma=realized_gamma,
                field=field, standardized=z, smooth_variance=variance,
                covariance=covariance, smooth_standardized=smooth_raw if identity else centered)


def _finalize_constrained_field_torch(
    standardized: Any, input_mean: Any, input_std: Any, identity_columns: np.ndarray,
) -> tuple[Any, np.ndarray]:
    """Tensor equivalent of winner-only normalization, with analytic curve scale."""
    import torch

    centered = standardized - standardized.mean(dim=0, keepdim=True)
    variance = (centered * centered).mean(dim=0)
    active = input_std > 0.0
    if bool((active & ((variance <= 0.0) | ~torch.isfinite(variance))).any()):
        raise ValueError("A selected variance-constrained surrogate has invalid variance.")
    identity = torch.as_tensor(identity_columns, device=standardized.device)
    rescale = active & ~identity
    safe_variance = torch.where(rescale, variance, torch.ones_like(variance))
    scale = safe_variance.reciprocal()
    z = centered * torch.sqrt(scale)
    z[:, identity] = standardized[:, identity]
    return z * input_std + input_mean, scale.detach().cpu().numpy()


def _project_noise(
    noise: np.ndarray,
    expression: np.ndarray,
    expression_t: Any,
    scale: float,
    device: str,
) -> np.ndarray:
    if device == "cpu":
        return (noise @ expression) / scale
    import torch

    weight = torch.as_tensor(noise, dtype=torch.float64, device="cuda")
    projected = (weight @ expression_t) / scale
    return projected.detach().cpu().numpy()


def run_batched_variogram_matching(
    X: np.ndarray,
    coords: np.ndarray,
    *,
    n_perms: int,
    seed: int,
    n_bins: int,
    n_plot_genes: int,
    var_names: list[str] | None,
    candidate_labels: np.ndarray,
    blur_mode: str,
    identity_index: int | None,
    extra_diagnostics: dict[str, Any],
    verbose: bool,
    progress_label: str | None,
    backend: str,
    perm_batch_size: int | None,
    weight_cache_bytes: int | None,
    collect_trace: bool,
    blur_weights: list[np.ndarray | csr_matrix] | None,
    adaptive_plan: _AdaptiveBlurPlan | None,
    fit_mode: str = "legacy",
) -> tuple[np.ndarray, dict[str, Any]]:
    if backend not in {"cpu", "cuda"}:
        raise ValueError(f"Batched variogram matching does not run backend {backend!r}")
    if backend == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "variogram backend 'cuda' was requested, but CUDA is not available. "
                "Use backend 'cpu' or 'sequential'."
            )
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
    fit_mode = _normalize_variogram_fit_mode(fit_mode)
    constrained = fit_mode == "variance_constrained"

    geom = _build_variogram_geometry(coords, n_bins)
    _sync_cuda(backend)
    t_var = time.perf_counter()
    gamma_x = _classical_variogram(X, geom)
    _sync_cuda(backend)
    timing_variogram += time.perf_counter() - t_var

    # Same generator and same first draws as the sequential loop: one row
    # permutation, then one N×N Gaussian matrix, repeated once per permutation.
    rng = np.random.default_rng(seed)
    n_plot = min(n_plot_genes, n_genes)
    plot_rng = np.random.default_rng(seed)
    plot_gene_indices = plot_rng.choice(n_genes, size=n_plot, replace=False)
    plot_gene_indices.sort()
    if var_names is not None:
        plot_gene_names = [str(var_names[i]) for i in plot_gene_indices]
    else:
        plot_gene_names = [f"gene_{i}" for i in plot_gene_indices]

    cache_bytes = (
        DEFAULT_VARIAGRAM_WEIGHT_CACHE_BYTES
        if weight_cache_bytes is None
        else int(weight_cache_bytes)
    )
    _sync_cuda(backend)
    t_smooth = time.perf_counter()
    if blur_mode == "adaptive":
        if adaptive_plan is None:
            raise ValueError("Adaptive batched matching requires a blur plan")
        smoother: _AdaptiveBatchSmoother | _UniformBatchSmoother = _AdaptiveBatchSmoother(
            coords,
            adaptive_plan,
            block_size=ADAPTIVE_BLUR_BLOCK_SIZE,
            cache_bytes=cache_bytes,
            device=backend,
        )
    elif blur_mode == "uniform":
        if not blur_weights:
            raise ValueError("Uniform batched matching requires blur weights")
        smoother = _UniformBatchSmoother(blur_weights, backend)
    else:
        raise ValueError(f"Unknown blur mode {blur_mode!r}")
    _sync_cuda(backend)
    timing_smooth += time.perf_counter() - t_smooth

    torch_variogram = _TorchVariogram(geom) if backend == "cuda" else None
    expression_t = None
    if backend == "cuda":
        import torch

        expression_t = torch.as_tensor(X, dtype=torch.float64, device="cuda")
    noise_scale = np.sqrt(float(n_cells))
    batch_size = _resolve_perm_batch_size(
        n_cells, n_genes, n_perms, perm_batch_size, backend
    )

    A_surr = np.empty((n_perms, n_cells, n_genes), dtype=np.float32)
    selection_counts = np.zeros((n_genes, n_candidates), dtype=np.int64)
    n_upper_boundary = 0
    n_pure_noise = 0
    n_identity_selections = 0
    trace_perms = trace_noise = None
    trace_gamma = trace_sse = trace_coefficient_sse = None
    trace_slope = trace_intercept = trace_best = None
    trace_realized_sse = None
    if collect_trace:
        trace_perms = np.empty((n_perms, n_cells), dtype=np.int64)
        trace_noise = np.empty((n_perms, n_cells, n_cells), dtype=np.float64)
        trace_gamma = np.empty((n_perms, n_candidates, geom.lag_centers.size, n_genes), dtype=np.float64)
        trace_sse = np.empty((n_perms, n_candidates, n_genes), dtype=np.float64)
        trace_coefficient_sse = np.empty_like(trace_sse)
        trace_slope = np.empty_like(trace_sse)
        trace_intercept = np.empty_like(trace_sse)
        trace_best = np.empty((n_perms, n_genes), dtype=np.int64)
        if constrained:
            trace_realized_sse = np.empty_like(trace_sse)
    gamma_target = gamma_x * input_variance

    slot0: dict[str, Any] | None = None
    desc = "variogram matching"
    if progress_label:
        desc = f"{desc}: {progress_label}"
    progress = tqdm(total=n_perms, desc=desc, unit="perm", disable=not verbose)
    offset = 0
    while offset < n_perms:
        take = min(batch_size, n_perms - offset)
        permutations = np.empty((take, n_cells), dtype=np.int64)
        noises: list[np.ndarray] = []
        for i in range(take):
            permutations[i] = rng.permutation(n_cells)
            noises.append(rng.standard_normal((n_cells, n_cells)))
        columns = _stack_permuted_columns(X, permutations)
        correlated = np.empty((take, n_cells, n_genes), dtype=np.float64)
        for i, noise in enumerate(noises):
            projected = _project_noise(noise, X, expression_t, noise_scale, backend)
            if constrained:
                projected = _standardize_noise_columns(projected)
            correlated[i] = projected
        noise_columns_np = np.transpose(correlated, (1, 0, 2)).reshape(n_cells, take * n_genes)
        mean_columns = np.tile(input_mean, take)
        std_columns = np.tile(input_std, take)
        gamma_target_columns = np.tile(gamma_target, (1, take))
        noise_columns_t = mean_columns_t = std_columns_t = None
        if backend == "cuda":
            import torch

            noise_columns_t = torch.as_tensor(noise_columns_np, dtype=torch.float64, device="cuda")
            mean_columns_t = torch.as_tensor(mean_columns, dtype=torch.float64, device="cuda")
            std_columns_t = torch.as_tensor(std_columns, dtype=torch.float64, device="cuda")

        column_t = None
        if backend == "cuda":
            import torch

            column_t = torch.as_tensor(columns, dtype=torch.float64, device="cuda")

        _sync_cuda(backend)
        t_var = time.perf_counter()
        if offset == 0:
            if torch_variogram is not None:
                assert column_t is not None
                gamma_pi = torch_variogram.evaluate(column_t[:, :n_genes])
            else:
                gamma_pi = _classical_variogram(columns[:, :n_genes], geom)
        _sync_cuda(backend)
        timing_variogram += time.perf_counter() - t_var

        n_cols = int(columns.shape[1])
        best_sse = np.full(n_cols, np.inf, dtype=np.float64)
        best_k = np.zeros(n_cols, dtype=np.int64)
        best_slope = np.empty(n_cols, dtype=np.float64)
        best_intercept = np.empty(n_cols, dtype=np.float64)
        best_final: np.ndarray | Any = None
        best_smooth0: np.ndarray | Any = None
        best_smooth_variance = np.empty(n_cols, dtype=np.float64) if constrained else None
        best_covariance = np.empty(n_cols, dtype=np.float64) if constrained else None
        best_z = None
        gamma_y0 = slope0 = intercept0 = sse0 = None
        realized_gamma0 = realized_sse0 = None
        if offset == 0:
            n_bins = int(geom.lag_centers.size)
            gamma_y0 = np.empty((n_candidates, n_bins, n_genes), dtype=np.float64)
            slope0 = np.empty((n_candidates, n_genes), dtype=np.float64)
            intercept0 = np.empty_like(slope0)
            sse0 = np.empty_like(slope0)
            realized_gamma0 = np.empty_like(gamma_y0)
            realized_sse0 = np.empty_like(slope0)

        for candidate in range(n_candidates):
            _sync_cuda(backend)
            t_smooth = time.perf_counter()
            if torch_variogram is not None:
                smoothed = smoother.apply_torch(column_t, candidate)
            else:
                smoothed = smoother.apply_numpy(columns, candidate)
            _sync_cuda(backend)
            timing_smooth += time.perf_counter() - t_smooth

            identity = identity_index is not None and candidate == int(identity_index)
            smooth_plot = None
            smooth_variance_k = covariance_k = standardized_k = None
            if constrained:
                realized_columns = _diagnostic_realized_columns(
                    n_cols,
                    plot_gene_indices,
                    record_slot0=offset == 0,
                    collect_trace=collect_trace,
                )
                _sync_cuda(backend)
                t_fit = time.perf_counter()
                if torch_variogram is not None:
                    scored = _score_variance_constrained_candidate_torch(
                        smoothed, noise_columns_t, np.tile(gamma_x, (1, take)),
                        geom.valid_bins, torch_variogram, identity=identity,
                        input_mean=mean_columns_t, input_std=std_columns_t,
                        gamma_target=gamma_target_columns, realized_columns=realized_columns,
                    )
                else:
                    scored = _score_variance_constrained_candidate(
                        smoothed, noise_columns_np, np.tile(gamma_x, (1, take)),
                        geom.valid_bins, geom, identity=identity,
                        input_mean=mean_columns, input_std=std_columns,
                        gamma_target=gamma_target_columns, realized_columns=realized_columns,
                    )
                _sync_cuda(backend)
                timing_fit += time.perf_counter() - t_fit - scored["timing_variogram_sec"]
                timing_variogram += scored["timing_variogram_sec"]
                gamma = scored["gamma_y"]
                slope = scored["smooth_multiplier"]
                intercept = scored["noise_variance"]
                sse = scored["theoretical_sse"]
                restored = scored["field"]
                gamma_final = scored["realized_gamma"]
                sse_realized = scored["realized_sse"]
                # Select on the normalized, noisy realization returned to the
                # permutation test; keep ``sse`` as the theoretical fit
                # diagnostic only.
                selection_sse = sse_realized
                smooth_plot = scored["smooth_standardized"]
                smooth_variance_k = scored["smooth_variance"]
                covariance_k = scored["covariance"]
                standardized_k = scored["standardized"]
            else:
                _sync_cuda(backend)
                t_var = time.perf_counter()
                if torch_variogram is not None:
                    gamma = torch_variogram.evaluate(smoothed)
                else:
                    gamma = _classical_variogram(smoothed, geom)
                _sync_cuda(backend)
                timing_variogram += time.perf_counter() - t_var

                _sync_cuda(backend)
                t_fit = time.perf_counter()
                sse, intercept, slope = _fit_candidate_columns(
                    gamma_x,
                    gamma,
                    geom.valid_bins,
                    identity=identity,
                )
                _sync_cuda(backend)
                timing_fit += time.perf_counter() - t_fit

                _sync_cuda(backend)
                t_var = time.perf_counter()
                if torch_variogram is not None:
                    assert noise_columns_t is not None
                    assert mean_columns_t is not None and std_columns_t is not None
                    restored, gamma_final, sse_realized = _score_restored_candidate_torch(
                        smoothed,
                        slope,
                        intercept,
                        noise_columns_t,
                        mean_columns_t,
                        std_columns_t,
                        gamma_target_columns,
                        geom.valid_bins,
                        torch_variogram,
                        identity=identity,
                    )
                else:
                    restored, gamma_final, sse_realized = _score_restored_candidate(
                        smoothed,
                        slope,
                        intercept,
                        noise_columns_np,
                        identity=identity,
                        input_mean=mean_columns,
                        input_std=std_columns,
                        gamma_target=gamma_target_columns,
                        geom=geom,
                    )
                _sync_cuda(backend)
                timing_variogram += time.perf_counter() - t_var
                selection_sse = sse_realized

            if offset == 0:
                assert gamma_y0 is not None and slope0 is not None
                assert intercept0 is not None and sse0 is not None
                gamma_y0[candidate] = gamma[:, :n_genes]
                slope0[candidate] = slope[:n_genes]
                intercept0[candidate] = intercept[:n_genes]
                sse0[candidate] = sse[:n_genes]
            if collect_trace:
                assert trace_gamma is not None and trace_coefficient_sse is not None
                assert trace_slope is not None and trace_intercept is not None
                trace_gamma[offset : offset + take, candidate] = _split_columns(gamma, take, n_genes)
                trace_coefficient_sse[offset : offset + take, candidate] = _split_columns(sse, take, n_genes)
                trace_slope[offset : offset + take, candidate] = _split_columns(slope, take, n_genes)
                trace_intercept[offset : offset + take, candidate] = _split_columns(intercept, take, n_genes)

            if offset == 0:
                assert realized_gamma0 is not None and realized_sse0 is not None
                realized_gamma0[candidate] = np.asarray(gamma_final)[:, :n_genes]
                realized_sse0[candidate] = np.asarray(sse_realized)[:n_genes]
            if collect_trace:
                assert trace_sse is not None
                trace_sse[offset : offset + take, candidate] = _split_columns(selection_sse, take, n_genes)
                if trace_realized_sse is not None:
                    trace_realized_sse[offset : offset + take, candidate] = _split_columns(
                        sse_realized, take, n_genes
                    )

            improved = _strictly_better(selection_sse, best_sse)
            if candidate == 0:
                improved = np.ones(n_cols, dtype=bool)
            if not np.any(improved):
                continue
            best_sse[improved] = selection_sse[improved]
            best_slope[improved] = slope[improved]
            best_intercept[improved] = intercept[improved]
            best_k[improved] = candidate
            if constrained:
                assert best_smooth_variance is not None and best_covariance is not None
                assert smooth_variance_k is not None and covariance_k is not None and standardized_k is not None
                assert smooth_plot is not None
                best_smooth_variance[improved] = smooth_variance_k[improved]
                best_covariance[improved] = covariance_k[improved]
                if torch_variogram is not None:
                    mask = torch.as_tensor(improved, device=standardized_k.device)
                    if best_z is None:
                        best_z = standardized_k.clone()
                    else:
                        best_z[:, mask] = standardized_k[:, mask]
                elif best_z is None:
                    best_z = np.array(standardized_k, copy=True)
                else:
                    best_z[:, improved] = standardized_k[:, improved]
            torch_fields = torch_variogram is not None
            if torch_fields:
                import torch

                if not constrained:
                    if best_final is None:
                        best_final = restored.clone()
                    else:
                        mask = torch.as_tensor(improved, device=best_final.device)
                        best_final[:, mask] = restored[:, mask]
                if offset == 0:
                    improved0 = improved[:n_genes]
                    smooth_source = smooth_plot if smooth_plot is not None else smoothed
                    if best_smooth0 is None:
                        best_smooth0 = smooth_source[:, :n_genes].clone()
                    elif np.any(improved0):
                        mask0 = torch.as_tensor(improved0, device=best_smooth0.device)
                        best_smooth0[:, mask0] = smooth_source[:, :n_genes][:, mask0]
            else:
                if not constrained:
                    if best_final is None:
                        best_final = np.array(restored, copy=True, order="C")
                    else:
                        best_final[:, improved] = np.asarray(restored)[:, improved]
                if offset == 0:
                    improved0 = improved[:n_genes]
                    smooth_source = smooth_plot if smooth_plot is not None else smoothed
                    if best_smooth0 is None:
                        best_smooth0 = np.array(np.asarray(smooth_source)[:, :n_genes], copy=True, order="C")
                    elif np.any(improved0):
                        best_smooth0[:, improved0] = np.asarray(smooth_source)[:, :n_genes][:, improved0]

        final_variogram_scale = None
        if constrained:
            assert best_z is not None
            identity_columns = np.zeros(n_cols, dtype=bool) if identity_index is None else best_k == identity_index
            if torch_variogram is not None:
                best_final, final_variogram_scale = _finalize_constrained_field_torch(
                    best_z, mean_columns_t, std_columns_t, identity_columns
                )
            else:
                best_final, final_variogram_scale = _finalize_constrained_field(
                    best_z, mean_columns, std_columns, identity_columns
                )

        if offset == 0:
            assert gamma_y0 is not None and slope0 is not None
            assert intercept0 is not None and sse0 is not None
            assert realized_gamma0 is not None and realized_sse0 is not None
            assert best_final is not None and best_smooth0 is not None
            numpy_fields = torch_variogram is None
            if numpy_fields:
                y_slot0 = np.array(best_smooth0, copy=True)
                x_slot0 = np.array(best_final[:, :n_genes], copy=True)
            else:
                y_slot0 = best_smooth0.detach().cpu().numpy()
                x_slot0 = best_final[:, :n_genes].detach().cpu().numpy()
            slot0 = {
                "x_pi": columns[:, :n_genes].copy(),
                "gamma_pi": gamma_pi,
                "gamma_y": gamma_y0,
                "slope": slope0,
                "intercept": intercept0,
                "sse": sse0,
                "realized_sse": realized_sse0,
                "realized_gamma": realized_gamma0,
                "best_k": best_k[:n_genes].copy(),
                "a_best": best_slope[:n_genes].copy(),
                "b_best": best_intercept[:n_genes].copy(),
                "y_best": y_slot0,
                "x_final": x_slot0,
                "smooth_variance_best": None if best_smooth_variance is None else best_smooth_variance[:n_genes].copy(),
                "covariance_best": None if best_covariance is None else best_covariance[:n_genes].copy(),
                "final_variogram_scale": None if final_variogram_scale is None else final_variogram_scale[:n_genes].copy(),
            }

        assert best_final is not None
        numpy_fields = torch_variogram is None
        for local in range(take):
            cols = slice(local * n_genes, (local + 1) * n_genes)
            if numpy_fields:
                final = np.asarray(best_final[:, cols])
            else:
                final = best_final[:, cols].detach().cpu().numpy()
            chosen = best_k[cols]
            A_surr[offset + local] = final
            selection_counts[np.arange(n_genes), chosen] += 1
            if constrained:
                n_upper, n_pure, n_identity = _selection_frequency_counts(
                    chosen,
                    best_intercept[cols],
                    best_slope[cols],
                    labels,
                    identity_index,
                )
                n_upper_boundary += n_upper
                n_pure_noise += n_pure
                n_identity_selections += n_identity
            if collect_trace:
                assert trace_perms is not None and trace_noise is not None and trace_best is not None
                trace_perms[offset + local] = permutations[local]
                trace_noise[offset + local] = noises[local]
                trace_best[offset + local] = chosen
        progress.update(take)
        offset += take
    progress.close()

    assert slot0 is not None
    diagnostics = _slot0_variogram_diagnostics(
        blur_mode=blur_mode,
        backend=backend,
        fit_mode=fit_mode,
        input_variance=input_variance,
        input_std=input_std,
        input_mean=input_mean,
        X=X,
        coords=coords,
        geom=geom,
        gamma_x=gamma_x,
        gamma_pi=slot0["gamma_pi"],
        gamma_y=slot0["gamma_y"],
        slope=slot0["slope"],
        intercept=slot0["intercept"],
        sse=slot0["sse"],
        realized_sse=slot0["realized_sse"],
        realized_gamma=slot0["realized_gamma"],
        best_k=slot0["best_k"],
        y_best=slot0["y_best"],
        a_best=slot0["a_best"],
        b_best=slot0["b_best"],
        x_pi=slot0["x_pi"],
        x_final=slot0["x_final"],
        labels=labels,
        plot_gene_indices=plot_gene_indices,
        plot_gene_names=plot_gene_names,
        extra_diagnostics=extra_diagnostics,
        smooth_variance_best=slot0["smooth_variance_best"],
        smooth_noise_covariance_best=slot0["covariance_best"],
        final_variogram_scale=slot0["final_variogram_scale"],
    )
    diagnostics["selection_counts"] = selection_counts
    diagnostics["selection_counts_columns"] = labels.astype(np.float64)
    if constrained:
        n_selections = float(n_perms * n_genes)
        diagnostics["upper_boundary_label"] = float(np.max(labels))
        diagnostics["upper_boundary_frequency"] = n_upper_boundary / n_selections
        diagnostics["pure_noise_frequency"] = n_pure_noise / n_selections
        diagnostics["identity_selection_frequency"] = n_identity_selections / n_selections
    diagnostics["variogram_perm_batch_size"] = int(batch_size)
    diagnostics["variogram_weight_cache"] = smoother.weight_cache
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
        if trace_realized_sse is not None:
            diagnostics["trace_realized_sse"] = trace_realized_sse
    return A_surr, diagnostics
