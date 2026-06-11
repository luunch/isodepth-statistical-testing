from __future__ import annotations

from typing import Any

import numpy as np


def _filter_genes_by_min_cells(
    a: np.ndarray,
    min_cells_per_gene: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    a_np = np.asarray(a, dtype=np.float32)
    if min_cells_per_gene <= 0:
        keep = np.ones(a_np.shape[1], dtype=bool)
        return a_np, keep

    nonzero_per_gene = (a_np != 0).sum(axis=0)
    keep = nonzero_per_gene >= min_cells_per_gene
    if keep.sum() == 0:
        raise ValueError(
            "Filtering removed all genes. Lower min_cells_per_gene or choose another matrix/layer."
        )
    return np.asarray(a_np[:, keep], dtype=np.float32), keep


def filter_genes_by_min_cells(a: np.ndarray, min_cells_per_gene: int = 0) -> np.ndarray:
    filtered, _ = _filter_genes_by_min_cells(a, min_cells_per_gene=min_cells_per_gene)
    return filtered


def _zscore_expression(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0, keepdims=True)
    sigma = a.std(axis=0, keepdims=True)
    return np.asarray((a - mu) / (sigma + 1e-8), dtype=np.float32)


def zscore_covariate(values: np.ndarray) -> np.ndarray:
    """Per-cell z-score of a 1D covariate on the supplied cell subset."""
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    mu = float(v.mean())
    sigma = float(v.std())
    if sigma < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return np.asarray((v - mu) / sigma, dtype=np.float32)


def midline_latent(S: np.ndarray) -> np.ndarray:
    """Fixed 1-D midline latent ``z-score(|x - median(x)|)`` from spatial coordinates.

    Numpy counterpart of :class:`methods.architectures.MidlineLatentSingle`.  The
    median uses the lower of the two middle values for even counts to match
    ``torch.median`` semantics, so this reproduces the midline bottleneck used by
    the covariate model exactly.  ``S`` is ``(N, 2+)``; column 0 is treated as ``x``.
    """
    x = np.asarray(S, dtype=np.float64)[:, 0]
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    med = float(np.sort(x)[(x.shape[0] - 1) // 2])
    depth = np.abs(x - med)
    return zscore_covariate(depth)


def celltype_expression_residuals(
    A: np.ndarray,
    cell_type_labels: np.ndarray,
    n_cell_types: int | None = None,
) -> np.ndarray:
    """Subtract per-cell-type mean expression; used for ``cell_type: together`` training."""
    a_np = np.asarray(A, dtype=np.float32)
    labels = np.asarray(cell_type_labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != a_np.shape[0]:
        raise ValueError(
            f"cell_type_labels length {labels.shape[0]} != n_cells {a_np.shape[0]}"
        )
    if n_cell_types is None:
        n_cell_types = int(labels.max()) + 1 if labels.size else 0
    type_means = np.zeros((int(n_cell_types), a_np.shape[1]), dtype=np.float32)
    for cell_type in range(int(n_cell_types)):
        mask = labels == cell_type
        if mask.any():
            type_means[cell_type] = a_np[mask].mean(axis=0)
    return np.asarray(a_np - type_means[labels], dtype=np.float32)


def log1p_expression(a: np.ndarray) -> np.ndarray:
    expression = np.asarray(a, dtype=np.float32)
    if np.any(expression < 0):
        raise ValueError("log1p transform requires non-negative expression values")
    return np.asarray(np.log1p(expression), dtype=np.float32)


_CPM_TARGET = 1e6


def normalize_total_expression(a: np.ndarray) -> np.ndarray:
    """Per-cell CPM normalization (counts per million).

    Each cell is rescaled so its total counts equal ``1e6`` (``a_ig / N_i * 10^6``).
    With ``log1p`` this yields standard log-CPM, ``log(a_ig / N_i * 10^6 + 1)``. This
    removes the per-cell sequencing-depth/library-size factor *before* ``log1p`` so a
    smooth spatial depth gradient cannot masquerade as spatial signal in the Gaussian
    (log-normal) modeling path. Cells with zero total are left as zeros.
    """
    counts = np.asarray(a, dtype=np.float32)
    if np.any(counts < 0):
        raise ValueError("normalize_total requires non-negative (count-like) expression values")
    cell_totals = counts.sum(axis=1)
    safe_totals = np.where(cell_totals > 0, cell_totals, 1.0)
    scale = np.where(cell_totals > 0, _CPM_TARGET / safe_totals, 0.0).astype(np.float32)
    return np.asarray(counts * scale[:, None], dtype=np.float32)


def poisson_low_rank_factorization(
    a: np.ndarray,
    q: int,
    *,
    seed: int = 0,
    max_iter: int = 250,
    lr: float = 5e-2,
    patience: int = 25,
    tol: float = 1e-4,
    size_factor_offset: bool = True,
    gene_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson GLM low-rank embedding (GLM-PCA style).

    Fits ``log E[Y_ig] = log s_i + beta_g + (L R^T)_ig`` by minimizing the Poisson
    negative log-likelihood, then returns the cell scores ``L`` (and loadings ``R``).

    - ``size_factor_offset`` adds the fixed per-cell exposure ``log s_i`` where
      ``s_i`` is the total counts of cell ``i`` (library size). This is the same
      depth handling used by RCTD/C-SIDE: depth lives in the mean as an offset
      rather than being divided out of the data.
    - ``gene_intercept`` adds a learnable per-gene baseline ``beta_g``.

    Without these terms the leading low-rank factor simply recovers library size, so
    both default to ``True`` to keep the returned ``L`` depth-free.
    """
    if q <= 0:
        raise ValueError("q must be > 0 for Poisson low-rank factorization")

    counts = np.asarray(a, dtype=np.float32)
    if counts.ndim != 2:
        raise ValueError(f"Expression matrix must be 2D, got shape {counts.shape}")
    if np.any(counts < 0):
        raise ValueError("Poisson low-rank factorization requires non-negative expression values")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Poisson low-rank factorization requires torch to be installed"
        ) from exc

    torch.manual_seed(seed)

    n_cells, n_genes = counts.shape
    k = 2 * int(q)
    init_rank = min(k, n_cells, n_genes)

    # Fixed per-cell size-factor offset (library size) and per-gene intercept init.
    cell_totals = counts.sum(axis=1)
    if size_factor_offset:
        log_size_factor = np.log(np.clip(cell_totals, 1.0, None)).astype(np.float32)
    else:
        log_size_factor = np.zeros(n_cells, dtype=np.float32)

    if gene_intercept:
        grand_total = float(cell_totals.sum())
        gene_rate = counts.sum(axis=0) / (grand_total if grand_total > 0 else 1.0)
        beta_init = np.log(np.clip(gene_rate, 1e-6, None)).astype(np.float32)
    else:
        beta_init = np.zeros(n_genes, dtype=np.float32)

    # Initialize the low-rank part from the SVD of the residual log-expression, i.e.
    # the part of log1p(counts) not already explained by the offset and intercept.
    residual = np.log1p(counts) - log_size_factor[:, None] - beta_init[None, :]
    u, s, vt = np.linalg.svd(residual, full_matrices=False)

    l_init = np.zeros((n_cells, k), dtype=np.float32)
    r_init = np.zeros((n_genes, k), dtype=np.float32)
    if init_rank > 0:
        sqrt_s = np.sqrt(np.maximum(s[:init_rank], 1e-8)).astype(np.float32)
        l_init[:, :init_rank] = u[:, :init_rank] * sqrt_s[None, :]
        r_init[:, :init_rank] = vt[:init_rank, :].T * sqrt_s[None, :]

    if init_rank < k:
        rng = np.random.default_rng(seed)
        l_init[:, init_rank:] = 1e-2 * rng.standard_normal((n_cells, k - init_rank)).astype(np.float32)
        r_init[:, init_rank:] = 1e-2 * rng.standard_normal((n_genes, k - init_rank)).astype(np.float32)

    counts_t = torch.tensor(counts, dtype=torch.float32)
    offset_t = torch.tensor(log_size_factor, dtype=torch.float32).unsqueeze(1)
    l_t = torch.nn.Parameter(torch.tensor(l_init, dtype=torch.float32))
    r_t = torch.nn.Parameter(torch.tensor(r_init, dtype=torch.float32))
    params = [l_t, r_t]
    if gene_intercept:
        beta_t = torch.nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        params.append(beta_t)
    else:
        beta_t = torch.tensor(beta_init, dtype=torch.float32)
    optimizer = torch.optim.Adam(params, lr=lr)

    best_loss = float("inf")
    best_l = l_init.copy()
    best_r = r_init.copy()
    patience_counter = 0

    for _ in range(max_iter):
        optimizer.zero_grad()
        eta = offset_t + beta_t.unsqueeze(0) + l_t @ r_t.T
        eta_clipped = torch.clamp(eta, min=-15.0, max=15.0)
        loss = (torch.exp(eta_clipped) - counts_t * eta_clipped).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        loss_value = float(loss.item())
        if best_loss - loss_value > tol:
            best_loss = loss_value
            best_l = l_t.detach().cpu().numpy().copy()
            best_r = r_t.detach().cpu().numpy().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    scores = np.linalg.norm(best_l, axis=0) * np.linalg.norm(best_r, axis=0)
    order = np.argsort(scores)[::-1]
    return (
        np.asarray(best_l[:, order], dtype=np.float32),
        np.asarray(best_r[:, order], dtype=np.float32),
    )


def apply_expression_transforms(
    a: np.ndarray,
    *,
    min_cells_per_gene: int = 0,
    normalize_total: bool = False,
    log1p: bool = False,
    standardize_expression: bool = True,
    q: int | None = None,
    seed: int = 0,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    transformed, keep_mask = _filter_genes_by_min_cells(a, min_cells_per_gene=min_cells_per_gene)
    metadata: dict[str, Any] = {
        "gene_keep_mask": keep_mask,
        "representation": "gene_expression",
    }

    if log1p and q is not None:
        raise ValueError("log1p cannot be combined with q because the Poisson low-rank factorization expects counts")
    if normalize_total and q is not None:
        raise ValueError(
            "normalize_total cannot be combined with q because the Poisson low-rank "
            "factorization handles library size via a size-factor offset, not by rescaling counts"
        )

    if normalize_total:
        transformed = normalize_total_expression(transformed)

    if log1p:
        transformed = log1p_expression(transformed)

    if q is not None:
        latent, _ = poisson_low_rank_factorization(transformed, q=q, seed=seed)
        transformed = latent
        metadata.update(
            {
                "representation": "poisson_low_rank_latent",
                "q": int(q),
                "latent_dim": int(transformed.shape[1]),
                "feature_names": [f"poisson_latent_{idx + 1}" for idx in range(transformed.shape[1])],
            }
        )

    if standardize_expression:
        transformed = _zscore_expression(transformed)

    transformed = np.asarray(transformed, dtype=np.float32)
    if return_metadata:
        metadata["normalize_total"] = bool(normalize_total)
        metadata["log1p"] = bool(log1p)
        metadata["standardize_expression"] = bool(standardize_expression)
        return transformed, metadata
    return transformed


def apply_expression_transforms_by_celltype(
    a: np.ndarray,
    cell_type_labels: np.ndarray,
    *,
    min_cells_per_gene: int = 0,
    normalize_total: bool = False,
    log1p: bool = False,
    standardize_expression: bool = True,
    q: int | None = None,
    seed: int = 0,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Apply expression transforms with Poisson ``q`` embedding computed per cell type.

    Gene filtering uses the full matrix so all types share the same gene columns.
    When ``q`` is set, ``poisson_low_rank_factorization`` and optional z-scoring run
    independently within each cell-type subset (rows are written back in original order).
    """
    if q is None:
        raise ValueError("apply_expression_transforms_by_celltype requires q to be set")
    if normalize_total:
        raise ValueError(
            "normalize_total cannot be combined with q because the Poisson low-rank "
            "factorization handles library size via a size-factor offset, not by rescaling counts"
        )

    filtered, keep_mask = _filter_genes_by_min_cells(a, min_cells_per_gene=min_cells_per_gene)
    labels = np.asarray(cell_type_labels, dtype=np.int64)
    if labels.shape[0] != filtered.shape[0]:
        raise ValueError(
            f"cell_type_labels length {labels.shape[0]} != expression rows {filtered.shape[0]}"
        )

    if log1p:
        raise ValueError("log1p cannot be combined with q because the Poisson low-rank factorization expects counts")

    n_cell_types = int(labels.max()) + 1 if labels.size else 0
    latent_dim = 2 * int(q)
    transformed = np.empty((filtered.shape[0], latent_dim), dtype=np.float32)

    for type_index in range(n_cell_types):
        mask = labels == type_index
        if not np.any(mask):
            continue
        counts_c = np.asarray(filtered[mask], dtype=np.float32)
        latent_c, _ = poisson_low_rank_factorization(counts_c, q=q, seed=int(seed) + type_index)
        if standardize_expression:
            latent_c = _zscore_expression(latent_c)
        transformed[mask] = np.asarray(latent_c, dtype=np.float32)

    metadata: dict[str, Any] = {
        "gene_keep_mask": keep_mask,
        "representation": "poisson_low_rank_latent",
        "q": int(q),
        "latent_dim": latent_dim,
        "q_by_celltype": True,
        "feature_names": [f"poisson_latent_{idx + 1}" for idx in range(latent_dim)],
        "log1p": bool(log1p),
        "standardize_expression": bool(standardize_expression),
    }
    transformed = np.asarray(transformed, dtype=np.float32)
    if return_metadata:
        return transformed, metadata
    return transformed
