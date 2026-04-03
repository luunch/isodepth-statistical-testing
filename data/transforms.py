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


def standardize_expression(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0, keepdims=True)
    sigma = a.std(axis=0, keepdims=True)
    return np.asarray((a - mu) / (sigma + 1e-8), dtype=np.float32)


def poisson_low_rank_factorization(
    a: np.ndarray,
    q: int,
    *,
    seed: int = 0,
    max_iter: int = 250,
    lr: float = 5e-2,
    patience: int = 25,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
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

    log_counts = np.log1p(counts)
    u, s, vt = np.linalg.svd(log_counts, full_matrices=False)

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
    l_t = torch.nn.Parameter(torch.tensor(l_init, dtype=torch.float32))
    r_t = torch.nn.Parameter(torch.tensor(r_init, dtype=torch.float32))
    optimizer = torch.optim.Adam((l_t, r_t), lr=lr)

    best_loss = float("inf")
    best_l = l_init.copy()
    best_r = r_init.copy()
    patience_counter = 0

    for _ in range(max_iter):
        optimizer.zero_grad()
        eta = l_t @ r_t.T
        eta_clipped = torch.clamp(eta, min=-15.0, max=15.0)
        loss = (torch.exp(eta_clipped) - counts_t * eta_clipped).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((l_t, r_t), max_norm=5.0)
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
    standardize: bool = True,
    q: int | None = None,
    seed: int = 0,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    transformed, keep_mask = _filter_genes_by_min_cells(a, min_cells_per_gene=min_cells_per_gene)
    metadata: dict[str, Any] = {
        "gene_keep_mask": keep_mask,
        "representation": "gene_expression",
    }

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

    if standardize:
        transformed = standardize_expression(transformed)

    transformed = np.asarray(transformed, dtype=np.float32)
    if return_metadata:
        metadata["standardize"] = bool(standardize)
        return transformed, metadata
    return transformed
