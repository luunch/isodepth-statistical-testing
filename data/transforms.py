from __future__ import annotations

import numpy as np


def filter_genes_by_min_cells(a: np.ndarray, min_cells_per_gene: int = 0) -> np.ndarray:
    if min_cells_per_gene <= 0:
        return np.asarray(a, dtype=np.float32)

    nonzero_per_gene = (a != 0).sum(axis=0)
    keep = nonzero_per_gene >= min_cells_per_gene
    if keep.sum() == 0:
        raise ValueError(
            "Filtering removed all genes. Lower min_cells_per_gene or choose another matrix/layer."
        )
    return np.asarray(a[:, keep], dtype=np.float32)


def standardize_expression(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0, keepdims=True)
    sigma = a.std(axis=0, keepdims=True)
    return np.asarray((a - mu) / (sigma + 1e-8), dtype=np.float32)


def apply_expression_transforms(
    a: np.ndarray,
    *,
    min_cells_per_gene: int = 0,
    standardize: bool = True,
) -> np.ndarray:
    transformed = filter_genes_by_min_cells(a, min_cells_per_gene=min_cells_per_gene)
    if standardize:
        transformed = standardize_expression(transformed)
    return np.asarray(transformed, dtype=np.float32)
