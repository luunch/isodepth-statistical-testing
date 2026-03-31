from __future__ import annotations

from typing import Iterable

import numpy as np


CANONICAL_METRIC_ALIASES = {
    "nll": "nll_gaussian_mse",
    "nll_gaussian_mse": "nll_gaussian_mse",
    "mse": "mse",
    "pearson": "pearson_corr_mean",
    "pearson_corr": "pearson_corr_mean",
    "pearson_corr_mean": "pearson_corr_mean",
    "spearman": "spearman_corr_mean",
    "spearman_corr": "spearman_corr_mean",
    "spearman_corr_mean": "spearman_corr_mean",
}


def _rankdata_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def canonicalize_metric_name(metric: str) -> str:
    try:
        return CANONICAL_METRIC_ALIASES[metric]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric '{metric}'") from exc


def metric_prefers_lower(metric: str) -> bool:
    return canonicalize_metric_name(metric) in {"nll_gaussian_mse", "mse"}


def permutation_p_value(metric: str, stat_true: float, stat_perm: np.ndarray) -> float:
    stat_perm = np.asarray(stat_perm, dtype=np.float64)
    if metric_prefers_lower(metric):
        return float((1 + np.sum(stat_perm <= stat_true)) / (stat_perm.size + 1))
    return float((1 + np.sum(stat_perm >= stat_true)) / (stat_perm.size + 1))


def _mean_gene_pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    x = y_true - y_true.mean(axis=0, keepdims=True)
    y = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((x**2).sum(axis=0) * (y**2).sum(axis=0))
    corr = np.divide((x * y).sum(axis=0), denom, out=np.zeros_like(denom), where=denom > 1e-12)
    return float(np.mean(corr))


def _mean_gene_spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ranks_true = np.apply_along_axis(_rankdata_1d, 0, y_true)
    ranks_pred = np.apply_along_axis(_rankdata_1d, 0, y_pred)
    return _mean_gene_pearson_corr(ranks_true, ranks_pred)


def compute_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric = canonicalize_metric_name(metric)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must match shapes, got {y_true.shape} vs {y_pred.shape}")

    mse = float(np.mean((y_pred - y_true) ** 2))
    if metric == "mse":
        return mse
    if metric == "nll_gaussian_mse":
        n_total = y_true.shape[0] * y_true.shape[1]
        return float((n_total / 2) * np.log(2 * np.pi * mse + 1e-12) + (n_total / 2))
    if metric == "pearson_corr_mean":
        return _mean_gene_pearson_corr(y_true, y_pred)
    if metric == "spearman_corr_mean":
        return _mean_gene_spearman_corr(y_true, y_pred)
    raise ValueError(f"Unsupported metric '{metric}'")


def compute_metric_batch(metric: str, y_true: np.ndarray, y_pred_batch: np.ndarray) -> np.ndarray:
    return np.asarray([compute_metric(metric, y_true, pred) for pred in y_pred_batch], dtype=np.float64)


def summarize_metric_distribution(values: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
