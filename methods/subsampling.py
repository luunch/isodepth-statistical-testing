from __future__ import annotations

import time

import numpy as np
import torch

from data.schemas import DatasetBundle, TestConfig, TestResult
from methods.metrics import canonicalize_metric_name, compute_metric, permutation_p_value
from methods.perturbation import score_depth_similarity
from methods.trainers import resolve_device, train_batched_isodepth_model, train_isodepth_model


def _extract_isodepth_from_model(model, S: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        s_t = torch.tensor(S, dtype=torch.float32, device=device)
        d = model.encoder(s_t).detach().cpu().numpy().reshape(-1)
    return np.asarray(d, dtype=np.float32)


def _extract_isodepth_batch(model, s_batched: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        s_batched_t = torch.tensor(s_batched, dtype=torch.float32, device=device)
        isodepth_batched = model.encoder(s_batched_t).detach().cpu().numpy().squeeze(-1)
    return np.asarray(isodepth_batched, dtype=np.float32)


def _summarize_scores(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _subset_size(n_cells: int, fraction: float) -> int:
    raw_size = int(round(float(fraction) * float(n_cells)))
    if n_cells <= 1:
        return 1
    return min(max(raw_size, 1), n_cells - 1)


def build_subset_masks(
    n_cells: int,
    subset_fractions: list[float],
    n_subsets: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total_subsets = len(subset_fractions) * n_subsets
    loss_mask_batched = np.zeros((total_subsets, n_cells, 1), dtype=np.float32)
    fraction_per_subset = np.zeros(total_subsets, dtype=np.float32)
    size_per_subset = np.zeros(total_subsets, dtype=np.int64)

    subset_index = 0
    for fraction in subset_fractions:
        subset_size = _subset_size(n_cells, fraction)
        for _ in range(n_subsets):
            selected = rng.choice(n_cells, size=subset_size, replace=False)
            loss_mask_batched[subset_index, selected, 0] = 1.0
            fraction_per_subset[subset_index] = float(fraction)
            size_per_subset[subset_index] = subset_size
            subset_index += 1

    return loss_mask_batched, fraction_per_subset, size_per_subset


def compute_masked_losses(
    predictions: np.ndarray,
    targets: np.ndarray,
    loss_mask_batched: np.ndarray,
    *,
    metric: str = "mse",
) -> np.ndarray:
    metric = canonicalize_metric_name(metric)
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    loss_mask = np.asarray(loss_mask_batched, dtype=np.float64)

    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must match shapes, got {predictions.shape} vs {targets.shape}"
        )
    if loss_mask.shape not in {
        (predictions.shape[0], predictions.shape[1], 1),
        predictions.shape,
    }:
        raise ValueError(
            "loss_mask_batched must have shape (M, N, 1) or match predictions, "
            f"got {loss_mask.shape} vs {predictions.shape}"
        )

    if loss_mask.shape[-1] == 1:
        loss_mask = np.repeat(loss_mask, predictions.shape[2], axis=2)

    active_counts = loss_mask.sum(axis=(1, 2))
    if np.any(active_counts <= 0):
        raise ValueError("Each model must have at least one active masked entry")

    squared_error = (predictions - targets) ** 2
    mse = (squared_error * loss_mask).sum(axis=(1, 2)) / active_counts
    if metric == "mse":
        return np.asarray(mse, dtype=np.float64)
    if metric == "nll_gaussian_mse":
        return np.asarray(
            (active_counts / 2.0) * np.log(2.0 * np.pi * mse + 1e-12) + (active_counts / 2.0),
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported masked-loss metric '{metric}'")


def compute_subset_correlations(
    d_true: np.ndarray,
    d_subset_batched: np.ndarray,
    loss_mask_batched: np.ndarray,
) -> np.ndarray:
    d_true = np.asarray(d_true, dtype=np.float32).reshape(-1)
    d_subset_batched = np.asarray(d_subset_batched, dtype=np.float32)
    subset_mask = np.asarray(loss_mask_batched, dtype=np.float32)[..., 0] > 0
    correlations = np.zeros(d_subset_batched.shape[0], dtype=np.float64)

    for index, selected in enumerate(subset_mask):
        if int(selected.sum()) < 2:
            correlations[index] = 0.0
            continue
        correlations[index] = score_depth_similarity(
            "spearman_corr_mean",
            d_true[selected],
            d_subset_batched[index, selected],
        )

    return correlations


def _fraction_summaries(
    values: np.ndarray,
    correlations: np.ndarray,
    null_stats: np.ndarray,
    fraction_per_subset: np.ndarray,
    size_per_subset: np.ndarray,
    metric: str,
) -> dict[str, dict[str, float | list[float]]]:
    summaries: dict[str, dict[str, float | list[float]]] = {}
    unique_fractions = np.unique(fraction_per_subset)
    for fraction in unique_fractions:
        mask = np.isclose(fraction_per_subset, fraction)
        stat_true = float(np.mean(values[mask]))
        stat_perm = np.asarray(null_stats[:, mask].mean(axis=1), dtype=np.float64)
        summaries[f"{float(fraction):.3f}"] = {
            "fraction": float(fraction),
            "subset_size": int(np.round(size_per_subset[mask].mean())),
            "n_subsets": int(mask.sum()),
            "loss_mean": stat_true,
            "loss_std": float(np.std(values[mask])),
            "correlation_mean": float(np.mean(correlations[mask])),
            "correlation_std": float(np.std(correlations[mask])),
            "p_value": float(permutation_p_value(metric, stat_true, stat_perm)),
            "null_mean": float(np.mean(stat_perm)),
            "null_std": float(np.std(stat_perm)),
            "observed_distribution": [float(x) for x in np.asarray(values[mask], dtype=np.float64).tolist()],
            "null_distribution": [float(x) for x in stat_perm.tolist()],
        }
    return summaries


def _primary_fraction_summary(
    fraction_summaries: dict[str, dict[str, float | list[float]]],
    subset_fractions: list[float],
) -> tuple[float, dict[str, float | list[float]]]:
    for fraction in subset_fractions:
        key = f"{float(fraction):.3f}"
        summary = fraction_summaries.get(key)
        if summary is not None:
            return float(fraction), summary
    raise ValueError("fraction_summaries must contain at least one configured subset fraction")


def _fraction_plot_rows(
    observed_isodepth_batched: np.ndarray,
    observed_losses: np.ndarray,
    fraction_per_subset: np.ndarray,
    loss_mask_batched: np.ndarray,
) -> list[dict[str, np.ndarray | float | int]]:
    rows: list[dict[str, np.ndarray | float | int]] = []
    unique_fractions = np.unique(fraction_per_subset)
    for fraction in unique_fractions:
        indices = np.flatnonzero(np.isclose(fraction_per_subset, fraction))
        fraction_losses = observed_losses[indices]
        low_local = int(np.argmin(fraction_losses))
        high_local = int(np.argmax(fraction_losses))
        low_index = int(indices[low_local])
        high_index = int(indices[high_local])
        rows.append(
            {
                "fraction": float(fraction),
                "lowest_index": low_index,
                "lowest_isodepth": np.asarray(observed_isodepth_batched[low_index], dtype=np.float32),
                "lowest_mask": np.asarray(loss_mask_batched[low_index, :, 0], dtype=np.float32),
                "lowest_stat": float(observed_losses[low_index]),
                "highest_index": high_index,
                "highest_isodepth": np.asarray(observed_isodepth_batched[high_index], dtype=np.float32),
                "highest_mask": np.asarray(loss_mask_batched[high_index, :, 0], dtype=np.float32),
                "highest_stat": float(observed_losses[high_index]),
            }
        )
    return rows


def _fraction_loss_summaries(
    observed_stat: float,
    scaled_losses: np.ndarray,
    correlations: np.ndarray,
    fraction_per_subset: np.ndarray,
    size_per_subset: np.ndarray,
    metric: str,
) -> dict[str, dict[str, float | list[float]]]:
    summaries: dict[str, dict[str, float | list[float]]] = {}
    unique_fractions = np.unique(fraction_per_subset)
    for fraction in unique_fractions:
        mask = np.isclose(fraction_per_subset, fraction)
        stat_perm = np.asarray(scaled_losses[mask], dtype=np.float64)
        summaries[f"{float(fraction):.3f}"] = {
            "fraction": float(fraction),
            "subset_size": int(np.round(size_per_subset[mask].mean())),
            "n_subsets": int(mask.sum()),
            "loss_mean": float(observed_stat),
            "loss_std": 0.0,
            "correlation_mean": float(np.mean(correlations[mask])),
            "correlation_std": float(np.std(correlations[mask])),
            "p_value": float(permutation_p_value(metric, observed_stat, stat_perm)),
            "null_mean": float(np.mean(stat_perm)),
            "null_std": float(np.std(stat_perm)),
            "observed_distribution": [float(observed_stat)],
            "null_distribution": [float(x) for x in stat_perm.tolist()],
        }
    return summaries


def _null_group_size(config: TestConfig) -> int:
    if config.batch_size is None:
        return config.n_nulls
    return min(config.batch_size, config.n_nulls)


def run_subsampling_test(
    dataset: DatasetBundle,
    config: TestConfig,
    device: torch.device | None = None,
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()

    true_model, pred_true = train_isodepth_model(
        dataset.S,
        dataset.A,
        config,
        device=device,
        seed_offset=0,
        model_label="true model",
    )
    true_isodepth = _extract_isodepth_from_model(true_model, dataset.S, device)
    observed_stat = float(compute_metric(metric, dataset.A, pred_true))

    loss_mask_batched, fraction_per_subset, size_per_subset = build_subset_masks(
        dataset.n_cells,
        config.subset_fractions,
        config.n_subsets,
        seed=config.seed,
    )
    total_subsets = loss_mask_batched.shape[0]
    s_batched = np.repeat(np.asarray(dataset.S, dtype=np.float32)[None, :, :], total_subsets, axis=0)
    a_batched = np.repeat(np.asarray(dataset.A, dtype=np.float32)[None, :, :], total_subsets, axis=0)

    observed_model, observed_predictions = train_batched_isodepth_model(
        s_batched,
        dataset.A,
        config,
        device=device,
        a_batched=a_batched,
        loss_mask_batched=loss_mask_batched,
        model_label=f"subsampling batch ({total_subsets} subset refits)",
    )
    observed_isodepth_batched = _extract_isodepth_batch(observed_model, s_batched, device)
    subset_losses = compute_masked_losses(
        observed_predictions,
        a_batched,
        loss_mask_batched,
        metric=metric,
    )
    scaled_losses = np.asarray(subset_losses / np.asarray(fraction_per_subset, dtype=np.float64), dtype=np.float64)
    observed_correlations = compute_subset_correlations(
        true_isodepth,
        observed_isodepth_batched,
        loss_mask_batched,
    )

    lowest_index = int(np.argmin(scaled_losses))
    highest_index = int(np.argmax(scaled_losses))
    fraction_summaries = _fraction_loss_summaries(
        observed_stat,
        scaled_losses,
        observed_correlations,
        fraction_per_subset,
        size_per_subset,
        metric,
    )
    primary_fraction, primary_summary = _primary_fraction_summary(
        fraction_summaries,
        [float(value) for value in config.subset_fractions],
    )
    stat_true = float(primary_summary["loss_mean"])
    stat_perm = np.asarray(primary_summary["null_distribution"], dtype=np.float64)
    fraction_plot_rows = _fraction_plot_rows(
        observed_isodepth_batched,
        scaled_losses,
        fraction_per_subset,
        loss_mask_batched,
    )

    return TestResult(
        method_name="subsampling_test",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=time.time() - start,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "pred_true": np.asarray(pred_true, dtype=np.float32),
            "true_isodepth": true_isodepth,
            "lowest_isodepth": np.asarray(observed_isodepth_batched[lowest_index], dtype=np.float32),
            "lowest_S": np.asarray(dataset.S, dtype=np.float32),
            "lowest_stat": float(scaled_losses[lowest_index]),
            "lowest_subset_index": lowest_index,
            "lowest_subset_mask": np.asarray(loss_mask_batched[lowest_index, :, 0], dtype=np.float32),
            "lowest_subset_fraction": float(fraction_per_subset[lowest_index]),
            "highest_isodepth": np.asarray(observed_isodepth_batched[highest_index], dtype=np.float32),
            "highest_S": np.asarray(dataset.S, dtype=np.float32),
            "highest_stat": float(scaled_losses[highest_index]),
            "highest_subset_index": highest_index,
            "highest_subset_mask": np.asarray(loss_mask_batched[highest_index, :, 0], dtype=np.float32),
            "highest_subset_fraction": float(fraction_per_subset[highest_index]),
            "observed_scores": np.asarray(scaled_losses, dtype=np.float64),
            "observed_correlations": np.asarray(observed_correlations, dtype=np.float64),
            "observed_summary": {
                "mean": float(observed_stat),
                "median": float(observed_stat),
                "std": 0.0,
                "min": float(observed_stat),
                "max": float(observed_stat),
            },
            "observed_correlation_summary": _summarize_scores(observed_correlations),
            "null_summary": _summarize_scores(stat_perm),
            "fraction_summaries": fraction_summaries,
            "primary_fraction": primary_fraction,
            "fraction_plot_rows": fraction_plot_rows,
            "subset_fractions": [float(value) for value in config.subset_fractions],
            "subset_fraction_per_subset": np.asarray(fraction_per_subset, dtype=np.float32),
            "subset_size_per_subset": np.asarray(size_per_subset, dtype=np.int64),
            "n_subsets": int(config.n_subsets),
            "n_nulls": int(config.n_perms),
            "summary_statistic": "scaled_subset_reconstruction_loss",
        },
    ).validate()


def run_comparison_subsampling_test(
    dataset: DatasetBundle,
    config: TestConfig,
    device: torch.device | None = None,
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()

    true_model, pred_true = train_isodepth_model(
        dataset.S,
        dataset.A,
        config,
        device=device,
        seed_offset=0,
        model_label="true model",
    )
    true_isodepth = _extract_isodepth_from_model(true_model, dataset.S, device)

    loss_mask_batched, fraction_per_subset, size_per_subset = build_subset_masks(
        dataset.n_cells,
        config.subset_fractions,
        config.n_subsets,
        seed=config.seed,
    )
    total_subsets = loss_mask_batched.shape[0]
    s_batched = np.repeat(np.asarray(dataset.S, dtype=np.float32)[None, :, :], total_subsets, axis=0)
    a_batched = np.repeat(np.asarray(dataset.A, dtype=np.float32)[None, :, :], total_subsets, axis=0)

    observed_model, observed_predictions = train_batched_isodepth_model(
        s_batched,
        dataset.A,
        config,
        device=device,
        a_batched=a_batched,
        loss_mask_batched=loss_mask_batched,
        model_label=f"subset selection batch ({total_subsets} subset refits)",
    )
    observed_isodepth_batched = _extract_isodepth_batch(observed_model, s_batched, device)
    observed_losses = compute_masked_losses(
        observed_predictions,
        a_batched,
        loss_mask_batched,
        metric=metric,
    )
    observed_correlations = compute_subset_correlations(
        true_isodepth,
        observed_isodepth_batched,
        loss_mask_batched,
    )
    stat_perm = np.zeros(config.n_nulls, dtype=np.float64)
    null_losses_per_subset = np.zeros((config.n_nulls, total_subsets), dtype=np.float64)
    rng = np.random.default_rng(config.seed + 100_000)
    group_size = _null_group_size(config)

    for group_start in range(0, config.n_nulls, group_size):
        group_stop = min(group_start + group_size, config.n_nulls)
        active_nulls = group_stop - group_start
        block_models = active_nulls * total_subsets
        s_group = np.repeat(np.asarray(dataset.S, dtype=np.float32)[None, :, :], block_models, axis=0)
        a_group = np.zeros((block_models, dataset.n_cells, dataset.n_genes), dtype=np.float32)
        mask_group = np.zeros((block_models, dataset.n_cells, 1), dtype=np.float32)

        for offset, null_index in enumerate(range(group_start, group_stop)):
            block_start = offset * total_subsets
            block_stop = block_start + total_subsets
            perm = rng.permutation(dataset.n_cells)
            a_perm = np.asarray(dataset.A[perm], dtype=np.float32)
            a_group[block_start:block_stop] = a_perm[None, :, :]
            mask_group[block_start:block_stop] = loss_mask_batched

        _, null_predictions = train_batched_isodepth_model(
            s_group,
            dataset.A,
            config,
            device=device,
            a_batched=a_group,
            loss_mask_batched=mask_group,
            model_label=f"subset null batch {group_start + 1}-{group_stop}/{config.n_nulls}",
        )
        grouped_losses = compute_masked_losses(
            null_predictions,
            a_group,
            mask_group,
            metric=metric,
        )
        grouped_losses = grouped_losses.reshape(active_nulls, total_subsets)
        null_losses_per_subset[group_start:group_stop] = grouped_losses
        stat_perm[group_start:group_stop] = grouped_losses.mean(axis=1)

    runtime_sec = time.time() - start
    lowest_index = int(np.argmin(observed_losses))
    highest_index = int(np.argmax(observed_losses))
    fraction_summaries = _fraction_summaries(
        observed_losses,
        observed_correlations,
        null_losses_per_subset,
        fraction_per_subset,
        size_per_subset,
        metric,
    )
    primary_fraction, primary_summary = _primary_fraction_summary(
        fraction_summaries,
        [float(value) for value in config.subset_fractions],
    )
    stat_true = float(primary_summary["loss_mean"])
    stat_perm = np.asarray(primary_summary["null_distribution"], dtype=np.float64)
    fraction_plot_rows = _fraction_plot_rows(
        observed_isodepth_batched,
        observed_losses,
        fraction_per_subset,
        loss_mask_batched,
    )

    return TestResult(
        method_name="comparison_subsampling_test",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "pred_true": np.asarray(pred_true, dtype=np.float32),
            "true_isodepth": true_isodepth,
            "lowest_isodepth": np.asarray(observed_isodepth_batched[lowest_index], dtype=np.float32),
            "lowest_S": np.asarray(dataset.S, dtype=np.float32),
            "lowest_stat": float(observed_losses[lowest_index]),
            "lowest_subset_index": lowest_index,
            "lowest_subset_mask": np.asarray(loss_mask_batched[lowest_index, :, 0], dtype=np.float32),
            "lowest_subset_fraction": float(fraction_per_subset[lowest_index]),
            "highest_isodepth": np.asarray(observed_isodepth_batched[highest_index], dtype=np.float32),
            "highest_S": np.asarray(dataset.S, dtype=np.float32),
            "highest_stat": float(observed_losses[highest_index]),
            "highest_subset_index": highest_index,
            "highest_subset_mask": np.asarray(loss_mask_batched[highest_index, :, 0], dtype=np.float32),
            "highest_subset_fraction": float(fraction_per_subset[highest_index]),
            "observed_scores": np.asarray(observed_losses, dtype=np.float64),
            "observed_correlations": np.asarray(observed_correlations, dtype=np.float64),
            "observed_summary": _summarize_scores(observed_losses),
            "observed_correlation_summary": _summarize_scores(observed_correlations),
            "null_summary": _summarize_scores(stat_perm),
            "fraction_summaries": fraction_summaries,
            "primary_fraction": primary_fraction,
            "fraction_plot_rows": fraction_plot_rows,
            "subset_fractions": [float(value) for value in config.subset_fractions],
            "subset_fraction_per_subset": np.asarray(fraction_per_subset, dtype=np.float32),
            "subset_size_per_subset": np.asarray(size_per_subset, dtype=np.int64),
            "n_subsets": int(config.n_subsets),
            "n_nulls": int(config.n_nulls),
            "summary_statistic": "mean_masked_subset_loss",
        },
    ).validate()


__all__ = [
    "build_subset_masks",
    "compute_masked_losses",
    "compute_subset_correlations",
    "run_comparison_subsampling_test",
    "run_subsampling_test",
]
