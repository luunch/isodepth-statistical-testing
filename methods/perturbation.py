from __future__ import annotations

import time

import numpy as np
import torch

from data.schemas import DatasetBundle, TestConfig, TestResult
from methods.metrics import (
    canonicalize_metric_name,
    compute_metric,
    compute_metric_batch,
    metric_prefers_lower,
    permutation_p_value,
)
from methods.trainers import get_training_metadata, resolve_device, train_parallel_isodepth_model


def perturb_coordinates(S: np.ndarray, delta: float, seed: int) -> np.ndarray:
    s = np.asarray(S, dtype=np.float32)
    mins = s.min(axis=0, keepdims=True)
    maxs = s.max(axis=0, keepdims=True)
    axis_range = np.maximum(maxs - mins, 1e-8)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=delta * axis_range, size=s.shape).astype(np.float32)
    return np.asarray(np.clip(s + noise, mins, maxs), dtype=np.float32)


def normalize_depth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    vmin = float(values.min())
    vmax = float(values.max())
    return np.asarray((values - vmin) / (vmax - vmin + 1e-8), dtype=np.float32)


def _as_single_feature(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def score_depth_similarity(metric: str, d_true: np.ndarray, d_perturbed: np.ndarray) -> float:
    metric = canonicalize_metric_name(metric)
    d_true_norm = normalize_depth(d_true)
    d_perturbed_norm = normalize_depth(d_perturbed)
    score = compute_metric(metric, _as_single_feature(d_true_norm), _as_single_feature(d_perturbed_norm))
    if metric in {"pearson_corr_mean", "spearman_corr_mean"}:
        score = abs(score)
    return float(score)


def _summarize_scores(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _delta_schedule(config: TestConfig) -> list[float]:
    return [float(value) for value in config.delta]


def _build_perturbation_batch(
    S: np.ndarray,
    config: TestConfig,
    *,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(S, dtype=np.float32)
    delta_values = _delta_schedule(config)
    total_perturbations = len(delta_values) * config.n_perms
    s_batched = np.zeros((total_perturbations + 1, s.shape[0], 2), dtype=np.float32)
    delta_per_score = np.zeros(total_perturbations, dtype=np.float32)
    s_batched[0] = s
    delta_stride = config.n_perms + 17

    model_index = 1
    for delta_index, delta in enumerate(delta_values):
        local_seed_base = seed_base + delta_index * delta_stride
        for perm_index in range(config.n_perms):
            s_batched[model_index] = perturb_coordinates(s, delta, local_seed_base + perm_index)
            delta_per_score[model_index - 1] = float(delta)
            model_index += 1
    return s_batched, delta_per_score


def _extract_isodepth_batch(model, s_batched: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        s_batched_t = torch.tensor(s_batched, dtype=torch.float32, device=device)
        isodepth_batched = model.encoder(s_batched_t).detach().cpu().numpy().squeeze(-1)
    return np.asarray(isodepth_batched, dtype=np.float32)


def _rerun_summary(model) -> dict[str, object]:
    metadata = get_training_metadata(model)
    return {
        "n_reruns": int(metadata["n_reruns"]),
        "selection_loss": str(metadata["selection_loss"]),
    }


def _rerun_index_and_loss(model, index: int) -> tuple[int, float]:
    metadata = get_training_metadata(model)
    return (
        int(metadata["best_rerun_index_per_model"][index]),
        float(metadata["best_train_loss_per_model"][index]),
    )


def _run_perturbation_batch(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: torch.device,
    seed_base: int,
    model_label: str,
    a_batched: np.ndarray | None = None,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    s_batched, delta_per_score = _build_perturbation_batch(S, config, seed_base=seed_base)
    model, _ = train_parallel_isodepth_model(
        S,
        A,
        config,
        s_batched=s_batched,
        a_batched=a_batched,
        device=device,
        model_label=model_label,
    )
    isodepth_batched = _extract_isodepth_batch(model, s_batched, device)
    return model, s_batched, isodepth_batched, delta_per_score


def _compute_perturbation_scores(metric: str, isodepth_batched: np.ndarray) -> np.ndarray:
    true_isodepth = np.asarray(isodepth_batched[0], dtype=np.float32)
    scores = np.zeros(isodepth_batched.shape[0] - 1, dtype=np.float64)
    for i in range(scores.shape[0]):
        scores[i] = score_depth_similarity(metric, true_isodepth, isodepth_batched[i + 1])
    return scores


def _null_group_size(config: TestConfig) -> int:
    if config.batch_size is None:
        return config.n_nulls
    return min(config.batch_size, config.n_nulls)


def _run_grouped_null_batches(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: torch.device,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + 100_000)
    delta_values = _delta_schedule(config)
    models_per_null = 1 + len(delta_values) * config.n_perms
    group_size = _null_group_size(config)
    stat_perm = np.zeros(config.n_nulls, dtype=np.float64)
    null_scores_per_perturbation = np.zeros((config.n_nulls, models_per_null - 1), dtype=np.float64)
    seed_stride = models_per_null + 17

    for group_start in range(0, config.n_nulls, group_size):
        group_stop = min(group_start + group_size, config.n_nulls)
        active_nulls = group_stop - group_start
        s_group = np.zeros((active_nulls * models_per_null, S.shape[0], 2), dtype=np.float32)
        a_group = np.zeros((active_nulls * models_per_null, A.shape[0], A.shape[1]), dtype=np.float32)

        for offset, null_index in enumerate(range(group_start, group_stop)):
            block_start = offset * models_per_null
            block_stop = block_start + models_per_null
            s_group[block_start:block_stop], _ = _build_perturbation_batch(
                S,
                config,
                seed_base=config.seed + (null_index + 1) * seed_stride,
            )
            perm = rng.permutation(A.shape[0])
            a_null = np.asarray(A[perm], dtype=np.float32)
            a_group[block_start:block_stop] = a_null[None, :, :]

        model, _ = train_parallel_isodepth_model(
            S,
            A,
            config,
            s_batched=s_group,
            a_batched=a_group,
            device=device,
            model_label=(
                f"parallel null perturbation batch {group_start + 1}-{group_stop}/{config.n_nulls}"
            ),
        )
        grouped_isodepth = _extract_isodepth_batch(model, s_group, device)
        grouped_isodepth = grouped_isodepth.reshape(active_nulls, models_per_null, S.shape[0])
        for offset, null_index in enumerate(range(group_start, group_stop)):
            null_scores = _compute_perturbation_scores(metric=metric, isodepth_batched=grouped_isodepth[offset])
            null_scores_per_perturbation[null_index] = null_scores
            stat_perm[null_index] = float(np.mean(null_scores))

    return stat_perm, null_scores_per_perturbation


def _delta_summaries(
    observed_scores: np.ndarray,
    null_scores_per_perturbation: np.ndarray,
    delta_per_score: np.ndarray,
    metric: str,
) -> dict[str, dict[str, float | list[float]]]:
    summaries: dict[str, dict[str, float | list[float]]] = {}
    unique_deltas = np.unique(delta_per_score)
    for delta in unique_deltas:
        mask = np.isclose(delta_per_score, delta)
        stat_true = float(np.mean(observed_scores[mask]))
        stat_perm = np.asarray(null_scores_per_perturbation[:, mask].mean(axis=1), dtype=np.float64)
        summaries[f"{float(delta):.6g}"] = {
            "delta": float(delta),
            "n_perturbations": int(mask.sum()),
            "score_mean": stat_true,
            "score_std": float(np.std(observed_scores[mask])),
            "p_value": float(permutation_p_value(metric, stat_true, stat_perm)),
            "null_mean": float(np.mean(stat_perm)),
            "null_std": float(np.std(stat_perm)),
            "observed_distribution": [float(x) for x in np.asarray(observed_scores[mask], dtype=np.float64).tolist()],
            "null_distribution": [float(x) for x in stat_perm.tolist()],
        }
    return summaries


def _primary_delta_summary(
    delta_summaries: dict[str, dict[str, float | list[float]]],
    delta_schedule: list[float],
) -> tuple[float, dict[str, float | list[float]]]:
    for delta in delta_schedule:
        key = f"{float(delta):.6g}"
        summary = delta_summaries.get(key)
        if summary is not None:
            return float(delta), summary
    raise ValueError("delta_summaries must contain at least one configured delta")


def _delta_plot_rows(
    observed_s_batched: np.ndarray,
    observed_isodepth_batched: np.ndarray,
    observed_scores: np.ndarray,
    delta_per_score: np.ndarray,
) -> list[dict[str, np.ndarray | float | int]]:
    rows: list[dict[str, np.ndarray | float | int]] = []
    unique_deltas = np.unique(delta_per_score)
    for delta in unique_deltas:
        indices = np.flatnonzero(np.isclose(delta_per_score, delta))
        delta_scores = observed_scores[indices]
        low_local = int(np.argmin(delta_scores))
        high_local = int(np.argmax(delta_scores))
        low_index = int(indices[low_local])
        high_index = int(indices[high_local])
        rows.append(
            {
                "delta": float(delta),
                "lowest_index": low_index,
                "lowest_isodepth": np.asarray(observed_isodepth_batched[low_index + 1], dtype=np.float32),
                "lowest_S": np.asarray(observed_s_batched[low_index + 1], dtype=np.float32),
                "lowest_stat": float(observed_scores[low_index]),
                "highest_index": high_index,
                "highest_isodepth": np.asarray(observed_isodepth_batched[high_index + 1], dtype=np.float32),
                "highest_S": np.asarray(observed_s_batched[high_index + 1], dtype=np.float32),
                "highest_stat": float(observed_scores[high_index]),
            }
        )
    return rows


def _delta_loss_summaries(
    observed_stat: float,
    null_losses: np.ndarray,
    delta_per_score: np.ndarray,
    metric: str,
) -> dict[str, dict[str, float | list[float]]]:
    summaries: dict[str, dict[str, float | list[float]]] = {}
    unique_deltas = np.unique(delta_per_score)
    for delta in unique_deltas:
        mask = np.isclose(delta_per_score, delta)
        stat_perm = np.asarray(null_losses[mask], dtype=np.float64)
        summaries[f"{float(delta):.6g}"] = {
            "delta": float(delta),
            "n_perturbations": int(mask.sum()),
            "score_mean": float(observed_stat),
            "score_std": 0.0,
            "p_value": float(permutation_p_value(metric, observed_stat, stat_perm)),
            "null_mean": float(np.mean(stat_perm)),
            "null_std": float(np.std(stat_perm)),
            "observed_distribution": [float(observed_stat)],
            "null_distribution": [float(x) for x in stat_perm.tolist()],
        }
    return summaries


def run_perturbation_test(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()
    observed_s_batched, delta_per_score = _build_perturbation_batch(
        dataset.S,
        config,
        seed_base=config.seed,
    )
    observed_model, observed_predictions = train_parallel_isodepth_model(
        dataset.S,
        dataset.A,
        config,
        s_batched=observed_s_batched,
        device=device,
        model_label=f"perturbation batch (observed + {config.n_perms} perturbed models)",
    )
    observed_model_losses = compute_metric_batch(metric, dataset.A, observed_predictions)
    observed_stat = float(observed_model_losses[0])
    null_losses = np.asarray(observed_model_losses[1:], dtype=np.float64)
    observed_isodepth_batched = _extract_isodepth_batch(
        observed_model,
        observed_s_batched,
        device,
    )
    delta_summaries = _delta_loss_summaries(
        observed_stat,
        null_losses,
        delta_per_score,
        metric,
    )
    primary_delta, primary_summary = _primary_delta_summary(delta_summaries, _delta_schedule(config))
    stat_true = float(primary_summary["score_mean"])
    stat_perm = np.asarray(primary_summary["null_distribution"], dtype=np.float64)
    runtime_sec = time.time() - start
    delta_plot_rows = _delta_plot_rows(
        observed_s_batched,
        observed_isodepth_batched,
        null_losses,
        delta_per_score,
    )
    if metric_prefers_lower(metric):
        extreme_index = int(np.argmin(null_losses))
        opposite_index = int(np.argmax(null_losses))
    else:
        extreme_index = int(np.argmax(null_losses))
        opposite_index = int(np.argmin(null_losses))
    true_rerun_index, true_train_loss = _rerun_index_and_loss(observed_model, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(observed_model, extreme_index + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(observed_model, opposite_index + 1)

    return TestResult(
        method_name="perturbation_test",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "true_isodepth": np.asarray(observed_isodepth_batched[0], dtype=np.float32),
            "rerun_summary": _rerun_summary(observed_model),
            "true_rerun_index": int(true_rerun_index),
            "true_train_loss": float(true_train_loss),
            "perturbed_isodepth": np.asarray(observed_isodepth_batched[1], dtype=np.float32),
            "perturbed_S": np.asarray(observed_s_batched[1], dtype=np.float32),
            "lowest_isodepth": np.asarray(observed_isodepth_batched[extreme_index + 1], dtype=np.float32),
            "lowest_S": np.asarray(observed_s_batched[extreme_index + 1], dtype=np.float32),
            "lowest_stat": float(null_losses[extreme_index]),
            "lowest_perm_index": int(extreme_index),
            "lowest_rerun_index": int(lowest_rerun_index),
            "lowest_train_loss": float(lowest_train_loss),
            "highest_isodepth": np.asarray(observed_isodepth_batched[opposite_index + 1], dtype=np.float32),
            "highest_S": np.asarray(observed_s_batched[opposite_index + 1], dtype=np.float32),
            "highest_stat": float(null_losses[opposite_index]),
            "highest_perm_index": int(opposite_index),
            "highest_rerun_index": int(highest_rerun_index),
            "highest_train_loss": float(highest_train_loss),
            "delta": _delta_schedule(config),
            "delta_summaries": delta_summaries,
            "primary_delta": primary_delta,
            "delta_plot_rows": delta_plot_rows,
            "perturb_target": config.perturb_target,
            "observed_scores": np.asarray(null_losses, dtype=np.float64),
            "observed_summary": {
                "mean": float(observed_stat),
                "median": float(observed_stat),
                "std": 0.0,
                "min": float(observed_stat),
                "max": float(observed_stat),
            },
            "null_summary": _summarize_scores(stat_perm),
            "summary_statistic": "reconstruction_loss",
            "n_nulls": int(config.n_perms),
        },
    ).validate()


def run_comparison_perturbation_test(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()

    observed_model, observed_s_batched, observed_isodepth_batched, delta_per_score = _run_perturbation_batch(
        dataset.S,
        dataset.A,
        config,
        device=device,
        seed_base=config.seed,
        model_label=f"parallel perturbation batch (observed + {config.n_perms} perturbed models)",
    )
    observed_scores = _compute_perturbation_scores(metric, observed_isodepth_batched)

    if metric_prefers_lower(metric):
        extreme_index = int(np.argmin(observed_scores))
        opposite_index = int(np.argmax(observed_scores))
    else:
        extreme_index = int(np.argmax(observed_scores))
        opposite_index = int(np.argmin(observed_scores))
    true_rerun_index, true_train_loss = _rerun_index_and_loss(observed_model, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(observed_model, extreme_index + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(observed_model, opposite_index + 1)

    _, null_scores_per_perturbation = _run_grouped_null_batches(
        dataset.S,
        dataset.A,
        config,
        device=device,
        metric=metric,
    )

    runtime_sec = time.time() - start
    observed_summary = _summarize_scores(observed_scores)
    delta_summaries = _delta_summaries(
        observed_scores,
        null_scores_per_perturbation,
        delta_per_score,
        metric,
    )
    primary_delta, primary_summary = _primary_delta_summary(delta_summaries, _delta_schedule(config))
    stat_true = float(primary_summary["score_mean"])
    stat_perm = np.asarray(primary_summary["null_distribution"], dtype=np.float64)
    null_summary = _summarize_scores(stat_perm)
    delta_plot_rows = _delta_plot_rows(
        observed_s_batched,
        observed_isodepth_batched,
        observed_scores,
        delta_per_score,
    )

    return TestResult(
        method_name="comparison_perturbation_test",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "true_isodepth": np.asarray(observed_isodepth_batched[0], dtype=np.float32),
            "rerun_summary": _rerun_summary(observed_model),
            "true_rerun_index": int(true_rerun_index),
            "true_train_loss": float(true_train_loss),
            "perturbed_isodepth": np.asarray(observed_isodepth_batched[1], dtype=np.float32),
            "perturbed_S": np.asarray(observed_s_batched[1], dtype=np.float32),
            "lowest_isodepth": np.asarray(observed_isodepth_batched[extreme_index + 1], dtype=np.float32),
            "lowest_S": np.asarray(observed_s_batched[extreme_index + 1], dtype=np.float32),
            "lowest_stat": float(observed_scores[extreme_index]),
            "lowest_perm_index": int(extreme_index),
            "lowest_rerun_index": int(lowest_rerun_index),
            "lowest_train_loss": float(lowest_train_loss),
            "highest_isodepth": np.asarray(observed_isodepth_batched[opposite_index + 1], dtype=np.float32),
            "highest_S": np.asarray(observed_s_batched[opposite_index + 1], dtype=np.float32),
            "highest_stat": float(observed_scores[opposite_index]),
            "highest_perm_index": int(opposite_index),
            "highest_rerun_index": int(highest_rerun_index),
            "highest_train_loss": float(highest_train_loss),
            "delta": _delta_schedule(config),
            "delta_summaries": delta_summaries,
            "primary_delta": primary_delta,
            "delta_plot_rows": delta_plot_rows,
            "perturb_target": config.perturb_target,
            "observed_scores": np.asarray(observed_scores, dtype=np.float64),
            "observed_summary": observed_summary,
            "null_summary": null_summary,
            "summary_statistic": "mean",
            "n_nulls": int(config.n_nulls),
        },
    ).validate()


__all__ = [
    "normalize_depth",
    "perturb_coordinates",
    "run_comparison_perturbation_test",
    "run_perturbation_test",
    "score_depth_similarity",
]
