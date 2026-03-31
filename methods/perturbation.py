from __future__ import annotations

import time

import numpy as np
import torch

from data.schemas import DatasetBundle, TestConfig, TestResult
from methods.metrics import canonicalize_metric_name, compute_metric, metric_prefers_lower, permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model


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


def _build_perturbation_batch(S: np.ndarray, config: TestConfig, *, seed_base: int) -> np.ndarray:
    s = np.asarray(S, dtype=np.float32)
    s_batched = np.zeros((config.n_perms + 1, s.shape[0], 2), dtype=np.float32)
    s_batched[0] = s
    for i in range(config.n_perms):
        s_batched[i + 1] = perturb_coordinates(s, config.delta, seed_base + i)
    return s_batched


def _extract_isodepth_batch(model, s_batched: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        s_batched_t = torch.tensor(s_batched, dtype=torch.float32, device=device)
        isodepth_batched = model.encoder(s_batched_t).detach().cpu().numpy().squeeze(-1)
    return np.asarray(isodepth_batched, dtype=np.float32)


def _run_perturbation_batch(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: torch.device,
    seed_base: int,
    model_label: str,
    a_batched: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    s_batched = _build_perturbation_batch(S, config, seed_base=seed_base)
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
    return s_batched, isodepth_batched


def _compute_perturbation_scores(metric: str, isodepth_batched: np.ndarray) -> np.ndarray:
    true_isodepth = np.asarray(isodepth_batched[0], dtype=np.float32)
    scores = np.zeros(isodepth_batched.shape[0] - 1, dtype=np.float64)
    for i in range(scores.shape[0]):
        scores[i] = score_depth_similarity(metric, true_isodepth, isodepth_batched[i + 1])
    return scores


def _summarize_scores(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


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
) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 100_000)
    models_per_null = config.n_perms + 1
    group_size = _null_group_size(config)
    stat_perm = np.zeros(config.n_nulls, dtype=np.float64)
    seed_stride = config.n_perms + 7

    for group_start in range(0, config.n_nulls, group_size):
        group_stop = min(group_start + group_size, config.n_nulls)
        active_nulls = group_stop - group_start
        s_group = np.zeros((active_nulls * models_per_null, S.shape[0], 2), dtype=np.float32)
        a_group = np.zeros((active_nulls * models_per_null, A.shape[0], A.shape[1]), dtype=np.float32)

        for offset, null_index in enumerate(range(group_start, group_stop)):
            block_start = offset * models_per_null
            block_stop = block_start + models_per_null
            s_group[block_start:block_stop] = _build_perturbation_batch(
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
            stat_perm[null_index] = float(np.mean(null_scores))

    return stat_perm


def run_perturbation_robustness_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()

    observed_s_batched, observed_isodepth_batched = _run_perturbation_batch(
        dataset.S,
        dataset.A,
        config,
        device=device,
        seed_base=config.seed,
        model_label=f"parallel perturbation batch (observed + {config.n_perms} perturbed models)",
    )
    observed_scores = _compute_perturbation_scores(metric, observed_isodepth_batched)
    stat_true = float(np.mean(observed_scores))

    if metric_prefers_lower(metric):
        extreme_index = int(np.argmin(observed_scores))
        opposite_index = int(np.argmax(observed_scores))
    else:
        extreme_index = int(np.argmax(observed_scores))
        opposite_index = int(np.argmin(observed_scores))

    stat_perm = _run_grouped_null_batches(dataset.S, dataset.A, config, device=device, metric=metric)

    runtime_sec = time.time() - start
    observed_summary = _summarize_scores(observed_scores)
    null_summary = _summarize_scores(stat_perm)

    return TestResult(
        method_name="perturbation_robustness",
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
            "perturbed_isodepth": np.asarray(observed_isodepth_batched[1], dtype=np.float32),
            "perturbed_S": np.asarray(observed_s_batched[1], dtype=np.float32),
            "lowest_isodepth": np.asarray(observed_isodepth_batched[extreme_index + 1], dtype=np.float32),
            "lowest_S": np.asarray(observed_s_batched[extreme_index + 1], dtype=np.float32),
            "lowest_stat": float(observed_scores[extreme_index]),
            "lowest_perm_index": int(extreme_index),
            "highest_isodepth": np.asarray(observed_isodepth_batched[opposite_index + 1], dtype=np.float32),
            "highest_S": np.asarray(observed_s_batched[opposite_index + 1], dtype=np.float32),
            "highest_stat": float(observed_scores[opposite_index]),
            "highest_perm_index": int(opposite_index),
            "delta": float(config.delta),
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
    "run_perturbation_robustness_method",
    "score_depth_similarity",
]
