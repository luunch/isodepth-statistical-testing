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
from methods.perturbation import run_comparison_perturbation_test, run_perturbation_test
from methods.subsampling import run_comparison_subsampling_test, run_subsampling_test
from methods.trainers import (
    extract_model_isodepth,
    get_training_metadata,
    resolve_device,
    train_isodepth_model,
    train_parallel_isodepth_model,
)


def _extract_isodepth_from_model(model, S: np.ndarray, device: torch.device) -> np.ndarray:
    isodepth = extract_model_isodepth(model, S, device)
    if isodepth.shape[1] == 1:
        return np.asarray(isodepth[:, 0], dtype=np.float32)
    return np.asarray(isodepth, dtype=np.float32)


def _extract_batched_isodepth(model, s_batched: torch.Tensor) -> np.ndarray:
    latent_dim = int(getattr(model, "latent_dim", 0))
    if latent_dim <= 0 or not hasattr(model, "encoder"):
        n_models, n_cells = int(s_batched.shape[0]), int(s_batched.shape[1])
        return np.zeros((n_models, n_cells, 0), dtype=np.float32)
    with torch.no_grad():
        isodepth_batched = model.encoder(s_batched).detach().cpu().numpy()
    return np.asarray(isodepth_batched, dtype=np.float32).reshape(s_batched.shape[0], s_batched.shape[1], latent_dim)


def _select_extreme_index(metric: str, stat_perm: np.ndarray) -> int:
    if metric_prefers_lower(metric):
        return int(np.argmin(stat_perm))
    return int(np.argmax(stat_perm))


def _select_low_high_indices(stat_perm: np.ndarray) -> tuple[int, int]:
    return int(np.argmin(stat_perm)), int(np.argmax(stat_perm))


def _build_permuted_coordinate_batch(
    S: np.ndarray,
    *,
    n_perms: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    s_t = torch.tensor(S, dtype=torch.float32, device=device)
    n_models = n_perms + 1
    s_batched = torch.zeros((n_models, S.shape[0], S.shape[1]), dtype=torch.float32, device=device)
    s_batched[0] = s_t
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutations: list[np.ndarray] = []
    for m in range(1, n_models):
        perm = torch.randperm(S.shape[0], generator=generator)
        permutations.append(perm.cpu().numpy())
        s_batched[m] = s_t[perm.to(device=device)]
    return s_batched, permutations


def _delta_p_value(stat_true: float, stat_perm: np.ndarray) -> float:
    stat_perm = np.asarray(stat_perm, dtype=np.float64)
    return float((1 + np.sum(stat_perm <= stat_true)) / (stat_perm.size + 1))


def _format_isodepth_for_artifact(isodepth: np.ndarray) -> np.ndarray:
    arr = np.asarray(isodepth, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return np.asarray(arr[:, 0], dtype=np.float32)
    return arr


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


def _summarize_exact_existence_step(
    dataset: DatasetBundle,
    s_batched: torch.Tensor,
    losses_k: np.ndarray,
    losses_k_plus_1: np.ndarray,
    isodepth_k_plus_1: np.ndarray,
    tested_dim: int,
    *,
    model_k,
    model_k_plus_1,
) -> dict[str, object]:
    stat_true = float(losses_k_plus_1[0] - losses_k[0])
    stat_perm = np.asarray(losses_k_plus_1[1:] - losses_k[1:], dtype=np.float64)
    p_value = _delta_p_value(stat_true, stat_perm)
    low_idx, high_idx = _select_low_high_indices(stat_perm)
    true_isodepth = _format_isodepth_for_artifact(isodepth_k_plus_1[0])
    lowest_isodepth = _format_isodepth_for_artifact(isodepth_k_plus_1[low_idx + 1])
    highest_isodepth = _format_isodepth_for_artifact(isodepth_k_plus_1[high_idx + 1])
    lowest_S = np.asarray(s_batched[low_idx + 1].detach().cpu().numpy(), dtype=np.float32)
    highest_S = np.asarray(s_batched[high_idx + 1].detach().cpu().numpy(), dtype=np.float32)
    true_rerun_index_k, true_train_loss_k = _rerun_index_and_loss(model_k, 0)
    true_rerun_index_k_plus_1, true_train_loss_k_plus_1 = _rerun_index_and_loss(model_k_plus_1, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(model_k_plus_1, low_idx + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(model_k_plus_1, high_idx + 1)
    return {
        "tested_dim": int(tested_dim),
        "previous_dim": int(tested_dim - 1),
        "test_type": "dimension_increase",
        "stat_true": stat_true,
        "stat_perm": stat_perm,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),  # overwritten by caller using config.alpha
        "loss_k_true": float(losses_k[0]),
        "loss_k_plus_1_true": float(losses_k_plus_1[0]),
        "true_isodepth": true_isodepth,
        "lowest_isodepth": lowest_isodepth,
        "lowest_S": lowest_S,
        "lowest_stat": float(stat_perm[low_idx]),
        "lowest_perm_index": int(low_idx),
        "highest_isodepth": highest_isodepth,
        "highest_S": highest_S,
        "highest_stat": float(stat_perm[high_idx]),
        "highest_perm_index": int(high_idx),
        "true_rerun_index": int(true_rerun_index_k_plus_1),
        "true_train_loss": float(true_train_loss_k_plus_1),
        "true_rerun_index_k": int(true_rerun_index_k),
        "true_train_loss_k": float(true_train_loss_k),
        "true_rerun_index_k_plus_1": int(true_rerun_index_k_plus_1),
        "true_train_loss_k_plus_1": float(true_train_loss_k_plus_1),
        "lowest_rerun_index": int(lowest_rerun_index),
        "lowest_train_loss": float(lowest_train_loss),
        "highest_rerun_index": int(highest_rerun_index),
        "highest_train_loss": float(highest_train_loss),
        "rerun_summary": _rerun_summary(model_k_plus_1),
        "null_summary": {
            "mean": float(np.mean(stat_perm)),
            "std": float(np.std(stat_perm)),
            "min": float(np.min(stat_perm)),
            "max": float(np.max(stat_perm)),
        },
        "n_cells": int(dataset.n_cells),
    }


def _summarize_exact_existence_first_step(
    existence_result: TestResult,
    *,
    alpha: float,
) -> dict[str, object]:
    return {
        "tested_dim": 1,
        "previous_dim": 0,
        "test_type": "existence",
        "stat_true": float(existence_result.stat_true),
        "p_value": float(existence_result.p_value),
        "significant": bool(float(existence_result.p_value) < alpha),
        "true_isodepth": np.asarray(existence_result.artifacts["true_isodepth"], dtype=np.float32),
        "lowest_isodepth": np.asarray(existence_result.artifacts["lowest_isodepth"], dtype=np.float32),
        "lowest_S": np.asarray(existence_result.artifacts["lowest_S"], dtype=np.float32),
        "lowest_stat": float(existence_result.artifacts["lowest_stat"]),
        "lowest_perm_index": int(existence_result.artifacts["lowest_perm_index"]),
        "highest_isodepth": np.asarray(existence_result.artifacts["highest_isodepth"], dtype=np.float32),
        "highest_S": np.asarray(existence_result.artifacts["highest_S"], dtype=np.float32),
        "highest_stat": float(existence_result.artifacts["highest_stat"]),
        "highest_perm_index": int(existence_result.artifacts["highest_perm_index"]),
        "rerun_summary": dict(existence_result.artifacts["rerun_summary"]),
        "true_rerun_index": int(existence_result.artifacts["true_rerun_index"]),
        "true_train_loss": float(existence_result.artifacts["true_train_loss"]),
        "lowest_rerun_index": int(existence_result.artifacts["lowest_rerun_index"]),
        "lowest_train_loss": float(existence_result.artifacts["lowest_train_loss"]),
        "highest_rerun_index": int(existence_result.artifacts["highest_rerun_index"]),
        "highest_train_loss": float(existence_result.artifacts["highest_train_loss"]),
        "null_summary": {
            "mean": float(np.mean(existence_result.stat_perm)),
            "std": float(np.std(existence_result.stat_perm)),
            "min": float(np.min(existence_result.stat_perm)),
            "max": float(np.max(existence_result.stat_perm)),
        },
        "n_cells": int(existence_result.n_cells),
        "alpha": float(alpha),
        "null_distribution": np.asarray(existence_result.stat_perm, dtype=np.float64),
        "observed_stat": float(existence_result.stat_true),
        "dimension_labels": ["d1"],
        "pred_true_k_plus_1": np.asarray(existence_result.artifacts["pred_true"], dtype=np.float32),
    }


def run_parallel_permutation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()
    model, predictions = train_parallel_isodepth_model(dataset.S, dataset.A, config, device=device)
    stats = compute_metric_batch(metric, dataset.A, predictions)
    stat_true = float(stats[0])
    stat_perm = np.asarray(stats[1:], dtype=np.float64)
    low_idx, high_idx = _select_low_high_indices(stat_perm)
    s_batched, _ = _build_permuted_coordinate_batch(dataset.S, n_perms=config.n_perms, seed=config.seed, device=device)
    isodepth_batched = _extract_batched_isodepth(model, s_batched)
    true_rerun_index, true_train_loss = _rerun_index_and_loss(model, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(model, low_idx + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(model, high_idx + 1)
    runtime_sec = time.time() - start

    return TestResult(
        method_name="parallel_permutation",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "model": model,
            "pred_true": np.asarray(predictions[0], dtype=np.float32),
            "true_isodepth": np.asarray(isodepth_batched[0, :, 0], dtype=np.float32),
            "rerun_summary": _rerun_summary(model),
            "true_rerun_index": int(true_rerun_index),
            "true_train_loss": float(true_train_loss),
            "lowest_isodepth": np.asarray(isodepth_batched[low_idx + 1, :, 0], dtype=np.float32),
            "lowest_S": np.asarray(s_batched[low_idx + 1].detach().cpu().numpy(), dtype=np.float32),
            "lowest_stat": float(stat_perm[low_idx]),
            "lowest_perm_index": low_idx,
            "lowest_rerun_index": int(lowest_rerun_index),
            "lowest_train_loss": float(lowest_train_loss),
            "highest_isodepth": np.asarray(isodepth_batched[high_idx + 1, :, 0], dtype=np.float32),
            "highest_S": np.asarray(s_batched[high_idx + 1].detach().cpu().numpy(), dtype=np.float32),
            "highest_stat": float(stat_perm[high_idx]),
            "highest_perm_index": high_idx,
            "highest_rerun_index": int(highest_rerun_index),
            "highest_train_loss": float(highest_train_loss),
        },
    ).validate()


def run_exact_existence_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()
    step_summaries: dict[str, dict[str, object]] = {}
    dimension_plot_rows: list[dict[str, object]] = []
    selected_spatial_dims = 0
    final_step_summary: dict[str, object] | None = None

    existence_result = run_parallel_permutation_method(dataset, config, device=device)
    first_step_summary = _summarize_exact_existence_first_step(existence_result, alpha=config.alpha)
    step_summaries["1"] = first_step_summary
    dimension_plot_rows.append(
        {
            "tested_dim": 1,
            "true_isodepth": first_step_summary["true_isodepth"],
            "lowest_isodepth": first_step_summary["lowest_isodepth"],
            "lowest_S": first_step_summary["lowest_S"],
            "lowest_stat": float(first_step_summary["lowest_stat"]),
            "highest_isodepth": first_step_summary["highest_isodepth"],
            "highest_S": first_step_summary["highest_S"],
            "highest_stat": float(first_step_summary["highest_stat"]),
            "dimension_labels": ["d1"],
            "p_value": float(first_step_summary["p_value"]),
            "significant": bool(first_step_summary["significant"]),
            "test_type": "existence",
        }
    )
    final_step_summary = first_step_summary
    if bool(first_step_summary["significant"]):
        selected_spatial_dims = 1
    else:
        runtime_sec = time.time() - start
        return TestResult(
            method_name="exact_existence",
            metric=metric,
            p_value=float(first_step_summary["p_value"]),
            stat_true=float(first_step_summary["observed_stat"]),
            stat_perm=np.asarray(first_step_summary["null_distribution"], dtype=np.float64),
            runtime_sec=runtime_sec,
            n_cells=dataset.n_cells,
            n_genes=dataset.n_genes,
            config={"test": config.__dict__.copy()},
            artifacts={
                "selected_spatial_dims": 0,
                "tested_spatial_dims": [1],
                "step_summaries": step_summaries,
                "dimension_plot_rows": dimension_plot_rows,
                "true_isodepth": np.asarray(first_step_summary["true_isodepth"], dtype=np.float32),
                "rerun_summary": dict(first_step_summary["rerun_summary"]),
                "true_rerun_index": int(first_step_summary["true_rerun_index"]),
                "true_train_loss": float(first_step_summary["true_train_loss"]),
                "lowest_isodepth": np.asarray(first_step_summary["lowest_isodepth"], dtype=np.float32),
                "lowest_S": np.asarray(first_step_summary["lowest_S"], dtype=np.float32),
                "lowest_stat": float(first_step_summary["lowest_stat"]),
                "lowest_perm_index": int(first_step_summary["lowest_perm_index"]),
                "lowest_rerun_index": int(first_step_summary["lowest_rerun_index"]),
                "lowest_train_loss": float(first_step_summary["lowest_train_loss"]),
                "highest_isodepth": np.asarray(first_step_summary["highest_isodepth"], dtype=np.float32),
                "highest_S": np.asarray(first_step_summary["highest_S"], dtype=np.float32),
                "highest_stat": float(first_step_summary["highest_stat"]),
                "highest_perm_index": int(first_step_summary["highest_perm_index"]),
                "highest_rerun_index": int(first_step_summary["highest_rerun_index"]),
                "highest_train_loss": float(first_step_summary["highest_train_loss"]),
                "null_summary": dict(first_step_summary["null_summary"]),
                "alpha": float(config.alpha),
                "max_spatial_dims": int(config.max_spatial_dims),
            },
        ).validate()

    for tested_dim in range(2, config.max_spatial_dims + 1):
        s_batched, _ = _build_permuted_coordinate_batch(
            dataset.S,
            n_perms=config.n_perms,
            seed=config.seed + tested_dim - 1,
            device=device,
        )
        s_batched_np = np.asarray(s_batched.detach().cpu().numpy(), dtype=np.float32)

        model_k, predictions_k = train_parallel_isodepth_model(
            dataset.S,
            dataset.A,
            config,
            device=device,
            s_batched=s_batched_np,
            latent_dim=tested_dim - 1,
            model_label=f"exact existence k={tested_dim - 1}",
        )
        model_k_plus_1, predictions_k_plus_1 = train_parallel_isodepth_model(
            dataset.S,
            dataset.A,
            config,
            device=device,
            s_batched=s_batched_np,
            latent_dim=tested_dim,
            model_label=f"exact existence k={tested_dim}",
        )
        losses_k = compute_metric_batch(metric, dataset.A, predictions_k)
        losses_k_plus_1 = compute_metric_batch(metric, dataset.A, predictions_k_plus_1)
        isodepth_k_plus_1 = _extract_batched_isodepth(model_k_plus_1, s_batched)

        step_summary = _summarize_exact_existence_step(
            dataset,
            s_batched,
            losses_k,
            losses_k_plus_1,
            isodepth_k_plus_1,
            tested_dim,
            model_k=model_k,
            model_k_plus_1=model_k_plus_1,
        )
        step_summary["p_value"] = float(step_summary["p_value"])
        step_summary["significant"] = bool(float(step_summary["p_value"]) < config.alpha)
        step_summary["alpha"] = float(config.alpha)
        step_summary["test_type"] = "dimension_increase"
        step_summary["null_distribution"] = np.asarray(step_summary.pop("stat_perm"), dtype=np.float64)
        step_summary["observed_delta"] = float(step_summary["stat_true"])
        step_summary["dimension_labels"] = [f"d{i + 1}" for i in range(tested_dim)]
        step_summary["pred_true_k"] = np.asarray(predictions_k[0], dtype=np.float32)
        step_summary["pred_true_k_plus_1"] = np.asarray(predictions_k_plus_1[0], dtype=np.float32)
        step_summaries[str(tested_dim)] = step_summary

        dimension_plot_rows.append(
            {
                "tested_dim": int(tested_dim),
                "true_isodepth": step_summary["true_isodepth"],
                "lowest_isodepth": step_summary["lowest_isodepth"],
                "lowest_S": step_summary["lowest_S"],
                "lowest_stat": float(step_summary["lowest_stat"]),
                "highest_isodepth": step_summary["highest_isodepth"],
                "highest_S": step_summary["highest_S"],
                "highest_stat": float(step_summary["highest_stat"]),
                "dimension_labels": list(step_summary["dimension_labels"]),
                "p_value": float(step_summary["p_value"]),
                "significant": bool(step_summary["significant"]),
                "test_type": "dimension_increase",
            }
        )
        final_step_summary = step_summary
        if bool(step_summary["significant"]):
            selected_spatial_dims = tested_dim
            continue
        break

    if final_step_summary is None:
        raise RuntimeError("exact_existence did not evaluate any dimensions")

    runtime_sec = time.time() - start
    stat_true = float(
        final_step_summary["observed_delta"]
        if "observed_delta" in final_step_summary
        else final_step_summary["observed_stat"]
    )
    stat_perm = np.asarray(final_step_summary["null_distribution"], dtype=np.float64)
    lowest_isodepth = final_step_summary["lowest_isodepth"]
    highest_isodepth = final_step_summary["highest_isodepth"]
    true_isodepth = final_step_summary["true_isodepth"]

    return TestResult(
        method_name="exact_existence",
        metric=metric,
        p_value=float(final_step_summary["p_value"]),
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "selected_spatial_dims": int(selected_spatial_dims),
            "tested_spatial_dims": [int(value) for value in range(1, len(step_summaries) + 1)],
            "step_summaries": step_summaries,
            "dimension_plot_rows": dimension_plot_rows,
            "true_isodepth": np.asarray(true_isodepth, dtype=np.float32),
            "rerun_summary": dict(final_step_summary["rerun_summary"]),
            "true_rerun_index": int(final_step_summary["true_rerun_index"]),
            "true_train_loss": float(final_step_summary["true_train_loss"]),
            "lowest_isodepth": np.asarray(lowest_isodepth, dtype=np.float32),
            "lowest_S": np.asarray(final_step_summary["lowest_S"], dtype=np.float32),
            "lowest_stat": float(final_step_summary["lowest_stat"]),
            "lowest_perm_index": int(final_step_summary["lowest_perm_index"]),
            "lowest_rerun_index": int(final_step_summary["lowest_rerun_index"]),
            "lowest_train_loss": float(final_step_summary["lowest_train_loss"]),
            "highest_isodepth": np.asarray(highest_isodepth, dtype=np.float32),
            "highest_S": np.asarray(final_step_summary["highest_S"], dtype=np.float32),
            "highest_stat": float(final_step_summary["highest_stat"]),
            "highest_perm_index": int(final_step_summary["highest_perm_index"]),
            "highest_rerun_index": int(final_step_summary["highest_rerun_index"]),
            "highest_train_loss": float(final_step_summary["highest_train_loss"]),
            "null_summary": dict(final_step_summary["null_summary"]),
            "alpha": float(config.alpha),
            "max_spatial_dims": int(config.max_spatial_dims),
        },
    ).validate()
def run_full_retraining_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
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
    stat_true = compute_metric(metric, dataset.A, pred_true)
    true_isodepth = _extract_isodepth_from_model(true_model, dataset.S, device)
    true_rerun_index, true_train_loss = _rerun_index_and_loss(true_model, 0)

    rng = np.random.default_rng(config.seed)
    stat_perm = np.zeros(config.n_perms, dtype=np.float64)
    lowest_stat = None
    lowest_isodepth = None
    lowest_S = None
    lowest_rerun_index = 0
    lowest_train_loss = 0.0
    highest_stat = None
    highest_isodepth = None
    highest_S = None
    highest_rerun_index = 0
    highest_train_loss = 0.0
    for i in range(config.n_perms):
        perm = rng.permutation(dataset.n_cells)
        s_perm = dataset.S[perm]
        model_perm, pred_perm = train_isodepth_model(
            s_perm,
            dataset.A,
            config,
            device=device,
            seed_offset=i + 1,
            model_label=f"permuted model {i + 1}/{config.n_perms}",
        )
        stat_perm[i] = compute_metric(metric, dataset.A, pred_perm)
        current_isodepth = _extract_isodepth_from_model(model_perm, s_perm, device)
        current_rerun_index, current_train_loss = _rerun_index_and_loss(model_perm, 0)
        if lowest_stat is None or stat_perm[i] < lowest_stat:
            lowest_stat = float(stat_perm[i])
            lowest_isodepth = current_isodepth
            lowest_S = np.asarray(s_perm, dtype=np.float32)
            lowest_rerun_index = int(current_rerun_index)
            lowest_train_loss = float(current_train_loss)
        if highest_stat is None or stat_perm[i] > highest_stat:
            highest_stat = float(stat_perm[i])
            highest_isodepth = current_isodepth
            highest_S = np.asarray(s_perm, dtype=np.float32)
            highest_rerun_index = int(current_rerun_index)
            highest_train_loss = float(current_train_loss)

    runtime_sec = time.time() - start
    return TestResult(
        method_name="full_retraining",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=float(stat_true),
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "model": true_model,
            "pred_true": pred_true,
            "true_isodepth": true_isodepth,
            "rerun_summary": _rerun_summary(true_model),
            "true_rerun_index": int(true_rerun_index),
            "true_train_loss": float(true_train_loss),
            "lowest_isodepth": np.asarray(lowest_isodepth, dtype=np.float32),
            "lowest_S": np.asarray(lowest_S, dtype=np.float32),
            "lowest_stat": float(lowest_stat),
            "lowest_perm_index": int(np.argmin(stat_perm)),
            "lowest_rerun_index": int(lowest_rerun_index),
            "lowest_train_loss": float(lowest_train_loss),
            "highest_isodepth": np.asarray(highest_isodepth, dtype=np.float32),
            "highest_S": np.asarray(highest_S, dtype=np.float32),
            "highest_stat": float(highest_stat),
            "highest_perm_index": int(np.argmax(stat_perm)),
            "highest_rerun_index": int(highest_rerun_index),
            "highest_train_loss": float(highest_train_loss),
        },
    ).validate()
def run_permutation_method(dataset: DatasetBundle, config: TestConfig) -> TestResult:
    device = resolve_device(config.device)
    print(f"device: {device}")

    if config.method == "comparison_perturbation_test":
        return run_comparison_perturbation_test(dataset, config, device=device)
    if config.method == "perturbation_test":
        return run_perturbation_test(dataset, config, device=device)
    if config.method == "comparison_subsampling_test":
        return run_comparison_subsampling_test(dataset, config, device=device)
    if config.method == "subsampling_test":
        return run_subsampling_test(dataset, config, device=device)
    if config.method == "parallel_permutation":
        return run_parallel_permutation_method(dataset, config, device=device)
    if config.method == "exact_existence":
        return run_exact_existence_method(dataset, config, device=device)
    if config.method == "full_retraining":
        return run_full_retraining_method(dataset, config, device=device)
    raise ValueError(f"Unsupported test.method '{config.method}'")
