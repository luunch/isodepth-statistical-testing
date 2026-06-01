from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch
from scipy.stats import spearmanr

from data.schemas import DatasetBundle, TestConfig, TestResult
from data.transforms import zscore_covariate
from methods.metrics import (
    canonicalize_metric_name,
    compute_metric,
    metric_prefers_lower,
    permutation_p_value,
)
from methods.perturbation import run_comparison_perturbation_test, run_perturbation_test
from methods.subsampling import (
    compute_masked_losses,
    run_comparison_subsampling_test,
    run_subsampling_test,
)
from methods.trainers import (
    extract_celltype_model_isodepth,
    extract_model_isodepth,
    get_training_metadata,
    offload_module_to_cpu,
    resolve_device,
    run_with_cuda_oom_retry,
    train_celltype_parallel_isodepth_model,
    train_fixed_covariate_model,
    train_isodepth_model,
    train_parallel_isodepth_model,
)


def _covariate_type_midline(config: TestConfig) -> bool:
    cov = getattr(config, "covariate", None)
    return cov is not None and getattr(cov, "type", None) == "midline"


def _covariate_type_obs_key(config: TestConfig) -> bool:
    """True when the covariate is a labeled obs-column key (not midline, not None)."""
    cov = getattr(config, "covariate", None)
    return cov is not None and getattr(cov, "is_obs_key", False)


def _has_covariate(config: TestConfig) -> bool:
    cov = getattr(config, "covariate", None)
    return cov is not None and getattr(cov, "type", None) is not None


def _obs_covariate_values(dataset: DatasetBundle, config: TestConfig) -> np.ndarray:
    cov_values = dataset.meta.get("covariate_values")
    if cov_values is None:
        obs_key = config.covariate.type
        raise ValueError(
            f"test.covariate obs key '{obs_key}' was specified but "
            "dataset.meta['covariate_values'] is missing.  "
            "Ensure load_dataset is called with covariate=config.covariate so the "
            "obs column is extracted during data loading."
        )
    return np.asarray(cov_values, dtype=np.float32)


def _train_covariate_artifacts(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    device: torch.device,
    metric: str,
    stat_perm: np.ndarray,
    *,
    covariate_values: np.ndarray | None = None,
    model_label: str = "covariate decoder",
) -> dict[str, object]:
    """Train a decoder-only covariate model on a cell subset and compare to ``stat_perm``."""
    if _covariate_type_midline(config):
        covariate_model, pred_covariate = train_isodepth_model(
            S,
            A,
            config,
            device=device,
            model_label=model_label,
        )
        stat_covariate = float(compute_metric(metric, A, pred_covariate))
        return {
            "stat_covariate": stat_covariate,
            "p_value_covariate": float(permutation_p_value(metric, stat_covariate, stat_perm)),
            "pred_true_covariate": np.asarray(pred_covariate, dtype=np.float32),
            "true_isodepth_covariate": np.asarray(
                _extract_isodepth_from_model(covariate_model, S, device), dtype=np.float32
            ),
        }
    if _covariate_type_obs_key(config):
        if covariate_values is None:
            obs_key = config.covariate.type
            raise ValueError(
                f"test.covariate obs key '{obs_key}' was specified but covariate_values is missing."
            )
        _, pred_covariate = train_fixed_covariate_model(
            covariate_values,
            A,
            config,
            device=device,
            model_label=model_label,
        )
        stat_covariate = float(compute_metric(metric, A, pred_covariate))
        return {
            "stat_covariate": stat_covariate,
            "p_value_covariate": float(permutation_p_value(metric, stat_covariate, stat_perm)),
            "pred_true_covariate": np.asarray(pred_covariate, dtype=np.float32),
            "true_isodepth_covariate": zscore_covariate(covariate_values),
        }
    return {}


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


def _extract_slot_isodepths(
    model,
    s_batched_np: np.ndarray,
    slot_indices: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Extract isodepth only for the given slot indices.

    Uses ``extract_model_isodepth`` per unique index so each forward pass uses
    the correct parallel slot in the compact model.

    Returns ``{slot_index: isodepth_array}`` where each value has shape ``(N,)``
    for 1-D latent or ``(N, latent_dim)`` otherwise.
    """
    latent_dim = int(getattr(model, "latent_dim", 0))
    if latent_dim <= 0 or not hasattr(model, "encoder"):
        n_cells = s_batched_np.shape[1]
        return {idx: np.zeros((n_cells, 0), dtype=np.float32) for idx in slot_indices}

    result: dict[int, np.ndarray] = {}
    for idx in dict.fromkeys(slot_indices):
        iso = extract_model_isodepth(model, s_batched_np[idx], device, slot_index=idx)
        if iso.ndim == 2 and iso.shape[1] == 1:
            iso = np.asarray(iso[:, 0], dtype=np.float32)
        else:
            iso = np.asarray(iso, dtype=np.float32)
        result[idx] = iso
    return result


def _select_extreme_index(metric: str, stat_perm: np.ndarray) -> int:
    if metric_prefers_lower(metric):
        return int(np.argmin(stat_perm))
    return int(np.argmax(stat_perm))


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


def _build_train_test_masks(
    n_cells: int,
    *,
    n_models: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_size = int(round(float(train_fraction) * float(n_cells)))
    if train_size <= 0 or train_size >= n_cells:
        raise ValueError(
            "cross_validation requires at least one train and one test cell after rounding "
            f"test.train_fraction={float(train_fraction):.4g} for n_cells={int(n_cells)}"
        )

    rng = np.random.default_rng(seed)
    train_indices = rng.choice(n_cells, size=train_size, replace=False)
    train_mask = np.zeros((n_models, n_cells, 1), dtype=np.float32)
    train_mask[:, train_indices, 0] = 1.0
    test_mask = 1.0 - train_mask
    return train_mask, test_mask


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


def _compute_spearman_matrix(
    isodepth_vectors: list[np.ndarray],
    labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Pairwise Spearman correlation matrix for a list of equal-length isodepth vectors."""
    n = len(isodepth_vectors)
    matrix = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(isodepth_vectors[i], isodepth_vectors[j])
            matrix[i, j] = rho
            matrix[j, i] = rho
    return matrix, labels


def _celltype_indices_by_descending_cell_count(
    cell_type_labels: np.ndarray,
    n_cell_types: int,
) -> list[int]:
    """Original cell-type indices sorted by within-type cell count (largest first)."""
    counts = [(c, int((cell_type_labels == c).sum())) for c in range(n_cell_types)]
    counts.sort(key=lambda item: item[1], reverse=True)
    return [c for c, _ in counts]


def _process_single_celltype_separate(
    dataset: DatasetBundle,
    config: TestConfig,
    device: torch.device,
    *,
    type_index: int,
    type_name: str,
    cell_type_labels: np.ndarray,
    metric: str,
    covariate_values: np.ndarray | None = None,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    """Train/evaluate one cell type; returns (per-type result dict, coord standardization)."""
    mask = cell_type_labels == type_index
    S_original_c = np.asarray(dataset.S[mask], dtype=np.float32)
    A_c = dataset.A[mask]
    n_c = int(mask.sum())

    mean_c = S_original_c.mean(axis=0)
    std_c = S_original_c.std(axis=0)
    safe_std_c = np.where(std_c > 1e-8, std_c, 1.0)
    S_c = np.asarray((S_original_c - mean_c) / safe_std_c, dtype=np.float32)

    type_config = replace(config, seed=config.seed + type_index)
    parallel_config = replace(type_config, covariate=None) if _has_covariate(config) else type_config
    model_c, training_outputs_c, s_batched_np_c = train_parallel_isodepth_model(
        S_c,
        A_c,
        parallel_config,
        device=device,
        model_label=f"separate {type_name} ({n_c} cells)",
    )
    stat_true_c = float(training_outputs_c.stat_true)
    stat_perm_c = training_outputs_c.stat_perm
    p_value_c = permutation_p_value(metric, stat_true_c, stat_perm_c)

    covariate_artifacts: dict[str, object] = {}
    if _has_covariate(config):
        cov_values_c = None
        if covariate_values is not None:
            cov_values_c = np.asarray(covariate_values[mask], dtype=np.float32)
        cov_label = config.covariate.type if config.covariate is not None else "covariate"
        covariate_artifacts = _train_covariate_artifacts(
            S_c,
            A_c,
            type_config,
            device,
            metric,
            stat_perm_c,
            covariate_values=cov_values_c,
            model_label=f"{cov_label} covariate decoder ({type_name})",
        )

    low_idx = int(training_outputs_c.best_null_index)
    high_idx = int(training_outputs_c.worst_null_index)
    slot_iso = _extract_slot_isodepths(
        model_c, s_batched_np_c, [0, low_idx + 1, high_idx + 1], device,
    )
    true_rerun_index, true_train_loss = _rerun_index_and_loss(model_c, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(model_c, low_idx + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(model_c, high_idx + 1)

    type_result = {
        "p_value": p_value_c,
        "stat_true": stat_true_c,
        "stat_perm": stat_perm_c,
        "n_cells": n_c,
        "model": model_c,
        "pred_true": np.asarray(training_outputs_c.pred_true, dtype=np.float32),
        "true_isodepth": np.asarray(slot_iso[0], dtype=np.float32),
        "rerun_summary": _rerun_summary(model_c),
        "true_rerun_index": int(true_rerun_index),
        "true_train_loss": float(true_train_loss),
        "lowest_isodepth": np.asarray(slot_iso[low_idx + 1], dtype=np.float32),
        "lowest_S": np.asarray(s_batched_np_c[low_idx + 1], dtype=np.float32),
        "lowest_stat": float(stat_perm_c[low_idx]),
        "lowest_perm_index": int(low_idx),
        "lowest_rerun_index": int(lowest_rerun_index),
        "lowest_train_loss": float(lowest_train_loss),
        "highest_isodepth": np.asarray(slot_iso[high_idx + 1], dtype=np.float32),
        "highest_S": np.asarray(s_batched_np_c[high_idx + 1], dtype=np.float32),
        "highest_stat": float(stat_perm_c[high_idx]),
        "highest_perm_index": int(high_idx),
        "highest_rerun_index": int(highest_rerun_index),
        "highest_train_loss": float(highest_train_loss),
        "S": S_c,
        "S_original": S_original_c,
        "A": A_c,
        **covariate_artifacts,
    }
    del s_batched_np_c
    type_result["model"] = offload_module_to_cpu(model_c)
    return type_result, (mean_c, safe_std_c)


def _run_celltype_separate_parallel_permutation(
    dataset: DatasetBundle, config: TestConfig, device: torch.device
) -> TestResult:
    """Per-cell-type independent isodepth models with within-type permutations."""
    metric = canonicalize_metric_name(config.metric)
    cell_type_labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    cell_type_names: list[str] = list(dataset.meta["cell_type_names"])
    n_cell_types = int(dataset.meta["n_cell_types"])
    type_order = _celltype_indices_by_descending_cell_count(cell_type_labels, n_cell_types)
    covariate_values = dataset.meta.get("covariate_values")
    if _covariate_type_obs_key(config) and covariate_values is None:
        obs_key = config.covariate.type
        raise ValueError(
            f"test.covariate obs key '{obs_key}' was specified but "
            "dataset.meta['covariate_values'] is missing.  "
            "Ensure load_dataset is called with covariate=config.covariate."
        )
    if covariate_values is not None:
        covariate_values = np.asarray(covariate_values, dtype=np.float32)

    start = time.time()
    per_type_results: dict[str, dict] = {}
    per_type_standardization: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    current_device = device

    for step_idx, type_index in enumerate(type_order):
        type_name = cell_type_names[type_index]
        n_c = int((cell_type_labels == type_index).sum())
        print(
            f"Cell type {step_idx + 1}/{n_cell_types}: {type_name} ({n_c} cells)",
            flush=True,
        )

        used_device = current_device

        def _train_current_type(train_device: torch.device) -> dict:
            nonlocal used_device
            used_device = train_device
            type_result, standardization = _process_single_celltype_separate(
                dataset,
                config,
                train_device,
                type_index=type_index,
                type_name=type_name,
                cell_type_labels=cell_type_labels,
                metric=metric,
                covariate_values=covariate_values,
            )
            per_type_standardization[type_name] = standardization
            return type_result

        per_type_results[type_name] = run_with_cuda_oom_retry(
            _train_current_type,
            current_device,
            label=f"cell type '{type_name}'",
        )
        current_device = used_device

    # --- Spearman correlation matrix across cell types ---
    # Evaluate each per-type model on the full dataset for comparable isodepth vectors.
    full_isodepths: list[np.ndarray] = []
    spearman_labels: list[str] = []

    for type_name in cell_type_names:
        mean_c, safe_std_c = per_type_standardization[type_name]
        S_full_standardized = np.asarray(
            (dataset.S - mean_c) / safe_std_c, dtype=np.float32,
        )
        model_c = per_type_results[type_name]["model"]
        iso_full = _extract_isodepth_from_model(model_c, S_full_standardized, device)
        full_isodepths.append(iso_full.reshape(-1))
        spearman_labels.append(type_name)

    spearman_matrix, spearman_labels = _compute_spearman_matrix(
        full_isodepths, spearman_labels,
    )

    runtime_sec = time.time() - start

    all_stat_true = [per_type_results[n]["stat_true"] for n in cell_type_names]
    all_stat_perm = np.concatenate([per_type_results[n]["stat_perm"] for n in cell_type_names])
    combined_stat_true = float(np.mean(all_stat_true))
    combined_p = float(np.mean([per_type_results[n]["p_value"] for n in cell_type_names]))

    return TestResult(
        method_name="parallel_permutation",
        metric=metric,
        p_value=combined_p,
        stat_true=combined_stat_true,
        stat_perm=all_stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "per_type_results": per_type_results,
            "cell_type_names": cell_type_names,
            "cell_type_labels": cell_type_labels,
            "n_cell_types": n_cell_types,
            "cell_type_mode": "separate",
            "spearman_matrix": spearman_matrix,
            "spearman_labels": spearman_labels,
        },
    ).validate()


def _run_celltype_parallel_permutation(
    dataset: DatasetBundle, config: TestConfig, device: torch.device
) -> TestResult:
    """Cell-type-specific variant: shared encoder + per-cell-type decoders."""
    metric = canonicalize_metric_name(config.metric)
    cell_type_labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    n_cell_types = int(dataset.meta["n_cell_types"])

    start = time.time()
    has_midline_covariate = _covariate_type_midline(config)
    has_obs_key_covariate = _covariate_type_obs_key(config)
    parallel_config = replace(config, covariate=None) if (has_midline_covariate or has_obs_key_covariate) else config
    model, training_outputs, s_batched_np = train_celltype_parallel_isodepth_model(
        dataset.S,
        dataset.A,
        parallel_config,
        cell_type_labels=cell_type_labels,
        n_cell_types=n_cell_types,
        device=device,
    )
    stat_true = float(training_outputs.stat_true)
    stat_perm = training_outputs.stat_perm
    p_value = permutation_p_value(metric, stat_true, stat_perm)
    if _has_covariate(config):
        cov_values = (
            _obs_covariate_values(dataset, config) if has_obs_key_covariate else None
        )
        covariate_artifacts = _train_covariate_artifacts(
            dataset.S,
            dataset.A,
            config,
            device,
            metric,
            stat_perm,
            covariate_values=cov_values,
            model_label="true layout covariate decoder (cell-type parallel)",
        )
    else:
        covariate_artifacts = {}

    low_idx = int(training_outputs.best_null_index)
    high_idx = int(training_outputs.worst_null_index)

    isodepth_shared = extract_celltype_model_isodepth(model, dataset.S, device)
    true_rerun_index, true_train_loss = _rerun_index_and_loss(model, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(model, low_idx + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(model, high_idx + 1)

    lowest_S_np = np.asarray(s_batched_np[low_idx + 1], dtype=np.float32)
    highest_S_np = np.asarray(s_batched_np[high_idx + 1], dtype=np.float32)

    lowest_isodepth = extract_celltype_model_isodepth(
        model,
        lowest_S_np,
        device,
        slot_index=low_idx + 1,
    )
    highest_isodepth = extract_celltype_model_isodepth(
        model,
        highest_S_np,
        device,
        slot_index=high_idx + 1,
    )

    runtime_sec = time.time() - start

    return TestResult(
        method_name="parallel_permutation",
        metric=metric,
        p_value=p_value,
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "model": model,
            "pred_true": np.asarray(training_outputs.pred_true, dtype=np.float32),
            "true_isodepth": np.asarray(isodepth_shared[:, 0], dtype=np.float32)
            if isodepth_shared.shape[1] == 1
            else np.asarray(isodepth_shared, dtype=np.float32),
            "rerun_summary": _rerun_summary(model),
            "true_rerun_index": int(true_rerun_index),
            "true_train_loss": float(true_train_loss),
            "lowest_isodepth": np.asarray(lowest_isodepth[:, 0], dtype=np.float32)
            if lowest_isodepth.shape[1] == 1
            else np.asarray(lowest_isodepth, dtype=np.float32),
            "lowest_S": lowest_S_np,
            "lowest_stat": float(stat_perm[low_idx]) if stat_perm.size else float("nan"),
            "lowest_perm_index": low_idx,
            "lowest_rerun_index": int(lowest_rerun_index),
            "lowest_train_loss": float(lowest_train_loss),
            "highest_isodepth": np.asarray(highest_isodepth[:, 0], dtype=np.float32)
            if highest_isodepth.shape[1] == 1
            else np.asarray(highest_isodepth, dtype=np.float32),
            "highest_S": highest_S_np,
            "highest_stat": float(stat_perm[high_idx]) if stat_perm.size else float("nan"),
            "highest_perm_index": high_idx,
            "highest_rerun_index": int(highest_rerun_index),
            "highest_train_loss": float(highest_train_loss),
            "cell_type_labels": cell_type_labels,
            "cell_type_names": dataset.meta["cell_type_names"],
            "n_cell_types": n_cell_types,
            **covariate_artifacts,
        },
    ).validate()


def run_parallel_permutation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    if dataset.meta.get("cell_type_mode") == "separate":
        return _run_celltype_separate_parallel_permutation(dataset, config, device)
    if dataset.meta.get("cell_type_labels") is not None:
        return _run_celltype_parallel_permutation(dataset, config, device)

    start = time.time()
    has_midline_covariate = _covariate_type_midline(config)
    has_obs_key_covariate = _covariate_type_obs_key(config)
    parallel_config = replace(config, covariate=None) if (has_midline_covariate or has_obs_key_covariate) else config
    model, training_outputs, s_batched_np = train_parallel_isodepth_model(
        dataset.S, dataset.A, parallel_config, device=device,
    )
    stat_true = float(training_outputs.stat_true)
    stat_perm = training_outputs.stat_perm
    p_value = permutation_p_value(metric, stat_true, stat_perm)
    if _has_covariate(config):
        cov_values = (
            _obs_covariate_values(dataset, config) if has_obs_key_covariate else None
        )
        covariate_artifacts = _train_covariate_artifacts(
            dataset.S,
            dataset.A,
            config,
            device,
            metric,
            stat_perm,
            covariate_values=cov_values,
            model_label=(
                f"obs-key covariate decoder ({config.covariate.type})"
                if has_obs_key_covariate
                else "true layout covariate decoder"
            ),
        )
    else:
        covariate_artifacts = {}
    low_idx = int(training_outputs.best_null_index)
    high_idx = int(training_outputs.worst_null_index)
    slot_iso = _extract_slot_isodepths(
        model, s_batched_np, [0, low_idx + 1, high_idx + 1], device,
    )
    true_rerun_index, true_train_loss = _rerun_index_and_loss(model, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(model, low_idx + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(model, high_idx + 1)
    runtime_sec = time.time() - start

    return TestResult(
        method_name="parallel_permutation",
        metric=metric,
        p_value=p_value,
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "model": model,
            "pred_true": np.asarray(training_outputs.pred_true, dtype=np.float32),
            "true_isodepth": np.asarray(slot_iso[0], dtype=np.float32),
            "rerun_summary": _rerun_summary(model),
            "true_rerun_index": int(true_rerun_index),
            "true_train_loss": float(true_train_loss),
            "lowest_isodepth": np.asarray(slot_iso[low_idx + 1], dtype=np.float32),
            "lowest_S": np.asarray(s_batched_np[low_idx + 1], dtype=np.float32),
            "lowest_stat": float(stat_perm[low_idx]),
            "lowest_perm_index": low_idx,
            "lowest_rerun_index": int(lowest_rerun_index),
            "lowest_train_loss": float(lowest_train_loss),
            "highest_isodepth": np.asarray(slot_iso[high_idx + 1], dtype=np.float32),
            "highest_S": np.asarray(s_batched_np[high_idx + 1], dtype=np.float32),
            "highest_stat": float(stat_perm[high_idx]),
            "highest_perm_index": high_idx,
            "highest_rerun_index": int(highest_rerun_index),
            "highest_train_loss": float(highest_train_loss),
            **covariate_artifacts,
        },
    ).validate()


def run_cross_validation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()
    n_models = config.n_perms + 1
    s_batched_pre, _ = _build_permuted_coordinate_batch(
        dataset.S,
        n_perms=config.n_perms,
        seed=config.seed,
        device=device,
    )
    s_batched_pre_np = np.asarray(s_batched_pre.detach().cpu().numpy(), dtype=np.float32)
    del s_batched_pre
    train_mask_batched, test_mask_batched = _build_train_test_masks(
        dataset.n_cells,
        n_models=n_models,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    model, training_outputs, s_batched_np = train_parallel_isodepth_model(
        dataset.S,
        dataset.A,
        config,
        device=device,
        s_batched=s_batched_pre_np,
        loss_mask_batched=train_mask_batched,
        metric_loss_mask_batched=test_mask_batched,
        model_label=f"cross validation batch (true + {config.n_perms} permuted models)",
    )
    stat_true = float(training_outputs.stat_true)
    stat_perm = training_outputs.stat_perm
    low_idx = int(training_outputs.best_null_index)
    high_idx = int(training_outputs.worst_null_index)
    slot_iso = _extract_slot_isodepths(
        model, s_batched_np, [0, low_idx + 1, high_idx + 1], device,
    )
    true_rerun_index, true_train_loss = _rerun_index_and_loss(model, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(model, low_idx + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(model, high_idx + 1)
    runtime_sec = time.time() - start

    return TestResult(
        method_name="cross_validation",
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
            "pred_true": np.asarray(training_outputs.pred_true, dtype=np.float32),
            "true_isodepth": np.asarray(slot_iso[0], dtype=np.float32),
            "rerun_summary": _rerun_summary(model),
            "true_rerun_index": int(true_rerun_index),
            "true_train_loss": float(true_train_loss),
            "lowest_isodepth": np.asarray(slot_iso[low_idx + 1], dtype=np.float32),
            "lowest_S": np.asarray(s_batched_np[low_idx + 1], dtype=np.float32),
            "lowest_stat": float(stat_perm[low_idx]),
            "lowest_perm_index": low_idx,
            "lowest_rerun_index": int(lowest_rerun_index),
            "lowest_train_loss": float(lowest_train_loss),
            "highest_isodepth": np.asarray(slot_iso[high_idx + 1], dtype=np.float32),
            "highest_S": np.asarray(s_batched_np[high_idx + 1], dtype=np.float32),
            "highest_stat": float(stat_perm[high_idx]),
            "highest_perm_index": high_idx,
            "highest_rerun_index": int(highest_rerun_index),
            "highest_train_loss": float(highest_train_loss),
            "held_out_losses": np.asarray(training_outputs.model_metrics, dtype=np.float64),
            "null_summary": {
                "mean": float(np.mean(stat_perm)),
                "std": float(np.std(stat_perm)),
                "min": float(np.min(stat_perm)),
                "max": float(np.max(stat_perm)),
            },
            "train_mask": np.asarray(train_mask_batched[0, :, 0], dtype=np.float32),
            "test_mask": np.asarray(test_mask_batched[0, :, 0], dtype=np.float32),
            "train_fraction": float(config.train_fraction),
            "test_fraction": float(1.0 - config.train_fraction),
            "train_size": int(np.sum(train_mask_batched[0, :, 0] > 0)),
            "test_size": int(np.sum(test_mask_batched[0, :, 0] > 0)),
            "observed_test_loss": float(stat_true),
        },
    ).validate()


def run_full_retraining_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    _has_midline = _covariate_type_midline(config)
    _has_obs_key = _covariate_type_obs_key(config)
    config_full = replace(config, covariate=None) if (_has_midline or _has_obs_key) else config

    start = time.time()
    true_model, pred_true = train_isodepth_model(
        dataset.S,
        dataset.A,
        config_full,
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
            config_full,
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
    p_value = permutation_p_value(metric, stat_true, stat_perm)
    if _has_midline or _has_obs_key:
        cov_values = _obs_covariate_values(dataset, config) if _has_obs_key else None
        covariate_artifacts = _train_covariate_artifacts(
            dataset.S,
            dataset.A,
            config,
            device,
            metric,
            stat_perm,
            covariate_values=cov_values,
            model_label=(
                f"obs-key covariate decoder ({config.covariate.type})"
                if _has_obs_key
                else "true model midline covariate"
            ),
        )
    else:
        covariate_artifacts = {}

    return TestResult(
        method_name="full_retraining",
        metric=metric,
        p_value=p_value,
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
            **covariate_artifacts,
        },
    ).validate()
def _run_permutation_method_on_device(
    dataset: DatasetBundle, config: TestConfig, device: torch.device
) -> TestResult:
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
    if config.method == "cross_validation":
        return run_cross_validation_method(dataset, config, device=device)
    if config.method == "full_retraining":
        return run_full_retraining_method(dataset, config, device=device)
    raise ValueError(f"Unsupported test.method '{config.method}'")


def run_permutation_method(dataset: DatasetBundle, config: TestConfig) -> TestResult:
    device = resolve_device(config.device)
    print(f"device: {device}")

    dispatch = lambda resolved_device: _run_permutation_method_on_device(  # noqa: E731
        dataset, config, resolved_device
    )
    # Separate cell-type mode retries OOM per cell type (and for the together model).
    if (
        config.method == "parallel_permutation"
        and dataset.meta.get("cell_type_mode") == "separate"
    ):
        return dispatch(device)
    return run_with_cuda_oom_retry(dispatch, device, label=config.method)
