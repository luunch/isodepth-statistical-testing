from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch
from scipy.stats import spearmanr

from data.h5ad_loader import preprocess_celltype_subset
from data.schemas import DatasetBundle, TestConfig, TestResult
from data.transforms import celltype_expression_residuals, midline_latent, zscore_covariate
from data import raw_coordinates_from_standardized, standardize_coordinate_batch
from methods.metrics import (
    canonicalize_metric_name,
    compute_metric,
    metric_prefers_lower,
    permutation_p_value,
)
from methods.block_permutation import (
    block_stats,
    build_block_permuted_coordinate_batch,
    hex_bin_ids,
    resolve_um_per_unit,
)
from methods.perturbation import run_comparison_perturbation_test, run_perturbation_test
from methods.subsampling import (
    compute_masked_losses,
    run_comparison_subsampling_test,
    run_subsampling_test,
)
from methods.trainers import (
    covariate_decoder_is_closed_form,
    extract_celltype_model_isodepth,
    extract_model_isodepth,
    fit_closed_form_decoder,
    fit_poisson_glm_irls,
    get_training_metadata,
    poisson_parametric_decoder_uses_irls,
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
    poisson_size_factors_np: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit a decoder-only covariate model on a cell subset and compare to ``stat_perm``.

    The covariate latent is fixed (midline depth or an obs column), so for linear and
    quadratic decoders the optimal decoder is the closed-form OLS fit and no neural
    network training is needed; only the ``nn`` decoder still trains iteratively.

    For Poisson NLL (``nll_poisson_mse``), the closed-form OLS is fitted on log-rate
    targets ``log(A / sf)`` so predictions live on the log-rate scale.  Evaluation uses
    ``poisson_size_factors_np`` (or row-sum defaults) to keep the metric comparable with
    the main isodepth model.
    """
    is_poisson = canonicalize_metric_name(metric) == "nll_poisson_mse"
    closed_form = covariate_decoder_is_closed_form(config)
    decoder_type = str(getattr(config, "decoder", "nn"))

    # For Poisson, the fixed-latent covariate is an exact Poisson GLM (log link +
    # exposure offset), fitted to its maximum-likelihood optimum with IRLS.  This
    # replaces the old gradient-descent path, which could diverge on small/sparse
    # cell-type subsets: the fixed, right-skewed midline latent fed through z² and the
    # exp() link produced exploding rates, leaving stat_covariate far above the
    # per-gene-mean floor (sometimes above its own initialization loss).  IRLS solves
    # the same model/objective stably.  Gaussian linear/quadratic still uses OLS;
    # ``decoder="nn"`` still trains iteratively.
    use_poisson_irls = poisson_parametric_decoder_uses_irls(config)
    if is_poisson:
        closed_form = False

    if _covariate_type_midline(config):
        if closed_form:
            isodepth_covariate = midline_latent(S)
            pred_covariate = fit_closed_form_decoder(isodepth_covariate, A, decoder_type)
        elif use_poisson_irls:
            isodepth_covariate = midline_latent(S)
            pred_covariate = fit_poisson_glm_irls(
                isodepth_covariate, A, decoder_type, size_factors=poisson_size_factors_np
            )
        else:
            # The midline covariate is a FIXED |x - median(x)| baseline.  It must not
            # inherit the main model's `encoder="midline"` restriction, otherwise the
            # covariate is built as a learnable midline encoder (priority branch in
            # train_batched_isodepth_model) and collapses onto the main model instead of
            # staying fixed.  Force encoder="mlp" so the fixed-midline covariate
            # (HybridMidlineParallelNet) is constructed.
            fixed_midline_config = replace(config, encoder="mlp")
            covariate_model, pred_covariate = train_isodepth_model(
                S,
                A,
                fixed_midline_config,
                device=device,
                model_label=model_label,
                poisson_size_factors_override=poisson_size_factors_np,
            )
            isodepth_covariate = _extract_isodepth_from_model(covariate_model, S, device)
        stat_covariate = float(compute_metric(
            metric, A, pred_covariate, poisson_size_factors=poisson_size_factors_np
        ))
        return {
            "stat_covariate": stat_covariate,
            "p_value_covariate": float(permutation_p_value(metric, stat_covariate, stat_perm)),
            "pred_true_covariate": np.asarray(pred_covariate, dtype=np.float32),
            "true_isodepth_covariate": np.asarray(isodepth_covariate, dtype=np.float32),
        }
    if _covariate_type_obs_key(config):
        if covariate_values is None:
            obs_key = config.covariate.type
            raise ValueError(
                f"test.covariate obs key '{obs_key}' was specified but covariate_values is missing."
            )
        latent_covariate = zscore_covariate(covariate_values)
        if closed_form:
            pred_covariate = fit_closed_form_decoder(latent_covariate, A, decoder_type)
        elif use_poisson_irls:
            pred_covariate = fit_poisson_glm_irls(
                latent_covariate, A, decoder_type, size_factors=poisson_size_factors_np
            )
        else:
            _, pred_covariate = train_fixed_covariate_model(
                covariate_values,
                A,
                config,
                device=device,
                model_label=model_label,
                poisson_size_factors=poisson_size_factors_np,
            )
        stat_covariate = float(compute_metric(
            metric, A, pred_covariate, poisson_size_factors=poisson_size_factors_np
        ))
        return {
            "stat_covariate": stat_covariate,
            "p_value_covariate": float(permutation_p_value(metric, stat_covariate, stat_perm)),
            "pred_true_covariate": np.asarray(pred_covariate, dtype=np.float32),
            "true_isodepth_covariate": latent_covariate,
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


def global_coordinate_permute_slot(
    S: np.ndarray,
    *,
    seed: int,
    slot: int = 1,
) -> np.ndarray:
    """Return globally shuffled coordinates for null slot ``slot`` (1-indexed).

    Matches the shuffle order used by ``_build_permuted_coordinate_batch`` for the
    same ``seed`` and ``slot``.
    """
    if slot < 1:
        raise ValueError(f"slot must be >= 1, got {slot}")
    S = np.asarray(S, dtype=np.float32)
    if S.ndim != 2 or S.shape[1] != 2:
        raise ValueError(f"S must be (n_cells, 2), got {S.shape}")
    s_t = torch.tensor(S, dtype=torch.float32, device=torch.device("cpu"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(S.shape[0], generator=generator)
    for _ in range(2, slot + 1):
        perm = torch.randperm(S.shape[0], generator=generator)
    return np.asarray(s_t[perm.numpy()].numpy(), dtype=np.float32)


def _validate_cross_validation_folds(n_cells: int, n_folds: int) -> None:
    if n_folds < 2:
        raise ValueError("cross_validation requires test.n_folds >= 2")
    if n_folds > n_cells:
        raise ValueError(
            f"cross_validation requires test.n_folds <= n_cells, got n_folds={n_folds} "
            f"for n_cells={n_cells}"
        )
    max_test_size = int(np.ceil(float(n_cells) / float(n_folds)))
    if n_cells - max_test_size < 1:
        raise ValueError(
            "cross_validation requires at least one train cell per fold for "
            f"n_cells={n_cells} and n_folds={n_folds}"
        )


def _build_kfold_assignments(
    n_cells: int,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-cell fold ids (0..k-1) and fold test sizes."""
    _validate_cross_validation_folds(n_cells, n_folds)
    rng = np.random.default_rng(seed)
    indices = np.arange(n_cells, dtype=np.int64)
    rng.shuffle(indices)

    fold_ids = np.zeros(n_cells, dtype=np.int64)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    base_size = n_cells // n_folds
    remainder = n_cells % n_folds
    pos = 0
    for fold_index in range(n_folds):
        size = base_size + (1 if fold_index < remainder else 0)
        fold_sizes[fold_index] = size
        fold_ids[indices[pos : pos + size]] = fold_index
        pos += size
    return fold_ids, fold_sizes


def _fold_train_test_masks(
    n_cells: int,
    n_models: int,
    fold_id: int,
    cell_fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    test_1d = (cell_fold_ids == fold_id).astype(np.float32)
    train_1d = 1.0 - test_1d
    if train_1d.sum() <= 0 or test_1d.sum() <= 0:
        raise ValueError(
            f"cross_validation fold {fold_id} produced empty train or test mask "
            f"for n_cells={n_cells}"
        )
    train_mask = np.zeros((n_models, n_cells, 1), dtype=np.float32)
    test_mask = np.zeros((n_models, n_cells, 1), dtype=np.float32)
    train_mask[:, :, 0] = train_1d
    test_mask[:, :, 0] = test_1d
    return train_mask, test_mask


def _aggregate_weighted_fold_losses(
    fold_true_losses: list[float],
    fold_perm_losses: list[np.ndarray],
    fold_weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    weights = np.asarray(fold_weights, dtype=np.float64)
    if weights.shape[0] != len(fold_true_losses):
        raise ValueError("fold_weights must match number of folds")
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("fold_weights must sum to a positive value")
    weights = weights / weight_sum
    true_values = np.asarray(fold_true_losses, dtype=np.float64)
    perm_matrix = np.stack([np.asarray(values, dtype=np.float64) for values in fold_perm_losses], axis=0)
    stat_true = float(np.sum(weights * true_values))
    stat_perm = np.sum(weights[:, None] * perm_matrix, axis=0)
    return stat_true, stat_perm


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


def _preprocess_separate_subset(
    dataset: DatasetBundle, A_c_raw: np.ndarray, type_index: int
) -> tuple[np.ndarray, list[str] | None, str | None]:
    """Apply per-cell-type expression preprocessing for separate mode.

    When the loader deferred preprocessing (``cell_type="separate"``), HVG
    selection, gene filtering and z-scoring are computed *within this cell type
    only*.  Falls back to the already-transformed matrix for datasets that did not
    defer (e.g. synthetic bundles or legacy callers).
    """
    A_c_raw = np.asarray(A_c_raw, dtype=np.float32)
    pp = dataset.meta.get("separate_preprocessing")
    if not pp:
        return (
            A_c_raw,
            dataset.meta.get("var_names"),
            dataset.meta.get("feature_space"),
        )
    params = {k: v for k, v in pp.items() if k != "seed"}
    seed = int(pp.get("seed", 0)) + int(type_index)
    return preprocess_celltype_subset(
        A_c_raw,
        dataset.meta.get("var_names"),
        seed=seed,
        **params,
    )


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
    A_c, var_names_c, feature_space_c = _preprocess_separate_subset(
        dataset, dataset.A[mask], type_index
    )
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

    # Separate mode trains a plain (non-cell-type) model per subset, which for
    # Poisson uses row-sum size factors N_i = Σ_g A_ig.  Model output is log-rate;
    # convert to expected counts (sf · exp(log_rate)) so gene-expression plots and
    # F-tests see predictions on the same scale as the raw expression.
    pred_true_c = np.asarray(training_outputs_c.pred_true, dtype=np.float32)
    if canonicalize_metric_name(metric) == "nll_poisson_mse":
        sf_row_c = np.asarray(A_c, dtype=np.float32).sum(axis=1, keepdims=True)
        pred_true_c = (sf_row_c * np.exp(pred_true_c)).astype(np.float32)
        pred_cov_c = covariate_artifacts.get("pred_true_covariate")
        if pred_cov_c is not None:
            covariate_artifacts["pred_true_covariate"] = (
                sf_row_c * np.exp(np.asarray(pred_cov_c, dtype=np.float32))
            ).astype(np.float32)

    type_result = {
        "p_value": p_value_c,
        "stat_true": stat_true_c,
        "stat_perm": stat_perm_c,
        "n_cells": n_c,
        "model": model_c,
        "pred_true": pred_true_c,
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
        "var_names": var_names_c,
        "feature_space": feature_space_c,
        "n_genes": int(np.asarray(A_c).shape[1]),
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

    is_poisson_ct = canonicalize_metric_name(metric) == "nll_poisson_mse"
    if is_poisson_ct:
        # For Poisson, train covariate on raw counts with the same per-gene
        # cell-type mean size factors used by the main isodepth model.
        expression_for_covariate = np.asarray(dataset.A, dtype=np.float32)
        ct_means = np.zeros((n_cell_types, dataset.A.shape[1]), dtype=np.float32)
        for ct in range(n_cell_types):
            mask = cell_type_labels == ct
            if mask.any():
                ct_means[ct] = expression_for_covariate[mask].mean(axis=0)
        sf_ct_np: np.ndarray | None = np.maximum(ct_means[cell_type_labels], 1e-3)
    else:
        expression_for_covariate = celltype_expression_residuals(
            dataset.A,
            cell_type_labels,
            n_cell_types=n_cell_types,
        )
        sf_ct_np = None

    if _has_covariate(config):
        cov_values = (
            _obs_covariate_values(dataset, config) if has_obs_key_covariate else None
        )
        covariate_artifacts = _train_covariate_artifacts(
            dataset.S,
            expression_for_covariate,
            config,
            device,
            metric,
            stat_perm,
            covariate_values=cov_values,
            model_label="true layout covariate decoder (cell-type parallel)",
            poisson_size_factors_np=sf_ct_np,
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

    # For Poisson, decoder output is log-rate.  Downstream code (gene-expression
    # plots, F-tests, residual ratios) expects predictions on the expression scale.
    # Convert: E[a] = sf * exp(log_rate).
    pred_true_np = np.asarray(training_outputs.pred_true, dtype=np.float32)
    if is_poisson_ct and sf_ct_np is not None:
        pred_true_np = (sf_ct_np * np.exp(pred_true_np)).astype(np.float32)
        pred_cov = covariate_artifacts.get("pred_true_covariate")
        if pred_cov is not None:
            covariate_artifacts["pred_true_covariate"] = (
                sf_ct_np * np.exp(np.asarray(pred_cov, dtype=np.float32))
            ).astype(np.float32)

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
            "pred_true": pred_true_np,
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

    # Convert Poisson log-rate predictions to expected counts for downstream plots.
    pred_true_np = np.asarray(training_outputs.pred_true, dtype=np.float32)
    is_poisson_plain = canonicalize_metric_name(metric) == "nll_poisson_mse"
    if is_poisson_plain:
        A_f = np.asarray(dataset.A, dtype=np.float32)
        sf_row = A_f.sum(axis=1, keepdims=True)  # (N, 1)
        pred_true_np = (sf_row * np.exp(pred_true_np)).astype(np.float32)
        pred_cov = covariate_artifacts.get("pred_true_covariate")
        if pred_cov is not None:
            covariate_artifacts["pred_true_covariate"] = (
                sf_row * np.exp(np.asarray(pred_cov, dtype=np.float32))
            ).astype(np.float32)

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
            "pred_true": pred_true_np,
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


def _cross_validation_fold_weights(fold_test_sizes: np.ndarray, n_cells: int) -> np.ndarray:
    return np.asarray(fold_test_sizes, dtype=np.float64) / float(n_cells)


def _per_fold_p_values(
    metric: str,
    fold_true_losses: list[float],
    fold_perm_losses: list[np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            permutation_p_value(metric, float(true_loss), np.asarray(perm_losses, dtype=np.float64))
            for true_loss, perm_losses in zip(fold_true_losses, fold_perm_losses)
        ],
        dtype=np.float64,
    )


def _extract_fold_true_isodepth_plain(model, S: np.ndarray, device: torch.device) -> np.ndarray:
    return _format_isodepth_for_artifact(extract_model_isodepth(model, S, device, slot_index=0))


def _extract_fold_true_isodepth_celltype(model, S: np.ndarray, device: torch.device) -> np.ndarray:
    return _format_isodepth_for_artifact(
        extract_celltype_model_isodepth(model, S, device, slot_index=0)
    )


def _cross_validation_common_artifacts(
    *,
    config: TestConfig,
    metric: str,
    stat_true: float,
    stat_perm: np.ndarray,
    fold_true_losses: list[float],
    fold_perm_losses: list[np.ndarray],
    fold_test_sizes: np.ndarray,
    fold_weights: np.ndarray,
    fold_true_isodepths: list[np.ndarray],
    per_fold_p_values: np.ndarray,
    primary_training_outputs,
    primary_model,
    primary_s_batched_np: np.ndarray,
    primary_train_mask: np.ndarray,
    primary_test_mask: np.ndarray,
    device: torch.device,
    covariate_artifacts: dict[str, object] | None = None,
    pred_true_np: np.ndarray | None = None,
    cell_type_isodepth: bool = False,
    true_coords: np.ndarray | None = None,
) -> dict[str, object]:
    low_idx = int(primary_training_outputs.best_null_index)
    high_idx = int(primary_training_outputs.worst_null_index)
    if cell_type_isodepth:
        true_s = true_coords if true_coords is not None else primary_s_batched_np[0]
        slot_iso = {
            0: extract_celltype_model_isodepth(primary_model, true_s, device, slot_index=0),
            low_idx + 1: extract_celltype_model_isodepth(
                primary_model,
                primary_s_batched_np[low_idx + 1],
                device,
                slot_index=low_idx + 1,
            ),
            high_idx + 1: extract_celltype_model_isodepth(
                primary_model,
                primary_s_batched_np[high_idx + 1],
                device,
                slot_index=high_idx + 1,
            ),
        }
        for key, iso in list(slot_iso.items()):
            if iso.ndim == 2 and iso.shape[1] == 1:
                slot_iso[key] = np.asarray(iso[:, 0], dtype=np.float32)
            else:
                slot_iso[key] = np.asarray(iso, dtype=np.float32)
    else:
        slot_iso = _extract_slot_isodepths(
            primary_model, primary_s_batched_np, [0, low_idx + 1, high_idx + 1], device,
        )
    true_rerun_index, true_train_loss = _rerun_index_and_loss(primary_model, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(primary_model, low_idx + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(primary_model, high_idx + 1)
    if pred_true_np is None:
        pred_true_np = np.asarray(primary_training_outputs.pred_true, dtype=np.float32)

    artifacts: dict[str, object] = {
        "model": primary_model,
        "pred_true": pred_true_np,
        "true_isodepth": np.asarray(slot_iso[0], dtype=np.float32),
        "rerun_summary": _rerun_summary(primary_model),
        "true_rerun_index": int(true_rerun_index),
        "true_train_loss": float(true_train_loss),
        "lowest_isodepth": np.asarray(slot_iso[low_idx + 1], dtype=np.float32),
        "lowest_S": np.asarray(primary_s_batched_np[low_idx + 1], dtype=np.float32),
        "lowest_stat": float(stat_perm[low_idx]),
        "lowest_perm_index": low_idx,
        "lowest_rerun_index": int(lowest_rerun_index),
        "lowest_train_loss": float(lowest_train_loss),
        "highest_isodepth": np.asarray(slot_iso[high_idx + 1], dtype=np.float32),
        "highest_S": np.asarray(primary_s_batched_np[high_idx + 1], dtype=np.float32),
        "highest_stat": float(stat_perm[high_idx]),
        "highest_perm_index": high_idx,
        "highest_rerun_index": int(highest_rerun_index),
        "highest_train_loss": float(highest_train_loss),
        "held_out_losses": np.asarray(primary_training_outputs.model_metrics, dtype=np.float64),
        "null_summary": {
            "mean": float(np.mean(stat_perm)),
            "std": float(np.std(stat_perm)),
            "min": float(np.min(stat_perm)),
            "max": float(np.max(stat_perm)),
        },
        "train_mask": np.asarray(primary_train_mask, dtype=np.float32),
        "test_mask": np.asarray(primary_test_mask, dtype=np.float32),
        "n_folds": int(config.n_folds),
        "fold_test_sizes": np.asarray(fold_test_sizes, dtype=np.int64),
        "fold_weights": np.asarray(fold_weights, dtype=np.float64),
        "per_fold_true_loss": np.asarray(fold_true_losses, dtype=np.float64),
        "per_fold_perm_loss": np.stack(
            [np.asarray(values, dtype=np.float64) for values in fold_perm_losses],
            axis=0,
        ),
        "per_fold_p_values": np.asarray(per_fold_p_values, dtype=np.float64),
        "per_fold_true_isodepth": np.stack(
            [np.asarray(values, dtype=np.float32) for values in fold_true_isodepths],
            axis=0,
        ),
        "train_size": int(np.sum(primary_train_mask > 0)),
        "test_size": int(np.sum(primary_test_mask > 0)),
        "observed_test_loss": float(stat_true),
    }
    if covariate_artifacts:
        artifacts.update(covariate_artifacts)
    return artifacts


def _run_kfold_cv_training_loop(
    *,
    dataset: DatasetBundle,
    config: TestConfig,
    metric: str,
    device: torch.device,
    s_batched_np: np.ndarray,
    cell_fold_ids: np.ndarray,
    fold_test_sizes: np.ndarray,
    train_fn,
    train_kwargs: dict,
    extract_fold_isodepth_fn=None,
) -> tuple[
    float,
    np.ndarray,
    object,
    object,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[float],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
]:
    n_models = config.n_perms + 1
    n_folds = int(config.n_folds)
    fold_true_losses: list[float] = []
    fold_perm_losses: list[np.ndarray] = []
    fold_true_isodepths: list[np.ndarray] = []
    primary_model = None
    primary_training_outputs = None
    primary_train_mask = None
    primary_test_mask = None

    for fold_id in range(n_folds):
        train_mask_batched, test_mask_batched = _fold_train_test_masks(
            dataset.n_cells,
            n_models=n_models,
            fold_id=fold_id,
            cell_fold_ids=cell_fold_ids,
        )
        model, training_outputs, s_batched_out = train_fn(
            dataset.S,
            dataset.A,
            config,
            device=device,
            s_batched=s_batched_np,
            loss_mask_batched=train_mask_batched,
            metric_loss_mask_batched=test_mask_batched,
            model_label=(
                f"cross validation fold {fold_id + 1}/{n_folds} "
                f"(true + {config.n_perms} permuted models)"
            ),
            **train_kwargs,
        )
        fold_true_losses.append(float(training_outputs.stat_true))
        fold_perm_losses.append(np.asarray(training_outputs.stat_perm, dtype=np.float64))
        if extract_fold_isodepth_fn is not None:
            fold_true_isodepths.append(
                np.asarray(
                    extract_fold_isodepth_fn(model, dataset.S, device),
                    dtype=np.float32,
                )
            )
        if fold_id == 0:
            primary_model = model
            primary_training_outputs = training_outputs
            primary_train_mask = np.asarray(train_mask_batched[0, :, 0], dtype=np.float32)
            primary_test_mask = np.asarray(test_mask_batched[0, :, 0], dtype=np.float32)
        else:
            del model

    assert primary_model is not None
    assert primary_training_outputs is not None
    assert primary_train_mask is not None
    assert primary_test_mask is not None

    fold_weights = _cross_validation_fold_weights(fold_test_sizes, dataset.n_cells)
    stat_true, stat_perm = _aggregate_weighted_fold_losses(
        fold_true_losses,
        fold_perm_losses,
        fold_weights,
    )
    per_fold_p_values = _per_fold_p_values(metric, fold_true_losses, fold_perm_losses)
    return (
        stat_true,
        stat_perm,
        primary_model,
        primary_training_outputs,
        s_batched_np,
        primary_train_mask,
        primary_test_mask,
        fold_true_losses,
        fold_perm_losses,
        fold_true_isodepths,
        per_fold_p_values,
    )


def _run_plain_cross_validation(
    dataset: DatasetBundle, config: TestConfig, device: torch.device
) -> TestResult:
    metric = canonicalize_metric_name(config.metric)
    start = time.time()
    _validate_cross_validation_folds(dataset.n_cells, config.n_folds)
    cell_fold_ids, fold_test_sizes = _build_kfold_assignments(
        dataset.n_cells,
        config.n_folds,
        config.seed,
    )

    s_batched_pre, _ = _build_permuted_coordinate_batch(
        dataset.S,
        n_perms=config.n_perms,
        seed=config.seed,
        device=device,
    )
    s_batched_np = np.asarray(s_batched_pre.detach().cpu().numpy(), dtype=np.float32)
    del s_batched_pre

    has_midline_covariate = _covariate_type_midline(config)
    has_obs_key_covariate = _covariate_type_obs_key(config)
    parallel_config = replace(config, covariate=None) if (has_midline_covariate or has_obs_key_covariate) else config

    (
        stat_true,
        stat_perm,
        model,
        training_outputs,
        s_batched_np,
        primary_train_mask,
        primary_test_mask,
        fold_true_losses,
        fold_perm_losses,
        fold_true_isodepths,
        per_fold_p_values,
    ) = _run_kfold_cv_training_loop(
        dataset=dataset,
        config=parallel_config,
        metric=metric,
        device=device,
        s_batched_np=s_batched_np,
        cell_fold_ids=cell_fold_ids,
        fold_test_sizes=fold_test_sizes,
        train_fn=train_parallel_isodepth_model,
        train_kwargs={},
        extract_fold_isodepth_fn=_extract_fold_true_isodepth_plain,
    )

    covariate_artifacts: dict[str, object] = {}
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

    pred_true_np = np.asarray(training_outputs.pred_true, dtype=np.float32)
    if canonicalize_metric_name(metric) == "nll_poisson_mse":
        sf_row = np.asarray(dataset.A, dtype=np.float32).sum(axis=1, keepdims=True)
        pred_true_np = (sf_row * np.exp(pred_true_np)).astype(np.float32)
        pred_cov = covariate_artifacts.get("pred_true_covariate")
        if pred_cov is not None:
            covariate_artifacts["pred_true_covariate"] = (
                sf_row * np.exp(np.asarray(pred_cov, dtype=np.float32))
            ).astype(np.float32)

    fold_weights = _cross_validation_fold_weights(fold_test_sizes, dataset.n_cells)
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
        artifacts=_cross_validation_common_artifacts(
            config=config,
            metric=metric,
            stat_true=stat_true,
            stat_perm=stat_perm,
            fold_true_losses=fold_true_losses,
            fold_perm_losses=fold_perm_losses,
            fold_test_sizes=fold_test_sizes,
            fold_weights=fold_weights,
            fold_true_isodepths=fold_true_isodepths,
            per_fold_p_values=per_fold_p_values,
            primary_training_outputs=training_outputs,
            primary_model=model,
            primary_s_batched_np=s_batched_np,
            primary_train_mask=primary_train_mask,
            primary_test_mask=primary_test_mask,
            device=device,
            covariate_artifacts=covariate_artifacts,
            pred_true_np=pred_true_np,
        ),
    ).validate()


def _run_celltype_together_cross_validation(
    dataset: DatasetBundle, config: TestConfig, device: torch.device
) -> TestResult:
    metric = canonicalize_metric_name(config.metric)
    cell_type_labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    n_cell_types = int(dataset.meta["n_cell_types"])
    start = time.time()
    _validate_cross_validation_folds(dataset.n_cells, config.n_folds)
    cell_fold_ids, fold_test_sizes = _build_kfold_assignments(
        dataset.n_cells,
        config.n_folds,
        config.seed,
    )

    s_batched_pre, _ = _build_permuted_coordinate_batch(
        dataset.S,
        n_perms=config.n_perms,
        seed=config.seed,
        device=device,
    )
    s_batched_np = np.asarray(s_batched_pre.detach().cpu().numpy(), dtype=np.float32)
    del s_batched_pre

    has_midline_covariate = _covariate_type_midline(config)
    has_obs_key_covariate = _covariate_type_obs_key(config)
    parallel_config = replace(config, covariate=None) if (has_midline_covariate or has_obs_key_covariate) else config

    (
        stat_true,
        stat_perm,
        model,
        training_outputs,
        s_batched_np,
        primary_train_mask,
        primary_test_mask,
        fold_true_losses,
        fold_perm_losses,
        fold_true_isodepths,
        per_fold_p_values,
    ) = _run_kfold_cv_training_loop(
        dataset=dataset,
        config=parallel_config,
        metric=metric,
        device=device,
        s_batched_np=s_batched_np,
        cell_fold_ids=cell_fold_ids,
        fold_test_sizes=fold_test_sizes,
        train_fn=train_celltype_parallel_isodepth_model,
        train_kwargs={
            "cell_type_labels": cell_type_labels,
            "n_cell_types": n_cell_types,
        },
        extract_fold_isodepth_fn=_extract_fold_true_isodepth_celltype,
    )

    is_poisson_ct = canonicalize_metric_name(metric) == "nll_poisson_mse"
    if is_poisson_ct:
        expression_for_covariate = np.asarray(dataset.A, dtype=np.float32)
        ct_means = np.zeros((n_cell_types, dataset.A.shape[1]), dtype=np.float32)
        for ct in range(n_cell_types):
            mask = cell_type_labels == ct
            if mask.any():
                ct_means[ct] = expression_for_covariate[mask].mean(axis=0)
        sf_ct_np: np.ndarray | None = np.maximum(ct_means[cell_type_labels], 1e-3)
    else:
        expression_for_covariate = celltype_expression_residuals(
            dataset.A,
            cell_type_labels,
            n_cell_types=n_cell_types,
        )
        sf_ct_np = None

    covariate_artifacts: dict[str, object] = {}
    if _has_covariate(config):
        cov_values = (
            _obs_covariate_values(dataset, config) if has_obs_key_covariate else None
        )
        covariate_artifacts = _train_covariate_artifacts(
            dataset.S,
            expression_for_covariate,
            config,
            device,
            metric,
            stat_perm,
            covariate_values=cov_values,
            model_label="true layout covariate decoder (cell-type parallel)",
            poisson_size_factors_np=sf_ct_np,
        )

    pred_true_np = np.asarray(training_outputs.pred_true, dtype=np.float32)
    if is_poisson_ct and sf_ct_np is not None:
        pred_true_np = (sf_ct_np * np.exp(pred_true_np)).astype(np.float32)
        pred_cov = covariate_artifacts.get("pred_true_covariate")
        if pred_cov is not None:
            covariate_artifacts["pred_true_covariate"] = (
                sf_ct_np * np.exp(np.asarray(pred_cov, dtype=np.float32))
            ).astype(np.float32)

    fold_weights = _cross_validation_fold_weights(fold_test_sizes, dataset.n_cells)
    runtime_sec = time.time() - start
    artifacts = _cross_validation_common_artifacts(
        config=config,
        metric=metric,
        stat_true=stat_true,
        stat_perm=stat_perm,
        fold_true_losses=fold_true_losses,
        fold_perm_losses=fold_perm_losses,
        fold_test_sizes=fold_test_sizes,
        fold_weights=fold_weights,
        fold_true_isodepths=fold_true_isodepths,
        per_fold_p_values=per_fold_p_values,
        primary_training_outputs=training_outputs,
        primary_model=model,
        primary_s_batched_np=s_batched_np,
        primary_train_mask=primary_train_mask,
        primary_test_mask=primary_test_mask,
        device=device,
        covariate_artifacts=covariate_artifacts,
        pred_true_np=pred_true_np,
        cell_type_isodepth=True,
        true_coords=np.asarray(dataset.S, dtype=np.float32),
    )
    artifacts["cell_type_labels"] = cell_type_labels
    artifacts["cell_type_names"] = dataset.meta["cell_type_names"]
    artifacts["n_cell_types"] = n_cell_types

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
        artifacts=artifacts,
    ).validate()


def _process_single_celltype_separate_cv(
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
    _validate_cross_validation_folds(n_c, type_config.n_folds)
    cell_fold_ids, fold_test_sizes = _build_kfold_assignments(
        n_c,
        type_config.n_folds,
        type_config.seed,
    )

    s_batched_pre, _ = _build_permuted_coordinate_batch(
        S_c,
        n_perms=type_config.n_perms,
        seed=type_config.seed,
        device=device,
    )
    s_batched_np = np.asarray(s_batched_pre.detach().cpu().numpy(), dtype=np.float32)
    del s_batched_pre

    subset_bundle = DatasetBundle(S=S_c, A=A_c).validate()
    (
        stat_true,
        stat_perm,
        model_c,
        training_outputs_c,
        s_batched_np,
        primary_train_mask,
        primary_test_mask,
        fold_true_losses,
        fold_perm_losses,
        fold_true_isodepths,
        per_fold_p_values,
    ) = _run_kfold_cv_training_loop(
        dataset=subset_bundle,
        config=parallel_config,
        metric=metric,
        device=device,
        s_batched_np=s_batched_np,
        cell_fold_ids=cell_fold_ids,
        fold_test_sizes=fold_test_sizes,
        train_fn=train_parallel_isodepth_model,
        train_kwargs={},
        extract_fold_isodepth_fn=_extract_fold_true_isodepth_plain,
    )
    p_value_c = permutation_p_value(metric, stat_true, stat_perm)

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
            stat_perm,
            covariate_values=cov_values_c,
            model_label=f"{cov_label} covariate decoder ({type_name})",
        )

    fold_weights = _cross_validation_fold_weights(fold_test_sizes, n_c)
    low_idx = int(training_outputs_c.best_null_index)
    high_idx = int(training_outputs_c.worst_null_index)
    slot_iso = _extract_slot_isodepths(
        model_c, s_batched_np, [0, low_idx + 1, high_idx + 1], device,
    )
    true_rerun_index, true_train_loss = _rerun_index_and_loss(model_c, 0)
    lowest_rerun_index, lowest_train_loss = _rerun_index_and_loss(model_c, low_idx + 1)
    highest_rerun_index, highest_train_loss = _rerun_index_and_loss(model_c, high_idx + 1)

    pred_true_c = np.asarray(training_outputs_c.pred_true, dtype=np.float32)
    if canonicalize_metric_name(metric) == "nll_poisson_mse":
        sf_row_c = np.asarray(A_c, dtype=np.float32).sum(axis=1, keepdims=True)
        pred_true_c = (sf_row_c * np.exp(pred_true_c)).astype(np.float32)
        pred_cov_c = covariate_artifacts.get("pred_true_covariate")
        if pred_cov_c is not None:
            covariate_artifacts["pred_true_covariate"] = (
                sf_row_c * np.exp(np.asarray(pred_cov_c, dtype=np.float32))
            ).astype(np.float32)

    type_result = {
        "p_value": p_value_c,
        "stat_true": stat_true,
        "stat_perm": stat_perm,
        "n_cells": n_c,
        "model": model_c,
        "pred_true": pred_true_c,
        "true_isodepth": np.asarray(slot_iso[0], dtype=np.float32),
        "rerun_summary": _rerun_summary(model_c),
        "true_rerun_index": int(true_rerun_index),
        "true_train_loss": float(true_train_loss),
        "lowest_isodepth": np.asarray(slot_iso[low_idx + 1], dtype=np.float32),
        "lowest_S": np.asarray(s_batched_np[low_idx + 1], dtype=np.float32),
        "lowest_stat": float(stat_perm[low_idx]),
        "lowest_perm_index": int(low_idx),
        "lowest_rerun_index": int(lowest_rerun_index),
        "lowest_train_loss": float(lowest_train_loss),
        "highest_isodepth": np.asarray(slot_iso[high_idx + 1], dtype=np.float32),
        "highest_S": np.asarray(s_batched_np[high_idx + 1], dtype=np.float32),
        "highest_stat": float(stat_perm[high_idx]),
        "highest_perm_index": int(high_idx),
        "highest_rerun_index": int(highest_rerun_index),
        "highest_train_loss": float(highest_train_loss),
        "n_folds": int(type_config.n_folds),
        "fold_test_sizes": np.asarray(fold_test_sizes, dtype=np.int64),
        "fold_weights": np.asarray(fold_weights, dtype=np.float64),
        "per_fold_true_loss": np.asarray(fold_true_losses, dtype=np.float64),
        "per_fold_perm_loss": np.stack(
            [np.asarray(values, dtype=np.float64) for values in fold_perm_losses],
            axis=0,
        ),
        "per_fold_p_values": np.asarray(per_fold_p_values, dtype=np.float64),
        "per_fold_true_isodepth": np.stack(
            [np.asarray(values, dtype=np.float32) for values in fold_true_isodepths],
            axis=0,
        ),
        "train_mask": np.asarray(primary_train_mask, dtype=np.float32),
        "test_mask": np.asarray(primary_test_mask, dtype=np.float32),
        "observed_test_loss": float(stat_true),
        "S": S_c,
        "S_original": S_original_c,
        "A": A_c,
        **covariate_artifacts,
    }
    del s_batched_np
    type_result["model"] = offload_module_to_cpu(model_c)
    return type_result, (mean_c, safe_std_c)


def _run_celltype_separate_cross_validation(
    dataset: DatasetBundle, config: TestConfig, device: torch.device
) -> TestResult:
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
            "dataset.meta['covariate_values'] is missing."
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
            type_result, standardization = _process_single_celltype_separate_cv(
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
        method_name="cross_validation",
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
            "n_folds": int(config.n_folds),
        },
    ).validate()


def run_cross_validation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    device = device or resolve_device(config.device)

    if dataset.meta.get("cell_type_mode") == "separate":
        return _run_celltype_separate_cross_validation(dataset, config, device)
    if dataset.meta.get("cell_type_labels") is not None:
        return _run_celltype_together_cross_validation(dataset, config, device)
    return _run_plain_cross_validation(dataset, config, device)


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


def _process_single_celltype_separate_block_permutation(
    dataset: DatasetBundle,
    config: TestConfig,
    device: torch.device,
    *,
    type_index: int,
    type_name: str,
    cell_type_labels: np.ndarray,
    metric: str,
    S_raw_full: np.ndarray,
    um_per_unit: float,
    radius_um: float,
    covariate_values: np.ndarray | None = None,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    """Train/evaluate one cell type with block-permuted null; returns (per-type result dict, coord standardization)."""
    mask = cell_type_labels == type_index
    S_raw_c = np.asarray(S_raw_full[mask], dtype=np.float32)
    # S_original_c mirrors _process_single_celltype_separate: globally z-scored subset,
    # used by save_combined_celltype_isodepth_grid whose tissue_limits come from dataset.S.
    S_original_c = np.asarray(dataset.S[mask], dtype=np.float32)
    A_c, var_names_c, feature_space_c = _preprocess_separate_subset(
        dataset, dataset.A[mask], type_index
    )
    n_c = int(mask.sum())

    # per-type coordinate standardization on raw physical coords for block-permutation batch
    mean_c = S_raw_c.mean(axis=0)
    std_c = S_raw_c.std(axis=0)
    safe_std_c = np.where(std_c > 1e-8, std_c, 1.0)
    S_c = np.asarray((S_raw_c - mean_c) / safe_std_c, dtype=np.float32)

    # build block-permuted batch for this cell type only, then standardize with per-type stats
    type_seed = config.seed + type_index
    s_batched_raw_c = build_block_permuted_coordinate_batch(
        S_raw_c,
        radius_um=radius_um,
        coordinate_um_per_unit=um_per_unit,
        n_perms=config.n_perms,
        seed=type_seed,
        block_jitter=config.block_jitter,
    )
    s_batched_c = ((s_batched_raw_c - mean_c) / safe_std_c).astype(np.float32)

    S_um_c = np.asarray(S_raw_c, dtype=np.float64) * um_per_unit
    block_ids_c = hex_bin_ids(S_um_c, radius_um, (0.0, 0.0))
    s_permuted_slot1_raw_c = np.asarray(s_batched_raw_c[1], dtype=np.float32)
    block_radius_units_c = float(radius_um / um_per_unit)

    type_config = replace(config, seed=type_seed)
    parallel_config = replace(type_config, covariate=None) if _has_covariate(config) else type_config
    model_c, training_outputs_c, s_batched_np_c = train_parallel_isodepth_model(
        S_c,
        A_c,
        parallel_config,
        device=device,
        s_batched=s_batched_c,
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

    pred_true_c = np.asarray(training_outputs_c.pred_true, dtype=np.float32)
    if canonicalize_metric_name(metric) == "nll_poisson_mse":
        sf_row_c = np.asarray(A_c, dtype=np.float32).sum(axis=1, keepdims=True)
        pred_true_c = (sf_row_c * np.exp(pred_true_c)).astype(np.float32)
        pred_cov_c = covariate_artifacts.get("pred_true_covariate")
        if pred_cov_c is not None:
            covariate_artifacts["pred_true_covariate"] = (
                sf_row_c * np.exp(np.asarray(pred_cov_c, dtype=np.float32))
            ).astype(np.float32)

    type_result = {
        "p_value": p_value_c,
        "stat_true": stat_true_c,
        "stat_perm": stat_perm_c,
        "n_cells": n_c,
        "model": model_c,
        "pred_true": pred_true_c,
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
        "var_names": var_names_c,
        "feature_space": feature_space_c,
        "n_genes": int(np.asarray(A_c).shape[1]),
        "S_raw": S_raw_c,
        "block_ids_true": block_ids_c,
        "s_permuted_slot1_raw": s_permuted_slot1_raw_c,
        "block_radius_units": block_radius_units_c,
        **covariate_artifacts,
    }
    del s_batched_np_c
    type_result["model"] = offload_module_to_cpu(model_c)
    return type_result, (mean_c, safe_std_c)


def _run_celltype_separate_block_permutation(
    dataset: DatasetBundle,
    config: TestConfig,
    device: torch.device,
    *,
    S_raw: np.ndarray,
    um_per_unit: float,
    radius_um: float,
    stats: dict,
    block_ids_true: np.ndarray,
    s_permuted_slot1_raw: np.ndarray,
) -> TestResult:
    """Per-cell-type independent block-permutation models (cell_type='separate')."""
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

        def _train_current_type(train_device: torch.device, _ti=type_index, _tn=type_name) -> dict:
            nonlocal used_device
            used_device = train_device
            type_result, standardization = _process_single_celltype_separate_block_permutation(
                dataset,
                config,
                train_device,
                type_index=_ti,
                type_name=_tn,
                cell_type_labels=cell_type_labels,
                metric=metric,
                S_raw_full=S_raw,
                um_per_unit=um_per_unit,
                radius_um=radius_um,
                covariate_values=covariate_values,
            )
            per_type_standardization[_tn] = standardization
            return type_result

        per_type_results[type_name] = run_with_cuda_oom_retry(
            _train_current_type,
            current_device,
            label=f"cell type '{type_name}'",
        )
        current_device = used_device

    # Spearman correlation matrix across cell types (evaluate each model on full dataset)
    full_isodepths: list[np.ndarray] = []
    spearman_labels: list[str] = []
    for type_name in cell_type_names:
        mean_c, safe_std_c = per_type_standardization[type_name]
        S_full_standardized = np.asarray(
            (S_raw - mean_c) / safe_std_c, dtype=np.float32,
        )
        model_c = per_type_results[type_name]["model"]
        iso_full = _extract_isodepth_from_model(model_c, S_full_standardized, device)
        full_isodepths.append(iso_full.reshape(-1))
        spearman_labels.append(type_name)
    spearman_matrix, spearman_labels = _compute_spearman_matrix(full_isodepths, spearman_labels)

    runtime_sec = time.time() - start

    all_stat_true = [per_type_results[n]["stat_true"] for n in cell_type_names]
    all_stat_perm = np.concatenate([per_type_results[n]["stat_perm"] for n in cell_type_names])
    combined_stat_true = float(np.mean(all_stat_true))
    combined_p = float(np.mean([per_type_results[n]["p_value"] for n in cell_type_names]))

    return TestResult(
        method_name="block_permutation",
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
            # overlay diagnostics (full-tissue view)
            "block_ids_true": block_ids_true,
            "s_permuted_slot1_raw": s_permuted_slot1_raw,
            "block_radius_units": float(radius_um / um_per_unit),
            "block_stats": stats,
        },
    ).validate()


def run_block_permutation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    """Block-permutation existence test.

    Breaks tissue-scale gradients by randomly permuting hex-block centroids
    while preserving within-block expression–coordinate coupling.
    """
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    um_per_unit = resolve_um_per_unit(
        config.coordinate_um_per_unit,
        dataset.meta.get("coordinate_um_per_unit"),
    )
    radius_um = float(config.block_radius)  # type: ignore[arg-type]

    # Hex tiling uses raw physical coordinates (pixels / µm), not z-scored training coords.
    S_raw = raw_coordinates_from_standardized(dataset.S, dataset.meta)

    # Diagnostics
    stats = block_stats(S_raw, radius_um, um_per_unit)
    if config.verbose:
        radius_units = radius_um / um_per_unit
        zscore_note = (
            " [coords z-scored for training; binning on raw physical coords]"
            if dataset.meta.get("coordinate_standardization") == "zscore"
            else ""
        )
        print(
            f"Block permutation: radius={radius_um:.1f} µm "
            f"(={radius_units:.2f} raw coord units, scale={um_per_unit:.4f} µm/unit)"
            f"{zscore_note}"
        )
        print(f"  {stats['n_blocks']} occupied blocks")
        if stats["n_blocks"] > 0:
            print(
                f"  cells/block: mean={stats['mean_cells']:.1f}, "
                f"median={stats['median_cells']:.1f}, "
                f"min={stats['min_cells']}, max={stats['max_cells']}"
            )

    # --- build block-permuted coordinate batch in raw space, then z-score for training ---
    cell_type_labels = None
    n_cell_types = None
    cell_type_mode = dataset.meta.get("cell_type_mode", "none")
    if cell_type_mode in ("together", "separate") and "cell_type_labels" in dataset.meta:
        cell_type_labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
        n_cell_types = int(dataset.meta["n_cell_types"])

    # Overlay diagnostics: block IDs and one sample permutation slot (full tissue)
    S_um = np.asarray(S_raw, dtype=np.float64) * um_per_unit
    block_ids_true = hex_bin_ids(S_um, radius_um, (0.0, 0.0))
    overlay_batch = build_block_permuted_coordinate_batch(
        S_raw,
        radius_um=radius_um,
        coordinate_um_per_unit=um_per_unit,
        n_perms=1,
        seed=config.seed,
        cell_type_labels=cell_type_labels,
        n_cell_types=n_cell_types,
        block_jitter=config.block_jitter,
    )
    s_permuted_slot1_raw = np.asarray(overlay_batch[1], dtype=np.float32)
    del overlay_batch

    # Separate mode: independent per-type slices with per-type block batches
    if cell_type_mode == "separate":
        return _run_celltype_separate_block_permutation(
            dataset, config, device,
            S_raw=S_raw,
            um_per_unit=um_per_unit,
            radius_um=radius_um,
            stats=stats,
            block_ids_true=block_ids_true,
            s_permuted_slot1_raw=s_permuted_slot1_raw,
        )

    # Together / plain: build full permutation batch and train a single model
    s_batched_raw = build_block_permuted_coordinate_batch(
        S_raw,
        radius_um=radius_um,
        coordinate_um_per_unit=um_per_unit,
        n_perms=config.n_perms,
        seed=config.seed,
        cell_type_labels=cell_type_labels,
        n_cell_types=n_cell_types,
        block_jitter=config.block_jitter,
    )
    s_batched = standardize_coordinate_batch(s_batched_raw, dataset.meta)

    # --- train ---
    start = time.time()
    has_midline_covariate = _covariate_type_midline(config)
    has_obs_key_covariate = _covariate_type_obs_key(config)
    parallel_config = (
        replace(config, covariate=None)
        if (has_midline_covariate or has_obs_key_covariate)
        else config
    )

    if cell_type_mode == "together" and cell_type_labels is not None:
        model, training_outputs, s_batched_np = train_celltype_parallel_isodepth_model(
            dataset.S,
            dataset.A,
            parallel_config,
            cell_type_labels=cell_type_labels,
            n_cell_types=n_cell_types,
            device=device,
            s_batched=s_batched,
        )
    else:
        model, training_outputs, s_batched_np = train_parallel_isodepth_model(
            dataset.S, dataset.A, parallel_config, device=device, s_batched=s_batched
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

    pred_true_np = np.asarray(training_outputs.pred_true, dtype=np.float32)
    is_poisson = canonicalize_metric_name(metric) == "nll_poisson_mse"
    if is_poisson:
        A_f = np.asarray(dataset.A, dtype=np.float32)
        sf_row = A_f.sum(axis=1, keepdims=True)
        pred_true_np = (sf_row * np.exp(pred_true_np)).astype(np.float32)
        pred_cov = covariate_artifacts.get("pred_true_covariate")
        if pred_cov is not None:
            covariate_artifacts["pred_true_covariate"] = (
                sf_row * np.exp(np.asarray(pred_cov, dtype=np.float32))
            ).astype(np.float32)

    return TestResult(
        method_name="block_permutation",
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
            "pred_true": pred_true_np,
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
            # diagnostic artifacts for the overlay plot
            "block_ids_true": block_ids_true,
            "s_permuted_slot1_raw": s_permuted_slot1_raw,
            "block_radius_units": float(radius_um / um_per_unit),
            "block_stats": stats,
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
    if config.method == "block_permutation":
        return run_block_permutation_method(dataset, config, device=device)
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
        config.method in {"parallel_permutation", "block_permutation"}
        and dataset.meta.get("cell_type_mode") == "separate"
    ):
        return dispatch(device)
    return run_with_cuda_oom_retry(dispatch, device, label=config.method)
