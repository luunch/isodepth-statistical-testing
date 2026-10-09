from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from data.h5ad_loader import preprocess_celltype_subset
from methods.covariate_loss_difference import (
    assemble_separate_loss_difference_artifacts,
    covariate_whitening_obs_key as _loss_diff_covariate_whitening_obs_key,
    covariate_whitening_values as _loss_diff_covariate_whitening_values,
    dataset_uses_loss_difference_whitening,
    run_loss_difference_parallel_training,
)
from methods.freedman_lane import (
    assemble_separate_freedman_lane_artifacts,
    dataset_uses_freedman_lane_whitening,
    whiten_expression_freedman_lane,
    _covariate_whitening_obs_key,
)
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
    assign_block_ids,
    block_stats,
    build_block_permuted_coordinate_batch,
    resolve_um_per_unit,
)
from methods.correlated_null_permutation import (
    build_correlated_null_adaptive_surrogates_from_dataset,
    build_correlated_null_surrogates_from_dataset,
)
from methods.binning import bin_dataset_to_pseudospots, parallel_config_for_binned_run
from methods.visium_pseudobulk import (
    build_visium_pseudobulk_dataset,
    parallel_config_for_visium_pseudobulk_run,
)
from methods.moran import maybe_compute_moran_artifacts
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


def _covariate_type_values(config: TestConfig) -> bool:
    """True when the covariate uses per-cell values from ``dataset.meta['covariate_values']``."""
    cov = getattr(config, "covariate", None)
    return cov is not None and getattr(cov, "is_values_covariate", False)


def _has_covariate(config: TestConfig) -> bool:
    cov = getattr(config, "covariate", None)
    return cov is not None and getattr(cov, "type", None) is not None


def _run_parallel_isodepth_training(
    dataset: DatasetBundle,
    *,
    train_fn,
    train_kwargs: dict,
    model_label: str,
    covariate_values: np.ndarray | None = None,
) -> tuple[object, object, np.ndarray, dict[str, object]]:
    """Train the parallel isodepth model, using h(d,n) when loss-difference is enabled."""
    if dataset_uses_loss_difference_whitening(dataset):
        if covariate_values is not None:
            values = np.asarray(covariate_values, dtype=np.float32)
            # Only squeeze a genuine single-column (N, 1) covariate to (N,); a
            # blanket reshape(-1) here would scramble a multi-covariate (N, K)
            # matrix (e.g. tumor proportion + log counts) into a flat length-N*K
            # vector instead of preserving per-cell rows.
            if values.ndim == 2 and values.shape[1] == 1:
                values = values.reshape(-1)
        else:
            values = _loss_diff_covariate_whitening_values(dataset)
        obs_key = _loss_diff_covariate_whitening_obs_key(dataset)
        model, outputs, s_batched_np, artifacts = run_loss_difference_parallel_training(
            train_fn,
            covariate_values=values,
            model_label=model_label,
            train_kwargs={**train_kwargs, "obs_key": obs_key},
        )
        return model, outputs, s_batched_np, artifacts

    model, outputs, s_batched_np = train_fn(
        **train_kwargs,
        model_label=model_label,
    )
    return model, outputs, s_batched_np, {}


def _values_covariate_from_dataset(dataset: DatasetBundle, config: TestConfig) -> np.ndarray:
    cov_values = dataset.meta.get("covariate_values")
    if cov_values is None:
        cov_type = config.covariate.type
        raise ValueError(
            f"test.covariate type '{cov_type}' was specified but "
            "dataset.meta['covariate_values'] is missing.  "
            "Ensure load_dataset is called with covariate=config.covariate so the "
            "covariate values are extracted during data loading."
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
    if _covariate_type_values(config):
        if covariate_values is None:
            cov_type = config.covariate.type
            raise ValueError(
                f"test.covariate type '{cov_type}' was specified but covariate_values is missing."
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


ExpressionNullBuilder = Callable[..., tuple[np.ndarray, dict[str, Any]]]


def _expression_null_batched_arrays(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    builder: ExpressionNullBuilder,
    dataset: DatasetBundle,
    **builder_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    A_surr, diagnostics = builder(dataset, config, A_override=A, **builder_kwargs)
    a_batched = np.concatenate(
        [np.asarray(A, dtype=np.float32)[np.newaxis, :, :], A_surr],
        axis=0,
    ).astype(np.float32)
    s_batched = np.tile(
        np.asarray(S, dtype=np.float32)[np.newaxis, :, :],
        (int(config.n_perms) + 1, 1, 1),
    ).astype(np.float32)
    return a_batched, s_batched, diagnostics


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


def _print_gene_filter_line(n_used: int, n_input: int, *, label: str | None = None) -> None:
    """One terminal line: share of genes kept, and how many the filters removed."""
    n_used = int(n_used)
    n_input = int(n_input)
    if n_input <= 0:
        return
    if n_used > n_input:
        n_input = n_used
    filtered = n_input - n_used
    prefix = f"Genes ({label})" if label else "Genes"
    print(
        f"{prefix}: using {n_used}/{n_input} ({n_used / n_input:.1%}); filtered out {filtered}",
        flush=True,
    )


def _print_dataset_gene_filter(dataset: DatasetBundle, *, label: str | None = None) -> None:
    n_used = int(dataset.meta.get("n_genes_kept", dataset.n_genes))
    n_input = int(dataset.meta.get("n_genes_input", n_used))
    _print_gene_filter_line(n_used, n_input, label=label)


def _preprocess_separate_subset(
    dataset: DatasetBundle,
    A_c_raw: np.ndarray,
    type_index: int,
    *,
    label: str | None = None,
) -> tuple[np.ndarray, list[str] | None, str | None]:
    """Apply per-cell-type expression preprocessing for separate mode.

    When the loader deferred preprocessing (``cell_type="separate"``), HVG
    selection, gene filtering and z-scoring are computed *within this cell type
    only*.  ``min_sparsity_percentage`` is a percent of this domain, so the
    implied expressing-cell threshold differs across domains.  Falls back to the
    already-transformed matrix for datasets that did not
    defer (e.g. synthetic bundles or legacy callers).
    """
    A_c_raw = np.asarray(A_c_raw, dtype=np.float32)
    n_matrix = int(A_c_raw.shape[1])
    n_file = int(dataset.meta.get("n_genes_input", n_matrix))
    pp = dataset.meta.get("separate_preprocessing")
    if not pp:
        _print_gene_filter_line(n_matrix, n_file, label=label)
        return (
            A_c_raw,
            dataset.meta.get("var_names"),
            dataset.meta.get("feature_space"),
        )
    params = {k: v for k, v in pp.items() if k != "seed"}
    seed = int(pp.get("seed", 0)) + int(type_index)
    gene_counts: dict[str, int] = {}
    transformed, var_names_c, feature_space = preprocess_celltype_subset(
        A_c_raw,
        dataset.meta.get("var_names"),
        seed=seed,
        gene_counts=gene_counts,
        **params,
    )
    n_kept = int(gene_counts.get("n_genes_kept", np.asarray(transformed).shape[1]))
    _print_gene_filter_line(n_kept, n_file, label=label)
    return transformed, var_names_c, feature_space


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
    expression_null_builder: ExpressionNullBuilder | None = None,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    """Train/evaluate one cell type; returns (per-type result dict, coord standardization)."""
    mask = cell_type_labels == type_index
    S_original_c = np.asarray(dataset.S[mask], dtype=np.float32)
    A_c, var_names_c, feature_space_c = _preprocess_separate_subset(
        dataset, dataset.A[mask], type_index, label=type_name
    )
    n_c = int(mask.sum())

    freedman_lane_type_artifacts: dict[str, object] = {}
    if dataset_uses_freedman_lane_whitening(dataset):
        fl_values = dataset.meta.get("covariate_whitening_values")
        if fl_values is None:
            obs_key = _covariate_whitening_obs_key(dataset)
            raise ValueError(
                f"data.covariate_whitening obs_key='{obs_key}' was specified but "
                "dataset.meta['covariate_whitening_values'] is missing."
            )
        A_c, freedman_lane_type_artifacts = whiten_expression_freedman_lane(
            A_c,
            np.asarray(fl_values[mask], dtype=np.float32),
            config,
            device,
            obs_key=_covariate_whitening_obs_key(dataset),
            model_label=f"Freedman–Lane covariate decoder ({type_name})",
        )

    mean_c = S_original_c.mean(axis=0)
    std_c = S_original_c.std(axis=0)
    safe_std_c = np.where(std_c > 1e-8, std_c, 1.0)
    S_c = np.asarray((S_original_c - mean_c) / safe_std_c, dtype=np.float32)

    type_config = replace(config, seed=config.seed + type_index)
    parallel_config = replace(type_config, covariate=None) if _has_covariate(config) else type_config
    loss_diff_artifacts: dict[str, object] = {}
    fl_values = dataset.meta.get("covariate_whitening_values")
    cov_values_c = (
        np.asarray(fl_values[mask], dtype=np.float32)
        if fl_values is not None and dataset_uses_loss_difference_whitening(dataset)
        else None
    )
    train_kwargs: dict[str, Any] = {
        "S": S_c,
        "A": A_c,
        "config": parallel_config,
        "device": device,
    }
    variogram_matching_diagnostics: dict[str, Any] | None = None
    a_batched_c: np.ndarray | None = None
    if expression_null_builder is not None:
        S_raw_c = raw_coordinates_from_standardized(dataset.S, dataset.meta)[mask]
        a_batched_c, s_batched_c, variogram_matching_diagnostics = _expression_null_batched_arrays(
            S_c,
            A_c,
            type_config,
            expression_null_builder,
            dataset,
            coords_override=S_raw_c,
            seed_offset=type_index,
            progress_label=type_name,
        )
        train_kwargs["s_batched"] = s_batched_c
        train_kwargs["a_batched"] = a_batched_c
    model_c, training_outputs_c, s_batched_np_c, loss_diff_artifacts = _run_parallel_isodepth_training(
        dataset,
        train_fn=train_parallel_isodepth_model,
        train_kwargs=train_kwargs,
        model_label=f"separate {type_name} h(d, n) ({n_c} cells)",
        covariate_values=cov_values_c,
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
        **freedman_lane_type_artifacts,
        **loss_diff_artifacts,
        **maybe_compute_moran_artifacts(
            type_config,
            dataset.meta,
            s_batched_np_c,
            A_c,
            a_batched=a_batched_c,
            coord_mean=mean_c,
            coord_std=safe_std_c,
        ),
    }
    if a_batched_c is not None:
        type_result["lowest_A"] = a_batched_c[low_idx + 1].copy()
        type_result["highest_A"] = a_batched_c[high_idx + 1].copy()
    if variogram_matching_diagnostics is not None:
        type_result["variogram_matching_diagnostics"] = variogram_matching_diagnostics
    del s_batched_np_c
    type_result["model"] = offload_module_to_cpu(model_c)
    return type_result, (mean_c, safe_std_c)


def _run_celltype_separate_parallel_permutation(
    dataset: DatasetBundle,
    config: TestConfig,
    device: torch.device,
    *,
    expression_null_builder: ExpressionNullBuilder | None = None,
    method_name: str = "parallel_permutation",
) -> TestResult:
    """Per-cell-type independent isodepth models with within-type permutations."""
    metric = canonicalize_metric_name(config.metric)
    cell_type_labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    cell_type_names: list[str] = list(dataset.meta["cell_type_names"])
    n_cell_types = int(dataset.meta["n_cell_types"])
    type_order = _celltype_indices_by_descending_cell_count(cell_type_labels, n_cell_types)
    covariate_values = dataset.meta.get("covariate_values")
    if _covariate_type_values(config) and covariate_values is None:
        cov_type = config.covariate.type
        raise ValueError(
            f"test.covariate type '{cov_type}' was specified but "
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
                expression_null_builder=expression_null_builder,
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

    separate_artifacts: dict[str, object] = {
        "per_type_results": per_type_results,
        "cell_type_names": cell_type_names,
        "cell_type_labels": cell_type_labels,
        "n_cell_types": n_cell_types,
        "cell_type_mode": "separate",
        "spearman_matrix": spearman_matrix,
        "spearman_labels": spearman_labels,
    }
    separate_artifacts.update(
        assemble_separate_freedman_lane_artifacts(
            dataset,
            per_type_results,
            cell_type_names,
            cell_type_labels,
        )
    )
    separate_artifacts.update(assemble_separate_loss_difference_artifacts(dataset, per_type_results))

    return TestResult(
        method_name=method_name,
        metric=metric,
        p_value=None,
        stat_true=None,
        stat_perm=None,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts=separate_artifacts,
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
    has_values_covariate = _covariate_type_values(config)
    parallel_config = replace(config, covariate=None) if (has_midline_covariate or has_values_covariate) else config
    loss_diff_artifacts: dict[str, object] = {}
    model, training_outputs, s_batched_np, loss_diff_artifacts = _run_parallel_isodepth_training(
        dataset,
        train_fn=train_celltype_parallel_isodepth_model,
        train_kwargs={
            "S": dataset.S,
            "A": dataset.A,
            "config": parallel_config,
            "cell_type_labels": cell_type_labels,
            "n_cell_types": n_cell_types,
            "device": device,
        },
        model_label="cell-type parallel isodepth h(d, n)",
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
            _values_covariate_from_dataset(dataset, config) if has_values_covariate else None
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

    moran_artifacts = maybe_compute_moran_artifacts(
        config, dataset.meta, s_batched_np, dataset.A,
    )

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
            **loss_diff_artifacts,
            **moran_artifacts,
        },
    ).validate()


def run_parallel_permutation_method(
    dataset: DatasetBundle,
    config: TestConfig,
    device: torch.device | None = None,
    *,
    expression_null_builder: ExpressionNullBuilder | None = None,
    method_name: str = "parallel_permutation",
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    if dataset.meta.get("cell_type_mode") == "separate":
        return _run_celltype_separate_parallel_permutation(
            dataset,
            config,
            device,
            expression_null_builder=expression_null_builder,
            method_name=method_name,
        )
    if dataset.meta.get("cell_type_labels") is not None:
        return _run_celltype_parallel_permutation(dataset, config, device)

    start = time.time()
    has_midline_covariate = _covariate_type_midline(config)
    has_values_covariate = _covariate_type_values(config)
    parallel_config = replace(config, covariate=None) if (has_midline_covariate or has_values_covariate) else config
    train_kwargs: dict[str, Any] = {
        "S": dataset.S,
        "A": dataset.A,
        "config": parallel_config,
        "device": device,
    }
    variogram_matching_diagnostics: dict[str, Any] | None = None
    a_batched: np.ndarray | None = None
    if expression_null_builder is not None:
        a_batched, s_batched, variogram_matching_diagnostics = _expression_null_batched_arrays(
            dataset.S,
            dataset.A,
            config,
            expression_null_builder,
            dataset,
        )
        train_kwargs["s_batched"] = s_batched
        train_kwargs["a_batched"] = a_batched
    loss_diff_artifacts: dict[str, object] = {}
    model, training_outputs, s_batched_np, loss_diff_artifacts = _run_parallel_isodepth_training(
        dataset,
        train_fn=train_parallel_isodepth_model,
        train_kwargs=train_kwargs,
        model_label="parallel isodepth h(d, n)",
    )
    stat_true = float(training_outputs.stat_true)
    stat_perm = training_outputs.stat_perm
    p_value = permutation_p_value(metric, stat_true, stat_perm)
    if _has_covariate(config):
        cov_values = (
            _values_covariate_from_dataset(dataset, config) if has_values_covariate else None
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
                f"values covariate decoder ({config.covariate.type})"
                if has_values_covariate
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

    moran_artifacts = maybe_compute_moran_artifacts(
        config,
        dataset.meta,
        s_batched_np,
        dataset.A,
        a_batched=a_batched,
    )

    artifacts: dict[str, object] = {
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
        **loss_diff_artifacts,
        **moran_artifacts,
    }
    if a_batched is not None:
        artifacts["lowest_A"] = a_batched[low_idx + 1].copy()
        artifacts["highest_A"] = a_batched[high_idx + 1].copy()
    if variogram_matching_diagnostics is not None:
        artifacts["variogram_matching_diagnostics"] = variogram_matching_diagnostics

    return TestResult(
        method_name=method_name,
        metric=metric,
        p_value=p_value,
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts=artifacts,
    ).validate()


def run_correlated_null_permutation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    """Variogram-matched correlated nulls with fixed coordinates (plain or separate mode)."""
    dataset.validate()
    config.validate()
    if (
        dataset.meta.get("cell_type_labels") is not None
        and dataset.meta.get("cell_type_mode") != "separate"
    ):
        raise NotImplementedError(
            "correlated_null_permutation currently supports plain datasets or "
            "cell_type='separate' only."
        )
    device = device or resolve_device(config.device)
    if config.verbose:
        print(
            "Correlated null permutation: fixed S, variogram-matched expression nulls "
            f"(sigma multiples={list(config.variogram_sigma_multiples)})",
            flush=True,
        )

    def _builder(
        ds: DatasetBundle,
        cfg: TestConfig,
        *,
        A_override: np.ndarray | None = None,
        coords_override: np.ndarray | None = None,
        seed_offset: int = 0,
        progress_label: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return build_correlated_null_surrogates_from_dataset(
            ds,
            cfg,
            A_override=A_override,
            coords_override=coords_override,
            seed_offset=seed_offset,
            progress_label=progress_label,
        )

    return run_parallel_permutation_method(
        dataset,
        config,
        device,
        expression_null_builder=_builder,
        method_name="correlated_null_permutation",
    )


def run_correlated_null_adaptive_permutation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    """STCompare adaptive locfit variogram-matched correlated nulls (separate or plain)."""
    dataset.validate()
    config.validate()
    if (
        dataset.meta.get("cell_type_labels") is not None
        and dataset.meta.get("cell_type_mode") != "separate"
    ):
        raise NotImplementedError(
            "correlated_null_adaptive_permutation currently supports plain datasets or "
            "cell_type='separate' only."
        )
    device = device or resolve_device(config.device)
    if config.verbose:
        print(
            "Correlated null adaptive permutation (STCompare locfit): fixed S, "
            f"nn candidates={list(config.variogram_nn_candidates)}",
            flush=True,
        )

    def _builder(
        ds: DatasetBundle,
        cfg: TestConfig,
        *,
        A_override: np.ndarray | None = None,
        coords_override: np.ndarray | None = None,
        seed_offset: int = 0,
        progress_label: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return build_correlated_null_adaptive_surrogates_from_dataset(
            ds,
            cfg,
            A_override=A_override,
            coords_override=coords_override,
            seed_offset=seed_offset,
            progress_label=progress_label,
        )

    return run_parallel_permutation_method(
        dataset,
        config,
        device,
        expression_null_builder=_builder,
        method_name="correlated_null_adaptive_permutation",
    )




def run_full_retraining_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    _has_midline = _covariate_type_midline(config)
    _has_values = _covariate_type_values(config)
    config_full = replace(config, covariate=None) if (_has_midline or _has_values) else config

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
    if _has_midline or _has_values:
        cov_values = _values_covariate_from_dataset(dataset, config) if _has_values else None
        covariate_artifacts = _train_covariate_artifacts(
            dataset.S,
            dataset.A,
            config,
            device,
            metric,
            stat_perm,
            covariate_values=cov_values,
            model_label=(
                f"values covariate decoder ({config.covariate.type})"
                if _has_values
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
        dataset, dataset.A[mask], type_index, label=type_name
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
        block_shape=config.block_shape,
    )
    s_batched_c = ((s_batched_raw_c - mean_c) / safe_std_c).astype(np.float32)

    S_um_c = np.asarray(S_raw_c, dtype=np.float64) * um_per_unit
    block_ids_c = assign_block_ids(
        S_um_c, radius_um, (0.0, 0.0), block_shape=config.block_shape,
    )
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
        "block_shape": config.block_shape,
        **covariate_artifacts,
        **maybe_compute_moran_artifacts(
            type_config,
            dataset.meta,
            s_batched_np_c,
            A_c,
            s_batched_native=s_batched_raw_c,
            coord_mean=mean_c,
            coord_std=safe_std_c,
        ),
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
    if _covariate_type_values(config) and covariate_values is None:
        cov_type = config.covariate.type
        raise ValueError(
            f"test.covariate type '{cov_type}' was specified but "
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

    return TestResult(
        method_name="block_permutation",
        metric=metric,
        p_value=None,
        stat_true=None,
        stat_perm=None,
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
            "block_shape": config.block_shape,
            "block_stats": stats,
        },
    ).validate()


def run_block_permutation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    """Block-permutation existence test.

    Breaks tissue-scale gradients by randomly permuting block centroids
    while preserving within-block expression–coordinate coupling.
    """
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)
    block_shape = config.block_shape

    um_per_unit = resolve_um_per_unit(
        config.coordinate_um_per_unit,
        dataset.meta.get("coordinate_um_per_unit"),
    )
    radius_um = float(config.block_radius)  # type: ignore[arg-type]

    # Block tiling uses raw physical coordinates (pixels / µm), not z-scored training coords.
    S_raw = raw_coordinates_from_standardized(dataset.S, dataset.meta)

    # Diagnostics
    stats = block_stats(
        S_raw, radius_um, um_per_unit, block_shape=block_shape,
    )
    if config.verbose:
        radius_units = radius_um / um_per_unit
        zscore_note = (
            " [coords z-scored for training; binning on raw physical coords]"
            if dataset.meta.get("coordinate_standardization") == "zscore"
            else ""
        )
        print(
            f"Block permutation: shape={block_shape}, radius={radius_um:.1f} µm "
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
    block_ids_true = assign_block_ids(
        S_um, radius_um, (0.0, 0.0), block_shape=block_shape,
    )
    overlay_batch = build_block_permuted_coordinate_batch(
        S_raw,
        radius_um=radius_um,
        coordinate_um_per_unit=um_per_unit,
        n_perms=1,
        seed=config.seed,
        cell_type_labels=cell_type_labels,
        n_cell_types=n_cell_types,
        block_jitter=config.block_jitter,
        block_shape=block_shape,
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
        block_shape=block_shape,
    )
    s_batched = standardize_coordinate_batch(s_batched_raw, dataset.meta)

    # --- train ---
    start = time.time()
    has_midline_covariate = _covariate_type_midline(config)
    has_values_covariate = _covariate_type_values(config)
    parallel_config = (
        replace(config, covariate=None)
        if (has_midline_covariate or has_values_covariate)
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
            _values_covariate_from_dataset(dataset, config) if has_values_covariate else None
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

    moran_artifacts = maybe_compute_moran_artifacts(
        config,
        dataset.meta,
        s_batched_np,
        dataset.A,
        s_batched_native=s_batched_raw,
    )

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
            "block_shape": block_shape,
            "block_stats": stats,
            **covariate_artifacts,
            **moran_artifacts,
        },
    ).validate()





def run_binning_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    """Bin cells/spots into pseudospots, then run the standard permutation test."""
    binned_dataset, binning_artifacts = bin_dataset_to_pseudospots(dataset, config)
    if binned_dataset.meta.get("cell_type_mode") != "separate":
        _print_dataset_gene_filter(binned_dataset)
    if config.verbose:
        summary = binning_artifacts["binning_summary"]
        print(
            "Binning permutation: "
            f"shape={summary['shape']}, "
            f"spot_distance={summary['spot_distance_um']:.1f} µm, "
            f"original cells={summary['original_n_cells']}, "
            f"pseudospots={summary['n_pseudospots']}, "
            f"cells/pseudospot median={summary['median_cells_per_pseudospot']:.1f}",
            flush=True,
        )
    inner_config = parallel_config_for_binned_run(config)
    result = run_parallel_permutation_method(binned_dataset, inner_config, device=device)
    result.method_name = "binning"
    result.config = {"test": config.__dict__.copy()}
    result.artifacts.update(binning_artifacts)
    return result.validate()


def run_visium_pseudobulk_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    """Aggregate cells into Visium-style pseudospots, then run the standard permutation test."""
    binned_dataset, pseudobulk_artifacts = build_visium_pseudobulk_dataset(dataset, config)
    if binned_dataset.meta.get("cell_type_mode") != "separate":
        _print_dataset_gene_filter(binned_dataset)
    if config.verbose:
        summary = pseudobulk_artifacts["visium_pseudobulk_summary"]
        print(
            "Visium pseudobulk: "
            f"spot_distance={summary['spot_distance_um']:.1f} µm, "
            f"spot_diameter={summary['spot_diameter_um']:.1f} µm, "
            f"original cells={summary['original_n_cells']}, "
            f"spots in tissue={summary['n_spots_in_tissue']}, "
            f"pseudospots={summary['n_pseudospots']}, "
            f"cells/pseudospot median={summary['median_cells_per_pseudospot']:.1f}, "
            f"cells dropped in gaps={summary['n_cells_dropped_in_gaps']} "
            f"({100.0 * summary['fraction_cells_dropped_in_gaps']:.1f}%)",
            flush=True,
        )
    inner_config = parallel_config_for_visium_pseudobulk_run(config)
    result = run_parallel_permutation_method(binned_dataset, inner_config, device=device)
    result.method_name = "visium_pseudobulk"
    result.config = {"test": config.__dict__.copy()}
    result.artifacts.update(pseudobulk_artifacts)
    return result.validate()


def _run_permutation_method_on_device(
    dataset: DatasetBundle, config: TestConfig, device: torch.device
) -> TestResult:
    if config.method == "parallel_permutation":
        return run_parallel_permutation_method(dataset, config, device=device)
    if config.method == "block_permutation":
        return run_block_permutation_method(dataset, config, device=device)
    if config.method == "binning":
        return run_binning_method(dataset, config, device=device)
    if config.method == "visium_pseudobulk":
        return run_visium_pseudobulk_method(dataset, config, device=device)
    if config.method == "correlated_null_permutation":
        return run_correlated_null_permutation_method(dataset, config, device=device)
    if config.method == "correlated_null_adaptive_permutation":
        return run_correlated_null_adaptive_permutation_method(dataset, config, device=device)
    if config.method == "full_retraining":
        return run_full_retraining_method(dataset, config, device=device)
    raise ValueError(f"Unsupported test.method '{config.method}'")


def run_permutation_method(dataset: DatasetBundle, config: TestConfig) -> TestResult:
    device = resolve_device(config.device)
    print(f"device: {device}")
    # Separate mode filters inside each domain, and binning / Visium pseudobulk
    # filter after aggregation. Those paths print their own line.
    if (
        dataset.meta.get("cell_type_mode") != "separate"
        and config.method not in {"binning", "visium_pseudobulk"}
    ):
        _print_dataset_gene_filter(dataset)

    dispatch = lambda resolved_device: _run_permutation_method_on_device(  # noqa: E731
        dataset, config, resolved_device
    )
    # Separate cell-type mode retries OOM per cell type (and for the together model).
    if (
        config.method in {
            "parallel_permutation",
            "block_permutation",
            "binning",
            "visium_pseudobulk",
            "correlated_null_permutation",
            "correlated_null_adaptive_permutation",
        }
        and dataset.meta.get("cell_type_mode") == "separate"
    ):
        return dispatch(device)
    return run_with_cuda_oom_retry(dispatch, device, label=config.method)
