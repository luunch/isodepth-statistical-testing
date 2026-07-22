"""Freedman–Lane covariate whitening before spatial permutation tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import torch

from data.schemas import DatasetBundle, TestConfig
from data.transforms import zscore_covariate
from methods.metrics import canonicalize_metric_name
from methods.trainers import (
    covariate_decoder_is_closed_form,
    fit_closed_form_decoder,
    fit_poisson_glm_irls,
    poisson_parametric_decoder_uses_irls,
    resolve_device,
    train_fixed_covariate_model,
)


def dataset_uses_freedman_lane_whitening(dataset: DatasetBundle) -> bool:
    cw = dataset.meta.get("covariate_whitening")
    return isinstance(cw, Mapping) and cw.get("method") == "freedman-lane"


def _covariate_whitening_obs_key(dataset: DatasetBundle) -> str:
    cw = dataset.meta.get("covariate_whitening")
    if not isinstance(cw, Mapping):
        raise ValueError("dataset.meta['covariate_whitening'] is missing.")
    obs_key = cw.get("obs_key")
    if not obs_key:
        raise ValueError("dataset.meta['covariate_whitening']['obs_key'] is missing.")
    return str(obs_key)


def _zscore_residuals(residuals: np.ndarray) -> np.ndarray:
    """Per-gene z-score of residual expression (same convention as expression transforms)."""
    arr = np.asarray(residuals, dtype=np.float32)
    mu = arr.mean(axis=0, keepdims=True)
    sigma = arr.std(axis=0, keepdims=True)
    return np.asarray((arr - mu) / (sigma + 1e-8), dtype=np.float32)


def _fit_freedman_lane_predictions(
    covariate_values: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    device: torch.device,
    *,
    poisson_size_factors_np: np.ndarray | None = None,
    model_label: str = "Freedman–Lane covariate decoder",
) -> np.ndarray:
    """Train decoder h(n) on covariate values and return expression predictions A'."""
    metric = canonicalize_metric_name(config.metric)
    is_poisson = metric == "nll_poisson_mse"
    closed_form = covariate_decoder_is_closed_form(config)
    decoder_type = str(getattr(config, "decoder", "nn"))
    use_poisson_irls = poisson_parametric_decoder_uses_irls(config)
    if is_poisson:
        closed_form = False

    latent_covariate = zscore_covariate(covariate_values)
    if closed_form:
        pred = fit_closed_form_decoder(latent_covariate, A, decoder_type)
    elif is_poisson and use_poisson_irls:
        pred = fit_poisson_glm_irls(
            latent_covariate,
            A,
            decoder_type,
            size_factors=poisson_size_factors_np,
        )
    else:
        _, pred = train_fixed_covariate_model(
            covariate_values,
            A,
            config,
            device=device,
            model_label=model_label,
            poisson_size_factors=poisson_size_factors_np,
        )
    return np.asarray(pred, dtype=np.float32)


def whiten_expression_freedman_lane(
    A: np.ndarray,
    covariate_values: np.ndarray,
    config: TestConfig,
    device: torch.device,
    *,
    obs_key: str = "covariate",
    poisson_size_factors_np: np.ndarray | None = None,
    model_label: str = "Freedman–Lane covariate decoder",
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit h(n) on one expression block, residualize, and per-gene re-standardize."""
    A = np.asarray(A, dtype=np.float32)
    values = np.asarray(covariate_values, dtype=np.float32).reshape(-1)
    if values.shape[0] != A.shape[0]:
        raise ValueError(
            f"covariate_whitening length {values.shape[0]} != expression rows {A.shape[0]}"
        )

    metric = canonicalize_metric_name(config.metric)
    is_poisson = metric == "nll_poisson_mse"

    pred = _fit_freedman_lane_predictions(
        values,
        A,
        config,
        device,
        poisson_size_factors_np=poisson_size_factors_np,
        model_label=model_label,
    )
    if is_poisson and poisson_size_factors_np is not None:
        sf = np.asarray(poisson_size_factors_np, dtype=np.float32)
        if sf.ndim == 1:
            sf = sf.reshape(-1, 1)
        pred = (sf * np.exp(pred)).astype(np.float32)

    whitened = _zscore_residuals(A - pred)
    latent = zscore_covariate(values)
    artifacts: dict[str, object] = {
        "freedman_lane_obs_key": obs_key,
        "freedman_lane_covariate_values": values,
        "freedman_lane_latent": latent,
        "freedman_lane_pred": pred,
    }
    return whitened, artifacts


def apply_freedman_lane_whitening(
    dataset: DatasetBundle,
    config: TestConfig,
    device: torch.device | None = None,
) -> tuple[DatasetBundle, dict[str, object]]:
    """Whitening step for together / globally preprocessed datasets."""
    if not dataset_uses_freedman_lane_whitening(dataset):
        return dataset, {}

    if dataset.meta.get("cell_type_mode") == "separate":
        raise ValueError(
            "apply_freedman_lane_whitening must not be called when "
            "data.cell_type_mode='separate'; whitening is applied per cell type "
            "after subset preprocessing."
        )

    covariate_values = dataset.meta.get("covariate_whitening_values")
    obs_key = _covariate_whitening_obs_key(dataset)
    if covariate_values is None:
        raise ValueError(
            f"data.covariate_whitening obs_key='{obs_key}' was specified but "
            "dataset.meta['covariate_whitening_values'] is missing.  "
            "Ensure load_dataset extracts the obs column during data loading."
        )

    device = device or resolve_device(config.device)
    whitened, artifacts = whiten_expression_freedman_lane(
        dataset.A,
        np.asarray(covariate_values, dtype=np.float32),
        config,
        device,
        obs_key=obs_key,
        model_label="Freedman–Lane covariate decoder",
    )

    new_dataset = replace(dataset, A=whitened)
    new_dataset.meta = dict(dataset.meta)
    new_dataset.validate()
    return new_dataset, artifacts


def assemble_separate_freedman_lane_artifacts(
    dataset: DatasetBundle,
    per_type_results: dict[str, dict],
    cell_type_names: list[str],
    cell_type_labels: np.ndarray,
) -> dict[str, object]:
    """Stitch per-cell-type Freedman–Lane diagnostics into one full-dataset artifact."""
    if not dataset_uses_freedman_lane_whitening(dataset):
        return {}

    fl_values = dataset.meta.get("covariate_whitening_values")
    if fl_values is None:
        return {}

    obs_key = _covariate_whitening_obs_key(dataset)
    n_cells = int(dataset.n_cells)
    pred_mean_full = np.zeros(n_cells, dtype=np.float32)
    for type_index, type_name in enumerate(cell_type_names):
        mask = cell_type_labels == type_index
        pred = per_type_results[type_name].get("freedman_lane_pred")
        if pred is None:
            continue
        pred_mean_full[mask] = np.asarray(pred, dtype=np.float32).mean(axis=1)

    values = np.asarray(fl_values, dtype=np.float32).reshape(-1)
    return {
        "freedman_lane_obs_key": obs_key,
        "freedman_lane_covariate_values": values,
        "freedman_lane_latent": zscore_covariate(values),
        "freedman_lane_pred": pred_mean_full.reshape(-1, 1),
    }
