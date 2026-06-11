"""Data loading, generation, and transformation utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from data.schemas import DataConfig, DatasetBundle


def load_h5ad_dataset_from_config(
    config: DataConfig,
    *,
    covariate_obs_key: Optional[str] = None,
) -> DatasetBundle:
    from data.h5ad_loader import load_dataset_from_config as _impl

    return _impl(config, covariate_obs_key=covariate_obs_key)


def load_h5ad_dataset(**kwargs) -> DatasetBundle:
    from data.h5ad_loader import load_h5ad_dataset as _impl

    return _impl(**kwargs)


def load_h5ad_as_permutation_dataset(**kwargs):
    from data.h5ad_loader import load_h5ad_as_permutation_dataset as _impl

    return _impl(**kwargs)


def generate_synthetic_dataset(config: DataConfig) -> DatasetBundle:
    from data.synthetic import generate_synthetic_dataset as _impl

    return _impl(config)


def _standardize_coordinates_inplace(dataset: DatasetBundle) -> DatasetBundle:
    """Per-axis z-score (mean 0, std 1) of ``dataset.S`` before any downstream training.

    Records ``coord_mean``, ``coord_std``, and ``coordinate_standardization='zscore'`` in
    ``dataset.meta`` so the transform is reproducible and inspectable.
    """
    s = np.asarray(dataset.S, dtype=np.float32)
    mean = s.mean(axis=0)
    std = s.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    standardized = (s - mean) / safe_std
    dataset.S = np.asarray(standardized, dtype=np.float32)
    dataset.meta["coordinate_standardization"] = "zscore"
    dataset.meta["coord_mean"] = np.asarray(mean, dtype=np.float32)
    dataset.meta["coord_std"] = np.asarray(std, dtype=np.float32)
    return dataset


def raw_coordinates_from_standardized(
    S: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Invert per-axis z-score using ``coord_mean`` / ``coord_std`` from ``meta``."""
    if meta.get("coordinate_standardization") != "zscore":
        return np.asarray(S, dtype=np.float32)
    mean = np.asarray(meta["coord_mean"], dtype=np.float32)
    std = np.asarray(meta["coord_std"], dtype=np.float32)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return (np.asarray(S, dtype=np.float32) * safe_std + mean).astype(np.float32)


def standardize_coordinate_batch(
    s_batched: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Apply the fixed true-layout z-score to every coordinate batch slot."""
    if meta.get("coordinate_standardization") != "zscore":
        return np.asarray(s_batched, dtype=np.float32)
    mean = np.asarray(meta["coord_mean"], dtype=np.float32)
    std = np.asarray(meta["coord_std"], dtype=np.float32)
    safe_std = np.where(std > 1e-8, std, 1.0)
    s = np.asarray(s_batched, dtype=np.float32)
    return ((s - mean[None, None, :]) / safe_std[None, None, :]).astype(np.float32)


def load_dataset(config: DataConfig, *, covariate=None) -> DatasetBundle:
    """Load a dataset from ``config``.

    Parameters
    ----------
    config:
        Data configuration.
    covariate:
        Optional :class:`~data.schemas.CovariateConfig`.  When set and the covariate
        is an obs-key type (not ``"midline"``), the corresponding ``adata.obs`` column
        is extracted and stored in ``dataset.meta["covariate_values"]``.  Raises
        ``ValueError`` if the key is absent from the h5ad file.
    """
    config.validate()
    covariate_obs_key: Optional[str] = None
    if covariate is not None and getattr(covariate, "is_obs_key", False):
        covariate_obs_key = covariate.type
    if config.source == "h5ad":
        dataset = load_h5ad_dataset_from_config(config, covariate_obs_key=covariate_obs_key)
    elif config.source == "synthetic":
        if covariate_obs_key is not None:
            raise ValueError(
                f"test.covariate obs key '{covariate_obs_key}' is only supported with "
                "data.source='h5ad'; synthetic data does not have an obs table."
            )
        dataset = generate_synthetic_dataset(config)
    else:
        raise ValueError(f"Unsupported data source '{config.source}'")
    if bool(getattr(config, "standardize_coordinates", True)):
        dataset = _standardize_coordinates_inplace(dataset)
    return dataset


def __getattr__(name: str):
    if name == "SpatialDataSimulator":
        from data.synthetic import SpatialDataSimulator

        return SpatialDataSimulator
    raise AttributeError(f"module 'data' has no attribute '{name}'")


__all__ = [
    "DataConfig",
    "DatasetBundle",
    "SpatialDataSimulator",
    "generate_synthetic_dataset",
    "load_dataset",
    "load_h5ad_dataset_from_config",
    "load_h5ad_dataset",
    "load_h5ad_as_permutation_dataset",
    "raw_coordinates_from_standardized",
    "standardize_coordinate_batch",
]
