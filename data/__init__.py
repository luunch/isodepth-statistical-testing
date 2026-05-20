"""Data loading, generation, and transformation utilities."""

from __future__ import annotations

import numpy as np

from data.schemas import DataConfig, DatasetBundle


def load_h5ad_dataset_from_config(config: DataConfig) -> DatasetBundle:
    from data.h5ad_loader import load_dataset_from_config as _impl

    return _impl(config)


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


def load_dataset(config: DataConfig) -> DatasetBundle:
    config.validate()
    if config.source == "h5ad":
        dataset = load_h5ad_dataset_from_config(config)
    elif config.source == "synthetic":
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
]
