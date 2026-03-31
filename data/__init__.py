"""Data loading, generation, and transformation utilities."""

from __future__ import annotations

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


def load_dataset(config: DataConfig) -> DatasetBundle:
    config.validate()
    if config.source == "h5ad":
        return load_h5ad_dataset_from_config(config)
    if config.source == "synthetic":
        return generate_synthetic_dataset(config)
    raise ValueError(f"Unsupported data source '{config.source}'")


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
