from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np


CANONICAL_METRICS = {
    "nll_gaussian_mse",
    "mse",
    "pearson_corr_mean",
    "spearman_corr_mean",
}

SUPPORTED_SYNTHETIC_MODES = {
    "checkerboard",
    "fourier",
    "noise",
    "radial",
}

SUPPORTED_PERMUTATION_METHODS = {
    "parallel_permutation",
    "full_retraining",
    "frozen_encoder",
    "gaston_mix_closed_form",
    "comparison_perturbation_test",
    "perturbation_test",
    "comparison_subsampling_test",
    "subsampling_test",
}


@dataclass
class DatasetBundle:
    S: np.ndarray
    A: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "DatasetBundle":
        s = np.asarray(self.S, dtype=np.float32)
        a = np.asarray(self.A, dtype=np.float32)

        if s.ndim != 2 or s.shape[1] != 2:
            raise ValueError(f"DatasetBundle.S must have shape (N, 2), got {s.shape}")
        if a.ndim != 2:
            raise ValueError(f"DatasetBundle.A must be 2D, got {a.shape}")
        if s.shape[0] != a.shape[0]:
            raise ValueError(
                f"DatasetBundle row mismatch: S has {s.shape[0]} rows but A has {a.shape[0]}"
            )
        if s.shape[0] == 0 or a.shape[1] == 0:
            raise ValueError("DatasetBundle must contain at least one cell and one gene")

        self.S = s
        self.A = a
        self.meta = dict(self.meta or {})
        return self

    @property
    def n_cells(self) -> int:
        return int(self.S.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.A.shape[1])


@dataclass
class DataConfig:
    h5ad: Optional[str] = None
    spatial_key: str = "spatial"
    obs_x_col: Optional[str] = None
    obs_y_col: Optional[str] = None
    layer: Optional[str] = None
    use_raw: bool = False
    min_cells_per_gene: int = 0
    standardize: bool = True
    q: Optional[int] = None
    max_cells: Optional[int] = None
    seed: int = 0
    source: str = "h5ad"
    mode: str = "radial"
    n_cells: int = 900
    n_genes: int = 20
    sigma: float = 0.1
    k: Optional[int] = None
    k_min: Optional[int] = None
    k_max: Optional[int] = None
    dependent_xy: bool = True
    poly_degree: int = 3

    def validate(self) -> "DataConfig":
        if self.source not in {"h5ad", "synthetic"}:
            raise ValueError(f"Unsupported data source '{self.source}'")
        if self.source == "h5ad" and not self.h5ad:
            raise ValueError("data.h5ad is required when data.source='h5ad'")
        if self.min_cells_per_gene < 0:
            raise ValueError("data.min_cells_per_gene must be >= 0")
        if self.q is not None and self.q <= 0:
            raise ValueError("data.q must be > 0 when provided")
        if self.source == "synthetic" and self.q is not None:
            raise ValueError("data.q is only supported when data.source='h5ad'")
        if self.k is not None and self.k <= 0:
            raise ValueError("data.k must be > 0 when provided")
        if self.k_min is not None and self.k_min <= 0:
            raise ValueError("data.k_min must be > 0 when provided")
        if self.k_max is not None and self.k_max <= 0:
            raise ValueError("data.k_max must be > 0 when provided")
        if self.poly_degree < 0:
            raise ValueError("data.poly_degree must be >= 0")
        if self.source != "synthetic" and self.k is not None:
            raise ValueError("data.k is only supported when data.source='synthetic'")
        if self.source != "synthetic" and self.k_min is not None:
            raise ValueError("data.k_min is only supported when data.source='synthetic'")
        if self.source != "synthetic" and self.k_max is not None:
            raise ValueError("data.k_max is only supported when data.source='synthetic'")
        if self.source != "synthetic" and self.dependent_xy is not True:
            raise ValueError("data.dependent_xy is only supported when data.source='synthetic'")
        if self.source != "synthetic" and self.poly_degree != 3:
            raise ValueError("data.poly_degree is only supported when data.source='synthetic'")
        if self.max_cells is not None and self.max_cells <= 0:
            raise ValueError("data.max_cells must be > 0 when provided")
        if self.n_cells <= 0 or self.n_genes <= 0:
            raise ValueError("Synthetic data requires positive n_cells and n_genes")
        if self.source == "synthetic":
            if self.mode not in SUPPORTED_SYNTHETIC_MODES:
                raise ValueError(
                    f"Unsupported synthetic data mode '{self.mode}'. Expected one of {sorted(SUPPORTED_SYNTHETIC_MODES)}"
                )
            if self.mode == "fourier":
                if self.k is not None:
                    if self.k_min is not None or self.k_max is not None:
                        raise ValueError("data.k cannot be combined with data.k_min or data.k_max")
                    self.k_min = 1
                    self.k_max = int(self.k)
                if self.k_min is None or self.k_max is None:
                    raise ValueError(
                        "data.k_min and data.k_max are required when data.source='synthetic' and data.mode='fourier'"
                    )
                if self.k_min > self.k_max:
                    raise ValueError("data.k_min must be <= data.k_max")
            else:
                if self.k is not None or self.k_min is not None or self.k_max is not None:
                    raise ValueError("data.k, data.k_min, and data.k_max are only supported when data.mode='fourier'")
                if self.dependent_xy is not True:
                    raise ValueError("data.dependent_xy is only supported when data.mode='fourier'")
        return self


@dataclass
class TestConfig:
    method: str = "parallel_permutation"
    metric: str = "nll_gaussian_mse"
    n_perms: int = 100
    n_nulls: int = 50
    epochs: int = 5000
    lr: float = 1e-3
    patience: int = 50
    seed: int = 0
    device: str = "auto"
    batch_size: Optional[int] = None
    n_experts: int = 1
    delta: list[float] = field(default_factory=lambda: [0.05])
    perturb_target: str = "coordinates"
    subset_fractions: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.9])
    n_subsets: int = 10
    verbose: bool = True

    def validate(self) -> "TestConfig":
        if self.method not in SUPPORTED_PERMUTATION_METHODS:
            raise ValueError(
                f"Unsupported test.method '{self.method}'. Expected one of {sorted(SUPPORTED_PERMUTATION_METHODS)}"
            )
        if self.metric not in CANONICAL_METRICS:
            raise ValueError(
                f"Unsupported metric '{self.metric}'. Expected one of {sorted(CANONICAL_METRICS)}"
            )
        if self.n_nulls <= 0:
            raise ValueError("test.n_nulls must be > 0")
        if self.epochs <= 0:
            raise ValueError("test.epochs must be > 0")
        if self.lr <= 0:
            raise ValueError("test.lr must be > 0")
        if self.patience <= 0:
            raise ValueError("test.patience must be > 0")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("test.batch_size must be > 0 when provided")
        if self.n_experts <= 0:
            raise ValueError("test.n_experts must be > 0")
        self.delta = [float(value) for value in self.delta]
        if not self.delta:
            raise ValueError("test.delta must contain at least one value")
        if any(delta <= 0.0 for delta in self.delta):
            raise ValueError("test.delta entries must be > 0")
        if self.perturb_target != "coordinates":
            raise ValueError("test.perturb_target currently only supports 'coordinates'")
        if self.n_subsets <= 0:
            raise ValueError("test.n_subsets must be > 0")

        self.subset_fractions = [float(value) for value in self.subset_fractions]
        if not self.subset_fractions:
            raise ValueError("test.subset_fractions must contain at least one fraction")
        if any(fraction <= 0.0 or fraction >= 1.0 for fraction in self.subset_fractions):
            raise ValueError("test.subset_fractions entries must lie strictly between 0 and 1")

        if self.method in {
            "parallel_permutation",
            "full_retraining",
            "frozen_encoder",
            "gaston_mix_closed_form",
            "comparison_perturbation_test",
            "perturbation_test",
            "subsampling_test",
        } and self.n_perms <= 0:
            raise ValueError("test.n_perms must be > 0")

        if self.method == "comparison_subsampling_test" and self.metric not in {
            "nll_gaussian_mse",
            "mse",
        }:
            raise ValueError(
                "test.metric for comparison_subsampling_test must be one of ['mse', 'nll_gaussian_mse']"
            )
        if self.method == "perturbation_test" and self.metric not in {
            "nll_gaussian_mse",
            "mse",
        }:
            raise ValueError(
                "test.metric for perturbation_test must be one of ['mse', 'nll_gaussian_mse']"
            )
        if self.method == "subsampling_test" and self.metric not in {
            "nll_gaussian_mse",
            "mse",
        }:
            raise ValueError(
                "test.metric for subsampling_test must be one of ['mse', 'nll_gaussian_mse']"
            )
        return self


@dataclass
class OutputConfig:
    out_dir: str = "results"
    run_name: str = "permutation"
    save_preds: bool = False
    save_perm_stats: bool = True

    def validate(self) -> "OutputConfig":
        if not self.out_dir:
            raise ValueError("output.out_dir is required")
        if not self.run_name:
            raise ValueError("output.run_name is required")
        return self


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    test: TestConfig = field(default_factory=TestConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> "RunConfig":
        self.data.validate()
        self.test.validate()
        self.output.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    method_name: str
    metric: str
    p_value: float
    stat_true: float
    stat_perm: np.ndarray
    runtime_sec: float
    n_cells: int
    n_genes: int
    config: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "TestResult":
        stat_perm = np.asarray(self.stat_perm, dtype=np.float64)
        if stat_perm.ndim != 1:
            raise ValueError("TestResult.stat_perm must be a 1D array")
        if stat_perm.size == 0:
            raise ValueError("TestResult.stat_perm must contain at least one permutation statistic")

        self.stat_perm = stat_perm
        self.p_value = float(self.p_value)
        self.stat_true = float(self.stat_true)
        self.runtime_sec = float(self.runtime_sec)
        self.n_cells = int(self.n_cells)
        self.n_genes = int(self.n_genes)
        self.config = dict(self.config or {})
        self.artifacts = dict(self.artifacts or {})
        return self

    def to_json_dict(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        artifacts: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "metric": self.metric,
            "p_value": float(self.p_value),
            "stat_true": float(self.stat_true),
            "stat_perm": [float(x) for x in np.asarray(self.stat_perm).tolist()],
            "runtime_sec": float(self.runtime_sec),
            "n_cells": int(self.n_cells),
            "n_genes": int(self.n_genes),
            "config": dict(config or self.config),
            "artifacts": dict(artifacts or {}),
        }


def run_config_from_mapping(mapping: Optional[Mapping[str, Any]]) -> RunConfig:
    mapping = dict(mapping or {})
    return RunConfig(
        data=DataConfig(**dict(mapping.get("data", {}))),
        test=TestConfig(**dict(mapping.get("test", {}))),
        output=OutputConfig(**dict(mapping.get("output", {}))),
    ).validate()


__all__ = [
    "CANONICAL_METRICS",
    "DataConfig",
    "DatasetBundle",
    "OutputConfig",
    "RunConfig",
    "SUPPORTED_SYNTHETIC_MODES",
    "SUPPORTED_PERMUTATION_METHODS",
    "TestConfig",
    "TestResult",
    "run_config_from_mapping",
]
