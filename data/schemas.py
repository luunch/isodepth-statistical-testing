from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from methods.architectures import SUPPORTED_DECODER_TYPES, SUPPORTED_ENCODER_TYPES


CANONICAL_METRICS = {
    "nll_gaussian_mse",
    "nll_poisson_mse",
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

SUPPORTED_SPATIAL_SHAPES = frozenset({"square", "semicircle", "circle", "triangle", "square_cutout"})

SUPPORTED_SAMPLING_BIAS = frozenset({"uniform", "normal", "anisotropic_normal"})

SUPPORTED_EXPRESSION_DISTRIBUTIONS = frozenset({"gaussian", "poisson"})


@dataclass
class SamplingBiasConfig:
    """Spatial cell-sampling bias for synthetic datasets.

    - ``type='uniform'`` is Lebesgue-uniform on the configured spatial shape (no ``variance``).
    - ``type='normal'`` is isotropic Gaussian centered at ``(0.5, 0.5)`` with per-axis
      variance ``variance`` (i.e. covariance ``variance * I_2``); ``variance`` must be a
      positive scalar.
    - ``type='anisotropic_normal'`` is a diagonal-covariance Gaussian centered at ``(0.5, 0.5)``
      with covariance ``diag(variance[0], variance[1])`` (per-axis variances `\sigma_x^2`,
      `\sigma_y^2`); ``variance`` must be a length-2 list/tuple of positive floats. Useful for
      midline-style densities (e.g. ``variance=[0.003, 0.05]`` produces a vertical strip).

    All Gaussian variants are rejection-sampled inside the spatial mask intersected with
    ``[0, 1]^2``.
    """

    type: str = "uniform"
    variance: Optional[Any] = None

    def validate(self) -> "SamplingBiasConfig":
        if self.type not in SUPPORTED_SAMPLING_BIAS:
            raise ValueError(
                f"Unsupported data.sampling_bias.type {self.type!r}. "
                f"Expected one of {sorted(SUPPORTED_SAMPLING_BIAS)}"
            )
        if self.type == "normal":
            if self.variance is None:
                raise ValueError("data.sampling_bias.variance is required when type='normal'")
            if isinstance(self.variance, (list, tuple)):
                raise ValueError(
                    "data.sampling_bias.variance must be a positive scalar when type='normal'; "
                    "use type='anisotropic_normal' for per-axis variances"
                )
            self.variance = float(self.variance)
            if self.variance <= 0.0:
                raise ValueError("data.sampling_bias.variance must be > 0 when provided")
        elif self.type == "anisotropic_normal":
            if self.variance is None:
                raise ValueError(
                    "data.sampling_bias.variance is required when type='anisotropic_normal' "
                    "and must be a length-2 list [\u03c3_x\u00b2, \u03c3_y\u00b2]"
                )
            if not isinstance(self.variance, (list, tuple)) or len(self.variance) != 2:
                raise ValueError(
                    "data.sampling_bias.variance must be a length-2 list/tuple "
                    "[\u03c3_x\u00b2, \u03c3_y\u00b2] when type='anisotropic_normal'"
                )
            self.variance = [float(self.variance[0]), float(self.variance[1])]
            if any(v <= 0.0 for v in self.variance):
                raise ValueError(
                    "data.sampling_bias.variance entries must be > 0 when type='anisotropic_normal'"
                )
        else:  # uniform
            if self.variance is not None:
                raise ValueError(
                    f"data.sampling_bias.variance is only supported when type in "
                    f"{{'normal', 'anisotropic_normal'}} (got type={self.type!r})"
                )
        return self

    def to_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"type": str(self.type)}
        if self.variance is not None:
            if isinstance(self.variance, (list, tuple)):
                meta["variance"] = [float(v) for v in self.variance]
            else:
                meta["variance"] = float(self.variance)
        return meta


_SAMPLING_BIAS_ALLOWED_KEYS = frozenset({"type", "variance"})


def _sampling_bias_from_raw(raw: Any) -> Optional[SamplingBiasConfig]:
    if raw is None:
        return None
    if isinstance(raw, SamplingBiasConfig):
        return raw
    if isinstance(raw, str):
        return SamplingBiasConfig(type=raw.strip())
    if isinstance(raw, Mapping):
        unknown = set(raw.keys()) - _SAMPLING_BIAS_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"data.sampling_bias has unsupported keys {sorted(unknown)}; "
                f"allowed: {sorted(_SAMPLING_BIAS_ALLOWED_KEYS)}"
            )
        return SamplingBiasConfig(**dict(raw))
    raise TypeError(
        "data.sampling_bias must be a mapping like {\"type\": \"uniform\"} or "
        "{\"type\": \"normal\", \"variance\": 0.05}, a string such as \"uniform\", or null; "
        f"got {type(raw).__name__}"
    )


SUPPORTED_KERNEL_TYPES = frozenset({"exp"})

_KERNEL_ALLOWED_KEYS = frozenset({"type", "distance", "max_interaction_distance"})


@dataclass
class KernelConfig:
    """Spatial autocorrelation kernel for synthetic datasets.

    Adds a correlated noise component to expression with covariance
    ``Σ = σ²(I + δ·K)`` where ``K_ij = exp(-d_ij / distance)`` and
    ``d_ij`` is the Euclidean distance between cells i and j in microns.

    Parameters
    ----------
    type : str
        Kernel type.  Currently only ``"exp"`` (exponential) is supported.
    distance : float
        Length-scale ``p`` in microns.  Controls how quickly spatial
        correlation decays: correlation halves at ``d ≈ 0.69 * distance``.
    max_interaction_distance : float or None
        Hard cutoff beyond which ``K_ij`` is set to zero.  Defaults to
        ``4 * distance`` (where ``K`` has decayed to < 2%), which bounds
        memory usage via a KDTree neighbor query instead of a full N×N
        distance matrix.
    """

    type: str = "exp"
    distance: float = 100.0
    max_interaction_distance: Optional[float] = None

    @property
    def effective_cutoff(self) -> float:
        """Upper distance (µm) beyond which kernel is treated as zero."""
        return (
            self.max_interaction_distance
            if self.max_interaction_distance is not None
            else 4.0 * self.distance
        )

    def validate(self) -> "KernelConfig":
        if self.type not in SUPPORTED_KERNEL_TYPES:
            raise ValueError(
                f"data.kernel.type {self.type!r} not supported; "
                f"expected one of {sorted(SUPPORTED_KERNEL_TYPES)}"
            )
        if self.distance <= 0:
            raise ValueError("data.kernel.distance must be > 0")
        if self.max_interaction_distance is not None:
            if self.max_interaction_distance <= 0:
                raise ValueError("data.kernel.max_interaction_distance must be > 0")
            if self.max_interaction_distance < self.distance:
                raise ValueError(
                    "data.kernel.max_interaction_distance should be >= distance "
                    f"(got {self.max_interaction_distance} < {self.distance})"
                )
        return self

    def to_meta(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"type": self.type, "distance": float(self.distance)}
        if self.max_interaction_distance is not None:
            m["max_interaction_distance"] = float(self.max_interaction_distance)
        return m


def _kernel_from_raw(raw: Any) -> Optional["KernelConfig"]:
    if raw is None:
        return None
    if isinstance(raw, KernelConfig):
        return raw
    if isinstance(raw, Mapping):
        unknown = set(raw.keys()) - _KERNEL_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"data.kernel has unsupported keys {sorted(unknown)}; "
                f"allowed: {sorted(_KERNEL_ALLOWED_KEYS)}"
            )
        return KernelConfig(**dict(raw))
    raise TypeError(
        "data.kernel must be a mapping like {\"type\": \"exp\", \"distance\": 150} or null; "
        f"got {type(raw).__name__}"
    )

SUPPORTED_PERMUTATION_METHODS = {
    "parallel_permutation",
    "block_permutation",
    "cross_validation",
    "full_retraining",
    "comparison_perturbation_test",
    "perturbation_test",
    "comparison_subsampling_test",
    "subsampling_test",
}

SUPPORTED_EXISTENCE_METHODS = {
    "parallel_permutation",
    "block_permutation",
    "cross_validation",
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
    top_var_genes: int = 0
    normalize_total: bool = False
    log1p: bool = False
    standardize_expression: bool = True
    q: Optional[int] = None
    max_cells: Optional[int] = None
    seed: int = 0
    source: str = "h5ad"
    mode: str = "radial"
    n_cells: int = 900
    n_genes: int = 20
    sigma: float = 0.1
    expression_distribution: str = "gaussian"
    mean_count: float = 5.0
    k: Optional[int] = None
    k_min: Optional[int] = None
    k_max: Optional[int] = None
    dependent_xy: bool = True
    poly_degree: int = 3
    side_length: Optional[int] = None
    shape: str = "square"
    sampling_bias: Optional[SamplingBiasConfig] = None
    scale: Optional[float] = None
    kernel: Optional[KernelConfig] = None
    delta: float = 0.0
    standardize_coordinates: bool = True
    cell_type: Any = False
    cell_type_key: str = "cell_type"
    min_cells_per_celltype: int = 1
    obs_filters: Optional[dict] = None
    obs_indices: Optional[str] = None
    obs_drop_na: Optional[list[str]] = None
    # Union[bool, int]: False = off, True = DBSCAN auto-K, int >= 2 = K-Means with that K
    spatial_region_split: Union[bool, int] = False
    spatial_region_split_eps: Optional[float] = None   # DBSCAN only; auto-detected if None
    spatial_region_split_eps_mult: float = 3.0          # DBSCAN only; multiplier for auto-eps
    spatial_region_split_min_samples: int = 10          # DBSCAN only
    spatial_region_split_min_cells: int = 50            # drop sub-regions smaller than this

    def __post_init__(self) -> None:
        if self.sampling_bias is not None and not isinstance(self.sampling_bias, SamplingBiasConfig):
            self.sampling_bias = _sampling_bias_from_raw(self.sampling_bias)
        if self.kernel is not None and not isinstance(self.kernel, KernelConfig):
            self.kernel = _kernel_from_raw(self.kernel)

    @property
    def cell_type_mode(self) -> str:
        """Returns ``'none'``, ``'together'``, or ``'separate'``."""
        ct = self.cell_type
        if ct is False or ct is None:
            return "none"
        if ct is True or ct == "together":
            return "together"
        if ct == "separate":
            return "separate"
        raise ValueError(
            f"Unsupported data.cell_type value {ct!r}; "
            "expected false, true, \"together\", or \"separate\""
        )

    def validate(self) -> "DataConfig":
        if self.source not in {"h5ad", "synthetic"}:
            raise ValueError(f"Unsupported data source '{self.source}'")
        if self.source == "h5ad" and not self.h5ad:
            raise ValueError("data.h5ad is required when data.source='h5ad'")
        if self.min_cells_per_gene < 0:
            raise ValueError("data.min_cells_per_gene must be >= 0")
        if self.top_var_genes < 0:
            raise ValueError("data.top_var_genes must be >= 0 (0 means use all genes)")
        if self.top_var_genes > 0 and self.source != "h5ad":
            raise ValueError("data.top_var_genes is only supported when data.source='h5ad'")
        if self.log1p and self.q is not None:
            raise ValueError("data.log1p cannot be combined with data.q")
        if self.normalize_total and self.q is not None:
            raise ValueError(
                "data.normalize_total cannot be combined with data.q; the Poisson low-rank "
                "factorization handles library size via a size-factor offset, not by rescaling counts"
            )
        if self.normalize_total and self.source != "h5ad":
            raise ValueError("data.normalize_total is only supported when data.source='h5ad'")
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
        if self.side_length is not None and self.side_length <= 0:
            raise ValueError("data.side_length must be > 0 when provided")
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
        if self.source != "synthetic" and self.side_length is not None:
            raise ValueError("data.side_length is only supported when data.source='synthetic'")
        if self.source != "synthetic" and self.shape != "square":
            raise ValueError("data.shape is only supported when data.source='synthetic'")
        if self.source != "synthetic" and self.sampling_bias is not None:
            raise ValueError("data.sampling_bias is only supported when data.source='synthetic'")
        if self.source != "synthetic" and self.expression_distribution != "gaussian":
            raise ValueError(
                "data.expression_distribution is only supported when data.source='synthetic'"
            )
        if self.source != "synthetic" and self.mean_count != 5.0:
            raise ValueError("data.mean_count is only supported when data.source='synthetic'")
        if self.expression_distribution not in SUPPORTED_EXPRESSION_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported data.expression_distribution '{self.expression_distribution}'. "
                f"Expected one of {sorted(SUPPORTED_EXPRESSION_DISTRIBUTIONS)}"
            )
        if self.mean_count <= 0:
            raise ValueError("data.mean_count must be > 0 when provided")
        if self.shape not in SUPPORTED_SPATIAL_SHAPES:
            raise ValueError(
                f"Unsupported data.shape '{self.shape}'. Expected one of {sorted(SUPPORTED_SPATIAL_SHAPES)}"
            )
        if self.max_cells is not None and self.max_cells <= 0:
            raise ValueError("data.max_cells must be > 0 when provided")
        _ = self.cell_type_mode
        if self.cell_type_mode != "none" and self.source != "h5ad":
            raise ValueError("data.cell_type is only supported when data.source='h5ad'")
        if self.min_cells_per_celltype < 1:
            raise ValueError("data.min_cells_per_celltype must be >= 1")
        if self.obs_filters is not None:
            if not isinstance(self.obs_filters, dict) or not self.obs_filters:
                raise ValueError("data.obs_filters must be a non-empty dict when provided")
        if self.obs_indices is not None and not str(self.obs_indices).strip():
            raise ValueError("data.obs_indices must be a non-empty path when provided")
        if self.obs_filters and self.obs_indices:
            raise ValueError("Provide either data.obs_filters or data.obs_indices, not both")
        if self.obs_drop_na is not None:
            if not isinstance(self.obs_drop_na, list) or not self.obs_drop_na:
                raise ValueError("data.obs_drop_na must be a non-empty list of obs column names when provided")
            if any(not isinstance(col, str) or not col.strip() for col in self.obs_drop_na):
                raise ValueError("data.obs_drop_na entries must be non-empty strings")
        # --- spatial_region_split validation ---
        rs = self.spatial_region_split
        if rs is not False:
            # isinstance check: True is an instance of int in Python, so check bool first
            if isinstance(rs, bool):
                if rs is not True:
                    raise ValueError(
                        "data.spatial_region_split must be false (off), true (DBSCAN auto-K), "
                        "or an integer >= 2 (K-Means); got False"
                    )
            elif isinstance(rs, int):
                if rs < 2:
                    raise ValueError(
                        f"data.spatial_region_split integer must be >= 2 (got {rs}); "
                        "use false to disable or true for DBSCAN auto-K"
                    )
            else:
                raise ValueError(
                    "data.spatial_region_split must be false (off), true (DBSCAN auto-K), "
                    f"or an integer >= 2 (K-Means); got {rs!r}"
                )
            if self.cell_type_mode == "together":
                raise ValueError(
                    "data.spatial_region_split is not compatible with data.cell_type='together'; "
                    "use data.cell_type=false (global preprocessing then spatial split) or "
                    "data.cell_type='separate' (deferred per-region preprocessing)"
                )
            if self.source != "h5ad":
                raise ValueError(
                    "data.spatial_region_split is only supported when data.source='h5ad'"
                )
        if self.spatial_region_split_eps is not None and float(self.spatial_region_split_eps) <= 0:
            raise ValueError("data.spatial_region_split_eps must be > 0 when provided")
        if self.spatial_region_split_eps_mult <= 0:
            raise ValueError("data.spatial_region_split_eps_mult must be > 0")
        if self.spatial_region_split_min_samples < 1:
            raise ValueError("data.spatial_region_split_min_samples must be >= 1")
        if self.spatial_region_split_min_cells < 1:
            raise ValueError("data.spatial_region_split_min_cells must be >= 1")
        # --- end spatial_region_split validation ---

        if self.n_cells <= 0 or self.n_genes <= 0:
            raise ValueError("Synthetic data requires positive n_cells and n_genes")
        if self.source == "synthetic":
            if self.sampling_bias is not None:
                if not isinstance(self.sampling_bias, SamplingBiasConfig):
                    self.sampling_bias = _sampling_bias_from_raw(self.sampling_bias)
                self.sampling_bias = self.sampling_bias.validate()
            if self.kernel is not None or self.delta != 0.0:
                if self.source != "synthetic":
                    raise ValueError("data.kernel and data.delta are only supported when data.source='synthetic'")
                if self.scale is None:
                    raise ValueError(
                        "data.scale (tissue width in microns) is required when data.kernel is set"
                    )
                if self.scale <= 0:
                    raise ValueError("data.scale must be > 0")
                if self.delta < 0:
                    raise ValueError("data.delta must be >= 0")
                if self.kernel is not None:
                    if not isinstance(self.kernel, KernelConfig):
                        self.kernel = _kernel_from_raw(self.kernel)
                    self.kernel = self.kernel.validate()
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
            if self.mode != "noise" and self.side_length is not None:
                raise ValueError("data.side_length is only supported when data.mode='noise'")
            if self.shape != "square" and self.side_length is not None:
                raise ValueError(
                    "data.side_length is only supported when data.shape='square'; "
                    "omit side_length for semicircle, circle, or triangle (the lattice is sized from data.n_cells)"
                )
            if (
                self.mode == "noise"
                and self.side_length is not None
                and self.shape == "square"
                and self.sampling_bias is None
            ):
                if self.n_cells % int(self.side_length) != 0:
                    raise ValueError(
                        "When data.side_length is set for noise mode, data.n_cells must be divisible by data.side_length"
                    )
        return self


SUPPORTED_COVARIATE_TYPES = frozenset({"midline"})

MIDLINE_COVARIATE = "midline"


@dataclass
class CovariateConfig:
    """Optional fixed bottleneck / isodepth specification (decoder-only training).

    Two modes:
    - ``type='midline'``: fixed depth ``d(x, y) = |x - median(x)|`` computed from spatial
      coordinates; no data key required.
    - ``type=<obs_key>`` (any other non-empty string): reads per-cell latent values from
      ``adata.obs[obs_key]`` in the h5ad file; the key must exist in ``obs`` or a
      ``ValueError`` is raised at load time.
    """

    type: Optional[str] = None

    @property
    def is_obs_key(self) -> bool:
        """True when the covariate is a labeled obs column (not midline and not None)."""
        return self.type is not None and self.type != MIDLINE_COVARIATE

    def validate(self) -> "CovariateConfig":
        if self.type is not None and not self.type.strip():
            raise ValueError(
                "test.covariate.type must be a non-empty string or null; got an empty string."
            )
        return self


@dataclass
class TestConfig:
    method: str = "parallel_permutation"
    metric: str = "nll_gaussian_mse"
    n_perms: int = 100
    train_fraction: float = 0.8
    n_folds: int = 5
    n_reruns: int = 30
    alpha: float = 0.05
    n_nulls: int = 50
    epochs: int = 5000
    lr: float = 1e-3
    patience: int = 0
    seed: int = 0
    device: str = "cuda"
    decoder: str = "nn"
    encoder: str = "mlp"
    midline_init_theta: float = 0.0
    batch_size: Optional[int] = None
    sgd_batch_size: Optional[int] = None
    sgd_cosine_lr_decay: bool = False
    sgd_cosine_eta_min: float = 0.0
    sgd_cosine_t_max_steps: Optional[int] = None
    max_wall_time_sec: Optional[float] = None
    record_loss_history: bool = False
    delta: list[float] = field(default_factory=lambda: [0.05])
    perturb_target: str = "coordinates"
    subset_fractions: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.9])
    verbose: bool = True
    covariate: Optional[CovariateConfig] = None
    recursive: bool = False
    max_gradients: int = 10
    block_radius: Optional[float] = None
    coordinate_um_per_unit: Optional[float] = None
    block_jitter: bool = True
    save_permutation_null_comparison: bool = False
    gaussian_pretrain_epochs: int = 0
    gaussian_pretrain_freeze_encoder: bool = False

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
        if self.train_fraction <= 0.0 or self.train_fraction >= 1.0:
            raise ValueError("test.train_fraction must lie strictly between 0 and 1")
        if self.n_reruns <= 0:
            raise ValueError("test.n_reruns must be > 0")
        if self.alpha <= 0.0 or self.alpha >= 1.0:
            raise ValueError("test.alpha must lie strictly between 0 and 1")
        if self.epochs <= 0:
            raise ValueError("test.epochs must be > 0")
        if self.lr <= 0:
            raise ValueError("test.lr must be > 0")
        if self.patience < 0:
            raise ValueError("test.patience must be >= 0 (0 disables early stopping)")
        if self.decoder not in SUPPORTED_DECODER_TYPES:
            raise ValueError(
                f"Unsupported test.decoder '{self.decoder}'. Expected one of {sorted(SUPPORTED_DECODER_TYPES)}"
            )
        if self.encoder not in SUPPORTED_ENCODER_TYPES:
            raise ValueError(
                f"Unsupported test.encoder '{self.encoder}'. Expected one of {sorted(SUPPORTED_ENCODER_TYPES)}"
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("test.batch_size must be > 0 when provided")
        if self.sgd_batch_size is not None and self.sgd_batch_size < 0:
            raise ValueError("test.sgd_batch_size must be >= 0 when provided")
        if self.sgd_cosine_lr_decay:
            if self.sgd_batch_size is None or self.sgd_batch_size <= 0:
                raise ValueError(
                    "test.sgd_cosine_lr_decay requires test.sgd_batch_size > 0 (minibatch cell SGD)"
                )
            if self.sgd_cosine_eta_min < 0.0 or self.sgd_cosine_eta_min > self.lr:
                raise ValueError(
                    "test.sgd_cosine_eta_min must satisfy 0 <= sgd_cosine_eta_min <= test.lr "
                    f"(got eta_min={self.sgd_cosine_eta_min}, lr={self.lr})"
                )
            if self.sgd_cosine_t_max_steps is not None and int(self.sgd_cosine_t_max_steps) <= 0:
                raise ValueError("test.sgd_cosine_t_max_steps must be > 0 when provided")
        if self.max_wall_time_sec is not None and float(self.max_wall_time_sec) <= 0:
            raise ValueError("test.max_wall_time_sec must be > 0 when provided")
        self.delta = [float(value) for value in self.delta]
        if not self.delta:
            raise ValueError("test.delta must contain at least one value")
        if any(delta <= 0.0 for delta in self.delta):
            raise ValueError("test.delta entries must be > 0")
        if self.perturb_target != "coordinates":
            raise ValueError("test.perturb_target currently only supports 'coordinates'")

        self.subset_fractions = [float(value) for value in self.subset_fractions]
        if not self.subset_fractions:
            raise ValueError("test.subset_fractions must contain at least one fraction")
        if any(fraction <= 0.0 or fraction >= 1.0 for fraction in self.subset_fractions):
            raise ValueError("test.subset_fractions entries must lie strictly between 0 and 1")

        if self.method in {
            "parallel_permutation",
            "block_permutation",
            "cross_validation",
            "full_retraining",
            "comparison_perturbation_test",
            "perturbation_test",
            "comparison_subsampling_test",
            "subsampling_test",
        } and self.n_perms <= 0:
            raise ValueError("test.n_perms must be > 0")

        if self.method == "block_permutation":
            if self.block_radius is None:
                raise ValueError(
                    "test.block_radius (in microns) is required when test.method='block_permutation'"
                )
            if float(self.block_radius) <= 0:
                raise ValueError("test.block_radius must be > 0")
        if self.block_radius is not None and float(self.block_radius) <= 0:
            raise ValueError("test.block_radius must be > 0 when provided")
        if self.save_permutation_null_comparison and self.method != "block_permutation":
            raise ValueError(
                "test.save_permutation_null_comparison requires test.method='block_permutation'"
            )
        if self.coordinate_um_per_unit is not None and float(self.coordinate_um_per_unit) <= 0:
            raise ValueError("test.coordinate_um_per_unit must be > 0 when provided")

        if self.method == "cross_validation":
            if self.n_folds < 2:
                raise ValueError("test.n_folds must be >= 2 for cross_validation")
            if self.metric not in {
                "nll_gaussian_mse",
                "nll_poisson_mse",
                "poisson",
                "mse",
            }:
                raise ValueError(
                    "test.metric for cross_validation must be one of "
                    "['mse', 'nll_gaussian_mse', 'nll_poisson_mse']"
                )
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

        if self.max_gradients <= 0:
            raise ValueError("test.max_gradients must be > 0")
        if self.recursive:
            if self.method != "parallel_permutation":
                raise ValueError(
                    "test.recursive requires test.method == 'parallel_permutation'"
                )
            if self.decoder not in {"linear", "quadratic"}:
                raise ValueError(
                    f"test.recursive requires a parametric decoder ('linear' or 'quadratic'); "
                    f"got '{self.decoder}'. The nn decoder does not support recursive SVG detection."
                )

        if self.covariate is not None:
            self.covariate.validate()

        if int(self.gaussian_pretrain_epochs) < 0:
            raise ValueError("test.gaussian_pretrain_epochs must be >= 0")
        if self.gaussian_pretrain_epochs > 0:
            if self.metric not in {"nll_poisson_mse", "poisson"}:
                raise ValueError(
                    "test.gaussian_pretrain_epochs > 0 requires test.metric='nll_poisson_mse'"
                )
            if self.gaussian_pretrain_epochs >= self.epochs:
                raise ValueError(
                    "test.gaussian_pretrain_epochs must be strictly less than test.epochs "
                    f"(got {self.gaussian_pretrain_epochs} >= {self.epochs})"
                )
            if self.sgd_batch_size is None or int(self.sgd_batch_size) <= 0:
                raise ValueError(
                    "test.gaussian_pretrain_epochs > 0 requires test.sgd_batch_size > 0 "
                    "(minibatch SGD; optimizer and batch order must persist across the switch)"
                )
            if self.encoder == "midline":
                raise ValueError(
                    "test.gaussian_pretrain_epochs is incompatible with test.encoder='midline'"
                )
            if self.covariate is not None and self.covariate.type == MIDLINE_COVARIATE:
                raise ValueError(
                    "test.gaussian_pretrain_epochs is incompatible with test.covariate='midline'"
                )
            if self.gaussian_pretrain_freeze_encoder and self.sgd_cosine_lr_decay:
                raise ValueError(
                    "test.gaussian_pretrain_freeze_encoder=true is incompatible with "
                    "test.sgd_cosine_lr_decay (optimizer is recreated at the Gaussian→Poisson switch)"
                )
        elif self.gaussian_pretrain_freeze_encoder:
            raise ValueError(
                "test.gaussian_pretrain_freeze_encoder requires test.gaussian_pretrain_epochs > 0"
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


def _covariate_config_from_raw(raw: Any) -> CovariateConfig:
    """Build ``CovariateConfig`` from JSON/YAML.

    Accepts:
    - ``"midline"`` (string shorthand)
    - ``{"type": "midline"}`` (mapping)
    - any non-empty string to use as an ``adata.obs`` key
    - ``{"type": "<obs_key>"}`` mapping for the same
    """
    if isinstance(raw, CovariateConfig):
        return raw
    if isinstance(raw, str):
        return CovariateConfig(type=raw.strip() or None)
    if isinstance(raw, Mapping):
        return CovariateConfig(**dict(raw))
    raise TypeError(
        "test.covariate must be a mapping like {\"type\": \"midline\"} or "
        "{\"type\": \"<obs_key>\"}, a string such as \"midline\" or an obs key, "
        f"or omitted; got {type(raw).__name__}"
    )


def run_config_from_mapping(mapping: Optional[Mapping[str, Any]]) -> RunConfig:
    mapping = dict(mapping or {})
    test_mapping = dict(mapping.get("test", {}))
    covariate_raw = test_mapping.pop("covariate", None)
    test_config = TestConfig(**test_mapping)
    if covariate_raw is not None:
        test_config.covariate = _covariate_config_from_raw(covariate_raw)
    return RunConfig(
        data=DataConfig(**dict(mapping.get("data", {}))),
        test=test_config.validate(),
        output=OutputConfig(**dict(mapping.get("output", {}))),
    ).validate()


__all__ = [
    "CANONICAL_METRICS",
    "CovariateConfig",
    "SUPPORTED_ENCODER_TYPES",
    "DataConfig",
    "DatasetBundle",
    "MIDLINE_COVARIATE",
    "OutputConfig",
    "RunConfig",
    "SUPPORTED_COVARIATE_TYPES",
    "SUPPORTED_EXISTENCE_METHODS",
    "SUPPORTED_SAMPLING_BIAS",
    "SamplingBiasConfig",
    "SUPPORTED_SPATIAL_SHAPES",
    "SUPPORTED_EXPRESSION_DISTRIBUTIONS",
    "SUPPORTED_SYNTHETIC_MODES",
    "SUPPORTED_PERMUTATION_METHODS",
    "TestConfig",
    "TestResult",
    "run_config_from_mapping",
]
