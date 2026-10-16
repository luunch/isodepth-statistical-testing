from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
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
    "linear_x",
    "noise",
    "radial",
}


def linear_x_n_signal_genes(n_genes: int, signal_fraction: float) -> int:
    """How many genes follow ``d = x`` under ``data.mode='linear_x'``.

    ``signal_fraction * n_genes`` is rounded half-to-even. A fraction in
    ``(0, 1)`` must land on a nonempty proper subset, so a requested signal
    level cannot silently become the global null or an all-signal panel.
    """
    n_genes = int(n_genes)
    fraction = float(signal_fraction)
    if n_genes <= 0:
        raise ValueError("n_genes must be positive")
    if not np.isfinite(fraction) or not (0.0 <= fraction <= 1.0):
        raise ValueError("signal_fraction must lie in [0, 1]")
    if fraction == 0.0:
        return 0
    if fraction == 1.0:
        return n_genes
    n_signal = int(np.rint(fraction * n_genes))
    if n_signal <= 0 or n_signal >= n_genes:
        raise ValueError(
            f"signal_fraction={fraction} with n_genes={n_genes} rounds to "
            f"{n_signal} signal genes; choose a fraction that selects a proper subset"
        )
    return n_signal

SUPPORTED_SPATIAL_SHAPES = frozenset(
    {"square", "semicircle", "circle", "triangle", "square_cutout", "kidney"}
)

SUPPORTED_BLOCK_SHAPES = frozenset({"hexagon", "square"})

SUPPORTED_SAMPLING_BIAS = frozenset({"uniform", "normal", "anisotropic_normal", "lattice"})

SUPPORTED_EXPRESSION_DISTRIBUTIONS = frozenset({"gaussian", "poisson"})
SUPPORTED_INPUT_EXPRESSION_REPRESENTATIONS = frozenset(
    {"raw_counts", "log1p_normalized"}
)


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
    - ``type='lattice'`` builds the full ``data.side_length`` × ``data.side_length`` square
      lattice (honouring ``data.lattice_cell_centers``), then uniformly subsamples
      ``data.n_cells`` lattice sites without replacement. Cells stay on the regular grid so
      block-permutation meshes stay aligned; use this instead of ``uniform`` when you want
      random sparsity without breaking block tiling.

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


SUPPORTED_KERNEL_TYPES = frozenset({"exp", "trunc", "smooth"})

_KERNEL_ALLOWED_KEYS = frozenset({
    "type",
    "distance",
    "max_interaction_distance",
    "per_gene_sigma_low",
    "per_gene_sigma_high",
})


@dataclass
class KernelConfig:
    """Spatial autocorrelation kernel for synthetic datasets.

    Two generative modes:

    - ``"exp"`` / ``"trunc"`` (with ``data.delta > 0``): correlated Gaussian noise
      with covariance ``Σ = σ²(I + δ·K)`` where ``d_ij`` is Euclidean distance
      between cells i and j in microns.
    - ``"smooth"``: draw white noise ``Z`` and replace each cell with a
      Gaussian-weighted local average over nearby cells
      (``Z'_i ∝ Σ_j Z_j exp(-‖S_i-S_j‖² / (2 p²))``).  Does not use ``delta``.
      When ``per_gene_sigma_low`` and ``per_gene_sigma_high`` are set, each gene
      draws its own bandwidth ``σ_g ~ Unif[low, high]`` (µm).  ``σ_g = 0`` leaves
      that gene unsmoothed.  ``distance`` is unused in that mode.

    Parameters
    ----------
    type : str
        Kernel type.  ``"exp"``: ``K_ij = exp(-d_ij / distance)``.
        ``"trunc"``: ``K_ij = 1`` for ``d_ij ≤ cutoff``, else ``0`` (flat compact
        support).  ``"smooth"``: Gaussian bandwidth smoother of white noise;
        ``distance`` is the bandwidth ``p`` (µm).  Cutoff is
        ``max_interaction_distance`` when set, otherwise ``distance`` for
        ``"trunc"`` and ``4 * distance`` for ``"exp"`` / ``"smooth"``.
    distance : float
        Length-scale ``p`` in microns.  For ``"exp"``, correlation halves at
        ``d ≈ 0.69 * distance``.  For ``"trunc"`` without
        ``max_interaction_distance``, this is also the hard cutoff.  For
        ``"smooth"``, this is the Gaussian bandwidth ``σ_bw``.
    max_interaction_distance : float or None
        Hard cutoff beyond which kernel weights are treated as zero.  Defaults to
        ``distance`` for ``"trunc"``, else ``4 * distance`` for ``"exp"`` /
        ``"smooth"``.
    """

    type: str = "exp"
    distance: float = 100.0
    max_interaction_distance: Optional[float] = None
    per_gene_sigma_low: Optional[float] = None
    per_gene_sigma_high: Optional[float] = None

    @property
    def effective_cutoff(self) -> float:
        """Upper distance (µm) beyond which kernel is treated as zero."""
        if self.max_interaction_distance is not None:
            return float(self.max_interaction_distance)
        if self.type == "trunc":
            return float(self.distance)
        return 4.0 * self.distance

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
        low = self.per_gene_sigma_low
        high = self.per_gene_sigma_high
        if (low is None) != (high is None):
            raise ValueError(
                "data.kernel.per_gene_sigma_low and data.kernel.per_gene_sigma_high "
                "must be set together"
            )
        if low is not None:
            if self.type != "smooth":
                raise ValueError(
                    "data.kernel.per_gene_sigma_low/high are only supported when "
                    "data.kernel.type='smooth'"
                )
            if float(low) < 0.0 or float(high) <= float(low):
                raise ValueError(
                    "data.kernel per-gene sigma requires 0 <= per_gene_sigma_low "
                    f"< per_gene_sigma_high (got low={low}, high={high})"
                )
        return self

    def to_meta(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"type": self.type, "distance": float(self.distance)}
        if self.max_interaction_distance is not None:
            m["max_interaction_distance"] = float(self.max_interaction_distance)
        if self.per_gene_sigma_low is not None and self.per_gene_sigma_high is not None:
            m["per_gene_sigma_low"] = float(self.per_gene_sigma_low)
            m["per_gene_sigma_high"] = float(self.per_gene_sigma_high)
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
    "binning",
    "visium_pseudobulk",
    "correlated_null_permutation",
    "correlated_null_adaptive_permutation",
    "full_retraining",
}

SUPPORTED_EXISTENCE_METHODS = {
    "parallel_permutation",
    "block_permutation",
    "binning",
    "visium_pseudobulk",
    "correlated_null_permutation",
    "correlated_null_adaptive_permutation",
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
    input_expression_representation: str = "raw_counts"
    input_normalization_target: Optional[float] = None
    make_var_names_unique: bool = False
    min_cells_per_gene: int = 0
    # Minimum percent of cells (0-100) in which a gene must be detected (nonzero).
    # 0 disables the filter. When min_cells_per_gene is also positive, a gene must
    # satisfy both. In cell_type="separate" mode the percentage is evaluated inside
    # each domain, so the implied cell count differs by domain size.
    min_sparsity_percentage: float = 0.0
    top_var_genes: int = 0
    # After the usual expression transforms, replace gene columns with this many
    # leading PCs (cell scores). 0 = off. Separate cell-type mode fits PCA
    # independently in each subset. Incompatible with data.q.
    n_expression_pcs: int = 0
    exclude_gene_patterns: Optional[list[str]] = None
    # Restrict analysis to exactly this fixed gene panel (e.g. all genes in a pathway/gene-set
    # of interest) instead of data-driven HVG selection. Every downstream preprocessing
    # statistic (min_cells_per_gene support, normalize_total size factors, per-gene z-score
    # mean/std, HVG dispersion if top_var_genes is also nonzero) is then computed with respect
    # to only these genes, not the full transcriptome. Genes absent from the dataset are
    # silently dropped (recorded in dataset meta as gene_list_missing_genes). Mutually
    # exclusive with top_var_genes > 0 (set top_var_genes=0 to use the panel as-is; gene_list
    # replaces HVG selection rather than adding to it).
    gene_list: Optional[list[str]] = None
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
    # linear_x only. Fraction of genes with mean ±x on [0, 1]; the rest are null.
    signal_fraction: Optional[float] = None
    expression_distribution: str = "gaussian"
    mean_count: float = 5.0
    k: Optional[int] = None
    k_min: Optional[int] = None
    k_max: Optional[int] = None
    dependent_xy: bool = True
    poly_degree: int = 3
    side_length: Optional[int] = None
    lattice_cell_centers: bool = False
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
    obs_numeric_filters: Optional[dict] = None
    obs_indices: Optional[str] = None
    obs_drop_na: Optional[list[str]] = None
    # Drop spots that have an immediate spatial neighbor with a different
    # cell_type_key label. On a Visium lattice the neighbor radius is
    # label_border_radius_mult times the median nearest-neighbor distance,
    # which keeps the first hex ring and excludes the next one. Applied before
    # coordinate z-scoring so those spots never enter a separate-layer test.
    drop_label_border_spots: bool = False
    label_border_radius_mult: float = 1.2
    # Largest-connected-component outlier removal (applied before spatial_region_split and
    # before coordinate z-scoring).  Per clone, spots not in the largest connected component
    # within this radius are dropped.  Radius is in µm; requires coordinate_um_per_unit.
    spatial_denoise_radius_um: Optional[float] = None
    # Union[bool, int]: False = off, True = DBSCAN auto-K, int >= 2 = K-Means with that K
    spatial_region_split: Union[bool, int] = False
    spatial_region_split_eps: Optional[float] = None   # DBSCAN only; auto-detected if None
    spatial_region_split_eps_mult: float = 3.0          # DBSCAN only; multiplier for auto-eps
    spatial_region_split_min_samples: int = 10          # DBSCAN only
    spatial_region_split_min_cells: int = 50            # drop sub-regions smaller than this
    # Axis-aligned crop applied on already-standardized coordinates (post-zscore), e.g.
    # {"x": {"gt": -3}} keeps only cells with standardized x > -3. Same operator syntax
    # as obs_numeric_filters ("gt"/"ge"/"lt"/"le"/"eq"/"ne"). Applied after
    # standardize_coordinates and after spatial_region_split.
    spatial_crop: Optional[dict] = None
    covariate_whitening: Optional[CovariateWhiteningConfig] = None

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
        if self.input_expression_representation not in SUPPORTED_INPUT_EXPRESSION_REPRESENTATIONS:
            raise ValueError(
                "Unsupported data.input_expression_representation "
                f"{self.input_expression_representation!r}; expected one of "
                f"{sorted(SUPPORTED_INPUT_EXPRESSION_REPRESENTATIONS)}"
            )
        if self.input_expression_representation != "raw_counts":
            if self.source != "h5ad":
                raise ValueError(
                    "data.input_expression_representation is only supported for h5ad data"
                )
            if self.layer is not None or self.use_raw:
                raise ValueError(
                    "data.input_expression_representation='log1p_normalized' currently "
                    "requires data.layer=null and data.use_raw=false"
                )
            if self.input_normalization_target is None:
                raise ValueError(
                    "data.input_normalization_target is required when "
                    "data.input_expression_representation='log1p_normalized'"
                )
        if (
            self.input_normalization_target is not None
            and float(self.input_normalization_target) <= 0.0
        ):
            raise ValueError("data.input_normalization_target must be > 0 when provided")
        if (
            self.input_expression_representation == "raw_counts"
            and self.input_normalization_target is not None
        ):
            raise ValueError(
                "data.input_normalization_target is only valid when "
                "data.input_expression_representation='log1p_normalized'"
            )
        if self.make_var_names_unique and self.source != "h5ad":
            raise ValueError("data.make_var_names_unique is only supported for h5ad data")
        if self.min_cells_per_gene < 0:
            raise ValueError("data.min_cells_per_gene must be >= 0")
        if self.min_sparsity_percentage is None:
            self.min_sparsity_percentage = 0.0
        try:
            self.min_sparsity_percentage = float(self.min_sparsity_percentage)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "data.min_sparsity_percentage must be a number between 0 and 100"
            ) from exc
        if not np.isfinite(self.min_sparsity_percentage) or not (
            0.0 <= self.min_sparsity_percentage <= 100.0
        ):
            raise ValueError(
                "data.min_sparsity_percentage must be between 0 and 100 "
                "(percent of cells in which the gene is expressed; 0 disables it)"
            )
        if self.top_var_genes < 0:
            raise ValueError("data.top_var_genes must be >= 0 (0 means use all genes)")
        if self.top_var_genes > 0 and self.source != "h5ad":
            raise ValueError("data.top_var_genes is only supported when data.source='h5ad'")
        if self.n_expression_pcs < 0:
            raise ValueError("data.n_expression_pcs must be >= 0 (0 means keep genes)")
        if self.n_expression_pcs > 0 and self.source != "h5ad":
            raise ValueError("data.n_expression_pcs is only supported when data.source='h5ad'")
        if self.n_expression_pcs > 0 and self.q is not None:
            raise ValueError("data.n_expression_pcs cannot be combined with data.q")
        if self.exclude_gene_patterns is not None:
            if self.source != "h5ad":
                raise ValueError("data.exclude_gene_patterns is only supported when data.source='h5ad'")
            if (
                not isinstance(self.exclude_gene_patterns, list)
                or not self.exclude_gene_patterns
            ):
                raise ValueError(
                    "data.exclude_gene_patterns must be a non-empty list of regex strings"
                )
            for pattern in self.exclude_gene_patterns:
                if not isinstance(pattern, str) or not pattern.strip():
                    raise ValueError("data.exclude_gene_patterns entries must be non-empty strings")
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"data.exclude_gene_patterns contains invalid regex {pattern!r}: {exc}"
                    ) from exc
        if self.gene_list is not None:
            if self.source != "h5ad":
                raise ValueError("data.gene_list is only supported when data.source='h5ad'")
            if not isinstance(self.gene_list, list) or not self.gene_list:
                raise ValueError("data.gene_list must be a non-empty list of gene symbol strings")
            for gene in self.gene_list:
                if not isinstance(gene, str) or not gene.strip():
                    raise ValueError("data.gene_list entries must be non-empty strings")
            if len(set(self.gene_list)) != len(self.gene_list):
                raise ValueError("data.gene_list must not contain duplicate gene names")
            if self.top_var_genes > 0:
                raise ValueError(
                    "data.gene_list cannot be combined with data.top_var_genes > 0; "
                    "gene_list replaces HVG selection rather than adding to it -- "
                    "set data.top_var_genes=0 to use the fixed gene panel as-is"
                )
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
        if self.source != "synthetic" and self.signal_fraction is not None:
            raise ValueError(
                "data.signal_fraction is only supported when data.source='synthetic'"
            )
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
        if self.obs_numeric_filters is not None:
            if not isinstance(self.obs_numeric_filters, dict) or not self.obs_numeric_filters:
                raise ValueError("data.obs_numeric_filters must be a non-empty dict when provided")
            allowed_ops = {"gt", "ge", "gte", "lt", "le", "lte", "eq", "ne"}
            for col, spec in self.obs_numeric_filters.items():
                if not isinstance(col, str) or not col.strip():
                    raise ValueError("data.obs_numeric_filters keys must be non-empty obs column names")
                if not isinstance(spec, Mapping) or not spec:
                    raise ValueError(
                        "data.obs_numeric_filters values must be non-empty mappings "
                        "such as {\"gt\": 0.8}"
                    )
                unknown_ops = set(spec.keys()) - allowed_ops
                if unknown_ops:
                    raise ValueError(
                        f"data.obs_numeric_filters[{col!r}] has unsupported operators "
                        f"{sorted(unknown_ops)}; allowed: {sorted(allowed_ops)}"
                    )
                for op, val in spec.items():
                    try:
                        float(val)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"data.obs_numeric_filters[{col!r}][{op!r}] must be numeric"
                        ) from exc
        if self.obs_indices is not None and not str(self.obs_indices).strip():
            raise ValueError("data.obs_indices must be a non-empty path when provided")
        if self.obs_filters and self.obs_indices:
            raise ValueError("Provide either data.obs_filters or data.obs_indices, not both")
        if self.obs_drop_na is not None:
            if not isinstance(self.obs_drop_na, list) or not self.obs_drop_na:
                raise ValueError("data.obs_drop_na must be a non-empty list of obs column names when provided")
            if any(not isinstance(col, str) or not col.strip() for col in self.obs_drop_na):
                raise ValueError("data.obs_drop_na entries must be non-empty strings")
        try:
            self.label_border_radius_mult = float(self.label_border_radius_mult)
        except (TypeError, ValueError) as exc:
            raise ValueError("data.label_border_radius_mult must be a number > 1") from exc
        if (
            not np.isfinite(self.label_border_radius_mult)
            or self.label_border_radius_mult <= 1.0
        ):
            raise ValueError(
                "data.label_border_radius_mult must be > 1 "
                "(1.2 selects immediate Visium neighbors and not the next ring)"
            )
        if self.drop_label_border_spots:
            if self.source != "h5ad":
                raise ValueError(
                    "data.drop_label_border_spots is only supported when data.source='h5ad'"
                )
            if self.cell_type_mode == "none":
                raise ValueError(
                    "data.drop_label_border_spots requires data.cell_type to be "
                    'true, "together", or "separate" so each spot has a layer label'
                )
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

        # --- spatial_crop validation ---
        if self.spatial_crop is not None:
            allowed_ops = {"gt", "ge", "gte", "lt", "le", "lte", "eq", "ne"}
            if not isinstance(self.spatial_crop, dict) or not self.spatial_crop:
                raise ValueError(
                    "data.spatial_crop must be a non-empty mapping such as "
                    '{"x": {"gt": -3}}'
                )
            for axis, ops in self.spatial_crop.items():
                if axis not in ("x", "y"):
                    raise ValueError(
                        f"data.spatial_crop key must be 'x' or 'y'; got '{axis}'"
                    )
                if not isinstance(ops, dict) or not ops:
                    raise ValueError(
                        f"data.spatial_crop['{axis}'] must be a non-empty mapping "
                        "such as {'gt': -3}"
                    )
                unknown_ops = set(ops.keys()) - allowed_ops
                if unknown_ops:
                    raise ValueError(
                        f"data.spatial_crop['{axis}'] has unsupported operators "
                        f"{sorted(unknown_ops)}; allowed: {sorted(allowed_ops)}"
                    )
        # --- end spatial_crop validation ---

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
                        "data.scale (tissue width in microns) is required when data.kernel is set "
                        "or data.delta != 0"
                    )
                if self.scale <= 0:
                    raise ValueError("data.scale must be > 0")
                if self.delta < 0:
                    raise ValueError("data.delta must be >= 0")
                if self.kernel is not None:
                    if not isinstance(self.kernel, KernelConfig):
                        self.kernel = _kernel_from_raw(self.kernel)
                    self.kernel = self.kernel.validate()
                    if self.kernel.type == "smooth" and self.delta != 0.0:
                        raise ValueError(
                            "data.kernel.type='smooth' does not use data.delta; "
                            "set data.delta=0 (got "
                            f"{self.delta})"
                        )
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
            if self.mode == "linear_x":
                if self.signal_fraction is None:
                    raise ValueError(
                        "data.signal_fraction is required when data.mode='linear_x'"
                    )
                self.signal_fraction = float(self.signal_fraction)
                if self.sigma <= 0.0:
                    raise ValueError("data.sigma must be > 0 when data.mode='linear_x'")
                if self.expression_distribution != "gaussian":
                    raise ValueError("data.mode='linear_x' requires data.expression_distribution='gaussian'")
                linear_x_n_signal_genes(self.n_genes, self.signal_fraction)
            elif self.signal_fraction is not None:
                raise ValueError(
                    "data.signal_fraction is only supported when data.mode='linear_x'"
                )
            if self.mode not in {"noise", "linear_x"} and self.side_length is not None:
                raise ValueError(
                    "data.side_length is only supported when data.mode is 'noise' or 'linear_x'"
                )
            if self.shape != "square" and self.side_length is not None:
                raise ValueError(
                    "data.side_length is only supported when data.shape='square'; "
                    "omit side_length for semicircle, circle, triangle, square_cutout, or kidney "
                    "(the lattice is sized from data.n_cells)"
                )
            if (
                self.mode in {"noise", "linear_x"}
                and self.side_length is not None
                and self.shape == "square"
                and self.sampling_bias is None
            ):
                if self.n_cells % int(self.side_length) != 0:
                    raise ValueError(
                        "When data.side_length is set, data.n_cells must be divisible by data.side_length"
                    )
            if (
                self.lattice_cell_centers
                and self.sampling_bias is not None
                and self.sampling_bias.type != "lattice"
            ):
                raise ValueError(
                    "data.lattice_cell_centers requires a regular lattice; use "
                    "data.sampling_bias.type='lattice' or omit data.sampling_bias"
                )
            if self.sampling_bias is not None and self.sampling_bias.type == "lattice":
                if self.shape != "square":
                    raise ValueError(
                        "data.sampling_bias.type='lattice' requires data.shape='square'"
                    )
                if self.side_length is None:
                    raise ValueError(
                        "data.sampling_bias.type='lattice' requires data.side_length"
                    )
                lattice_capacity = int(self.side_length) ** 2
                if self.n_cells > lattice_capacity:
                    raise ValueError(
                        f"data.n_cells={self.n_cells} exceeds the {lattice_capacity} sites "
                        f"on a data.side_length={self.side_length} square lattice"
                    )
        if self.covariate_whitening is not None:
            self.covariate_whitening.validate()
        return self

MIDLINE_COVARIATE = "midline"
TOTAL_COUNTS_COVARIATE = "total_counts"

SUPPORTED_COVARIATE_WHITENING_METHODS = frozenset({"freedman-lane", "loss-difference"})
COVARIATE_WHITENING_METHOD_ALIASES = {
    "freeman-lane": "freedman-lane",
    "freedman_lane": "freedman-lane",
    "loss_difference": "loss-difference",
    "nested-covariate": "loss-difference",
}


@dataclass
class CovariateWhiteningConfig:
    """Expression whitening applied before the statistical test.

    ``obs_key`` may be a single obs column name or a list of columns.  For
    ``loss-difference``, multiple keys are concatenated as a multi-dimensional
    fixed covariate ``n`` in ``h(d, n)``.  ``freedman-lane`` currently requires a
    single key.
    """

    method: str
    obs_key: str | list[str]

    @property
    def obs_keys(self) -> list[str]:
        if isinstance(self.obs_key, str):
            return [self.obs_key]
        return list(self.obs_key)

    def validate(self) -> "CovariateWhiteningConfig":
        method = _normalize_covariate_whitening_method(self.method)
        if method not in SUPPORTED_COVARIATE_WHITENING_METHODS:
            raise ValueError(
                f"Unsupported data.covariate_whitening method '{self.method}'. "
                f"Expected one of {sorted(SUPPORTED_COVARIATE_WHITENING_METHODS)}"
            )
        keys = self.obs_keys
        if not keys:
            raise ValueError(
                "data.covariate_whitening requires a non-empty obs_key (string or list)."
            )
        cleaned: list[str] = []
        for key in keys:
            text = str(key).strip()
            if not text:
                raise ValueError(
                    "data.covariate_whitening obs_key entries must be non-empty strings."
                )
            cleaned.append(text)
        if method == "freedman-lane" and len(cleaned) > 1:
            raise ValueError(
                "data.covariate_whitening method 'freedman-lane' currently supports "
                "a single obs_key; use 'loss-difference' for multiple covariates."
            )
        self.method = method
        self.obs_key = cleaned[0] if len(cleaned) == 1 else cleaned
        return self

    @property
    def is_freedman_lane(self) -> bool:
        return self.method == "freedman-lane"

    @property
    def is_loss_difference(self) -> bool:
        return self.method == "loss-difference"


def _normalize_covariate_whitening_method(method: str) -> str:
    normalized = str(method).strip().lower()
    return COVARIATE_WHITENING_METHOD_ALIASES.get(normalized, normalized)


def _normalize_obs_key_spec(raw: Any) -> str | list[str]:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    raise TypeError(
        "data.covariate_whitening obs_key must be a string or list of strings; "
        f"got {type(raw).__name__}"
    )


def _covariate_whitening_from_raw(
    raw: Any,
    *,
    obs_key_fallback: Optional[str] = None,
) -> CovariateWhiteningConfig:
    if isinstance(raw, CovariateWhiteningConfig):
        return raw.validate()
    if isinstance(raw, str):
        method = _normalize_covariate_whitening_method(raw)
        obs_key = obs_key_fallback
        if not obs_key:
            raise ValueError(
                "data.covariate_whitening_obs_key is required when "
                "data.covariate_whitening is a method string."
            )
        return CovariateWhiteningConfig(
            method=method,
            obs_key=_normalize_obs_key_spec(obs_key),
        ).validate()
    if isinstance(raw, Mapping):
        method_raw = raw.get("method", raw.get("type"))
        if method_raw is None:
            raise ValueError(
                "data.covariate_whitening mapping must include 'method'."
            )
        obs_key = raw.get("obs_key", raw.get("column", obs_key_fallback))
        if obs_key is None and "obs_keys" in raw:
            obs_key = raw.get("obs_keys")
        if not obs_key:
            raise ValueError(
                "data.covariate_whitening mapping must include 'obs_key' "
                "(string or list)."
            )
        return CovariateWhiteningConfig(
            method=str(method_raw),
            obs_key=_normalize_obs_key_spec(obs_key),
        ).validate()
    raise TypeError(
        "data.covariate_whitening must be a method string, a mapping with "
        "'method' and 'obs_key', or omitted; got "
        f"{type(raw).__name__}"
    )


@dataclass
class CovariateConfig:
    """Optional fixed bottleneck / isodepth specification (decoder-only training).

    Three modes:
    - ``type='midline'``: fixed depth ``d(x, y) = |x - median(x)|`` computed from spatial
      coordinates; no data key required.
    - ``type='total_counts'``: per-cell ``log1p(sum_g counts)`` computed from the raw
      expression matrix before normalization; no ``obs`` column required.
    - ``type=<obs_key>`` (any other non-empty string): reads per-cell latent values from
      ``adata.obs[obs_key]`` in the h5ad file; the key must exist in ``obs`` or a
      ``ValueError`` is raised at load time.
    """

    type: Optional[str] = None

    @property
    def is_builtin(self) -> bool:
        """True for built-in covariate types (``midline``, ``total_counts``)."""
        return self.type in (MIDLINE_COVARIATE, TOTAL_COUNTS_COVARIATE)

    @property
    def is_obs_key(self) -> bool:
        """True when the covariate is a labeled obs column (not a built-in type)."""
        return self.type is not None and not self.is_builtin

    @property
    def is_values_covariate(self) -> bool:
        """True when per-cell values are stored in ``dataset.meta['covariate_values']``."""
        return self.is_obs_key or self.type == TOTAL_COUNTS_COVARIATE

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
    n_reruns: int = 30
    alpha: float = 0.05
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
    record_loss_history: bool = True
    verbose: bool = True
    covariate: Optional[CovariateConfig] = None
    recursive: bool = False
    max_gradients: int = 10
    block_radius: Optional[float] = None
    block_shape: str = "hexagon"
    bin_shape: str = "hexagon"
    bin_spot_distance_um: float = 100.0
    visium_spot_distance_um: float = 100.0
    visium_spot_diameter_um: float = 55.0
    # Override for the fallback (non-synthetic) tissue-membership radius used to decide
    # which candidate Visium spots overlap tissue. Default heuristic (1.5x median nearest-
    # neighbor spacing of the dataset's own cells) is tuned for densely-packed full tissue
    # sections; it is too tight for sparse/scattered cell-type subsets (e.g. a single
    # immune-cell DBSCAN region) where real gaps between same-type cells are still tissue.
    # When set, matches the connectivity radius used to define such regions.
    visium_tissue_radius_um: Optional[float] = None
    coordinate_um_per_unit: Optional[float] = None
    block_jitter: bool = True
    save_permutation_null_comparison: bool = False
    moran: bool = False
    moran_neighbor_radius_um: float = 30.0
    gaussian_pretrain_epochs: int = 0
    gaussian_pretrain_freeze_encoder: bool = False
    variogram_sigma_multiples: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    )
    variogram_n_bins: int = 13
    variogram_n_plot_genes: int = 10
    variogram_nn_candidates: list[float] = field(
        default_factory=lambda: [
            0.0,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]
    )
    # ``cuda`` is the default matcher. ``sequential`` is the reference loop.
    # ``cpu`` batches on the CPU. CUDA products stay float64.
    variogram_backend: str = "cuda"
    variogram_perm_batch_size: Optional[int] = None
    # ``legacy`` is the nonnegative coefficient fit with final moment restoration.
    # ``variance_constrained`` is the notebook fit a + b*gamma_Y with a + b*Var(Y) = 1,
    # selecting on that theoretical SSE, then restoring the winning field's moments.
    variogram_fit_mode: str = "legacy"

    def validate(self) -> "TestConfig":
        if self.method not in SUPPORTED_PERMUTATION_METHODS:
            raise ValueError(
                f"Unsupported test.method '{self.method}'. Expected one of {sorted(SUPPORTED_PERMUTATION_METHODS)}"
            )
        if self.metric not in CANONICAL_METRICS:
            raise ValueError(
                f"Unsupported metric '{self.metric}'. Expected one of {sorted(CANONICAL_METRICS)}"
            )
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

        if self.method in {
            "parallel_permutation",
            "block_permutation",
            "binning",
            "visium_pseudobulk",
            "correlated_null_permutation",
            "correlated_null_adaptive_permutation",
            "full_retraining",
        } and self.n_perms <= 0:
            raise ValueError("test.n_perms must be > 0")

        if self.method in {"correlated_null_permutation", "correlated_null_adaptive_permutation"}:
            if self.metric in {"nll_poisson_mse", "poisson"}:
                raise ValueError("Correlated Gaussian nulls do not support a Poisson metric")
            if int(self.variogram_n_bins) <= 0:
                raise ValueError("test.variogram_n_bins must be > 0")
            if int(self.variogram_n_plot_genes) <= 0:
                raise ValueError("test.variogram_n_plot_genes must be > 0")
        if self.method == "correlated_null_permutation":
            self.variogram_sigma_multiples = [float(v) for v in self.variogram_sigma_multiples]
            if not self.variogram_sigma_multiples:
                raise ValueError("test.variogram_sigma_multiples must be non-empty")
            if any(v <= 0 for v in self.variogram_sigma_multiples):
                raise ValueError("test.variogram_sigma_multiples entries must be > 0")
        if self.method == "correlated_null_adaptive_permutation":
            self.variogram_nn_candidates = [float(v) for v in self.variogram_nn_candidates]
            if not self.variogram_nn_candidates:
                raise ValueError("test.variogram_nn_candidates must be non-empty")
            if np.sum(np.asarray(self.variogram_nn_candidates) == 0.0) != 1:
                raise ValueError(
                    "test.variogram_nn_candidates must contain exactly one nn=0 identity candidate"
                )
            if any((v < 0 or v > 1) for v in self.variogram_nn_candidates):
                raise ValueError("test.variogram_nn_candidates entries must lie in [0, 1]")
        if self.method in {"correlated_null_permutation", "correlated_null_adaptive_permutation"}:
            backend = str(self.variogram_backend).strip().lower()
            if backend == "gpu":
                backend = "cuda"
            if backend not in {"sequential", "cpu", "cuda"}:
                raise ValueError(
                    "test.variogram_backend must be 'sequential', 'cpu', or 'cuda' "
                    f"(got {self.variogram_backend!r})"
                )
            self.variogram_backend = backend
            if self.variogram_perm_batch_size is not None:
                self.variogram_perm_batch_size = int(self.variogram_perm_batch_size)
                if self.variogram_perm_batch_size <= 0:
                    raise ValueError("test.variogram_perm_batch_size must be > 0 when provided")
            fit_mode = str(self.variogram_fit_mode).strip().lower()
            if fit_mode not in {"legacy", "variance_constrained"}:
                raise ValueError(
                    "test.variogram_fit_mode must be 'legacy' or 'variance_constrained' "
                    f"(got {self.variogram_fit_mode!r})"
                )
            self.variogram_fit_mode = fit_mode

        if self.method == "block_permutation":
            if self.block_radius is None:
                raise ValueError(
                    "test.block_radius (in microns) is required when test.method='block_permutation'"
                )
            if float(self.block_radius) <= 0:
                raise ValueError("test.block_radius must be > 0")
        if self.block_radius is not None and float(self.block_radius) <= 0:
            raise ValueError("test.block_radius must be > 0 when provided")
        if self.block_shape not in SUPPORTED_BLOCK_SHAPES:
            raise ValueError(
                f"Unsupported test.block_shape '{self.block_shape}'. "
                f"Expected one of {sorted(SUPPORTED_BLOCK_SHAPES)}"
            )
        self.bin_shape = "hexagon" if str(self.bin_shape) == "hexagonal" else str(self.bin_shape)
        if self.bin_shape not in SUPPORTED_BLOCK_SHAPES:
            raise ValueError(
                f"Unsupported test.bin_shape '{self.bin_shape}'. "
                f"Expected one of {sorted(SUPPORTED_BLOCK_SHAPES)}"
            )
        if float(self.bin_spot_distance_um) <= 0:
            raise ValueError("test.bin_spot_distance_um must be > 0")
        if float(self.visium_spot_distance_um) <= 0:
            raise ValueError("test.visium_spot_distance_um must be > 0")
        if float(self.visium_spot_diameter_um) <= 0:
            raise ValueError("test.visium_spot_diameter_um must be > 0")
        if float(self.visium_spot_diameter_um) >= float(self.visium_spot_distance_um):
            raise ValueError(
                "test.visium_spot_diameter_um must be strictly less than "
                "test.visium_spot_distance_um so neighboring spot circles cannot overlap "
                f"(got diameter={self.visium_spot_diameter_um}, distance={self.visium_spot_distance_um})"
            )
        if self.method == "visium_pseudobulk" and self.coordinate_um_per_unit is None:
            raise ValueError(
                "test.coordinate_um_per_unit is required when test.method='visium_pseudobulk'"
            )
        if self.visium_tissue_radius_um is not None and float(self.visium_tissue_radius_um) <= 0:
            raise ValueError("test.visium_tissue_radius_um must be > 0 when provided")
        if self.save_permutation_null_comparison and self.method != "block_permutation":
            raise ValueError(
                "test.save_permutation_null_comparison requires test.method='block_permutation'"
            )
        if self.coordinate_um_per_unit is not None and float(self.coordinate_um_per_unit) <= 0:
            raise ValueError("test.coordinate_um_per_unit must be > 0 when provided")

        if self.moran:
            if float(self.moran_neighbor_radius_um) <= 0:
                raise ValueError("test.moran_neighbor_radius_um must be > 0 when test.moran is true")

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
    # Embed cell-by-gene decoder predictions in result.json. Off by default:
    # those matrices are unreadable as JSON and dominate the file size.
    save_preds: bool = False
    save_perm_stats: bool = True

    def validate(self) -> "OutputConfig":
        if not self.out_dir:
            raise ValueError("output.out_dir is required")
        if not self.run_name:
            raise ValueError("output.run_name is required")
        return self


@dataclass
class BlurredPermutationsConfig:
    """Optional post-hoc spatially blurred null reference distributions."""

    enabled: bool = True
    sigmas_um: list[float] = field(default_factory=lambda: [20.0, 50.0])
    n_per_sigma: int = 20
    cutoff_mult: float = 4.0
    within_cell_type: bool = True

    def validate(self) -> "BlurredPermutationsConfig":
        self.sigmas_um = [float(value) for value in self.sigmas_um]
        if not self.sigmas_um:
            raise ValueError("blurred_permutations.sigmas_um must not be empty")
        if any(value <= 0.0 for value in self.sigmas_um):
            raise ValueError("blurred_permutations.sigmas_um entries must be > 0")
        if len(set(self.sigmas_um)) != len(self.sigmas_um):
            raise ValueError("blurred_permutations.sigmas_um must not contain duplicates")
        if int(self.n_per_sigma) <= 0:
            raise ValueError("blurred_permutations.n_per_sigma must be > 0")
        if float(self.cutoff_mult) <= 0.0:
            raise ValueError("blurred_permutations.cutoff_mult must be > 0")
        self.n_per_sigma = int(self.n_per_sigma)
        self.cutoff_mult = float(self.cutoff_mult)
        return self


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    test: TestConfig = field(default_factory=TestConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    blurred_permutations: Optional[BlurredPermutationsConfig] = None

    def validate(self) -> "RunConfig":
        self.data.validate()
        self.test.validate()
        self.output.validate()
        if self.blurred_permutations is not None:
            self.blurred_permutations.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    method_name: str
    metric: str
    p_value: Optional[float]
    stat_true: Optional[float]
    stat_perm: Optional[np.ndarray]
    runtime_sec: float
    n_cells: int
    n_genes: int
    config: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "TestResult":
        self.config = dict(self.config or {})
        self.artifacts = dict(self.artifacts or {})
        separate_cell_types = self.artifacts.get("cell_type_mode") == "separate"

        if self.stat_perm is None:
            if not separate_cell_types:
                raise ValueError("TestResult.stat_perm is required unless cell_type_mode='separate'")
        else:
            stat_perm = np.asarray(self.stat_perm, dtype=np.float64)
            if stat_perm.ndim != 1:
                raise ValueError("TestResult.stat_perm must be a 1D array")
            if stat_perm.size == 0:
                raise ValueError("TestResult.stat_perm must contain at least one permutation statistic")
            self.stat_perm = stat_perm

        if self.p_value is None:
            if not separate_cell_types:
                raise ValueError("TestResult.p_value is required unless cell_type_mode='separate'")
        else:
            self.p_value = float(self.p_value)
        if self.stat_true is None:
            if not separate_cell_types:
                raise ValueError("TestResult.stat_true is required unless cell_type_mode='separate'")
        else:
            self.stat_true = float(self.stat_true)
        self.runtime_sec = float(self.runtime_sec)
        self.n_cells = int(self.n_cells)
        self.n_genes = int(self.n_genes)
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
            "p_value": None if self.p_value is None else float(self.p_value),
            "stat_true": None if self.stat_true is None else float(self.stat_true),
            "stat_perm": None if self.stat_perm is None else [float(x) for x in np.asarray(self.stat_perm).tolist()],
            "runtime_sec": float(self.runtime_sec),
            "n_cells": int(self.n_cells),
            "n_genes": int(self.n_genes),
            "config": dict(config or self.config),
            "artifacts": dict(artifacts or {}),
        }


def _covariate_config_from_raw(raw: Any) -> CovariateConfig:
    """Build ``CovariateConfig`` from JSON/YAML.

    Accepts:
    - ``"midline"`` or ``"total_counts"`` (string shorthand)
    - ``{"type": "midline"}`` or ``{"type": "total_counts"}`` (mapping)
    - any other non-empty string to use as an ``adata.obs`` key
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
    data_mapping = dict(mapping.get("data", {}))
    whitening_obs_key = data_mapping.pop("covariate_whitening_obs_key", None)
    whitening_raw = data_mapping.pop("covariate_whitening", None)
    if whitening_raw is None and whitening_obs_key is not None:
        raise ValueError(
            "data.covariate_whitening_obs_key requires data.covariate_whitening to be set."
        )
    data_config = DataConfig(**data_mapping)
    if whitening_raw is not None:
        data_config.covariate_whitening = _covariate_whitening_from_raw(
            whitening_raw,
            obs_key_fallback=whitening_obs_key,
        )

    test_mapping = dict(mapping.get("test", {}))
    # Legacy: migrate test.freedman_lane / test.freedman-lane into data.covariate_whitening.
    legacy_lane = test_mapping.pop("freedman_lane", None)
    if "freedman-lane" in test_mapping:
        legacy_lane = test_mapping.pop("freedman-lane")
    if legacy_lane is not None and data_config.covariate_whitening is None:
        data_config.covariate_whitening = _covariate_whitening_from_raw(
            "freedman-lane",
            obs_key_fallback=str(legacy_lane),
        )

    covariate_raw = test_mapping.pop("covariate", None)
    test_config = TestConfig(**test_mapping)
    if covariate_raw is not None:
        test_config.covariate = _covariate_config_from_raw(covariate_raw)
    blurred_raw = mapping.get("blurred_permutations")
    blurred_config = (
        None
        if blurred_raw is None
        else BlurredPermutationsConfig(**dict(blurred_raw)).validate()
    )
    return RunConfig(
        data=data_config.validate(),
        test=test_config.validate(),
        output=OutputConfig(**dict(mapping.get("output", {}))),
        blurred_permutations=blurred_config,
    ).validate()


__all__ = [
    "CANONICAL_METRICS",
    "BlurredPermutationsConfig",
    "CovariateConfig",
    "CovariateWhiteningConfig",
    "SUPPORTED_ENCODER_TYPES",
    "DataConfig",
    "DatasetBundle",
    "MIDLINE_COVARIATE",
    "TOTAL_COUNTS_COVARIATE",
    "OutputConfig",
    "RunConfig",
    "SUPPORTED_COVARIATE_TYPES",
    "SUPPORTED_EXISTENCE_METHODS",
    "SUPPORTED_SAMPLING_BIAS",
    "SamplingBiasConfig",
    "SUPPORTED_SPATIAL_SHAPES",
    "SUPPORTED_EXPRESSION_DISTRIBUTIONS",
    "SUPPORTED_SYNTHETIC_MODES",
    "PERMUTATION_BLURRING_BINNING",
    "PERMUTATION_BLURRING_MODES",
    "PERMUTATION_BLURRING_NOISE_MODELS",
    "SUPPORTED_PERMUTATION_METHODS",
    "TestConfig",
    "TestResult",
    "run_config_from_mapping",
]
