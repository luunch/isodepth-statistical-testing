from __future__ import annotations

import re
import warnings
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data.schemas import DataConfig, DatasetBundle
from data.transforms import apply_expression_transforms, apply_expression_transforms_by_celltype


DEFAULT_OBS_COORD_CANDIDATES = [
    ("x", "y"),
    ("X", "Y"),
    ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ("array_row", "array_col"),
]


def _select_hvg_mask(adata: ad.AnnData, n_top: int) -> np.ndarray:
    """Seurat HVG mask computed on a normalize+log1p copy (safe for raw counts)."""
    import scanpy as sc

    tmp = ad.AnnData(adata.X)
    tmp.var_names = adata.var_names.copy()
    tmp.obs_names = adata.obs_names.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(
        tmp,
        flavor="seurat",
        n_top_genes=min(int(n_top), adata.n_vars - 1),
    )
    return tmp.var["highly_variable"].to_numpy()


def _select_hvg_mask_from_counts(counts: np.ndarray, n_top: int) -> np.ndarray:
    """Seurat HVG mask over a raw-count matrix ``(N, G)`` (numpy entry point)."""
    counts = np.asarray(counts, dtype=np.float32)
    n_genes = counts.shape[1]
    if n_top <= 0 or int(n_top) >= n_genes:
        return np.ones(n_genes, dtype=bool)
    tmp = ad.AnnData(counts)
    tmp.var_names = [f"gene_{i}" for i in range(n_genes)]
    return _select_hvg_mask(tmp, int(n_top))


def preprocess_celltype_subset(
    counts: np.ndarray,
    var_names: Optional[list[str]],
    *,
    min_cells_per_gene: int = 0,
    top_var_genes: int = 0,
    normalize_total: bool = False,
    log1p: bool = False,
    standardize_expression: bool = True,
    q: Optional[int] = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[str], str]:
    """Apply the full expression-preprocessing pipeline to one cell-type subset.

    Mirrors the global ordering used by :func:`load_h5ad_dataset` (HVG selection on
    raw counts, then gene filtering + normalize/log1p/standardize/q), but every
    statistic (HVG dispersion, ``min_cells_per_gene`` support, per-gene z-score
    mean/std, Poisson low-rank factorization) is computed *within this subset only*.

    Returns ``(A_c, var_names_c, feature_space)`` where ``var_names_c`` are the
    surviving gene names (or Poisson-latent feature names when ``q`` is set).
    """
    counts = np.asarray(counts, dtype=np.float32)
    n_genes_full = counts.shape[1]
    if var_names is None:
        var_names = [f"gene_{i}" for i in range(n_genes_full)]
    var_names = [str(v) for v in var_names]
    if len(var_names) != n_genes_full:
        raise ValueError(
            f"var_names length {len(var_names)} != counts genes {n_genes_full}"
        )

    surviving = np.arange(n_genes_full, dtype=np.int64)
    work = counts
    if top_var_genes and int(top_var_genes) > 0 and int(top_var_genes) < work.shape[1]:
        hvg_mask = _select_hvg_mask_from_counts(work, int(top_var_genes))
        work = work[:, hvg_mask]
        surviving = surviving[hvg_mask]

    transformed, transform_meta = apply_expression_transforms(
        work,
        min_cells_per_gene=min_cells_per_gene,
        normalize_total=normalize_total,
        log1p=log1p,
        standardize_expression=standardize_expression,
        q=q,
        seed=seed,
        return_metadata=True,
    )
    keep_mask = np.asarray(transform_meta["gene_keep_mask"], dtype=bool)
    surviving = surviving[keep_mask]

    feature_space = str(transform_meta["representation"])
    if "feature_names" in transform_meta:
        var_names_c = [str(name) for name in transform_meta["feature_names"]]
    else:
        var_names_c = [var_names[int(i)] for i in surviving]
    return np.asarray(transformed, dtype=np.float32), var_names_c, feature_space


def _extract_coordinates(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    obs_x_col: Optional[str] = None,
    obs_y_col: Optional[str] = None,
) -> np.ndarray:
    if obs_x_col and obs_y_col:
        if obs_x_col not in adata.obs.columns or obs_y_col not in adata.obs.columns:
            raise ValueError(
                f"Requested obs columns '{obs_x_col}'/'{obs_y_col}' not found in adata.obs"
            )
        return np.asarray(adata.obs[[obs_x_col, obs_y_col]].to_numpy(), dtype=np.float32)

    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"adata.obsm['{spatial_key}'] must be 2D with at least 2 columns, got {coords.shape}"
            )
        return np.asarray(coords[:, :2], dtype=np.float32)

    for x_col, y_col in DEFAULT_OBS_COORD_CANDIDATES:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return np.asarray(adata.obs[[x_col, y_col]].to_numpy(), dtype=np.float32)

    raise ValueError(
        "Could not find spatial coordinates. Provide spatial_key if using adata.obsm, "
        "or provide obs_x_col/obs_y_col for columns in adata.obs."
    )


def _extract_expression(
    adata: ad.AnnData,
    *,
    layer: Optional[str] = None,
    use_raw: bool = False,
) -> np.ndarray:
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        x = adata.layers[layer]
    elif use_raw:
        if adata.raw is None:
            raise ValueError("use_raw requested but adata.raw is None")
        x = adata.raw.X
    else:
        x = adata.X

    if sp.issparse(x):
        x = x.toarray()

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expression matrix must be 2D, got shape {x.shape}")
    return x


MOSTA_BIN50_UM_PER_UNIT = 25.0  # Stereo-seq bin50: 50 DNBs × 0.5 µm/DNB pitch
# CosMx SMI global pixel coords (AtoMx ReadMe / NanoString): 120 nm/px edge → 0.12028 µm/px
COSMX_UM_PER_UNIT = 0.12028


def _detect_coordinate_um_per_unit(
    adata: ad.AnnData,
    *,
    h5ad_path: Optional[str] = None,
) -> Optional[float]:
    """Try to detect microns-per-coordinate-unit from file metadata.

    Priority:
    - ``uns['stereo_seq']['coordinate_um_per_unit']`` (or legacy ``uns['mosta']``)
    - ``uns['spatial'][library]['metadata']['coordinate_um_per_unit']``
    - Visium ``uns['spatial']`` scalefactors
    - MOSTA processed ``*.MOSTA.h5ad`` filenames (bin50 grid → 25 µm/unit)
    - CosMx NSCLC ``*cosmx*`` / ``*nsclc*`` h5ad paths (global px → 0.12028 µm/unit)

    Returns ``None`` when no recognisable scale entry is found.
    """
    for meta_key in ("stereo_seq", "mosta"):
        stereo_meta = adata.uns.get(meta_key)
        if isinstance(stereo_meta, dict) and "coordinate_um_per_unit" in stereo_meta:
            try:
                val = float(stereo_meta["coordinate_um_per_unit"])
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass

    if h5ad_path is not None and str(h5ad_path).endswith(".MOSTA.h5ad"):
        return MOSTA_BIN50_UM_PER_UNIT

    if h5ad_path is not None:
        path_lower = str(h5ad_path).lower()
        if "cosmx" in path_lower or "nsclc" in path_lower:
            return COSMX_UM_PER_UNIT

    sp = adata.uns.get("spatial") if "spatial" in adata.uns else None
    if not isinstance(sp, dict):
        return None
    for lib_val in sp.values():
        if not isinstance(lib_val, dict):
            continue
        metadata = lib_val.get("metadata")
        if isinstance(metadata, dict) and "coordinate_um_per_unit" in metadata:
            try:
                val = float(metadata["coordinate_um_per_unit"])
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
        sf = lib_val.get("scalefactors")
        if not isinstance(sf, dict):
            continue
        if "microns_per_pixel" in sf:
            try:
                val = float(sf["microns_per_pixel"])
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
        if "spot_diameter_fullres" in sf:
            try:
                spot_px = float(sf["spot_diameter_fullres"])
                if spot_px > 0:
                    return 55.0 / spot_px
            except (TypeError, ValueError):
                pass
    return None


def _uns_keys_from_io_registry_error(error: BaseException) -> list[str]:
    """Parse ``/uns/<key>`` paths mentioned in an anndata ``IORegistryError``."""
    keys: list[str] = []
    for match in re.finditer(r"from (/uns/[^/\s]+)", str(error)):
        key = match.group(1).split("/")[-1]
        if key not in keys:
            keys.append(key)
    return keys


def _strip_uns_entries(
    h5ad_path: str,
    *,
    uns_keys_to_remove: Optional[list[str]] = None,
    remove_all_uns: bool = False,
) -> None:
    """Remove unreadable ``uns`` entries from ``h5ad_path`` in place."""
    import h5py

    with h5py.File(h5ad_path, "a") as f:
        if "uns" not in f:
            return
        if remove_all_uns:
            del f["uns"]
        elif uns_keys_to_remove:
            for key in uns_keys_to_remove:
                if key in f["uns"]:
                    del f["uns"][key]


def _safe_read_h5ad(h5ad_path: str) -> ad.AnnData:
    """Read h5ad, cleaning unreadable ``uns`` entries on failure.

    Some scanpy-written files store ``uns/log1p/base`` with ``encoding_type='null'``, which
    older anndata builds cannot decode. When that happens we delete the offending ``uns`` keys
    from the file in place and retry (our loader only needs ``X``, ``obs``, ``obsm``, ``var``).
    """
    try:
        return ad.read_h5ad(h5ad_path)
    except Exception as e:
        if "IORegistryError" not in type(e).__name__:
            raise

        remove_keys = _uns_keys_from_io_registry_error(e)
        if "log1p" not in remove_keys:
            remove_keys.append("log1p")

        warnings.warn(
            f"anndata could not read {h5ad_path} ({e}); "
            f"removing uns keys in place and retrying: {remove_keys}"
        )
        _strip_uns_entries(h5ad_path, uns_keys_to_remove=remove_keys)
        try:
            return ad.read_h5ad(h5ad_path)
        except Exception as e2:
            if "IORegistryError" not in type(e2).__name__:
                raise
            warnings.warn(
                f"anndata still could not read {h5ad_path} ({e2}); "
                "removing all uns entries in place and retrying."
            )
            _strip_uns_entries(h5ad_path, remove_all_uns=True)
            return ad.read_h5ad(h5ad_path)


def _apply_obs_subset(
    adata: ad.AnnData,
    *,
    obs_filters: Optional[dict] = None,
    obs_numeric_filters: Optional[dict] = None,
    obs_indices: Optional[str] = None,
    obs_drop_na: Optional[list[str]] = None,
) -> ad.AnnData:
    """Subset ``adata`` by global indices, obs equality/numeric filters, and non-null columns."""
    mask = np.ones(adata.n_obs, dtype=bool)

    if obs_indices is not None:
        idx = np.load(obs_indices)
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"obs_indices file is empty: {obs_indices}")
        index_mask = np.zeros(adata.n_obs, dtype=bool)
        index_mask[idx] = True
        mask &= index_mask

    if obs_filters:
        for col, val in obs_filters.items():
            if col not in adata.obs.columns:
                raise ValueError(
                    f"obs_filters key '{col}' not in adata.obs; "
                    f"available: {list(adata.obs.columns)}"
            )
            mask &= adata.obs[col].astype(str).to_numpy() == str(val)

    if obs_numeric_filters:
        allowed_ops = {"gt", "ge", "gte", "lt", "le", "lte", "eq", "ne"}
        for col, spec in obs_numeric_filters.items():
            if col not in adata.obs.columns:
                raise ValueError(
                    f"obs_numeric_filters key '{col}' not in adata.obs; "
                    f"available: {list(adata.obs.columns)}"
                )
            if not isinstance(spec, dict) or not spec:
                raise ValueError(
                    f"obs_numeric_filters['{col}'] must be a non-empty mapping "
                    "such as {'gt': 0.8}"
                )
            unknown_ops = set(spec.keys()) - allowed_ops
            if unknown_ops:
                raise ValueError(
                    f"obs_numeric_filters['{col}'] has unsupported operators "
                    f"{sorted(unknown_ops)}; allowed: {sorted(allowed_ops)}"
                )
            values = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(dtype=np.float64)
            col_mask = np.isfinite(values)
            for op, threshold_raw in spec.items():
                threshold = float(threshold_raw)
                if op == "gt":
                    col_mask &= values > threshold
                elif op in {"ge", "gte"}:
                    col_mask &= values >= threshold
                elif op == "lt":
                    col_mask &= values < threshold
                elif op in {"le", "lte"}:
                    col_mask &= values <= threshold
                elif op == "eq":
                    col_mask &= values == threshold
                elif op == "ne":
                    col_mask &= values != threshold
            mask &= col_mask

    if obs_drop_na:
        for col in obs_drop_na:
            if col not in adata.obs.columns:
                raise ValueError(
                    f"obs_drop_na key '{col}' not in adata.obs; "
                    f"available: {list(adata.obs.columns)}"
                )
            mask &= adata.obs[col].notna().to_numpy()

    if mask.all():
        return adata
    if not mask.any():
        raise ValueError(
            "Obs subset matched no cells "
            f"(obs_filters={obs_filters}, obs_numeric_filters={obs_numeric_filters}, "
            f"obs_indices={obs_indices}, obs_drop_na={obs_drop_na})"
        )

    sub = adata[mask]
    if getattr(sub, "isbacked", False):
        return sub.to_memory()
    return sub.copy()


def _compile_gene_exclusion_patterns(patterns: Optional[list[str]]) -> list[re.Pattern[str]]:
    if not patterns:
        return []
    return [re.compile(str(pattern)) for pattern in patterns]


def _apply_gene_exclusions(
    adata: ad.AnnData,
    patterns: Optional[list[str]],
) -> tuple[ad.AnnData, dict]:
    compiled = _compile_gene_exclusion_patterns(patterns)
    if not compiled:
        return adata, {
            "exclude_gene_patterns": None,
            "excluded_gene_count": 0,
            "excluded_gene_names": [],
        }

    var_names = np.asarray([str(name) for name in adata.var_names], dtype=object)
    exclude_mask = np.zeros(var_names.shape[0], dtype=bool)
    for regex in compiled:
        exclude_mask |= np.fromiter(
            (bool(regex.search(str(name))) for name in var_names),
            dtype=bool,
            count=var_names.shape[0],
        )
    if exclude_mask.all():
        raise ValueError(
            "data.exclude_gene_patterns removed all genes; "
            f"patterns={patterns}"
        )
    if not exclude_mask.any():
        return adata, {
            "exclude_gene_patterns": [str(pattern) for pattern in patterns],
            "excluded_gene_count": 0,
            "excluded_gene_names": [],
        }

    kept = ~exclude_mask
    excluded = [str(name) for name in var_names[exclude_mask].tolist()]
    filtered = adata[:, kept].copy()
    return filtered, {
        "exclude_gene_patterns": [str(pattern) for pattern in patterns],
        "excluded_gene_count": int(exclude_mask.sum()),
        "excluded_gene_names": excluded,
    }


def load_h5ad_dataset(
    *,
    h5ad_path: str,
    spatial_key: str = "spatial",
    obs_x_col: Optional[str] = None,
    obs_y_col: Optional[str] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
    min_cells_per_gene: int = 0,
    top_var_genes: int = 0,
    exclude_gene_patterns: Optional[list[str]] = None,
    normalize_total: bool = False,
    log1p: bool = False,
    standardize_expression: bool = True,
    q: Optional[int] = None,
    max_cells: Optional[int] = None,
    seed: int = 0,
    cell_type=False,
    cell_type_key: str = "cell_type",
    min_cells_per_celltype: int = 1,
    covariate_obs_key: Optional[str] = None,
    compute_total_counts_covariate: bool = False,
    covariate_whitening_obs_key: Optional[str] = None,
    obs_filters: Optional[dict] = None,
    obs_numeric_filters: Optional[dict] = None,
    obs_indices: Optional[str] = None,
    obs_drop_na: Optional[list[str]] = None,
) -> DatasetBundle:
    needs_subset = bool(obs_filters or obs_numeric_filters or obs_indices or obs_drop_na)
    if needs_subset:
        adata = ad.read_h5ad(h5ad_path, backed="r")
        adata = _apply_obs_subset(
            adata,
            obs_filters=obs_filters,
            obs_numeric_filters=obs_numeric_filters,
            obs_indices=obs_indices,
            obs_drop_na=obs_drop_na,
        )
    else:
        adata = _safe_read_h5ad(h5ad_path)
    if getattr(adata, "isbacked", False):
        adata = adata.to_memory()
    adata, gene_exclusion_meta = _apply_gene_exclusions(adata, exclude_gene_patterns)
    # In cell_type="separate" mode every expression statistic (HVG dispersion,
    # gene support, z-score mean/std) must be computed *within each cell type*,
    # not across the pooled multi-type matrix.  Defer HVG selection and all
    # expression transforms to per-cell-type processing (see
    # methods.permutation._process_single_celltype_separate); keep raw counts here.
    defer_preprocessing = isinstance(cell_type, str) and cell_type == "separate"
    if top_var_genes and int(top_var_genes) > 0 and not defer_preprocessing:
        n_top = int(top_var_genes)
        if n_top >= adata.n_vars:
            warnings.warn(
                f"data.top_var_genes={n_top} >= number of available genes ({adata.n_vars}); "
                "keeping all genes."
            )
        else:
            hvg_mask = _select_hvg_mask(adata, n_top)
            adata = adata[:, hvg_mask].copy()
    s = _extract_coordinates(
        adata,
        spatial_key=spatial_key,
        obs_x_col=obs_x_col,
        obs_y_col=obs_y_col,
    )
    a = _extract_expression(adata, layer=layer, use_raw=use_raw)

    if s.shape[0] != a.shape[0]:
        raise ValueError(
            f"Coordinate rows ({s.shape[0]}) do not match expression rows ({a.shape[0]})."
        )

    covariate_values: Optional[np.ndarray] = None
    if compute_total_counts_covariate:
        from data.transforms import total_counts_covariate_values

        covariate_values = total_counts_covariate_values(a)
        covariate_obs_key = "total_counts"
    elif covariate_obs_key is not None:
        if covariate_obs_key not in adata.obs.columns:
            raise ValueError(
                f"test.covariate key '{covariate_obs_key}' not found in adata.obs columns. "
                f"Available obs columns: {list(adata.obs.columns)}"
            )
        covariate_values = np.asarray(adata.obs[covariate_obs_key].values, dtype=np.float32)

    covariate_whitening_values: Optional[np.ndarray] = None
    if covariate_whitening_obs_key is not None:
        if covariate_whitening_obs_key not in adata.obs.columns:
            raise ValueError(
                f"data.covariate_whitening obs key '{covariate_whitening_obs_key}' "
                f"not found in adata.obs columns. "
                f"Available obs columns: {list(adata.obs.columns)}"
            )
        covariate_whitening_values = np.asarray(
            adata.obs[covariate_whitening_obs_key].values, dtype=np.float32
        )

    calicost_tumor_proportion_values: Optional[np.ndarray] = None
    if "calicost_tumor_proportion" in adata.obs.columns:
        calicost_tumor_proportion_values = np.asarray(
            adata.obs["calicost_tumor_proportion"].values, dtype=np.float32
        )

    cell_type_labels: Optional[np.ndarray] = None
    cell_type_names: Optional[list] = None
    cell_type_mode = "none"
    if cell_type is True or (isinstance(cell_type, str) and cell_type in ("together", "separate")):
        cell_type_mode = "separate" if cell_type == "separate" else "together"
        if cell_type_key not in adata.obs.columns:
            raise ValueError(
                f"data.cell_type_key '{cell_type_key}' not found in adata.obs columns. "
                f"Available: {list(adata.obs.columns)}"
            )
        raw_labels = adata.obs[cell_type_key].values
        unique_types = sorted(set(str(v) for v in raw_labels))
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        cell_type_labels = np.array([type_to_idx[str(v)] for v in raw_labels], dtype=np.int64)
        cell_type_names = unique_types

        if min_cells_per_celltype > 1:
            counts = np.bincount(cell_type_labels, minlength=len(cell_type_names))
            keep_types = [i for i, cnt in enumerate(counts) if cnt >= min_cells_per_celltype]
            if not keep_types:
                raise ValueError(
                    f"No cell types have >= {min_cells_per_celltype} cells"
                )
            keep_mask = np.isin(cell_type_labels, keep_types)
            s = s[keep_mask]
            a = a[keep_mask]
            if covariate_values is not None:
                covariate_values = covariate_values[keep_mask]
            if covariate_whitening_values is not None:
                covariate_whitening_values = covariate_whitening_values[keep_mask]
            if calicost_tumor_proportion_values is not None:
                calicost_tumor_proportion_values = calicost_tumor_proportion_values[keep_mask]
            old_to_new = {old: new for new, old in enumerate(keep_types)}
            cell_type_labels = np.array(
                [old_to_new[v] for v in cell_type_labels[keep_mask]], dtype=np.int64
            )
            cell_type_names = [cell_type_names[i] for i in keep_types]

    # When not in cell-type training mode, still load labels for plotting if the key exists.
    plot_cell_type_labels: Optional[np.ndarray] = None
    plot_cell_type_names: Optional[list] = None
    if cell_type_mode == "none" and cell_type_key in adata.obs.columns:
        raw_plot_labels = adata.obs[cell_type_key].values
        unique_plot_types = sorted(set(str(v) for v in raw_plot_labels))
        plot_type_to_idx = {t: i for i, t in enumerate(unique_plot_types)}
        plot_cell_type_labels = np.array(
            [plot_type_to_idx[str(v)] for v in raw_plot_labels], dtype=np.int64
        )
        plot_cell_type_names = unique_plot_types

    if max_cells is not None and max_cells < s.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(s.shape[0], size=max_cells, replace=False)
        s = s[idx]
        a = a[idx]
        if cell_type_labels is not None:
            cell_type_labels = cell_type_labels[idx]
        if plot_cell_type_labels is not None:
            plot_cell_type_labels = plot_cell_type_labels[idx]
        if covariate_values is not None:
            covariate_values = covariate_values[idx]
        if covariate_whitening_values is not None:
            covariate_whitening_values = covariate_whitening_values[idx]
        if calicost_tumor_proportion_values is not None:
            calicost_tumor_proportion_values = calicost_tumor_proportion_values[idx]

    if defer_preprocessing:
        # Keep raw counts; per-cell-type preprocessing happens downstream.  Mark
        # every gene as retained so var_names lines up with the raw matrix.
        a = np.asarray(a, dtype=np.float32)
        transform_meta = {
            "gene_keep_mask": np.ones(a.shape[1], dtype=bool),
            "representation": "raw_counts_deferred",
        }
    elif cell_type_mode == "separate" and cell_type_labels is not None and q is not None:
        a, transform_meta = apply_expression_transforms_by_celltype(
            a,
            cell_type_labels,
            min_cells_per_gene=min_cells_per_gene,
            log1p=log1p,
            standardize_expression=standardize_expression,
            q=q,
            seed=seed,
            return_metadata=True,
        )
    else:
        a, transform_meta = apply_expression_transforms(
            a,
            min_cells_per_gene=min_cells_per_gene,
            normalize_total=normalize_total,
            log1p=log1p,
            standardize_expression=standardize_expression,
            q=q,
            seed=seed,
            return_metadata=True,
        )

    raw_var_names = np.asarray(adata.var_names, dtype=object)
    keep_mask = np.asarray(transform_meta["gene_keep_mask"], dtype=bool)
    if "feature_names" in transform_meta:
        feature_names = [str(name) for name in transform_meta["feature_names"]]
    else:
        feature_names = [str(name) for name in raw_var_names[keep_mask]]

    coordinate_um_per_unit = _detect_coordinate_um_per_unit(adata, h5ad_path=h5ad_path)

    meta = {
        "source": "h5ad",
        "h5ad": h5ad_path,
        "spatial_key": spatial_key,
        "obs_filters": obs_filters,
        "obs_numeric_filters": obs_numeric_filters,
        "obs_indices": obs_indices,
        "obs_drop_na": obs_drop_na,
        "obs_x_col": obs_x_col,
        "obs_y_col": obs_y_col,
        "layer": layer,
        "use_raw": use_raw,
        "cell_type_key": str(cell_type_key),
        "min_cells_per_gene": int(min_cells_per_gene),
        "top_var_genes": int(top_var_genes),
        "exclude_gene_patterns": gene_exclusion_meta["exclude_gene_patterns"],
        "excluded_gene_count": int(gene_exclusion_meta["excluded_gene_count"]),
        "excluded_gene_names": list(gene_exclusion_meta["excluded_gene_names"]),
        "normalize_total": bool(normalize_total),
        "log1p": bool(log1p),
        "standardize_expression": bool(standardize_expression),
        "q": None if q is None else int(q),
        "max_cells": None if max_cells is None else int(max_cells),
        "seed": int(seed),
        "feature_space": str(transform_meta["representation"]),
        "var_names": feature_names,
    }
    if coordinate_um_per_unit is not None:
        meta["coordinate_um_per_unit"] = float(coordinate_um_per_unit)
    if transform_meta.get("q_by_celltype"):
        meta["q_by_celltype"] = True
    if defer_preprocessing:
        # Parameters the per-cell-type path needs (these live on DataConfig, which
        # the permutation code does not otherwise see).
        meta["separate_preprocessing"] = {
            "min_cells_per_gene": int(min_cells_per_gene),
            "top_var_genes": int(top_var_genes),
            "normalize_total": bool(normalize_total),
            "log1p": bool(log1p),
            "standardize_expression": bool(standardize_expression),
            "q": None if q is None else int(q),
            "seed": int(seed),
        }
    if cell_type_labels is not None and cell_type_names is not None:
        meta["cell_type_labels"] = cell_type_labels
        meta["cell_type_names"] = cell_type_names
        meta["n_cell_types"] = len(cell_type_names)
        meta["cell_type_mode"] = cell_type_mode
    if covariate_values is not None:
        meta["covariate_values"] = np.asarray(covariate_values, dtype=np.float32)
        meta["covariate_obs_key"] = covariate_obs_key
    if covariate_whitening_values is not None:
        meta["covariate_whitening_values"] = np.asarray(
            covariate_whitening_values, dtype=np.float32
        )
        meta["covariate_whitening_obs_key"] = covariate_whitening_obs_key
    if calicost_tumor_proportion_values is not None:
        meta["calicost_tumor_proportion"] = np.asarray(
            calicost_tumor_proportion_values, dtype=np.float32
        )
    if plot_cell_type_labels is not None and plot_cell_type_names is not None:
        meta["plot_cell_type_labels"] = plot_cell_type_labels
        meta["plot_cell_type_names"] = plot_cell_type_names
    return DatasetBundle(S=s, A=a, meta=meta).validate()


def load_h5ad_as_permutation_dataset(**kwargs) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_h5ad_dataset(**kwargs)
    return dataset.S, dataset.A


def load_dataset_from_config(
    config: DataConfig,
    *,
    covariate_obs_key: Optional[str] = None,
    compute_total_counts_covariate: bool = False,
    covariate_whitening_obs_key: Optional[str] = None,
) -> DatasetBundle:
    config.validate()
    if config.source != "h5ad":
        raise ValueError(f"load_dataset_from_config only supports h5ad source, got {config.source}")
    return load_h5ad_dataset(
        h5ad_path=config.h5ad,
        spatial_key=config.spatial_key,
        obs_x_col=config.obs_x_col,
        obs_y_col=config.obs_y_col,
        layer=config.layer,
        use_raw=config.use_raw,
        min_cells_per_gene=config.min_cells_per_gene,
        top_var_genes=config.top_var_genes,
        exclude_gene_patterns=config.exclude_gene_patterns,
        normalize_total=config.normalize_total,
        log1p=config.log1p,
        standardize_expression=config.standardize_expression,
        q=config.q,
        max_cells=config.max_cells,
        seed=config.seed,
        cell_type=config.cell_type,
        cell_type_key=config.cell_type_key,
        min_cells_per_celltype=config.min_cells_per_celltype,
        covariate_obs_key=covariate_obs_key,
        compute_total_counts_covariate=compute_total_counts_covariate,
        covariate_whitening_obs_key=covariate_whitening_obs_key,
        obs_filters=config.obs_filters,
        obs_numeric_filters=config.obs_numeric_filters,
        obs_indices=config.obs_indices,
        obs_drop_na=config.obs_drop_na,
    )
