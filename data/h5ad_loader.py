from __future__ import annotations

import warnings
from typing import Optional

import anndata as ad
import numpy as np
import scipy.sparse as sp

from data.schemas import DataConfig, DatasetBundle
from data.transforms import apply_expression_transforms


DEFAULT_OBS_COORD_CANDIDATES = [
    ("x", "y"),
    ("X", "Y"),
    ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ("array_row", "array_col"),
]


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


def _safe_read_h5ad(h5ad_path: str) -> ad.AnnData:
    """Read h5ad, cleaning unreadable uns entries on failure."""
    try:
        return ad.read_h5ad(h5ad_path)
    except Exception as e:
        if "IORegistryError" not in type(e).__name__:
            raise
        import h5py
        warnings.warn(
            f"anndata could not read {h5ad_path} ({e}); "
            "stripping unreadable uns entries and retrying."
        )
        with h5py.File(h5ad_path, "a") as f:
            if "uns" in f:
                for key in list(f["uns"].keys()):
                    if key.startswith("_"):
                        del f["uns"][key]
        return ad.read_h5ad(h5ad_path)


def load_h5ad_dataset(
    *,
    h5ad_path: str,
    spatial_key: str = "spatial",
    obs_x_col: Optional[str] = None,
    obs_y_col: Optional[str] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
    min_cells_per_gene: int = 0,
    log1p: bool = False,
    standardize_expression: bool = True,
    q: Optional[int] = None,
    max_cells: Optional[int] = None,
    seed: int = 0,
    cell_type=False,
    cell_type_key: str = "cell_type",
    min_cells_per_celltype: int = 1,
) -> DatasetBundle:
    adata = _safe_read_h5ad(h5ad_path)
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
            old_to_new = {old: new for new, old in enumerate(keep_types)}
            cell_type_labels = np.array(
                [old_to_new[v] for v in cell_type_labels[keep_mask]], dtype=np.int64
            )
            cell_type_names = [cell_type_names[i] for i in keep_types]

    if max_cells is not None and max_cells < s.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(s.shape[0], size=max_cells, replace=False)
        s = s[idx]
        a = a[idx]
        if cell_type_labels is not None:
            cell_type_labels = cell_type_labels[idx]

    a, transform_meta = apply_expression_transforms(
        a,
        min_cells_per_gene=min_cells_per_gene,
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

    meta = {
        "source": "h5ad",
        "h5ad": h5ad_path,
        "spatial_key": spatial_key,
        "obs_x_col": obs_x_col,
        "obs_y_col": obs_y_col,
        "layer": layer,
        "use_raw": use_raw,
        "min_cells_per_gene": int(min_cells_per_gene),
        "log1p": bool(log1p),
        "standardize_expression": bool(standardize_expression),
        "q": None if q is None else int(q),
        "max_cells": None if max_cells is None else int(max_cells),
        "seed": int(seed),
        "feature_space": str(transform_meta["representation"]),
        "var_names": feature_names,
    }
    if cell_type_labels is not None and cell_type_names is not None:
        meta["cell_type_labels"] = cell_type_labels
        meta["cell_type_names"] = cell_type_names
        meta["n_cell_types"] = len(cell_type_names)
        meta["cell_type_mode"] = cell_type_mode
    return DatasetBundle(S=s, A=a, meta=meta).validate()


def load_h5ad_as_permutation_dataset(**kwargs) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_h5ad_dataset(**kwargs)
    return dataset.S, dataset.A


def load_dataset_from_config(config: DataConfig) -> DatasetBundle:
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
        log1p=config.log1p,
        standardize_expression=config.standardize_expression,
        q=config.q,
        max_cells=config.max_cells,
        seed=config.seed,
        cell_type=config.cell_type,
        cell_type_key=config.cell_type_key,
        min_cells_per_celltype=config.min_cells_per_celltype,
    )
