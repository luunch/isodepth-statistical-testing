from __future__ import annotations

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


def load_h5ad_dataset(
    *,
    h5ad_path: str,
    spatial_key: str = "spatial",
    obs_x_col: Optional[str] = None,
    obs_y_col: Optional[str] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
    min_cells_per_gene: int = 0,
    standardize: bool = True,
    max_cells: Optional[int] = None,
    seed: int = 0,
) -> DatasetBundle:
    adata = ad.read_h5ad(h5ad_path)
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

    if max_cells is not None and max_cells < s.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(s.shape[0], size=max_cells, replace=False)
        s = s[idx]
        a = a[idx]

    a = apply_expression_transforms(
        a,
        min_cells_per_gene=min_cells_per_gene,
        standardize=standardize,
    )

    meta = {
        "source": "h5ad",
        "h5ad": h5ad_path,
        "spatial_key": spatial_key,
        "obs_x_col": obs_x_col,
        "obs_y_col": obs_y_col,
        "layer": layer,
        "use_raw": use_raw,
        "min_cells_per_gene": int(min_cells_per_gene),
        "standardize": bool(standardize),
        "max_cells": None if max_cells is None else int(max_cells),
        "seed": int(seed),
        "var_names": list(map(str, adata.var_names[: a.shape[1]])),
    }
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
        standardize=config.standardize,
        max_cells=config.max_cells,
        seed=config.seed,
    )
