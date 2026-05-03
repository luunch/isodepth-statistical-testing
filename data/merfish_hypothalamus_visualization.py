from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from analysis.plots import (
    expression_colormap_for_index,
    save_gene_spatial_contour_grid,
    save_multi_gene_spatial_expression_grid,
    save_spatial_binned_density_plot,
    save_spatial_pointcloud_kde_plot,
    save_spatial_principal_axes_plot,
)
from data import load_h5ad_dataset
from data.schemas import DataConfig, DatasetBundle


def normalize_gene_token(name: str) -> str:
    return str(name).split("|")[0].strip().upper()


def resolve_gene_column(
    var_names: Sequence[str],
    requested: str,
) -> tuple[str, int] | None:
    """Map a requested symbol to ``(resolved_name, column_index)`` in ``var_names``."""
    for i, vn in enumerate(var_names):
        if str(vn) == requested:
            return str(vn), i
    req_norm = normalize_gene_token(requested)
    for i, vn in enumerate(var_names):
        if normalize_gene_token(vn) == req_norm:
            return str(vn), i
    return None


def spatial_coordinate_views(
    S: np.ndarray,
    n_perms: int,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Same shuffle semantics as ``_build_permuted_coordinate_batch`` on CPU."""
    S = np.asarray(S, dtype=np.float32)
    n = int(S.shape[0])
    device = torch.device("cpu")
    s_t = torch.tensor(S, dtype=torch.float32, device=device)
    views: list[tuple[str, np.ndarray]] = [("True", S.copy())]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for k in range(int(n_perms)):
        perm = torch.randperm(n, generator=generator)
        s_perm = np.asarray(s_t[perm].numpy(), dtype=np.float32)
        views.append((f"Permutation {k + 1}", s_perm))
    return views


def resolve_genes_for_dataset(
    dataset: DatasetBundle,
    requested_genes: Sequence[str],
) -> tuple[dict[str, tuple[str, int]], list[str]]:
    var_names = dataset.meta.get("var_names")
    if not var_names:
        raise ValueError("dataset.meta must contain 'var_names'")
    resolved: dict[str, tuple[str, int]] = {}
    missing: list[str] = []
    for req in requested_genes:
        hit = resolve_gene_column(var_names, req)
        if hit is not None:
            resolved[req] = hit
        else:
            missing.append(req)
    return resolved, missing


def run_merfish_hypothalamus_visualization(
    dataset: DatasetBundle,
    *,
    requested_genes: Sequence[str],
    n_perms: int,
    seed: int,
    out_dir: Path,
    show_contours: bool = True,
    hide_zero_expression: bool = False,
    single_combined_plot: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_map, missing = resolve_genes_for_dataset(dataset, requested_genes)
    for gene in missing:
        warnings.warn(f"Gene not found in dataset after transforms: {gene!r}", stacklevel=2)

    spatial_views = spatial_coordinate_views(dataset.S, n_perms=n_perms, seed=seed)

    summary: dict[str, Any] = {
        "requested_genes": list(requested_genes),
        "resolved": {k: {"var_name": v[0], "column_index": v[1]} for k, v in resolved_map.items()},
        "missing": missing,
        "n_perms": int(n_perms),
        "seed": int(seed),
        "n_cells": int(dataset.n_cells),
        "show_contours": bool(show_contours),
        "hide_zero_expression": bool(hide_zero_expression),
        "single_combined_plot": bool(single_combined_plot),
        "plots": [],
    }

    pc_path = out_dir / "spatial_principal_coordinates.png"
    save_spatial_principal_axes_plot(dataset.S, pc_path)
    summary["spatial_principal_coordinates_plot"] = str(pc_path)
    kde_path = out_dir / "spatial_pointcloud_kde.png"
    save_spatial_pointcloud_kde_plot(dataset.S, kde_path)
    summary["spatial_pointcloud_kde_plot"] = str(kde_path)
    binned_density_path = out_dir / "spatial_binned_density.png"
    save_spatial_binned_density_plot(dataset.S, binned_density_path)
    summary["spatial_binned_density_plot"] = str(binned_density_path)

    if single_combined_plot and resolved_map:
        combined_requested = list(resolved_map.keys())
        combined_labels = [f"{req} ({resolved_map[req][0]})" for req in combined_requested]
        combined_matrix = np.column_stack(
            [np.asarray(dataset.A[:, resolved_map[req][1]], dtype=np.float32) for req in combined_requested]
        )
        ztag = "_nozero" if hide_zero_expression else ""
        combined_out = out_dir / f"genes_relative_expression_grid{ztag}.png"
        save_multi_gene_spatial_expression_grid(
            dataset.S,
            combined_matrix,
            combined_labels,
            combined_out,
            show_contours=show_contours,
            hide_zero_expression=hide_zero_expression,
            figure_title="Relative expression by gene (true spatial coordinates)",
        )
        summary["plots"].append(str(combined_out))
    else:
        for gene_idx, (requested, (var_name, col_idx)) in enumerate(resolved_map.items()):
            z = np.asarray(dataset.A[:, col_idx], dtype=np.float32)
            if hide_zero_expression:
                # Keep in sync with analysis.plots._EXPRESSION_ZERO_EPS
                nz = np.abs(z) > 1e-15
                z_scale = z[nz]
                if z_scale.size:
                    vmin = float(np.min(z_scale))
                    vmax = float(np.max(z_scale))
                else:
                    vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.min(z))
                vmax = float(np.max(z))
            if vmax <= vmin:
                vmax = vmin + 1e-8

            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in requested)
            suffix = "contours" if show_contours else "scatter"
            ztag = "_nozero" if hide_zero_expression else ""
            out_png = out_dir / f"gene_{safe}_{suffix}{ztag}.png"
            save_gene_spatial_contour_grid(
                spatial_views,
                z,
                out_png,
                vmin=vmin,
                vmax=vmax,
                colorbar_label="Expression",
                figure_title=f"{requested} ({var_name})",
                show_contours=show_contours,
                hide_zero_expression=hide_zero_expression,
                cmap=expression_colormap_for_index(gene_idx),
            )
            summary["plots"].append(str(out_png))

    summary_path = out_dir / "resolved_genes.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote summary to {summary_path}")
    print(f"  spatial PCs: {summary['spatial_principal_coordinates_plot']}")
    print(f"  spatial KDE: {summary['spatial_pointcloud_kde_plot']}")
    print(f"  spatial binned density: {summary['spatial_binned_density_plot']}")
    for path in summary["plots"]:
        print(f"  plot: {path}")

    return summary


def load_data_config_from_mapping(mapping: Mapping[str, Any]) -> DataConfig:
    payload = dict(mapping.get("data", {}))
    return DataConfig(**payload).validate()


def default_visualization_section() -> dict[str, Any]:
    return {
        "n_perms": 5,
        "genes": ["Syt2", "Ano3", "Etv1", "Man1a", "Htr2c"],
        "out_dir": "results/merfishhypothalamus_visualization",
    }


def load_dataset_for_visualization(data_cfg: DataConfig) -> DatasetBundle:
    return load_h5ad_dataset(
        h5ad_path=data_cfg.h5ad or "",
        spatial_key=data_cfg.spatial_key,
        obs_x_col=data_cfg.obs_x_col,
        obs_y_col=data_cfg.obs_y_col,
        layer=data_cfg.layer,
        use_raw=data_cfg.use_raw,
        min_cells_per_gene=data_cfg.min_cells_per_gene,
        log1p=data_cfg.log1p,
        standardize=data_cfg.standardize,
        q=data_cfg.q,
        max_cells=data_cfg.max_cells,
        seed=data_cfg.seed,
    )


def load_visualization_payload(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, config_path
