"""Overlay blurred-permutation losses on an existing null-distribution plot.

Does not recompute p-values. Reconstructs the first ``n_blur`` coordinate
permutations from the original ``test.seed``, applies a Gaussian blur of
each requested ``sigma_um`` on the shuffled layout, trains those slots with
the original isodepth hyperparameters, and draws the resulting losses on
the matching metric-distribution plot. CLI defaults remain 20 µm and 50 µm;
``--from-config`` reads a run JSON's ``blurred_permutations`` block instead
(10/20/50 µm × 20 in the historical-dataset configs).

Together mode trains the shared-encoder / per-type-decoder model and can
blur within cell type. Separate mode preprocesses and trains each retained
type independently. The 60 extra losses are effect-size overlays only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")

from analysis.plots import save_combined_celltype_metric_distribution, save_metric_distribution_plot
from data import load_dataset, raw_coordinates_from_standardized
from data.schemas import DatasetBundle, TestResult, run_config_from_mapping
from experiments.configuration import (
    _resolve_config_relative_paths,
    _resolve_covariate_label,
    load_json_config,
)
from methods.block_permutation import resolve_um_per_unit
from methods.metrics import canonicalize_metric_name, gaussian_mse_from_nll
from methods.permutation import (
    _build_permuted_coordinate_batch,
    _preprocess_separate_subset,
)
from methods.spatial_blur import (
    DEFAULT_BLUR_CUTOFF_MULT,
    blur_expression_slots_within_groups,
)
from methods.trainers.gpu_selection import resolve_device
from methods.trainers.isodepth import (
    train_celltype_parallel_isodepth_model,
    train_parallel_isodepth_model,
)

DEFAULT_CONFIGS = (
    "configs/jfanmerfish/jfan_raw_B1_coord_perm.json",
    "configs/jfanmerfish/jfan_raw_B2_coord_perm.json",
    "configs/jfanmerfish/jfan_raw_B3_coord_perm.json",
)
DEFAULT_SIGMAS_UM = (20.0, 50.0)
DEFAULT_N_BLUR = 20
OVERLAY_COLORS = ("seagreen", "darkorange", "mediumorchid")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=None,
        help="Run config JSON (repeatable). Defaults to the three raw JFan B1/B2/B3 coordinate-permutation configs.",
    )
    parser.add_argument(
        "--sigma-um",
        type=float,
        action="append",
        dest="sigmas_um",
        default=None,
        help="Gaussian blur σ in µm (repeatable). Defaults to 20 and 50.",
    )
    parser.add_argument("--n-blur", type=int, default=DEFAULT_N_BLUR)
    parser.add_argument(
        "--cutoff-mult",
        type=float,
        default=DEFAULT_BLUR_CUTOFF_MULT,
        help="Gaussian cutoff in units of σ (default 4).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training and redraw from a previously saved sidecar JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the sidecar JSON already has that σ.",
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help=(
            "Read sigmas/count/cutoff from the config's blurred_permutations block. "
            "If the block is absent or disabled, skip the config."
        ),
    )
    return parser.parse_args()


def _load_run_config(config_path: Path):
    raw = _resolve_config_relative_paths(load_json_config(str(config_path)), str(config_path))
    return run_config_from_mapping(raw)


def _result_json_path(run_config) -> Path:
    return Path(run_config.output.out_dir) / run_config.output.run_name / (
        f"{run_config.output.run_name}_result.json"
    )


def _sidecar_path(run_config) -> Path:
    return Path(run_config.output.out_dir) / run_config.output.run_name / (
        f"{run_config.output.run_name}_blurred_perm_losses.json"
    )


def _metric_plot_path(run_config) -> Path:
    return Path(run_config.output.out_dir) / run_config.output.run_name / (
        f"{run_config.output.run_name}_metric_distribution.png"
    )


def _load_result_metrics(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "method_name": str(payload["method_name"]),
        "metric": str(payload["metric"]),
        "p_value": float(payload["p_value"]),
        "stat_true": float(payload["stat_true"]),
        "stat_perm": np.asarray(payload["stat_perm"], dtype=np.float64),
        "n_cells": int(payload["n_cells"]),
        "n_genes": int(payload["n_genes"]),
        "runtime_sec": float(payload.get("runtime_sec", 0.0)),
    }


def _median_nn_um(S_um: np.ndarray) -> float:
    from scipy.spatial import KDTree

    if S_um.shape[0] < 2:
        return float("nan")
    distances, _ = KDTree(S_um).query(S_um, k=2)
    return float(np.median(np.asarray(distances, dtype=np.float64)[:, 1]))


def _neighbor_fraction(S_um: np.ndarray, radius_um: float) -> float:
    from scipy.spatial import KDTree

    if S_um.shape[0] == 0:
        return 0.0
    tree = KDTree(S_um)
    counts = [len(nbrs) - 1 for nbrs in tree.query_ball_point(S_um, r=float(radius_um))]
    return float(np.mean(np.asarray(counts, dtype=np.float64) > 0))


def _sigma_close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) < 1e-9


def _normalize_sidecar(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"overlays": []}
    overlays = payload.get("overlays")
    if isinstance(overlays, list):
        return payload
    if "stat_blurred_perm" in payload:
        return {
            "n_blur": payload.get("n_blur"),
            "seed": payload.get("seed"),
            "cutoff_mult": payload.get("cutoff_mult"),
            "coordinate_um_per_unit": payload.get("coordinate_um_per_unit"),
            "median_nn_um": payload.get("median_nn_um"),
            "perm_indices": payload.get("perm_indices"),
            "overlays": [
                {
                    "sigma_um": payload.get("sigma_um"),
                    "n_blur": payload.get("n_blur"),
                    "cutoff_mult": payload.get("cutoff_mult"),
                    "frac_cells_with_neighbor_within_cutoff": payload.get(
                        "frac_cells_with_neighbor_within_cutoff"
                    ),
                    "stat_blurred_perm": payload.get("stat_blurred_perm"),
                    "stat_blurred_perm_mean": payload.get("stat_blurred_perm_mean"),
                    "stat_blurred_perm_std": payload.get("stat_blurred_perm_std"),
                }
            ],
        }
    return {"overlays": []}


def _overlay_for_sigma(sidecar: dict[str, Any], sigma_um: float) -> dict[str, Any] | None:
    for entry in sidecar.get("overlays") or []:
        if _sigma_close(float(entry.get("sigma_um", -1.0)), sigma_um) and entry.get("stat_blurred_perm"):
            return entry
    return None


def _upsert_overlay(sidecar: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    overlays = [
        item
        for item in (sidecar.get("overlays") or [])
        if not _sigma_close(float(item.get("sigma_um", -1.0)), float(entry["sigma_um"]))
    ]
    overlays.append(entry)
    overlays.sort(key=lambda item: float(item["sigma_um"]))
    sidecar = dict(sidecar)
    sidecar["overlays"] = overlays
    sidecar["n_blur"] = entry.get("n_blur", sidecar.get("n_blur"))
    sidecar["seed"] = entry.get("seed", sidecar.get("seed"))
    sidecar["cutoff_mult"] = entry.get("cutoff_mult", sidecar.get("cutoff_mult"))
    sidecar["coordinate_um_per_unit"] = entry.get(
        "coordinate_um_per_unit", sidecar.get("coordinate_um_per_unit")
    )
    sidecar["median_nn_um"] = entry.get("median_nn_um", sidecar.get("median_nn_um"))
    sidecar["perm_indices"] = entry.get("perm_indices", sidecar.get("perm_indices"))
    return sidecar


def blur_settings_from_run_config(run_config) -> dict[str, Any] | None:
    blur = getattr(run_config, "blurred_permutations", None)
    if blur is None or not bool(getattr(blur, "enabled", True)):
        return None
    return {
        "sigmas_um": [float(value) for value in blur.sigmas_um],
        "n_blur": int(blur.n_per_sigma),
        "cutoff_mult": float(blur.cutoff_mult),
        "within_cell_type": bool(blur.within_cell_type),
    }


def _run_dir(run_config) -> Path:
    return Path(run_config.output.out_dir) / run_config.output.run_name


def _safe_type_name(type_name: str) -> str:
    return str(type_name).replace(" ", "_").replace("/", "_")


def _payload_metrics(
    payload: dict[str, Any],
    *,
    type_summary: dict[str, Any] | None = None,
    n_genes: int | None = None,
) -> dict[str, Any]:
    artifacts = payload.get("artifacts") or {}
    source = type_summary if type_summary is not None else {**artifacts, **payload}
    metrics = {
        "method_name": str(payload["method_name"]),
        "metric": str(payload["metric"]),
        "p_value": float(source["p_value"]),
        "stat_true": float(source["stat_true"]),
        "stat_perm": np.asarray(source["stat_perm"], dtype=np.float64),
        "n_cells": int(source.get("n_cells", payload["n_cells"])),
        "n_genes": int(
            n_genes if n_genes is not None else source.get("n_genes", payload["n_genes"])
        ),
        "runtime_sec": float(payload.get("runtime_sec", 0.0)),
    }
    if source.get("stat_covariate") is not None:
        metrics["stat_covariate"] = float(source["stat_covariate"])
    if source.get("p_value_covariate") is not None:
        metrics["p_value_covariate"] = float(source["p_value_covariate"])
    return metrics


def _separate_subset_bundle(
    dataset: DatasetBundle, type_index: int
) -> tuple[DatasetBundle, np.ndarray]:
    labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    mask = labels == int(type_index)
    S_original = np.asarray(dataset.S[mask], dtype=np.float32)
    A_subset, var_names, feature_space = _preprocess_separate_subset(
        dataset, dataset.A[mask], int(type_index)
    )
    mean = S_original.mean(axis=0)
    std = S_original.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    S_subset = np.asarray((S_original - mean) / safe_std, dtype=np.float32)
    raw_full = raw_coordinates_from_standardized(dataset.S, dataset.meta)
    raw_subset = np.asarray(raw_full[mask], dtype=np.float32)
    meta = dict(dataset.meta)
    meta.pop("cell_type_labels", None)
    meta.pop("cell_type_names", None)
    meta.pop("n_cell_types", None)
    meta.pop("cell_type_mode", None)
    meta["coord_mean"] = np.asarray(mean, dtype=np.float32)
    meta["coord_std"] = np.asarray(safe_std, dtype=np.float32)
    if var_names is not None:
        meta["var_names"] = list(var_names)
    if feature_space is not None:
        meta["feature_space"] = feature_space
    return DatasetBundle(S=S_subset, A=A_subset, meta=meta).validate(), raw_subset


@dataclass
class OverlayTarget:
    label: str
    plot_path: Path
    sidecar_path: Path
    metrics: dict[str, Any]
    seed: int
    dataset: DatasetBundle | None = None
    raw_coordinates: np.ndarray | None = None
    training_cell_type_labels: np.ndarray | None = None
    n_cell_types: int | None = None
    blur_group_labels: np.ndarray | None = None


def _bind_training_data(
    target: OverlayTarget,
    *,
    dataset: DatasetBundle,
    raw_coordinates: np.ndarray | None,
    training_cell_type_labels: np.ndarray | None,
    n_cell_types: int | None,
    blur_group_labels: np.ndarray | None,
) -> OverlayTarget:
    target.dataset = dataset
    target.raw_coordinates = raw_coordinates
    target.training_cell_type_labels = training_cell_type_labels
    target.n_cell_types = n_cell_types
    target.blur_group_labels = blur_group_labels
    if dataset is not None:
        target.metrics = dict(target.metrics)
        target.metrics["n_cells"] = int(dataset.n_cells)
        target.metrics["n_genes"] = int(dataset.n_genes)
    return target


def build_overlay_targets(
    run_config,
    result_payload: dict[str, Any],
    *,
    dataset: DatasetBundle | None = None,
    within_cell_type: bool = True,
) -> list[OverlayTarget]:
    mode = run_config.data.cell_type_mode
    run_dir = _run_dir(run_config)
    run_name = run_config.output.run_name
    if mode == "separate":
        names = list(
            (result_payload.get("artifacts") or {}).get("cell_type_names")
            or (dataset.meta.get("cell_type_names") if dataset is not None else None)
            or []
        )
        if not names:
            raise ValueError("separate-mode overlay needs artifacts.cell_type_names")
        summaries = (result_payload.get("artifacts") or {}).get("per_type_summaries") or {}
        dataset_names = (
            [str(name) for name in dataset.meta.get("cell_type_names") or []]
            if dataset is not None
            else []
        )
        targets: list[OverlayTarget] = []
        for type_name in names:
            summary = summaries.get(type_name)
            if summary is None:
                raise ValueError(
                    f"Missing artifacts.per_type_summaries[{type_name!r}] in result JSON"
                )
            safe_name = _safe_type_name(type_name)
            type_dir = run_dir / safe_name
            type_index = dataset_names.index(type_name) if type_name in dataset_names else None
            target = OverlayTarget(
                label=str(type_name),
                plot_path=type_dir / f"{safe_name}_metric_distribution.png",
                sidecar_path=type_dir / f"{safe_name}_blurred_perm_losses.json",
                metrics=_payload_metrics(result_payload, type_summary=summary),
                seed=int(run_config.test.seed)
                + (int(type_index) if type_index is not None else 0),
            )
            if dataset is not None:
                if type_name not in dataset_names:
                    raise ValueError(
                        f"Result cell type {type_name!r} is missing from the reloaded dataset"
                    )
                subset, raw_coords = _separate_subset_bundle(dataset, int(type_index))
                _bind_training_data(
                    target,
                    dataset=subset,
                    raw_coordinates=raw_coords,
                    training_cell_type_labels=None,
                    n_cell_types=None,
                    blur_group_labels=None,
                )
                target.seed = int(run_config.test.seed) + int(type_index)
            targets.append(target)
        return targets

    together = mode == "together"
    labels = None
    n_cell_types = None
    blur_groups = None
    if dataset is not None:
        labels = dataset.meta.get("cell_type_labels")
        n_cell_types = dataset.meta.get("n_cell_types")
        if together:
            if labels is None or n_cell_types is None:
                raise ValueError(
                    "together-mode overlay requires cell_type_labels in dataset.meta"
                )
            labels = np.asarray(labels, dtype=np.int64)
        if within_cell_type and labels is not None:
            blur_groups = np.asarray(labels, dtype=np.int64)
    target = OverlayTarget(
        label=run_name,
        plot_path=_metric_plot_path(run_config),
        sidecar_path=_sidecar_path(run_config),
        metrics=_payload_metrics(
            result_payload,
            n_genes=None if dataset is None else dataset.n_genes,
        ),
        seed=int(run_config.test.seed),
    )
    if dataset is not None:
        _bind_training_data(
            target,
            dataset=dataset,
            raw_coordinates=None,
            training_cell_type_labels=labels if together else None,
            n_cell_types=int(n_cell_types) if together else None,
            blur_group_labels=blur_groups,
        )
    return [target]


def _train_blurred_perm_losses(
    dataset: DatasetBundle,
    run_config,
    *,
    n_blur: int,
    sigma_um: float,
    cutoff_mult: float,
    raw_coordinates: np.ndarray | None = None,
    training_cell_type_labels: np.ndarray | None = None,
    n_cell_types: int | None = None,
    blur_group_labels: np.ndarray | None = None,
    seed: int | None = None,
    model_label: str | None = None,
) -> dict[str, Any]:
    test = run_config.test
    device = resolve_device(test.device)
    if device.type != "cuda":
        raise RuntimeError(
            f"Blur-overlay training requires CUDA, got {device}. "
            "Activate isodepth_env on a GPU node."
        )
    print(
        f"Training device: {device} ({torch.cuda.get_device_name(device)})",
        flush=True,
    )
    um_per_unit = resolve_um_per_unit(
        test.coordinate_um_per_unit,
        dataset.meta.get("coordinate_um_per_unit"),
    )
    effective_seed = int(test.seed if seed is None else seed)
    s_batched, permutations = _build_permuted_coordinate_batch(
        dataset.S,
        n_perms=int(n_blur),
        seed=effective_seed,
        device=torch.device("cpu"),
    )
    s_perm = np.asarray(s_batched[1:], dtype=np.float32)
    s_true_raw = (
        np.asarray(raw_coordinates, dtype=np.float32)
        if raw_coordinates is not None
        else raw_coordinates_from_standardized(dataset.S, dataset.meta)
    )
    s_perm_raw = np.stack([s_true_raw[p] for p in permutations], axis=0)
    s_perm_um = s_perm_raw.astype(np.float64) * float(um_per_unit)
    a_batched = blur_expression_slots_within_groups(
        s_perm_um,
        dataset.A,
        sigma_um,
        cutoff_mult=cutoff_mult,
        group_labels=blur_group_labels,
    )
    s_true_um = np.asarray(s_true_raw, dtype=np.float64) * float(um_per_unit)
    train_config = replace(
        test,
        covariate=None,
        n_perms=int(n_blur),
        record_loss_history=False,
        verbose=True,
    )
    train_kwargs = {
        "S": dataset.S,
        "A": dataset.A,
        "config": train_config,
        "device": device,
        "s_batched": s_perm,
        "a_batched": a_batched,
        "model_label": model_label
        or f"{n_blur} × {sigma_um:g} µm-blurred permutations",
    }
    if training_cell_type_labels is None:
        _, outputs, _ = train_parallel_isodepth_model(**train_kwargs)
    else:
        if n_cell_types is None:
            raise ValueError("n_cell_types is required for together cell-type training")
        _, outputs, _ = train_celltype_parallel_isodepth_model(
            **train_kwargs,
            cell_type_labels=np.asarray(training_cell_type_labels, dtype=np.int64),
            n_cell_types=int(n_cell_types),
        )
    losses = np.asarray(outputs.model_metrics, dtype=np.float64)
    r_max = float(cutoff_mult) * float(sigma_um)
    return {
        "sigma_um": float(sigma_um),
        "n_blur": int(n_blur),
        "cutoff_mult": float(cutoff_mult),
        "seed": effective_seed,
        "coordinate_um_per_unit": float(um_per_unit),
        "median_nn_um": _median_nn_um(s_true_um),
        "frac_cells_with_neighbor_within_cutoff": _neighbor_fraction(s_true_um, r_max),
        "perm_indices": [np.asarray(p, dtype=np.int64).tolist() for p in permutations],
        "stat_blurred_perm": [float(x) for x in losses.tolist()],
        "stat_blurred_perm_mean": float(np.mean(losses)),
        "stat_blurred_perm_std": float(np.std(losses)),
    }


def _n_total_for_overlay(target: OverlayTarget) -> int | None:
    if target.dataset is not None:
        return int(target.dataset.n_cells) * int(target.dataset.n_genes)
    npz_path = target.plot_path.with_name(
        target.plot_path.name.replace("_metric_distribution.png", "_isodepths.npz")
    )
    if not npz_path.is_file():
        return None
    with np.load(npz_path) as handle:
        if "A" not in handle.files:
            return None
        shape = np.asarray(handle["A"]).shape
    return int(shape[0]) * int(shape[1])


def _align_metrics_to_overlay_scale(
    metrics: dict[str, Any],
    overlay_entries: list[dict[str, Any]],
    n_total: int | None,
) -> dict[str, Any]:
    """Put legacy NLL result stats on the current MSE overlay axis."""
    if canonicalize_metric_name(str(metrics["metric"])) != "nll_gaussian_mse":
        return metrics
    overlay_vals = np.concatenate(
        [
            np.asarray(entry["stat_blurred_perm"], dtype=np.float64)
            for entry in overlay_entries
        ]
    )
    obs = float(metrics["stat_true"])
    if obs < 10.0 or float(np.max(np.abs(overlay_vals))) > 10.0:
        return metrics
    if n_total is None or int(n_total) <= 0:
        raise ValueError(
            "Cannot convert legacy NLL result stats to MSE without n_cells * n_genes"
        )
    aligned = dict(metrics)
    aligned["stat_true"] = float(gaussian_mse_from_nll(obs, n_total))
    aligned["stat_perm"] = np.asarray(
        gaussian_mse_from_nll(np.asarray(metrics["stat_perm"], dtype=np.float64), n_total),
        dtype=np.float64,
    )
    if metrics.get("stat_covariate") is not None:
        aligned["stat_covariate"] = float(
            gaussian_mse_from_nll(float(metrics["stat_covariate"]), n_total)
        )
    aligned["n_genes"] = int(n_total) // max(int(metrics["n_cells"]), 1)
    return aligned


def _overlay_plot_tuples(
    overlay_entries: list[dict[str, Any]],
) -> list[tuple[np.ndarray, str, str]]:
    overlays: list[tuple[np.ndarray, str, str]] = []
    for index, entry in enumerate(overlay_entries):
        color = OVERLAY_COLORS[index % len(OVERLAY_COLORS)]
        overlays.append(
            (
                np.asarray(entry["stat_blurred_perm"], dtype=np.float64),
                f"{float(entry['sigma_um']):g} µm blurred perms",
                color,
            )
        )
    return overlays


def _write_overlay_plot(
    metrics: dict[str, Any],
    overlay_entries: list[dict[str, Any]],
    plot_path: Path,
    *,
    covariate_label: str | None = None,
) -> Path:
    artifacts: dict[str, Any] = {}
    if metrics.get("stat_covariate") is not None and metrics.get("p_value_covariate") is not None:
        artifacts["stat_covariate"] = float(metrics["stat_covariate"])
        artifacts["p_value_covariate"] = float(metrics["p_value_covariate"])
        if covariate_label:
            artifacts["covariate_label"] = covariate_label
    result = TestResult(
        method_name=str(metrics["method_name"]),
        metric=str(metrics["metric"]),
        p_value=float(metrics["p_value"]),
        stat_true=float(metrics["stat_true"]),
        stat_perm=np.asarray(metrics["stat_perm"], dtype=np.float64),
        runtime_sec=float(metrics.get("runtime_sec", 0.0)),
        n_cells=int(metrics["n_cells"]),
        n_genes=int(metrics["n_genes"]),
        config={},
        artifacts=artifacts,
    ).validate()
    backup = plot_path.with_name(plot_path.stem + "_pre_blur_overlay.png")
    if plot_path.exists() and not backup.exists():
        shutil.copy2(plot_path, backup)
    return save_metric_distribution_plot(
        result, plot_path, overlays=_overlay_plot_tuples(overlay_entries)
    )


def _process_overlay_target(
    target: OverlayTarget,
    run_config,
    *,
    sigmas_um: list[float],
    n_blur: int,
    cutoff_mult: float,
    plot_only: bool,
    force: bool,
) -> Path:
    sidecar = _normalize_sidecar(
        json.loads(target.sidecar_path.read_text(encoding="utf-8"))
        if target.sidecar_path.is_file()
        else None
    )
    for sigma_um in sigmas_um:
        existing = None if force else _overlay_for_sigma(sidecar, sigma_um)
        if existing is not None:
            print(
                f"Reusing sidecar {sigma_um:g} µm overlay for {target.label}",
                flush=True,
            )
            continue
        if plot_only:
            raise FileNotFoundError(
                f"--plot-only missing {sigma_um:g} µm overlay in {target.sidecar_path}"
            )
        if target.dataset is None:
            raise RuntimeError(
                f"Dataset was not loaded before training blur overlays for {target.label}"
            )
        print(
            f"Training {n_blur} blurred permutation slots for {target.label} "
            f"(σ={sigma_um:g} µm, n_cells={target.dataset.n_cells}, "
            f"n_genes={target.dataset.n_genes}) ...",
            flush=True,
        )
        entry = _train_blurred_perm_losses(
            target.dataset,
            run_config,
            n_blur=n_blur,
            sigma_um=sigma_um,
            cutoff_mult=cutoff_mult,
            raw_coordinates=target.raw_coordinates,
            training_cell_type_labels=target.training_cell_type_labels,
            n_cell_types=target.n_cell_types,
            blur_group_labels=target.blur_group_labels,
            seed=target.seed,
            model_label=(
                f"{n_blur} × {sigma_um:g} µm-blurred permutations ({target.label})"
            ),
        )
        sidecar = _upsert_overlay(sidecar, entry)
        target.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        target.sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        print(f"Wrote {target.sidecar_path} (σ={sigma_um:g} µm)", flush=True)

    overlay_entries = []
    for sigma_um in sigmas_um:
        entry = _overlay_for_sigma(sidecar, sigma_um)
        if entry is None:
            raise FileNotFoundError(
                f"Missing {sigma_um:g} µm overlay in {target.sidecar_path}"
            )
        overlay_entries.append(entry)

    aligned_metrics = _align_metrics_to_overlay_scale(
        target.metrics,
        overlay_entries,
        _n_total_for_overlay(target),
    )
    _write_overlay_plot(
        aligned_metrics,
        overlay_entries,
        target.plot_path,
        covariate_label=_resolve_covariate_label(run_config),
    )
    parts = [
        f"σ={float(entry['sigma_um']):g} mean={float(np.mean(entry['stat_blurred_perm'])):.6g}"
        for entry in overlay_entries
    ]
    print(
        f"Updated {target.plot_path}  |  observed={aligned_metrics['stat_true']:.6g}  "
        f"{'  '.join(parts)}  p={target.metrics['p_value']:.4g} (unchanged)",
        flush=True,
    )
    return target.plot_path


def _sidecar_overlay_entries(
    target: OverlayTarget, sigmas_um: list[float]
) -> list[dict[str, Any]]:
    sidecar = _normalize_sidecar(
        json.loads(target.sidecar_path.read_text(encoding="utf-8"))
        if target.sidecar_path.is_file()
        else None
    )
    entries: list[dict[str, Any]] = []
    for sigma_um in sigmas_um:
        entry = _overlay_for_sigma(sidecar, sigma_um)
        if entry is None:
            raise FileNotFoundError(
                f"Missing {sigma_um:g} µm overlay in {target.sidecar_path}"
            )
        entries.append(entry)
    return entries


def _write_combined_separate_overlay_plot(
    run_config,
    payload: dict[str, Any],
    targets: list[OverlayTarget],
    sigmas_um: list[float],
) -> Path | None:
    """Redraw the separate-mode combined grid with per-type requested-σ overlays."""
    names = [target.label for target in targets]
    summaries = (payload.get("artifacts") or {}).get("per_type_summaries") or {}
    if not names:
        return None
    per_type: dict[str, dict[str, Any]] = {}
    overlays_by_type: dict[str, list[tuple[np.ndarray, str, str]]] = {}
    for target in targets:
        summary = summaries.get(target.label) or {}
        entries = _sidecar_overlay_entries(target, sigmas_um)
        metrics = _align_metrics_to_overlay_scale(
            _payload_metrics(payload, type_summary=summary),
            entries,
            _n_total_for_overlay(target),
        )
        type_row: dict[str, Any] = {
            "stat_perm": metrics["stat_perm"],
            "stat_true": metrics["stat_true"],
            "p_value": metrics["p_value"],
            "n_cells": metrics["n_cells"],
        }
        if metrics.get("stat_covariate") is not None:
            type_row["stat_covariate"] = metrics["stat_covariate"]
        if metrics.get("p_value_covariate") is not None:
            type_row["p_value_covariate"] = metrics["p_value_covariate"]
        per_type[target.label] = type_row
        overlays_by_type[target.label] = _overlay_plot_tuples(entries)

    combined = _run_dir(run_config) / (
        f"{run_config.output.run_name}_combined_metric_distribution.png"
    )
    backup = combined.with_name(combined.stem + "_pre_blur_overlay.png")
    if combined.exists() and not backup.exists():
        shutil.copy2(combined, backup)
    save_combined_celltype_metric_distribution(
        per_type,
        names,
        combined,
        metric=str(payload["metric"]),
        covariate_label=_resolve_covariate_label(run_config),
        overlays_by_type=overlays_by_type,
    )
    print(f"Updated combined overlay plot {combined}", flush=True)
    return combined


def process_config(
    config_path: Path,
    *,
    sigmas_um: list[float],
    n_blur: int,
    cutoff_mult: float,
    plot_only: bool,
    force: bool,
    within_cell_type: bool = True,
) -> list[Path]:
    run_config = _load_run_config(config_path)
    result_path = _result_json_path(run_config)
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Missing result JSON (run the existence test first): {result_path}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    planned = build_overlay_targets(
        run_config,
        payload,
        within_cell_type=within_cell_type,
    )
    needs_training = False
    if not plot_only:
        if force:
            needs_training = True
        else:
            for target in planned:
                sidecar = _normalize_sidecar(
                    json.loads(target.sidecar_path.read_text(encoding="utf-8"))
                    if target.sidecar_path.is_file()
                    else None
                )
                if any(
                    _overlay_for_sigma(sidecar, sigma_um) is None for sigma_um in sigmas_um
                ):
                    needs_training = True
                    break

    dataset = None
    if needs_training:
        print(f"Loading dataset for {config_path.name} ...", flush=True)
        dataset = load_dataset(run_config.data, covariate=run_config.test.covariate)
    targets = build_overlay_targets(
        run_config,
        payload,
        dataset=dataset,
        within_cell_type=within_cell_type,
    )
    plot_paths = [
        _process_overlay_target(
            target,
            run_config,
            sigmas_um=sigmas_um,
            n_blur=n_blur,
            cutoff_mult=cutoff_mult,
            plot_only=plot_only,
            force=force,
        )
        for target in targets
    ]
    if run_config.data.cell_type_mode == "separate" and len(targets) > 1:
        combined = _write_combined_separate_overlay_plot(
            run_config, payload, targets, sigmas_um
        )
        if combined is not None:
            plot_paths.append(combined)
    elif len(plot_paths) == 1:
        combined = _run_dir(run_config) / (
            f"{run_config.output.run_name}_combined_metric_distribution.png"
        )
        if plot_paths[0] != combined:
            shutil.copy2(plot_paths[0], combined)
            print(f"Copied overlay onto {combined}", flush=True)
    return plot_paths


def main() -> None:
    args = _parse_args()
    config_values = args.configs or list(DEFAULT_CONFIGS)
    for raw_path in config_values:
        config_path = Path(raw_path)
        if not config_path.is_absolute():
            config_path = (REPO / config_path).resolve()
        run_config = _load_run_config(config_path)
        if args.from_config:
            settings = blur_settings_from_run_config(run_config)
            if settings is None:
                print(
                    f"Skipping {config_path.name}: no enabled blurred_permutations block",
                    flush=True,
                )
                continue
            sigmas_um = settings["sigmas_um"]
            n_blur = settings["n_blur"]
            cutoff_mult = settings["cutoff_mult"]
            within_cell_type = settings["within_cell_type"]
        else:
            sigmas_um = [float(x) for x in (args.sigmas_um or list(DEFAULT_SIGMAS_UM))]
            n_blur = int(args.n_blur)
            cutoff_mult = float(args.cutoff_mult)
            within_cell_type = True
        process_config(
            config_path,
            sigmas_um=sigmas_um,
            n_blur=n_blur,
            cutoff_mult=cutoff_mult,
            plot_only=bool(args.plot_only),
            force=bool(args.force),
            within_cell_type=bool(within_cell_type),
        )


if __name__ == "__main__":
    main()
