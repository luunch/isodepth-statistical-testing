"""Encoder-source × decoder-loss grid after matched pretrain (diagnostic).

Works from a JSON spec for any single-region h5ad load (e.g. hypothalamus MERFISH)
or a DLPFC Layer-3 prepared h5ad (``data.prepared_h5ad``).

Phase A — encoder pretrain (500 epochs total, encoder never frozen except when
we snapshot for Phase B):

  1. **poisson_500** — 500 epochs Poisson NLL (encoder+decoder).
  2. **gauss100_poiss400** — 100 epochs Gaussian MSE on log-CPM z-scored counts,
     then 400 epochs Poisson NLL with the **same Adam state and minibatch order**
     (Gaussian initialization for Poisson; encoder stays trainable throughout).
  3. **gaussian_500** — 500 epochs Gaussian MSE (encoder+decoder).

Phase B — freeze encoder snapshot, decoder-only (500 epochs) under Gaussian or
Poisson loss.  SGD batch size 128 throughout.

Usage:
    python scripts/dlpfc_layer3_encoder_loss_grid.py
    python scripts/dlpfc_layer3_encoder_loss_grid.py \\
        --spec configs/experiments/hypothalamus_encoder_loss_grid.json
    python scripts/dlpfc_layer3_encoder_loss_grid.py --n-reruns 10  # smoke test
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import optim

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.plots import _plot_spatial_isodepth  # noqa: E402
from data import load_dataset  # noqa: E402
from data.schemas import DataConfig  # noqa: E402
from data.spatial_regions import dbscan_middle_region_mask  # noqa: E402
from data.transforms import gaussian_log_cpm_targets_from_counts  # noqa: E402
from methods.architectures import ParallelIsoDepthNet  # noqa: E402
from methods.trainers.gpu_selection import resolve_device  # noqa: E402
from methods.trainers.isodepth import (  # noqa: E402
    _compute_reconstruction_loss_per_model,
    _set_torch_seed,
)
from scripts.dlpfc_layer3_gaussian_axis_poisson import (  # noqa: E402
    _data_config_from_spec,
    _resolve,
    prepare_layer3_h5ad,
)
from scripts.dlpfc_layer3_gaussian_poisson_schedule import (  # noqa: E402
    _expand_coords,
    _new_adam_and_batch_gen,
    _poisson_size_factors,
    _select_rerun,
    _sgd_epoch,
    _train_poisson_free_tail_with_history,
)
from scripts.liver_lobule_sweep import load_spec  # noqa: E402

DEFAULT_SPEC = REPO / "configs/experiments/dlpfc_layer3_encoder_loss_grid.json"

EncoderSource = Literal["poisson_500", "gauss100_poiss400", "gaussian_500"]
DecoderLoss = Literal["gaussian", "poisson"]

ENCODER_LABELS: dict[EncoderSource, str] = {
    "poisson_500": "500 ep Poisson",
    "gauss100_poiss400": "100 ep Gauss → 400 ep Poiss",
    "gaussian_500": "500 ep Gaussian",
}

ENCODER_COLORS: dict[EncoderSource, str] = {
    "poisson_500": "#2166ac",
    "gauss100_poiss400": "#762a83",
    "gaussian_500": "#b2182b",
}


def _h5ad_config_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _build_data_config_from_spec(spec: dict) -> DataConfig:
    """Load config for a direct h5ad or a DLPFC Layer-3 prepared h5ad."""
    data = spec["data"]
    pp = spec.get("preprocessing", {})
    if data.get("prepared_h5ad"):
        prepared = prepare_layer3_h5ad(spec, force=False)
        return _data_config_from_spec(data, pp, prepared)

    h5ad_path = _resolve(data["h5ad"])
    cell_type = data.get("cell_type", False)
    return DataConfig(
        source="h5ad",
        h5ad=_h5ad_config_path(h5ad_path),
        spatial_key=str(data.get("spatial_key", "spatial")),
        obs_x_col=data.get("obs_x_col"),
        obs_y_col=data.get("obs_y_col"),
        layer=data.get("layer"),
        use_raw=bool(data.get("use_raw", False)),
        min_cells_per_gene=int(pp.get("min_cells_per_gene", data.get("min_cells_per_gene", 0))),
        top_var_genes=int(pp.get("top_var_genes", data.get("top_var_genes", 0))),
        normalize_total=bool(pp.get("normalize_total", data.get("normalize_total", False))),
        log1p=bool(pp.get("log1p", data.get("log1p", False))),
        standardize_expression=bool(
            pp.get("standardize_expression", data.get("standardize_expression", False))
        ),
        standardize_coordinates=bool(
            pp.get("standardize_coordinates", data.get("standardize_coordinates", True))
        ),
        cell_type=cell_type,
        cell_type_key=str(data.get("cell_type_key", "cell_type")),
        min_cells_per_celltype=int(data.get("min_cells_per_celltype", 1)),
        obs_filters=data.get("obs_filters"),
        obs_numeric_filters=data.get("obs_numeric_filters"),
        obs_indices=data.get("obs_indices"),
        obs_drop_na=data.get("obs_drop_na"),
        max_cells=data.get("max_cells"),
        seed=int(data.get("seed", pp.get("seed", 42))),
    ).validate()


def _load_grid_arrays(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = _build_data_config_from_spec(spec)
    dataset = load_dataset(data_cfg)
    return (
        np.asarray(dataset.S, dtype=np.float32),
        np.asarray(dataset.A, dtype=np.float32),
    )


def _standardize_spatial_subset(spatial: np.ndarray) -> np.ndarray:
    spatial_sub = np.asarray(spatial, dtype=np.float32)
    mean = spatial_sub.mean(axis=0)
    std = spatial_sub.std(axis=0)
    return np.asarray(
        (spatial_sub - mean) / np.where(std > 1e-8, std, 1.0),
        dtype=np.float32,
    )


def _apply_spatial_subset(
    spatial: np.ndarray,
    counts: np.ndarray,
    spec: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    subset_cfg = spec.get("spatial_subset")
    if not subset_cfg:
        return spatial, counts, None

    mode = str(subset_cfg.get("mode", "dbscan_middle"))
    if mode != "dbscan_middle":
        raise ValueError(
            f"spatial_subset.mode must be 'dbscan_middle' (got '{mode}')"
        )

    min_cells = int(subset_cfg.get("min_cells", 50))
    mask, meta = dbscan_middle_region_mask(
        spatial,
        eps=subset_cfg.get("eps"),
        eps_mult=float(subset_cfg.get("eps_mult", 3.0)),
        min_samples=int(subset_cfg.get("min_samples", 10)),
        min_cells=min_cells,
        axis=str(subset_cfg.get("axis", "pc1")),
    )
    n_kept = int(meta["n_cells_after"])
    if n_kept < min_cells:
        raise ValueError(
            f"spatial_subset kept only {n_kept} cells (min_cells={min_cells})"
        )

    spatial_sub = _standardize_spatial_subset(spatial[mask])
    counts_sub = np.asarray(counts[mask], dtype=np.float32)
    print(
        f"[grid] spatial_subset DBSCAN middle region (cluster "
        f"{meta['selected_cluster']}, {len(meta['cluster_sizes'])} valid): "
        f"{meta['n_cells_before']} → {n_kept} cells",
        flush=True,
    )
    return spatial_sub, counts_sub, meta


def _plot_dbscan_subset(
    spatial: np.ndarray,
    subset_meta: dict[str, Any],
    out_path: Path,
    *,
    title: str,
) -> None:
    cluster_ids = np.asarray(subset_meta["cluster_ids"], dtype=np.int64)
    selected = int(subset_meta["selected_cluster"])
    valid_clusters = sorted(int(c) for c in subset_meta["cluster_sizes"])

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if int(subset_meta.get("noise_cells", 0)) > 0:
        noise = cluster_ids == -1
        ax.scatter(
            spatial[noise, 0],
            spatial[noise, 1],
            s=8,
            c="0.75",
            alpha=0.5,
            linewidths=0,
            label="noise",
        )
    for i, cluster in enumerate(valid_clusters):
        m = cluster_ids == cluster
        color = "#d62728" if cluster == selected else cmap(i % 10)
        size = 14 if cluster == selected else 8
        label = f"r{valid_clusters.index(cluster)} (n={int(m.sum())})"
        if cluster == selected:
            label += " [middle]"
        ax.scatter(
            spatial[m, 0],
            spatial[m, 1],
            s=size,
            c=[color],
            alpha=0.85 if cluster == selected else 0.55,
            linewidths=0,
            label=label,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("spatial x (standardized)")
    ax.set_ylabel("spatial y (standardized)")
    ax.set_title(f"{title}\nDBSCAN regions — middle cluster {selected} kept")
    ax.legend(loc="best", fontsize=8, markerscale=0.8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] wrote {out_path}", flush=True)


def _eval_gaussian_loss(
    model: ParallelIsoDepthNet,
    gaussian_targets_t: torch.Tensor,
    *,
    coords_t: torch.Tensor,
    fixed_latent_t: torch.Tensor | None = None,
) -> np.ndarray:
    with torch.no_grad():
        if fixed_latent_t is not None:
            output = model.decoder(fixed_latent_t)
        else:
            output = model(coords_t)
        loss_per_model = _compute_reconstruction_loss_per_model(
            output, gaussian_targets_t, None, poisson_size_factors=None
        )
    return loss_per_model.detach().cpu().numpy().astype(np.float64)


def _eval_poisson_loss(
    model: ParallelIsoDepthNet,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    *,
    coords_t: torch.Tensor,
    fixed_latent_t: torch.Tensor | None = None,
) -> np.ndarray:
    with torch.no_grad():
        if fixed_latent_t is not None:
            output = model.decoder(fixed_latent_t)
        else:
            output = model(coords_t)
        loss_per_model = _compute_reconstruction_loss_per_model(
            output, targets_t, None, poisson_size_factors=size_factors_t
        )
    return loss_per_model.detach().cpu().numpy().astype(np.float64)


def _train_gaussian_phase_with_history(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    gaussian_targets_t: torch.Tensor,
    *,
    gaussian_epochs: int,
    lr: float,
    sgd_batch_size: int,
    seed: int,
    optimizer: optim.Optimizer | None = None,
    minibatch_generator: torch.Generator | None = None,
) -> tuple[np.ndarray, optim.Optimizer, torch.Generator]:
    n_cells = coords_t.shape[1]
    if optimizer is None or minibatch_generator is None:
        optimizer, minibatch_generator = _new_adam_and_batch_gen(model, lr=lr, seed=seed)
    history: list[np.ndarray] = []
    if gaussian_epochs <= 0:
        return np.empty((0, coords_t.shape[0]), dtype=np.float64), optimizer, minibatch_generator
    print(
        f"    Gaussian phase: {gaussian_epochs} epochs (MSE log-CPM, SGD-{sgd_batch_size})",
        flush=True,
    )
    for _ in range(gaussian_epochs):
        _sgd_epoch(
            model,
            coords_t,
            gaussian_targets_t,
            None,
            optimizer=optimizer,
            sgd_batch_size=sgd_batch_size,
            n_cells=n_cells,
            minibatch_generator=minibatch_generator,
        )
        history.append(
            _eval_gaussian_loss(model, gaussian_targets_t, coords_t=coords_t)
        )
    return np.asarray(history), optimizer, minibatch_generator


def _train_poisson_joint_with_history(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    *,
    poisson_epochs: int,
    lr: float,
    sgd_batch_size: int,
    seed: int,
    optimizer: optim.Optimizer | None = None,
    minibatch_generator: torch.Generator | None = None,
) -> tuple[np.ndarray, optim.Optimizer, torch.Generator]:
    return _train_poisson_free_tail_with_history(
        model,
        coords_t,
        targets_t,
        size_factors_t,
        poisson_epochs=poisson_epochs,
        lr=lr,
        sgd_batch_size=sgd_batch_size,
        seed=seed,
        optimizer=optimizer,
        minibatch_generator=minibatch_generator,
    )


def _train_decoder_phase(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    *,
    decoder_loss: DecoderLoss,
    gaussian_targets_t: torch.Tensor,
    poisson_targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    fixed_latent_t: torch.Tensor,
    decoder_epochs: int,
    lr: float,
    sgd_batch_size: int,
    seed: int,
) -> np.ndarray:
    n_cells = coords_t.shape[1]
    optimizer = optim.Adam(model.decoder.parameters(), lr=lr, foreach=False)
    minibatch_generator = torch.Generator(device="cpu")
    minibatch_generator.manual_seed(seed)
    print(
        f"    Decoder-only ({decoder_loss}): {decoder_epochs} epochs (SGD-{sgd_batch_size})",
        flush=True,
    )
    history: list[np.ndarray] = []
    for _ in range(decoder_epochs):
        if decoder_loss == "gaussian":
            _sgd_epoch(
                model,
                coords_t,
                gaussian_targets_t,
                None,
                optimizer=optimizer,
                sgd_batch_size=sgd_batch_size,
                n_cells=n_cells,
                minibatch_generator=minibatch_generator,
                fixed_latent_t=fixed_latent_t,
            )
            history.append(
                _eval_gaussian_loss(
                    model,
                    gaussian_targets_t,
                    coords_t=coords_t,
                    fixed_latent_t=fixed_latent_t,
                )
            )
        else:
            _sgd_epoch(
                model,
                coords_t,
                poisson_targets_t,
                size_factors_t,
                optimizer=optimizer,
                sgd_batch_size=sgd_batch_size,
                n_cells=n_cells,
                minibatch_generator=minibatch_generator,
                fixed_latent_t=fixed_latent_t,
            )
            history.append(
                _eval_poisson_loss(
                    model,
                    poisson_targets_t,
                    size_factors_t,
                    coords_t=coords_t,
                    fixed_latent_t=fixed_latent_t,
                )
            )
    return np.asarray(history)


def _pretrain_encoder(
    source: EncoderSource,
    *,
    coords_t: torch.Tensor,
    gaussian_targets_t: torch.Tensor,
    poisson_targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    n_reruns: int,
    n_genes: int,
    pretrain_epochs: int,
    warm_gaussian_epochs: int,
    lr: float,
    sgd_batch_size: int,
    decoder: str,
    seed: int,
    device: torch.device,
) -> tuple[ParallelIsoDepthNet, np.ndarray, torch.Tensor]:
    _set_torch_seed(seed)
    model = ParallelIsoDepthNet(
        n_reruns, n_genes, latent_dim=1, decoder_type=decoder
    ).to(device)

    if source == "poisson_500":
        print(f"[pretrain] {source}: {pretrain_epochs} Poisson epochs", flush=True)
        history, _, _ = _train_poisson_joint_with_history(
            model,
            coords_t,
            poisson_targets_t,
            size_factors_t,
            poisson_epochs=pretrain_epochs,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
        )
    elif source == "gauss100_poiss400":
        poisson_epochs = pretrain_epochs - warm_gaussian_epochs
        if poisson_epochs <= 0:
            raise ValueError(
                f"warm_gaussian_epochs={warm_gaussian_epochs} must be < "
                f"pretrain_epochs={pretrain_epochs}"
            )
        print(
            f"[pretrain] {source}: {warm_gaussian_epochs} Gaussian + "
            f"{poisson_epochs} Poisson (encoder free throughout)",
            flush=True,
        )
        gauss_hist, optimizer, gen = _train_gaussian_phase_with_history(
            model,
            coords_t,
            gaussian_targets_t,
            gaussian_epochs=warm_gaussian_epochs,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
        )
        poiss_hist, _, _ = _train_poisson_joint_with_history(
            model,
            coords_t,
            poisson_targets_t,
            size_factors_t,
            poisson_epochs=poisson_epochs,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
            optimizer=optimizer,
            minibatch_generator=gen,
        )
        history = np.concatenate([gauss_hist, poiss_hist], axis=0)
    elif source == "gaussian_500":
        print(f"[pretrain] {source}: {pretrain_epochs} Gaussian epochs", flush=True)
        history, _, _ = _train_gaussian_phase_with_history(
            model,
            coords_t,
            gaussian_targets_t,
            gaussian_epochs=pretrain_epochs,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown encoder source '{source}'")

    with torch.no_grad():
        fixed_latent = model.encoder(coords_t).detach()
    return model, history, fixed_latent


def _cell_key(source: EncoderSource, decoder_loss: DecoderLoss) -> str:
    return f"{source}__dec_{decoder_loss}"


def _summarize_cell(
    *,
    source: EncoderSource,
    decoder_loss: DecoderLoss,
    pretrain_history: np.ndarray,
    decoder_history: np.ndarray,
    eval_poisson: np.ndarray,
    eval_gaussian: np.ndarray,
) -> dict[str, Any]:
    train_final = decoder_history[-1]
    sel_min = _select_rerun(decoder_history, "min")
    sel_median = _select_rerun(decoder_history, "median")
    return {
        "cell_key": _cell_key(source, decoder_loss),
        "encoder_source": source,
        "decoder_loss": decoder_loss,
        "pretrain_final_median": float(np.median(pretrain_history[-1])),
        "train_final_median": float(np.median(train_final)),
        "train_final_min": float(np.min(train_final)),
        "eval_poisson_nll_median": float(np.median(eval_poisson)),
        "eval_gaussian_mse_median": float(np.median(eval_gaussian)),
        "eval_poisson_nll_at_min_train": float(eval_poisson[sel_min]),
        "eval_gaussian_mse_at_min_train": float(eval_gaussian[sel_min]),
        "min_rerun_index": int(sel_min),
        "median_rerun_index": int(sel_median),
    }


def _plot_pretrain_curves(
    histories: dict[EncoderSource, np.ndarray],
    out_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for source, history in histories.items():
        epochs = np.arange(history.shape[0])
        ax.plot(
            epochs,
            np.median(history, axis=1),
            color=ENCODER_COLORS[source],
            lw=2.0,
            label=ENCODER_LABELS[source],
        )
    ax.set_xlabel("Pretrain epoch")
    ax.set_ylabel("Training loss (median over reruns)")
    ax.set_title(f"{title} — Phase A pretrain (SGD-128)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] wrote {out_path}", flush=True)


def _plot_decoder_grid(
    decoder_histories: dict[str, np.ndarray],
    cells: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    pretrain_epochs: int,
) -> None:
    sources: list[EncoderSource] = ["poisson_500", "gauss100_poiss400", "gaussian_500"]
    losses: list[DecoderLoss] = ["gaussian", "poisson"]
    lookup = {row["cell_key"]: row for row in cells}
    fig, axes = plt.subplots(len(sources), len(losses), figsize=(11, 9), squeeze=False)
    for row, source in enumerate(sources):
        for col, dec_loss in enumerate(losses):
            ax = axes[row, col]
            key = _cell_key(source, dec_loss)
            history = decoder_histories[key]
            epochs = pretrain_epochs + np.arange(history.shape[0])
            median_curve = np.median(history, axis=1)
            ax.plot(epochs, median_curve, color=ENCODER_COLORS[source], lw=2.0)
            ax.axvline(pretrain_epochs, color="0.5", ls=":", lw=1.0)
            row_meta = lookup[key]
            ax.set_title(
                f"{ENCODER_LABELS[source]}\n"
                f"train={dec_loss}, final={row_meta['train_final_median']:.4f}",
                fontsize=9,
            )
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("Train loss (median)")
            if row == len(sources) - 1:
                ax.set_xlabel("Global epoch")
    fig.suptitle(f"{title} — Phase B decoder-only (SGD-128)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] wrote {out_path}", flush=True)


def _plot_isodepths(
    spatial: np.ndarray,
    encoder_axes: dict[EncoderSource, np.ndarray],
    cells: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> None:
    sources: list[EncoderSource] = ["poisson_500", "gauss100_poiss400", "gaussian_500"]
    fig, axes = plt.subplots(1, len(sources), figsize=(4.0 * len(sources), 4.2), squeeze=False)
    for col, source in enumerate(sources):
        poiss_row = next(
            row for row in cells
            if row["encoder_source"] == source and row["decoder_loss"] == "poisson"
        )
        slot = int(poiss_row["median_rerun_index"])
        depth = encoder_axes[source][slot]
        ax = axes[0, col]
        _plot_spatial_isodepth(
            ax,
            spatial,
            np.asarray(depth, dtype=np.float32),
            (
                f"{ENCODER_LABELS[source]}\n"
                f"Poiss dec loss={poiss_row['train_final_median']:.4f}"
            ),
        )
    fig.suptitle(
        f"{title} — frozen encoder axes (median rerun; identical across decoder losses)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] wrote {out_path}", flush=True)


def _write_cross_metric_csv(cells: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "encoder_source",
        "decoder_loss",
        "pretrain_final_median",
        "train_final_median",
        "train_final_min",
        "eval_poisson_nll_median",
        "eval_gaussian_mse_median",
        "eval_poisson_nll_at_min_train",
        "eval_gaussian_mse_at_min_train",
        "min_rerun_index",
        "median_rerun_index",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in cells:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"[grid] wrote {out_path}", flush=True)


def _write_summary(summary: dict[str, Any], out_dir: Path, run_name: str) -> None:
    json_path = out_dir / f"{run_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[grid] wrote {json_path}", flush=True)


def run(spec: dict, args: argparse.Namespace) -> dict[str, Any]:
    sched = spec["schedule"]
    device = resolve_device(args.device or sched.get("device", "cuda"))
    out_dir = _resolve(spec["output"]["out_dir"]) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrain_epochs = int(args.pretrain_epochs or sched["pretrain_epochs"])
    warm_gaussian_epochs = int(args.warm_gaussian_epochs or sched["warm_gaussian_epochs"])
    decoder_epochs = int(args.decoder_epochs or sched["decoder_epochs"])
    n_reruns = int(args.n_reruns or sched["n_reruns"])
    sgd_batch_size = int(args.sgd_batch_size or sched["sgd_batch_size"])
    lr = float(args.lr or sched["lr"])
    decoder = str(args.decoder or sched.get("decoder", "nn"))
    seed = int(args.seed or sched["seed"])

    if sgd_batch_size <= 0:
        raise ValueError("sgd_batch_size must be > 0 (this experiment requires SGD minibatches)")

    spatial_raw, counts_raw = _load_grid_arrays(spec)
    spatial, counts, spatial_subset_meta = _apply_spatial_subset(
        spatial_raw,
        counts_raw,
        spec,
    )
    n_cells, n_genes = counts.shape

    print(
        f"[grid] n={n_cells} genes={n_genes} reruns={n_reruns} "
        f"pretrain={pretrain_epochs} decoder={decoder_epochs} "
        f"warm_gaussian={warm_gaussian_epochs} sgd_batch_size={sgd_batch_size} "
        f"device={device}",
        flush=True,
    )

    coords_t = _expand_coords(spatial, n_reruns, device)
    poisson_targets_t = torch.tensor(counts, dtype=torch.float32, device=device)
    gaussian_targets_np = gaussian_log_cpm_targets_from_counts(counts)
    gaussian_targets_t = torch.tensor(gaussian_targets_np, dtype=torch.float32, device=device)
    size_factors_t = _poisson_size_factors(counts, device)

    encoder_sources: list[EncoderSource] = [
        "poisson_500",
        "gauss100_poiss400",
        "gaussian_500",
    ]
    decoder_losses: list[DecoderLoss] = ["gaussian", "poisson"]

    pretrain_histories: dict[EncoderSource, np.ndarray] = {}
    frozen_latents: dict[EncoderSource, torch.Tensor] = {}
    post_pretrain_states: dict[EncoderSource, dict[str, torch.Tensor]] = {}

    for source in encoder_sources:
        model, pretrain_hist, fixed_latent = _pretrain_encoder(
            source,
            coords_t=coords_t,
            gaussian_targets_t=gaussian_targets_t,
            poisson_targets_t=poisson_targets_t,
            size_factors_t=size_factors_t,
            n_reruns=n_reruns,
            n_genes=n_genes,
            pretrain_epochs=pretrain_epochs,
            warm_gaussian_epochs=warm_gaussian_epochs,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            decoder=decoder,
            seed=seed,
            device=device,
        )
        pretrain_histories[source] = pretrain_hist
        frozen_latents[source] = fixed_latent
        post_pretrain_states[source] = copy.deepcopy(model.state_dict())

    decoder_histories: dict[str, np.ndarray] = {}
    cells: list[dict[str, Any]] = []
    encoder_axes: dict[EncoderSource, np.ndarray] = {
        source: frozen_latents[source].detach().cpu().numpy()[:, :, 0]
        for source in encoder_sources
    }

    for source in encoder_sources:
        fixed_latent = frozen_latents[source]
        for dec_loss in decoder_losses:
            dec_model = ParallelIsoDepthNet(
                n_reruns, n_genes, latent_dim=1, decoder_type=decoder
            ).to(device)
            dec_model.load_state_dict(post_pretrain_states[source])
            print(
                f"[decoder] {source} → train decoder with {dec_loss} loss",
                flush=True,
            )
            dec_hist = _train_decoder_phase(
                dec_model,
                coords_t,
                decoder_loss=dec_loss,
                gaussian_targets_t=gaussian_targets_t,
                poisson_targets_t=poisson_targets_t,
                size_factors_t=size_factors_t,
                fixed_latent_t=fixed_latent,
                decoder_epochs=decoder_epochs,
                lr=lr,
                sgd_batch_size=sgd_batch_size,
                seed=seed,
            )
            eval_poisson = _eval_poisson_loss(
                dec_model,
                poisson_targets_t,
                size_factors_t,
                coords_t=coords_t,
                fixed_latent_t=fixed_latent,
            )
            eval_gaussian = _eval_gaussian_loss(
                dec_model,
                gaussian_targets_t,
                coords_t=coords_t,
                fixed_latent_t=fixed_latent,
            )
            key = _cell_key(source, dec_loss)
            decoder_histories[key] = dec_hist
            cells.append(
                _summarize_cell(
                    source=source,
                    decoder_loss=dec_loss,
                    pretrain_history=pretrain_histories[source],
                    decoder_history=dec_hist,
                    eval_poisson=eval_poisson,
                    eval_gaussian=eval_gaussian,
                )
            )

    plot_title = str(spec.get("plot_title", "DLPFC Layer 3 encoder × decoder grid"))
    if spatial_subset_meta is not None:
        plot_title = (
            f"{plot_title} (DBSCAN middle, n={spatial_subset_meta['n_cells_after']})"
        )
    prefix = args.run_name
    if spatial_subset_meta is not None:
        _plot_dbscan_subset(
            spatial_raw,
            spatial_subset_meta,
            out_dir / f"{prefix}_dbscan_regions.png",
            title=plot_title,
        )
    _plot_pretrain_curves(
        pretrain_histories,
        out_dir / f"{prefix}_pretrain_loss_curves.png",
        title=plot_title,
    )
    _plot_decoder_grid(
        decoder_histories,
        cells,
        out_dir / f"{prefix}_decoder_loss_curves.png",
        title=plot_title,
        pretrain_epochs=pretrain_epochs,
    )
    _plot_isodepths(
        spatial,
        encoder_axes,
        cells,
        out_dir / f"{prefix}_isodepths.png",
        title=plot_title,
    )
    _write_cross_metric_csv(cells, out_dir / f"{prefix}_cross_metric.csv")

    summary: dict[str, Any] = {
        "experiment": spec.get("experiment_name", "encoder_loss_grid"),
        "run_name": args.run_name,
        "h5ad": spec.get("data", {}).get("h5ad") or spec.get("data", {}).get("prepared_h5ad"),
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_reruns": n_reruns,
        "pretrain_epochs": pretrain_epochs,
        "warm_gaussian_epochs": warm_gaussian_epochs,
        "decoder_epochs": decoder_epochs,
        "sgd_batch_size": sgd_batch_size,
        "lr": lr,
        "seed": seed,
        "decoder": decoder,
        "gaussian_targets": "log_cpm_zscore",
        "spatial_subset": None
        if spatial_subset_meta is None
        else {
            k: v
            for k, v in spatial_subset_meta.items()
            if k != "cluster_ids"
        },
        "cells": cells,
    }
    _write_summary(summary, out_dir, args.run_name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--warm-gaussian-epochs", type=int, default=None)
    parser.add_argument("--decoder-epochs", type=int, default=None)
    parser.add_argument("--n-reruns", type=int, default=None)
    parser.add_argument("--sgd-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    if args.run_name is None:
        args.run_name = str(spec["output"]["run_name"])
    summary = run(spec, args)

    print("\n=== loss summary (median over reruns) ===")
    for row in summary["cells"]:
        print(
            f"  {row['encoder_source']:22s} dec={row['decoder_loss']:8s}  "
            f"train={row['train_final_median']:.5f}  "
            f"eval_poiss={row['eval_poisson_nll_median']:.5f}  "
            f"eval_gauss={row['eval_gaussian_mse_median']:.5f}"
        )


if __name__ == "__main__":
    main()
