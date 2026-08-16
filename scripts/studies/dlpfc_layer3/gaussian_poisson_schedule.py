"""DLPFC Layer 3: Gaussian→Poisson schedule sweep with fair fork at switch.

For each switch epoch m (from spec ``gaussian_epochs_list``, default
{0, 20, 200, 500} with ``total_epochs``=1000), four schedule branches plus a
covariate-matched reference:

  1. **gaussian_free** — m epochs Gaussian (MSE), then Poisson encoder+decoder
  2. **gaussian_frozen** — m epochs Gaussian, then freeze encoder snapshot,
     Poisson decoder-only
  3. **poisson_frozen** — m epochs Poisson encoder+decoder from init, then freeze
     encoder snapshot at m, Poisson decoder-only (standardized frozen control)
  4. **reference_frozen** — freeze exported ``gaussian_isodepth`` (z-scored, same as
     covariate experiment); Poisson decoder-only for ``total_epochs`` (run once)

Optional spec key ``gaussian_preprocessing`` overrides the expression preprocessing
used for the **Gaussian (MSE) phase only** (e.g. normalize_total + log1p +
standardize for log-CPM), while the Poisson phases continue on raw counts.  This
makes the in-run Gaussian axis match the preprocessing of the exported
``gaussian_isodepth`` reference instead of fitting MSE on raw counts.

100 parallel reruns, SGD batch size 128. Loss curves use global epoch x with the
decoder-only segment starting at x=m.

**PITFALL (fixed)**: ``gaussian_free`` must not recreate Adam / replay the minibatch
seed at the Gaussian→Poisson switch.  The Poisson tail continues the same optimizer
and batch order on the post-Gaussian model (frozen branches still clone before the
tail and train decoder-only).

Outputs (under results/dlpfc_new/layer3_gaussian_poisson_schedule/):
  - *_loss_curves.png                 — full log scale + zoomed converged tail
  - *_loss_diff_curves.png           — gaussian free − gaussian frozen
  - *_frozen_pretrain_diff_curves.png — gaussian frozen − poisson frozen
  - *_isodepths_{min,median}.png     — 4×4 panels (incl. reference_frozen row)
  - *_summary.json / .csv
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

from experiments.core.paths import repo_root
REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from analysis.plots import _plot_spatial_isodepth  # noqa: E402
from data import load_dataset  # noqa: E402
from data.transforms import zscore_covariate  # noqa: E402
from methods.architectures import (  # noqa: E402
    ParallelDecoderOnlyNetFixed,
    ParallelIsoDepthNet,
)
from methods.trainers.gpu_selection import resolve_device  # noqa: E402
from methods.trainers.isodepth import (  # noqa: E402
    _compute_reconstruction_loss_per_model,
    _set_torch_seed,
)
from scripts.studies.dlpfc_layer3.gaussian_axis_poisson import (  # noqa: E402
    _data_config_from_spec,
    _resolve,
    prepare_layer3_h5ad,
)
from experiments.core.study_spec import load_spec  # noqa: E402

DEFAULT_SPEC = REPO / "configs/experiments/dlpfc_layer3_gaussian_poisson_schedule.json"
BranchKind = Literal[
    "gaussian_free", "gaussian_frozen", "poisson_frozen", "reference_frozen"
]

M_COLORS: dict[int, Any] = {}


def _build_m_colors(m_values: list[int]) -> dict[int, Any]:
    ms = sorted({int(m) for m in m_values})
    if len(ms) <= 1:
        return {m: "#b2182b" for m in ms}
    cmap = plt.get_cmap("coolwarm")
    return {m: cmap(i / (len(ms) - 1)) for i, m in enumerate(ms)}

BRANCH_STYLE = {
    "gaussian_free": "-",
    "gaussian_frozen": "--",
    "poisson_frozen": ":",
    "reference_frozen": "-.",
}

REFERENCE_KEY = "reference_frozen"


def _plot_title(spec: dict) -> str:
    if spec.get("plot_title"):
        return str(spec["plot_title"])
    label = spec.get("data", {}).get("layer_label", "subset")
    return f"{label} — Gaussian vs Poisson pretrain, free vs frozen encoder"


def _covariate_baseline_paths(spec: dict) -> list[Path]:
    baseline = spec.get("covariate_baseline")
    if baseline:
        explicit = baseline.get("paths")
        if explicit:
            return [_resolve(p) for p in explicit]
        run_name = str(baseline.get("run_name", "layer3_poisson_gaussian_covariate"))
    else:
        run_name = "layer3_poisson_gaussian_covariate"
    out_dir = _resolve(spec["output"]["out_dir"])
    return [
        out_dir / run_name / "experiment_summary.json",
        out_dir / run_name / f"{run_name}_result.json",
    ]


def _branch_key(m: int, kind: BranchKind) -> str:
    return f"m{m}_{kind}"


def _expand_coords(s_np: np.ndarray, n_models: int, device: torch.device) -> torch.Tensor:
    s = torch.tensor(np.asarray(s_np, dtype=np.float32), device=device)
    return s.unsqueeze(0).expand(n_models, -1, -1).contiguous()


def _poisson_size_factors(a_np: np.ndarray, device: torch.device) -> torch.Tensor:
    sf = np.asarray(a_np, dtype=np.float32).sum(axis=1, keepdims=True)
    return torch.tensor(sf, dtype=torch.float32, device=device)


def _clone_model(model: ParallelIsoDepthNet) -> ParallelIsoDepthNet:
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict())
    return clone


def _sgd_epoch(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor | None,
    *,
    optimizer: optim.Optimizer,
    sgd_batch_size: int,
    n_cells: int,
    minibatch_generator: torch.Generator,
    fixed_latent_t: torch.Tensor | None = None,
) -> None:
    n_models = coords_t.shape[0]
    permutation = torch.randperm(n_cells, generator=minibatch_generator)
    for start in range(0, n_cells, sgd_batch_size):
        batch_indices = permutation[start : start + sgd_batch_size].to(device=coords_t.device)
        batch_a = targets_t.index_select(0, batch_indices)
        batch_sf = None
        if size_factors_t is not None:
            batch_sf = size_factors_t.index_select(0, batch_indices)

        optimizer.zero_grad()
        if fixed_latent_t is not None:
            batch_output = model.decoder(fixed_latent_t.index_select(1, batch_indices))
        else:
            batch_s = coords_t.index_select(1, batch_indices)
            batch_output = model(batch_s)
        batch_loss_per_model = _compute_reconstruction_loss_per_model(
            batch_output,
            batch_a,
            None,
            poisson_size_factors=batch_sf,
        )
        total_loss = batch_loss_per_model.sum() / n_models
        total_loss.backward()
        optimizer.step()


def _eval_poisson_loss(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    *,
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


def _eval_reference_poisson_loss(
    model: ParallelDecoderOnlyNetFixed,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        output = model()
        loss_per_model = _compute_reconstruction_loss_per_model(
            output, targets_t, None, poisson_size_factors=size_factors_t
        )
    return loss_per_model.detach().cpu().numpy().astype(np.float64)


def _sgd_reference_epoch(
    model: ParallelDecoderOnlyNetFixed,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    *,
    optimizer: optim.Optimizer,
    sgd_batch_size: int,
    n_cells: int,
    minibatch_generator: torch.Generator,
) -> None:
    n_models = model.M
    permutation = torch.randperm(n_cells, generator=minibatch_generator)
    for start in range(0, n_cells, sgd_batch_size):
        batch_indices = permutation[start : start + sgd_batch_size].to(device=targets_t.device)
        batch_a = targets_t.index_select(0, batch_indices)
        batch_sf = size_factors_t.index_select(0, batch_indices)
        batch_latent = model.encoder.latent_values.index_select(0, batch_indices)
        batch_latent = batch_latent.unsqueeze(0).expand(n_models, -1, -1)

        optimizer.zero_grad()
        batch_output = model.decoder(batch_latent)
        batch_loss_per_model = _compute_reconstruction_loss_per_model(
            batch_output,
            batch_a,
            None,
            poisson_size_factors=batch_sf,
        )
        total_loss = batch_loss_per_model.sum() / n_models
        total_loss.backward()
        optimizer.step()


def _load_reference_latent(spec: dict, n_cells: int) -> np.ndarray:
    import anndata as ad

    prepared = _resolve(spec["data"]["prepared_h5ad"])
    cov_key = str(spec["data"]["covariate_obs_key"])
    adata = ad.read_h5ad(prepared)
    if cov_key not in adata.obs.columns:
        gaussian_npz = _resolve(spec["artifacts"]["gaussian_isodepth_npz"])
        from scripts.studies.dlpfc_layer3.gaussian_axis_poisson import _align_gaussian_isodepth

        values = _align_gaussian_isodepth(gaussian_npz, n_cells)
    else:
        values = np.asarray(adata.obs[cov_key].to_numpy(), dtype=np.float32).reshape(-1)
    if values.size != n_cells:
        raise ValueError(
            f"Reference covariate length {values.size} != dataset n_cells {n_cells}"
        )
    return zscore_covariate(values)


def _train_reference_frozen(
    reference_latent: np.ndarray,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    *,
    n_reruns: int,
    n_genes: int,
    poisson_epochs: int,
    lr: float,
    sgd_batch_size: int,
    decoder: str,
    seed: int,
    device: torch.device,
) -> tuple[ParallelDecoderOnlyNetFixed, np.ndarray]:
    n_cells = targets_t.shape[0]
    _set_torch_seed(seed)
    model = ParallelDecoderOnlyNetFixed(
        n_reruns,
        n_genes,
        reference_latent,
        latent_dim=1,
        decoder_type=decoder,
    ).to(device)
    optimizer = optim.Adam(model.decoder.parameters(), lr=lr, foreach=False)
    minibatch_generator = torch.Generator(device="cpu")
    minibatch_generator.manual_seed(seed)
    print(
        f"    reference_frozen: {poisson_epochs} decoder-only epochs on exported "
        f"gaussian_isodepth (SGD-{sgd_batch_size})",
        flush=True,
    )
    history: list[np.ndarray] = []
    for _ in range(poisson_epochs):
        _sgd_reference_epoch(
            model,
            targets_t,
            size_factors_t,
            optimizer=optimizer,
            sgd_batch_size=sgd_batch_size,
            n_cells=n_cells,
            minibatch_generator=minibatch_generator,
        )
        history.append(_eval_reference_poisson_loss(model, targets_t, size_factors_t))
    return model, np.asarray(history)


def _new_adam_and_batch_gen(
    model: ParallelIsoDepthNet, *, lr: float, seed: int
) -> tuple[optim.Optimizer, torch.Generator]:
    optimizer = optim.Adam(model.parameters(), lr=lr, foreach=False)
    minibatch_generator = torch.Generator(device="cpu")
    minibatch_generator.manual_seed(seed)
    return optimizer, minibatch_generator


def _train_gaussian_phase(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    *,
    gaussian_epochs: int,
    lr: float,
    sgd_batch_size: int,
    seed: int,
    optimizer: optim.Optimizer | None = None,
    minibatch_generator: torch.Generator | None = None,
) -> tuple[optim.Optimizer, torch.Generator]:
    """Run the Gaussian (MSE) segment; returns optimizer + batch generator for reuse.

    For ``gaussian_free``, the Poisson tail must continue the **same** Adam state and
    minibatch order.  Recreating either at the switch (old behaviour) made the true
    encoder fail to adapt after log-CPM warm-up.
    """
    n_cells = coords_t.shape[1]
    if optimizer is None or minibatch_generator is None:
        optimizer, minibatch_generator = _new_adam_and_batch_gen(model, lr=lr, seed=seed)
    if gaussian_epochs <= 0:
        return optimizer, minibatch_generator
    print(f"    Gaussian phase: {gaussian_epochs} epochs (MSE, SGD-{sgd_batch_size})", flush=True)
    for _ in range(gaussian_epochs):
        _sgd_epoch(
            model,
            coords_t,
            targets_t,
            None,
            optimizer=optimizer,
            sgd_batch_size=sgd_batch_size,
            n_cells=n_cells,
            minibatch_generator=minibatch_generator,
        )
    return optimizer, minibatch_generator


def _train_poisson_free_tail_with_history(
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
    """Poisson encoder+decoder tail; optionally continues an existing optimizer."""
    if poisson_epochs <= 0:
        if optimizer is None or minibatch_generator is None:
            optimizer, minibatch_generator = _new_adam_and_batch_gen(model, lr=lr, seed=seed)
        return np.empty((0, coords_t.shape[0]), dtype=np.float64), optimizer, minibatch_generator
    n_cells = coords_t.shape[1]
    if optimizer is None or minibatch_generator is None:
        optimizer, minibatch_generator = _new_adam_and_batch_gen(model, lr=lr, seed=seed)
    print(
        f"    Poisson free (encoder+decoder): {poisson_epochs} epochs (SGD-{sgd_batch_size})",
        flush=True,
    )
    history: list[np.ndarray] = []
    for _ in range(poisson_epochs):
        _sgd_epoch(
            model,
            coords_t,
            targets_t,
            size_factors_t,
            optimizer=optimizer,
            sgd_batch_size=sgd_batch_size,
            n_cells=n_cells,
            minibatch_generator=minibatch_generator,
        )
        history.append(
            _eval_poisson_loss(model, coords_t, targets_t, size_factors_t)
        )
    return np.asarray(history), optimizer, minibatch_generator


def _train_poisson_free_phase(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    sgd_batch_size: int,
    seed: int,
) -> None:
    if epochs <= 0:
        return
    n_cells = coords_t.shape[1]
    optimizer = optim.Adam(model.parameters(), lr=lr, foreach=False)
    minibatch_generator = torch.Generator(device="cpu")
    minibatch_generator.manual_seed(seed)
    print(
        f"    Poisson free (encoder+decoder): {epochs} epochs (SGD-{sgd_batch_size})",
        flush=True,
    )
    for _ in range(epochs):
        _sgd_epoch(
            model,
            coords_t,
            targets_t,
            size_factors_t,
            optimizer=optimizer,
            sgd_batch_size=sgd_batch_size,
            n_cells=n_cells,
            minibatch_generator=minibatch_generator,
        )


def _train_poisson_decoder_phase(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    fixed_latent_t: torch.Tensor,
    *,
    poisson_epochs: int,
    lr: float,
    sgd_batch_size: int,
    seed: int,
    label: str,
) -> np.ndarray:
    n_cells = coords_t.shape[1]
    optimizer = optim.Adam(model.decoder.parameters(), lr=lr, foreach=False)
    minibatch_generator = torch.Generator(device="cpu")
    minibatch_generator.manual_seed(seed)
    print(
        f"    Poisson frozen encoder ({label}): {poisson_epochs} decoder-only epochs "
        f"(SGD-{sgd_batch_size})",
        flush=True,
    )
    history: list[np.ndarray] = []
    for _ in range(poisson_epochs):
        _sgd_epoch(
            model,
            coords_t,
            targets_t,
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
                coords_t,
                targets_t,
                size_factors_t,
                fixed_latent_t=fixed_latent_t,
            )
        )
    return np.asarray(history)


def _select_rerun(history: np.ndarray, mode: str) -> int:
    finals = history[-1]
    if mode == "min":
        return int(np.argmin(finals))
    if mode == "median":
        return int(np.argsort(finals)[len(finals) // 2])
    raise ValueError(f"Unsupported rerun selection mode '{mode}'")


def _encoder_axis(model: ParallelIsoDepthNet, coords_t: torch.Tensor, slot: int) -> np.ndarray:
    with torch.no_grad():
        latent = model.encoder(coords_t)[slot, :, 0].detach().cpu().numpy()
    return latent.astype(np.float64)


def _latent_axis(fixed_latent_t: torch.Tensor, slot: int) -> np.ndarray:
    return fixed_latent_t[slot, :, 0].detach().cpu().numpy().astype(np.float64)


def _summarize_branch(
    *,
    m: int,
    kind: BranchKind,
    poisson_tail_epochs: int,
    history: np.ndarray,
) -> dict[str, Any]:
    finals = history[-1]
    sel_min = _select_rerun(history, "min")
    sel_median = _select_rerun(history, "median")
    return {
        "schedule_key": _branch_key(m, kind),
        "freeze_epoch": m,
        "branch": kind,
        "poisson_tail_epochs": poisson_tail_epochs,
        "median_final_loss": float(np.median(finals)),
        "min_final_loss": float(np.min(finals)),
        "q25_final_loss": float(np.percentile(finals, 25)),
        "q75_final_loss": float(np.percentile(finals, 75)),
        "min_rerun_index": sel_min,
        "median_rerun_index": sel_median,
        "min_final_loss_selected": float(finals[sel_min]),
        "median_final_loss_selected": float(finals[sel_median]),
    }


def _plot_loss_curves(
    branches: list[dict[str, Any]],
    histories: dict[str, np.ndarray],
    out_path: Path,
    *,
    title: str,
    reference_history: np.ndarray | None = None,
    zoom_final_cap: float = 3.0,
    zoom_tail_fraction: float = 0.5,
) -> None:
    series: list[tuple[np.ndarray, np.ndarray, str, str, str]] = []
    for row in branches:
        m = int(row["freeze_epoch"])
        kind = row["branch"]
        if kind == "reference_frozen":
            continue
        history = histories[row["schedule_key"]]
        median_curve = np.median(history, axis=1)
        epochs = m + np.arange(history.shape[0])
        color = M_COLORS[m]
        ls = BRANCH_STYLE[kind]
        label = f"m={m} {kind} (final={median_curve[-1]:.4f})"
        series.append((epochs, median_curve, color, ls, label))
    if reference_history is not None:
        ref_median = np.median(reference_history, axis=1)
        series.append(
            (
                np.arange(reference_history.shape[0]),
                ref_median,
                "#4daf4a",
                BRANCH_STYLE["reference_frozen"],
                f"reference_frozen (final={ref_median[-1]:.4f})",
            )
        )

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6))
    tail_vals: list[float] = []
    tail_x_min: list[float] = []
    for epochs, median_curve, color, ls, label in series:
        ax_full.plot(epochs, median_curve, color=color, lw=2.0, ls=ls, label=label)
        ax_zoom.plot(epochs, median_curve, color=color, lw=2.0, ls=ls, label=label)
        if float(median_curve[-1]) <= zoom_final_cap:
            tail_n = max(1, int(len(median_curve) * zoom_tail_fraction))
            tail_vals.extend(median_curve[-tail_n:].tolist())
            tail_x_min.append(float(epochs[max(0, len(epochs) - tail_n)]))

    ax_full.set_yscale("log")
    ax_full.set_xlabel("Global epoch")
    ax_full.set_ylabel("Poisson NLL (median, log)")
    ax_full.set_title("Full training (log scale)")
    ax_full.legend(loc="upper right", fontsize=5, ncol=2)
    ax_full.grid(alpha=0.3, which="both")

    if tail_vals:
        pad = max(0.002, 0.05 * (max(tail_vals) - min(tail_vals)))
        ax_zoom.set_ylim(min(tail_vals) - pad, max(tail_vals) + pad)
    if tail_x_min:
        ax_zoom.set_xlim(left=min(tail_x_min))
    ax_zoom.set_xlabel("Global epoch")
    ax_zoom.set_ylabel("Poisson NLL (median)")
    ax_zoom.set_title("Converged tail (well-behaved branches)")
    ax_zoom.legend(loc="best", fontsize=5, ncol=2)
    ax_zoom.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[schedule] wrote {out_path}", flush=True)


def _plot_pair_diff_curves(
    m_values: list[int],
    histories: dict[str, np.ndarray],
    *,
    left_kind: BranchKind,
    right_kind: BranchKind,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in m_values:
        left_hist = histories[_branch_key(m, left_kind)]
        right_hist = histories[_branch_key(m, right_kind)]
        n = min(left_hist.shape[0], right_hist.shape[0])
        diff = np.median(left_hist[:n], axis=1) - np.median(right_hist[:n], axis=1)
        ax.plot(m + np.arange(n), diff, color=M_COLORS[m], lw=2.0, label=f"m={m}")
    ax.axhline(0.0, color="0.4", lw=1.0, ls=":")
    ax.set_xlabel("Global epoch (tail starts at m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[schedule] wrote {out_path}", flush=True)


def _plot_isodepths(
    spatial: np.ndarray,
    m_values: list[int],
    branches: list[dict[str, Any]],
    axes_by_key: dict[str, np.ndarray],
    out_path: Path,
    *,
    title: str,
    selection_label: str,
    rerun_key: str,
) -> None:
    lookup = {row["schedule_key"]: row for row in branches}
    kinds: list[BranchKind] = [
        "gaussian_free",
        "gaussian_frozen",
        "poisson_frozen",
        "reference_frozen",
    ]
    row_labels = ["gauss free", "gauss frz", "poiss frz", "ref frz"]
    fig, plot_axes = plt.subplots(
        len(kinds), len(m_values), figsize=(3.0 * len(m_values), 3.2 * len(kinds)), squeeze=False
    )
    for row_idx, kind in enumerate(kinds):
        for col, m in enumerate(m_values):
            ax = plot_axes[row_idx, col]
            if kind == "reference_frozen":
                row = lookup[REFERENCE_KEY]
                depth = axes_by_key[REFERENCE_KEY]
                title = f"ref frz (covariate)\nNLL={row[rerun_key]:.4f}"
            else:
                key = _branch_key(m, kind)
                row = lookup[key]
                depth = axes_by_key[key]
                title = f"{row_labels[row_idx]} m={m}\nNLL={row[rerun_key]:.4f}"
            _plot_spatial_isodepth(ax, spatial, np.asarray(depth, dtype=np.float32), title)
    fig.suptitle(f"{title} — isodepths ({selection_label} rerun, SGD-128)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[schedule] wrote {out_path}", flush=True)


def _write_summary(summary: dict, out_dir: Path, run_name: str) -> None:
    json_path = out_dir / f"{run_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    csv_path = out_dir / f"{run_name}_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
    print(f"[schedule] wrote {json_path}", flush=True)
    print(f"[schedule] wrote {csv_path}", flush=True)


def _append_branch(
    *,
    m: int,
    kind: BranchKind,
    history: np.ndarray,
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    fixed_latent: torch.Tensor | None,
    poisson_tail: int,
    histories: dict[str, np.ndarray],
    models: dict[str, ParallelIsoDepthNet],
    frozen_latents: dict[str, torch.Tensor],
    branches: list[dict[str, Any]],
) -> None:
    key = _branch_key(m, kind)
    histories[key] = history
    models[key] = model
    if kind not in ("gaussian_free", "reference_frozen"):
        assert fixed_latent is not None
        frozen_latents[key] = fixed_latent
    branches.append(
        _summarize_branch(
            m=m,
            kind=kind,
            poisson_tail_epochs=poisson_tail,
            history=history,
        )
    )


def run(spec: dict, args: argparse.Namespace) -> dict:
    sched = spec["schedule"]
    device = resolve_device(args.device or sched.get("device", "cuda"))
    out_dir = _resolve(spec["output"]["out_dir"]) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = int(args.total_epochs or sched["total_epochs"])
    m_values = (
        [int(args.gaussian_epochs)]
        if args.gaussian_epochs is not None
        else [int(m) for m in sched["gaussian_epochs_list"]]
    )
    n_reruns = int(args.n_reruns or sched["n_reruns"])
    sgd_batch_size = int(args.sgd_batch_size or sched["sgd_batch_size"])
    lr = float(args.lr or sched["lr"])
    decoder = str(args.decoder or sched.get("decoder", "nn"))
    seed = int(args.seed or sched["seed"])

    M_COLORS.clear()
    M_COLORS.update(_build_m_colors(m_values))

    prepared = prepare_layer3_h5ad(spec, force=False)
    data_cfg = _data_config_from_spec(spec["data"], spec["preprocessing"], prepared)
    dataset = load_dataset(data_cfg)
    n_cells, n_genes = dataset.A.shape

    spatial = np.asarray(dataset.S, dtype=np.float32)

    print(
        f"[schedule] n={n_cells} genes={n_genes} reruns={n_reruns} "
        f"total_epochs={total_epochs} sgd_batch_size={sgd_batch_size} device={device}",
        flush=True,
    )

    coords_t = _expand_coords(dataset.S, n_reruns, device)
    targets_t = torch.tensor(np.asarray(dataset.A, dtype=np.float32), device=device)
    size_factors_t = _poisson_size_factors(dataset.A, device)

    # Optional: train the Gaussian (MSE) phase on a different preprocessing
    # (e.g. normalize_total + log1p + standardize) while the Poisson phases keep
    # raw counts.  This makes the in-run Gaussian axis match the preprocessing of
    # the exported gaussian_isodepth / covariate reference.
    gaussian_pp = spec.get("gaussian_preprocessing")
    if gaussian_pp is not None:
        merged_pp = {**spec["preprocessing"], **gaussian_pp}
        gauss_data_cfg = _data_config_from_spec(spec["data"], merged_pp, prepared)
        gauss_dataset = load_dataset(gauss_data_cfg)
        if gauss_dataset.A.shape != dataset.A.shape:
            raise ValueError(
                f"Gaussian-phase targets shape {gauss_dataset.A.shape} != raw targets "
                f"shape {dataset.A.shape}; preprocessing must not change cell/gene counts."
            )
        gaussian_targets_t = torch.tensor(
            np.asarray(gauss_dataset.A, dtype=np.float32), device=device
        )
        print(
            f"[schedule] Gaussian MSE phase preprocessing override: {gaussian_pp}",
            flush=True,
        )
    else:
        gaussian_targets_t = targets_t

    histories: dict[str, np.ndarray] = {}
    models: dict[str, ParallelIsoDepthNet | ParallelDecoderOnlyNetFixed] = {}
    frozen_latents: dict[str, torch.Tensor] = {}
    branches: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    covariate_epochs = int(sched.get("covariate_epochs", 500))

    reference_latent = _load_reference_latent(spec, n_cells)
    ref_model, ref_history = _train_reference_frozen(
        reference_latent,
        targets_t,
        size_factors_t,
        n_reruns=n_reruns,
        n_genes=n_genes,
        poisson_epochs=total_epochs,
        lr=lr,
        sgd_batch_size=sgd_batch_size,
        decoder=decoder,
        seed=seed,
        device=device,
    )
    histories[REFERENCE_KEY] = ref_history
    models[REFERENCE_KEY] = ref_model
    ref_branch = _summarize_branch(
        m=0,
        kind="reference_frozen",
        poisson_tail_epochs=total_epochs,
        history=ref_history,
    )
    ref_branch["schedule_key"] = REFERENCE_KEY
    if covariate_epochs <= total_epochs:
        ref_at_cov = ref_history[covariate_epochs - 1]
        ref_branch["median_loss_at_covariate_epochs"] = float(np.median(ref_at_cov))
        ref_branch["min_loss_at_covariate_epochs"] = float(np.min(ref_at_cov))
    branches.append(ref_branch)

    for m in m_values:
        if m < 0 or m > total_epochs:
            raise ValueError(f"freeze_epoch={m} must be in [0, {total_epochs}]")
        poisson_tail = total_epochs - m
        print(f"[schedule] m={m}: three branches (gauss free / gauss frz / poiss frz)", flush=True)

        # --- Gaussian pretrain path ---
        _set_torch_seed(seed)
        gauss_model = ParallelIsoDepthNet(
            n_reruns, n_genes, latent_dim=1, decoder_type=decoder
        ).to(device)
        optimizer, gen = _train_gaussian_phase(
            gauss_model,
            coords_t,
            gaussian_targets_t,
            gaussian_epochs=m,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
        )
        with torch.no_grad():
            gauss_switch_latent = gauss_model.encoder(coords_t).detach()

        # Frozen branch: snapshot post-Gaussian weights before any Poisson free steps.
        gauss_frozen_model = _clone_model(gauss_model)
        gauss_free_hist, _, _ = _train_poisson_free_tail_with_history(
            gauss_model,
            coords_t,
            targets_t,
            size_factors_t,
            poisson_epochs=poisson_tail,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
            optimizer=optimizer,
            minibatch_generator=gen,
        )
        gauss_free_model = gauss_model
        gauss_frozen_hist = _train_poisson_decoder_phase(
            gauss_frozen_model,
            coords_t,
            targets_t,
            size_factors_t,
            gauss_switch_latent,
            poisson_epochs=poisson_tail,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
            label="after Gaussian",
        )

        _append_branch(
            m=m,
            kind="gaussian_free",
            history=gauss_free_hist,
            model=gauss_free_model,
            coords_t=coords_t,
            fixed_latent=None,
            poisson_tail=poisson_tail,
            histories=histories,
            models=models,
            frozen_latents=frozen_latents,
            branches=branches,
        )
        _append_branch(
            m=m,
            kind="gaussian_frozen",
            history=gauss_frozen_hist,
            model=gauss_frozen_model,
            coords_t=coords_t,
            fixed_latent=gauss_switch_latent,
            poisson_tail=poisson_tail,
            histories=histories,
            models=models,
            frozen_latents=frozen_latents,
            branches=branches,
        )

        # --- Poisson pretrain control (freeze at same m) ---
        _set_torch_seed(seed)
        poiss_model = ParallelIsoDepthNet(
            n_reruns, n_genes, latent_dim=1, decoder_type=decoder
        ).to(device)
        _train_poisson_free_phase(
            poiss_model,
            coords_t,
            targets_t,
            size_factors_t,
            epochs=m,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
        )
        with torch.no_grad():
            poiss_switch_latent = poiss_model.encoder(coords_t).detach()

        poiss_frozen_model = _clone_model(poiss_model)
        poiss_frozen_hist = _train_poisson_decoder_phase(
            poiss_frozen_model,
            coords_t,
            targets_t,
            size_factors_t,
            poiss_switch_latent,
            poisson_epochs=poisson_tail,
            lr=lr,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
            label="after Poisson pretrain",
        )
        _append_branch(
            m=m,
            kind="poisson_frozen",
            history=poiss_frozen_hist,
            model=poiss_frozen_model,
            coords_t=coords_t,
            fixed_latent=poiss_switch_latent,
            poisson_tail=poisson_tail,
            histories=histories,
            models=models,
            frozen_latents=frozen_latents,
            branches=branches,
        )

        lookup = {b["schedule_key"]: b for b in branches if b["freeze_epoch"] == m}
        comparisons.append(
            {
                "freeze_epoch": m,
                "poisson_tail_epochs": poisson_tail,
                "median_gaussian_free": lookup[_branch_key(m, "gaussian_free")]["median_final_loss"],
                "median_gaussian_frozen": lookup[_branch_key(m, "gaussian_frozen")][
                    "median_final_loss"
                ],
                "median_poisson_frozen": lookup[_branch_key(m, "poisson_frozen")][
                    "median_final_loss"
                ],
                "median_reference_frozen": ref_branch["median_final_loss"],
                "median_reference_at_covariate_epochs": ref_branch.get(
                    "median_loss_at_covariate_epochs"
                ),
                "median_gaussian_free_minus_frozen": (
                    lookup[_branch_key(m, "gaussian_free")]["median_final_loss"]
                    - lookup[_branch_key(m, "gaussian_frozen")]["median_final_loss"]
                ),
                "median_gaussian_frozen_minus_poisson_frozen": (
                    lookup[_branch_key(m, "gaussian_frozen")]["median_final_loss"]
                    - lookup[_branch_key(m, "poisson_frozen")]["median_final_loss"]
                ),
                "median_gaussian_frozen_minus_reference": (
                    lookup[_branch_key(m, "gaussian_frozen")]["median_final_loss"]
                    - ref_branch["median_final_loss"]
                ),
            }
        )

    prefix = args.run_name
    plot_title = _plot_title(spec)
    _plot_loss_curves(
        branches,
        histories,
        out_dir / f"{prefix}_loss_curves.png",
        title=plot_title,
        reference_history=ref_history,
    )
    _plot_pair_diff_curves(
        m_values,
        histories,
        left_kind="gaussian_free",
        right_kind="gaussian_frozen",
        ylabel="median NLL (gaussian free − gaussian frozen)",
        title="Free vs frozen after Gaussian pretrain",
        out_path=out_dir / f"{prefix}_loss_diff_curves.png",
    )
    _plot_pair_diff_curves(
        m_values,
        histories,
        left_kind="gaussian_frozen",
        right_kind="poisson_frozen",
        ylabel="median NLL (gaussian frozen − poisson frozen)",
        title="Frozen-decoder tail: Gaussian vs Poisson pretrain at same m",
        out_path=out_dir / f"{prefix}_frozen_pretrain_diff_curves.png",
    )
    for mode, rerun_key in (
        ("min", "min_final_loss_selected"),
        ("median", "median_final_loss_selected"),
    ):
        axes_by_key: dict[str, np.ndarray] = {}
        for row in branches:
            key = row["schedule_key"]
            slot = int(row[f"{mode}_rerun_index"])
            if row["branch"] == "gaussian_free":
                axes_by_key[key] = _encoder_axis(models[key], coords_t, slot)  # type: ignore[arg-type]
            elif row["branch"] == "reference_frozen":
                axes_by_key[key] = reference_latent.astype(np.float64)
            else:
                axes_by_key[key] = _latent_axis(frozen_latents[key], slot)
        _plot_isodepths(
            spatial,
            m_values,
            branches,
            axes_by_key,
            out_dir / f"{prefix}_isodepths_{mode}.png",
            title=plot_title,
            selection_label=mode,
            rerun_key=rerun_key,
        )

    covariate_stat: float | None = None
    for cov_path in _covariate_baseline_paths(spec):
        if not cov_path.exists():
            continue
        with open(cov_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        raw = payload.get("stat_covariate")
        if raw is None:
            raw = payload.get("covariate_artifacts", {}).get("stat_covariate")
        if raw is not None:
            covariate_stat = float(raw)
            break

    summary = {
        "experiment": spec.get("experiment_name", "dlpfc_layer3_gaussian_poisson_schedule"),
        "run_name": args.run_name,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_reruns": n_reruns,
        "total_epochs": total_epochs,
        "covariate_epochs": covariate_epochs,
        "gaussian_preprocessing": gaussian_pp,
        "sgd_batch_size": sgd_batch_size,
        "lr": lr,
        "seed": seed,
        "reference_frozen": ref_branch,
        "covariate_stat_from_prior_run": covariate_stat,
        "branches": branches,
        "comparisons": comparisons,
    }
    _write_summary(summary, out_dir, args.run_name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument("--gaussian-epochs", type=int, default=None)
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

    ref = summary["reference_frozen"]
    print("\n=== reference_frozen (exported gaussian_isodepth) ===")
    print(f"  final median NLL={ref['median_final_loss']:.5f}")
    if "median_loss_at_covariate_epochs" in ref:
        print(
            f"  median NLL @ epoch {summary['covariate_epochs']}="
            f"{ref['median_loss_at_covariate_epochs']:.5f}"
        )
    if summary.get("covariate_stat_from_prior_run") is not None:
        print(
            f"  prior covariate stat_covariate="
            f"{summary['covariate_stat_from_prior_run']:.5f}"
        )

    sweep_label = spec.get("data", {}).get("layer_label", "subset")
    print(f"\n=== {sweep_label} schedule sweep ===")
    for row in summary["comparisons"]:
        print(
            f"m={row['freeze_epoch']:3d}  "
            f"g_free={row['median_gaussian_free']:.5f}  "
            f"g_frz={row['median_gaussian_frozen']:.5f}  "
            f"p_frz={row['median_poisson_frozen']:.5f}  "
            f"ref={row['median_reference_frozen']:.5f}  "
            f"Δ(g_frz−ref)={row['median_gaussian_frozen_minus_reference']:+.5f}"
        )


if __name__ == "__main__":
    main()
