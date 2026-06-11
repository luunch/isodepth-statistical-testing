"""DLPFC Layer 3: warm-start vs cold-start Poisson, compared under the Poisson loss.

Question: if a Poisson isodepth model is *warm-started* at the frozen Gaussian
isodepth axis (encoder reproduces the Gaussian axis + decoder pre-fit on it) and
then trained freely, does it stay near the Gaussian solution or converge to the
same place a cold-started Poisson model does?

Three trainings, all evaluated under the SAME Poisson NLL, all recorded from
epoch 0 (no 500-epoch head start), each running ``n_reruns`` parallel inits:

  A. Frozen Gaussian axis  -- encoder held EXACTLY at the (z-scored) Gaussian
     isodepth; only the Poisson decoder trains. This is the "how low can Poisson
     go if the axis never moves" floor and supplies the warm decoder for B.
  B. Warm-start            -- encoder pretrained to the Gaussian axis + decoder
     copied from A, then the full model (encoder+decoder) trains freely.
  C. Cold-start            -- default initialization, full model trains freely.

A, B, C share identical random initialization (same seed) so the only
differences are the warm encoder/decoder of B and the frozen encoder of A.

Outputs (under results/dlpfc_new/layer3_poisson_warmstart/):
  - layer3_poisson_warmstart_loss_curves_{min,median}.png
  - layer3_poisson_warmstart_isodepths_{min,median}.png
  - layer3_poisson_warmstart_summary.json / .csv

Each loss/isodepth figure duplicates the same layout twice: once highlighting the
minimum-final-loss rerun (dark) and once the median-final-loss rerun.

Usage:
    python scripts/dlpfc_layer3_poisson_warmstart.py
    python scripts/dlpfc_layer3_poisson_warmstart.py --epochs 1500 --n-reruns 100
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import optim

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.plots import _plot_spatial_isodepth  # noqa: E402
from data import load_dataset  # noqa: E402
from data.transforms import zscore_covariate  # noqa: E402
from methods.architectures import ParallelIsoDepthNet  # noqa: E402
from methods.trainers.gpu_selection import resolve_device  # noqa: E402
from methods.trainers.isodepth import (  # noqa: E402
    _compute_reconstruction_loss_per_model,
    _set_torch_seed,
)
from scripts.dlpfc_layer3_gaussian_axis_poisson import (  # noqa: E402
    DEFAULT_SPEC,
    _data_config_from_spec,
    _resolve,
    prepare_layer3_h5ad,
)
from scripts.liver_lobule_sweep import load_spec  # noqa: E402


def _expand_coords(s_np: np.ndarray, n_models: int, device: torch.device) -> torch.Tensor:
    """(N, 2) numpy coords -> (M, N, 2) tensor broadcast across the M parallel inits."""
    s = torch.tensor(np.asarray(s_np, dtype=np.float32), device=device)
    return s.unsqueeze(0).expand(n_models, -1, -1).contiguous()


def _poisson_size_factors(a_np: np.ndarray, device: torch.device) -> torch.Tensor:
    sf = np.asarray(a_np, dtype=np.float32).sum(axis=1, keepdims=True)
    return torch.tensor(sf, dtype=torch.float32, device=device)


def _train_parallel(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    *,
    params,
    epochs: int,
    lr: float,
    fixed_latent_t: torch.Tensor | None = None,
    record: bool = True,
) -> np.ndarray | None:
    """Full-batch Adam. Returns per-epoch Poisson NLL per rerun, shape (epochs, M).

    If ``fixed_latent_t`` is given the encoder is bypassed (decoder-only on a
    frozen latent); otherwise the full ``model(coords)`` path is used.
    """
    optimizer = optim.Adam(params, lr=lr, foreach=False)
    n_models = coords_t.shape[0]
    history: list[np.ndarray] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model.decoder(fixed_latent_t) if fixed_latent_t is not None else model(coords_t)
        loss_per_model = _compute_reconstruction_loss_per_model(
            output, targets_t, None, poisson_size_factors=size_factors_t
        )
        total_loss = loss_per_model.sum() / n_models
        total_loss.backward()
        optimizer.step()
        if record:
            history.append(loss_per_model.detach().cpu().numpy().astype(np.float64))
    return np.asarray(history) if record else None


def _pretrain_encoder_to_axis(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    target_latent_t: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> float:
    """Regress encoder(coords) -> target axis (MSE). Returns final |corr| of the
    best-fitting rerun to the target axis (warm-start quality diagnostic)."""
    optimizer = optim.Adam(model.encoder.parameters(), lr=lr, foreach=False)
    for _ in range(epochs):
        optimizer.zero_grad()
        latent = model.encoder(coords_t)
        loss = ((latent - target_latent_t) ** 2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        latent = model.encoder(coords_t)[:, :, 0].detach().cpu().numpy()
    target = target_latent_t[0, :, 0].detach().cpu().numpy()
    corrs = [abs(np.corrcoef(latent[m], target)[0, 1]) for m in range(latent.shape[0])]
    return float(np.nanmax(corrs))


def _select_rerun(history: np.ndarray, mode: str) -> int:
    """Index of the rerun selected by final-epoch loss (``min`` or ``median``)."""
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


def _abs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(spearmanr(a, b).statistic))


def _selection_summary(
    *,
    mode: str,
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    hist_c: np.ndarray,
    model_b: ParallelIsoDepthNet,
    model_c: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    target_axis: np.ndarray,
) -> tuple[dict, dict]:
    sel_a = _select_rerun(hist_a, mode)
    sel_b = _select_rerun(hist_b, mode)
    sel_c = _select_rerun(hist_c, mode)
    axis_b = _encoder_axis(model_b, coords_t, sel_b)
    axis_c = _encoder_axis(model_c, coords_t, sel_c)
    prefix = f"{mode}_"
    stats = {
        f"{prefix}final_loss_frozen_gaussian": float(hist_a[-1, sel_a]),
        f"{prefix}final_loss_warm": float(hist_b[-1, sel_b]),
        f"{prefix}final_loss_cold": float(hist_c[-1, sel_c]),
        f"{prefix}abs_spearman_warm_vs_gaussian": _abs_spearman(axis_b, target_axis),
        f"{prefix}abs_spearman_cold_vs_gaussian": _abs_spearman(axis_c, target_axis),
        f"{prefix}abs_spearman_warm_vs_cold": _abs_spearman(axis_b, axis_c),
        f"{prefix}rerun_index_frozen_gaussian": sel_a,
        f"{prefix}rerun_index_warm": sel_b,
        f"{prefix}rerun_index_cold": sel_c,
    }
    axes = {
        "A: frozen Gaussian axis": target_axis,
        "B: warm-start": axis_b,
        "C: cold-start": axis_c,
    }
    experiments = {
        "A: frozen Gaussian axis": (hist_a, sel_a, "#1f4e79"),
        "B: warm-start": (hist_b, sel_b, "#1b7837"),
        "C: cold-start": (hist_c, sel_c, "#b2182b"),
    }
    return stats, {"stats": stats, "axes": axes, "experiments": experiments}


def run(spec: dict, args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    out_dir = _resolve(spec["output"]["out_dir"]) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_layer3_h5ad(spec, force=False)
    data_cfg = _data_config_from_spec(spec["data"], spec["preprocessing"], prepared)
    dataset = load_dataset(data_cfg)
    n_cells, n_genes = dataset.A.shape

    cov_key = str(spec["data"]["covariate_obs_key"])
    gaussian_axis = np.asarray(
        ad.read_h5ad(prepared).obs[cov_key].to_numpy(), dtype=np.float64
    ).reshape(-1)
    if gaussian_axis.size != n_cells:
        raise ValueError(
            f"Gaussian axis length {gaussian_axis.size} != dataset cells {n_cells}."
        )
    target_axis = zscore_covariate(gaussian_axis).reshape(-1)

    print(
        f"[warmstart] n={n_cells} genes={n_genes} reruns={args.n_reruns} "
        f"epochs={args.epochs} pretrain_epochs={args.pretrain_epochs} device={device}",
        flush=True,
    )

    n_models = int(args.n_reruns)
    coords_t = _expand_coords(dataset.S, n_models, device)
    targets_t = torch.tensor(np.asarray(dataset.A, dtype=np.float32), device=device)
    size_factors_t = _poisson_size_factors(dataset.A, device)
    target_latent_t = (
        torch.tensor(target_axis, dtype=torch.float32, device=device)
        .reshape(1, n_cells, 1)
        .expand(n_models, -1, -1)
        .contiguous()
    )

    def _new_model() -> ParallelIsoDepthNet:
        _set_torch_seed(args.seed)
        return ParallelIsoDepthNet(
            n_models, n_genes, latent_dim=1, decoder_type=args.decoder
        ).to(device)

    # --- A. Frozen Gaussian axis (decoder-only on the EXACT z-scored axis) ---
    print("[warmstart] A: frozen Gaussian axis (Poisson decoder only)", flush=True)
    model_a = _new_model()
    hist_a = _train_parallel(
        model_a,
        coords_t,
        targets_t,
        size_factors_t,
        params=model_a.decoder.parameters(),
        epochs=args.epochs,
        lr=args.lr,
        fixed_latent_t=target_latent_t,
    )

    # --- B. Warm-start: encoder pretrained to axis + decoder copied from A ---
    print("[warmstart] B: warm-start (encoder->axis, decoder<-A), then free", flush=True)
    model_b = _new_model()
    warm_corr = _pretrain_encoder_to_axis(
        model_b, coords_t, target_latent_t, epochs=args.pretrain_epochs, lr=args.pretrain_lr
    )
    model_b.decoder.load_state_dict(model_a.decoder.state_dict())
    print(f"[warmstart]   encoder->axis warm-start |corr|={warm_corr:.4f}", flush=True)
    hist_b = _train_parallel(
        model_b,
        coords_t,
        targets_t,
        size_factors_t,
        params=model_b.parameters(),
        epochs=args.epochs,
        lr=args.lr,
    )

    # --- C. Cold-start: default init, full model ---
    print("[warmstart] C: cold-start (default init), free", flush=True)
    model_c = _new_model()
    hist_c = _train_parallel(
        model_c,
        coords_t,
        targets_t,
        size_factors_t,
        params=model_c.parameters(),
        epochs=args.epochs,
        lr=args.lr,
    )

    spatial = np.asarray(dataset.S, dtype=np.float32)
    summary: dict = {
        "experiment": spec.get("experiment_name", "dlpfc_layer3_poisson_warmstart"),
        "run_name": args.run_name,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_reruns": n_models,
        "epochs": int(args.epochs),
        "pretrain_epochs": int(args.pretrain_epochs),
        "warmstart_encoder_axis_abs_corr": warm_corr,
    }

    for mode in ("min", "median"):
        mode_stats, bundle = _selection_summary(
            mode=mode,
            hist_a=hist_a,
            hist_b=hist_b,
            hist_c=hist_c,
            model_b=model_b,
            model_c=model_c,
            coords_t=coords_t,
            target_axis=target_axis,
        )
        summary.update(mode_stats)
        selection_label = "minimum" if mode == "min" else "median"
        _plot_loss_curves(
            bundle["experiments"],
            out_dir / f"{args.run_name}_loss_curves_{mode}.png",
            mode_stats,
            selection_label=selection_label,
        )
        _plot_isodepth_comparison(
            spatial,
            bundle["axes"],
            mode_stats,
            out_dir / f"{args.run_name}_isodepths_{mode}.png",
            selection_label=selection_label,
        )

    _write_summary(summary, out_dir, args.run_name)
    return summary


def _summary_key(prefix: str, field: str) -> str:
    return f"{prefix}{field}"


def _plot_isodepth_comparison(
    spatial: np.ndarray,
    axes: dict[str, np.ndarray],
    summary: dict,
    out_path: Path,
    *,
    selection_label: str,
) -> None:
    """Side-by-side spatial scatter of the three isodepth axes."""
    prefix = "min_" if selection_label == "minimum" else "median_"
    panel_titles = {
        "A: frozen Gaussian axis": (
            f"A: frozen Gaussian axis\n"
            f"Poisson NLL={summary[_summary_key(prefix, 'final_loss_frozen_gaussian')]:.4f}"
        ),
        "B: warm-start": (
            f"B: warm-start\n"
            f"Poisson NLL={summary[_summary_key(prefix, 'final_loss_warm')]:.4f}  "
            f"|rho|(gauss)={summary[_summary_key(prefix, 'abs_spearman_warm_vs_gaussian')]:.3f}"
        ),
        "C: cold-start": (
            f"C: cold-start\n"
            f"Poisson NLL={summary[_summary_key(prefix, 'final_loss_cold')]:.4f}  "
            f"|rho|(gauss)={summary[_summary_key(prefix, 'abs_spearman_cold_vs_gaussian')]:.3f}"
        ),
    }
    fig, plot_axes = plt.subplots(1, 3, figsize=(18, 5.5), squeeze=False)
    for ax, (label, depth) in zip(plot_axes.flat, axes.items()):
        _plot_spatial_isodepth(
            ax,
            spatial,
            np.asarray(depth, dtype=np.float32),
            panel_titles.get(label, label),
        )
    fig.suptitle(
        f"DLPFC Layer 3 \u2014 converged isodepth axes ({selection_label} rerun, Poisson loss)\n"
        f"|rho|(warm,cold)={summary[_summary_key(prefix, 'abs_spearman_warm_vs_cold')]:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[warmstart] wrote {out_path}", flush=True)


def _plot_loss_curves(
    experiments: dict,
    out_path: Path,
    summary: dict,
    *,
    selection_label: str,
) -> None:
    prefix = "min_" if selection_label == "minimum" else "median_"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = list(experiments.items())
    # Per-experiment panels: log y so the full descent AND the fine converged
    # separation are both visible (frozen/cold start ~110, warm starts ~1.07).
    for ax, (label, (history, selected, color)) in zip(axes.flat[:3], panels):
        epochs = np.arange(history.shape[0])
        for m in range(history.shape[1]):
            ax.plot(epochs, history[:, m], color="0.82", lw=0.35, zorder=1, alpha=0.7)
        ax.plot(
            epochs,
            history[:, selected],
            color=color,
            lw=2.0,
            zorder=3,
            label=f"{selection_label} rerun (final={history[-1, selected]:.4f})",
        )
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Poisson NLL (log)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3, which="both")

    # Overlay zoomed to the converged tail so the warm/cold/frozen final gap shows.
    ax = axes.flat[3]
    tail_vals: list[float] = []
    for label, (history, selected, color) in panels:
        epochs = np.arange(history.shape[0])
        ax.plot(epochs, history[:, selected], color=color, lw=2.0, label=label)
        tail_start = max(1, int(history.shape[0] * 0.2))
        tail_vals.append(float(history[tail_start:, selected].max()))
        tail_vals.append(float(history[-1, selected]))
    finals = [
        summary[_summary_key(prefix, "final_loss_frozen_gaussian")],
        summary[_summary_key(prefix, "final_loss_warm")],
        summary[_summary_key(prefix, "final_loss_cold")],
    ]
    lo = min(finals) - 0.004
    hi = max(tail_vals)
    ax.set_ylim(lo, hi)
    ax.set_xlim(left=max(1, int(panels[0][1][0].shape[0] * 0.2)))
    ax.set_title(f"{selection_label.capitalize()} rerun per experiment (converged tail)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Poisson NLL")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"DLPFC Layer 3 \u2014 warm-start vs cold-start Poisson ({selection_label} rerun)\n"
        f"|rho|(warm,gauss)={summary[_summary_key(prefix, 'abs_spearman_warm_vs_gaussian')]:.3f}  "
        f"|rho|(cold,gauss)={summary[_summary_key(prefix, 'abs_spearman_cold_vs_gaussian')]:.3f}  "
        f"|rho|(warm,cold)={summary[_summary_key(prefix, 'abs_spearman_warm_vs_cold')]:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[warmstart] wrote {out_path}", flush=True)


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
    print(f"[warmstart] wrote {json_path}", flush=True)
    print(f"[warmstart] wrote {csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--run-name", default="layer3_poisson_warmstart")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--pretrain-epochs", type=int, default=500)
    parser.add_argument("--n-reruns", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-lr", type=float, default=1e-2)
    parser.add_argument("--decoder", default="nn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    summary = run(spec, args)

    print("\n=== Layer 3 warm-start vs cold-start Poisson ===")
    print(f"warm-start encoder->axis |corr|: {summary['warmstart_encoder_axis_abs_corr']:.4f}")
    for mode, label in (("min", "minimum"), ("median", "median")):
        print(f"\n[{label} rerun]")
        print(
            "final Poisson NLL:  "
            f"frozen={summary[f'{mode}_final_loss_frozen_gaussian']:.5f}  "
            f"warm={summary[f'{mode}_final_loss_warm']:.5f}  "
            f"cold={summary[f'{mode}_final_loss_cold']:.5f}"
        )
        print(
            "converged-axis |Spearman|:  "
            f"warm-vs-gauss={summary[f'{mode}_abs_spearman_warm_vs_gaussian']:.3f}  "
            f"cold-vs-gauss={summary[f'{mode}_abs_spearman_cold_vs_gaussian']:.3f}  "
            f"warm-vs-cold={summary[f'{mode}_abs_spearman_warm_vs_cold']:.3f}"
        )


if __name__ == "__main__":
    main()
