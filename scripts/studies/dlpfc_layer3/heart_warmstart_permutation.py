"""Symmetric Gaussian warm-start permutation test on MOSTA E10.5 Heart.

Question: Poisson existence tests have lower power than Gaussian. Is that power
deficit a genuine specificity gain, or an optimization artifact (cold Poisson SGD
getting stuck in bad basins so the *true* model can't beat its permutation null)?

This script runs three permutation existence tests on the SAME Heart subset with
the SAME permutation set (slot 0 = true coordinates; slots 1..n_perms = spatially
permuted coordinates; expression shared), differing only in how each model is
trained:

  1. **poisson_cold**   — random init, Poisson NLL on raw counts (the baseline).
  2. **gaussian_cpm**   — random init, MSE on log-CPM + standardized expression
                          (the high-power reference).
  3. **poisson_warm**   — random init, short MSE warm-up on log-CPM, then switch to
                          Poisson NLL on raw counts (encoder stays trainable).

Crucially the warm-up is applied SYMMETRICALLY to the true and permuted models, so
calibration is preserved: permuted data has no spatial structure for the warm-up to
latch onto. If poisson_warm recovers power toward gaussian_cpm, the Poisson deficit
was optimization, not specificity.

Each slot uses ``n_reruns`` independent inits; the per-slot statistic is the best
(lowest) final loss over reruns, matching the main pipeline's rerun selection.

Rerun layout (symmetric for true and every permutation):
  - Slot 0 (true coordinates): ``n_reruns`` parallel models → take ``min`` loss.
  - Slots 1..n_perms (permuted coordinates): each gets its own ``n_reruns`` → ``min`` each.
  Default ``n_reruns`` is 10 (``schedule.n_reruns_permutation``), matching ``run_config`` configs.

Outputs (results/mouse-organogenesis/E10.5_heart_warmstart_permutation/):
  - *_null_distributions.png   — null histogram + true stat per method
  - *_isodepths.png            — final true isodepth (best Poisson/MSE rerun) per method
  - *_poisson_warm_switch_isodepths.png — poisson_warm at Gaussian→Poisson handoff vs final
  - *_summary.json / .csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

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
from methods.architectures import ParallelIsoDepthNet  # noqa: E402
from methods.metrics import permutation_p_value  # noqa: E402
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

DEFAULT_SPEC = (
    REPO
    / "configs/experiments/mouse_organogenesis_E10.5_heart_gaussian_cpm_poisson_schedule.json"
)


def _build_permuted_coords(
    S: np.ndarray, n_perms: int, seed: int, device: torch.device
) -> torch.Tensor:
    """(n_perms+1, N, 2): slot 0 true coords, slots 1.. spatially permuted."""
    n_cells = S.shape[0]
    s_t = torch.tensor(np.asarray(S, dtype=np.float32), device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    slots = torch.zeros((n_perms + 1, n_cells, 2), dtype=torch.float32, device=device)
    slots[0] = s_t
    for k in range(1, n_perms + 1):
        perm = torch.randperm(n_cells, generator=generator).to(device=device)
        slots[k] = s_t[perm]
    return slots


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
) -> None:
    n_models = coords_t.shape[0]
    permutation = torch.randperm(n_cells, generator=minibatch_generator)
    for start in range(0, n_cells, sgd_batch_size):
        batch_indices = permutation[start : start + sgd_batch_size].to(device=coords_t.device)
        batch_s = coords_t.index_select(1, batch_indices)
        batch_a = targets_t.index_select(0, batch_indices)
        batch_sf = None if size_factors_t is None else size_factors_t.index_select(0, batch_indices)
        optimizer.zero_grad()
        batch_output = model(batch_s)
        loss_per_model = _compute_reconstruction_loss_per_model(
            batch_output, batch_a, None, poisson_size_factors=batch_sf
        )
        (loss_per_model.sum() / n_models).backward()
        optimizer.step()


def _train(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor | None,
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
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for _ in range(epochs):
        _sgd_epoch(
            model,
            coords_t,
            targets_t,
            size_factors_t,
            optimizer=optimizer,
            sgd_batch_size=sgd_batch_size,
            n_cells=n_cells,
            minibatch_generator=gen,
        )


def _train_warm_then_poisson(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    cpm_t: torch.Tensor,
    raw_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    *,
    warm_epochs: int,
    poisson_epochs: int,
    n_perms: int,
    n_reruns: int,
    lr: float,
    sgd_batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Gaussian warm-up then Poisson fine-tuning without resetting Adam or batch order.

    Two separate ``_train`` calls would recreate the optimizer (and replay the same
    minibatch seed) at the switch, which breaks Poisson adaptation for the true slot.

    Returns ``(warm_mse_per_model, isodepth_at_switch, warm_stat_true, best_rerun_at_switch)``.
    """
    n_cells = coords_t.shape[1]
    optimizer = optim.Adam(model.parameters(), lr=lr, foreach=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for _ in range(warm_epochs):
        _sgd_epoch(
            model,
            coords_t,
            cpm_t,
            None,
            optimizer=optimizer,
            sgd_batch_size=sgd_batch_size,
            n_cells=n_cells,
            minibatch_generator=gen,
        )
    warm_mse_per_model = _eval_loss_per_model(model, coords_t, cpm_t, None)
    per_slot_warm = _per_slot_best(warm_mse_per_model, n_perms, n_reruns)
    warm_stat_true = float(per_slot_warm[0])
    best_rerun_at_switch = _best_true_rerun_index(warm_mse_per_model, n_perms, n_reruns)
    isodepth_at_switch = _true_isodepth_for_model(model, coords_t, best_rerun_at_switch)
    for _ in range(poisson_epochs):
        _sgd_epoch(
            model,
            coords_t,
            raw_t,
            size_factors_t,
            optimizer=optimizer,
            sgd_batch_size=sgd_batch_size,
            n_cells=n_cells,
            minibatch_generator=gen,
        )
    return warm_mse_per_model, isodepth_at_switch, warm_stat_true, best_rerun_at_switch


def _eval_loss_per_model(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    targets_t: torch.Tensor,
    size_factors_t: torch.Tensor | None,
    *,
    cell_chunk: int = 128,
) -> np.ndarray:
    """Mean loss over (cells, genes) per model, evaluated in cell chunks (memory-safe)."""
    n_models = coords_t.shape[0]
    n_cells = coords_t.shape[1]
    n_genes = targets_t.shape[1]
    total = torch.zeros(n_models, dtype=torch.float64, device=coords_t.device)
    with torch.no_grad():
        for start in range(0, n_cells, cell_chunk):
            sl = slice(start, start + cell_chunk)
            output = model(coords_t[:, sl, :])
            tg = targets_t[sl]
            if size_factors_t is not None:
                sf = size_factors_t[sl]
                el = sf * torch.exp(output) - tg * output
            else:
                el = (output - tg) ** 2
            total += el.sum(dim=(1, 2)).to(torch.float64)
    return (total / (n_cells * n_genes)).cpu().numpy()


def _expand_coords_with_reruns(
    slots: torch.Tensor, n_reruns: int
) -> torch.Tensor:
    """Repeat each coordinate slot ``n_reruns`` times: true and nulls treated identically.

    Input ``slots``: ``(n_perms + 1, N, 2)`` with slot 0 = true coords.
    Output: ``((n_perms + 1) * n_reruns, N, 2)`` in slot-major order so
    ``reshape(n_perms + 1, n_reruns)`` yields all reruns for each slot.
    """
    if n_reruns <= 0:
        raise ValueError(f"n_reruns must be > 0, got {n_reruns}")
    return slots.repeat_interleave(n_reruns, dim=0).contiguous()


def _per_slot_best(stat_per_model: np.ndarray, n_perms: int, n_reruns: int) -> np.ndarray:
    """Reshape (slots*n_reruns,) -> (slots,) taking min loss per slot (true + each null)."""
    expected = (n_perms + 1) * n_reruns
    if stat_per_model.size != expected:
        raise ValueError(
            f"expected {expected} model losses for {n_perms + 1} slots x {n_reruns} reruns, "
            f"got {stat_per_model.size}"
        )
    reshaped = stat_per_model.reshape(n_perms + 1, n_reruns)
    return reshaped.min(axis=1)


def _true_isodepth_for_model(
    model: ParallelIsoDepthNet,
    coords_t: torch.Tensor,
    model_index: int,
) -> np.ndarray:
    with torch.no_grad():
        latent = model.encoder(coords_t)
    return latent[model_index, :, 0].detach().cpu().numpy().astype(np.float64)


def _best_true_rerun_index(
    stat_per_model: np.ndarray, n_perms: int, n_reruns: int
) -> int:
    slot0_stats = stat_per_model.reshape(n_perms + 1, n_reruns)[0]
    return int(np.argmin(slot0_stats))


def _run_method(
    name: str,
    *,
    coords_t: torch.Tensor,
    raw_t: torch.Tensor,
    cpm_t: torch.Tensor,
    size_factors_t: torch.Tensor,
    n_perms: int,
    n_reruns: int,
    n_genes: int,
    total_epochs: int,
    warm_epochs: int,
    lr: float,
    sgd_batch_size: int,
    decoder: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    n_models = (n_perms + 1) * n_reruns
    _set_torch_seed(seed)
    model = ParallelIsoDepthNet(n_models, n_genes, latent_dim=1, decoder_type=decoder).to(device)

    isodepth_at_poisson_switch: np.ndarray | None = None
    warm_stat_true: float | None = None
    true_best_rerun_at_switch: int | None = None

    if name == "poisson_cold":
        _train(model, coords_t, raw_t, size_factors_t, epochs=total_epochs, lr=lr,
               sgd_batch_size=sgd_batch_size, seed=seed)
        eval_targets, eval_sf, metric = raw_t, size_factors_t, "nll_poisson_mse"
    elif name == "gaussian_cpm":
        _train(model, coords_t, cpm_t, None, epochs=total_epochs, lr=lr,
               sgd_batch_size=sgd_batch_size, seed=seed)
        eval_targets, eval_sf, metric = cpm_t, None, "mse"
    elif name == "poisson_warm":
        warm_mse_per_model, isodepth_at_poisson_switch, warm_stat_true, true_best_rerun_at_switch = (
            _train_warm_then_poisson(
                model,
                coords_t,
                cpm_t,
                raw_t,
                size_factors_t,
                warm_epochs=warm_epochs,
                poisson_epochs=total_epochs - warm_epochs,
                n_perms=n_perms,
                n_reruns=n_reruns,
                lr=lr,
                sgd_batch_size=sgd_batch_size,
                seed=seed,
            )
        )
        eval_targets, eval_sf, metric = raw_t, size_factors_t, "nll_poisson_mse"
    else:
        raise ValueError(f"unknown method {name}")

    stat_per_model = _eval_loss_per_model(model, coords_t, eval_targets, eval_sf)
    per_slot = _per_slot_best(stat_per_model, n_perms, n_reruns)
    stat_true = float(per_slot[0])
    stat_perm = per_slot[1:]
    p_value = permutation_p_value(metric, stat_true, stat_perm)
    perm_mean = float(stat_perm.mean())
    perm_std = float(stat_perm.std())
    z = (perm_mean - stat_true) / perm_std if perm_std > 0 else 0.0

    # True-model (slot 0) isodepth from its best rerun, for the spatial panels.
    best_rerun = _best_true_rerun_index(stat_per_model, n_perms, n_reruns)
    true_model_index = best_rerun  # slot 0 occupies rows [0, n_reruns)
    isodepth_true = _true_isodepth_for_model(model, coords_t, true_model_index)

    print(
        f"  {name:14s} metric={metric:16s} stat_true={stat_true:.5f} "
        f"perm_mean={perm_mean:.5f} p={p_value:.4f}",
        flush=True,
    )
    if warm_stat_true is not None:
        print(
            f"    poisson_warm @ switch: warm MSE={warm_stat_true:.5f} "
            f"(rerun {true_best_rerun_at_switch})",
            flush=True,
        )

    result: dict[str, Any] = {
        "method": name,
        "metric": metric,
        "stat_true": stat_true,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
        "p_value": p_value,
        "z_effect": z,
        "n_better_or_equal_perms": int(np.sum(stat_perm <= stat_true)),
        "stat_perm": stat_perm.tolist(),
        "n_reruns_per_slot": int(n_reruns),
        "true_best_rerun_index": int(best_rerun),
        "isodepth_true": isodepth_true.tolist(),
    }
    if isodepth_at_poisson_switch is not None:
        result["warm_stat_true"] = warm_stat_true
        result["true_best_rerun_at_switch"] = true_best_rerun_at_switch
        result["isodepth_at_poisson_switch"] = isodepth_at_poisson_switch.tolist()
    return result


def _plot(results: list[dict], out_path: Path, *, n_perms: int) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.2), squeeze=False)
    for ax, res in zip(axes[0], results):
        perms = np.asarray(res["stat_perm"], dtype=np.float64)
        finite = perms[np.isfinite(perms)]
        ax.hist(finite, bins=40, color="#bdbdbd", edgecolor="none")
        ax.axvline(res["stat_true"], color="#c0392b", lw=2.0,
                   label=f"true = {res['stat_true']:.4f}")
        ax.set_title(
            f"{res['method']}\n({res['metric']})\np={res['p_value']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("permutation-null statistic (lower = better fit)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"E10.5 Heart — permutation existence test ({n_perms} perms): "
        "cold Poisson vs Gaussian(log-CPM) vs warm-start Poisson",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[warmstart-perm] wrote {out_path}", flush=True)


def _plot_poisson_warm_switch_isodepths(
    poisson_warm: dict,
    gaussian_cpm: dict,
    spatial: np.ndarray,
    out_path: Path,
    *,
    warm_epochs: int,
) -> None:
    """Compare poisson_warm isodepth at Gaussian handoff vs after Poisson vs pure Gaussian."""
    switch = np.asarray(poisson_warm["isodepth_at_poisson_switch"], dtype=np.float32)
    final = np.asarray(poisson_warm["isodepth_true"], dtype=np.float32)
    gauss = np.asarray(gaussian_cpm["isodepth_true"], dtype=np.float32)
    warm_mse = poisson_warm.get("warm_stat_true")
    warm_title = (
        f"poisson_warm @ switch\n(Gaussian MSE, epoch {warm_epochs})"
        + (f"\nMSE={warm_mse:.4f}" if warm_mse is not None else "")
    )
    panels = [
        (switch, warm_title),
        (
            final,
            f"poisson_warm @ final\n(Poisson NLL)  p={poisson_warm['p_value']:.4f}",
        ),
        (
            gauss,
            f"gaussian_cpm @ final\n(MSE)  p={gaussian_cpm['p_value']:.4f}",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), squeeze=False)
    for ax, (depth, title) in zip(axes[0], panels):
        _plot_spatial_isodepth(ax, spatial, depth, title)
    fig.suptitle(
        "E10.5 Heart — poisson_warm isodepth at Gaussian→Poisson switch vs after Poisson",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[warmstart-perm] wrote {out_path}", flush=True)


def _plot_isodepths(results: list[dict], spatial: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4.2), squeeze=False)
    for ax, res in zip(axes[0], results):
        depth = np.asarray(res["isodepth_true"], dtype=np.float32)
        title = f"{res['method']}\n({res['metric']})  p={res['p_value']:.4f}"
        _plot_spatial_isodepth(ax, spatial, depth, title)
    fig.suptitle(
        "E10.5 Heart — true-model isodepth (best rerun) by training scale",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[warmstart-perm] wrote {out_path}", flush=True)


def run(spec: dict, args: argparse.Namespace) -> dict:
    sched = spec["schedule"]
    device = resolve_device(args.device or sched.get("device", "cuda"))
    out_dir = _resolve(spec["output"]["out_dir"]) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_perms = int(args.n_perms)
    n_reruns = int(args.n_reruns if args.n_reruns is not None else sched.get("n_reruns_permutation", 10))
    if n_reruns < 10:
        print(f"[warmstart-perm] warning: n_reruns={n_reruns} < 10; main pipeline uses 10", flush=True)
    total_epochs = int(args.total_epochs)
    warm_epochs = int(args.warm_epochs)
    lr = float(sched["lr"])
    sgd_batch_size = int(sched["sgd_batch_size"])
    decoder = str(sched.get("decoder", "nn"))
    seed = int(sched["seed"])

    prepared = prepare_layer3_h5ad(spec, force=False)
    raw_dataset = load_dataset(_data_config_from_spec(spec["data"], spec["preprocessing"], prepared))
    merged_pp = {**spec["preprocessing"], **spec["gaussian_preprocessing"]}
    cpm_dataset = load_dataset(_data_config_from_spec(spec["data"], merged_pp, prepared))
    if raw_dataset.A.shape != cpm_dataset.A.shape:
        raise ValueError("raw and log-CPM matrices differ in shape")

    n_cells, n_genes = raw_dataset.A.shape
    print(
        f"[warmstart-perm] n={n_cells} genes={n_genes} n_perms={n_perms} n_reruns={n_reruns} "
        f"total_epochs={total_epochs} warm_epochs={warm_epochs} device={device}",
        flush=True,
    )

    slots = _build_permuted_coords(raw_dataset.S, n_perms, seed, device)
    coords_t = _expand_coords_with_reruns(slots, n_reruns)
    raw_t = torch.tensor(np.asarray(raw_dataset.A, dtype=np.float32), device=device)
    cpm_t = torch.tensor(np.asarray(cpm_dataset.A, dtype=np.float32), device=device)
    size_factors_t = torch.tensor(
        np.asarray(raw_dataset.A, dtype=np.float32).sum(axis=1, keepdims=True), device=device
    )

    results: list[dict] = []
    for name in ("poisson_cold", "gaussian_cpm", "poisson_warm"):
        print(f"[warmstart-perm] training {name}", flush=True)
        results.append(
            _run_method(
                name,
                coords_t=coords_t,
                raw_t=raw_t,
                cpm_t=cpm_t,
                size_factors_t=size_factors_t,
                n_perms=n_perms,
                n_reruns=n_reruns,
                n_genes=n_genes,
                total_epochs=total_epochs,
                warm_epochs=warm_epochs,
                lr=lr,
                sgd_batch_size=sgd_batch_size,
                decoder=decoder,
                seed=seed,
                device=device,
            )
        )

    prefix = args.run_name
    _plot(results, out_dir / f"{prefix}_null_distributions.png", n_perms=n_perms)
    _plot_isodepths(
        results,
        np.asarray(raw_dataset.S, dtype=np.float32),
        out_dir / f"{prefix}_isodepths.png",
    )
    by_method = {res["method"]: res for res in results}
    if "poisson_warm" in by_method and "gaussian_cpm" in by_method:
        _plot_poisson_warm_switch_isodepths(
            by_method["poisson_warm"],
            by_method["gaussian_cpm"],
            np.asarray(raw_dataset.S, dtype=np.float32),
            out_dir / f"{prefix}_poisson_warm_switch_isodepths.png",
            warm_epochs=warm_epochs,
        )

    summary = {
        "experiment": "heart_warmstart_permutation",
        "run_name": args.run_name,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_perms": n_perms,
        "n_reruns": n_reruns,
        "total_epochs": total_epochs,
        "warm_epochs": warm_epochs,
        "lr": lr,
        "sgd_batch_size": sgd_batch_size,
        "seed": seed,
        "rerun_selection": "min_train_loss_per_slot",
        "n_slots": int(n_perms + 1),
        "n_reruns_per_slot": n_reruns,
        "methods": [
            {
                k: v
                for k, v in res.items()
                if k not in ("stat_perm", "isodepth_true", "isodepth_at_poisson_switch", "z_effect")
            }
            for res in results
        ],
    }
    json_path = out_dir / f"{prefix}_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    csv_path = out_dir / f"{prefix}_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "metric", "stat_true", "perm_mean", "p_value"])
        for res in results:
            writer.writerow([
                res["method"], res["metric"], f"{res['stat_true']:.6f}",
                f"{res['perm_mean']:.6f}", f"{res['p_value']:.4f}",
            ])
    print(f"[warmstart-perm] wrote {json_path}", flush=True)
    print(f"[warmstart-perm] wrote {csv_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--run-name", default="E10.5_heart_warmstart_permutation")
    parser.add_argument("--n-perms", type=int, default=499)
    parser.add_argument("--n-reruns", type=int, default=None,
                        help="best-of-k inits per slot (default: schedule n_reruns_permutation or 10)")
    parser.add_argument("--total-epochs", type=int, default=500)
    parser.add_argument("--warm-epochs", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    summary = run(spec, args)

    print("\n=== Heart warm-start permutation test ===")
    for res in summary["methods"]:
        print(f"{res['method']:14s}  p={res['p_value']:.4f}  (metric={res['metric']})")


if __name__ == "__main__":
    main()
