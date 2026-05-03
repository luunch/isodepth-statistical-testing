from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_dataset
from experiments.configuration import build_run_config
from methods.architectures import IsoDepthNet
from methods.trainers import resolve_device


DEFAULT_CONFIG_PATH = "configs/merfish_hypothalamus_batchsize_comparison.json"
DEFAULT_SPEC_PATH = "configs/experiments/batchsize_comparison.json"


@dataclass
class BatchSizeComparisonSpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    base_epochs: int
    batch_sizes: list[int]
    device: str | None = None

    def validate(self) -> "BatchSizeComparisonSpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if int(self.base_epochs) <= 0:
            raise ValueError("base_epochs must be > 0")
        if any(int(value) <= 0 for value in self.batch_sizes):
            raise ValueError("batch_sizes entries must be > 0")
        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.base_epochs = int(self.base_epochs)
        self.batch_sizes = [int(value) for value in self.batch_sizes]
        self.device = None if self.device is None else str(self.device)
        return self


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_experiment_section(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    section = payload.get("experiment", {})
    if not isinstance(section, dict):
        raise ValueError("config.experiment must be an object when provided")
    return section


def load_batchsize_comparison_spec(path: str | Path) -> BatchSizeComparisonSpec:
    spec_path = _resolve_repo_path(str(path))
    with spec_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return BatchSizeComparisonSpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(str(payload["base_config"])),
        output_root=_resolve_repo_path(str(payload["output_root"])),
        base_epochs=int(payload["base_epochs"]),
        batch_sizes=[int(value) for value in payload.get("batch_sizes", [512, 256])],
        device=payload.get("device"),
    ).validate()


def _resolve_base_epochs(experiment_section: Mapping[str, Any], test_epochs: int) -> tuple[int, str]:
    """Epoch budget for the full-batch baseline and for scaling mini-batch runs.

    Precedence: ``experiment.base_epochs`` → ``experiment.epochs`` → ``test.epochs``.
    """
    if experiment_section.get("base_epochs") is not None:
        value = int(experiment_section["base_epochs"])
        return value, "experiment.base_epochs"
    if experiment_section.get("epochs") is not None:
        value = int(experiment_section["epochs"])
        return value, "experiment.epochs"
    return int(test_epochs), "test.epochs"


def _resolve_batch_sizes(experiment_section: Mapping[str, Any]) -> list[int]:
    raw = experiment_section.get("batch_sizes", [512, 256])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("experiment.batch_sizes must be a list of positive integers")
    sizes: list[int] = []
    for item in raw:
        b = int(item)
        if b <= 0:
            raise ValueError(f"experiment.batch_sizes entries must be > 0, got {b}")
        sizes.append(b)
    return sizes


def _build_schedule(n_cells: int, base_epochs: int, batch_sizes: list[int]) -> list[dict[str, Any]]:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if base_epochs <= 0:
        raise ValueError("base_epochs must be positive")

    schedule: list[dict[str, Any]] = []
    schedule.append(
        {
            "label": "true_full_batch",
            "batch_size": 0,
            "steps_per_epoch": 1,
            "effective_epochs": int(base_epochs),
            "planned_total_updates": int(base_epochs),
        }
    )
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError(f"experiment.batch_sizes entries must be > 0, got {batch_size}")
        steps_per_epoch = int(math.ceil(float(n_cells) / float(batch_size)))
        effective_epochs = int(max(1, round(float(base_epochs) / float(steps_per_epoch))))
        schedule.append(
            {
                "label": f"batch_{batch_size}",
                "batch_size": int(batch_size),
                "steps_per_epoch": steps_per_epoch,
                "effective_epochs": effective_epochs,
                "planned_total_updates": int(effective_epochs * steps_per_epoch),
            }
        )
    return schedule


def _train_single_setting(
    S: np.ndarray,
    A: np.ndarray,
    *,
    decoder: str,
    lr: float,
    patience: int,
    device: torch.device,
    seed: int,
    effective_epochs: int,
    sgd_batch_size: int,
) -> dict[str, Any]:
    _set_global_seed(seed)
    n_cells, n_genes = A.shape
    model = IsoDepthNet(n_genes, latent_dim=1, decoder_type=decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, foreach=False)

    s_all = torch.tensor(S, dtype=torch.float32, device=device)
    a_all = torch.tensor(A, dtype=torch.float32, device=device)
    minibatch_generator = torch.Generator(device="cpu")
    minibatch_generator.manual_seed(seed)
    resolved_batch = 0 if sgd_batch_size <= 0 else min(int(sgd_batch_size), int(n_cells))

    loss_history: list[float] = []
    epoch_elapsed_sec: list[float] = []
    best_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    executed_epochs = 0
    steps_per_epoch = 1 if resolved_batch <= 0 else int(math.ceil(float(n_cells) / float(resolved_batch)))

    t_train0 = time.perf_counter()
    for epoch in range(int(effective_epochs)):
        if resolved_batch <= 0:
            optimizer.zero_grad()
            preds = model(s_all)
            train_loss = torch.mean((preds - a_all) ** 2)
            train_loss.backward()
            optimizer.step()
        else:
            permutation = torch.randperm(n_cells, generator=minibatch_generator)
            for start in range(0, n_cells, resolved_batch):
                batch_indices = permutation[start : start + resolved_batch].to(device=device)
                batch_s = s_all.index_select(0, batch_indices)
                batch_a = a_all.index_select(0, batch_indices)
                optimizer.zero_grad()
                batch_preds = model(batch_s)
                batch_loss = torch.mean((batch_preds - batch_a) ** 2)
                batch_loss.backward()
                optimizer.step()

        with torch.no_grad():
            full_preds = model(s_all)
            full_loss = float(torch.mean((full_preds - a_all) ** 2).detach().cpu().item())
        loss_history.append(full_loss)
        epoch_elapsed_sec.append(float(time.perf_counter() - t_train0))
        executed_epochs = epoch + 1

        if full_loss < (best_loss - 1e-8):
            best_loss = full_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    cumulative_updates = [(i + 1) * steps_per_epoch for i in range(len(loss_history))]

    return {
        "loss_history": [float(value) for value in loss_history],
        "epoch_elapsed_sec": [float(value) for value in epoch_elapsed_sec],
        "cumulative_gradient_updates": [int(u) for u in cumulative_updates],
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "executed_epochs": int(executed_epochs),
        "actual_total_updates": int(executed_epochs * steps_per_epoch),
        "steps_per_epoch": int(steps_per_epoch),
        "resolved_batch_size": int(resolved_batch),
    }


def _render_loss_line_plot(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    metadata_lines: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    for label, x_vals, y_vals in series:
        ax.plot(x_vals, y_vals, linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right")

    fig.subplots_adjust(bottom=0.35)
    metadata_text = "\n".join(metadata_lines)
    fig.text(
        0.01,
        0.01,
        metadata_text,
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _runtime_bar_labels(record: dict[str, Any]) -> str:
    label = str(record.get("label", ""))
    if label == "true_full_batch":
        return "Full batch"
    batch_size = record.get("batch_size")
    if batch_size is not None:
        return f"Batch {int(batch_size)}"
    return label


def _render_runtime_plot(
    records: list[dict[str, Any]],
    metadata_lines: list[str],
    out_path: Path,
) -> None:
    labels = [_runtime_bar_labels(record) for record in records]
    times_sec = [float(record["wall_time_sec"]) for record in records]

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    x_pos = np.arange(len(labels), dtype=np.float64)
    n_bars = max(len(labels), 1)
    bar_colors = plt.cm.tab10(np.linspace(0.0, 1.0, n_bars, endpoint=False))
    bars = ax.bar(x_pos, times_sec, color=bar_colors, edgecolor="0.2", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("Training wall time by batch regime")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    ymax = max(times_sec) if times_sec else 1.0
    for bar, seconds in zip(bars, times_sec):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02 * ymax,
            f"{seconds:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.subplots_adjust(bottom=0.22)
    metadata_text = "\n".join(metadata_lines)
    fig.text(
        0.01,
        0.01,
        metadata_text,
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_record_for_json(record: dict[str, Any]) -> dict[str, Any]:
    """Scalars only — no per-epoch series (those live in the plot files)."""
    return {
        "label": record["label"],
        "batch_size": int(record["batch_size"]),
        "effective_epochs": int(record["effective_epochs"]),
        "steps_per_epoch": int(record["steps_per_epoch"]),
        "planned_total_updates": int(record["planned_total_updates"]),
        "executed_epochs": int(record["executed_epochs"]),
        "actual_total_updates": int(record["actual_total_updates"]),
        "best_loss": float(record["best_loss"]),
        "best_epoch": int(record["best_epoch"]),
        "wall_time_sec": float(record["wall_time_sec"]),
        "resolved_batch_size": int(record["resolved_batch_size"]),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-size loss comparison: full batch vs mini-batches.")
    parser.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Path to experiment spec JSON (base_config + experiment overrides).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to legacy JSON configuration file (used when --spec is not provided).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run-name override for output folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved schedule and exit without training.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    spec_path: Path | None = None
    if args.spec:
        spec_path = _resolve_repo_path(args.spec)
        spec = load_batchsize_comparison_spec(spec_path)
        run_config = build_run_config(str(spec.base_config), {})
        run_name = str(args.run_name) if args.run_name else str(spec.experiment_name)
        out_dir = str(_resolve_repo_path(args.out_dir)) if args.out_dir else str(spec.output_root)
        experiment_section = {
            "base_epochs": int(spec.base_epochs),
            "batch_sizes": list(spec.batch_sizes),
        }
        run_config = replace(
            run_config,
            output=replace(run_config.output, run_name=run_name, out_dir=out_dir),
        )
        if spec.device:
            run_config = replace(
                run_config,
                test=replace(run_config.test, device=str(spec.device)),
            )
    else:
        config_path = _resolve_repo_path(args.config)
        experiment_section = _extract_experiment_section(config_path)
        run_config = build_run_config(str(config_path), {})
        if args.run_name:
            run_config = replace(run_config, output=replace(run_config.output, run_name=args.run_name))
        if args.out_dir:
            out_dir_override = str(_resolve_repo_path(args.out_dir))
            run_config = replace(run_config, output=replace(run_config.output, out_dir=out_dir_override))

    dataset = load_dataset(run_config.data)
    batch_sizes = _resolve_batch_sizes(experiment_section)
    base_epochs, base_epochs_source = _resolve_base_epochs(
        experiment_section,
        int(run_config.test.epochs),
    )
    schedule = _build_schedule(dataset.n_cells, base_epochs, batch_sizes)

    print(f"Loaded dataset from: {run_config.data.h5ad}")
    if spec_path is not None:
        print(f"Loaded experiment spec: {spec_path}")
    print(f"Requested device: {run_config.test.device}")
    print(f"n_cells={dataset.n_cells}, n_genes={dataset.n_genes}")
    print(f"Base epoch budget: {base_epochs} (from {base_epochs_source})")
    print(f"Mini-batch sizes to compare: {batch_sizes if batch_sizes else '(none — full batch only)'}")
    print("Computed schedule:")
    print(json.dumps(schedule, indent=2))
    if args.dry_run:
        return

    device = resolve_device(run_config.test.device)
    print(f"Resolved device: {device}")

    out_dir = Path(run_config.output.out_dir) / run_config.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    series_epoch: list[tuple[str, np.ndarray, np.ndarray]] = []
    series_updates: list[tuple[str, np.ndarray, np.ndarray]] = []
    series_time: list[tuple[str, np.ndarray, np.ndarray]] = []
    for item in schedule:
        label = str(item["label"])
        seed_for_run = int(run_config.test.seed)
        _sync_cuda_if_needed(device)
        t_wall0 = time.perf_counter()
        train_summary = _train_single_setting(
            dataset.S,
            dataset.A,
            decoder=str(run_config.test.decoder),
            lr=float(run_config.test.lr),
            patience=int(run_config.test.patience),
            device=device,
            seed=seed_for_run,
            effective_epochs=int(item["effective_epochs"]),
            sgd_batch_size=int(item["batch_size"]),
        )
        _sync_cuda_if_needed(device)
        wall_time_sec = float(time.perf_counter() - t_wall0)
        train_summary["wall_time_sec"] = wall_time_sec

        plot_label = f"{label} (batch={item['batch_size']}, epochs={item['effective_epochs']})"
        losses = np.asarray(train_summary["loss_history"], dtype=np.float64)
        n = len(losses)
        epochs_x = np.arange(1, n + 1, dtype=np.float64)
        updates_x = np.asarray(train_summary["cumulative_gradient_updates"], dtype=np.float64)
        time_x = np.asarray(train_summary["epoch_elapsed_sec"], dtype=np.float64)
        series_epoch.append((plot_label, epochs_x, losses))
        series_updates.append((plot_label, updates_x, losses))
        series_time.append((plot_label, time_x, losses))
        run_records.append({**item, **train_summary})
        print(f"{label}: wall_time_sec={wall_time_sec:.3f}")

    metadata_lines = [
        f"dataset={Path(str(run_config.data.h5ad)).name}",
        f"n_cells={dataset.n_cells}, n_genes={dataset.n_genes}",
        f"seed={run_config.test.seed}, lr={run_config.test.lr}, patience={run_config.test.patience}",
        f"decoder={run_config.test.decoder}, loss=mse",
        f"base_epochs={base_epochs} ({base_epochs_source}), baseline=batch_size=0 (full batch)",
        "schedule="
        + ", ".join(
            [
                f"{record['label']}:batch={record['batch_size']},epochs={record['effective_epochs']},"
                f"steps/epoch={record['steps_per_epoch']},planned_updates={record['planned_total_updates']}"
                for record in run_records
            ]
        ),
    ]

    stem = f"{run_config.output.run_name}_batchsize"
    plot_epoch_path = out_dir / f"{stem}_loss_vs_epoch.png"
    plot_updates_path = out_dir / f"{stem}_loss_vs_gradient_updates.png"
    plot_time_path = out_dir / f"{stem}_loss_vs_time.png"
    _render_loss_line_plot(
        series_epoch,
        title="Training loss vs epoch",
        xlabel="Epoch",
        ylabel="Training MSE loss",
        metadata_lines=metadata_lines,
        out_path=plot_epoch_path,
    )
    _render_loss_line_plot(
        series_updates,
        title="Training loss vs cumulative gradient updates",
        xlabel="Cumulative gradient updates",
        ylabel="Training MSE loss",
        metadata_lines=metadata_lines,
        out_path=plot_updates_path,
    )
    _render_loss_line_plot(
        series_time,
        title="Training loss vs elapsed wall time",
        xlabel="Elapsed time (s)",
        ylabel="Training MSE loss",
        metadata_lines=metadata_lines,
        out_path=plot_time_path,
    )

    runtime_plot_path = out_dir / f"{stem}_wall_time.png"
    _render_runtime_plot(run_records, metadata_lines, runtime_plot_path)

    results_payload = {
        "spec_path": None if spec_path is None else str(spec_path),
        "dataset": {
            "h5ad": str(run_config.data.h5ad),
            "n_cells": int(dataset.n_cells),
            "n_genes": int(dataset.n_genes),
        },
        "training": {
            "seed": int(run_config.test.seed),
            "lr": float(run_config.test.lr),
            "patience": int(run_config.test.patience),
            "decoder": str(run_config.test.decoder),
            "base_epochs": int(base_epochs),
            "base_epochs_source": base_epochs_source,
            "batch_sizes": list(batch_sizes),
            "loss": "mse",
            "device": str(device),
        },
        "runs": [_run_record_for_json(r) for r in run_records],
        "artifacts": {
            "loss_vs_epoch": str(plot_epoch_path),
            "loss_vs_gradient_updates": str(plot_updates_path),
            "loss_vs_time": str(plot_time_path),
            "runtime_plot": str(runtime_plot_path),
        },
    }
    json_path = out_dir / f"{run_config.output.run_name}_batchsize_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)

    print(f"Saved loss vs epoch: {plot_epoch_path}")
    print(f"Saved loss vs gradient updates: {plot_updates_path}")
    print(f"Saved loss vs time: {plot_time_path}")
    print(f"Saved wall-time plot: {runtime_plot_path}")
    print(f"Saved comparison JSON: {json_path}")


if __name__ == "__main__":
    main()
