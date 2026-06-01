from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import spearmanr

from analysis.plots import _bias_detection_pearson, _plot_spatial_isodepth
from data import load_dataset
from data.schemas import TestConfig
from experiments.configuration import build_run_config
from methods.metrics import (
    canonicalize_metric_name,
    permutation_p_value,
)
from methods.permutation import _extract_slot_isodepths
from methods.trainers import get_training_metadata, resolve_device, train_parallel_isodepth_model


DEFAULT_CONFIG_PATH = "configs/merfish_hypothalamus_batchsize_comparison.json"
DEFAULT_SPEC_PATH = "configs/experiments/batchsize_comparison.json"
# Upper bound on the epoch loop when training until time_budget_sec elapses.
_TIME_MODE_EPOCH_CAP = 10_000_000


@dataclass
class BatchSizeComparisonSpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    base_epochs: int | None
    batch_sizes: list[int]
    device: str | None = None
    time_budget_sec: float | None = None
    n_perms: int | None = None
    n_reruns: int | None = None

    def validate(self) -> "BatchSizeComparisonSpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if self.base_epochs is not None and int(self.base_epochs) <= 0:
            raise ValueError("base_epochs must be > 0 when provided")
        if any(int(value) <= 0 for value in self.batch_sizes):
            raise ValueError("batch_sizes entries must be > 0")
        if self.n_perms is not None and int(self.n_perms) <= 0:
            raise ValueError("n_perms must be > 0 when provided")
        if self.n_reruns is not None and int(self.n_reruns) <= 0:
            raise ValueError("n_reruns must be > 0 when provided")
        if self.time_budget_sec is not None and float(self.time_budget_sec) <= 0:
            raise ValueError("time_budget_sec must be > 0 when provided")
        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.base_epochs = None if self.base_epochs is None else int(self.base_epochs)
        self.batch_sizes = [int(value) for value in self.batch_sizes]
        self.device = None if self.device is None else str(self.device)
        self.time_budget_sec = None if self.time_budget_sec is None else float(self.time_budget_sec)
        self.n_perms = None if self.n_perms is None else int(self.n_perms)
        self.n_reruns = None if self.n_reruns is None else int(self.n_reruns)
        return self


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()



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
    raw_epochs = payload.get("base_epochs")
    raw_time = payload.get("time_budget_sec", payload.get("time"))
    return BatchSizeComparisonSpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(str(payload["base_config"])),
        output_root=_resolve_repo_path(str(payload["output_root"])),
        base_epochs=None if raw_epochs is None else int(raw_epochs),
        batch_sizes=[int(value) for value in payload.get("batch_sizes", [512, 256])],
        device=payload.get("device"),
        time_budget_sec=None if raw_time is None else float(raw_time),
        n_perms=payload.get("n_perms"),
        n_reruns=payload.get("n_reruns"),
    ).validate()


def _resolve_time_budget_sec(experiment_section: Mapping[str, Any]) -> float | None:
    if experiment_section.get("time_budget_sec") is not None:
        value = float(experiment_section["time_budget_sec"])
    elif experiment_section.get("time") is not None:
        value = float(experiment_section["time"])
    else:
        return None
    if value <= 0:
        raise ValueError("time_budget_sec must be > 0 when provided")
    return value


def _resolve_base_epochs(experiment_section: Mapping[str, Any], test_epochs: int) -> tuple[int, str]:
    """Full-batch gradient-update budget (mini-batch regimes are scaled to match).

    Precedence: ``experiment.base_epochs`` -> ``experiment.epochs`` -> ``test.epochs``.
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


def _steps_per_epoch(n_cells: int, batch_size: int) -> int:
    if batch_size <= 0:
        return 1
    return int(math.ceil(float(n_cells) / float(batch_size)))


def _epochs_for_equal_updates(base_updates: int, steps_per_epoch: int) -> int:
    if steps_per_epoch <= 1:
        return int(base_updates)
    return max(1, int(math.ceil(float(base_updates) / float(steps_per_epoch))))


def _build_regime_list(batch_sizes: list[int]) -> list[dict[str, Any]]:
    regimes: list[dict[str, Any]] = [
        {"label": "true_full_batch", "batch_size": 0},
    ]
    for batch_size in batch_sizes:
        regimes.append({"label": f"batch_{batch_size}", "batch_size": int(batch_size)})
    return regimes


def _build_fixed_update_schedule(
    n_cells: int,
    base_updates: int,
    batch_sizes: list[int],
) -> list[dict[str, Any]]:
    """Fixed update budget per regime (used when time_budget_sec is not set)."""
    schedule: list[dict[str, Any]] = []
    for regime in _build_regime_list(batch_sizes):
        batch_size = int(regime["batch_size"])
        steps = _steps_per_epoch(n_cells, batch_size) if batch_size > 0 else 1
        effective_epochs = (
            int(base_updates) if batch_size == 0 else _epochs_for_equal_updates(base_updates, steps)
        )
        schedule.append(
            {
                **regime,
                "steps_per_epoch": steps,
                "effective_epochs": effective_epochs,
                "planned_total_updates": int(effective_epochs) * steps,
            }
        )
    return schedule


def _resolve_regime_test_config(
    run_config_test: TestConfig,
    *,
    batch_size: int,
    n_perms: int,
    n_reruns: int,
    n_cells: int,
    base_updates: int,
    time_budget_sec: float | None,
    record_loss_history: bool = False,
) -> TestConfig:
    if time_budget_sec is not None:
        return replace(
            run_config_test,
            epochs=_TIME_MODE_EPOCH_CAP,
            n_perms=n_perms,
            n_reruns=n_reruns,
            sgd_batch_size=batch_size if batch_size > 0 else None,
            max_wall_time_sec=float(time_budget_sec),
            record_loss_history=record_loss_history,
        )
    steps = _steps_per_epoch(n_cells, batch_size) if batch_size > 0 else 1
    effective_epochs = base_updates if batch_size == 0 else _epochs_for_equal_updates(base_updates, steps)
    return replace(
        run_config_test,
        epochs=int(effective_epochs),
        n_perms=n_perms,
        n_reruns=n_reruns,
        sgd_batch_size=batch_size if batch_size > 0 else None,
        max_wall_time_sec=None,
        record_loss_history=record_loss_history,
    )


def _run_permutation_regime(
    S: np.ndarray,
    A: np.ndarray,
    *,
    test_config: TestConfig,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    """Run the full parallel permutation framework for one batch-size regime.

    Returns a dict with p_value, stat_true, stat_perm, true_isodepth, and timing info.
    """
    t0 = time.perf_counter()
    model, training_outputs, s_batched_np = train_parallel_isodepth_model(S, A, test_config, device=device)
    wall_time_sec = time.perf_counter() - t0

    metric = canonicalize_metric_name(test_config.metric)
    stat_true = float(training_outputs.stat_true)
    stat_perm = training_outputs.stat_perm
    p_value = permutation_p_value(metric, stat_true, stat_perm)

    slot_iso = _extract_slot_isodepths(model, s_batched_np, [0], device)
    true_isodepth = slot_iso[0]
    metadata = get_training_metadata(model)
    executed_epochs = int(metadata.get("executed_epochs") or test_config.epochs)
    executed_gradient_steps = int(
        metadata.get("executed_gradient_steps") or executed_epochs
    )
    loss_history = metadata.get("loss_history")
    loss_history_elapsed_sec = metadata.get("loss_history_elapsed_sec")
    loss_history_gradient_updates = metadata.get("loss_history_gradient_updates")

    return {
        "label": label,
        "p_value": float(p_value),
        "stat_true": float(stat_true),
        "stat_perm": stat_perm,
        "true_isodepth": np.asarray(true_isodepth, dtype=np.float32),
        "wall_time_sec": float(wall_time_sec),
        "n_perms": int(test_config.n_perms),
        "n_reruns": int(test_config.n_reruns),
        "executed_epochs": executed_epochs,
        "executed_gradient_steps": executed_gradient_steps,
        "stopped_by_time": bool(metadata.get("stopped_by_time", False)),
        "loss_history": None if loss_history is None else np.asarray(loss_history, dtype=np.float64),
        "loss_history_elapsed_sec": None
        if loss_history_elapsed_sec is None
        else np.asarray(loss_history_elapsed_sec, dtype=np.float64),
        "loss_history_gradient_updates": None
        if loss_history_gradient_updates is None
        else np.asarray(loss_history_gradient_updates, dtype=np.int64),
    }


def _correlation_to_synthetic(
    learned: np.ndarray,
    true: np.ndarray,
) -> tuple[float, float]:
    learned_flat = np.asarray(learned, dtype=np.float64).reshape(-1)
    true_flat = np.asarray(true, dtype=np.float64).reshape(-1)
    pearson = _bias_detection_pearson(learned_flat, true_flat)
    if learned_flat.size != true_flat.size or learned_flat.size < 2:
        return pearson, float("nan")
    spearman = float(spearmanr(learned_flat, true_flat).correlation)
    return pearson, spearman


def _resolve_synthetic_true_isodepth(dataset_meta: Mapping[str, Any], *, is_synthetic: bool) -> np.ndarray | None:
    if not is_synthetic:
        return None
    raw_true = dataset_meta.get("synthetic_true_curve")
    if raw_true is None:
        return None
    return np.asarray(raw_true, dtype=np.float32).reshape(-1)


def _loss_plot_label(record: dict[str, Any]) -> str:
    batch_size = int(record.get("batch_size", 0))
    if batch_size == 0:
        return "full batch"
    return f"batch={batch_size}"


def _build_loss_plot_series(
    regime_results: list[dict[str, Any]],
) -> tuple[
    list[tuple[str, np.ndarray, np.ndarray]],
    list[tuple[str, np.ndarray, np.ndarray]],
    list[tuple[str, np.ndarray, np.ndarray]],
]:
    series_epoch: list[tuple[str, np.ndarray, np.ndarray]] = []
    series_updates: list[tuple[str, np.ndarray, np.ndarray]] = []
    series_time: list[tuple[str, np.ndarray, np.ndarray]] = []
    for record in regime_results:
        losses = record.get("loss_history")
        if losses is None or len(losses) == 0:
            continue
        plot_label = _loss_plot_label(record)
        loss_arr = np.asarray(losses, dtype=np.float64)
        n = len(loss_arr)
        epochs_x = np.arange(1, n + 1, dtype=np.float64)
        updates_x = np.asarray(record["loss_history_gradient_updates"], dtype=np.float64)
        time_x = np.asarray(record["loss_history_elapsed_sec"], dtype=np.float64)
        series_epoch.append((plot_label, epochs_x, loss_arr))
        series_updates.append((plot_label, updates_x, loss_arr))
        series_time.append((plot_label, time_x, loss_arr))
    return series_epoch, series_updates, series_time


def _render_loss_line_plot(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float | None, float | None] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    for label, x_vals, y_vals in series:
        ax.plot(x_vals, y_vals, linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        lo, hi = ylim
        if lo is not None and hi is not None:
            ax.set_ylim(lo, hi)
        elif lo is not None:
            ax.set_ylim(bottom=lo)
        elif hi is not None:
            ax.set_ylim(top=hi)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_isodepth_grid(
    panels: list[tuple[str, np.ndarray]],
    spatial: np.ndarray,
    *,
    title: str,
    out_path: Path,
) -> None:
    """One subplot per (label, isodepth) panel, all on the same `spatial` (N, 2) coordinates."""
    if not panels:
        return
    n_panels = len(panels)
    n_cols = min(n_panels, 4)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )
    coords = np.asarray(spatial, dtype=np.float32)
    for index, (label, depth) in enumerate(panels):
        ax = axes[index // n_cols][index % n_cols]
        _plot_spatial_isodepth(ax, coords, np.asarray(depth, dtype=np.float32), label)
    for extra in range(n_panels, n_rows * n_cols):
        axes[extra // n_cols][extra % n_cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_metric_distribution_grid(
    regime_results: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    out_path: Path,
) -> None:
    """One subplot per regime showing null distribution histogram + true statistic line."""
    n = len(regime_results)
    n_cols = min(n, 3)
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )
    for idx, record in enumerate(regime_results):
        ax = axes[idx // n_cols][idx % n_cols]
        stat_perm = np.asarray(record["stat_perm"], dtype=np.float64)
        stat_true = float(record["stat_true"])
        p_value = float(record["p_value"])
        label = str(record["label"])

        ax.hist(stat_perm, bins=30, alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(stat_true, color="red", linewidth=2.0, linestyle="--",
                   label=f"True: {stat_true:.4g}")
        ax.set_title(f"{label}\np = {p_value:.4g}", fontsize=10)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, linewidth=0.5)
    for extra in range(n, n_rows * n_cols):
        axes[extra // n_cols][extra % n_cols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_correlation_bar_chart(
    regime_results: list[dict[str, Any]],
    *,
    title: str,
    out_path: Path,
) -> None:
    """Bar chart of Pearson r vs synthetic ground truth across regimes."""
    labels = [str(r["label"]) for r in regime_results]
    pearson_values = [float(r["synthetic_pearson"]) for r in regime_results]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(labels)), 5.0))
    bars = ax.bar(range(len(labels)), pearson_values, color="seagreen", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r vs synthetic isodepth")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, value in zip(bars, pearson_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.03 if value >= 0 else -0.07),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_pvalue_bar_chart(
    regime_results: list[dict[str, Any]],
    *,
    title: str,
    out_path: Path,
) -> None:
    """Bar chart of p-values across regimes."""
    labels = [str(r["label"]) for r in regime_results]
    p_values = [float(r["p_value"]) for r in regime_results]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(labels)), 5.0))
    bars = ax.bar(range(len(labels)), p_values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(0.05, color="red", linewidth=1.5, linestyle="--", label="α = 0.05")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("p-value")
    ax.set_ylim(0, max(max(p_values) * 1.15, 0.1))
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, pv in zip(bars, p_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{pv:.4g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_record_for_json(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": record["label"],
        "batch_size": int(record["batch_size"]),
        "executed_epochs": record.get("executed_epochs"),
        "executed_gradient_steps": record.get("executed_gradient_steps"),
        "stopped_by_time": bool(record.get("stopped_by_time", False)),
        "steps_per_epoch": record.get("steps_per_epoch"),
        "planned_total_updates": record.get("planned_total_updates"),
        "p_value": float(record["p_value"]),
        "stat_true": float(record["stat_true"]),
        "stat_perm_mean": float(np.mean(record["stat_perm"])),
        "stat_perm_std": float(np.std(record["stat_perm"])),
        "n_perms": int(record["n_perms"]),
        "n_reruns": int(record["n_reruns"]),
        "wall_time_sec": float(record["wall_time_sec"]),
        "synthetic_pearson": record.get("synthetic_pearson"),
        "synthetic_spearman": record.get("synthetic_spearman"),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-size comparison with permutation test per regime.")
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
            "base_epochs": spec.base_epochs,
            "batch_sizes": list(spec.batch_sizes),
            "n_perms": spec.n_perms,
            "n_reruns": spec.n_reruns,
            "time_budget_sec": spec.time_budget_sec,
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
        experiment_section = dict(_extract_experiment_section(config_path))
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
    if base_epochs <= 0:
        raise ValueError(
            "base_epochs is required (the parallel permutation trainer is epoch-based). "
            "Set experiment.base_epochs, experiment.epochs, or rely on test.epochs in the base config."
        )

    n_perms = int(experiment_section.get("n_perms") or run_config.test.n_perms)
    n_reruns = int(experiment_section.get("n_reruns") or run_config.test.n_reruns)
    time_budget_sec = _resolve_time_budget_sec(experiment_section)
    base_updates = int(base_epochs)
    use_time_budget = time_budget_sec is not None
    n_regimes = 1 + len(batch_sizes)

    print(f"Loaded dataset from: {run_config.data.h5ad if run_config.data.source == 'h5ad' else run_config.data.source}")
    if spec_path is not None:
        print(f"Loaded experiment spec: {spec_path}")
    print(f"Requested device: {run_config.test.device}")
    print(f"n_cells={dataset.n_cells}, n_genes={dataset.n_genes}")
    print(f"Permutation framework: n_perms={n_perms}, n_reruns={n_reruns}")
    if use_time_budget:
        print(
            f"Training mode: run until {time_budget_sec:.0f}s per regime "
            f"({time_budget_sec / 3600.0:.2f}h); {n_regimes} regimes "
            f"(~{time_budget_sec * n_regimes / 3600.0:.2f}h total if each uses the full budget)"
        )
    else:
        print(f"Training mode: fixed update budget = {base_updates} (from {base_epochs_source})")
    print(f"Mini-batch sizes to compare: {batch_sizes if batch_sizes else '(none — full batch only)'}")
    regimes = _build_regime_list(batch_sizes)
    if args.dry_run:
        if use_time_budget:
            print("Regimes (each trains until time_budget_sec elapses):")
            print(json.dumps(regimes, indent=2))
        else:
            schedule = _build_fixed_update_schedule(dataset.n_cells, base_updates, batch_sizes)
            print("Fixed update schedule (equal gradient steps per regime):")
            print(json.dumps(schedule, indent=2))
        return

    device = resolve_device(run_config.test.device)
    print(f"Resolved device: {device}")

    out_dir_path = Path(run_config.output.out_dir) / run_config.output.run_name
    out_dir_path.mkdir(parents=True, exist_ok=True)

    regime_results: list[dict[str, Any]] = []
    isodepth_panels: list[tuple[str, np.ndarray]] = []
    metric = canonicalize_metric_name(run_config.test.metric)
    is_synthetic = str(run_config.data.source) == "synthetic"
    true_isodepth_arr = _resolve_synthetic_true_isodepth(dataset.meta, is_synthetic=is_synthetic)

    for item in regimes:
        label = str(item["label"])
        sgd_batch = int(item["batch_size"])
        steps_per_epoch = _steps_per_epoch(dataset.n_cells, sgd_batch) if sgd_batch > 0 else 1
        regime_config = _resolve_regime_test_config(
            run_config.test,
            batch_size=sgd_batch,
            n_perms=n_perms,
            n_reruns=n_reruns,
            n_cells=int(dataset.n_cells),
            base_updates=base_updates,
            time_budget_sec=time_budget_sec,
            record_loss_history=True,
        )

        print(f"\n{'='*60}")
        print(f"Running regime: {label} (sgd_batch_size={sgd_batch or 'full'})")
        if use_time_budget:
            print(
                f"  max_wall_time_sec={time_budget_sec:.0f}, steps/epoch={steps_per_epoch}, "
                f"n_perms={n_perms}, n_reruns={n_reruns}"
            )
        else:
            print(
                f"  epochs={regime_config.epochs}, steps/epoch={steps_per_epoch}, "
                f"planned_updates={int(regime_config.epochs) * steps_per_epoch}, "
                f"n_perms={n_perms}, n_reruns={n_reruns}"
            )
        print(f"{'='*60}")

        _sync_cuda_if_needed(device)
        result = _run_permutation_regime(
            dataset.S,
            dataset.A,
            test_config=regime_config,
            device=device,
            label=label,
        )
        _sync_cuda_if_needed(device)

        result["batch_size"] = sgd_batch
        result["steps_per_epoch"] = steps_per_epoch
        if not use_time_budget:
            result["planned_total_updates"] = int(regime_config.epochs) * steps_per_epoch
        if true_isodepth_arr is not None:
            pearson, spearman = _correlation_to_synthetic(result["true_isodepth"], true_isodepth_arr)
            result["synthetic_pearson"] = float(pearson)
            result["synthetic_spearman"] = float(spearman)
        regime_results.append(result)

        panel_label = (
            f"full batch (p={result['p_value']:.3g})"
            if sgd_batch == 0
            else f"batch={sgd_batch} (p={result['p_value']:.3g})"
        )
        if result.get("synthetic_pearson") is not None:
            panel_label += f", r={result['synthetic_pearson']:.3f}"
        isodepth_panels.append(
            (panel_label, np.asarray(result["true_isodepth"], dtype=np.float32))
        )
        corr_msg = ""
        if result.get("synthetic_pearson") is not None:
            corr_msg = (
                f", pearson={result['synthetic_pearson']:.4g}, "
                f"spearman={result['synthetic_spearman']:.4g}"
            )
        print(
            f"  -> p_value={result['p_value']:.4g}, stat_true={result['stat_true']:.4g}, "
            f"wall_time={result['wall_time_sec']:.1f}s, "
            f"updates={result['executed_gradient_steps']}{corr_msg}"
        )

    stem = f"{run_config.output.run_name}_batchsize"

    series_epoch, series_updates, series_time = _build_loss_plot_series(regime_results)
    plot_epoch_path = out_dir_path / f"{stem}_loss_vs_epoch.png"
    plot_updates_path = out_dir_path / f"{stem}_loss_vs_gradient_updates.png"
    plot_time_path = out_dir_path / f"{stem}_loss_vs_time.png"
    if series_epoch:
        _render_loss_line_plot(
            series_epoch,
            title="Training loss vs epoch (true layout, slot 0)",
            xlabel="Epoch",
            ylabel="Training MSE loss",
            out_path=plot_epoch_path,
        )
        _render_loss_line_plot(
            series_updates,
            title="Training loss vs cumulative gradient updates (true layout, slot 0)",
            xlabel="Cumulative gradient updates",
            ylabel="Training MSE loss",
            out_path=plot_updates_path,
        )
        _render_loss_line_plot(
            series_time,
            title="Training loss vs elapsed wall time (true layout, slot 0)",
            xlabel="Elapsed time (s)",
            ylabel="Training MSE loss",
            out_path=plot_time_path,
        )

    # Metric distribution grid
    plot_metric_dist_path = out_dir_path / f"{stem}_metric_distributions.png"
    _render_metric_distribution_grid(
        regime_results,
        metric=metric,
        title=f"Null distributions per batch-size regime ({metric})",
        out_path=plot_metric_dist_path,
    )

    # P-value bar chart
    plot_pvalue_path = out_dir_path / f"{stem}_pvalues.png"
    _render_pvalue_bar_chart(
        regime_results,
        title="Permutation test p-value per batch-size regime",
        out_path=plot_pvalue_path,
    )

    # Isodepth grid
    grid_panels: list[tuple[str, np.ndarray]] = []
    if true_isodepth_arr is not None:
        grid_panels.append(("True isodepth (synthetic)", true_isodepth_arr))
    grid_panels.extend(isodepth_panels)
    plot_isodepth_grid_path = out_dir_path / f"{stem}_learned_isodepths.png"
    _render_isodepth_grid(
        grid_panels,
        np.asarray(dataset.S, dtype=np.float32),
        title="Learned isodepth per regime"
        + (" (with synthetic ground truth)" if true_isodepth_arr is not None else ""),
        out_path=plot_isodepth_grid_path,
    )

    plot_synthetic_corr_path: Path | None = None
    if true_isodepth_arr is not None and all(
        record.get("synthetic_pearson") is not None for record in regime_results
    ):
        plot_synthetic_corr_path = out_dir_path / f"{stem}_synthetic_correlations.png"
        _render_correlation_bar_chart(
            regime_results,
            title="Pearson correlation to synthetic isodepth per batch-size regime",
            out_path=plot_synthetic_corr_path,
        )

    # NPZ with isodepths
    isodepths_npz_path = out_dir_path / f"{stem}_learned_isodepths.npz"
    npz_payload: dict[str, np.ndarray] = {
        "S": np.asarray(dataset.S, dtype=np.float32),
    }
    for record in regime_results:
        safe_key = str(record["label"]).replace(" ", "_")
        npz_payload[f"isodepth__{safe_key}"] = np.asarray(record["true_isodepth"], dtype=np.float32)
    if true_isodepth_arr is not None:
        npz_payload["true_isodepth"] = true_isodepth_arr
    np.savez(isodepths_npz_path, **npz_payload)

    # JSON results
    artifacts_payload: dict[str, str] = {
        "metric_distributions": str(plot_metric_dist_path),
        "pvalues_plot": str(plot_pvalue_path),
        "learned_isodepths_plot": str(plot_isodepth_grid_path),
        "learned_isodepths_data": str(isodepths_npz_path),
    }
    if series_epoch:
        artifacts_payload["loss_vs_epoch"] = str(plot_epoch_path)
        artifacts_payload["loss_vs_gradient_updates"] = str(plot_updates_path)
        artifacts_payload["loss_vs_time"] = str(plot_time_path)
    if plot_synthetic_corr_path is not None:
        artifacts_payload["synthetic_correlations_plot"] = str(plot_synthetic_corr_path)

    results_payload = {
        "spec_path": None if spec_path is None else str(spec_path),
        "dataset": {
            "source": str(run_config.data.source),
            "h5ad": str(run_config.data.h5ad) if run_config.data.source == "h5ad" else None,
            "n_cells": int(dataset.n_cells),
            "n_genes": int(dataset.n_genes),
        },
        "training": {
            "seed": int(run_config.test.seed),
            "lr": float(run_config.test.lr),
            "patience": int(run_config.test.patience),
            "decoder": str(run_config.test.decoder),
            "training_mode": "time_budget_per_regime" if use_time_budget else "fixed_updates",
            "base_updates": None if use_time_budget else int(base_updates),
            "base_updates_source": None if use_time_budget else base_epochs_source,
            "time_budget_sec": time_budget_sec,
            "time_budget_per_regime": True,
            "n_perms": int(n_perms),
            "n_reruns": int(n_reruns),
            "metric": str(metric),
            "batch_sizes": list(batch_sizes),
            "device": str(device),
        },
        "runs": [_run_record_for_json(r) for r in regime_results],
        "synthetic_ground_truth_available": true_isodepth_arr is not None,
        "artifacts": artifacts_payload,
    }
    json_path = out_dir_path / f"{run_config.output.run_name}_batchsize_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)

    print(f"\nSaved metric distributions: {plot_metric_dist_path}")
    print(f"Saved p-value bar chart: {plot_pvalue_path}")
    if series_epoch:
        print(f"Saved loss vs epoch: {plot_epoch_path}")
        print(f"Saved loss vs gradient updates: {plot_updates_path}")
        print(f"Saved loss vs time: {plot_time_path}")
    if plot_synthetic_corr_path is not None:
        print(f"Saved synthetic correlation bar chart: {plot_synthetic_corr_path}")
    print(f"Saved learned isodepth grid: {plot_isodepth_grid_path}")
    print(f"Saved learned isodepth arrays: {isodepths_npz_path}")
    print(f"Saved comparison JSON: {json_path}")
    print("\nSummary:")
    for record in regime_results:
        corr_suffix = ""
        if record.get("synthetic_pearson") is not None:
            corr_suffix = (
                f"  pearson={record['synthetic_pearson']:.4g}  "
                f"spearman={record['synthetic_spearman']:.4g}"
            )
        print(
            f"  {record['label']:20s}  p={record['p_value']:.4g}  "
            f"stat_true={record['stat_true']:.4g}  wall={record['wall_time_sec']:.1f}s  "
            f"updates={record['executed_gradient_steps']}{corr_suffix}"
        )


if __name__ == "__main__":
    main()
