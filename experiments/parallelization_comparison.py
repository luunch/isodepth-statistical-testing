from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.plots import save_parallelization_grid, save_parallelization_paired_comparison
from data import load_dataset
from data.schemas import RunConfig, run_config_from_mapping
from experiments.configuration import build_run_config
from methods.metrics import canonicalize_metric_name, compute_metric
from methods.trainers import (
    build_parallel_initial_state,
    extract_model_isodepth,
    extract_parallel_slot_initial_state,
    get_training_metadata,
    resolve_device,
    train_isodepth_model,
    train_parallel_isodepth_model,
)

@dataclass
class ParallelizationComparisonSpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    n_perms: int
    epochs: int
    device: str = "cpu"

    def validate(self) -> "ParallelizationComparisonSpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if int(self.n_perms) <= 0:
            raise ValueError("n_perms must be > 0")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be > 0")
        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.n_perms = int(self.n_perms)
        self.epochs = int(self.epochs)
        self.device = str(self.device or "cpu")
        return self


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_parallelization_comparison_spec(path: str | Path) -> ParallelizationComparisonSpec:
    spec_path = Path(path).resolve()
    with open(spec_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return ParallelizationComparisonSpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(payload["base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        n_perms=int(payload["n_perms"]),
        epochs=int(payload["epochs"]),
        device=str(payload.get("device", "cpu")),
    ).validate()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _build_permuted_coordinate_batch(
    S: np.ndarray,
    *,
    n_perms: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    spatial = np.asarray(S, dtype=np.float32)
    n_models = int(n_perms) + 1
    n_cells = int(spatial.shape[0])
    s_batched = np.zeros((n_models, n_cells, 2), dtype=np.float32)
    s_batched[0] = spatial
    rng = np.random.default_rng(int(seed))
    permutations: list[np.ndarray] = []
    for model_index in range(1, n_models):
        perm = rng.permutation(n_cells).astype(np.int64)
        permutations.append(perm)
        s_batched[model_index] = spatial[perm]
    return s_batched, permutations


def _build_effective_run_config(base_run_config: RunConfig, spec: ParallelizationComparisonSpec) -> RunConfig:
    mapping = copy.deepcopy(base_run_config.to_dict())
    mapping.setdefault("test", {})
    mapping.setdefault("output", {})
    mapping["test"]["n_perms"] = int(spec.n_perms)
    mapping["test"]["n_reruns"] = 1
    mapping["test"]["epochs"] = int(spec.epochs)
    mapping["test"]["patience"] = int(spec.epochs) + 1
    mapping["test"]["device"] = str(spec.device)
    mapping["test"]["verbose"] = False
    return run_config_from_mapping(mapping)


def _slot_titles(n_models: int) -> list[str]:
    titles = ["True data"]
    titles.extend(f"Permutation {index}" for index in range(1, n_models))
    return titles


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"Correlation inputs must match sizes, got {x.size} and {y.size}")
    if x.size == 0:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _run_sequential_branch(
    s_batched: np.ndarray,
    A: np.ndarray,
    run_config: RunConfig,
    *,
    metric: str,
    reference_initial_state: Mapping[str, Any],
    device,
) -> dict[str, Any]:
    n_models = int(s_batched.shape[0])
    decoder_type = str(run_config.test.decoder)
    latent_dim = 1

    predictions = np.zeros((n_models, A.shape[0], A.shape[1]), dtype=np.float32)
    isodepths = np.zeros((n_models, A.shape[0]), dtype=np.float32)
    per_slot: list[dict[str, Any]] = []
    for model_index in range(n_models):
        slot_initial_state = extract_parallel_slot_initial_state(
            reference_initial_state,
            slot_index=model_index,
            n_genes=A.shape[1],
            latent_dim=latent_dim,
            decoder_type=decoder_type,
            device=device,
        )
        model, prediction = train_isodepth_model(
            s_batched[model_index],
            A,
            run_config.test,
            device=device,
            seed_offset=0,
            latent_dim=latent_dim,
            model_label=f"sequential comparison model {model_index + 1}/{n_models}",
            initial_state=slot_initial_state,
            gradient_scale_divisor=float(n_models),
        )
        predictions[model_index] = np.asarray(prediction, dtype=np.float32)
        current_isodepth = extract_model_isodepth(model, s_batched[model_index], device)
        isodepths[model_index] = np.asarray(current_isodepth[:, 0], dtype=np.float32)
        metadata = get_training_metadata(model)
        per_slot.append(
            {
                "model_index": int(model_index),
                "perm_index": None if model_index == 0 else int(model_index - 1),
                "metric": float(compute_metric(metric, A, prediction)),
                "train_loss": float(metadata["best_train_loss_per_model"][0]),
                "selected_rerun_index": int(metadata["best_rerun_index_per_model"][0]),
            }
        )
    return {
        "predictions": predictions,
        "isodepths": isodepths,
        "per_slot": per_slot,
    }


def _run_parallel_branch(
    s_batched: np.ndarray,
    A: np.ndarray,
    run_config: RunConfig,
    *,
    metric: str,
    reference_initial_state: Mapping[str, Any],
    device,
) -> dict[str, Any]:
    model, predictions = train_parallel_isodepth_model(
        s_batched[0],
        A,
        run_config.test,
        device=device,
        s_batched=s_batched,
        latent_dim=1,
        model_label=f"parallel comparison batch (true + {run_config.test.n_perms} permutations)",
        initial_state=reference_initial_state,
        gradient_scale_divisor=float(s_batched.shape[0]),
    )
    metadata = get_training_metadata(model)
    with torch.no_grad():
        slot_depths = np.asarray(
            model.encoder(torch.tensor(s_batched, dtype=torch.float32, device=device)).detach().cpu().numpy()[:, :, 0],
            dtype=np.float32,
        )
    per_slot: list[dict[str, Any]] = []
    for model_index, prediction in enumerate(np.asarray(predictions, dtype=np.float32)):
        per_slot.append(
            {
                "model_index": int(model_index),
                "perm_index": None if model_index == 0 else int(model_index - 1),
                "metric": float(compute_metric(metric, A, prediction)),
                "train_loss": float(metadata["best_train_loss_per_model"][model_index]),
                "selected_rerun_index": int(metadata["best_rerun_index_per_model"][model_index]),
            }
        )
    return {
        "predictions": np.asarray(predictions, dtype=np.float32),
        "isodepths": slot_depths,
        "per_slot": per_slot,
    }


def run_parallelization_comparison(
    spec: ParallelizationComparisonSpec,
) -> dict[str, Any]:
    base_run_config = build_run_config(str(spec.base_config), {})
    run_config = _build_effective_run_config(base_run_config, spec)
    if int(run_config.test.sgd_batch_size or 0) != 0:
        raise ValueError("parallelization comparison requires test.sgd_batch_size == 0")

    dataset = load_dataset(run_config.data)
    device = resolve_device(run_config.test.device)
    n_models = int(run_config.test.n_perms) + 1
    metric = canonicalize_metric_name(run_config.test.metric)
    s_batched, _ = _build_permuted_coordinate_batch(
        dataset.S,
        n_perms=run_config.test.n_perms,
        seed=run_config.test.seed,
    )
    reference_initial_state = build_parallel_initial_state(
        n_models,
        dataset.n_genes,
        latent_dim=1,
        decoder_type=str(run_config.test.decoder),
        seed=run_config.test.seed,
        device=device,
    )

    parallel_branch = _run_parallel_branch(
        s_batched,
        dataset.A,
        run_config,
        metric=metric,
        reference_initial_state=reference_initial_state,
        device=device,
    )
    sequential_branch = _run_sequential_branch(
        s_batched,
        dataset.A,
        run_config,
        metric=metric,
        reference_initial_state=reference_initial_state,
        device=device,
    )

    panel_titles = _slot_titles(n_models)
    out_dir = spec.output_root / spec.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    parallel_grid_path = save_parallelization_grid(
        s_batched,
        parallel_branch["isodepths"],
        out_dir / f"{spec.experiment_name}_parallel_grid.png",
        panel_titles=panel_titles,
        figure_title="Parallel permutation comparison",
    )
    sequential_grid_path = save_parallelization_grid(
        s_batched,
        sequential_branch["isodepths"],
        out_dir / f"{spec.experiment_name}_sequential_grid.png",
        panel_titles=panel_titles,
        figure_title="Sequential permutation comparison",
    )
    paired_grid_path = save_parallelization_paired_comparison(
        s_batched,
        parallel_branch["isodepths"],
        sequential_branch["isodepths"],
        out_dir / f"{spec.experiment_name}_paired_comparison.png",
        row_titles=panel_titles,
    )

    per_slot: list[dict[str, Any]] = []
    for model_index in range(n_models):
        parallel_prediction = np.asarray(parallel_branch["predictions"][model_index], dtype=np.float32)
        sequential_prediction = np.asarray(sequential_branch["predictions"][model_index], dtype=np.float32)
        parallel_isodepth = np.asarray(parallel_branch["isodepths"][model_index], dtype=np.float32)
        sequential_isodepth = np.asarray(sequential_branch["isodepths"][model_index], dtype=np.float32)
        per_slot.append(
            {
                "model_index": int(model_index),
                "perm_index": None if model_index == 0 else int(model_index - 1),
                "parallel_metric": float(parallel_branch["per_slot"][model_index]["metric"]),
                "sequential_metric": float(sequential_branch["per_slot"][model_index]["metric"]),
                "metric_abs_diff": float(
                    abs(parallel_branch["per_slot"][model_index]["metric"] - sequential_branch["per_slot"][model_index]["metric"])
                ),
                "prediction_mse": float(np.mean((parallel_prediction - sequential_prediction) ** 2)),
                "isodepth_mse": float(np.mean((parallel_isodepth - sequential_isodepth) ** 2)),
                "isodepth_corr": float(_safe_corr(parallel_isodepth, sequential_isodepth)),
                "parallel_train_loss": float(parallel_branch["per_slot"][model_index]["train_loss"]),
                "sequential_train_loss": float(sequential_branch["per_slot"][model_index]["train_loss"]),
                "parallel_selected_rerun_index": int(parallel_branch["per_slot"][model_index]["selected_rerun_index"]),
                "sequential_selected_rerun_index": int(sequential_branch["per_slot"][model_index]["selected_rerun_index"]),
            }
        )

    metric_abs_diffs = np.asarray([row["metric_abs_diff"] for row in per_slot], dtype=np.float64)
    prediction_mse = np.asarray([row["prediction_mse"] for row in per_slot], dtype=np.float64)
    isodepth_mse = np.asarray([row["isodepth_mse"] for row in per_slot], dtype=np.float64)
    isodepth_corr = np.asarray([row["isodepth_corr"] for row in per_slot], dtype=np.float64)
    payload = {
        "experiment_name": spec.experiment_name,
        "base_config_path": str(spec.base_config),
        "output_root": str(spec.output_root),
        "device": str(device),
        "epochs_override": int(spec.epochs),
        "permutation_seed": int(run_config.test.seed),
        "n_perms": int(run_config.test.n_perms),
        "n_models": int(n_models),
        "effective_run_config": run_config.to_dict(),
        "panel_titles": panel_titles,
        "parallel_grid_plot": str(parallel_grid_path),
        "sequential_grid_plot": str(sequential_grid_path),
        "paired_comparison_plot": str(paired_grid_path),
        "per_slot": per_slot,
        "summary": {
            "max_metric_abs_diff": float(metric_abs_diffs.max()),
            "mean_metric_abs_diff": float(metric_abs_diffs.mean()),
            "max_prediction_mse": float(prediction_mse.max()),
            "mean_prediction_mse": float(prediction_mse.mean()),
            "max_isodepth_mse": float(isodepth_mse.max()),
            "mean_isodepth_mse": float(isodepth_mse.mean()),
            "min_isodepth_corr": float(isodepth_corr.min()),
            "mean_isodepth_corr": float(isodepth_corr.mean()),
        },
    }

    result_path = out_dir / f"{spec.experiment_name}_comparison_result.json"
    payload["comparison_result_path"] = str(result_path)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(_json_compatible(payload), handle, indent=2)
    return _json_compatible(payload)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare parallel and sequential permutation training.")
    parser.add_argument("--spec", required=True, help="Path to the comparison spec JSON")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    spec = load_parallelization_comparison_spec(args.spec)
    payload = run_parallelization_comparison(spec)
    print(f"Saved comparison outputs to: {Path(payload['comparison_result_path']).parent}")
    print(f"Result JSON: {payload['comparison_result_path']}")
    print(f"Max metric abs diff: {float(payload['summary']['max_metric_abs_diff']):.6g}")
    print(f"Mean isodepth corr: {float(payload['summary']['mean_isodepth_corr']):.6g}")


if __name__ == "__main__":
    main()
