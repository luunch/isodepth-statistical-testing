from __future__ import annotations

from experiments.core.paths import repo_root

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.plots import (
    compute_isodepth_bias_detection_similarity,
    compute_isodepth_cross_correlation_matrices,
    save_isodepth_bias_detection_figure,
    save_isodepth_cross_correlation_matrix_figure,
    save_spatial_pointcloud_kde_plot,
    save_synthetic_true_curve_plot,
)
from data import load_dataset
from data.schemas import RunConfig, run_config_from_mapping
from experiments.configuration import build_run_config
from methods.metrics import canonicalize_metric_name, compute_metric
from methods.trainers import (
    build_batched_isodepth_initial_state,
    train_celltype_parallel_isodepth_model,
    train_parallel_isodepth_model,
    get_training_metadata,
    resolve_device,
)

_ALLOWED_DEVICES = frozenset({"cuda", "cpu"})


@dataclass
class IsodepthBiasDetectionSpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    n_perms: int
    epochs: int
    devices: list[str]
    verbose: bool = False

    def validate(self) -> "IsodepthBiasDetectionSpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if int(self.n_perms) <= 0:
            raise ValueError("n_perms must be > 0")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be > 0")
        if not self.devices:
            raise ValueError("devices must be a non-empty list of 'cuda' and/or 'cpu'")
        for d in self.devices:
            if d not in _ALLOWED_DEVICES:
                raise ValueError(f"Unsupported device {d!r}; allowed: {sorted(_ALLOWED_DEVICES)}")
        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.n_perms = int(self.n_perms)
        self.epochs = int(self.epochs)
        self.verbose = bool(self.verbose)
        return self


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _parse_devices_payload(payload: Mapping[str, Any]) -> list[str]:
    if "devices" in payload and payload["devices"] is not None:
        raw = payload["devices"]
        if isinstance(raw, str):
            return [raw]
        return [str(x) for x in raw]
    if "device" in payload and payload["device"] is not None:
        raw = payload["device"]
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return [str(raw)]
    return ["cpu"]


def load_isodepth_bias_detection_spec(path: str | Path) -> IsodepthBiasDetectionSpec:
    spec_path = Path(path).resolve()
    with open(spec_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return IsodepthBiasDetectionSpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(payload["base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        n_perms=int(payload["n_perms"]),
        epochs=int(payload["epochs"]),
        devices=_parse_devices_payload(payload),
        verbose=bool(payload.get("verbose", False)),
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


def _build_effective_run_config(base_run_config: RunConfig, spec: IsodepthBiasDetectionSpec, device_str: str) -> RunConfig:
    mapping = copy.deepcopy(base_run_config.to_dict())
    mapping.setdefault("test", {})
    mapping.setdefault("output", {})
    mapping["test"]["n_perms"] = int(spec.n_perms)
    mapping["test"]["n_reruns"] = 1
    mapping["test"]["epochs"] = int(spec.epochs)
    mapping["test"]["patience"] = int(spec.epochs) + 1
    mapping["test"]["device"] = str(device_str)
    mapping["test"]["verbose"] = bool(spec.verbose)
    return run_config_from_mapping(mapping)


def _slot_titles(n_models: int) -> list[str]:
    titles = ["True data"]
    titles.extend(f"Permutation {index}" for index in range(1, n_models))
    return titles


def _run_parallel_branch(
    s_batched: np.ndarray,
    A: np.ndarray,
    run_config: RunConfig,
    *,
    metric: str,
    reference_initial_state: Mapping[str, Any] | None,
    device,
    cell_type_labels: np.ndarray | None = None,
    n_cell_types: int = 0,
) -> dict[str, Any]:
    if cell_type_labels is not None and n_cell_types > 0:
        model, training_outputs, _ = train_celltype_parallel_isodepth_model(
            s_batched[0],
            A,
            run_config.test,
            cell_type_labels=cell_type_labels,
            n_cell_types=n_cell_types,
            device=device,
            s_batched=s_batched,
            latent_dim=1,
            model_label=f"cell-type parallel permutation batch (true + {run_config.test.n_perms} permutations)",
        )
    else:
        model, training_outputs, _ = train_parallel_isodepth_model(
            s_batched[0],
            A,
            run_config.test,
            device=device,
            s_batched=s_batched,
            latent_dim=1,
            model_label=f"parallel permutation batch (true + {run_config.test.n_perms} permutations)",
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
    for model_index, metric_value in enumerate(np.asarray(training_outputs.model_metrics, dtype=np.float64)):
        per_slot.append(
            {
                "model_index": int(model_index),
                "perm_index": None if model_index == 0 else int(model_index - 1),
                "metric": float(metric_value),
                "train_loss": float(metadata["best_train_loss_per_model"][model_index]),
                "selected_rerun_index": int(metadata["best_rerun_index_per_model"][model_index]),
            }
        )
    return {
        "predictions": {
            "true": np.asarray(training_outputs.pred_true, dtype=np.float32),
            "best_null": np.asarray(training_outputs.pred_best_null, dtype=np.float32),
            "worst_null": np.asarray(training_outputs.pred_worst_null, dtype=np.float32),
        },
        "model_metrics": np.asarray(training_outputs.model_metrics, dtype=np.float64),
        "isodepths": slot_depths,
        "per_slot": per_slot,
    }


def run_isodepth_bias_detection(spec: IsodepthBiasDetectionSpec) -> dict[str, Any]:
    """Train parallel permutation slots on each requested device (same coordinate permutations) and summarize bias signal."""
    base_run_config = build_run_config(str(spec.base_config), {})

    dataset = load_dataset(base_run_config.data)
    n_models = int(spec.n_perms) + 1
    panel_titles = _slot_titles(n_models)

    s_batched, _ = _build_permuted_coordinate_batch(
        dataset.S,
        n_perms=int(spec.n_perms),
        seed=base_run_config.test.seed,
    )

    isodepths_by_device: dict[str, np.ndarray] = {}
    per_device_payload: dict[str, Any] = {}

    probe_config = _build_effective_run_config(base_run_config, spec, spec.devices[0])
    if int(probe_config.test.sgd_batch_size or 0) != 0:
        raise ValueError("isodepth bias detection requires test.sgd_batch_size == 0")

    effective_run_config_dict = probe_config.to_dict()

    cell_type_mode = base_run_config.data.cell_type_mode
    cell_type_labels: np.ndarray | None = None
    n_cell_types: int = 0
    if cell_type_mode in ("together",):
        cell_type_labels = dataset.meta.get("cell_type_labels")
        n_cell_types = int(dataset.meta.get("n_cell_types", 0))
        if cell_type_labels is None or n_cell_types == 0:
            raise ValueError(
                f"data.cell_type={base_run_config.data.cell_type!r} but dataset "
                "does not contain cell_type_labels / n_cell_types"
            )

    for device_str in spec.devices:
        run_config = _build_effective_run_config(base_run_config, spec, device_str)
        device = resolve_device(run_config.test.device)
        metric = canonicalize_metric_name(run_config.test.metric)
        effective_run_config_dict = run_config.to_dict()

        reference_initial_state: dict[str, torch.Tensor] | None = None
        if cell_type_labels is None:
            reference_initial_state = build_batched_isodepth_initial_state(
                run_config.test,
                total_models=n_models,
                n_genes=dataset.n_genes,
                latent_dim=1,
                device=device,
            )

        parallel_branch = _run_parallel_branch(
            s_batched,
            dataset.A,
            run_config,
            metric=metric,
            reference_initial_state=reference_initial_state,
            device=device,
            cell_type_labels=cell_type_labels,
            n_cell_types=n_cell_types,
        )
        isodepths_by_device[device_str] = np.asarray(parallel_branch["isodepths"], dtype=np.float32)
        per_device_payload[device_str] = {
            "device_torch": str(device),
            "per_slot": parallel_branch["per_slot"],
        }

    out_dir = spec.output_root / spec.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling_distribution_plot_path = save_spatial_pointcloud_kde_plot(
        np.asarray(dataset.S, dtype=np.float32),
        out_dir / f"{spec.experiment_name}_sampling_distribution.png",
    )

    synthetic_true_isodepth_path: Path | None = None
    synthetic_true_isodepth_plot_path: Path | None = None
    if str(base_run_config.data.source) == "synthetic":
        true_curve = dataset.meta.get("synthetic_true_curve")
        if true_curve is not None:
            true_curve_arr = np.asarray(true_curve, dtype=np.float32).reshape(-1)
            coords_arr = np.asarray(dataset.S, dtype=np.float32)
            synthetic_true_isodepth_path = (
                out_dir / f"{spec.experiment_name}_synthetic_true_isodepth.npz"
            )
            np.savez(
                synthetic_true_isodepth_path,
                S=coords_arr,
                true_isodepth=true_curve_arr,
            )
            synthetic_true_isodepth_plot_path = save_synthetic_true_curve_plot(
                dataset,
                out_dir / f"{spec.experiment_name}_synthetic_true_isodepth.png",
            )

    bias_plot_path = save_isodepth_bias_detection_figure(
        s_batched,
        isodepths_by_device,
        out_dir / f"{spec.experiment_name}_bias_detection.png",
        device_order=list(spec.devices),
        panel_titles=panel_titles,
        figure_title="Isodepth bias detection (true coordinates vs permutations)",
    )
    similarity_by_device = compute_isodepth_bias_detection_similarity(isodepths_by_device)
    for device_str, device_rows in similarity_by_device.items():
        if device_str in per_device_payload:
            per_device_payload[device_str]["per_slot_similarity"] = device_rows
            perm_pearsons = [row["pearson"] for row in device_rows if row["model_index"] != 0]
            per_device_payload[device_str]["similarity_summary"] = {
                "mean_perm_pearson": float(np.nanmean(perm_pearsons)) if perm_pearsons else float("nan"),
                "max_perm_pearson": float(np.nanmax(perm_pearsons)) if perm_pearsons else float("nan"),
            }

    cross_corr_matrices = compute_isodepth_cross_correlation_matrices(isodepths_by_device)
    cross_corr_matrix_plot_path = save_isodepth_cross_correlation_matrix_figure(
        cross_corr_matrices,
        out_dir / f"{spec.experiment_name}_cross_correlation_matrix.png",
        panel_titles=panel_titles,
        figure_title="Pairwise Pearson cross-correlation of slot isodepths",
    )
    cross_corr_matrix_data_path = (
        out_dir / f"{spec.experiment_name}_cross_correlation_matrix.npz"
    )
    np.savez(
        cross_corr_matrix_data_path,
        panel_titles=np.asarray(panel_titles, dtype=object),
        **{f"matrix__{device_str}": np.asarray(matrix, dtype=np.float64)
           for device_str, matrix in cross_corr_matrices.items()},
    )
    for device_str, matrix in cross_corr_matrices.items():
        if device_str in per_device_payload:
            per_device_payload[device_str]["cross_correlation_matrix"] = (
                np.asarray(matrix, dtype=np.float64).tolist()
            )

    cross_slot: list[dict[str, Any]] | None = None
    summary_extra: dict[str, Any] = {}
    if len(spec.devices) >= 2:
        d0, d1 = spec.devices[0], spec.devices[1]
        id0 = isodepths_by_device[d0]
        id1 = isodepths_by_device[d1]
        mses = []
        max_abs = []
        cross_slot = []
        for model_index in range(n_models):
            delta = np.asarray(id0[model_index] - id1[model_index], dtype=np.float64).reshape(-1)
            mse = float(np.mean(delta**2))
            mx = float(np.max(np.abs(delta)))
            mses.append(mse)
            max_abs.append(mx)
            cross_slot.append(
                {
                    "model_index": int(model_index),
                    "perm_index": None if model_index == 0 else int(model_index - 1),
                    "isodepth_mse_between_devices": mse,
                    "isodepth_max_abs_between_devices": mx,
                }
            )
        summary_extra = {
            "devices_compared_for_difference": [d0, d1],
            "max_isodepth_mse_between_devices": float(np.max(mses)),
            "mean_isodepth_mse_between_devices": float(np.mean(mses)),
            "max_isodepth_abs_between_devices": float(np.max(max_abs)),
        }

    payload = {
        "experiment_name": spec.experiment_name,
        "base_config_path": str(spec.base_config),
        "output_root": str(spec.output_root),
        "devices": list(spec.devices),
        "epochs_override": int(spec.epochs),
        "permutation_seed": int(base_run_config.test.seed),
        "n_perms": int(spec.n_perms),
        "n_models": int(n_models),
        "effective_run_config": effective_run_config_dict,
        "panel_titles": panel_titles,
        "bias_detection_plot": str(bias_plot_path),
        "cross_correlation_matrix_plot": str(cross_corr_matrix_plot_path),
        "cross_correlation_matrix_data": str(cross_corr_matrix_data_path),
        "sampling_distribution_plot": str(sampling_distribution_plot_path),
        "synthetic_true_isodepth_path": (
            str(synthetic_true_isodepth_path) if synthetic_true_isodepth_path is not None else None
        ),
        "synthetic_true_isodepth_plot": (
            str(synthetic_true_isodepth_plot_path)
            if synthetic_true_isodepth_plot_path is not None
            else None
        ),
        "per_device": per_device_payload,
        "cross_device_slots": cross_slot,
        "summary": summary_extra,
    }

    result_path = out_dir / f"{spec.experiment_name}_bias_detection_result.json"
    payload["bias_detection_result_path"] = str(result_path)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(_json_compatible(payload), handle, indent=2)
    return _json_compatible(payload)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isodepth bias detection: compare learned depths on true vs permuted coordinates across devices."
    )
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    spec = load_isodepth_bias_detection_spec(args.spec)
    payload = run_isodepth_bias_detection(spec)
    print(f"Saved outputs to: {Path(payload['bias_detection_result_path']).parent}")
    print(f"Result JSON: {payload['bias_detection_result_path']}")
    print(f"Figure: {payload['bias_detection_plot']}")


if __name__ == "__main__":
    main()
