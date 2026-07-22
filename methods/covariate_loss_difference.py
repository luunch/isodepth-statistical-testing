"""Loss-difference covariate adjustment for parallel permutation tests.

For each slot (true layout + permuted nulls), train:
  h(d(x,y), n(x,y))   — n is fixed per cell and never permuted;
                         only spatial coordinates are shuffled across slots.

The test statistic is the per-slot h(d,n) loss directly.  Under the null
(expression driven solely by n, not spatial position) the joint model cannot
exploit permuted coordinates and all slot losses are approximately equal.
Under the alternative the true slot achieves a lower loss, yielding a
significant p-value via the standard lower-tail permutation test.

No separate covariate-only h(n) model is needed: h(n) loss is the same
constant for every slot, so subtracting it never changes the slot ranking
or the resulting p-value.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import torch

from data.schemas import DatasetBundle, TestConfig
from data.transforms import zscore_covariate
from methods.trainers import BatchedTrainingOutputs, resolve_device


def dataset_uses_loss_difference_whitening(dataset: DatasetBundle) -> bool:
    cw = dataset.meta.get("covariate_whitening")
    return isinstance(cw, Mapping) and cw.get("method") == "loss-difference"


def covariate_whitening_obs_key(dataset: DatasetBundle) -> str:
    cw = dataset.meta.get("covariate_whitening")
    if not isinstance(cw, Mapping):
        raise ValueError("dataset.meta['covariate_whitening'] is missing.")
    obs_key = cw.get("obs_key")
    if not obs_key:
        raise ValueError("dataset.meta['covariate_whitening']['obs_key'] is missing.")
    return str(obs_key)


def covariate_whitening_values(dataset: DatasetBundle) -> np.ndarray:
    obs_key = covariate_whitening_obs_key(dataset)
    values = dataset.meta.get("covariate_whitening_values")
    if values is None:
        raise ValueError(
            f"data.covariate_whitening obs_key='{obs_key}' was specified but "
            "dataset.meta['covariate_whitening_values'] is missing.  "
            "Ensure load_dataset extracts the obs column during data loading."
        )
    return np.asarray(values, dtype=np.float32).reshape(-1)


def run_loss_difference_parallel_training(
    train_fn: Callable[..., tuple[object, BatchedTrainingOutputs, np.ndarray]],
    *,
    covariate_values: np.ndarray,
    model_label: str,
    train_kwargs: dict,
) -> tuple[object, BatchedTrainingOutputs, np.ndarray, dict[str, object]]:
    """Train h(d, n) for all slots and return outputs as the loss-difference statistic.

    The covariate n is fixed per cell across all slots.  Only coordinates are
    permuted, so permuted slots measure the loss when spatial structure is
    destroyed while n remains informative.  The returned outputs can be used
    directly with the standard permutation p-value.
    """
    obs_key = str(train_kwargs.get("obs_key", "covariate"))
    fn_kwargs = {k: v for k, v in train_kwargs.items() if k != "obs_key"}
    model, outputs, s_batched_np = train_fn(
        **fn_kwargs,
        fixed_covariate_values=np.asarray(covariate_values, dtype=np.float32),
        model_label=model_label,
    )
    artifacts = loss_difference_artifacts_from_outputs(
        covariate_values=covariate_values,
        obs_key=obs_key,
        joint_outputs=outputs,
    )
    return model, outputs, s_batched_np, artifacts


def loss_difference_artifacts_from_outputs(
    *,
    covariate_values: np.ndarray,
    obs_key: str,
    joint_outputs: BatchedTrainingOutputs,
) -> dict[str, object]:
    values = np.asarray(covariate_values, dtype=np.float32).reshape(-1)
    return {
        "loss_difference_obs_key": obs_key,
        "loss_difference_covariate_values": values,
        "loss_difference_covariate_latent": zscore_covariate(values),
        "loss_difference_joint_metrics": np.asarray(joint_outputs.model_metrics, dtype=np.float64),
    }


def run_plain_parallel_with_loss_difference(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    covariate_values: np.ndarray,
    obs_key: str = "covariate",
    device: torch.device | None = None,
    model_label: str = "parallel isodepth h(d, n)",
) -> tuple[object, BatchedTrainingOutputs, np.ndarray, dict[str, object]]:
    from methods.trainers import train_parallel_isodepth_model

    device = device or resolve_device(config.device)
    return run_loss_difference_parallel_training(
        train_parallel_isodepth_model,
        covariate_values=covariate_values,
        model_label=model_label,
        train_kwargs={
            "S": S,
            "A": A,
            "config": config,
            "device": device,
            "obs_key": obs_key,
        },
    )


def run_celltype_parallel_with_loss_difference(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    cell_type_labels: np.ndarray,
    n_cell_types: int,
    covariate_values: np.ndarray,
    obs_key: str = "covariate",
    device: torch.device | None = None,
    model_label: str = "cell-type parallel isodepth h(d, n)",
) -> tuple[object, BatchedTrainingOutputs, np.ndarray, dict[str, object]]:
    from methods.trainers import train_celltype_parallel_isodepth_model

    device = device or resolve_device(config.device)
    return run_loss_difference_parallel_training(
        train_celltype_parallel_isodepth_model,
        covariate_values=covariate_values,
        model_label=model_label,
        train_kwargs={
            "S": S,
            "A": A,
            "config": config,
            "cell_type_labels": cell_type_labels,
            "n_cell_types": n_cell_types,
            "device": device,
            "obs_key": obs_key,
        },
    )


def assemble_separate_loss_difference_artifacts(
    dataset: DatasetBundle,
    per_type_results: dict[str, dict],
) -> dict[str, object]:
    """Aggregate per-cell-type loss-difference diagnostics into one artifact."""
    if not dataset_uses_loss_difference_whitening(dataset):
        return {}

    values = covariate_whitening_values(dataset)
    obs_key = covariate_whitening_obs_key(dataset)
    joint_metrics: list[np.ndarray] = []
    for type_data in per_type_results.values():
        joint = type_data.get("loss_difference_joint_metrics")
        if joint is None:
            continue
        joint_metrics.append(np.asarray(joint, dtype=np.float64))

    if not joint_metrics:
        return {}

    joint_mean = np.stack(joint_metrics, axis=0).mean(axis=0)
    return {
        "loss_difference_obs_key": obs_key,
        "loss_difference_covariate_values": values,
        "loss_difference_covariate_latent": zscore_covariate(values),
        "loss_difference_joint_metrics": joint_mean,
    }
