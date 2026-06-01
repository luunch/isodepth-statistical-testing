from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from data.schemas import TestConfig
from data.transforms import zscore_covariate
from methods.architectures import (
    CellTypeIsoDepthNet,
    DecoderOnlyNet,
    DecoderOnlyNetFixed,
    HybridMidlineLatent,
    HybridMidlineParallelNet,
    IsoDepthNet,
    ParallelCellTypeIsoDepthNet,
    ParallelDecoderOnlyNetFixed,
    ParallelIsoDepthNet,
    ParallelLinear,
    ParallelQuadraticDecoder,
)
from methods.metrics import canonicalize_metric_name, compute_metric, metric_prefers_lower
from methods.trainers.gpu_selection import offload_module_to_cpu, resolve_device

_DEFAULT_FINALIZE_CHUNK_SIZE = 128


@dataclass(frozen=True)
class BatchedTrainingOutputs:
    """Minimal post-training artifacts: metrics for all models and three prediction matrices."""

    model_metrics: np.ndarray
    pred_true: np.ndarray
    pred_best_null: np.ndarray
    pred_worst_null: np.ndarray
    best_null_index: int
    worst_null_index: int

    @property
    def stat_true(self) -> float:
        return float(self.model_metrics[0])

    @property
    def stat_perm(self) -> np.ndarray:
        return np.asarray(self.model_metrics[1:], dtype=np.float64)


def _is_midline_covariate(config: TestConfig) -> bool:
    cov = getattr(config, "covariate", None)
    return cov is not None and getattr(cov, "type", None) == "midline"


def _is_obs_key_covariate(config: TestConfig) -> bool:
    """True when the covariate is a labeled obs-column (not midline, not None)."""
    cov = getattr(config, "covariate", None)
    return cov is not None and getattr(cov, "is_obs_key", False)


def _parallel_slot_count(model: nn.Module) -> int:
    for _name, tensor in model.named_parameters():
        if tensor.ndim > 0:
            return int(tensor.shape[0])
    raise ValueError("Cannot infer parallel slot count: model has no parameters with ndim >= 1.")


def _resolve_decoder_type(config: TestConfig) -> str:
    return str(getattr(config, "decoder", "nn"))


def _clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _load_initial_state(
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor] | None,
    *,
    device: torch.device,
) -> None:
    if initial_state is None:
        return
    current_state = model.state_dict()
    loaded_state: dict[str, torch.Tensor] = {}
    for name, tensor in current_state.items():
        if name not in initial_state:
            raise ValueError(f"initial_state is missing parameter '{name}'")
        source_tensor = initial_state[name].detach()
        if source_tensor.shape == tensor.shape:
            loaded_state[name] = source_tensor.to(device=device)
            continue
        if tensor.ndim == source_tensor.ndim + 1 and tensor.shape[0] == 1 and tensor.shape[1:] == source_tensor.shape:
            loaded_state[name] = source_tensor.unsqueeze(0).to(device=device)
            continue
        raise ValueError(
            f"initial_state parameter '{name}' has shape {tuple(source_tensor.shape)} but expected {tuple(tensor.shape)}"
        )
    model.load_state_dict(loaded_state)


def build_batched_isodepth_initial_state(
    config: TestConfig,
    *,
    total_models: int,
    n_genes: int,
    latent_dim: int = 1,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """State dict matching ``train_batched_isodepth_model`` (ParallelIsoDepthNet or HybridMidlineParallelNet)."""
    resolved_device = device or torch.device("cpu")
    decoder_type = _resolve_decoder_type(config)
    n_reruns = int(config.n_reruns)
    _set_torch_seed(int(config.seed))
    if _is_midline_covariate(config):
        if latent_dim != 1:
            raise ValueError("covariate type midline requires latent_dim == 1")
        model = HybridMidlineParallelNet(
            total_models,
            n_genes,
            slot_split=n_reruns,
            latent_dim=1,
            decoder_type=decoder_type,
        ).to(resolved_device)
    else:
        model = ParallelIsoDepthNet(
            total_models,
            n_genes,
            latent_dim=latent_dim,
            decoder_type=decoder_type,
        ).to(resolved_device)
    return _clone_state_dict(model.state_dict())


def build_parallel_initial_state(
    n_models: int,
    n_genes: int,
    *,
    latent_dim: int = 1,
    decoder_type: str = "nn",
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """RNG-aligned weights for plain ``ParallelIsoDepthNet`` (no midline covariate)."""
    resolved_device = device or torch.device("cpu")
    m = int(n_models)
    if m <= 1:
        _set_torch_seed(int(seed))
        model = ParallelIsoDepthNet(
            max(m, 1),
            n_genes,
            latent_dim=latent_dim,
            decoder_type=decoder_type,
        ).to(resolved_device)
        return _clone_state_dict(model.state_dict())
    cfg = TestConfig(
        method="parallel_permutation",
        metric="mse",
        n_perms=m - 1,
        n_reruns=1,
        epochs=1,
        patience=2,
        lr=1e-3,
        seed=int(seed),
        device="cpu",
        decoder=decoder_type,
        sgd_batch_size=0,
        verbose=False,
        covariate=None,
    )
    cfg.validate()
    return build_batched_isodepth_initial_state(
        cfg,
        total_models=m,
        n_genes=int(n_genes),
        latent_dim=latent_dim,
        device=device,
    )


def extract_parallel_slot_initial_state(
    parallel_state: Mapping[str, torch.Tensor],
    *,
    slot_index: int,
    n_genes: int,
    latent_dim: int = 1,
    decoder_type: str = "nn",
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    resolved_device = device or torch.device("cpu")
    model = IsoDepthNet(
        n_genes,
        latent_dim=latent_dim,
        decoder_type=decoder_type,
    ).to(resolved_device)
    single_state = model.state_dict()
    extracted_state: dict[str, torch.Tensor] = {}
    for name, tensor in single_state.items():
        source = parallel_state.get(name)
        if source is None:
            raise ValueError(f"parallel_state is missing parameter '{name}'")
        detached_source = source.detach().cpu()
        if detached_source.shape == tensor.shape:
            extracted_state[name] = detached_source.clone()
        elif detached_source.ndim > 0 and detached_source.shape[0] > int(slot_index):
            extracted_state[name] = detached_source[int(slot_index)].clone()
        else:
            raise ValueError(
                f"Cannot extract slot {int(slot_index)} for parameter '{name}' with shape {tuple(detached_source.shape)}"
            )
    return extracted_state


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_predictions(model: nn.Module, S: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        s_t = torch.tensor(S, dtype=torch.float32, device=device)
        preds = model(s_t).detach().cpu().numpy()
    return np.asarray(preds, dtype=np.float32)


def _parallel_model_slot_count(model: nn.Module) -> int:
    parallel_m = getattr(model, "M", None)
    if parallel_m is not None:
        return int(parallel_m)
    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, nn.Sequential) and len(encoder) > 0:
        first_layer = encoder[0]
        layer_m = getattr(first_layer, "M", None)
        if layer_m is not None:
            return int(layer_m)
    return _parallel_slot_count(model)


def _encoder_expects_batched_spatial_input(model: nn.Module) -> bool:
    if isinstance(
        model,
        (ParallelIsoDepthNet, ParallelCellTypeIsoDepthNet, HybridMidlineParallelNet),
    ):
        return True
    return getattr(model, "M", None) is not None


def extract_model_isodepth(
    model: nn.Module,
    S: np.ndarray,
    device: torch.device,
    *,
    slot_index: int = 0,
) -> np.ndarray:
    latent_dim = int(getattr(model, "latent_dim", 0))
    if latent_dim <= 0 or not hasattr(model, "encoder"):
        return np.zeros((S.shape[0], 0), dtype=np.float32)
    if hasattr(model, "parameters"):
        try:
            inference_device = next(model.parameters()).device
        except StopIteration:
            inference_device = device
    else:
        inference_device = device
    with torch.no_grad():
        s_t = torch.tensor(S, dtype=torch.float32, device=inference_device)
        if _encoder_expects_batched_spatial_input(model):
            n_slots = _parallel_model_slot_count(model)
            n_cells = int(S.shape[0])
            s_batched = s_t.unsqueeze(0).expand(n_slots, n_cells, 2).contiguous()
            d = model.encoder(s_batched).detach().cpu().numpy()
            return np.asarray(d[slot_index], dtype=np.float32).reshape(n_cells, latent_dim)
        d = model.encoder(s_t).detach().cpu().numpy()
    return np.asarray(d, dtype=np.float32).reshape(S.shape[0], latent_dim)


def _prepare_loss_mask(
    loss_mask_batched: np.ndarray | None,
    *,
    n_models: int,
    n_cells: int,
    n_genes: int,
    device: torch.device,
) -> torch.Tensor | None:
    if loss_mask_batched is None:
        return None

    loss_mask_np = np.asarray(loss_mask_batched, dtype=np.float32)
    valid_shapes = {
        (n_models, n_cells, 1),
        (n_models, n_cells, n_genes),
    }
    if loss_mask_np.shape not in valid_shapes:
        raise ValueError(
            "loss_mask_batched must have shape (M, N, 1) or (M, N, G), "
            f"got {loss_mask_np.shape}"
        )

    if np.any(loss_mask_np < 0):
        raise ValueError("loss_mask_batched must be non-negative")

    if loss_mask_np.shape[-1] == 1:
        loss_mask_np = np.repeat(loss_mask_np, n_genes, axis=2)

    active_counts = loss_mask_np.sum(axis=(1, 2))
    if np.any(active_counts <= 0):
        raise ValueError("Each model must have at least one active entry in loss_mask_batched")

    return torch.tensor(loss_mask_np, dtype=torch.float32, device=device)


def _snapshot_parallel_model_state(model: nn.Module, n_models: int) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        detached = tensor.detach().cpu().clone()
        if detached.ndim > 0 and detached.shape[0] == n_models:
            state[name] = detached
    return state


def _update_parallel_model_snapshot(
    snapshot: dict[str, torch.Tensor],
    model: nn.Module,
    improved_mask: np.ndarray,
    n_models: int,
) -> None:
    if not np.any(improved_mask):
        return

    improved_indices = np.flatnonzero(improved_mask)
    for name, tensor in model.state_dict().items():
        detached = tensor.detach().cpu()
        if detached.ndim == 0 or detached.shape[0] != n_models:
            continue
        snapshot[name][improved_indices] = detached[improved_indices]


def _restore_parallel_model_snapshot(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    current_state = model.state_dict()
    restored_state = {}
    for name, tensor in current_state.items():
        if name in snapshot:
            restored_state[name] = snapshot[name].to(device=device)
        else:
            restored_state[name] = tensor
    model.load_state_dict(restored_state)


def _snapshot_parallel_model_state_on_device(
    model: nn.Module, n_models: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """GPU-resident snapshot: clones only the parallel-batch tensors, staying on device."""
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        detached = tensor.detach()
        if detached.ndim > 0 and detached.shape[0] == n_models:
            state[name] = detached.to(device=device).clone()
    return state


def _update_parallel_model_snapshot_on_device(
    snapshot: dict[str, torch.Tensor],
    model: nn.Module,
    improved_indices: np.ndarray,
    n_models: int,
) -> None:
    """In-place update of a GPU-resident snapshot: copies only improved rows on device."""
    if len(improved_indices) == 0:
        return
    for name, tensor in model.state_dict().items():
        if tensor.ndim == 0 or tensor.shape[0] != n_models or name not in snapshot:
            continue
        snapshot[name][improved_indices] = tensor.detach()[improved_indices]


def _restore_parallel_model_snapshot_on_device(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    """Restore model weights from a GPU-resident snapshot (no device transfer needed)."""
    current_state = model.state_dict()
    restored_state = {
        name: snapshot[name] if name in snapshot else tensor
        for name, tensor in current_state.items()
    }
    model.load_state_dict(restored_state)


def _snapshot_hybrid_midline_state(
    model: HybridMidlineParallelNet,
    total_models: int,
    slot_split: int,
) -> dict[str, torch.Tensor]:
    """Snapshot decoder rows (size ``total_models``) and permutation-encoder rows (size ``total_models - slot_split``)."""
    state: dict[str, torch.Tensor] = {}
    p_enc = total_models - slot_split
    for name, tensor in model.state_dict().items():
        detached = tensor.detach().cpu().clone()
        if detached.ndim == 0:
            continue
        if name.startswith("decoder") and detached.shape[0] == total_models:
            state[name] = detached
        elif "encoder_perm" in name and p_enc > 0 and detached.shape[0] == p_enc:
            state[name] = detached
    return state


def _update_hybrid_midline_snapshot(
    snapshot: dict[str, torch.Tensor],
    model: HybridMidlineParallelNet,
    improved_mask: np.ndarray,
    total_models: int,
    slot_split: int,
) -> None:
    if not np.any(improved_mask):
        return
    improved_indices = np.flatnonzero(improved_mask)
    p_enc = total_models - slot_split
    for name, tensor in model.state_dict().items():
        detached = tensor.detach().cpu()
        if detached.ndim == 0 or name not in snapshot:
            continue
        if name.startswith("decoder") and detached.shape[0] == total_models:
            snapshot[name][improved_indices] = detached[improved_indices]
        elif "encoder_perm" in name and p_enc > 0 and detached.shape[0] == p_enc:
            enc_imp = improved_indices[improved_indices >= slot_split]
            if enc_imp.size:
                enc_rows = enc_imp - slot_split
                snapshot[name][enc_rows] = detached[enc_rows]


def _restore_hybrid_midline_snapshot(
    model: HybridMidlineParallelNet,
    snapshot: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    _restore_parallel_model_snapshot(model, snapshot, device)


def _compact_hybrid_midline_parallel_model(
    expanded_model: HybridMidlineParallelNet,
    *,
    selected_indices: np.ndarray,
    n_models: int,
    n_genes: int,
    latent_dim: int,
    decoder_type: str,
    device: torch.device,
) -> HybridMidlineParallelNet:
    slot_exp = int(expanded_model.slot_split)
    compact = HybridMidlineParallelNet(
        n_models,
        n_genes,
        slot_split=1,
        latent_dim=latent_dim,
        decoder_type=decoder_type,
    ).to(device)
    sel = np.asarray(selected_indices, dtype=np.int64)
    exp_sd = expanded_model.state_dict()
    comp_sd = compact.state_dict()

    for name in comp_sd:
        if name.startswith("decoder"):
            src = exp_sd[name]
            if src.ndim == 0:
                continue
            gathered = torch.stack([src[int(sel[i])].clone() for i in range(n_models)], dim=0)
            comp_sd[name].copy_(gathered.to(device=device))
        elif "encoder_perm" in name and n_models > 1:
            src = exp_sd[name]
            if src.ndim == 0:
                continue
            rows: list[torch.Tensor] = []
            for j in range(n_models - 1):
                exp_slot = int(sel[j + 1])
                enc_row = exp_slot - slot_exp
                rows.append(src[int(enc_row)].clone())
            stacked = torch.stack(rows, dim=0)
            comp_sd[name].copy_(stacked.to(device=device))

    compact.load_state_dict(comp_sd)
    return compact


def _compute_reconstruction_loss_per_model(
    output: torch.Tensor,
    targets: torch.Tensor,
    loss_mask_t: torch.Tensor | None,
) -> torch.Tensor:
    # targets may be (M, N, G) or (N, G); broadcasting handles both.
    squared_error = (output - targets) ** 2
    if loss_mask_t is not None:
        squared_error = squared_error * loss_mask_t
        active_counts = loss_mask_t.sum(dim=(1, 2)).clamp_min(1.0)
        return squared_error.sum(dim=(1, 2)) / active_counts
    return squared_error.mean(dim=(1, 2))


def _broadcast_targets(targets: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 2:
        return targets
    if targets.shape[0] == output.shape[0]:
        return targets
    raise ValueError(
        f"targets must have shape (N, G) or (M, N, G) matching output, "
        f"got {tuple(targets.shape)} vs output {tuple(output.shape)}"
    )


def _masked_metric_from_mse(
    mse: torch.Tensor,
    active_counts: torch.Tensor,
    *,
    metric: str,
) -> torch.Tensor:
    metric = canonicalize_metric_name(metric)
    if metric == "mse":
        return mse
    if metric == "nll_gaussian_mse":
        return (active_counts / 2.0) * torch.log(2.0 * torch.pi * mse + 1e-12) + (active_counts / 2.0)
    raise ValueError(f"Unsupported masked-loss metric '{metric}'")


def _compute_masked_metric_per_model(
    output: torch.Tensor,
    targets: torch.Tensor,
    loss_mask_t: torch.Tensor,
    *,
    metric: str,
) -> torch.Tensor:
    targets_b = _broadcast_targets(targets, output)
    squared_error = (output - targets_b) ** 2
    loss_mask = loss_mask_t
    if loss_mask.shape[-1] == 1:
        loss_mask = loss_mask.expand(-1, -1, output.shape[-1])
    active_counts = loss_mask.sum(dim=(1, 2)).clamp_min(1.0)
    mse = squared_error.mul(loss_mask).sum(dim=(1, 2)) / active_counts
    return _masked_metric_from_mse(mse, active_counts, metric=metric)


def _null_extreme_indices(model_metrics: np.ndarray, metric: str) -> tuple[int, int]:
    stat_perm = np.asarray(model_metrics[1:], dtype=np.float64)
    if stat_perm.size == 0:
        return 0, 0
    if metric_prefers_lower(metric):
        return int(np.argmin(stat_perm)), int(np.argmax(stat_perm))
    return int(np.argmax(stat_perm)), int(np.argmin(stat_perm))


def _parallel_linear_slice_forward(
    layer: ParallelLinear,
    x: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    weight = layer.weight[start:stop]
    bias = layer.bias[start:stop]
    return torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)


def _parallel_module_stack_slice_forward(
    stack: nn.Module,
    x: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    if isinstance(stack, ParallelLinear):
        return _parallel_linear_slice_forward(stack, x, start, stop)
    if isinstance(stack, ParallelQuadraticDecoder):
        return _parallel_linear_slice_forward(
            stack.linear, torch.cat([x, x ** 2], dim=-1), start, stop
        )
    if not isinstance(stack, nn.Sequential):
        raise TypeError(f"Unsupported parallel module stack: {type(stack)!r}")
    out = x
    for module in stack:
        if isinstance(module, nn.ReLU):
            out = module(out)
        elif isinstance(module, ParallelLinear):
            out = _parallel_linear_slice_forward(module, out, start, stop)
        else:
            raise TypeError(f"Unsupported module in parallel sequential: {type(module)!r}")
    return out


def _hybrid_midline_latent_slice_forward(
    encoder: HybridMidlineLatent,
    x: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    m = stop - start
    n = x.shape[1]
    out = torch.zeros(m, n, encoder.latent_dim, device=x.device, dtype=x.dtype)
    slot_split = int(encoder.slot_split)
    midline_start = max(start, 0)
    midline_stop = min(stop, slot_split)
    if midline_start < midline_stop:
        local_start = midline_start - start
        local_stop = midline_stop - start
        out[local_start:local_stop] = encoder.midline(x[local_start:local_stop])
    if encoder.encoder_perm is not None:
        perm_global_start = max(start, slot_split)
        if perm_global_start < stop:
            local_start = perm_global_start - start
            enc_start = perm_global_start - slot_split
            enc_stop = stop - slot_split
            out[local_start:] = _parallel_module_stack_slice_forward(
                encoder.encoder_perm,
                x[local_start:],
                enc_start,
                enc_stop,
            )
    return out


def _encode_parallel_model_slice(
    model: ParallelIsoDepthNet | HybridMidlineParallelNet,
    x: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    if isinstance(model, HybridMidlineParallelNet):
        return _hybrid_midline_latent_slice_forward(model.encoder, x, start, stop)
    return _parallel_module_stack_slice_forward(model.encoder, x, start, stop)


def _forward_parallel_model_slice(
    model: ParallelIsoDepthNet | HybridMidlineParallelNet,
    x: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    latent = _encode_parallel_model_slice(model, x, start, stop)
    return _parallel_module_stack_slice_forward(model.decoder, latent, start, stop)


def _forward_parallel_celltype_slice(
    model: ParallelCellTypeIsoDepthNet,
    x: torch.Tensor,
    cell_type_indices: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    if model._sort_idx is None or model._sort_idx.device != x.device:
        model._build_routing_cache(cell_type_indices)

    latent = _parallel_module_stack_slice_forward(model.encoder, x, start, stop)
    m, n, _ = x.shape
    sorted_latent = latent[:, model._sort_idx, :]
    sorted_output = torch.empty(m, n, model.G, device=x.device, dtype=x.dtype)
    for cell_type_index, (type_start, type_stop) in enumerate(model._type_offsets):
        if type_start == type_stop:
            continue
        sorted_output[:, type_start:type_stop, :] = _parallel_module_stack_slice_forward(
            model.decoders[cell_type_index],
            sorted_latent[:, type_start:type_stop, :],
            start,
            stop,
        )
    return sorted_output[:, model._unsort_idx, :]


def _finalize_batched_parallel_training(
    model: nn.Module,
    s_batched_t: torch.Tensor,
    a_t: torch.Tensor,
    loss_mask_t: torch.Tensor | None,
    A: np.ndarray,
    config: TestConfig,
    *,
    n_models: int,
    n_reruns: int,
    n_genes: int,
    latent_dim: int,
    device: torch.device,
    metric_loss_mask_t: torch.Tensor | None = None,
    chunk_size: int = _DEFAULT_FINALIZE_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, BatchedTrainingOutputs, np.ndarray]:
    """Chunked post-training: rerun selection, per-model metrics, three prediction matrices."""
    total_models = int(s_batched_t.shape[0])
    metric_name = canonicalize_metric_name(config.metric)
    chunk_size = max(1, min(int(chunk_size), total_models))
    a_np = np.asarray(A, dtype=np.float32)

    slot_train_losses = np.empty(total_models, dtype=np.float64)
    for start in range(0, total_models, chunk_size):
        stop = min(start + chunk_size, total_models)
        mask_chunk = None if loss_mask_t is None else loss_mask_t[start:stop]
        with torch.no_grad():
            output = _forward_parallel_model_slice(model, s_batched_t[start:stop], start, stop)
            losses = _compute_reconstruction_loss_per_model(output, a_t, mask_chunk)
        slot_train_losses[start:stop] = losses.detach().cpu().numpy().astype(np.float64)

    train_loss_per_rerun = slot_train_losses.reshape(n_models, n_reruns)
    best_rerun_index_per_model = np.argmin(train_loss_per_rerun, axis=1).astype(np.int64)
    selected_slot_indices = (np.arange(n_models, dtype=np.int64) * n_reruns) + best_rerun_index_per_model

    eval_mask_t = metric_loss_mask_t if metric_loss_mask_t is not None else loss_mask_t
    model_metrics = np.empty(n_models, dtype=np.float64)
    if eval_mask_t is None:
        for model_index in range(n_models):
            slot = int(selected_slot_indices[model_index])
            with torch.no_grad():
                pred = _forward_parallel_model_slice(
                    model,
                    s_batched_t[slot : slot + 1],
                    slot,
                    slot + 1,
                )[0].detach().cpu().numpy().astype(np.float32)
            model_metrics[model_index] = compute_metric(metric_name, a_np, pred)
    else:
        for model_index in range(n_models):
            slot = int(selected_slot_indices[model_index])
            with torch.no_grad():
                output = _forward_parallel_model_slice(
                    model,
                    s_batched_t[slot : slot + 1],
                    slot,
                    slot + 1,
                )
                if a_t.ndim == 2:
                    targets = a_t
                else:
                    targets = a_t[slot : slot + 1]
                metric_value = _compute_masked_metric_per_model(
                    output,
                    targets,
                    eval_mask_t[slot : slot + 1],
                    metric=metric_name,
                )
            model_metrics[model_index] = float(metric_value[0].detach().cpu().numpy())

    best_null_index, worst_null_index = _null_extreme_indices(model_metrics, metric_name)
    slots_for_predictions = {0}
    if n_models > 1:
        slots_for_predictions.add(int(best_null_index + 1))
        slots_for_predictions.add(int(worst_null_index + 1))
    stored_predictions: dict[int, np.ndarray] = {}
    for model_index in slots_for_predictions:
        slot = int(selected_slot_indices[model_index])
        with torch.no_grad():
            pred = _forward_parallel_model_slice(
                model,
                s_batched_t[slot : slot + 1],
                slot,
                slot + 1,
            )[0].detach().cpu().numpy().astype(np.float32)
        stored_predictions[model_index] = pred

    pred_true = stored_predictions[0]
    pred_best_null = stored_predictions.get(int(best_null_index + 1), pred_true)
    pred_worst_null = stored_predictions.get(int(worst_null_index + 1), pred_true)

    outputs = BatchedTrainingOutputs(
        model_metrics=model_metrics,
        pred_true=pred_true,
        pred_best_null=pred_best_null,
        pred_worst_null=pred_worst_null,
        best_null_index=best_null_index,
        worst_null_index=worst_null_index,
    )

    with torch.no_grad():
        true_rerun_isodepths = (
            _encode_parallel_model_slice(model, s_batched_t[:n_reruns], 0, n_reruns)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return (
        selected_slot_indices,
        train_loss_per_rerun,
        outputs,
        true_rerun_isodepths,
    )



def _celltype_expression_predictions(
    model: ParallelCellTypeIsoDepthNet,
    s_batched_t: torch.Tensor,
    cell_type_t: torch.Tensor,
    type_means: np.ndarray,
    cell_type_labels_np: np.ndarray,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> torch.Tensor:
    if start is None or stop is None:
        residual = model(s_batched_t, cell_type_t)
    else:
        residual = _forward_parallel_celltype_slice(model, s_batched_t, cell_type_t, start, stop)
    return residual + torch.tensor(
        type_means[cell_type_labels_np],
        dtype=residual.dtype,
        device=residual.device,
    ).unsqueeze(0)


def _finalize_celltype_parallel_training(
    model: ParallelCellTypeIsoDepthNet,
    s_batched_t: torch.Tensor,
    a_batched_t: torch.Tensor,
    cell_type_t: torch.Tensor,
    A: np.ndarray,
    type_means: np.ndarray,
    cell_type_labels_np: np.ndarray,
    config: TestConfig,
    *,
    n_models: int,
    n_reruns: int,
    latent_dim: int,
    chunk_size: int = _DEFAULT_FINALIZE_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, BatchedTrainingOutputs, np.ndarray]:
    total_models = int(s_batched_t.shape[0])
    metric_name = canonicalize_metric_name(config.metric)
    chunk_size = max(1, min(int(chunk_size), total_models))
    a_np = np.asarray(A, dtype=np.float32)

    slot_train_losses = np.empty(total_models, dtype=np.float64)
    for start in range(0, total_models, chunk_size):
        stop = min(start + chunk_size, total_models)
        with torch.no_grad():
            output = _forward_parallel_celltype_slice(model, s_batched_t[start:stop], cell_type_t, start, stop)
            losses = (output - a_batched_t[start:stop]).pow(2).mean(dim=(1, 2))
        slot_train_losses[start:stop] = losses.detach().cpu().numpy().astype(np.float64)

    train_loss_per_rerun = slot_train_losses.reshape(n_models, n_reruns)
    selected_slot_indices = (
        np.arange(n_models, dtype=np.int64) * n_reruns
        + np.argmin(train_loss_per_rerun, axis=1).astype(np.int64)
    )

    model_metrics = np.empty(n_models, dtype=np.float64)
    for model_index in range(n_models):
        slot = int(selected_slot_indices[model_index])
        with torch.no_grad():
            pred = _celltype_expression_predictions(
                model,
                s_batched_t[slot : slot + 1],
                cell_type_t,
                type_means,
                cell_type_labels_np,
                start=slot,
                stop=slot + 1,
            )[0].detach().cpu().numpy().astype(np.float32)
        model_metrics[model_index] = compute_metric(metric_name, a_np, pred)

    best_null_index, worst_null_index = _null_extreme_indices(model_metrics, metric_name)
    slots_for_predictions = {0, int(best_null_index + 1), int(worst_null_index + 1)}
    stored_predictions: dict[int, np.ndarray] = {}
    for model_index in slots_for_predictions:
        slot = int(selected_slot_indices[model_index])
        with torch.no_grad():
            pred = _celltype_expression_predictions(
                model,
                s_batched_t[slot : slot + 1],
                cell_type_t,
                type_means,
                cell_type_labels_np,
                start=slot,
                stop=slot + 1,
            )[0].detach().cpu().numpy().astype(np.float32)
        stored_predictions[model_index] = pred

    outputs = BatchedTrainingOutputs(
        model_metrics=model_metrics,
        pred_true=stored_predictions[0],
        pred_best_null=stored_predictions[int(best_null_index + 1)],
        pred_worst_null=stored_predictions[int(worst_null_index + 1)],
        best_null_index=best_null_index,
        worst_null_index=worst_null_index,
    )

    with torch.no_grad():
        true_rerun_isodepths = (
            _parallel_module_stack_slice_forward(model.encoder, s_batched_t[:n_reruns], 0, n_reruns)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return selected_slot_indices, train_loss_per_rerun, outputs, true_rerun_isodepths

def _repeat_batched_inputs(array: np.ndarray | None, n_reruns: int) -> np.ndarray | None:
    if array is None:
        return None
    if n_reruns == 1:
        return np.asarray(array)
    return np.repeat(np.asarray(array), n_reruns, axis=0)


def _wall_time_exceeded(training_start: float, max_wall_time_sec: float | None) -> bool:
    return max_wall_time_sec is not None and (time.perf_counter() - training_start) >= float(max_wall_time_sec)


def _resolve_sgd_batch_size(config: TestConfig, n_cells: int) -> int | None:
    if config.sgd_batch_size is None or config.sgd_batch_size == 0:
        return None
    return min(int(config.sgd_batch_size), int(n_cells))


def _sgd_steps_per_epoch(n_cells: int, sgd_batch_size: int) -> int:
    return (int(n_cells) + int(sgd_batch_size) - 1) // int(sgd_batch_size)


def _attach_training_metadata(
    model: nn.Module,
    *,
    n_reruns: int,
    best_train_loss_per_model: np.ndarray,
    best_rerun_index_per_model: np.ndarray,
    train_loss_per_rerun: np.ndarray,
    true_rerun_isodepths: Optional[np.ndarray] = None,
    executed_epochs: int | None = None,
    executed_gradient_steps: int | None = None,
    stopped_by_time: bool = False,
    loss_history: list[float] | None = None,
    loss_history_elapsed_sec: list[float] | None = None,
    loss_history_gradient_updates: list[int] | None = None,
) -> None:
    metadata = {
        "n_reruns": int(n_reruns),
        "selection_loss": "training_reconstruction_loss",
        "best_train_loss_per_model": np.asarray(best_train_loss_per_model, dtype=np.float64),
        "best_rerun_index_per_model": np.asarray(best_rerun_index_per_model, dtype=np.int64),
        "train_loss_per_rerun": np.asarray(train_loss_per_rerun, dtype=np.float64),
    }
    if loss_history is not None:
        metadata["loss_history"] = [float(value) for value in loss_history]
    if loss_history_elapsed_sec is not None:
        metadata["loss_history_elapsed_sec"] = [float(value) for value in loss_history_elapsed_sec]
    if loss_history_gradient_updates is not None:
        metadata["loss_history_gradient_updates"] = [int(value) for value in loss_history_gradient_updates]
    if executed_epochs is not None:
        metadata["executed_epochs"] = int(executed_epochs)
    if executed_gradient_steps is not None:
        metadata["executed_gradient_steps"] = int(executed_gradient_steps)
    if stopped_by_time:
        metadata["stopped_by_time"] = True
    if true_rerun_isodepths is not None:
        metadata["true_rerun_isodepths"] = np.asarray(true_rerun_isodepths, dtype=np.float32)
    model.training_metadata = metadata


def get_training_metadata(model: nn.Module) -> dict[str, Any]:
    metadata = getattr(model, "training_metadata", None)
    if not isinstance(metadata, dict):
        return {
            "n_reruns": 1,
            "selection_loss": "training_reconstruction_loss",
            "best_train_loss_per_model": np.zeros(1, dtype=np.float64),
            "best_rerun_index_per_model": np.zeros(1, dtype=np.int64),
            "train_loss_per_rerun": np.zeros((1, 1), dtype=np.float64),
        "true_rerun_isodepths": None,
        "executed_epochs": None,
        "executed_gradient_steps": None,
        "stopped_by_time": False,
        "loss_history": None,
        "loss_history_elapsed_sec": None,
        "loss_history_gradient_updates": None,
    }
    return {
        "n_reruns": int(metadata.get("n_reruns", 1)),
        "selection_loss": str(metadata.get("selection_loss", "training_reconstruction_loss")),
        "best_train_loss_per_model": np.asarray(metadata.get("best_train_loss_per_model", [0.0]), dtype=np.float64),
        "best_rerun_index_per_model": np.asarray(metadata.get("best_rerun_index_per_model", [0]), dtype=np.int64),
        "train_loss_per_rerun": np.asarray(metadata.get("train_loss_per_rerun", [[0.0]]), dtype=np.float64),
        "true_rerun_isodepths": None
        if metadata.get("true_rerun_isodepths") is None
        else np.asarray(metadata.get("true_rerun_isodepths"), dtype=np.float32),
        "executed_epochs": metadata.get("executed_epochs"),
        "executed_gradient_steps": metadata.get("executed_gradient_steps"),
        "stopped_by_time": bool(metadata.get("stopped_by_time", False)),
        "loss_history": None
        if metadata.get("loss_history") is None
        else np.asarray(metadata.get("loss_history"), dtype=np.float64),
        "loss_history_elapsed_sec": None
        if metadata.get("loss_history_elapsed_sec") is None
        else np.asarray(metadata.get("loss_history_elapsed_sec"), dtype=np.float64),
        "loss_history_gradient_updates": None
        if metadata.get("loss_history_gradient_updates") is None
        else np.asarray(metadata.get("loss_history_gradient_updates"), dtype=np.int64),
    }


def _compact_parallel_model(
    expanded_model: nn.Module,
    *,
    selected_indices: np.ndarray,
    n_models: int,
    n_genes: int,
    latent_dim: int,
    decoder_type: str,
    device: torch.device,
) -> nn.Module:
    if isinstance(expanded_model, HybridMidlineParallelNet):
        return _compact_hybrid_midline_parallel_model(
            expanded_model,
            selected_indices=selected_indices,
            n_models=n_models,
            n_genes=n_genes,
            latent_dim=latent_dim,
            decoder_type=decoder_type,
            device=device,
        )

    n_expanded = _parallel_slot_count(expanded_model)
    compact_model = ParallelIsoDepthNet(
        n_models,
        n_genes,
        latent_dim=latent_dim,
        decoder_type=decoder_type,
    ).to(device)
    expanded_state = expanded_model.state_dict()
    compact_state = compact_model.state_dict()
    slot_indices = [int(index) for index in np.asarray(selected_indices, dtype=np.int64).tolist()]
    restored_state: dict[str, torch.Tensor] = {}
    for name, tensor in compact_state.items():
        source = expanded_state[name].detach().cpu()
        if source.ndim > 0 and source.shape[0] == n_expanded:
            restored_state[name] = source[slot_indices].clone().to(device=device)
        else:
            restored_state[name] = source.clone().to(device=device)
    compact_model.load_state_dict(restored_state)
    return compact_model


def _compact_single_decoder_only_model(
    expanded_model: HybridMidlineParallelNet,
    *,
    selected_index: int,
    n_genes: int,
    latent_dim: int,
    decoder_type: str,
    device: torch.device,
) -> DecoderOnlyNet:
    compact_model = DecoderOnlyNet(
        n_genes,
        latent_dim=latent_dim,
        decoder_type=decoder_type,
    ).to(device)
    idx = int(selected_index)
    slot_state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, tensor in expanded_model.decoder.state_dict().items():
            if tensor.ndim > 0 and tensor.shape[0] > idx:
                slot_state[name] = tensor[idx].detach().clone().to(device=device)
            else:
                slot_state[name] = tensor.detach().clone().to(device=device)
    compact_model.decoder.load_state_dict(slot_state)
    return compact_model


def _compact_fixed_covariate_model(
    expanded_model: ParallelDecoderOnlyNetFixed,
    *,
    selected_index: int,
    n_genes: int,
    decoder_type: str,
    device: torch.device,
) -> DecoderOnlyNetFixed:
    values_np = expanded_model.encoder.latent_values.detach().cpu().numpy()
    compact_model = DecoderOnlyNetFixed(
        n_genes,
        values_np,
        latent_dim=1,
        decoder_type=decoder_type,
    ).to(device)
    idx = int(selected_index)
    slot_state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, tensor in expanded_model.decoder.state_dict().items():
            if tensor.ndim > 0 and tensor.shape[0] > idx:
                slot_state[name] = tensor[idx].detach().clone().to(device=device)
            else:
                slot_state[name] = tensor.detach().clone().to(device=device)
    compact_model.decoder.load_state_dict(slot_state)
    return compact_model


def _compact_single_model(
    expanded_model: ParallelIsoDepthNet,
    *,
    selected_index: int,
    n_genes: int,
    latent_dim: int,
    decoder_type: str,
    device: torch.device,
) -> IsoDepthNet:
    compact_model = IsoDepthNet(
        n_genes,
        latent_dim=latent_dim,
        decoder_type=decoder_type,
    ).to(device)
    parallel_layers = [
        expanded_model.encoder[0],
        expanded_model.encoder[2],
        expanded_model.encoder[4],
    ]
    single_layers = [
        compact_model.encoder[0],
        compact_model.encoder[2],
        compact_model.encoder[4],
    ]
    with torch.no_grad():
        for parallel_layer, single_layer in zip(parallel_layers, single_layers):
            single_layer.weight.copy_(parallel_layer.weight[selected_index])
            single_layer.bias.copy_(parallel_layer.bias[selected_index])
        compact_model.decoder.load_state_dict({
            name: tensor[selected_index].detach().cpu()
            for name, tensor in expanded_model.decoder.state_dict().items()
        })
    return compact_model


def train_batched_isodepth_model(
    s_batched: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    a_batched: Optional[np.ndarray] = None,
    loss_mask_batched: Optional[np.ndarray] = None,
    metric_loss_mask_batched: Optional[np.ndarray] = None,
    latent_dim: int = 1,
    model_label: str = "parallel isodepth batch",
    initial_state: Mapping[str, torch.Tensor] | None = None,
    gradient_scale_divisor: float | None = None,
) -> tuple[nn.Module, BatchedTrainingOutputs]:
    device = device or resolve_device(config.device)
    _set_torch_seed(config.seed)

    s_batched_np = np.asarray(s_batched, dtype=np.float32)
    if s_batched_np.ndim != 3 or s_batched_np.shape[-1] != 2:
        raise ValueError(f"s_batched must have shape (M, N, 2), got {s_batched_np.shape}")

    n_models, n_cells, _ = s_batched_np.shape
    n_cells_a, n_genes = A.shape
    if n_cells != n_cells_a:
        raise ValueError(f"s_batched and A must have matching cell counts, got {n_cells} vs {n_cells_a}")

    if latent_dim <= 0:
        raise ValueError("latent_dim must be >= 1")
    if _is_midline_covariate(config) and latent_dim != 1:
        raise ValueError("covariate type midline requires latent_dim == 1")
    decoder_type = _resolve_decoder_type(config)
    if gradient_scale_divisor is not None and float(gradient_scale_divisor) <= 0.0:
        raise ValueError("gradient_scale_divisor must be > 0 when provided")

    # When a_batched is None, every parallel slot uses the same expression
    # targets A.  Keep a single (N, G) tensor on GPU instead of (M, N, G)
    # to avoid duplicating ~total_models × N × G × 4 bytes of VRAM.
    broadcast_a = a_batched is None
    if not broadcast_a:
        base_a_batched = np.asarray(a_batched, dtype=np.float32)
        if base_a_batched.ndim != 3:
            raise ValueError(f"a_batched must have shape (M, N, G), got {base_a_batched.shape}")
        if base_a_batched.shape != (n_models, n_cells, n_genes):
            raise ValueError(
                "a_batched must match (M, N, G) for the supplied s_batched and A, "
                f"got {base_a_batched.shape} vs {(n_models, n_cells, n_genes)}"
            )

    n_reruns = int(config.n_reruns)
    expanded_s_batched = _repeat_batched_inputs(s_batched_np, n_reruns)
    expanded_loss_mask = _repeat_batched_inputs(loss_mask_batched, n_reruns)

    total_models = int(expanded_s_batched.shape[0])
    s_batched_t = torch.tensor(expanded_s_batched, dtype=torch.float32, device=device)
    if broadcast_a:
        a_t = torch.tensor(np.asarray(A, dtype=np.float32), dtype=torch.float32, device=device)
    else:
        expanded_a_batched = _repeat_batched_inputs(base_a_batched, n_reruns)
        a_t = torch.tensor(expanded_a_batched, dtype=torch.float32, device=device)
    loss_mask_t = _prepare_loss_mask(
        expanded_loss_mask,
        n_models=total_models,
        n_cells=n_cells,
        n_genes=n_genes,
        device=device,
    )
    expanded_metric_loss_mask = _repeat_batched_inputs(metric_loss_mask_batched, n_reruns)
    metric_loss_mask_t = _prepare_loss_mask(
        expanded_metric_loss_mask,
        n_models=total_models,
        n_cells=n_cells,
        n_genes=n_genes,
        device=device,
    )

    if _is_midline_covariate(config):
        model = HybridMidlineParallelNet(
            total_models,
            n_genes,
            slot_split=n_reruns,
            latent_dim=1,
            decoder_type=decoder_type,
        ).to(device)
    else:
        model = ParallelIsoDepthNet(
            total_models,
            n_genes,
            latent_dim=latent_dim,
            decoder_type=decoder_type,
        ).to(device)
    _load_initial_state(model, initial_state, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, foreach=False)
    sgd_batch_size = _resolve_sgd_batch_size(config, n_cells)
    lr_scheduler_step: lr_scheduler.CosineAnnealingLR | None = None
    if sgd_batch_size is not None and config.sgd_cosine_lr_decay:
        steps_per_epoch = _sgd_steps_per_epoch(n_cells, sgd_batch_size)
        t_max = config.sgd_cosine_t_max_steps
        if t_max is None:
            t_max = int(config.epochs) * steps_per_epoch
        t_max = max(int(t_max), 1)
        lr_scheduler_step = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=float(config.sgd_cosine_eta_min),
        )
    active_mask_t = torch.ones(total_models, dtype=torch.float32, device=device)
    minibatch_generator = None
    if sgd_batch_size is not None:
        minibatch_generator = torch.Generator(device="cpu")
        minibatch_generator.manual_seed(config.seed)

    use_patience = config.patience > 0
    best_loss_per_model = np.full(total_models, np.inf, dtype=np.float64)
    patience_counter_per_model = np.zeros(total_models, dtype=np.int64)
    active_mask_np = np.ones(total_models, dtype=bool)
    if use_patience:
        if isinstance(model, HybridMidlineParallelNet):
            best_state = _snapshot_hybrid_midline_state(model, total_models, n_reruns)
        else:
            best_state = _snapshot_parallel_model_state(model, total_models)
    iterator = tqdm(range(config.epochs), disable=not config.verbose)
    training_start = time.perf_counter()
    executed_epochs = 0
    executed_gradient_steps = 0
    stopped_by_time = False
    record_loss_history = bool(getattr(config, "record_loss_history", False))
    loss_history: list[float] = []
    loss_history_elapsed_sec: list[float] = []
    loss_history_gradient_updates: list[int] = []
    max_wall_time_sec = getattr(config, "max_wall_time_sec", None)
    for epoch in iterator:
        if _wall_time_exceeded(training_start, max_wall_time_sec):
            stopped_by_time = True
            if config.verbose:
                print(
                    f"[time-stop] {model_label} stopped at epoch {epoch} "
                    f"after {executed_gradient_steps} gradient steps "
                    f"(max_wall_time_sec={float(max_wall_time_sec):.3f})"
                )
            break
        active_mask_t.copy_(torch.from_numpy(active_mask_np.astype(np.float32)))
        active_count = float(active_mask_np.sum())
        if sgd_batch_size is None:
            optimizer.zero_grad()
            output = model(s_batched_t)
            loss_per_model = _compute_reconstruction_loss_per_model(output, a_t, loss_mask_t)
            divisor = float(gradient_scale_divisor) if gradient_scale_divisor is not None else max(active_count, 1.0)
            total_loss = (loss_per_model * active_mask_t).sum() / divisor
            total_loss.backward()
            optimizer.step()
            executed_gradient_steps += 1
        else:
            permutation = torch.randperm(n_cells, generator=minibatch_generator)
            for start in range(0, n_cells, sgd_batch_size):
                if _wall_time_exceeded(training_start, max_wall_time_sec):
                    stopped_by_time = True
                    break
                batch_indices = permutation[start : start + sgd_batch_size].to(device=device)
                batch_s = s_batched_t.index_select(1, batch_indices)
                batch_a = a_t.index_select(-2, batch_indices)
                batch_mask = None if loss_mask_t is None else loss_mask_t.index_select(1, batch_indices)

                optimizer.zero_grad()
                batch_output = model(batch_s)
                batch_loss_per_model = _compute_reconstruction_loss_per_model(batch_output, batch_a, batch_mask)
                divisor = float(gradient_scale_divisor) if gradient_scale_divisor is not None else max(active_count, 1.0)
                batch_total_loss = (batch_loss_per_model * active_mask_t).sum() / divisor
                batch_total_loss.backward()
                optimizer.step()
                if lr_scheduler_step is not None:
                    lr_scheduler_step.step()
                executed_gradient_steps += 1
            if stopped_by_time:
                if config.verbose:
                    print(
                        f"[time-stop] {model_label} stopped during epoch {epoch + 1} "
                        f"after {executed_gradient_steps} gradient steps "
                        f"(max_wall_time_sec={float(max_wall_time_sec):.3f})"
                    )
                break

        executed_epochs = epoch + 1

        if use_patience:
            with torch.no_grad():
                output = model(s_batched_t)
                loss_per_model = _compute_reconstruction_loss_per_model(output, a_t, loss_mask_t)
            loss_values = loss_per_model.detach().cpu().numpy().astype(np.float64)
            improved_mask = active_mask_np & (loss_values < (best_loss_per_model - 1e-5))
            if np.any(improved_mask):
                best_loss_per_model[improved_mask] = loss_values[improved_mask]
                patience_counter_per_model[improved_mask] = 0
                if isinstance(model, HybridMidlineParallelNet):
                    _update_hybrid_midline_snapshot(
                        best_state,
                        model,
                        improved_mask,
                        total_models,
                        n_reruns,
                    )
                else:
                    _update_parallel_model_snapshot(best_state, model, improved_mask, total_models)

            stalled_mask = active_mask_np & ~improved_mask
            patience_counter_per_model[stalled_mask] += 1
            active_mask_np = patience_counter_per_model < config.patience

            if not np.any(active_mask_np):
                if config.verbose:
                    print(
                        f"[early-stop] {model_label} stopped at epoch {epoch + 1} "
                        f"(all {total_models} batched models exhausted patience={config.patience})"
                    )
                break

        if record_loss_history:
            if use_patience:
                true_loss = float(loss_per_model[0].detach().cpu().item())
            else:
                with torch.no_grad():
                    hist_output = model(s_batched_t)
                    hist_loss_per_model = _compute_reconstruction_loss_per_model(
                        hist_output, a_t, loss_mask_t
                    )
                true_loss = float(hist_loss_per_model[0].detach().cpu().item())
            loss_history.append(true_loss)
            loss_history_elapsed_sec.append(float(time.perf_counter() - training_start))
            loss_history_gradient_updates.append(int(executed_gradient_steps))

    if use_patience:
        if isinstance(model, HybridMidlineParallelNet):
            _restore_hybrid_midline_snapshot(model, best_state, device)
        else:
            _restore_parallel_model_snapshot(model, best_state, device)
    selected_slot_indices, train_loss_per_rerun, training_outputs, true_rerun_isodepths = (
        _finalize_batched_parallel_training(
            model,
            s_batched_t,
            a_t,
            loss_mask_t,
            A,
            config,
            n_models=n_models,
            n_reruns=n_reruns,
            n_genes=n_genes,
            latent_dim=latent_dim,
            device=device,
            metric_loss_mask_t=metric_loss_mask_t,
        )
    )
    compact_model = _compact_parallel_model(
        model,
        selected_indices=selected_slot_indices,
        n_models=n_models,
        n_genes=n_genes,
        latent_dim=latent_dim,
        decoder_type=decoder_type,
        device=device,
    )
    best_rerun_index_per_model = np.argmin(train_loss_per_rerun, axis=1).astype(np.int64)
    best_train_loss_per_model = train_loss_per_rerun[
        np.arange(n_models, dtype=np.int64), best_rerun_index_per_model
    ]
    _attach_training_metadata(
        compact_model,
        n_reruns=n_reruns,
        best_train_loss_per_model=best_train_loss_per_model,
        best_rerun_index_per_model=best_rerun_index_per_model,
        train_loss_per_rerun=train_loss_per_rerun,
        true_rerun_isodepths=true_rerun_isodepths,
        executed_epochs=executed_epochs,
        executed_gradient_steps=executed_gradient_steps,
        stopped_by_time=stopped_by_time,
        loss_history=loss_history if record_loss_history else None,
        loss_history_elapsed_sec=loss_history_elapsed_sec if record_loss_history else None,
        loss_history_gradient_updates=loss_history_gradient_updates if record_loss_history else None,
    )
    return compact_model, training_outputs


# ---------------------------------------------------------------------------
# OOM-split helpers: train in chunks when GPU memory is insufficient
# ---------------------------------------------------------------------------

def _merge_chunked_training_results(
    chunk_models: list[nn.Module],
    chunk_outputs: list[BatchedTrainingOutputs],
    s_batched_np: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    latent_dim: int,
    device: torch.device,
) -> tuple[nn.Module, BatchedTrainingOutputs]:
    """Merge results from independently-trained chunks into a single result.

    Each chunk trained a contiguous subset of the ``(n_perms + 1)`` permutation
    models.  This function reassembles the compact model, metrics, and the
    three prediction matrices so the caller sees the same interface as a
    single-batch run.
    """
    n_perm_models = s_batched_np.shape[0]
    n_genes = A.shape[1]
    metric_name = canonicalize_metric_name(config.metric)
    decoder_type = _resolve_decoder_type(config)

    all_metrics = np.concatenate(
        [out.model_metrics for out in chunk_outputs],
    )

    best_null_index, worst_null_index = _null_extreme_indices(all_metrics, metric_name)

    pred_true = chunk_outputs[0].pred_true

    chunk_sizes = [out.model_metrics.shape[0] for out in chunk_outputs]

    def _chunk_for_global(global_model_idx: int) -> tuple[int, int]:
        offset = 0
        for ci, cs in enumerate(chunk_sizes):
            if global_model_idx < offset + cs:
                return ci, global_model_idx - offset
            offset += cs
        raise IndexError(f"global model index {global_model_idx} out of range")

    def _pred_for_global(global_model_idx: int) -> np.ndarray:
        ci, local_idx = _chunk_for_global(global_model_idx)
        cm = chunk_models[ci]
        cm.to(device)
        s_t = torch.tensor(
            s_batched_np[global_model_idx : global_model_idx + 1],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            pred = _forward_parallel_model_slice(cm, s_t, local_idx, local_idx + 1)
        pred_np = pred[0].detach().cpu().numpy().astype(np.float32)
        offload_module_to_cpu(cm)
        return pred_np

    pred_best_null = _pred_for_global(best_null_index + 1)
    pred_worst_null = _pred_for_global(worst_null_index + 1)

    outputs = BatchedTrainingOutputs(
        model_metrics=all_metrics,
        pred_true=pred_true,
        pred_best_null=pred_best_null,
        pred_worst_null=pred_worst_null,
        best_null_index=best_null_index,
        worst_null_index=worst_null_index,
    )

    is_hybrid = isinstance(chunk_models[0], HybridMidlineParallelNet)
    if is_hybrid:
        merged_model = HybridMidlineParallelNet(
            n_perm_models, n_genes, slot_split=1,
            latent_dim=latent_dim, decoder_type=decoder_type,
        )
    else:
        merged_model = ParallelIsoDepthNet(
            n_perm_models, n_genes,
            latent_dim=latent_dim, decoder_type=decoder_type,
        )
    merged_state = merged_model.state_dict()
    offset = 0
    for cm in chunk_models:
        chunk_slot_count = _parallel_slot_count(cm)
        chunk_state = cm.state_dict()
        for name in merged_state:
            src = chunk_state[name]
            if src.ndim > 0 and src.shape[0] == chunk_slot_count:
                merged_state[name][offset : offset + chunk_slot_count] = src
        offset += chunk_slot_count
    merged_model.load_state_dict(merged_state)

    n_reruns = int(config.n_reruns)
    all_train_loss_per_model = np.concatenate(
        [get_training_metadata(m)["best_train_loss_per_model"] for m in chunk_models],
    )
    all_best_rerun_index = np.concatenate(
        [get_training_metadata(m)["best_rerun_index_per_model"] for m in chunk_models],
    )
    all_train_loss_per_rerun = np.concatenate(
        [get_training_metadata(m)["train_loss_per_rerun"] for m in chunk_models],
    )
    true_rerun_isodepths = get_training_metadata(chunk_models[0]).get("true_rerun_isodepths")

    _attach_training_metadata(
        merged_model,
        n_reruns=n_reruns,
        best_train_loss_per_model=all_train_loss_per_model,
        best_rerun_index_per_model=all_best_rerun_index,
        train_loss_per_rerun=all_train_loss_per_rerun,
        true_rerun_isodepths=true_rerun_isodepths,
    )
    return merged_model, outputs


def train_parallel_isodepth_model(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    s_batched: Optional[np.ndarray] = None,
    a_batched: Optional[np.ndarray] = None,
    loss_mask_batched: Optional[np.ndarray] = None,
    metric_loss_mask_batched: Optional[np.ndarray] = None,
    latent_dim: int = 1,
    model_label: Optional[str] = None,
    initial_state: Mapping[str, torch.Tensor] | None = None,
    gradient_scale_divisor: float | None = None,
) -> tuple[nn.Module, BatchedTrainingOutputs, np.ndarray]:
    """Returns (compact_model, training_outputs, s_batched_np).

    ``s_batched_np`` is the ``(n_perms + 1, N, 2)`` coordinate array that was
    used for training (generated or passed in).  Callers can reuse it for
    post-training isodepth extraction without re-building the permutation batch.
    """
    device = device or resolve_device(config.device)
    if s_batched is None:
        n_models = config.n_perms + 1
        n_cells = A.shape[0]
        s_t = torch.tensor(S, dtype=torch.float32, device=device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)

        s_batched_t = torch.zeros((n_models, n_cells, 2), dtype=torch.float32, device=device)
        s_batched_t[0] = s_t
        for model_index in range(1, n_models):
            perm = torch.randperm(n_cells, generator=generator)
            s_batched_t[model_index] = s_t[perm.to(device=device)]
        s_batched = s_batched_t.detach().cpu().numpy()
        if model_label is None:
            model_label = f"parallel isodepth batch (true + {config.n_perms} permuted models)"
    elif model_label is None:
        model_label = "parallel isodepth batch"

    s_batched_np = np.asarray(s_batched, dtype=np.float32)

    def _slice_optional(arr: np.ndarray | None, start: int, end: int) -> np.ndarray | None:
        return None if arr is None else arr[start:end]

    def _train_chunk(chunk_s: np.ndarray, chunk_a_batched, chunk_loss_mask,
                     chunk_metric_mask, label: str) -> tuple[nn.Module, BatchedTrainingOutputs]:
        return train_batched_isodepth_model(
            chunk_s, A, config, device=device,
            a_batched=chunk_a_batched,
            loss_mask_batched=chunk_loss_mask,
            metric_loss_mask_batched=chunk_metric_mask,
            latent_dim=latent_dim, model_label=label,
            initial_state=None,
            gradient_scale_divisor=gradient_scale_divisor,
        )

    # -- try full batch first; on OOM, halve until a chunk size works --------
    n_perm_models = s_batched_np.shape[0]
    try:
        model, outputs = train_batched_isodepth_model(
            s_batched_np, A, config, device=device,
            a_batched=a_batched,
            loss_mask_batched=loss_mask_batched,
            metric_loss_mask_batched=metric_loss_mask_batched,
            latent_dim=latent_dim, model_label=model_label,
            initial_state=initial_state,
            gradient_scale_divisor=gradient_scale_divisor,
        )
        return model, outputs, s_batched_np
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    n_chunks = 2
    first_model: nn.Module | None = None
    first_outputs: BatchedTrainingOutputs | None = None
    while n_chunks <= n_perm_models:
        chunk_size = math.ceil(n_perm_models / n_chunks)
        if config.verbose:
            print(f"[OOM split] Trying {n_chunks} chunks of ~{chunk_size} perm models")
        try:
            first_model, first_outputs = _train_chunk(
                s_batched_np[:chunk_size],
                _slice_optional(a_batched, 0, chunk_size),
                _slice_optional(loss_mask_batched, 0, chunk_size),
                _slice_optional(metric_loss_mask_batched, 0, chunk_size),
                f"{model_label} [chunk 1/{n_chunks}]",
            )
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_chunks *= 2

    if first_model is None:
        raise RuntimeError(
            f"CUDA OOM even with 1 model per chunk "
            f"({n_perm_models} perm models, {A.shape[0]} cells, {A.shape[1]} genes)"
        )

    offload_module_to_cpu(first_model)

    chunk_models: list[nn.Module] = [first_model]
    chunk_outputs_list: list[BatchedTrainingOutputs] = [first_outputs]

    actual_n_chunks = math.ceil(n_perm_models / chunk_size)
    for chunk_idx in range(1, actual_n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_perm_models)
        if start >= n_perm_models:
            break
        if config.verbose:
            print(f"[OOM split] Training chunk {chunk_idx + 1}/{actual_n_chunks} "
                  f"(models {start}-{end - 1})")
        cm, co = _train_chunk(
            s_batched_np[start:end],
            _slice_optional(a_batched, start, end),
            _slice_optional(loss_mask_batched, start, end),
            _slice_optional(metric_loss_mask_batched, start, end),
            f"{model_label} [chunk {chunk_idx + 1}/{actual_n_chunks}]",
        )
        offload_module_to_cpu(cm)
        chunk_models.append(cm)
        chunk_outputs_list.append(co)

    merged_model, merged_outputs = _merge_chunked_training_results(
        chunk_models, chunk_outputs_list,
        s_batched_np, A, config,
        latent_dim=latent_dim, device=device,
    )
    return merged_model, merged_outputs, s_batched_np


def train_isodepth_model(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    seed_offset: int = 0,
    latent_dim: int = 1,
    model_label: str = "model",
    initial_state: Mapping[str, torch.Tensor] | None = None,
    gradient_scale_divisor: float | None = None,
) -> tuple[nn.Module, np.ndarray]:
    device = device or resolve_device(config.device)
    if latent_dim <= 0:
        raise ValueError("latent_dim must be >= 1")

    effective_config = replace(config, seed=config.seed + seed_offset)
    parallel_model, training_outputs = train_batched_isodepth_model(
        np.asarray(S, dtype=np.float32)[None, :, :],
        A,
        effective_config,
        device=device,
        latent_dim=latent_dim,
        model_label=model_label,
        initial_state=initial_state,
        gradient_scale_divisor=gradient_scale_divisor,
    )
    metadata = get_training_metadata(parallel_model)
    dec_type = _resolve_decoder_type(config)
    if isinstance(parallel_model, HybridMidlineParallelNet):
        model = _compact_single_decoder_only_model(
            parallel_model,
            selected_index=0,
            n_genes=A.shape[1],
            latent_dim=latent_dim,
            decoder_type=dec_type,
            device=device,
        )
    else:
        model = _compact_single_model(
            parallel_model,
            selected_index=0,
            n_genes=A.shape[1],
            latent_dim=latent_dim,
            decoder_type=dec_type,
            device=device,
        )
    _attach_training_metadata(
        model,
        n_reruns=metadata["n_reruns"],
        best_train_loss_per_model=metadata["best_train_loss_per_model"],
        best_rerun_index_per_model=metadata["best_rerun_index_per_model"],
        train_loss_per_rerun=metadata["train_loss_per_rerun"],
        true_rerun_isodepths=metadata["true_rerun_isodepths"],
    )
    return model, np.asarray(training_outputs.pred_true, dtype=np.float32)


def train_celltype_parallel_isodepth_model(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    cell_type_labels: np.ndarray,
    n_cell_types: int,
    device: Optional[torch.device] = None,
    s_batched: Optional[np.ndarray] = None,
    latent_dim: int = 1,
    model_label: Optional[str] = None,
) -> tuple[nn.Module, BatchedTrainingOutputs, np.ndarray]:
    """Train a shared-encoder + per-cell-type-decoder model in parallel across permutations.

    Same permutation structure as ``train_parallel_isodepth_model`` but uses
    ``ParallelCellTypeIsoDepthNet`` with C cell-type-specific decoder heads.

    Returns (compact_model, training_outputs, s_batched_np).
    """
    device = device or resolve_device(config.device)
    _set_torch_seed(config.seed)

    n_cells, n_genes = A.shape
    cell_type_labels_np = np.asarray(cell_type_labels, dtype=np.int64)
    if cell_type_labels_np.shape[0] != n_cells:
        raise ValueError(
            f"cell_type_labels length {cell_type_labels_np.shape[0]} != n_cells {n_cells}"
        )

    if s_batched is None:
        n_models = config.n_perms + 1
        s_t = torch.tensor(S, dtype=torch.float32, device=device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        s_batched_t = torch.zeros((n_models, n_cells, 2), dtype=torch.float32, device=device)
        s_batched_t[0] = s_t
        for model_index in range(1, n_models):
            perm = torch.randperm(n_cells, generator=generator)
            s_batched_t[model_index] = s_t[perm.to(device=device)]
        s_batched_np = s_batched_t.detach().cpu().numpy()
        if model_label is None:
            model_label = f"cell-type parallel batch (true + {config.n_perms} permuted)"
    else:
        s_batched_np = np.asarray(s_batched, dtype=np.float32)
        n_models = s_batched_np.shape[0]
        if model_label is None:
            model_label = "cell-type parallel batch"

    decoder_type = _resolve_decoder_type(config)
    n_reruns = int(config.n_reruns)
    total_models = n_models * n_reruns

    expanded_s_batched = _repeat_batched_inputs(s_batched_np, n_reruns)
    s_batched_t = torch.tensor(expanded_s_batched, dtype=torch.float32, device=device)

    # Subtract per-cell-type mean expression so decoders cannot trivially memorize
    # type means and the encoder is forced to capture within-type spatial variation.
    a_np = np.asarray(A, dtype=np.float32)
    type_means = np.zeros((n_cell_types, n_genes), dtype=np.float32)
    for c in range(n_cell_types):
        mask = cell_type_labels_np == c
        if mask.any():
            type_means[c] = a_np[mask].mean(axis=0)
    a_residual = a_np - type_means[cell_type_labels_np]

    a_batched_t = torch.tensor(
        np.repeat(a_residual[None, :, :], total_models, axis=0),
        dtype=torch.float32,
        device=device,
    )
    cell_type_t = torch.tensor(cell_type_labels_np, dtype=torch.long, device=device)

    model = ParallelCellTypeIsoDepthNet(
        total_models,
        n_cell_types,
        n_genes,
        latent_dim=latent_dim,
        decoder_type=decoder_type,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, foreach=False)

    use_patience = config.patience > 0
    best_loss_per_model = np.full(total_models, np.inf, dtype=np.float64)
    patience_counter_per_model = np.zeros(total_models, dtype=np.int64)
    active_mask_np = np.ones(total_models, dtype=bool)
    active_mask_t = torch.ones(total_models, dtype=torch.float32, device=device)
    if use_patience:
        best_state = _snapshot_parallel_model_state(model, total_models)

    iterator = tqdm(range(config.epochs), disable=not config.verbose)
    for epoch in iterator:
        active_mask_t.copy_(torch.from_numpy(active_mask_np.astype(np.float32)))
        active_count = float(active_mask_np.sum())

        optimizer.zero_grad()
        output = model(s_batched_t, cell_type_t)
        loss_per_model = (output - a_batched_t).pow(2).mean(dim=(1, 2))
        total_loss = (loss_per_model * active_mask_t).sum() / max(active_count, 1.0)
        total_loss.backward()
        optimizer.step()

        if use_patience:
            with torch.no_grad():
                output = model(s_batched_t, cell_type_t)
                loss_per_model = (output - a_batched_t).pow(2).mean(dim=(1, 2))
            loss_values = loss_per_model.detach().cpu().numpy().astype(np.float64)

            improved_mask = active_mask_np & (loss_values < (best_loss_per_model - 1e-5))
            if np.any(improved_mask):
                best_loss_per_model[improved_mask] = loss_values[improved_mask]
                patience_counter_per_model[improved_mask] = 0
                _update_parallel_model_snapshot(best_state, model, improved_mask, total_models)

            stalled_mask = active_mask_np & ~improved_mask
            patience_counter_per_model[stalled_mask] += 1
            active_mask_np = patience_counter_per_model < config.patience

            if not np.any(active_mask_np):
                if config.verbose:
                    print(
                        f"[early-stop] {model_label} stopped at epoch {epoch + 1} "
                        f"(all {total_models} models exhausted patience={config.patience})"
                    )
                break

    if use_patience:
        _restore_parallel_model_snapshot(model, best_state, device)

    selected_slot_indices, train_loss_per_rerun, training_outputs, true_rerun_isodepths = (
        _finalize_celltype_parallel_training(
            model,
            s_batched_t,
            a_batched_t,
            cell_type_t,
            A,
            type_means,
            cell_type_labels_np,
            config,
            n_models=n_models,
            n_reruns=n_reruns,
            latent_dim=latent_dim,
        )
    )

    compact_model = ParallelCellTypeIsoDepthNet(
        n_models,
        n_cell_types,
        n_genes,
        latent_dim=latent_dim,
        decoder_type=decoder_type,
    ).to(device)
    expanded_state = model.state_dict()
    compact_state = compact_model.state_dict()
    slot_indices = [int(idx) for idx in selected_slot_indices.tolist()]
    for name, tensor in compact_state.items():
        source = expanded_state[name].detach().cpu()
        if source.ndim > 0 and source.shape[0] == total_models:
            compact_state[name] = source[slot_indices].clone().to(device=device)
        else:
            compact_state[name] = source.clone().to(device=device)
    compact_model.load_state_dict(compact_state)

    best_rerun_index_per_model = np.argmin(train_loss_per_rerun, axis=1).astype(np.int64)
    best_train_loss_per_model = train_loss_per_rerun[
        np.arange(n_models, dtype=np.int64), best_rerun_index_per_model
    ]

    _attach_training_metadata(
        compact_model,
        n_reruns=n_reruns,
        best_train_loss_per_model=best_train_loss_per_model,
        best_rerun_index_per_model=best_rerun_index_per_model,
        train_loss_per_rerun=train_loss_per_rerun,
        true_rerun_isodepths=true_rerun_isodepths,
    )
    return compact_model, training_outputs, s_batched_np


def extract_celltype_model_isodepth(
    model: ParallelCellTypeIsoDepthNet,
    S: np.ndarray,
    device: torch.device,
    *,
    slot_index: int = 0,
) -> np.ndarray:
    """Extract isodepth from a specific parallel slot's encoder weights."""
    return extract_model_isodepth(model, S, device, slot_index=slot_index)


def train_fixed_covariate_model(
    covariate_values: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    model_label: str = "obs-key covariate",
) -> tuple[nn.Module, np.ndarray]:
    """Train a decoder-only model using pre-computed per-cell latent values from ``adata.obs``.

    The latent values are stored as a frozen buffer in :class:`DecoderOnlyNetFixed`;
    only the decoder weights are updated during training.  Supports both full-batch and
    SGD mini-batch training (``config.sgd_batch_size``).

    Runs ``config.n_reruns`` independent decoder initializations in parallel via
    :class:`ParallelDecoderOnlyNetFixed`, selects the rerun with the lowest training
    reconstruction loss, and returns a compact :class:`DecoderOnlyNetFixed` (same rule
    as parallel null and midline covariate training).

    Parameters
    ----------
    covariate_values:
        Per-cell covariate values of shape ``(N,)`` or ``(N, 1)``.
    A:
        Expression matrix of shape ``(N, G)``.
    config:
        Test configuration (uses ``epochs``, ``lr``, ``seed``, ``n_reruns``,
        ``sgd_batch_size``, ``verbose``, and ``decoder``).
    device:
        Target device; resolved from ``config.device`` if omitted.
    model_label:
        Label shown in the tqdm progress bar.

    Returns
    -------
    model:
        Trained :class:`DecoderOnlyNetFixed` for the best parallel rerun.
    pred:
        Expression predictions of shape ``(N, G)`` from the best rerun's decoder.
    """
    device = device or resolve_device(config.device)
    a_np = np.asarray(A, dtype=np.float32)
    n_cells, n_genes = a_np.shape
    n_reruns = int(config.n_reruns)
    dec_type = _resolve_decoder_type(config)
    values_np = zscore_covariate(covariate_values).reshape(n_cells, 1)

    _set_torch_seed(config.seed)
    parallel_model = ParallelDecoderOnlyNetFixed(
        n_reruns,
        n_genes,
        values_np,
        latent_dim=1,
        decoder_type=dec_type,
    ).to(device)
    a_t = torch.tensor(a_np, dtype=torch.float32, device=device)
    optimizer = optim.Adam(parallel_model.decoder.parameters(), lr=config.lr, foreach=False)
    sgd_batch_size = _resolve_sgd_batch_size(config, n_cells)

    use_patience = config.patience > 0
    use_best_state = sgd_batch_size is not None
    best_loss_per_model = np.full(n_reruns, np.inf, dtype=np.float64)
    patience_counter_per_model = np.zeros(n_reruns, dtype=np.int64)
    active_mask_np = np.ones(n_reruns, dtype=bool)
    active_mask_t = torch.ones(n_reruns, dtype=torch.float32, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_state_gpu: dict[str, torch.Tensor] | None = None
    if use_best_state:
        best_state_gpu = _snapshot_parallel_model_state_on_device(parallel_model, n_reruns, device)
    elif use_patience:
        best_state = _snapshot_parallel_model_state(parallel_model, n_reruns)

    minibatch_generator: Optional[torch.Generator] = None
    if sgd_batch_size is not None:
        minibatch_generator = torch.Generator(device="cpu")
        minibatch_generator.manual_seed(config.seed)

    iterator = tqdm(range(config.epochs), disable=not config.verbose, desc=model_label)
    for epoch in iterator:
        active_mask_t.copy_(torch.from_numpy(active_mask_np.astype(np.float32)))
        active_count = float(active_mask_np.sum())
        if sgd_batch_size is None:
            optimizer.zero_grad()
            output = parallel_model()
            loss_per_model = _compute_reconstruction_loss_per_model(output, a_t, None)
            total_loss = (loss_per_model * active_mask_t).sum() / max(active_count, 1.0)
            total_loss.backward()
            optimizer.step()
        else:
            permutation = torch.randperm(n_cells, generator=minibatch_generator)
            for start in range(0, n_cells, sgd_batch_size):
                batch_indices = permutation[start : start + sgd_batch_size].to(device=device)
                batch_latent = parallel_model.encoder.latent_values.index_select(0, batch_indices)
                batch_latent = batch_latent.unsqueeze(0).expand(n_reruns, -1, -1)
                batch_a = a_t.index_select(0, batch_indices)

                optimizer.zero_grad()
                batch_output = parallel_model.decoder(batch_latent)
                batch_loss_per_model = _compute_reconstruction_loss_per_model(
                    batch_output,
                    batch_a,
                    None,
                )
                batch_total_loss = (batch_loss_per_model * active_mask_t).sum() / max(active_count, 1.0)
                batch_total_loss.backward()
                optimizer.step()

        if use_best_state or use_patience:
            with torch.no_grad():
                output = parallel_model()
                loss_per_model = _compute_reconstruction_loss_per_model(output, a_t, None)
            loss_values = loss_per_model.detach().cpu().numpy().astype(np.float64)
            improved_mask = active_mask_np & (loss_values < (best_loss_per_model - 1e-5))
            if np.any(improved_mask):
                best_loss_per_model[improved_mask] = loss_values[improved_mask]
                if use_best_state:
                    assert best_state_gpu is not None
                    _update_parallel_model_snapshot_on_device(
                        best_state_gpu,
                        parallel_model,
                        np.flatnonzero(improved_mask),
                        n_reruns,
                    )
                else:
                    assert best_state is not None
                    _update_parallel_model_snapshot(
                        best_state,
                        parallel_model,
                        improved_mask,
                        n_reruns,
                    )
                if use_patience:
                    patience_counter_per_model[improved_mask] = 0

            if use_patience:
                stalled_mask = active_mask_np & ~improved_mask
                patience_counter_per_model[stalled_mask] += 1
                active_mask_np = patience_counter_per_model < config.patience
                if not np.any(active_mask_np):
                    if config.verbose:
                        print(
                            f"[early-stop] {model_label} stopped at epoch {epoch + 1} "
                            f"(all {n_reruns} reruns exhausted patience={config.patience})"
                        )
                    break

    if use_best_state and best_state_gpu is not None:
        _restore_parallel_model_snapshot_on_device(parallel_model, best_state_gpu)
    elif use_patience and best_state is not None:
        _restore_parallel_model_snapshot(parallel_model, best_state, device)

    with torch.no_grad():
        output = parallel_model()
        train_loss_per_rerun = _compute_reconstruction_loss_per_model(
            output,
            a_t,
            None,
        ).detach().cpu().numpy().astype(np.float64)

    best_rerun_index = int(np.argmin(train_loss_per_rerun))
    best_train_loss = float(train_loss_per_rerun[best_rerun_index])
    compact_model = _compact_fixed_covariate_model(
        parallel_model,
        selected_index=best_rerun_index,
        n_genes=n_genes,
        decoder_type=dec_type,
        device=device,
    )
    with torch.no_grad():
        pred = compact_model.decoder(compact_model.encoder.latent_values).detach().cpu().numpy()

    _attach_training_metadata(
        compact_model,
        n_reruns=n_reruns,
        best_train_loss_per_model=np.asarray([best_train_loss], dtype=np.float64),
        best_rerun_index_per_model=np.asarray([best_rerun_index], dtype=np.int64),
        train_loss_per_rerun=train_loss_per_rerun,
    )
    if n_reruns > 1 and config.verbose:
        print(
            f"{model_label}: selected rerun {best_rerun_index + 1}/{n_reruns} "
            f"(train MSE={best_train_loss:.6f})"
        )
    return compact_model, np.asarray(pred, dtype=np.float32)
