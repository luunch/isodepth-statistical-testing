from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.schemas import TestConfig
from methods.architectures import IsoDepthNet, ParallelIsoDepthNet


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device but CUDA is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device but MPS is not available")
    return resolved


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_predictions(model: nn.Module, S: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        s_t = torch.tensor(S, dtype=torch.float32, device=device)
        preds = model(s_t).detach().cpu().numpy()
    return np.asarray(preds, dtype=np.float32)


def extract_model_isodepth(model: nn.Module, S: np.ndarray, device: torch.device) -> np.ndarray:
    latent_dim = int(getattr(model, "latent_dim", 0))
    if latent_dim <= 0 or not hasattr(model, "encoder"):
        return np.zeros((S.shape[0], 0), dtype=np.float32)
    with torch.no_grad():
        s_t = torch.tensor(S, dtype=torch.float32, device=device)
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


def _compute_reconstruction_loss_per_model(
    output: torch.Tensor,
    targets: torch.Tensor,
    loss_mask_t: torch.Tensor | None,
) -> torch.Tensor:
    squared_error = (output - targets) ** 2
    if loss_mask_t is not None:
        squared_error = squared_error * loss_mask_t
        active_counts = loss_mask_t.sum(dim=(1, 2)).clamp_min(1.0)
        return squared_error.sum(dim=(1, 2)) / active_counts
    return squared_error.mean(dim=(1, 2))


def _repeat_batched_inputs(array: np.ndarray | None, n_reruns: int) -> np.ndarray | None:
    if array is None:
        return None
    if n_reruns == 1:
        return np.asarray(array)
    return np.repeat(np.asarray(array), n_reruns, axis=0)


def _resolve_sgd_batch_size(config: TestConfig, n_cells: int) -> int | None:
    if config.sgd_batch_size is None or config.sgd_batch_size == 0:
        return None
    return min(int(config.sgd_batch_size), int(n_cells))


def _attach_training_metadata(
    model: nn.Module,
    *,
    n_reruns: int,
    best_train_loss_per_model: np.ndarray,
    best_rerun_index_per_model: np.ndarray,
    train_loss_per_rerun: np.ndarray,
) -> None:
    model.training_metadata = {
        "n_reruns": int(n_reruns),
        "selection_loss": "training_reconstruction_loss",
        "best_train_loss_per_model": np.asarray(best_train_loss_per_model, dtype=np.float64),
        "best_rerun_index_per_model": np.asarray(best_rerun_index_per_model, dtype=np.int64),
        "train_loss_per_rerun": np.asarray(train_loss_per_rerun, dtype=np.float64),
    }


def get_training_metadata(model: nn.Module) -> dict[str, Any]:
    metadata = getattr(model, "training_metadata", None)
    if not isinstance(metadata, dict):
        return {
            "n_reruns": 1,
            "selection_loss": "training_reconstruction_loss",
            "best_train_loss_per_model": np.zeros(1, dtype=np.float64),
            "best_rerun_index_per_model": np.zeros(1, dtype=np.int64),
            "train_loss_per_rerun": np.zeros((1, 1), dtype=np.float64),
        }
    return {
        "n_reruns": int(metadata.get("n_reruns", 1)),
        "selection_loss": str(metadata.get("selection_loss", "training_reconstruction_loss")),
        "best_train_loss_per_model": np.asarray(metadata.get("best_train_loss_per_model", [0.0]), dtype=np.float64),
        "best_rerun_index_per_model": np.asarray(metadata.get("best_rerun_index_per_model", [0]), dtype=np.int64),
        "train_loss_per_rerun": np.asarray(metadata.get("train_loss_per_rerun", [[0.0]]), dtype=np.float64),
    }


def _compact_parallel_model(
    expanded_model: ParallelIsoDepthNet,
    *,
    selected_indices: np.ndarray,
    n_models: int,
    n_genes: int,
    latent_dim: int,
    device: torch.device,
) -> ParallelIsoDepthNet:
    compact_model = ParallelIsoDepthNet(n_models, n_genes, latent_dim=latent_dim).to(device)
    expanded_state = expanded_model.state_dict()
    compact_state = compact_model.state_dict()
    slot_indices = [int(index) for index in np.asarray(selected_indices, dtype=np.int64).tolist()]
    restored_state: dict[str, torch.Tensor] = {}
    for name, tensor in compact_state.items():
        source = expanded_state[name].detach().cpu()
        if source.ndim > 0 and source.shape[0] == expanded_model.decoder.weight.shape[0]:
            restored_state[name] = source[slot_indices].clone().to(device=device)
        else:
            restored_state[name] = source.clone().to(device=device)
    compact_model.load_state_dict(restored_state)
    return compact_model


def _compact_single_model(
    expanded_model: ParallelIsoDepthNet,
    *,
    selected_index: int,
    n_genes: int,
    latent_dim: int,
    device: torch.device,
) -> IsoDepthNet:
    compact_model = IsoDepthNet(n_genes, latent_dim=latent_dim).to(device)
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
        compact_model.decoder.weight.copy_(expanded_model.decoder.weight[selected_index])
        compact_model.decoder.bias.copy_(expanded_model.decoder.bias[selected_index])
    return compact_model


def train_batched_isodepth_model(
    s_batched: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    a_batched: Optional[np.ndarray] = None,
    loss_mask_batched: Optional[np.ndarray] = None,
    latent_dim: int = 1,
    model_label: str = "parallel isodepth batch",
) -> tuple[nn.Module, np.ndarray]:
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

    if a_batched is None:
        base_a_batched = np.repeat(np.asarray(A, dtype=np.float32)[None, :, :], n_models, axis=0)
    else:
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
    expanded_a_batched = _repeat_batched_inputs(base_a_batched, n_reruns)
    expanded_loss_mask = _repeat_batched_inputs(loss_mask_batched, n_reruns)

    total_models = int(expanded_s_batched.shape[0])
    s_batched_t = torch.tensor(expanded_s_batched, dtype=torch.float32, device=device)
    a_batched_t = torch.tensor(expanded_a_batched, dtype=torch.float32, device=device)
    loss_mask_t = _prepare_loss_mask(
        expanded_loss_mask,
        n_models=total_models,
        n_cells=n_cells,
        n_genes=n_genes,
        device=device,
    )

    model = ParallelIsoDepthNet(total_models, n_genes, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, foreach=False)
    sgd_batch_size = _resolve_sgd_batch_size(config, n_cells)
    active_mask_t = torch.ones(total_models, dtype=torch.float32, device=device)
    minibatch_generator = None
    if sgd_batch_size is not None:
        minibatch_generator = torch.Generator(device="cpu")
        minibatch_generator.manual_seed(config.seed)

    best_loss_per_model = np.full(total_models, np.inf, dtype=np.float64)
    patience_counter_per_model = np.zeros(total_models, dtype=np.int64)
    active_mask_np = np.ones(total_models, dtype=bool)
    best_state = _snapshot_parallel_model_state(model, total_models)
    iterator = tqdm(range(config.epochs), disable=not config.verbose)
    for epoch in iterator:
        active_mask_t.copy_(torch.from_numpy(active_mask_np.astype(np.float32)))
        active_count = float(active_mask_np.sum())
        if sgd_batch_size is None:
            optimizer.zero_grad()
            output = model(s_batched_t)
            loss_per_model = _compute_reconstruction_loss_per_model(output, a_batched_t, loss_mask_t)
            total_loss = (loss_per_model * active_mask_t).sum() / max(active_count, 1.0)
            total_loss.backward()
            optimizer.step()
        else:
            permutation = torch.randperm(n_cells, generator=minibatch_generator)
            for start in range(0, n_cells, sgd_batch_size):
                batch_indices = permutation[start : start + sgd_batch_size].to(device=device)
                batch_s = s_batched_t.index_select(1, batch_indices)
                batch_a = a_batched_t.index_select(1, batch_indices)
                batch_mask = None if loss_mask_t is None else loss_mask_t.index_select(1, batch_indices)

                optimizer.zero_grad()
                batch_output = model(batch_s)
                batch_loss_per_model = _compute_reconstruction_loss_per_model(batch_output, batch_a, batch_mask)
                batch_total_loss = (batch_loss_per_model * active_mask_t).sum() / max(active_count, 1.0)
                batch_total_loss.backward()
                optimizer.step()

        with torch.no_grad():
            output = model(s_batched_t)
            loss_per_model = _compute_reconstruction_loss_per_model(output, a_batched_t, loss_mask_t)
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
                    f"(all {total_models} batched models exhausted patience={config.patience})"
                )
            break

    _restore_parallel_model_snapshot(model, best_state, device)
    with torch.no_grad():
        expanded_predictions = model(s_batched_t).detach().cpu().numpy().astype(np.float32)
        final_loss_values = _compute_reconstruction_loss_per_model(
            model(s_batched_t),
            a_batched_t,
            loss_mask_t,
        ).detach().cpu().numpy().astype(np.float64)

    train_loss_per_rerun = final_loss_values.reshape(n_models, n_reruns)
    best_rerun_index_per_model = np.argmin(train_loss_per_rerun, axis=1).astype(np.int64)
    selected_slot_indices = (np.arange(n_models, dtype=np.int64) * n_reruns) + best_rerun_index_per_model
    predictions = expanded_predictions.reshape(n_models, n_reruns, n_cells, n_genes)[
        np.arange(n_models, dtype=np.int64),
        best_rerun_index_per_model,
    ]
    compact_model = _compact_parallel_model(
        model,
        selected_indices=selected_slot_indices,
        n_models=n_models,
        n_genes=n_genes,
        latent_dim=latent_dim,
        device=device,
    )
    best_train_loss_per_model = train_loss_per_rerun[np.arange(n_models, dtype=np.int64), best_rerun_index_per_model]
    _attach_training_metadata(
        compact_model,
        n_reruns=n_reruns,
        best_train_loss_per_model=best_train_loss_per_model,
        best_rerun_index_per_model=best_rerun_index_per_model,
        train_loss_per_rerun=train_loss_per_rerun,
    )
    return compact_model, np.asarray(predictions, dtype=np.float32)


def train_parallel_isodepth_model(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    s_batched: Optional[np.ndarray] = None,
    a_batched: Optional[np.ndarray] = None,
    loss_mask_batched: Optional[np.ndarray] = None,
    latent_dim: int = 1,
    model_label: Optional[str] = None,
) -> tuple[nn.Module, np.ndarray]:
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

    return train_batched_isodepth_model(
        s_batched,
        A,
        config,
        device=device,
        a_batched=a_batched,
        loss_mask_batched=loss_mask_batched,
        latent_dim=latent_dim,
        model_label=model_label,
    )


def train_isodepth_model(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    seed_offset: int = 0,
    latent_dim: int = 1,
    model_label: str = "model",
) -> tuple[nn.Module, np.ndarray]:
    device = device or resolve_device(config.device)
    if latent_dim <= 0:
        raise ValueError("latent_dim must be >= 1")

    effective_config = replace(config, seed=config.seed + seed_offset)
    parallel_model, predictions = train_batched_isodepth_model(
        np.asarray(S, dtype=np.float32)[None, :, :],
        A,
        effective_config,
        device=device,
        latent_dim=latent_dim,
        model_label=model_label,
    )
    metadata = get_training_metadata(parallel_model)
    model = _compact_single_model(
        parallel_model,
        selected_index=0,
        n_genes=A.shape[1],
        latent_dim=latent_dim,
        device=device,
    )
    _attach_training_metadata(
        model,
        n_reruns=metadata["n_reruns"],
        best_train_loss_per_model=metadata["best_train_loss_per_model"],
        best_rerun_index_per_model=metadata["best_rerun_index_per_model"],
        train_loss_per_rerun=metadata["train_loss_per_rerun"],
    )
    return model, np.asarray(predictions[0], dtype=np.float32)
