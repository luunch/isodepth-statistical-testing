from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.schemas import TestConfig
from methods.architectures import GastonMixNet, IsoDepthNet, ParallelIsoDepthNet


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


def _fit_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    patience: int,
    verbose: bool,
    model_label: str = "model",
) -> None:
    best_loss = float("inf")
    patience_counter = 0
    iterator = tqdm(range(epochs), disable=not verbose)
    for epoch in iterator:
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(
                    f"[early-stop] {model_label} stopped at epoch {epoch + 1} "
                    f"(best_loss={best_loss:.6g}, patience={patience})"
                )
            break


def evaluate_predictions(model: nn.Module, S: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        s_t = torch.tensor(S, dtype=torch.float32, device=device)
        preds = model(s_t).detach().cpu().numpy()
    return np.asarray(preds, dtype=np.float32)


def train_isodepth_model(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    seed_offset: int = 0,
    model_label: str = "model",
) -> tuple[nn.Module, np.ndarray]:
    device = device or resolve_device(config.device)
    _set_torch_seed(config.seed + seed_offset)

    model = IsoDepthNet(A.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    s_t = torch.tensor(S, dtype=torch.float32, device=device)
    a_t = torch.tensor(A, dtype=torch.float32, device=device)

    _fit_model(
        model,
        optimizer,
        criterion,
        s_t,
        a_t,
        epochs=config.epochs,
        patience=config.patience,
        verbose=config.verbose,
        model_label=model_label,
    )
    return model, evaluate_predictions(model, S, device)


def _reset_module_parameters(module: nn.Module) -> None:
    for child in module.children():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def train_frozen_decoder_model(
    trained_model: nn.Module,
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    seed_offset: int = 0,
    model_label: str = "model",
) -> tuple[nn.Module, np.ndarray]:
    device = device or resolve_device(config.device)
    _set_torch_seed(config.seed + seed_offset)

    model = deepcopy(trained_model).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    _reset_module_parameters(model.decoder)

    optimizer = optim.Adam(model.decoder.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    s_t = torch.tensor(S, dtype=torch.float32, device=device)
    a_t = torch.tensor(A, dtype=torch.float32, device=device)

    _fit_model(
        model,
        optimizer,
        criterion,
        s_t,
        a_t,
        epochs=config.epochs,
        patience=config.patience,
        verbose=config.verbose,
        model_label=model_label,
    )
    return model, evaluate_predictions(model, S, device)


def train_parallel_isodepth_model(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    s_batched: Optional[np.ndarray] = None,
    a_batched: Optional[np.ndarray] = None,
    loss_mask_batched: Optional[np.ndarray] = None,
    model_label: Optional[str] = None,
) -> tuple[nn.Module, np.ndarray]:
    device = device or resolve_device(config.device)
    if s_batched is None:
        n_models = config.n_perms + 1
        N = A.shape[0]
        s_t = torch.tensor(S, dtype=torch.float32, device=device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)

        s_batched_t = torch.zeros((n_models, N, 2), dtype=torch.float32, device=device)
        s_batched_t[0] = s_t
        for m in range(1, n_models):
            perm = torch.randperm(N, generator=generator)
            s_batched_t[m] = s_t[perm.to(device=device)]
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
        model_label=model_label,
    )


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


def train_batched_isodepth_model(
    s_batched: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    a_batched: Optional[np.ndarray] = None,
    loss_mask_batched: Optional[np.ndarray] = None,
    model_label: str = "parallel isodepth batch",
) -> tuple[nn.Module, np.ndarray]:
    device = device or resolve_device(config.device)
    _set_torch_seed(config.seed)

    s_batched_np = np.asarray(s_batched, dtype=np.float32)
    if s_batched_np.ndim != 3 or s_batched_np.shape[-1] != 2:
        raise ValueError(f"s_batched must have shape (M, N, 2), got {s_batched_np.shape}")

    N, G = A.shape
    if s_batched_np.shape[1] != N:
        raise ValueError(
            f"s_batched and A must have matching cell counts, got {s_batched_np.shape[1]} vs {N}"
        )

    n_models = s_batched_np.shape[0]
    s_batched_t = torch.tensor(s_batched_np, dtype=torch.float32, device=device)
    if a_batched is None:
        a_t = torch.tensor(A, dtype=torch.float32, device=device)
        a_batched_t = a_t.unsqueeze(0).expand(n_models, -1, -1)
    else:
        a_batched_np = np.asarray(a_batched, dtype=np.float32)
        if a_batched_np.ndim != 3:
            raise ValueError(f"a_batched must have shape (M, N, G), got {a_batched_np.shape}")
        if a_batched_np.shape != (n_models, N, G):
            raise ValueError(
                "a_batched must match (M, N, G) for the supplied s_batched and A, "
                f"got {a_batched_np.shape} vs {(n_models, N, G)}"
            )
        a_batched_t = torch.tensor(a_batched_np, dtype=torch.float32, device=device)

    loss_mask_t = _prepare_loss_mask(
        loss_mask_batched,
        n_models=n_models,
        n_cells=N,
        n_genes=G,
        device=device,
    )

    model = ParallelIsoDepthNet(n_models, G).to(device)
    # The foreach Adam path has been unstable on some CUDA and MPS environments.
    optimizer = optim.Adam(model.parameters(), lr=config.lr, foreach=False)
    criterion = nn.MSELoss(reduction="none")

    best_loss = float("inf")
    patience_counter = 0
    iterator = tqdm(range(config.epochs), disable=not config.verbose)
    for epoch in iterator:
        optimizer.zero_grad()
        output = model(s_batched_t)
        loss = criterion(output, a_batched_t)
        if loss_mask_t is not None:
            loss = loss * loss_mask_t
            active_counts = loss_mask_t.sum(dim=(1, 2)).clamp_min(1.0)
            loss_per_model = loss.sum(dim=(1, 2)) / active_counts
        else:
            loss_per_model = loss.mean(dim=(1, 2))
        total_loss = loss_per_model.sum()
        total_loss.backward()
        optimizer.step()

        current_loss = float(total_loss.item())
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            if config.verbose:
                print(
                    f"[early-stop] {model_label} stopped at epoch {epoch + 1} "
                    f"(best_loss={best_loss:.6g}, patience={config.patience})"
                )
            break

    with torch.no_grad():
        predictions = model(s_batched_t).detach().cpu().numpy()

    return model, np.asarray(predictions, dtype=np.float32)


def train_gaston_mix_model(
    S: np.ndarray,
    A: np.ndarray,
    config: TestConfig,
    *,
    device: Optional[torch.device] = None,
    seed_offset: int = 0,
    model_label: str = "model",
) -> tuple[nn.Module, np.ndarray, np.ndarray, list[np.ndarray]]:
    device = device or resolve_device(config.device)
    _set_torch_seed(config.seed + seed_offset)

    model = GastonMixNet(A.shape[1], P=config.n_experts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    s_t = torch.tensor(S, dtype=torch.float32, device=device)
    a_t = torch.tensor(A, dtype=torch.float32, device=device)

    best_loss = float("inf")
    patience_counter = 0
    iterator = tqdm(range(config.epochs), disable=not config.verbose)
    for epoch in iterator:
        optimizer.zero_grad()
        output, _, _ = model(s_t)
        loss = criterion(output, a_t)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            if config.verbose:
                print(
                    f"[early-stop] {model_label} stopped at epoch {epoch + 1} "
                    f"(best_loss={best_loss:.6g}, patience={config.patience})"
                )
            break

    model.eval()
    with torch.no_grad():
        predictions, gates, isodepths = model(s_t)

    return (
        model,
        predictions.detach().cpu().numpy().astype(np.float32),
        gates.detach().cpu().numpy().astype(np.float32),
        [d.detach().cpu().numpy().astype(np.float32) for d in isodepths],
    )
