from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

TensorLossFn = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def reset_parameters(module: nn.Module) -> None:
    for layer in module.modules():
        if hasattr(layer, "reset_parameters") and layer is not module:
            layer.reset_parameters()


def train_with_early_stopping(
    model: nn.Module,
    s_t: torch.Tensor,
    a_t: torch.Tensor,
    *,
    epochs: int = 5000,
    lr: float = 1e-3,
    patience: int = 50,
    loss_fn: TensorLossFn | None = None,
    params: list[torch.nn.Parameter] | None = None,
    foreach: bool | None = None,
) -> float:
    if loss_fn is None:
        criterion = nn.MSELoss()

        def loss_fn(inner_model: nn.Module, inner_s: torch.Tensor, inner_a: torch.Tensor) -> torch.Tensor:
            return criterion(inner_model(inner_s), inner_a)

    optimizer_kwargs = {"lr": lr}
    if foreach is not None:
        optimizer_kwargs["foreach"] = foreach

    optimizer = optim.Adam(model.parameters() if params is None else params, **optimizer_kwargs)

    best_loss = float("inf")
    patience_counter = 0

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, s_t, a_t)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_loss
