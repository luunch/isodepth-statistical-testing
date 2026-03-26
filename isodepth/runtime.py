from __future__ import annotations

import os
import numpy as np
import torch


def choose_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int | None = None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_results_dir(path: str = "results") -> str:
    os.makedirs(path, exist_ok=True)
    return path
