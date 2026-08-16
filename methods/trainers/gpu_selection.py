from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from typing import TypeVar

import torch

T = TypeVar("T")

# Skip GPUs with less than this much free memory when auto-picking (MiB).
DEFAULT_MIN_FREE_MIB = 1024.0


def _parse_cuda_visible_devices() -> list[int] | None:
    """Physical GPU indices exposed to this process, or ``None`` if unset."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped or stripped.lower() in {"none", "no", "null"}:
        return []
    indices: list[int] = []
    for part in stripped.split(","):
        token = part.strip()
        if not token:
            continue
        indices.append(int(token))
    return indices


def query_gpu_free_memory_mib() -> dict[int, float] | None:
    """Return ``{physical_gpu_index: free_mib}`` via ``nvidia-smi``, or ``None`` if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not (result.stdout or "").strip():
        return None

    free_by_index: dict[int, float] = {}
    for line in (result.stdout or "").strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            free_by_index[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return free_by_index or None


def physical_to_torch_cuda_index(physical_index: int) -> int:
    visible = _parse_cuda_visible_devices()
    if visible is None:
        return int(physical_index)
    return int(visible.index(int(physical_index)))


def torch_cuda_index_to_physical(torch_index: int) -> int:
    visible = _parse_cuda_visible_devices()
    if visible is None:
        return int(torch_index)
    return int(visible[int(torch_index)])


def rank_cuda_device_indices(
    *,
    preferred_physical_index: int | None = None,
    min_free_mib: float = DEFAULT_MIN_FREE_MIB,
) -> list[int]:
    """
    Rank **torch** CUDA device indices by free memory (descending).

    When ``nvidia-smi`` is unavailable, returns ``[0]`` or the preferred index only.
    """
    if not torch.cuda.is_available():
        return []

    visible_physical = _parse_cuda_visible_devices()
    n_torch = torch.cuda.device_count()
    if n_torch <= 0:
        return []

    if visible_physical is not None and len(visible_physical) == 0:
        return []

    free_by_physical = query_gpu_free_memory_mib()

    candidates: list[tuple[int, float]] = []
    if visible_physical is None:
        physical_indices = list(range(n_torch))
    else:
        physical_indices = list(visible_physical)

    for physical in physical_indices:
        try:
            torch_idx = physical_to_torch_cuda_index(physical)
        except ValueError:
            continue
        if torch_idx < 0 or torch_idx >= n_torch:
            continue
        free_mib = float("inf")
        if free_by_physical is not None:
            free_mib = free_by_physical.get(int(physical), 0.0)
        candidates.append((torch_idx, free_mib))

    if not candidates:
        return [0] if n_torch > 0 else []

    usable = [(idx, free) for idx, free in candidates if free >= min_free_mib]
    pool = usable if usable else candidates

    pool.sort(key=lambda item: item[1], reverse=True)
    ranked = [idx for idx, _ in pool]

    if preferred_physical_index is not None:
        try:
            preferred_torch = physical_to_torch_cuda_index(preferred_physical_index)
        except ValueError:
            preferred_torch = None
        if preferred_torch is not None and preferred_torch in ranked:
            ranked = [preferred_torch] + [idx for idx in ranked if idx != preferred_torch]

    # Deduplicate while preserving order.
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in ranked:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def cuda_fallback_devices(
    device: torch.device,
    *,
    min_free_mib: float = DEFAULT_MIN_FREE_MIB,
) -> list[torch.device]:
    """Ordered CUDA devices to try: ``device`` first, then others by free memory."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return [device]

    preferred_physical: int | None = None
    if device.index is not None:
        preferred_physical = torch_cuda_index_to_physical(int(device.index))

    ranked = rank_cuda_device_indices(
        preferred_physical_index=preferred_physical,
        min_free_mib=min_free_mib,
    )
    if not ranked:
        return [device]

    devices = [torch.device(f"cuda:{idx}") for idx in ranked]
    if device.index is not None:
        primary = torch.device(f"cuda:{int(device.index)}")
        if primary not in devices:
            devices = [primary] + devices
        else:
            devices = [primary] + [d for d in devices if d != primary]
    return devices


def pick_best_cuda_device(*, min_free_mib: float = DEFAULT_MIN_FREE_MIB) -> torch.device:
    ranked = rank_cuda_device_indices(min_free_mib=min_free_mib)
    if ranked:
        return torch.device(f"cuda:{ranked[0]}")
    return torch.device("cuda")


def resolve_device(device: str, *, min_free_mib: float = DEFAULT_MIN_FREE_MIB) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return pick_best_cuda_device(min_free_mib=min_free_mib)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device but CUDA is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device but MPS is not available")

    if resolved.type == "cuda" and resolved.index is None:
        return pick_best_cuda_device(min_free_mib=min_free_mib)
    return resolved


def module_inference_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    """Device where ``module`` parameters live; used for forward passes after CPU offload."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback


def offload_module_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    """Move a module to CPU and release cached CUDA memory if possible."""
    module = module.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return module


def _clear_cuda_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def run_with_cuda_oom_retry(
    fn: Callable[[torch.device], T],
    device: torch.device,
    *,
    min_free_mib: float = DEFAULT_MIN_FREE_MIB,
    label: str = "operation",
) -> T:
    """Run ``fn(device)``; on CUDA OOM, retry on other GPUs ranked by free memory."""
    if device.type != "cuda":
        return fn(device)

    candidates = cuda_fallback_devices(device, min_free_mib=min_free_mib)
    last_error: torch.cuda.OutOfMemoryError | None = None
    for attempt, candidate in enumerate(candidates):
        if attempt > 0:
            print(
                f"CUDA OOM during {label}; retrying on {candidate} "
                f"({attempt + 1}/{len(candidates)})",
                file=sys.stderr,
            )
        try:
            return fn(candidate)
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            _clear_cuda_memory(candidate)
            continue

    if last_error is not None:
        raise last_error
    return fn(device)
