from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


SUPPORTED_DECODER_TYPES = {"linear", "nn", "quadratic"}


class QuadraticDecoder(nn.Module):
    """Polynomial degree-2 decoder: output = Linear([z; z²]) + b."""

    def __init__(self, latent_dim: int, G: int):
        super().__init__()
        self.linear = nn.Linear(2 * latent_dim, G)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([z, z ** 2], dim=-1))


class ParallelQuadraticDecoder(nn.Module):
    """Batched (M parallel) polynomial degree-2 decoder."""

    def __init__(self, M: int, latent_dim: int, G: int):
        super().__init__()
        self.linear = ParallelLinear(M, 2 * latent_dim, G)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([z, z ** 2], dim=-1))


def _build_decoder(latent_dim: int, G: int, *, decoder_type: str) -> nn.Module:
    if decoder_type == "linear":
        return nn.Linear(latent_dim, G)
    if decoder_type == "quadratic":
        return QuadraticDecoder(latent_dim, G)
    if decoder_type == "nn":
        return nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, G),
        )
    raise ValueError(
        f"Unsupported decoder_type '{decoder_type}'. Expected one of {sorted(SUPPORTED_DECODER_TYPES)}"
    )


def _build_parallel_decoder(M: int, latent_dim: int, G: int, *, decoder_type: str) -> nn.Module:
    if decoder_type == "linear":
        return ParallelLinear(M, latent_dim, G)
    if decoder_type == "quadratic":
        return ParallelQuadraticDecoder(M, latent_dim, G)
    if decoder_type == "nn":
        return nn.Sequential(
            ParallelLinear(M, latent_dim, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, G),
        )
    raise ValueError(
        f"Unsupported decoder_type '{decoder_type}'. Expected one of {sorted(SUPPORTED_DECODER_TYPES)}"
    )


class IsoDepthNet(nn.Module):
    def __init__(self, G: int, latent_dim: int = 1, decoder_type: str = "nn"):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.decoder_type = str(decoder_type)
        self.encoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.latent_dim),
        )
        self.decoder = _build_decoder(self.latent_dim, G, decoder_type=self.decoder_type)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ParallelLinear(nn.Module):
    def __init__(self, M: int, in_f: int, out_f: int):
        super().__init__()
        self.M = M
        self.weight = nn.Parameter(torch.empty(M, out_f, in_f))
        self.bias = nn.Parameter(torch.empty(M, out_f))
        self.reset_parameters()

    def reset_parameters(self):
        for m in range(self.M):
            nn.init.kaiming_uniform_(self.weight[m], a=np.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[m])
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias[m], -bound, bound)

    def forward(self, x):
        return torch.bmm(x, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)


class ParallelIsoDepthNet(nn.Module):
    def __init__(self, M: int, G: int, latent_dim: int = 1, decoder_type: str = "nn"):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.decoder_type = str(decoder_type)
        self.encoder = nn.Sequential(
            ParallelLinear(M, 2, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, self.latent_dim),
        )
        self.decoder = _build_parallel_decoder(M, self.latent_dim, G, decoder_type=self.decoder_type)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MidlineLatent(nn.Module):
    """Fixed 1D depth per layout: c = median(x), d(x, y) = |x - c|, z-scored across cells.

    Expects ``x`` as column 0 of spatial coordinates. One median per batch row ``m`` (parallel models).
    """

    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = 1

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s[..., 0]
        med = x.median(dim=1).values.unsqueeze(1)
        depth = (x - med).abs()
        mu = depth.mean(dim=1, keepdim=True)
        sigma = depth.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
        return ((depth - mu) / sigma).unsqueeze(-1)


class MidlineLatentSingle(nn.Module):
    """Same as ``MidlineLatent`` for a single layout of shape ``(N, 2)``."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = 1

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s[:, 0]
        med = x.median()
        depth = (x - med).abs()
        mu = depth.mean()
        sigma = depth.std(unbiased=False)
        if float(sigma) < 1e-8:
            return torch.zeros((s.shape[0], 1), device=s.device, dtype=s.dtype)
        return ((depth - mu) / sigma).unsqueeze(-1)


class HybridMidlineLatent(nn.Module):
    """First ``slot_split`` batch rows: fixed midline depth |x-median(x)|; remaining rows: learned parallel encoder."""

    def __init__(self, slot_split: int, M: int, latent_dim: int = 1):
        super().__init__()
        self.slot_split = int(slot_split)
        self.M = int(M)
        self.latent_dim = int(latent_dim)
        if not (1 <= self.slot_split <= self.M):
            raise ValueError(f"Require 1 <= slot_split <= M; got slot_split={self.slot_split}, M={self.M}")
        self.midline = MidlineLatent()
        p_enc = self.M - self.slot_split
        self.encoder_perm = None
        if p_enc > 0:
            self.encoder_perm = nn.Sequential(
                ParallelLinear(p_enc, 2, 20),
                nn.ReLU(),
                ParallelLinear(p_enc, 20, 20),
                nn.ReLU(),
                ParallelLinear(p_enc, 20, self.latent_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, n, _ = x.shape
        if m != self.M:
            raise ValueError(f"Expected batch dimension {self.M}, got {m}")
        out = torch.zeros(m, n, self.latent_dim, device=x.device, dtype=x.dtype)
        out[: self.slot_split] = self.midline(x[: self.slot_split])
        if self.encoder_perm is not None:
            out[self.slot_split :] = self.encoder_perm(x[self.slot_split :])
        return out


class HybridMidlineParallelNet(nn.Module):
    """Covariate ``midline``: true-layout instances use fixed midline bottleneck (decoder-only path); other parallel
    instances use a full learned encoder + decoder. Typical expanded batch: ``slot_split = n_reruns`` so every rerun
    of the true layout is midline; permutation slots train encoder+decoder."""

    def __init__(self, M: int, G: int, *, slot_split: int, latent_dim: int = 1, decoder_type: str = "nn"):
        super().__init__()
        if int(latent_dim) != 1:
            raise ValueError("HybridMidlineParallelNet currently requires latent_dim=1.")
        self.M = int(M)
        self.slot_split = int(slot_split)
        self.latent_dim = 1
        self.decoder_type = str(decoder_type)
        self.encoder = HybridMidlineLatent(self.slot_split, self.M, latent_dim=1)
        self.decoder = _build_parallel_decoder(self.M, self.latent_dim, G, decoder_type=self.decoder_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class CellTypeIsoDepthNet(nn.Module):
    """Shared encoder + per-cell-type decoders for a single layout.

    The encoder maps (x, y) -> latent_dim (shared across all cell types).
    Each cell type c has its own decoder h_c: latent_dim -> G.
    Forward requires ``cell_type_indices`` of shape (N,) with values in [0, C).
    """

    def __init__(self, n_cell_types: int, G: int, latent_dim: int = 1, decoder_type: str = "nn"):
        super().__init__()
        self.n_cell_types = int(n_cell_types)
        self.G = int(G)
        self.latent_dim = int(latent_dim)
        self.decoder_type = str(decoder_type)
        self.encoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.latent_dim),
        )
        self.decoders = nn.ModuleList([
            _build_decoder(self.latent_dim, G, decoder_type=self.decoder_type)
            for _ in range(self.n_cell_types)
        ])

    def forward(self, x: torch.Tensor, cell_type_indices: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        N, G = x.shape[0], self.G
        output = torch.zeros(N, G, device=x.device, dtype=x.dtype)
        for c in range(self.n_cell_types):
            mask = cell_type_indices == c
            if mask.any():
                output[mask] = self.decoders[c](latent[mask])
        return output


class ParallelCellTypeIsoDepthNet(nn.Module):
    """Batched (M parallel models) variant with shared encoder + per-cell-type decoders.

    Encoder: ParallelLinear stack mapping (M, N, 2) -> (M, N, latent_dim) shared across cell types.
    Decoders: C decoder heads, each a ParallelLinear stack mapping (M, N_c, latent_dim) -> (M, N_c, G).
    Forward requires ``cell_type_indices`` of shape (N,) with values in [0, C).

    Optimization: on first forward call, pre-computes a sorted-by-type order and offset table
    so subsequent calls use contiguous slicing instead of per-type boolean masks.
    """

    def __init__(self, M: int, n_cell_types: int, G: int, latent_dim: int = 1, decoder_type: str = "nn"):
        super().__init__()
        self.M = int(M)
        self.n_cell_types = int(n_cell_types)
        self.latent_dim = int(latent_dim)
        self.decoder_type = str(decoder_type)
        self.G = int(G)
        self.encoder = nn.Sequential(
            ParallelLinear(M, 2, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, self.latent_dim),
        )
        self.decoders = nn.ModuleList([
            _build_parallel_decoder(M, self.latent_dim, G, decoder_type=self.decoder_type)
            for _ in range(self.n_cell_types)
        ])
        self._sort_idx: torch.Tensor | None = None
        self._unsort_idx: torch.Tensor | None = None
        self._type_offsets: list[tuple[int, int]] | None = None

    def _build_routing_cache(self, cell_type_indices: torch.Tensor) -> None:
        """Pre-compute sorted order and per-type slice boundaries."""
        ct_np = cell_type_indices.detach().cpu().numpy()
        sort_idx = np.argsort(ct_np, kind="stable")
        unsort_idx = np.empty_like(sort_idx)
        unsort_idx[sort_idx] = np.arange(len(sort_idx))

        offsets: list[tuple[int, int]] = []
        pos = 0
        sorted_ct = ct_np[sort_idx]
        for c in range(self.n_cell_types):
            count = int(np.sum(sorted_ct == c))
            offsets.append((pos, pos + count))
            pos += count

        device = cell_type_indices.device
        self._sort_idx = torch.from_numpy(sort_idx).long().to(device)
        self._unsort_idx = torch.from_numpy(unsort_idx).long().to(device)
        self._type_offsets = offsets

    def forward(self, x: torch.Tensor, cell_type_indices: torch.Tensor) -> torch.Tensor:
        if self._sort_idx is None or self._sort_idx.device != x.device:
            self._build_routing_cache(cell_type_indices)

        latent = self.encoder(x)
        M, N, _ = x.shape

        sorted_latent = latent[:, self._sort_idx, :]

        sorted_output = torch.empty(M, N, self.G, device=x.device, dtype=x.dtype)
        for c, (start, end) in enumerate(self._type_offsets):
            if start == end:
                continue
            sorted_output[:, start:end, :] = self.decoders[c](sorted_latent[:, start:end, :])

        output = sorted_output[:, self._unsort_idx, :]
        return output


class DecoderOnlyNet(nn.Module):
    """Single-layout decoder-only model with midline fixed depth."""

    def __init__(self, G: int, latent_dim: int = 1, decoder_type: str = "linear"):
        super().__init__()
        if int(latent_dim) != 1:
            raise ValueError("DecoderOnlyNet (midline) requires latent_dim=1.")
        self.latent_dim = 1
        self.decoder_type = str(decoder_type)
        self.encoder = MidlineLatentSingle()
        self.decoder = _build_decoder(self.latent_dim, G, decoder_type=self.decoder_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class FixedLatentSingle(nn.Module):
    """Fixed per-cell latent values read from an obs column.

    The stored ``latent_values`` buffer (shape ``(N, 1)``) is returned unchanged for
    every forward call; the spatial coordinate input is ignored.  This mirrors
    ``MidlineLatentSingle`` but uses data-driven values instead of computing
    ``|x - median(x)|`` from coordinates.
    """

    def __init__(self, values: np.ndarray) -> None:
        super().__init__()
        v = torch.tensor(np.asarray(values, dtype=np.float32).reshape(-1, 1))
        self.register_buffer("latent_values", v)
        self.latent_dim = 1

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.latent_values


class DecoderOnlyNetFixed(nn.Module):
    """Decoder-only model whose encoder is a fixed obs-column latent (``FixedLatentSingle``).

    Suitable for any pre-computed per-cell covariate stored in ``adata.obs``.  Only the
    decoder weights are updated during training; the latent values are frozen buffers.
    """

    def __init__(
        self,
        G: int,
        values: np.ndarray,
        latent_dim: int = 1,
        decoder_type: str = "nn",
    ) -> None:
        super().__init__()
        if int(latent_dim) != 1:
            raise ValueError("DecoderOnlyNetFixed requires latent_dim=1.")
        self.latent_dim = 1
        self.decoder_type = str(decoder_type)
        self.encoder = FixedLatentSingle(values)
        self.decoder = _build_decoder(self.latent_dim, G, decoder_type=self.decoder_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ParallelDecoderOnlyNetFixed(nn.Module):
    """``n_reruns`` parallel decoder-only models sharing one fixed obs-column latent."""

    def __init__(
        self,
        M: int,
        G: int,
        values: np.ndarray,
        *,
        latent_dim: int = 1,
        decoder_type: str = "nn",
    ) -> None:
        super().__init__()
        if int(latent_dim) != 1:
            raise ValueError("ParallelDecoderOnlyNetFixed requires latent_dim=1.")
        self.M = int(M)
        self.latent_dim = 1
        self.decoder_type = str(decoder_type)
        self.encoder = FixedLatentSingle(values)
        self.decoder = _build_parallel_decoder(self.M, self.latent_dim, G, decoder_type=self.decoder_type)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        del x
        latent = self.encoder.latent_values
        batched_latent = latent.unsqueeze(0).expand(self.M, -1, -1)
        return self.decoder(batched_latent)
