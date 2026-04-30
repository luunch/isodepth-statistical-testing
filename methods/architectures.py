from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


SUPPORTED_DECODER_TYPES = {"linear", "nn"}


def _build_decoder(latent_dim: int, G: int, *, decoder_type: str) -> nn.Module:
    if decoder_type == "linear":
        return nn.Linear(latent_dim, G)
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
    def __init__(self, G: int, latent_dim: int = 1, decoder_type: str = "linear"):
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
