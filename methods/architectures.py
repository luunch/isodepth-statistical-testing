from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class IsoDepthNet(nn.Module):
    def __init__(self, G: int, latent_dim: int = 1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.latent_dim),
        )
        self.decoder = nn.Linear(self.latent_dim, G)

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
    def __init__(self, M: int, G: int, latent_dim: int = 1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            ParallelLinear(M, 2, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, self.latent_dim),
        )
        self.decoder = ParallelLinear(M, self.latent_dim, G)

    def forward(self, x):
        return self.decoder(self.encoder(x))
