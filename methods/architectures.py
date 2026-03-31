from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class IsoDepthNet(nn.Module):
    def __init__(self, G: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, G),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class GastonMixNet(nn.Module):
    def __init__(self, G: int, P: int = 2):
        super().__init__()
        self.P = P
        self.router = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, P),
            nn.Softmax(dim=1),
        )
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, 20),
                    nn.ReLU(),
                    nn.Linear(20, 1),
                    nn.ReLU(),
                )
                for _ in range(P)
            ]
        )
        self.experts = nn.ModuleList([nn.Linear(1, G) for _ in range(P)])

    def forward(self, x):
        gates = self.router(x)
        output = 0
        isodepths = []
        for p in range(self.P):
            g_p = gates[:, p : p + 1]
            d_p = self.encoders[p](x)
            pred_p = self.experts[p](d_p)
            output += g_p * pred_p
            isodepths.append(d_p)
        return output, gates, isodepths


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
    def __init__(self, M: int, G: int):
        super().__init__()
        self.encoder = nn.Sequential(
            ParallelLinear(M, 2, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, 1),
        )
        self.decoder = nn.Sequential(
            ParallelLinear(M, 1, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, 20),
            nn.ReLU(),
            ParallelLinear(M, 20, G),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
