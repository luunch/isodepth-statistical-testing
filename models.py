import numpy as np
import torch
import torch.nn as nn

class IsoDepthNet(nn.Module):
    def __init__(self, G):
        super(IsoDepthNet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 1))
        self.decoder = nn.Sequential(nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, G))
    def forward(self, x):
        return self.decoder(self.encoder(x))

class GastonMixNet(nn.Module):
    def __init__(self, G, P=2):
        super(GastonMixNet, self).__init__()
        self.P = P
        
        # Routing Network: R^2 -> [0,1]^P (Softmax ensures sum to 1)
        # 2 hidden layers of size 20
        self.router = nn.Sequential(
            nn.Linear(2, 20), nn.ReLU(), 
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, P), nn.Softmax(dim=1)
        )
        
        # Local Encoders: P distinct networks R^2 -> R^1
        # Each has 1 hidden layer of size 20 and a final ReLU for the isodepth
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 20), nn.ReLU(), 
                nn.Linear(20, 1), nn.ReLU()
            ) 
            for _ in range(P)
        ])
        
        # Linear Experts (Decoder): P distinct linear layers (d -> Genes)
        self.experts = nn.ModuleList([
            nn.Linear(1, G) for _ in range(P)
        ])

    def forward(self, x):
        gates = self.router(x) # Shape: [N, P]
        
        output = 0
        isodepths = []
        # Continuous routing: Sum across all P experts
        for p in range(self.P):
            g_p = gates[:, p:p+1]           # Gate prob for expert p
            d_p = self.encoders[p](x)       # Isodepth for expert p
            pred_p = self.experts[p](d_p)   # Linear prediction
            
            output += g_p * pred_p          # Blend predictions
            isodepths.append(d_p)
            
        return output, gates, isodepths

class ParallelLinear(nn.Module):
    """
    A batched linear layer that maintains M independent weight matrices.
    Input shape:  [M, N, in_features]
    Output shape: [M, N, out_features]
    """
    def __init__(self, M, in_f, out_f):
        super().__init__()
        self.M = M
        self.weight = nn.Parameter(torch.empty(M, out_f, in_f))
        self.bias = nn.Parameter(torch.empty(M, out_f))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize each of the M models using standard Kaiming/He initialization
        for m in range(self.M):
            nn.init.kaiming_uniform_(self.weight[m], a=np.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[m])
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias[m], -bound, bound)

    def forward(self, x):
        # x: [M, N, in_f]
        # weights: [M, out_f, in_f]
        # We use bmm (Batch Matrix Multiplication) to apply M different weights to M different data blocks
        return torch.bmm(x, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)

class ParallelIsoDepthNet(nn.Module):
    def __init__(self, M, G):
        super().__init__()
        # M is the number of parallel models (1 True + M-1 Permutations)
        self.encoder = nn.Sequential(
            ParallelLinear(M, 2, 20), nn.ReLU(),
            ParallelLinear(M, 20, 20), nn.ReLU(),
            ParallelLinear(M, 20, 1)
        )
        self.decoder = nn.Sequential(
            ParallelLinear(M, 1, 20), nn.ReLU(),
            ParallelLinear(M, 20, 20), nn.ReLU(),
            ParallelLinear(M, 20, G)
        )

    def forward(self, x):
        # x shape: [M, N, 2]
        isodepth = self.encoder(x)
        output = self.decoder(isodepth)
        return output
