import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import kstest
from data_manager import SpatialDataSimulator

# 1. Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 2. Parallel Linear Layer
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

# 3. Parallel GASTON Network
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

# 4. Parallel Training Logic
def run_parallel_permutation_test(S, A, M=100, epochs=500, lr=1e-3):
    """
    Runs M permutations in parallel using a single batched neural network.
    Model 0 is the True data. Models 1 to M-1 are permuted data.
    """
    N, G = A.shape
    
    # Prepare Batched Data
    # S_batched: [M, N, 2]
    # A_batched: [M, N, G]
    S_t = torch.tensor(S, dtype=torch.float32).to(device)
    A_t = torch.tensor(A, dtype=torch.float32).to(device)
    
    # Construct permutations: model 0 is identity, others are random
    S_batched = torch.zeros((M, N, 2), device=device)
    S_batched[0] = S_t # True mapping
    for m in range(1, M):
        perm = torch.randperm(N)
        S_batched[m] = S_t[perm]
        
    A_batched = A_t.unsqueeze(0).expand(M, -1, -1) # A is same for all, S is permuted
    
    # Initialize Parallel Model
    model = ParallelIsoDepthNet(M, G).to(device)
    
    # Use foreach=True for a significant speedup in optimizer updates
    # (Generally safe on CPU/CUDA, but can be problematic on MPS)
    use_foreach = (device.type != 'mps')
    optimizer = optim.Adam(model.parameters(), lr=lr, foreach=use_foreach)
    criterion = nn.MSELoss(reduction='none') # Don't reduce so we can get loss per model
    
    print(f"Training {M} models in parallel for {epochs} epochs...")
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(S_batched)
        loss = criterion(output, A_batched) # [M, N, G]
        
        # Mean loss per model
        loss_per_model = loss.mean(dim=(1, 2)) # [M]
        total_loss = loss_per_model.sum()
        total_loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_output = model(S_batched)
        final_mse = criterion(final_output, A_batched).mean(dim=(1, 2)).cpu().numpy()
        
    # Calculate NLL (Negative Log-Likelihood) for each model
    n_total = N * G
    nll = (n_total / 2) * np.log(2 * np.pi * final_mse + 1e-12) + (n_total / 2)
    
    L_true = nll[0]
    L_perm = nll[1:]
    
    p_value = (1 + np.sum(L_perm <= L_true)) / M
    return p_value, L_true, L_perm, model

# 5. Q-Q Plot Analysis
def run_parallel_qq_analysis(n_trials=20, n_perms=100, N=400, G=10):
    simulator = SpatialDataSimulator(N, G, device=device)
    
    p_values = []
    print(f"--- STARTING PARALLEL Q-Q ANALYSIS ({n_trials} Trials, {n_perms} Permutations each) ---")
    
    for i in range(n_trials):
        print(f"\nTrial {i+1}/{n_trials}")
        S, A = simulator.generate(mode="noise")
        
        # We use a slightly higher M for the test to ensure good resolution
        p, _, _, _ = run_parallel_permutation_test(S, A, M=n_perms, epochs=400)
        p_values.append(p)
        print(f"Trial {i+1} P-value: {p:.4f}")

    # Statistical Test
    ks_res = kstest(p_values, 'uniform')
    fpr = np.sum(np.array(p_values) <= 0.05) / n_trials
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(p_values, bins=10, range=(0, 1), density=True, alpha=0.6, color='emerald', edgecolor='black')
    plt.axhline(1, color='red', linestyle='--')
    plt.title(f"P-value Density\nKS p={ks_res.pvalue:.4f} | FPR={fpr:.2%}")
    plt.xlabel("P-value")
    
    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, 1, n_trials), np.sort(p_values), 'o-', markersize=4)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f"Q-Q Plot ({n_perms} Permutations)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Observed P-values")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("parallel_qq_plot.png")
    print(f"\nAnalysis complete. Plot saved to 'parallel_qq_plot.png'")
    print(f"Final Calibration Summary:")
    print(f"  - Kolmogorov-Smirnov test: p={ks_res.pvalue:.4f}")
    print(f"  - FPR at alpha=0.05: {fpr:.2%}")

if __name__ == "__main__":
    run_parallel_qq_analysis(n_trials=20, n_perms=100)
