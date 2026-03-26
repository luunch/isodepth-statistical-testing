import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kstest
from data_manager import SpatialDataSimulator
from isodepth import choose_device, empirical_p_value, ensure_results_dir, gaussian_nll_from_mse

# 1. Setup
device = choose_device(prefer_mps=True)
print(f"Using device: {device}")

# 2. Parallel Linear Layer
from models import ParallelIsoDepthNet

# 4. Parallel Training Logic
def run_parallel_permutation_test(S, A, M=100, epochs=5000, lr=1e-3, patience=50):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=use_foreach)
    criterion = nn.MSELoss(reduction='none') # Don't reduce so we can get loss per model
    
    best_loss = float('inf')
    patience_counter = 0

    print(f"Training {M} models in parallel for up to {epochs} epochs...")
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(S_batched)
        loss = criterion(output, A_batched) # [M, N, G]
        
        # Mean loss per model
        loss_per_model = loss.mean(dim=(1, 2)) # [M]
        total_loss = loss_per_model.sum()
        total_loss.backward()
        optimizer.step()
        
        current_loss = total_loss.item()
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
        
    with torch.no_grad():
        final_output = model(S_batched)
        final_mse = criterion(final_output, A_batched).mean(dim=(1, 2)).cpu().numpy()
        
    # Calculate NLL (Negative Log-Likelihood) for each model
    n_total = N * G
    nll = gaussian_nll_from_mse(final_mse, n_total)
    
    L_true = nll[0]
    L_perm = nll[1:]
    
    p_value = empirical_p_value(L_perm, L_true)
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
        p, _, _, _ = run_parallel_permutation_test(S, A, M=n_perms, epochs=5000)
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
    ensure_results_dir("results")
    plt.savefig("results/parallel_qq_plot.png")
    print(f"\nAnalysis complete. Plot saved to 'results/parallel_qq_plot.png'")
    print(f"Final Calibration Summary:")
    print(f"  - Kolmogorov-Smirnov test: p={ks_res.pvalue:.4f}")
    print(f"  - FPR at alpha=0.05: {fpr:.2%}")

if __name__ == "__main__":
    run_parallel_qq_analysis(n_trials=20, n_perms=100)
