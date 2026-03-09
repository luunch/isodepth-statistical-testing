import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from data_manager import SpatialDataSimulator
from models import IsoDepthNet

def calculate_laplacian_smoothness(Z):
    """
    Calculates the Dirichlet energy (Laplacian smoothness) of a 2D grid.
    Z: 2D numpy array of shape (gridsize, gridsize)
    Returns: The sum of squared differences between adjacent pixels.
    """
    diff_x = Z[1:, :] - Z[:-1, :]
    diff_y = Z[:, 1:] - Z[:, :-1]
    energy = np.sum(diff_x**2) + np.sum(diff_y**2)
    return energy

def train_and_get_isodepth(S, A, device, epochs=5000, patience=50):
    model = IsoDepthNet(A.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    S_t = torch.tensor(S, dtype=torch.float32).to(device)
    A_t = torch.tensor(A, dtype=torch.float32).to(device)
    
    best_loss = float('inf')
    patience_counter = 0

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(S_t), A_t)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
        
    model.eval()
    with torch.no_grad():
        d_learned = model.encoder(S_t).cpu().numpy().flatten()
        
    raw_range = d_learned.max() - d_learned.min()
        
    # Normalize to [0, 1] for scale-invariant comparison
    d_norm = (d_learned - d_learned.min()) / (raw_range + 1e-8)
    
    gridsize = int(np.sqrt(S.shape[0]))
    Z = d_norm.reshape(gridsize, gridsize)
    
    return Z, raw_range

def run_permuted_smoothness_trials(n_trials=20, N=400, G=10, epochs=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    simulator = SpatialDataSimulator(N=N, G=G, device=device)
    
    modes = ["radial", "noise"]
    
    original_smoothness = {}
    original_Z = {}
    
    permuted_smoothness = {mode: [] for mode in modes}
    
    # Dictionaries to track the most extreme permutations
    best_perm_Z = {mode: None for mode in modes}
    worst_perm_Z = {mode: None for mode in modes}
    best_perm_smooth = {mode: float('inf') for mode in modes}
    worst_perm_smooth = {mode: float('-inf') for mode in modes}
    
    print(f"--- Running {n_trials} permutation trials for Laplacian Smoothness ---")
    for mode in modes:
        print(f"\n--- Mode: {mode.capitalize()} ---")
        S, A = simulator.generate(mode=mode)
        
        # Original (just once)
        print("Training on original data...")
        Z_orig, raw_orig = train_and_get_isodepth(S, A, device, epochs=epochs)
        orig_smooth = calculate_laplacian_smoothness(Z_orig)
        original_smoothness[mode] = orig_smooth
        original_Z[mode] = Z_orig
        print(f"Original {mode.capitalize()} Smoothness: {orig_smooth:.4f} (Raw range: {raw_orig:.4f})")
        
        # Permuted (n_trials times)
        print(f"Running {n_trials} permutations...")
        for i in range(n_trials):
            perm = np.random.permutation(A.shape[0])
            A_perm = A[perm]
            
            Z_perm, raw_perm = train_and_get_isodepth(S, A_perm, device, epochs=epochs)
            perm_smooth = calculate_laplacian_smoothness(Z_perm)
            permuted_smoothness[mode].append(perm_smooth)
            print(f"  Trial {i+1}/{n_trials}: {perm_smooth:.4f} (Raw range: {raw_perm:.4f})")
            
            # Track best (smoothest = lowest) and worst (roughest = highest)
            if perm_smooth < best_perm_smooth[mode]:
                best_perm_smooth[mode] = perm_smooth
                best_perm_Z[mode] = Z_perm
            if perm_smooth > worst_perm_smooth[mode]:
                worst_perm_smooth[mode] = perm_smooth
                worst_perm_Z[mode] = Z_perm
            
    # --- Visualization 1: Distribution ---
    plt.figure(figsize=(10, 6))
    
    # Plotting Permuted Distributions
    data = []
    labels = []
    for mode in modes:
        data.extend(permuted_smoothness[mode])
        labels.extend([mode.capitalize()] * n_trials)
        
    sns.boxplot(x=labels, y=data, color="lightblue", showfliers=False)
    sns.stripplot(x=labels, y=data, color="blue", alpha=0.5, jitter=True)
    
    # Plotting Original Values as Red Stars
    for i, mode in enumerate(modes):
        plt.scatter(i, original_smoothness[mode], color="red", marker="*", s=200, label="Original" if i==0 else "", zorder=5)
        
    plt.title("Laplacian Smoothness: Original vs. Permuted Null Distribution")
    plt.ylabel("Smoothness Score (Lower is Smoother)")
    plt.xlabel("Underlying Dataset Structure")
    plt.legend()
    
    os.makedirs("results", exist_ok=True)
    save_path_dist = "results/permuted_smoothness_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path_dist)
    plt.close()
    
    # --- Visualization 2: Extreme Isodepths ---
    fig, axes = plt.subplots(len(modes), 3, figsize=(15, 5 * len(modes)))
    if len(modes) == 1:
        axes = [axes] # Handle single row properly if needed
    
    for idx, mode in enumerate(modes):
        ax_orig = axes[idx][0]
        im = ax_orig.imshow(original_Z[mode], cmap="viridis", extent=[0,1,0,1], origin="lower")
        ax_orig.set_title(f"{mode.capitalize()}: Original\nScore: {original_smoothness[mode]:.2f}")
        plt.colorbar(im, ax=ax_orig, fraction=0.046, pad=0.04)
        
        ax_best = axes[idx][1]
        im = ax_best.imshow(best_perm_Z[mode], cmap="viridis", extent=[0,1,0,1], origin="lower")
        ax_best.set_title(f"{mode.capitalize()}: Smoothest Perm\nScore: {best_perm_smooth[mode]:.2f}")
        plt.colorbar(im, ax=ax_best, fraction=0.046, pad=0.04)
        
        ax_worst = axes[idx][2]
        im = ax_worst.imshow(worst_perm_Z[mode], cmap="viridis", extent=[0,1,0,1], origin="lower")
        ax_worst.set_title(f"{mode.capitalize()}: Roughest Perm\nScore: {worst_perm_smooth[mode]:.2f}")
        plt.colorbar(im, ax=ax_worst, fraction=0.046, pad=0.04)
        
    plt.suptitle("Isodepth Visualization: Original vs Extremes of Permutation")
    plt.tight_layout()
    save_path_extremes = "results/permuted_extremes.png"
    plt.savefig(save_path_extremes)
    plt.close()
    
    print(f"\nAnalysis complete! Plots saved to '{save_path_dist}' and '{save_path_extremes}'")
    
    # Print Summary Statistics
    print("\n--- Summary Statistics ---")
    for mode in modes:
        orig = original_smoothness[mode]
        perm_mean = np.mean(permuted_smoothness[mode])
        perm_std = np.std(permuted_smoothness[mode])
        
        # Calculate a simple empirical p-value
        p_val = np.sum(np.array(permuted_smoothness[mode]) <= orig) / n_trials
        
        print(f"{mode.capitalize()}:")
        print(f"  Original Smoothness: {orig:.4f}")
        print(f"  Permuted Smoothness: {perm_mean:.4f} ± {perm_std:.4f}")
        print(f"  Empirical p-value (Perm <= Orig): {p_val:.4f}")

if __name__ == "__main__":
    run_permuted_smoothness_trials(n_trials=1000, N=400, G=10, epochs=500)