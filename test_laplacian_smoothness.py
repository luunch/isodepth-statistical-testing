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
        
    # Normalize to [0, 1] for scale-invariant comparison
    d_norm = (d_learned - d_learned.min()) / (d_learned.max() - d_learned.min() + 1e-8)
    
    gridsize = int(np.sqrt(S.shape[0]))
    Z = d_norm.reshape(gridsize, gridsize)
    
    return Z

def run_smoothness_trials(n_trials=20, N=400, G=10, epochs=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    simulator = SpatialDataSimulator(N=N, G=G, device=device)
    
    modes = ["radial", "noise"]
    results = {mode: [] for mode in modes}
    
    print(f"--- Running {n_trials} trials for Laplacian Smoothness ---")
    for i in range(n_trials):
        print(f"\nTrial {i+1}/{n_trials}")
        for mode in modes:
            S, A = simulator.generate(mode=mode)
            Z = train_and_get_isodepth(S, A, device, epochs=epochs)
            smoothness = calculate_laplacian_smoothness(Z)
            results[mode].append(smoothness)
            print(f"  {mode.capitalize()} Smoothness: {smoothness:.4f}")
            
    # Visualization
    plt.figure(figsize=(10, 6))
    
    data = []
    labels = []
    for mode in modes:
        data.extend(results[mode])
        labels.extend([mode.capitalize()] * n_trials)
        
    sns.boxplot(x=labels, y=data, hue=labels, palette="Set2", showfliers=False, legend=False)
    sns.stripplot(x=labels, y=data, color=".25", size=6, alpha=0.7)
    
    plt.title("Laplacian Smoothness of Learned Isodepth Maps")
    plt.ylabel("Smoothness Score (Lower is Smoother)")
    plt.xlabel("Underlying Dataset Structure")
    
    os.makedirs("results", exist_ok=True)
    save_path = "results/laplacian_smoothness_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"\nAnalysis complete! Plot saved to '{save_path}'")
    
    # Print Summary Statistics
    print("\n--- Summary Statistics (Mean ± Std) ---")
    for mode in modes:
        mean_val = np.mean(results[mode])
        std_val = np.std(results[mode])
        print(f"{mode.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    run_smoothness_trials(n_trials=20, N=400, G=10, epochs=5000)