import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from data_manager import SpatialDataSimulator
from isodepth import (
    choose_device,
    empirical_p_value,
    ensure_results_dir,
    gaussian_nll_from_mse,
    train_with_early_stopping,
)

# 1. Setup
device = choose_device(prefer_mps=True)

# 2. Network Architecture
from models import IsoDepthNet

# 3. Training Function
def train_model(S, A, epochs=5000, lr=1e-3, patience=50):
    """Trains the full GASTON model (encoder + decoder) from scratch."""
    model = IsoDepthNet(A.shape[1]).to(device)
    criterion = nn.MSELoss()
    
    S_t = torch.tensor(S, dtype=torch.float32).to(device)
    A_t = torch.tensor(A, dtype=torch.float32).to(device)
    
    train_with_early_stopping(model, S_t, A_t, epochs=epochs, lr=lr, patience=patience)

    with torch.no_grad():
        mse = criterion(model(S_t), A_t).item()

    n_total = A.shape[0] * A.shape[1]
    nll = gaussian_nll_from_mse(mse, n_total)
    return nll, model

# 4. Full Retraining Permutation Test
def full_retraining_permutation_test(S, A, M=20, epochs=5000):
    print("Step 1: Training full model on true data...")
    L_true, true_model = train_model(S, A, epochs=epochs)
    
    perm_losses = []
    print(f"Step 2: Running {M} permutations with FULL retraining...")
    for i in tqdm(range(M)):
        # Permute spatial coordinates
        perm = np.random.permutation(S.shape[0])
        S_perm = S[perm]
        
        # Retrain EVERYTHING from scratch
        L_perm, _ = train_model(S_perm, A, epochs=epochs)
        perm_losses.append(L_perm)
        
    p_value = empirical_p_value(np.array(perm_losses), L_true)
    return p_value, L_true, np.array(perm_losses), true_model

# 5. Visualization
def visualize_results(S, A, model, L_true, L_perm, title=""):
    plt.figure(figsize=(12, 5))
    gridsize = int(np.sqrt(S.shape[0]))
    
    # 1. Visualize Learned Isodepth Map
    with torch.no_grad():
        S_t = torch.tensor(S, dtype=torch.float32).to(device)
        d_learned = model.encoder(S_t).cpu().numpy().flatten()
        # Normalize for visualization
        d_learned = (d_learned - d_learned.min()) / (d_learned.max() - d_learned.min() + 1e-8)
        Z_d = d_learned.reshape(gridsize, gridsize)

    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(Z_d, cmap="viridis", extent=[0,1,0,1], origin="lower")
    ax1.contour(Z_d, levels=8, colors="white", linewidths=1, extent=[0,1,0,1], alpha=0.6)
    plt.colorbar(im, ax=ax1, label="Depth Value")
    ax1.set_title(f"{title}: Learned Isodepth Map")

    # 2. Visualize Null Distribution
    ax2 = plt.subplot(1, 2, 2)
    sns.histplot(L_perm, ax=ax2, color="skyblue", kde=True)
    ax2.axvline(L_true, color="red", linestyle="--", label=f"True NLL (p={empirical_p_value(L_perm, L_true):.3f})")
    ax2.set_title("Null Distribution (Full Retraining)")
    ax2.legend()
    
    plt.tight_layout()
    ensure_results_dir("results")
    plt.savefig(f"results/full_retrain_{title.lower()}_results.png")
    plt.close()

# 6. Execution
if __name__ == "__main__":
    # Parameters
    N, G, M = 400, 10, 20  # Reduced N and G for faster execution during testing
    
    simulator = SpatialDataSimulator(N=N, G=G, device=device)
    
    for control in ["Positive", "Negative"]:
        print(f"\n--- {control.upper()} CONTROL ---")
        mode = "radial" if control == "Positive" else "noise"
        
        S, A = simulator.generate(mode=mode)
        
        p, L_true, L_perm, model = full_retraining_permutation_test(S, A, M=M)
        
        print(f"Final P-value: {p:.4f}")
        visualize_results(S, A, model, L_true, L_perm, title=control)
