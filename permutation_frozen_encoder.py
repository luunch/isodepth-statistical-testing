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
    reset_parameters,
    set_global_seed,
    train_with_early_stopping,
)

# 1. Setup
device = choose_device(prefer_mps=False)

# 2. Network Architecture
from models import IsoDepthNet

# 3. Specialized Training Functions
def train_full_model(S, A, epochs=5000, patience=50):
    set_global_seed(42)
    model = IsoDepthNet(A.shape[1]).to(device)
    S_t, A_t = torch.tensor(S).to(device), torch.tensor(A).to(device)
    train_with_early_stopping(model, S_t, A_t, epochs=epochs, lr=1e-3, patience=patience)
    return model

def train_frozen_decoder(model, S, A, epochs=5000, patience=50):
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Reset decoder weights for fresh start on permuted data
    reset_parameters(model.decoder)
    criterion = nn.MSELoss()
    S_t, A_t = torch.tensor(S).to(device), torch.tensor(A).to(device)
    train_with_early_stopping(
        model,
        S_t,
        A_t,
        epochs=epochs,
        lr=1e-3,
        patience=patience,
        params=list(model.decoder.parameters()),
    )

    with torch.no_grad():
        mse = criterion(model(S_t), A_t).item()
    n_total = A.shape[0] * A.shape[1]
    nll = gaussian_nll_from_mse(mse, n_total)
    return nll

# 4. Specialized Permutation Test
def frozen_permutation_test(S, A, M=20):
    print("Step 1: Training full model on true data...")
    model = train_full_model(S, A)

    with torch.no_grad():
        mse = nn.MSELoss()(model(torch.tensor(S).to(device)), torch.tensor(A).to(device)).item()
    n_total = A.shape[0] * A.shape[1]
    L_true = gaussian_nll_from_mse(mse, n_total)

    perm_losses = []
    print(f"Step 2: Running {M} permutations with GASTON encoder...")
    for _ in tqdm(range(M)):
        perm = np.random.permutation(S.shape[0])
        S_perm = S[perm]

        L_perm = train_frozen_decoder(model, S_perm, A)
        perm_losses.append(L_perm)

    p_value = empirical_p_value(np.array(perm_losses), L_true)
    return p_value, L_true, np.array(perm_losses), model

# 5. Visualization
def visualize_results(S, A, model, L_true, L_perm, title=""):
    plt.figure(figsize=(12, 5))
    gridsize = int(np.sqrt(S.shape[0]))
    with torch.no_grad():
        d_learned = model.encoder(torch.tensor(S).to(device)).cpu().numpy().flatten()
        d_learned = (d_learned - d_learned.min()) / (d_learned.max() - d_learned.min() + 1e-8)
        Z_d = d_learned.reshape(gridsize, gridsize)

    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(Z_d, cmap="viridis", extent=[0,1,0,1], origin="lower")
    ax1.contour(Z_d, levels=8, colors="white", linewidths=1, extent=[0,1,0,1], alpha=0.6)
    plt.colorbar(im, ax=ax1, label="Depth Value")
    ax1.set_title(f"{title}: Fixed Isodepth Map")

    ax2 = plt.subplot(1, 2, 2)
    sns.histplot(L_perm, ax=ax2, color="salmon", kde=False)
    p_val = empirical_p_value(L_perm, L_true)
    ax2.axvline(L_true, color="red", linestyle="--", label=f"True NLL (p={p_val:.3f})")
    ax2.set_title("GASTON Permutation Results")
    ax2.legend()
    plt.tight_layout()
    ensure_results_dir("results")
    plt.savefig(f"results/gaston_{title.lower()}_results.png")
    plt.close()

def visualize_gene_data(S, A, gene_indices=range(10), title=""):
    plt.figure(figsize=(20, 8))
    gridsize = int(np.sqrt(S.shape[0]))
    rows, cols = 2, 5
    for i, g_idx in enumerate(gene_indices):
        if i < rows * cols:
            ax = plt.subplot(rows, cols, i + 1)
            im = ax.imshow(A[:, g_idx].reshape(gridsize, gridsize), cmap="magma", extent=[0, 1, 0, 1], origin="lower")
            ax.set_title(f"Gene {g_idx}")
            ax.axis("off")
            if i % cols == (cols - 1):
                plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle(f"{title}: Spatial Expression of Genes 0-9")
    plt.tight_layout()
    ensure_results_dir("results")
    plt.savefig(f"results/gaston_{title.lower()}_genes.png")
    plt.close()

# 6. Execution
if __name__ == "__main__":
    N, G, M = 900, 20, 30

    simulator = SpatialDataSimulator(N=N, G=G, device=device)
    for control in ["Positive", "Negative"]:
        print(f"- {control.upper()} CONTROL ---")
        mode = "radial" if control == "Positive" else "noise"

        S, A = simulator.generate(mode=mode)

        p, L_true, L_perm, model = frozen_permutation_test(S, A, M)
        print(f"P-value: {p:.4f}")
        visualize_results(S, A, model, L_true, L_perm, title=control)
        visualize_gene_data(S, A, gene_indices=range(min(10, A.shape[1])), title=control)
