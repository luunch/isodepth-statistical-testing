import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Simulate Spatial Data

def simulate_data(N=400, G=20, sigma=0.1, positive=True):
    gridsize = int(np.sqrt(N))
    N = gridsize**2 
    
    coords = np.linspace(0, 1, gridsize)
    x, y = np.meshgrid(coords, coords)
    S = np.stack([x.ravel(), y.ravel()], axis=1)

    if positive:
        d = np.sqrt((S[:, 0] - 0.5)**2 + (S[:, 1] - 0.5)**2)
        H = np.zeros((N, G))
        for g in range(G):
            coeffs = np.random.randn(4)
            H[:, g] = np.polyval(coeffs, d)
        A = H + sigma * np.random.randn(N, G)
    else:
        A = np.random.randn(N, G)

    # Standardize gene expression
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)

    return S.astype(np.float32), A.astype(np.float32)

# 3. Define Neural Network

class IsoDepthNet(nn.Module):
    def __init__(self, G):
        super(IsoDepthNet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 1))
        self.decoder = nn.Sequential(nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, G))

    def forward(self, x):
        return self.decoder(self.encoder(x))

# 4. Train network

def train_model(S, A, epochs=500, lr=1e-3):
    torch.manual_seed(42)
    S_tensor = torch.tensor(S).to(device)
    A_tensor = torch.tensor(A).to(device)
    model = IsoDepthNet(A.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(S_tensor), A_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mse = criterion(model(S_tensor), A_tensor).item()
        
    n_total = A.shape[0] * A.shape[1]
    nll = (n_total / 2) * np.log(2 * np.pi * mse + 1e-12) + (n_total / 2)
    return nll, model

# 5. Permutation Test

def permutation_test(S, A, M=20):
    L_true, model = train_model(S, A)
    perm_losses = []
    print(f"Running {M} permutations")
    for _ in tqdm(range(M)):
        perm = np.random.permutation(A.shape[0])
        L_perm, _ = train_model(S, A[perm])
        perm_losses.append(L_perm)
    
    perm_losses = np.array(perm_losses)
    p_value = (1 + np.sum(perm_losses <= L_true)) / (M + 1)
    return p_value, L_true, perm_losses, model

# 6. Visualization

def visualize_results(S, A, model, L_true, L_perm, title=""):
    plt.figure(figsize=(12, 5))
    gridsize = int(np.sqrt(S.shape[0]))
    with torch.no_grad():
        d_learned = model.encoder(torch.tensor(S).to(device)).cpu().numpy().flatten()
        d_learned = (d_learned - d_learned.min()) / (d_learned.max() - d_learned.min() + 1e-8)
        Z_d = d_learned.reshape(gridsize, gridsize)

    # 1. Learned Isodepth
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(Z_d, cmap="viridis", extent=[0, 1, 0, 1], origin="lower")
    ax1.contour(Z_d, levels=8, colors="white", linewidths=1, extent=[0, 1, 0, 1], alpha=0.6)
    plt.colorbar(im, ax=ax1, label="Depth Value")
    ax1.set_title(f"{title}: Learned Isodepths")
    
    # 2. NLL Distribution
    ax2 = plt.subplot(1, 2, 2)
    sns.histplot(L_perm, ax=ax2, color="skyblue", kde=False)
    p_val = (1 + np.sum(L_perm <= L_true)) / (len(L_perm) + 1)
    ax2.axvline(L_true, color="red", linestyle="--", label=f"True NLL (p={p_val:.3f})")
    ax2.set_title("Permutation Test Results")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.lower()}_results.png")
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
    plt.savefig(f"{title.lower()}_genes.png")
    plt.close()

# 7. Execution

if __name__ == "__main__":
    N, G, M = 900, 30, 100
    
    for control_type in ["Positive", "Negative"]:
        print(f"\n--- {control_type.upper()} CONTROL ---")
        is_pos = (control_type == "Positive")
        S, A = simulate_data(N, G, positive=is_pos)
        p, L_true, L_perm, model = permutation_test(S, A, M)
        print(f"P-value: {p:.4f}")
        
        visualize_results(S, A, model, L_true, L_perm, title=control_type)
        visualize_gene_data(S, A, gene_indices=range(10), title=control_type)
