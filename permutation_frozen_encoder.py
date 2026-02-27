import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup
#np.random.seed(0)
#torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Simulation
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
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)
    return S.astype(np.float32), A.astype(np.float32)

# 3. Network Architecture
class IsoDepthNet(nn.Module):
    def __init__(self, G):
        super(IsoDepthNet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 1))
        self.decoder = nn.Sequential(nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, G))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# 4. Specialized Training Functions
def train_full_model(S, A, epochs=500):
    torch.manual_seed(42)
    model = IsoDepthNet(A.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    S_t, A_t = torch.tensor(S).to(device), torch.tensor(A).to(device)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(S_t), A_t)
        loss.backward()
        optimizer.step()
    return model

def train_frozen_decoder(model, S, A, epochs=500):
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Reset decoder weights for fresh start on permuted data
    for layer in model.decoder:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    S_t, A_t = torch.tensor(S).to(device), torch.tensor(A).to(device)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(S_t), A_t)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        mse = criterion(model(S_t), A_t).item()
    n_total = A.shape[0] * A.shape[1]
    nll = (n_total / 2) * np.log(2 * np.pi * mse + 1e-12) + (n_total / 2)
    return nll

# 5. Specialized Permutation Test
def frozen_permutation_test(S, A, M=20):
    print("Step 1: Training full model on true data...")
    model = train_full_model(S, A)
    
    with torch.no_grad():
        mse = nn.MSELoss()(model(torch.tensor(S).to(device)), torch.tensor(A).to(device)).item()
    n_total = A.shape[0] * A.shape[1]
    L_true = (n_total / 2) * np.log(2 * np.pi * mse + 1e-12) + (n_total / 2)
    
    perm_losses = []
    print(f"Step 2: Running {M} permutations with FROZEN encoder...")
    for _ in tqdm(range(M)):
        perm = np.random.permutation(S.shape[0])
        S_perm = S[perm]
        
        L_perm = train_frozen_decoder(model, S_perm, A)
        perm_losses.append(L_perm)
        
    p_value = (1 + np.sum(np.array(perm_losses) <= L_true)) / (M + 1)
    return p_value, L_true, np.array(perm_losses), model

# 6. Visualization
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
    p_val = (1 + np.sum(L_perm <= L_true)) / (len(L_perm) + 1)
    ax2.axvline(L_true, color="red", linestyle="--", label=f"True NLL (p={p_val:.3f})")
    ax2.set_title("Frozen Encoder Permutation Results")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"frozen_{title.lower()}_results.png")
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
    plt.savefig(f"frozen_{title.lower()}_genes.png")
    plt.close()

# 7. Execution
if __name__ == "__main__":
    N, G, M = 900, 20, 30
    for control in ["Positive", "Negative"]:
        print(f"- {control.upper()} CONTROL ---")
        S, A = simulate_data(N, G, positive=(control=="Positive"))
        p, L_true, L_perm, model = frozen_permutation_test(S, A, M)
        print(f"P-value: {p:.4f}")
        visualize_results(S, A, model, L_true, L_perm, title=control)
        visualize_gene_data(S, A, gene_indices=range(10), title=control)
