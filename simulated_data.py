import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 1. Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Simulate Spatial Data

def simulate_data(N=400, G=20, sigma=0.5, positive=True):
    gridsize = int(np.sqrt(N))
    N = gridsize**2 # Ensures N matches the actual grid size
    
    # Grid construction
    S = np.array([[i, j] for i in range(1, gridsize + 1) for j in range(1, gridsize + 1)])
    x = S[:, 0] / gridsize
    y = S[:, 1] / gridsize

    if positive:
        d = x**2 + y**2
        polynomial_degree = 5 
        gene_coeff = np.random.randn(polynomial_degree + 1, G)
        H = np.vander(d, N=polynomial_degree+1) @ gene_coeff
        noise = sigma * np.random.randn(N, G)
        A = H + noise
    
    else:
        A = np.random.randn(N, G)

    return S.astype(np.float32), A.astype(np.float32)

# 3. Define GASTON-style Neural Network

class IsoDepthNet(nn.Module):
    """
    Neural network approximating: f_g(x,y) = h_g(d(x,y))
    Matches architecture from Appendix 1.1: 2 -> 20 -> 20 -> 1 -> 20 -> 20 -> G
    """
    def __init__(self, G):
        super(IsoDepthNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1)   # 1D isodepth bottleneck d_theta
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, G)   # h_theta'
        )

    def forward(self, x):
        d = self.encoder(x)
        out = self.decoder(d)
        return out

# 4. Train network and compute Negative Log-Likelihood

def train_model(S, A, epochs=300, lr=1e-3):
    """
    Trains network and returns final exact Gaussian negative log-likelihood.
    """
    S_tensor = torch.tensor(S).to(device)
    A_tensor = torch.tensor(A).to(device)

    N, G = A.shape

    model = IsoDepthNet(G).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train model
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(S_tensor)
        loss = criterion(output, A_tensor)
        loss.backward()
        optimizer.step()

    # Compute exact Gaussian Negative Log-Likelihood
    with torch.no_grad():
        output = model(S_tensor)
        mse = criterion(output, A_tensor).item()
        
    # NLL for Gaussian: (n/2)*log(2*pi*sigma^2) + (n/2) 
    # where n = N*G and empirical sigma^2 = MSE
    n_total = N * G
    nll = (n_total / 2) * np.log(2 * np.pi * mse) + (n_total / 2)

    return nll

def simulate_data_poly(N=400, G=20, sigma=0.5):
    gridsize = int(np.sqrt(N))

    coords = np.linspace(0, 1, gridsize)
    x, y = np.meshgrid(coords, coords)
    x_centered = x.ravel() - 0.5
    y_centered = y.ravel() - 0.5
    S = np.stack([x_centered, y_centered], axis=1)
    
    # 2. Define a clean distance metric (isodepth)
    d = np.sqrt(S[:,0]**2 + S[:,1]**2) 
    
    poly_degree = 3
    gene_coeff = np.random.uniform(-1, 1, (poly_degree + 1, G))
    
    # 4. Generate H
    H = np.zeros((N, G))
    for p in range(poly_degree + 1):
        H += np.outer(d**p, gene_coeff[p, :])
        
    noise = sigma * np.random.randn(N, G)
    A = H + noise
    return S.astype(np.float32), A.astype(np.float32)

# 5. Permutation Test

def permutation_test(S, A, M=30):
    L_true = train_model(S, A)

    perm_losses = []

    print(f"Running {M} permutations")
    for _ in tqdm(range(M)):
        # Permute gene expression values across cells
        perm = np.random.permutation(A.shape[0])
        A_perm = A[perm]

        L_perm = train_model(S, A_perm)
        perm_losses.append(L_perm)

    perm_losses = np.array(perm_losses)

    # extreme values under the null hypothesis are smaller than L_true.
    p_value = (1 + np.sum(perm_losses <= L_true)) / (M + 1)

    return p_value, L_true, perm_losses

# 6. Execution

if __name__ == "__main__":
    N = 400
    G = 20
    sigma = 0.5
    M = 100

    print("Positive Control")
    S_pos, A_pos = simulate_data(N, G, sigma, positive=True)
    p_pos, L_true_pos, L_perm_pos = permutation_test(S_pos, A_pos, M)

    print(f"\nTrue NLL: {L_true_pos:.4f}")
    print(f"Mean permuted NLL: {L_perm_pos.mean():.4f}")
    print(f"P-value: {p_pos:.4f}")

    print("Negative Control")
    S_neg, A_neg = simulate_data(N, G, sigma, positive=False)
    p_neg, L_true_neg, L_perm_neg = permutation_test(S_neg, A_neg, M)

    print(f"\nTrue NLL: {L_true_neg:.4f}")
    print(f"Mean permuted NLL: {L_perm_neg.mean():.4f}")
    print(f"P-value: {p_neg:.4f}")