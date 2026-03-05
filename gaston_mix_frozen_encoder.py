import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_manager import SpatialDataSimulator

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Network Architecture: GASTON-MIX
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

# 3. Phase 2: Train the Deep Learning Mapmaker
def train_full_model(S, A, P=3, epochs=500):
    torch.manual_seed(42)
    model = GastonMixNet(A.shape[1], P=P).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    S_t, A_t = torch.tensor(S).to(device), torch.tensor(A).to(device)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        output, _, _ = model(S_t)
        loss = criterion(output, A_t)
        loss.backward()
        optimizer.step()
        
    return model

# 4. The Closed-Form Permutation Test
def closed_form_permutation_test(S, A, P=3, M=1000):
    print("Step 1: Training GASTON-MIX on true data...")
    model = train_full_model(S, A, P=P)
    model.eval()
    
    S_t = torch.tensor(S).to(device)
    # Center A to ensure intercept-only model is a baseline
    A_t = torch.tensor(A).to(device)
    A_t = A_t - A_t.mean(dim=0, keepdim=True)
    
    with torch.no_grad():
        _, gates, isodepths = model(S_t)
    
    print("Step 2: Constructing Design Matrix (X) and Projection Matrix (I - H)...")
    # Build X matrix of size [N, 2P + 1]
    X_cols = []
    # 1. Global intercept
    X_cols.append(torch.ones((S_t.shape[0], 1), device=device))
    
    for p in range(P):
        g_p = gates[:, p:p+1]  # [N, 1]
        d_p = isodepths[p]     # [N, 1]
        # 2. Gated Isodepth
        X_cols.append(g_p * d_p)  
        # 3. Gated Intercept (already partially covered by global if P=1, but distinct for mixture)
        X_cols.append(g_p)        
    
    X = torch.cat(X_cols, dim=1)  # Shape: [N, 2P + 1]
    
    # Use QR decomposition for numerical stability instead of explicit inversion
    # H = Q Q^T where Q is the orthogonal basis for the column space of X
    Q, _ = torch.linalg.qr(X)
    
    print(f"Step 3: Running {M} lightning-fast closed-form permutations...")
    
    # Calculate True Loss: L = ||A - Q(Q^T A)||_F^2
    # This is equivalent to ||(I - QQ^T)A||_F^2
    QtA = torch.matmul(Q.T, A_t)
    proj_A = torch.matmul(Q, QtA)
    residuals_true = A_t - proj_A
    L_true = torch.sum(residuals_true ** 2).item()
    
    perm_losses = np.zeros(M)
    for m in tqdm(range(M)):
        # Shuffle spatial indices
        perm_idx = torch.randperm(A_t.shape[0])
        A_perm = A_t[perm_idx]
        
        # Calculate permuted loss analytically
        QtA_perm = torch.matmul(Q.T, A_perm)
        proj_A_perm = torch.matmul(Q, QtA_perm)
        residuals_perm = A_perm - proj_A_perm
        L_perm = torch.sum(residuals_perm ** 2).item()
        perm_losses[m] = L_perm
        
    p_value = (1 + np.sum(perm_losses <= L_true)) / (M + 1)
    return p_value, L_true, perm_losses, model

# 5. Execution
if __name__ == "__main__":
    N, G, M, P = 900, 20, 1000, 2
    
    simulator = SpatialDataSimulator(N, G, device=device)

    for control in ["Checkerboard", "Noise"]:
        print(f"\n--- {control.upper()} CONTROL ---")
        mode = control.lower()
        
        S, A = simulator.generate(mode=mode)
        
        # Visualize Input Data
        simulator.visualize_genes(S, A, title=f"{control} Control", save_path=f"gaston_mix_{control.lower()}_genes.png")

        p, L_true, L_perm, model = closed_form_permutation_test(S, A, P=P, M=M)
        
        print(f"Final P-value: {p:.4f}")
        
        # Visualize Distribution
        simulator.visualize_permutation(L_true, L_perm, title=control, save_path=f"gaston_mix_{control.lower()}_dist.png")
        
        # Visualize Learned Spatial Components
        simulator.visualize_gaston_mix_results(S, model, title=control, save_path=f"gaston_mix_{control.lower()}_maps.png")
