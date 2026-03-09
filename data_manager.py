import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class SpatialDataSimulator:
    def __init__(self, N=900, G=20, sigma=0.1, device="cpu"):
        self.N_requested = N
        self.gridsize = int(np.sqrt(N))
        self.N = self.gridsize**2
        self.G = G
        self.sigma = sigma
        self.device = device
        
        # Pre-compute coordinates
        coords = np.linspace(0, 1, self.gridsize)
        x, y = np.meshgrid(coords, coords)
        self.S = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)

    def generate(self, mode="radial", seed=None):
        """
        Generates simulated spatial gene expression data (Continuous/Gaussian).
        Modes: 'radial', 'checkerboard', 'noise'
        """
        if seed is not None:
            np.random.seed(seed)
            
        if mode == "radial":
            # Radial distance from center
            d = np.sqrt((self.S[:, 0] - 0.5)**2 + (self.S[:, 1] - 0.5)**2)
            H = self._apply_expression_manifold(d)
            A = H + self.sigma * np.random.randn(self.N, self.G)
        
        elif mode == "checkerboard":
            # 3x3 grid with alternating x and y gradients
            d = np.zeros(self.N)
            for i in range(self.N):
                xi, yi = self.S[i]
                col = min(int(xi * 3), 2)
                row = min(int(yi * 3), 2)
                if (row + col) % 2 == 0:
                    d[i] = xi
                else:
                    d[i] = yi
            H = self._apply_expression_manifold(d)
            A = H + self.sigma * np.random.randn(self.N, self.G)
            
        else:
            A = np.random.randn(self.N, self.G)

        # Standardize gene expression
        A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)
        return self.S, A.astype(np.float32)

    def _apply_expression_manifold(self, d):
        H = np.zeros((self.N, self.G))
        for g in range(self.G):
            coeffs = np.random.randn(4)
            H[:, g] = np.polyval(coeffs, d)
        return H

    def visualize_genes(self, S, A, title="Data", n_genes=10, save_path=None):
        rows, cols = 2, 5
        plt.figure(figsize=(20, 8))
        for i in range(min(n_genes, A.shape[1])):
            ax = plt.subplot(rows, cols, i + 1)
            im = ax.imshow(A[:, i].reshape(self.gridsize, self.gridsize), 
                           cmap="magma", extent=[0, 1, 0, 1], origin="lower")
            ax.set_title(f"Gene {i}")
            ax.axis("off")
            if i % cols == (cols - 1):
                plt.colorbar(im, ax=ax, shrink=0.8)
        plt.suptitle(f"{title}: Spatial Expression Patterns")
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.close()

    def visualize_permutation(self, L_true, L_perm, title="Permutation Test", save_path=None):
        plt.figure(figsize=(6, 5))
        sns.histplot(L_perm, color="salmon", kde=False)
        p_val = (1 + np.sum(L_perm <= L_true)) / (len(L_perm) + 1)
        plt.axvline(L_true, color="red", linestyle="--", label=f"True Loss (p={p_val:.4f})")
        plt.title(f"{title}: Null Distribution")
        plt.xlabel("Loss")
        plt.legend()
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.close()

    def visualize_gaston_mix_results(self, S, model, title="GASTON-MIX", save_path=None):
        model.eval()
        S_t = torch.tensor(S).to(self.device)
        with torch.no_grad():
            _, gates, isodepths = model(S_t)
            gates = gates.cpu().numpy()
            isodepths = [d.cpu().numpy().flatten() for d in isodepths]
        
        P = len(isodepths)
        fig, axes = plt.subplots(2, P, figsize=(P*4, 8))
        for p in range(P):
            ax_g = axes[0, p]
            Z_g = gates[:, p].reshape(self.gridsize, self.gridsize)
            im_g = ax_g.imshow(Z_g, cmap="Blues", extent=[0,1,0,1], origin="lower", vmin=0, vmax=1)
            ax_g.set_title(f"Expert {p} Gate Weight")
            plt.colorbar(im_g, ax=ax_g)
            
            ax_d = axes[1, p]
            d_learned = isodepths[p]
            d_norm = (d_learned - d_learned.min()) / (d_learned.max() - d_learned.min() + 1e-8)
            Z_d = d_norm.reshape(self.gridsize, self.gridsize)
            im_d = ax_d.imshow(Z_d, cmap="viridis", extent=[0,1,0,1], origin="lower")
            ax_d.contour(Z_d, levels=8, colors="white", linewidths=1, extent=[0,1,0,1], alpha=0.4)
            ax_d.set_title(f"Expert {p} Isodepth")
            plt.colorbar(im_d, ax=ax_d)
        plt.suptitle(f"{title}: Spatial Mixture Decomposition")
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.close()
