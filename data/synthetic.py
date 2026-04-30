from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from data.schemas import DataConfig, DatasetBundle


class SpatialDataSimulator:
    def __init__(
        self,
        N: int = 900,
        G: int = 20,
        sigma: float = 0.1,
        device: str = "cpu",
        poly_degree: int = 3,
        side_length: Optional[int] = None,
    ):
        self.N_requested = int(N)
        self.side_length = side_length
        self.G = int(G)
        self.sigma = sigma
        self.device = device
        self.poly_degree = poly_degree

        if side_length is None:
            self.gridsize = int(np.sqrt(self.N_requested))
            self.N = self.gridsize**2
            self.grid_height = self.gridsize
            self.grid_width = self.gridsize
            coords = np.linspace(0, 1, self.gridsize)
            x, y = np.meshgrid(coords, coords)
            self.S = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
        else:
            self.grid_width = int(side_length)
            if self.grid_width <= 0:
                raise ValueError("side_length must be positive when provided")
            if self.N_requested % self.grid_width != 0:
                raise ValueError("When data.side_length is set, data.n_cells must be divisible by data.side_length")
            self.grid_height = self.N_requested // self.grid_width
            self.N = self.N_requested
            x_coords = np.linspace(0, 1, self.grid_width)
            y_coords = np.linspace(0, 1, self.grid_height)
            x, y = np.meshgrid(x_coords, y_coords)
            self.S = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)

    def generate(
        self,
        mode: str = "radial",
        seed: Optional[int] = None,
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        dependent_xy: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)

        if mode == "radial":
            d = np.sqrt((self.S[:, 0] - 0.5) ** 2 + (self.S[:, 1] - 0.5) ** 2)
            H = self._apply_expression_manifold(d)
            A = H + self.sigma * np.random.randn(self.N, self.G)
        elif mode == "checkerboard":
            d = np.zeros(self.N)
            for i in range(self.N):
                xi, yi = self.S[i]
                col = min(int(xi * 3), 2)
                row = min(int(yi * 3), 2)
                d[i] = xi if (row + col) % 2 == 0 else yi
            H = self._apply_expression_manifold(d)
            A = H + self.sigma * np.random.randn(self.N, self.G)
        elif mode == "fourier":
            if k_min is None or k_max is None:
                raise ValueError("k_min and k_max must be provided when mode='fourier'")
            d = self._generate_fourier_latent(k_min, k_max, dependent_xy=dependent_xy)
            H = self._apply_expression_manifold(d)
            A = H + self.sigma * np.random.randn(self.N, self.G)
        elif mode == "noise":
            d = np.zeros(self.N, dtype=np.float64)
            A = np.random.randn(self.N, self.G)
        else:
            raise ValueError(f"Unsupported synthetic data mode '{mode}'")

        A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)
        return self.S, A.astype(np.float32), d.astype(np.float32)

    def _generate_fourier_latent(self, k_min: int, k_max: int, *, dependent_xy: bool = True) -> np.ndarray:
        x = self.S[:, 0]
        y = self.S[:, 1]
        d_raw = np.zeros(self.N, dtype=np.float64)

        if dependent_xy:
            for k1 in range(k_min, k_max + 1):
                for k2 in range(k_min, k_max + 1):
                    coeffs = np.random.randn(2)
                    angle = 2.0 * np.pi * (k1 * x + k2 * y)
                    d_raw += coeffs[0] * np.cos(angle)
                    d_raw += coeffs[1] * np.sin(angle)
        else:
            for frequency in range(k_min, k_max + 1):
                coeffs = np.random.randn(4)
                angle = 2.0 * np.pi * frequency
                d_raw += coeffs[0] * np.sin(angle * x)
                d_raw += coeffs[1] * np.cos(angle * x)
                d_raw += coeffs[2] * np.sin(angle * y)
                d_raw += coeffs[3] * np.cos(angle * y)

        d_min = float(d_raw.min())
        d_max = float(d_raw.max())
        if d_max - d_min <= 1e-12:
            return np.zeros(self.N, dtype=np.float64)
        return (d_raw - d_min) / (d_max - d_min)

    def _apply_expression_manifold(self, d: np.ndarray) -> np.ndarray:
        H = np.zeros((self.N, self.G))
        for g in range(self.G):
            coeffs = np.random.randn(self.poly_degree + 1)
            H[:, g] = np.polyval(coeffs, d)
        return H

    def visualize_genes(self, S, A, title: str = "Data", n_genes: int = 10, save_path: Optional[str] = None):
        rows, cols = 2, 5
        plt.figure(figsize=(20, 8))
        for i in range(min(n_genes, A.shape[1])):
            ax = plt.subplot(rows, cols, i + 1)
            im = ax.imshow(
                A[:, i].reshape(self.grid_height, self.grid_width),
                cmap="magma",
                extent=[0, 1, 0, 1],
                origin="lower",
            )
            ax.set_title(f"Gene {i}")
            ax.axis("off")
            if i % cols == (cols - 1):
                plt.colorbar(im, ax=ax, shrink=0.8)
        plt.suptitle(f"{title}: Spatial Expression Patterns")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def visualize_permutation(self, L_true, L_perm, title: str = "Permutation Test", save_path: Optional[str] = None):
        plt.figure(figsize=(6, 5))
        sns.histplot(L_perm, color="salmon", kde=False)
        p_val = (1 + np.sum(L_perm <= L_true)) / (len(L_perm) + 1)
        plt.axvline(L_true, color="red", linestyle="--", label=f"True Loss (p={p_val:.4f})")
        plt.title(f"{title}: Null Distribution")
        plt.xlabel("Loss")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def visualize_gaston_mix_results(self, S, model, title: str = "GASTON-MIX", save_path: Optional[str] = None):
        model.eval()
        S_t = torch.tensor(S).to(self.device)
        with torch.no_grad():
            _, gates, isodepths = model(S_t)
            gates = gates.cpu().numpy()
            isodepths = [d.cpu().numpy().flatten() for d in isodepths]

        P = len(isodepths)
        fig, axes = plt.subplots(2, P, figsize=(P * 4, 8))
        for p in range(P):
            ax_g = axes[0, p]
            Z_g = gates[:, p].reshape(self.grid_height, self.grid_width)
            im_g = ax_g.imshow(Z_g, cmap="Blues", extent=[0, 1, 0, 1], origin="lower", vmin=0, vmax=1)
            ax_g.set_title(f"Expert {p} Gate Weight")
            plt.colorbar(im_g, ax=ax_g)

            ax_d = axes[1, p]
            d_learned = isodepths[p]
            d_norm = (d_learned - d_learned.min()) / (d_learned.max() - d_learned.min() + 1e-8)
            Z_d = d_norm.reshape(self.grid_height, self.grid_width)
            im_d = ax_d.imshow(Z_d, cmap="viridis", extent=[0, 1, 0, 1], origin="lower")
            ax_d.contour(Z_d, levels=8, colors="white", linewidths=1, extent=[0, 1, 0, 1], alpha=0.4)
            ax_d.set_title(f"Expert {p} Isodepth")
            plt.colorbar(im_d, ax=ax_d)
        plt.suptitle(f"{title}: Spatial Mixture Decomposition")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()


def generate_synthetic_dataset(config: DataConfig) -> DatasetBundle:
    config.validate()
    if config.source != "synthetic":
        raise ValueError(f"generate_synthetic_dataset requires data.source='synthetic', got {config.source}")

    simulator = SpatialDataSimulator(
        N=config.n_cells,
        G=config.n_genes,
        sigma=config.sigma,
        device="cpu",
        poly_degree=config.poly_degree,
        side_length=config.side_length,
    )
    s, a, true_curve = simulator.generate(
        mode=config.mode,
        seed=config.seed,
        k_min=config.k_min,
        k_max=config.k_max,
        dependent_xy=config.dependent_xy,
    )
    meta = {
        "source": "synthetic",
        "mode": config.mode,
        "seed": int(config.seed),
        "sigma": float(config.sigma),
        "poly_degree": int(config.poly_degree),
        "n_cells_requested": int(config.n_cells),
        "n_cells_generated": int(s.shape[0]),
        "n_genes": int(a.shape[1]),
        "synthetic_true_curve": np.asarray(true_curve, dtype=np.float32),
        "grid_height": int(simulator.grid_height),
        "grid_width": int(simulator.grid_width),
    }
    if config.mode == "fourier":
        meta["k_min"] = int(config.k_min)
        meta["k_max"] = int(config.k_max)
        meta["dependent_xy"] = bool(config.dependent_xy)
        meta["fourier_basis"] = "interaction_xy" if config.dependent_xy else "independent_xy"
    if config.side_length is not None:
        meta["side_length"] = int(config.side_length)
        meta["other_side_length"] = int(s.shape[0] // config.side_length)
    return DatasetBundle(S=s, A=a, meta=meta).validate()
