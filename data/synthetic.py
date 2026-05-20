from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from data.schemas import DataConfig, DatasetBundle, SamplingBiasConfig


def _point_in_spatial_shape(shape: str, x: float, y: float, *, eps: float = 1e-9) -> bool:
    """Shapes live in the unit square; lattice points are tested in [0, 1]^2."""
    if shape == "square":
        return x >= -eps and y >= -eps and x <= 1.0 + eps and y <= 1.0 + eps
    if shape == "circle":
        return (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25 + eps
    if shape == "semicircle":
        # Diameter from (0, 0) to (1, 0); disk center (0.5, 0), radius 0.5
        return y >= -eps and (x - 0.5) ** 2 + y**2 <= 0.25 + eps
    if shape == "triangle":
        # Right triangle with vertices (0, 0), (1, 0), (0, 1)
        return x >= -eps and y >= -eps and x + y <= 1.0 + eps
    if shape == "square_cutout":
        # Unit square with the bottom semicircle removed (same disk as ``semicircle``,
        # i.e. diameter on the bottom edge from (0,0) to (1,0), bulge into y > 0).
        if not (x >= -eps and y >= -eps and x <= 1.0 + eps and y <= 1.0 + eps):
            return False
        in_removed_semicircle = y >= -eps and (x - 0.5) ** 2 + y**2 <= 0.25 + eps
        return not in_removed_semicircle
    raise ValueError(f"unknown spatial shape {shape!r}")


def _spatial_shape_mask(xy: np.ndarray, shape: str, *, eps: float = 1e-9) -> np.ndarray:
    """Vectorized membership in the same regions as ``_point_in_spatial_shape`` (for rejection sampling)."""
    x = xy[:, 0]
    y = xy[:, 1]
    if shape == "square":
        return (x >= -eps) & (y >= -eps) & (x <= 1.0 + eps) & (y <= 1.0 + eps)
    if shape == "circle":
        return (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25 + eps
    if shape == "semicircle":
        return (y >= -eps) & ((x - 0.5) ** 2 + y**2 <= 0.25 + eps)
    if shape == "triangle":
        return (x >= -eps) & (y >= -eps) & (x + y <= 1.0 + eps)
    if shape == "square_cutout":
        in_square = (x >= -eps) & (y >= -eps) & (x <= 1.0 + eps) & (y <= 1.0 + eps)
        in_removed = (y >= -eps) & ((x - 0.5) ** 2 + y**2 <= 0.25 + eps)
        return in_square & ~in_removed
    raise ValueError(f"unknown spatial shape {shape!r}")


def _grid_dims_from_xy_extent(s_arr: np.ndarray) -> tuple[int, int]:
    """Match plot aspect hints to ``_fit_shaped_lattice`` (grid_height, grid_width)."""
    sx = float(s_arr[:, 0].max() - s_arr[:, 0].min()) + 1e-9
    sy = float(s_arr[:, 1].max() - s_arr[:, 1].min()) + 1e-9
    ar = sx / sy
    if ar >= 1.0:
        gw = 100
        gh = max(1, int(round(100 / ar)))
    else:
        gh = 100
        gw = max(1, int(round(100 * ar)))
    return gh, gw


def _sample_uniform_on_shape(n_cells: int, shape: str, rng: np.random.RandomState) -> np.ndarray:
    """IID uniform samples on the spatial mask (Lebesgue-uniform), via rejection in ``[0,1]^2``."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    chunks: list[np.ndarray] = []
    remaining = n_cells
    while remaining > 0:
        batch = max(min(remaining * 10, 500_000), 4096)
        cand = rng.uniform(0.0, 1.0, size=(batch, 2))
        mask = _spatial_shape_mask(cand, shape)
        accepted = cand[mask]
        if accepted.shape[0] == 0:
            continue
        take = min(int(accepted.shape[0]), remaining)
        chunks.append(accepted[:take].astype(np.float32))
        remaining -= take
    return np.concatenate(chunks, axis=0)


def _sample_normal_on_shape(
    n_cells: int,
    shape: str,
    variance: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Isotropic Gaussian centered at (0.5, 0.5) with covariance ``variance * I_2``,
    rejection-sampled inside the spatial mask intersected with ``[0,1]^2``."""
    if variance <= 0.0:
        raise ValueError("variance must be > 0")
    return _sample_anisotropic_normal_on_shape(
        n_cells, shape, (float(variance), float(variance)), rng
    )


def _sample_anisotropic_normal_on_shape(
    n_cells: int,
    shape: str,
    variance_xy: tuple[float, float] | list[float],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Diagonal-covariance Gaussian centered at (0.5, 0.5) with covariance
    ``diag(variance_xy[0], variance_xy[1])``, rejection-sampled inside the spatial
    mask intersected with ``[0, 1]^2``."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if len(variance_xy) != 2:
        raise ValueError("variance_xy must be a length-2 sequence")
    var_x = float(variance_xy[0])
    var_y = float(variance_xy[1])
    if var_x <= 0.0 or var_y <= 0.0:
        raise ValueError("variance_xy entries must be > 0")
    sigma_x = float(np.sqrt(var_x))
    sigma_y = float(np.sqrt(var_y))
    chunks: list[np.ndarray] = []
    remaining = n_cells
    while remaining > 0:
        batch = max(min(remaining * 20, 500_000), 4096)
        cand = np.stack(
            [
                rng.normal(loc=0.5, scale=sigma_x, size=batch),
                rng.normal(loc=0.5, scale=sigma_y, size=batch),
            ],
            axis=1,
        )
        in_unit = (
            (cand[:, 0] >= 0.0)
            & (cand[:, 0] <= 1.0)
            & (cand[:, 1] >= 0.0)
            & (cand[:, 1] <= 1.0)
        )
        mask = in_unit & _spatial_shape_mask(cand, shape)
        accepted = cand[mask]
        if accepted.shape[0] == 0:
            continue
        take = min(int(accepted.shape[0]), remaining)
        chunks.append(accepted[:take].astype(np.float32))
        remaining -= take
    return np.concatenate(chunks, axis=0)


def _lattice_points_for_resolution(shape: str, K: int) -> list[tuple[float, float]]:
    """Integer lattice (i/K, j/K) intersected with the shape."""
    if K <= 0:
        return []
    pts: list[tuple[float, float]] = []
    for i in range(K + 1):
        for j in range(K + 1):
            x = i / K
            y = j / K
            if _point_in_spatial_shape(shape, x, y):
                pts.append((x, y))
    return pts


def _fit_shaped_lattice(
    n_cells: int, shape: str, seed: int
) -> tuple[np.ndarray, int, int, int]:
    """
    Choose the smallest K such that the K-step grid has at least n_cells points in the
    shape; subsample uniformly to exactly n_cells when there are more.

    Returns (S, grid_height, grid_width, lattice_K) where grid_* are integers whose ratio
    matches the spatial extent of S (for plot aspect hints).
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    max_K = max(50_000, int(n_cells**0.5 * 200))
    K = 1
    while K <= max_K:
        pts = _lattice_points_for_resolution(shape, K)
        if len(pts) >= n_cells:
            break
        K += 1
    else:
        raise ValueError(
            f"Could not fit {n_cells} lattice points in shape={shape!r} (exceeded max K={max_K})"
        )
    rng = np.random.RandomState(int(seed))
    if len(pts) == n_cells:
        idx = np.arange(n_cells, dtype=np.int64)
    else:
        idx = rng.choice(len(pts), size=n_cells, replace=False)
    chosen = [pts[i] for i in idx]
    s_arr = np.asarray(chosen, dtype=np.float32)
    gh, gw = _grid_dims_from_xy_extent(s_arr)
    return s_arr, gh, gw, K


class SpatialDataSimulator:
    def __init__(
        self,
        N: int = 900,
        G: int = 20,
        sigma: float = 0.1,
        device: str = "cpu",
        poly_degree: int = 3,
        side_length: Optional[int] = None,
        shape: str = "square",
        lattice_seed: int = 0,
        sampling_bias: Optional[SamplingBiasConfig] = None,
    ):
        self.N_requested = int(N)
        self.side_length = side_length
        self.G = int(G)
        self.sigma = sigma
        self.device = device
        self.poly_degree = poly_degree
        self.shape = shape
        self.lattice_seed = int(lattice_seed)
        self.lattice_resolution: Optional[int] = None

        if sampling_bias is not None:
            rng = np.random.RandomState(int(lattice_seed))
            if sampling_bias.type == "uniform":
                self.S = _sample_uniform_on_shape(self.N_requested, shape, rng)
            elif sampling_bias.type == "normal":
                if sampling_bias.variance is None:
                    raise ValueError("sampling_bias.variance is required when type='normal'")
                self.S = _sample_normal_on_shape(
                    self.N_requested, shape, float(sampling_bias.variance), rng
                )
            elif sampling_bias.type == "anisotropic_normal":
                if sampling_bias.variance is None or not isinstance(sampling_bias.variance, (list, tuple)):
                    raise ValueError(
                        "sampling_bias.variance must be a length-2 list when type='anisotropic_normal'"
                    )
                self.S = _sample_anisotropic_normal_on_shape(
                    self.N_requested, shape, tuple(sampling_bias.variance), rng
                )
            else:
                raise ValueError(
                    f"Unsupported sampling_bias.type {sampling_bias.type!r}"
                )
            self.N = int(self.S.shape[0])
            self.grid_height, self.grid_width = _grid_dims_from_xy_extent(self.S)
            return

        if shape != "square":
            self.S, self.grid_height, self.grid_width, K = _fit_shaped_lattice(
                self.N_requested, shape, self.lattice_seed
            )
            self.lattice_resolution = int(K)
            self.N = int(self.S.shape[0])
        elif side_length is None:
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
        shape=config.shape,
        lattice_seed=config.seed,
        sampling_bias=config.sampling_bias,
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
    if config.sampling_bias is not None:
        meta["sampling_bias"] = config.sampling_bias.to_meta()
    if config.shape != "square":
        meta["shape"] = str(config.shape)
        if simulator.lattice_resolution is not None:
            meta["lattice_resolution"] = int(simulator.lattice_resolution)
    if config.mode == "fourier":
        meta["k_min"] = int(config.k_min)
        meta["k_max"] = int(config.k_max)
        meta["dependent_xy"] = bool(config.dependent_xy)
        meta["fourier_basis"] = "interaction_xy" if config.dependent_xy else "independent_xy"
    if config.side_length is not None and config.shape == "square" and config.sampling_bias is None:
        meta["side_length"] = int(config.side_length)
        meta["other_side_length"] = int(s.shape[0] // config.side_length)
    return DatasetBundle(S=s, A=a, meta=meta).validate()
