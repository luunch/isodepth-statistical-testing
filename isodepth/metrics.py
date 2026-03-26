from __future__ import annotations

import numpy as np


def gaussian_nll_from_mse(mse: float, n_total: int) -> float:
    return (n_total / 2.0) * np.log(2 * np.pi * mse + 1e-12) + (n_total / 2.0)


def empirical_p_value(null_losses: np.ndarray, observed_loss: float) -> float:
    return (1 + np.sum(null_losses <= observed_loss)) / (len(null_losses) + 1)


def laplacian_smoothness(z_grid: np.ndarray) -> float:
    diff_x = z_grid[1:, :] - z_grid[:-1, :]
    diff_y = z_grid[:, 1:] - z_grid[:, :-1]
    return float(np.sum(diff_x**2) + np.sum(diff_y**2))
