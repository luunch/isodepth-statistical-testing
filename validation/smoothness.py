from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import TestConfig
from data.synthetic import SpatialDataSimulator
from methods.trainers import resolve_device, train_isodepth_model


def calculate_laplacian_smoothness(Z: np.ndarray) -> float:
    diff_x = Z[1:, :] - Z[:-1, :]
    diff_y = Z[:, 1:] - Z[:, :-1]
    return float(np.sum(diff_x**2) + np.sum(diff_y**2))


def train_and_get_isodepth(
    S: np.ndarray,
    A: np.ndarray,
    device: torch.device,
    *,
    epochs: int = 5000,
    patience: int = 50,
    seed: int = 0,
) -> np.ndarray:
    config = TestConfig(
        method="full_retraining",
        epochs=epochs,
        patience=patience,
        seed=seed,
        device=str(device),
        verbose=False,
    )
    model, _ = train_isodepth_model(S, A, config, device=device)
    with torch.no_grad():
        s_t = torch.tensor(S, dtype=torch.float32, device=device)
        d_learned = model.encoder(s_t).detach().cpu().numpy().flatten()

    d_norm = (d_learned - d_learned.min()) / (d_learned.max() - d_learned.min() + 1e-8)
    gridsize = int(np.sqrt(S.shape[0]))
    return d_norm.reshape(gridsize, gridsize)


def run_smoothness_trials(
    n_trials: int = 20,
    N: int = 400,
    G: int = 10,
    epochs: int = 5000,
    patience: int = 50,
    out_dir: str = "results",
) -> dict[str, list[float]]:
    device = resolve_device("auto")
    simulator = SpatialDataSimulator(N=N, G=G, device=str(device))

    modes = ["radial", "noise"]
    results = {mode: [] for mode in modes}

    for i in range(n_trials):
        for mode in modes:
            S, A = simulator.generate(mode=mode, seed=i)
            Z = train_and_get_isodepth(S, A, device, epochs=epochs, patience=patience, seed=i)
            results[mode].append(calculate_laplacian_smoothness(Z))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    for mode in modes:
        data.extend(results[mode])
        labels.extend([mode.capitalize()] * n_trials)

    sns.boxplot(x=labels, y=data, hue=labels, palette="Set2", showfliers=False, legend=False)
    sns.stripplot(x=labels, y=data, color=".25", size=6, alpha=0.7)
    plt.title("Laplacian Smoothness of Learned Isodepth Maps")
    plt.ylabel("Smoothness Score (Lower is Smoother)")
    plt.xlabel("Underlying Dataset Structure")
    plt.tight_layout()
    plt.savefig(out_path / "laplacian_smoothness_comparison.png")
    plt.close()
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Laplacian smoothness validation trials.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-cells", type=int, default=400)
    parser.add_argument("--n-genes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--out-dir", default="results")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_smoothness_trials(
        n_trials=args.n_trials,
        N=args.n_cells,
        G=args.n_genes,
        epochs=args.epochs,
        patience=args.patience,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
