from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.synthetic import SpatialDataSimulator
from methods.trainers import resolve_device
from validation.smoothness import calculate_laplacian_smoothness, train_and_get_isodepth


def run_permuted_smoothness_trials(
    n_trials: int = 20,
    N: int = 400,
    G: int = 10,
    epochs: int = 500,
    patience: int = 50,
    out_dir: str = "results",
) -> dict[str, dict[str, float | list[float]]]:
    device = resolve_device("auto")
    simulator = SpatialDataSimulator(N=N, G=G, device=str(device))
    modes = ["radial", "noise"]

    original_smoothness: dict[str, float] = {}
    original_Z: dict[str, np.ndarray] = {}
    permuted_smoothness = {mode: [] for mode in modes}

    best_perm_Z = {mode: None for mode in modes}
    worst_perm_Z = {mode: None for mode in modes}
    best_perm_smooth = {mode: float("inf") for mode in modes}
    worst_perm_smooth = {mode: float("-inf") for mode in modes}

    rng = np.random.default_rng(0)
    for mode in modes:
        S, A = simulator.generate(mode=mode, seed=0)
        Z_orig = train_and_get_isodepth(S, A, device, epochs=epochs, patience=patience, seed=0)
        orig_smooth = calculate_laplacian_smoothness(Z_orig)
        original_smoothness[mode] = orig_smooth
        original_Z[mode] = Z_orig

        for i in range(n_trials):
            perm = rng.permutation(A.shape[0])
            A_perm = A[perm]
            Z_perm = train_and_get_isodepth(S, A_perm, device, epochs=epochs, patience=patience, seed=i + 1)
            perm_smooth = calculate_laplacian_smoothness(Z_perm)
            permuted_smoothness[mode].append(perm_smooth)
            if perm_smooth < best_perm_smooth[mode]:
                best_perm_smooth[mode] = perm_smooth
                best_perm_Z[mode] = Z_perm
            if perm_smooth > worst_perm_smooth[mode]:
                worst_perm_smooth[mode] = perm_smooth
                worst_perm_Z[mode] = Z_perm

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    for mode in modes:
        data.extend(permuted_smoothness[mode])
        labels.extend([mode.capitalize()] * n_trials)
    sns.boxplot(x=labels, y=data, color="lightblue", showfliers=False)
    sns.stripplot(x=labels, y=data, color="blue", alpha=0.5, jitter=True)
    for i, mode in enumerate(modes):
        plt.scatter(i, original_smoothness[mode], color="red", marker="*", s=200, label="Original" if i == 0 else "", zorder=5)
    plt.title("Laplacian Smoothness: Original vs. Permuted Null Distribution")
    plt.ylabel("Smoothness Score (Lower is Smoother)")
    plt.xlabel("Underlying Dataset Structure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "permuted_smoothness_comparison.png")
    plt.close()

    fig, axes = plt.subplots(len(modes), 3, figsize=(15, 5 * len(modes)))
    if len(modes) == 1:
        axes = [axes]
    for idx, mode in enumerate(modes):
        for j, (title, image, score) in enumerate(
            [
                ("Original", original_Z[mode], original_smoothness[mode]),
                ("Smoothest Perm", best_perm_Z[mode], best_perm_smooth[mode]),
                ("Roughest Perm", worst_perm_Z[mode], worst_perm_smooth[mode]),
            ]
        ):
            ax = axes[idx][j]
            im = ax.imshow(image, cmap="viridis", extent=[0, 1, 0, 1], origin="lower")
            ax.set_title(f"{mode.capitalize()}: {title}\nScore: {score:.2f}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path / "permuted_extremes.png")
    plt.close()

    return {
        mode: {
            "original_smoothness": original_smoothness[mode],
            "permuted_smoothness": permuted_smoothness[mode],
            "best_perm_smoothness": best_perm_smooth[mode],
            "worst_perm_smoothness": worst_perm_smooth[mode],
        }
        for mode in modes
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run permuted smoothness validation trials.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-cells", type=int, default=400)
    parser.add_argument("--n-genes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--out-dir", default="results")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_permuted_smoothness_trials(
        n_trials=args.n_trials,
        N=args.n_cells,
        G=args.n_genes,
        epochs=args.epochs,
        patience=args.patience,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
