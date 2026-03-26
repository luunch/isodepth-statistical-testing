import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_manager import SpatialDataSimulator
from isodepth import (
    choose_device,
    empirical_p_value,
    ensure_results_dir,
    gaussian_nll_from_mse,
    set_global_seed,
    train_with_early_stopping,
)
from models import IsoDepthNetMulti, ParallelIsoDepthNetMulti


def evaluate_nll(model, s_t, a_t):
    criterion = nn.MSELoss()
    with torch.no_grad():
        mse = float(criterion(model(s_t), a_t).item())
    n_total = int(a_t.shape[0] * a_t.shape[1])
    return gaussian_nll_from_mse(mse, n_total), mse


def train_multidepth_model(S, A, K, device, epochs, lr, patience, seed, hidden=32):
    set_global_seed(seed)
    model = IsoDepthNetMulti(A.shape[1], K=K, hidden=hidden).to(device)

    s_t = torch.tensor(S, dtype=torch.float32, device=device)
    a_t = torch.tensor(A, dtype=torch.float32, device=device)

    train_with_early_stopping(model, s_t, a_t, epochs=epochs, lr=lr, patience=patience)
    nll, mse = evaluate_nll(model, s_t, a_t)
    return model, nll, mse, s_t, a_t


def parallel_permutation_nll_multidepth(S, A, K, device, M, epochs, lr, patience, seed, hidden=32):
    """
    Runs true + M permutations in one batched parallel training.
    Returns (L_true, L_perm, p_value).
    """
    set_global_seed(seed)
    n, g = A.shape
    total_models = M + 1

    s_t = torch.tensor(S, dtype=torch.float32, device=device)
    a_t = torch.tensor(A, dtype=torch.float32, device=device)

    s_batched = torch.zeros((total_models, n, 2), dtype=torch.float32, device=device)
    s_batched[0] = s_t
    for m in range(1, total_models):
        perm = torch.randperm(n, device=device)
        s_batched[m] = s_t[perm]

    a_batched = a_t.unsqueeze(0).expand(total_models, -1, -1)

    model = ParallelIsoDepthNetMulti(total_models, g, K=K, hidden=hidden).to(device)
    use_foreach = device.type != "mps"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=use_foreach)
    criterion = nn.MSELoss(reduction="none")

    best_loss = float("inf")
    patience_counter = 0
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(s_batched)
        loss = criterion(output, a_batched)
        loss_per_model = loss.mean(dim=(1, 2))
        total_loss = loss_per_model.sum()
        total_loss.backward()
        optimizer.step()

        current_loss = float(total_loss.item())
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    with torch.no_grad():
        final_output = model(s_batched)
        final_mse = criterion(final_output, a_batched).mean(dim=(1, 2)).detach().cpu().numpy()

    n_total = int(n * g)
    nll = gaussian_nll_from_mse(final_mse, n_total)
    l_true = float(nll[0])
    l_perm = np.array(nll[1:], dtype=np.float64)
    p_value = empirical_p_value(l_perm, l_true)
    return l_true, l_perm, p_value


def permutation_test_multidepth(S, A, K, device, M, epochs, lr, patience, seed, hidden=32):
    # Keep a standalone true model for node-importance ablations.
    model, _, mse_true, s_t, a_t = train_multidepth_model(
        S, A, K, device, epochs, lr, patience, seed, hidden=hidden
    )
    l_true, perm_losses, p_value = parallel_permutation_nll_multidepth(
        S, A, K, device, M, epochs, lr, patience, seed + 50_000, hidden=hidden
    )
    return model, l_true, mse_true, perm_losses, p_value, s_t, a_t


def latent_dimension_importance(model, s_t, a_t):
    """
    Per-dimension ablation importance:
    delta_mse[k] = mse(with z_k zeroed) - baseline_mse.
    Higher is more important.
    """
    criterion = nn.MSELoss()
    model.eval()

    with torch.no_grad():
        pred, latent = model(s_t, return_latent=True)
        baseline_mse = float(criterion(pred, a_t).item())

        k_dim = latent.shape[1]
        delta = np.zeros(k_dim, dtype=np.float64)

        for k in range(k_dim):
            latent_ablate = latent.clone()
            latent_ablate[:, k] = 0.0
            pred_ablate = model.decoder(latent_ablate)
            ablate_mse = float(criterion(pred_ablate, a_t).item())
            delta[k] = ablate_mse - baseline_mse

    return baseline_mse, delta


def parse_k_values(raw):
    out = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("No K values provided.")
    return sorted(set(out))


def plot_summary(agg, trial_rows, out_dir, mode):
    k_vals = np.array([x["K"] for x in agg], dtype=int)

    p_mean = np.array([x["p_mean"] for x in agg], dtype=float)
    p_std = np.array([x["p_std"] for x in agg], dtype=float)

    # More useful than raw NLL: margin between null and observed.
    margin_mean = np.array([x["perm_minus_true_nll_mean"] for x in agg], dtype=float)
    margin_std = np.array([x["perm_minus_true_nll_std"] for x in agg], dtype=float)

    helpful_mean = np.array([x["helpful_mean"] for x in agg], dtype=float)
    helpful_frac_mean = np.array([x["helpful_frac_mean"] for x in agg], dtype=float)

    # Seed-level points for uncertainty visualization.
    trial_k = np.array([r["K"] for r in trial_rows], dtype=int)
    trial_p = np.array([r["p_value"] for r in trial_rows], dtype=float)
    trial_margin = np.array([r["mean_perm_nll"] - r["true_nll"] for r in trial_rows], dtype=float)
    trial_helpful_frac = np.array([r["helpful_dims"] / max(1, r["total_dims"]) for r in trial_rows], dtype=float)

    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax1.errorbar(k_vals, p_mean, yerr=p_std, marker="o", capsize=4)
    ax1.scatter(trial_k, trial_p, alpha=0.35, s=25, color="black")
    ax1.axhline(0.05, color="red", linestyle=":", label="alpha=0.05")
    ax1.set_title(f"P-value vs Bottleneck Width (mode={mode})")
    ax1.set_xlabel("K (isodepth dimensions)")
    ax1.set_ylabel("Permutation p-value")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.errorbar(k_vals, margin_mean, yerr=margin_std, marker="o", color="purple", capsize=4)
    ax2.scatter(trial_k, trial_margin, alpha=0.35, s=25, color="black")
    ax2.axhline(0, color="red", linestyle=":", label="No separation")
    ax2.set_title(f"Null-Observed NLL Margin vs K (mode={mode})")
    ax2.set_xlabel("K (isodepth dimensions)")
    ax2.set_ylabel("mean(perm NLL) - true NLL")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(k_vals, helpful_mean, marker="o", linewidth=2)
    ax3.set_title(f"Helpful Dimensions Count (mode={mode})")
    ax3.set_xlabel("K (isodepth dimensions)")
    ax3.set_ylabel("Helpful dimensions (mean across seeds)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(k_vals, helpful_frac_mean, marker="o", linewidth=2, color="teal")
    ax4.scatter(trial_k, trial_helpful_frac, alpha=0.35, s=25, color="black")
    ax4.set_ylim(-0.02, 1.02)
    ax4.set_title(f"Helpful Fraction vs K (mode={mode})")
    ax4.set_xlabel("K (isodepth dimensions)")
    ax4.set_ylabel("helpful_dims / K")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"multidepth_k_summary_{mode}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Sweep multi-isodepth bottleneck width and test usefulness.")
    parser.add_argument("--k-values", type=str, default="1,2,3,4,6,8", help="Comma-separated K values")
    parser.add_argument("--mode", type=str, choices=["noise", "radial", "checkerboard"], default="radial")
    parser.add_argument("--n", type=int, default=400, help="Number of spatial spots")
    parser.add_argument("--g", type=int, default=10, help="Number of genes")
    parser.add_argument("--perms", type=int, default=20, help="Permutation count per run")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds per K")
    parser.add_argument("--epochs", type=int, default=1500, help="Training epochs")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden width")
    parser.add_argument(
        "--helpful-threshold-frac",
        type=float,
        default=0.01,
        help="Dimension is helpful if ablation delta_mse > baseline_mse * threshold",
    )
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    device = choose_device(prefer_mps=True)
    out_dir = ensure_results_dir("results")

    print("--- MULTI-ISODEPTH BOTTLENECK SWEEP ---")
    print(f"Device: {device}")
    print(
        f"Mode={args.mode} | K={k_values} | N={args.n} G={args.g} | perms={args.perms} seeds={args.seeds} "
        f"epochs={args.epochs}"
    )

    trial_rows = []

    for k in k_values:
        print(f"\nEvaluating K={k}")
        for seed in range(args.seeds):
            set_global_seed(seed)
            simulator = SpatialDataSimulator(N=args.n, G=args.g, device=device)
            S, A = simulator.generate(mode=args.mode, seed=seed)

            model, l_true, mse_true, perm_losses, p_value, s_t, a_t = permutation_test_multidepth(
                S,
                A,
                K=k,
                device=device,
                M=args.perms,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                seed=seed,
                hidden=args.hidden,
            )

            baseline_mse, delta = latent_dimension_importance(model, s_t, a_t)
            helpful = int(np.sum(delta > baseline_mse * args.helpful_threshold_frac))

            row = {
                "K": k,
                "seed": seed,
                "mode": args.mode,
                "p_value": float(p_value),
                "true_nll": float(l_true),
                "true_mse": float(mse_true),
                "mean_perm_nll": float(np.mean(perm_losses)),
                "std_perm_nll": float(np.std(perm_losses)),
                "helpful_dims": helpful,
                "total_dims": int(k),
                "mean_delta_mse": float(np.mean(delta)),
                "max_delta_mse": float(np.max(delta)),
            }
            trial_rows.append(row)
            print(
                f"  seed={seed} p={row['p_value']:.4f} true_nll={row['true_nll']:.2f} "
                f"helpful={helpful}/{k}"
            )

    agg = []
    for k in k_values:
        k_rows = [r for r in trial_rows if r["K"] == k]
        p_vals = np.array([r["p_value"] for r in k_rows], dtype=float)
        nlls = np.array([r["true_nll"] for r in k_rows], dtype=float)
        perm_nlls = np.array([r["mean_perm_nll"] for r in k_rows], dtype=float)
        helpful = np.array([r["helpful_dims"] for r in k_rows], dtype=float)
        helpful_frac = np.array([r["helpful_dims"] / max(1, r["total_dims"]) for r in k_rows], dtype=float)

        agg.append(
            {
                "K": k,
                "p_mean": float(np.mean(p_vals)),
                "p_std": float(np.std(p_vals)),
                "nll_mean": float(np.mean(nlls)),
                "nll_std": float(np.std(nlls)),
                "helpful_mean": float(np.mean(helpful)),
                "helpful_frac_mean": float(np.mean(helpful_frac)),
                "perm_minus_true_nll_mean": float(np.mean(perm_nlls - nlls)),
                "perm_minus_true_nll_std": float(np.std(perm_nlls - nlls)),
            }
        )

    plot_summary(agg, trial_rows, out_dir, args.mode)

    print("\nSaved:")
    print(f"  - Plot: {os.path.join(out_dir, f'multidepth_k_summary_{args.mode}.png')}")

    print("\nAggregate summary:")
    for x in agg:
        print(
            f"  K={x['K']}: p={x['p_mean']:.4f}±{x['p_std']:.4f}, "
            f"NLL={x['nll_mean']:.2f}±{x['nll_std']:.2f}, "
            f"helpful={x['helpful_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
