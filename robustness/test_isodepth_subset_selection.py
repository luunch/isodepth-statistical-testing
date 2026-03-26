import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

# Ensure repository root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_manager import SpatialDataSimulator
from isodepth import choose_device, ensure_results_dir, set_global_seed, train_with_early_stopping
from models import IsoDepthNet, ParallelIsoDepthNet


def parse_modes(raw: str):
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    allowed = {"noise", "radial", "checkerboard"}
    invalid = [m for m in modes if m not in allowed]
    if invalid:
        raise ValueError(f"Invalid modes: {invalid}. Allowed: {sorted(allowed)}")
    if not modes:
        raise ValueError("No modes provided.")
    return modes


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    corr, _ = stats.spearmanr(x, y)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def train_isodepth_model(
    s: np.ndarray,
    a: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
):
    set_global_seed(seed)
    model = IsoDepthNet(a.shape[1]).to(device)

    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    a_t = torch.tensor(a, dtype=torch.float32, device=device)

    train_with_early_stopping(model, s_t, a_t, epochs=epochs, lr=lr, patience=patience)
    model.eval()

    with torch.no_grad():
        d = model.encoder(s_t).detach().cpu().numpy().flatten()

    return model, d


def train_parallel_isodepth_models(
    s_batched: np.ndarray,
    a_batched: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
):
    """
    Train M independent IsoDepth models in parallel.
    s_batched: [M, N_sub, 2]
    a_batched: [M, N_sub, G]
    Returns latent isodepths: [M, N_sub]
    """
    set_global_seed(seed)
    m_models, _, g = a_batched.shape

    s_t = torch.tensor(s_batched, dtype=torch.float32, device=device)
    a_t = torch.tensor(a_batched, dtype=torch.float32, device=device)

    model = ParallelIsoDepthNet(m_models, g).to(device)
    use_foreach = device.type != "mps"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=use_foreach)
    criterion = nn.MSELoss(reduction="none")

    best_loss = float("inf")
    patience_counter = 0
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(s_t)
        loss = criterion(out, a_t)
        loss_per_model = loss.mean(dim=(1, 2))
        total_loss = loss_per_model.sum()
        total_loss.backward()
        optimizer.step()

        current = float(total_loss.item())
        if current < best_loss - 1e-5:
            best_loss = current
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    with torch.no_grad():
        d_batched = model.encoder(s_t).squeeze(-1).detach().cpu().numpy()
    return d_batched


def subset_identifiability_test(
    s: np.ndarray,
    a: np.ndarray,
    subset_frac: float,
    m_subsets: int,
    k_null: int,
    epochs: int,
    lr: float,
    patience: int,
    align_sign: bool,
    use_parallel: bool,
    seed: int,
    device: torch.device,
):
    start_time = time.time()

    # Baseline isodepth from all spots.
    _, d_full = train_isodepth_model(s, a, device, epochs, lr, patience, seed=seed)
    print("Phase 1/3: trained full-data baseline isodepth.")

    n = s.shape[0]
    subset_size = max(5, int(round(subset_frac * n)))

    idx_subset_list = []
    d_subset_list = []

    log_every = max(1, m_subsets // 10)
    if use_parallel:
        s_sub_batched = np.zeros((m_subsets, subset_size, 2), dtype=np.float32)
        a_sub_batched = np.zeros((m_subsets, subset_size, a.shape[1]), dtype=np.float32)
        print(f"Phase 2/3: preparing and training {m_subsets} subsets in parallel (subset size={subset_size}/{n})...")

        for m in range(m_subsets):
            rng = np.random.default_rng(seed + 10_000 + m)
            idx = rng.choice(n, size=subset_size, replace=False)

            idx_subset_list.append(idx)
            s_sub_batched[m] = s[idx]
            a_sub_batched[m] = a[idx]

            if (m + 1) % log_every == 0 or (m + 1) == m_subsets:
                print(f"  subset prep: {m + 1}/{m_subsets}")

        d_sub_batched = train_parallel_isodepth_models(
            s_batched=s_sub_batched,
            a_batched=a_sub_batched,
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            seed=seed + 20_000,
        )
    else:
        d_sub_batched = None
        print(f"Phase 2/3: training on {m_subsets} subsets sequentially (subset size={subset_size}/{n})...")
        for m in range(m_subsets):
            rng = np.random.default_rng(seed + 10_000 + m)
            idx = rng.choice(n, size=subset_size, replace=False)
            idx_subset_list.append(idx)
            s_sub = s[idx]
            a_sub = a[idx]
            _, d_sub = train_isodepth_model(
                s_sub,
                a_sub,
                device,
                epochs,
                lr,
                patience,
                seed=seed + 20_000 + m,
            )
            d_subset_list.append(d_sub)
            if (m + 1) % log_every == 0 or (m + 1) == m_subsets:
                print(f"  subset retrain progress: {m + 1}/{m_subsets}")

    obs_corrs = np.zeros(m_subsets, dtype=np.float64)
    for m in range(m_subsets):
        idx = idx_subset_list[m]
        d_sub = d_sub_batched[m] if use_parallel else d_subset_list[m]
        d_ref = d_full[idx]
        corr = spearman_safe(d_ref, d_sub)
        if align_sign and corr < 0:
            d_sub = -d_sub
            corr = -corr
        if use_parallel:
            d_subset_list.append(d_sub)
        else:
            d_subset_list[m] = d_sub
        obs_corrs[m] = corr
        if (m + 1) % log_every == 0 or (m + 1) == m_subsets:
            print(f"  subset correlation pass: {m + 1}/{m_subsets} (running mean Spearman={np.mean(obs_corrs[:m+1]):.4f})")

    t_obs = float(np.mean(obs_corrs))

    null_corrs = np.zeros(k_null, dtype=np.float64)
    null_log_every = max(1, k_null // 10)
    print(f"Phase 3/3: building null distribution with {k_null} permutations...")

    for k in range(k_null):
        corr_k = np.zeros(m_subsets, dtype=np.float64)
        for m in range(m_subsets):
            idx = idx_subset_list[m]
            d_ref = d_full[idx]
            d_sub = d_subset_list[m]

            perm_rng = np.random.default_rng(seed + 30_000 + (k * m_subsets + m))
            perm = perm_rng.permutation(len(d_sub))
            corr_k[m] = spearman_safe(d_ref, d_sub[perm])

        null_corrs[k] = float(np.mean(corr_k))

        if (k + 1) % null_log_every == 0 or (k + 1) == k_null:
            print(f"  null progress: {k + 1}/{k_null}")

    p_value = float((1 + np.sum(null_corrs >= t_obs)) / (k_null + 1))
    elapsed = time.time() - start_time
    print(f"Completed subset selection test in {elapsed:.1f}s.")

    return {
        "t_obs": t_obs,
        "p_value": p_value,
        "obs_corrs": obs_corrs,
        "null_corrs": null_corrs,
        "subset_size": subset_size,
        "n_total": n,
    }


def save_outputs(result: dict, out_dir: str, mode: str, subset_frac: float):
    tag = f"{mode}_subset{subset_frac:.2f}".replace(".", "p")

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(result["null_corrs"], bins=30, color="lightsteelblue", edgecolor="black")
    ax1.axvline(result["t_obs"], color="red", linestyle="--", label=f"Observed mean={result['t_obs']:.3f}")
    ax1.set_title("Null Distribution of Mean Spearman")
    ax1.set_xlabel("Mean Spearman")
    ax1.set_ylabel("Count")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(result["obs_corrs"], bins=20, color="darkseagreen", edgecolor="black")
    ax2.axvline(np.mean(result["obs_corrs"]), color="black", linestyle=":", label="Observed mean")
    ax2.set_title("Observed Spearman Across Subsets")
    ax2.set_xlabel("Spearman(d_A, d|_A)")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"subset_summary_{tag}.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def save_mode_comparison(summary_rows, out_dir: str, subset_frac: float):
    tag = f"subset{subset_frac:.2f}".replace(".", "p")
    modes = [row["mode"] for row in summary_rows]
    t_obs = [row["t_obs"] for row in summary_rows]
    p_vals = [row["p_value"] for row in summary_rows]

    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(modes, t_obs, color="mediumseagreen", alpha=0.85)
    ax1.set_title("Observed Subset Stability by Mode")
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("Observed Mean Spearman (T_obs)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(modes, p_vals, color="cornflowerblue", alpha=0.85)
    ax2.axhline(0.05, color="red", linestyle=":", label="alpha=0.05")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("P-value by Mode")
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("One-sided p-value")
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"subset_mode_comparison_{tag}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Subset-selection test for isodepth global identifiability.")
    parser.add_argument(
        "--modes",
        type=str,
        default="radial",
        help="Comma-separated dataset modes, e.g. radial,noise",
    )
    parser.add_argument("--n", type=int, default=400, help="Number of spatial spots")
    parser.add_argument("--g", type=int, default=10, help="Number of genes")
    parser.add_argument("--subset-frac", type=float, default=0.7, help="Subset fraction |A|/n")
    parser.add_argument("--m-subsets", type=int, default=30, help="Number of subset retraining runs (M)")
    parser.add_argument("--k-null", type=int, default=500, help="Number of null permutations (K)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-align-sign",
        action="store_true",
        help="Disable sign alignment between d|_A and d_A before Spearman.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential retraining instead of parallel batched retraining.",
    )

    args = parser.parse_args()

    if not (0 < args.subset_frac <= 1):
        raise ValueError("--subset-frac must be in (0, 1].")

    device = choose_device(prefer_mps=True)
    out_dir = ensure_results_dir("results")
    modes = parse_modes(args.modes)

    print("--- ISODEPTH SUBSET SELECTION TEST ---")
    print(
        f"Device={device} modes={modes} subset_frac={args.subset_frac} "
        f"M={args.m_subsets} K={args.k_null} method={'sequential' if args.sequential else 'parallel'}"
    )

    summary_rows = []
    for i, mode in enumerate(modes):
        print(f"\n=== Mode: {mode} ===")
        simulator = SpatialDataSimulator(N=args.n, G=args.g, device=device)
        s, a = simulator.generate(mode=mode, seed=args.seed + i)

        result = subset_identifiability_test(
            s=s,
            a=a,
            subset_frac=args.subset_frac,
            m_subsets=args.m_subsets,
            k_null=args.k_null,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            align_sign=not args.no_align_sign,
            use_parallel=not args.sequential,
            seed=args.seed + i,
            device=device,
        )

        plot_path = save_outputs(result, out_dir, mode, args.subset_frac)

        reject = result["p_value"] < args.alpha
        print("Mode result:")
        print(f"  Subset size |A|: {result['subset_size']}/{result['n_total']}")
        print(f"  Observed mean stability T_obs: {result['t_obs']:.4f}")
        print(f"  One-sided p-value: {result['p_value']:.6f}")
        print(f"  Reject H0 at alpha={args.alpha}: {reject}")
        print("  Saved files:")
        print(f"    - {plot_path}")

        summary_rows.append(
            {
                "mode": mode,
                "t_obs": float(result["t_obs"]),
                "p_value": float(result["p_value"]),
                "subset_size": int(result["subset_size"]),
                "n_total": int(result["n_total"]),
                "reject_h0": int(reject),
            }
        )

    comp_plot = save_mode_comparison(summary_rows, out_dir, args.subset_frac)
    print("\nCross-mode comparison:")
    print(f"  - {comp_plot}")
    for row in summary_rows:
        print(
            f"  {row['mode']}: T_obs={row['t_obs']:.4f}, "
            f"p={row['p_value']:.6f}, reject={bool(row['reject_h0'])}"
        )


if __name__ == "__main__":
    main()
