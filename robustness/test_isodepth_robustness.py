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
    s_batched: [M, N, 2]
    a_batched: [M, N, G]
    Returns latent isodepths: [M, N]
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
        loss = criterion(out, a_t)  # [M, N, G]
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
        d_batched = model.encoder(s_t).squeeze(-1).detach().cpu().numpy()  # [M, N]
    return d_batched


def perturb_dataset(
    s: np.ndarray,
    a: np.ndarray,
    perturb_target: str,
    delta: float,
    rng: np.random.Generator,
):
    s_delta = s.copy()
    a_delta = a.copy()

    if perturb_target in {"coords", "both"}:
        # Coordinates are in [0,1], so delta is interpreted as absolute std on that scale.
        s_delta = s_delta + rng.normal(loc=0.0, scale=delta, size=s_delta.shape)
        s_delta = np.clip(s_delta, 0.0, 1.0)

    if perturb_target in {"expression", "both"}:
        gene_std = a_delta.std(axis=0, keepdims=True) + 1e-8
        noise = rng.normal(loc=0.0, scale=delta, size=a_delta.shape) * gene_std
        a_delta = a_delta + noise

    return s_delta.astype(np.float32), a_delta.astype(np.float32)


def robustness_test(
    s: np.ndarray,
    a: np.ndarray,
    perturb_target: str,
    delta: float,
    m_perturb: int,
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
    n = s.shape[0]
    perturb_log_every = max(1, m_perturb // 10)
    d_deltas = np.zeros((m_perturb, n), dtype=np.float64)

    if use_parallel:
        print("Phase 1/3: building perturbed datasets for parallel training.")
        g = a.shape[1]
        total_models = m_perturb + 1
        s_batched = np.zeros((total_models, n, 2), dtype=np.float32)
        a_batched = np.zeros((total_models, n, g), dtype=np.float32)
        s_batched[0] = s.astype(np.float32)
        a_batched[0] = a.astype(np.float32)

        for m in range(m_perturb):
            rng = np.random.default_rng(seed + 10_000 + m)
            s_delta, a_delta = perturb_dataset(s, a, perturb_target, delta, rng)
            s_batched[m + 1] = s_delta
            a_batched[m + 1] = a_delta
            if (m + 1) % perturb_log_every == 0 or (m + 1) == m_perturb:
                print(f"  perturb dataset prep: {m + 1}/{m_perturb}")

        print(f"Phase 2/3: training {total_models} models in parallel (1 baseline + {m_perturb} perturbed)...")
        d_all = train_parallel_isodepth_models(
            s_batched=s_batched,
            a_batched=a_batched,
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            seed=seed + 20_000,
        )
        d_base = d_all[0]
        d_deltas = d_all[1:]
    else:
        print("Phase 1/3: training baseline isodepth model.")
        _, d_base = train_isodepth_model(s, a, device, epochs, lr, patience, seed=seed)
        print(f"Phase 2/3: running {m_perturb} perturbed retraining runs (sequential mode)...")
        for m in range(m_perturb):
            rng = np.random.default_rng(seed + 10_000 + m)
            s_delta, a_delta = perturb_dataset(s, a, perturb_target, delta, rng)
            _, d_delta = train_isodepth_model(
                s_delta,
                a_delta,
                device,
                epochs,
                lr,
                patience,
                seed=seed + 20_000 + m,
            )
            d_deltas[m] = d_delta
            if (m + 1) % perturb_log_every == 0 or (m + 1) == m_perturb:
                print(f"  perturb retrain progress: {m + 1}/{m_perturb}")

    obs_corrs = np.zeros(m_perturb, dtype=np.float64)
    for m in range(m_perturb):
        d_delta = d_deltas[m]
        corr = spearman_safe(d_base, d_delta)
        if align_sign and corr < 0:
            d_delta = -d_delta
            corr = -corr
            d_deltas[m] = d_delta

        obs_corrs[m] = corr
        if (m + 1) % perturb_log_every == 0 or (m + 1) == m_perturb:
            print(
                f"  perturb correlation pass: {m + 1}/{m_perturb} "
                f"(running mean Spearman={np.mean(obs_corrs[: m + 1]):.4f})"
            )

    s_obs = float(np.mean(obs_corrs))

    null_corrs = np.zeros(k_null, dtype=np.float64)
    null_log_every = max(1, k_null // 10)
    print(f"Phase 3/3: building null with {k_null} permutations...")
    for k in range(k_null):
        perm_rng = np.random.default_rng(seed + 30_000 + k)
        perm = perm_rng.permutation(s.shape[0])

        corr_k = np.zeros(m_perturb, dtype=np.float64)
        for m in range(m_perturb):
            corr_k[m] = spearman_safe(d_base, d_deltas[m][perm])
        null_corrs[k] = float(np.mean(corr_k))
        if (k + 1) % null_log_every == 0 or (k + 1) == k_null:
            print(f"  null progress: {k + 1}/{k_null}")

    p_value = float((1 + np.sum(null_corrs >= s_obs)) / (k_null + 1))
    elapsed = time.time() - start_time
    print(f"Completed robustness test in {elapsed:.1f}s.")
    return {
        "s_obs": s_obs,
        "p_value": p_value,
        "obs_corrs": obs_corrs,
        "null_corrs": null_corrs,
    }


def save_outputs(result: dict, out_dir: str, mode: str, perturb_target: str, delta: float):
    tag = f"{mode}_{perturb_target}_delta{delta:.3f}".replace(".", "p")

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(result["null_corrs"], bins=30, color="lightsteelblue", edgecolor="black")
    ax1.axvline(result["s_obs"], color="red", linestyle="--", label=f"Observed mean={result['s_obs']:.3f}")
    ax1.set_title("Null Distribution of Mean Spearman")
    ax1.set_xlabel("Mean Spearman")
    ax1.set_ylabel("Count")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(result["obs_corrs"], bins=20, color="darkseagreen", edgecolor="black")
    ax2.axvline(np.mean(result["obs_corrs"]), color="black", linestyle=":", label="Observed mean")
    ax2.set_title("Observed Spearman Across Perturbations")
    ax2.set_xlabel("Spearman(d, d_delta)")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"robustness_summary_{tag}.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def save_mode_comparison(summary_rows, out_dir: str, perturb_target: str, delta: float):
    tag = f"{perturb_target}_delta{delta:.3f}".replace(".", "p")
    modes = [row["mode"] for row in summary_rows]
    s_obs = [row["s_obs"] for row in summary_rows]
    p_vals = [row["p_value"] for row in summary_rows]

    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(modes, s_obs, color="seagreen", alpha=0.8)
    ax1.set_title("Observed Stability by Mode")
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("Observed Mean Spearman")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(modes, p_vals, color="steelblue", alpha=0.8)
    ax2.axhline(0.05, color="red", linestyle=":", label="alpha=0.05")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("P-value by Mode")
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("One-sided p-value")
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"robustness_mode_comparison_{tag}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Permutation-based robustness test for isodepth stability.")
    parser.add_argument(
        "--modes",
        type=str,
        default="radial",
        help="Comma-separated dataset modes, e.g. radial,noise",
    )
    parser.add_argument("--n", type=int, default=400, help="Number of spatial spots")
    parser.add_argument("--g", type=int, default=10, help="Number of genes")
    parser.add_argument("--delta", type=float, default=0.05, help="Gaussian perturbation magnitude")
    parser.add_argument(
        "--perturb-target",
        choices=["coords", "expression", "both"],
        default="coords",
        help="Where perturbation is applied",
    )
    parser.add_argument("--m-perturb", type=int, default=30, help="Number of perturbed retraining runs (M)")
    parser.add_argument("--k-null", type=int, default=500, help="Number of null permutations (K)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-align-sign",
        action="store_true",
        help="Disable sign alignment between d and d_delta before computing Spearman.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential retraining instead of parallel batched retraining.",
    )

    args = parser.parse_args()

    device = choose_device(prefer_mps=True)
    out_dir = ensure_results_dir("results")
    modes = parse_modes(args.modes)

    print("--- ISODEPTH ROBUSTNESS TEST ---")
    print(
        f"Device={device} modes={modes} perturb={args.perturb_target} delta={args.delta} "
        f"M={args.m_perturb} K={args.k_null} method={'sequential' if args.sequential else 'parallel'}"
    )

    summary_rows = []
    for i, mode in enumerate(modes):
        print(f"\n=== Mode: {mode} ===")
        simulator = SpatialDataSimulator(N=args.n, G=args.g, device=device)
        s, a = simulator.generate(mode=mode, seed=args.seed + i)

        result = robustness_test(
            s=s,
            a=a,
            perturb_target=args.perturb_target,
            delta=args.delta,
            m_perturb=args.m_perturb,
            k_null=args.k_null,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            align_sign=not args.no_align_sign,
            use_parallel=not args.sequential,
            seed=args.seed + i,
            device=device,
        )

        plot_path = save_outputs(result, out_dir, mode, args.perturb_target, args.delta)
        reject = result["p_value"] < args.alpha

        print("Mode result:")
        print(f"  Observed mean stability S_obs: {result['s_obs']:.4f}")
        print(f"  One-sided p-value: {result['p_value']:.6f}")
        print(f"  Reject H0 at alpha={args.alpha}: {reject}")
        print("  Saved files:")
        print(f"    - {plot_path}")

        summary_rows.append(
            {
                "mode": mode,
                "s_obs": float(result["s_obs"]),
                "p_value": float(result["p_value"]),
                "reject_h0": int(reject),
            }
        )

    comp_plot = save_mode_comparison(summary_rows, out_dir, args.perturb_target, args.delta)
    print("\nCross-mode comparison:")
    print(f"  - {comp_plot}")
    for row in summary_rows:
        print(
            f"  {row['mode']}: S_obs={row['s_obs']:.4f}, "
            f"p={row['p_value']:.6f}, reject={bool(row['reject_h0'])}"
        )


if __name__ == "__main__":
    main()
