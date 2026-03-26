import argparse
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_manager import SpatialDataSimulator
from full_retraining_gaston import full_retraining_permutation_test
from isodepth import choose_device, ensure_results_dir, set_global_seed
from parallel_full_retraining import run_parallel_permutation_test


def parse_list(raw: str, cast=int) -> list:
    items = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            items.append(cast(x))
    return items


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["mode", "M"], as_index=False)
        .agg(
            mean_seq_p=("p_seq", "mean"),
            mean_par_p=("p_par", "mean"),
            mean_abs_diff=("abs_diff", "mean"),
            max_abs_diff=("abs_diff", "max"),
            corr=("pair_key", lambda _: np.nan),
            sig_agreement=("sig_agree", "mean"),
        )
    )

    # Correlation per (mode, M) separately for clarity.
    corrs = []
    for (mode, m), block in df.groupby(["mode", "M"]):
        if len(block) > 1:
            corr = float(np.corrcoef(block["p_seq"], block["p_par"])[0, 1])
        else:
            corr = np.nan
        corrs.append({"mode": mode, "M": m, "corr": corr})

    corr_df = pd.DataFrame(corrs)
    grouped = grouped.drop(columns=["corr"]).merge(corr_df, on=["mode", "M"], how="left")
    return grouped


def plot_results(df: pd.DataFrame, out_dir: str) -> None:
    # 1) Scatter: sequential vs parallel, faceted by mode
    modes = sorted(df["mode"].unique())
    n_modes = len(modes)

    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), squeeze=False)
    for i, mode in enumerate(modes):
        ax = axes[0, i]
        sub = df[df["mode"] == mode]
        sns.scatterplot(data=sub, x="p_seq", y="p_par", hue="M", palette="viridis", s=70, ax=ax)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"P-value Agreement ({mode})")
        ax.set_xlabel("Sequential p-value")
        ax.set_ylabel("Parallel p-value")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pvalue_agreement_scatter.png"))
    plt.close()

    # 2) Absolute difference vs M and mode
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="M", y="abs_diff", hue="mode")
    plt.title("|Sequential - Parallel| by Permutation Count")
    plt.xlabel("M (number of permutations)")
    plt.ylabel("Absolute p-value difference")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pvalue_absdiff_by_m.png"))
    plt.close()

    # 3) Mean p-values across methods and modes
    long_df = pd.melt(
        df,
        id_vars=["mode", "M", "trial"],
        value_vars=["p_seq", "p_par"],
        var_name="method",
        value_name="p_value",
    )
    long_df["method"] = long_df["method"].map({"p_seq": "Sequential", "p_par": "Parallel"})

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=long_df, x="M", y="p_value", hue="method", style="mode", markers=True, dashes=False)
    plt.title("P-value Trends: Sequential vs Parallel")
    plt.xlabel("M (number of permutations)")
    plt.ylabel("P-value")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pvalue_trends_seq_vs_par.png"))
    plt.close()


def run_benchmark(
    modes: Iterable[str],
    m_values: Iterable[int],
    trials: int,
    n: int,
    g: int,
    epochs: int,
    alpha: float,
    dataset_seed: int,
) -> pd.DataFrame:
    device = choose_device(prefer_mps=True)
    print(f"Using device: {device}")

    rows = []

    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        for m in m_values:
            print(f"  M={m}")
            for trial in range(trials):
                # Build paired dataset once per trial.
                data_seed = dataset_seed + trial
                set_global_seed(data_seed)
                simulator = SpatialDataSimulator(N=n, G=g, device=device)
                S, A = simulator.generate(mode=mode, seed=data_seed)

                # Keep method randomness controlled but independent from data sampling.
                seq_seed = 100_000 + 1000 * m + trial
                par_seed = 200_000 + 1000 * m + trial

                set_global_seed(seq_seed)
                p_seq, _, _, _ = full_retraining_permutation_test(S, A, M=m, epochs=epochs)

                set_global_seed(par_seed)
                p_par, _, _, _ = run_parallel_permutation_test(S, A, M=m + 1, epochs=epochs)

                abs_diff = abs(float(p_seq) - float(p_par))
                sig_agree = int((p_seq <= alpha) == (p_par <= alpha))

                rows.append(
                    {
                        "mode": mode,
                        "M": int(m),
                        "trial": int(trial),
                        "pair_key": f"{mode}|{m}|{trial}",
                        "p_seq": float(p_seq),
                        "p_par": float(p_par),
                        "abs_diff": float(abs_diff),
                        "sig_agree": sig_agree,
                    }
                )
                print(
                    f"    trial={trial} p_seq={p_seq:.4f} p_par={p_par:.4f} "
                    f"|diff|={abs_diff:.4f} agree@{alpha}={bool(sig_agree)}"
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark p-value agreement between sequential and parallel tests.")
    parser.add_argument("--modes", type=str, default="noise,radial,checkerboard", help="Comma-separated modes")
    parser.add_argument("--m-values", type=str, default="20,50,100", help="Comma-separated permutation counts")
    parser.add_argument("--trials", type=int, default=5, help="Trials per (mode, M)")
    parser.add_argument("--n", type=int, default=400, help="Number of spots")
    parser.add_argument("--g", type=int, default=10, help="Number of genes")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--dataset-seed", type=int, default=42, help="Base seed for dataset generation")
    args = parser.parse_args()

    modes = parse_list(args.modes, cast=str)
    m_values = parse_list(args.m_values, cast=int)

    out_dir = ensure_results_dir("results")

    print("--- P-VALUE AGREEMENT BENCHMARK ---")
    print(
        f"Modes={modes} | M={m_values} | Trials={args.trials} | N={args.n} G={args.g} | Epochs={args.epochs}"
    )

    df = run_benchmark(
        modes=modes,
        m_values=m_values,
        trials=args.trials,
        n=args.n,
        g=args.g,
        epochs=args.epochs,
        alpha=args.alpha,
        dataset_seed=args.dataset_seed,
    )

    summary = summarize(df)
    plot_results(df, out_dir)

    print("\nSaved files:")
    print(f"  - {os.path.join(out_dir, 'pvalue_agreement_scatter.png')}")
    print(f"  - {os.path.join(out_dir, 'pvalue_absdiff_by_m.png')}")
    print(f"  - {os.path.join(out_dir, 'pvalue_trends_seq_vs_par.png')}")

    print("\nSummary table:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
