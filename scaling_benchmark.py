import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from data_manager import SpatialDataSimulator
from isodepth import choose_device, ensure_results_dir

# Define worker functions at the top level so they can be pickled by multiprocessing
def run_sequential_worker(q, S, A, M, epochs):
    from full_retraining_gaston import full_retraining_permutation_test
    import time
    start = time.time()
    try:
        p, L_true, L_perm, _ = full_retraining_permutation_test(S, A, M=M, epochs=epochs)
        q.put(("success", time.time() - start, p))
    except Exception as e:
        q.put(("error", str(e), None))

def run_parallel_worker(q, S, A, M, epochs):
    from parallel_full_retraining import run_parallel_permutation_test
    import time
    start = time.time()
    try:
        # Pass M+1 to parallel_test to match sequential's M permutations + 1 true
        p, L_true, L_perm, _ = run_parallel_permutation_test(S, A, M=M+1, epochs=epochs)
        q.put(("success", time.time() - start, p))
    except Exception as e:
        q.put(("error", str(e), None))

def run_with_timeout(worker_func, args, timeout_secs):
    """Runs a function in a separate process and kills it if it exceeds timeout_secs."""
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    
    p = ctx.Process(target=worker_func, args=(q, *args))
    p.start()
    p.join(timeout_secs)
    
    if p.is_alive():
        print(f"  [!] Timeout reached (> {timeout_secs}s). Terminating process.")
        p.terminate()
        p.join()
        return None, None  # Represents a timeout
    
    if not q.empty():
        status, val1, val2 = q.get()
        if status == "success":
            return val1, val2 # Time, p-value
        else:
            print(f"  [!] Error in process: {val1}")
            return None, None
    return None, None

def run_scaling_benchmark(N=400, G=10, epochs=3000, timeout=120, mode="noise"):
    device = choose_device(prefer_mps=True)
    print(f"--- SCALING BENCHMARK (N={N}, G={G}, Epochs={epochs}, Timeout={timeout}s, Mode={mode}) ---")
    print(f"Detected Device (for info): {device}\n")

    # Generate the single dataset to evaluate
    simulator = SpatialDataSimulator(N, G, device=device)
    S, A = simulator.generate(mode=mode, seed=42)
    
    # We use smaller epochs to allow hitting higher Ms within the 2-minute window
    M_values = [10, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
    
    results = []
    
    # Track if a method has timed out so we can skip higher M values for it
    skip_seq = False
    skip_par = False
    
    for M in M_values:
        print(f"\nEvaluating M = {M} permutations...")
        
        # --- Sequential ---
        if not skip_seq:
            print("  Running Sequential...")
            seq_time, seq_p = run_with_timeout(run_sequential_worker, (S, A, M, epochs), timeout)
            if seq_time is None:
                skip_seq = True
                results.append({"M": M, "Method": "Sequential", "Time (s)": np.nan, "P-value": np.nan, "Status": "Timeout"})
            else:
                results.append({"M": M, "Method": "Sequential", "Time (s)": seq_time, "P-value": seq_p, "Status": "Completed"})
                print(f"  -> Finished in {seq_time:.2f}s (p={seq_p:.4f})")
        else:
            results.append({"M": M, "Method": "Sequential", "Time (s)": np.nan, "P-value": np.nan, "Status": "Skipped (Prev Timeout)"})
            
        # --- Parallel ---
        if not skip_par:
            print("  Running Parallel...")
            par_time, par_p = run_with_timeout(run_parallel_worker, (S, A, M, epochs), timeout)
            if par_time is None:
                skip_par = True
                results.append({"M": M, "Method": "Parallel", "Time (s)": np.nan, "P-value": np.nan, "Status": "Timeout"})
            else:
                results.append({"M": M, "Method": "Parallel", "Time (s)": par_time, "P-value": par_p, "Status": "Completed"})
                print(f"  -> Finished in {par_time:.2f}s (p={par_p:.4f})")
        else:
            results.append({"M": M, "Method": "Parallel", "Time (s)": np.nan, "P-value": np.nan, "Status": "Skipped (Prev Timeout)"})

    # --- Process Results & Plot ---
    df = pd.DataFrame(results)
    print("\n--- FINAL RESULTS ---")
    print(df.to_string(index=False))
    
    ensure_results_dir("results")
    
    # Plot 1: Execution Time vs M
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[df["Status"] == "Completed"], x="M", y="Time (s)", hue="Method", marker="o", linewidth=2.5, markersize=8)
    plt.axhline(y=timeout, color='red', linestyle='--', label=f'Timeout ({timeout}s)')
    plt.title(f"Scaling Performance: Sequential vs Parallel (Epochs={epochs})")
    plt.xlabel("Number of Permutations (M)")
    plt.ylabel("Execution Time (Seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    time_plot_path = "results/scaling_time_comparison.png"
    plt.savefig(time_plot_path)
    plt.close()
    
    # Plot 2: P-values vs M
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df[df["Status"] == "Completed"], x="M", y="P-value", hue="Method", s=100, alpha=0.8)
    sns.lineplot(data=df[df["Status"] == "Completed"], x="M", y="P-value", hue="Method", alpha=0.3, legend=False)
    plt.axhline(y=0.05, color='red', linestyle=':', label='Significance Threshold (0.05)')
    plt.title("Stability of P-value across M Permutations")
    plt.xlabel("Number of Permutations (M)")
    plt.ylabel("Calculated P-value")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    pval_plot_path = "results/scaling_pvalues.png"
    plt.savefig(pval_plot_path)
    plt.close()

    # Plot 3: Sequential vs Parallel p-value agreement
    completed = df[df["Status"] == "Completed"]
    paired = completed.pivot(index="M", columns="Method", values="P-value").dropna()
    agreement_plot_path = "results/scaling_pvalue_agreement.png"
    agreement_summary = None

    if {"Sequential", "Parallel"}.issubset(set(completed["Method"].unique())) and not paired.empty:
        seq_vals = paired["Sequential"].to_numpy()
        par_vals = paired["Parallel"].to_numpy()
        delta = np.abs(seq_vals - par_vals)

        if len(seq_vals) > 1:
            corr = float(np.corrcoef(seq_vals, par_vals)[0, 1])
        else:
            corr = float("nan")

        agreement_summary = {
            "mean_abs_diff": float(np.mean(delta)),
            "max_abs_diff": float(np.max(delta)),
            "corr": corr,
            "n_points": int(len(delta)),
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: direct agreement scatter with identity line
        axes[0].scatter(seq_vals, par_vals, s=70, alpha=0.8, color="teal")
        axes[0].plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Ideal y=x")
        axes[0].set_xlim(-0.02, 1.02)
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_xlabel("Sequential p-value")
        axes[0].set_ylabel("Parallel p-value")
        axes[0].set_title("P-value Agreement")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        # Right: absolute difference vs permutation count
        axes[1].plot(paired.index.to_numpy(), delta, marker="o", linewidth=2, color="darkorange")
        axes[1].set_xlabel("Number of Permutations (M)")
        axes[1].set_ylabel("|Sequential - Parallel|")
        axes[1].set_title("Absolute P-value Difference Across M")
        axes[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(agreement_plot_path)
        plt.close()
    
    print(f"\nPlots saved to '{time_plot_path}' and '{pval_plot_path}'")
    if agreement_summary is not None:
        print(f"Agreement plot saved to '{agreement_plot_path}'")
        print("P-value agreement summary:")
        print(f"  - Paired points: {agreement_summary['n_points']}")
        print(f"  - Mean |difference|: {agreement_summary['mean_abs_diff']:.4f}")
        print(f"  - Max |difference|:  {agreement_summary['max_abs_diff']:.4f}")
        if not np.isnan(agreement_summary["corr"]):
            print(f"  - Correlation:       {agreement_summary['corr']:.4f}")
    else:
        print("Agreement plot skipped (need completed results for both Sequential and Parallel at the same M values).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scale Testing with Timeouts")
    parser.add_argument("--epochs", type=int, default=3000, help="Epochs per training run")
    parser.add_argument("--timeout", type=int, default=120, help="Max time allowed per run in seconds")
    parser.add_argument("--mode", type=str, choices=["noise", "radial", "checkerboard"], default="noise", help="Dataset mode")
    
    args = parser.parse_args()
    
    # Required for multiprocessing on some platforms/CUDA
    mp.set_start_method('spawn', force=True)
    run_scaling_benchmark(epochs=args.epochs, timeout=args.timeout, mode=args.mode)
