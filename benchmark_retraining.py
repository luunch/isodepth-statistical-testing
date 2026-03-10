import time
import torch
import numpy as np
import pandas as pd
from data_manager import SpatialDataSimulator

# Import both methods
from full_retraining_gaston import full_retraining_permutation_test as sequential_test
from parallel_full_retraining import run_parallel_permutation_test as parallel_test

def benchmark_retraining(M=20, N=400, G=10, epochs=5000, mode="both"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--- BENCHMARKING RETRAINING METHODS (M={M}, N={N}, G={G}, Epochs={epochs}, Mode={mode}) ---")

    print(f"Running on: {device}")
    
    simulator = SpatialDataSimulator(N, G, device=device)
    S, A = simulator.generate(mode="noise", seed=42)
    
    duration_seq, duration_par = None, None
    p_seq, L_true_seq = None, None
    p_par, L_true_par = None, None
    
    # 1. Sequential Benchmark
    if mode in ["sequential", "both"]:
        print("\nStarting Sequential Retraining...")
        start_seq = time.time()
        p_seq, L_true_seq, _, _ = sequential_test(S, A, M=M, epochs=epochs)
        end_seq = time.time()
        duration_seq = end_seq - start_seq
    
    # 2. Parallel Benchmark
    if mode in ["parallel", "both"]:
        print("\nStarting Parallel (Batched) Retraining...")
        start_par = time.time()
        # Note: parallel_test takes M as total models (including true), so we pass M+1 to match sequential's M permutations
        p_par, L_true_par, _, _ = parallel_test(S, A, M=M+1, epochs=epochs)
        end_par = time.time()
        duration_par = end_par - start_par
    
    # Summary Table
    results = {"Method": [], "Total Time (s)": [], "Time per Permutation (s)": [], "P-value": []}
    
    if duration_seq is not None:
        results["Method"].append("Sequential")
        results["Total Time (s)"].append(duration_seq)
        results["Time per Permutation (s)"].append(duration_seq / M)
        results["P-value"].append(p_seq)
        
    if duration_par is not None:
        results["Method"].append("Parallel (Batched)")
        results["Total Time (s)"].append(duration_par)
        results["Time per Permutation (s)"].append(duration_par / M)
        results["P-value"].append(p_par)
        
    df = pd.DataFrame(results)
    print("\n--- BENCHMARK RESULTS ---")
    print(df.to_string(index=False))
    
    if mode == "both":
        speedup = duration_seq / duration_par
        print(f"\nObservation:")
        print(f"Parallel retraining is {speedup:.2f}x faster than sequential retraining.")
        
        # Consistency Check
        print(f"\nConsistency Check:")
        print(f"Sequential True NLL: {L_true_seq:.4f}")
        print(f"Parallel True NLL:   {L_true_par:.4f}")
        print(f"Difference:          {abs(L_true_seq - L_true_par):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Retraining Methods")
    parser.add_argument("--mode", type=str, choices=["sequential", "parallel", "both"], default="both", help="Which method(s) to benchmark")
    parser.add_argument("--M", type=int, default=100, help="Number of permutations")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs to train")
    
    args = parser.parse_args()
    benchmark_retraining(M=args.M, N=400, G=10, epochs=args.epochs, mode=args.mode)