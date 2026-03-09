import time
import torch
import numpy as np
import pandas as pd
from data_manager import SpatialDataSimulator

# Import both methods
from full_retraining_gaston import full_retraining_permutation_test as sequential_test
from parallel_full_retraining import run_parallel_permutation_test as parallel_test

def benchmark_retraining(M=20, N=400, G=10, epochs=5000):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--- BENCHMARKING RETRAINING METHODS (M={M}, N={N}, G={G}, Epochs={epochs}) ---")

    print(f"Running on: {device}")
    
    simulator = SpatialDataSimulator(N, G, device=device)
    S, A = simulator.generate(mode="noise", seed=42)
    
    # 1. Sequential Benchmark
    print("\nStarting Sequential Retraining...")
    start_seq = time.time()
    p_seq, L_true_seq, L_perm_seq, _ = sequential_test(S, A, M=M, epochs=epochs)
    end_seq = time.time()
    duration_seq = end_seq - start_seq
    
    # 2. Parallel Benchmark
    print("\nStarting Parallel (Batched) Retraining...")
    start_par = time.time()
    # Note: parallel_test takes M as total models (including true), so we pass M+1 to match sequential's M permutations
    p_par, L_true_par, L_perm_par, _ = parallel_test(S, A, M=M+1, epochs=epochs)
    end_par = time.time()
    duration_par = end_par - start_par
    
    # Summary Table
    results = {
        "Method": ["Sequential", "Parallel (Batched)"],
        "Total Time (s)": [duration_seq, duration_par],
        "Time per Permutation (s)": [duration_seq / M, duration_par / M],
        "Speedup": [1.0, duration_seq / duration_par],
        "P-value": [p_seq, p_par]
    }
    
    df = pd.DataFrame(results)
    print("\n--- BENCHMARK RESULTS ---")
    print(df.to_string(index=False))
    
    print(f"\nObservation:")
    print(f"Parallel retraining is {df.iloc[1]['Speedup']:.2f}x faster than sequential retraining.")
    
    # Consistency Check
    print(f"\nConsistency Check:")
    print(f"Sequential True NLL: {L_true_seq:.4f}")
    print(f"Parallel True NLL:   {L_true_par:.4f}")
    print(f"Difference:          {abs(L_true_seq - L_true_par):.4f}")

if __name__ == "__main__":
    # We use a smaller M and epochs for a quick benchmark
    benchmark_retraining(M=100, N=400, G=10, epochs=5000)