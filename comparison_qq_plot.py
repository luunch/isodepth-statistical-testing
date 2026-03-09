import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import kstest
from data_manager import SpatialDataSimulator

# Import the three different test methods
from gaston_mix_frozen_encoder import closed_form_permutation_test as gaston_mix_test
from permutation_frozen_encoder import frozen_permutation_test as gaston_test
from full_retraining_gaston import full_retraining_permutation_test as full_retrain_test

def run_comparison_qq(methods=['gaston_frozen', 'gaston_mix_frozen', 'full_retrain'], 
                      n_trials=20, N=400, G=10, 
                      M_gaston_mix=100, M_gaston=20, M_full=20):
    """
    Runs a Q-Q plot analysis for a specified subset of GASTON permutation test methods.
    Checks for p-value calibration under the null hypothesis (Gaussian Noise).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = SpatialDataSimulator(N, G, device=device)
    
    print(f"--- PREPARING BASE DATA (Gaussian Noise) ---")
    S, A_base = simulator.generate(mode="noise", seed=42)

    results = []
    print(f"\n--- STARTING COMPARISON Q-Q ANALYSIS ({n_trials} Trials) ---")
    
    for i in range(n_trials):
        print(f"\nTrial {i+1}/{n_trials}")
        
        # Create a new null dataset for each trial by permuting the noise
        perm = np.random.permutation(S.shape[0])
        A = A_base[perm]
        
        # Center A for this trial
        A = A - A.mean(axis=0, keepdims=True)
        
        # 1. Frozen Encoder GASTON-Mix
        if 'gaston_mix_frozen' in methods:
            p_mix, _, _, _ = gaston_mix_test(S, A, P=2, M=M_gaston_mix)
            results.append({"method": "GASTON-Mix (Frozen)", "p": p_mix})
        
        # 2. Frozen Encoder GASTON
        if 'gaston_frozen' in methods:
            p_gaston, _, _, _ = gaston_test(S, A, M=M_gaston)
            results.append({"method": "GASTON (Frozen)", "p": p_gaston})
            
        # 3. Full Retraining GASTON
        if 'full_retrain' in methods:
            # Full retraining is slow, so we use lower M and epochs if needed
            p_full, _, _, _ = full_retrain_test(S, A, M=M_full, epochs=5000)
            results.append({"method": "GASTON (Full Retrain)", "p": p_full})

    df = pd.DataFrame(results)
    
    # Pre-calculate K-S tests for each method
    ks_results = {}
    for method in df['method'].unique():
        p_vals = df[df['method'] == method]['p']
        res = kstest(p_vals, 'uniform')
        ks_results[method] = res

    # Visualization
    plt.figure(figsize=(14, 6))
    
    # 1. P-value Histograms
    plt.subplot(1, 2, 1)
    for method in df['method'].unique():
        data = df[df['method'] == method]['p']
        label = f"{method} (KS p={ks_results[method].pvalue:.4f})"
        plt.hist(data, bins=10, range=(0, 1), alpha=0.3, label=label, density=True)
    plt.axhline(1, color='black', linestyle='--', label='Ideal Uniform')
    plt.title(f"P-value Density (Null Distribution)")
    plt.xlabel("P-value")
    plt.ylabel("Density")
    plt.legend()

    # 2. Q-Q Plot
    plt.subplot(1, 2, 2)
    expected_p = np.linspace(0, 1, n_trials)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['method'].unique())))
    for (method, color) in zip(df['method'].unique(), colors):
        observed_p = np.sort(df[df['method'] == method]['p'])
        label = f"{method} (KS p={ks_results[method].pvalue:.4f})"
        plt.plot(expected_p, observed_p, 'o-', label=label, color=color, markersize=5, alpha=0.7)
        
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Calibration')
    
    plt.title(f"P-value Q-Q Plot ({n_trials} Trials)")
    plt.xlabel("Theoretical Quantiles (Uniform)")
    plt.ylabel("Observed P-values")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    import os
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison_qq_plot.png")
    print("\nSaved comparison Q-Q plot to 'results/comparison_qq_plot.png'")
    
    # Summary Statistics
    print(f"\n--- CALIBRATION SUMMARY ---")
    for method in df['method'].unique():
        p_vals = df[df['method'] == method]['p']
        fpr = np.sum(p_vals <= 0.05) / n_trials
        ks_p = ks_results[method].pvalue
        print(f"{method}:")
        print(f"  - Observed False Positive Rate at alpha=0.05: {fpr:.2%}")
        print(f"  - Kolmogorov-Smirnov test (for Uniformity): p={ks_p:.4f}")

if __name__ == "__main__":
    # Example: Specify which methods to compare
    # Choices: 'gaston_frozen', 'gaston_mix_frozen', 'full_retrain'
    my_methods = ['gaston_frozen', 'gaston_mix_frozen', 'full_retrain']
    
    run_comparison_qq(
        methods=my_methods, 
        n_trials=50,      # Small number for quick verification
        N=400, 
        G=10, 
        M_gaston_mix=100, 
        M_gaston=30, 
        M_full=30
    )
