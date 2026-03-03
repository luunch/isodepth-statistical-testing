import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from data_manager import SpatialDataSimulator
from gaston_mix_frozen_encoder import closed_form_permutation_test as gaston_mix_test
from permutation_frozen_encoder import frozen_permutation_test as gaston_test

def run_qq_honesty_test(n_trials=30, N=900, G=20, M_gaston_mix=200, M_gaston=20):
    """
    Runs a Q-Q plot analysis to check for statistical honesty (Uniformity of P-values).
    Focuses on the NOISE mode for both methods.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = SpatialDataSimulator(N, G, device=device)
    
    results = []

    print(f"--- STARTING Q-Q HONESTY TEST ({n_trials} Trials on Noise) ---")
    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}")
        S, A = simulator.generate(mode="noise", seed=42+i)
        
        # 1. Test GASTON-Mix
        p_gaston_mix, _, _, _ = gaston_mix_test(S, A, P=3, M=M_gaston_mix)
        results.append({"method": "GASTON-Mix", "p": p_gaston_mix})
        
        # 2. Test GASTON
        p_gaston, _, _, _ = gaston_test(S, A, M=M_gaston)
        results.append({"method": "GASTON", "p": p_gaston})

    df = pd.DataFrame(results)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # 1. Histogram
    plt.subplot(1, 2, 1)
    for method in ["GASTON-Mix", "GASTON"]:
        data = df[df['method'] == method]['p']
        plt.hist(data, bins=10, range=(0, 1), alpha=0.4, label=method, density=True)
    plt.axhline(1, color='black', linestyle='--', label='Ideal Uniform')
    plt.title("P-value Density (Noise)")
    plt.xlabel("P-value")
    plt.ylabel("Density")
    plt.legend()

    # 2. Q-Q Plot
    plt.subplot(1, 2, 2)
    expected_p = np.linspace(0, 1, n_trials)
    
    for method, color in zip(["GASTON-Mix", "GASTON"], ["blue", "orange"]):
        observed_p = np.sort(df[df['method'] == method]['p'])
        plt.plot(expected_p, observed_p, 'o', label=method, color=color, markersize=5)
        
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Calibration')
    
    plt.title("P-value Q-Q Plot (Noise)")
    plt.xlabel("Theoretical Quantiles (Uniform)")
    plt.ylabel("Observed P-values")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("qq_honesty_test.png")
    plt.show()

    # Summary Statistics
    print("\n--- HONESTY SUMMARY (False Positive Rates) ---")
    for method in ["GASTON-Mix", "GASTON"]:
        p_vals = df[df['method'] == method]['p']
        fpr = np.sum(p_vals <= 0.05) / n_trials
        print(f"{method}: Observed FPR at alpha=0.05: {fpr:.2%} (Target: 5.00%)")

if __name__ == "__main__":
    # n_trials=30 provides a much clearer Q-Q curve
    run_qq_honesty_test(n_trials=30, M_gaston_mix=200, M_gaston=20)
