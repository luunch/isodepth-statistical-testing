import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from data_manager import SpatialDataSimulator
from gaston_mix_frozen_encoder import closed_form_permutation_test

def run_power_test(n_trials=20, N=900, G=20, M=200, P=3):
    """
    Runs the Negative Control (pure noise) multiple times to check 
    if the p-value distribution is Uniform(0, 1).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = SpatialDataSimulator(N, G, device=device)
    
    p_values = []
    
    print(f"Starting Power Test: {n_trials} trials on Negative Control (Noise)")
    print(f"Each trial uses M={M} permutations.")
    
    for i in range(n_trials):
        # Generate new random noise for each trial
        S, A = simulator.generate(positive=False, seed=42 + i)
        
        # Run the permutation test
        p, _, _, _ = closed_form_permutation_test(S, A, P=P, M=M)
        p_values.append(p)
        
        print(f"Trial {i+1}/{n_trials} | P-value: {p:.4f}")

    # Visualization
    plt.figure(figsize=(10, 5))
    
    # 1. Histogram of P-values
    plt.subplot(1, 2, 1)
    plt.hist(p_values, bins=10, range=(0, 1), color='skyblue', edgecolor='black', alpha=0.7)
    plt.axhline(n_trials/10, color='red', linestyle='--', label='Theoretical Uniform')
    plt.title("P-value Distribution (Noise)")
    plt.xlabel("P-value")
    plt.ylabel("Frequency")
    plt.legend()

    # 2. Cumulative Distribution (Q-Q Plot style)
    plt.subplot(1, 2, 2)
    sorted_p = np.sort(p_values)
    expected_p = np.linspace(0, 1, len(p_values))
    plt.plot(expected_p, sorted_p, marker='o', linestyle='none', color='blue')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title("Cumulative P-value Distribution")
    plt.xlabel("Expected Quantile")
    plt.ylabel("Observed P-value")

    plt.tight_layout()
    plt.savefig("power_test_results.png")
    plt.show()

    # Statistical Check
    n_sig = np.sum(np.array(p_values) <= 0.05)
    false_positive_rate = n_sig / n_trials
    print(f"\n--- POWER TEST SUMMARY ---")
    print(f"Total Trials: {n_trials}")
    print(f"Significant results (p <= 0.05): {n_sig}")
    print(f"Observed False Positive Rate: {false_positive_rate:.2%}")
    print(f"Expected False Positive Rate: 5.00%")
    
    if false_positive_rate > 0.15:
        print("WARNING: The test may be OVERFITTING. P-values are clustering too low.")
    else:
        print("SUCCESS: The test appears statistically honest.")

if __name__ == "__main__":
    # We use fewer permutations (M=200) and trials (20) to keep it fast
    run_power_test(n_trials=20, M=200)
