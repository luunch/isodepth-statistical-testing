import json, numpy as np
from scipy.stats import spearmanr
from data.schemas import DataConfig
from data import load_dataset

with open("results/hypothalamus_existence_one_perm/hypothalamus_existence_one_perm_result.json") as f:
    saved = json.load(f)
arts = saved["artifacts"]
iso = np.array(arts["true_isodepth"], dtype=np.float64).reshape(-1)
cov = np.array(arts["true_isodepth_covariate"], dtype=np.float64).reshape(-1)

cfg = json.load(open("configs/hypothalamus_existence.json"))
dataset = load_dataset(DataConfig(**cfg["data"]))
A = np.asarray(dataset.A, dtype=np.float64)
names = list(dataset.meta.get("var_names") or [f"gene_{i}" for i in range(A.shape[1])])
G = A.shape[1]

rhos_iso = np.array([spearmanr(A[:, g], iso).statistic for g in range(G)])
rhos_cov = np.array([spearmanr(A[:, g], cov).statistic for g in range(G)])

print(f"N genes: {G}")
print(f"\n--- PER-GENE |rho| summary ---")
print(f"Isodepth:  mean={np.abs(rhos_iso).mean():.4f}  median={np.median(np.abs(rhos_iso)):.4f}  max={np.abs(rhos_iso).max():.4f}")
print(f"Midline:   mean={np.abs(rhos_cov).mean():.4f}  median={np.median(np.abs(rhos_cov)):.4f}  max={np.abs(rhos_cov).max():.4f}")

# For each gene, which coordinate has higher |rho|?
iso_wins = np.sum(np.abs(rhos_iso) > np.abs(rhos_cov))
cov_wins = G - iso_wins
print(f"\n--- Which coordinate has higher |rho| per gene? ---")
print(f"Isodepth wins: {iso_wins}/{G} genes ({100*iso_wins/G:.1f}%)")
print(f"Midline wins:  {cov_wins}/{G} genes ({100*cov_wins/G:.1f}%)")

# proxy for NLL reduction: sum of rho^2 (variance explained) across all genes
var_exp_iso = np.sum(rhos_iso**2)
var_exp_cov = np.sum(rhos_cov**2)
print(f"\n--- Sum of rho^2 across all {G} genes (proxy for joint reconstruction) ---")
print(f"Isodepth:  {var_exp_iso:.3f}")
print(f"Midline:   {var_exp_cov:.3f}")
print(f"Ratio (iso/cov): {var_exp_iso/var_exp_cov:.2f}x")

print(f"\n--- Top-5 genes by iso |rho| and their midline |rho| ---")
for i in np.argsort(-np.abs(rhos_iso))[:5]:
    marker = "<<" if abs(rhos_cov[i]) > abs(rhos_iso[i]) else ""
    print(f"  {names[i]:<20}  rho_iso={rhos_iso[i]:+.4f}  rho_mid={rhos_cov[i]:+.4f}  {marker}")
