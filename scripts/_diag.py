import json, numpy as np
from scipy.stats import spearmanr, pearsonr
from data.schemas import DataConfig
from data import load_dataset

with open("results/hypothalamus_existence_one_perm/hypothalamus_existence_one_perm_result.json") as f:
    saved = json.load(f)
arts = saved["artifacts"]
iso = np.array(arts["true_isodepth"], dtype=np.float64)
cov = np.array(arts["true_isodepth_covariate"], dtype=np.float64)

cfg = json.load(open("configs/hypothalamus_existence.json"))
dataset = load_dataset(DataConfig(**cfg["data"]))
A = np.asarray(dataset.A, dtype=np.float64)
names = list(dataset.meta.get("var_names") or [f"gene_{i}" for i in range(A.shape[1])])

rhos_iso = np.array([spearmanr(A[:, g], iso).statistic for g in range(A.shape[1])])
rhos_cov = np.array([spearmanr(A[:, g], cov).statistic for g in range(A.shape[1])])

print(f"Isodepth  range=[{iso.min():.3f}, {iso.max():.3f}]  std={iso.std():.3f}")
print(f"Midline   range=[{cov.min():.3f}, {cov.max():.3f}]  std={cov.std():.3f}")
print(f"Pearson(iso, midline) = {pearsonr(iso, cov).statistic:.4f}")

print("\n-- Top-5 genes vs ISODEPTH --")
for i in np.argsort(-np.abs(rhos_iso))[:5]:
    print(f"  {names[i]:<30}  rho={rhos_iso[i]:+.4f}")
print(f"  distribution: max={np.abs(rhos_iso).max():.4f}  p90={np.percentile(np.abs(rhos_iso),90):.4f}  median={np.median(np.abs(rhos_iso)):.4f}")
print(f"  genes |rho|>0.3: {(np.abs(rhos_iso)>0.3).sum()}  |rho|>0.2: {(np.abs(rhos_iso)>0.2).sum()}  |rho|>0.1: {(np.abs(rhos_iso)>0.1).sum()}")

print("\n-- Top-5 genes vs MIDLINE --")
for i in np.argsort(-np.abs(rhos_cov))[:5]:
    print(f"  {names[i]:<30}  rho={rhos_cov[i]:+.4f}")
print(f"  distribution: max={np.abs(rhos_cov).max():.4f}  p90={np.percentile(np.abs(rhos_cov),90):.4f}  median={np.median(np.abs(rhos_cov)):.4f}")
print(f"  genes |rho|>0.3: {(np.abs(rhos_cov)>0.3).sum()}  |rho|>0.2: {(np.abs(rhos_cov)>0.2).sum()}  |rho|>0.1: {(np.abs(rhos_cov)>0.1).sum()}")
