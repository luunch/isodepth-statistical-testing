"""Per-population histogram of gene sparsity (% of spots with zero counts)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scratch_diag_corrnull import ds, labels, names, _preprocess_separate_subset

OUT = ("results/maynard_dlpfc/maynard_dlpfc_151673_correlated_null_adaptive_separate_gaussian_linear_p99/"
       "maynard_dlpfc_151673_gene_sparsity_histograms.png")

RUN_MIN_CELLS = 3  # min_cells_per_gene used by the run saved in this folder
ds.meta["separate_preprocessing"] = {**ds.meta["separate_preprocessing"], "min_cells_per_gene": RUN_MIN_CELLS}

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
bins = np.linspace(0, 100, 51)
for ax, t in zip(axes.flat, names):
    ti = names.index(t)
    m = labels == ti
    n = int(m.sum())
    A_c = np.asarray(_preprocess_separate_subset(ds, ds.A[m], ti)[0], dtype=np.float64)
    sparsity = 100 * np.isclose(A_c, A_c.min(0)).mean(0)
    cut20 = 100 * (1 - 20 / n)
    kept20 = (sparsity <= cut20).sum()
    kept5 = (sparsity <= 95).sum()
    ax.hist(sparsity, bins=bins, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.axvline(cut20, color="darkorange", ls="--", label=f"detected in >=20 spots: {kept20} kept")
    ax.axvline(95, color="crimson", ls=":", label=f"detected in >=5% of spots: {kept5} kept")
    ax.set_title(f"{t} ({n} spots, {len(sparsity)} genes)\nmedian sparsity {np.median(sparsity):.1f}%")
    ax.set_xlabel("Sparsity (% of spots with zero counts)")
    ax.set_ylabel("Number of genes")
    ax.legend(fontsize=8, loc="upper left")
    print(t, n, len(sparsity), f"median={np.median(sparsity):.1f}", f"kept@20spots={kept20}", f"kept@5%={kept5}")
for ax in axes.flat[len(names):]:
    ax.axis("off")
fig.suptitle(f"DLPFC 151673: per-gene sparsity within each population (genes used by this run, min_cells_per_gene={RUN_MIN_CELLS})")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(OUT)
