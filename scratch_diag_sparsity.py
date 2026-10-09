"""Does the selected blur bandwidth track gene sparsity?"""
import sys
import numpy as np
from scipy.stats import spearmanr
from scratch_diag_corrnull import ds, labels, names, coords_all, cfg, _preprocess_separate_subset
from methods.correlated_null_permutation import build_correlated_null_adaptive_surrogates

for t in sys.argv[1:]:
    ti = names.index(t)
    m = labels == ti
    A_c = np.asarray(_preprocess_separate_subset(ds, ds.A[m], ti)[0], dtype=np.float64)
    c = coords_all[m]
    zero_frac = (np.isclose(A_c, A_c.min(0))).mean(0)
    _, diag = build_correlated_null_adaptive_surrogates(
        A_c, c, n_perms=8, seed=cfg.test.seed + ti,
        nn_candidates=list(cfg.test.variogram_nn_candidates),
        n_bins=cfg.test.variogram_n_bins, verbose=False, backend="cpu",
    )
    sc = diag["selection_counts"].astype(float)
    lab = np.asarray(diag["selection_counts_columns"])
    wide = sc[:, lab >= 0.3].sum(1) / sc.sum(1)
    print(f"{t}: Spearman(zero fraction, share of wide picks) = {spearmanr(zero_frac, wide)[0]:.3f}")
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, .9), (.9, 1.01)]:
        g = (zero_frac >= lo) & (zero_frac < hi)
        if g.any():
            print(f"  zero fraction {lo:.2f}-{hi:.2f}: {g.sum():4d} genes, wide-pick share {wide[g].mean():.3f}")
