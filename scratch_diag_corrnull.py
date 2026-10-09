"""Diagnose why correlated-null surrogates are easier to fit than real data."""
import sys
import numpy as np
from data import load_dataset, raw_coordinates_from_standardized
from experiments.configuration import build_run_config
from data.h5ad_loader import preprocess_celltype_subset


def _preprocess_separate_subset(dataset, A_c_raw, type_index):
    A_c_raw = np.asarray(A_c_raw, dtype=np.float32)
    pp = dataset.meta.get("separate_preprocessing")
    if not pp:
        return A_c_raw, None, None
    params = {k: v for k, v in pp.items() if k != "seed"}
    return preprocess_celltype_subset(
        A_c_raw, dataset.meta.get("var_names"),
        seed=int(pp.get("seed", 0)) + int(type_index), **params,
    )
from methods.correlated_null_permutation import build_correlated_null_adaptive_surrogates

cfg = build_run_config("configs/maynard_dlpfc/maynard_dlpfc_151673_correlated_null_adaptive_separate_linear_p99.json", {})
ds = load_dataset(cfg.data, covariate=cfg.test.covariate)
labels = np.asarray(ds.meta["cell_type_labels"])
names = list(ds.meta["cell_type_names"])
coords_all = raw_coordinates_from_standardized(ds.S, ds.meta)


def smooth_rank1_mse(X, c, deg=4):
    """Best rank-1 approx of the projection of X onto a smooth polynomial basis of coords.

    Proxy for a 1-D isodepth + linear decoder: MSE = 1 - (top singular value^2)/total.
    """
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    cs = (c - c.mean(0)) / c.std(0)
    cols = [np.ones(len(c))]
    for i in range(1, deg + 1):
        for j in range(i + 1):
            cols.append(cs[:, 0] ** (i - j) * cs[:, 1] ** j)
    B, _ = np.linalg.qr(np.stack(cols, 1))
    F = B @ (B.T @ X)
    s = np.linalg.svd(F, compute_uv=False)
    tot = (X ** 2).sum()
    return 1 - s[0] ** 2 / tot, 1 - (F ** 2).sum() / tot


def pc1_share(X):
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    s = np.linalg.svd(X, compute_uv=False)
    return s[0] ** 2 / (s ** 2).sum()


for t in (sys.argv[1:] if __name__ == "__main__" else []):
    ti = names.index(t)
    m = labels == ti
    A_c, _, _ = _preprocess_separate_subset(ds, ds.A[m], ti)
    A_c = np.asarray(A_c, dtype=np.float64)
    c = coords_all[m]
    surr, diag = build_correlated_null_adaptive_surrogates(
        A_c, c, n_perms=8, seed=cfg.test.seed + ti,
        nn_candidates=list(cfg.test.variogram_nn_candidates),
        n_bins=cfg.test.variogram_n_bins, verbose=False, backend="cpu",
    )
    r1, full = smooth_rank1_mse(A_c, c)
    print(f"{t} n={m.sum()} G={A_c.shape[1]}")
    print(f"  REAL : rank1-smooth MSE={r1:.4f}  full-smooth MSE={full:.4f}  PC1 share={pc1_share(A_c):.4f}")
    rs = [smooth_rank1_mse(s.astype(np.float64), c) for s in surr]
    pcs = [pc1_share(s.astype(np.float64)) for s in surr]
    print(f"  SURR : rank1-smooth MSE={np.mean([r[0] for r in rs]):.4f} (min {min(r[0] for r in rs):.4f}, max {max(r[0] for r in rs):.4f})"
          f"  full-smooth MSE={np.mean([r[1] for r in rs]):.4f}  PC1 share={np.mean(pcs):.4f}")
    rng = np.random.default_rng(0)
    sh = [smooth_rank1_mse(A_c[rng.permutation(len(c))], c)[0] for _ in range(20)]
    print(f"  SHUFFLE: rank1-smooth MSE mean={np.mean(sh):.4f} min={np.min(sh):.4f}")
    capped = [v for v in cfg.test.variogram_nn_candidates if v <= 0.1]
    surr_c, _ = build_correlated_null_adaptive_surrogates(
        A_c, c, n_perms=8, seed=cfg.test.seed + ti, nn_candidates=capped,
        n_bins=cfg.test.variogram_n_bins, verbose=False, backend="cpu",
    )
    rc = [smooth_rank1_mse(s.astype(np.float64), c)[0] for s in surr_c]
    print(f"  SURR nn<=0.1: rank1-smooth MSE mean={np.mean(rc):.4f} min={np.min(rc):.4f} max={np.max(rc):.4f}")
    sel = diag["selection_counts"].sum(0)
    print("  selection share by nn:", dict(zip(np.round(diag["selection_counts_columns"], 2), np.round(sel / sel.sum(), 3))))
