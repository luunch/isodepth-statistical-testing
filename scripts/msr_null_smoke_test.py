"""MSR null utilities — shared across smoke-test scripts."""
from __future__ import annotations
import sys, time
from dataclasses import replace
from pathlib import Path

import anndata as ad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.schemas import TestConfig
from methods.metrics import permutation_p_value
from methods.trainers import resolve_device, train_parallel_isodepth_model

N_PERMS   = 19
N_RERUNS  = 3
EPOCHS    = 150
SEED      = 42
K_NEIGHBORS = 6

H5AD   = REPO / "data/h5ad/starmap_mvc_BY3.h5ad"
OUTDIR = REPO / "results/msr_null_smoke"
OUTDIR.mkdir(parents=True, exist_ok=True)


def base_cfg(n_perms: int = N_PERMS, seed: int = SEED) -> TestConfig:
    return TestConfig(
        method="parallel_permutation", metric="nll_gaussian_mse",
        n_perms=n_perms, epochs=EPOCHS, n_reruns=N_RERUNS,
        sgd_batch_size=128, lr=1e-3, seed=seed, device="cuda",
        decoder="nn", verbose=False,
    )


def load_starmap(depth_filter=None):
    adata = ad.read_h5ad(H5AD)
    if depth_filter is not None:
        adata = adata[adata.obs["cortical_depth"] == depth_filter].copy()
    import scipy.sparse as sp
    raw = adata.layers["counts"]
    counts = np.array(raw.toarray() if sp.issparse(raw) else raw, dtype=np.float32)
    S_raw  = np.array(adata.obsm["spatial"], dtype=np.float32)
    row_sums = counts.sum(axis=1, keepdims=True).clip(1.0)
    A = np.log1p(counts / row_sums * 1e6)
    A = ((A - A.mean(0, keepdims=True)) / A.std(0, keepdims=True).clip(1e-8)).astype(np.float32)
    S = ((S_raw - S_raw.mean(0, keepdims=True)) / S_raw.std(0, keepdims=True).clip(1e-8)).astype(np.float32)
    lbl = f"L5 only (N={len(adata)})" if depth_filter is not None else f"full cortex (N={len(adata)})"
    return S, A, lbl


def _msr_basis(S, A, k=K_NEIGHBORS, return_eigvals: bool = False):
    """Build kNN graph → Omega → eigendecomposition → (V, C) [+ eigvals]."""
    from sklearn.neighbors import NearestNeighbors
    N, G = A.shape
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(S)
    _, idxs = nbrs.kneighbors(S)
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in idxs[i, 1:]:
            W[i, j] = 1.0; W[j, i] = 1.0
    rm = W.mean(1, keepdims=True); cm = W.mean(0, keepdims=True); gm = W.mean()
    Omega = W - rm - cm + gm
    t = time.time()
    eigvals, eigvecs = np.linalg.eigh(Omega)
    print(f"    Eigendecomp: {time.time()-t:.1f}s", flush=True)
    keep = np.abs(eigvals) > 1e-8
    eig_kept = eigvals[keep]
    V = eigvecs[:, keep]
    print(f"    K={V.shape[1]} non-trivial MEMs ({eig_kept.min():.3f} … {eig_kept.max():.3f})", flush=True)
    C = V.T @ A.astype(np.float64)
    if return_eigvals:
        return V, C, eig_kept
    return V, C


def _mem_characteristic_scales_um(
    eigvals: np.ndarray,
    calibration_um: float,
) -> np.ndarray:
    """
    Map Moran eigenvalues to an approximate spatial scale in microns.

    Low |λ| → smooth / long-range MEM; high |λ| → local / short-range MEM.
    scale_um ≈ calibration_um / sqrt(|λ|).
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    return calibration_um / np.sqrt(np.clip(np.abs(eigvals), 1e-8, None))


def _joint_trunc_mode_mask(
    eigvals: np.ndarray,
    truncate_scale_um: float,
    calibration_um: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (scales_um, rand_mask, calibration_um used)."""
    cal_um = float(calibration_um if calibration_um is not None else truncate_scale_um)
    scales_um = _mem_characteristic_scales_um(eigvals, cal_um)
    rand_mask = scales_um > float(truncate_scale_um)
    return scales_um, rand_mask, cal_um


def estimate_pooled_autocorr_length_um(
    S: np.ndarray,
    A: np.ndarray,
    *,
    um_per_unit: float = 1000.0,
    n_bins: int = 40,
    n_est_pairs: int = 500_000,
    seed: int = 0,
) -> tuple[float, dict]:
    """
    Estimate pooled spatial autocorrelation length from expression vs distance.

    Pools gene-wise products (z-scored A ⇒ mean_i·j is cross-cell covariance),
    bins by pairwise distance, returns the **half-max** distance in microns:
    first bin center where c(d) ≤ 0.5 · max(c).

    Returns (length_um, diagnostics dict with c_hat curve).
    """
    coords = np.asarray(S, dtype=np.float64)
    um = float(um_per_unit)
    A64 = np.asarray(A, dtype=np.float64)
    N, G = A64.shape

    rng = np.random.default_rng(seed)
    n_est = int(min(n_est_pairs, N * (N - 1) // 2))
    ii = rng.integers(0, N, size=n_est)
    jj = rng.integers(0, N, size=n_est)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    d_est = np.linalg.norm(coords[ii] - coords[jj], axis=1) * um
    prod_est = (A64[ii] * A64[jj]).mean(axis=1)

    d_hi = float(d_est.max())
    bins = np.linspace(0.0, d_hi, n_bins + 1)
    idx = np.clip(np.digitize(d_est, bins) - 1, 0, n_bins - 1)
    csum = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    np.add.at(csum, idx, prod_est)
    np.add.at(cnt, idx, 1.0)
    c_hat = np.where(cnt > 0, csum / np.maximum(cnt, 1.0), 0.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    c_max = float(c_hat.max())
    half_level = 0.5 * c_max
    below = np.where(c_hat <= half_level)[0]
    half_um = float(centers[below[0]]) if below.size > 0 else d_hi

    diag = {
        "c_max": c_max,
        "c_hat_first_bin": float(c_hat[0]),
        "half_max_um": half_um,
        "centers_um": centers,
        "c_hat": c_hat,
    }
    return half_um, diag


def _log_joint_trunc_split(
    scales_um: np.ndarray,
    rand_mask: np.ndarray,
    truncate_scale_um: float,
    K_modes: int,
) -> None:
    n_rand = int(rand_mask.sum())
    n_keep = K_modes - n_rand
    print(
        f"    Mode split: randomize {n_rand}/{K_modes} long-range modes "
        f"(scale>{truncate_scale_um}µm), keep {n_keep} short-range fixed",
        flush=True,
    )
    if n_rand > 0:
        print(
            f"    Randomized scale range: "
            f"{scales_um[rand_mask].min():.1f} … {scales_um[rand_mask].max():.1f} µm",
            flush=True,
        )
    if n_keep > 0:
        print(
            f"    Kept scale range: "
            f"{scales_um[~rand_mask].min():.1f} … {scales_um[~rand_mask].max():.1f} µm",
            flush=True,
        )


def build_msr_surrogates(S, A, n_surrogates, seed, k=K_NEIGHBORS):
    """Plain MSR: per-gene independent sign-flip of spectral coefficients."""
    N, G = A.shape
    print(f"    MSR setup: N={N}, G={G}, k={k}", flush=True)
    t0 = time.time()
    V, C = _msr_basis(S, A, k)
    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    ts = time.time()
    for i in range(n_surrogates):
        signs = rng.integers(0, 2, size=C.shape) * 2 - 1
        surrogates[i] = (V @ (C * signs)).astype(np.float32)
    print(f"    Surrogates generated: {time.time()-ts:.1f}s  |  total MSR: {time.time()-t0:.1f}s", flush=True)
    orig_var = (A**2).mean(); surr_var = (surrogates[0]**2).mean()
    print(f"    Variance check: orig={orig_var:.4f}  surrogate[0]={surr_var:.4f} (should match)", flush=True)
    return surrogates


def build_msr_recolored_surrogates(S, A, n_surrogates, seed, k=K_NEIGHBORS):
    """MSR with gene-gene covariance recoloring."""
    from scipy.linalg import cholesky as sp_chol, solve_triangular
    N, G = A.shape
    print(f"    MSR-RC setup: N={N}, G={G}, k={k}", flush=True)
    t0 = time.time()
    V, C = _msr_basis(S, A, k)
    K_modes = V.shape[1]
    A64 = A.astype(np.float64)
    Sigma = (A64.T @ A64) / N
    eps = max(1e-9, 1e-6 * float(np.trace(Sigma)) / G)
    L = sp_chol(Sigma + eps * np.eye(G), lower=True)
    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    ts = time.time()
    for i in range(n_surrogates):
        signs  = rng.integers(0, 2, size=(K_modes, G)) * 2 - 1
        A_star = V @ (C * signs)
        Sig_s  = (A_star.T @ A_star) / N
        eps_s  = max(1e-9, 1e-6 * float(np.trace(Sig_s)) / G)
        L_star = sp_chol(Sig_s + eps_s * np.eye(G), lower=True)
        Z      = solve_triangular(L_star.T, A_star.T, lower=False).T
        A_rc   = Z @ L.T
        mu = A_rc.mean(0, keepdims=True); std = A_rc.std(0, keepdims=True).clip(1e-8)
        surrogates[i] = ((A_rc - mu) / std).astype(np.float32)
    print(f"    Surrogates: {time.time()-ts:.1f}s  |  total MSR-RC: {time.time()-t0:.1f}s", flush=True)
    surr_var = (surrogates[0]**2).mean()
    print(f"    Variance check: orig={(A**2).mean():.4f}  surr[0]={surr_var:.4f}  (should be ~1.0)", flush=True)
    if G <= 300:
        corr_o = np.corrcoef(A.T); corr_s = np.corrcoef(surrogates[0].T)
        m = np.triu_indices(G, k=1)
        print(f"    Corr check: orig off-diag mean={corr_o[m].mean():.4f}  surr[0]={corr_s[m].mean():.4f}  (should be close)", flush=True)
    return surrogates


def build_spectral_gp_surrogates(S, A, n_surrogates, seed, k=K_NEIGHBORS):
    """
    Parametric Moran-spectral GP surrogates: fresh draws from estimated SA distribution.

    Unlike MSR sign-flip, this draws NEW spectral coefficients from the estimated
    spectral density. Each surrogate is a genuine new GP realization with the same
    SA as the real data — it does NOT scramble the spatial orientation, so the
    optimization landscape has the same difficulty as real data.

    This fixes the kernel-noise false positive that MSR (sign-flip) produces:
    MSR sign-flips make surrogates harder to fit by scrambling coherent spatial
    modes, giving real data a systematic optimization advantage even with no gradient.
    Fresh draws remove that asymmetry.
    """
    N, G = A.shape
    print(f"    SpGP setup: N={N}, G={G}, k={k}", flush=True)
    t0 = time.time()
    V, C = _msr_basis(S, A, k)
    K_modes = V.shape[1]

    # Estimate per-eigenvector spectral variance by pooling across genes
    # (valid when all genes share the same SA kernel)
    spectral_var = (C**2).mean(axis=1)          # (K,)  mean power per MEM
    std_w        = np.sqrt(spectral_var[:, np.newaxis])  # (K, 1)
    print(f"    Spectral var: sum={spectral_var.sum():.1f}  (expected ≈ {N})  "
          f"mean={spectral_var.mean():.4f}", flush=True)

    rng        = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    ts         = time.time()
    for i in range(n_surrogates):
        Z     = rng.standard_normal((K_modes, G)) * std_w   # fresh draw
        A_new = V @ Z
        mu    = A_new.mean(0, keepdims=True); std = A_new.std(0, keepdims=True).clip(1e-8)
        surrogates[i] = ((A_new - mu) / std).astype(np.float32)

    print(f"    Surrogates: {time.time()-ts:.1f}s  |  total SpGP: {time.time()-t0:.1f}s", flush=True)
    surr_var = (surrogates[0]**2).mean()
    print(f"    Variance check: orig={(A**2).mean():.4f}  surr[0]={surr_var:.4f}  (should be ~1.0)", flush=True)
    return surrogates


def build_rank_matched_msr_surrogates(
    S: np.ndarray,
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    Rank-matched MSR surrogates (BrainSMASH-style).

    The key difference from all other MSR variants: instead of generating new
    expression values (Gaussian draws in MEM basis), we **reassign the original
    expression values** spatially using the MSR field as a smooth reference ordering.

    For each surrogate and gene g:
      1.  Generate a smooth MSR reference field r[:,g] via sign-flip (same SA, fresh orientation).
      2.  Rank-match: assign the k-th sorted real expression value to the cell with rank k
          in the MSR reference field.

    Preserves (exactly):
      - Marginal distribution of each gene (same set of values, just rearranged)
      - Per-gene variance (trivially, same values)

    Preserves (approximately):
      - Per-gene spatial autocorrelation (smooth reference field drives the spatial ordering)

    Destroys:
      - Cross-gene spatial alignment (each gene gets an independent reference orientation)
      - Gene-gene covariance structure (spatial positions differ per gene)

    Why this may calibrate better than fresh-draw MSR / SpGP:
      Fresh-draw methods replace expression values with new GP samples; those new values
      may be harder or easier to fit along a 1D axis depending on the basis mismatch.
      Rank-matching never generates new values — it only rearranges existing ones —
      so the model faces the same absolute value landscape as on real data.
    """
    N, G = A.shape
    print(f"    Rank-MSR setup: N={N}, G={G}, k={k}", flush=True)
    t0 = time.time()
    V, C = _msr_basis(S, A, k)

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    A32 = A.astype(np.float32)

    # Pre-sort real values per gene once — O(N log N × G)
    sorted_vals = np.sort(A32, axis=0)   # (N, G) ascending

    ts = time.time()
    for i in range(n_surrogates):
        signs   = rng.integers(0, 2, size=C.shape) * 2 - 1   # (K, G) ±1
        ref     = (V @ (C * signs)).astype(np.float32)        # (N, G) smooth reference
        # rank order of reference per gene: argsort gives cell indices sorted low→high
        ranks   = np.argsort(ref, axis=0)                     # (N, G)
        surr    = np.empty_like(A32)
        for g in range(G):
            surr[ranks[:, g], g] = sorted_vals[:, g]
        surrogates[i] = surr
    print(f"    Surrogates: {time.time()-ts:.1f}s  |  total Rank-MSR: {time.time()-t0:.1f}s", flush=True)

    # Sanity checks
    orig_var = float((A32 ** 2).mean())
    surr_var = float((surrogates[0] ** 2).mean())
    print(f"    Variance check: orig={orig_var:.4f}  surr[0]={surr_var:.4f}  (should match)", flush=True)
    # Sorted values preserved per gene
    if G <= 100:
        orig_sorted = np.sort(A32, axis=0)
        surr_sorted = np.sort(surrogates[0], axis=0)
        max_diff = float(np.abs(orig_sorted - surr_sorted).max())
        print(f"    Value preservation: max|sorted_orig - sorted_surr|={max_diff:.2e}  (should be ~0)", flush=True)

    return surrogates


def build_joint_truncated_msr_surrogates(
    S: np.ndarray,
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    k: int = K_NEIGHBORS,
    *,
    truncate_scale_um: float = 60.0,
    calibration_um: float | None = None,
) -> np.ndarray:
    """
    Joint (full-vector) MSR with long-range MEM truncation.

    Unlike per-gene MSR, applies the **same** sign-flip on each Moran mode across
    all genes — gene vectors rotate together in spectral space, preserving
    cross-gene structure within each kept mode.

    Truncation matches block-permutation scale: only modes with characteristic
    spatial scale **greater than** ``truncate_scale_um`` are randomized (sign-flip).
    Short-range modes (scale ≤ truncate_scale_um) are left fixed, preserving
    local SA within the block radius.

    Scale assignment: ``scale_k = calibration_um / sqrt(|λ_k|)`` from Moran
    eigenvalues.  When ``calibration_um == truncate_scale_um`` the cutoff is
    always ``|λ_k| < 1`` regardless of truncate — set ``calibration_um`` to a
    fixed reference (e.g. 100 µm) when sweeping truncate.
    """
    N, G = A.shape
    cal_um = float(calibration_um if calibration_um is not None else truncate_scale_um)
    print(
        f"    Joint trunc-MSR: N={N}, G={G}, k={k}, "
        f"truncate>{truncate_scale_um}µm, cal={cal_um}µm",
        flush=True,
    )
    t0 = time.time()
    V, C, eigvals = _msr_basis(S, A, k, return_eigvals=True)
    K_modes = V.shape[1]
    scales_um, rand_mask, cal_um = _joint_trunc_mode_mask(eigvals, truncate_scale_um, calibration_um)
    _log_joint_trunc_split(scales_um, rand_mask, truncate_scale_um, K_modes)
    n_rand = int(rand_mask.sum())

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    ts = time.time()
    for i in range(n_surrogates):
        signs = np.ones(K_modes, dtype=np.float64)
        if n_rand > 0:
            signs[rand_mask] = rng.integers(0, 2, size=n_rand) * 2 - 1
        # Joint flip: same sign per mode for all genes
        surrogates[i] = (V @ (C * signs[:, np.newaxis])).astype(np.float32)

    print(
        f"    Surrogates: {time.time()-ts:.1f}s  |  total joint trunc-MSR: {time.time()-t0:.1f}s",
        flush=True,
    )
    orig_var = float((A ** 2).mean())
    surr_var = float((surrogates[0] ** 2).mean())
    print(f"    Variance check: orig={orig_var:.4f}  surr[0]={surr_var:.4f}", flush=True)
    if G <= 300:
        corr_o = np.corrcoef(A.T)
        corr_s = np.corrcoef(surrogates[0].T)
        m = np.triu_indices(G, k=1)
        print(
            f"    Gene corr: orig off-diag mean={corr_o[m].mean():.4f}  "
            f"surr[0]={corr_s[m].mean():.4f}  (joint MSR preserves more than per-gene)",
            flush=True,
        )
    return surrogates


def build_joint_truncated_rank_msr_surrogates(
    S: np.ndarray,
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    k: int = K_NEIGHBORS,
    *,
    truncate_scale_um: float = 60.0,
    calibration_um: float | None = None,
    shared_rank: bool = False,
) -> np.ndarray:
    """
    Rank-matched joint truncated MSR surrogates.

    Combines joint truncated MSR with BrainSMASH-style rank assignment:

    1. Build the same truncated joint sign-flip reference as ``build_joint_truncated_msr_surrogates``
       (short-range MEMs fixed, long-range MEMs jointly sign-flipped across genes).
    2. Instead of using the reference *values*, rank-match the original expression:
       assign the k-th sorted real value per gene to the cell with rank k in the
       reference field.

    Preserves (exactly):
      - Per-gene marginal distribution (same multiset of values per gene)
      - Short-range spectral content in the reference ordering (local SA)

    Destroys (approximately):
      - Long-range cross-gene spatial alignment (joint flips on global modes)

    ``shared_rank=True``: one spatial permutation for all genes, derived from the
    mean joint reference field across genes — stronger destruction of cross-gene
    alignment while keeping marginals.
    """
    N, G = A.shape
    cal_um = float(calibration_um if calibration_um is not None else truncate_scale_um)
    mode = "shared" if shared_rank else "per-gene"
    print(
        f"    Joint trunc rank-MSR ({mode}): N={N}, G={G}, k={k}, "
        f"truncate>{truncate_scale_um}µm, cal={cal_um}µm",
        flush=True,
    )
    t0 = time.time()
    V, C, eigvals = _msr_basis(S, A, k, return_eigvals=True)
    K_modes = V.shape[1]
    scales_um, rand_mask, _ = _joint_trunc_mode_mask(eigvals, truncate_scale_um, calibration_um)
    n_rand = int(rand_mask.sum())
    _log_joint_trunc_split(scales_um, rand_mask, truncate_scale_um, K_modes)

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    A32 = A.astype(np.float32)
    sorted_vals = np.sort(A32, axis=0)

    ts = time.time()
    for i in range(n_surrogates):
        signs = np.ones(K_modes, dtype=np.float64)
        if n_rand > 0:
            signs[rand_mask] = rng.integers(0, 2, size=n_rand) * 2 - 1
        ref = (V @ (C * signs[:, np.newaxis])).astype(np.float32)
        surr = np.empty_like(A32)
        if shared_rank:
            agg = ref.mean(axis=1)
            ranks = np.argsort(agg)
            for g in range(G):
                surr[ranks, g] = sorted_vals[:, g]
        else:
            ranks = np.argsort(ref, axis=0)
            for g in range(G):
                surr[ranks[:, g], g] = sorted_vals[:, g]
        surrogates[i] = surr

    print(
        f"    Surrogates: {time.time()-ts:.1f}s  |  total joint trunc rank-MSR: {time.time()-t0:.1f}s",
        flush=True,
    )
    orig_var = float((A32 ** 2).mean())
    surr_var = float((surrogates[0] ** 2).mean())
    print(f"    Variance check: orig={orig_var:.4f}  surr[0]={surr_var:.4f}", flush=True)
    if G <= 300:
        orig_sorted = np.sort(A32, axis=0)
        surr_sorted = np.sort(surrogates[0], axis=0)
        max_diff = float(np.abs(orig_sorted - surr_sorted).max())
        print(f"    Value preservation: max diff={max_diff:.2e}  (should be ~0)", flush=True)

    return surrogates


def build_joint_truncated_partial_rank_msr_surrogates(
    S: np.ndarray,
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    k: int = K_NEIGHBORS,
    *,
    truncate_scale_um: float = 60.0,
    calibration_um: float | None = None,
    shared_rank: bool = False,
) -> np.ndarray:
    """
    Partial rank-match: short-range MEM component fixed, only long-range rank-matched.

    Decompose ``A = V_keep @ C_keep + V_rand @ C_rand`` in Moran space.

    Surrogate construction per draw:
      - ``A_short = V_keep @ C_keep`` is **identical** to real data (all short-range modes fixed).
      - ``A_long = V_rand @ C_rand`` is the real long-range contribution.
      - Build joint truncated reference ``ref_long`` via sign-flip on rand modes only.
      - Rank-match sorted ``A_long`` into ``ref_long`` ordering (per gene or shared).
      - ``A_surr = A_short + A_surr_long``.

    Compared to full rank-MSR, short-range structure is preserved exactly in the
    surrogate expression (not just in the reference ordering). Only the long-range
    additive component is permuted.
    """
    N, G = A.shape
    cal_um = float(calibration_um if calibration_um is not None else truncate_scale_um)
    mode = "shared" if shared_rank else "per-gene"
    print(
        f"    Joint trunc partial-rank ({mode}): N={N}, G={G}, k={k}, "
        f"truncate>{truncate_scale_um}µm, cal={cal_um}µm",
        flush=True,
    )
    t0 = time.time()
    V, C, eigvals = _msr_basis(S, A, k, return_eigvals=True)
    K_modes = V.shape[1]
    scales_um, rand_mask, _ = _joint_trunc_mode_mask(eigvals, truncate_scale_um, calibration_um)
    n_rand = int(rand_mask.sum())
    _log_joint_trunc_split(scales_um, rand_mask, truncate_scale_um, K_modes)

    keep_mask = ~rand_mask
    V_keep = V[:, keep_mask]
    V_rand = V[:, rand_mask]
    C_keep = C[keep_mask, :]
    C_rand = C[rand_mask, :]

    A_short = (V_keep @ C_keep).astype(np.float32)
    A_long = (V_rand @ C_rand).astype(np.float32)
    sorted_long = np.sort(A_long, axis=0)

    short_frac = float((A_short ** 2).sum() / max((A.astype(np.float32) ** 2).sum(), 1e-12))
    print(f"    Short-range energy fraction: {short_frac:.3f} of ||A||²", flush=True)

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    ts = time.time()
    for i in range(n_surrogates):
        signs = np.ones(n_rand, dtype=np.float64)
        if n_rand > 0:
            signs = rng.integers(0, 2, size=n_rand) * 2 - 1
        ref_long = (V_rand @ (C_rand * signs[:, np.newaxis])).astype(np.float32)
        surr_long = np.empty_like(A_long)
        if shared_rank:
            ranks = np.argsort(ref_long.mean(axis=1))
            for g in range(G):
                surr_long[ranks, g] = sorted_long[:, g]
        else:
            ranks = np.argsort(ref_long, axis=0)
            for g in range(G):
                surr_long[ranks[:, g], g] = sorted_long[:, g]
        surrogates[i] = A_short + surr_long

    print(
        f"    Surrogates: {time.time()-ts:.1f}s  |  total partial-rank: {time.time()-t0:.1f}s",
        flush=True,
    )
    A32 = A.astype(np.float32)
    orig_var = float((A32 ** 2).mean())
    surr_var = float((surrogates[0] ** 2).mean())
    print(f"    Variance check: orig={orig_var:.4f}  surr[0]={surr_var:.4f}", flush=True)
    short_match = float(np.abs(A_short - (V_keep @ C_keep)).max())
    print(f"    A_short fixed in surr: max|A_short - V_keep@C_keep|={short_match:.2e}", flush=True)
    if G <= 300 and n_rand > 0:
        long_sorted_match = float(np.abs(np.sort(A_long, axis=0) - np.sort(surrogates[0] - A_short, axis=0)).max())
        print(f"    A_long marginals preserved: max diff={long_sorted_match:.2e}", flush=True)

    return surrogates


def build_variogram_matched_surrogates(
    S: np.ndarray,
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    *,
    n_bins: int = 40,
    cutoff_corr: float = 0.02,
    n_est_pairs: int = 2_000_000,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Non-parametric SA-preserving null: empirical covariance-vs-distance + Cholesky draws.

    The non-parametric cousin of the exp-kernel Cholesky-GP null.  Instead of assuming
    an exponential kernel with a known correlation length ρ, it estimates the spatial
    covariance directly from the data as a function of distance, then resamples fresh
    GP realizations matching that empirical structure.

    For a stationary isotropic GP, covariance and (semi)variogram are dual:
        γ(d) = c(0) − c(d),   with c(0)=1 for z-scored expression.
    So matching c(d) ≡ matching the empirical variogram (BrainSMASH idea), but built
    in cell space so the surrogates are exchangeable with what isodepth actually sees.

    Steps
    -----
    1. Estimate c_hat(d): for a random subsample of cell pairs, compute the per-pair
       gene-pooled product (A_i · A_j)/G (z-scored A ⇒ this is the pooled cross-cell
       covariance), then bin by distance.  c_hat(0)=1 by construction.
    2. Determine a data-driven cutoff: the first distance where c_hat decays to
       ``cutoff_corr`` — beyond that, covariance is treated as 0.  No ρ assumed; the
       decay length is read off the empirical curve.
    3. Build a sparse stationary covariance C with C_ij = interp(c_hat)(d_ij) for
       d_ij < cutoff, diagonal = 1.
    4. Cholesky factor L (eigen-clip to nearest PSD if the empirical C is indefinite).
    5. Draw σ·L·Z per gene independently, z-score per gene.

    Assumptions: stationarity + isotropy (covariance depends only on distance) and a
    finite correlation range — the same minimal assumptions as variogram methods.
    No parametric kernel and no fixed correlation length.
    """
    from scipy.spatial import KDTree

    N, G = A.shape
    print(f"    Variogram setup: N={N}, G={G}, n_bins={n_bins}, cutoff_corr={cutoff_corr}", flush=True)
    t0 = time.time()
    coords = S.astype(np.float64)
    A64 = A.astype(np.float64)

    # 1. Estimate covariance-vs-distance from random pairs (full range, unbiased, cheap)
    rng_est = np.random.default_rng(seed + 12345)
    n_est = int(min(n_est_pairs, N * (N - 1) // 2))
    ii = rng_est.integers(0, N, size=n_est)
    jj = rng_est.integers(0, N, size=n_est)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    d_est = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    prod_est = (A64[ii] * A64[jj]).mean(axis=1)          # gene-pooled product
    d_lo, d_hi = 0.0, float(d_est.max())
    bins = np.linspace(d_lo, d_hi, n_bins + 1)
    idx = np.clip(np.digitize(d_est, bins) - 1, 0, n_bins - 1)
    csum = np.zeros(n_bins); cnt = np.zeros(n_bins)
    np.add.at(csum, idx, prod_est)
    np.add.at(cnt, idx, 1.0)
    c_hat = np.where(cnt > 0, csum / np.maximum(cnt, 1.0), 0.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # 2. Data-driven cutoff: first bin center where covariance falls to cutoff_corr
    below = np.where(c_hat <= cutoff_corr)[0]
    cutoff = float(centers[below[0]]) if below.size > 0 else d_hi
    decay_half = centers[np.argmin(np.abs(c_hat - 0.5))] if c_hat.max() >= 0.5 else float("nan")
    print(f"    Empirical c(d): c[0bin]={c_hat[0]:.3f}  half-corr≈{decay_half:.3f}  "
          f"cutoff(≤{cutoff_corr})={cutoff:.3f}  (units of standardized S)", flush=True)

    # 3. Build sparse stationary covariance from cutoff neighbors
    t_c = time.time()
    tree = KDTree(coords)
    cpairs = tree.query_pairs(cutoff, output_type="ndarray")
    cd = np.linalg.norm(coords[cpairs[:, 0]] - coords[cpairs[:, 1]], axis=1)
    cvals = np.interp(cd, centers, c_hat, left=1.0, right=0.0)
    C = np.eye(N, dtype=np.float64)
    C[cpairs[:, 0], cpairs[:, 1]] = cvals
    C[cpairs[:, 1], cpairs[:, 0]] = cvals
    print(f"    C built: {cpairs.shape[0]} neighbor pairs within cutoff  [{time.time()-t_c:.1f}s]", flush=True)

    # 4. Cholesky with PSD repair
    t_f = time.time()
    try:
        L = np.linalg.cholesky(C)
        psd_note = "direct"
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        w_clip = np.clip(w, 1e-6, None)
        C_psd = (V * w_clip) @ V.T
        L = np.linalg.cholesky(C_psd + 1e-8 * np.eye(N))
        psd_note = f"eigen-clipped (min eig was {w.min():.3e})"
    print(f"    Cholesky: {psd_note}  [{time.time()-t_f:.1f}s]", flush=True)

    # 5. Fresh draws, per-gene z-score
    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    ts = time.time()
    for i in range(n_surrogates):
        Z = rng.standard_normal((N, G))
        noise = float(sigma) * (L @ Z)
        mu = noise.mean(axis=0, keepdims=True)
        std = noise.std(axis=0, keepdims=True).clip(1e-8)
        surrogates[i] = ((noise - mu) / std).astype(np.float32)
    print(f"    Surrogates: {time.time()-ts:.1f}s  |  total variogram: {time.time()-t0:.1f}s", flush=True)
    print(f"    Variance check: surr[0]={(surrogates[0]**2).mean():.4f}  (should be ~1.0)", flush=True)

    return surrogates


def _build_exp_kernel_cholesky(
    S: np.ndarray,
    scale_um: float,
    kernel_distance_um: float,
    delta: float,
    max_interaction_distance_um: float | None = None,
) -> np.ndarray:
    """Cholesky of C = I + δ·K on micron coords — mirrors SpatialDataSimulator._build_cholesky."""
    from scipy.spatial import KDTree

    N = S.shape[0]
    S_um = S.astype(np.float64) * float(scale_um)
    r_max = float(max_interaction_distance_um) if max_interaction_distance_um is not None else 4.0 * float(kernel_distance_um)
    p = float(kernel_distance_um)
    tree = KDTree(S_um)
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    C = np.eye(N, dtype=np.float64)
    np.fill_diagonal(C, 1.0 + delta)
    if pairs.shape[0] > 0:
        d_vals = np.linalg.norm(S_um[pairs[:, 0]] - S_um[pairs[:, 1]], axis=1)
        k_vals = delta * np.exp(-d_vals / p)
        C[pairs[:, 0], pairs[:, 1]] += k_vals
        C[pairs[:, 1], pairs[:, 0]] += k_vals
    return np.linalg.cholesky(C)


def build_kernel_cholesky_surrogates(
    S: np.ndarray,
    G: int,
    n_surrogates: int,
    seed: int,
    *,
    scale_um: float,
    kernel_distance_um: float,
    delta: float,
    sigma: float = 1.0,
    max_interaction_distance_um: float | None = None,
) -> np.ndarray:
    """
    Generative-matched null: fresh draws from N(0, σ²·C) in cell space.

    Mirrors SpatialDataSimulator._draw_correlated_noise + per-gene z-score exactly.
    Each surrogate gene is an independent draw; cross-gene covariance is zero.
    Uses the same exponential kernel Cholesky as the synthetic kernel-noise simulator,
    not Moran MEMs — so the null matches what isodepth sees (smooth fields on fixed S).
    """
    N = S.shape[0]
    print(f"    Cholesky-GP setup: N={N}, G={G}, ρ={kernel_distance_um}µm, δ={delta}, σ={sigma}", flush=True)
    t0 = time.time()
    t_chol = time.time()
    L = _build_exp_kernel_cholesky(S, scale_um, kernel_distance_um, delta, max_interaction_distance_um)
    print(f"    Cholesky: {time.time()-t_chol:.1f}s  (L shape {L.shape})", flush=True)

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, N, G), dtype=np.float32)
    ts = time.time()
    for i in range(n_surrogates):
        Z = rng.standard_normal((N, G))
        noise = float(sigma) * (L @ Z)
        mu = noise.mean(axis=0, keepdims=True)
        std = noise.std(axis=0, keepdims=True).clip(1e-8)
        surrogates[i] = ((noise - mu) / std).astype(np.float32)
    print(f"    Surrogates: {time.time()-ts:.1f}s  |  total Cholesky-GP: {time.time()-t0:.1f}s", flush=True)
    surr_var = (surrogates[0] ** 2).mean()
    print(f"    Variance check: surr[0]={surr_var:.4f}  (should be ~1.0)", flush=True)
    return surrogates


def build_kernel_cholesky_surrogates_from_meta(
    S: np.ndarray,
    G: int,
    n_surrogates: int,
    seed: int,
    meta: dict,
) -> np.ndarray:
    """Build Cholesky-GP surrogates using params stored in synthetic dataset meta."""
    kernel = meta.get("kernel") or {}
    distance = float(kernel.get("distance", 30.0))
    scale_um = float(meta.get("scale_um", 1000.0))
    delta = float(meta.get("delta", 0.1))
    sigma = float(meta.get("sigma", 0.5))
    max_dist = kernel.get("max_interaction_distance")
    max_um = float(max_dist) if max_dist is not None else None
    return build_kernel_cholesky_surrogates(
        S, G, n_surrogates, seed,
        scale_um=scale_um,
        kernel_distance_um=distance,
        delta=delta,
        sigma=sigma,
        max_interaction_distance_um=max_um,
    )


def run_coord_shuffle(S, A, label):
    print(f"\n  [Coord shuffle] {label}", flush=True)
    t0 = time.time()
    cfg = base_cfg(N_PERMS)
    device = resolve_device(cfg.device)
    _, outputs, _ = train_parallel_isodepth_model(S, A, cfg, device=device, model_label=f"coord_shuffle_{label}")
    elapsed = time.time() - t0
    p = permutation_p_value(cfg.metric, outputs.stat_true, outputs.stat_perm)
    nm = float(outputs.stat_perm.mean()); ns = float(outputs.stat_perm.std())
    z = (nm - outputs.stat_true) / (ns + 1e-12)
    print(f"    stat_true={outputs.stat_true:.6f}  null_mean={nm:.6f}  null_std={ns:.2e}  z={z:.2f}  p={p:.3f}  [{elapsed:.0f}s]", flush=True)
    return dict(label=label, method="coord_shuffle", stat_true=outputs.stat_true,
                stat_perm=outputs.stat_perm, null_mean=nm, null_std=ns, z=z, p=p, elapsed=elapsed)


def run_msr_null(S, A, label):
    print(f"\n  [MSR null] {label}", flush=True)
    t0 = time.time(); device = resolve_device("cuda")
    surrogates = build_msr_surrogates(S, A, N_PERMS, seed=SEED+500)
    cfg0 = base_cfg(n_perms=0, seed=SEED)
    _, out_true, _ = train_parallel_isodepth_model(S, A, cfg0, device=device, model_label=f"MSR_true_{label}")
    stat_true = out_true.stat_true
    print(f"    stat_true={stat_true:.6f}", flush=True)
    stat_perm = np.empty(N_PERMS, dtype=np.float64)
    for i in range(N_PERMS):
        cfg_i = base_cfg(n_perms=0, seed=SEED+i+1)
        _, out_i, _ = train_parallel_isodepth_model(S, surrogates[i], cfg_i, device=device, model_label=f"MSR_perm{i+1}_{label}")
        stat_perm[i] = out_i.stat_true
        print(f"    perm {i+1:2d}/{N_PERMS}: stat={stat_perm[i]:.6f}", flush=True)
    elapsed = time.time() - t0
    p = permutation_p_value(cfg0.metric, stat_true, stat_perm)
    nm = float(stat_perm.mean()); ns = float(stat_perm.std())
    z = (nm - stat_true) / (ns + 1e-12)
    print(f"    null_mean={nm:.6f}  null_std={ns:.2e}  z={z:.2f}  p={p:.3f}  [{elapsed:.0f}s]", flush=True)
    return dict(label=label, method="msr", stat_true=stat_true, stat_perm=stat_perm,
                null_mean=nm, null_std=ns, z=z, p=p, elapsed=elapsed)
