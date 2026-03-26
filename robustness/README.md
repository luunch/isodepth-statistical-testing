# Isodepth Robustness Tests

This folder includes two hypothesis tests for isodepth stability and identifiability.

## 1) Perturbation Robustness Test

Script: `robustness/test_isodepth_robustness.py`

- Train baseline isodepth `d` on original data.
- Build `M` perturbed datasets and retrain to get `d_delta,m`.
- Compute observed stability `S_obs = mean_m Spearman(d, d_delta,m)`.
- Build null by permuting indices of each `d_delta,m` for `K` permutations.
- Compute one-sided p-value: `P(S_null >= S_obs)`.

Run:

```bash
python robustness/test_isodepth_robustness.py \
  --modes radial,noise \
  --perturb-target coords \
  --delta 0.05 \
  --m-perturb 30 \
  --k-null 500
```

## 2) Subset Selection Test (Global Identifiability)

Script: `robustness/test_isodepth_subset_selection.py`

- Train baseline isodepth `d` on full data.
- Sample `M` subsets `A_m` and retrain to get `d_{A_m}`.
- Compute observed statistic `T_obs = mean_m Spearman(d|_{A_m}, d_{A_m})`.
- Build null by permuting indices within each subset prediction for `K` permutations.
- Compute one-sided p-value: `P(T_null >= T_obs)`.

Run:

```bash
python robustness/test_isodepth_subset_selection.py \
  --modes radial,noise \
  --subset-frac 0.7 \
  --m-subsets 30 \
  --k-null 500
```

## Notes

- Both scripts support `--modes noise,radial,checkerboard` (comma-separated).
- Each script writes per-mode plots plus a cross-mode comparison plot.
- Both scripts print progress updates during long runs.
- Both scripts use parallelized GASTON (`ParallelIsoDepthNet`) for the expensive repeated retraining stage.
- Add `--sequential` to force non-parallel retraining for validation/debug comparisons.
- By default, sign alignment is applied before Spearman (`d` vs `-d` symmetry). Disable with `--no-align-sign`.
