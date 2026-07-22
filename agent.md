# Agent notes — isodepth-statistical-testing

## Synthetic spatial noise

Two generative modes on `data.kernel`:

1. **`exp` / `trunc` + `data.delta > 0`**: Cholesky of `C = I + δ·K` (GP-style correlated noise). Requires `data.scale` (µm).
2. **`smooth`**: Gaussian-smoothed white noise over micron coordinates; `kernel.distance` is bandwidth `σ_bw` (µm). Requires `data.delta == 0` and `data.scale`. Cutoff defaults to `4 * distance`.

Implementation: `data/synthetic.py` (`_draw_smoothed_noise`, `_draw_correlated_noise`). Configs: `configs/synthetic/smooth_noise_*.json` (mirrored from `kernel_noise_*` scale: ~4–5k cells, σ=0.5, bw=15 µm).

Do not confuse `data.delta` (kernel SA strength) with `test.delta` (coordinate perturbation).

## `parallel_permutation` / `separate` mode: batch-size-dependent training noise

In `cell_type="separate"` mode, per-type preprocessing (gene filtering, HVG selection,
z-scoring of `S` and `A`) is deferred and computed independently per cell type on that
type's own cells (`data/h5ad_loader.py` `defer_preprocessing` branch;
`_process_single_celltype_separate` in `methods/permutation.py`). So a data-level change
that only removes cells from one cell type (e.g. a `spatial_crop` filter) leaves the
`(S, A)` inputs for *other* cell types bit-identical across runs.

Despite that, `stat_true`/null-distribution values for an *unaffected* cell type can still
shift measurably (~1%) between two runs that only differ in `n_perms`. Cause: the trainer
batches every permutation slot (`true` + all nulls) times `n_reruns` into one tensor,
`total_models = n_models * n_reruns` (`methods/trainers/isodepth.py`), so different
`n_perms` values change the total batch size fed through CUDA. No
`torch.use_deterministic_algorithms`/cuDNN-deterministic flag is set anywhere in
`methods/`, so kernel selection / floating-point reduction order depends on batch shape.
Tiny per-step numerical differences compound over hundreds of SGD epochs into genuinely
different local optima, even for logically identical per-cell-type sub-problems.

Practical implication: don't compare `stat_true` / null-distribution spread across two
runs with different `n_perms` (or different `n_reruns`) as if they were noise-free —
especially when eyeballing "how many permutations beat the true loss," since that count
depends on both the number of permutations *and* the true-vs-null gap (in σ units), and
the gap itself is not perfectly reproducible across batch sizes. To isolate a real data
effect (e.g. from a crop) from this training noise, rerun both configs with matching
`n_perms`, or use `experiments/real_data_existence_consistency_sweep` to measure the
seed-to-seed noise floor for `stat_true` before trusting small differences.
