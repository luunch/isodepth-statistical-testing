# Agent notes — isodepth-statistical-testing

## Loss history recording

`TestConfig.record_loss_history` defaults to `True`. When on, trainers store
`loss_history` (true-slot scalar per epoch) and `loss_history_per_slot`
(stacked array, shape `(epochs, total_slots)`) in `model.training_metadata`.
Chunk merge concatenates per-slot histories along the slot axis (trimmed to
shared epoch length).

## Calicost clone-specific configs

`calicost_clone_label` values are float-like strings (`"1.0"`, `"2.0"`). Restrict to one
clone with `data.obs_filters` (string equality). Omit `obs_numeric_filters` when skipping
tumor-proportion gating; keep `obs_drop_na` + loss-difference whitening on
`calicost_tumor_proportion` unless whitening is also dropped. Example:
`configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2.json` (~199 clone-2 spots;
~197 after tumor-prop NA drop).

For posthoc GSEA, invoke the script as a module from repo root because filepath invocation
imports `experiments` before inserting repo root into `sys.path`. If Scanpy/Numba cache
writes fail, set writable temp caches:
`env NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mplconfig /weka/home/ajain71/miniforge3/envs/isodepth_env/bin/python -m scripts.posthoc.postprocess_gsea_isodepth ...`.
Run completed for
`results/calicost/HT306P1_S1H1Fc2U1Z1Bs1/loss_diff_clone2_linear_gt0p5_cropy`;
outputs are in `gsea_isodepth/`.

Sensitivity config excluding MT/ribo/MALAT1/stress genes:
`configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_no_mtribo_stress.json`.
Completed run:
`results/calicost/HT306P1_S1H1Fc2U1Z1Bs1/loss_diff_clone2_linear_gt0p5_cropy_no_mtribo_stress`.
It remains significant (`p_value=0.004`), and the filtered true isodepth is nearly identical
to the original gt0p5/cropy isodepth (Spearman 0.994, Pearson 0.997). Posthoc GSEA remains
dominated by negative-end EMT/ECM, IFN/MHC/allograft, IL6/JAK/STAT3, complement/inflammation;
MT/MALAT1 no longer dominate the bottom-ranked genes.

## Loss curves (restored after scripts/experiments reorg)

`test.record_loss_history` defaults to **True**. Training stores per-epoch true loss and
`loss_history_per_slot`; runs write `{type}_loss_curve.png` + `{type}_loss_history.npz`
(separate mode) or `{run_name}_loss_curve.png` / `_loss_history.npz` (combined). Set
`record_loss_history: false` to skip (saves memory/time on huge n_perms × epochs jobs).

## Scripts / experiments layout (2026-08 cleanup)

Reorganized `scripts/` and `experiments/` on branch `cleanup/scripts-experiments`.
Archive snapshot: `archive/pre-cleanup-20260806`. Layout contract:
`docs/SCRIPTS_EXPERIMENTS_LAYOUT.md` and `scripts/README.md`.

- Shared helpers: `experiments/core/{study_io,dataset_cache,study_spec,paths}.py`
- Studies live under `experiments/studies/<name>/`; flat `python -m experiments.*` paths are shims
- Scripts live under `scripts/{data_prep,regen,posthoc,studies}/`; top-level `scripts/*.py` are shims
- WIP from before this cleanup was stashed on `main` as `wip-before-scripts-experiments-reorg-20260806`

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
