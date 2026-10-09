# Isodepth Statistical Testing

This repository now uses a refactored pipeline for running isodepth-based statistical tests with a stable configuration model and a clean separation between data loading, methods, and orchestration.

## Active Pipeline

The active pipeline is built around these files and folders:

- `configs/`: JSON config files
- `data/`: data loaders, transforms, schemas, and dataset assets
- `methods/`: model architectures, training code, test methods, and metrics
- `run_permutation.py`: generic entry point that executes a config-defined dataset and test method
- `experiments/configuration.py`: shared config merge and output writing logic
- `experiments/core/`: shared study helpers (`study_io`, `dataset_cache`, `study_spec`)
- `experiments/studies/`: multi-run sweep families (flat `python -m experiments.*` shims remain)
- `scripts/`: utilities under `data_prep/`, `regen/`, `posthoc/`, `studies/<topic>/` (see `scripts/README.md`)
- `data/schemas.py`: shared `DatasetBundle`, `TestConfig`, `RunConfig`, and `TestResult` contracts
- `validation/`: smoothness and related validation checks

`run_permutation.py` is the only Python entry point that should live at the
repository root. Put all other standalone utilities, diagnostics, and
regeneration helpers under `scripts/` (not the repo root). Layout details:
[`docs/SCRIPTS_EXPERIMENTS_LAYOUT.md`](docs/SCRIPTS_EXPERIMENTS_LAYOUT.md).

Legacy prototypes live under `old/` (gitignored). Pre-cleanup snapshot:
`archive/pre-cleanup-20260806`.

## Data Layout

All current and future `.h5ad` files should live under:

```text
data/h5ad/
```

Example:

```text
data/h5ad/hippocampal_pyramidal.h5ad
```

Both `h5ad` and synthetic sources are normalized into the same `DatasetBundle` structure:

- `S`: `np.ndarray` with shape `(N, 2)`
- `A`: `np.ndarray` with shape `(N, G)`
- `meta`: source-specific metadata

## Config Model

The runner supports:

- a first-class JSON config via `--config`
- CLI overrides for quick edits
- reproducible saved outputs using the merged effective config

The effective config is:

```text
JSON config + explicitly provided CLI overrides
```

Top-level config sections:

- `data`
- `test`
- `output`

## Example Config

Create a config file like this and save it as `configs/my_run.json`:

```json
{
  "data": {
    "source": "h5ad",
    "h5ad": "data/h5ad/hippocampal_pyramidal.h5ad",
    "spatial_key": "spatial",
    "obs_x_col": null,
    "obs_y_col": null,
    "layer": null,
    "use_raw": false,
    "min_cells_per_gene": 1,
    "standardize_expression": true,
    "standardize_coordinates": true,
    "q": null,
    "max_cells": null,
    "seed": 0
  },
  "test": {
    "method": "parallel_permutation",
    "metric": "nll_gaussian_mse",
    "n_perms": 100,
    "epochs": 5000,
    "decoder": "nn",
    "lr": 0.001,
    "patience": 50,
    "seed": 0,
    "device": "auto",
    "batch_size": null,
    "sgd_batch_size": 0,
    "verbose": true
  },
  "output": {
    "out_dir": "results",
    "run_name": "parallel_permutation",
    "save_preds": false,
    "save_perm_stats": true
  }
}
```

Supported `test.method` values:

- `parallel_permutation`
- `block_permutation`
- `binning`
- `visium_pseudobulk`
- `correlated_null_permutation`
- `correlated_null_adaptive_permutation`
- `full_retraining`

## Running A Config

Run the config file:

```bash
python run_permutation.py \
  --config configs/my_run.json
```

Run the bundled example config:

```bash
python run_permutation.py \
  --config configs/noise_existence.json
```

Run a config with CLI overrides:

```bash
python run_permutation.py \
  --config configs/my_run.json \
  --h5ad data/h5ad/hippocampal_pyramidal.h5ad \
  --method parallel_permutation \
  --n-perms 200 \
  --out-dir results/hippocampal_run \
  --run-name hippocampal_parallel \
  --quiet
```

Run without a config file:

```bash
python run_permutation.py \
  --h5ad data/h5ad/hippocampal_pyramidal.h5ad \
  --method parallel_permutation \
  --n-perms 100 \
  --epochs 5000 \
  --lr 1e-3 \
  --patience 50 \
  --out-dir results
```

## Key CLI Flags

- `--config`: path to a JSON config file. This is the main entry point for reproducible runs.
- `--data-source`: dataset source. Supported values are `h5ad` and `synthetic`.
- `--h5ad`: input `.h5ad` path when `data.source = "h5ad"`.
- `--spatial-key`: key inside `adata.obsm` used for spatial coordinates. Default is `spatial`.
- `--obs-x-col`, `--obs-y-col`: explicit coordinate columns from `adata.obs`. Use these if coordinates are not in `adata.obsm`.
- `--layer`: expression layer to read from `adata.layers[...]` instead of `adata.X`.
- `--use-raw`: use `adata.raw.X` as the expression matrix.
- `--no-use-raw`: explicitly disable `adata.raw.X` usage and fall back to `adata.X` or `--layer`.

`--use-raw` details:
If `--layer` is provided, that takes precedence.
If `--use-raw` is set, the loader uses `adata.raw.X`.
If neither is set, the loader uses `adata.X`.

- `--min-cells-per-gene`: drop genes expressed in fewer than this many cells before running the test.
- `--log1p`: apply `log(1 + x)` after loading/filtering and before any standardization.
- `--no-log1p`: leave expression values on their original scale before standardization.
- `--standardize-expression`: z-score each gene after loading/filtering and any optional `log1p`.
- `--no-standardize-expression`: skip z-scoring and keep the post-transform feature scale.
- `--standardize-coordinates`: z-score each spatial axis (x, y) before training so the encoder sees N(0,1) inputs.
- `--no-standardize-coordinates`: keep raw spatial coordinates as loaded.
- `--q`: if provided, replace the expression matrix with a Poisson low-rank latent embedding of width `2*q`.

`--log1p` details:
This controls the same `data.log1p` field from the config.
The default pipeline behavior is log transform disabled.
`--log1p` cannot be combined with `--q`, because the Poisson low-rank transform expects counts.

`--standardize-expression` details:
This controls the same `data.standardize_expression` field from the config.
The default pipeline behavior is standardization enabled.
Use `--no-standardize-expression` only if you intentionally want the post-filter/post-`log1p` feature scale without z-scoring.

`--standardize-coordinates` details:
This controls the same `data.standardize_coordinates` field from the config.
The default pipeline behavior is coordinate standardization enabled (per-axis z-score on `dataset.S`).
Use `--no-standardize-coordinates` only if you want to preserve the raw spatial scale (e.g. for visualization-only runs).

`--q` details:
The loader first filters genes, then fits a Poisson low-rank factorization with `log(lambda) = L R^T` and returns the top `2*q` latent dimensions from `L`.
After projection, the latent representation is treated as approximately Gaussian and can be used directly by the downstream Gaussian-error models in this repository.
This option is intended for `h5ad` count-valued inputs and is not supported for the synthetic generator.

- `--max-cells`: randomly subsample at most this many cells/spots from the dataset.
- `--seed`: random seed used for dataset subsampling and permutation generation.
- `--mode`: synthetic data mode. Supported values are `radial`, `checkerboard`, `noise`, and `fourier`.
- `--n-cells`: requested number of synthetic cells/spots.
- `--n-genes`: requested number of synthetic genes.
- `--sigma`: synthetic noise scale.
- `--k`: legacy shorthand for the Fourier frequency band when `--mode fourier`. It maps to `k_min = 1` and `k_max = k`, and the generator samples a coupled 2D Fourier basis over terms of the form `cos(2π(k1 x + k2 y))` and `sin(2π(k1 x + k2 y))`.
- `data.dependent_xy`: Fourier-only boolean flag. `true` uses the coupled basis `k1 x + k2 y`; `false` uses independent `x`-only and `y`-only sine/cosine terms.

- `--method`: test method. Supported values are `parallel_permutation`, `block_permutation`, `binning`, `visium_pseudobulk`, `correlated_null_permutation`, `correlated_null_adaptive_permutation`, and `full_retraining`.
- `--metric`: one of `nll_gaussian_mse`, `mse`, `pearson_corr_mean`, `spearman_corr_mean`.
- `--n-perms`: number of permutations for existence-style methods.
- `--n-reruns`: number of parallel reruns trained per dataset instance inside the batched GASTON model. The minimum training reconstruction loss is selected independently for the observed dataset and every transformed/null dataset. Default is `30`.
- `--alpha`: one-sided permutation significance threshold. Default is `0.05`.
- `--epochs`: training epochs for learned-model methods.
- `--lr`: learning rate.
- `--patience`: early stopping patience. `0` disables early stopping; minibatch training still restores the best full-data epoch for each restart.
- `--device`: compute device such as `cpu`, `cuda`, `mps`, or `auto`.
- `--decoder`: decoder architecture for isodepth models. Supported values are `linear` and `nn`. Default is `nn` when omitted from config and CLI.
- `--batch-size`: optional `test.batch_size` override.
- `--sgd-batch-size`: optional minibatch size for stochastic gradient descent over cells during model fitting. Leave unset or set to `0` to keep the current full-batch training behavior. Each independent model receives its own mean reconstruction-loss gradient, without division by the number of parallel models. This convention is shared by learned isodepth models, models with appended covariates, cell-type models, and fixed-covariate decoders.
  Gaussian linear/quadratic decoders also honor this setting: OLS is fit within each minibatch for encoder updates, while checkpoint selection and final decoder fitting use the full training data. In separate cell-type mode, the batch size applies independently within each cell-type subset.

- `--out-dir`: parent output directory.
- `--run-name`: run-specific subdirectory name under `out_dir`, and the filename prefix for saved artifacts.
- `--save-preds`: legacy compatibility flag. Prediction arrays are not written to `results/`.
- `--no-save-preds`: legacy compatibility flag. Prediction arrays are not written to `results/`.
- `--save-perm-stats`: legacy compatibility flag. Permutation arrays are not written to `results/`.
- `--no-save-perm-stats`: legacy compatibility flag. Permutation arrays are not written to `results/`.
- `--verbose`: enable training/progress output.
- `--quiet`: disable training/progress output.

## Output Format

Each run writes:

- standardized result JSON
- `<run_name>_isodepth_triptych.png`
- `<run_name>_metric_distribution.png`
- `<run_name>_true_curve.png` for synthetic `radial`, `fourier`, and `noise` datasets
- the full merged effective config embedded in the result JSON

Typical output directory contents:

```text
results/
  <run_name>/
    <run_name>_true_curve.png
    <run_name>_isodepth_triptych.png
    <run_name>_metric_distribution.png
    <run_name>_result.json
```

## Result Schema

The result JSON includes these standard fields:

- `method_name`
- `metric`
- `p_value`
- `stat_true`
- `stat_perm`
- `runtime_sec`
- `n_cells`
- `n_genes`
- `config`
- `artifacts`

## Experiment Sweeps

Broader multi-run studies live under `experiments/` and sit above the single-run `run_permutation.py` pipeline.

The existence power-vs-sigma workflow uses:

- [configs/synthetic_existence_base.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/synthetic_existence_base.json) as the reusable base existence config
- [configs/experiments/existence_sigma_study.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/experiments/existence_sigma_study.json) as the sweep spec
- `python -m experiments.existence_sigma_sweep --spec configs/experiments/existence_sigma_study.json` to launch the study
- `python -m experiments.existence_sigma_analysis --spec configs/experiments/existence_sigma_study.json` to aggregate saved result JSONs into CSV summaries and experiment-level plots

The Fourier `k_max` sweep uses:

- [configs/experiments/fourier_kmax_study.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/experiments/fourier_kmax_study.json) as the sweep spec
- `python -m experiments.fourier_kmax_sweep --spec configs/experiments/fourier_kmax_study.json` to launch the study
- `python -m experiments.fourier_kmax_analysis --spec configs/experiments/fourier_kmax_study.json` to aggregate saved result JSONs into `per_run_results.csv`, `summary_by_kmax.csv`, and `pvalue_vs_kmax.png`

The Fourier existence-only `k_max` study uses:

- [configs/synthetic/fourier_existence.json](/home/jhu/ajain71/scratch_uchitra1/users/ajain71/isodepth-statistical-testing/configs/synthetic/fourier_existence.json) as the existence base config
- [configs/experiments/fourier_kmax_existence_study.json](/home/ajain71/scratch_uchitra1/users/ajain71/isodepth-statistical-testing/configs/experiments/fourier_kmax_existence_study.json) as the sweep spec
- `python -m experiments.fourier_kmax_sweep --spec configs/experiments/fourier_kmax_existence_study.json` to launch the study
- `python -m experiments.fourier_kmax_analysis --spec configs/experiments/fourier_kmax_existence_study.json` to analyze the study

The real-data existence consistency study uses:

- [configs/mouse_hippocampus_existence.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/mouse_hippocampus_existence.json) as the default base real-data existence config (override via `base_config` in the spec for other datasets)
- [configs/experiments/consistency_study.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/experiments/consistency_study.json) as the sweep spec (`experiment_name` is `consistency_study`; runner: `run_consistency_study.sh`)
- `python -m experiments.real_data_existence_consistency_sweep --spec configs/experiments/consistency_study.json` to launch the repeat sweep
- `python -m experiments.real_data_existence_consistency_analysis --spec configs/experiments/consistency_study.json` to aggregate saved result JSONs into repeat-level CSVs and plots for p-values, true losses, null loss distributions, and the true-isodepth Spearman matrix

This study keeps the real dataset fixed, reruns the same `parallel_permutation` existence test with different seeds, and records:

- one p-value per repeat
- one true-data loss per repeat
- the full null loss distribution for each repeat
- the pairwise Spearman agreement across all saved true-data isodepth vectors

Sweep outputs are written under:

```text
results/experiments/<experiment_name>/
  manifest.json
  runs/<run_name>/...
  analysis/
    per_run_results.csv
    summary_by_condition.csv
    power_vs_sigma.png
    null_rejection_vs_sigma.png
    fourier_power_heatmap.png
    fourier_null_rejection_heatmap.png
    pvalue_by_repeat.png
    true_loss_by_repeat.png
    null_loss_distributions_by_repeat.png
    null_loss_density_overlay.png
    true_isodepth_spearman_matrix.png
    isodepth_spearman_histogram.png
```

## Recursive SVG Gradient Discovery

Setting `test.recursive = true` enables iterative spatial gradient peeling.

**Requirements:**
- `test.method` must be `parallel_permutation`.
- `test.decoder` must be `linear` or `quadratic` (the F-test requires a parametric decoder df). The `nn` decoder raises an error.
- `cell_type = "none"` (default), `"together"`, and `"separate"` are all supported.

**How it works (plain / `together` mode):**

1. Run the permutation test on the full gene set.
2. If `p_value >= alpha`, stop — no spatial structure found.
3. Identify significant SVGs: per-gene F-test on the decoder predictions + Benjamini-Hochberg correction at `q < alpha`.
4. If none pass, stop.
5. Record gradient `k`, remove its SVGs from the dataset, repeat on remaining genes.
6. Stop after `max_gradients` iterations or when no genes remain.

Each gene is assigned to exactly one gradient. The covariate is carried through fitting but is not peeled recursively.

**How it works (`separate` mode):**

1. Run the full separate-mode permutation test (one model per cell type); save all standard per-type plots and CSVs.
2. For each cell type with `p_value < alpha`, run the gradient-peeling loop **independently** on that cell type's data and save outputs in `<TypeName>/recursive/`.
3. Cell types with `p_value >= alpha` are skipped (no `recursive/` folder created).
4. A top-level `<run_name>_celltype_recursive_summary.json` records which types were processed and the number of gradients found per type.

**Config parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recursive` | `false` | Enable recursive mode |
| `alpha` | `0.05` | Dataset significance threshold AND SVG BH q-value threshold |
| `max_gradients` | `10` | Safety cap on number of gradients |

Example config for plain mode (`configs/hypothalamus_recursive.json`):

```json
{
  "test": {
    "method": "parallel_permutation",
    "decoder": "quadratic",
    "recursive": true,
    "alpha": 0.05,
    "max_gradients": 10
  }
}
```

Example config for separate cell-type mode (`configs/hypothalamus_celltype_recursive.json`):

```json
{
  "data": { "cell_type": "separate", "cell_type_key": "dominant_cell_type" },
  "test": {
    "method": "parallel_permutation",
    "decoder": "quadratic",
    "recursive": true,
    "alpha": 0.05,
    "max_gradients": 10
  }
}
```

Run:

```bash
python run_permutation.py --config configs/hypothalamus_recursive.json
python run_permutation.py --config configs/hypothalamus_celltype_recursive.json
```

**Output layout (plain / `together` mode):**

```text
results/<run_name>/recursive/
  combined_sig_genes.csv          # all SVGs: gene, p_value, q_value, corresponding_gradient
  recursive_summary.json          # per-gradient summary + paths
  gradient_1/
    gradient_1_dataset.png        # spatial cell scatter coloured by expression
    gradient_1_isodepth.png       # isodepth triptych (true, best null, worst null)
    gradient_1_isodepth.npy       # isodepth vector (float32)
    gradient_1_metric_distribution.png
    gradient_1_null_distribution.npy
    gradient_1_top_genes.png      # top genes by |Spearman ρ| vs gradient
    gradient_1_sig_genes.csv
    gradient_1_result.json
  gradient_2/
    ...
```

**Output layout (`separate` mode):**

```text
results/<run_name>/
  <run_name>_result.json                         # standard separate result
  <run_name>_celltype_recursive_summary.json     # NEW: per-type recursive summary
  TypeA/                                         # standard per-type outputs (unchanged)
    TypeA_dataset.png
    TypeA_isodepth.png
    TypeA_metric_distribution.png
    TypeA_gene_expression_vs_isodepth.png
    TypeA_isodepth_sig_genes.csv
    recursive/                                   # NEW: only for types with p < alpha
      combined_sig_genes.csv
      recursive_summary.json
      gradient_1/
        gradient_1_dataset.png
        gradient_1_isodepth.png
        gradient_1_isodepth.npy
        gradient_1_metric_distribution.png
        gradient_1_null_distribution.npy
        gradient_1_top_genes.png
        gradient_1_sig_genes.csv
        gradient_1_result.json
      gradient_2/  ...
  TypeB/                                         # no recursive/ if p >= alpha
    ...
```

## Notes

- The default `test.decoder` is `nn` (multilayer perceptron); use `linear` for the linear decoder when needed.
- The default metric is `nll_gaussian_mse`.
- The active runner supports `data.source = "h5ad"` and `data.source = "synthetic"`.
- The shared method/config interfaces are designed so future tests and metrics can reuse the same contracts.

## Validation

Smoothness checks are available under `validation/`:

- `python validation/smoothness.py`
- `python validation/permuted_smoothness.py`

### Independent-model training and reproducibility

Parallel model losses are averaged within each model over the included cells/genes,
then **summed across independent models** for backpropagation. Adding permutations,
restarts, or covariates does not introduce a different loss-scale divisor. Stopping
other slots or splitting a batch for GPU memory also does not rescale an active
model's gradients. The former `gradient_scale_divisor` trainer argument has been
removed, including its use in the bias diagnostic.

All neural trainers use the same checkpoint rule: minibatch training restores the
best full-data training epoch per restart, even with `patience=0`; full-batch training
restores the best epoch when early stopping is enabled and otherwise uses the final
epoch. The same optional cosine minibatch learning-rate schedule applies to fixed
covariate decoders. Checkpointed spatial loss histories report the retained best
loss per slot, so the final recorded ordering agrees with restart selection.
Gaussian warm-up checkpoints are compared separately from the final Poisson phase.

Gradient-scale invariance assumes identical initial weights and data batches. A
fixed seed alone does not guarantee identical weights when model shapes or counts
change, and adding a covariate can change the architecture and objective. Floating
point kernels may also differ with batch shape. The regression tests explicitly
copy weights to isolate the scaling guarantee.

Results produced before this correction retain their original scores. Recompute
both observed and permutation models to obtain results under the corrected
training procedure; do not combine newly trained observed models with old nulls.
