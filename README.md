# Isodepth Statistical Testing

This repository now uses a refactored pipeline for running isodepth-based statistical tests with a stable configuration model and a clean separation between data loading, methods, and orchestration.

## Active Pipeline

The active pipeline is built around these files and folders:

- `configs/`: JSON config files
- `data/`: data loaders, transforms, schemas, and dataset assets
- `methods/`: model architectures, training code, test methods, and metrics
- `run_permutation.py`: generic entry point that executes a config-defined dataset and test method
- `experiments/configuration.py`: shared config merge and output writing logic
- `data/schemas.py`: shared `DatasetBundle`, `TestConfig`, `RunConfig`, and `TestResult` contracts
- `validation/`: smoothness and related validation checks

Legacy scripts that are not part of this pipeline are archived under `old/`.

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
    "standardize": true,
    "max_cells": null,
    "seed": 0
  },
  "test": {
    "method": "parallel_permutation",
    "metric": "nll_gaussian_mse",
    "n_perms": 100,
    "n_nulls": 50,
    "epochs": 5000,
    "lr": 0.001,
    "patience": 50,
    "seed": 0,
    "device": "auto",
    "batch_size": null,
    "n_experts": 2,
    "delta": 0.05,
    "perturb_target": "coordinates",
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

There are ready-made perturbation examples at [noise_perturbation.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/noise_perturbation.json), [radial_perturbation.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/radial_perturbation.json), and [mouse_hippocampus_perturbation.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/mouse_hippocampus_perturbation.json).

Supported `test.method` values:

- `parallel_permutation`
- `full_retraining`
- `frozen_encoder`
- `gaston_mix_closed_form`
- `perturbation_robustness`

## Running A Config

Run the config file:

```bash
python run_permutation.py \
  --config configs/my_run.json
```

Run the bundled example config:

```bash
python run_permutation.py \
  --config configs/noise_perturbation.json
```

Run a config with CLI overrides:

```bash
python run_permutation.py \
  --config configs/my_run.json \
  --h5ad data/h5ad/hippocampal_pyramidal.h5ad \
  --method frozen_encoder \
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
- `--standardize`: z-score each gene after loading/filtering.
- `--no-standardize`: leave expression values on their original scale.

`--standardize` details:
This controls the same `data.standardize` field from the config.
The default pipeline behavior is standardization enabled.
Use `--no-standardize` only if you intentionally want raw-scale expression values.

- `--max-cells`: randomly subsample at most this many cells/spots from the dataset.
- `--seed`: random seed used for dataset subsampling and permutation generation.
- `--mode`: synthetic data mode. Current examples include `radial`, `checkerboard`, and `noise`.
- `--n-cells`: requested number of synthetic cells/spots.
- `--n-genes`: requested number of synthetic genes.
- `--sigma`: synthetic noise scale.

- `--method`: test method. Supported values are `parallel_permutation`, `full_retraining`, `frozen_encoder`, `gaston_mix_closed_form`, `perturbation_robustness`.
- `--metric`: one of `nll_gaussian_mse`, `mse`, `pearson_corr_mean`, `spearman_corr_mean`.
- `--n-perms`: number of perturbations for `perturbation_robustness`, or number of permutations for existence-style methods.
- `--n-nulls`: number of null datasets for `perturbation_robustness`.
- `--epochs`: training epochs for learned-model methods.
- `--lr`: learning rate.
- `--patience`: early stopping patience.
- `--device`: compute device such as `cpu`, `cuda`, `mps`, or `auto`.
- `--batch-size`: optional grouping cap for `perturbation_robustness` null datasets. Each grouped null batch packs up to `batch_size * (n_perms + 1)` models into one parallel trainer call. Leave unset to batch all null datasets together.
- `--n-experts`: number of GASTON-MIX experts. Only used by `gaston_mix_closed_form`.
- `--delta`: perturbation scale for `perturbation_robustness`. Noise is sampled as a fraction of each spatial axis range.
- `--perturb-target`: perturbation target for `perturbation_robustness`. The current implementation only supports `coordinates`.

- `--out-dir`: parent output directory.
- `--run-name`: run-specific subdirectory name under `out_dir`, and the filename prefix for saved artifacts.
- `--save-preds`: save prediction arrays when the selected method exposes them.
- `--no-save-preds`: disable prediction array saving.
- `--save-perm-stats`: save `perm_stats.npy`.
- `--no-save-perm-stats`: disable `perm_stats.npy` saving.
- `--verbose`: enable training/progress output.
- `--quiet`: disable training/progress output.

## Output Format

Each run writes:

- standardized result JSON
- `S.npy`
- `A.npy`
- `perm_stats.npy` when enabled
- `<run_name>_isodepth_triptych.png`
- `<run_name>_metric_distribution.png`
- optional predictions when enabled
- the full merged effective config embedded in the result JSON

Typical output directory contents:

```text
results/
  <run_name>/
    S.npy
    A.npy
    perm_stats.npy
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

## Perturbation Robustness

`perturbation_robustness` measures whether the learned isodepth is stable after perturbing the spatial coordinates and retraining the model.

- On the observed dataset, fit one original model plus `n_perms` perturbed-coordinate refits and compute the perturbation score for each refit.
- Summarize those observed perturbation scores by their mean. This is the reported `stat_true`.
- Build a null distribution by destroying the spatial-expression pairing with random permutations of `A`.
- For each null dataset, rerun the same perturb-and-refit pipeline and summarize the resulting perturbation scores by their mean.
- Compare the observed mean perturbation score to that null distribution to obtain the p-value.

Metric semantics for `perturbation_robustness`:

- `mse` and `nll_gaussian_mse` compare min-max normalized isodepth vectors, where lower is better.
- `pearson_corr_mean` and `spearman_corr_mean` compare min-max normalized single-column vectors and use the absolute correlation, where higher is better.

Example config:

```json
{
  "test": {
    "method": "perturbation_robustness",
    "metric": "spearman_corr_mean",
    "n_perms": 200,
    "n_nulls": 50,
    "delta": 0.05,
    "perturb_target": "coordinates"
  }
}
```

## Notes

- The default metric is `nll_gaussian_mse`.
- The active runner supports `data.source = "h5ad"` and `data.source = "synthetic"`.
- The shared method/config interfaces are designed so future tests and metrics can reuse the same contracts.

## Validation

Smoothness checks are available under `validation/`:

- `python validation/smoothness.py`
- `python validation/permuted_smoothness.py`
