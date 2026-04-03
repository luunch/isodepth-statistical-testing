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
    "q": null,
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
    "delta": [0.05],
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

There are ready-made perturbation examples at [noise_perturbation.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/noise_perturbation.json), [noise_comparison_perturbation.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/noise_comparison_perturbation.json), [radial_comparison_perturbation.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/radial_comparison_perturbation.json), and [mouse_hippocampus_comparison_perturbation.json](/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/configs/mouse_hippocampus_comparison_perturbation.json).

Supported `test.method` values:

- `parallel_permutation`
- `full_retraining`
- `frozen_encoder`
- `gaston_mix_closed_form`
- `comparison_perturbation_test`
- `perturbation_test`
- `comparison_subsampling_test`
- `subsampling_test`

## Running A Config

Run the config file:

```bash
python run_permutation.py \
  --config configs/my_run.json
```

Run the bundled example config:

```bash
python run_permutation.py \
  --config configs/noise_comparison_perturbation.json
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
- `--q`: if provided, replace the expression matrix with a Poisson low-rank latent embedding of width `2*q`.

`--standardize` details:
This controls the same `data.standardize` field from the config.
The default pipeline behavior is standardization enabled.
Use `--no-standardize` only if you intentionally want raw-scale expression values.

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

- `--method`: test method. Supported values are `parallel_permutation`, `full_retraining`, `frozen_encoder`, `gaston_mix_closed_form`, `comparison_perturbation_test`, `perturbation_test`, `comparison_subsampling_test`, `subsampling_test`.
- `--metric`: one of `nll_gaussian_mse`, `mse`, `pearson_corr_mean`, `spearman_corr_mean`.
- `--n-perms`: number of perturbations for `comparison_perturbation_test` and `perturbation_test`, number of transformed trials for `subsampling_test`, or number of permutations for existence-style methods.
- `--n-nulls`: number of null datasets for `comparison_perturbation_test` and `comparison_subsampling_test`. This field is ignored by the direct transformed-data methods.
- `--epochs`: training epochs for learned-model methods.
- `--lr`: learning rate.
- `--patience`: early stopping patience.
- `--device`: compute device such as `cpu`, `cuda`, `mps`, or `auto`.
- `--batch-size`: optional grouping cap for `comparison_perturbation_test` null datasets. Each grouped null batch packs up to `batch_size * (n_perms + 1)` models into one parallel trainer call. Leave unset to batch all null datasets together.
- `--n-experts`: number of GASTON-MIX experts. Only used by `gaston_mix_closed_form`.
- `--delta`: comma-separated perturbation scales for perturbation methods. Noise is sampled as a fraction of each spatial axis range.
- `--perturb-target`: perturbation target for perturbation methods. The current implementation only supports `coordinates`.
- `--subset-fractions`: comma-separated subset fractions for subsampling methods. The default schedule is `0.5,0.7,0.9`.
- `--n-subsets`: number of repeated subsets drawn at each subset fraction for `comparison_subsampling_test` and `subsampling_test`.

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

## Comparison Perturbation Test

`comparison_perturbation_test` measures whether the learned isodepth is stable after perturbing the spatial coordinates and retraining the model.

- On the observed dataset, fit one original model plus `n_perms` perturbed-coordinate refits at each delta in the `delta` list and compute the perturbation score for each refit.
- For each delta, summarize the observed perturbation scores at that delta by their mean. Each delta gets its own `stat_true`, null distribution, and p-value.
- Build a null distribution by destroying the spatial-expression pairing with random permutations of `A`.
- For each null dataset, rerun the same perturb-and-refit pipeline and keep a separate delta-specific null mean for every delta.
- Compare each observed delta-specific mean perturbation score to its matching delta-specific null distribution to obtain the p-value.

Metric semantics for `comparison_perturbation_test`:

- `mse` and `nll_gaussian_mse` compare min-max normalized isodepth vectors, where lower is better.
- `pearson_corr_mean` and `spearman_corr_mean` compare min-max normalized single-column vectors and use the absolute correlation, where higher is better.

Example config:

```json
{
  "test": {
    "method": "comparison_perturbation_test",
    "metric": "spearman_corr_mean",
    "n_perms": 200,
    "n_nulls": 50,
    "delta": [0.05, 0.1, 0.5],
    "perturb_target": "coordinates"
  }
}
```

Multi-delta perturbation plots:

- The isodepth figure becomes a 3-column grid with one row per delta.
- Each row contains the full-data isodepth, the lowest-metric perturbed refit for that delta, and the highest-metric perturbed refit for that delta.
- A separate scatter plot shows delta on the x-axis and per-delta p-value on the y-axis.

## Perturbation Test

`perturbation_test` measures how unusual the original-data reconstruction loss is relative to refits on directly perturbed coordinate datasets.

- Fit one baseline model on the original `(S, A)` dataset and use its reconstruction loss as `stat_true`.
- For each configured delta, generate `n_perms` perturbed coordinate datasets, refit the model on each perturbed `S`, and evaluate reconstruction loss against the observed `A`.
- Use those perturbed-data losses directly as the delta-specific null distribution.
- Report one p-value per delta and expose the first configured delta as the top-level `stat_true`, `stat_perm`, and `p_value`.

Metric semantics for `perturbation_test`:

- `mse` and `nll_gaussian_mse` are supported, where lower is better.
- Correlation metrics are rejected because this method tests reconstruction loss directly.

Example config:

```json
{
  "test": {
    "method": "perturbation_test",
    "metric": "nll_gaussian_mse",
    "n_perms": 200,
    "delta": [0.05, 0.1, 0.5],
    "perturb_target": "coordinates"
  }
}
```

## Comparison Subsampling Test

`comparison_subsampling_test` measures whether the full-data isodepth can be recovered from repeated subsets of spots.

- Fit a baseline model on the full dataset and extract the reference isodepth `d`.
- Draw repeated subsets at one or more subset fractions.
- Refit isodepth models in parallel while restricting each model's training loss to its selected subset.
- Compute the masked reconstruction loss for each subset refit and summarize those observed losses by their mean within each subset fraction. Each fraction gets its own `stat_true`, null distribution, and p-value.
- Build a null distribution by permuting the expression rows, reusing the same subset masks, and recomputing the fraction-specific mean masked loss for each null replicate.
- Record the absolute Spearman agreement between each subset isodepth `d_A` and `d` on the selected spots as a diagnostic artifact.
- Compare each observed fraction-specific mean loss to its matching fraction-specific null distribution to obtain the p-value.

Metric semantics for `comparison_subsampling_test`:

- `mse` and `nll_gaussian_mse` are supported for hypothesis testing, where lower is better.
- Correlation metrics are not used as the primary test statistic for this method.

Recommended default schedule:

- `subset_fractions = [0.5, 0.7, 0.9]`
- `n_subsets = 10`
- total observed subset refits per run = `30`

Plots for `comparison_subsampling_test`:

- The isodepth figure becomes a 3-column grid with one row per subset fraction.
- Each row contains the full-data isodepth, the lowest-loss subset refit for that fraction, and the highest-loss subset refit for that fraction.
- A separate scatter plot shows subset fraction on the x-axis and per-fraction p-value on the y-axis.

## Subsampling Test

`subsampling_test` measures how unusual the full-data reconstruction loss is relative to refits on directly subsampled coordinate sets.

- Fit one baseline model on the full dataset and use its full-data reconstruction loss as `stat_true`.
- Draw repeated subsets at each configured subset fraction.
- Refit the model with training loss restricted to each sampled subset.
- Evaluate each subset refit on its selected spots, scale the subset loss by dividing by the subset fraction, and use those scaled losses as the fraction-specific null distribution.
- Report one p-value per subset fraction and expose the first configured fraction as the top-level `stat_true`, `stat_perm`, and `p_value`.

Metric semantics for `subsampling_test`:

- `mse` and `nll_gaussian_mse` are supported, where lower is better.
- Correlation diagnostics are retained as artifacts only and are not the test statistic.

Example config:

```json
{
  "test": {
    "method": "subsampling_test",
    "metric": "mse",
    "n_perms": 50,
    "subset_fractions": [0.5, 0.7, 0.9],
    "n_subsets": 10
  }
}
```

Example config:

```json
{
  "test": {
    "method": "comparison_subsampling_test",
    "metric": "mse",
    "n_nulls": 50,
    "subset_fractions": [0.5, 0.7, 0.9],
    "n_subsets": 10
  }
}
```

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
```

## Notes

- The default metric is `nll_gaussian_mse`.
- The active runner supports `data.source = "h5ad"` and `data.source = "synthetic"`.
- The shared method/config interfaces are designed so future tests and metrics can reuse the same contracts.

## Validation

Smoothness checks are available under `validation/`:

- `python validation/smoothness.py`
- `python validation/permuted_smoothness.py`
